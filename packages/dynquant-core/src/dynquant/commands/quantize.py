"""``dynquant quantize`` -- allocate widths and encode the weights.

What this writes, precisely
---------------------------
A directory holding the model with **quantized values in the compute dtype**: each
weight has been encoded to its allocated width and decoded back, so every number
in it is exactly a number the packed checkpoint would hold, but the file stores it
as bf16/fp16. ``AutoModelForCausalLM.from_pretrained`` reads it with no DynQuant
installed, and any accuracy measured on it is the accuracy of the quantized model.
Its *size* is not the quantized size, and this command says so on every run rather
than leaving it to be inferred -- the research supplement reported storage savings
from exactly this path while the model it loaded was fp16 all along.

The size the packed form *will* occupy is not a guess: it is the bit map's own
accounting, scales and offsets and packing overhead included, printed next to the
directory size. The packed writer is P9; ``dynquant.runtime.pack_model`` already
keeps the weights packed in VRAM for an in-process evaluation, which is what
``dynquant eval --map`` uses -- so pair ``--save-map`` with this command when the
memory figure is the point, and quantize the values when the accuracy is.

Where the widths come from
--------------------------
Three ways, in the order they are checked:

* ``--map FILE`` -- a map written earlier by ``dynquant inspect --save-map``. The
  reviewed map is then the one that gets applied, with no second allocation in
  between that could differ from the one that was looked at.
* ``--uniform BITS`` -- every module at one width. The control arm; structural
  roles keep their floors, because a 2-bit router is not a low-precision model,
  it is a broken one.
* ``--stats DIR --target 3.0`` -- allocate now.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import _shared

if TYPE_CHECKING:
    import argparse

    from dynquant.allocate.knapsack import BitMap

__all__ = ["run"]


@dataclass(slots=True)
class _Stored:
    """What the packed form will occupy, and where that figure came from.

    Separate from :class:`BitMap` because the ``--map`` path has no ``BitMap``: the
    accounting in the file was produced by whatever wrote it and is reported as the
    file's rather than recomputed. But it is still *there* -- ``--save-map`` records
    it -- and declining to print it is how the directory size ends up being read as
    the quantized size, which is the single number the research supplement got wrong.
    """

    nbytes: int
    average_bits: float
    provenance: str
    violations: tuple[Any, ...] = ()

    @classmethod
    def from_bit_map(cls, bit_map: BitMap) -> _Stored:
        return cls(
            nbytes=bit_map.nbytes,
            average_bits=bit_map.average_bits,
            provenance="this allocation",
            violations=bit_map.violations,
        )

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any], source: str) -> _Stored | None:
        """The file's own numbers, or ``None`` if it did not record them.

        A hand-written or stage-script map carries only widths, and inventing the
        accounting for it would need the graph its writer used.
        """
        nbytes, average = metadata.get("nbytes"), metadata.get("average_bits")
        if nbytes is None or average is None:
            return None
        return cls(
            nbytes=int(nbytes),
            average_bits=float(average),
            provenance=f"{source}, as recorded",
            violations=tuple(metadata.get("violations", ())),
        )


@dataclass(slots=True)
class _Widths:
    """Where the widths came from, resolved once.

    ``bit_map`` is ``None`` on the ``--map`` path and ``stored`` is not: a file's
    accounting can be reported without being recomputed, and only a live ``BitMap``
    can be written back out as one.
    """

    bits: dict[str, int]
    source: str
    allocator: str
    bit_map: BitMap | None = None
    stored: _Stored | None = None


def run(args: argparse.Namespace) -> int:
    from dynquant._version import __version__
    from dynquant.errors import DynQuantError
    from dynquant.quant.quantizer import quantize_model

    if args.output is None and not args.dry_run:
        raise DynQuantError("-o/--output is required unless --dry-run is given")

    model = _shared.load_model(
        args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )

    resolved = _resolve_widths(model, args)
    bits, bit_map, stored = resolved.bits, resolved.bit_map, resolved.stored
    _shared.check_map_covers(model, bits)

    print(
        f"\n{len(bits)} modules to quantize, widths from {resolved.source}"
        + (f"\n{bit_map.summary()}" if bit_map is not None else ""),
        flush=True,
    )
    if stored is not None:
        print(
            f"  packed form would occupy {stored.nbytes / 2**30:.3f} GiB at "
            f"{stored.average_bits:.4f} average bits ({stored.provenance})",
            flush=True,
        )
        if bit_map is None:
            # An allocated map already listed these through ``BitMap.summary()``.
            _print_violations(stored)
    else:
        print(
            f"  {resolved.source} records only widths, so the packed size it implies "
            f"is not known here. Re-run `dynquant inspect --save-map` to get it.",
            flush=True,
        )

    if args.dry_run:
        print("\n--dry-run: no weight was touched", flush=True)
        if args.save_map and bit_map is not None:
            written = _shared.write_bit_maps(
                args.save_map,
                {_map_key(args, bit_map): bit_map},
                model=args.model,
                stats=args.stats,
                allocator=resolved.allocator,
                group_size=args.group_size,
            )
            print(f"-> wrote {written}", flush=True)
        return 0

    started = time.time()
    report = quantize_model(
        model,
        bits,
        group_size=args.group_size,
        symmetric=args.symmetric,
        in_place=True,
        compute_device=getattr(args, "compute_device", "auto"),
        progress=None if args.quiet else _shared.progress_printer("quantize"),
    )
    elapsed = time.time() - started
    print(f"\n{report.summary()}\n  {elapsed:.0f}s", flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\nwriting {out} ...", flush=True)
    model.save_pretrained(str(out))
    _save_tokenizer(args, out)

    layers = {
        name: {
            "bits": layer.bits,
            "num_params": layer.num_params,
            "rmse": layer.rmse,
            "relative_error": layer.relative_error,
            "clipped_fraction": layer.clipped_fraction,
            "clip_improvement": layer.clip_improvement,
        }
        for name, layer in report.layers.items()
    }
    maps: dict[str, BitMap] = {}
    if bit_map is not None:
        maps[_map_key(args, bit_map)] = bit_map
    written = _shared.write_bit_maps(
        out,
        maps,
        model=args.model,
        stats=args.stats,
        allocator=resolved.allocator,
        group_size=args.group_size,
        extra={
            "storage": "compute-dtype values, quantized then decoded (not packed)",
            "quantized": {
                "bits": bits,
                "seconds": round(elapsed, 1),
                "symmetric": args.symmetric,
                "layers": layers,
            },
        },
    )

    directory_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(
        f"-> wrote {out} ({directory_bytes / 2**30:.3f} GiB on disk) and {written.name}\n"
        f"   Storage is the compute dtype: every value is quantized, the container is "
        f"not.\n"
        f"   Accuracy measured on this directory is the quantized model's accuracy; "
        f"its size is not the\n   quantized size"
        + (
            f" -- that is {stored.nbytes / 2**30:.3f} GiB, and "
            f"`dynquant eval --map` measures it in VRAM."
            if stored is not None
            else "."
        ),
        flush=True,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "dynquant_core": __version__,
                    "model": args.model,
                    "output": str(out),
                    "allocator": resolved.allocator,
                    "group_size": args.group_size,
                    "modules": len(bits),
                    "seconds": round(elapsed, 1),
                    "average_bits": None if stored is None else stored.average_bits,
                    "packed_nbytes": None if stored is None else stored.nbytes,
                    "packed_size_from": None if stored is None else stored.provenance,
                    "directory_nbytes": directory_bytes,
                    "layers": layers,
                },
                indent=2,
            )
        )
    return 0


def _print_violations(stored: _Stored) -> None:
    """The floors a map file recorded as breached, in ``BitMap.summary()``'s shape.

    Printed rather than skipped on the ``--map`` path because a breached floor is
    the one thing in a bit map that is a risk to the model rather than an accounting
    detail, and the map may have been written by a run whose output nobody still
    has. Violations arrive here as JSON objects rather than ``FloorViolation``
    instances, so the fields are read by key -- and a file that recorded a count but
    not the details still gets the count.
    """
    if not stored.violations:
        return
    rows = [v for v in stored.violations if isinstance(v, dict)]
    print(f"  {len(stored.violations)} floors breached to fit the budget:", flush=True)
    for v in sorted(rows, key=lambda v: -int(v.get("num_params", 0)))[:5]:
        print(
            f"    {v.get('name', '?')} ({v.get('role', '?')}) {v.get('floor_bits')}b -> "
            f"{v.get('assigned_bits')}b, {int(v.get('num_params', 0)) / 1e6:.1f}M params",
            flush=True,
        )


def _resolve_widths(model: Any, args: argparse.Namespace) -> _Widths:
    """Where the widths come from, and everything that follows from that.

    A map read from a file has no ``BitMap`` behind it -- the accounting in the file
    was produced by whatever wrote it, and recomputing it here would need the graph
    the writer used. So the file's own numbers are carried as :class:`_Stored` and
    reported as the file's; the only claim made about them is where they came from.
    """
    from dynquant.errors import DynQuantError

    if args.map is not None:
        bits, metadata = _shared.read_bit_map(args.map, key=args.map_key)
        group_size = metadata.get("group_size")
        if group_size is not None and int(group_size) != args.group_size:
            raise DynQuantError(
                f"{args.map} was built at group_size={group_size} but --group-size is "
                f"{args.group_size}. The widths were priced against that grouping; "
                f"applying them at another changes the stored size and the error of "
                f"every tensor."
            )
        origin = f"{args.map}"
        if "map_key" in metadata:
            origin += f" [{metadata['map_key']}]"
        return _Widths(
            bits=bits,
            source=origin,
            allocator=str(metadata.get("allocator", "file")),
            stored=_Stored.from_metadata(metadata, origin),
        )

    graph_inputs = None
    if args.uniform is not None:
        from dynquant.graph.classify import classify_model

        graph = classify_model(model, overrides=_shared.parse_overrides(args.role))
        bit_map = _shared.uniform_map(graph, args.uniform, group_size=args.group_size)
        return _Widths(
            bits=bit_map.bits,
            source=f"uniform {args.uniform} bit",
            allocator="uniform",
            bit_map=bit_map,
            stored=_Stored.from_bit_map(bit_map),
        )

    given = [args.target is not None, args.target_size is not None, args.target_ratio is not None]
    if sum(given) != 1:
        raise DynQuantError(
            "give exactly one of --target, --target-size, --target-ratio, --uniform or --map"
        )

    graph_inputs = _shared.build_inputs(
        model,
        stats=args.stats,
        moments=args.moments,
        group_size=args.group_size,
        symmetric=args.symmetric,
        overrides=_shared.parse_overrides(args.role),
        soft_floors=not args.hard_floors,
        verbose=not args.quiet,
    )
    bit_map = _shared.allocate(
        graph_inputs,
        target_bits=args.target,
        target_size=args.target_size,
        target_ratio=args.target_ratio,
    )
    return _Widths(
        bits=bit_map.bits,
        source=bit_map.target_label,
        allocator=graph_inputs.allocator,
        bit_map=bit_map,
        stored=_Stored.from_bit_map(bit_map),
    )


def _map_key(args: argparse.Namespace, bit_map: BitMap) -> str:
    if args.target is not None:
        return f"{args.target:.2f}"
    if args.uniform is not None:
        return f"uniform-{args.uniform}"
    return bit_map.target_label or "map"


def _save_tokenizer(args: argparse.Namespace, out: Path) -> None:
    """Copy the tokenizer across, so the output directory is loadable on its own.

    Best-effort by design: a directory without a tokenizer is still a valid model
    directory, and failing a quantization that already succeeded because a
    tokenizer file could not be found would be the wrong trade. The reason is
    printed either way -- an evaluation that then has to be pointed at the original
    tokenizer should know why.
    """
    source = args.tokenizer or args.model
    try:
        tokenizer = _shared.load_tokenizer(source, trust_remote_code=args.trust_remote_code)
        tokenizer.save_pretrained(str(out))
    except Exception as exc:  # noqa: BLE001 -- see the docstring
        print(f"  note: no tokenizer copied from {source} ({exc})", flush=True)
