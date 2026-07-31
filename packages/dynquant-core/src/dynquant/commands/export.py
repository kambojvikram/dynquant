"""``dynquant export`` -- write the packed checkpoint that servers actually load.

The difference from ``dynquant quantize`` is the container, not the arithmetic.
Both encode with the same clipping search at the same widths, so the values are
identical; ``quantize`` decodes them back into a bf16 tensor so plain transformers
can read the directory, and this writes the packed form -- one ``qweight`` /
``scales`` / ``offsets`` triple per module, at the width the allocator chose.

    dynquant export Qwen/Qwen3-1.7B --map maps.json -o qwen3-dynquant-3bit
    vllm serve qwen3-dynquant-3bit

The second line needs nothing else: ``pip install dynquant`` registers the
quantization method through vLLM's plugin entry point, and ``config.json`` in the
exported directory names it. See
:mod:`dynquant.integration.vllm_plugin` for what happens on the vLLM side and
:mod:`dynquant.quant.checkpoint` for the format.

Widths come from the same three places ``quantize`` takes them from -- ``--map``,
``--uniform``, or ``--stats`` plus a target -- and that resolution is shared code,
not a parallel implementation, so a map inspected for one command means the same
thing to the other.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from . import _shared
from .quantize import _map_key, _print_violations, _resolve_widths, _save_tokenizer

if TYPE_CHECKING:
    import argparse

__all__ = ["run"]


def run(args: argparse.Namespace) -> int:
    from dynquant._version import __version__
    from dynquant.errors import DynQuantError
    from dynquant.quant.checkpoint import export_packed_checkpoint

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
        f"\n{len(bits)} modules to pack, widths from {resolved.source}"
        + (f"\n{bit_map.summary()}" if bit_map is not None else ""),
        flush=True,
    )
    if stored is not None:
        print(
            f"  predicted {stored.nbytes / 2**30:.3f} GiB at {stored.average_bits:.4f} "
            f"average bits ({stored.provenance})",
            flush=True,
        )
        if bit_map is None:
            _print_violations(stored)

    if args.dry_run:
        print("\n--dry-run: no weight was touched", flush=True)
        return 0

    out = Path(args.output)
    started = time.time()
    report = export_packed_checkpoint(
        model,
        bits,
        output_dir=out,
        group_size=args.group_size,
        symmetric=args.symmetric,
        compute_device=args.compute_device,
        max_shard_bytes=_parse_size(args.max_shard_size),
        provenance={
            "dynquant_core": __version__,
            "model": args.model,
            "stats": args.stats,
            "allocator": resolved.allocator,
            "widths_from": resolved.source,
            "predicted_nbytes": None if stored is None else stored.nbytes,
        },
        progress=None if args.quiet else _shared.progress_printer("pack"),
    )
    elapsed = time.time() - started
    _save_tokenizer(args, out)

    # The map goes in beside the checkpoint. `config.json` already carries every
    # width the loader needs; this is the provenance -- which stats file, which
    # allocator, which floors were breached -- and none of that survives in the
    # loader's view of the directory.
    if bit_map is not None:
        _shared.write_bit_maps(
            out,
            {_map_key(args, bit_map): bit_map},
            model=args.model,
            stats=args.stats,
            allocator=resolved.allocator,
            group_size=args.group_size,
        )

    directory_bytes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\n{report.summary()}\n  {elapsed:.0f}s", flush=True)
    if stored is not None:
        # The allocator prices a map before anything is encoded; this is the same
        # figure measured from the tensors that were written. They should agree to
        # the byte, and a gap means the pricing model and the writer disagree
        # about something -- which is worth seeing rather than averaging away.
        delta = report.packed_bytes - stored.nbytes
        if delta:
            print(
                f"  note: packed weights came out {delta:+,} B against the map's "
                f"prediction of {stored.nbytes:,} B",
                flush=True,
            )
    print(
        f"-> wrote {out} ({directory_bytes / 2**30:.3f} GiB on disk)\n"
        f"   Serve it with `vllm serve {out}` -- the plugin registers itself, so no "
        f"flag and no vLLM patch.",
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
                    "symmetric": args.symmetric,
                    "modules": report.quantized_modules,
                    "tied": report.tied,
                    "seconds": round(elapsed, 1),
                    "average_bits": report.average_bits,
                    "packed_nbytes": report.packed_bytes,
                    "dense_nbytes": report.dense_bytes,
                    "directory_nbytes": directory_bytes,
                    "files": list(report.files),
                },
                indent=2,
            )
        )
    return 0


_UNITS = {
    "": 1,
    "B": 1,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
    "KIB": 1024,
    "MIB": 1024**2,
    "GIB": 1024**3,
    "TIB": 1024**4,
}


def _parse_size(text: str | None) -> int:
    """``"4GiB"`` -> bytes.

    Accepts the same spellings ``--target-size`` does, because a user who learned
    one on this CLI will type it at the other.
    """
    from dynquant.quant.checkpoint import DEFAULT_SHARD_BYTES

    if text is None:
        return DEFAULT_SHARD_BYTES

    from dynquant.errors import DynQuantError

    raw = text.strip().upper().replace(" ", "")
    for suffix in sorted(_UNITS, key=len, reverse=True):
        if suffix and raw.endswith(suffix):
            number, unit = raw[: -len(suffix)], suffix
            break
    else:
        number, unit = raw, ""
    try:
        value = float(number)
    except ValueError as exc:
        raise DynQuantError(
            f"could not read {text!r} as a size. Write it as a number with an optional "
            f"unit, e.g. 4GiB, 500MB, or 2000000000."
        ) from exc
    if value <= 0:
        raise DynQuantError(f"--max-shard-size must be positive, got {text!r}")
    return int(value * _UNITS[unit])
