#!/usr/bin/env python3
"""S3: build the quantization arms, at bytes that are equal by construction.

What S3 has to establish is not that DynQuant is accurate at 3 bits -- that is S4's
number -- but that whatever accuracy it has comes from *where it put the bits*. Three
things can produce a win at a nominal target and only one of them is the method:

* a bigger file (a 3.3-bit arm beating a 3.0-bit one is arithmetic, not allocation),
* the allocator's structure alone (floors and ROI ordering, with no signal at all),
* the signal (which module the fine-tune said was fragile).

So each anchor here is a *byte count*, taken from the uniform arm, and every other arm
at that anchor is allocated to that exact size. The comparison is then "same bytes,
different assignment" with nothing left to argue about, and the driver refuses to
publish arms whose realized sizes drift apart -- see :func:`check_matched`. Matched
bytes is a property this script enforces, not a sentence a report asserts.

The four arms at each anchor
----------------------------
=========== ============================================ ==========================
arm         how it is allocated                          what its gap to ``dq`` buys
=========== ============================================ ==========================
``rtn``     every module at one width                    the whole method
``rank``    real signal, rank-product ordering           the sensitivity estimator
``shuf``    signal permuted within role, sensitivity on  the signal itself
``dq``      real signal, measured sensitivity            --
=========== ============================================ ==========================

``shuf`` is the control that matters and the one it is easy to get wrong. Permuting
*within role* rather than globally holds every structural fact fixed -- role floors,
group sizes, the number of modules of each kind, and on a dense model the parameter
counts too, since all 32 ``o_proj`` are the same shape. What it destroys is only which
module a measurement belongs to. An arm that still wins after that was never reading
the signal. Phase 2 learned this from the row-order control, which *lost* 1.28 points
at identical bytes; a control that is merely "no stats" would have conflated the
signal with the ranking machinery that consumes it.

Not covered here, and it is a real gap: **row granularity.** Phase 2's largest single
lever was per-row allocation, and it lives in ``experiments/four_point/p2_rowbody.py``,
not in ``dynquant.allocate`` -- ``--target`` and ``--map`` are module-granularity only.
Every number this driver produces is therefore scoped to module granularity, the same
scope as the "signal is 12% of the margin" split. Promoting the row allocator into the
package would subsume the open per-partition-width question as well, and both are
decisions above this script's pay grade.

What this script does not do
----------------------------
It does not evaluate. ``scripts/run_s1_headroom.py`` is the campaign's only evaluator
and takes ``name=path``; a second one would be a second set of prompt-assembly and
parsing decisions that no results table would show. This driver prints the exact
invocation and ``--evaluate`` runs it.

Usage::

    python scripts/run_s3_allocate.py \\
        --model /workspace/runs/s2/phi4-mini.tulu3/merged \\
        --stats /workspace/runs/s2/phi4-mini.tulu3/stats \\
        --arm-dir /workspace/runs/s3/phi4-mini \\
        --out experiments/phase3/s3_allocation/phi4-mini
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

#: The widths whose uniform arms anchor the byte budgets. 3 and 4 because those are
#: the two regimes phase 2 separated: at 4 bits the role floors are affordable and
#: allocation has little to do, at 3 bits they are not and it decides the model.
ANCHORS = (3, 4)

#: Which signal file each arm allocates from, and whether it gets measured
#: sensitivity. ``None`` for the map means the arm is not allocated at all.
ARMS: dict[str, dict[str, Any]] = {
    "rtn": {"variant": None, "moments": False},
    "rank": {"variant": "signal", "moments": False},
    "shuf": {"variant": "shuffled", "moments": True},
    "dq": {"variant": "signal", "moments": True},
}

#: How far two arms at one anchor may differ and still be called matched, as a
#: fraction of the anchor. ``--target-size`` is a ceiling rather than an equality, so
#: an allocator that cannot spend the last few bits lands under it; a tenth of a
#: percent is far below the smallest accuracy difference these arms will show, and
#: anything larger is a size advantage wearing a method's name.
MATCH_TOLERANCE = 0.001

#: The stats fields that are a *measurement of this module* and therefore travel
#: together when the control permutes them. Everything not listed -- the name, the
#: parameter count, the role, the estimator that produced it -- describes the module's
#: identity rather than its behaviour and stays put, which is what makes the permuted
#: arm structurally identical to the real one.
PERMUTED_FIELDS = (
    "activation_rms_ema",
    "grad_norm_count",
    "grad_norm_mean",
    "grad_norm_var",
    "coherence_ema",
    "routing_hits",
    "forward_calls",
)


@dataclass(frozen=True, slots=True)
class Arm:
    """One measurement point: a name, a directory, and what it cost in bytes."""

    name: str
    anchor: int
    kind: str
    map_path: Path
    map_key: str
    nbytes: int
    average_bits: float
    violations: int
    directory: Path | None = None

    @property
    def label(self) -> str:
        """The record name S1 files this arm under. No dot: ``resolve_model`` bans it."""
        return f"{self.kind}{self.anchor}"


# ---------------------------------------------------------------------------
# Derived signal files
# ---------------------------------------------------------------------------


def permutation_within_role(stats: Any, seed: int, *, moments: Any = None) -> dict[str, str]:
    """Target module -> the module whose measurements it will carry.

    One permutation, applied to every artifact the allocator reads, because a module
    holding one donor's scalars and another's channel vectors carries a signal that
    belongs to no module at all -- a third distribution rather than a relabelled one.

    Grouped by role, and when the moments are given by channel shape as well. The shape
    refinement is not a weakening of "within role": on a dense model every member of a
    role has the same geometry, so the groups are unchanged. It exists because nothing
    guarantees that, and because a channel vector of the wrong length is not an error
    the consumer reports. ``_shapes_agree`` rejects it and ``_moments_for`` returns
    ``None``, so the module silently leaves the sensitivity table and is priced from the
    proxy instead -- the control would then be ablating *coverage*, which is structural,
    rather than correspondence, which is what it is for.

    A derangement is not attempted and would be the wrong thing to attempt: with a fixed
    seed the permutation is reproducible, and forcing every module to move would make
    the control depend on a rejection loop whose behaviour at small role sizes is its own
    confound. What is reported instead is how many modules actually moved, so a
    permutation that happened to be near-identity is visible rather than assumed away.

    Singleton roles -- a tied embedding is one -- are fixed points by construction. That
    is not a flaw in the control: there is no other module of that role to swap with, and
    the arm is honest about the fact in ``moved``.
    """

    def shape(name: str) -> tuple[int | None, int | None]:
        if moments is None:
            return (None, None)
        x = moments.input_sq.get(name)
        d = moments.output_grad_sq.get(name)
        return (None if x is None else int(x.numel()), None if d is None else int(d.numel()))

    groups: dict[tuple[Any, ...], list[str]] = {}
    for name, layer in stats.layers.items():
        groups.setdefault((layer.role or "", *shape(name)), []).append(name)

    rng = random.Random(seed)
    permutation: dict[str, str] = {}
    for _key, names in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ordered = sorted(names)
        donors = list(ordered)
        rng.shuffle(donors)
        permutation.update(zip(ordered, donors, strict=True))
    return permutation


def shuffle_stats(stats: Any, permutation: Mapping[str, str]) -> Any:
    """Relabel the per-module scalars, keeping name, role and parameter count in place."""
    permuted = dict(stats.layers)
    for target, donor in permutation.items():
        source = stats.layers[donor]
        permuted[target] = replace(
            stats.layers[target],
            **{field: getattr(source, field) for field in PERMUTED_FIELDS},
        )
    return replace(stats, layers=permuted)


def shuffle_moments(moments: Any, permutation: Mapping[str, str]) -> Any:
    """Relabel the channel moments under the same permutation the scalars got.

    Without this the ``shuf`` arm is not a control. ``--moments`` builds a measured
    sensitivity table, and the knapsack prices a width change from that table whenever
    the module has an entry, falling back to the stats-derived score only when it does
    not. Phi's moments cover all 129 quantizable modules, so permuting the stats alone
    changes nothing the allocator consults: the arm would have allocated identically to
    the treatment and reported a null with the ablation never having happened.
    """
    from dynquant.signals.moments import ChannelMoments

    out = ChannelMoments()
    for target, donor in permutation.items():
        for field_name in ("input_sq", "output_grad_sq"):
            source = getattr(moments, field_name).get(donor)
            if source is not None:
                getattr(out, field_name)[target] = source
    # Anything the stats never named keeps its own measurement rather than vanishing:
    # dropping a module from the moments would change which modules the allocator can
    # price, which is a structural difference and not a relabelling.
    for field_name in ("input_sq", "output_grad_sq"):
        for name, tensor in getattr(moments, field_name).items():
            getattr(out, field_name).setdefault(name, tensor)
    out.observations.update(moments.observations)
    return out


def _write_if_changed(path: Path, write: Callable[[Path], None]) -> bool:
    """Write through a scratch file, and keep the original when the bytes are identical.

    This is about the mtime, not the I/O. Both variants are a deterministic function of
    the S2 stats and the seed, so rewriting them unconditionally leaves them permanently
    newer than any map derived from them -- and ``--reuse-maps``, which asks whether a map
    postdates its inputs, could then never answer yes for the three arms that read a
    variant. Writing only on a real change makes the timestamp mean "the content changed",
    which is the question the reuse guard is actually asking. If a serializer turns out not
    to be byte-stable the only cost is that nothing is ever reused, which is the safe
    direction to fail in.
    """
    scratch = path.with_name(path.name + ".tmp")
    try:
        write(scratch)
        if path.is_file() and scratch.read_bytes() == path.read_bytes():
            return False
        scratch.replace(path)
        return True
    finally:
        scratch.unlink(missing_ok=True)


def write_variants(
    stats_path: Path, moments_path: Path, destination: Path, seed: int
) -> dict[str, dict[str, Any]]:
    """Write the derived signal files, and say how much each one actually differs.

    The real signal is copied rather than referenced so that the arms are reproducible
    from one directory after the fact, and so that ``dq`` and ``shuf`` are read through
    the same loader -- a control that differs from its treatment in how it was *parsed*
    is not a control.
    """
    from dynquant.signals.moments import load_moments, save_moments
    from dynquant.signals.schema import load_stats, save_stats

    destination.mkdir(parents=True, exist_ok=True)
    stats = load_stats(stats_path)
    moments = load_moments(moments_path)

    written: dict[str, dict[str, Any]] = {}
    signal_stats = destination / "dynquant_stats.signal.json"
    signal_moments = destination / "dynquant_moments.signal.safetensors"
    changed = _write_if_changed(signal_stats, lambda p: save_stats(stats, p))
    changed |= _write_if_changed(signal_moments, lambda p: save_moments(moments, p))
    written["signal"] = {
        "stats": str(signal_stats),
        "moments": str(signal_moments),
        "seed": None,
        "moved": None,
        "rewritten": changed,
    }

    permutation = permutation_within_role(stats, seed, moments=moments)
    moved = sum(1 for target, donor in permutation.items() if target != donor)
    shuffled_stats = destination / f"dynquant_stats.shuffled-{seed}.json"
    shuffled_moments = destination / f"dynquant_moments.shuffled-{seed}.safetensors"
    changed = _write_if_changed(
        shuffled_stats, lambda p: save_stats(shuffle_stats(stats, permutation), p)
    )
    changed |= _write_if_changed(
        shuffled_moments, lambda p: save_moments(shuffle_moments(moments, permutation), p)
    )
    written["shuffled"] = {
        "stats": str(shuffled_stats),
        "moments": str(shuffled_moments),
        "seed": seed,
        "moved": moved,
        "total": len(stats.layers),
        "rewritten": changed,
    }
    if moved == 0:
        raise SystemExit(
            f"the shuffled control moved 0 of {len(stats.layers)} modules at seed {seed}: "
            f"it is an identity permutation and would measure nothing. Pass --seed."
        )
    return written


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


def _run(cmd: list[str], *, what: str) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"{what} failed with exit {proc.returncode}")


def _inspect_cmd(args: argparse.Namespace, save_map: Path) -> list[str]:
    """The invariant part of every allocation: same model, same group size, same roles.

    ``-m dynquant`` on ``sys.executable`` rather than the console script, so the
    subprocess is guaranteed to be this environment's package and not whichever
    ``dynquant`` happens to be first on ``PATH``.
    """
    cmd = [
        sys.executable,
        "-m",
        "dynquant",
        "inspect",
        args.model,
        "--group-size",
        str(args.group_size),
        "--save-map",
        str(save_map),
        "--json",
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    return cmd


def anchor_cmd(args: argparse.Namespace, save_map: Path) -> list[str]:
    """One ``--uniform`` carrying every anchor width, not one flag per width.

    The flag is ``nargs="+"``, so a repeated option overwrites rather than accumulates
    and only the last anchor is written. Nothing announces that: the run proceeds, the
    4-bit anchor is there, and the 3-bit arm goes missing several minutes later when
    something asks for it.
    """
    return [*_inspect_cmd(args, save_map), "--uniform", *[str(w) for w in ANCHORS]]


def _inputs_mtime(args: argparse.Namespace) -> float:
    """When the newest of this run's *sources* was last written.

    The sources are S2's stats and moments and the merged checkpoint -- deliberately not
    the signal/shuffled variants the arms are actually allocated from. Those are derived
    by this same run and are a pure function of the sources and the seed, so stamping a
    map against them would be asking whether the map predates a file the run just
    regenerated: always yes, and nothing about whether the numbers changed. The variant
    paths are still checked for *identity*, so an arm cannot reuse another arm's map.
    """
    newest = 0.0
    for candidate in (args.stats, args.moments, args.model):
        root = Path(candidate)
        entries = sorted(root.iterdir()) if root.is_dir() else [root]
        for entry in entries:
            if entry.is_file():
                newest = max(newest, entry.stat().st_mtime)
    return newest


def _map_mismatch(
    payload: dict[str, Any],
    args: argparse.Namespace,
    *,
    keys: list[str],
    stats: str | None,
    allocator: str,
) -> str | None:
    """Why the map on disk is not the map this invocation would have written."""
    if payload.get("schema") != "dynquant_allocation_v1":
        return f"schema is {payload.get('schema')!r}"
    for field, want in (("model", args.model), ("stats", stats), ("group_size", args.group_size)):
        if payload.get(field) != want:
            return f"{field} is {payload.get(field)!r}, not {want!r}"
    if payload.get("allocator") != allocator:
        return f"allocator is {payload.get('allocator')!r}, not {allocator!r}"
    missing = [key for key in keys if key not in payload.get("maps", {})]
    if missing:
        return f"no map for {missing}"
    return None


def _reusable(
    save_map: Path,
    args: argparse.Namespace,
    *,
    keys: list[str],
    stats: str | None,
    allocator: str,
) -> dict[str, Any] | None:
    """The map already on disk, if it can be shown to postdate every input it names.

    A moments-priced map is about 1 h 45 m of CPU on an 8B model, so a run that only
    wants to quantize and evaluate should not spend seven hours rebuilding six maps it
    already has. But skipping a completed step is a resume guard, and the failure mode of
    a resume guard is not "it did not skip" -- it is that existence-on-disk cannot see
    that an artifact *predates its own input*, so a stale map gets stapled into a fresh
    run and every downstream number silently describes the wrong allocation.

    So existence is not the test. The map has to name this model, this group size, this
    stats file and this allocator -- which is what distinguishes ``rank`` from ``dq``, and
    is otherwise invisible in a finished map's bit widths -- and its mtime has to be later
    than every source file the run reads. Anything else rebuilds, and says which check
    failed rather than reporting a bare "rebuilding".
    """
    if not args.reuse_maps or not save_map.is_file():
        return None
    try:
        payload = json.loads(save_map.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  rebuilding {save_map.name}: unreadable ({exc})", flush=True)
        return None
    reason = _map_mismatch(payload, args, keys=keys, stats=stats, allocator=allocator)
    if reason is None and save_map.stat().st_mtime <= _inputs_mtime(args):
        reason = "it predates an input it summarises"
    if reason is not None:
        print(f"  rebuilding {save_map.name}: {reason}", flush=True)
        return None
    print(f"  reusing {save_map.name}", flush=True)
    return payload


def allocate_anchors(args: argparse.Namespace, work: Path) -> dict[int, Arm]:
    """Build the uniform arms, whose sizes every other arm is then held to."""
    save_map = work / "map.rtn.json"
    payload = _reusable(
        save_map,
        args,
        keys=[f"uniform-{width}" for width in ANCHORS],
        stats=None,
        allocator="rank_product",
    )
    if payload is None:
        _run(anchor_cmd(args, save_map), what="allocating the uniform anchors")
        payload = json.loads(save_map.read_text(encoding="utf-8"))
    anchors: dict[int, Arm] = {}
    for width in ANCHORS:
        key = f"uniform-{width}"
        if key not in payload["maps"]:
            raise SystemExit(
                f"the anchor allocation wrote {sorted(payload['maps'])} but not {key!r}. "
                f"Every other arm is sized against these, so a missing one is a missing "
                f"anchor rather than a missing row."
            )
        body = payload["maps"][key]
        anchors[width] = Arm(
            name=f"rtn{width}",
            anchor=width,
            kind="rtn",
            map_path=save_map,
            map_key=key,
            nbytes=int(body["nbytes"]),
            average_bits=float(body["average_bits"]),
            violations=len(body["violations"]),
        )
        print(
            f"  anchor {width}b: {anchors[width].nbytes / 1e9:.3f} GB "
            f"at {anchors[width].average_bits:.4f} stored bits/weight",
            flush=True,
        )
    return anchors


def allocate_arm(
    args: argparse.Namespace,
    work: Path,
    *,
    kind: str,
    anchor: Arm,
    variants: dict[str, dict[str, Any]],
) -> Arm:
    """Allocate one arm to the anchor's exact byte count."""
    spec = ARMS[kind]
    save_map = work / f"map.{kind}{anchor.anchor}.json"
    key = str(anchor.nbytes)

    variant = variants[spec["variant"]]
    moments = variant["moments"] if spec["moments"] else None
    payload = _reusable(
        save_map,
        args,
        keys=[key],
        stats=variant["stats"],
        allocator="sensitivity" if spec["moments"] else "rank_product",
    )
    if payload is None:
        cmd = _inspect_cmd(args, save_map)
        cmd += ["--stats", variant["stats"], "--target-size", key]
        if spec["moments"]:
            # The variant's own moments, never ``args.moments``: the sensitivity table is
            # what the knapsack prices from whenever a module has an entry, so an arm that
            # permuted the stats and then read the real moments would allocate exactly like
            # the treatment and report a null it never tested.
            cmd += ["--moments", moments]
        _run(cmd, what=f"allocating {kind}{anchor.anchor}")
        payload = json.loads(save_map.read_text(encoding="utf-8"))

    body = payload["maps"][key]
    return Arm(
        name=f"{kind}{anchor.anchor}",
        anchor=anchor.anchor,
        kind=kind,
        map_path=save_map,
        map_key=key,
        nbytes=int(body["nbytes"]),
        average_bits=float(body["average_bits"]),
        violations=len(body["violations"]),
    )


def check_matched(arms: list[Arm], anchor: Arm) -> None:
    """Refuse to publish arms whose sizes drifted apart.

    ``--target-size`` is a ceiling. An allocator that cannot spend the last few bits
    lands under it, and two arms that differ by a percent are not a comparison of
    assignments -- the larger one has a size advantage that will be read as accuracy.
    This is the guard that makes "at matched bytes" a fact about the run rather than a
    sentence in the report.
    """
    worst = max(arms, key=lambda a: abs(a.nbytes - anchor.nbytes))
    drift = abs(worst.nbytes - anchor.nbytes) / anchor.nbytes
    print(
        f"\n  anchor {anchor.anchor}b: widest drift {worst.name} "
        f"{worst.nbytes - anchor.nbytes:+d} B ({drift:.5%})",
        flush=True,
    )
    if drift > MATCH_TOLERANCE:
        raise SystemExit(
            f"arm {worst.name} is {drift:.3%} off the {anchor.anchor}-bit anchor "
            f"({worst.nbytes} vs {anchor.nbytes} bytes), over the {MATCH_TOLERANCE:.3%} "
            f"tolerance. These arms are not byte-matched and any accuracy difference "
            f"between them is confounded with size."
        )


# ---------------------------------------------------------------------------
# Materializing and evaluating
# ---------------------------------------------------------------------------


def quantize_arm(args: argparse.Namespace, arm: Arm) -> Arm:
    """Write the arm's weights, so the evaluator can load it as a plain causal LM."""
    directory = Path(args.arm_dir) / arm.name
    cmd = [
        sys.executable,
        "-m",
        "dynquant",
        "quantize",
        args.model,
        "--map",
        str(arm.map_path),
        "--map-key",
        arm.map_key,
        "--group-size",
        str(args.group_size),
        "--output",
        str(directory),
        "--json",
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    _run(cmd, what=f"quantizing {arm.name}")
    return replace(arm, directory=directory)


def evaluate(args: argparse.Namespace, arms: list[Arm], out: Path) -> None:
    """Hand the arms to the campaign's evaluator, which is not this file.

    The unquantized merge goes in as ``bf16``. Without it every arm is measured only
    against the others and the table can say which allocation is best but not what any
    of them cost -- and "quantization is free here" and "quantization destroyed this
    model equally in all four arms" have the same shape when the ceiling is missing.
    It is not an allocated arm and so does not appear in ``arms.json``; it is the same
    weights S2 produced, scored through the same evaluator.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("run_s1_headroom.py")),
        "--out",
        str(out),
        "--models",
        f"bf16={args.model}",
        *[f"{arm.label}={arm.directory}" for arm in arms],
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    print("\nEvaluate with:\n  " + " ".join(cmd), flush=True)
    if args.evaluate:
        _run(cmd, what="evaluating the arms")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="the merged fine-tune S2 produced")
    parser.add_argument("--stats", required=True, help="that arm's signal file or directory")
    parser.add_argument(
        "--moments",
        help="the channel-moment sidecar (default: dynquant_moments.safetensors beside --stats)",
    )
    parser.add_argument("--arm-dir", required=True, help="where the quantized arms are written")
    parser.add_argument("--out", required=True, help="where the records are written")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--seed", type=int, default=0, help="the shuffled control's permutation seed"
    )
    parser.add_argument("--anchors", type=int, nargs="*", default=list(ANCHORS))
    parser.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    parser.add_argument("--limit", type=int, help="passed through to the evaluator")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--keep-weights",
        action="store_true",
        help=(
            "keep each arm's directory after it is evaluated. Off by default because "
            "the quantized values are stored at compute dtype: eight arms of an 8B "
            "model is over 100 GB, and the map beside it rebuilds any of them"
        ),
    )
    parser.add_argument("--evaluate", action="store_true", help="also run the evaluator")
    parser.add_argument(
        "--reuse-maps",
        action="store_true",
        help=(
            "skip allocating any map already on disk that names this model, group size, "
            "stats file and allocator *and* is newer than all of them. Off by default: a "
            "resume guard that cannot tell a fresh map from one predating its own input "
            "is worse than the seven hours it saves"
        ),
    )
    parser.add_argument(
        "--allocate-only",
        action="store_true",
        help="stop after the maps, before any weights are written",
    )
    args = parser.parse_args(argv)

    stats_path = Path(args.stats)
    if args.moments is None:
        directory = stats_path if stats_path.is_dir() else stats_path.parent
        args.moments = str(directory / "dynquant_moments.safetensors")
    if not Path(args.moments).is_file():
        raise SystemExit(f"no channel moments at {args.moments}: the dq arm cannot be allocated")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "maps"
    work.mkdir(parents=True, exist_ok=True)

    variants = write_variants(stats_path, Path(args.moments), work, args.seed)
    print(
        f"\nshuffled control: {variants['shuffled']['moved']}/{variants['shuffled']['total']} "
        f"modules carry another module's measurements",
        flush=True,
    )

    anchors = allocate_anchors(args, work)
    built: list[Arm] = []
    for width in args.anchors:
        anchor = anchors[width]
        at_anchor = [anchor] if "rtn" in args.arms else []
        for kind in args.arms:
            if kind == "rtn":
                continue
            at_anchor.append(allocate_arm(args, work, kind=kind, anchor=anchor, variants=variants))
        check_matched(at_anchor, anchor)
        built.extend(at_anchor)

    if not args.allocate_only:
        built = [quantize_arm(args, arm) for arm in built]

    record = out / "arms.json"
    record.write_text(
        json.dumps(
            {
                "model": args.model,
                "stats": str(stats_path),
                "moments": args.moments,
                "group_size": args.group_size,
                "seed": args.seed,
                "tolerance": MATCH_TOLERANCE,
                "variants": variants,
                "arms": [
                    {
                        "name": arm.name,
                        "anchor": arm.anchor,
                        "kind": arm.kind,
                        "map": str(arm.map_path),
                        "map_key": arm.map_key,
                        "nbytes": arm.nbytes,
                        "average_bits": arm.average_bits,
                        "violations": arm.violations,
                        "directory": str(arm.directory) if arm.directory else None,
                    }
                    for arm in built
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n-> wrote {record}", flush=True)

    if args.allocate_only:
        return 0

    evaluate(args, built, out)
    if not args.keep_weights and args.evaluate:
        for arm in built:
            if arm.directory and Path(arm.directory).is_dir():
                shutil.rmtree(arm.directory)
        print(f"removed {len(built)} arm directories; --keep-weights retains them", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
