"""The seven-arm panel as one table, assembled from what the run left on disk.

Nothing here re-scores anything or recomputes a size. Every number is a number some
process already wrote down, and an arm that has not run shows up as a missing row rather
than as a gap the reader has to notice.

Where the size column comes from, and why not the record
--------------------------------------------------------

Not from the eval record, which does not carry one and should not. Four of the six
quantized arms are scored on a model ``compressed-tensors`` wrote, and the other two are
scored by encoding the allocator's widths back into bf16 weights -- so the *resident* size
of a DynQuant arm is the fp16 size, and a column filled from the loaded model would report
16 bits for the arm whose compression is the claim.

The honest source is ``arms.json``: for a baseline the size is its format's own accounting
at that width, and for a DynQuant arm it is the byte count the allocator realised and
``check_matched`` already held against the baselines' anchor. This script re-states that
drift rather than trusting that the check ran, because a manifest can be assembled by hand
and a table that only prints matched sizes is not the same as one that checks them.

The fp16 row is the single derived number. It is ``params * 2`` with ``params`` read from a
baseline's ``.quant.json`` side file -- the count that arm was itself sized against, so the
ceiling and the arms beneath it count the same tensors and the ratio between them is real.
No literal parameter count appears anywhere here: one written for this model would be
silently wrong for the next one.

Two families, corrected separately
----------------------------------

Twelve comparisons at alpha=0.05 expect half a false positive, and the headline of this
panel is one of the twelve. So each block carries a Holm-adjusted p and the verdict follows
the adjusted one.

The split into two blocks is not a way of shrinking the multiplier on the claim. The blocks
answer different kinds of question. "Does DynQuant beat GPTQ at these bytes" is a hypothesis
test and belongs in a corrected family. "What did quantizing to 4 bits cost" is a
measurement whose answer is the interval; nobody doubts the sign. Both are corrected anyway,
each within its own block, and the block sizes are printed so a reader who disagrees with
the split can multiply by twelve instead.

Where the per-source block sits
-------------------------------

``--sources`` turns the head-to-head family into a per-dataset decomposition. It is a
decomposition and not a second experiment: the same twelve hits are being re-partitioned,
so a method that wins on one source and loses on the other has not thereby produced two
findings to choose between. It is here because the aggregate raises the question and
cannot answer it -- the records store 12,000 hits in one order and a per-source total,
and nothing that says which item is which.

The labels are therefore checked, not trusted. A source vector is only accepted if, for
every arm, the hits at the positions it calls a given source sum to exactly the
``by_source`` count that arm wrote during its own run. Eight integer agreements on this
panel, and a shuffle reconstructed from the wrong seed would have to reproduce all eight.

Run::

    python experiments/phase4/panel_table.py --arms runs/s4/arms
    python experiments/phase4/panel_table.py --arms runs/s4/arms --json > table.json
    python experiments/phase4/panel_table.py --arms runs/s4/arms --sources runs/s4/arms/sources.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

# The same bootstrap `dispatch_delta.py` and `probe_dispatch_agreement.py` carry, and
# for the same reason: nothing installs `dynquant` on the box that runs these, so a
# script that imports core and inserts nothing works only where something else already
# put the source on the path. Under pytest that something is conftest, which is why the
# suite could not have caught this. The imports here are inside functions, so the
# failure waits until the table is asked for real records -- `--help` succeeds.
CORE_SRC = Path(__file__).resolve().parents[2] / "packages" / "dynquant-core" / "src"
if str(CORE_SRC) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(CORE_SRC))

#: Head-to-head at matched bytes. This is the claim, and the family it is corrected in.
HEAD_TO_HEAD: tuple[tuple[str, str, str], ...] = (
    ("dq_4b", "gptq_4b", "4b  DynQuant vs GPTQ"),
    ("dq_4b", "awq_4b", "4b  DynQuant vs AWQ"),
    ("gptq_4b", "awq_4b", "4b  GPTQ vs AWQ"),
    ("dq_3b", "gptq_3b", "3b  DynQuant vs GPTQ"),
    ("dq_3b", "awq_3b", "3b  DynQuant vs AWQ"),
    ("gptq_3b", "awq_3b", "3b  GPTQ vs AWQ"),
)

#: What each method cost against the unquantized model it was built from.
AGAINST_CEILING: tuple[tuple[str, str, str], ...] = (
    ("gptq_4b", "bf16", "4b  GPTQ vs bf16"),
    ("awq_4b", "bf16", "4b  AWQ vs bf16"),
    ("dq_4b", "bf16", "4b  DynQuant vs bf16"),
    ("gptq_3b", "bf16", "3b  GPTQ vs bf16"),
    ("awq_3b", "bf16", "3b  AWQ vs bf16"),
    ("dq_3b", "bf16", "3b  DynQuant vs bf16"),
)

#: The arm the fidelity block asks about agreement with: the unquantized model every
#: quantized arm in the panel was built from.
CEILING = "bf16"

FP16_BYTES_PER_PARAM = 2


def compact(count: int) -> str:
    """A parameter count at a width a reader can compare across rows.

    Used for the breached-floor mass, which is the one place here that counts parameters
    and where they span four orders of magnitude between roles -- fixed G units printed a
    real million-parameter role as ``0.00G``, a mass the budget did take, rendered as
    though it had taken nothing. The width histogram counts *modules* and is printed as an
    integer; the two are not interchangeable and reading one as the other is what this
    formatter was originally, wrongly, applied to.
    """
    if count >= 1_000_000_000:
        return f"{count / 1e9:.2f}G"
    if count >= 1_000_000:
        return f"{count / 1e6:.0f}M"
    return f"{count / 1e3:.0f}K"


def standard_error(accuracy: float, total: int) -> float:
    """One binomial SE, in percentage points."""
    return math.sqrt(accuracy * (1.0 - accuracy) / total) * 100.0 if total else 0.0


def chi_square_sf(statistic: float, df: int) -> float:
    """Upper tail of a chi-square, without pulling scipy in for one number.

    This file's neighbours already refuse that dependency -- ``compare_paired`` sums an
    exact binomial tail with :func:`math.comb` rather than calling a library -- and the
    reason holds here: the script runs on whatever interpreter is beside a results
    directory, including one on a box that has torch and nothing else.

    Regularised incomplete gamma by series below the transition and continued fraction
    above it, which is the standard split and is accurate to well past the precision any
    p-value here is quoted at. ``df`` is small by construction: it is one less than the
    number of datasets in the eval mixture.
    """
    if df < 1:
        raise ValueError(f"df must be at least 1, got {df}")
    if statistic <= 0:
        return 1.0
    a, x = df / 2.0, statistic / 2.0
    log_gamma = math.lgamma(a)
    if x < a + 1.0:
        # Series for the *lower* tail, subtracted. Converges fast when x is left of the peak.
        term = total = 1.0 / a
        for i in range(1, 1000):
            term *= x / (a + i)
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - log_gamma))
    # Continued fraction for the upper tail directly (Lentz), which is the stable side here.
    tiny = 1e-300
    b, c, d = x + 1.0 - a, 1.0 / tiny, 1.0 / (x + 1.0 - a)
    result = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return min(1.0, max(0.0, math.exp(-x + a * math.log(x) - log_gamma) * result))


def cochran_q(deltas: list[float], errors: list[float]) -> tuple[float, float, float]:
    """Heterogeneity across independent estimates of the same quantity.

    Returns ``(pooled, q, p)``. The pooled estimate is inverse-variance weighted, which is
    *not* the same as the whole-panel delta and is not offered as a replacement for it --
    the panel's own number weights items equally, this one weights sources by precision.
    It is here because Q is a spread around a centre and the centre has to be named.

    Independence is the caller's to guarantee and is real here: the subsets partition the
    eval set, so no item contributes to two of them. Running this over overlapping slices
    would understate Q, and understating Q is the direction that manufactures agreement.
    """
    if len(deltas) != len(errors):
        raise ValueError("deltas and errors must be the same length")
    if len(deltas) < 2:
        raise ValueError("heterogeneity needs at least two estimates")
    if any(error <= 0 for error in errors):
        raise ValueError("a zero standard error would carry infinite weight")
    weights = [1.0 / error**2 for error in errors]
    pooled = sum(w * d for w, d in zip(weights, deltas, strict=True)) / sum(weights)
    q = sum(w * (d - pooled) ** 2 for w, d in zip(weights, deltas, strict=True))
    return pooled, q, chi_square_sf(q, len(deltas) - 1)


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the order given.

    Step-down rather than plain Bonferroni: it controls the same family-wise error rate
    and is uniformly more powerful, so using Bonferroni here would be discarding real
    findings for no gain in rigour. Monotonicity is enforced on the way up, which is what
    makes the adjusted values interpretable as p-values rather than as a sorted list of
    multiplied numbers.
    """
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def load_panel(out: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The manifest plus every record it names that exists."""
    manifest_path = out / "arms.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"{manifest_path} does not exist. The table is assembled from the manifest the "
            f"run writes, not from whatever json files are in the directory -- a stray "
            f"record from another run would otherwise enter the panel as an arm."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: dict[str, dict[str, Any]] = {}
    for arm in manifest["arms"]:
        found = resolve_stored(out, arm.get("record"))
        if found is not None:
            records[arm["label"]] = json.loads(found.read_text(encoding="utf-8"))
    return manifest, records


def resolve_stored(out: Path, stored: str | None) -> Path | None:
    """A path the manifest names, found from beside the manifest if need be.

    The driver writes ``str(out / ...)``, so a run launched with a relative ``--out`` stores
    relative paths -- which resolve against whatever directory the *table* is later run from,
    not the one the panel ran in. Read literally, a manifest moved off the box, or simply read
    from a different cwd, is a panel in which no arm was scored: seven arms, seven silent
    misses, and a table that prints ``0/7`` for a run that finished.

    So a stored path that is not a file is retried beside the manifest. That cannot pick up a
    foreign file, because the writer's own invariant is that these artifacts sit in ``out`` --
    the same fact the retry uses.

    Every stored path goes through here, and the manifest names three kinds: the record, the
    saved bit map, and the ``.quant.json`` side file derived from the record's name. Fixing
    only the record is the version of this bug that is worse than not fixing it at all -- the
    arms score, so the table looks whole, while the fp16 ceiling silently loses its parameter
    count and every DynQuant arm silently loses its allocation. Both of those print as absence,
    and absence of a floor breach is exactly what §12 pre-registered 4 bits to show.

    The retry is the *longest* tail of the stored path that exists under ``out``, not the bare
    filename, and the difference is not academic: records are written to ``out/<label>.json``
    and maps to ``out/maps/<label>.json``, so a map retried by filename alone lands on the
    record of the same arm. That file parses, carries no ``maps`` key, and yields no
    allocation -- a wrong file read successfully, reported as a missing allocation. Longest
    tail first means the more specific location always wins, so ``maps/dq_4b.json`` can only
    resolve to a map.
    """
    if not stored:
        return None
    direct = Path(stored)
    if direct.is_file():
        return direct
    parts = direct.parts
    for start in range(len(parts)):
        candidate = out.joinpath(*parts[start:])
        if candidate.is_file():
            return candidate
    return None


def check_pairable(records: dict[str, dict[str, Any]]) -> str | None:
    """Whether the directory is homogeneous, as a banner over the whole table.

    ``arms_lfm2 run`` already checks this as each arm lands. Checked again because this
    script can be pointed at a directory the run did not assemble -- a resumed panel, a
    hand-merged one -- and pairing two hit vectors that describe different problem sets
    produces a number rather than an error.

    Advisory now, and that is the change. It used to gate every row in every block: one
    arm disagreeing with the ceiling blanked `GPTQ vs AWQ` too, a comparison neither the
    stale arm nor the ceiling is part of. A panel is a set of pairs and comparability is a
    property of a pair, so `print_comparisons` decides row by row and this says only that
    somewhere in the directory two arms do not agree.
    """
    from dynquant.commands.evaluate import problem_set_difference

    if not records:
        return None
    reference = "bf16" if "bf16" in records else next(iter(records))
    for label, record in records.items():
        differed = problem_set_difference(records[reference], record)
        if differed:
            return (
                f"{label} was not scored under the same settings as {reference}: {differed} "
                f"as ({reference}, {label}). Their hit vectors are not paired."
            )
    return None


def infer_params(out: Path, manifest: dict[str, Any]) -> int | None:
    """The parameter count the baselines sized themselves against.

    Read from a ``.quant.json`` side file rather than counted here, so the fp16 row and
    the quantized rows are denominated in the same tensors. Returns ``None`` when no
    baseline arm has run, and the fp16 size column then says so instead of guessing.
    """
    for arm in manifest["arms"]:
        if arm.get("kind") not in ("gptq", "awq"):
            continue
        record = resolve_stored(out, arm.get("record"))
        if record is None:
            continue
        side = record.with_suffix(".quant.json")
        if side.is_file():
            payload = json.loads(side.read_text(encoding="utf-8"))
            if payload.get("params"):
                return int(payload["params"])
    return None


def linearization_of(out: Path, arm: dict[str, Any]) -> dict[str, Any] | None:
    """What ``llm-compressor`` did to this arm's expert banks, per the run that did it.

    From the ``.quant.json`` side file that `baselines_lfm2.score` writes beside the eval
    record. The two describe one object: `quantize` linearises, calibrates and returns the
    model *in memory*, and `score` hands that same object to `evaluate.run` without a save
    or a reload. So ``banks_after`` is not a claim about a checkpoint the scorer later
    opened -- it is a count taken in the same process on the weights that were then scored.

    Returns ``None`` for an arm with no such file, which is every arm this repository
    quantizes itself. That absence is the point: see `dispatch_of`.
    """
    record = resolve_stored(out, arm.get("record"))
    if record is None:
        return None
    side = record.with_suffix(".quant.json")
    if not side.is_file():
        return None
    payload = json.loads(side.read_text(encoding="utf-8"))
    report = payload.get("linearization")
    return report if isinstance(report, dict) else None


def allocation_of(out: Path, arm: dict[str, Any]) -> dict[str, Any] | None:
    """The allocator's own account of a DynQuant arm: widths and breached floors.

    The floor violations are the part that cannot be recovered later. Two arms can land on
    the same average bits with and without a breached expert bank, and only one of them is
    a knapsack result -- so an allocation that reports zero breaches at 4 bits and a
    breached ``gate_up`` at 3 is the pre-registered prediction being confirmed or not.
    """
    path, key = resolve_stored(out, arm.get("map")), str(arm.get("target_bytes"))
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload.get("maps", {}).get(key)
    if entry is None:
        return None
    return {
        "average_bits": entry.get("average_bits"),
        "nbytes": entry.get("nbytes"),
        "histogram": entry.get("histogram", {}),
        "violations": entry.get("violations", []),
        # Carried through rather than summarised here, because "which price chose these
        # widths" is the question the rest of this dict cannot answer. Absent on a map
        # written before the field existed, which reads correctly as "unknown".
        "pricing": entry.get("pricing"),
    }


def rows(
    out: Path,
    manifest: dict[str, Any],
    records: dict[str, dict[str, Any]],
    params: int | None,
) -> list[dict[str, Any]]:
    """One row per planned arm, in the order the panel planned them."""
    built = []
    for arm in manifest["arms"]:
        label = arm["label"]
        record = records.get(label)
        nbytes = arm.get("nbytes")
        if nbytes is None and arm.get("kind") == "ceiling" and params:
            nbytes = params * FP16_BYTES_PER_PARAM
        detail = (record or {}).get("detail") or {}
        target = arm.get("target_bytes")
        built.append(
            {
                "label": label,
                "kind": arm.get("kind"),
                "anchor": arm.get("anchor"),
                "nbytes": nbytes,
                "target_bytes": target,
                "drift": (nbytes - target) / target if (nbytes and target) else None,
                "bits_per_param": (nbytes * 8 / params) if (nbytes and params) else None,
                "accuracy": (record or {}).get("accuracy"),
                "correct": (record or {}).get("correct"),
                "total": (record or {}).get("total"),
                "unparseable": (record or {}).get("unparseable"),
                "errored": detail.get("errored"),
                "exact": detail.get("exact"),
                "unfinished": detail.get("unfinished_reasoning"),
                "by_source": detail.get("by_source") or {},
                "apply": ((record or {}).get("packed") or {}).get("apply"),
                # Two fields for one value because a missing key and a `null` are
                # different facts and `.get` cannot tell them apart -- which is exactly
                # the distinction `_comparability` also cannot make. See `dispatch_of`.
                "experts": (record or {}).get("experts"),
                "experts_recorded": record is not None and "experts" in record,
                # What the quantizer did to this arm's banks, from the record it wrote in
                # the same process. Recovers the dispatch when the eval record cannot.
                "linearization": linearization_of(out, arm),
                "allocation": allocation_of(out, arm),
                "seconds": (record or {}).get("seconds"),
            }
        )
    return built


def print_sizes(built: list[dict[str, Any]], params: int | None, tolerance: float) -> None:
    header = (
        f"{'arm':10s} {'method':8s} {'GiB':>8s} {'b/param':>8s} {'off anchor':>12s} "
        f"{'vs bf16':>8s} {'apply':>7s}"
    )
    print(header)
    print("-" * len(header))
    ceiling = next((row["nbytes"] for row in built if row["kind"] == "ceiling"), None)
    for row in built:
        nbytes = row["nbytes"]
        gib = f"{nbytes / 2**30:.3f}" if nbytes else "--"
        bpp = f"{row['bits_per_param']:.4f}" if row["bits_per_param"] else "--"
        if row["drift"] is None:
            drift = "--"
        else:
            drift = f"{row['drift'] * 100:+.4f}%" + ("!" if abs(row["drift"]) > tolerance else "")
        ratio = f"{ceiling / nbytes:.2f}x" if (ceiling and nbytes) else "--"
        kind = row["kind"] or "--"
        apply_mode = row["apply"] or "--"
        print(
            f"{row['label']:10s} {kind:8s} {gib:>8s} {bpp:>8s} {drift:>12s} "
            f"{ratio:>8s} {apply_mode:>7s}"
        )
    if params:
        print(f"  denominated in {params:,} parameters, from a baseline's own accounting")
    else:
        print("  no baseline .quant.json found: the bf16 size and b/param columns are unavailable")
    if any(row["drift"] is not None and abs(row["drift"]) > tolerance for row in built):
        raise SystemExit(
            f"an arm marked ! is further than {tolerance:.3%} from its anchor, so the panel "
            f"is not byte-matched and every accuracy difference in it is confounded with "
            f"size. The rest of the table is not printed for a panel that cannot support one."
        )


def print_accuracy(built: list[dict[str, Any]], chance: float | None) -> None:
    header = (
        f"{'arm':10s} {'exec match':>11s} {'+-1SE':>6s} {'correct':>11s} "
        f"{'exact':>7s} {'no query':>9s} {'sql error':>10s} {'unfinished':>11s} {'min':>6s}"
    )
    print(header)
    print("-" * len(header))
    for row in built:
        if row["accuracy"] is None:
            print(f"{row['label']:10s} {'not run':>11s}")
            continue
        se = standard_error(row["accuracy"], row["total"] or 0)
        minutes = f"{row['seconds'] / 60:.0f}" if row["seconds"] else "--"
        unfinished = row["unfinished"]
        counted = f"{row['correct']}/{row['total']}"
        # A ! rather than a footnote: a non-zero count here caps the headline above it,
        # and that is a fact about the row it sits on, not about the table.
        flagged = f"{unfinished if unfinished is not None else '--'}{'!' if unfinished else ''}"
        exact, no_query, errored = (
            f"{row[field] if row[field] is not None else '--'}"
            for field in ("exact", "unparseable", "errored")
        )
        print(
            f"{row['label']:10s} {row['accuracy'] * 100:10.2f}% {se:6.2f} {counted:>11s} "
            f"{exact:>7s} {no_query:>9s} {errored:>10s} {flagged:>11s} {minutes:>6s}"
        )
    if chance:
        print(f"{'(guessing)':10s} {chance * 100:10.2f}%")


def dispatch_of(row: dict[str, Any]) -> tuple[str, str]:
    """Which arithmetic this arm ran, from its record and from the record beside it.

    `_pin_experts_dispatch` writes one of exactly two things, so an eval record is in one
    of exactly three states. A model whose config carries `_experts_implementation` gets
    `{found, ran}` -- what the dispatch was on arrival and what it was when the scorer
    ran. A model whose config does not gets `null`, which is a dense model and not a
    linearised one: `linearize_moe` rewrites modules and leaves the config alone, so a
    baseline records `{grouped_mm, eager}` like everyone else. A record written before the
    field existed has no key at all.

    The last two are the same absence to `_comparability`, which reads
    `record.get("experts")` and treats a non-dict as `_ABSENT`. That is deliberate for
    `null` -- a dense model has no dispatch and never will, so absence has to pair with
    absence -- and it takes a missing key along with it. So "we know this model had
    nothing to dispatch" and "we do not know what this ran" pair with each other and the
    guard stays quiet, which is the one straddle a panel can make without being told.

    But a missing field is not always a missing fact. An arm quantized by `baselines_lfm2`
    has a `.quant.json` beside it reporting `banks_after: 0`, counted in the process that
    then scored that same object, and a model with no batched bank has no grouped kernel
    to take: the arithmetic is the loop, whatever the config says. That is recovered rather
    than recorded, and the difference is not pedantic -- the guard still refuses it,
    because a record that cannot state what it ran is not certified by a neighbour that
    can. What it changes is which arms are genuinely unknown, and that set turns out to be
    exactly the arms worth re-scoring.

    Returns the text and one of `recorded`, `dense`, `recovered`, `unknown`.
    """
    if row["experts_recorded"]:
        experts = row["experts"]
        if not isinstance(experts, dict):
            return "none (dense)", "dense"
        return f"{experts.get('ran')} (from {experts.get('found')})", "recorded"
    report = row["linearization"]
    if isinstance(report, dict) and report.get("banks_after") == 0:
        return f"loop ({report.get('banks_before')} banks -> 0)", "recovered"
    return "not recorded", "unknown"


def arithmetic_of(row: dict[str, Any]) -> str | None:
    """Which *class* of expert arithmetic this arm ran, or `None` if nothing says.

    Coarser than `dispatch_of` on purpose. The panel's comparisons do not care whether a
    bank was indexed by `eager` or by a `ModuleList` that `linearize_moe` produced -- both
    take one expert at a time -- they care that neither is `grouped_mm`, which batches and
    which disagrees with the indexed path on 1.24% of teacher-forced tokens on this model.
    So `eager` and the linearised loop collapse to `indexed`.

    That collapse rests on a measurement, and on a small one: same four-layer model, one
    side on `eager` and one through `linearize_moe`, output bitwise identical. Bitwise is
    a strong result at any scale -- there is no numeric difference for a top-k router to
    turn discrete -- but it is a CPU fp32 model, and the whole reason this function exists
    is that a tiny-scale agreement between dispatches did not survive to 8B once. Treat
    `indexed` as one class and the 8B check as still owed.
    """
    _, state = dispatch_of(row)
    if state == "dense":
        return "dense"
    if state == "recovered":
        return "indexed"
    if state == "recorded":
        ran = (row["experts"] or {}).get("ran")
        return {"eager": "indexed", "grouped_mm": "grouped"}.get(ran, ran)
    return None


def print_dispatch(built: list[dict[str, Any]]) -> None:
    """Which arithmetic each arm ran, per its own record.

    On LFM2.5-8B-A1B the two experts dispatches disagree on 1.24% of teacher-forced
    tokens -- 0.29x the effect quantization itself has -- so this is not a provenance
    note, it is a column on the panel's main axis.
    """
    scored = [row for row in built if row["accuracy"] is not None]
    if not scored:
        return
    header = f"{'arm':10s} {'experts dispatch':>26s}"
    print(header)
    print("-" * len(header))
    recovered, silent, eager = [], [], []
    for row in scored:
        text, state = dispatch_of(row)
        if state == "recovered":
            recovered.append(row["label"])
        elif state == "unknown":
            silent.append(row["label"])
        elif state == "recorded" and (row["experts"] or {}).get("ran") == "eager":
            eager.append(row["label"])
        print(f"{row['label']:10s} {text:>26s}")
    if recovered:
        print()
        print(
            f"  {len(recovered)} arm(s) carry no dispatch field but were linearised to "
            f"zero banks: {', '.join(recovered)}."
        )
        print(
            "  That count was taken in the process that went on to score the same object, "
            "so the arithmetic is"
        )
        print(
            "  the loop and nothing had to be assumed about a checkpoint. Recovered, not "
            "recorded -- what that"
        )
        print(
            "  costs is this line, not the comparison: a dispatch difference is priced on "
            "the row, not refused."
        )
    if silent:
        print()
        print(
            f"  {len(silent)} arm(s) carry no dispatch field and no linearization report: "
            f"{', '.join(silent)}."
        )
        print(
            "  Nothing written down says what these ran, and to `_comparability` the "
            "absence is the same one a"
        )
        print("  dense model's `null` makes, so nothing flags them. They are the re-score set.")
    if recovered and eager:
        # The one claim the panel makes that no measurement on this model supports. It
        # only becomes load-bearing once both buckets are occupied, which is precisely
        # the state the eager re-score produces -- so the note appears exactly when the
        # table stops showing any other caveat, and does not nag before then.
        print()
        print(
            f"  {len(eager)} arm(s) ran `eager` and {len(recovered)} ran the linearised "
            f"loop. The panel treats those"
        )
        print(
            "  as one class -- both index one expert at a time -- on a four-layer CPU "
            "fp32 model where the two"
        )
        print(
            "  were bitwise identical. Bitwise is strong, but section 8 of the report is "
            "an agreement at small"
        )
        print(
            "  scale that did not survive to 8B. Nothing below is marked for dispatch; "
            "that rests on this."
        )


def print_by_source(built: list[dict[str, Any]]) -> None:
    """Per-source accuracy, because one number over three datasets can hide a collapse.

    A method that damages one source's distribution and leaves the others alone moves the
    headline by a couple of points and moves one column here by twenty. The mixture exists
    so that is visible; printing only the mixture would waste it.
    """
    sources = sorted({name for row in built for name in row["by_source"]})
    if not sources:
        return
    header = f"{'arm':10s}" + "".join(f"{name:>18s}" for name in sources)
    print(header)
    print("-" * len(header))
    for row in built:
        if not row["by_source"]:
            continue
        cells = ""
        for name in sources:
            pair = row["by_source"].get(name)
            if not pair or not pair[1]:
                cells += f"{'--':>18s}"
            else:
                correct, total = pair
                cells += f"{f'{correct / total * 100:.1f}% ({total})':>18s}"
        print(f"{row['label']:10s}{cells}")


def describe_pricing(pricing: dict[str, Any]) -> str:
    """One line saying how much of the map the measured signal actually decided.

    In parameters first and modules second, because the two disagree by an order of
    magnitude on a MoE and the count is the flattering one: 44 of 133 modules sounds
    like an edge case and is 91.5% of the checkpoint. A missing scale is not a smaller
    number -- it is the case where the proxy-priced modules have no calibrated
    relationship to the measured ones at all, so their order among themselves is
    whatever the score said and their order against the rest is arbitrary. That
    deserves the word, not a blank.
    """
    proxied = int(pricing.get("proxied_modules", 0))
    measured = int(pricing.get("measured_modules", 0))
    if not proxied:
        return f"all {measured} modules from measured sensitivity"
    share = float(pricing.get("proxied_share", 0.0)) * 100
    if not measured:
        return f"all {proxied} modules from the score proxy -- nothing was measured"
    scale = pricing.get("scale")
    how = (
        f"rescaled by {scale:.3e}"
        if scale is not None
        else "NO COMMON SCALE, their order is arbitrary"
    )
    return (
        f"{measured} modules measured, {proxied} from the score proxy "
        f"= {share:.1f}% of parameters ({how})"
    )


def print_allocation(built: list[dict[str, Any]]) -> None:
    """What the allocator did, for the arms that had one.

    A DynQuant arm's average bits and its breached floors are the only evidence of whether
    the budget was still binding. Zero breaches means the floors fitted and the allocation
    was a knapsack over the slack; a breach names the role the budget could not afford.
    Both were predicted before either arm ran, and neither can be recovered from the
    accuracy afterwards.
    """
    for row in built:
        report = row["allocation"]
        if not report:
            continue
        # Modules, not parameters. The saved map's histogram counts tensors -- checked
        # against a real one rather than assumed -- and this line said "params" and ran the
        # counts through a billions-scale formatter, so a 187-module width printed as
        # `0K`: the allocator's whole answer, rendered as though it had assigned nothing.
        # The parameter mass is not in the map, but it is in `violations`, which is where
        # the question that needs it is actually asked.
        widths = "  ".join(
            f"{width}b {int(modules)}"
            for width, modules in sorted(report["histogram"].items(), key=lambda kv: int(kv[0]))
        )
        total = sum(int(modules) for modules in report["histogram"].values())
        print(f"{row['label']}: {report['average_bits']:.4f} avg bits over the quantized set")
        print(f"  widths, modules at each: {widths}   ({total} quantized)")
        # Which of the two prices chose those widths, before the widths are discussed.
        # On this architecture the Gauss-Newton estimate does not exist for a batched
        # expert bank -- the bank's forward spans two matmuls and a non-linearity, so
        # no module boundary gives the pairing the Kronecker form needs -- and the
        # banks are 91.5% of the parameters. Reading the DynQuant arms as "measured
        # sensitivity decided this" without that line is reading them wrong.
        pricing = report.get("pricing")
        if pricing:
            print(f"  priced: {describe_pricing(pricing)}")
        breaches = report["violations"]
        if not breaches:
            print("  floors: none breached -- the budget was not binding on any role")
            continue
        by_role: dict[str, list[int]] = {}
        for breach in breaches:
            by_role.setdefault(str(breach["role"]), []).append(int(breach["num_params"]))
        print(f"  floors: {len(breaches)} breached")
        for role, counts in sorted(by_role.items()):
            print(f"    {role:16s} {len(counts):3d} tensors  {compact(sum(counts))} params")


def print_comparisons(
    title: str,
    family: tuple[tuple[str, str, str], ...],
    records: dict[str, dict[str, Any]],
    arithmetic: dict[str, str | None],
    *,
    explain_arithmetic: bool = True,
) -> list[dict[str, Any]]:
    from dynquant.commands.evaluate import problem_set_difference
    from dynquant.eval.compare import compare_paired

    print(title)
    header = (
        f"{'comparison':28s} {'delta':>7s} {'95% CI':>18s} {'flips':>11s} "
        f"{'p':>10s} {'p (Holm)':>10s}  verdict"
    )
    print(header)
    print("-" * len(header))

    # Classify first and print afterwards, so the block reads in the family's declared
    # order on a partial panel. Printing skips inside the loop and computed rows after it
    # put the one comparison a half-finished run *can* make at the bottom, under five
    # placeholders -- readable only once nothing is missing, which is exactly when the
    # ordering stops mattering.
    computed: list[dict[str, Any]] = []
    rows: list[dict[str, Any] | str] = []
    for left, right, question in family:
        if left not in records or right not in records:
            rows.append(f"{question:28s} (needs both arms)")
            continue
        a, b = records[left], records[right]
        # These two against each other. The fields are named in the row because the reader
        # has to act on them -- "not comparable" alone sends someone to diff two 120 KB
        # records to learn that one of them was scored at a different `--limit`.
        differed = problem_set_difference(a, b)
        if differed:
            rows.append(f"{question:28s} (not comparable: {', '.join(sorted(differed))})")
            continue
        if not a.get("hits") or not b.get("hits"):
            rows.append(f"{question:28s} (no per-item hits recorded)")
            continue
        paired = compare_paired(a["hits"], b["hits"], label_a=left, label_b=right)
        entry = {
            "left": left,
            "right": right,
            "question": question,
            "paired": paired,
            # Flagged when the two arms did not demonstrably run the same class of expert
            # arithmetic -- including when one of them simply does not say. An unknown is
            # not a pass: the confound this guards against is 0.29x the effect being
            # measured, so "probably the same" is not a basis for a verdict.
            "same_arithmetic": (
                arithmetic.get(left) is not None and arithmetic.get(left) == arithmetic.get(right)
            ),
        }
        computed.append(entry)
        rows.append(entry)

    adjusted = holm([entry["paired"].p_value for entry in computed])
    for entry, p_adj in zip(computed, adjusted, strict=True):
        entry["p_adjusted"] = p_adj
        entry["separated"] = p_adj < 0.05
    for row in rows:
        if isinstance(row, str):
            print(row)
            continue
        paired = row["paired"]
        low, high = paired.interval_points
        print(
            f"{row['question']:28s} {paired.delta_points:+6.2f} "
            f"{f'[{low:+.2f}, {high:+.2f}]':>18s} "
            f"{f'{paired.a_only}/{paired.b_only}':>11s} "
            f"{paired.p_value:10.3g} {row['p_adjusted']:10.3g}  "
            f"{'separated' if row['separated'] else 'NOT separated'}"
            f"{'' if row['same_arithmetic'] else '  !'}"
        )
    mixed = [entry for entry in computed if not entry["same_arithmetic"]]
    if mixed and not explain_arithmetic:
        # A per-source block re-partitions the hits the block above already tested, so its
        # flagged set is the same set. The marker still has to appear on the row -- a
        # reader scanning one block should not have to know that -- but the four lines of
        # explanation would be the third copy on one screen.
        print()
        print(
            f"  ! {len(mixed)} comparison(s) mix expert arithmetic, the same ones flagged "
            f"in the block above."
        )
    elif mixed:
        print()
        print(
            f"  ! {len(mixed)} comparison(s) pair arms that are not known to have run the "
            f"same expert arithmetic:"
        )
        for entry in mixed:
            left, right = entry["left"], entry["right"]
            print(
                f"      {entry['question'].strip():26s} "
                f"{left} = {arithmetic.get(left) or 'unrecorded'}, "
                f"{right} = {arithmetic.get(right) or 'unrecorded'}"
            )
        print(
            "    On this model the grouped and indexed dispatches disagree on 1.24% of "
            "teacher-forced tokens,"
        )
        print(
            "    0.29x what quantizing to 4 bits moves, so a flagged delta carries a term "
            "that is not the"
        )
        print(
            "    method. The verdict is what the hits say; it is not yet a statement about "
            "quantization."
        )
    if computed:
        # Say when the family is short. Holm's multiplier is the number of comparisons it
        # actually corrected, so a block read at four of six arms is corrected *less* than
        # the same block will be at seven -- an adjusted p quoted from a running panel can
        # only move one way when the panel finishes, and it is the unfavourable way. The
        # count alone does not carry that; a reader has to know the family size to notice
        # the difference, and the reader here is someone glancing at a table mid-run.
        short = len(computed) < len(family)
        print(
            f"  Holm-adjusted over {len(computed)} of {len(family)} comparisons in this"
            + (
                " block -- a short family, so these adjusted p are weaker than the finished panel's"
                if short
                else " block"
            )
        )
    return computed


def load_sources(
    out: Path, given: str | None, records: dict[str, dict[str, Any]]
) -> tuple[list[str] | None, str | None]:
    """Per-item source labels, or a reason there are none.

    Verified against the records rather than accepted: every arm wrote a per-source
    correct/total during its own run, and a label vector that disagrees with any of them
    is the wrong vector. Returning the reason instead of raising keeps a bad or missing
    ``sources.json`` from taking the rest of the table down with it -- the aggregate rows
    are what the panel is for, and they do not need this.
    """
    path = Path(given) if given else out / "sources.json"
    if not path.exists():
        return None, None if given is None else f"{path} does not exist"
    try:
        labels = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"{path}: {exc}"
    if not isinstance(labels, list) or not all(isinstance(name, str) for name in labels):
        return None, f"{path}: expected a json list of strings"

    checked = 0
    for label, record in sorted(records.items()):
        hits = record.get("hits")
        stored = (record.get("detail") or {}).get("by_source")
        if not hits or not stored:
            continue
        if len(hits) != len(labels):
            return None, f"{label} has {len(hits)} hits against {len(labels)} labels"
        for name, pair in sorted(stored.items()):
            correct, total = int(pair[0]), int(pair[1])
            mine = sum(1 for hit, src in zip(hits, labels, strict=True) if src == name and hit)
            count = sum(1 for src in labels if src == name)
            if (mine, count) != (correct, total):
                return None, (
                    f"{label}/{name}: labels say {mine}/{count}, the record says {correct}/{total}"
                )
            checked += 1
    if not checked:
        return None, "no arm records both hits and a per-source breakdown to check against"
    return labels, None


def restrict(
    records: dict[str, dict[str, Any]], labels: list[str], name: str
) -> dict[str, dict[str, Any]]:
    """The same records with every hit vector cut down to one source's items."""
    subset: dict[str, dict[str, Any]] = {}
    for label, record in records.items():
        hits = record.get("hits")
        if not hits or len(hits) != len(labels):
            continue
        kept = [hit for hit, src in zip(hits, labels, strict=True) if src == name]
        subset[label] = {**record, "hits": kept}
    return subset


#: Why the accuracy blocks are worth printing per source.
ACCURACY_BY_SOURCE_NOTE = (
    "  The per-source blocks re-partition the same hits the block above tests, so a "
    "method\n  ahead on one source and behind on the other has produced one result "
    "with structure,\n  not two results to choose between. Each block is Holm-"
    "corrected within itself."
)

#: Why the same partition is worth running again on the fidelity indicator. Not a
#: restatement: these two blocks answer different questions and can disagree, and it is
#: the disagreement that carries the finding.
FIDELITY_BY_SOURCE_NOTE = (
    "  Accuracy and fidelity re-partitioned the same way answer different questions. A "
    "point\n  of agreement with the ceiling moves accuracy by 2c-1, and c is not the same "
    "on both\n  sources here, so an accuracy margin can vary by source while the method's "
    "own\n  tracking of the ceiling does not. When that happens the spread above is the "
    "ceiling's\n  arithmetic rather than the method's behaviour, and only this block can "
    "tell them apart."
)


def print_source_blocks(
    labels: list[str] | None,
    why_not: str | None,
    records: dict[str, dict[str, Any]],
    arithmetic: dict[str, str | None],
    *,
    question: str = "head to head",
    closing: str = ACCURACY_BY_SOURCE_NOTE,
) -> dict[str, list[dict[str, Any]]]:
    """One `print_comparisons` block per source, over whatever indicator ``records`` carry.

    Parameterised on the question rather than duplicated because the partition, the family
    and the within-block Holm correction are the same work either way -- a second copy of
    this loop for the fidelity indicator would be a second place for the restriction to be
    got wrong, and the two would agree right up until one of them was edited.
    """
    if labels is None:
        if why_not:
            print(f"per-source {question}: unavailable -- {why_not}")
            print()
        return {}
    blocks: dict[str, list[dict[str, Any]]] = {}
    for name in sorted(set(labels)):
        count = sum(1 for src in labels if src == name)
        blocks[name] = print_comparisons(
            f"{question}, on {name} alone ({count:,} of {len(labels):,} items)",
            HEAD_TO_HEAD,
            restrict(records, labels, name),
            arithmetic,
            explain_arithmetic=False,
        )
        print()
    print(closing)
    print()
    return blocks


def print_heterogeneity(
    blocks: dict[str, list[dict[str, Any]]],
    family: tuple[tuple[str, str, str], ...] = HEAD_TO_HEAD,
    *,
    question: str = "is the margin the same on every source?",
) -> list[dict[str, Any]]:
    """Whether each comparison's per-source deltas differ by more than their own noise.

    The block above prints two intervals per comparison and leaves the reader to compare
    them by eye, which is the one thing overlapping intervals are known not to support.
    This is the test: independent per-source deltas, inverse-variance pooled, Cochran's Q
    on the spread.

    What a significant row licenses is narrow. It says the method's margin is not the same
    on both datasets -- not that the aggregate is wrong, and not that either per-source
    number is the "real" one. The aggregate stays the panel's claim because it is the one
    the arms were run to answer; this says where that claim's margin comes from.

    The family is corrected within itself, and it is a family: one test per comparison,
    asked only because the panel already ran. A row that separates here at an uncorrected p
    and not at the adjusted one is a row that did not separate.
    """
    if len(blocks) < 2:
        # One source is not a mixture, and the caller printing zero blocks has already said
        # why. Silence rather than a header with nothing under it.
        return []
    questions: dict[str, str] = {}
    gathered: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for source, entries in blocks.items():
        for entry in entries:
            key = (entry["left"], entry["right"])
            questions[repr(key)] = entry["question"]
            gathered.setdefault(repr(key), []).append((source, entry))

    usable = {key: found for key, found in gathered.items() if len(found) == len(blocks)}
    if not usable:
        return []

    print(f"{question} ({', '.join(sorted(blocks))})")
    header = (
        f"{'comparison':28s} {'pooled':>7s} {'spread':>15s} {'Q':>7s} "
        f"{'p':>10s} {'p (Holm)':>10s}  verdict"
    )
    print(header)
    print("-" * len(header))

    computed: list[dict[str, Any]] = []
    for key, found in usable.items():
        found = sorted(found, key=lambda pair: pair[0])
        deltas = [entry["paired"].delta_points for _, entry in found]
        errors = [entry["paired"].standard_error_points for _, entry in found]
        if any(error <= 0 for error in errors):
            # No flips on a source: the delta is exactly zero with no width, and a weight of
            # infinity would decide the pooled estimate by itself. Skipped and named, which
            # is what the row would otherwise silently be.
            print(f"{questions[key]:28s} (a source produced no flips at all)")
            continue
        pooled, q, p_value = cochran_q(deltas, errors)
        computed.append(
            {
                "left": found[0][1]["left"],
                "right": found[0][1]["right"],
                "question": questions[key],
                "sources": {source: entry["paired"].delta_points for source, entry in found},
                "pooled_points": pooled,
                "q": q,
                "df": len(found) - 1,
                "p_value": p_value,
                # Inherited, not re-derived. The comparison is the same comparison; if its
                # arms did not demonstrably run the same expert arithmetic then neither did
                # the per-source halves, and a heterogeneity row is a statement about two
                # deltas that each carry the confound.
                "same_arithmetic": found[0][1]["same_arithmetic"],
            }
        )

    for entry, p_adj in zip(computed, holm([e["p_value"] for e in computed]), strict=True):
        entry["p_adjusted"] = p_adj
        entry["heterogeneous"] = p_adj < 0.05
        spread = ", ".join(f"{delta:+.2f}" for _, delta in sorted(entry["sources"].items()))
        print(
            f"{entry['question']:28s} {entry['pooled_points']:+6.2f} {spread:>15s} "
            f"{entry['q']:7.2f} {entry['p_value']:10.3g} {p_adj:10.3g}  "
            f"{'HETEROGENEOUS' if entry['heterogeneous'] else 'consistent'}"
            f"{'' if entry['same_arithmetic'] else '  !'}"
        )

    flagged = [entry for entry in computed if entry["heterogeneous"]]
    print()
    if flagged:
        print("  A heterogeneous row means the margin is not one number. It does not demote the")
        print("  aggregate, which is what the arms were run to measure -- it says which part of")
        print("  the mixture that aggregate is coming from, and that a differently weighted")
        print("  mixture would report a different margin.")
    else:
        print("  No row separates: every margin is consistent with being the same on every")
        print("  source. Consistent is not the same as equal, and these subsets are small")
        print("  enough that a real difference the size of the aggregate would often fail")
        print("  to show here.")
    if computed and len(computed) < len(family):
        # The same warning the block above carries, and it bites harder here. Holm's
        # multiplier is the number of comparisons actually corrected, so a row that clears
        # 0.05 over a half-run family can fail over the finished one -- on this panel's
        # five arms the single heterogeneous row is 0.0359 over three comparisons and
        # 0.0717 over six. That is not a small movement near the boundary; it is the
        # verdict. A reader glancing at a running panel has no way to see it from the row.
        print(
            f"  Holm-adjusted over {len(computed)} of {len(family)} comparisons -- a short "
            f"family. A verdict"
        )
        print(
            "  here is provisional in one direction: finishing the panel can only raise "
            "these adjusted"
        )
        print("  p, so a row that reads HETEROGENEOUS now may read consistent then.")
    if any(not entry["same_arithmetic"] for entry in computed):
        print("  ! flags carry down from the block above: both halves of a flagged comparison")
        print("    carry the same unpriced dispatch term, so its spread does too.")
    print()
    return computed


def agreement_records(
    records: dict[str, dict[str, Any]], ceiling: str = CEILING
) -> dict[str, dict[str, Any]]:
    """Every arm's record with ``hits`` replaced by "answered the way the ceiling did".

    Everything but ``hits`` is carried through untouched, so the derived records pair under
    the same rules the real ones do and can go straight to `print_comparisons`. That is the
    point of deriving records rather than writing a second comparison path: the question,
    the family, the Holm correction and the mixed-arithmetic mark are all the same, and only
    the indicator differs.
    """
    base = records.get(ceiling)
    if not base or not base.get("hits"):
        return {}
    derived: dict[str, dict[str, Any]] = {}
    for label, record in records.items():
        # An arm agrees with itself on every item by construction, and a row of 100.00% in
        # a fidelity table reads like a result rather than like a tautology.
        if label == ceiling:
            continue
        hits = record.get("hits")
        if not hits or len(hits) != len(base["hits"]):
            continue
        derived[label] = {
            **record,
            "hits": [bool(hit) == bool(fact) for hit, fact in zip(hits, base["hits"], strict=True)],
        }
    return derived


def _percent(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.2f}%"


def print_fidelity(
    built: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    ceiling: str = CEILING,
) -> list[dict[str, Any]]:
    """How often each arm answered the way the model it was built from did.

    A quantized arm either matches the ceiling on an item or flips it -- a hit is a boolean
    and there is no third case -- so accuracy is an exact function of two fidelities and the
    ceiling's own accuracy ``c``::

        accuracy = c * agree_where_right + (1 - c) * (1 - agree_where_wrong)

    which is why this is a block rather than a footnote. The two columns pull in opposite
    directions: tracking the ceiling more closely wins the items it got right and loses the
    items it got wrong. A method whose advantage is really fidelity therefore appears in the
    accuracy table as a margin that changes sign with the difficulty of the subset, and gets
    read as a margin that is unstable. On this panel it is +1.18 points where bf16 is right
    and -2.22 where it is wrong -- two rows, one cause.

    Reported per arm and not only head-to-head because the levels carry their own meaning:
    two arms can separate on fidelity while both sit far enough below the ceiling that the
    difference between them is not what limits either.
    """
    base = records.get(ceiling)
    if not base or not base.get("hits"):
        print(f"fidelity: no {ceiling} arm with per-item hits, so there is nothing to agree with")
        return []
    truth = [bool(hit) for hit in base["hits"]]
    right = [index for index, hit in enumerate(truth) if hit]
    wrong = [index for index, hit in enumerate(truth) if not hit]
    print(
        f"fidelity: how often each arm answered the way {ceiling} did "
        f"({len(right):,} it got right, {len(wrong):,} it got wrong)"
    )
    header = (
        f"{'arm':10s} {'accuracy':>9s} {'agrees':>8s} {'where right':>12s} {'where wrong':>12s}"
    )
    print(header)
    print("-" * len(header))

    computed: list[dict[str, Any]] = []
    for row in built:
        label = row["label"]
        if label == ceiling:
            continue
        hits = records.get(label, {}).get("hits")
        if not hits or len(hits) != len(truth):
            print(f"{label:10s} {'not run' if not hits else 'not pairable':>9s}")
            continue
        agree = [bool(hit) == fact for hit, fact in zip(hits, truth, strict=True)]
        entry = {
            "label": label,
            "ceiling": ceiling,
            "accuracy": row["accuracy"],
            "agreement": sum(agree) / len(agree),
            "agreement_where_ceiling_right": (
                sum(agree[index] for index in right) / len(right) if right else None
            ),
            "agreement_where_ceiling_wrong": (
                sum(agree[index] for index in wrong) / len(wrong) if wrong else None
            ),
            "ceiling_accuracy": len(right) / len(truth),
        }
        computed.append(entry)
        accuracy = _percent(entry["accuracy"])
        print(
            f"{label:10s} {accuracy:>9s} {entry['agreement'] * 100:7.2f}% "
            f"{_percent(entry['agreement_where_ceiling_right']):>12s} "
            f"{_percent(entry['agreement_where_ceiling_wrong']):>12s}"
        )
    if computed:
        share = len(right) / len(truth)
        print()
        print(
            f"  A hit either matches {ceiling} or flips it, so accuracy = c * (where right) + "
            f"(1 - c) * (1 -"
        )
        print(
            f"  (where wrong)) exactly, with c = {100.0 * share:.2f}% the ceiling's own "
            f"accuracy. The two columns"
        )
        print(
            "  trade against each other, and a fidelity gain is worth more where the ceiling "
            "is higher:"
        )
        print(f"  one point of it moves accuracy by 2c - 1 = {2.0 * share - 1.0:.2f} points here.")
    return computed


def as_json(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The computed comparisons, serialised for a reader that is not this terminal.

    ``question`` and ``same_arithmetic`` are carried, not dropped. The printed block shows
    both -- the question is the row label and ``same_arithmetic`` is the trailing ``!`` --
    so a serialisation that omits them hands a downstream consumer a delta and a verdict
    with no way to know the row was flagged. On this model the two expert dispatches
    disagree on 1.24% of teacher-forced tokens, 0.29x the effect being measured, and a
    model card generated from this json is exactly the artifact that must not lose it.
    """
    return [
        {
            "left": entry["left"],
            "right": entry["right"],
            "question": entry["question"],
            **entry["paired"].as_dict(),
            "p_adjusted": entry["p_adjusted"],
            "separated": entry["separated"],
            "same_arithmetic": entry["same_arithmetic"],
        }
        for entry in entries
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="assemble the phase-4 panel table")
    parser.add_argument("--arms", required=True, help="the --out directory arms_lfm2 run wrote")
    parser.add_argument("--json", action="store_true", help="emit the assembled table as json")
    parser.add_argument(
        "--json-out",
        help=(
            "also write the assembled table to this path. The same payload --json prints, "
            "in a file a downstream tool can read without having to find where the human "
            "output stopped"
        ),
    )
    parser.add_argument(
        "--sources",
        help="json list of per-item source labels; defaults to sources.json beside the arms",
    )
    args = parser.parse_args(argv)

    out = Path(args.arms)
    manifest, records = load_panel(out)
    params = infer_params(out, manifest)
    built = rows(out, manifest, records, params)
    pairable = check_pairable(records)
    tolerance = float(manifest.get("tolerance", 0.001))

    chance = next((r.get("chance") for r in records.values() if r.get("chance")), None)
    scored = sum(1 for row in built if row["accuracy"] is not None)
    print(f"panel: {scored}/{len(built)} arms scored   model: {manifest.get('model')}")
    anchors = manifest.get("anchors") or {}
    ordered = sorted(anchors.items(), key=lambda kv: int(kv[0]), reverse=True)
    print("anchors: " + ", ".join(f"{w}b -> {int(b):,} B" for w, b in ordered))
    print()
    print_sizes(built, params, tolerance)
    print()
    print_accuracy(built, chance)
    print()
    print_dispatch(built)
    print()
    print_by_source(built)
    print()
    print_allocation(built)
    print()

    if pairable is not None:
        print(f"NOT PAIRED: {pairable}")
        print()
    # One map for every block, built from the same rows the dispatch census printed, so
    # the flag and the census cannot disagree about what an arm ran.
    arithmetic = {row["label"]: arithmetic_of(row) for row in built}
    head = print_comparisons("head to head, at matched bytes", HEAD_TO_HEAD, records, arithmetic)
    print()
    labels, why_not = load_sources(out, args.sources, records)
    blocks = print_source_blocks(labels, why_not, records, arithmetic)
    spread = print_heterogeneity(blocks)
    ceiling = print_comparisons("what each method cost", AGAINST_CEILING, records, arithmetic)
    print()
    fidelity = print_fidelity(built, records)
    print()
    # The same family the head-to-head block ran, on the indicator above. Worth its own
    # rows because the two can disagree: a pair can separate on accuracy and not on
    # fidelity, which would say the arms differ in *which* items they get right rather
    # than in how closely either tracks the model they were both built from.
    agreed = agreement_records(records)
    fidelity_head = print_comparisons(
        f"the same comparisons, on agreement with {CEILING} instead of accuracy",
        HEAD_TO_HEAD,
        agreed,
        arithmetic,
        explain_arithmetic=False,
    )
    print()
    # `why_not` is deliberately dropped here: the accuracy call above already printed the
    # reason there are no source blocks, and printing it a second time would read as a
    # second failure.
    fidelity_blocks = print_source_blocks(
        labels,
        None,
        agreed,
        arithmetic,
        question=f"agreement with {CEILING}",
        closing=FIDELITY_BY_SOURCE_NOTE,
    )
    fidelity_spread = print_heterogeneity(
        fidelity_blocks,
        question="is the fidelity margin the same on every source?",
    )
    for line in (
        "delta = left minus right, percentage points, on the same problems in the same order.",
        "CI and p are McNemar exact over the discordant pairs; flips = only-left-right /",
        "only-right-right. The verdict follows the Holm-adjusted p within its own block.",
        "Sizes are the manifest's, not the loaded model's: a DynQuant arm is scored by",
        "encoding its widths back into bf16, so it holds fp16 and claims the allocator's",
        "bytes -- the size the same map writes when packed to disk, short by the rank-1",
        "tensors the budget does not price (205 KB here: norms, layernorms and expert",
        "bias, 0.005% of the 4b anchor and 21x inside this panel's match tolerance).",
    ):
        print(line)

    # Built once and sent to however many destinations were asked for. A second
    # construction for the file would be a second copy of the table, and the whole
    # reason a downstream card reads this rather than the records is that there is
    # exactly one place the panel's numbers are assembled.
    payload = {
        "model": manifest.get("model"),
        "anchors": anchors,
        "params": params,
        "pairable": pairable is None,
        "pairing_error": pairable,
        "arms": built,
        "head_to_head": as_json(head),
        "head_to_head_by_source": {name: as_json(entries) for name, entries in blocks.items()},
        # Already plain values -- no `as_json` pass, because there is no PairedComparison
        # underneath to flatten. Carried for the same reason the per-source blocks are: a
        # consumer that reads only the aggregate cannot tell a margin that holds everywhere
        # from one that lives in a single dataset, and on this panel those are two different
        # comparisons in the same table.
        "source_heterogeneity": spread,
        "against_ceiling": as_json(ceiling),
        # Carried because a card that reads only the accuracy delta cannot tell a method
        # that is more accurate from one that is more faithful to the model it was built
        # from -- and on this panel that distinction is the finding, not a nuance.
        "fidelity": fidelity,
        "fidelity_head_to_head": as_json(fidelity_head),
        # The accuracy spread and the fidelity spread are the pair that has to be read
        # together -- one of them varying while the other does not is a statement about the
        # ceiling, not about the method -- so a consumer that gets one and not the other
        # can draw the opposite conclusion from the same panel.
        "fidelity_head_to_head_by_source": {
            name: as_json(entries) for name, entries in fidelity_blocks.items()
        },
        "fidelity_source_heterogeneity": fidelity_spread,
    }
    serialised = json.dumps(payload, indent=2, default=str)
    if args.json:
        print(serialised)
    if args.json_out:
        Path(args.json_out).write_text(serialised + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
