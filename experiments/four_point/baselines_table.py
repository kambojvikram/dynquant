"""The external comparison table: DynQuant against GPTQ, AWQ and RTN.

Reads the records :mod:`stage8_baselines` and :mod:`stage5_quantize` wrote and prints
two things -- a row per arm, and a paired test per comparison that matters.

Why paired tests rather than a column of accuracies
---------------------------------------------------
Every arm scores the same test split in the same order, so the arms are matched on
problem and the right test is McNemar's on the discordant pairs. An unpaired comparison
of two accuracies is roughly twice as wide on this data, which at the 4-bit end is the
difference between "these methods are indistinguishable" and "we could not tell".
Those are different claims and only the first one is honest when the pairing exists.

Why the size column is bits and not "4-bit"
--------------------------------------------
The arms do not cost what their names say. A "4-bit g128" GPTQ checkpoint keeps its
embedding and LM head in fp16 -- the convention every published checkpoint follows --
and pays an fp16 scale plus a zero point per group of 128 on everything else. On
Mistral-7B that is 4.595 bits per weight, not 4. DynQuant's 4.25 quantizes the
embedding and the head too and its 4.25 already includes metadata. Ranking these arms
by their nominal width would credit the baselines for bytes they are still spending, so
the table sorts by what they measurably cost.

Which leaves the comparison at 3 bits reading two things at once -- DynQuant 3.25 against
GPTQ 3.6249 is a method difference plus a 0.37-bit budget difference, and the row cannot
separate them. ``run_isosize.sh`` closes that from the other side by allocating DynQuant
at the baselines' measured width; those arms are picked up here automatically and compared
against every baseline sharing their budget, and against the nominal DynQuant arm below
them, which is what isolates the budget gap.

On a tied-embedding model that budget gap stops being a footnote
----------------------------------------------------------------
``ignore=["lm_head"]`` costs the baselines 0.35 bits on Mistral-7B, where the LM head is
5.5% of the checkpoint. Qwen3.5-2B shares one ``[248320, 2048]`` tensor between
``embed_tokens`` and ``lm_head`` -- 508,559,360 parameters, 27.02% of the model's
1,881,825,088, counted from the checkpoint. (The 27.05% quoted elsewhere is the residue
*back-solved* from the measured effective bits, which also picks up the norms and a few
small modules llm-compressor skips; the two are not the same quantity.) So the
same convention leaves "4-bit g128" measuring
7.3605 bits and "3-bit" measuring 6.6253, with most of the stored bits in tensors nobody
quantized. Comparing DynQuant's 4.2486 against that is not a method comparison, and no
iso arm fixes it from the DynQuant side because matching 7.36 bits means asking the
allocator for a budget no one would ship.

``run_tiedhead.sh`` fixes it from the baseline side, re-running the same recipes with the
tie quantized: 4.1597 and 3.1522 bits, just *under* DynQuant's own width. Those arms are
picked up here automatically and appear as a second panel, paired against the nearest
DynQuant arm and against their own fp16-head twin. Both panels are kept -- the default one
is a real finding about the convention, and the tied-head one is the only place the
allocator is compared on method alone.
"""

from __future__ import annotations

import json
from math import sqrt

from common import RUN_DIR as RUNS
from scipy.stats import binomtest

# Order is the order the table prints in: the ceiling, then each budget with DynQuant
# first and its external competitors under it.
ARMS: list[tuple[str, str]] = [
    ("fine-tuned bf16", "stage8_fp16"),
    ("DynQuant 4.25b", "stage8_dq_4p25"),
    ("GPTQ 4b g128", "stage8_gptq_4b"),
    ("AWQ 4b g128", "stage8_awq_4b"),
    ("bnb NF4", "stage8_nf4_4b"),
    ("RTN 4b g128", "stage8_rtn_4b"),
    ("DynQuant 3.25b", "stage8_dq_3p25"),
    ("GPTQ 3b g128", "stage8_gptq_3b"),
    ("AWQ 3b g128", "stage8_awq_3b"),
    ("RTN 3b g128", "stage8_rtn_3b"),
]

# Left arm is DynQuant in every row, so a positive delta always means DynQuant ahead.
# Stated once here rather than inferred from the sign at read time.
COMPARISONS: list[tuple[str, str]] = [
    ("stage8_dq_4p25", "stage8_gptq_4b"),
    ("stage8_dq_4p25", "stage8_awq_4b"),
    ("stage8_dq_4p25", "stage8_nf4_4b"),
    ("stage8_dq_4p25", "stage8_rtn_4b"),
    ("stage8_gptq_4b", "stage8_rtn_4b"),
    ("stage8_dq_3p25", "stage8_gptq_3b"),
    ("stage8_dq_3p25", "stage8_awq_3b"),
    ("stage8_dq_3p25", "stage8_rtn_3b"),
    ("stage8_gptq_3b", "stage8_rtn_3b"),
    ("stage8_fp16", "stage8_dq_4p25"),
    ("stage8_fp16", "stage8_gptq_4b"),
    ("stage8_fp16", "stage8_dq_3p25"),
    ("stage8_fp16", "stage8_gptq_3b"),
]


QUANT_SUFFIX = "_quant.json"
"""The per-layer error companion :mod:`stage5_quantize` writes beside every DynQuant arm.

Excluded from both globs by name. It has to be: ``stage8_dq_iso*.json`` matches
``stage8_dq_iso6p63_quant.json`` too, and that record holds ``layers`` and ``nbytes`` but no
``hits`` -- so the arm list picked it up as an arm and :func:`provenance` died on ``KeyError:
'hits'`` before printing a single row. A glob over a directory that holds two record shapes has
to discriminate between them; matching the accuracy shape by prefix alone does not.
"""

ISO_PREFIX = "stage8_dq_iso"
"""Equal-byte DynQuant arms written by ``run_isosize.sh``.

Discovered by glob rather than listed in :data:`ARMS`, because the width each one matches
is a property of the model -- the baselines' accounted width depends on vocabulary share
and whether the embedding is tied -- so the set is ``iso4p60``/``iso3p62`` on Mistral and
something else on the next checkpoint. Listing them would mean editing this file per
model, which is how a table ends up silently missing an arm that ran.
"""


def load(name: str) -> dict | None:
    path = RUNS / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def bits_of(record: dict) -> float:
    """Stored bits per weight, whichever stage wrote the record.

    stage 5 calls it ``average_bits`` and stage 8 calls it ``accounted_bits``; they are
    the same quantity under the same convention -- total stored bits including metadata,
    divided by parameters -- which is the only reason it is safe to read either into one
    column. The bf16 arm has neither key and is 16 by definition.
    """
    for key in ("average_bits", "accounted_bits"):
        if key in record:
            return float(record[key])
    return 16.0


def gib_of(record: dict) -> float:
    """Stored size in GiB, whichever of the three writers produced the record.

    Three keys for one quantity, because three different stages write these arms:
    ``accounted_gib`` from stage 8's byte accounting, ``quantized_gib`` from stage 5's
    manifest, and ``bytes_on_disk`` for the bf16 arm, which has no quantized size and is
    measured by walking its checkpoint. The first two already include metadata; the third
    is the whole directory, which is ~34 KB of tokenizer above the tensor bytes and is
    reported rather than corrected because it is what the checkpoint actually costs.

    Returning 0.0 for a record carrying none of them is deliberate -- an arm with no size
    field is a reporting gap, and a zero in the column is visible where a silently
    plausible number computed from a missing ``params`` key was not. That is the bug this
    replaced: the DynQuant rows printed 0.000 GiB for exactly that reason.
    """
    for key in ("accounted_gib", "quantized_gib"):
        if key in record:
            return float(record[key])
    if "bytes_on_disk" in record:
        return float(record["bytes_on_disk"]) / 2**30
    return 0.0


def mcnemar(left: list[bool], right: list[bool]) -> tuple[float, int, int, float, float, float]:
    # strict: two hit vectors of different lengths are two different problem sets, and
    # pairing them silently would report a method difference that is a harness one.
    only_left = sum(1 for a, b in zip(left, right, strict=True) if a and not b)
    only_right = sum(1 for a, b in zip(left, right, strict=True) if b and not a)
    n = only_left + only_right
    p = binomtest(only_left, n, 0.5).pvalue if n else 1.0
    delta = 100.0 * (only_left - only_right) / len(left)
    if n:
        se = 100.0 * sqrt(n) / len(left)
        lo, hi = delta - 1.96 * se, delta + 1.96 * se
    else:
        lo = hi = 0.0
    return delta, only_left, only_right, lo, hi, p


def iso_arms() -> list[tuple[str, str]]:
    """The equal-byte arms present on disk, widest first."""
    found: list[tuple[float, str, str]] = []
    for path in RUNS.glob(f"{ISO_PREFIX}*.json"):
        if path.name.endswith(QUANT_SUFFIX):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        found.append((bits_of(record), record.get("label", path.stem), path.stem))
    return [(label, name) for _, label, name in sorted(found, reverse=True)]


def iso_comparisons(
    data: dict[str, dict | None], iso: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Every comparison an equal-byte arm makes possible, paired by measured width.

    Matched on ``bits`` within a tolerance rather than by name, because "same budget" is
    the only reason these arms exist and name-matching would pair a 4.60-bit arm with a
    row labelled "4-bit" that costs 4.5953. The 0.05 tolerance is set to include bnb-NF4
    at 4.5671 -- its metadata is a doubly-quantized per-64 absmax rather than a per-128
    fp16 scale, so it lands 0.03 bits below the g128 family and is still the same budget
    for any purpose a reader cares about.

    Each iso arm is also paired against the nominal DynQuant arm below it, since the
    difference between those two rows is what the extra bits bought and is the only way to
    read the budget gap separately from the method gap.
    """
    pairs: list[tuple[str, str]] = []
    for _, name in iso:
        record = data.get(name)
        if record is None:
            continue
        width = bits_of(record)
        for _, other in ARMS:
            neighbour = data.get(other)
            if neighbour is None or "accounted_bits" not in neighbour:
                continue
            if abs(bits_of(neighbour) - width) <= 0.05:
                pairs.append((name, other))
        # The nominal DynQuant arm at the next budget down, if there is one.
        below = [
            (bits_of(data[n]), n)
            for _, n in ARMS
            if data.get(n) is not None and "average_bits" in data[n] and bits_of(data[n]) < width
        ]
        if below:
            pairs.append((name, max(below)[1]))
        pairs.append(("stage8_fp16", name))
    return pairs


HEAD_SUFFIX = "_head"
"""Baseline arms run with ``--include-head``, written by ``run_tiedhead.sh``.

Globbed for the same reason the iso arms are: they exist only on a model that ties
``lm_head`` to ``embed_tokens``, so the set is empty on Mistral-7B and six arms on
Qwen3.5-2B, and a hardcoded list would have to be edited per model.

These are the panel that makes a tied model comparable at all. With the standard
``ignore=["lm_head"]`` the shared 508.6M tensor -- 27.02% of Qwen3.5-2B -- stays fp16, so
"4-bit g128" measures 7.3605 bits against DynQuant's 4.2486 and the accuracy column is
comparing budgets, not methods. Quantizing it drops the same recipes to 4.1597 and
3.1522, just under DynQuant's own width.
"""


def head_arms() -> list[tuple[str, str]]:
    """The tied-head baseline arms present on disk, widest first."""
    found: list[tuple[float, str, str]] = []
    for path in RUNS.glob(f"stage8_*{HEAD_SUFFIX}.json"):
        if path.name.endswith(QUANT_SUFFIX):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        found.append((bits_of(record), record.get("label", path.stem), path.stem))
    return [(label, name) for _, label, name in sorted(found, reverse=True)]


def head_comparisons(
    data: dict[str, dict | None], head: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """The two questions a tied-head arm answers, per arm.

    First, DynQuant against it at nearly equal bytes -- paired to the *nearest* DynQuant
    arm by measured width rather than by a tolerance window, because 4.1597 against 4.2486
    is 0.089 bits apart and no window that admits that pair would exclude anything else on
    a two-budget table. The 0.25 cap is there only so an arm at some unrelated width does
    not get silently paired with whatever is closest. The remaining gap runs *against*
    DynQuant, which is worth saying in the writeup: the baseline is the smaller model in
    these rows.

    Second, the arm against its own default-panel twin -- ``gptq 4b`` against ``gptq 4b
    +head`` -- with the default on the left, so a positive delta is the accuracy the
    baseline gives up by quantizing the tied tensor. That is the price DynQuant is paying
    on the same tensor, measured on the baselines' own methods, and it is the number that
    says whether the default panel's size gap is a free lunch or a real trade.
    """
    pairs: list[tuple[str, str]] = []
    for _, name in head:
        record = data.get(name)
        if record is None:
            continue
        width = bits_of(record)
        nearest = [
            (abs(bits_of(data[n]) - width), n)
            for _, n in ARMS
            if data.get(n) is not None and "average_bits" in data[n]
        ]
        if nearest and min(nearest)[0] <= 0.25:
            pairs.append((min(nearest)[1], name))
        twin = name[: -len(HEAD_SUFFIX)]
        if data.get(twin) is not None:
            pairs.append((twin, name))
    return pairs


def arm_label(record: dict) -> str:
    """An arm's own name, with any trailing "vs ..." purpose clause dropped.

    ``run_isosize.sh`` labels its arms by what they are *for* -- "dynquant iso-3.6244b vs 3-bit
    baselines" -- which is the right label in the arm table and the wrong one inside a comparison,
    where it yields "... vs 3-bit baselines vs rtn 3-bit g128" and the reader has to guess which
    " vs " is the comparison. Keep only the part left of the first " vs ".
    """
    return record.get("label", "?").split(" vs ")[0].strip()


def provenance(data: dict[str, dict | None], rows: list[tuple[str, str]]) -> list[str]:
    """Reasons not to believe a row, checked before the row is printed.

    The first check exists because it already fired. An RTN 4-bit arm scored *exactly*
    the bf16 arm's 2907/3080 with an identical hit vector, because llm-compressor's
    ``oneshot`` fits scales without rounding the weights and the in-process path never
    saved, so the evaluation ran on the original checkpoint. Four-bit quantization of a
    7 B model cannot leave 3080 predictions untouched, so hit-vector equality with the
    bf16 arm is a sound and very cheap test for "this arm was never quantized" -- and it
    is the one thing a reader cannot spot from an accuracy column, since the number looks
    excellent rather than broken.

    The second check is the positive form of the same question. Arms quantized after the
    fix carry ``materialized_modules`` from ``stage8_baselines.materialize_quantization``,
    which asserts every probed weight sits on its own quantization grid. An arm without
    that field predates the guard and is trusted only on the equality check above.
    """
    reference = data.get("stage8_fp16")
    notes: list[str] = []
    for _, name in rows:
        record = data.get(name)
        if record is None or name == "stage8_fp16":
            continue
        if reference is not None and record["hits"] == reference["hits"]:
            notes.append(
                f"  !! {name}: hit vector is identical to bf16 on all {len(record['hits'])} "
                "problems -- this arm was NOT quantized, do not report it"
            )
        elif record.get("method") in METHODS_NEEDING_PROOF and "materialized_modules" not in record:
            notes.append(
                f"  ?  {name}: no materialization proof in the record (quantized before the "
                "guard existed); differs from bf16, so it did quantize, but unverified"
            )
    return notes


METHODS_NEEDING_PROOF = {"gptq", "awq", "rtn"}
"""The llm-compressor arms. bnb-NF4 quantizes on load and DynQuant writes its own
dequantized values, so neither goes through the materialization path."""


def main() -> None:
    # Missing is reported over ARMS only: an absent iso arm means run_isosize.sh has not
    # been run for this model, which is a different statement from "an arm of the
    # comparison is missing" and should not read as a gap in the table.
    data = {name: load(name) for _, name in ARMS}
    missing = [name for name, record in data.items() if record is None]
    if missing:
        print(f"not yet measured: {', '.join(missing)}\n")

    iso = iso_arms()
    head = head_arms()
    data.update({name: load(name) for _, name in iso + head})
    rows = ARMS + iso + head

    notes = provenance(data, rows)
    if notes:
        print("provenance warnings:")
        print("\n".join(notes), "\n")

    print(
        f"| {'arm':22s} | {'bits':>6s} | {'GiB':>6s} | {'accuracy':>8s} | "
        f"{'±1SE':>5s} | {'correct':>11s} | {'unparsed':>8s} |"
    )
    print(f"|{'-' * 24}|{'-' * 8}|{'-' * 8}|{'-' * 10}|{'-' * 7}|{'-' * 13}|{'-' * 10}|")
    for label, name in rows:
        record = data.get(name)
        if record is None:
            print(
                f"| {label:22s} | {'--':>6s} | {'--':>6s} | {'--':>8s} | "
                f"{'--':>5s} | {'--':>11s} | {'--':>8s} |"
            )
            continue
        bits = bits_of(record)
        total, accuracy = record["total"], record["accuracy"]
        gib = gib_of(record)
        se = 100.0 * sqrt(accuracy * (1 - accuracy) / total)
        print(
            f"| {label:22s} | {bits:6.3f} | {gib:6.3f} | {100 * accuracy:7.2f}% | "
            f"{se:5.2f} | {record['correct']:5d} / {total:<5d} | "
            f"{record['unparseable']:8d} |"
        )

    # Two passes, because the label column cannot have a constant width. An iso arm's label is
    # generated from its measured width ("dynquant iso-4.5949b vs 4-bit baselines"), so on
    # Mistral the joined label runs past 50 characters. The previous fixed `42.42s` truncated it
    # exactly where the right-hand arm name begins, leaving five rows that all read
    # "...iso-3.6244b vs 3-bit baselines vs" with no way to tell which baseline each was against
    # -- and those five rows had five different deltas, two of them separating. A table whose
    # rows cannot be told apart is worse than no table.
    measured: list[tuple[str, float, float, float, int, int, float]] = []
    for left_name, right_name in (
        COMPARISONS + iso_comparisons(data, iso) + head_comparisons(data, head)
    ):
        left, right = data.get(left_name), data.get(right_name)
        if left is None or right is None:
            continue
        delta, only_left, only_right, lo, hi, p = mcnemar(left["hits"], right["hits"])
        measured.append(
            (f"{arm_label(left)} vs {arm_label(right)}", delta, lo, hi, only_left, only_right, p)
        )

    width = max([len("comparison")] + [len(row[0]) for row in measured])
    print(
        f"\n| {'comparison':{width}s} | {'delta':>7s} | {'paired 95% CI':>18s} | "
        f"{'flips':>11s} | {'p':>9s} | verdict |"
    )
    print(f"|{'-' * (width + 2)}|{'-' * 9}|{'-' * 20}|{'-' * 13}|{'-' * 11}|{'-' * 9}|")
    for label, delta, lo, hi, only_left, only_right, p in measured:
        verdict = "separated" if p < 0.05 else "not separated"
        print(
            f"| {label:{width}s} | {delta:+7.2f} | [{lo:+7.2f}, {hi:+7.2f}] | "
            f"{only_left:5d} / {only_right:<5d} | {p:9.2g} | {verdict} |"
        )


if __name__ == "__main__":
    main()
