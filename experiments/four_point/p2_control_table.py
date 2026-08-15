"""The phase-2 replicate's control panel: what does GPTQ's grid choice buy it here?

This is a *replicate*, and every number it prints has to be read as one. The panel
these arms are named after -- ``p2_rb_agg`` 89.57% against ``gptq_3b_head`` 88.03%,
the +1.54 at p < 0.0001 the campaign's headline rests on -- was measured on a
fine-tuned checkpoint that no longer exists, on a box that was destroyed, under
torch 2.13 / transformers 5.14. No phase-2 arm record was ever committed, because
``runs/`` is ignored, so the per-item ``hits`` those paired tests were computed from
are gone too. The original numbers survive only as prose in
``RESULTS-external-comparison.md``.

So this run rebuilds the checkpoint from the same regime and re-runs the arms on it,
and the control is read *within* this run. That is the whole point of the panel:
``stage8_gptq_3b_head`` against ``stage8_gptq_3b_head_asym`` is one flag apart on one
checkpoint, and it prices the single choice the original comparison never controlled
for. Every published GPTQ arm in this project ran ``--symmetric auto``, which for
GPTQ is symmetric; DynQuant's quantizer is asymmetric and always was. A delta between
them therefore spans two differences at once -- how the bits were allocated, and
whether a zero point is stored per group -- and on Mistral-7B that second difference
alone was worth 69.4 points. Until the same question is asked here, +1.54 is not
attributable to allocation.

Comparing this run's absolute accuracies against the original panel's is not valid
and the table does not do it. What is valid is the sign and size of a comparison
*within* this run, because both arms of every row share a checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from baselines_table import RUNS, bits_of, gib_of, load, mcnemar

# Holm is imported rather than reimplemented. A second copy of a statistic agrees
# with the first until the day it does not, and the phase-4 panel already owns the
# definition this campaign has been reporting under.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase4"))
from panel_table import holm

ARMS: list[tuple[str, str]] = [
    ("fine-tuned bf16", "stage8_fp16"),
    ("DynQuant per-row body", "p2_rb_agg"),
    ("GPTQ 3b g128 +head, symmetric", "stage8_gptq_3b_head"),
    ("GPTQ 3b g128 +head, asymmetric", "stage8_gptq_3b_head_asym"),
]

# One family, Holm-corrected together. The left arm is named first and a positive
# delta always means the left arm ahead, stated here rather than inferred from a sign
# at read time.
FAMILY: list[tuple[str, str, str]] = [
    ("stage8_gptq_3b_head_asym", "stage8_gptq_3b_head", "the grid alone, GPTQ against itself"),
    ("p2_rb_agg", "stage8_gptq_3b_head", "the published comparison, replicated"),
    ("p2_rb_agg", "stage8_gptq_3b_head_asym", "the same claim against the better GPTQ"),
    ("stage8_fp16", "p2_rb_agg", "headroom above DynQuant"),
    ("stage8_fp16", "stage8_gptq_3b_head", "headroom above symmetric GPTQ"),
    ("stage8_fp16", "stage8_gptq_3b_head_asym", "headroom above asymmetric GPTQ"),
]


def scheme_of(record: dict[str, Any]) -> str:
    """What the arm ran, read back off the record rather than off its name.

    ``stage8_baselines`` resolves ``--symmetric auto`` to the method's own default
    before writing, so this is the answer and not the request. A record carrying no
    ``symmetric`` key is a DynQuant arm, whose asymmetry is a property of the
    quantizer rather than a recipe field there is anything to record.
    """
    if "symmetric" not in record:
        return "n/a"
    grid = "symmetric" if record["symmetric"] else "asymmetric"
    order = record.get("actorder")
    return grid if order is None else f"{grid}+{order}"


def check_control_pair(sym: dict[str, Any], asym: dict[str, Any]) -> None:
    """Refuse to print a table whose control pair does not straddle the flag.

    Two checks rather than one. The recorded flag proves the recipe was built the way
    the arm claims; the widths prove it reached the quantizer, because an asymmetric
    grid stores a zero point per group and cannot cost the same as a symmetric one.
    An arm that silently ignored ``--symmetric no`` would pass the first and fail the
    second, and it would otherwise produce a perfectly readable table of one
    configuration compared against itself.
    """
    if bool(sym["symmetric"]) is not True or bool(asym["symmetric"]) is not False:
        raise SystemExit(
            f"the control pair does not straddle the flag: "
            f"{scheme_of(sym)} against {scheme_of(asym)}"
        )
    if bits_of(sym) == bits_of(asym):
        raise SystemExit(
            f"both control arms account to {bits_of(sym):.4f} bits; a stored zero point "
            "per group has to show up in the width, so the flag did not reach the recipe"
        )


def arm_rows(data: dict[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
    rows = []
    for label, name in ARMS:
        record = data[name]
        assert record is not None
        rows.append(
            {
                "name": name,
                "label": label,
                "scheme": scheme_of(record),
                "accuracy": record["accuracy"],
                "correct": record["correct"],
                "total": record["total"],
                "bits": bits_of(record),
                "nbytes": record.get("nbytes") or int(gib_of(record) * 2**30),
            }
        )
    return rows


def comparison_rows(data: dict[str, dict[str, Any] | None]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right, why in FAMILY:
        a, b = data[left], data[right]
        assert a is not None and b is not None
        delta, a_only, b_only, low, high, p = mcnemar(a["hits"], b["hits"])
        rows.append(
            {
                "left": left,
                "right": right,
                "why": why,
                "delta_points": round(delta, 4),
                "a_only": a_only,
                "b_only": b_only,
                "discordant": a_only + b_only,
                "ci_low_points": round(low, 4),
                "ci_high_points": round(high, 4),
                # float(), not p. binomtest returns a numpy scalar and json.dumps
                # refuses one, so the JSON this table writes would die on the last
                # line after every arm had already been paid for.
                "p_value": float(p),
            }
        )
    adjusted = holm([row["p_value"] for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["p_adjusted"] = float(value)
        # Separation is claimed off the corrected p and not the raw one. Six
        # comparisons on one test split is a family whether or not it is called one.
        row["separated"] = bool(value < 0.05)
    return rows


def main() -> int:
    data = {name: load(name) for _, name in ARMS}
    missing = [name for _, name in ARMS if data[name] is None]
    if missing:
        raise SystemExit(f"not on disk in {RUNS}: {', '.join(missing)}")
    sym, asym = data["stage8_gptq_3b_head"], data["stage8_gptq_3b_head_asym"]
    assert sym is not None and asym is not None
    check_control_pair(sym, asym)

    arms = arm_rows(data)
    comparisons = comparison_rows(data)

    print(f"\nphase-2 replicate, {RUNS}\n")
    head = f"{'arm':<32} {'scheme':<12} {'acc %':>7} {'bits':>7} {'bytes':>15} {'n':>6}"
    print(head)
    print("-" * len(head))
    for row in arms:
        print(
            f"{row['label']:<32} {row['scheme']:<12} {100 * row['accuracy']:>7.2f} "
            f"{row['bits']:>7.4f} {row['nbytes']:>15,} {row['total']:>6}"
        )

    print(f"\nMcNemar, Holm-corrected over all {len(comparisons)} comparisons\n")
    head = (
        f"{'comparison':<54} {'delta':>7} {'discordant':>11} {'95% CI':>18} "
        f"{'p':>9} {'Holm':>9}  sep"
    )
    print(head)
    print("-" * len(head))
    for row in comparisons:
        pair = f"{row['left']} vs {row['right']}"
        interval = f"[{row['ci_low_points']:+.2f}, {row['ci_high_points']:+.2f}]"
        discordant = f"{row['a_only']}/{row['b_only']}"
        print(
            f"{pair:<54} {row['delta_points']:>+7.2f} {discordant:>11} "
            f"{interval:>18} {row['p_value']:>9.2g} {row['p_adjusted']:>9.2g}  "
            f"{'yes' if row['separated'] else 'no'}"
        )
    print()
    for row in comparisons:
        print(f"  {row['left']} vs {row['right']}: {row['why']}")

    out = RUNS / "p2_control_table.json"
    out.write_text(
        json.dumps(
            {
                "run_dir": str(RUNS),
                "replicate": True,
                "replicate_of": "the phase-2 panel in RESULTS-external-comparison.md",
                "arms": arms,
                "comparisons": comparisons,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
