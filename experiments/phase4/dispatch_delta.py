"""What the experts dispatch costs, from two panels that differ in nothing else.

The eager re-score exists to make seven arms comparable. It also, for nothing, produces the
measurement this campaign has owed since section 8 of the packed-MoE report: the same
model, the same 12,000 items, the same seed, the same everything, scored once on the
dispatch `post_init` chose and once on `eager`. Three arms keep their expert banks --
`bf16`, `dq_4b`, `dq_3b` -- so three paired comparisons fall out of a run that was already
going to happen.

They fall out only if someone collects them, and the re-score writes over the records it
re-scores. So the sequence is `cp -a panel panel_grouped_mm`, then re-score `panel` in
place, then this script over the two directories. It is written down here rather than in a
report because a measurement that depends on remembering a `cp` is a measurement that will
be lost.

What this is not: a comparison of `eager` against the linearised loop. Three dispatches
appear in this campaign and they make three pairs, and the report has already been caught
once carrying a number from one pair to another. This script measures `grouped_mm` against
`eager` on arms that own a bank. The four baselines have no bank left and cannot appear in
either column.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CORE_SRC = REPO_ROOT / "packages" / "dynquant-core" / "src"
# `HERE` as well as the package source, because `holm` lives in the sibling script and this
# file is loaded two ways: as `__main__`, where the interpreter puts its own directory on
# the path, and by `importlib` from the tests, where it does not.
for entry in (str(CORE_SRC), str(HERE)):  # pragma: no cover - import bootstrap
    if entry not in sys.path:
        sys.path.insert(0, entry)


def dispatch_of(record: dict[str, Any]) -> str | None:
    """What the record says it ran, or ``None`` if it does not say.

    ``None`` is not "no dispatch". It is a record written before ``dynquant eval`` learned
    to write the field, which is what every arm of the first LFM2.5 panel is. The whole
    point of the re-score is to turn one side of each pair from ``None`` into a name, so
    the absence is expected on the ``--before`` side and is a failure on the ``--after``
    side.
    """
    experts = record.get("experts")
    if not experts:
        return None
    ran = experts.get("ran")
    return None if ran is None else str(ran)


def check_pair(before: dict[str, Any], after: dict[str, Any]) -> str | None:
    """Why these two records cannot be read as a dispatch measurement, or ``None``.

    Two refusals, and they are refusals for opposite reasons.

    The first is the ordinary one: if the two runs asked different questions -- a different
    split, a different ``--limit``, a different decode budget -- their hit vectors are not
    element-wise the same items and pairing them is arithmetic on unrelated vectors.
    ``problem_set_difference`` is exactly that test and it deliberately ignores the expert
    dispatch, which is the one field these two records are *supposed* to differ on.

    The second is the one worth the file. If the two records agree on the dispatch, there
    is no measurement here -- and the number it would print is not noise, it is a
    convincing zero. A re-score whose ``--experts-impl`` never took effect produces two
    runs of the same computation, `compare_paired` reports a delta near zero, and that
    reads as "the dispatch is free", which is the strongest form of the claim this whole
    exercise exists to stop asserting without evidence. A fix that measures zero may be a
    fix that never ran, so the zero is refused rather than printed.
    """
    from dynquant.commands.evaluate import problem_set_difference

    moved = problem_set_difference(before, after)
    if moved:
        return f"not the same problem set: {', '.join(sorted(moved))}"

    left, right = dispatch_of(before), dispatch_of(after)
    if right is None:
        return "the after record does not say what dispatch it ran, so it was not re-scored"
    if left == right:
        return (
            f"both records ran {right!r}, so there is no dispatch difference to measure. "
            "A delta computed here would be a zero produced by the re-score not having "
            "happened, which is indistinguishable from the dispatch being free"
        )
    if len(before.get("hits") or ()) == 0 or len(after.get("hits") or ()) == 0:
        return "no per-item hits recorded, so the comparison cannot be paired"
    return None


def load_panel(directory: Path) -> dict[str, dict[str, Any]]:
    """Every eval record in a panel directory, keyed by label.

    Sidecars are excluded by suffix rather than by an allow-list of labels, because the
    point of pointing this at two directories is that it does not need to know which arms
    the panel planned.
    """
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name in {"arms.json", "sources.json"} or path.name.endswith(".quant.json"):
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if "hits" not in record and "accuracy" not in record:
            continue
        records[record.get("label") or path.stem] = record
    return records


def compare(before_dir: Path, after_dir: Path, only: list[str] | None) -> list[dict[str, Any]]:
    """One row per arm scored in both directories, refusals included.

    Refusals are rows and not exceptions. A panel where two of three arms re-scored is
    still two measurements, and the third's reason belongs beside them where it can be
    read against the two that worked.
    """
    from dynquant.eval.compare import compare_paired

    before, after = load_panel(before_dir), load_panel(after_dir)
    labels = [label for label in before if label in after]
    if only is not None:
        labels = [label for label in labels if label in only]

    rows: list[dict[str, Any]] = []
    for label in labels:
        a, b = before[label], after[label]
        why_not = check_pair(a, b)
        row: dict[str, Any] = {
            "label": label,
            "from": dispatch_of(a) or "unrecorded",
            "to": dispatch_of(b) or "unrecorded",
            "seconds_before": a.get("seconds"),
            "seconds_after": b.get("seconds"),
            "why_not": why_not,
        }
        if why_not is None:
            row["paired"] = compare_paired(
                a["hits"],
                b["hits"],
                label_a=f"{label}:{row['from']}",
                label_b=f"{label}:{row['to']}",
            )
        rows.append(row)
    return rows


def print_rows(rows: list[dict[str, Any]]) -> None:
    """The table, and the two things it is evidence about.

    Accuracy and wall clock in one table because they are two readings of one change and
    the campaign has a number for neither. Section 8 priced the linearised loop against
    ``grouped_mm`` at 1.9-2.3x from the panel's own timings and could not separate the
    dispatch from the dequantization, because every linearised arm was also a quantized
    one. These pairs have no such confound: the weights are identical on both sides.
    """
    # Deferred, and imported by name rather than by path, because the bootstrap above put
    # this file's own directory on the path for exactly this line.
    from panel_table import holm

    print("experts dispatch, paired over one problem set")
    print()
    print(
        f"{'arm':10s} {'dispatch':>26s} {'acc before':>11s} {'acc after':>10s} "
        f"{'delta':>8s} {'p':>9s} {'p (Holm)':>9s} {'seconds':>18s}"
    )

    measured = [row for row in rows if row["why_not"] is None]
    adjusted = holm([row["paired"].p_value for row in measured])
    for row, p_adj in zip(measured, adjusted, strict=True):
        paired = row["paired"]
        before_s, after_s = row["seconds_before"], row["seconds_after"]
        clock = "--"
        if before_s and after_s:
            clock = f"{before_s:,.0f} -> {after_s:,.0f} ({after_s / before_s:.2f}x)"
        print(
            f"{row['label']:10s} {row['from'] + ' -> ' + row['to']:>26s} "
            f"{paired.accuracy_a * 100:10.2f}% {paired.accuracy_b * 100:9.2f}% "
            f"{paired.delta_points:+8.2f} {paired.p_value:9.4f} {p_adj:9.4f} {clock:>18s}"
        )
    for row in rows:
        if row["why_not"] is not None:
            print(f"{row['label']:10s} {row['why_not']}")

    if not measured:
        print()
        print("  Nothing measured. Every row above says why, and none of the reasons is a result.")
        return

    print()
    print(
        "  The delta is before minus after, so a positive number means the dispatch the panel used"
    )
    print(
        "  scored higher than the one a download runs. Sign matters here in a way it did "
        "not for the"
    )
    print(
        "  token-agreement probe, which could only say the two computations differ and not which is"
    )
    print("  better; this can, because both sides are scored against the same golds.")
    print()
    print(
        "  Weights are identical across each pair, so the seconds column is the dispatch "
        "cost with no"
    )
    print(
        "  dequantization confounded into it -- which is what section 8's 1.9-2.3x could "
        "not separate."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before",
        required=True,
        help="the panel as it was scored, copied aside before the re-score overwrote it",
    )
    parser.add_argument("--after", required=True, help="the re-scored panel")
    parser.add_argument(
        "--labels",
        default=None,
        metavar="LABEL[,LABEL...]",
        help="restrict to these arms (default: every arm present in both directories)",
    )
    args = parser.parse_args(argv)

    only = [part.strip() for part in args.labels.split(",")] if args.labels else None
    rows = compare(Path(args.before), Path(args.after), only)
    if not rows:
        raise SystemExit(
            f"no arm is scored in both {args.before} and {args.after}. This wants the "
            "pre-re-score copy of a panel directory and the re-scored one, not two "
            "different panels."
        )
    print_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
