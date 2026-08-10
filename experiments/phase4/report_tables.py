"""Turn a `panel_table --json-out` payload into the markdown tables the reports carry.

The statistics have a tool now; the *tables* did not. §13's rows were typed into markdown by
hand from the terminal block above them, which is the same transcription step the tool was
written to remove, moved one file to the right -- and it is the step where a refresh goes
wrong, because a re-run changes every adjusted p at once and a reader cannot tell a stale cell
from a fresh one.

This computes nothing. It reads the payload `panel_table` already writes for the model cards
and formats it, so a second consumer of that payload is a check on it rather than a second
copy of the arithmetic. Anything it cannot find in the payload it refuses to print.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

#: The three partitions `panel_table` runs Cochran's Q over, and what each one is called in
#: prose. Keyed by the payload's own field names so a renamed block fails here rather than
#: printing an empty table.
SPREADS = {
    "source": ("source_heterogeneity", "by source"),
    "difficulty": ("difficulty_heterogeneity", "by the ceiling's own answer"),
    "crossed": ("source_and_difficulty_heterogeneity", "by source and difficulty"),
}

#: The stratified McNemar blocks, in the order §13 reads them.
BLOCKS = (
    ("head_to_head_by_difficulty", "difficulty"),
    ("head_to_head_by_source_and_difficulty", "source and difficulty"),
)


def label(question: str) -> str:
    """One comparison name, whitespace collapsed, for printing and for matching alike.

    The panel pads its question strings to a fixed column, so they arrive carrying interior
    runs of spaces that a markdown cell should not show and that nobody typing `--strata`
    would reproduce. Both uses go through this, so a name that prints is a name that matches.
    """
    return " ".join(question.split())


def marker(entry: dict[str, Any]) -> str:
    """The terminal block's trailing `!`, which the markdown must not quietly drop.

    It flags a comparison whose two arms did not demonstrably run the same expert arithmetic.
    On this panel the two dispatches disagree with each other on 1.24% of teacher-forced
    tokens, which is a large fraction of the margin being reported -- and the rows carrying
    the flag are the headline rows. A generated table that omitted it would present them as
    unconfounded, and would do so more convincingly than the hand-typed table it replaces.
    """
    return "" if entry["same_arithmetic"] else " !"


def signed(value: float) -> str:
    return f"{value:+.2f}"


def probability(value: float) -> str:
    """Three significant figures, which is where a Holm-adjusted p stops being informative.

    `%.3g` rather than a fixed number of places because this column spans 0.570 to 5.44e-19,
    and the hand-typed tables it replaces had drifted into using both conventions in the same
    section.
    """
    return f"{value:.3g}"


def notes(entries: list[dict[str, Any]]) -> str:
    """What a block says beneath itself: the confound legend, and how short the family was.

    Holm's multiplier is the number of comparisons actually corrected, so an adjusted p that
    stood against three is not the same claim as one that stood against six -- on this panel's
    own five-arm run that difference moved a row from 0.0359 to 0.0717, which is the verdict
    rather than a decimal place. `panel_table` prints the warning once under the block; a row
    pasted into a report travels alone.
    """
    lines = []
    if any(not entry["same_arithmetic"] for entry in entries):
        lines.append(
            "`!` -- the two arms did not demonstrably run the same expert arithmetic, so "
            "the margin carries the dispatch difference as well as the method."
        )
    corrected, declared = entries[0]["holm_corrected"], entries[0]["holm_family"]
    if corrected < declared:
        lines.append(
            f"Holm-adjusted over {corrected} of {declared} comparisons -- a short family, "
            f"so these adjusted *p* are weaker than the finished panel's."
        )
    return "\n\n" + "\n\n".join(lines) if lines else ""


def spread_table(entries: list[dict[str, Any]], caption: str) -> str:
    """One Cochran block: pooled margin, the per-subset spread, Q and the adjusted p."""
    subsets = sorted({name for entry in entries for name in entry["sources"]})
    header = " / ".join(subsets)
    lines = [
        f"| comparison | pooled | {header} | Q | *p* (Holm) | |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for entry in entries:
        cells = " / ".join(signed(entry["sources"][name]) for name in subsets)
        verdict = "heterogeneous" if entry["heterogeneous"] else "consistent"
        lines.append(
            f"| {label(entry['question'])} | {signed(entry['pooled_points'])} | {cells} "
            f"| {entry['q']:.2f} | {probability(entry['p_adjusted'])} "
            f"| {verdict}{marker(entry)} |"
        )
    head = f"Cochran's Q {caption}, df={entries[0]['df']}:"
    return head + "\n\n" + "\n".join(lines) + notes(entries)


def strata_table(payload: dict[str, Any], question: str) -> str:
    """Every stratum one comparison was cut into, with the size each row rests on.

    The size is summed from the row's own four discordance counts rather than taken from the
    panel, because a stratum is a subset and the whole point of the table is that its rows do
    not all rest on the same n -- a row over 869 items and a row over 7 917 read identically
    otherwise.
    """
    lines = [
        f"| stratum | n | {label(question)} | *p* |",
        "|---|---:|---:|---:|",
    ]
    found = False
    flagged = False
    for field, _ in BLOCKS:
        for stratum, entries in (payload.get(field) or {}).items():
            for entry in entries:
                if label(entry["question"]) != label(question):
                    continue
                found = True
                total = (
                    entry["both_right"] + entry["a_only"] + entry["b_only"] + entry["both_wrong"]
                )
                flagged = flagged or not entry["same_arithmetic"]
                lines.append(
                    f"| {stratum} | {total:,} | {signed(entry['delta_points'])}"
                    f"{marker(entry)} | {probability(entry['p_value'])} |"
                )
    if not found:
        raise SystemExit(f"no stratified rows for {question!r} in this payload")
    if flagged:
        # The legend, and only the legend: these p are McNemar's own and uncorrected, because
        # a stratum table decomposes one comparison rather than assembling a family of them.
        lines.append("")
        lines.append(
            "`!` -- the two arms did not demonstrably run the same expert arithmetic, so "
            "the margin carries the dispatch difference as well as the method."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", required=True, type=Path, help="panel_table --json-out")
    parser.add_argument(
        "--strata",
        help="emit the per-stratum rows for one comparison, named exactly as the table names it",
    )
    parser.add_argument(
        "--spread",
        choices=sorted(SPREADS),
        action="append",
        help="emit one Cochran block; repeatable, and all three if omitted",
    )
    args = parser.parse_args(argv)

    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    parts = []
    if args.strata:
        parts.append(strata_table(payload, args.strata))
    for name in args.spread or sorted(SPREADS):
        field, caption = SPREADS[name]
        entries = payload.get(field)
        if not entries:
            print(f"# no {name} spread in this payload -- the panel had one subset, or fewer")
            continue
        parts.append(spread_table(entries, caption))
    print("\n\n".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
