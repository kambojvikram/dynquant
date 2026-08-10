"""The formatter must read the payload and nothing else.

`report_tables.py` exists so §13's markdown stops being typed out of a terminal block. That is
only worth having if the tables it prints are the payload's own numbers -- a formatter that
rounded, re-derived or reordered anything would replace a transcription error with a quieter
one, and the report would then carry it under the authority of having been generated.

So these tests assert printed cells against the payload they came from, and assert that the
payload fields the formatter reads are fields `panel_table` actually writes.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "phase4" / "report_tables.py"
PANEL_TABLE = REPO_ROOT / "experiments" / "phase4" / "panel_table.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tables() -> Any:
    return _load("_dq_report_tables", SCRIPT)


def _mcnemar(
    question: str,
    delta: float,
    p: float,
    counts: tuple[int, int, int, int],
    *,
    same_arithmetic: bool = True,
) -> dict[str, Any]:
    both_right, a_only, b_only, both_wrong = counts
    return {
        "question": question,
        "delta_points": delta,
        "p_value": p,
        "both_right": both_right,
        "a_only": a_only,
        "b_only": b_only,
        "both_wrong": both_wrong,
        "same_arithmetic": same_arithmetic,
    }


def _spread(
    question: str,
    pooled: float,
    sources: dict[str, float],
    q: float,
    df: int,
    p: float,
    *,
    same_arithmetic: bool = True,
    family: tuple[int, int] = (6, 6),
) -> dict[str, Any]:
    corrected, declared = family
    return {
        "question": question,
        "pooled_points": pooled,
        "sources": sources,
        "q": q,
        "df": df,
        "p_adjusted": p,
        "heterogeneous": p < 0.05,
        "same_arithmetic": same_arithmetic,
        "holm_corrected": corrected,
        "holm_family": declared,
    }


# Deliberately unequal strata, unequal deltas, and p spanning both formatting regimes. A
# fixture whose rows all looked alike would pass a formatter that printed any one of them.
PAYLOAD: dict[str, Any] = {
    "head_to_head_by_difficulty": {
        "ceiling-right": [
            _mcnemar("A vs B", 1.1769, 1.3768e-06, (9373, 360, 241, 137), same_arithmetic=False)
        ],
        "ceiling-wrong": [
            _mcnemar("A vs B", -2.2234, 0.00966, (0, 402, 444, 1043), same_arithmetic=False)
        ],
    },
    "head_to_head_by_source_and_difficulty": {
        "wikisql/ceiling-right": [_mcnemar("A vs B", 1.5789, 1.26e-09, (7400, 300, 175, 42))],
        "wikisql/ceiling-wrong": [_mcnemar("A vs B", -3.2353, 0.00993, (0, 220, 253, 547))],
    },
    # The difficulty block is flagged and short; the crossed block is neither. Both conditions
    # print once per block, so a fixture where every block carried them could not tell a
    # formatter that reads the payload from one that prints them unconditionally.
    "difficulty_heterogeneity": [
        _spread(
            "A vs B",
            0.9153,
            {"ceiling-right": 1.1769, "ceiling-wrong": -2.2234},
            15.1686,
            1,
            0.000295,
            same_arithmetic=False,
            family=(2, 6),
        ),
        _spread(
            "C vs D",
            0.3400,
            {"ceiling-right": 0.3900, "ceiling-wrong": -0.1600},
            0.3200,
            1,
            0.570,
            family=(2, 6),
        ),
    ],
    "source_and_difficulty_heterogeneity": [
        _spread(
            "A vs B",
            1.0381,
            {
                "gretel/ceiling-right": -0.2735,
                "gretel/ceiling-wrong": -1.0357,
                "wikisql/ceiling-right": 1.5789,
                "wikisql/ceiling-wrong": -3.2353,
            },
            24.7002,
            3,
            5.3512e-05,
        ),
    ],
    "source_heterogeneity": [
        _spread("A vs B", 0.7290, {"gretel": -0.4897, "wikisql": 1.0294}, 6.3178, 1, 0.03586),
    ],
}


def _run(tables: Any, payload: dict[str, Any], tmp_path: Path, *argv: str) -> str:
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    buffer = StringIO()
    with redirect_stdout(buffer):
        assert tables.main(["--payload", str(path), *argv]) == 0
    return buffer.getvalue()


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _rows(printed: str, width: int) -> list[list[str]]:
    """Table rows only, so a legend line under the table cannot be indexed as if it were one."""
    rows = [_cells(line) for line in printed.splitlines() if line.startswith("| ")]
    return [row for row in rows if len(row) == width and not row[0].startswith("---")]


def _strata_only(printed: str) -> str:
    """--strata alone also emits every spread block, and those carry a legend of their own."""
    head, _, _ = printed.partition("Cochran")
    assert head != printed, "the spread blocks stopped printing, so this split guards nothing"
    return head


def test_every_spread_cell_is_the_payloads_own_number(tables: Any, tmp_path: Path) -> None:
    """Turns red when: the formatter rounds, rescales or re-derives a spread it was handed.

    Including the header, which names the subsets the payload carries rather than a list this
    file keeps -- that is what lets a wider partition widen the table instead of silently
    dropping its extra cells into a column with no name over them.
    """
    printed = _run(tables, PAYLOAD, tmp_path, "--spread", "crossed")
    entry = PAYLOAD["source_and_difficulty_heterogeneity"][0]
    subsets = sorted(entry["sources"])

    header = next(line for line in printed.splitlines() if line.startswith("| comparison"))
    assert _cells(header)[2] == " / ".join(subsets), header

    row = next(line for line in printed.splitlines() if line.startswith("| A vs B"))
    cells = _cells(row)
    assert cells[1] == f"{entry['pooled_points']:+.2f}"
    assert cells[2] == " / ".join(f"{entry['sources'][name]:+.2f}" for name in subsets)
    assert cells[3] == f"{entry['q']:.2f}"
    assert cells[4] == f"{entry['p_adjusted']:.3g}"
    assert cells[5] == "heterogeneous"
    assert f"df={entry['df']}" in printed


def test_a_consistent_row_is_not_called_heterogeneous(tables: Any, tmp_path: Path) -> None:
    """Turns red when: the verdict stops following the payload's own flag.

    Both rows of this block carry a real Q and a real p; only the flag separates them, which is
    the point -- a verdict recomputed from a threshold living in the formatter would be a
    second copy of a decision the panel already made.
    """
    printed = _run(tables, PAYLOAD, tmp_path, "--spread", "difficulty")
    verdicts = {
        _cells(line)[0]: _cells(line)[5]
        for line in printed.splitlines()
        if line.startswith("| A vs B") or line.startswith("| C vs D")
    }
    assert verdicts == {"A vs B": "heterogeneous !", "C vs D": "consistent"}


def test_the_confound_flag_survives_into_the_markdown(tables: Any, tmp_path: Path) -> None:
    """Turns red when: a flagged comparison prints as an unflagged one.

    The rows that carry this flag on the real panel are the headline rows -- bf16 and dq load
    one checkpoint while gptq and awq load llm-compressor outputs, and the two dispatches
    disagree with each other on 1.24% of teacher-forced tokens, which is a large fraction of
    the margin the row reports. A generated table that dropped it would state the finding more
    confidently than the hand-typed one it replaces, which is the wrong direction for a tool
    whose entire justification is that generated numbers are more trustworthy.
    """
    printed = _run(tables, PAYLOAD, tmp_path, "--spread", "difficulty")
    flagged = {row[0] for row in _rows(printed, 6) if row[-1].endswith("!")}
    assert flagged == {"A vs B"}, printed
    assert "did not demonstrably run the same expert arithmetic" in printed


def test_a_block_with_nothing_to_qualify_prints_no_qualifications(
    tables: Any, tmp_path: Path
) -> None:
    """Turns red when: the legend and the family note are printed unconditionally.

    Both are conditions on reading the rows above them. A note under every block is a note a
    reader stops reading, and "Holm-adjusted over 6 of 6" under a finished panel would say the
    opposite of what the warning exists to say.
    """
    printed = _run(tables, PAYLOAD, tmp_path, "--spread", "crossed")
    assert "!" not in printed, printed
    assert "same expert arithmetic" not in printed
    assert "short family" not in printed


def test_a_short_family_says_how_short(tables: Any, tmp_path: Path) -> None:
    """Turns red when: adjusted p from a half-run panel are pasted with no note of the family.

    Holm's multiplier is the number of comparisons actually corrected, so the same row over
    three comparisons and over six is two different claims -- on this panel's five-arm run the
    one heterogeneous row read 0.0359 and 0.0717. That is the verdict, not a decimal place, and
    a row pasted into a report travels without the block's closing warning.
    """
    printed = _run(tables, PAYLOAD, tmp_path, "--spread", "difficulty")
    entry = PAYLOAD["difficulty_heterogeneity"][0]
    assert f"over {entry['holm_corrected']} of {entry['holm_family']} comparisons" in printed


def test_the_flag_marks_the_strata_that_carry_it_and_not_the_rest(
    tables: Any, tmp_path: Path
) -> None:
    """Turns red when: the strata table flags every row, or none, instead of the flagged ones.

    A per-stratum decomposition can mix them -- the difficulty cut and the crossed cut are read
    off different blocks -- so a legend under a table whose rows are unmarked, or a mark on
    every row, both destroy the only information the flag carries.
    """
    strata = _strata_only(_run(tables, PAYLOAD, tmp_path, "--strata", "A vs B"))
    marked = {row[0] for row in _rows(strata, 4) if row[2].endswith("!")}
    assert marked == {"ceiling-right", "ceiling-wrong"}, strata
    assert "same expert arithmetic" in strata

    clean = {
        field: {
            stratum: [{**entry, "same_arithmetic": True} for entry in entries]
            for stratum, entries in PAYLOAD[field].items()
        }
        for field, _ in tables.BLOCKS
    }
    strata = _strata_only(_run(tables, {**PAYLOAD, **clean}, tmp_path, "--strata", "A vs B"))
    assert "!" not in strata, strata
    assert "same expert arithmetic" not in strata


def test_each_stratum_carries_the_size_it_rests_on(tables: Any, tmp_path: Path) -> None:
    """Turns red when: n is taken from the panel instead of summed from the row's own counts.

    The four strata here are four different sizes, as the real ones are -- 10 111 against
    1 020. A formatter reading a panel-wide total would print one number four times and every
    row would still look right on its own.
    """
    printed = _run(tables, PAYLOAD, tmp_path, "--strata", "A vs B")
    sizes = {}
    for line in printed.splitlines():
        cells = _cells(line)
        if len(cells) == 4 and cells[0] not in {"stratum", "---"}:
            sizes[cells[0]] = cells[1]

    expected = {}
    for field in ("head_to_head_by_difficulty", "head_to_head_by_source_and_difficulty"):
        for stratum, entries in PAYLOAD[field].items():
            entry = entries[0]
            total = entry["both_right"] + entry["a_only"] + entry["b_only"] + entry["both_wrong"]
            expected[stratum] = f"{total:,}"

    assert sizes == expected
    assert len(set(sizes.values())) == len(sizes), f"fixture does not discriminate: {sizes}"


def test_the_padding_the_panel_prints_with_does_not_reach_the_markdown(
    tables: Any, tmp_path: Path
) -> None:
    """Turns red when: a name prints one way and has to be typed another.

    `panel_table` pads its question strings to a fixed terminal column, so they arrive with
    interior runs of spaces. Printing them raw puts "4b  DynQuant vs GPTQ" in a cell; matching
    them raw means --strata only works if the padding is reproduced exactly, which is both
    invisible in the terminal and impossible to guess from the rendered table.
    """
    padded = json.loads(json.dumps(PAYLOAD).replace("A vs B", "4b  A vs B"))
    printed = _strata_only(_run(tables, padded, tmp_path, "--strata", "4b A vs B"))
    assert "4b A vs B" in printed
    assert "4b  A vs B" not in printed

    spread = _run(tables, padded, tmp_path, "--spread", "difficulty")
    assert "| 4b A vs B |" in spread, spread


def test_a_comparison_the_payload_does_not_carry_is_refused(tables: Any, tmp_path: Path) -> None:
    """Turns red when: an unmatched --strata prints a header with no rows under it.

    An empty table is the failure mode that reaches a report, because it reads as a finding --
    "this comparison was cut into no strata" -- rather than as a typo in a flag.
    """
    with pytest.raises(SystemExit, match="no stratified rows"):
        _run(tables, PAYLOAD, tmp_path, "--strata", "E vs F")


def test_a_block_the_panel_did_not_produce_is_skipped_not_faked(
    tables: Any, tmp_path: Path
) -> None:
    """Turns red when: a null block prints an empty table instead of saying it is absent.

    A single-source panel has no crossed block at all, and `panel_table` writes null for it
    rather than an empty list. Formatting that as a header with no rows would put a partition
    into the report that was never measured.
    """
    payload = {**PAYLOAD, "source_and_difficulty_heterogeneity": None}
    printed = _run(tables, payload, tmp_path, "--spread", "crossed")
    assert "no crossed spread" in printed
    assert "| comparison |" not in printed


def test_the_fields_the_formatter_reads_are_fields_the_panel_writes(tables: Any) -> None:
    """Turns red when: a payload key is renamed in panel_table and not here.

    The formatter is a second consumer of one payload, which is a check on that payload only
    while both agree on its names. Nothing else in the suite would notice them diverging: the
    formatter would print "no spread in this payload" for every block and exit 0.
    """
    source = PANEL_TABLE.read_text(encoding="utf-8")
    emitted = set(re.findall(r'"([a-z0-9_]+)":', source))
    wanted = {field for field, _ in tables.SPREADS.values()} | {f for f, _ in tables.BLOCKS}
    missing = sorted(wanted - emitted)
    assert not missing, f"{missing} are read from the payload and never written into it"
