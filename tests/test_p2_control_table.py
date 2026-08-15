"""The phase-2 replicate's control table has to refuse a control that is not one.

A table comparing two GPTQ arms is only a scheme control if the two arms really
differ in the scheme. Two ways that quietly stops being true: an arm gets rerun under
the wrong name, or ``--symmetric no`` reaches the CLI and not the recipe. Both produce
a table that prints cleanly, with six comparisons, a Holm correction, and one
configuration compared against itself in the row the whole run exists to fill.

These tests fail that diff here rather than after the GPU hours. The run they protect
costs a fine-tune, four arms and roughly a day, and a defect in it does not look like
a defect -- it looks like a finding that the grid is worth nothing on this model.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FOUR_POINT = REPO_ROOT / "experiments" / "four_point"


@pytest.fixture(scope="module")
def control_table(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Loaded with ``common`` stubbed out, because importing it reaches the network.

    ``common`` builds the CaseHOLD task at module scope, and ``baselines_table`` --
    which owns ``mcnemar`` and the byte accounting this module reuses -- imports
    ``RUN_DIR`` from it. Stubbing ``common`` and nothing else keeps the statistics
    under test real; stubbing ``baselines_table`` would have replaced them with a
    second copy, which is the failure this suite is about.
    """
    runs = tmp_path_factory.mktemp("runs")
    stub = types.ModuleType("common")
    stub.RUN_DIR = runs  # type: ignore[attr-defined]
    saved = {name: sys.modules.get(name) for name in ("common", "p2_control_table")}
    sys.modules["common"] = stub
    sys.path.insert(0, str(FOUR_POINT))
    try:
        module = importlib.import_module("p2_control_table")
        yield module
    finally:
        sys.path.remove(str(FOUR_POINT))
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def arm(symmetric: bool | None, bits: float, hits: list[bool]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "accuracy": sum(hits) / len(hits),
        "correct": sum(hits),
        "total": len(hits),
        "hits": hits,
        "accounted_bits": bits,
        "nbytes": 1,
    }
    if symmetric is not None:
        record["symmetric"] = symmetric
        record["actorder"] = None
    return record


def test_a_control_pair_that_does_not_straddle_the_flag_is_refused(control_table: Any) -> None:
    """Both arms symmetric is the mislabelled-rerun case, and it must not print.

    Nothing else in the pipeline notices it. The arm names differ, both records are
    well formed, McNemar runs, and the row reports the grid as worth whatever noise
    separates two fits of the same recipe.
    """
    with pytest.raises(SystemExit, match="does not straddle the flag"):
        control_table.check_control_pair(
            arm(True, 3.15, [True, False]), arm(True, 3.15, [True, True])
        )


def test_equal_widths_are_refused_even_when_both_flags_are_right(control_table: Any) -> None:
    """The flag reaching the record is not the flag reaching the quantizer.

    An asymmetric grid stores a zero point per group, so it cannot account to the same
    width as a symmetric one on the same model. Equal widths mean the recipe was built
    from the flag and then not applied -- which is exactly the shape of the bug that
    made ``oneshot`` fit scales without rounding weights, and it reads as a result.
    """
    with pytest.raises(SystemExit, match="did not reach the recipe"):
        control_table.check_control_pair(
            arm(True, 3.15, [True, False]), arm(False, 3.15, [True, True])
        )


def test_a_real_control_pair_is_accepted(control_table: Any) -> None:
    """The negative controls above are only meaningful next to a positive one."""
    control_table.check_control_pair(
        arm(True, 3.1522, [True, False]), arm(False, 3.2891, [True, True])
    )


def test_the_scheme_is_read_off_the_record_not_off_the_name(control_table: Any) -> None:
    assert control_table.scheme_of(arm(True, 3.15, [True])) == "symmetric"
    assert control_table.scheme_of(arm(False, 3.28, [True])) == "asymmetric"
    # DynQuant's arms carry no scheme, because its asymmetry is a property of the
    # quantizer and not a recipe field. Reporting "symmetric" for a missing key would
    # put the wrong label on the one arm the whole panel is about.
    assert control_table.scheme_of(arm(None, 3.01, [True])) == "n/a"


def test_activation_reordering_is_carried_into_the_scheme(control_table: Any) -> None:
    """The Mistral campaign's collapse was act-ordering, not the grid, and a table
    that prints both configurations as "asymmetric" cannot tell the two apart."""
    record = arm(False, 3.28, [True])
    record["actorder"] = "group"
    assert control_table.scheme_of(record) == "asymmetric+group"


def test_the_family_asks_both_halves_of_the_control(control_table: Any) -> None:
    """Dropping either row leaves a panel that cannot answer its own question.

    GPTQ against itself prices the grid. DynQuant against the *asymmetric* GPTQ is the
    claim under the control. A family carrying only DynQuant against the symmetric arm
    is the uncontrolled comparison the original panel already made.
    """
    pairs = {(left, right) for left, right, _ in control_table.FAMILY}
    assert ("stage8_gptq_3b_head_asym", "stage8_gptq_3b_head") in pairs
    assert ("p2_rb_agg", "stage8_gptq_3b_head_asym") in pairs
    assert ("p2_rb_agg", "stage8_gptq_3b_head") in pairs


def test_separation_is_claimed_off_the_corrected_p(control_table: Any) -> None:
    """Six comparisons on one test split is a family whether or not it is called one.

    The fixture puts one comparison just under 0.05 raw. Read uncorrected it separates;
    Holm over six moves it above the line. A diff that sets ``separated`` from
    ``p_value`` turns this red, and that diff is invisible in every other row.
    """
    # 15 discordant, all one way: p = 2 ** -15 * 2 = 6.1e-05 raw, which survives Holm.
    strong_left = [True] * 15 + [True] * 85
    strong_right = [False] * 15 + [True] * 85
    # 5 discordant, all one way: p = 0.0625 raw -- under 0.05 it is not, so the pair
    # below carries the boundary instead.
    weak_left = [True] * 6 + [True] * 94
    weak_right = [False] * 6 + [True] * 94

    data = {
        "stage8_fp16": arm(None, 16.0, strong_left),
        "p2_rb_agg": arm(None, 3.01, weak_left),
        "stage8_gptq_3b_head": arm(True, 3.1522, strong_right),
        "stage8_gptq_3b_head_asym": arm(False, 3.2891, weak_right),
    }
    rows = control_table.comparison_rows(data)
    assert len(rows) == len(control_table.FAMILY)
    for row in rows:
        assert row["p_adjusted"] >= row["p_value"]
        assert row["separated"] is bool(row["p_adjusted"] < 0.05)
        # Both have to survive json.dumps, which refuses the numpy scalars scipy
        # returns. The table writes its JSON on the last line, after every arm has
        # been paid for, so a numpy float here is a whole panel lost at the finish.
        assert type(row["p_value"]) is float
        assert type(row["p_adjusted"]) is float
    # At least one row has to be moved by the correction, or the fixture is not
    # exercising it and this test would pass against an uncorrected table.
    assert any(row["p_adjusted"] > row["p_value"] for row in rows)


def test_a_positive_delta_always_means_the_left_arm_ahead(control_table: Any) -> None:
    """The sign convention is the table's, not the reader's.

    Every row names the left arm first, so a reader never has to work out which
    direction a delta points from which arm happens to be better.
    """
    data = {
        "stage8_fp16": arm(None, 16.0, [True] * 10),
        "p2_rb_agg": arm(None, 3.01, [True] * 8 + [False] * 2),
        "stage8_gptq_3b_head": arm(True, 3.1522, [True] * 6 + [False] * 4),
        "stage8_gptq_3b_head_asym": arm(False, 3.2891, [True] * 7 + [False] * 3),
    }
    rows = {(r["left"], r["right"]): r for r in control_table.comparison_rows(data)}
    assert rows[("p2_rb_agg", "stage8_gptq_3b_head")]["delta_points"] > 0
    assert rows[("stage8_fp16", "p2_rb_agg")]["delta_points"] > 0
    assert rows[("stage8_gptq_3b_head_asym", "stage8_gptq_3b_head")]["delta_points"] > 0


def test_the_panel_it_writes_survives_a_round_trip(control_table: Any) -> None:
    """End to end, because the JSON is written on the last line of the run.

    Everything above tests a function. This tests the artifact, and it is the one that
    catches a type that prints fine and does not serialize -- ``scipy`` hands back
    numpy scalars, ``json.dumps`` refuses them, and the traceback would arrive after
    the fine-tune and all four arms had been paid for.
    """
    import json

    hits = {
        "stage8_fp16": [True] * 90 + [False] * 10,
        "p2_rb_agg": [True] * 88 + [False] * 12,
        "stage8_gptq_3b_head": [True] * 84 + [False] * 16,
        "stage8_gptq_3b_head_asym": [True] * 86 + [False] * 14,
    }
    schemes: dict[str, tuple[bool | None, float]] = {
        "stage8_fp16": (None, 16.0),
        "p2_rb_agg": (None, 3.0102),
        "stage8_gptq_3b_head": (True, 3.1522),
        "stage8_gptq_3b_head_asym": (False, 3.2891),
    }
    for name, (symmetric, bits) in schemes.items():
        record = arm(symmetric, bits, hits[name])
        (control_table.RUNS / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")

    assert control_table.main() == 0
    written = json.loads((control_table.RUNS / "p2_control_table.json").read_text(encoding="utf-8"))
    assert written["replicate"] is True
    assert [row["name"] for row in written["arms"]] == [name for _, name in control_table.ARMS]
    assert len(written["comparisons"]) == len(control_table.FAMILY)
    assert all("p_adjusted" in row for row in written["comparisons"])
