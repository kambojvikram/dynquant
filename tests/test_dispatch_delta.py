"""The dispatch measurement the re-score produces for nothing, and its one dangerous zero.

Three arms keep their expert banks, so re-scoring them on ``--experts-impl eager`` scores
the same weights over the same 12,000 items twice with the dispatch as the only
difference. That is the ``grouped_mm``-against-``eager`` number section 8 of the
packed-MoE report owes, and it falls out of a run that has to happen anyway.

Every failure worth covering here prints a plausible number rather than crashing:

* a re-score whose ``--experts-impl`` never took effect, which pairs a computation with
  itself and reports a delta of zero -- read as "the dispatch is free", which is the
  claim this campaign spent a section retracting;
* an ``--after`` directory that is not the re-scored one, paired anyway;
* two records over different problem sets, paired into a p-value;
* one unusable arm taking the two usable ones down with it.

Nothing here loads a model. The joint hit patterns are laid out by hand, so every count in
the output is a count this file chose.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "phase4" / "dispatch_delta.py"
PANEL = REPO_ROOT / "experiments" / "phase4" / "panel_table.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def delta() -> Any:
    return _load("dispatch_delta", SCRIPT)


def _hits(
    both_right: int, a_only: int, b_only: int, both_wrong: int
) -> tuple[list[bool], list[bool]]:
    """Two hit vectors realising one joint pattern.

    Laid out as a joint distribution rather than as two independent vectors because the
    paired test reads only the discordant cells, and a fixture that sets the two marginals
    without saying how they overlap has not specified the thing under test.
    """
    a = [True] * both_right + [True] * a_only + [False] * b_only + [False] * both_wrong
    b = [True] * both_right + [False] * a_only + [True] * b_only + [False] * both_wrong
    return a, b


def _record(label: str, hits: list[bool], *, ran: str | None, seconds: float) -> dict[str, Any]:
    """One eval record, carrying only what this script reads.

    ``experts`` is omitted entirely rather than set to ``None`` when ``ran`` is ``None``,
    because that is the shape of every record the first LFM2.5 panel wrote -- the field
    did not exist yet -- and the absence is what the ``--before`` side is expected to have.
    """
    record: dict[str, Any] = {
        "label": label,
        "task": "text2sql",
        "backend": "transformers",
        "split": "test",
        "shots": 2,
        "shot_seed": 0,
        "limit": len(hits),
        "decode": {"max_new_tokens": 1024},
        "detail": {"prompt_style": "chat"},
        "accuracy": sum(hits) / len(hits),
        "seconds": seconds,
        "hits": hits,
    }
    if ran is not None:
        record["experts"] = {"found": "grouped_mm", "ran": ran}
    return record


def _write(directory: Path, records: dict[str, dict[str, Any]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for label, record in records.items():
        (directory / f"{label}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return directory


def _panels(tmp_path: Path) -> tuple[Path, Path]:
    """A panel and its re-score: two arms that moved, one baseline that could not.

    ``gptq_4b`` is in both directories untouched, which is what a real re-score leaves
    behind -- the baselines are reused, not re-run -- and it is the arm that proves a
    refusal does not cost the arms beside it.
    """
    before_hits, after_hits = {}, {}
    for label, pattern in (("bf16", (80, 8, 4, 8)), ("dq_4b", (78, 6, 5, 11))):
        before_hits[label], after_hits[label] = _hits(*pattern)
    baseline = [True] * 82 + [False] * 18

    before = _write(
        tmp_path / "panel_grouped_mm",
        {
            "bf16": _record("bf16", before_hits["bf16"], ran=None, seconds=10_307.6),
            "dq_4b": _record("dq_4b", before_hits["dq_4b"], ran=None, seconds=10_011.2),
            "gptq_4b": _record("gptq_4b", baseline, ran=None, seconds=19_805.0),
        },
    )
    after = _write(
        tmp_path / "panel",
        {
            "bf16": _record("bf16", after_hits["bf16"], ran="eager", seconds=17_240.0),
            "dq_4b": _record("dq_4b", after_hits["dq_4b"], ran="eager", seconds=16_880.0),
            "gptq_4b": _record("gptq_4b", baseline, ran=None, seconds=19_805.0),
        },
    )
    return before, after


def _run(delta: Any, before: Path, after: Path, labels: str | None = None) -> str:
    argv = ["--before", str(before), "--after", str(after)]
    if labels is not None:
        argv += ["--labels", labels]
    buffer = StringIO()
    with redirect_stdout(buffer):
        assert delta.main(argv) == 0
    return buffer.getvalue()


def _row(printed: str, label: str) -> str:
    matches = [line for line in printed.splitlines() if line.startswith(label + " ")]
    assert len(matches) == 1, (label, matches)
    return matches[0]


def test_the_delta_is_before_minus_after_and_the_clock_comes_with_it(
    delta: Any, tmp_path: Path
) -> None:
    """The measurement, in the direction the reader needs and with its cost beside it.

    Sign is the whole point of doing this instead of re-reading the token-agreement probe:
    that probe could say the two dispatches compute different things and not which one is
    right. This can, because both sides are scored against the same golds -- so the row
    has to state its direction and mean it. Before minus after, before being the dispatch
    the panel used and after being the one a download runs.

    The seconds column rides along because the weights are identical across the pair,
    which makes it the dispatch cost with no dequantization confounded into it -- the one
    thing section 8's 1.9-2.3x could not separate.

    Turns red when: the subtraction flips, the arms scored on one dispatch stop being
    column A, or the timing pair stops being printed and the free half of the measurement
    is silently dropped.
    """
    before, after = _panels(tmp_path)

    printed = _run(delta, before, after)
    row = _row(printed, "bf16")
    assert "grouped_mm -> eager" not in row, "the before side of this panel recorded nothing"
    assert "unrecorded -> eager" in row
    assert "88.00%" in row and "84.00%" in row
    assert "+4.00" in row, "before minus after: the panel's dispatch scored higher"
    assert "10,308 -> 17,240 (1.67x)" in row
    assert "delta is before minus after" in printed


def test_a_re_score_that_did_not_move_the_dispatch_is_refused_not_reported_as_zero(
    delta: Any, tmp_path: Path
) -> None:
    """The reason this file exists.

    If ``--experts-impl`` silently fails to take effect, the re-score scores the same
    computation twice. The two hit vectors are then identical, the delta is exactly zero,
    the p-value is 1.0, and the table reads as a clean demonstration that the dispatch
    costs nothing -- which is the claim four places in this codebase asserted on a
    one-layer measurement and had to retract. A zero produced by the fix not having run is
    indistinguishable, from the output alone, from a zero that is true.

    So it is refused, and the refusal names the dispatch both sides ran.

    Turns red when: the equal-dispatch case starts producing a row, or the refusal stops
    saying which dispatch was on both sides and an operator cannot tell whether the flag
    was ignored or the copy was taken after the re-score instead of before.
    """
    before, after = _panels(tmp_path)
    for directory in (before, after):
        path = directory / "bf16.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["experts"] = {"found": "grouped_mm", "ran": "grouped_mm"}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    printed = _run(delta, before, after)
    row = _row(printed, "bf16")
    assert "both records ran 'grouped_mm'" in row
    assert "no dispatch difference to measure" in row
    assert "%" not in row, "no accuracy, so no zero to misread"
    assert "0.00" not in row


def test_an_after_record_that_never_learned_the_field_is_refused(
    delta: Any, tmp_path: Path
) -> None:
    """Pointing this at two copies of the same panel is a mistake with a plausible output.

    Both directories then hold records written before ``dynquant eval`` wrote ``experts``,
    both sides say ``None``, and every pair is a vector against itself. The equal-dispatch
    refusal above would not catch it, because ``None == None`` is agreement on nothing
    rather than agreement on a dispatch -- so the absence is refused first and by its own
    message, which says the after side was not re-scored rather than that the two agree.

    Turns red when: an unrecorded after side starts pairing, which would make the most
    likely operator error the one that prints the most convincing table.
    """
    before, after = _panels(tmp_path)
    path = after / "dq_4b.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["experts"]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    printed = _run(delta, before, after)
    row = _row(printed, "dq_4b")
    assert "does not say what dispatch it ran" in row
    assert "was not re-scored" in row


def test_records_over_different_problem_sets_are_refused_and_the_field_is_named(
    delta: Any, tmp_path: Path
) -> None:
    """The ordinary refusal, kept for the ordinary reason.

    A dispatch difference does not stop a pairing; a ``--limit`` or a decode-budget
    difference does, because then element *i* of the two vectors is not the same item.
    ``problem_set_difference`` draws exactly that line and this script inherits it rather
    than restating it, which is what stops the two copies of the rule from drifting apart.

    Turns red when: the check goes back to comparing whole ``_comparability`` dicts, which
    would refuse every pair here for differing on the dispatch -- the one difference the
    script is looking for.
    """
    before, after = _panels(tmp_path)
    path = after / "bf16.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["decode"]["max_new_tokens"] = 320
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    printed = _run(delta, before, after)
    row = _row(printed, "bf16")
    assert "not the same problem set: decode.max_new_tokens" in row
    assert "experts.ran" not in row, "the dispatch difference is the point, not a reason to refuse"


def test_a_refused_arm_does_not_take_the_arms_beside_it_down(delta: Any, tmp_path: Path) -> None:
    """A reused baseline is in both directories and can never be measured. That is fine.

    The four ``llm-compressor`` arms have no batched bank left, so they are reused rather
    than re-scored and both copies of their record are the same file. They will always
    refuse. If that refusal were an exception the script would be unusable on the only
    directory pair it will ever be pointed at, and a panel where two of three arms moved
    is still two measurements.

    Turns red when: a refusal becomes fatal, or the Holm multiplier starts counting
    refused rows -- three tests' worth of correction applied to two tests.
    """
    before, after = _panels(tmp_path)

    printed = _run(delta, before, after)
    assert "does not say what dispatch it ran" in _row(printed, "gptq_4b")
    assert "+4.00" in _row(printed, "bf16")
    assert "%" in _row(printed, "dq_4b")

    rows = delta.compare(before, after, None)
    measured = [row for row in rows if row["why_not"] is None]
    assert [row["label"] for row in measured] == ["bf16", "dq_4b"]

    # Recomputed here over the two that paired. If the refused baseline were counted the
    # multiplier would be three and every adjusted p in the table would be wrong by a
    # third -- which is a correction nobody would notice being applied to a comparison
    # that was never made.
    holm = _load("panel_table", PANEL).holm
    for row, adjusted in zip(measured, holm([r["paired"].p_value for r in measured]), strict=True):
        assert f"{adjusted:9.4f}" in _row(printed, row["label"])


def test_two_directories_with_no_arm_in_common_is_an_error_and_says_which_two(
    delta: Any, tmp_path: Path
) -> None:
    """Empty output would read as "nothing moved". It means "you pointed at two panels".

    Turns red when: the empty case starts printing a table with no rows, which is the
    same screen a run where every arm refused would produce and means something else
    entirely.
    """
    before, _ = _panels(tmp_path)
    other = _write(
        tmp_path / "elsewhere",
        {"phi4_4b": _record("phi4_4b", [True] * 8, ran="eager", seconds=1.0)},
    )

    with pytest.raises(SystemExit) as caught:
        _run(delta, before, other)
    assert "no arm is scored in both" in str(caught.value)
    assert str(before) in str(caught.value)
