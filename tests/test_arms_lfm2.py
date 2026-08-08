"""The seven-arm panel has to be right before it runs, because running it is seven hours.

`baselines_lfm2` is one arm and this is the thing that spends all of them, so the failures
worth covering here are the ones that survive the run and enter the table:

* an anchor taken from DynQuant's format rather than the baselines', which hands the arm
  whose accuracy is the claim 2.3% more bytes than the arms it is compared against;
* a DynQuant arm that lands under its ceiling and is reported as byte-matched anyway;
* a map written under one key and read under another, so the eval scores a budget nobody
  planned;
* and a scoring contract that differs between arm kinds in a field the pairing check reads.

Nothing here loads a model, allocates anything, or touches a GPU. Every function under test
is a plan, a command line, or a comparison of two integers.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dynquant.commands.evaluate import DECODE_PAIRING_FIELDS, PAIRING_FIELDS
from dynquant.quant.pack import stored_bits

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "experiments" / "phase4" / "arms_lfm2.py"


@pytest.fixture(scope="module")
def arms() -> Any:
    spec = importlib.util.spec_from_file_location("_dq_arms_lfm2", DRIVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dq_arms_lfm2"] = module
    spec.loader.exec_module(module)
    return module


RUN = [
    "run",
    "--model",
    "/runs/s4/merged",
    "--stats",
    "/runs/s4/dynquant_stats.json",
    "--out",
    "/runs/s4/arms",
]


def _args(arms: Any, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse through the driver's own parser, so a renamed flag fails here."""
    return arms.build_parser().parse_args(argv or RUN)  # type: ignore[no-any-return]


# --- who sets the budget --------------------------------------------------------------


@pytest.mark.parametrize(
    ("bits", "dynquant", "baselines"), [(4, 4.25, 4.15625), (3, 3.25, 3.1484375)]
)
def test_the_two_formats_do_not_cost_the_same_and_the_pin_is_not_a_no_op(
    bits: int, dynquant: float, baselines: float
) -> None:
    """Both accountings, on one row, so the size the pin removes is a measured number.

    This is the premise the whole panel rests on and it is easy to assume away. DynQuant
    writes an fp16 scale *and* an fp16 offset per group, 32 bits; ``compressed-tensors``
    writes an fp16 scale and a zero point packed at the weight's own width, ``16 + bits``.
    At a group of 128 that is 4.25 bits per parameter against 4.15625 -- so anchoring the
    panel on DynQuant's own uniform arm would let it spend **2.3% more bytes** than the
    methods it is being compared against, inside the arm whose accuracy is the claim.

    Pinned against literals rather than recomputed from the same expressions the source
    uses, because a test that re-derives both sides passes when both sides change together
    -- which is exactly the change that would silently move the anchor.

    Turns red when: either format's per-group overhead changes. That is a real event and it
    should stop here, because the report quotes these two numbers.
    """
    row = 128  # one group, so the per-parameter cost is the whole story
    assert stored_bits(1, row, bits, group_size=128) / row == dynquant
    assert (row * bits + (row // 128) * (16 + bits)) / row == baselines
    assert dynquant > baselines


def test_the_dynquant_arms_are_pinned_to_the_baselines_byte_count(arms: Any) -> None:
    """Every arm at a width shares one target, and it came from the baselines.

    The grouping is the claim "at matched bytes", expressed before anything runs. A panel
    where each method requested its own natural size would still produce a table, and the
    table would read as an accuracy comparison.

    Turns red when: an arm at a width gets a target that is not its group's, or the widths
    stop being drawn from one budget map.
    """
    budgets = {4: 4_399_629_312, 3: 3_332_904_576}

    planned = arms.plan_arms(budgets)

    for width, budget in budgets.items():
        at_width = [arm for arm in planned if arm.anchor == width]
        assert [arm.kind for arm in at_width] == ["gptq", "awq", "dq"]
        assert {arm.target_bytes for arm in at_width} == {budget}


def test_the_ceiling_comes_first_and_is_the_only_arm_without_a_budget(arms: Any) -> None:
    """Order is a cost decision, not a style one.

    The unquantized arm is the one that can fail for a reason that has nothing to do with
    quantization -- a bad merge, a tokenizer that does not round-trip, a decode budget that
    truncates every answer. Six quantization passes before discovering that is six passes
    thrown away. It is also the only arm with no target, because there is nothing to match
    it to: it is what the others are measured against.

    Turns red when: the ceiling moves out of first place, or acquires a budget.
    """
    planned = arms.plan_arms({4: 4_399_629_312, 3: 3_332_904_576})

    assert planned[0].kind == "ceiling"
    assert [arm.target_bytes for arm in planned].count(None) == 1
    assert planned[0].target_bytes is None
    assert [arm.label for arm in planned[1:]] == [
        "gptq_4b",
        "awq_4b",
        "dq_4b",
        "gptq_3b",
        "awq_3b",
        "dq_3b",
    ]


# --- the scoring contract -------------------------------------------------------------


def test_every_arm_is_scored_under_the_same_contract_except_its_name(arms: Any) -> None:
    """Seven records that differ in one field, and that field is the label.

    ``eval_flags`` exists so the shot count, the shot seed, the prompt style and the decode
    budget cannot drift between arm kinds. They are what make two hit vectors describe the
    same 400 problems; a panel where the GPTQ arms saw two shots and the DynQuant arms saw
    three is not a comparison of quantization.

    Turns red when: a per-arm-kind override appears, or a field is dropped from one caller.
    """
    args = _args(arms)

    ceiling = arms.eval_flags(args, "bf16")
    quantized = arms.eval_flags(args, "dq_3b")

    assert ceiling[:2] == ["--label", "bf16"]
    assert quantized[:2] == ["--label", "dq_3b"]
    assert ceiling[2:] == quantized[2:]


def test_the_pairing_fields_are_pinned_on_the_command_line_or_by_the_command(arms: Any) -> None:
    """What ``--compare`` refuses to pair across has to be decided here, not per arm.

    ``task`` and ``backend`` are fixed by the command itself -- every arm runs
    ``eval --task text2sql`` on the transformers path -- so they cannot drift and are not
    flags. The rest are, and are passed explicitly rather than left to the eval parser's
    default: the record writes what was *resolved*, so an arm whose split was inherited
    records the same value under a different provenance and pairs by luck.

    ``limit`` is the one that may legitimately be absent. Absent means every arm inherits
    the same task default, which is what pairing needs; what would break it is one arm
    carrying the flag and another not, so the assertion is on it moving for all or none.

    ``max_new_tokens`` used to be in that category and is not. Inheriting is only safe when
    there is one default to inherit, and there are two: the CLI's task spec answers 320
    while the in-process chat config answers 384. The panel therefore states its budget on
    every command, and the number is the one the ceiling was measured at.

    Turns red when: a new pairing field is added to the eval contract and no flag here
    carries it, one of these stops being passed, or the decode budget goes back to being
    inherited.
    """
    flags = arms.eval_flags(_args(arms), "dq_4b")

    fixed_by_the_command = {"task", "backend"}
    optional = {"limit"}
    for name in (*PAIRING_FIELDS, *DECODE_PAIRING_FIELDS):
        if name in fixed_by_the_command or name in optional:
            continue
        assert f"--{name.replace('_', '-')}" in flags, f"{name} is left to the eval default"

    assert flags[flags.index("--max-new-tokens") + 1] == "1024"
    for kind in ("bf16", "gptq_4b", "dq_3b"):
        assert arms.eval_flags(_args(arms), kind).count("--max-new-tokens") == 1

    with_limit = arms.eval_flags(_args(arms, [*RUN, "--limit", "400"]), "dq_4b")
    assert "--limit" in with_limit
    assert "--limit" not in flags


# --- matched bytes --------------------------------------------------------------------


def test_an_arm_that_missed_its_budget_is_refused_rather_than_reported(arms: Any) -> None:
    """Under the ceiling is still off the anchor, and it is the direction that happens.

    ``--target-size`` is a ceiling: an allocator that cannot spend the last bits lands
    *below* it, and a signed check would let a smaller arm through while catching an
    impossible larger one. A percent is not a rounding difference at 4.4 GB -- it is 44 MB
    of extra weights, which reads as accuracy.

    Turns red when: the comparison stops taking an absolute value, the tolerance loosens, or
    the breach downgrades to a warning.
    """
    anchor = 4_399_629_312
    tolerated = int(anchor * (1 - arms.MATCH_TOLERANCE / 2))
    breached = int(anchor * (1 - arms.MATCH_TOLERANCE * 2))

    arms.check_matched(arms.Arm("dq_4b", "dq", 4, anchor, nbytes=tolerated))

    with pytest.raises(SystemExit, match="off its anchor"):
        arms.check_matched(arms.Arm("dq_4b", "dq", 4, anchor, nbytes=breached))

    # The ceiling has neither number and must not be compared to itself.
    arms.check_matched(arms.Arm("bf16", "ceiling", None, None))


def test_the_realised_size_is_read_back_from_the_map_not_taken_from_the_request(
    arms: Any, tmp_path: Path
) -> None:
    """``target_bytes`` is what was asked for; this is what the allocator spent.

    Taking the request as the answer would make :func:`check_matched` a tautology -- it
    would compare a number against itself, print a reassuring ``+0 B``, and never fire. The
    map below is deliberately short of its key by more than the tolerance, which is the
    state the whole guard exists to catch.

    Turns red when: ``nbytes`` starts coming from the arm instead of the file.
    """
    save_map = tmp_path / "dq_4b.json"
    save_map.write_text(
        json.dumps({"maps": {"4399629312": {"nbytes": 4_300_000_000, "bits": {}}}}),
        encoding="utf-8",
    )

    assert arms.map_nbytes(save_map, "4399629312") == 4_300_000_000

    with pytest.raises(SystemExit, match="different budget"):
        arms.map_nbytes(save_map, "3332904576")


def test_the_map_is_written_and_read_under_the_same_key(arms: Any) -> None:
    """``inspect`` keys the map on the raw ``--target-size`` string it was handed.

    Not on a parsed byte count, not on a normalised unit -- the literal argument. So
    ``--target-size 4.1GiB`` and ``--target-size 4402341478`` produce different keys for the
    same allocation, and the two commands here are the only place that coupling is visible.
    A mismatch is not a crash at allocation time; it is a crash at eval time, after the
    allocation has already run.

    Turns red when: either command formats the budget differently, or the eval starts
    guessing the key from the map's single entry.
    """
    args = _args(arms)
    arm = arms.Arm("dq_3b", "dq", 3, 3_332_904_576)

    inspect = arms.dq_inspect_cmd(args, arm, Path("/maps/dq_3b.json"))
    evaluate = arms.dq_eval_cmd(args, arm, Path("/maps/dq_3b.json"), Path("/out/dq_3b.json"))

    written = inspect[inspect.index("--target-size") + 1]
    read = evaluate[evaluate.index("--map-key") + 1]
    assert written == read == "3332904576"


def test_a_dynquant_arm_is_scored_through_its_map_and_never_through_a_written_copy(
    arms: Any,
) -> None:
    """No ``quantize`` step, because a decoded copy is 16.9 GB of disk that means nothing.

    ``dynquant quantize`` writes bf16 -- identical numerics to ``eval --map``, at fp16 size,
    in a directory named for a 3-bit arm. Six of those is 100 GB on a box with 102 GB free,
    and the first reader to check a folder size gets the wrong footprint for the arm.

    Turns red when: the DynQuant path grows a materialisation step, or the eval stops
    passing the map.
    """
    args = _args(arms)
    arm = arms.Arm("dq_4b", "dq", 4, 4_399_629_312)

    evaluate = arms.dq_eval_cmd(args, arm, Path("/maps/dq_4b.json"), Path("/out/dq_4b.json"))

    assert "quantize" not in evaluate
    assert evaluate[evaluate.index("-m") + 1 : evaluate.index("-m") + 3] == ["dynquant", "eval"]
    assert "--map" in evaluate


# --- one stack ------------------------------------------------------------------------


def test_the_panel_refuses_an_interpreter_that_cannot_run_the_baselines(
    arms: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every arm runs under one Python, and it is checked before the ceiling, not at GPTQ.

    ``sys.executable`` drives all seven subprocesses so the panel cannot be scored by two
    transformers versions -- a difference in the instrument, reported as a difference
    between methods. The box has two environments and only one has llm-compressor, so the
    wrong launch is a live possibility rather than a hypothetical.

    The check has to happen before the first arm: the ceiling is an hour of generation, and
    failing after it is an hour spent to learn something knowable at startup.

    Turns red when: the check moves below the arm loop, or the baselines are given their own
    interpreter.
    """
    monkeypatch.setattr(arms.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(SystemExit, match="not importable"):
        arms.require_one_stack()

    monkeypatch.setattr(arms.importlib.util, "find_spec", lambda name: object())
    arms.require_one_stack()


def test_every_subprocess_is_this_interpreter(arms: Any) -> None:
    """Not ``python``, not the console script -- the interpreter that is running.

    ``dynquant`` on ``PATH`` is whichever environment was activated last, which on a box
    with two of them is a coin flip; and there is no bare ``python`` on this one at all. The
    baselines are invoked by file path for the same reason: ``experiments/`` is not a
    package and an import would depend on the working directory.

    Turns red when: any command hardcodes an interpreter name or reaches for the entry point.
    """
    args = _args(arms)
    commands = [
        arms.ceiling_cmd(args, arms.Arm("bf16", "ceiling", None, None), Path("/out/bf16.json")),
        arms.baseline_cmd(args, arms.Arm("gptq_4b", "gptq", 4, 1), Path("/out/gptq_4b.json")),
        arms.dq_inspect_cmd(args, arms.Arm("dq_4b", "dq", 4, 1), Path("/maps/dq_4b.json")),
        arms.dq_eval_cmd(args, arms.Arm("dq_4b", "dq", 4, 1), Path("/m.json"), Path("/o.json")),
    ]

    for command in commands:
        assert command[0] == sys.executable
    assert commands[1][1].endswith("baselines_lfm2.py")


# --- resume ----------------------------------------------------------------------------
#
# `--resume` exists because the panel is seven hours and a crash in arm six should not
# re-spend arms one through five. It is also the only path where a record enters the
# manifest without this run having produced it, so both failures below are its alone.


def _record(**overrides: Any) -> dict[str, Any]:
    """A record shaped like the one `dynquant eval` writes, in the fields that pair."""
    record = {
        "task": "text2sql",
        "backend": "torch",
        "split": "test",
        "shots": 2,
        "shot_seed": 0,
        "limit": 400,
        "decode": {"max_new_tokens": 384, "batch_size": 8},
        "detail": {"prompt_style": "chat", "by_source": {}},
        "accuracy": 0.5,
        "hits": [1, 0],
    }
    record.update(overrides)
    return record


def _panel(arms: Any, tmp_path: Path, records: dict[str, dict[str, Any]]) -> list[Any]:
    """Arms carrying written records, as `do_run` leaves them before the manifest."""
    built = []
    for label, payload in records.items():
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        arm = arms.Arm(label, "ceiling" if label == "bf16" else "dq", None, None)
        arm.record = str(path)
        built.append(arm)
    return built


def test_two_arms_scored_on_different_problem_sets_are_refused(arms: Any, tmp_path: Path) -> None:
    """A reused record's only claim to provenance is its filename, so the records are read.

    ``eval_flags`` makes the seven commands identical, which proves nothing about a record
    this run did not write. The case below is the realistic one: an arm kept from a 200-item
    smoke run resumed into a 400-item panel. Both files are valid, both carry ``hits``, and
    the McNemar that pairs them would silently compare two different problem sets.

    Refused at run time rather than at compare time because that is where the operator can
    still act -- deleting one record costs one arm, discovering it later costs the panel.

    Turns red when: the check stops reading the files, or only checks resumed arms.
    """
    panel = _panel(
        arms,
        tmp_path,
        {"bf16": _record(), "dq_4b": _record(), "dq_3b": _record(limit=200)},
    )

    with pytest.raises(SystemExit, match="dq_3b was not scored under the same settings as bf16"):
        arms.check_pairable(panel)


def test_the_arms_may_differ_in_everything_the_pairing_does_not_read(
    arms: Any, tmp_path: Path
) -> None:
    """Read through the eval's own ``_comparability``, not by diffing the records.

    Every arm's record legitimately differs -- accuracy, runtime, the quantized size, the
    batch size that fit in VRAM. A whole-record comparison would fire on all seven runs of
    a correct panel, which is a guard nobody keeps. Delegating to the eval command's own
    flattener is also what keeps this from being a second copy of the contract: a field
    added to ``PAIRING_FIELDS`` reaches this check without anyone editing it.

    Turns red when: the check compares records directly, or reimplements the field list.
    """
    panel = _panel(
        arms,
        tmp_path,
        {
            "bf16": _record(accuracy=0.61, decode={"max_new_tokens": 384, "batch_size": 8}),
            "dq_4b": _record(accuracy=0.58, decode={"max_new_tokens": 384, "batch_size": 2}),
        },
    )

    arms.check_pairable(panel)

    assert "batch_size" not in PAIRING_FIELDS
    assert "max_new_tokens" in DECODE_PAIRING_FIELDS


def test_an_arm_asked_a_different_kind_of_question_is_refused(arms: Any, tmp_path: Path) -> None:
    """The framing is resolved from the tokenizer, so it can differ with the flags equal.

    Every arm is launched by this driver with the same ``--prompt-style``, which is
    ``auto``, which the tokenizer answers. A quantized checkpoint saved with a tokenizer
    that lost its chat template is then asked bare-text questions while the ceiling is
    asked chat questions, and the gap arrives in the table wearing quantization's costume.

    Covered here rather than only in the eval's own tests because it is the reason
    ``check_pairable`` reads through ``_comparability`` instead of a field list written
    when only the CLI-level settings were known.

    Turns red when: the panel's guard stops reading the detail block.
    """
    panel = _panel(
        arms,
        tmp_path,
        {
            "bf16": _record(),
            "gptq_4b": _record(detail={"prompt_style": "completion", "by_source": {}}),
        },
    )

    with pytest.raises(SystemExit, match="cannot be paired"):
        arms.check_pairable(panel)


ANCHORS = {4: 4_399_629_312, 3: 3_332_904_576}
PANEL = ("bf16", "gptq_4b", "awq_4b", "dq_4b", "gptq_3b", "awq_3b", "dq_3b")


def _resumable(
    arms: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    records: dict[str, dict[str, Any]],
) -> tuple[Path, list[str]]:
    """A directory `do_run --resume` can walk, and the list of arms it decided to spend.

    Every DynQuant arm gets a map at its anchor, because a resumed one is still weighed;
    ``records`` decides which arms are already scored, and anything absent from it is an
    arm the run would have to launch. ``_run`` records rather than raises so a test can
    assert on *where* the run stopped, not only that it did.
    """
    out = tmp_path / "arms"
    (out / "maps").mkdir(parents=True)
    for label, width in (("dq_4b", 4), ("dq_3b", 3)):
        (out / "maps" / f"{label}.json").write_text(
            json.dumps({"maps": {str(ANCHORS[width]): {"nbytes": ANCHORS[width], "bits": {}}}}),
            encoding="utf-8",
        )
    for label, payload in records.items():
        (out / f"{label}.json").write_text(json.dumps(payload), encoding="utf-8")

    spent: list[str] = []
    monkeypatch.setattr(arms, "require_one_stack", lambda: None)
    monkeypatch.setattr(arms, "anchor_bytes", lambda model, group_size: ANCHORS)
    monkeypatch.setattr(arms, "_run", lambda cmd, what: spent.append(what))
    return out, spent


def test_a_resumed_arm_is_weighed_even_though_it_is_not_run(
    arms: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skipping the work must not skip the evidence the work was at matched bytes.

    The whole panel is an at-matched-bytes claim, and the manifest is where it is recorded.
    A resume that only restored ``record`` left ``nbytes`` at ``None``, so a resumed row
    read as an arm that never claimed a size rather than one whose size stopped being
    checked -- and ``check_matched`` never ran, so a map that had drifted since would pass.

    Turns red when: the resume branch short-circuits the arm before it is priced.
    """
    out, spent = _resumable(arms, tmp_path, monkeypatch, {one: _record() for one in PANEL})

    assert arms.do_run(_args(arms, [*RUN[:-1], str(out), "--resume"])) == 0
    assert spent == [], "a resumed arm must not be re-run"

    manifest = json.loads((out / "arms.json").read_text(encoding="utf-8"))
    priced = {arm["label"]: arm["nbytes"] for arm in manifest["arms"]}
    assert priced == {
        "bf16": None,
        "gptq_4b": 4_399_629_312,
        "awq_4b": 4_399_629_312,
        "dq_4b": 4_399_629_312,
        "gptq_3b": 3_332_904_576,
        "awq_3b": 3_332_904_576,
        "dq_3b": 3_332_904_576,
    }
    assert all(arm["record"] for arm in manifest["arms"])


def test_the_run_itself_refuses_a_panel_it_cannot_pair(
    arms: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check has to be wired into the run, not merely available to it.

    Same fixture as the pricing test with one record moved to a 200-item limit, because a
    guard nothing calls is the failure mode a unit test on the guard cannot see.

    The manifest is written after every arm, so the refusal does not leave the panel
    unreadable -- but the arm that caused it must not be in the copy on disk. Its record
    file is still sitting in the directory, and an entry naming it is all the table needs to
    read it back into the comparison the guard just refused.

    Turns red when: the call is dropped from ``do_run``, or moved after the manifest write.
    """
    records = {one: _record() for one in PANEL}
    records["dq_3b"] = _record(limit=200)
    out, spent = _resumable(arms, tmp_path, monkeypatch, records)

    with pytest.raises(SystemExit, match="cannot be paired"):
        arms.do_run(_args(arms, [*RUN[:-1], str(out), "--resume"]))

    assert spent == [], "nothing should re-run"
    written = json.loads((out / "arms.json").read_text(encoding="utf-8"))
    named = {one["label"]: one["record"] for one in written["arms"]}
    assert named["dq_3b"] is None, "the unpairable arm must not be named in the manifest"
    assert (out / "dq_3b.json").is_file(), "its record is on disk; only the entry is withheld"
    assert all(named[one] is not None for one in PANEL if one != "dq_3b")


def test_a_resumed_dynquant_arm_whose_allocation_is_gone_is_refused(
    arms: Any, tmp_path: Path
) -> None:
    """The record says what it scored; only the map says what it cost.

    Resuming into a directory whose ``maps/`` was cleared is the ordinary way to reach this
    -- the records are the expensive artefacts and the maps look like scratch. Without the
    map there is no realised size, so the arm cannot be shown to be byte-matched and the
    honest outcome is a refusal, not a row with the request copied into the size column.

    Turns red when: a missing map falls through to a traceback, or to the request.
    """
    with pytest.raises(SystemExit, match="does not exist"):
        arms.map_nbytes(tmp_path / "dq_4b.json", "4399629312")


# --- decode budget ----------------------------------------------------------------------
#
# The panel's roof. Every arm generates SQL under the same `--max-new-tokens`, and a query
# cut mid-clause scores as a syntax error rather than as an answer -- so a budget that binds
# is a floor under every arm's accuracy, and it binds unevenly because a damaged arm rambles.
# The ceiling runs first and is checked for censoring so the roof is known to be the model's
# before six quantization passes are spent under it.


def test_a_ceiling_that_ran_out_of_budget_is_refused(arms: Any, tmp_path: Path) -> None:
    """A censored ceiling is not a headroom, and the six arms beneath it are not comparable.

    Quantized arms may legitimately run past the cap -- deliberating longer *is* damage, and
    the record says so. The ceiling is the one arm where it means the opposite: the number
    every difference in the table is measured against was set by the flag rather than by the
    model, and the flag is the same for all seven, so the whole panel tilts together.

    The message carries the count and the budget because the fix is a specific larger
    number, and the operator reading it has an hour of ceiling and six hours of arms in
    front of them.

    Turns red when: the ceiling stops being checked, or the check reports the failure
    without saying what budget produced it.
    """
    record = tmp_path / "bf16.json"
    record.write_text(
        json.dumps(
            _record(
                total=400,
                accuracy=0.6125,
                detail={"prompt_style": "chat", "unfinished_reasoning": 10, "by_source": {}},
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match=r"10/400 generations \(2\.5%\).*at 384 new tokens"):
        arms.check_uncensored(record)


def test_a_ceiling_that_closed_inside_the_budget_passes(arms: Any, tmp_path: Path) -> None:
    """Zero unfinished is the whole condition -- no tolerance, and no other reason to fail.

    Paired with the refusal above so the guard is pinned from both sides: one that fired on
    every ceiling would be removed within a run, and one that never fired would not be
    noticed at all. The record here is the ordinary shape, including an accuracy well short
    of 100%, because being wrong is not being censored.

    Turns red when: a threshold is introduced, or the check reads a field that an
    uncensored record does not carry.
    """
    record = tmp_path / "bf16.json"
    record.write_text(json.dumps(_record(total=400, accuracy=0.6125)), encoding="utf-8")

    arms.check_uncensored(record)


def test_the_run_stops_at_the_ceiling_before_a_quantized_arm_is_spent(
    arms: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordering is the point of the check, so the test is on what did not run.

    ``plan_arms`` puts the ceiling first for exactly this: the censoring verdict is
    available after one arm and invalidates the other six, and six hours is the cost of
    learning it afterwards. A ``check_uncensored`` called at the end of ``do_run`` would
    pass every unit test above and still let the whole panel run under a lowered roof.

    Turns red when: the ceiling check moves after the loop, or the panel is reordered so a
    quantized arm runs first.
    """
    censored = _record(
        total=400, detail={"prompt_style": "chat", "unfinished_reasoning": 10, "by_source": {}}
    )
    out, spent = _resumable(arms, tmp_path, monkeypatch, {"bf16": censored})

    with pytest.raises(SystemExit, match="the bf16 ceiling left"):
        arms.do_run(_args(arms, [*RUN[:-1], str(out), "--resume"]))

    assert spent == []


def test_the_decode_budget_is_a_pairing_field_the_records_carry(arms: Any, tmp_path: Path) -> None:
    """The budget the ceiling was cleared at is the budget the other six have to be scored at.

    ``check_uncensored`` establishes that 1024 was enough for the ceiling. That says nothing
    about an arm scored at 320, which is what the eval's own task spec answers when the flag
    is absent -- and a resumed record from before the budget was pinned carries exactly that.
    So the guard that clears the roof and the guard that pairs the arms are two halves of
    one claim, and this is the half that catches the leftover.

    Turns red when: ``max_new_tokens`` leaves ``DECODE_PAIRING_FIELDS``, or the panel's
    pairing check stops reading the decode block.
    """
    panel = _panel(
        arms,
        tmp_path,
        {"bf16": _record(), "dq_4b": _record(decode={"max_new_tokens": 320, "batch_size": 8})},
    )

    with pytest.raises(SystemExit, match=re.escape("decode.max_new_tokens")):
        arms.check_pairable(panel)


def test_a_mismatched_arm_stops_the_run_before_the_next_one_is_launched(
    arms: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checked after every arm, not once at the end, because the end is six hours away.

    The realistic shape: a resume into a directory holding one record from an earlier,
    narrower run. Pairing at the end would catch it -- after launching the five arms that
    were not resumed, which is the entire cost the check exists to avoid. Here only the
    ceiling and a stale ``gptq_4b`` are on disk, so an end-of-loop check would spend
    ``awq_4b`` and everything after it.

    Turns red when: ``check_pairable`` moves back out of the loop, or is passed the whole
    panel including arms that have not been scored yet.
    """
    records = {"bf16": _record(), "gptq_4b": _record(limit=200)}
    out, spent = _resumable(arms, tmp_path, monkeypatch, records)

    with pytest.raises(SystemExit, match="gptq_4b was not scored under the same settings"):
        arms.do_run(_args(arms, [*RUN[:-1], str(out), "--resume"]))

    assert spent == []


def test_a_quantized_arm_that_ran_out_of_budget_is_recorded_not_refused(
    arms: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deliberating past the cap is a result for a quantized arm and a defect for the ceiling.

    The same number means two different things depending on which arm produced it, which is
    why the check is wired to one arm rather than applied to the record schema. A 3-bit arm
    that stops closing its queries inside a budget the bf16 model cleared comfortably is the
    damage this campaign exists to measure -- refusing it would delete the finding, and the
    eval already records the count so the table can report it.

    Turns red when: the censoring check is applied to every arm instead of the ceiling.
    """
    records = {one: _record() for one in PANEL}
    records["dq_3b"] = _record(
        total=400, detail={"prompt_style": "chat", "unfinished_reasoning": 37, "by_source": {}}
    )
    out, spent = _resumable(arms, tmp_path, monkeypatch, records)

    assert arms.do_run(_args(arms, [*RUN[:-1], str(out), "--resume"])) == 0
    assert spent == []


# --- the commands parse -------------------------------------------------------------------
#
# Everything above checks what the driver *decides*. This checks that the programs it drives
# accept what it says, which is a different failure and a more expensive one: a renamed flag
# is a clean argparse exit five hours into a seven-hour panel, on the arm that happens to
# carry it, with the arms before it already spent.


def test_the_dynquant_arms_apply_their_map_the_only_way_this_model_allows(
    arms: Any, tmp_path: Path
) -> None:
    """Packing has nothing to replace for 91.5% of this checkpoint.

    The expert banks are bare 3-D parameters, and the packed runtime swaps ``Linear`` and
    ``Embedding`` *modules*, so ``--map-apply pack`` -- the default, and right for every
    dense model -- reaches 8.5% of the parameters here and refuses on the rest. Encoding
    runs the same encoder at the same widths and writes the reconstruction back, which is
    what the accuracy panel is measuring; the byte figure comes from the map either way.

    Asserted on the DynQuant arms only. If the flag ever became a panel-wide default it
    would silently change what every future dense campaign measures in VRAM, and this test
    is the record that it was chosen for this architecture.

    Turns red when: an arm reverts to the packed default and would die four hours in, or
    the flag is spelled somewhere the eval parser does not accept -- the parse test below
    catches that half.
    """
    for label, width in (("dq_4b", 4), ("dq_3b", 3)):
        arm = arms.Arm(label=label, kind="dq", anchor=width, target_bytes=ANCHORS[width])
        cmd = arms.dq_eval_cmd(_args(arms), arm, tmp_path / "maps.json", tmp_path / f"{label}.json")
        assert cmd[cmd.index("--map-apply") + 1] == "encode"

    ceiling = arms.Arm(label="bf16", kind="ceiling", anchor=None, target_bytes=None)
    assert "--map-apply" not in arms.ceiling_cmd(_args(arms), ceiling, tmp_path / "bf16.json")


def test_every_command_the_panel_issues_is_accepted_by_the_program_it_runs(
    arms: Any, tmp_path: Path
) -> None:
    """Parsed by the real parsers, not compared against a list of flag names written here.

    The driver builds eleven command lines across four builders, and every one of them
    crosses a package boundary: ``dynquant eval`` and ``dynquant inspect`` live in
    `dynquant-core`, the baseline arms live in a sibling experiment script. Nothing makes
    those move together -- `--map-key` could be renamed in a core refactor that no test in
    core notices, because core's own tests do not know this panel exists.

    Handing the argv to the actual `build_parser` of the actual target is the only version
    of this check that cannot go stale. A hand-written list of expected flags would be a
    third copy of the contract, and it would agree with the driver right up until the
    parser changed underneath both.

    Turns red when: a flag is renamed or dropped on either side of the boundary, or a
    required argument is added to a subcommand the panel calls.
    """
    from dynquant.cli import build_parser as core_parser

    spec = importlib.util.spec_from_file_location(
        "_dq_baselines_for_arms", DRIVER.parent / "baselines_lfm2.py"
    )
    assert spec and spec.loader
    baselines = importlib.util.module_from_spec(spec)
    sys.modules["_dq_baselines_for_arms"] = baselines
    spec.loader.exec_module(baselines)

    args = _args(arms)
    issued = 0
    for arm in arms.plan_arms(ANCHORS):
        record, save_map = tmp_path / f"{arm.label}.json", tmp_path / f"{arm.label}.map.json"
        if arm.kind == "ceiling":
            built = [(core_parser(), arms.ceiling_cmd(args, arm, record))]
        elif arm.kind == "dq":
            built = [
                (core_parser(), arms.dq_inspect_cmd(args, arm, save_map)),
                (core_parser(), arms.dq_eval_cmd(args, arm, save_map, record)),
            ]
        else:
            built = [(baselines.build_parser(), arms.baseline_cmd(args, arm, record))]
        for parser, cmd in built:
            # Past `sys.executable` and past `-m dynquant` or the script path -- argv as the
            # target sees it, which is where a wrong flag would actually be rejected.
            argv = cmd[3:] if cmd[1] == "-m" else cmd[2:]
            parser.parse_args(argv)
            issued += 1

    assert issued == 9, (
        "one ceiling, four baselines, and two DynQuant arms that allocate before they score"
    )


def test_where_the_panel_loads_is_one_passthrough_and_it_reaches_every_arm(
    arms: Any, tmp_path: Path
) -> None:
    """Seven arms on one device, chosen once, and the allocator deliberately not on it.

    The default is `cuda` because that is where the panel runs. The flag exists because the
    box it runs on holds the GPU for hours at a time under the fine-tune that produces the
    signal, and a driver that can only be exercised on the hardware the real run needs is a
    driver first exercised during the real run. `CUDA_VISIBLE_DEVICES=` would hide the GPU
    from the arms that read it from the environment and change nothing for the one that
    passes `device_map="cuda"` regardless, which was `baselines_lfm2 run` until this flag.

    `dq_inspect_cmd` is excluded on purpose rather than by omission. Allocation reads a
    stats file and a config; it never loads weights, `inspect` already defaults to cpu, and
    routing it to the panel's device would put the one CPU-only step of the panel on the
    GPU the passthrough exists to keep clear.

    Turns red when: an arm kind stops taking the flag, one of them hardcodes a device, or
    the allocation starts loading onto the panel's device.
    """
    assert _args(arms).device == "cuda"

    args = _args(arms, [*RUN, "--device", "cpu"])
    loading, allocating = 0, 0
    for arm in arms.plan_arms(ANCHORS):
        record, save_map = tmp_path / f"{arm.label}.json", tmp_path / f"{arm.label}.map.json"
        if arm.kind == "ceiling":
            built = [arms.ceiling_cmd(args, arm, record)]
        elif arm.kind == "dq":
            built = [arms.dq_eval_cmd(args, arm, save_map, record)]
            allocation = arms.dq_inspect_cmd(args, arm, save_map)
            assert "--device" not in allocation
            allocating += 1
        else:
            built = [arms.baseline_cmd(args, arm, record)]
        for cmd in built:
            assert cmd[cmd.index("--device") + 1] == "cpu"
            assert cmd.count("--device") == 1
            loading += 1

    assert (loading, allocating) == (7, 2)


def test_a_failing_arm_still_reports_how_long_it_took(
    arms: Any, capsys: Any, monkeypatch: Any
) -> None:
    """The duration is printed before the return code is checked, not after.

    An eval record carries its own ``seconds``; a quantization pass carries nothing, so the
    panel log is the only place a calibration's cost is written down. The arm most worth
    timing is the one that died -- an arm that fell over 90 minutes into a GPTQ pass and an
    arm that refused in 3 seconds want different fixes, and after the fact the log is all
    there is to tell them apart.

    Turns red when: the elapsed line moves below the ``raise``, or is dropped.
    """
    monkeypatch.setattr(arms.subprocess, "run", lambda cmd: SimpleNamespace(returncode=7))
    with pytest.raises(SystemExit):
        arms._run(["true"], what="gptq_3b quantization")

    out = capsys.readouterr().out
    assert "[gptq_3b quantization] exit 7 after " in out
    assert out.rstrip().endswith("s")
