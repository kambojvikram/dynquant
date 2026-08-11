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
import os
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


def test_a_rescored_dispatch_does_not_stop_the_panel(
    arms: Any, tmp_path: Path, capsys: Any
) -> None:
    """The shape of `--rescore bf16,dq_4b,dq_3b --experts-impl eager`, checked before it runs.

    Three arms score again under a dispatch the four reused baselines predate, so the
    re-scored records carry `experts.ran` and the reused ones carry no `experts` key at
    all. Every other field agrees, because it is the same driver, the same flags and the
    same items. If that combination raised, the driver would stop on the second arm -- the
    first *reused* one -- having already spent the ceiling re-score and reached none of the
    DynQuant arms it was launched for.

    It does not raise, because the disagreement is not about the problem set: the hits are
    over the same items in the same order and they pair. It is about how the answers were
    computed, which `panel_table` marks per comparison and prices against the 1.24% of
    tokens the two dispatches disagree on. The note is asserted too -- a difference that
    the driver neither refuses nor mentions is one the operator finds in the table.

    Turns red when: the expert block goes back into the refusal, or the note stops naming
    the arms and the log no longer says the panel is straddling two dispatches.
    """
    panel = _panel(
        arms,
        tmp_path,
        {
            "bf16": _record(experts={"found": "grouped_mm", "ran": "eager"}),
            "gptq_4b": _record(),
            "awq_4b": _record(),
            "dq_4b": _record(experts={"found": "grouped_mm", "ran": "eager"}),
        },
    )

    arms.check_pairable(panel)

    note = capsys.readouterr().out
    assert "2 arm(s) disagree with bf16 on the expert dispatch" in note
    assert "gptq_4b, awq_4b" in note
    assert "dq_4b" not in note, "an arm that agrees with the reference is not a straddle"


def test_a_moved_problem_set_still_stops_it_when_the_dispatch_moved_too(
    arms: Any, tmp_path: Path
) -> None:
    """Holding one field out of the refusal must not hold the record out of it.

    The dangerous version of the change above is one that notices any expert difference
    and stops looking. Then a record from a 200-item smoke run resumed into a 400-item
    panel walks in behind a dispatch difference, and the guard that exists for exactly that
    case waves it through.

    The message is asserted as well as the raise. An operator reads it to decide which file
    to delete, and one that led with `experts.ran` would send them to re-run an arm whose
    real defect is its `limit`.

    Turns red when: the held-out field short-circuits the whole comparison, or the reported
    difference starts including the field the driver decided not to refuse.
    """
    panel = _panel(
        arms,
        tmp_path,
        {
            "bf16": _record(experts={"found": "grouped_mm", "ran": "eager"}),
            "gptq_4b": _record(limit=200),
        },
    )

    with pytest.raises(SystemExit, match="cannot be paired") as caught:
        arms.check_pairable(panel)

    assert "'limit'" in str(caught.value)
    assert "experts.ran" not in str(caught.value)


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


def _resume_args(arms: Any, model: Path, stats: Path, out: Path) -> argparse.Namespace:
    return _args(
        arms,
        ["run", "--model", str(model), "--stats", str(stats), "--out", str(out), "--resume"],
    )


def test_resuming_into_another_panels_directory_refuses(arms: Any, tmp_path: Path) -> None:
    """The manifest names the inputs; a resume that changes them is a different panel.

    ``check_pairable`` reads ``_comparability``, and no field in it names the model. Seven
    records scored on two merges at the same task, split, shots and limit pair perfectly and
    table as a comparison between quantizers. Nothing downstream can notice, because by then
    the only surviving evidence of which weights produced a number is the directory it sits
    in -- and the directory is the thing being reused.

    Turns red when: the manifest's model/stats/moments/group_size stop being compared against
    the current run's, or the comparison stops raising.
    """
    out = tmp_path / "panel"
    out.mkdir()
    (out / "arms.json").write_text(
        json.dumps(
            {
                "model": "/runs/s4/some-other-merge",
                "stats": str(tmp_path / "dynquant_stats.json"),
                "moments": None,
                "group_size": 128,
            }
        ),
        encoding="utf-8",
    )
    args = _resume_args(arms, tmp_path / "merged", tmp_path / "dynquant_stats.json", out)

    with pytest.raises(SystemExit) as caught:
        arms.check_resumable(out, args, [], rescore=frozenset())

    message = str(caught.value)
    assert "some-other-merge" in message
    assert "model" in message


def test_a_dynquant_record_older_than_the_signal_is_stale_and_a_baseline_is_not(
    arms: Any, tmp_path: Path
) -> None:
    """Freshness is charged per arm, because only two of the seven read the signal file.

    A record that exists is the whole of what ``--resume`` checks, and existing says nothing
    about *when*. Regenerate the stats in place -- rerun the bank census, fix a key, extend
    the moments -- and the two DynQuant records still sitting in the directory were allocated
    from the file as it used to be. They pair, they table, and the arm carrying the claim is
    the one scored against the superseded signal.

    The four baselines never open that file, so charging it against them would price the
    cheapest correct fix -- rerun two arms -- as a whole new panel, which is how a guard
    teaches people to pass ``--resume`` less often rather than more carefully.

    Turns red when: the mtime comparison goes, or the stats charge stops being scoped to
    ``kind == "dq"``.
    """
    out = tmp_path / "panel"
    out.mkdir()
    stats = tmp_path / "dynquant_stats.json"
    stats.write_text("{}", encoding="utf-8")
    for label in ("dq_4b", "gptq_4b"):
        record = out / f"{label}.json"
        record.write_text("{}", encoding="utf-8")
        os.utime(record, (0, 0))
    # The model directory is never created, so `config.json` does not exist and the model
    # charge stays out of the way -- this asserts about the stats charge alone.
    args = _resume_args(arms, tmp_path / "merged", stats, out)
    panel = [arms.Arm("dq_4b", "dq", 4, 1), arms.Arm("gptq_4b", "gptq", 4, 1)]

    with pytest.raises(SystemExit) as caught:
        arms.check_resumable(out, args, panel, rescore=frozenset())

    message = str(caught.value)
    assert "dq_4b.json predates the stats file" in message
    assert "gptq_4b" not in message


def test_every_scoring_arm_names_the_experts_dispatch(arms: Any, tmp_path: Path) -> None:
    """The confound that had already contaminated four landed arms of this panel.

    bf16 and the dq arms encode in place and keep ``post_init``'s ``grouped_mm``; GPTQ and
    AWQ arrive from ``llm-compressor`` with their expert banks rewritten into per-expert
    ``Linear`` modules, so they compute what eager computes. On LFM2.5-8B-A1B the two
    dispatches disagree on 1.24% of teacher-forced tokens, 0.29x the quantization effect,
    which is the same order as the dq-minus-GPTQ margin the panel exists to report. Every
    arm therefore says ``eager`` out loud, and a panel run under a future default that
    changed underneath it is not a thing this driver can produce.

    The allocation command is excluded on purpose: ``dynquant inspect`` scores nothing, so
    a dispatch flag on it would be a flag with no meaning, and asserting its absence is
    what keeps the pin attached to the commands that generate tokens.

    Turns red when: the flag leaves ``eval_flags``, or its value stops being pinned and
    starts being inherited from whatever ``dynquant eval`` currently defaults to.
    """
    args = _args(arms)
    scoring = {
        "bf16": arms.ceiling_cmd(
            args, arms.Arm("bf16", "ceiling", None, None), tmp_path / "b.json"
        ),
        "gptq_4b": arms.baseline_cmd(args, arms.Arm("gptq_4b", "gptq", 4, 1), tmp_path / "g.json"),
        "dq_4b": arms.dq_eval_cmd(
            args, arms.Arm("dq_4b", "dq", 4, 1), tmp_path / "m.json", tmp_path / "d.json"
        ),
    }
    for label, command in scoring.items():
        assert "--experts-impl" in command, label
        assert command[command.index("--experts-impl") + 1] == "eager", label

    allocating = arms.dq_inspect_cmd(args, arms.Arm("dq_4b", "dq", 4, 1), tmp_path / "m.json")
    assert "--experts-impl" not in allocating


# --- scoring again without allocating again -------------------------------------------


def test_a_rescored_arm_keeps_its_allocation_and_scores_again(
    arms: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one operation the eager re-score needs, and the one a plain rerun cannot do.

    Section 8 of the packed-MoE report calls for three arms scored again under a pinned
    experts dispatch at the same anchors on the same items. Deleting their records and
    resuming would have re-run ``dynquant inspect`` too, so each DynQuant arm would have
    carried a freshly derived bit map into a comparison whose entire subject is the
    dispatch. The allocator is deterministic and the map would very likely have come back
    identical -- which is the argument that makes this easy to get wrong, because "very
    likely identical" is not a control, and any later change to budget accounting turns it
    into two changes measured as one.

    The arm is still weighed from the map it kept: reusing an allocation is not trusting it.

    Turns red when: ``--rescore`` starts allocating, or stops scoring, or skips the weighing.
    """
    out, spent = _resumable(arms, tmp_path, monkeypatch, {one: _record() for one in PANEL})

    assert arms.do_run(_args(arms, [*RUN[:-1], str(out), "--rescore", "bf16,dq_4b"])) == 0
    assert spent == ["bf16", "dq_4b"], spent
    assert "dq_4b allocation" not in spent

    manifest = json.loads((out / "arms.json").read_text(encoding="utf-8"))
    priced = {arm["label"]: arm["nbytes"] for arm in manifest["arms"]}
    assert priced["dq_4b"] == ANCHORS[4]
    assert priced["dq_3b"] == ANCHORS[3], "an arm not named must still be resumed, not run"


def test_a_rescored_arm_with_no_map_is_refused_rather_than_allocated(
    arms: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure mode the flag exists to prevent, arriving through the flag itself.

    Falling back to allocation when the map is missing would be the friendly behaviour and
    would defeat the point: the caller asked for the previous allocation and would be given
    a new one, in the run where that difference is the thing being controlled for.

    Turns red when: the missing-map branch is softened into a warning or a fallback.
    """
    out, spent = _resumable(arms, tmp_path, monkeypatch, {one: _record() for one in PANEL})
    (out / "maps" / "dq_4b.json").unlink()

    with pytest.raises(SystemExit, match=r"--rescore dq_4b reuses the allocation"):
        arms.do_run(_args(arms, [*RUN[:-1], str(out), "--rescore", "dq_4b"]))
    assert "dq_4b allocation" not in spent


def test_rescore_refuses_a_label_the_panel_does_not_plan(arms: Any) -> None:
    """A typo that rescores nothing and exits 0 is the quietest way to not do the work.

    Turns red when: unknown labels are filtered instead of refused.
    """
    with pytest.raises(SystemExit, match=r"--rescore names \['dq4b'\]"):
        arms.rescored_labels("dq4b,bf16", arms.plan_arms(ANCHORS))
    assert arms.rescored_labels(" dq_4b , bf16 ", arms.plan_arms(ANCHORS)) == {"dq_4b", "bf16"}


def test_a_rescored_arm_is_checked_against_its_map_not_its_record(
    arms: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The staleness check has to follow the artifact being kept, which moved.

    ``check_resumable`` exists because a skip-if-output-exists guard cannot see that its
    output predates its input. Under ``--rescore`` the output being kept is no longer the
    record -- that is about to be rewritten, so its age says nothing -- it is the map. A
    check still pointed at the record would read a *fresh* timestamp, pass, and let a bit
    map derived from a superseded signal file be scored as though it came from this one.
    That is the original defect surviving the fix for it, one artifact to the left.

    The two assertions are the discrimination, not the refusal: the same directory has to
    pass under ``--resume`` and fail under ``--rescore``, because under ``--resume`` the
    record is what carries forward and the record is new enough.

    Turns red when: the check stops distinguishing the two, in either direction.
    """
    out, spent = _resumable(arms, tmp_path, monkeypatch, {one: _record() for one in PANEL})
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}", encoding="utf-8")
    stats = tmp_path / "dynquant_stats.json"
    stats.write_text("{}", encoding="utf-8")

    os.utime(merged / "config.json", (500, 500))
    os.utime(out / "maps" / "dq_4b.json", (1000, 1000))
    os.utime(stats, (2000, 2000))
    for label in PANEL:
        os.utime(out / f"{label}.json", (3000, 3000))

    argv = ["run", "--model", str(merged), "--stats", str(stats), "--out", str(out)]
    with pytest.raises(SystemExit, match=r"maps.dq_4b\.json predates the stats file"):
        arms.do_run(_args(arms, [*argv, "--rescore", "dq_4b"]))

    assert arms.do_run(_args(arms, [*argv, "--resume"])) == 0
    assert spent == [], "the record postdates every input, so a resume has nothing to redo"


# --------------------------------------------------------------------------- controls
#
# The seven-arm panel says DynQuant beats GPTQ by 19.13 points at 3 bits. It does not say
# what earned them: a mixed-width map holding routers at 8 bits and expert
# down-projections at 2 would beat a uniform recipe whether or not the widths were chosen
# by the fine-tune. A control arm is the only thing that can tell those apart, and the
# failures worth covering are the ones that would make it look like one without being one.


def _completing(arms: Any, monkeypatch: pytest.MonkeyPatch, spent: list[str]) -> None:
    """`_resumable`'s fake, plus the record an eval would have written.

    Only the eval half writes anything: an allocation is identified by `--save-map` and its
    output is pre-placed by the caller, since a fake cannot allocate. Keyed on the flag
    rather than on the arm's kind so it stays right for a command this driver grows later.
    """

    def spend(cmd: list[str], what: str) -> None:
        spent.append(what)
        if "--out" in cmd:
            Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(_record()), encoding="utf-8")

    monkeypatch.setattr(arms, "_run", spend)


def test_a_panel_that_asked_for_no_control_is_the_panel_that_was_banked(arms: Any) -> None:
    """Adding the arm must not change the run of anyone who does not want it.

    The seven-arm panel is committed under `experiments/phase4/results/` and every table in
    the report is built from it. A planner that grew an eighth arm by default would make
    every one of those a partial panel -- and the driver would spend an hour on an arm
    nobody asked for on the next resume, into a directory whose manifest would then no
    longer describe what is banked beside it.

    Turns red when: the controls stop being opt-in.
    """
    assert [arm.label for arm in arms.plan_arms(ANCHORS)] == list(PANEL)
    assert all(arm.null_mode is None for arm in arms.plan_arms(ANCHORS))


def test_a_control_is_appended_so_a_resume_scores_only_the_new_arm(
    arms: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of appending rather than inserting, tested through the thing it protects.

    Seven records already on disk and a control added: the run must reuse all seven, spend
    exactly the new arm's allocation and its eval, and leave the seven rows of the manifest
    reading what they read before. An arm inserted at its anchor -- next to `dq_3b`, where
    it belongs conceptually -- would reorder the manifest, and the manifest is the order the
    table prints its rows in, so every banked table would need regenerating to be compared
    against a re-run.

    Turns red when: a control lands anywhere but the end, or a resume re-spends a real arm.
    """
    out, spent = _resumable(arms, tmp_path, monkeypatch, {one: _record() for one in PANEL})
    _completing(arms, monkeypatch, spent)
    # A fake cannot allocate, so the map the control will be weighed against has to be here
    # already -- the same way `_resumable` pre-writes the two real DynQuant arms' maps.
    (out / "maps" / "dq_3b_shuf.json").write_text(
        json.dumps({"maps": {str(ANCHORS[3]): {"nbytes": ANCHORS[3], "bits": {}}}}),
        encoding="utf-8",
    )

    argv = [*RUN[:-1], str(out), "--resume", "--score-null", "shuffle"]
    assert arms.do_run(_args(arms, argv)) == 0
    assert spent == ["dq_3b_shuf allocation", "dq_3b_shuf"]

    manifest = json.loads((out / "arms.json").read_text(encoding="utf-8"))
    assert [arm["label"] for arm in manifest["arms"]] == [*PANEL, "dq_3b_shuf"]
    assert all(arm["record"] for arm in manifest["arms"])


def test_the_manifest_says_which_arms_are_controls_and_which_are_not(
    arms: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A control that is only a control in its filename is the failure this prevents.

    Downstream reads the manifest, not the directory listing. An arm whose signal was
    nulled and whose manifest entry looks like every other DynQuant arm is a bit map that
    becomes a headline -- and the reader who would have caught it is looking at a table
    built from the file that does not say.

    Turns red when: the provenance stops reaching the manifest, or reaches the real arms.
    """
    out, spent = _resumable(arms, tmp_path, monkeypatch, {one: _record() for one in PANEL})
    _completing(arms, monkeypatch, spent)
    for mode in ("shuf", "unif"):
        (out / "maps" / f"dq_3b_{mode}.json").write_text(
            json.dumps({"maps": {str(ANCHORS[3]): {"nbytes": ANCHORS[3], "bits": {}}}}),
            encoding="utf-8",
        )

    argv = [*RUN[:-1], str(out), "--resume", "--score-null", "shuffle,uniform", "--null-seed", "7"]
    assert arms.do_run(_args(arms, argv)) == 0

    manifest = json.loads((out / "arms.json").read_text(encoding="utf-8"))
    marked = {arm["label"]: arm.get("score_null") for arm in manifest["arms"]}
    assert marked["dq_3b_shuf"] == {"mode": "shuffle", "seed": 7}
    assert marked["dq_3b_unif"] == {"mode": "uniform", "seed": 7}
    assert [label for label, spec in marked.items() if spec] == ["dq_3b_shuf", "dq_3b_unif"]


def test_the_control_differs_from_the_arm_it_controls_in_the_allocation_only(
    arms: Any, tmp_path: Path
) -> None:
    """One change, and it is at allocation time.

    A control that also scored differently would answer nothing: the margin against it
    would carry the signal *and* whatever else moved. So the eval command has to be the
    real arm's command with a different label and a different map, and the allocation
    command has to be the real arm's with the null appended -- which is also why the null
    goes on last, after `--moments`. The allocator applies the null to whatever the scoring
    and the sensitivity pricing produced; applied earlier it would leave the 8.5% of this
    checkpoint's parameters that are priced by measured `dL` still priced by their own
    moments, and the arm would be a partial control reported as a whole one.

    Turns red when: a flag that is not the null differs between the two, in either command.
    """
    planned = arms.plan_arms(ANCHORS, nulls=("shuffle",), null_anchor=3, null_seed=3)
    real = next(arm for arm in planned if arm.label == "dq_3b")
    control = planned[-1]
    args = _args(arms, [*RUN, "--moments", "/runs/s4/moments.json"])

    real_alloc = arms.dq_inspect_cmd(args, real, tmp_path / "dq_3b.json")
    control_alloc = arms.dq_inspect_cmd(args, control, tmp_path / "dq_3b_shuf.json")
    assert control_alloc[-4:] == ["--score-null", "shuffle", "--null-seed", "3"]
    assert control_alloc[:-4] == [
        part.replace("dq_3b.json", "dq_3b_shuf.json") for part in real_alloc
    ]
    assert control_alloc.index("--moments") < control_alloc.index("--score-null")

    real_eval = arms.dq_eval_cmd(args, real, tmp_path / "dq_3b.json", tmp_path / "dq_3b.out")
    control_eval = arms.dq_eval_cmd(
        args, control, tmp_path / "dq_3b_shuf.json", tmp_path / "dq_3b_shuf.out"
    )
    assert "--score-null" not in control_eval
    renamed = [part.replace("dq_3b", "dq_3b_shuf") for part in real_eval]
    assert control_eval == renamed


def test_a_control_is_planned_at_the_same_anchor_as_the_arm_it_decomposes(arms: Any) -> None:
    """Matched bytes, or the control is a size comparison wearing a signal's name.

    The whole decomposition is subtraction: the real arm minus the control is the signal's
    share only if the two spent the same bytes. They are pinned to the same anchor by
    construction here, and the run still puts both through `check_matched` -- this asserts
    the plan, and `check_matched` asserts the realisation.

    Turns red when: a control is planned at a budget its reference arm did not run at.
    """
    planned = arms.plan_arms(ANCHORS, nulls=("shuffle", "uniform"))
    at_three = [arm for arm in planned if arm.anchor == 3]
    assert {arm.target_bytes for arm in at_three} == {ANCHORS[3]}
    assert [arm.label for arm in at_three] == [
        "gptq_3b",
        "awq_3b",
        "dq_3b",
        "dq_3b_shuf",
        "dq_3b_unif",
    ]

    four = arms.plan_arms(ANCHORS, nulls=("shuffle",), null_anchor=4)[-1]
    assert (four.label, four.anchor, four.target_bytes) == ("dq_4b_shuf", 4, ANCHORS[4])


def test_an_unknown_null_mode_is_refused_against_the_packages_own_list(arms: Any) -> None:
    """The list of modes lives in one place, and it is not this driver.

    `choices=` on the parser would be a second copy of a registry: a mode added to
    `dynquant.score.null` and not here is unreachable from the panel, and one removed there
    is accepted here and fails an hour into a run. So the driver reads `NULL_MODES` and the
    refusal quotes it, which also means the message stays right when the tuple changes.

    Turns red when: the driver starts keeping its own list of modes.
    """
    from dynquant.score.null import NULL_MODES

    with pytest.raises(SystemExit) as caught:
        arms.check_null_modes("shufle")
    assert str(list(NULL_MODES)) in str(caught.value)
    assert arms.check_null_modes(",".join(NULL_MODES)) == tuple(NULL_MODES)
    assert arms.check_null_modes("") == ()


def test_a_repeated_null_mode_is_refused_before_two_arms_share_a_record(arms: Any) -> None:
    """Two arms with one label is the silent version of running one arm.

    The record is `out/<label>.json` and the map is `out/maps/<label>.json`, so a duplicated
    mode plans two arms that write to the same two paths. The second overwrites the first,
    the manifest lists both, and the table prints two identical rows as though the control
    had been replicated.

    Turns red when: duplicates are accepted, or silently collapsed to one.
    """
    with pytest.raises(SystemExit, match="repeats a mode"):
        arms.check_null_modes("shuffle,shuffle")


def test_a_control_is_charged_against_the_stats_file_like_every_other_dynquant_arm(
    arms: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The staleness guard has to reach the new arm, and it reaches it by kind.

    `check_resumable` charges the stats file against DynQuant arms and not the baselines,
    because only DynQuant reads it. A control reads it too -- it is the *nulled* signal, not
    no signal -- so a control map written before the signal file it claims to have nulled is
    exactly as stale as a real arm's, and for the same reason.

    Turns red when: the guard starts keying on the label, which is where the controls differ.
    """
    out, spent = _resumable(arms, tmp_path, monkeypatch, {one: _record() for one in PANEL})
    control_map = out / "maps" / "dq_3b_shuf.json"
    control_map.write_text(
        json.dumps({"maps": {str(ANCHORS[3]): {"nbytes": ANCHORS[3], "bits": {}}}}),
        encoding="utf-8",
    )
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / "config.json").write_text("{}", encoding="utf-8")
    stats = tmp_path / "dynquant_stats.json"
    stats.write_text("{}", encoding="utf-8")

    os.utime(merged / "config.json", (500, 500))
    for label in PANEL:
        os.utime(out / f"{label}.json", (3000, 3000))
    (out / "dq_3b_shuf.json").write_text(json.dumps(_record()), encoding="utf-8")
    os.utime(out / "dq_3b_shuf.json", (1000, 1000))
    os.utime(stats, (2000, 2000))

    argv = [
        "run",
        "--model",
        str(merged),
        "--stats",
        str(stats),
        "--out",
        str(out),
        "--resume",
        "--score-null",
        "shuffle",
    ]
    with pytest.raises(SystemExit, match=r"dq_3b_shuf\.json predates the stats file"):
        arms.do_run(_args(arms, argv))
    assert spent == []
