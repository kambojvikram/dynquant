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
import sys
from pathlib import Path
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

    ``limit`` and ``max_new_tokens`` are the two that may legitimately be absent. Absent
    means every arm inherits the same task default, which is what pairing needs; what would
    break it is one arm carrying the flag and another not, so the assertion is on the pair
    moving together.

    Turns red when: a new pairing field is added to the eval contract and no flag here
    carries it, or one of these stops being passed.
    """
    flags = arms.eval_flags(_args(arms), "dq_4b")

    fixed_by_the_command = {"task", "backend"}
    optional = {"limit", "max_new_tokens"}
    for name in (*PAIRING_FIELDS, *DECODE_PAIRING_FIELDS):
        if name in fixed_by_the_command or name in optional:
            continue
        assert f"--{name.replace('_', '-')}" in flags, f"{name} is left to the eval default"

    wider = _args(arms, [*RUN, "--limit", "400", "--max-new-tokens", "1024"])
    with_both = arms.eval_flags(wider, "dq_4b")
    assert "--limit" in with_both and "--max-new-tokens" in with_both
    assert "--limit" not in flags and "--max-new-tokens" not in flags


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
