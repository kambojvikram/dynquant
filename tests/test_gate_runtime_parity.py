"""The G4 gate must be configured the way the campaign is configured.

``scripts/gate_runtime_parity.py`` exists to certify that serving an arm through vLLM
scores the same as running it through ``transformers``. That certificate is only worth
something if the gate evaluates the task the way ``dynquant eval`` does. It used to
resolve its own splits -- ``args.split or "test"``, ``--shot-split`` defaulted to
``"train"`` -- and carry an ``if args.task == "ifeval"`` branch around the registry.
Under that arrangement the gate could certify a three-shot MBPP drawn from the scored
split while the campaign ran a three-shot MBPP drawn from ``prompt``, and every number
either produced would look exactly as expected.

So both halves are pinned here: the gate offers every task the campaign scores, and the
namespace it hands to the shared resolvers actually satisfies them. The second is the
one that fails expensively in the wild -- a missing attribute raises ``AttributeError``
inside ``_task_kwargs`` *after* both a model load and, on the served arm, a vLLM engine
build, which is twenty minutes and a GPU-hour into a run that was never going to work.

Nothing here needs a GPU, a checkpoint or a network: the gate's argument parser and the
two resolvers are pure, so the whole boundary is reachable from CPU CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from dynquant.commands.evaluate import TASKS, _resolve_splits, _task_kwargs
from dynquant.errors import DynQuantError
from dynquant.eval.compare import PairedComparison

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE = REPO_ROOT / "scripts" / "gate_runtime_parity.py"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("_dq_gate", GATE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dq_gate"] = module
    spec.loader.exec_module(module)
    return module


def _args(gate, *extra: str):
    return gate._parse_args(["--model", "m", *extra])


def test_the_gate_offers_every_task_the_campaign_scores(gate) -> None:
    """A task in the registry but not in the gate's choices is a task the campaign
    serves through a runtime nothing ever compared against ``transformers``."""
    (action,) = [a for a in gate._build_parser()._actions if a.dest == "task"]
    assert set(action.choices) == set(TASKS)


@pytest.mark.parametrize("task", sorted(TASKS))
def test_the_gates_namespace_satisfies_the_resolvers_it_shares(gate, task: str) -> None:
    """Every attribute ``_resolve_splits`` and ``_task_kwargs`` read must exist on the
    gate's namespace. They are two separate argument parsers over one pair of
    functions, and nothing but this test connects them."""
    spec = TASKS[task]
    extra = ["--allow-execution"] if spec.executes_code else []
    args = _args(gate, "--task", task, *extra)

    split, shot_split, n_shots = _resolve_splits(spec, args)
    assert split == spec.split
    assert shot_split == spec.shot_split
    assert n_shots == (spec.shots if spec.takes_shots else 0)

    sent = _task_kwargs(spec, args, ["exemplar"])
    assert ("shots" in sent) == spec.takes_shots
    assert ("allow_execution" in sent) == spec.executes_code


def test_the_gate_drops_unverifiable_items_where_the_command_would_raise(gate) -> None:
    """The one place the gate deliberately differs, so it is asserted rather than
    left to a default. A missing ``langdetect`` makes ``raise`` produce no number at
    all; both arms drop the same keys, so the pairing -- the only thing this script
    measures -- survives."""
    assert _args(gate).on_unverifiable == "drop"
    sent = _task_kwargs(TASKS["ifeval"], _args(gate, "--task", "ifeval"), [])
    assert sent["on_unverifiable"] == "drop"


def test_a_code_task_still_has_to_opt_into_running_the_generations(gate) -> None:
    """The gate is not an exemption. Scoring HumanEval means executing what the model
    wrote, and the refusal has to happen here too."""
    with pytest.raises(DynQuantError, match="--allow-execution"):
        _task_kwargs(TASKS["humaneval"], _args(gate, "--task", "humaneval"), [])


# --------------------------------------------------------------------------
# The verdict
#
# Equivalence testing has three outcomes and only two of them are a failure, so the
# gate has to name which failure it found. It got that wrong on the first real run:
# a 23-point gap with p=0.0002 was reported as "too few problems to tell", which
# points the operator at scoring *more* problems -- the one action that cannot help,
# and the expensive one.
# --------------------------------------------------------------------------


def _paired(*, both_right: int, a_only: int, b_only: int, both_wrong: int, p: float):
    return PairedComparison(
        label_a="transformers",
        label_b="vllm",
        both_right=both_right,
        a_only=a_only,
        b_only=b_only,
        both_wrong=both_wrong,
        p_value=p,
    )


def test_arms_that_agree_within_the_bound_pass(gate) -> None:
    """The whole point of the gate: a small measured difference is a pass.

    Turns red when: the gate starts demanding identical scores. That is not the claim
    -- the serving-parity report already measured that vLLM and transformers share
    only 9 of 32 greedy tokens on some fp16 prompts, because ties near a decision
    boundary break differently under different kernel orders, while top-1 agreement
    stays at 100%. A gate demanding equality fails on a correct integration.
    """
    paired = _paired(both_right=600, a_only=5, b_only=5, both_wrong=390, p=1.0)
    assert gate._judge(paired, chance=0.0, max_delta=1.0) == []


def test_a_difference_that_is_real_but_tiny_still_passes(gate) -> None:
    """Equivalence, not significance -- the distinction the whole verdict rests on.

    At 10k problems a 0.40-point gap has an interval that excludes zero, so a
    significance test would fail it. It is also four times inside the bound, which is
    the statement the campaign needs: the runtimes may differ, but not by enough to
    change a conclusion.

    Turns red when: the judge is rewritten around ``paired.separated()`` or the
    p-value, which is the natural-looking mistake and would fail every sufficiently
    large honest run.
    """
    paired = _paired(both_right=4000, a_only=60, b_only=20, both_wrong=5920, p=1e-5)
    low, high = paired.interval_points
    assert low > 0.0, "the fixture is only interesting if it does exclude zero"
    assert high < 1.0

    assert gate._judge(paired, chance=0.0, max_delta=1.0) == []


def test_a_real_disagreement_is_not_reported_as_too_little_data(gate) -> None:
    """The G4 smoke run, replayed exactly: 37.00% vs 60.00% on 100 GSM8K problems.

    Those counts came off the box, and the cause was real -- the transformers arm
    merged the checkpoint's ``repetition_penalty: 1.1``. The gate found it and then
    described it as a sample-size problem, because it branched on how *wide* the
    interval was before asking whether it excluded zero. Width cannot tell the two
    apart. Acting on that message means scoring more problems, which narrows the
    interval around a difference that is genuinely there and fails again, later and
    more expensively.

    Turns red when: the width check is put back in front of the exclusion check.
    """
    paired = _paired(both_right=30, a_only=7, b_only=30, both_wrong=33, p=0.0001911)
    assert paired.delta_points == pytest.approx(-23.0)

    (failure,) = gate._judge(paired, chance=0.0, max_delta=1.0)
    assert "the runtimes disagree" in failure
    assert "More problems will not change this" in failure
    assert "too few problems" not in failure


def test_an_interval_too_wide_to_conclude_anything_says_so(gate) -> None:
    """The third outcome, and the reason "not significantly different" cannot pass.

    On 40 problems the interval spans 35 points. Nothing was measured; a gate that
    accepted this would certify any pair of arms that were scored briefly enough.

    Turns red when: the wide-and-inconclusive case starts passing, or starts being
    reported as a disagreement.
    """
    paired = _paired(both_right=15, a_only=5, b_only=8, both_wrong=12, p=0.58)
    low, high = paired.interval_points
    assert low < 0.0 < high, "the fixture is only interesting if it contains zero"

    (failure,) = gate._judge(paired, chance=0.0, max_delta=1.0)
    assert "too few problems to tell" in failure
    assert "40 problems" in failure
    assert "the runtimes disagree" not in failure


def test_two_equally_destroyed_arms_do_not_pass_by_agreeing(gate) -> None:
    """Perfect agreement at the chance floor is agreement about nothing.

    Turns red when: the floor check is dropped. It is the one condition that makes
    the equivalence mean the runtimes both work, rather than that they both failed.
    """
    paired = _paired(both_right=25, a_only=0, b_only=0, both_wrong=75, p=1.0)
    failures = gate._judge(paired, chance=0.25, max_delta=1.0)

    assert len(failures) == 2, "both arms are at the floor, so both are named"
    assert all("measured nothing" in failure for failure in failures)
    assert {"transformers", "vllm"} == {failure.split()[0] for failure in failures}
