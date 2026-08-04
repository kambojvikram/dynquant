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
