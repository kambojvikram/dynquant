"""Scoring a model the CLI could not have loaded, through the evaluator the CLI uses.

``dynquant eval`` takes a path. A post-training baseline does not have one: llm-compressor
returns a quantized model object, and at 3 bits ``compressed-tensors`` has no packed format,
so the only checkpoint a path could point at is a dequantized bf16 copy four times the size
of the arm it represents. The alternative -- a second evaluator beside the command -- is what
``experiments/four_point`` did, and the cost of it is that its records cannot be paired
against a ``dynquant eval`` record without arguing that two implementations of the same
settings agree.

So ``run`` takes an optional model. The tests here are about the two ways that seam fails
without saying so, not about the arithmetic below it:

* the passed model is dropped and the *path* is loaded instead -- which scores the
  unquantized checkpoint and reports it under the quantized arm's label,
* and ``--map`` is honoured alongside it, which quantizes an already-quantized model.

Both produce a number. Neither produces an error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from dynquant.commands import evaluate
from dynquant.errors import DynQuantError


class _Result:
    """Enough of a task result for ``run`` to build a record from."""

    label = "stub"
    total = 3
    hits: ClassVar[list[bool]] = [True, False, True]
    predictions: ClassVar[list[str]] = []
    unparseable = 0

    @property
    def accuracy(self) -> float:
        return sum(1 for hit in self.hits if hit) / self.total

    def summary(self) -> str:
        return "stub: 2/3"

    def as_dict(self) -> dict[str, Any]:
        return {"stub": True}


class _Spec:
    """A task spec that loads nothing and records what it was handed to score.

    Substituted for a real entry in ``TASKS`` rather than driving a real task, because what
    is under test is which *model object* reaches ``evaluate`` -- and a real task would add a
    dataset download and a generation loop between the assertion and the thing it asserts.
    """

    key = "stub"
    chance = 0.0
    max_new_tokens = 8
    max_prompt_tokens = 32
    batch_size = 1
    split = "test"
    shot_split = None
    add_special_tokens = True
    takes_shots = False
    takes_style = False
    unverifiable = False
    executes_code = False
    detail = False

    def __init__(self) -> None:
        self.scored: Any = None

    def load(self, split: str | None) -> list[Any]:
        return [object(), object(), object()]

    def evaluate(self, model: Any, *_args: Any, **_kwargs: Any) -> _Result:
        self.scored = model
        return _Result()

    def unscored_count(self, result: Any) -> int:
        return 0

    def detail_of(self, result: Any) -> dict[str, Any] | None:
        return None


@pytest.fixture
def wired(monkeypatch: pytest.MonkeyPatch) -> _Spec:
    """Register the stub task and make loading anything from disk a failure.

    ``_load_runtime`` raising is the assertion, not a convenience. A seam that quietly falls
    through to it is the failure mode with the worst consequence in this campaign: the record
    still carries the quantized arm's label and its accuracy is the ceiling's.
    """
    spec = _Spec()
    monkeypatch.setitem(evaluate.TASKS, "stub", spec)  # type: ignore[arg-type]

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("_load_runtime was called even though a model was passed")

    monkeypatch.setattr(evaluate, "_load_runtime", refuse)
    monkeypatch.setattr(evaluate._shared, "load_tokenizer", lambda *_a, **_k: object())
    return spec


def _args(**overrides: Any) -> argparse.Namespace:
    """A namespace shaped the way the real parser shapes one.

    Built here rather than parsed, because the stub task is not a ``--task`` choice the real
    parser accepts. ``experiments/phase4/baselines_lfm2.py`` does the opposite -- it parses
    through :func:`dynquant.cli.build_parser` -- and that split is deliberate: the driver has
    to inherit every default the CLI has, and this test has to reach a task the CLI has never
    heard of.
    """
    base: dict[str, Any] = {
        "task": "stub",
        "backend": "transformers",
        "split": None,
        "shots": None,
        "shot_split": None,
        "shot_seed": 0,
        "limit": None,
        "batch_size": None,
        "max_new_tokens": None,
        "max_prompt_tokens": None,
        "model": "/some/checkpoint#gptq-4b",
        "tokenizer": None,
        "trust_remote_code": False,
        "device": "cuda",
        "dtype": "bfloat16",
        "map": None,
        "group_size": 128,
        "label": None,
        "quiet": True,
        "keep_predictions": None,
        "out": None,
        "json": False,
        "compare": None,
        "min_accuracy": None,
        "allow_execution": False,
        "prompt_style": "auto",
        "exec_timeout": None,
        "exec_memory_mb": None,
        "on_unverifiable": "raise",
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_the_passed_model_is_the_one_that_gets_scored(wired: _Spec) -> None:
    """The whole point, and the failure with no symptom.

    If ``run`` reached for the path anyway, the arm would be scored from an unquantized
    checkpoint under a quantized label -- a plausible number, in the direction that flatters
    quantization, with nothing in the record to contradict it.

    Turns red when: the ``model is None`` guard is dropped, or the parameter is accepted and
    then rebound by the loader.
    """
    sentinel = object()
    assert evaluate.run(_args(), model=sentinel) == 0
    assert wired.scored is sentinel


def test_a_passed_model_plus_a_bit_map_is_refused(wired: _Spec) -> None:
    """Two quantizers cannot both own the same weights.

    ``--map`` builds the packed runtime *from* an unquantized model. Handed one that arrives
    already quantized, either behaviour is a guess about which the caller meant, and the
    wrong guess reports a doubly-quantized model under one method's name.

    Turns red when: the map is silently ignored, which is the tempting implementation because
    the passed model makes ``_load_runtime`` -- where packing happens -- unreachable.
    """
    with pytest.raises(DynQuantError, match="cannot also be"):
        evaluate.run(_args(map="widths.json"), model=object())


def test_the_record_does_not_claim_to_have_measured_a_packing(wired: _Spec, tmp_path: Path) -> None:
    """``packed`` is where this command reports what *it* packed, and it packed nothing.

    A caller with its own size accounting writes it beside this record. Filling the field in
    from somewhere else would put a number under a key whose meaning is "measured here".

    Turns red when: ``packed`` is left carrying a value from a previous branch, or the
    in-memory path starts synthesising one.
    """
    out = tmp_path / "arm.json"
    assert evaluate.run(_args(out=str(out)), model=object()) == 0

    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["packed"] is None
    # The provenance field is the caller's to set and is recorded verbatim -- the driver
    # qualifies it with the recipe, because the weights are no longer what the path holds.
    assert record["model"] == "/some/checkpoint#gptq-4b"
    assert record["hits"] == [True, False, True]


def test_a_path_is_still_loaded_when_no_model_is_passed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI's own path, unchanged.

    Asserted because the guard added for the in-memory case sits directly on it, and a
    mis-written condition would make ``dynquant eval <dir>`` stop loading anything -- which
    every other test in this file has monkeypatched away.

    Turns red when: the ``model is None`` branch inverts, or stops calling ``_load_runtime``.
    """
    spec = _Spec()
    monkeypatch.setitem(evaluate.TASKS, "stub", spec)  # type: ignore[arg-type]
    loaded = object()
    monkeypatch.setattr(evaluate, "_load_runtime", lambda *_a, **_k: (loaded, {"gib": 1.0}))
    monkeypatch.setattr(evaluate._shared, "load_tokenizer", lambda *_a, **_k: object())

    assert evaluate.run(_args()) == 0
    assert spec.scored is loaded
