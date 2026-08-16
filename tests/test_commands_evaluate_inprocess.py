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


class _Spec(evaluate._TaskSpec):
    """A task spec that loads nothing and records what it was handed to score.

    Substituted for a real entry in ``TASKS`` rather than driving a real task, because what
    is under test is which *model object* reaches ``evaluate`` -- and a real task would add a
    dataset download and a generation loop between the assertion and the thing it asserts.

    A subclass and not a stand-alone stub with the same attribute names. The two are
    interchangeable right up to the day a field is added to `_TaskSpec`: the copy keeps
    passing every test that does not touch the new field and fails the command with an
    `AttributeError` from a line no test in this file is about. Inheriting means a new
    capability arrives here with the default the real class gives it, and a field that is
    *renamed* breaks the constructor call below rather than the command.
    """

    def __init__(self) -> None:
        super().__init__(
            "stub",
            shots=0,
            chance=0.0,
            max_new_tokens=8,
            max_prompt_tokens=32,
            batch_size=1,
            shot_split=None,
        )
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


# --- the comparability contract ---------------------------------------------------------


def test_an_unusable_out_is_refused_before_anything_is_loaded(wired: _Spec, tmp_path: Path) -> None:
    """The destination is checked first, because it is the last thing ``run`` uses.

    The record is written on the final line of :func:`run`, so every way ``--out`` can be
    wrong surfaces after the decode sweep has been paid for. That is not hypothetical: a
    control arm on this campaign pointed ``--out`` at the panel *directory* rather than a file
    inside it, quantized for seven minutes, generated for eleven hours, printed its accuracy
    to stdout, and died in ``write_text`` -- eleven GPU-hours for a number that reached no
    file.

    ``wired`` makes ``_load_runtime`` raise, so the ordering is what is being asserted and not
    just the message: a check that ran anywhere below the model load would surface as that
    ``AssertionError`` instead of the refusal.

    Turns red when: the call leaves the top of ``run``, or a directory stops being rejected.
    """
    with pytest.raises(DynQuantError, match="is a directory"):
        evaluate.run(_args(out=str(tmp_path)))


def test_a_writable_destination_is_accepted_and_its_parent_made(tmp_path: Path) -> None:
    """The check has to be quiet on the paths the campaign actually uses.

    ``run`` creates ``destination.parent`` itself, so a nested ``--out`` is legal and must not
    be turned into an error by the guard that precedes it. The unwritable case is exercised
    through a parent that is a file rather than through permissions, which do not mean the
    same thing across the platforms this suite runs on.

    Turns red when: the guard starts rejecting a path ``run`` would have written, or stops
    reporting one it could not.
    """
    evaluate.check_out_is_writable(None)
    nested = tmp_path / "arms" / "s4" / "gptq_3b_asym.json"
    evaluate.check_out_is_writable(str(nested))
    assert nested.parent.is_dir()
    # And the probe leaves nothing behind. ``arms_lfm2``'s ``--resume`` skips an arm whose
    # record ``is_file()``, so an empty file here would let a crashed arm be skipped as
    # done -- and be found much later as a corrupt record rather than a missing one.
    assert not nested.exists()

    # An existing record is left exactly as it was, because append mode is a probe here
    # and not a write, and because deleting one would destroy the arm it belongs to.
    kept = tmp_path / "already.json"
    kept.write_text('{"accuracy": 0.5}', encoding="utf-8")
    evaluate.check_out_is_writable(str(kept))
    assert kept.read_text(encoding="utf-8") == '{"accuracy": 0.5}'

    blocked = tmp_path / "notadir"
    blocked.write_text("", encoding="utf-8")
    with pytest.raises(DynQuantError, match="cannot be written"):
        evaluate.check_out_is_writable(str(blocked / "arm.json"))


def _record(**overrides: Any) -> dict[str, Any]:
    """A record shaped the way ``run`` writes one."""
    base: dict[str, Any] = {
        "task": "text2sql",
        "backend": "transformers",
        "split": "test",
        "shots": 2,
        "shot_seed": 0,
        "limit": 400,
        "label": "arm",
        "decode": {"max_new_tokens": 1024, "batch_size": 32, "greedy": True},
        "hits": [True, False, True],
    }
    base.update(overrides)
    return base


def test_two_arms_at_different_decode_budgets_do_not_pair(tmp_path: Path) -> None:
    """The setting that was worth 52 points on this task, and was not in the contract.

    A 256-token cap on a model that deliberates before answering scored 5.50% on text2sql;
    the same model on the same 400 problems at 1024 scores 57.75%. Paired against each
    other those two records would have produced a McNemar test with p < 1e-50 and a
    quantization story attached to it.

    Turns red when: ``DECODE_PAIRING_FIELDS`` empties, or the comparison stops reaching
    into ``decode`` -- which is the easy mistake, because every other field is top-level.
    """
    other = tmp_path / "ceiling.json"
    other.write_text(
        json.dumps(_record(label="ceiling", decode={"max_new_tokens": 256, "greedy": True})),
        encoding="utf-8",
    )
    with pytest.raises(DynQuantError, match=r"decode\.max_new_tokens=256"):
        evaluate._compare(_record(), str(other))


def test_a_record_written_before_the_budget_was_recorded_does_not_pair_silently(
    tmp_path: Path,
) -> None:
    """Absent is not equal, and absent is not ``None`` either.

    ``split`` is legitimately ``None`` for a single-set dataset, so absence cannot be
    spelled that way -- and a guard that used ``.get()`` for both sides would compare
    ``None`` against ``None`` on a record that never wrote the field, and pass.

    Turns red when: the sentinel is replaced by ``None``, or the missing field is filled in
    from a default rather than refused.
    """
    other = tmp_path / "old.json"
    payload = _record(label="old")
    del payload["decode"]
    other.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DynQuantError, match="absent"):
        evaluate._compare(_record(), str(other))


def test_a_matched_pair_still_compares(tmp_path: Path, capsys: Any) -> None:
    """The guard has to let the campaign's own arms through.

    Every arm in the six-arm panel differs from the bf16 ceiling in exactly one thing --
    the weights -- and a contract that refused those would be a contract nobody could use.

    Turns red when: a field is added to the contract that the arms cannot hold constant.
    """
    other = tmp_path / "ceiling.json"
    other.write_text(
        json.dumps(_record(label="ceiling", hits=[True, True, True])), encoding="utf-8"
    )
    assert evaluate._compare(_record(), str(other))
    capsys.readouterr()


def test_a_top_level_field_missing_from_this_run_is_a_bug_not_a_match(tmp_path: Path) -> None:
    """The sentinel has to work on the flat fields too, and one test did not prove it.

    ``split`` is ``None`` for a single-set dataset. So a record that never wrote ``split``
    and one that wrote ``None`` into it are indistinguishable under ``.get()`` -- the guard
    compares ``None`` to ``None``, agrees, and the "that is a bug in `dynquant eval`" check
    above it never fires. Every other field in the contract has the same hole, and the
    consequence is a paired test run across a setting nobody checked.

    Found by mutation: deleting the sentinel from the ``PAIRING_FIELDS`` comprehension left
    the suite green, because the only test of it deleted ``decode``, which is read on the
    other branch.

    Turns red when: the sentinel is dropped from the top-level read.
    """
    other = tmp_path / "single_set.json"
    other.write_text(json.dumps(_record(label="other", split=None)), encoding="utf-8")

    mine = _record()
    del mine["split"]
    with pytest.raises(DynQuantError, match="bug in `dynquant eval`"):
        evaluate._compare(mine, str(other))


class _Config:
    """Just enough of a transformers config to carry an experts dispatch."""

    def __init__(self, implementation: str) -> None:
        self._experts_implementation = implementation


class _MoE:
    """A model exposing the two things ``use_eager_experts`` drives, and nothing else.

    Deliberately not a transformers model. The production helper reads
    ``config._experts_implementation`` and calls ``set_experts_implementation``; a stub
    that implements exactly those two is what proves the pin goes through the supported
    seam rather than assigning the private attribute behind the model's back -- which
    would leave ``post_init``'s bookkeeping stale and the model computing something no
    field of it admits to.
    """

    def __init__(self, implementation: str = "grouped_mm") -> None:
        self.config = _Config(implementation)
        self.moved: list[str] = []

    def set_experts_implementation(self, implementation: str) -> None:
        self.moved.append(implementation)
        self.config._experts_implementation = implementation


def test_a_moe_is_pinned_to_eager_and_the_record_says_which(wired: _Spec, tmp_path: Path) -> None:
    """The confound this field exists for, in the configuration that produced it.

    A packed arm runs on ``eager`` because ``pack_model`` forces it there; an encoded arm
    runs on whatever ``post_init`` chose, which in transformers 5.14.1 is ``grouped_mm``.
    On LFM2.5-8B-A1B those two disagree on 1.24% of teacher-forced tokens, 0.29x the
    effect of quantizing that model to 4 bits -- the same order as the margins the panel
    reports. Pinning here is what makes a set of arms one experiment.

    Turns red when: the pin moves inside ``_load_runtime``, where a model passed through
    ``model=`` never reaches it, or the dispatch stops being written into the record.
    """
    model = _MoE()
    out = tmp_path / "arm.json"
    assert evaluate.run(_args(out=str(out)), model=model) == 0

    assert model.moved == ["eager"]
    assert model.config._experts_implementation == "eager"
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["experts"] == {"found": "grouped_mm", "ran": "eager"}


def test_a_model_with_no_experts_dispatch_records_that_rather_than_guessing(
    wired: _Spec, tmp_path: Path
) -> None:
    """Dense models and linearised baselines are the common case, and must say so.

    ``llm-compressor`` rewrites an expert bank into per-expert ``Linear`` modules, so a
    GPTQ or AWQ arm has no dispatch left to pin and already computes what ``eager``
    computes. Writing ``None`` rather than omitting the key is the difference between
    "there was nothing to choose" and "this run predates the field" -- the same record
    shape, two different claims.

    Turns red when: the helper invents a value for a model that has no dispatch, which
    would stop a dense arm pairing against the record of another dense arm.
    """
    out = tmp_path / "arm.json"
    assert evaluate.run(_args(out=str(out)), model=object()) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["experts"] is None


def test_auto_leaves_the_models_own_choice_and_still_records_it(
    wired: _Spec, tmp_path: Path
) -> None:
    """The escape hatch has to be a real one, or the dispatch cost cannot be measured.

    Measuring what the pin costs means running the same arm both ways, so ``auto`` has to
    reach the scorer unmoved -- and has to write down what ran anyway, because a record
    that names the dispatch only when it is ``eager`` cannot report the comparison it was
    made for.

    Turns red when: ``auto`` decays into an alias for ``eager``, or the recording is moved
    onto the pinning branch.
    """
    model = _MoE()
    out = tmp_path / "arm.json"
    assert evaluate.run(_args(out=str(out), experts_impl="auto"), model=model) == 0

    assert model.moved == []
    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["experts"] == {"found": "grouped_mm", "ran": "grouped_mm"}


def test_two_arms_on_different_experts_dispatches_do_not_pair(tmp_path: Path) -> None:
    """The four arms this campaign had already landed, and no record could have caught.

    bf16 and dq were scored on ``grouped_mm``; GPTQ and AWQ, whose banks ``llm-compressor``
    had linearised, computed eager. The dq-minus-GPTQ margin was +0.64 points against a
    dispatch effect worth roughly 0.45 if accuracy tracked token agreement. Nothing in
    either record said the two had run different arithmetic.

    Turns red when: ``EXPERTS_PAIRING_FIELDS`` empties, or the comparison stops reaching
    into the ``experts`` block -- the easy mistake, since two of the three nested blocks
    it reads are about settings and this one is about the computation.
    """
    other = tmp_path / "panel.json"
    other.write_text(
        json.dumps(_record(label="panel", experts={"found": "grouped_mm", "ran": "grouped_mm"})),
        encoding="utf-8",
    )
    with pytest.raises(DynQuantError, match=r"experts\.ran='grouped_mm'"):
        evaluate._compare(_record(experts={"found": "eager", "ran": "eager"}), str(other))


def test_two_arms_that_both_have_no_dispatch_still_pair(tmp_path: Path, capsys: Any) -> None:
    """Absence has to stay pairable, or the field breaks every dense comparison there is.

    A dense model has no experts dispatch and never will, so ``None`` on both sides is not
    a mismatch. It also must not be spelled as the string ``'None'``, which would compare
    unequal to the absence in every record written before this field existed -- retiring
    the campaign's own history to fix a MoE problem dense models do not have.

    Turns red when: ``experts.ran`` is promoted out of ``_OPTIONAL_COMPARABILITY``, or a
    ``None`` block starts being read as a value rather than as absence.
    """
    other = tmp_path / "dense.json"
    other.write_text(json.dumps(_record(label="dense", experts=None)), encoding="utf-8")
    assert evaluate._compare(_record(), str(other))
    capsys.readouterr()
