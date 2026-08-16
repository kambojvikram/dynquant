"""The audio side-channel: what it must not change, and what it must not get wrong.

Two halves, and the first is the more important one. Widening a shared harness is
only safe if the widening is unreachable from the path everything else takes, so the
first block asserts that a text task's journey through ``generate_batched`` is
byte-for-byte the journey it took before -- including that a backend written against
the *old* signature is never handed the new argument.

The second block is about the failure mode audio actually has. Every mistake
available here -- padding the wrong axis, permuting the ids without permuting the
spectrograms, slicing the continuation at the wrong width -- produces no exception.
It produces a model answering about the wrong clip, which scores above chance and
below the truth and reads as a weak checkpoint. So each one is asserted against a
stub that records exactly what reached ``generate``.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from _decode_stub import PAD_ID, StubModel, StubTokenizer  # noqa: E402

from dynquant.eval.harness import (  # noqa: E402
    AudioPrompt,
    EvalBackend,
    EvalConfig,
    TransformersBackend,
    encode_prompts,
    generate_batched,
)
from dynquant.eval.omni import OmniThinkerBackend, batch_features  # noqa: E402


class StubProcessor:
    """A processor is a tokenizer plus feature extractors; the tests only need the first."""

    def __init__(self) -> None:
        self.tokenizer = StubTokenizer()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.tokenizer(*args, **kwargs)

    def batch_decode(self, *args: Any, **kwargs: Any) -> list[str]:
        return self.tokenizer.batch_decode(*args, **kwargs)


def _features(value: float, frames: int = 3) -> dict[str, torch.Tensor]:
    """One example's processor output: a constant spectrogram and its mask.

    Constant, and different per example, so a row that ends up beside the wrong
    prompt says which prompt it belonged to instead of merely being wrong.
    """
    return {
        "input_features": torch.full((1, 2, frames), float(value)),
        "feature_attention_mask": torch.ones((1, frames), dtype=torch.long),
    }


# --------------------------------------------------------------------------
# What the widening must not change
# --------------------------------------------------------------------------


class _OldStyleBackend(EvalBackend):
    """A backend written before ``extras`` existed.

    `EvalBackend` is public and the subclasses that matter are the ones outside this
    package, so the compatibility that counts is not "a defaulted keyword is
    harmless" -- it is that a text task never passes the keyword at all. This class
    would raise `TypeError` if it did.
    """

    name = "old-style"

    def __init__(self) -> None:
        self.seen = 0

    def generate_ids(  # type: ignore[override]
        self,
        prompt_ids: Any,
        config: EvalConfig,
        *,
        progress: Any = None,
    ) -> list[list[int]]:
        self.seen += 1
        return [[PAD_ID + 1] for _ in prompt_ids]


def test_a_text_task_never_hands_a_backend_the_new_argument() -> None:
    backend = _OldStyleBackend()
    tokenizer = StubTokenizer()
    out = generate_batched(backend, tokenizer, ["one two", "three"], EvalConfig())
    assert len(out) == 2
    assert backend.seen == 1


def test_a_batch_with_no_audio_reports_no_extras_at_all() -> None:
    """Empty rather than a tuple of ``None``, so ``if encoded.extras`` is the whole test."""
    encoded = encode_prompts(StubTokenizer(), ["one two", [4, 5]], EvalConfig())
    assert encoded.extras == ()
    assert encoded.ids[1] == [4, 5]


def test_an_audio_prompt_keeps_its_features_beside_its_own_ids() -> None:
    audio = AudioPrompt(ids=(7, 8, 9), features=_features(1.0))
    encoded = encode_prompts(StubTokenizer(), ["one two", audio, [4]], EvalConfig())
    assert encoded.ids == [encoded.ids[0], [7, 8, 9], [4]]
    assert encoded.extras[0] is None
    assert encoded.extras[1] is audio.features
    assert encoded.extras[2] is None


def test_an_audio_prompt_that_does_not_fit_is_refused_rather_than_cut() -> None:
    """Front-truncation is right for text and catastrophic here.

    The ids reserve one position per encoder frame. Slicing some of them off while
    the spectrogram stays whole offsets the audio against the text, and nothing in
    ``generate`` notices -- so the number comes back low and looks like the model.
    """
    audio = AudioPrompt(ids=tuple(range(1, 12)), features=_features(1.0))
    with pytest.raises(ValueError, match="max_prompt_tokens=4"):
        encode_prompts(StubTokenizer(), [audio], EvalConfig(max_prompt_tokens=4))


def test_the_reference_backend_refuses_audio_and_names_the_one_that_takes_it() -> None:
    backend = TransformersBackend(StubModel(StubTokenizer(), "ok"), StubTokenizer())
    with pytest.raises(NotImplementedError, match="OmniThinkerBackend"):
        backend.generate_ids([[1, 2]], EvalConfig(), extras=[_features(1.0)])


# --------------------------------------------------------------------------
# Batching the encoder inputs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("axis", [1, 2])
def test_the_ragged_axis_is_found_rather_than_assumed(axis: int) -> None:
    """``[batch, mels, frames]`` and ``[batch, frames, mels]`` are both real layouts.

    A rule like "pad the last axis" is right for one and silently wrong for the
    other: it does not raise, it hands the encoder a spectrogram whose frames have
    been zero-extended along the mel axis.
    """
    shapes = [[1, 2, 2], [1, 2, 2]]
    shapes[0][axis] = 5
    rows = [{"x": torch.ones(*shape)} for shape in shapes]
    out = batch_features(rows)["x"]
    assert tuple(out.shape) == (2, *[5 if i == axis else 2 for i in (1, 2)])
    assert bool((out[1].movedim(axis - 1, 0)[2:] == 0).all())


def test_the_padding_a_batch_added_is_marked_in_the_companion_mask() -> None:
    batched = batch_features([_features(1.0, frames=6), _features(2.0, frames=2)])
    assert tuple(batched["input_features"].shape) == (2, 2, 6)
    assert batched["feature_attention_mask"][1].tolist() == [1, 1, 0, 0, 0, 0]
    assert bool((batched["input_features"][1, :, 2:] == 0).all())


def test_two_ragged_axes_are_an_error_rather_than_a_guess() -> None:
    with pytest.raises(ValueError, match="varies along axes"):
        batch_features([{"x": torch.ones(1, 3, 4)}, {"x": torch.ones(1, 5, 6)}])


def test_features_may_not_carry_the_ids_the_harness_owns() -> None:
    """A second copy of the ids would reach ``generate`` as a keyword and win."""
    with pytest.raises(ValueError, match="input_ids"):
        batch_features([{"input_ids": torch.ones(1, 3, dtype=torch.long)}])


def test_examples_that_present_different_inputs_cannot_be_batched() -> None:
    with pytest.raises(ValueError, match="disagree on which keys"):
        batch_features([_features(1.0), {"input_features": torch.ones(1, 2, 3)}])


# --------------------------------------------------------------------------
# The backend
# --------------------------------------------------------------------------


class RecordingModel(StubModel):
    """The stub, plus the one tensor it does not otherwise keep.

    ``StubModel.fed`` records the batch as *words*, which is the right resolution for
    asserting truncation. Here the question is finer -- which row of the spectrogram
    stack sits beside which row of the ids -- so the ids themselves are kept.
    """

    def __init__(self, tokenizer: StubTokenizer, reply: str) -> None:
        super().__init__(tokenizer, reply)
        self.batches: list[torch.Tensor] = []

    def generate(self, *, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.batches.append(input_ids.clone())
        return super().generate(input_ids=input_ids, **kwargs)


def _stub_pair(reply: str = "reply", vocab: int = 24) -> tuple[StubProcessor, RecordingModel]:
    """A processor whose ids 1..``vocab`` are real words, and a model on top of it.

    Registered up front because the stub decodes every batch it is fed, and a prompt
    built from bare integers would otherwise name ids the tokenizer has never issued.
    """
    processor = StubProcessor()
    processor.tokenizer.encode_words(" ".join(f"w{i}" for i in range(1, vocab + 1)))
    return processor, RecordingModel(processor.tokenizer, reply)


def _run(prompts: list[AudioPrompt], **config_kwargs: Any) -> tuple[RecordingModel, list[str]]:
    processor, model = _stub_pair()
    backend = OmniThinkerBackend(model, processor)
    config = EvalConfig(max_new_tokens=4, batch_size=8, **config_kwargs)
    return model, generate_batched(backend, processor, prompts, config)


def test_the_spectrograms_are_permuted_with_the_ids_they_belong_to() -> None:
    """The test this file exists for.

    Prompts are sorted long-to-short so the first batch is the memory high-water mark
    and an OOM arrives in the first minute rather than the fortieth. That permutation
    has to reach the features too, and a version that applied it to the ids alone
    would raise nothing: every row would still have *a* spectrogram, just another
    example's, and the run would score somewhere between chance and the truth.

    So each prompt carries the same marker in both halves -- its ids are all
    ``i + 1`` and its features are filled with ``i`` -- and every row of what reached
    ``generate`` is checked to carry the one marker twice.
    """
    prompts = [
        AudioPrompt(ids=tuple([index + 1] * (index + 3)), features=_features(index))
        for index in range(5)
    ]
    model, _ = _run(prompts)

    assert model.calls == 1, "one batch, so the ordering is the only variable"
    fed_features = model.kwargs[0]["input_features"]
    assert tuple(fed_features.shape) == (5, 2, 3)
    for row, ids in enumerate(model.batches[0].tolist()):
        real = {token for token in ids if token != PAD_ID}
        assert len(real) == 1, f"row {row} mixes prompts: {ids}"
        assert fed_features[row].unique().tolist() == [float(real.pop() - 1)]


def test_the_batch_is_left_padded_so_every_row_ends_on_a_real_token() -> None:
    prompts = [
        AudioPrompt(ids=(1, 1, 1), features=_features(0)),
        AudioPrompt(ids=(2,), features=_features(1)),
    ]
    model, _ = _run(prompts)
    fed = model.batches[0].tolist()
    assert [row[-1] for row in fed] == [1, 2]
    assert fed[1][0] == PAD_ID, "the shorter row should be padded on the left"


def test_a_prompt_arriving_without_its_encoder_inputs_is_refused() -> None:
    processor, model = _stub_pair()
    backend = OmniThinkerBackend(model, processor)
    with pytest.raises(ValueError, match="one feature mapping per prompt"):
        backend.generate_ids([[1, 2]], EvalConfig())
    with pytest.raises(ValueError, match="carry no encoder inputs"):
        backend.generate_ids([[1, 2], [3]], EvalConfig(), extras=[_features(1.0), None])


def test_a_keyword_the_model_does_not_declare_is_dropped_before_generate() -> None:
    """``return_audio`` switches off a vocoder this evaluation does not score.

    The composite checkpoint declares it and the Thinker alone does not, and both are
    legitimate things to hand this backend -- so it is filtered against the signature
    rather than passed and hoped for. The stub declares no such parameter, which is
    the Thinker's case.
    """
    model, _ = _run([AudioPrompt(ids=(1, 2), features=_features(0))])
    assert "return_audio" not in model.kwargs[0]


def test_the_continuation_is_located_by_confirming_the_prompt_not_by_trusting_it() -> None:
    """``[:, width:]`` is right by convention and catastrophic when the convention
    does not hold: it would slice past the end of every answer and score a working
    model at zero, silently. So the prefix is compared against what was sent."""

    class Rewriting(RecordingModel):
        def generate(self, *, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            return torch.roll(super().generate(input_ids=input_ids, **kwargs), 1, dims=1)

    processor = StubProcessor()
    processor.tokenizer.encode_words(" ".join(f"w{i}" for i in range(1, 25)))
    backend = OmniThinkerBackend(Rewriting(processor.tokenizer, "w9 w10 w11 w12"), processor)
    with pytest.raises(RuntimeError, match="are not the prompt that was sent"):
        backend.generate_ids([[1, 2, 3]], EvalConfig(max_new_tokens=2), extras=[_features(0)])
