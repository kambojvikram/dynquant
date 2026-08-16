"""A backend for a model whose prompt arrives as ids *and* encoder inputs.

Why this is a separate class
----------------------------
:class:`~dynquant.eval.harness.TransformersBackend` batches token ids and nothing
else, and that is the whole reason it can be shared by every text task: left-pad,
build the mask, slice at the prompt width. None of those three steps has a
model-specific answer.

Audio breaks that. The processor emits a mel spectrogram alongside the ids, ragged
in the frame axis, with a companion mask that has to be padded the same way and a
placeholder run inside the ids whose length is a function of the frame count. Which
axis is ragged and what the pad value means are properties of the *processor*, not
of generation, so putting them in the shared backend would put one model's layout
on the path every published arm takes.

So the shared path keeps its two lines and this class carries the modality. It is
reachable only through :class:`~dynquant.eval.harness.AudioPrompt`, which no text
task constructs.

What it is careful about
------------------------
**The halves stay together.** Prompts are still sorted long-to-short, so the first
batch is the memory high-water mark and an OOM arrives in the first minute rather
than the fortieth. The permutation is applied to the features in the same
expression that applies it to the ids -- not in a second loop over the same index
list, which is where a reordering bug would be invisible and would score every
example against another example's audio.

**Padding is measured, not assumed.** The ragged axis is *found* by comparing the
shapes in the chunk, and more than one ragged axis is an error. A rule like "pad
the last axis" would be right for ``[batch, mels, frames]`` and silently wrong for
``[batch, frames, mels]`` -- and wrong here does not raise, it feeds the encoder a
transposed-looking spectrogram and returns a plausible low score.

**The prompt width is confirmed, not trusted.** ``generate`` returning
prompt-plus-continuation is a convention, not a guarantee, and a wrapper that
returns only the continuation would make ``[:, width:]`` slice away the answer and
score zero. So the prefix is compared against what was sent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from dynquant._logging import get_logger

from .harness import (
    EvalBackend,
    EvalConfig,
    _stopping_criteria,
    greedy_generation_config,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = ["OmniThinkerBackend", "batch_features"]

_log = get_logger(__name__)

RESERVED_KEYS = ("input_ids", "attention_mask")
"""Keys an :class:`~dynquant.eval.harness.AudioPrompt`'s ``features`` must not carry.

The ids travel in ``AudioPrompt.ids`` and the harness pads them. A second copy
inside ``features`` would reach ``generate`` as a keyword and win, so the model
would read un-padded, un-sorted ids while the harness sliced the output at a width
computed from the padded ones. Refused rather than merged: there is no reading of
two disagreeing copies that is safe to guess at.
"""


def batch_features(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine per-example processor outputs into one batch.

    Each row is what the processor returned for a single conversation, batch axis
    included, so every tensor has ``shape[0] == 1`` and the batch is a concatenation
    along that axis once the ragged axis is padded to a common width.

    Padding is with zeros, which is correct for both halves of the pair it exists
    for: a zero mel frame is silence, and a zero in ``feature_attention_mask`` is
    exactly the flag that tells the encoder to ignore it. A key whose shapes already
    agree is concatenated untouched.

    Non-tensor values are passed through when every row agrees and refused when they
    do not, because a scalar that differs per example is a per-example setting and
    collapsing it to the first row's value would apply one example's setting to the
    whole batch.
    """
    if not rows:
        return {}

    keys = list(rows[0])
    for index, row in enumerate(rows[1:], start=1):
        if list(row) != keys:
            raise ValueError(
                f"processor outputs disagree on which keys they carry: example 0 has "
                f"{sorted(keys)}, example {index} has {sorted(row)}. Every example in a "
                f"batch has to present the same inputs, or `generate` sees a keyword for "
                f"some rows and not others."
            )

    batched: dict[str, Any] = {}
    for key in keys:
        if key in RESERVED_KEYS:
            raise ValueError(
                f"processor output carries {key!r}, which the harness owns. Put the ids in "
                f"AudioPrompt.ids and leave {list(RESERVED_KEYS)} out of `features`; see "
                f"dynquant.eval.omni.RESERVED_KEYS."
            )
        values = [row[key] for row in rows]
        if not all(torch.is_tensor(value) for value in values):
            unique = {repr(value) for value in values}
            if len(unique) != 1:
                raise ValueError(
                    f"{key!r} is not a tensor and differs across the batch ({sorted(unique)[:3]}"
                    f"...). A per-example setting cannot be batched by taking the first one."
                )
            batched[key] = values[0]
            continue
        batched[key] = _cat_padded(key, values)
    return batched


def _cat_padded(key: str, values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate along the batch axis, zero-padding the one axis that varies."""
    leading = {int(value.shape[0]) for value in values}
    if leading != {1}:
        raise ValueError(
            f"{key!r} has batch sizes {sorted(leading)}; each AudioPrompt's features must be "
            f"the processor's output for one example, which carries a batch axis of 1."
        )
    dims = {value.dim() for value in values}
    if len(dims) != 1:
        raise ValueError(f"{key!r} has ranks {sorted(dims)} across the batch; they must agree.")

    ndim = values[0].dim()
    ragged = [
        axis for axis in range(1, ndim) if len({int(value.shape[axis]) for value in values}) > 1
    ]
    if not ragged:
        return torch.cat(list(values), dim=0)
    if len(ragged) > 1:
        raise ValueError(
            f"{key!r} varies along axes {ragged}. One ragged axis is a length -- frames, "
            f"patches -- and can be padded with a mask to say where the padding is. Two is "
            f"a layout this function cannot pad without knowing what the axes mean."
        )

    axis = ragged[0]
    width = max(int(value.shape[axis]) for value in values)
    padded = []
    for value in values:
        short = width - int(value.shape[axis])
        if short:
            # `F.pad` counts pairs from the *last* dimension backwards.
            spec = [0, 0] * (ndim - 1 - axis) + [0, short]
            value = torch.nn.functional.pad(value, spec)
        padded.append(value)
    return torch.cat(padded, dim=0)


class OmniThinkerBackend(EvalBackend):
    """``generate`` on a multi-modal checkpoint, scored on its text output.

    The class is named for what it measures rather than for what it loads. A
    Qwen3-Omni checkpoint is a Thinker (the MoE that reads audio and writes text)
    and a Talker (the head that turns that text back into speech); DynQuant's
    allocation covers the Thinker, which holds ~97% of the parameters, and the
    number a quantization arm has to move is the Thinker's. Speech output is
    therefore switched off -- see ``return_audio`` below -- so the run is not paying
    a vocoder to synthesise audio nothing scores.
    """

    name = "omni-thinker"

    def __init__(self, model: Any, processor: Any, **generate_kwargs: Any) -> None:
        self._model = model
        self._processor = processor
        # A processor delegates `pad_token_id` in some transformers versions and not
        # in others; the inner tokenizer has it in all of them.
        self._tokenizer = getattr(processor, "tokenizer", processor)
        pad_id = getattr(self._tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self._tokenizer, "eos_token_id", None)
        if pad_id is None:
            raise ValueError(
                "the processor's tokenizer has neither pad_token_id nor eos_token_id, so "
                "there is no id to left-pad a batch with"
            )
        self._pad_id = int(pad_id)
        self._generate_kwargs = dict(generate_kwargs)
        self._generate_kwargs.setdefault("return_audio", False)
        self._generate_kwargs = _accepted_by(model.generate, self._generate_kwargs)

    @torch.no_grad()
    def generate_ids(
        self,
        prompt_ids: Sequence[Sequence[int]],
        config: EvalConfig,
        *,
        progress: Callable[[int, int], None] | None = None,
        extras: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> list[list[int]]:
        if extras is None or len(extras) != len(prompt_ids):
            raise ValueError(
                f"this backend needs one feature mapping per prompt; got "
                f"{0 if extras is None else len(extras)} for {len(prompt_ids)} prompts. "
                f"A prompt without its encoder inputs would be answered from the text "
                f"alone, which scores like a weak model rather than like a missing input."
            )
        missing = [index for index, extra in enumerate(extras) if extra is None]
        if missing:
            raise ValueError(
                f"{len(missing)} of {len(extras)} prompts carry no encoder inputs "
                f"(first at index {missing[0]}). Mixing plain-text and audio prompts in "
                f"one call would batch a row with no spectrogram against rows that have "
                f"one; build them as separate calls."
            )
        # An empty `missing` is the proof that no entry is ``None``. Rebinding states
        # that in the type rather than leaving it in the control flow, so
        # `batch_features` is never handed an optional it has no way to interpret. The
        # filter cannot drop anything here, so the indices `chunk` carries stay valid.
        present: list[Mapping[str, Any]] = [extra for extra in extras if extra is not None]

        model = self._model
        order = sorted(range(len(prompt_ids)), key=lambda i: -len(prompt_ids[i]))
        outputs: list[list[int] | None] = [None] * len(prompt_ids)

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        criteria = _stopping_criteria(self._tokenizer, config)
        greedy = greedy_generation_config(model, config, self._pad_id)

        try:
            for start in range(0, len(order), config.batch_size):
                chunk = order[start : start + config.batch_size]
                width = max(len(prompt_ids[i]) for i in chunk)
                input_ids = torch.tensor(
                    [
                        [self._pad_id] * (width - len(prompt_ids[i])) + list(prompt_ids[i])
                        for i in chunk
                    ],
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.tensor(
                    [[0] * (width - len(prompt_ids[i])) + [1] * len(prompt_ids[i]) for i in chunk],
                    dtype=torch.long,
                    device=device,
                )
                # One expression, one index list: the features are permuted by the same
                # `chunk` that permuted the ids, so the two cannot drift apart.
                features = batch_features([present[i] for i in chunk])
                features = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in features.items()
                }

                generated = _text_ids(
                    model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        generation_config=greedy,
                        stopping_criteria=criteria,
                        **features,
                        **self._generate_kwargs,
                    )
                )
                rows = _continuations(generated, input_ids, config)
                for position, row in zip(chunk, rows.tolist(), strict=True):
                    outputs[position] = [int(i) for i in row]

                if progress is not None:
                    progress(min(start + config.batch_size, len(order)), len(order))
        finally:
            if was_training:
                model.train()

        if any(row is None for row in outputs):
            missing_count = sum(row is None for row in outputs)
            raise RuntimeError(
                f"generation covered {len(outputs) - missing_count}/{len(outputs)} prompts"
            )
        return [row for row in outputs if row is not None]


def _text_ids(generated: Any) -> torch.Tensor:
    """The token ids out of whatever shape ``generate`` chose to return them in.

    A multi-modal ``generate`` may return a tensor, a ``(text_ids, waveform)`` pair,
    or a ``ModelOutput``. Unwrapping by inspection rather than by version check,
    because the alternative is a version table that goes stale silently and whose
    failure mode is indexing a waveform as though it were ids.
    """
    if torch.is_tensor(generated):
        return generated
    sequences = getattr(generated, "sequences", None)
    if torch.is_tensor(sequences):
        return sequences
    if isinstance(generated, (tuple, list)) and generated and torch.is_tensor(generated[0]):
        return generated[0]
    raise RuntimeError(
        f"generate returned {type(generated).__name__}, which carries no token ids this "
        f"function recognises (a tensor, a `.sequences`, or a tuple starting with one)."
    )


def _continuations(
    generated: torch.Tensor, input_ids: torch.Tensor, config: EvalConfig
) -> torch.Tensor:
    """Just the new tokens, having checked which convention was used.

    ``[:, width:]`` is right when ``generate`` returns prompt-plus-continuation and
    catastrophic when it returns the continuation alone -- it would slice past the
    end of every answer and score a working model at zero, with nothing raised. So
    the prompt is looked for rather than assumed: it is the exact tensor that was
    sent, and a false match on a whole padded batch is not a risk worth modelling.
    """
    width = input_ids.shape[1]
    if generated.shape[1] >= width and torch.equal(generated[:, :width], input_ids):
        return generated[:, width:]
    if generated.shape[1] <= config.max_new_tokens:
        _log.info(
            "generate returned %d columns against a %d-token prompt; reading the whole "
            "output as the continuation",
            generated.shape[1],
            width,
        )
        return generated
    raise RuntimeError(
        f"generate returned {generated.shape[1]} columns for a {width}-token prompt and the "
        f"first {width} are not the prompt that was sent. The continuation cannot be located "
        f"by slicing, and guessing an offset would silently score the model on the wrong "
        f"tokens."
    )


def _accepted_by(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop keywords ``fn`` does not declare.

    ``return_audio`` is the reason this exists. It is the switch that stops the
    vocoder running for an evaluation that scores text, and it is declared by the
    composite ``...ForConditionalGeneration`` and not by the Thinker on its own --
    both of which are legitimate things to hand this backend. Passing it to the
    Thinker raises deep inside ``forward`` with a message about an unexpected
    keyword, which is a confusing way to learn that a speed-up did not apply.

    Only *declared* names count, even when the signature also has ``**kwargs``.
    ``generate`` has had ``**kwargs`` throughout and forwards whatever it does not
    recognise into ``forward``, so its presence says nothing about what is consumed
    -- treating it as "accepts anything" would put the keyword back on the path this
    function exists to keep it off.
    """
    import inspect

    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover -- C-implemented callable
        return dict(kwargs)
    return {
        key: value
        for key, value in kwargs.items()
        if key in parameters and parameters[key].kind is not inspect.Parameter.VAR_KEYWORD
    }
