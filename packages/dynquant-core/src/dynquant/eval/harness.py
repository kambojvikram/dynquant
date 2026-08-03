"""Batched greedy generation, shared by every task.

Nothing here is task-specific: it takes prompts, returns continuations. The two
decisions that matter for measurement validity live here rather than in the task,
so no task can accidentally make itself incomparable.

**Left padding.** Decoder-only generation continues from the last position, so a
right-padded batch would continue from pad tokens. ``transformers`` warns about
this and then produces plausible-looking garbage for every sequence in the batch
except the longest, which is the worst failure mode there is -- the number comes
out low but not zero, and looks like a real result.

**Left truncation.** A prompt that will not fit is cut at the *front*. Everything a
few-shot prompt needs the model to act on -- the question, the options, the trailing
cue -- sits at the end; the exemplars at the front are the expendable part. Cutting
from the right instead removes the cue, so the model is scored on a question it was
never shown and returns something near chance however well it knows the task. That
reads as quantization damage rather than as a harness bug.

Truncation is also counted and logged, because the dangerous case is not truncation
but *silent* truncation: a run where 8% of prompts lost their question still
produces a plausible number, and nothing in the output says why it is low.

**Length-sorted batching.** Prompts are batched by length, so a batch's padding is
bounded by the spread within it rather than by the longest prompt in the dataset.
Results are restored to input order before returning, so this is invisible to the
caller and cannot change any score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from dynquant._logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["EvalConfig", "generate_batched"]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """How to decode. Identical across every measurement point, by construction."""

    max_new_tokens: int = 320
    batch_size: int = 32
    stop_sequences: tuple[str, ...] = ()
    """Cut the continuation at the first of these. Applied to the decoded text
    rather than as token ids, because a stop string can tokenize differently
    depending on what precedes it."""

    max_prompt_tokens: int = 2048
    limit: int | None = None
    """Score only the first N examples. For smoke runs; a real number needs the
    whole set, and the reported result says which was used."""

    add_special_tokens: bool = True
    """Let the tokenizer prepend its own BOS.

    Must be ``False`` for a prompt built by ``apply_chat_template``, because the
    template already emits the BOS token itself. Leaving it on gives Llama-3 and
    Gemma-3 prompts *two* leading BOS tokens, which no error reports and which costs a
    few points of instruction-following -- exactly the size of effect this package
    exists to measure, arriving from the harness instead of from the weights.

    A few-shot prompt is raw text with no template, so it wants the default."""

    early_stop: bool = True
    """Stop a sequence once it has emitted a stop string, instead of decoding to
    ``max_new_tokens`` and discarding the tail.

    Purely a speed setting: the text is truncated at the stop either way, so this
    cannot change a score. It is a setting rather than unconditional because it is
    only safe if the stopping criterion is evaluated *per sequence* -- a criterion
    that halted the whole batch when one member finished would silently truncate
    the others, which reads as a low-but-plausible score rather than as a crash.
    :func:`generate_batched` verifies the per-sequence shape before using it and
    falls back to full-length decoding otherwise."""


@torch.no_grad()
def generate_batched(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    config: EvalConfig,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> list[str]:
    """Greedy-decode a continuation for each prompt.

    Returns continuations only -- the prompt is stripped by slicing the token
    sequence, not by string matching, so a tokenizer that normalises whitespace
    cannot leave a fragment of the prompt in the answer.
    """
    if not prompts:
        return []

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    original_side = tokenizer.padding_side
    original_truncation = getattr(tokenizer, "truncation_side", "right")
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    truncated = 0

    # Sort long-to-short so the first batch is the memory high-water mark: an OOM
    # then happens immediately rather than 40 minutes into a run.
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    outputs: list[str | None] = [None] * len(prompts)

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    criteria = _stopping_criteria(tokenizer, config)

    try:
        for start in range(0, len(order), config.batch_size):
            chunk = order[start : start + config.batch_size]
            raw = [prompts[i] for i in chunk]
            batch = tokenizer(
                raw,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_prompt_tokens,
                add_special_tokens=config.add_special_tokens,
            ).to(device)
            if batch["input_ids"].shape[1] >= config.max_prompt_tokens:
                # Only pay for the exact count on batches that could have lost
                # something -- the sort puts those first, so this is rare.
                truncated += sum(
                    len(ids) > config.max_prompt_tokens
                    for ids in tokenizer(
                        raw, truncation=False, add_special_tokens=config.add_special_tokens
                    )["input_ids"]
                )

            generated = model.generate(
                **batch,
                max_new_tokens=config.max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=pad_id,
                stopping_criteria=criteria,
            )
            # Every row shares one prompt width because padding is left and the
            # batch is padded to a common length, so a single slice is correct.
            continuations = generated[:, batch["input_ids"].shape[1] :]
            for position, text in zip(
                chunk,
                tokenizer.batch_decode(continuations, skip_special_tokens=True),
                strict=True,
            ):
                outputs[position] = _truncate(text, config.stop_sequences)

            if progress is not None:
                progress(min(start + config.batch_size, len(order)), len(order))
    finally:
        tokenizer.padding_side = original_side
        tokenizer.truncation_side = original_truncation
        if was_training:
            model.train()

    if truncated:
        _log.warning(
            "%d/%d prompts exceeded max_prompt_tokens=%d and were cut at the front. "
            "The trailing cue survives, but the leading exemplars did not -- raise "
            "max_prompt_tokens or use fewer shots if this is a large fraction.",
            truncated,
            len(prompts),
            config.max_prompt_tokens,
        )

    # Every position must have been written: the batches partition `order`, which
    # is a permutation of the input indices. A gap would mean a silently dropped
    # prompt, which would shift every subsequent answer against its gold label.
    if any(text is None for text in outputs):
        missing = sum(text is None for text in outputs)
        raise RuntimeError(f"generation covered {len(prompts) - missing}/{len(prompts)} prompts")
    return [text for text in outputs if text is not None]


def _stopping_criteria(tokenizer: Any, config: EvalConfig) -> Any | None:
    """A per-sequence stop-string criterion, or ``None`` if one cannot be trusted.

    ``StopStringCriteria`` matches a stop string only as a *suffix* of the sequence
    so far, which is what makes it usable here: the few-shot prompt contains the
    stop string between exemplars, and a criterion that matched anywhere would
    declare every sequence finished before it produced a single token.

    The returned tensor is checked to be one entry per sequence. A criterion that
    collapsed the batch to a single verdict would end everyone's generation when
    the first sequence finished, truncating the rest mid-answer -- a failure that
    produces a plausible number rather than an error, so it is checked rather than
    assumed.
    """
    if not (config.early_stop and config.stop_sequences):
        return None
    try:
        from transformers import StoppingCriteriaList
        from transformers.generation import (  # type: ignore[attr-defined]
            StopStringCriteria,
        )
    except ImportError:
        return None

    try:
        criterion = StopStringCriteria(
            tokenizer=tokenizer, stop_strings=list(config.stop_sequences)
        )
        probe = torch.zeros((3, 4), dtype=torch.long)
        verdict = criterion(probe, None)
    # BLE001: deliberately blind. This probe exists to decide whether an *optional*
    # speed-up is trustworthy, and every possible failure -- a renamed argument, a
    # tokenizer without the vocabulary the criterion wants, an outright bug -- has
    # the same correct answer: decode to max_new_tokens instead. Narrowing this
    # would let some transformers version turn a slower evaluation into no
    # evaluation at all.
    except Exception as exc:  # noqa: BLE001  # pragma: no cover -- version-dependent
        _log.warning("stop-string criterion unavailable (%s); decoding to max_new_tokens", exc)
        return None

    if not (hasattr(verdict, "shape") and tuple(verdict.shape) == (3,)):
        _log.warning(
            "stop-string criterion is not per-sequence (verdict %r); decoding to max_new_tokens",
            getattr(verdict, "shape", verdict),
        )
        return None
    return StoppingCriteriaList([criterion])  # type: ignore[no-untyped-call]


def _truncate(text: str, stops: Sequence[str]) -> str:
    """Cut at the earliest stop sequence.

    Earliest, not first-listed: the model may emit several, and stopping at
    whichever appears first in the *text* is the only order-independent rule.
    """
    cut = len(text)
    for stop in stops:
        index = text.find(stop)
        if index != -1:
            cut = min(cut, index)
    return text[:cut]
