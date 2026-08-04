"""Batched greedy generation, shared by every task.

Nothing here is task-specific: it takes prompts, returns continuations. The three
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

**One encoder and one decoder, whatever produces the tokens.** A task hands
:func:`generate_batched` either a ``transformers`` model or an :class:`EvalBackend`
-- a vLLM engine, say -- and the signature is the same either way, so no task can
be scorable through one path and not the other. Prompts are turned into ids *here*,
the backend is handed those exact ids and returns continuation ids, and the decode
and stop-string truncation happen *here* too. The backend's only responsibility is
the tokens in between.

That split is the whole point. Evaluating the same checkpoint through two runtimes
is only a runtime comparison if everything except the runtime is shared, and the
ways two engines can quietly disagree about a *prompt* -- one prepending its own
BOS, one truncating from the other end, one detokenizing with special tokens left
in -- all produce a score a few points off rather than an error. None of them is
expressible here, because there is only one implementation of each.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from dynquant._logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "NEUTRAL_DECODE",
    "EncodedPrompts",
    "EvalBackend",
    "EvalConfig",
    "TransformersBackend",
    "encode_prompts",
    "generate_batched",
    "greedy_generation_config",
]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EvalConfig:
    """How to decode. Identical across every measurement point, by construction."""

    max_new_tokens: int = 320

    batch_size: int = 32
    """How many prompts to pad into one tensor. A ``transformers``-path knob only --
    a serving engine schedules its own concurrency and
    :class:`~dynquant.eval.backends.VllmBackend` says so and ignores this."""

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
    :class:`TransformersBackend` verifies the per-sequence shape before using it and
    falls back to full-length decoding otherwise."""


class EncodedPrompts(NamedTuple):
    """Prompt ids as every backend must see them, and what encoding cost."""

    ids: list[list[int]]
    truncated: int
    """How many prompts lost tokens off the front."""


def encode_prompts(tokenizer: Any, prompts: Sequence[str], config: EvalConfig) -> EncodedPrompts:
    """Tokenize, count the overlong ones, and cut them at the front.

    Truncation is done by slicing rather than by asking the tokenizer, so it does
    not depend on the caller's ``truncation_side`` -- a tokenizer object shared with
    a training pipeline arrives set to whatever that pipeline wanted, and the
    harness mutating it back and forth is how a run picks up a setting from whoever
    touched it last. ``ids[-limit:]`` is what ``truncation_side="left"`` does, on a
    sequence that already has its special tokens.
    """
    if not prompts:
        return EncodedPrompts(ids=[], truncated=0)

    rows = tokenizer(list(prompts), add_special_tokens=config.add_special_tokens)["input_ids"]
    limit = config.max_prompt_tokens
    truncated = sum(len(row) > limit for row in rows)
    if truncated:
        _log.warning(
            "%d/%d prompts exceeded max_prompt_tokens=%d and were cut at the front. "
            "The trailing cue survives, but the leading exemplars did not -- raise "
            "max_prompt_tokens or use fewer shots if this is a large fraction.",
            truncated,
            len(prompts),
            limit,
        )
    return EncodedPrompts(ids=[[int(i) for i in row[-limit:]] for row in rows], truncated=truncated)


class EvalBackend(ABC):
    """Something that turns prompt ids into continuation ids.

    Deliberately narrow. A backend does not tokenize, does not detokenize, does not
    apply stop strings and does not decide what "greedy" means beyond asking its
    engine for it -- :func:`generate_batched` owns all of that, identically for
    every backend. What is left is the only thing a runtime comparison is entitled
    to vary.
    """

    name: str = "backend"
    """Recorded in the run fingerprint, so a results table says which runtime
    produced each row."""

    @abstractmethod
    def generate_ids(
        self,
        prompt_ids: Sequence[Sequence[int]],
        config: EvalConfig,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[list[int]]:
        """Greedy continuations, one per prompt, **in input order**."""


#: Decode settings that are neutral by definition, checked rather than assumed.
#:
#: Each key is a field whose non-neutral value would change a greedy score -- by
#: choosing a different search (``penalty_alpha`` turns greedy into contrastive
#: search), by rewriting the logits before the argmax, or by moving where the
#: sequence stops -- paired with the value that means "not applied".
#:
#: This *is* the mechanism, and it did not start out that way. The list was written as
#: a tripwire on the theory that building a fresh ``GenerationConfig`` already made
#: every unnamed field the library's neutral default. That theory held on transformers
#: 4.x and stopped holding in the 5.x line, which fills a passed config's unset fields
#: from ``model.generation_config`` -- so :func:`greedy_generation_config` now pins
#: every key here explicitly and the neutrality is a property of this dict rather than
#: of the library's defaults.
#:
#: It also remains a tripwire: a release changing one of these defaults turns a test red
#: instead of quietly re-decoding the campaign, and the vLLM arm's
#: :meth:`~dynquant.eval.backends.VllmBackend._sampling_params` can be read against it.
NEUTRAL_DECODE: dict[str, Any] = {
    "do_sample": False,
    "num_beams": 1,
    "penalty_alpha": None,
    "num_return_sequences": 1,
    "repetition_penalty": 1.0,
    "encoder_repetition_penalty": 1.0,
    "no_repeat_ngram_size": 0,
    "length_penalty": 1.0,
    "guidance_scale": None,
    "sequence_bias": None,
    "min_length": 0,
    "min_new_tokens": None,
    "stop_strings": None,
    "bad_words_ids": None,
    "forced_bos_token_id": None,
    "forced_eos_token_id": None,
    "suppress_tokens": None,
    "begin_suppress_tokens": None,
    "exponential_decay_length_penalty": None,
}


def greedy_generation_config(model: Any, config: EvalConfig, pad_id: int) -> Any:
    """Decode settings pinned field by field, never inherited from the checkpoint.

    ``model.generate`` merges ``model.generation_config`` *under* the kwargs it is
    given, so every field the call site does not name is whatever the checkpoint author
    chose for chat. Qwen2.5-1.5B-Instruct ships ``repetition_penalty: 1.1``. On a
    five-shot GSM8K prompt -- repetitive by construction, since every exemplar repeats
    the same scaffolding and the arithmetic reuses digits -- that is worth 19 points
    against the same weights served through vLLM, whose sampling parameters apply no
    penalty. Neither arm raised anything and both numbers looked like results.

    An earlier version of this function passed an explicit ``GenerationConfig`` and
    stopped there, on the documented understanding that doing so *replaces* the
    checkpoint's config rather than layering over it, making every unnamed field the
    library's neutral default. On transformers 4.x that is what happens, and the fix
    worked. In the 5.x line a passed config's unset fields are filled from
    ``model.generation_config`` instead, so the penalty came back and the arm scored as
    though nothing had been done -- which is exactly what was measured: 42/100 on a
    block where 4.56.2 and vLLM both scored 61 and 60, recovered to 61/100 by pinning
    this one field, per-problem identical to pinning all of :data:`NEUTRAL_DECODE`.

    So every neutral field is named explicitly. The objection to that -- "the next
    checkpoint will ship a different field" -- is real and is why :data:`NEUTRAL_DECODE`
    enumerates the whole class rather than the one field that bit us. What it cannot
    cover is a field transformers adds later, and no formulation available here can:
    ``GenerationConfig`` has no "give me nothing but greedy" constructor. The guard for
    that is the gate, which scores the same checkpoint through two independent
    implementations and fails when they part company.

    The token ids are the deliberate exception. They are facts about the tokenizer, not
    decode settings, and reading them off the checkpoint is exactly right -- dropping
    ``eos_token_id`` would make every sequence run to ``max_new_tokens``. That case is
    warned about rather than repaired, because there is nowhere better to get the id
    from: the previous code passed no ``eos_token_id`` at all and let ``generate``
    merge this same attribute, so inventing a fallback here would be a silent change
    to which token ends a sequence, on top of a run that is merely slow.
    """
    from transformers import GenerationConfig

    source = getattr(model, "generation_config", None)
    eos_token_id = getattr(source, "eos_token_id", None)
    if eos_token_id is None:
        _log.warning(
            "the checkpoint's generation_config names no eos token, so every sequence "
            "will decode to max_new_tokens=%d. Scores are unaffected -- the text is cut "
            "at the stop sequences either way -- but the run will take several times "
            "longer than it needs to.",
            config.max_new_tokens,
        )
    return GenerationConfig(
        max_new_tokens=config.max_new_tokens,
        pad_token_id=pad_id,
        eos_token_id=eos_token_id,
        bos_token_id=getattr(source, "bos_token_id", None),
        **NEUTRAL_DECODE,
    )


class TransformersBackend(EvalBackend):
    """``model.generate`` on a padded batch. The reference path."""

    name = "transformers"

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer
        pad_id = tokenizer.pad_token_id
        self._pad_id = int(tokenizer.eos_token_id if pad_id is None else pad_id)

    @torch.no_grad()
    def generate_ids(
        self,
        prompt_ids: Sequence[Sequence[int]],
        config: EvalConfig,
        *,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[list[int]]:
        model = self._model
        # Sort long-to-short so the first batch is the memory high-water mark: an OOM
        # then happens immediately rather than 40 minutes into a run.
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
                # Padded on the left, so every row's last position is a real token
                # and one slice at `width` extracts every continuation.
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
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=greedy,
                    stopping_criteria=criteria,
                )
                for position, row in zip(chunk, generated[:, width:].tolist(), strict=True):
                    outputs[position] = [int(i) for i in row]

                if progress is not None:
                    progress(min(start + config.batch_size, len(order)), len(order))
        finally:
            if was_training:
                model.train()

        return _require_complete(outputs)


def generate_batched(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    config: EvalConfig,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> list[str]:
    """Greedy-decode a continuation for each prompt.

    ``model`` is either a ``transformers`` model or an :class:`EvalBackend`; the
    task calling this cannot tell the difference, which is what makes every task
    scorable through every runtime.

    Returns continuations only -- the prompt is stripped by slicing the token
    sequence, not by string matching, so a tokenizer that normalises whitespace
    cannot leave a fragment of the prompt in the answer.
    """
    if not prompts:
        return []

    encoded = encode_prompts(tokenizer, prompts, config)
    backend = model if isinstance(model, EvalBackend) else TransformersBackend(model, tokenizer)
    continuations = backend.generate_ids(encoded.ids, config, progress=progress)

    if len(continuations) != len(prompts):
        raise RuntimeError(
            f"{backend.name} returned {len(continuations)} continuations for {len(prompts)} prompts"
        )
    return [
        _truncate(text, config.stop_sequences)
        for text in tokenizer.batch_decode(continuations, skip_special_tokens=True)
    ]


def _require_complete(outputs: Sequence[list[int] | None]) -> list[list[int]]:
    """Every position must have been written.

    The batches partition a permutation of the input indices, so a gap would mean a
    silently dropped prompt -- which shifts every subsequent answer against its gold
    label and lands the score near chance for a reason that looks nothing like a
    batching bug.
    """
    if any(row is None for row in outputs):
        missing = sum(row is None for row in outputs)
        raise RuntimeError(f"generation covered {len(outputs) - missing}/{len(outputs)} prompts")
    return [row for row in outputs if row is not None]


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
        from transformers.generation import StopStringCriteria
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
    return StoppingCriteriaList([criterion])


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
