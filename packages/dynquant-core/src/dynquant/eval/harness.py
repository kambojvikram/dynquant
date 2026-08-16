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

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeAlias

import torch

from dynquant._logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "NEUTRAL_DECODE",
    "AudioPrompt",
    "EncodedPrompts",
    "EvalBackend",
    "EvalConfig",
    "Prompt",
    "TransformersBackend",
    "chat_prompt_style",
    "encode_prompts",
    "generate_batched",
    "greedy_generation_config",
    "reasoning_state",
    "render_chat",
    "strip_reasoning",
]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AudioPrompt:
    """Prompt ids whose meaning depends on tensors travelling beside them.

    An audio model's processor does not turn a conversation into ids and then take
    audio separately. It emits both together: the ids contain a run of placeholder
    tokens standing in for the waveform, and the encoder fills those positions from
    ``features``. The two are a single object with two halves, and every operation
    the harness performs on ids -- padding, batching, sorting -- has to keep the
    halves together or the model attends to audio belonging to a different example.

    So the halves are carried in one value rather than in two parallel sequences a
    caller could zip up wrong.

    ``features`` is whatever the processor returned for *this one* example, batch
    axis included (shape ``[1, ...]``), and it is handed to ``generate`` unread.
    Naming the keys here would mean this module knowing which model it is serving,
    which is the one thing :class:`EvalBackend` exists to keep out of the harness.
    """

    ids: tuple[int, ...]
    features: Mapping[str, Any]

    def __len__(self) -> int:
        return len(self.ids)


Prompt: TypeAlias = "str | list[int] | AudioPrompt"
"""What a task may hand the harness: text to tokenize, tokens already, or tokens
that only mean something alongside audio.

Few-shot prompts are text -- they are text in the dataset and text is what the model
was trained on. A *chat* prompt is ids, because the framing is made of control tokens
and :func:`render_chat` is the only thing entitled to emit them. An
:class:`AudioPrompt` is ids too, for the same reason and one more: the processor that
emitted them also emitted the encoder inputs those ids refer to."""


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

    extras: tuple[Mapping[str, Any] | None, ...] = ()
    """Per-prompt encoder inputs, aligned to :attr:`ids`, or empty when no prompt
    carried any.

    Defaulted so every existing construction of this tuple keeps its meaning, and
    empty rather than a tuple of ``None`` so that "nothing in this batch had audio"
    is one falsy check rather than a scan.
    """


def encode_prompts(tokenizer: Any, prompts: Sequence[Prompt], config: EvalConfig) -> EncodedPrompts:
    """Tokenize, count the overlong ones, and cut them at the front.

    Truncation is done by slicing rather than by asking the tokenizer, so it does
    not depend on the caller's ``truncation_side`` -- a tokenizer object shared with
    a training pipeline arrives set to whatever that pipeline wanted, and the
    harness mutating it back and forth is how a run picks up a setting from whoever
    touched it last. ``ids[-limit:]`` is what ``truncation_side="left"`` does, on a
    sequence that already has its special tokens.

    A prompt that arrives as ids is passed through and truncated identically. Only
    :func:`render_chat` produces those, and it produces them precisely so that the
    control tokens framing the turn never have to survive a trip through text.

    An :class:`AudioPrompt` is the one kind that is *refused* rather than cut. Its
    ids contain a run of placeholder positions the encoder fills from the tensors
    beside them, and the count has to match exactly; slicing the front removes some
    of those positions while the tensors stay whole, so the model reads the audio
    offset against the text. That failure raises nothing and has no signature in the
    output -- it scores low and looks like a weak model -- so an overlong audio
    prompt raises here instead, naming the two numbers the caller has to reconcile.
    """
    if not prompts:
        return EncodedPrompts(ids=[], truncated=0)

    rows: list[list[int]] = [[] for _ in prompts]
    text_at = [index for index, prompt in enumerate(prompts) if isinstance(prompt, str)]
    if text_at:
        # One batched call, as before: tokenizing per prompt is measurably slower on
        # the datasets this runs on, and the split here is by prompt *kind*, which is
        # uniform within a task in every current caller.
        encoded = tokenizer(
            [prompts[index] for index in text_at],
            add_special_tokens=config.add_special_tokens,
        )["input_ids"]
        for index, row in zip(text_at, encoded, strict=True):
            rows[index] = [int(token) for token in row]
    extras: list[Mapping[str, Any] | None] = [None] * len(prompts)
    for index, prompt in enumerate(prompts):
        if isinstance(prompt, AudioPrompt):
            rows[index] = [int(token) for token in prompt.ids]
            extras[index] = prompt.features
        elif not isinstance(prompt, str):
            rows[index] = [int(token) for token in prompt]
    limit = config.max_prompt_tokens
    overlong = [
        index
        for index, extra in enumerate(extras)
        if extra is not None and len(rows[index]) > limit
    ]
    if overlong:
        longest = max(len(rows[index]) for index in overlong)
        raise ValueError(
            f"{len(overlong)}/{len(prompts)} audio prompts are longer than "
            f"max_prompt_tokens={limit} (longest {longest}). An audio prompt cannot be cut "
            f"at the front: its ids reserve one position per encoder frame, and dropping "
            f"some of those while the encoder inputs stay whole offsets the audio against "
            f"the text without raising. Raise max_prompt_tokens above {longest}, or shorten "
            f"the prompt -- fewer shots, or shorter clips."
        )
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
    return EncodedPrompts(
        ids=[row[-limit:] for row in rows],
        truncated=truncated,
        extras=tuple(extras) if any(extra is not None for extra in extras) else (),
    )


def chat_prompt_style(tokenizer: Any) -> Literal["chat-template", "raw"]:
    """``"chat-template"`` if this tokenizer can frame a chat turn, else ``"raw"``.

    Decided by asking the tokenizer to do it, not by reading
    ``tokenizer.chat_template``. That attribute is a property of one *implementation* --
    the Jinja-backed one -- and not of the capability. ``MistralCommonBackend``, which
    ``AutoTokenizer`` returns for any Mistral checkpoint shipping a ``tekken.json``,
    leaves it ``None`` while ``apply_chat_template`` works perfectly and renders
    ``<s>[INST]...[/INST]``.

    That gap is not cosmetic and it does not announce itself. Under the attribute check
    an instruct checkpoint reads as a base checkpoint, so it is handed bare text, and an
    instruct model given bare text continues it rather than answering it. On IFEval that
    put Ministral-8B-Instruct at 24.77% with 195 of 541 generations empty, against
    Phi-4-mini's 68.76% with none -- a number low enough to look like a broken model and
    stable enough to look real. The one thing that made it visible was
    ``IfevalResult.prompt_style``, which is why the tasks record it.

    A base checkpoint still gets ``"raw"``, and that is still a genuine difference in
    measurement rather than a fallback to paper over: it has no template because it was
    never taught a turn structure. The probe distinguishes "will not" from "cannot",
    which the attribute could not.

    Broad ``except`` on purpose. This asks "does this work", and a tokenizer with no
    template is entitled to refuse in whatever way it likes -- ``ValueError`` from
    transformers today, and the whole point of probing is to stop encoding today's
    implementation detail as tomorrow's capability test.
    """
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": "ping"}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:  # noqa: BLE001 -- a capability probe; see docstring
        return "raw"
    return "chat-template" if isinstance(rendered, str) and rendered.strip() else "raw"


def render_chat(tokenizer: Any, messages: Sequence[dict[str, str]]) -> Prompt:
    """One chat turn, as the ids the model was trained on.

    Ids, not text, because the rendered text is not always re-tokenizable back into
    what it renders. ``MistralCommonBackend`` renders ``<s>[INST]...[/INST]`` and then
    tokenizes that string as *literal characters* -- ``tekken`` never parses control
    tokens out of user text, which is a deliberate injection guard and not a bug. The
    frame therefore survives rendering and dies on encoding: no BOS, no ``[INST]``,
    seventeen tokens where there should be ten. transformers says so out loud
    (``apply_chat_template(..., tokenize=False)`` "is unsafe ... don't encode the
    output manually"), in a warning that a batch eval buries.

    What that cost is the point. Handed the de-tokenized frame, Ministral-8B-Instruct
    returned *nothing at all* for 120 of 164 HumanEval problems and 84 of 541 IFEval
    prompts -- 23.17% and 37.52%, both stable, neither an error. Correctly framing an
    instruct checkpoint (:func:`chat_prompt_style`) and then discarding the framing on
    the way to the model is a harness bug that reads exactly like a damaged model,
    which is the failure mode this package exists to be able to rule out.

    Round-tripping is lossless for the Jinja-backed tokenizers -- Phi-4-mini gives the
    same ten ids either way -- so this changes no already-collected number for them.
    It removes the round trip rather than repairing it, because "re-tokenizing rendered
    text reproduces the render" is an assumption no tokenizer promises to keep.

    Falls back to text only if the tokenizer will not produce ids, which no real
    transformers tokenizer does; the fallback exists so a caller's stub cannot crash a
    run over a capability it never claimed.
    """
    turn = [dict(message) for message in messages]
    try:
        encoded = tokenizer.apply_chat_template(turn, tokenize=True, add_generation_prompt=True)
    except Exception:  # noqa: BLE001 -- see the fallback in the docstring
        encoded = None
    ids = _as_token_ids(encoded)
    if ids is not None:
        return ids
    return str(tokenizer.apply_chat_template(turn, tokenize=False, add_generation_prompt=True))


def _as_token_ids(encoded: Any) -> list[int] | None:
    """Normalise what ``apply_chat_template(tokenize=True)`` returns, or give up.

    It returns a ``BatchEncoding`` on some versions, a bare list on others, a tensor
    when asked, and either one row or a batch of one. ``None`` means "that was not ids"
    -- including the empty case, so an unusable render falls back rather than sending
    the model a zero-length prompt.
    """
    if encoded is None:
        return None
    if isinstance(encoded, str):
        return None
    if hasattr(encoded, "keys") and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if not isinstance(encoded, (list, tuple)) or not encoded:
        return None
    if isinstance(encoded[0], (list, tuple)):
        if len(encoded) != 1:
            return None
        encoded = encoded[0]
    if not encoded or not all(isinstance(token, int) for token in encoded):
        return None
    return [int(token) for token in encoded]


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
        extras: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> list[list[int]]:
        """Greedy continuations, one per prompt, **in input order**.

        ``extras`` carries per-prompt encoder inputs -- audio features, image
        patches -- aligned to ``prompt_ids``, for the backends that can consume
        them. It is defaulted, and :func:`generate_batched` only passes it when a
        prompt actually carried something, so a backend written before this
        argument existed is never handed it and keeps working unchanged on every
        text task. A backend that cannot consume extras should refuse them by name
        rather than ignore them: silently dropping the audio leaves a model
        answering a question about a clip it was never given, at a score that looks
        like a bad model rather than a missing input.
        """


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
        extras: Sequence[Mapping[str, Any] | None] | None = None,
    ) -> list[list[int]]:
        if extras and any(extra is not None for extra in extras):
            raise NotImplementedError(
                "TransformersBackend takes ids only. These prompts carry encoder inputs "
                "(audio features, say), and batching those is model-specific -- which axis "
                "is ragged, which mask goes with it -- so it is not something this class "
                "can do generically. Use dynquant.eval.omni.OmniThinkerBackend, or a "
                "backend for whatever modality produced them."
            )
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
    prompts: Sequence[Prompt],
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
    # Passed only when there is something to pass. A defaulted keyword is already
    # compatible with any backend defined in this package, but `EvalBackend` is a
    # public base class and the subclasses that matter most are the ones written
    # elsewhere -- so a text task must not become the first thing to hand one an
    # argument its signature never declared.
    continuations = (
        backend.generate_ids(encoded.ids, config, progress=progress, extras=encoded.extras)
        if encoded.extras
        else backend.generate_ids(encoded.ids, config, progress=progress)
    )

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


_REASONING_OPEN = re.compile(r"<think\s*>", re.IGNORECASE)
_REASONING_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Return the answer region of a generation, dropping any reasoning trace.

    A reasoning model answers in two parts and only the second is an answer. Handing an
    extractor the whole thing does not merely add noise: a trace is where the model
    argues *against* candidates, so the first plausible-looking answer in it is
    frequently one it went on to reject. Measured on ``LFM2.5-8B-A1B`` over the
    text-to-SQL mixture, cutting the trace moved 32 items from 6.2% to 40.6% -- the
    extractor had been reading queries out of the deliberation, one of them immediately
    followed by "But note that column name is ...".

    Three cases, and the third is the one that carries the honesty of the metric:

    - **No trace at all** -- returned unchanged. This is every non-reasoning model, so
      wiring this into an extractor cannot move a number that has already been
      collected, which is why all of them use it rather than only the one that needed it.
    - **A closed trace** -- everything after the last close tag. Last, not first, to
      match what the model's own chat template does when it strips a previous turn
      (``content.split("</think>")[-1]``); the convention is the model's, not this
      package's.
    - **An open trace that never closed** -- ``""``. The model spent its whole token
      budget thinking and never answered, so there is no answer region, and the caller
      counts it unparseable. Returning the trace instead would score the model on a
      query it had not finished reasoning about and report a truncated budget as a wrong
      answer -- which reads as quantization damage rather than as a decode setting.
    """
    closed = list(_REASONING_CLOSE.finditer(text))
    if closed:
        return text[closed[-1].end() :]
    return "" if _REASONING_OPEN.search(text) else text


def reasoning_state(text: str) -> Literal["absent", "closed", "unclosed"]:
    """Which of :func:`strip_reasoning`'s three cases this generation is.

    The classification the extractor throws away, and the record needs. An unclosed
    trace reaches the scorer as an empty answer region and is counted unparseable --
    correctly, since the model never answered -- but that makes it indistinguishable
    from a generation that emitted prose instead of SQL. The two have opposite causes:
    one is a decode budget that was too small, the other is a model that has stopped
    complying with the format.

    Which matters because the second is the first visible sign of quantization damage,
    and it is what a quantized arm is being watched for. An arm that reasons more
    verbosely than the baseline -- entirely plausible, since quantization perturbs the
    token distribution -- loses items to the cap and posts them as "stopped answering".
    Tallying this separately is what lets the comparison say which of the two happened
    instead of assuming.

    ``absent`` for every model that does not emit a trace, so the tally is zero
    throughout and costs nothing to carry.
    """
    if _REASONING_CLOSE.search(text):
        return "closed"
    return "unclosed" if _REASONING_OPEN.search(text) else "absent"


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
