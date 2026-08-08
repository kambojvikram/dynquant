#!/usr/bin/env python3
"""S2: the phase-3 fine-tunes, and the signal map they exist to produce.

One (model, dataset) pair per invocation. The fine-tune is not the deliverable -- the
deliverable is ``stats/dynquant_stats.json``, the plasticity/saliency map harvested by
:class:`~dynquant.DynQuantCallback` while the weights move. S3 allocates bits from it and
S4 scores what that cost. The fine-tuned checkpoint matters only because a quantization
regression has to be measured against a model that actually learned something: S1 showed
all eight arms have headroom, and S2 is what turns that headroom into a gain to lose.

Why this is not ``stage2_finetune.py``
--------------------------------------
That script trains a *classification* task -- CaseHOLD, Banking77 -- where an example is
one prompt and one completion and the whole of the loss masking is "everything before the
last token is prompt". Phase 3 trains a chat mixture: multi-turn conversations where the
supervised spans are interleaved with the unsupervised ones, and where the turn structure
is made of control tokens rather than text.

Masking, and the assumption underneath it
-----------------------------------------
Only assistant content is supervised. Training on the user turns would spend capacity
modelling the prompt distribution and would put user text into the very gradient signal
that is supposed to be measuring which weights matter for *answering*.

Finding those spans means knowing where each assistant turn starts in the tokenized
sequence, and the only honest way to get that is to ask the tokenizer to render the
conversation prefix and take its length. That works if and only if rendering is
**prefix-stable**: the ids for ``messages[:i+1]`` must begin with the ids for
``messages[:i]``. No tokenizer promises that. A template that emits a trailing whitespace
before the assistant content, or a BPE merge that spans the turn boundary, breaks it by
one or two tokens -- and one token of drift means the loss is computed against the
sequence shifted off its labels, which trains a *worse* model without failing.

So it is checked, per conversation, per turn (:func:`mask_conversation`), and a
conversation that fails is dropped and counted rather than mis-masked. If the drop rate
exceeds ``--max-drop-rate`` the run refuses to start: a fine-tune quietly trained on 40%
of its data looks exactly like a fine-tune that trained on all of it and learned less.

One tokenizer on the panel cannot be walked this way at all. ``mistral_common``, behind
Ministral-8B-Instruct, refuses to render *any* conversation ending in an assistant message
-- it is validating a serving request, and every prefix the walk asks for is exactly the
shape it rejects. So there is a second mode, ``assemble``, which renders each assistant
turn *open* with ``continue_final_message=True`` -- the training shape, and what the
refusal's own error message points at -- and closes it with a terminator measured once
from three synthetic renders. Which mode a tokenizer needs is not an attribute anywhere;
``--mask-mode auto`` measures both on a sample of real rows and takes the winner.

Everything goes through ``apply_chat_template(tokenize=True)``. Rendering to text and
re-tokenizing is what cost S1 two re-runs and up to 56 points of HumanEval -- for
``MistralCommonBackend`` the rendered frame is the *characters* ``<s>[INST]...[/INST]``
and ``tekken`` will not read control tokens back out of user text. See
``dynquant.eval.harness.render_chat``. The same discipline applies on the training side
for the same reason, and here the failure is quieter still: a model trained on a frame it
will never be evaluated under.

Usage::

    # cheap: build the data, report the masking statistics, touch no GPU
    python scripts/run_s2_finetune.py --model phi4-mini --dataset tulu3 --dry-run

    python scripts/run_s2_finetune.py --model phi4-mini --dataset tulu3 --out runs/s2
    python scripts/run_s2_finetune.py --model ministral-8b --dataset tulu3 --out runs/s2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_s1_headroom import MODELS

#: The chat mixtures phase 3 trains on. ``column`` is where the conversation lives;
#: its *shape* is not recorded here because two of these ship ShareGPT-style
#: ``{"from", "value"}`` turns and one ships ``{"role", "content"}``, and a registry that
#: asserts which would be wrong the first time a dataset is re-uploaded.
#: :func:`to_messages` reads the shape off the row instead.
#:
#: ``name`` is the Hub *config* and is only present where the repo has no default one.
#: SmolTalk ships fourteen configs and no default, so ``load_dataset(repo, split=...)``
#: raises ``ValueError: Config name is missing`` -- after the model is on the GPU, since
#: the driver loads the model first. ``all`` is the full 1.1M-row mixture, which is what
#: "SmolTalk" names; the other thirteen are its constituent subsets and picking one would
#: quietly turn a mixture arm into a single-source arm.
DATASETS: dict[str, dict[str, str]] = {
    "tulu3": {"repo": "allenai/tulu-3-sft-mixture", "split": "train", "column": "messages"},
    "smoltalk": {
        "repo": "HuggingFaceTB/smoltalk",
        "name": "all",
        "split": "train",
        "column": "messages",
    },
    "openthoughts3": {
        "repo": "open-thoughts/OpenThoughts3-1.2M",
        "split": "train",
        "column": "conversations",
    },
    # Not a Hub conversation dataset: three text-to-SQL corpora, mixed and turned into
    # single-turn conversations by `load_text2sql`. `builder` marks that, and `repo` is
    # the three repos joined so the run manifest names what was actually trained on
    # rather than one of them.
    "text2sql": {
        "repo": "gretelai/synthetic_text_to_sql+Salesforce/wikisql+b-mc2/sql-create-context",
        "split": "train",
        "column": "messages",
        "builder": "text2sql",
    },
}

#: Full fine-tuning moves every weight and needs the smaller step; LoRA moves a low-rank
#: residual through a fixed scaling and needs roughly an order of magnitude more. Carried
#: over unchanged from ``experiments/four_point/stage2_finetune.py`` -- the regimes are
#: the same two, and two files disagreeing about the learning rate would show up as a
#: model-dependent effect that is nothing of the kind.
LR_BY_REGIME = {"full fine-tune": 1e-5, "lora": 1e-4}

#: ShareGPT speaker names, mapped onto the roles a chat template understands.
_SHAREGPT_ROLES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "system": "system",
}

#: Sources whose presence in a training mixture makes an eval arm's absolute number
#: uninterpretable. Not dropped by default -- see :func:`report_sources`.
_CONTAMINATING = ("gsm8k", "humaneval", "mbpp")


# --------------------------------------------------------------------------
# Turning a dataset row into a conversation
# --------------------------------------------------------------------------


def to_messages(row: Any, column: str) -> list[dict[str, str]] | None:
    """Normalise one row into ``[{"role", "content"}, ...]``, or ``None`` if it is not one.

    Two shapes are in the phase-3 mixtures: ``{"role", "content"}`` (Tulu-3, SmolTalk) and
    ShareGPT's ``{"from", "value"}`` (OpenThoughts3). Read off the row rather than declared
    per dataset, because a re-upload that changes the shape would otherwise produce a
    column of ``None`` and an empty training set rather than an error anyone could act on.
    """
    turns = row.get(column) if isinstance(row, dict) else None
    if not isinstance(turns, (list, tuple)) or not turns:
        return None

    messages: list[dict[str, str]] = []
    for turn in turns:
        if not isinstance(turn, dict):
            return None
        if "role" in turn and "content" in turn:
            role, content = turn["role"], turn["content"]
        elif "from" in turn and "value" in turn:
            role, content = _SHAREGPT_ROLES.get(str(turn["from"]).lower()), turn["value"]
        else:
            return None
        if role is None or not isinstance(content, str):
            return None
        messages.append({"role": str(role), "content": content})
    return messages


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------


#: How far the render of a *finished* conversation may disagree with the render of the same
#: turn mid-conversation before the disagreement is treated as drift rather than as a
#: terminator. Two tokens: Phi-4-mini differs by one (``<|endoftext|>`` after ``<|end|>``),
#: and a template that closes with an EOT plus an EOS would differ by two.
TERMINATOR_TOKENS = 2

#: ``template`` renders the conversation and locates the assistant spans inside it by
#: rendering prefixes. ``assemble`` never renders a conversation ending in an assistant turn
#: at all -- it appends a synthetic user turn and takes each assistant span as the
#: difference between two renders that a serving validator accepts. See
#: :func:`mask_conversation`.
MASK_MODES = ("template", "assemble")


class MaskError(Exception):
    """The tokenizer's rendering could not be aligned to turn boundaries.

    Carries the reason so the driver can report *which* assumption failed across the
    dataset rather than a single count of drops -- "the template is not prefix-stable" and
    "this conversation is too long" call for opposite responses.
    """


def _render(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    generation_prompt: bool,
    continue_final: bool = False,
) -> list[int]:
    """One prefix of a conversation, as the ids the model is trained on.

    ``tokenize=True`` throughout. The text path is not merely slower here, it is wrong for
    any tokenizer whose frame is control tokens it will not parse back out of text.

    ``continue_final`` renders the last message as an *open* turn -- header and content, no
    terminator, no following header. It is what :func:`_assemble_conversation` is built on.
    """
    try:
        encoded = tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=True,
            add_generation_prompt=generation_prompt,
            **({"continue_final_message": True} if continue_final else {}),
        )
    except MaskError:
        raise
    except Exception as exc:
        raise MaskError(f"apply_chat_template raised {type(exc).__name__}") from exc
    ids = _as_token_ids(encoded)
    if ids is None:
        raise MaskError("apply_chat_template(tokenize=True) did not return token ids")
    return ids


def _shared_prefix(left: list[int], right: list[int]) -> int:
    """How many leading tokens the two sequences agree on."""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _shared_suffix(left: list[int], right: list[int]) -> int:
    """How many trailing tokens the two sequences agree on."""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[-1 - index] == right[-1 - index]:
        index += 1
    return index


def _as_token_ids(encoded: Any) -> list[int] | None:
    """Normalise what ``apply_chat_template(tokenize=True)`` returns, or give up.

    A ``BatchEncoding`` on some versions, a bare list on others, one row or a batch of one.
    Deliberately the same normalisation as ``dynquant.eval.harness._as_token_ids``: the
    training and evaluation sides have to agree about what a rendered turn *is*, and the
    cheapest way to guarantee that is for both to accept exactly the same shapes.
    """
    if encoded is None or isinstance(encoded, str):
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


def mask_conversation(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    max_len: int,
    mode: str = "template",
) -> dict[str, list[int]]:
    """Tokenize a conversation and supervise only its assistant content.

    Returns ``{"input_ids", "labels"}`` where ``labels`` is ``input_ids`` on assistant
    spans and ``-100`` everywhere else. Raises :class:`MaskError` rather than returning
    a best guess -- every failure mode here produces a sequence that trains without
    complaining and trains the wrong thing.

    ``mode="template"`` takes the entire sequence from the chat template and locates the
    assistant spans inside it by rendering prefixes (:func:`_walk_template`).
    ``mode="assemble"`` never asks for a conversation ending in an assistant turn at all,
    taking each span as the difference between two renders that a serving validator will
    accept (:func:`_assemble_conversation`) -- which is the only way to mask anything for
    ``mistral_common``. Neither mode is right for every tokenizer and neither can be chosen
    from an attribute, so the driver measures both against the tokenizer it was actually
    given -- see :func:`probe_mask_modes`.
    """
    if not messages:
        raise MaskError("empty conversation")
    if not any(message["role"] == "assistant" for message in messages):
        raise MaskError("no assistant turn to supervise")

    if mode == "template":
        input_ids, spans = _walk_template(tokenizer, messages, max_len=max_len)
    elif mode == "assemble":
        input_ids, spans = _assemble_conversation(tokenizer, messages, max_len=max_len)
    else:
        raise MaskError(f"unknown mask mode {mode!r}")

    labels = [-100] * len(input_ids)
    for start, end in spans:
        labels[start:end] = input_ids[start:end]

    if all(label == -100 for label in labels):
        # A batch of these gives a mean over zero elements. The loss is NaN, the optimizer
        # takes a step into it, and the signal map is harvested from a model made of NaN.
        raise MaskError("no supervised tokens")
    return {"input_ids": input_ids, "labels": labels}


def _walk_template(
    tokenizer: Any, messages: list[dict[str, str]], *, max_len: int
) -> tuple[list[int], list[tuple[int, int]]]:
    """Render the conversation once, then locate each assistant span inside that rendering.

    The span for the assistant turn at index ``i`` begins at the length of ``messages[:i]``
    rendered *with* a generation prompt, so the ``<|assistant|>`` header stays unsupervised
    -- the harness emits it, and the model is never asked to produce it. That length is
    checked against the sequence token for token: an off-by-one here shifts every label in
    the conversation and nothing downstream would say so.

    The span *ends* where ``messages[:i+1]`` stops agreeing with the full sequence, not at
    its length, and the two are allowed to differ by :data:`TERMINATOR_TOKENS`. A template
    asked to render a conversation that ends in an assistant turn renders it as a *finished*
    conversation: Phi-4-mini closes with ``<|end|><|endoftext|>`` where the same turn
    mid-conversation is followed by ``<|end|><|user|>``. That is one trailing token of
    disagreement, in 2% of Tulu-3, and it is not drift -- the two renders agree everywhere
    the model will ever see them. Taking the agreeing prefix supervises the turn terminator
    (which does have to be produced, or generation never stops) without supervising a
    sequence terminator that only belongs after the last turn.

    The allowance is bounded rather than unlimited because the two failures look alike from
    one end and nothing alike from the other: a conversation terminator disagrees in the
    last token or two, while a merge spanning the turn boundary disagrees from the boundary
    onwards. Anything past the bound is rejected as the drift it is.
    """
    input_ids = _render(tokenizer, messages, generation_prompt=False)
    if len(input_ids) > max_len:
        # Dropped, never truncated. Cutting the front removes BOS and the system frame;
        # cutting the back removes the assistant content that is the entire supervision.
        # Either produces a sequence that trains on something the model will never see.
        raise MaskError(f"too long: {len(input_ids)} > {max_len}")

    spans: list[tuple[int, int]] = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt = _render(tokenizer, messages[:index], generation_prompt=True)
        through = _render(tokenizer, messages[: index + 1], generation_prompt=False)

        start = len(prompt)
        if start == 0 or input_ids[:start] != prompt:
            # The template is not prefix-stable for this conversation. Every label from
            # here on would be shifted; there is no partial answer worth keeping.
            raise MaskError(f"turn {index}: rendering is not prefix-stable")
        end = _shared_prefix(through, input_ids)
        if len(through) - end > TERMINATOR_TOKENS:
            raise MaskError(f"turn {index}: rendering is not prefix-stable")
        if not start < end <= len(input_ids):
            raise MaskError(f"turn {index}: span {start}:{end} outside {len(input_ids)} tokens")
        spans.append((start, end))
    return input_ids, spans


#: A three-turn conversation whose contents do not matter and whose *shape* does: user,
#: assistant, user, so that no two adjacent turns share a role. ``mistral_common`` merges
#: adjacent same-role messages into one turn, which is what sank an earlier design that
#: appended a probe user turn to a prefix already ending in one.
_PROBE = (
    {"role": "user", "content": "a"},
    {"role": "assistant", "content": "b"},
    {"role": "user", "content": "c"},
)

#: ``id(tokenizer) -> (tokenizer, terminator)``. The tokenizer is held so the id cannot be
#: recycled onto a different object while the entry lives.
_TERMINATORS: dict[int, tuple[Any, list[int]]] = {}


def _assistant_terminator(tokenizer: Any) -> list[int]:
    """The tokens that close an assistant turn, measured from three legal renders.

    No render, on any tokenizer here, ever ends *between* an assistant terminator and the
    next turn's header -- the two are adjacent constants, so no arithmetic over render
    lengths can separate them. This measures the terminator by content instead:

    ``open``
        ``[user, assistant]`` continued -- header and content, and nothing after them.
    ``closed``
        ``[user, assistant, user]`` with a generation prompt -- the same, then the
        terminator, then a whole user turn.
    ``lone``
        ``[user]`` with a generation prompt -- that same user turn on its own, after
        whatever the template opens a conversation with.

    ``closed`` and ``lone`` end with the identical user frame, so the suffix they share
    reaches back exactly to the terminator and stops: on the other side of it ``lone`` has
    the conversation opener (``<s>``) and ``closed`` has the terminator itself (``</s>``).
    What is left between ``open`` and that point is the terminator, taken from the
    template's own output. It comes out as ``<|end|>`` (200020) for Phi-4-mini -- whose
    ``eos_token_id`` is a *different* token, ``<|endoftext|>`` (199999), which is why this
    is measured rather than read off the tokenizer -- and ``</s>`` (2) for Ministral.

    Every subsequent turn re-checks it: the sequence assembled with this terminator has to
    be a prefix of the next turn's prompt, which the template renders itself. A wrong
    terminator therefore costs drops, not silently mistrained spans.
    """
    cached = _TERMINATORS.get(id(tokenizer))
    if cached is not None and cached[0] is tokenizer:
        return cached[1]

    messages = [dict(message) for message in _PROBE]
    open_turn = _render(tokenizer, messages[:2], generation_prompt=False, continue_final=True)
    closed = _render(tokenizer, messages, generation_prompt=True)
    lone = _render(tokenizer, messages[2:], generation_prompt=True)
    end = len(closed) - _shared_suffix(closed, lone)
    if closed[: len(open_turn)] != open_turn or not len(open_turn) < end <= len(closed):
        raise MaskError("cannot measure the assistant turn terminator")

    terminator = closed[len(open_turn) : end]
    _TERMINATORS[id(tokenizer)] = (tokenizer, terminator)
    return terminator


def _assemble_conversation(
    tokenizer: Any, messages: list[dict[str, str]], *, max_len: int
) -> tuple[list[int], list[tuple[int, int]]]:
    """Build the sequence out of renders the template will actually agree to produce.

    ``mistral_common`` -- the tokenizer backend behind Ministral-8B-Instruct -- refuses to
    render *any* conversation ending in an assistant message: with
    ``add_generation_prompt=False`` it raises ``InvalidMessageStructureException`` ("Expected
    last role User or Tool ... for serving"), and with ``True`` it raises a ``ValueError``
    naming ``continue_final_message``. It is validating a serving request, and every prefix
    the per-turn walk needs is exactly the shape it rejects, so ``template`` mode drops
    100% of an SFT mixture on it.

    What it does accept is ``continue_final_message=True``, which is what its error message
    asks for and what the assistant-final shape means during training rather than serving:
    render the last turn *open*, header and content, no terminator. So each assistant turn
    at index ``i`` needs only two renders, both legal on both backends:

    ``prompt``
        ``messages[:i]`` with a generation prompt -- literally the serving request the model
        would receive before writing this turn, so the supervised span starts at its end and
        the assistant header stays out of it.
    ``through``
        ``messages[:i+1]`` continued -- the same thing plus this turn's content.

    :func:`_assistant_terminator` supplies the one piece neither render contains, and the
    sequence grows as ``through + terminator``.

    The obvious-looking alternative is ``return_assistant_tokens_mask=True``, which is worse
    than useless here: it needs ``{% generation %}`` markers in the template, and Phi-4-mini
    has none, so it returns a full-length mask of zeros and reports success. A driver that
    trusted it would train on no supervised tokens at all and never say so.

    Guessing was the first design and it does not survive contact with the data. Four
    hypotheses died against these two tokenizers, each within one dry run:

    * that the terminator is ``eos_token_id`` -- Phi-4-mini's turns end in ``<|end|>``
      (200020), its ``eos_token_id`` is ``<|endoftext|>`` (199999);
    * that message content can be tokenized on its own and matched -- ``mistral_common``
      strips trailing whitespace before framing, so ``"...Yosemite National Park "``
      assembles with a space token the template never emits (28 of 3000 Tulu-3
      conversations, plus 2 whose content is the empty string);
    * that ``generation_prompt=False`` renders a prefix of what ``True`` renders --
      Phi-4-mini *closes* a conversation it is not being asked to continue, ending
      ``<|end|><|endoftext|>`` where the training sequence needs the next header;
    * that a synthetic user turn can be appended to any prefix to make it renderable --
      ``mistral_common`` merges adjacent same-role messages, so appending one to a prefix
      that already ends in a user turn produces a single merged turn and no frame at all.
      That one survived a 3000-row dry run at a 95% success rate, because the merge
      separator and the turn opener are each exactly one token and the two errors cancelled.

    What is checked, per turn, is alignment: ``prompt`` must begin with the sequence built so
    far -- which is what re-verifies the terminator against the template's own output -- and
    ``through`` must begin with ``prompt``. Either failing means a merge crossed a turn
    boundary, and the conversation is dropped rather than supervised at an offset.

    The sequence ends at the last assistant turn. Trailing user or tool turns are dropped
    rather than rendered, which is what they are worth to a fine-tune: there is nothing to
    predict after them.

    Phi-4-mini accepts both modes, which makes it the cross-check: on 500 Tulu-3 rows, all
    495 that both modes mask come back token-identical, with identical span starts and
    identical ends on every span but the last. The single difference is that ``template``
    mode's sequence carries a trailing ``<|endoftext|>`` -- the document terminator Phi
    appends to a conversation it is not asked to continue -- inside its final span. That is
    a difference of scope, not of alignment, and it is left alone: it is what each template
    says a *finished* conversation is, Ministral has no equivalent token, and forcing either
    into the other's shape trains a model on a frame it is not served under. It is also why
    :func:`choose_mask_mode` breaks a tie toward ``template`` -- the mode that asks the
    template itself.
    """
    terminator = _assistant_terminator(tokenizer)
    sequence: list[int] = []
    spans: list[tuple[int, int]] = []

    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt = _render(tokenizer, messages[:index], generation_prompt=True)
        through = _render(
            tokenizer, messages[: index + 1], generation_prompt=False, continue_final=True
        )
        if prompt[: len(sequence)] != sequence or through[: len(prompt)] != prompt:
            raise MaskError(f"turn {index}: assembly disagrees with the template")
        start, end = len(prompt), len(through) + len(terminator)
        if start == 0 or not start < end:
            raise MaskError(f"turn {index}: span {start}:{end} supervises nothing")
        sequence = [*through, *terminator]
        spans.append((start, end))

    if len(sequence) > max_len:
        raise MaskError(f"too long: {len(sequence)} > {max_len}")
    return sequence, spans


def build_dataset(
    rows: Any,
    tokenizer: Any,
    *,
    column: str,
    max_len: int,
    mode: str = "template",
) -> tuple[list[dict[str, list[int]]], Counter]:
    """Mask every row, returning the usable ones and a census of why the rest went.

    The census is the point of returning it: "dropped 12%" and "dropped 12%, all of them
    for prefix instability" are a shrug and a stop-work respectively.
    """
    kept: list[dict[str, list[int]]] = []
    dropped: Counter = Counter()
    for row in rows:
        messages = to_messages(row, column)
        if messages is None:
            dropped["unreadable row"] += 1
            continue
        try:
            kept.append(mask_conversation(tokenizer, messages, max_len=max_len, mode=mode))
        except MaskError as failure:
            dropped[_reason(str(failure))] += 1
    return kept, dropped


def _reason(message: str) -> str:
    """Collapse a per-turn failure into the class of failure, for the census."""
    if "not prefix-stable" in message:
        return "not prefix-stable"
    if "assembly disagrees" in message:
        return "assembly disagrees with the template"
    if message.startswith("too long"):
        return "too long"
    if "supervises nothing" in message:
        return "empty assistant turn"
    if "span" in message:
        return "span out of range"
    # "apply_chat_template raised InvalidMessageStructureException" is already a class --
    # it names both the call that failed and the way it failed, and there are few of each.
    return message


def probe_mask_modes(
    rows: Any, tokenizer: Any, *, column: str, max_len: int, sample: int = 32
) -> dict[str, int]:
    """How many of the first ``sample`` conversations each mode can mask.

    The mode cannot be read off the tokenizer. ``mistral_common``'s refusal is not an
    attribute, a flag, or a template feature -- it is a validator that only fires on the
    exact shape the per-turn walk asks for, and it fires the same way on tokenizers that
    would otherwise be fine. So this asks, on real rows, the same way the chat-template
    capability check in the eval harness asks the tokenizer rather than reading
    ``chat_prompt_style``.
    """
    counts = dict.fromkeys(MASK_MODES, 0)
    seen = 0
    for row in rows:
        if seen >= sample:
            break
        messages = to_messages(row, column)
        if messages is None:
            continue
        seen += 1
        for mode in MASK_MODES:
            try:
                mask_conversation(tokenizer, messages, max_len=max_len, mode=mode)
            except MaskError:
                continue
            counts[mode] += 1
    return counts


def choose_mask_mode(counts: dict[str, int]) -> str:
    """Ties go to ``template``: it assumes nothing about where a turn ends."""
    return "assemble" if counts.get("assemble", 0) > counts.get("template", 0) else "template"


def collate(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, Any]:
    """Right-pad. Training reads the whole sequence at once, so unlike generation the
    padding side is free -- and right padding keeps label alignment obvious."""
    import torch

    width = max(len(row["input_ids"]) for row in batch)
    input_ids, labels, mask = [], [], []
    for row in batch:
        pad = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        labels.append(row["labels"] + [-100] * pad)
        mask.append([1] * len(row["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }


# --------------------------------------------------------------------------
# The mixture itself
# --------------------------------------------------------------------------


def report_sources(rows: Any) -> tuple[Counter, list[str]]:
    """Census the mixture's ``source`` column, and name any that overlap an eval task.

    Tulu-3 contains ``ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k``: GSM8K-derived
    training data, against a campaign that evaluates GSM8K. It is reported and kept.

    Kept, because phase 3 measures *quantization damage* -- fine-tuned fp16 against
    fine-tuned-and-quantized, both trained on the same mixture -- and contamination
    inflates both sides of that comparison equally. What it does invalidate is any claim
    about the absolute GSM8K number, so the report has to carry the caveat with it. The
    alternative, silently dropping the source, would change the mixture the paper says it
    trained on for a comparison that does not need it changed.
    """
    counts: Counter = Counter()
    for row in rows:
        source = row.get("source") if isinstance(row, dict) else None
        if isinstance(source, str):
            counts[source] += 1
    flagged = [
        source for source in counts if any(marker in source.lower() for marker in _CONTAMINATING)
    ]
    return counts, sorted(flagged)


def load_rows(spec: dict[str, str], *, examples: int, seed: int) -> Any:
    """Load the mixture and take a shuffled subsample of it.

    Shuffled before the limit, always. These splits arrive grouped -- Tulu-3 is ordered by
    ``source`` -- so ``select(range(n))`` on the raw split takes one subset of one source
    and calls it a sample of the mixture. That failure has already been paid for once in
    this project on a label-sorted classification split, where it read as a destroyed
    model rather than as a bad sample.
    """
    if spec.get("builder") == "text2sql":
        return load_text2sql_rows(examples=examples, seed=seed)

    from datasets import load_dataset

    dataset = load_dataset(spec["repo"], spec.get("name"), split=spec["split"])
    dataset = dataset.shuffle(seed=seed)
    if examples > 0 and examples < len(dataset):
        dataset = dataset.select(range(examples))
    return dataset


def load_text2sql_rows(*, examples: int, seed: int) -> list[dict[str, Any]]:
    """The text-to-SQL mixture, as single-turn conversations.

    Two properties this has to preserve and neither is visible downstream if it breaks.

    The user turn comes from :func:`dynquant.eval.text2sql.instruction`, which is the
    same function the chat evaluation calls -- so the model is asked at eval time in the
    words it was trained on. A driver that rendered its own phrasing here would produce a
    fine-tune that scores badly for a reason no arm of the comparison could reveal,
    because every arm would share it.

    ``load_text2sql`` balances and interleaves the three sources itself, so the
    ``examples`` limit takes a proportional sample rather than a prefix of whichever
    source came first. It also drops the evaluation's "gold must return rows" rule for
    ``train``: an empty result set is still correct supervision, and enforcing it here
    would discard most of two sources.
    """
    from dynquant.eval.text2sql import instruction, load_text2sql

    items = load_text2sql("train", limit=examples if examples > 0 else None, seed=seed)
    return [
        {
            "messages": [
                {"role": "user", "content": instruction(item)},
                {"role": "assistant", "content": item.gold},
            ],
            # Read by `report_sources`, so the run manifest records the realised mixture.
            # The balance is a quota over *admitted* items, and admission rates differ by
            # source, so the achieved split is worth recording rather than assuming.
            "source": f"text2sql/{item.source}",
        }
        for item in items
    ]


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def resolve_regime(lora_rank: int) -> str:
    return "lora" if lora_rank > 0 else "full fine-tune"


def run_dir(out: Path, model: str, dataset: str) -> Path:
    """Where this cell's artifacts go: one directory per (model, dataset), always.

    Derived from both, and returned rather than read from the environment, because an
    env-derived run directory has already sent four Mistral arms into a Qwen directory in
    this project -- where they were evaluated with a Qwen tokenizer and looked measured.
    """
    return out / f"{model}.{dataset}"


def _git_head(repo: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def resolve_bank_measurement(model: Any, requested: bool | None) -> bool:
    """Decide whether to measure batched expert banks, refusing to guess on an MoE.

    ``requested`` is tri-state on purpose. ``True``/``False`` are the caller's decision and
    are returned unchanged. ``None`` means the flag was never passed, and what happens then
    depends on whether this model has expert banks at all:

    - **No banks.** Returns ``False``. Every dense model takes this path, so nothing about
      the existing panel changes and no caller has to learn a new flag.
    - **Banks present.** Raises. Not because the right answer is unknowable, but because
      both answers fail in ways the run will not show you. Measuring costs a gradient
      buffer for the whole expert mass and can exhaust a smaller card; not measuring
      completes normally and writes a stats file whose UNMEASURED share is most of the
      checkpoint. The second is the dangerous one, because it produces numbers.

    The tracker already warns in the second case. That warning arrives after the model is
    loaded and the hooks are attached, in a log nobody reads until the results look wrong.
    This arrives before the first step, as a non-zero exit.
    """
    if requested is not None:
        return requested

    # Imported here rather than at module scope, matching the rest of this script: it is
    # runnable on a box where dynquant is not installed, and argparse --help should not
    # depend on that.
    from dynquant.graph.experts import batched_expert_params

    banked = sum(
        param.numel() for module in model.modules() for _, param in batched_expert_params(module)
    )
    if not banked:
        return False

    raise SystemExit(
        f"this model keeps {banked:,} parameters in batched expert banks, and neither "
        "--measure-expert-banks nor --no-measure-expert-banks was passed. Measuring them "
        "costs one gradient buffer for that mass; not measuring them means the allocator "
        "scores them neutrally and their bit widths come from role floors rather than from "
        "the signal. Choose explicitly -- there is no safe default."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--dataset", default="tulu3", choices=sorted(DATASETS))
    parser.add_argument("--out", default="runs/s2", help="parent of the per-cell run directory")
    parser.add_argument("--repo", default=".", help="the dynquant checkout, for the commit stamp")
    parser.add_argument("--examples", type=int, default=50_000, help="0 uses the whole split")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=None, help="default depends on the regime")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--estimator", default="outer_exact")
    parser.add_argument(
        "--measure-expert-banks",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "measure the 3-D expert tensors an MoE stores its experts in. Unset is an "
            "error on a model that has them, rather than a default, because both answers "
            "are defensible and the wrong one is nearly invisible: off, the run completes "
            "and writes a stats file in which most of the checkpoint is UNMEASURED. On "
            "LFM2.5-8B-A1B that is 88.4%% of the weights allocated by role floors alone, "
            "in a campaign whose claim is that the signal decides the allocation. It costs "
            "one gradient buffer for the expert mass (~15.5 GB in bf16 on that model), so "
            "--no-measure-expert-banks is the right call on a card that cannot hold it"
        ),
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help=(
            "0 fine-tunes every weight. Defaults to LoRA because an 8B does not full-tune "
            "on one card -- 16 GB of bf16 parameters, 16 GB of gradients and ~64 GB of "
            "fp32 AdamW moments is ~96 GB before a single activation -- and running the "
            "two panel models under different regimes would make their arms incomparable"
        ),
    )
    parser.add_argument("--lora-alpha", type=int, default=0, help="default 2x rank")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--mask-mode",
        default="auto",
        choices=("auto", *MASK_MODES),
        help=(
            "'auto' runs both modes over --mask-probe conversations and takes the one that "
            "masks more of them; 'template' finds assistant spans inside the template's own "
            "rendering; 'assemble' hand-builds the assistant turns and verifies them against "
            "the template, for a tokenizer that will not render a conversation ending in an "
            "assistant message"
        ),
    )
    parser.add_argument(
        "--mask-probe",
        type=int,
        default=32,
        help="conversations to try each mask mode on when --mask-mode is 'auto'",
    )
    parser.add_argument(
        "--max-drop-rate",
        type=float,
        default=0.05,
        help=(
            "refuse to train if more than this fraction of conversations failed to mask. "
            "Counts assumptions that broke, not conversations over --max-len"
        ),
    )
    parser.add_argument(
        "--max-length-drop-rate",
        type=float,
        default=0.15,
        help=(
            "refuse to train if more than this fraction is dropped for exceeding --max-len. "
            "Separate from --max-drop-rate because it is a budget, not a broken assumption: "
            "it is a known function of a flag, and the fix is to change the flag"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build the data, report the masking census, and stop before touching a GPU",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    entry = MODELS[args.model]
    repo_id = str(entry["repo"])
    if entry["gated"] and not (
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    ):
        print(
            f"{args.model} ({repo_id}) is gated=manual on the Hub and no HF_TOKEN is set. "
            f"The licence is accepted per account through the web UI and cannot be "
            f"accepted from here.",
            flush=True,
        )
        return 2

    spec = DATASETS[args.dataset]
    destination = run_dir(Path(args.out), args.model, args.dataset)
    destination.mkdir(parents=True, exist_ok=True)
    print(f"s2: {args.model} x {args.dataset} -> {destination}", flush=True)

    random.seed(args.seed)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_rows(spec, examples=args.examples, seed=args.seed)
    sources, flagged = report_sources(rows)
    if sources:
        top = ", ".join(f"{name} {count}" for name, count in sources.most_common(8))
        print(f"mixture: {len(sources)} sources, top: {top}", flush=True)
    for source in flagged:
        print(
            f"NOTE: {source} ({sources[source]} rows) overlaps an evaluated task. Kept -- "
            f"both sides of the quantization comparison train on it -- but the absolute "
            f"number for that task is not a claim about the benchmark.",
            flush=True,
        )

    probe: dict[str, int] = {}
    mask_mode = args.mask_mode
    if mask_mode == "auto":
        probe = probe_mask_modes(
            rows,
            tokenizer,
            column=spec["column"],
            max_len=args.max_len,
            sample=args.mask_probe,
        )
        mask_mode = choose_mask_mode(probe)
        scores = ", ".join(f"{name} {count}" for name, count in probe.items())
        print(f"mask mode: {mask_mode} (of {args.mask_probe} probed: {scores})", flush=True)

    started = time.time()
    dataset, dropped = build_dataset(
        rows,
        tokenizer,
        column=spec["column"],
        max_len=args.max_len,
        mode=mask_mode,
    )
    total = len(dataset) + sum(dropped.values())
    rate = (sum(dropped.values()) / total) if total else 0.0
    # Split, because the two halves call for opposite responses. "Too long" is a budget:
    # it is a known function of --max-len that anyone can reason about and change. Anything
    # else is an assumption about the tokenizer that turned out to be false.
    length_rate = (dropped["too long"] / total) if total else 0.0
    mask_rate = rate - length_rate
    print(
        f"masked {len(dataset)}/{total} conversations in {time.time() - started:.0f}s "
        f"({rate:.2%} dropped: {mask_rate:.2%} unmaskable, {length_rate:.2%} over --max-len)",
        flush=True,
    )
    for reason, count in dropped.most_common():
        print(f"  dropped {count:>7} : {reason}", flush=True)
    supervised = sum(sum(label != -100 for label in row["labels"]) for row in dataset)
    tokens = sum(len(row["input_ids"]) for row in dataset)
    share = f"{supervised / tokens:.1%}" if tokens else "n/a"
    print(f"{tokens} tokens, {supervised} supervised ({share})", flush=True)

    census = {
        "model": repo_id,
        # The config too, where there is one: "HuggingFaceTB/smoltalk" alone does not say
        # which of fourteen subsets trained the model, and a census that cannot answer that
        # cannot be compared against a later run.
        "dataset": spec["repo"] if spec.get("name") is None else f"{spec['repo']}:{spec['name']}",
        "examples_requested": args.examples,
        "conversations_seen": total,
        "conversations_kept": len(dataset),
        "drop_rate": round(rate, 6),
        "unmaskable_rate": round(mask_rate, 6),
        "over_max_len_rate": round(length_rate, 6),
        "dropped_by_reason": dict(dropped),
        "tokens": tokens,
        "supervised_tokens": supervised,
        "mask_mode": mask_mode,
        "mask_mode_requested": args.mask_mode,
        "mask_mode_probe": probe,
        "max_len": args.max_len,
        "sources": dict(sources),
        "sources_overlapping_an_eval_task": flagged,
    }
    (destination / "mask_census.json").write_text(json.dumps(census, indent=2), encoding="utf-8")

    if mask_rate > args.max_drop_rate:
        print(
            f"\nREFUSING to train: {mask_rate:.2%} of conversations could not be masked, over "
            f"the --max-drop-rate of {args.max_drop_rate:.2%}. A fine-tune quietly trained on "
            f"{1 - rate:.0%} of its data is indistinguishable from one that trained on all of "
            f"it and learned less. If the census above is dominated by 'not prefix-stable' or "
            f"by a template that raised, try --mask-mode assemble.",
            flush=True,
        )
        return 3
    if length_rate > args.max_length_drop_rate:
        print(
            f"\nREFUSING to train: {length_rate:.2%} of conversations are over --max-len "
            f"({args.max_len}), past the --max-length-drop-rate of "
            f"{args.max_length_drop_rate:.2%}. Dropping the longest conversations in a "
            f"mixture selects for short answers, and at this share it stops being a tail. "
            f"Raise --max-len, or raise the ceiling if the shorter mixture is what you want.",
            flush=True,
        )
        return 3
    if args.dry_run:
        print(f"\ndry run: census written to {destination / 'mask_census.json'}", flush=True)
        return 0
    if not dataset:
        print("nothing to train on", flush=True)
        return 3

    return _train(args, repo_id, destination, dataset, tokenizer, census)


def _train(
    args: argparse.Namespace,
    repo_id: str,
    destination: Path,
    dataset: list[dict[str, list[int]]],
    tokenizer: Any,
    census: dict[str, Any],
) -> int:
    """Everything from here down needs a GPU, which is why it is behind ``--dry-run``."""
    import torch
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    from dynquant import DynQuantCallback
    from dynquant.constants import STATS_FILENAME

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    regime = resolve_regime(args.lora_rank)
    lr = args.lr if args.lr is not None else LR_BY_REGIME[regime]
    print(f"{repo_id}: {regime}, lr {lr:g}, {len(dataset)} conversations", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False  # incompatible with the backward hooks, unused in training
    if regime == "lora":
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=args.lora_rank,
            # Twice the rank unless overridden: the update is scaled by alpha/r, so tying
            # them keeps the effective step size fixed as the rank is varied.
            lora_alpha=args.lora_alpha or 2 * args.lora_rank,
            lora_dropout=args.lora_dropout,
            # `all-linear` rather than a hand-written list: the projection names differ per
            # architecture -- `gate_up_proj` on Phi, `gate_proj`/`up_proj` on Ministral --
            # and a list that misses a family trains fewer modules than intended while
            # looking like it worked.
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        whole = sum(p.numel() for p in model.parameters())
        print(
            f"LoRA r={config.r} alpha={config.lora_alpha}: {trainable / 1e6:.1f}M trainable "
            f"of {whole / 1e9:.2f}B ({trainable / whole:.2%})",
            flush=True,
        )

    measure_banks = resolve_bank_measurement(model, args.measure_expert_banks)

    # Created before the callback is constructed, and that is load-bearing rather than
    # tidiness. ``StatsFile.save`` decides between "directory" and "file path" with
    # ``Path.is_dir()`` -- a filesystem test, not a reading of the argument -- so the same
    # command writes ``stats/dynquant_stats.json`` when the directory happens to exist and a
    # file literally named ``stats`` when it does not. Making it exist first pins the answer.
    stats_dir = destination / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_file = stats_dir / STATS_FILENAME
    callback = DynQuantCallback(
        stats_dir,
        grad_estimator=args.estimator,
        log_every=50,
        subsample_tokens=256,
        measure_expert_banks=bool(measure_banks),
    )

    training_args = TrainingArguments(
        output_dir=str(destination / "trainer"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        # Off deliberately: checkpointing replays the forward pass during backward, so a
        # module's forward hook fires twice per step while its backward hook fires once.
        # The stashed activation would then be the recomputed one and the saliency EMA
        # would double-count. Memory is not the constraint here -- LoRA is.
        gradient_checkpointing=False,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,  # type: ignore[arg-type] -- a list of dicts is a map dataset
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
        callbacks=[callback],
    )

    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started

    merged = save_outputs(model, tokenizer, destination, regime=regime)
    print(f"\nsaved fine-tuned model to {merged}", flush=True)

    tracked = len(callback.tracker) if callback.tracker is not None else 0
    record = {
        **census,
        "regime": regime,
        "lora_rank": args.lora_rank,
        "estimator": args.estimator,
        "epochs": args.epochs,
        "lr": lr,
        "effective_batch": args.batch * args.accum,
        "steps": result.global_step,
        "train_loss": result.training_loss,
        "seconds": round(elapsed, 1),
        "tracked_modules": tracked,
        "stats_file": str(stats_file),
        "stats_modules": _modules_in(stats_file),
        "output": str(merged),
        "commit": _git_head(Path(args.repo)),
    }
    (destination / "s2_finetune.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8"
    )
    print(f"-> wrote {destination / 's2_finetune.json'}", flush=True)
    if tracked == 0:
        # The checkpoint is the by-product; this is the deliverable. A run that trained
        # and tracked nothing has to fail here rather than be discovered at S3.
        print("NO MODULES TRACKED: the signal map is empty", flush=True)
        return 4
    if record["stats_modules"] != tracked:
        # Checked against the file on disk, not against the tracker in memory. The tracker
        # reporting N modules proves the hooks fired; only reading the file back proves S3
        # will find them, at the path this record claims they are at.
        print(
            f"SIGNAL MAP NOT WHERE IT IS RECORDED: {stats_file} holds "
            f"{record['stats_modules']} modules, the tracker saw {tracked}",
            flush=True,
        )
        return 5
    print(f"-> signal map: {tracked} modules at {stats_file}", flush=True)
    return 0


def save_outputs(model: Any, tokenizer: Any, destination: Path, *, regime: str) -> Path:
    """Write what the arm produced, and return the directory downstream stages load.

    That directory holds a **merged** model, not an adapter. Every stage after this one
    opens it as a plain causal LM -- S3 quantizes it, S4 evaluates it -- so publishing
    adapter weights as the deliverable would have S4 silently score the *base* model, an
    arm that looks measured and is not.

    The adapter is written anyway, beside the merge and never in its place. Order is the
    whole point: ``merge_adapters`` folds the deltas into the base weights and unloads in
    place, so the line below is the last moment the adapter exists as a separable object.
    It costs a few percent of the merge's size and rebuilds it exactly, which is what
    makes it worth writing at all -- these arms run on a box whose ``/workspace`` is not
    a volume, where losing ``merged/`` costs a 13-hour fine-tune and losing ``adapter/``
    costs a two-minute re-merge.
    """
    # Imported here, as everything heavy in this script is: `--dry-run` and `--help`
    # have to work on a box with no torch installed.
    from dynquant.integration.peft_utils import merge_adapters

    if regime == "lora":
        model.save_pretrained(str(destination / "adapter"))
    saved = merge_adapters(model) if regime == "lora" else model
    saved.config.use_cache = True
    merged = destination / "merged"
    saved.save_pretrained(str(merged))
    tokenizer.save_pretrained(str(merged))
    return merged


def _modules_in(stats_file: Path) -> int:
    """How many modules the stats file on disk actually holds. ``-1`` if unreadable."""
    try:
        payload = json.loads(stats_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return -1
    layers = payload.get("layers")
    return len(layers) if isinstance(layers, dict) else -1


if __name__ == "__main__":
    raise SystemExit(main())
