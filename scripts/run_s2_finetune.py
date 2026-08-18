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

#: torchrun's rank for this process, or -1 when launched directly. Read once, at import,
#: because it decides three separate things -- which card the replica lands on, whether
#: this process writes the run's artifacts, and what a plain `python` invocation means
#: -- and reading it three times invites the three to disagree.
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "-1"))

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
    #
    # Three, not four: the registry also holds Spider, which is scored and never trained
    # on. Naming it here would put a corpus in the provenance of a run that never read it.
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
MASK_MODES = ("template", "assemble", "seam")


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
    ``mistral_common``. ``mode="seam"`` splits on the rendered *text* and tokenizes each
    side, which is the only way to mask anything for a template whose generation prompt
    ends inside a BPE merge (:func:`_seam_conversation`). No mode is right for every
    tokenizer and none can be chosen from an attribute, so the driver measures all of them
    against the tokenizer it was actually given -- see :func:`probe_mask_modes`.
    """
    if not messages:
        raise MaskError("empty conversation")
    if not any(message["role"] == "assistant" for message in messages):
        raise MaskError("no assistant turn to supervise")

    if mode == "template":
        input_ids, spans = _walk_template(tokenizer, messages, max_len=max_len)
    elif mode == "assemble":
        input_ids, spans = _assemble_conversation(tokenizer, messages, max_len=max_len)
    elif mode == "seam":
        input_ids, spans = _seam_conversation(tokenizer, messages, max_len=max_len)
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


def _render_text(tokenizer: Any, messages: list[dict[str, str]], *, generation_prompt: bool) -> str:
    """One prefix of a conversation, as the string the template emits."""
    try:
        rendered = tokenizer.apply_chat_template(
            [dict(message) for message in messages],
            tokenize=False,
            add_generation_prompt=generation_prompt,
        )
    except MaskError:
        raise
    except Exception as exc:
        raise MaskError(f"apply_chat_template raised {type(exc).__name__}") from exc
    if isinstance(rendered, (list, tuple)) and len(rendered) == 1:
        rendered = rendered[0]
    if not isinstance(rendered, str):
        raise MaskError("apply_chat_template(tokenize=False) did not return a string")
    return rendered


def _encode(tokenizer: Any, text: str) -> list[int]:
    """Tokenize a fragment as a fragment: no BOS, no EOS, no framing of its own."""
    if not text:
        return []
    ids = _as_token_ids(tokenizer(text, add_special_tokens=False))
    if ids is None:
        raise MaskError("the tokenizer did not return ids for a rendered fragment")
    return ids


def _seam_conversation(
    tokenizer: Any, messages: list[dict[str, str]], *, max_len: int
) -> tuple[list[int], list[tuple[int, int]]]:
    """Split on the rendered text, and tokenize each side of the split separately.

    Both other modes require the generation prompt's tokens to be a *token* prefix of the
    turn that follows it. Qwen3.5 breaks that, and not by a bug: its generation prompt
    ends ``<think>\n`` and its rendered assistant turn continues ``\n</think>``, so the
    template's own output contains ``\n\n`` across the boundary and BPE merges it into
    one token. The prompt tokenizes ``<think>``, ``\n``; the full render tokenizes
    ``<think>``, ``\n\n``. The string is a prefix and the tokens are not, and the last
    prompt token differs -- so ``template`` and ``assemble`` both reject every row of the
    mixture, which is what they should do rather than supervise at an offset.

    This mode asks the weaker and more useful question. The seam is placed where *serving*
    puts it: the prompt is tokenized exactly as the harness will tokenize it, the rest is
    tokenized as its own fragment, and the two are concatenated. Training then sees
    ``<think>``, ``\n`` and learns to emit ``\n``, ``</think>`` -- which is exactly the
    continuation the model is asked for at inference, from exactly the tokens it will have
    in context.

    The cost is that the sequence is not the tokenizer's canonical encoding of its own
    text: one token pair at each seam is split where BPE would have merged it. That is a
    real difference and it is the *right* one -- canonical encoding is not what generation
    produces, and a model trained on the canonical merge would be trained on a prompt
    token it can never be given.

    Going through text is otherwise the exact mistake that cost S1 two re-runs, so it is
    permitted only where it is checked: every turn re-encodes the whole prompt and compares
    it against what ``tokenize=True`` produced. A tokenizer that does not read its own
    control tokens back out of text fails that on the first turn and the mode masks
    nothing, which is what sends the probe to ``assemble``.

    Multi-turn carries a bounded cost from that split. Each seam leaves its two tokens in
    the context where serving would have one, so the second supervised turn's prompt runs a
    token longer than the canonical encoding of the same text and the third two longer --
    the token before each span is still exactly the one serving hands the model, which is
    what the gradient is computed against, but the deeper context drifts by one token per
    seam already passed. It is a real cost and it is the reason this mode sorts last in
    :func:`choose_mask_mode`; on a single-turn mixture there is exactly one seam and it is
    the last thing in the prompt, so nothing drifts at all.

    Chaining across turns is strict: each turn's prompt must extend the text already
    committed, verbatim. A template that re-closes a conversation it is not being asked to
    continue -- Phi-4-mini ends ``<|end|><|endoftext|>`` mid-conversation -- fails that and
    the row is dropped. No allowance is made for it here, because such a template does not
    have this mode's problem in the first place and ``template`` masks it correctly with
    the terminator allowance it already has. The probe picks whichever mode works.
    """
    committed = ""
    ids: list[int] = []
    spans: list[tuple[int, int]] = []

    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prompt_text = _render_text(tokenizer, messages[:index], generation_prompt=True)
        through_text = _render_text(tokenizer, messages[: index + 1], generation_prompt=False)
        if not prompt_text.startswith(committed):
            raise MaskError(f"turn {index}: the template rewrote text it had already emitted")
        if not through_text.startswith(prompt_text):
            raise MaskError(f"turn {index}: rendering is not prefix-stable")
        if _encode(tokenizer, prompt_text) != _render(
            tokenizer, messages[:index], generation_prompt=True
        ):
            # The guard that makes a text path admissible at all. `mistral_common` renders
            # its frame as the characters `[INST]` and `tekken` will not read them back out
            # of user text, so re-encoding a render there silently produces a sequence with
            # no BOS and no frame -- a model trained on it is trained under a frame it will
            # never be served with, and the damage surfaces two stages later looking like
            # quantization loss. Rather than assume a tokenizer round-trips its own control
            # tokens, ask it: the text path is used only where it agrees with the
            # `tokenize=True` path on the prompt, and the row is dropped where it does not.
            raise MaskError(f"turn {index}: the text path does not round-trip the frame")

        ids += _encode(tokenizer, prompt_text[len(committed) :])
        start = len(ids)
        ids += _encode(tokenizer, through_text[len(prompt_text) :])
        if len(ids) == start:
            raise MaskError(f"turn {index}: supervises nothing")
        spans.append((start, len(ids)))
        committed = through_text

    if len(ids) > max_len:
        # Checked once at the end rather than per turn: the whole sequence is what has to
        # fit, and a per-turn check would report the wrong turn as the long one.
        raise MaskError(f"too long: {len(ids)} > {max_len}")
    return ids, spans


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
    """The mode that masked the most rows; ties broken by how much each mode assumes.

    ``template`` first because it assumes nothing -- the sequence is the template's own
    output, token for token. Then ``assemble``, which supplies the turn terminator itself
    and re-verifies it against the template every turn. Then ``seam``, which is the only
    one that emits a sequence the tokenizer would not have produced from its own text, and
    so is the one to reach for last and only when the others reach nothing.
    """
    return max(MASK_MODES, key=lambda mode: (counts.get(mode, 0), -MASK_MODES.index(mode)))


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


def load_rows(
    spec: dict[str, str], *, examples: int, seed: int, sources: list[str] | None = None
) -> tuple[Any, dict[str, int]]:
    """Load the mixture and take a shuffled subsample of it.

    Shuffled before the limit, always. These splits arrive grouped -- Tulu-3 is ordered by
    ``source`` -- so ``select(range(n))`` on the raw split takes one subset of one source
    and calls it a sample of the mixture. That failure has already been paid for once in
    this project on a label-sorted classification split, where it read as a destroyed
    model rather than as a bad sample.

    Returns the rows and, beside them, how many rows the loader dropped for appearing in
    the evaluation set, per source. Returned rather than logged, and returned rather than
    stashed on the module, because it goes into ``mask_census.json`` -- and a census that
    reported a stale count from a previous call would be worse than one that reported
    none.
    """
    if spec.get("builder") == "text2sql":
        return load_text2sql_rows(examples=examples, seed=seed, sources=sources)

    from datasets import load_dataset

    dataset = load_dataset(spec["repo"], spec.get("name"), split=spec["split"])
    dataset = dataset.shuffle(seed=seed)
    if examples > 0 and examples < len(dataset):
        dataset = dataset.select(range(examples))
    # No decontamination on this path, and the empty dict says so rather than implying a
    # clean result: these mixtures are name-checked against `_CONTAMINATING` and kept.
    return dataset, {}


def load_text2sql_rows(
    *, examples: int, seed: int, sources: list[str] | None = None
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The text-to-SQL mixture, as single-turn conversations.

    Two properties this has to preserve and neither is visible downstream if it breaks.

    The user turn comes from :func:`dynquant.eval.text2sql.instruction`, which is the
    same function the chat evaluation calls -- so the model is asked at eval time in the
    words it was trained on. A driver that rendered its own phrasing here would produce a
    fine-tune that scores badly for a reason no arm of the comparison could reveal,
    because every arm would share it.

    ``load_text2sql`` balances and interleaves the sources itself, so the ``examples``
    limit takes a proportional sample rather than a prefix of whichever source came
    first. It also drops the evaluation's "gold must return rows" rule for ``train``: an
    empty result set is still correct supervision, and enforcing it here would discard
    most of two sources.

    ``sources`` is ``None`` for the campaign default, which is the three corpora
    :data:`~dynquant.eval.text2sql_sources.DEFAULT_TRAIN` names and does *not* include
    Spider. Naming Spider explicitly is a supported and different experiment: its train
    and dev splits use disjoint databases by construction, which is what the benchmark is
    for, so training on ``spider/train`` and scoring on ``spider/validation`` is the
    standard Spider protocol rather than a leak. It is not the default because a run that
    quietly acquired it would report in-domain accuracy under the same name as the
    held-out number. Whichever list is used, the decontamination filter runs against the
    *evaluation* questions and its per-source drop count is returned, so an actual
    question overlap is counted rather than argued about.
    """
    from dynquant.eval.text2sql import instruction, load_text2sql
    from dynquant.eval.text2sql_sources import SourceTally

    tallies: dict[str, SourceTally] = {}
    items = load_text2sql(
        "train",
        sources=sources,
        limit=examples if examples > 0 else None,
        seed=seed,
        tallies=tallies,
    )
    # Decontamination is on by default for `train`, and this is where it becomes visible.
    # `b-mc2/sql-create-context` holds 189 of the 200 WikiSQL items this campaign scores;
    # `load_text2sql` drops them, and a run whose manifest does not say how many were
    # dropped cannot be told apart from a run where the filter silently stopped working.
    decontaminated = {
        name: tally.contaminated for name, tally in tallies.items() if tally.contaminated
    }
    for name, dropped in decontaminated.items():
        print(
            f"decontaminated: dropped {dropped} {name} rows that ask a question the "
            f"evaluation asks",
            flush=True,
        )
    rows = [
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
    return rows, decontaminated


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
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "load the weights from this directory instead of the Hub. The registry entry "
            "still names the run; this only says where its bytes are"
        ),
    )
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
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="checkpoint interval; 0 disables checkpointing entirely",
    )
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
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help=(
            "QLoRA: hold the frozen base in bitsandbytes NF4 and train the adapter "
            "against it. Buys the base at about a quarter of its bf16 footprint, which "
            "is what makes a 27B trainable on one card -- and under torchrun, what "
            "makes each rank's full replica affordable. The cost is paid at merge "
            "time, not here; see _merge_onto_full_precision. Requires --lora-rank > 0."
        ),
    )
    parser.add_argument("--lora-alpha", type=int, default=0, help="default 2x rank")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=(
            "recompute each block's activations during backward instead of storing them. "
            "Trades roughly 30%% of step time for an activation footprint that stops "
            "scaling with depth -- the difference between 93 GiB and a few on a 64-layer "
            "27B at 3072 tokens. Safe for the signals: the tracker drops forwards that "
            "fire inside a backward, so a replay is not counted as an observation"
        ),
    )
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
    parser.add_argument(
        "--train-sources",
        default=None,
        help=(
            "comma-separated text2sql corpora to train on, e.g. "
            "spider,gretel,wikisql,create-context. Default: the campaign's three, which "
            "exclude spider. Naming spider trains on its train split and leaves its "
            "validation split -- the scored one -- untouched; the databases are disjoint "
            "across those splits, and the decontamination filter reports any question that "
            "is not. Only applies to the text2sql dataset."
        ),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def resolve_model_source(args: argparse.Namespace, entry: dict[str, Any]) -> str:
    """Where the weights come from: ``--model-path`` if given, otherwise the Hub repo.

    The whole point is to stop a second copy being fetched over a checkout that is already
    on disk -- 16.9 GB for LFM2.5-8B-A1B, at a rate that stalled outright, while the
    complete file sat in ``/workspace/models``. A local path is also why the gate check is
    skipped here: a gated model already downloaded needs no token to read.

    Checked for ``config.json`` rather than merely for existence, because the failure this
    guards against is a *plausible* wrong path -- a run directory, a parent, a merge that
    has not been written yet -- and ``from_pretrained`` on one of those either raises a
    hundred lines later or silently reaches the Hub for the name it was given.
    """
    if args.model_path is None:
        return str(entry["repo"])
    path = Path(args.model_path).resolve()
    if not (path / "config.json").is_file():
        raise SystemExit(
            f"--model-path {path} has no config.json, so it is not a checkpoint directory. "
            f"Omit the flag to fetch {entry['repo']} from the Hub."
        )
    print(f"weights: {path} (registry entry {entry['repo']} not fetched)", flush=True)
    return str(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    entry = MODELS[args.model]
    repo_id = resolve_model_source(args, entry)
    if (
        args.model_path is None
        and entry["gated"]
        and not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
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

    train_sources = (
        [name.strip() for name in args.train_sources.split(",") if name.strip()]
        if args.train_sources
        else None
    )
    rows, decontaminated = load_rows(
        spec, examples=args.examples, seed=args.seed, sources=train_sources
    )
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
        # The registry identity, not `repo_id`: with `--model-path` those differ, and a
        # census that recorded only the resolved path could not be compared against a run
        # that fetched the same model from the Hub.
        "model": str(entry["repo"]),
        "model_path": None if args.model_path is None else str(Path(args.model_path).resolve()),
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
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "sources": dict(sources),
        # What was *asked* for, beside the realised mixture above. `None` records that
        # the run took the registry default rather than that it chose the same list --
        # a later reader comparing two runs needs to know whether a difference in the
        # realised mixture came from the request or from the admission rates.
        "train_sources_requested": train_sources,
        "sources_overlapping_an_eval_task": flagged,
        # What that check actually checked. An empty list above is only as strong as this
        # list, and on the text-to-SQL mixture it is worth nothing: the markers are
        # phase 3's, no SQL corpus name contains one, and the empty result was a check
        # that could not fire rather than a mixture that passed. Recorded so a later
        # reader can tell those two apart without reading this file.
        "contamination_markers": list(_CONTAMINATING),
        # The check that can. Names are a weak test for a mixture whose sources share a
        # repo with the evaluation and differ only by split; this is the per-source count
        # of rows dropped for asking a question the evaluation asks. Empty means the
        # loader had no decontamination step to run, not that it ran and found nothing --
        # `load_rows` returns `{}` on the generic Hub path.
        "decontaminated": decontaminated,
    }
    if LOCAL_RANK <= 0:
        # Every rank builds the same census from the same seeded loader, so every rank
        # would write the same bytes to the same path at the same time. Identical
        # content is no defence against interleaved writes -- a truncated JSON here
        # loses the record of what the run trained on, which is not recoverable from
        # the checkpoint.
        (destination / "mask_census.json").write_text(
            json.dumps(census, indent=2), encoding="utf-8"
        )

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
    if args.load_4bit and regime != "lora":
        # NF4 base weights are frozen and not trainable, so a "full fine-tune" that
        # loaded them would update nothing and still report a loss curve. Refuse
        # rather than train a model that cannot move.
        print(
            "--load-4bit is QLoRA: it freezes the base in NF4 and trains an adapter, "
            "so it needs --lora-rank > 0. Given rank 0, there would be nothing "
            "trainable in the model at all.",
            flush=True,
        )
        return 3
    lr = args.lr if args.lr is not None else LR_BY_REGIME[regime]
    print(f"{repo_id}: {regime}, lr {lr:g}, {len(dataset)} conversations", flush=True)

    # Under torchrun every rank holds a full replica and each has to land on *its*
    # card. "cuda" names device 0 for all of them, so a two-rank job puts two replicas
    # on GPU 0 and leaves GPU 1 idle -- an OOM on a model this size, and on a smaller
    # one a run that reports two ranks while using one card.
    device_map = "cuda" if LOCAL_RANK < 0 else {"": LOCAL_RANK}

    load_kwargs: dict[str, Any] = {}
    if args.load_4bit:
        from transformers import BitsAndBytesConfig

        # NF4 with double quantization: QLoRA as published. The compute dtype is bf16 to
        # match the rest of the run -- the weight is stored in 4 bits and every matmul is
        # still done in bf16, so the adapter's gradients, and the activation moments the
        # tracker reads off the same tensors, are in the dtype this arm claims.
        #
        # Deliberately no `prepare_model_for_kbit_training`. It exists to turn on
        # gradient checkpointing and to upcast non-quantized parameters to fp32, and this
        # run wants neither: checkpointing replays the forward, firing each saliency hook
        # twice per step against one backward, and the upcast would put this model's
        # untied 248k x 5120 embedding *and* head into fp32 for ~10 GiB that buys nothing
        # -- peft creates the LoRA parameters in fp32 either way.
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=args.trust_remote_code,
        **load_kwargs,
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
        **warmup_kwargs(TrainingArguments),
        weight_decay=0.0,
        bf16=True,
        # Was unconditionally off, for a reason that has since been fixed one layer down.
        # Checkpointing replays a block's forward during backward and module forward hooks
        # fire on the replay, so the saliency EMA used to count each micro-batch twice.
        # `signals.tracker._in_backward` now separates the two -- the graph-task id is the
        # only thing that differs between them -- and a replayed forward is dropped.
        #
        # Still off by default, because every arm of the campaign so far trained without
        # it and the flag costs ~30% of step time on models that do not need it. It is not
        # optional on a 27B: at 3072 tokens and batch 1, saving every layer's activations
        # needs 93 GiB, which is what a 96 GiB card OOMs on.
        gradient_checkpointing=args.gradient_checkpointing,
        # Non-reentrant, and that matters under LoRA rather than being a style choice. The
        # reentrant implementation runs the block's forward under `no_grad` and recovers
        # the graph from the block's *inputs*, so a block whose inputs are all frozen --
        # which under QLoRA is every block, the embedding being frozen too -- produces no
        # gradient at all and the run trains nothing while reporting a falling loss. The
        # usual fix is `enable_input_require_grads`; not needing it is cleaner.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        # Was "no". A twelve-hour run then had nothing to fall back on, and a box that
        # died at step 1800 cost all 1800 -- which is exactly what happened once, on a
        # host whose GPU turned out to be power-capped to a sixth of its clock.
        #
        # A checkpoint here is worth more than recoverable weights. ``on_save`` calls
        # ``tracker.save(checkpoint)``, so every checkpoint carries a complete signal map
        # as of that step, and the signal map is the artifact the panel cannot be run
        # without and cannot be recomputed without retraining. Only the LoRA adapter is
        # written, so three of them do not reach a gigabyte.
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=3,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        # Every module in this model runs on every forward -- it is dense, and the
        # linear-attention layers are unconditional -- so the unused-parameter search has
        # nothing to find and is pure per-step overhead. It would be wrong rather than
        # merely slow on an MoE, where a rank whose batch routes to no expert *e*
        # produces no gradient for *e* and DDP would hang waiting for one.
        ddp_find_unused_parameters=False,
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

    if LOCAL_RANK > 0:
        # Everything below writes this run's artifacts, and they are one set of files
        # per run rather than one per rank. Two ranks calling `save_pretrained` on the
        # same directory interleave shard writes; two ranks reloading a bf16 27B to
        # merge onto costs twice the host RAM. Rank 0 does it.
        #
        # Safe to leave now: every collective this script issues has already run.
        # `tracker.save` all-reduces the signals and *then* returns None off rank 0, and
        # it is called from `on_train_end` -- inside `trainer.train()`, which returned.
        print(f"rank {LOCAL_RANK}: trained {result.global_step} steps, rank 0 writes", flush=True)
        return 0

    merged = save_outputs(
        model,
        tokenizer,
        destination,
        regime=regime,
        full_precision_base=repo_id if args.load_4bit else None,
        trust_remote_code=args.trust_remote_code,
    )
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
    if measure_banks:
        # Only when they were asked for. Under --no-measure-expert-banks their absence is
        # the requested outcome, and a gate that fired anyway would make the flag unusable
        # on the card it exists for.
        missing = banked_entries_missing(model, stats_file)
        if missing:
            print(
                f"EXPERT BANKS REQUESTED BUT NOT IN THE SIGNAL MAP: {len(missing)} tensors "
                f"absent, first few {missing[:4]}. The allocator would score this mass "
                f"neutrally and set its widths from role floors.",
                flush=True,
            )
            return 6
    print(f"-> signal map: {tracked} modules at {stats_file}", flush=True)
    return 0


#: Warmup as a fraction of the run. Every phase-3 arm has trained under this number and
#: it is a fraction rather than a step count on purpose: the campaign runs mixtures that
#: differ by 5x in size, and a fixed step count would mean a different thing in each.
WARMUP_FRACTION = 0.03


def warmup_kwargs(training_arguments: Any) -> dict[str, float]:
    """Spell the warmup fraction the way the installed ``transformers`` spells it.

    transformers 5 removed ``warmup_ratio`` and folded it into ``warmup_steps``, which now
    reads any value below 1 as a fraction of the run. The two spellings are not
    interchangeable in the direction that matters: passing 0.03 to a 4.x ``warmup_steps``
    raises nothing and warms up for 0.03 *steps* -- which is none -- so a 27B QLoRA run
    would take its first optimizer step at the full learning rate and the only evidence
    would be a loss curve nobody has a reference for.

    So the spelling is chosen from what the class *declares* rather than by passing one
    and catching the failure. Catching would work for 5 (``warmup_ratio`` is a TypeError
    there) and is exactly the case that fails silently on 4.

    Turns on the presence of the field, not on ``transformers.__version__``: a version
    string is a claim about a distribution and this is a question about a class.
    """
    import dataclasses

    try:
        fields = {field.name for field in dataclasses.fields(training_arguments)}
    except TypeError:  # not a dataclass -- a test double, or a future rewrite
        fields = set(vars(training_arguments))
    if "warmup_ratio" in fields:
        return {"warmup_ratio": WARMUP_FRACTION}
    if "warmup_steps" not in fields:
        raise RuntimeError(
            "TrainingArguments declares neither warmup_ratio nor warmup_steps; "
            "refusing to train without a warmup rather than guessing at the spelling"
        )
    return {"warmup_steps": WARMUP_FRACTION}


def save_outputs(
    model: Any,
    tokenizer: Any,
    destination: Path,
    *,
    regime: str,
    full_precision_base: str | None = None,
    trust_remote_code: bool = False,
) -> Path:
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
    if regime != "lora":
        saved = model
    elif full_precision_base is None:
        saved = merge_adapters(model)
    else:
        saved = _merge_onto_full_precision(
            destination / "adapter", full_precision_base, trust_remote_code
        )
    saved.config.use_cache = True
    merged = destination / "merged"
    saved.save_pretrained(str(merged))
    tokenizer.save_pretrained(str(merged))
    return merged


def _merge_onto_full_precision(adapter: Path, base: str, trust_remote_code: bool) -> Any:
    """Fold a QLoRA adapter into bf16 base weights rather than into the NF4 ones.

    ``merge_and_unload`` on a bitsandbytes base does the arithmetic right and stores the
    answer wrong: it dequantizes each ``Linear4bit``, adds ``BA``, and quantizes the sum
    straight back to NF4. What comes out is a 4-bit model wearing bf16's dtype, every
    weight already snapped to one of sixteen levels per block.

    Downstream that is not a small error, it is a different experiment. S3 would allocate
    bit widths for tensors bitsandbytes had already quantized, and S4 would report
    DynQuant's cost against a baseline that is not full precision -- two quantizations'
    damage summed and attributed to one of them. The comparison the campaign exists to
    make would be measuring NF4 plus DynQuant against fp16 while calling it DynQuant.

    So the adapter goes onto the weights it is a low-rank residual *of*: the base is
    reloaded in bf16 and merged there. On CPU, because the card is still holding the
    trained replica and a 27B in bf16 is ~46 GiB against the box's 251 GiB of host RAM.
    It costs a few minutes of memory bandwidth once, at the end of a multi-hour run.

    Only reached under ``--load-4bit``. A bf16 run merges in place, where the base
    already *is* the full-precision weights and there is nothing to reload.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    print(f"merging the adapter onto bf16 weights reloaded from {base}", flush=True)
    full = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    wrapped = PeftModel.from_pretrained(full, str(adapter), device_map="cpu")
    # `safe_merge` reconstructs into a scratch tensor and refuses on a non-finite value,
    # which is the one failure mode that would otherwise reach the Hub silently.
    return wrapped.merge_and_unload(safe_merge=True)


def banked_entries_missing(model: Any, stats_file: Path) -> list[str]:
    """Expert-bank tensors the stats file does not carry, by the name S3 will look up.

    ``stats_modules == tracked`` already checks that the file holds what the tracker saw.
    It cannot check *this*: the banks are 91.5% of LFM2.5-8B-A1B and 22 of its modules, so
    a run that hooked none of them still writes hundreds of attention and dense entries and
    still matches its own tracker exactly. The count agrees and the mass is gone.

    The keys are rebuilt from the model rather than read from a list, using the same
    ``canonical_name(module) + "." + parameter`` the tracker writes and
    :func:`classify_model` looks up. Anything else would be a second spelling of the
    contract, and a gate that checks a name nobody uses passes for the wrong reason -- the
    failure this whole flag exists to prevent, one level up.

    Returns the missing names rather than a boolean so the message can say which, because
    "some banks are absent" and "layer 7's ``w1`` is absent" send a reader to different
    places.
    """
    from dynquant.graph.experts import batched_expert_params
    from dynquant.integration.peft_utils import canonical_name

    try:
        payload = json.loads(stats_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ["<stats file unreadable>"]
    layers = payload.get("layers")
    if not isinstance(layers, dict):
        return ["<stats file has no layers map>"]

    missing = []
    for name, module in model.named_modules():
        for param_name, _ in batched_expert_params(module):
            key = f"{canonical_name(name)}.{param_name}"
            if key not in layers:
                missing.append(key)
    return missing


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
