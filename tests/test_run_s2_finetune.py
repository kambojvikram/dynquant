"""The S2 driver has to be right before it is run, because running it is the cost.

Two fine-tunes, ~30 GPU-hours, and the deliverable is not the checkpoint -- it is the
signal map harvested while the weights move. Every failure covered here produces a run
that finishes, saves a model, writes a stats file, and is wrong:

* labels shifted by one token, because a chat template turned out not to be prefix-stable;
* the turn frame delivered as text, so the model trains on characters it will never be
  evaluated under (this one has already happened, on the evaluation side, and cost 56
  points of HumanEval);
* a mixture subsampled off the front of a split that arrives grouped by source;
* a conversation with nothing supervised in it, whose loss is a mean over zero elements;
* an assistant turn hand-assembled with the wrong terminator on the end of it, or a
  terminator mistaken for drift and half the mixture dropped over one token.

None of these raise. All of them are cheap to pin from CPU CI, because the masking is a
pure function of a tokenizer and a list of turns.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "scripts" / "run_s2_finetune.py"

# The driver reaches `run_s1_headroom` for `MODELS`, which imports the eval command,
# which imports transformers -- so the module cannot even load without it, and in the
# core-only `test` matrix job that is a collection error rather than a skip. The two
# pinned-transformers jobs are where these run. Declared here rather than inside the
# fixture so collection reports "skipped" instead of erroring on every test in the file.
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def s2():
    spec = importlib.util.spec_from_file_location("_dq_s2", DRIVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dq_s2"] = module
    spec.loader.exec_module(module)
    return module


def _enc(text: str) -> list[int]:
    """Content ids, offset well past every control id used below, so a control token in
    an output can only have come from a template and never from a character that
    happens to collide with one."""
    return [1000 + ord(character) for character in text]


class _Templated:
    """A Jinja-backed tokenizer: role headers are control tokens, rendering is prefix-stable.

    Records every ``tokenize`` it was asked for, so a regression to the text path is
    visible as data rather than only as a wrong number.
    """

    BOS, USER, ASSISTANT, END = 4, 1, 2, 3
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "</s>"
    eos_token_id = 3  # == END: this template ends an assistant turn with its EOS

    def __init__(self) -> None:
        self.tokenize_flags: list[bool] = []

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False, "message content must not be framed twice"
        return {"input_ids": _enc(text)}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
    ):
        self.tokenize_flags.append(tokenize)
        assert not (add_generation_prompt and continue_final_message)
        ids = [self.BOS]
        for index, message in enumerate(messages):
            ids.append(self.USER if message["role"] == "user" else self.ASSISTANT)
            ids += _enc(message["content"])
            if not (continue_final_message and index == len(messages) - 1):
                ids.append(self.END)
        if add_generation_prompt:
            ids.append(self.ASSISTANT)
        if not tokenize:
            return "".join(f"<|{m['role']}|>{m['content']}" for m in messages)
        return ids


class _Drifting(_Templated):
    """Prefix-stable *almost*: one BPE merge spans the assistant turn boundary.

    The header and the first content token become a single id whenever the conversation
    is rendered through its assistant turn, so the prefix lengths measured from the
    generation-prompt render are one too many. This is the realistic version of the
    failure -- it is off by one token, it is invisible in a decoded string, and it shifts
    every label in the conversation.
    """

    MERGED = 900

    def apply_chat_template(self, messages, *, tokenize=False, **kwargs):
        ids = super().apply_chat_template(messages, tokenize=tokenize, **kwargs)
        if not tokenize or kwargs.get("add_generation_prompt") or not messages:
            return ids
        if messages[-1]["role"] != "assistant":
            return ids
        where = len(ids) - 1 - ids[::-1].index(self.ASSISTANT)
        return [*ids[:where], self.MERGED, *ids[where + 2 :]]


class _Terminating(_Templated):
    """Renders a *finished* conversation with a sequence terminator on the end.

    Phi-4-mini: an assistant turn at the end of the conversation is closed with
    ``<|end|><|endoftext|>`` where the same turn mid-conversation is followed by
    ``<|end|><|user|>``. One trailing token of disagreement, on 2% of Tulu-3, and it is not
    drift -- the two renders agree everywhere the model will ever see them.
    """

    EOT = 5

    def apply_chat_template(self, messages, *, tokenize=False, **kwargs):
        ids = super().apply_chat_template(messages, tokenize=tokenize, **kwargs)
        if not tokenize or kwargs.get("add_generation_prompt") or not messages:
            return ids
        if messages[-1]["role"] != "assistant" or kwargs.get("continue_final_message"):
            return ids
        return [*ids, self.EOT]


class _LateDrifting(_Templated):
    """Diverges from the full sequence by more than a terminator's worth of tokens.

    The generation-prompt prefix still matches exactly, so the start check passes and only
    the bound on the end can catch this. It is the case that says the terminator allowance
    is an allowance and not a hole.
    """

    def apply_chat_template(self, messages, *, tokenize=False, **kwargs):
        ids = super().apply_chat_template(messages, tokenize=tokenize, **kwargs)
        if not tokenize or kwargs.get("add_generation_prompt") or not messages:
            return ids
        if messages[-1]["role"] != "assistant" or kwargs.get("continue_final_message"):
            return ids
        return [*ids[:-3], 901, 902, 903]


class _WrongEos(_Templated):
    """``eos_token_id`` is not what this template ends an assistant turn with.

    Phi-4-mini is exactly this: ``eos_token_id`` is ``<|endoftext|>`` (199999) and the turn
    terminator is ``<|end|>`` (200020). The assemble mode's terminator is a hypothesis, so
    the case that matters is what a wrong one costs.
    """

    eos_token_id = 99


class _SeamMerging:
    """A thinking template whose generation prompt ends inside a BPE merge.

    Qwen3.5, reduced to the one property that matters. Its generation prompt opens a
    reasoning block and ends ``<think>\n``; the rendered assistant turn continues
    ``\n</think>``, so the template's own output contains ``\n\n`` across the boundary
    and the tokenizer merges the pair into a single id. The prompt's *text* is a prefix of
    the full render's and its *tokens* are not -- the last prompt token differs -- which is
    what ``template`` and ``assemble`` are right to refuse and what ``seam`` exists for.

    Its frame is written as sentinel characters that it reads back as its own control ids.
    That is not a convenience: it is the property that makes the text path admissible here
    and inadmissible for :class:`_MistralShaped`, and the seam mode checks for it per turn
    rather than assuming it.
    """

    USER, ASSISTANT, END, NL, NLNL = 1, 2, 3, 6, 7
    FRAME: ClassVar[dict[str, int]] = {
        "\x01": USER,
        "\x02": ASSISTANT,
        "\x03": END,
        "\n": NL,
    }

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "</s>"
    eos_token_id = 3

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False, "message content must not be framed twice"
        return {"input_ids": self._encode(text)}

    @classmethod
    def _encode(cls, text: str) -> list[int]:
        ids: list[int] = []
        index = 0
        while index < len(text):
            if text.startswith("\n\n", index):  # the merge, and the whole point
                ids.append(cls.NLNL)
                index += 2
                continue
            ids.append(cls.FRAME.get(text[index], 1000 + ord(text[index])))
            index += 1
        return ids

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
    ):
        assert not (add_generation_prompt and continue_final_message)
        parts = []
        for message in messages:
            if message["role"] == "user":
                parts.append(f"\x01{message['content']}\x03")
            else:
                # The closed, empty reasoning block a thinking template emits for a turn it
                # is not being asked to continue -- this is where the `\n\n` comes from.
                parts.append(f"\x02\n\n{message['content']}\x03")
        if add_generation_prompt:
            parts.append("\x02\n")
        text = "".join(parts)
        return self._encode(text) if tokenize else text


class _MistralShaped:
    """``MistralCommonBackend``: refuses the render, merges turns, frames in unreadable text.

    Three independent behaviours in one tokenizer, each of which broke a design.

    First, it validates a *serving* request: ``apply_chat_template`` on any conversation
    ending in an assistant message raises -- ``InvalidMessageStructureException`` with
    ``add_generation_prompt=False``, a ``ValueError`` naming ``continue_final_message`` with
    ``True``. Every prefix the per-turn walk asks for is that shape, so the walk cannot run
    at all. ``continue_final_message=True`` is the one way in, and it renders the assistant
    turn open -- which is exactly what the training side wants.

    Second, it *merges* adjacent messages that share a role, joining them with a separator
    inside a single frame. Appending a synthetic user turn to a prefix that already ends in
    one therefore does not add a frame, it extends a turn. That design passed 95% of a
    3000-row dry run, because this template's separator and its turn opener are both one
    token and the two errors cancelled to zero.

    Third, the frame is ids ``BOS INST ... INST_END`` but the *text* render is the
    characters ``[INST]...[/INST]``, and ``tekken`` never promotes text to a control token --
    a deliberate injection guard. So the frame survives rendering and dies on re-encoding,
    and a driver that went through text would train a model on a frame it will never be
    served under.

    Returns a ``BatchEncoding``-shaped mapping rather than a bare list, which is the other
    thing ``apply_chat_template`` does depending on the version.
    """

    BOS, EOS, INST, INST_END = 1, 2, 3, 4
    SEP = 800  # what adjacent same-role contents are joined with
    pad_token = None
    pad_token_id = 0
    eos_token = "</s>"
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False, "message content must not be framed twice"
        return {"input_ids": _enc(text)}

    @staticmethod
    def _merged(messages):
        """Adjacent same-role messages arrive as one turn, contents joined."""
        turns: list[dict] = []
        for message in messages:
            if turns and turns[-1]["role"] == message["role"]:
                turns[-1] = {
                    "role": message["role"],
                    "content": [*turns[-1]["content"], message["content"]],
                }
            else:
                turns.append({"role": message["role"], "content": [message["content"]]})
        return turns

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
    ):
        turns = self._merged(messages)
        if turns and turns[-1]["role"] == "assistant" and not continue_final_message:
            raise ValueError("Expected last role User or Tool ... for serving")
        if not tokenize:
            return "<s>" + "".join(
                f"[INST]{'  '.join(t['content'])}[/INST]"
                if t["role"] == "user"
                else "  ".join(t["content"])
                for t in turns
            )
        ids = [self.BOS]
        for index, turn in enumerate(turns):
            content: list[int] = []
            for part in turn["content"]:
                content += [*([self.SEP] if content else []), *_enc(part)]
            open_turn = continue_final_message and index == len(turns) - 1
            if turn["role"] == "user":
                ids += [self.INST, *content, self.INST_END]
            else:
                ids += content if open_turn else [*content, self.EOS]
        # add_generation_prompt is a no-op: the frame already ends after [/INST].
        return {"input_ids": ids}


def _spans(labels: list[int]) -> list[tuple[int, int]]:
    """The supervised ranges, as (start, end) half-open pairs."""
    spans, start = [], None
    for index, label in enumerate(labels):
        if label != -100 and start is None:
            start = index
        elif label == -100 and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(labels)))
    return spans


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------


def test_only_assistant_content_is_supervised(s2) -> None:
    """The loss covers what the model has to produce and nothing else.

    Turns red when: the mask inverts, widens to the user turns, or stops being applied --
    all of which train a model that is worse and none of which fail.
    """
    tokenizer = _Templated()
    row = s2.mask_conversation(
        tokenizer,
        [{"role": "user", "content": "ab"}, {"role": "assistant", "content": "xy"}],
        max_len=64,
    )

    ids, labels = row["input_ids"], row["labels"]
    assert ids == [
        _Templated.BOS,
        _Templated.USER,
        *_enc("ab"),
        _Templated.END,
        _Templated.ASSISTANT,
        *_enc("xy"),
        _Templated.END,
    ]
    supervised = [ids[index] for index, label in enumerate(labels) if label != -100]
    assert supervised == [*_enc("xy"), _Templated.END]
    assert all(labels[index] in (-100, ids[index]) for index in range(len(ids)))


def test_the_assistant_header_is_not_supervised_but_the_terminator_is(s2) -> None:
    """The two boundary choices, stated as a test because both are arguable.

    The ``<|assistant|>`` header is context: the harness emits it as the generation prompt,
    so the model is never asked to produce it. The turn terminator is the opposite -- it is
    precisely what has to be produced for generation to stop, and a model trained without
    it runs to ``max_new_tokens`` on every prompt.

    Turns red when: the span moves to the header, or stops covering the terminator.
    """
    tokenizer = _Templated()
    row = s2.mask_conversation(
        tokenizer,
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        max_len=64,
    )
    ((start, end),) = _spans(row["labels"])

    assert row["input_ids"][start - 1] == _Templated.ASSISTANT, "header stays context"
    assert row["input_ids"][end - 1] == _Templated.END, "terminator is supervised"


def test_every_assistant_turn_is_supervised_not_just_the_last(s2) -> None:
    """A multi-turn conversation is several training examples sharing a prefix.

    Turns red when: the walk collapses to the final exchange, which silently throws away
    most of the supervision in a mixture whose conversations average more than one turn.
    """
    tokenizer = _Templated()
    conversation = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    row = s2.mask_conversation(tokenizer, conversation, max_len=64)

    ids = row["input_ids"]
    supervised = [ids[index] for index, label in enumerate(row["labels"]) if label != -100]
    assert supervised == [*_enc("b"), _Templated.END, *_enc("d"), _Templated.END]
    assert len(_spans(row["labels"])) == 2


def test_a_conversation_terminator_is_not_mistaken_for_drift(s2) -> None:
    """The 2% of Tulu-3 that Phi-4-mini refuses to call prefix-stable.

    Rendering ``messages[:i+1]`` renders a *finished* conversation, so it ends with a
    sequence terminator that the same turn does not have mid-conversation. Insisting the
    shorter render be a prefix of the longer therefore drops conversations over a token
    that the model will never see in that position -- and here that alone was the
    difference between a 5.67% drop rate and the 5.00% ceiling.

    The final turn is the one place the terminator *is* supervised, because there the
    through-render is the training sequence: it is the end of the sequence, and emitting it
    there is correct.

    Turns red when: the allowance goes (the first span disappears), or stops being applied
    per turn (the last span loses its terminator).
    """
    tokenizer = _Terminating()
    row = s2.mask_conversation(
        tokenizer,
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ],
        max_len=64,
    )

    ids = row["input_ids"]
    first, last = _spans(row["labels"])
    assert ids[first[0] : first[1]] == [*_enc("b"), _Templated.END], "turn terminator only"
    assert ids[last[0] : last[1]] == [*_enc("d"), _Templated.END, _Terminating.EOT]


def test_drift_larger_than_a_terminator_is_still_rejected(s2) -> None:
    """The allowance is an allowance, not a hole.

    A conversation terminator disagrees in the last token or two; a merge spanning the turn
    boundary disagrees from the boundary onwards. This tokenizer's prompt prefix matches
    exactly, so nothing but the bound on the end catches it.

    Turns red when: the bound is widened to "whatever disagrees", which would silently
    accept a template that drifts and supervise a truncated span of every turn.
    """
    with pytest.raises(s2.MaskError, match="not prefix-stable"):
        s2.mask_conversation(
            _LateDrifting(),
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
                {"role": "assistant", "content": "d"},
            ],
            max_len=64,
        )


def test_assemble_mode_reproduces_what_the_template_would_have_rendered(s2) -> None:
    """On a tokenizer both modes can handle, they must agree token for token.

    Assemble hand-builds the assistant turns instead of reading them out of a render, so
    the question it has to answer is whether the sequence it builds is the same sequence.
    If it is not, one of the two panel models trains on a different frame from the other
    and the S4 comparison is across two things at once.

    Turns red when: assembly drops the terminator, doubles the frame, or loses a turn.
    """
    conversation = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    by_template = s2.mask_conversation(_Templated(), conversation, max_len=64, mode="template")
    by_assembly = s2.mask_conversation(_Templated(), conversation, max_len=64, mode="assemble")

    assert by_assembly == by_template


def test_assemble_mode_masks_what_the_template_will_not_render(s2) -> None:
    """``mistral_common`` refuses every prefix the per-turn walk needs.

    It is validating a serving request -- "expected last role User or Tool" -- and an SFT
    conversation ends in an assistant message by definition. On the real tokenizer this
    refused 3000 of 3000 Tulu-3 conversations, which is a whole panel model lost.

    Turns red when: assemble stops handling the refusal, or starts asking for a render the
    validator rejects.
    """
    conversation = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
        {"role": "user", "content": "and"},
        {"role": "assistant", "content": "so"},
    ]
    with pytest.raises(s2.MaskError):
        s2.mask_conversation(_MistralShaped(), conversation, max_len=64, mode="template")

    row = s2.mask_conversation(_MistralShaped(), conversation, max_len=64, mode="assemble")

    ids = row["input_ids"]
    assert ids == [
        _MistralShaped.BOS,
        _MistralShaped.INST,
        *_enc("hi"),
        _MistralShaped.INST_END,
        *_enc("yo"),
        _MistralShaped.EOS,
        _MistralShaped.INST,
        *_enc("and"),
        _MistralShaped.INST_END,
        *_enc("so"),
        _MistralShaped.EOS,
    ]
    supervised = [ids[index] for index, label in enumerate(row["labels"]) if label != -100]
    assert supervised == [*_enc("yo"), _MistralShaped.EOS, *_enc("so"), _MistralShaped.EOS]


def test_the_two_modes_differ_only_by_the_document_terminator(s2) -> None:
    """Where both modes work they must agree, and where they do not, by a known token.

    This is the cross-check that makes assemble mode trustworthy: on a tokenizer the walk
    handles, its answer can be compared against a mode that reads the spans straight out of
    the template's own render. On 495 maskable Tulu-3 rows through the real Phi-4-mini the
    two are token-identical and span-identical, and differ only in that template mode
    carries a trailing ``<|endoftext|>`` -- the document terminator Phi appends to a
    conversation it is not being asked to continue -- and supervises it.

    That difference is kept rather than reconciled. It is what each template says a finished
    conversation *is*: Phi ends the document, ``mistral_common`` has no such token, and
    forcing either into the other's shape would train a model on a frame it is not served
    under. The tie in :func:`probe_mask_modes` goes to template for the same reason.

    Turns red when: the modes disagree about alignment rather than about that one token --
    a span start moving, a turn appearing or vanishing, the prefix diverging.
    """
    conversation = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    by_template = s2.mask_conversation(_Terminating(), conversation, max_len=64, mode="template")
    by_assembly = s2.mask_conversation(_Terminating(), conversation, max_len=64, mode="assemble")

    template_ids, assembly_ids = by_template["input_ids"], by_assembly["input_ids"]
    assert template_ids == [*assembly_ids, _Terminating.EOT]

    template_spans, assembly_spans = _spans(by_template["labels"]), _spans(by_assembly["labels"])
    assert [start for start, _ in template_spans] == [start for start, _ in assembly_spans]
    assert template_spans[:-1] == assembly_spans[:-1]
    assert template_spans[-1][1] == assembly_spans[-1][1] + 1


def test_the_supervised_span_ends_where_the_model_has_to_stop(s2) -> None:
    """The terminator is inside the span, not after it.

    A span that stops at the last content token trains a model that never emits the token
    the harness stops on, so generation runs to the length cap and every answer is scored
    against a trailing hallucination. That failure has already happened once on this
    project, on the eval side, and cost 24 GSM8K points before anyone noticed it was a stop
    condition and not the model.

    The terminator cannot be read off the tokenizer -- ``_Templated`` ends turns with its
    ``eos_token_id``, ``_MistralShaped`` does not have to -- so it is measured, and this is
    the property that measurement exists to deliver.

    Turns red when: the span end loses the terminator, or the terminator measurement starts
    returning the empty sequence (which would report success and supervise content only).
    """
    conversation = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    for tokenizer, terminator in ((_Templated(), _Templated.END), (_MistralShaped(), 2)):
        assert s2._assistant_terminator(tokenizer) == [terminator]

        row = s2.mask_conversation(tokenizer, conversation, max_len=64, mode="assemble")
        ids, labels = row["input_ids"], row["labels"]
        ends = [ids[end - 1] for _, end in _spans(labels)]
        assert ends == [terminator, terminator], type(tokenizer).__name__


def test_a_terminator_that_cannot_be_measured_is_refused_not_guessed(s2) -> None:
    """When the three probe renders do not line up, there is no fallback worth having.

    Every fallback available is a guess that has already been wrong on a real tokenizer:
    ``eos_token_id`` is the wrong token on Phi-4-mini, and an empty terminator silently
    trains a model that cannot stop. So a tokenizer whose terminator cannot be measured
    fails masking, lands in the census, and trips ``--max-drop-rate``.

    Turns red when: a fallback is added, or the measurement stops checking that the open
    render is a prefix of the closed one.
    """

    class _Unmeasurable(_Templated):
        def apply_chat_template(self, messages, *, tokenize=False, **kwargs):
            ids = super().apply_chat_template(messages, tokenize=tokenize, **kwargs)
            if tokenize and kwargs.get("continue_final_message"):
                return [*ids, 907]  # the open turn is not a prefix of the closed one
            return ids

    with pytest.raises(s2.MaskError, match="cannot measure"):
        s2.mask_conversation(
            _Unmeasurable(),
            [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
            max_len=64,
            mode="assemble",
        )


def test_a_wrong_eos_token_does_not_reach_the_supervised_span(s2) -> None:
    """``eos_token_id`` is not the turn terminator, and assembly must not care that it isn't.

    Phi-4-mini is the live counterexample: its turns end in ``<|end|>`` (200020) while its
    ``eos_token_id`` is ``<|endoftext|>`` (199999). An earlier design took the span end to be
    the content plus that token, and would have supervised a span ending one token short on
    every Phi turn. This tokenizer has a deliberately absurd ``eos_token_id`` (99, a token
    it never emits) and must still mask exactly as the honest one does.

    Turns red when: anything in the assemble path starts reading a terminator off the
    tokenizer again instead of measuring the turn from two renders the template produced.
    """
    conversation = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]

    lying = s2.mask_conversation(_WrongEos(), conversation, max_len=64, mode="assemble")
    honest = s2.mask_conversation(_Templated(), conversation, max_len=64, mode="assemble")

    assert lying == honest
    assert 99 not in lying["input_ids"]


def test_the_mode_is_measured_against_the_tokenizer_not_read_off_it(s2) -> None:
    """Nothing on a tokenizer says which mode it needs.

    ``mistral_common``'s refusal is not an attribute, a flag, or a template feature -- it is
    a validator that only fires on the exact shape the walk asks for. So the driver asks by
    trying, the same way the eval harness asks the tokenizer to render a probe conversation
    rather than reading ``chat_prompt_style``. Ties go to ``template``, which assumes
    nothing about where a turn ends.

    Turns red when: the choice reverts to an attribute lookup or a hardcoded per-model map,
    either of which is right until the next tokenizer.
    """
    rows = [
        {
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ]
        }
    ] * 4

    refusing = s2.probe_mask_modes(rows, _MistralShaped(), column="messages", max_len=64)
    # `seam` is 0 for the same reason it exists: it goes through text, and this backend
    # renders its frame as characters it will not read back. It refuses rather than
    # producing an unframed sequence.
    assert refusing == {"template": 0, "assemble": 4, "seam": 0}
    assert s2.choose_mask_mode(refusing) == "assemble"

    fine = s2.probe_mask_modes(rows, _Templated(), column="messages", max_len=64)
    assert fine == {"template": 4, "assemble": 4, "seam": 0}
    assert s2.choose_mask_mode(fine) == "template"


def test_a_template_that_is_not_prefix_stable_is_dropped_not_mis_masked(s2) -> None:
    """One token of drift shifts every label after it, and nothing downstream says so.

    Measuring a span as "the length of the shorter render" is only valid if the shorter
    render is a genuine prefix of the longer. No tokenizer promises that, so it is checked
    token for token, per turn.

    Turns red when: the prefix check is dropped or weakened to comparing lengths -- which
    is the same check for a merge that keeps the count and changes an id, and no check at
    all for one that does not.
    """
    with pytest.raises(s2.MaskError, match="not prefix-stable"):
        s2.mask_conversation(
            _Drifting(),
            [{"role": "user", "content": "ab"}, {"role": "assistant", "content": "xy"}],
            max_len=64,
        )


def test_no_render_asks_for_text(s2) -> None:
    """The training side of the bug that cost S1 two re-runs and 56 points of HumanEval.

    ``MistralCommonBackend`` renders its frame as the *characters* ``[INST]...[/INST]`` and
    ``tekken`` will not read control tokens back out of user text, so a driver that went
    through the text path would train Ministral on a sequence with no BOS and no ``[INST]``
    -- a frame it will never be served under, and damage that would surface at S4 looking
    like quantization loss.

    Asserted as the property rather than as its symptom, and over both walk modes, because
    the text path is a *fallback* in most codebases: it fires on the tokenizers CI does not
    exercise. ``seam`` is the deliberate exception and is asserted below it.

    Turns red when: a ``tokenize=False`` call is reintroduced anywhere in the masking path.
    """
    conversation = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    for mode in ("template", "assemble"):
        tokenizer = _Templated()
        s2.mask_conversation(tokenizer, conversation, max_len=64, mode=mode)
        assert tokenizer.tokenize_flags, f"{mode} rendered nothing"
        assert all(tokenizer.tokenize_flags), f"{mode} asked for text"

    # `seam` is the one mode that does go through text, because a generation prompt ending
    # inside a BPE merge cannot be located any other way. It pays for that with a check
    # rather than an assumption: it re-encodes the render and compares against what
    # `tokenize=True` produced, so on a tokenizer that does not read its own frame back out
    # of text -- which is what this stub is -- it refuses instead of training unframed.
    tokenizer = _Templated()
    with pytest.raises(s2.MaskError, match="round-trip the frame"):
        s2.mask_conversation(tokenizer, conversation, max_len=64, mode="seam")


def test_a_generation_prompt_ending_inside_a_merge_masks_under_seam_alone(s2) -> None:
    """The Qwen3.5 case, and the reason a third mode exists at all.

    Both walk modes ask for the generation prompt's tokens to be a prefix of the turn's.
    This template makes that false without being wrong: it opens a reasoning block ending
    ``<think>\n`` and continues it ``\n</think>``, so its own output holds ``\n\n``
    across the boundary and BPE merges it. The refusals are correct -- the alternative is
    supervising at an offset -- but they refuse *every* row, so the mixture is untrainable
    until something masks it.

    Turns red when: seam stops handling a boundary merge, or a walk mode starts pretending
    it can.
    """
    conversation = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]

    for mode in ("template", "assemble"):
        with pytest.raises(s2.MaskError):
            s2.mask_conversation(_SeamMerging(), conversation, max_len=64, mode=mode)

    masked = s2.mask_conversation(_SeamMerging(), conversation, max_len=64, mode="seam")
    assert sum(1 for label in masked["labels"] if label != -100) > 0


def test_the_seam_is_placed_where_serving_places_it(s2) -> None:
    """The unsupervised half must be, token for token, what the harness will send.

    This is the property that justifies emitting a sequence the tokenizer would not produce
    from its own text. One pair at the seam is split where BPE would have merged it, and
    that split is the *correct* one: at inference the model is handed the generation prompt
    and asked to continue it, so the token it must learn to follow is the prompt's last
    token -- not the merged token that only exists in a render of text the model never
    sees. Training on the canonical merge would train on a prompt token that cannot occur.

    Multi-turn is where the mode's cost is legible, so it is asserted rather than avoided.
    Each seam leaves its split pair behind in the context, so the *second* assistant turn's
    prompt runs one token longer than what serving would tokenize for the same text, and
    the third two longer. What holds at every turn is the part that carries the gradient:
    the token immediately before the supervised span is the one serving will actually hand
    the model. This mixture is single-turn, so the drift is measured here and paid nowhere.

    Turns red when: the seam moves off the generation-prompt boundary, the prompt side
    starts being tokenized as a fragment of the full text rather than on its own, or the
    multi-turn drift grows past one token per seam already passed.
    """
    tokenizer = _SeamMerging()
    conversation = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]

    masked = s2.mask_conversation(tokenizer, conversation, max_len=64, mode="seam")
    ids, labels = masked["input_ids"], masked["labels"]
    starts = [
        i for i in range(len(labels)) if labels[i] != -100 and (i == 0 or labels[i - 1] == -100)
    ]
    assert len(starts) == 2, "one supervised span per assistant turn"

    for seams_passed, (turn, start) in enumerate(zip((1, 3), starts, strict=True)):
        served = tokenizer.apply_chat_template(
            conversation[:turn], tokenize=True, add_generation_prompt=True
        )
        # The load-bearing one, and it holds at every turn: the model learns to continue
        # from the token it will actually be given.
        assert ids[start - 1] == served[-1] == tokenizer.NL, f"turn {turn}: wrong last token"
        if seams_passed == 0:
            assert ids[:start] == served, f"turn {turn}: the prompt is not what serving sends"
        else:
            assert len(ids[:start]) == len(served) + seams_passed, f"turn {turn}: drifted"

    # And the merge really was in play, so this is not a template the walk would have
    # handled anyway: the canonical encoding of the same text is exactly one token shorter
    # per seam.
    canonical = tokenizer.apply_chat_template(conversation, tokenize=True)
    assert len(canonical) == len(ids) - len(starts)

    assert labels[-1] != -100, "the last turn is supervised through its end"


def test_seam_is_the_last_resort_and_not_the_first(s2) -> None:
    """A mode that emits a non-canonical sequence must not win a tie.

    ``seam`` is the only mode whose output the tokenizer would not have produced from its
    own text. That is a real cost, paid deliberately for templates that leave no
    alternative, and it must not be paid for a template that has one. So the probe's tie
    break orders the modes by how much each assumes, and seam is last.

    Turns red when: the tie break reverts to a two-way comparison, which silently promotes
    whichever mode was added most recently.
    """
    rows = [
        {
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ]
        }
    ] * 4

    merging = s2.probe_mask_modes(rows, _SeamMerging(), column="messages", max_len=64)
    assert merging == {"template": 0, "assemble": 0, "seam": 4}
    assert s2.choose_mask_mode(merging) == "seam"

    assert s2.choose_mask_mode({"template": 4, "assemble": 4, "seam": 4}) == "template"
    assert s2.choose_mask_mode({"template": 0, "assemble": 4, "seam": 4}) == "assemble"
    assert s2.choose_mask_mode({"template": 3, "assemble": 3, "seam": 4}) == "seam"


def test_the_warmup_fraction_is_spelled_the_way_the_installed_transformers_spells_it(
    s2,
) -> None:
    """The two spellings differ in the direction that fails silently.

    transformers 5 removed ``warmup_ratio`` and folded it into ``warmup_steps``, which
    reads any value below 1 as a fraction of the run. Passing 0.03 to a 4.x
    ``warmup_steps`` raises nothing and warms up for 0.03 steps -- which is none -- so a
    multi-hour QLoRA run would take its first optimizer step at full learning rate and
    leave no evidence but a loss curve nobody has a reference for. The 5 direction is the
    loud one: ``warmup_ratio`` is a ``TypeError`` there, which is how this was found.

    So the field is chosen by what the class declares, and asserted against a stand-in for
    each generation rather than against whichever one CI happens to have installed.

    Turns red when: the choice moves to a try/except (right for 5, silent on 4), to a
    version string, or to one hardcoded spelling.
    """
    import dataclasses

    @dataclasses.dataclass
    class _Four:
        warmup_steps: int = 0
        warmup_ratio: float = 0.0

    @dataclasses.dataclass
    class _Five:
        warmup_steps: float = 0.0

    assert s2.warmup_kwargs(_Four) == {"warmup_ratio": s2.WARMUP_FRACTION}
    assert s2.warmup_kwargs(_Five) == {"warmup_steps": s2.WARMUP_FRACTION}
    # Both must be constructible with what they were handed -- the point of the exercise.
    assert _Four(**s2.warmup_kwargs(_Four)).warmup_ratio == s2.WARMUP_FRACTION
    assert _Five(**s2.warmup_kwargs(_Five)).warmup_steps == s2.WARMUP_FRACTION
    # And a fraction, not a step count: the campaign's mixtures differ by 5x in size.
    assert 0 < s2.WARMUP_FRACTION < 1


def test_a_transformers_with_neither_spelling_refuses_rather_than_warms_up_from_zero(
    s2,
) -> None:
    """A third rewrite is not something to guess through.

    Warmup is not decoration on a QLoRA run at this scale, and the failure mode of getting
    it silently wrong is a worse first step and no message. If neither field is there, the
    driver has no supported way to ask for a warmup and says so.

    Turns red when: the unknown case starts falling through to a default.
    """
    import dataclasses

    @dataclasses.dataclass
    class _Later:
        schedule: str = "cosine"

    with pytest.raises(RuntimeError, match="neither warmup_ratio nor warmup_steps"):
        s2.warmup_kwargs(_Later)


def test_an_overlong_conversation_is_dropped_not_truncated(s2) -> None:
    """Truncation has no correct end to cut from here.

    The front holds BOS and the system frame; the back holds the assistant content that is
    the entire supervision. A driver that trims either trains on a sequence the model will
    never see, at full cost and without a warning.

    Turns red when: a truncation path appears, or the limit stops being enforced.
    """
    with pytest.raises(s2.MaskError, match="too long"):
        s2.mask_conversation(
            _Templated(),
            [{"role": "user", "content": "a" * 50}, {"role": "assistant", "content": "b"}],
            max_len=16,
        )


def test_a_conversation_with_nothing_to_supervise_is_dropped(s2) -> None:
    """An all-``-100`` row makes the loss a mean over zero elements.

    One of these in a batch is a NaN loss, a step taken into it, and a signal map harvested
    from a model made of NaN -- which is the deliverable, not the checkpoint.

    Turns red when: the guard goes, or the drop moves downstream of the collator where the
    batch is already assembled.
    """
    with pytest.raises(s2.MaskError):
        s2.mask_conversation(_Templated(), [{"role": "user", "content": "a"}], max_len=64)
    with pytest.raises(s2.MaskError):
        s2.mask_conversation(_Templated(), [], max_len=64)


# --------------------------------------------------------------------------
# Reading the mixture
# --------------------------------------------------------------------------


def test_both_conversation_shapes_normalise_to_the_same_turns(s2) -> None:
    """Tulu-3 and SmolTalk ship ``{"role", "content"}``; OpenThoughts3 ships ShareGPT.

    Read off the row rather than declared per dataset, so a re-upload that changes shape
    is a normalisation that still works rather than a column of ``None`` and an empty
    training set.

    Turns red when: either shape stops being recognised, or the speaker map loses an alias.
    """
    expected = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    plain = {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
    sharegpt = {"conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "a"}]}

    assert s2.to_messages(plain, "messages") == expected
    assert s2.to_messages(sharegpt, "conversations") == expected


def test_a_row_that_is_not_a_conversation_is_counted_not_raised(s2) -> None:
    """One malformed row should not end a pass over a million of them.

    Turns red when: the normaliser starts raising, which would make a single bad row cost
    the whole build.
    """
    assert s2.to_messages({"messages": []}, "messages") is None
    assert s2.to_messages({"messages": "not a list"}, "messages") is None
    assert s2.to_messages({"messages": [{"speaker": "x"}]}, "messages") is None
    assert s2.to_messages({}, "messages") is None


def test_the_census_says_how_many_went_and_why(s2) -> None:
    """ "Dropped 12%" and "dropped 12%, all of them for prefix instability" are a shrug and
    a stop-work respectively, so the reason is carried, not just the count.

    Turns red when: the census collapses to a total, or a reason stops being classified
    and every drop lands in its own bucket with a token count in the key.
    """
    rows = [
        {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
        {
            "messages": [
                {"role": "user", "content": "a" * 99},
                {"role": "assistant", "content": "b"},
            ]
        },
        {"messages": "junk"},
    ]
    kept, dropped = s2.build_dataset(rows, _Templated(), column="messages", max_len=32)

    assert len(kept) == 1
    assert dropped["too long"] == 1
    assert dropped["unreadable row"] == 1


def test_a_source_overlapping_an_eval_task_is_named(s2) -> None:
    """Tulu-3 carries ``tulu_v3.9_open_math_2_gsm8k_50k`` and the campaign evaluates GSM8K.

    Kept -- both sides of the quantization comparison train on it -- but the absolute
    number stops being a claim about the benchmark, and that caveat only survives if the
    driver writes it down at the time.

    Turns red when: the check goes, or narrows to an exact repo name that a re-upload
    would rename past.
    """
    rows = [{"source": "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k"}] * 3 + [
        {"source": "ai2-adapt-dev/flan_v2_converted"}
    ] * 7
    counts, flagged = s2.report_sources(rows)

    assert counts["ai2-adapt-dev/flan_v2_converted"] == 7
    assert flagged == ["ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k"]


def test_the_subsample_is_shuffled_before_it_is_limited(s2, monkeypatch) -> None:
    """These splits arrive grouped -- Tulu-3 is ordered by source.

    ``select(range(n))`` on the raw split takes one subset of one source and calls it a
    sample of the mixture. This project has already paid for that once, on a label-sorted
    classification split, where it read as a destroyed model rather than a bad sample.

    Turns red when: the shuffle moves after the limit, or goes.
    """
    order: list[str] = []

    class _Split:
        def __len__(self) -> int:
            return 100

        def shuffle(self, seed: int):
            order.append(f"shuffle({seed})")
            return self

        def select(self, indices):
            order.append(f"select({len(list(indices))})")
            return self

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=lambda repo, name, split: _Split()),
    )
    s2.load_rows({"repo": "r", "split": "train", "column": "messages"}, examples=10, seed=7)

    assert order == ["shuffle(7)", "select(10)"]


def test_every_registered_dataset_names_a_config_the_hub_will_accept(s2, monkeypatch) -> None:
    """SmolTalk ships fourteen configs and no default one.

    ``load_dataset("HuggingFaceTB/smoltalk", split="train")`` raises ``ValueError: Config
    name is missing``, and the driver loads the *model* before it loads the rows -- so the
    arm would have burned a GPU load and a queue slot to discover a one-word registry
    omission. The registry now carries ``name`` where the repo has no default, and this
    pins that the loader actually forwards it.

    Turns red when: a dataset is registered without the config its repo requires, or
    ``load_rows`` stops passing ``name`` through.
    """
    seen: list[tuple[str, str | None]] = []

    class _Split:
        def __len__(self) -> int:
            return 4

        def shuffle(self, seed: int):
            return self

        def select(self, indices):
            return self

    def _load(repo, name, split):
        seen.append((repo, name))
        return _Split()

    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=_load))
    for spec in s2.DATASETS.values():
        # Entries with a `builder` are not one Hub conversation repo and do not take this
        # path -- text2sql mixes three corpora and executes every gold. Their contract is
        # pinned by the test below instead of by widening this one until it means nothing.
        if spec.get("builder"):
            continue
        s2.load_rows(spec, examples=0, seed=0)

    assert seen == [
        ("allenai/tulu-3-sft-mixture", None),
        ("HuggingFaceTB/smoltalk", "all"),
        ("open-thoughts/OpenThoughts3-1.2M", None),
    ]


def test_the_text2sql_entry_names_the_corpora_it_actually_reads(s2) -> None:
    """Its ``repo`` is the *training* Hub ids joined by ``+``, and nothing loads it.

    That string exists to be copied into the run manifest, so it is the only record of
    what an arm was trained on -- and being inert, it cannot fail loudly when the
    mixture changes underneath it. A manifest naming two corpora for a run that read
    three is worse than no manifest: it reads as measured provenance.

    Keyed to ``DEFAULT_TRAIN`` and not to ``SOURCES``, which is the distinction Spider
    introduced. Spider is scored and never trained on, so joining the whole registry here
    would put a fourth corpus in the provenance of a run that never opened it -- and the
    reader who then discounts the Spider column as contaminated would be discounting the
    one column that is clean.

    Turns red when: a source is added to, removed from or renamed in ``DEFAULT_TRAIN`` and
    the registry entry is not updated with it.
    """
    from dynquant.eval.text2sql_sources import DEFAULT_TRAIN, SOURCES

    spec = s2.DATASETS["text2sql"]
    assert spec["builder"] == "text2sql"
    assert spec["repo"] == "+".join(SOURCES[name].repo for name in DEFAULT_TRAIN)


# --------------------------------------------------------------------------
# The run itself
# --------------------------------------------------------------------------


def test_the_run_directory_names_both_the_model_and_the_dataset(s2, tmp_path) -> None:
    """An env-derived run directory has already sent four Mistral arms into a Qwen
    directory in this project, where they were evaluated with a Qwen tokenizer and looked
    measured.

    Turns red when: either half of the identity drops out of the path, which lets two
    cells write over each other.
    """
    first = s2.run_dir(tmp_path, "phi4-mini", "tulu3")
    assert first != s2.run_dir(tmp_path, "ministral-8b", "tulu3")
    assert first != s2.run_dir(tmp_path, "phi4-mini", "smoltalk")
    assert "phi4-mini" in first.name and "tulu3" in first.name


def test_the_panel_is_the_same_one_s1_screened(s2) -> None:
    """Imported, not re-declared. Two lists of four models would agree today and diverge
    the first time one is edited, and the divergence would look like a missing arm.

    Turns red when: the driver grows its own copy of the panel.
    """
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("_dq_s1_check", REPO_ROOT / "scripts" / "run_s1_headroom.py")
    assert spec and spec.loader
    s1 = module_from_spec(spec)
    spec.loader.exec_module(s1)
    assert s2.MODELS is s1.MODELS or s2.MODELS == s1.MODELS


def test_a_gated_model_stops_before_anything_is_loaded(s2, monkeypatch, capsys) -> None:
    """Licence acceptance is per-account on the Hub and cannot be done from the box.

    Failing at the top costs nothing; failing after the mixture is built and tokenized
    costs an hour of CPU to reach the same 401.

    Turns red when: the token check moves below the tokenizer or the dataset load.
    """
    import transformers

    for variable in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(variable, raising=False)

    def _no(*args, **kwargs):
        raise AssertionError("nothing should be loaded for a gated model without a token")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", _no)
    monkeypatch.setattr(s2, "load_rows", _no)

    assert s2.main(["--model", "llama31-8b", "--dry-run"]) == 2
    assert "gated=manual" in capsys.readouterr().out


def test_too_many_unmaskable_conversations_refuses_to_train(
    s2, monkeypatch, tmp_path, capsys
) -> None:
    """A fine-tune quietly trained on 60% of its data looks exactly like one that trained
    on all of it and learned less -- and the second reading is the one a results table
    invites.

    Turns red when: the ceiling stops being enforced, or the refusal degrades to a warning
    that a 30-hour run scrolls past.
    """
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Drifting())
    monkeypatch.setattr(
        s2,
        "load_rows",
        # `(rows, decontaminated)`: the second element is the per-source count of rows
        # dropped for asking a question the evaluation asks, and it is `{}` here because
        # this mixture has no decontamination step -- not because one ran and found none.
        lambda spec, examples, seed, sources=None: (
            [
                {
                    "messages": [
                        {"role": "user", "content": "a"},
                        {"role": "assistant", "content": "b"},
                    ]
                }
            ]
            * 4,
            {},
        ),
    )

    code = s2.main(
        ["--model", "phi4-mini", "--out", str(tmp_path), "--dry-run", "--mask-mode", "template"]
    )

    assert code == 3
    out = capsys.readouterr().out
    assert "REFUSING to train" in out
    assert "--mask-mode assemble" in out, "the refusal names a remedy that works"


def test_auto_mode_rescues_a_tokenizer_the_walk_cannot_handle(s2, monkeypatch, tmp_path) -> None:
    """End to end: the mode the driver picks is the one the run then uses.

    Ministral is half the phase-3 panel and ``template`` mode drops 100% of its data. The
    unit above proves the probe reads the tokenizer correctly; this proves the answer is
    carried into the build and written down, because a choice made and not recorded is one
    nobody can check against the model that came out.

    Turns red when: the probe stops being wired into ``main``, or its answer stops reaching
    the census.
    """
    import json

    import transformers

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _MistralShaped()
    )
    monkeypatch.setattr(
        s2,
        "load_rows",
        lambda spec, examples, seed, sources=None: (
            [
                {
                    "messages": [
                        {"role": "user", "content": "a"},
                        {"role": "assistant", "content": "b"},
                    ]
                }
            ]
            * 3,
            {},
        ),
    )
    monkeypatch.setattr(s2, "_train", lambda *a, **k: pytest.fail("dry run trained"))

    code = s2.main(["--model", "ministral-8b", "--out", str(tmp_path), "--dry-run"])

    assert code == 0
    census = json.loads(
        (s2.run_dir(tmp_path, "ministral-8b", "tulu3") / "mask_census.json").read_text(
            encoding="utf-8"
        )
    )
    assert census["mask_mode"] == "assemble"
    assert census["mask_mode_requested"] == "auto"
    assert census["mask_mode_probe"] == {"template": 0, "assemble": 3, "seam": 0}
    assert census["conversations_kept"] == 3
    assert census["drop_rate"] == 0.0


def test_a_dry_run_writes_the_census_and_touches_no_gpu(s2, monkeypatch, tmp_path) -> None:
    """The census is what makes the ceiling above actionable, and it is worth having on
    disk from the cheap pass rather than only from the expensive one.

    Turns red when: ``--dry-run`` starts training, or stops writing the census.
    """
    import json

    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: _Templated())
    monkeypatch.setattr(
        s2,
        "load_rows",
        lambda spec, examples, seed, sources=None: (
            [
                {
                    "source": "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",
                    "messages": [
                        {"role": "user", "content": "a"},
                        {"role": "assistant", "content": "b"},
                    ],
                }
            ]
            * 3,
            {},
        ),
    )
    monkeypatch.setattr(s2, "_train", lambda *a, **k: pytest.fail("dry run trained"))

    code = s2.main(["--model", "phi4-mini", "--out", str(tmp_path), "--dry-run"])

    assert code == 0
    census = json.loads(
        (s2.run_dir(tmp_path, "phi4-mini", "tulu3") / "mask_census.json").read_text(
            encoding="utf-8"
        )
    )
    assert census["conversations_kept"] == 3
    assert census["drop_rate"] == 0.0
    assert census["supervised_tokens"] > 0
    assert census["mask_mode"] == "template", "ties go to the mode that assumes least"
    assert census["sources_overlapping_an_eval_task"] == [
        "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k"
    ]
    # And what that list was checked against, because an empty one is worth exactly as
    # much as the markers behind it. On the text-to-SQL mixture it was worth nothing: no
    # SQL corpus name contains "gsm8k", "humaneval" or "mbpp", so the empty result was a
    # check that could not fire rather than a mixture that passed.
    assert census["contamination_markers"] == list(s2._CONTAMINATING)
    # The check that can fire, and it reports per source rather than as a boolean. Empty
    # here because this mixture takes the generic Hub path, which has no filter to run.
    assert census["decontaminated"] == {}


def test_the_signal_map_is_counted_from_the_file_not_from_the_tracker(s2, tmp_path) -> None:
    """The deliverable is a file S3 will open, so the check has to read that file.

    ``StatsFile.save`` chooses between "directory" and "file path" with ``Path.is_dir()`` --
    a filesystem test rather than a reading of its argument -- so the callback writes
    ``stats/dynquant_stats.json`` when that directory exists and a file literally named
    ``stats`` when it does not. Either is loadable; what is not acceptable is a run whose
    output path depends on what a previous run happened to leave behind, or a record that
    names a path the map is not at. A tracker reporting N modules proves the hooks fired; it
    does not prove S3 can find them.

    Turns red when: the counter starts trusting the in-memory tracker, or stops treating an
    unreadable or wrongly-shaped stats file as a failure.
    """
    import json

    from dynquant.constants import STATS_FILENAME

    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    good = stats_dir / STATS_FILENAME
    good.write_text(json.dumps({"layers": {"a": {}, "b": {}, "c": {}}}), encoding="utf-8")
    assert s2._modules_in(good) == 3

    assert s2._modules_in(tmp_path / "absent.json") == -1, "a missing map is not zero modules"
    truncated = tmp_path / "truncated.json"
    truncated.write_text('{"layers": {"a"', encoding="utf-8")
    assert s2._modules_in(truncated) == -1
    wrong_shape = tmp_path / "wrong.json"
    wrong_shape.write_text(json.dumps({"layers": []}), encoding="utf-8")
    assert s2._modules_in(wrong_shape) == -1


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


class _Mergeable:
    """A PEFT-shaped stub. ``is_peft_model`` duck-types on ``peft_config``, so this
    reaches the merge branch without ``peft`` installed."""

    def __init__(self, log: list[tuple[str, str]]) -> None:
        self.peft_config = {"default": object()}
        self.config = types.SimpleNamespace(use_cache=False)
        self._log = log
        self.merged = False

    def save_pretrained(self, path: str) -> None:
        self._log.append(("adapter" if not self.merged else "merged-as-peft", path))

    def merge_and_unload(self, safe_merge: bool = True) -> _Merged:
        self.merged = True
        return _Merged(self._log)


class _Merged:
    def __init__(self, log: list[tuple[str, str]]) -> None:
        self.config = types.SimpleNamespace(use_cache=False)
        self._log = log

    def save_pretrained(self, path: str) -> None:
        self._log.append(("model", path))


class _Tok:
    def __init__(self, log: list[tuple[str, str]]) -> None:
        self._log = log

    def save_pretrained(self, path: str) -> None:
        self._log.append(("tokenizer", path))


def test_the_adapter_is_saved_before_the_merge_consumes_it(s2, tmp_path) -> None:
    """Both artifacts are written, and the adapter is written *first*.

    ``merge_adapters`` calls PEFT's ``merge_and_unload``, which folds the deltas into the
    base weights and detaches the adapter in place. So the ordering is not a preference:
    after that call there is no adapter left to save, and the same line moved two
    statements down would write a second copy of the merged model under a directory named
    ``adapter/`` -- 7.2 GB of the wrong thing, under a name that says the right thing.
    That is the failure this pins, and it is invisible in a run log.

    Why save it at all, when ``merged/`` is what every downstream stage opens: it is a few
    percent of the size and reconstructs the merge exactly, and these arms run where
    ``/workspace`` is not a volume.

    Turns red when: the adapter save is dropped, moved after the merge, or starts being
    written *instead* of the merged model -- which would have S4 score the base weights.
    """
    log: list[tuple[str, str]] = []
    model = _Mergeable(log)

    merged = s2.save_outputs(model, _Tok(log), tmp_path, regime="lora")

    assert merged == tmp_path / "merged"
    assert log == [
        ("adapter", str(tmp_path / "adapter")),
        ("model", str(tmp_path / "merged")),
        ("tokenizer", str(tmp_path / "merged")),
    ], "the adapter must be written before merge_and_unload detaches it"
    assert model.merged, "the deliverable must be merged, not an adapter"


def test_a_full_finetune_writes_no_adapter_directory(s2, tmp_path) -> None:
    """There is no adapter to write when nothing was adapted.

    Turns red when: the adapter save stops being conditional on the regime and an empty
    or bogus ``adapter/`` appears beside a full fine-tune's merge.
    """
    log: list[tuple[str, str]] = []
    model = _Merged(log)

    merged = s2.save_outputs(model, _Tok(log), tmp_path, regime="full")

    assert merged == tmp_path / "merged"
    assert [kind for kind, _ in log] == ["model", "tokenizer"]
    assert model.config.use_cache is True, "the saved model has to be generation-ready"


# --------------------------------------------------------------------------
# Collation
# --------------------------------------------------------------------------


def test_padding_keeps_labels_aligned_and_unsupervised(s2) -> None:
    """Right padding, and the pad positions masked out of the loss on both sides.

    A pad token left at its own id in ``labels`` is supervision to emit padding; an
    attention mask that covers the pad lets it into the attention pattern. Both are
    silent.

    Turns red when: the padding side flips, or the pad stops being ``-100`` in the labels.
    """
    batch = s2.collate(
        [{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [8], "labels": [8]}],
        pad_id=0,
    )

    assert batch["input_ids"].tolist() == [[5, 6, 7], [8, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 6, 7], [8, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


# --- choosing whether to measure the expert mass --------------------------------------


class _Experts(torch.nn.Module):
    """A batched bank, shaped the way every MoE family on transformers 5.x stores one.

    The class name matters: ``is_expert_container`` tests the ``Experts`` suffix *and* the
    presence of 3-D parameters, so a stand-in named anything else is not a bank.
    """

    def __init__(self, experts: int = 4, hidden: int = 8, inter: int = 16) -> None:
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.zeros(experts, 2 * inter, hidden))
        self.down_proj = torch.nn.Parameter(torch.zeros(experts, hidden, inter))


class _Sparse(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = _Experts()


def test_a_dense_model_needs_no_decision(s2) -> None:
    """The flag must cost the existing panel nothing.

    Four dense models have already been run through this driver. If an unset flag raised
    for them too, every one of those commands would have to be rewritten to say "no" to a
    question their architecture never poses.

    Turns red when: the no-banks case starts demanding the flag, or stops defaulting off.
    """
    dense = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))

    assert s2.resolve_bank_measurement(dense, None) is False


def test_an_moe_with_no_flag_refuses_to_start(s2) -> None:
    """The failure this whole tri-state exists to prevent.

    Defaulting off is what produced the 11.6%-coverage run: it completed, saved a model,
    and wrote a stats file in which 88.4% of the checkpoint was UNMEASURED and allocated
    on role floors. That run looked exactly like a successful one. Defaulting *on* is not
    safe either -- it silently commits a gradient buffer for the entire expert mass. So
    the driver refuses, before the first step rather than in a warning after it.

    Turns red when: the unset case starts picking either answer on an MoE.
    """
    with pytest.raises(SystemExit) as caught:
        s2.resolve_bank_measurement(_Sparse(), None)

    message = str(caught.value)
    # The count, so the message says how much is at stake rather than that something is.
    assert f"{4 * 2 * 16 * 8 + 4 * 8 * 16:,}" in message
    assert "--measure-expert-banks" in message


def test_an_explicit_answer_is_never_second_guessed(s2) -> None:
    """Both explicit answers pass through, including on a model that has banks.

    ``--no-measure-expert-banks`` is a real choice -- it is the only way to run this model
    on a card that cannot hold the extra gradient buffer -- so the refusal must key on the
    flag being *absent*, not on the model having banks.

    Turns red when: the guard keys on the architecture instead of on the flag, which would
    make the opt-out unreachable.
    """
    banked = _Sparse()

    assert s2.resolve_bank_measurement(banked, True) is True
    assert s2.resolve_bank_measurement(banked, False) is False


# --- where the weights come from -------------------------------------------------------
#
# The LFM2.5-8B-A1B fine-tune spent twenty-two minutes re-fetching a 16.9 GB checkpoint
# that was already complete on the same disk, then stalled with the download silent for
# seventeen of them. The driver had no way to be told about the local copy; these cover the
# flag that gives it one.


def test_the_default_source_is_the_registry_repo(s2) -> None:
    """No flag, no change: the Hub id is still what gets loaded.

    Turns red when: the resolver starts reading a path, a cache directory or an environment
    variable when it was not asked to.
    """
    args = argparse.Namespace(model_path=None)
    assert s2.resolve_model_source(args, {"repo": "LiquidAI/LFM2.5-8B-A1B"}) == (
        "LiquidAI/LFM2.5-8B-A1B"
    )


def test_a_model_path_without_a_config_is_refused(s2, tmp_path) -> None:
    """The plausible wrong path, which is the only kind anyone actually passes.

    A run directory, its parent, a merge not yet written: each exists, and each would send
    `from_pretrained` either a hundred lines deep into a stack trace or -- worse -- back to
    the Hub for the name it was given, which is the fetch this flag exists to prevent.

    Turns red when: the check degrades to `is_dir()`, or to nothing at all.
    """
    empty = tmp_path / "not-a-checkpoint"
    empty.mkdir()
    args = argparse.Namespace(model_path=str(empty))

    with pytest.raises(SystemExit, match=r"no config.json"):
        s2.resolve_model_source(args, {"repo": "LiquidAI/LFM2.5-8B-A1B"})


def test_a_model_path_replaces_the_repo_and_is_absolute(s2, tmp_path, monkeypatch) -> None:
    """Resolved, not passed through -- the run is launched from a directory that changes.

    Turns red when: the path is handed on relative, or appended to the repo id rather than
    replacing it.
    """
    checkpoint = tmp_path / "models" / "LFM2.5-8B-A1B"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(model_path="models/LFM2.5-8B-A1B")

    resolved = s2.resolve_model_source(args, {"repo": "LiquidAI/LFM2.5-8B-A1B"})

    assert Path(resolved) == checkpoint.resolve()
    assert Path(resolved).is_absolute()


# --- the expert banks actually reaching the stats file ---------------------------------
#
# `stats_modules == tracked` proves the file holds what the tracker saw. On an MoE that is
# not the same question as whether the *mass* was measured: LFM2.5-8B-A1B keeps 91.5% of
# its parameters in 22 batched banks, so a run that hooked none of them still writes every
# attention and dense entry, still matches its own tracker exactly, and still leaves the
# allocator to set widths for nine tenths of the checkpoint from role floors alone.
#
# Six hours of fine-tune stand between the flag and the answer, so the gate has to be able
# to say the mass is missing without being told which names to expect.


class _Banked(torch.nn.Module):
    """:class:`_Sparse` with a dense module beside the bank, so "all" and "the banks" differ."""

    def __init__(self) -> None:
        super().__init__()
        self.experts = _Experts()
        self.attn = torch.nn.Linear(8, 8)


def _stats(tmp_path, layers: dict[str, dict]) -> Path:
    path = tmp_path / "dynquant_stats.json"
    path.write_text(json.dumps({"version": 2, "layers": layers}), encoding="utf-8")
    return path


def test_a_bank_absent_from_the_stats_file_is_named(s2, tmp_path) -> None:
    """The failure the module-count check cannot see, and the exact tensors that are gone.

    Named rather than counted because the two readings send a person to different places: a
    whole bank missing is a flag that did not take effect, one tensor of one bank missing is
    a hook that raised on a shape. The count is identical in both.

    Turns red when: the gate starts comparing totals, or reports a boolean.
    """
    model = _Banked()
    complete = _stats(
        tmp_path,
        {"experts.gate_up_proj": {}, "experts.down_proj": {}, "attn": {}},
    )
    assert s2.banked_entries_missing(model, complete) == []

    partial = tmp_path / "partial.json"
    partial.write_text(
        json.dumps({"version": 2, "layers": {"experts.gate_up_proj": {}, "attn": {}}}),
        encoding="utf-8",
    )
    assert s2.banked_entries_missing(model, partial) == ["experts.down_proj"]


def test_the_expected_keys_are_the_ones_the_allocator_looks_up(s2, tmp_path) -> None:
    """Rebuilt from the model through `canonical_name`, not spelled out a second time.

    A gate that invents its own naming passes for the wrong reason -- it checks a key nobody
    writes and nobody reads. This asserts the keys it demands are the ones
    ``classify_model`` produces for the same tensors, so the two cannot drift apart while
    both look right.

    The PEFT prefix is the case that would actually break it: under LoRA the modules are
    named ``base_model.model.model...``, and a gate keyed on the raw module name would call
    every bank missing on exactly the runs this campaign does.

    Turns red when: the gate stops canonicalising, or joins the parameter with anything but
    a dot.
    """
    from dynquant.graph.classify import classify_model

    model = _Banked()
    # A config, because the graph needs one to decide which axis of a bank is the input
    # and refuses the tensor when it cannot -- the gate needs no such thing, which is why
    # the two have to be checked against each other rather than assumed equal.
    config = types.SimpleNamespace(hidden_size=8, moe_intermediate_size=16)
    graph = classify_model(model, config=config)
    through_the_graph = {name for name in graph.modules if name.startswith("experts.")}
    assert through_the_graph == {"experts.gate_up_proj", "experts.down_proj"}

    empty = _stats(tmp_path, {})
    assert set(s2.banked_entries_missing(model, empty)) == through_the_graph

    # And under the wrapper the campaign actually trains through. PEFT names the same bank
    # `base_model.model.model...`; the stats file does not, because the tracker canonicalises
    # before writing. A gate keyed on the raw name would report every bank missing on every
    # LoRA run -- which is all of them.
    wrapped = torch.nn.Module()
    wrapped.base_model = torch.nn.Module()
    wrapped.base_model.model = torch.nn.Module()
    wrapped.base_model.model.model = model
    assert s2.banked_entries_missing(wrapped, empty) == [
        "model.experts.gate_up_proj",
        "model.experts.down_proj",
    ]


def test_an_unreadable_stats_file_is_a_failure_not_a_pass(s2, tmp_path) -> None:
    """No file, no JSON, no ``layers`` key -- none of those mean the banks are present.

    An exception swallowed into an empty list here would report a clean gate on a run whose
    deliverable does not exist, which is the one outcome worse than the gate firing.

    Turns red when: the reader returns ``[]`` on a missing or malformed file.
    """
    assert s2.banked_entries_missing(_Banked(), tmp_path / "absent.json") != []

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert s2.banked_entries_missing(_Banked(), broken) != []

    no_layers = tmp_path / "no_layers.json"
    no_layers.write_text(json.dumps({"version": 2}), encoding="utf-8")
    assert s2.banked_entries_missing(_Banked(), no_layers) != []


def test_a_dense_model_has_nothing_to_miss(s2, tmp_path) -> None:
    """The four dense panel models run this gate too, and it must never fire for them.

    Turns red when: the gate starts demanding an entry for every module rather than for
    every batched expert tensor.
    """
    assert s2.banked_entries_missing(torch.nn.Linear(8, 8), _stats(tmp_path, {})) == []


# --------------------------------------------------------------------------
# QLoRA and torchrun
# --------------------------------------------------------------------------


def test_the_qlora_merge_reloads_the_base_instead_of_folding_into_nf4(s2, tmp_path) -> None:
    """The merge under ``--load-4bit`` must not go through the quantized weights.

    ``merge_and_unload`` on a bitsandbytes base dequantizes, adds ``BA``, and requantizes
    to NF4 -- so the "full precision" model handed to S3 would already be 4-bit, and every
    number S4 reported would be NF4-plus-DynQuant charged to DynQuant alone. The test
    watches which object gets merged: the one reloaded from the base repo, never the
    trained one still holding ``Linear4bit`` modules.

    Turns red when: ``save_outputs`` merges in place while a full-precision base is named.
    """
    reloaded: list[str] = []
    merged_from: list[object] = []

    class _Wrapped:
        def merge_and_unload(self, **kwargs: object) -> object:
            assert kwargs.get("safe_merge") is True
            return _Saved("reloaded-base")

    class _Peft:
        @staticmethod
        def from_pretrained(model: object, adapter: str, **_: object) -> _Wrapped:
            merged_from.append(model)
            return _Wrapped()

    class _AutoModel:
        @staticmethod
        def from_pretrained(repo: str, **_: object) -> str:
            reloaded.append(repo)
            return "bf16-base"

    peft = types.ModuleType("peft")
    peft.PeftModel = _Peft  # type: ignore[attr-defined]
    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = _AutoModel  # type: ignore[attr-defined]

    saved = _run_save_outputs(
        s2,
        tmp_path,
        full_precision_base="Qwen/Qwen3.8-27B",
        modules={"peft": peft, "transformers": transformers},
    )

    assert reloaded == ["Qwen/Qwen3.8-27B"], "the bf16 base was never reloaded"
    assert merged_from == ["bf16-base"], "the adapter was folded into the trained NF4 model"
    assert (saved / "config.json").read_text("utf-8") == "reloaded-base"


def test_a_bf16_run_still_merges_in_place(s2, tmp_path) -> None:
    """Without ``--load-4bit`` there is nothing to reload -- the base already is bf16.

    Turns red when: the QLoRA path becomes unconditional and every bf16 arm pays a
    46 GiB host reload it does not need.
    """
    saved = _run_save_outputs(s2, tmp_path, full_precision_base=None, modules={})
    assert (saved / "config.json").read_text("utf-8") == "merged-in-place"


class _Saved:
    """A stand-in for whatever ``save_outputs`` ends up writing.

    ``filename`` differs between the model and the tokenizer because both are saved into
    ``merged/`` and the tokenizer is saved second: sharing a filename would have the
    tokenizer overwrite the marker the assertion reads, and the test would report the
    wrong merge path rather than a failure.
    """

    def __init__(self, marker: str, filename: str = "config.json") -> None:
        self.marker = marker
        self.filename = filename
        self.config = types.SimpleNamespace(use_cache=False)

    def save_pretrained(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / self.filename).write_text(self.marker, encoding="utf-8")


def _run_save_outputs(s2, tmp_path: Path, *, full_precision_base: str | None, modules: dict):
    """Drive ``save_outputs`` in the lora regime with both merge paths stubbed."""
    model = _Saved("adapter")
    peft_utils = types.ModuleType("dynquant.integration.peft_utils")
    peft_utils.merge_adapters = lambda _model: _Saved("merged-in-place")  # type: ignore[attr-defined]

    stubs = {"dynquant.integration.peft_utils": peft_utils, **modules}
    saved_modules = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        return s2.save_outputs(
            model,
            _Saved("tokenizer", filename="tokenizer_config.json"),
            tmp_path,
            regime="lora",
            full_precision_base=full_precision_base,
        )
    finally:
        for name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_the_replica_lands_on_the_rank_that_owns_it(s2) -> None:
    """``device_map`` under torchrun has to name *this* rank's card, not device 0.

    ``"cuda"`` resolves to device 0 in every process, so a two-rank job stacks two
    replicas on GPU 0 and leaves GPU 1 idle. On a 27B that is an OOM; on a small model it
    is a run that reports two ranks and uses one card, which is worse because it finishes.

    Turns red when: the LOCAL_RANK branch is dropped and every rank asks for "cuda".
    """
    source = DRIVER.read_text(encoding="utf-8")
    assert 'device_map = "cuda" if LOCAL_RANK < 0 else {"": LOCAL_RANK}' in source
    assert 'device_map="cuda"' not in source


def test_only_rank_zero_writes_the_runs_artifacts(s2) -> None:
    """One set of files per run, not one per rank.

    Both writes are guarded because both are unrecoverable: interleaved shard writes from
    two ranks calling ``save_pretrained`` on one directory cost the fine-tune, and a
    truncated ``mask_census.json`` costs the record of what the run trained on.

    Turns red when: either guard is removed, or the tail returns before the ranks have
    finished the collective inside ``trainer.train()``.
    """
    source = DRIVER.read_text(encoding="utf-8")
    assert "if LOCAL_RANK > 0:" in source
    assert "if LOCAL_RANK <= 0:" in source
    tail = source.index("if LOCAL_RANK > 0:")
    assert source.index("trainer.train()") < tail, "ranks must not skip the signal all-reduce"
    assert tail < source.index("merged = save_outputs("), "rank 0 alone writes the merge"


def test_qlora_without_an_adapter_is_refused(s2) -> None:
    """NF4 base weights are frozen, so rank 0 would train nothing and still log a loss.

    Turns red when: ``--load-4bit`` stops requiring ``--lora-rank``.
    """
    source = DRIVER.read_text(encoding="utf-8")
    assert 'if args.load_4bit and regime != "lora":' in source


def test_the_qlora_load_does_not_upcast_the_model(s2) -> None:
    """``prepare_model_for_kbit_training`` is deliberately absent, and has to stay absent.

    It upcasts every non-quantized parameter to fp32, which on this model puts a 248k x
    5120 embedding and an untied head into fp32 for nothing -- peft creates the LoRA
    parameters in fp32 either way. It also turns on gradient checkpointing as a side
    effect, which is a decision this driver makes per run rather than inherits from a
    helper: see the test below.

    Turns red when: someone adds the call because a tutorial has it.
    """
    source = DRIVER.read_text(encoding="utf-8")
    assert "prepare_model_for_kbit_training" not in source.replace(
        "`prepare_model_for_kbit_training`", ""
    )


def test_gradient_checkpointing_is_a_flag_that_defaults_to_off(s2) -> None:
    """It used to be hardcoded off, for a reason that has since been fixed a layer down.

    Checkpointing replays a block's forward during backward and module forward hooks fire
    on the replay, so the saliency EMA counted each micro-batch twice. That is now handled
    where it belongs -- ``signals.tracker._in_backward`` drops a forward that fires inside
    a backward -- so the driver no longer has to refuse the memory saving to protect the
    signal. It is not optional on a 27B, where storing 64 layers of activations at 3072
    tokens needs 93 GiB and a 96 GiB card OOMs on the forward.

    Off by default all the same: every arm run so far trained without it, and the recompute
    costs step time on models that have the memory to spare.

    Turns red when: the default flips silently, or the flag stops reaching
    ``TrainingArguments`` -- which would look exactly like a card that got smaller.
    """
    parser = s2.build_parser()
    base = ["--model", "qwen38-27b", "--out", "runs"]
    assert parser.parse_args(base).gradient_checkpointing is False
    assert parser.parse_args([*base, "--gradient-checkpointing"]).gradient_checkpointing is True

    source = DRIVER.read_text(encoding="utf-8")
    assert "gradient_checkpointing=args.gradient_checkpointing" in source
    # Non-reentrant is load-bearing under QLoRA rather than stylistic: the reentrant
    # implementation recovers the graph from a block's inputs, and under QLoRA every
    # block's inputs are frozen, so the run would train nothing while reporting a loss.
    assert '"use_reentrant": False' in source


def test_naming_spider_trains_on_it_and_still_scores_the_held_out_split(s2, monkeypatch) -> None:
    """``--train-sources`` reaches ``load_text2sql``'s ``sources``, and only that.

    The split it asks for stays ``"train"``. Spider's train and validation splits use
    disjoint databases by construction -- that is what the benchmark is -- so training on
    one and scoring on the other is the standard protocol; passing the *split* through
    would be the leak this looks like.

    Turns red when: the argument is dropped on the floor, which would train the campaign
    default while the manifest recorded the requested list, or when it starts changing
    the split.
    """
    seen: dict[str, object] = {}

    def _load(split, *, sources=None, limit=None, seed=0, tallies=None, **kwargs):
        seen.update(split=split, sources=sources, limit=limit)
        return []

    module = types.ModuleType("dynquant.eval.text2sql")
    module.load_text2sql = _load  # type: ignore[attr-defined]
    module.instruction = lambda item: "q"  # type: ignore[attr-defined]
    sources_mod = types.ModuleType("dynquant.eval.text2sql_sources")
    sources_mod.SourceTally = type("SourceTally", (), {"contaminated": 0})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dynquant.eval.text2sql", module)
    monkeypatch.setitem(sys.modules, "dynquant.eval.text2sql_sources", sources_mod)

    s2.load_text2sql_rows(examples=10_000, seed=0, sources=["spider", "gretel", "wikisql"])
    assert seen == {"split": "train", "sources": ["spider", "gretel", "wikisql"], "limit": 10_000}

    s2.load_text2sql_rows(examples=10_000, seed=0)
    assert seen["sources"] is None, "the default must stay the registry's, not a copy of it"


def test_the_default_training_mixture_still_excludes_spider(s2) -> None:
    """The knob is opt-in. Acquiring Spider silently would report in-domain accuracy
    under the name of the held-out number, and nothing downstream could tell.

    Turns red when: ``DEFAULT_TRAIN`` grows spider, which would change what every
    previously published arm in this campaign trained on without changing its manifest.
    """
    from dynquant.eval.text2sql_sources import DEFAULT_TRAIN

    assert DEFAULT_TRAIN == ("gretel", "wikisql", "create-context")
    assert "spider" not in DEFAULT_TRAIN
