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

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "scripts" / "run_s2_finetune.py"


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
    assert refusing == {"template": 0, "assemble": 4}
    assert s2.choose_mask_mode(refusing) == "assemble"

    fine = s2.probe_mask_modes(rows, _Templated(), column="messages", max_len=64)
    assert fine == {"template": 4, "assemble": 4}
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

    Asserted as the property rather than as its symptom, and over both modes, because the
    text path is a *fallback* in most codebases: it fires on the tokenizers CI does not
    exercise.

    Turns red when: a ``tokenize=False`` call is reintroduced anywhere in the masking path.
    """
    conversation = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    for mode in s2.MASK_MODES:
        tokenizer = _Templated()
        s2.mask_conversation(tokenizer, conversation, max_len=64, mode=mode)
        assert tokenizer.tokenize_flags, f"{mode} rendered nothing"
        assert all(tokenizer.tokenize_flags), f"{mode} asked for text"


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
        types.SimpleNamespace(load_dataset=lambda repo, split: _Split()),
    )
    s2.load_rows({"repo": "r", "split": "train", "column": "messages"}, examples=10, seed=7)

    assert order == ["shuffle(7)", "select(10)"]


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
        lambda spec, examples, seed: (
            [
                {
                    "messages": [
                        {"role": "user", "content": "a"},
                        {"role": "assistant", "content": "b"},
                    ]
                }
            ]
            * 4
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
        lambda spec, examples, seed: (
            [
                {
                    "messages": [
                        {"role": "user", "content": "a"},
                        {"role": "assistant", "content": "b"},
                    ]
                }
            ]
            * 3
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
    assert census["mask_mode_probe"] == {"template": 0, "assemble": 3}
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
        lambda spec, examples, seed: (
            [
                {
                    "source": "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k",
                    "messages": [
                        {"role": "user", "content": "a"},
                        {"role": "assistant", "content": "b"},
                    ],
                }
            ]
            * 3
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
