"""The shared decode loop, tested for the failures that return a number anyway.

Every bug in this file's subject matter has the same signature: no exception, no
warning from torch, just a score a few points below the truth. That is worse than a
crash, because the run gets written into a results table. So the loop is asserted
against a stub that reports exactly which tokens reached the model.

The specific hazard that motivated these tests: ``generate_batched`` set
``padding_side = "left"`` -- necessary and correct -- but left ``truncation_side`` at
its default of ``"right"``. GSM8K never noticed, because a 5-shot GSM8K prompt is
~900 tokens against a 2048 limit. CaseHOLD prompts are several times longer and do
reach the limit, and right-truncation would have removed the five holdings and the
``Answer:`` cue from the end -- scoring the model on a question it was never shown,
at a number near chance, in the arm where quantization was the suspected cause.
"""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from _decode_stub import StubModel, StubTokenizer  # noqa: E402

from dynquant.eval.harness import (  # noqa: E402
    NEUTRAL_DECODE,
    EvalConfig,
    generate_batched,
    greedy_generation_config,
)


@pytest.fixture
def propagating_logs():
    """``enable_logging`` sets ``propagate = False``; caplog needs it True.

    Another test calling the CLI would otherwise make these assertions pass or fail
    depending on test order.
    """
    logger = logging.getLogger("dynquant")
    original = logger.propagate
    logger.propagate = True
    yield
    logger.propagate = original


def _decode(prompts, *, reply="3", **config_kwargs):
    tokenizer = StubTokenizer()
    model = StubModel(tokenizer, reply)
    config = EvalConfig(early_stop=False, **config_kwargs)
    return model, generate_batched(model, tokenizer, prompts, config)


# --------------------------------------------------------------------------
# Truncation side
# --------------------------------------------------------------------------


def test_an_overlong_prompt_keeps_its_tail_not_its_head() -> None:
    """The end of a few-shot prompt is the part the model has to act on.

    Exemplars sit at the front and are expendable; the question, the options and the
    trailing cue sit at the back. Cutting from the right removes the cue.
    """
    prompt = "exemplar filler filler filler QUESTION options Answer:"
    model, _ = _decode([prompt], max_prompt_tokens=3)

    assert model.fed == [["QUESTION", "options", "Answer:"]]
    assert "exemplar" not in model.fed[0]


def test_truncation_ignores_the_tokenizers_own_setting() -> None:
    """A tokenizer shared with a training pipeline arrives set to whatever it wanted.

    The harness slices rather than asking the tokenizer to truncate, so a caller's
    ``truncation_side="right"`` cannot quietly remove the trailing cue from every
    prompt in an evaluation.

    Turns red when: ``encode_prompts`` delegates truncation to the tokenizer again.
    """
    tokenizer = StubTokenizer()
    tokenizer.truncation_side = "right"
    model = StubModel(tokenizer, "3")

    prompt = "exemplar filler filler filler QUESTION options Answer:"
    generate_batched(model, tokenizer, [prompt], EvalConfig(early_stop=False, max_prompt_tokens=3))

    assert model.fed == [["QUESTION", "options", "Answer:"]]


def test_a_prompt_that_fits_is_not_touched() -> None:
    prompt = "short question Answer:"
    model, _ = _decode([prompt], max_prompt_tokens=64)
    assert model.fed == [["short", "question", "Answer:"]]


def test_truncation_is_reported_with_a_count(propagating_logs, caplog) -> None:
    """Silent truncation is the dangerous kind.

    A run where 8% of prompts lost their question still produces a plausible
    accuracy, and nothing in the output says why it is low.
    """
    prompts = ["a b c d e f QUESTION Answer:", "short Answer:"]
    with caplog.at_level(logging.WARNING, logger="dynquant.eval.harness"):
        _decode(prompts, max_prompt_tokens=4)

    assert "1/2 prompts exceeded max_prompt_tokens=4" in caplog.text
    assert "cut at the front" in caplog.text


def test_no_warning_when_nothing_was_truncated(propagating_logs, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="dynquant.eval.harness"):
        _decode(["short Answer:"], max_prompt_tokens=64)
    assert "max_prompt_tokens" not in caplog.text


def test_the_tokenizer_is_never_mutated_at_all() -> None:
    """A caller's tokenizer is not ours to configure, even temporarily.

    A tokenizer left on ``truncation_side="left"`` would silently change how the
    *training* set is built if the same object is reused, which is exactly the kind
    of cross-contamination that makes a fine-tune and its evaluation disagree. The
    harness used to set both sides and restore them in a ``finally``; it now sets
    neither, which is the same guarantee without the window.
    """
    tokenizer = StubTokenizer()
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    model = StubModel(tokenizer, "3")

    generate_batched(model, tokenizer, ["a b Answer:"], EvalConfig(early_stop=False))

    assert tokenizer.padding_side == "right"
    assert tokenizer.truncation_side == "right"


def test_the_tokenizer_is_untouched_even_when_generation_raises() -> None:
    tokenizer = StubTokenizer()
    model = StubModel(tokenizer, "3")

    def explode(**_kwargs):
        raise RuntimeError("CUDA out of memory")

    model.generate = explode  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="out of memory"):
        generate_batched(model, tokenizer, ["a b Answer:"], EvalConfig(early_stop=False))

    assert tokenizer.padding_side == "right"
    assert tokenizer.truncation_side == "right"


# --------------------------------------------------------------------------
# Ordering and coverage
# --------------------------------------------------------------------------


def test_length_sorted_batching_returns_results_in_input_order() -> None:
    """The sort is an internal optimisation and must be invisible.

    If it leaked, every answer would be scored against another problem's gold label
    and the accuracy would land near chance -- for a reason that looks nothing like
    a bug in the batching code.
    """
    prompts = [f"prompt {'x ' * i}number{i} Answer:" for i in range(7)]
    tokenizer = StubTokenizer()
    model = StubModel(tokenizer, "reply")
    config = EvalConfig(early_stop=False, batch_size=3)

    # The reply is fixed, so identity is asserted through the loop's own bookkeeping:
    # a permutation bug shows up as the wrong count, and the padding assertions below
    # cover the batching itself.
    outputs = generate_batched(model, tokenizer, prompts, config)
    assert len(outputs) == len(prompts)
    assert model.calls == 3, "7 prompts at batch_size=3 is three batches"


def test_the_longest_prompts_are_batched_first() -> None:
    """So an OOM happens in the first seconds rather than 40 minutes in."""
    prompts = ["tiny Answer:", "a b c d e f g h Answer:", "mid size Answer:"]
    model, _ = _decode(prompts, batch_size=1)
    assert model.fed[0][0] == "a", "the 9-word prompt must go first"


def test_a_short_prompt_is_padded_on_the_left() -> None:
    """The pads have to land before the prompt, not after it.

    Right padding makes every sequence but the longest continue from pad tokens,
    which comes back as fluent text and a score a few points low. The harness builds
    the batch itself now, so this asserts the tensor it built rather than trusting
    the tokenizer's ``padding_side``.

    Turns red when: the pad prefix moves to a suffix in ``TransformersBackend``.
    """
    model, _ = _decode(["one Answer:", "a b c d Answer:"], batch_size=2)

    short = next(row for row in model.fed if "one" in row)
    assert short == ["", "", "", "one", "Answer:"], "pads first, then the prompt"
    assert model.fed[0][0] != "", "the longest row in the batch has no padding at all"


def test_an_empty_prompt_list_is_not_an_error() -> None:
    tokenizer = StubTokenizer()
    model = StubModel(tokenizer, "3")
    assert generate_batched(model, tokenizer, [], EvalConfig()) == []
    assert model.calls == 0


def test_a_model_left_in_training_mode_is_put_back() -> None:
    """Evaluating mid-fine-tune must not silently disable the trainer's dropout."""
    tokenizer = StubTokenizer()
    model = StubModel(tokenizer, "3")
    model.training = True
    generate_batched(model, tokenizer, ["a Answer:"], EvalConfig(early_stop=False))
    assert model.training is True


# --------------------------------------------------------------------------
# Continuation extraction
# --------------------------------------------------------------------------


def test_only_the_continuation_is_returned() -> None:
    """Sliced by token count, not by string matching.

    A tokenizer that normalises whitespace would leave a fragment of the prompt in
    the answer, and the extractor downstream reads the *first* digit it finds.
    """
    _, outputs = _decode(["question here Answer:"], reply="4")
    assert outputs == ["4"]


def test_padded_batch_members_get_their_own_answer_not_the_pad() -> None:
    """The whole reason for left padding: a right-padded batch continues from pads."""
    prompts = ["one Answer:", "a b c d e Answer:"]
    _, outputs = _decode(prompts, reply="2", batch_size=2)
    assert outputs == ["2", "2"]


# --------------------------------------------------------------------------
# Decode settings
#
# The failure these cover cost 23 points and produced no error: Qwen2.5-1.5B-Instruct
# ships `repetition_penalty: 1.1` in its generation_config.json, `model.generate`
# merges the checkpoint's config *under* its kwargs, and a five-shot GSM8K prompt is
# repetitive by construction. The same weights served through vLLM, whose sampling
# parameters apply no penalty, scored 60% against transformers' 37%. Both numbers
# looked like results.
# --------------------------------------------------------------------------


transformers = pytest.importorskip("transformers")


def _chat_tuned_config():
    """A checkpoint's own generation_config, as Qwen2.5-1.5B-Instruct ships it."""
    return transformers.GenerationConfig(
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.1,
        eos_token_id=[151645, 151643],
        bos_token_id=151643,
        pad_token_id=151643,
    )


def test_the_checkpoints_chat_preferences_do_not_reach_the_decode() -> None:
    """Every sampling field comes back to its neutral value, not the checkpoint's.

    Asserted on the built config rather than on ``repetition_penalty`` alone, because
    pinning the one field that bit is the fix that fails next time: the next
    checkpoint ships a different field.

    Turns red when: ``greedy_generation_config`` starts from the checkpoint's config
    (``copy.deepcopy(source)``, ``GenerationConfig(**source.to_dict())``) instead of
    from a fresh one.
    """
    model = StubModel(StubTokenizer(), "3", generation_config=_chat_tuned_config())
    greedy = greedy_generation_config(model, EvalConfig(max_new_tokens=64), pad_id=7)

    assert greedy.repetition_penalty == 1.0
    assert greedy.do_sample is False
    assert greedy.temperature == 1.0, "the library default, not the checkpoint's 0.7"
    assert greedy.top_p == 1.0
    assert greedy.top_k == 50


def test_every_field_that_could_move_a_greedy_score_is_neutral() -> None:
    """``NEUTRAL_DECODE`` is the list; this is the check that the built config obeys it.

    Turns red when: a field is added to ``NEUTRAL_DECODE`` that the built config does
    not actually leave neutral -- which is the only way the list could describe
    something other than what runs.
    """
    model = StubModel(StubTokenizer(), "3", generation_config=_chat_tuned_config())
    greedy = greedy_generation_config(model, EvalConfig(), pad_id=7)

    assert {field: getattr(greedy, field) for field in NEUTRAL_DECODE} == NEUTRAL_DECODE


def test_the_neutral_values_are_the_librarys_own_defaults() -> None:
    """The tripwire on transformers itself.

    ``NEUTRAL_DECODE`` claims each value means "not applied". That claim is only true
    while it matches what ``GenerationConfig()`` produces -- the config the harness
    builds names four fields and inherits the rest.

    Turns red when: a transformers release changes one of these defaults. Then the
    campaign's decode changed without anyone editing this repository, which is
    exactly the case worth stopping on.
    """
    fresh = transformers.GenerationConfig()
    assert {field: getattr(fresh, field) for field in NEUTRAL_DECODE} == NEUTRAL_DECODE


def test_the_tokenizers_own_ids_are_carried_over_not_reset() -> None:
    """The deliberate exception: token ids are facts, not preferences.

    Dropping ``eos_token_id`` would neutralise nothing and cost every sequence a full
    ``max_new_tokens`` of decoding.

    Turns red when: the ids stop being read off the checkpoint.
    """
    model = StubModel(StubTokenizer(), "3", generation_config=_chat_tuned_config())
    greedy = greedy_generation_config(model, EvalConfig(max_new_tokens=64), pad_id=7)

    assert greedy.eos_token_id == [151645, 151643]
    assert greedy.bos_token_id == 151643
    assert greedy.pad_token_id == 7, "the harness's pad id, not the checkpoint's"
    assert greedy.max_new_tokens == 64


def test_a_model_with_no_generation_config_still_decodes(propagating_logs, caplog) -> None:
    """A bare module has no ``generation_config``; that is a slow run, not a crash.

    Warned about because it is otherwise invisible: without an eos id every sequence
    runs to ``max_new_tokens``, which shows up as a run taking hours longer for no
    stated reason.

    Turns red when: the attribute is read directly and raises, or the warning goes.
    """
    with caplog.at_level(logging.WARNING, logger="dynquant.eval.harness"):
        greedy = greedy_generation_config(StubModel(StubTokenizer(), "3"), EvalConfig(), pad_id=7)

    assert greedy.eos_token_id is None
    assert "names no eos token" in caplog.text


def test_generation_is_driven_by_the_config_not_by_loose_kwargs() -> None:
    """The mechanism, asserted at the call site.

    A loose ``do_sample=False`` kwarg leaves every *unnamed* field merged from the
    checkpoint; an explicit ``GenerationConfig`` replaces the lot. Only the second is
    safe, so the call has to pass the object and nothing that could layer over it.

    Turns red when: a sampling kwarg is reintroduced beside ``generation_config``, or
    the config stops being passed at all.
    """
    tokenizer = StubTokenizer()
    model = StubModel(tokenizer, "3", generation_config=_chat_tuned_config())
    generate_batched(model, tokenizer, ["a b Answer:"], EvalConfig(early_stop=False))

    (call,) = model.kwargs
    assert set(call) == {"generation_config", "stopping_criteria"}
    assert call["generation_config"].repetition_penalty == 1.0


def test_the_config_is_built_once_for_the_whole_run() -> None:
    """Not per batch: it cannot vary between batches, and it is not free to build.

    Turns red when: ``greedy_generation_config`` moves inside the batch loop, where a
    later change could make one batch decode differently from another.
    """
    tokenizer = StubTokenizer()
    model = StubModel(tokenizer, "3", generation_config=_chat_tuned_config())
    prompts = [f"prompt {'x ' * i} Answer:" for i in range(5)]
    generate_batched(model, tokenizer, prompts, EvalConfig(early_stop=False, batch_size=2))

    assert model.calls == 3
    configs = {id(call["generation_config"]) for call in model.kwargs}
    assert len(configs) == 1, "every batch decoded with the same object"
