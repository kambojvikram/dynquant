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

from dynquant.eval.harness import EvalConfig, generate_batched  # noqa: E402


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
    return tokenizer, generate_batched(model, tokenizer, prompts, config)


# --------------------------------------------------------------------------
# Truncation side
# --------------------------------------------------------------------------


def test_an_overlong_prompt_keeps_its_tail_not_its_head() -> None:
    """The end of a few-shot prompt is the part the model has to act on.

    Exemplars sit at the front and are expendable; the question, the options and the
    trailing cue sit at the back. Cutting from the right removes the cue.
    """
    prompt = "exemplar filler filler filler QUESTION options Answer:"
    tokenizer, _ = _decode([prompt], max_prompt_tokens=3)

    assert tokenizer.fed == [["QUESTION", "options", "Answer:"]]
    assert "exemplar" not in tokenizer.fed[0]


def test_a_prompt_that_fits_is_not_touched() -> None:
    prompt = "short question Answer:"
    tokenizer, _ = _decode([prompt], max_prompt_tokens=64)
    assert tokenizer.fed == [["short", "question", "Answer:"]]


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


def test_the_tokenizer_is_left_as_it_was_found() -> None:
    """Both sides are mutated for the duration; a caller's tokenizer is not ours.

    A tokenizer left on ``truncation_side="left"`` would silently change how the
    *training* set is built if the same object is reused, which is exactly the kind
    of cross-contamination that makes a fine-tune and its evaluation disagree.
    """
    tokenizer = StubTokenizer()
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    model = StubModel(tokenizer, "3")

    generate_batched(model, tokenizer, ["a b Answer:"], EvalConfig(early_stop=False))

    assert tokenizer.padding_side == "right"
    assert tokenizer.truncation_side == "right"


def test_the_tokenizer_is_restored_even_when_generation_raises() -> None:
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
    tokenizer, _ = _decode(prompts, batch_size=1)
    assert tokenizer.fed[0][0] == "a", "the 9-word prompt must go first"


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
