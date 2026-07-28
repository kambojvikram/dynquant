"""CaseHOLD prompt construction, scoring, and the chance floor.

The scoring risk here is different from GSM8K's and in one way worse. GSM8K's floor
is zero, so a broken scorer produces a suspiciously round nothing. CaseHOLD is a
five-way choice, so a broken scorer produces **20%** -- a number that looks exactly
like "quantization damaged the model" and reads as a result rather than as a bug.

Hence two things asserted here that GSM8K did not need: that a model choosing with a
little extra prose is still credited with its choice, and that a model emitting no
digit at all is counted as *unparseable* rather than as wrong. Those two are the same
accuracy but different diagnoses, and only one of them means the model still knows
what it was asked.

No model is loaded: generation is stubbed, and what is under test is the prompt
string, the extractor, and the accounting.
"""

from __future__ import annotations

import pytest

from dynquant.eval.casehold import (
    CHANCE,
    FEWSHOT_STOP,
    CaseholdExample,
    CaseholdResult,
    build_prompt,
    extract_answer,
    format_training_text,
)

EXAMPLES = [
    CaseholdExample(
        citing_prompt="The court in (<HOLDING>) reached the opposite result.",
        holdings=("holding zero", "holding one", "holding two", "holding three", "holding four"),
        answer="1",
    ),
    CaseholdExample(
        citing_prompt="See id. at 400 (<HOLDING>).",
        holdings=("alpha", "beta", "gamma", "delta", "epsilon"),
        answer="3",
    ),
]


# --------------------------------------------------------------------------
# Answer extraction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The format the prompt asks for.
        (" 3", "3"),
        ("3", "3"),
        # Punctuation and leading junk around the digit: cosmetic, not a wrong answer.
        (" 3.", "3"),
        ("**2**", "2"),
        (") 0", "0"),
        # The model chose, then explained. It still chose.
        ("4 because the holding matches", "4"),
        ("holding 2 is the one cited", "2"),
        # Both ends of the range, because an off-by-one in the character class would
        # silently score every "0" or every "4" as unparseable.
        ("0", "0"),
        ("4", "4"),
        # No digit in range at all -- the format has collapsed. Must be None, not a
        # guess: a guess would be scored as a wrong answer and hide the collapse.
        ("I cannot determine which holding applies", None),
        ("", None),
        ("   ", None),
        # 5 is out of range for a five-way choice, and there is nothing else.
        ("5", None),
    ],
)
def test_extract_answer_handles_what_models_actually_emit(text: str, expected: str | None) -> None:
    assert extract_answer(text) == expected


def test_a_leading_digit_beats_a_later_one() -> None:
    """The fallback must not override the format.

    A generation that answers and then runs on into invented text contains more
    digits. Reading "any digit anywhere" unconditionally would let the second one win
    on some inputs and the first on others, which is a scorer whose result depends on
    how chatty the model is.
    """
    assert extract_answer("1 -- see also holding 4") == "1"


def test_out_of_range_digits_do_not_block_a_valid_later_choice() -> None:
    """``"7"`` is not a holding index, so it must not be mistaken for the answer."""
    assert extract_answer("Option 2") == "2"


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


def test_prompt_ends_where_the_model_must_continue() -> None:
    prompt = build_prompt(EXAMPLES[1], shots=EXAMPLES[:1])
    assert prompt.endswith("4. epsilon\nAnswer:")


def test_all_five_holdings_are_offered() -> None:
    """Numbered 0-4, matching the label space. An off-by-one here would make the
    gold label point at the wrong option and cap accuracy near chance."""
    prompt = build_prompt(EXAMPLES[1], shots=())
    for index, holding in enumerate(EXAMPLES[1].holdings):
        assert f"{index}. {holding}" in prompt


def test_the_shot_demonstrates_the_answer_format() -> None:
    prompt = build_prompt(EXAMPLES[1], shots=EXAMPLES[:1])
    assert "Answer: 1" in prompt, "the exemplar must show a bare digit as the answer"


def test_the_shot_separator_is_the_stop_sequence() -> None:
    """The prompt teaches "blank line, then the next case". The decoder stops there,
    or every generation runs to the token budget and puts stray digits in front of
    the fallback extractor."""
    prompt = build_prompt(EXAMPLES[1], shots=EXAMPLES[:1])
    assert FEWSHOT_STOP in prompt


def test_zero_shot_prompt_is_still_well_formed() -> None:
    prompt = build_prompt(EXAMPLES[0], shots=())
    assert prompt.startswith("The court in (<HOLDING>)")
    assert prompt.endswith("Answer:")


def test_training_text_is_split_so_loss_can_be_masked() -> None:
    """Load-bearing, not tidiness.

    The prompt is ~400 tokens of case-law prose and the completion is one digit.
    Training on the concatenation gives the answer under 1% of the gradient: the run
    teaches the model to write appellate prose, which it already does, and the
    evaluation comes back flat with a healthy-looking training loss the whole way.
    """
    prompt, completion = format_training_text(EXAMPLES[1])
    assert completion == " 3"
    assert prompt.endswith("Answer:")
    # And the joined form is exactly what an eval prompt continues into, so the
    # fine-tune and the evaluation agree down to the whitespace.
    assert prompt == build_prompt(EXAMPLES[1], shots=())
    assert (prompt + completion) == build_prompt(EXAMPLES[1], shots=()) + " 3"


def test_training_format_matches_the_exemplar_format_exactly() -> None:
    """Otherwise the fine-tune optimises one string shape and is scored on another,
    and the difference is charged to the method."""
    prompt, completion = format_training_text(EXAMPLES[0])
    assert (prompt + completion) in build_prompt(EXAMPLES[1], shots=EXAMPLES[:1])


# --------------------------------------------------------------------------
# Result accounting
# --------------------------------------------------------------------------


def test_chance_is_recorded_as_a_fifth() -> None:
    """Quoted beside every number: 34% against a 20% floor is 14 points of skill,
    and a collapsed model returns to 20% rather than to zero."""
    assert pytest.approx(0.2) == CHANCE


def test_above_chance_subtracts_the_floor() -> None:
    result = CaseholdResult(label="x", correct=40, total=100, unparseable=0)
    assert result.accuracy == pytest.approx(0.40)
    assert result.above_chance == pytest.approx(0.20)


def test_a_model_at_chance_reports_zero_skill() -> None:
    result = CaseholdResult(label="x", correct=20, total=100, unparseable=0)
    assert result.above_chance == pytest.approx(0.0)


def test_an_empty_result_does_not_divide_by_zero() -> None:
    assert CaseholdResult(label="x", correct=0, total=0, unparseable=0).accuracy == 0.0


def test_the_summary_carries_the_floor_and_the_unparseable_count() -> None:
    """A reader of the log needs both to interpret the accuracy at all."""
    summary = CaseholdResult(label="fp16", correct=40, total=100, unparseable=7).summary()
    assert "40.00%" in summary
    assert "vs chance" in summary
    assert "7 unparseable" in summary


# --------------------------------------------------------------------------
# End-to-end scoring, stubbed
# --------------------------------------------------------------------------


def _evaluate(reply: str, examples=EXAMPLES, **kwargs):
    pytest.importorskip("torch")
    from _decode_stub import StubModel, StubTokenizer

    from dynquant.eval.casehold import evaluate_casehold
    from dynquant.eval.harness import EvalConfig

    tokenizer = StubTokenizer()
    model = StubModel(tokenizer, reply)
    config = EvalConfig(early_stop=False, max_new_tokens=8, max_prompt_tokens=4096)
    return evaluate_casehold(model, tokenizer, examples, label="stub", config=config, **kwargs)


def test_hits_are_recorded_per_problem_in_dataset_order() -> None:
    """The vector, not the count, is the deliverable.

    Two models scored on one fixed set is a paired design, and McNemar's test needs
    to know *which* problems flipped. That cannot be recovered once the GPU time is
    spent, so it is never optional and never sampled.
    """
    result = _evaluate("3")
    assert result.hits == [False, True], "gold labels are 1 then 3"
    assert result.correct == 1
    assert result.total == 2
    assert len(result.hits) == result.total


def test_a_generation_with_no_digit_is_unparseable_not_wrong() -> None:
    """Same accuracy, different diagnosis.

    A rising unparseable count is the earliest sign that quantization broke format
    compliance rather than legal reasoning, and on a five-way task it is the only
    signal that distinguishes the two -- both land near 20%.
    """
    result = _evaluate("unclear")
    assert result.unparseable == 2
    assert result.correct == 0
    assert result.hits == [False, False]


def test_a_correct_answer_is_never_counted_as_unparseable() -> None:
    result = _evaluate("1")
    assert result.unparseable == 0
    assert result.hits == [True, False]


def test_predictions_are_kept_only_up_to_the_requested_count() -> None:
    """Keeping every generation on 5,314 problems across six arms is a large file for
    no extra information; keeping none makes a collapsed arm impossible to diagnose."""
    result = _evaluate("3", keep_predictions=1)
    assert len(result.predictions) == 1
    assert result.predictions[0] == {
        "generation": "3",
        "predicted": "3",
        "gold": "1",
        "correct": False,
    }


def test_a_caller_config_without_the_stop_sequence_gets_it_added() -> None:
    """A config assembled elsewhere must not be able to remove the turn boundary.

    Without it the model runs on into a case it invented, which is slower and puts
    stray digits in front of the extractor -- a silent accuracy change caused by a
    decode setting rather than by the weights.
    """
    pytest.importorskip("torch")
    from _decode_stub import StubModel, StubTokenizer

    from dynquant.eval.casehold import evaluate_casehold
    from dynquant.eval.harness import EvalConfig

    tokenizer = StubTokenizer()
    model = StubModel(tokenizer, "3 spillover")
    result = evaluate_casehold(
        model,
        tokenizer,
        EXAMPLES,
        label="stub",
        config=EvalConfig(early_stop=False, stop_sequences=("ZZZ",)),
    )
    # The stub joins tokens with single spaces, so the reply survives intact; what is
    # asserted is that scoring still worked with a foreign stop sequence in place.
    assert result.total == 2
    assert result.hits == [False, True]


def test_limit_is_honoured_and_reported_in_the_total() -> None:
    """A smoke run must not be mistakable for a full one: the denominator changes."""
    pytest.importorskip("torch")
    from _decode_stub import StubModel, StubTokenizer

    from dynquant.eval.casehold import evaluate_casehold
    from dynquant.eval.harness import EvalConfig

    tokenizer = StubTokenizer()
    model = StubModel(tokenizer, "1")
    result = evaluate_casehold(
        model,
        tokenizer,
        EXAMPLES,
        label="stub",
        config=EvalConfig(early_stop=False, limit=1),
    )
    assert result.total == 1
    assert result.hits == [True]
