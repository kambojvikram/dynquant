"""GSM8K prompt construction and answer scoring.

A scoring bug is the most expensive kind of bug in this project, because it does
not look like a bug. It looks like a result. If the extractor mishandles
``1,000``, every model scores a couple of points low, the ranking between them
survives, and the conclusion "3-bit costs 6 points" is wrong by an amount nobody
can see. So the parser is tested against the shapes models actually emit, not
against the shapes the format asks for.

No model is loaded here: generation is stubbed, and what is under test is the
prompt string, the extractor, and the match rule.
"""

from __future__ import annotations

import pytest

from dynquant.eval.gsm8k import (
    FEWSHOT_STOP,
    Gsm8kExample,
    build_prompt,
    extract_answer,
    format_training_text,
)
from dynquant.eval.harness import EvalConfig, _truncate

EXAMPLES = [
    Gsm8kExample(question="Two plus two?", reasoning="Add them.", answer="4"),
    Gsm8kExample(question="Three times three?", reasoning="Multiply.", answer="9"),
]


# --------------------------------------------------------------------------
# Answer extraction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The format the prompt asks for.
        ("She has 18 left.\n#### 18", "18"),
        # Thousands separators, currency and trailing punctuation: all cosmetic,
        # all things a model emits unprompted, none of them a wrong answer.
        ("#### 1,000", "1000"),
        ("#### $1,234.50", "1234.50"),
        ("#### 42.", "42"),
        ("#### 75%", "75"),
        # Negative results occur in GSM8K ("how much did he lose").
        ("#### -5", "-5"),
        # No marker at all -- the model reasoned and forgot the format. Taking the
        # last number scores the arithmetic rather than the compliance.
        ("First 3 apples, then 4 more, so 7 apples in total", "7"),
        # A marker plus trailing chatter: the marked value wins over the later one.
        ("#### 12\n\nQuestion: what about", "12"),
        # Nothing numeric: must be None, not 0. Scored as unparseable, which is a
        # distinct failure mode from being wrong.
        ("I am not sure about this one.", None),
        ("", None),
    ],
)
def test_extract_answer_handles_what_models_actually_emit(text: str, expected: str | None) -> None:
    assert extract_answer(text) == expected


def test_marked_answer_beats_a_later_number() -> None:
    """The fallback must not override the marker.

    A generation that answers and then starts the next problem contains a second
    number. Reading "last number in the text" unconditionally would score that
    second problem's figure as this problem's answer.
    """
    text = "The total is 18.\n#### 18\n\nQuestion: Bob has 99 marbles"
    assert extract_answer(text) == "18"


def test_numeric_equality_ignores_harmless_formatting() -> None:
    """``1000`` and ``1000.00`` are one answer; string equality says otherwise."""
    from dynquant.eval.gsm8k import _numeric_match

    assert _numeric_match("1000", "1000")
    assert _numeric_match("1000.00", "1000")
    assert _numeric_match("-5.0", "-5")
    assert not _numeric_match("1000", "10000")
    assert not _numeric_match("18", "81")


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


def test_prompt_ends_where_the_model_must_continue() -> None:
    prompt = build_prompt(EXAMPLES[1], shots=EXAMPLES[:1])
    assert prompt.endswith("Question: Three times three?\nAnswer:")
    assert "#### 4" in prompt, "the shot must demonstrate the answer marker"


def test_the_shot_separator_is_the_stop_sequence() -> None:
    """Otherwise the model runs on into a problem it invented.

    The prompt teaches "blank line, then the next Question:". The decoder has to
    stop on exactly that string, or every generation continues until the token
    budget runs out -- slow, and it puts extra numbers in front of the fallback
    extractor.
    """
    prompt = build_prompt(EXAMPLES[1], shots=EXAMPLES[:1])
    assert FEWSHOT_STOP in prompt


def test_zero_shot_prompt_is_still_well_formed() -> None:
    assert build_prompt(EXAMPLES[0], shots=()) == "Question: Two plus two?\nAnswer:"


def test_training_text_is_split_so_loss_can_be_masked() -> None:
    """Concatenating them would train on the question tokens too."""
    prompt, completion = format_training_text(EXAMPLES[0])
    assert prompt == "Question: Two plus two?\nAnswer:"
    assert completion == " Add them.\n#### 4"
    # And the joined form is exactly what an eval prompt continues into, so the
    # fine-tune and the evaluation agree on the format down to the whitespace.
    assert (prompt + completion).startswith(build_prompt(EXAMPLES[0], shots=()))


# --------------------------------------------------------------------------
# Decode truncation
# --------------------------------------------------------------------------


def test_truncation_takes_the_earliest_stop_not_the_first_listed() -> None:
    """Order-independence: the caller's tuple order must not change the answer."""
    text = "answer 5\nEND_B here\nEND_A there"
    assert _truncate(text, ("END_A", "END_B")) == "answer 5\n"
    assert _truncate(text, ("END_B", "END_A")) == "answer 5\n"


def test_truncation_leaves_text_without_a_stop_alone() -> None:
    assert _truncate("no stop here", ("END",)) == "no stop here"


def test_eval_config_defaults_to_greedy_shaped_settings() -> None:
    """Sampling would make small accuracy gaps unmeasurable without many seeds."""
    config = EvalConfig()
    assert config.batch_size > 0
    assert config.max_new_tokens >= 256, "GSM8K chains need room to finish"
