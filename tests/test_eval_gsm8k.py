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
    evaluate_gsm8k,
    extract_answer,
    format_training_text,
)
from dynquant.eval.harness import EvalConfig, _truncate

from ._decode_stub import StubModel, StubTokenizer

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
    assert "\n\nQuestion:" in prompt, "exemplars are blank-line separated"


def test_the_stop_matches_a_run_on_the_model_writes_inline() -> None:
    """The prompt's separator is not the stop, because models do not write it back.

    Qwen2.5-1.5B-Instruct finishes an answer and starts a problem it invented on the
    same line -- "the answer is 366. Question: There are 12 more green apples..." --
    so a stop of "\\n\\nQuestion:", the separator :func:`build_prompt` actually uses,
    matches nothing. Generation then runs to the token budget through two or three
    invented problems and the fallback extractor answers one of *those*.

    The text below is a real generation from the runtime-parity gate, shortened.
    """
    observed = (
        "So in total, it had 60 + 180 + 126 = 366 downloads.\n\n"
        "Therefore, the answer is 366. Question: There are 12 more green apples than "
        "red apples in a bowl.\nAnswer: Red apples: 16. Green apples: 16 + 12 = 28. "
        "Total: 16 + 28 = 44 apples.\n\nTherefore, the answer is 44."
    )
    assert "\n\nQuestion:" not in observed, "the model never writes the separator back"
    assert extract_answer(observed) == "44", "untruncated, the extractor answers the wrong problem"
    assert extract_answer(_truncate(observed, (FEWSHOT_STOP,))) == "366"


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


def test_the_task_adds_its_stop_to_a_config_that_arrives_without_one() -> None:
    """Callers build the ``EvalConfig``; the stop is the task's business, not theirs.

    ``scripts/gate_runtime_parity.py`` hands in a config it assembled from command
    line flags, which carries ``stop_sequences=()``. If the task does not merge its
    own stop into it, nothing cuts the generation: the run-on below survives to the
    extractor, which reads the last number in the text and answers a problem the
    model invented. The score that comes back is wrong rather than missing.
    """
    tokenizer = StubTokenizer()
    model = StubModel(
        tokenizer,
        reply="Therefore the answer is 366. Question: There are 12 apples Answer: 44",
    )
    example = Gsm8kExample(question="How many downloads?", reasoning="Add them.", answer="366")

    result = evaluate_gsm8k(
        model,
        tokenizer,
        [example],
        label="stub",
        config=EvalConfig(max_new_tokens=32, batch_size=2),
        keep_predictions=1,
    )

    assert result.predictions[0]["generation"] == "Therefore the answer is 366."
    assert result.correct == 1, "the stop never reached the config and 44 was scored"


def test_eval_config_defaults_to_greedy_shaped_settings() -> None:
    """Sampling would make small accuracy gaps unmeasurable without many seeds."""
    config = EvalConfig()
    assert config.batch_size > 0
    assert config.max_new_tokens >= 256, "GSM8K chains need room to finish"
