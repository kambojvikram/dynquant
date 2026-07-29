"""Banking77 prompt construction, scoring, and the intent taxonomy.

The scoring risk here is a third one, distinct from GSM8K's zero floor and
CaseHOLD's 20%. The answer is a *multi-digit* index into a 77-line list, so two
failure modes exist that a single-digit task cannot have:

**Truncation.** Gold ``41`` against a generation of ``"4"`` is a wrong answer, and
a first-digit extractor would silently score it correct roughly one time in ten.
**Out-of-range.** Nothing stops a model emitting ``"81"``. It is a digit string, so
a naive extractor accepts it as an answer that merely happens to be wrong -- when
what it actually means is that the model has lost the list it was given. That is
the same accuracy and a different diagnosis, and only the second one says the
weights are damaged.

The taxonomy itself is a correctness invariant rather than presentation: ``INTENTS``
is indexed by the gold label, so a reordering would score every answer against the
wrong intent while looking entirely healthy.

No model is loaded: generation is stubbed, and what is under test is the prompt
string, the extractor, and the accounting.
"""

from __future__ import annotations

import pytest

from dynquant.eval.banking77 import (
    CHANCE,
    FEWSHOT_STOP,
    HEADER,
    INTENTS,
    N_INTENTS,
    Banking77Example,
    Banking77Result,
    build_prompt,
    deterministic_order,
    extract_answer,
    format_training_text,
)

EXAMPLES = [
    Banking77Example(text="My card still has not arrived.", answer="11"),
    Banking77Example(text="Why was I charged twice for one purchase?", answer="63"),
]


# --------------------------------------------------------------------------
# The taxonomy
# --------------------------------------------------------------------------


def test_the_taxonomy_is_77_distinct_intents_in_label_order() -> None:
    """``INTENTS[gold]`` is the answer, so length and order are correctness."""
    assert len(INTENTS) == N_INTENTS == 77
    assert len(set(INTENTS)) == 77
    # Spot-checks at both ends and at the one upstream entry that is capitalised,
    # which is where an off-by-one during transcription would surface.
    assert INTENTS[0] == "activate my card"
    assert INTENTS[11] == "card arrival"
    assert INTENTS[51] == "refund not showing up"
    assert INTENTS[63] == "transaction charged twice"
    assert INTENTS[76] == "wrong exchange rate for cash withdrawal"


def test_intents_are_rendered_as_english_not_as_identifiers() -> None:
    """Upstream ships snake_case with a stray capital and a question mark. The model
    has to tell 77 near-synonyms apart by reading them."""
    assert not any("_" in intent for intent in INTENTS)
    assert not any("?" in intent for intent in INTENTS)
    assert all(intent == intent.lower() for intent in INTENTS)


def test_the_chance_floor_is_one_in_seventy_seven() -> None:
    """1.3%, not 20% and not zero. Against a floor this low a collapsed model is
    unmistakable -- but so is a small real regression, which is the useful half."""
    assert pytest.approx(1 / 77) == CHANCE


def test_the_header_numbers_every_intent_exactly_once() -> None:
    """The model answers with an index, so the list that numbers it is not optional
    context -- it is the definition of the answer space."""
    for index, intent in enumerate(INTENTS):
        assert f"\n{index}. {intent}" in f"\n{HEADER}"
    assert HEADER.count("\n") == N_INTENTS + 1  # two instruction lines, 77 entries


# --------------------------------------------------------------------------
# Answer extraction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The format the prompt asks for.
        (" 41", "41"),
        ("41", "41"),
        ("0", "0"),
        ("76", "76"),
        # Punctuation and emphasis around the index: cosmetic, not a wrong answer.
        (" 41.", "41"),
        ("**7**", "7"),
        (") 63", "63"),
        # Chosen, but with prose around it. Scoring this as a failure would charge a
        # formatting wobble to quantization.
        ("Intent: 41", "41"),
        ("The answer is 63.", "63"),
        # Out of range, so skipped rather than accepted -- then the in-range index
        # that follows is the model's actual choice.
        ("81 no, 63", "63"),
    ],
)
def test_extract_answer_reads_the_chosen_index(text: str, expected: str) -> None:
    assert extract_answer(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "I am not sure which intent applies.",
        # A digit string naming no intent. The model has stopped indexing the list
        # rather than indexing it wrongly, and the two are different diagnoses.
        "81",
        "1000",
    ],
)
def test_out_of_range_and_absent_indices_are_unparseable_not_wrong(text: str) -> None:
    assert extract_answer(text) is None


def test_a_truncated_index_is_not_silently_credited() -> None:
    """Gold 41 against a generation of "4" is intent 4, not intent 41. A first-digit
    extractor would score this correct about one time in ten."""
    assert extract_answer("4") == "4"
    assert extract_answer("4") != EXAMPLES[0].answer


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


def test_the_prompt_carries_the_header_once_ahead_of_the_shots() -> None:
    """Once, not per exemplar: four copies of 77 lines would treble the prompt for
    no information the model does not already have in front of it."""
    prompt = build_prompt(EXAMPLES[0], EXAMPLES[1:])
    assert prompt.count("0. activate my card") == 1
    assert prompt.startswith(HEADER)
    assert prompt.index(HEADER) < prompt.index(EXAMPLES[1].text)


def test_shots_are_answered_and_the_query_is_not() -> None:
    prompt = build_prompt(EXAMPLES[0], EXAMPLES[1:])
    assert f"Customer query: {EXAMPLES[1].text}\nIntent: {EXAMPLES[1].answer}" in prompt
    assert prompt.endswith(f"Customer query: {EXAMPLES[0].text}\nIntent:")
    assert FEWSHOT_STOP in prompt  # exemplars are blank-line separated


def test_zero_shot_still_carries_the_header() -> None:
    """Without it the index numbering is unknowable and the score is chance."""
    prompt = build_prompt(EXAMPLES[0], [])
    assert prompt == f"{HEADER}\n\nCustomer query: {EXAMPLES[0].text}\nIntent:"


def test_training_text_matches_the_evaluation_format() -> None:
    """The fine-tune must train against the same 77 lines it is scored against.
    Train on one numbering and evaluate on another and the run measures the
    mismatch."""
    prompt, completion = format_training_text(EXAMPLES[0])
    assert prompt == build_prompt(EXAMPLES[0], [])
    assert completion == " 11"


def test_training_splits_prompt_from_completion() -> None:
    """The prompt is ~615 tokens of taxonomy and the completion is one index. Train
    on the concatenation and the answer carries well under 1% of the gradient."""
    prompt, completion = format_training_text(EXAMPLES[1])
    assert completion.strip() == EXAMPLES[1].answer
    assert EXAMPLES[1].answer not in prompt.rsplit("Intent:", 1)[1]


# --------------------------------------------------------------------------
# Load order
# --------------------------------------------------------------------------

# Upstream ships both splits sorted by label: the test split is 77 contiguous
# blocks of 40. This reproduces that shape so the shuffle can be tested without a
# network round trip -- which is the whole reason deterministic_order is separate
# from the loader.
LABEL_SORTED = [
    Banking77Example(text=f"query {label}/{i}", answer=str(label))
    for label in range(N_INTENTS)
    for i in range(40)
]


def test_a_prefix_of_the_raw_order_is_a_single_intent() -> None:
    """Not a test of our code -- a record of the trap. A ``--limit 32`` smoke run
    against the raw order scored 3.12% where the real figure was 41%, because it
    had asked the model 32 questions with the same answer."""
    assert len({e.answer for e in LABEL_SORTED[:32]}) == 1


def test_shuffling_makes_a_prefix_representative() -> None:
    """The property ``--limit`` depends on: a prefix is a sample of the split, not
    a sample of one class.

    Thirty-two draws over 77 equally sized classes hit ~26 of them in expectation;
    the bar is set well below that because what is being tested is the difference
    between "a sample" and "one block", not the seed's particular draw."""
    prefix = deterministic_order(LABEL_SORTED)[:32]
    assert len({e.answer for e in prefix}) >= 20


def test_the_order_is_the_same_on_every_call() -> None:
    """The paired analysis compares per-problem hit vectors position by position, so
    an order that varied between arms would pair each problem with a different one
    and report a quantization effect that is a shuffling artifact."""
    first = deterministic_order(LABEL_SORTED)
    second = deterministic_order(LABEL_SORTED)
    assert [e.text for e in first] == [e.text for e in second]


def test_shuffling_neither_drops_nor_duplicates() -> None:
    shuffled = deterministic_order(LABEL_SORTED)
    assert sorted(e.text for e in shuffled) == sorted(e.text for e in LABEL_SORTED)
    assert [e.answer for e in LABEL_SORTED[:40]] == ["0"] * 40, "input was mutated in place"


# --------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------


def test_result_separates_wrong_answers_from_lost_format() -> None:
    result = Banking77Result(label="q3", correct=1, total=4, unparseable=2)
    result.hits = [True, False, False, False]
    assert result.accuracy == pytest.approx(0.25)
    assert result.above_chance == pytest.approx(0.25 - 1 / 77)
    summary = result.summary()
    assert "1/4 exact match" in summary
    assert "2 unparseable" in summary


def test_an_empty_result_does_not_divide_by_zero() -> None:
    assert Banking77Result(label="empty", correct=0, total=0, unparseable=0).accuracy == 0.0
