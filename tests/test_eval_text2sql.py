"""Text-to-SQL scoring: what counts as the same answer, and what counts as no answer.

The failure mode this task is exposed to is not "the scorer is too strict". It is the
opposite, and it is silent: **two queries that both return nothing compare equal.** An
evaluation set full of empty result sets scores a model that emits ``SELECT 0`` at
nearly the ceiling, and the resulting table looks like a normal set of numbers. That is
why :func:`~dynquant.eval.text2sql.load_text2sql` keeps only items whose gold query
returns rows, and why the first test here is about that rather than about accuracy.

The rest is the boundary between three verdicts that a single "wrong" count would
conflate: a query that ran and answered differently, a query that would not run, and a
generation with no query in it at all. Only the third means the model has stopped doing
the task, and on a zero-floor benchmark it is the earliest sign that quantization broke
format compliance rather than accuracy.

No model is loaded. Generation is stubbed; what is under test is the extractor, the
result comparison, the sandbox, and the accounting.
"""

from __future__ import annotations

import inspect

import pytest

from dynquant.eval.harness import strip_reasoning
from dynquant.eval.text2sql import (
    FEWSHOT_STOP,
    Text2SqlExample,
    Text2SqlResult,
    build_prompt,
    execution_match,
    extract_sql,
    format_training_text,
    instruction,
    run_query,
)

CONTEXT = (
    "CREATE TABLE staff (id INT, name TEXT, dept TEXT, salary INT); "
    "INSERT INTO staff VALUES (1, 'ada', 'eng', 100), (2, 'bo', 'eng', 90), "
    "(3, 'cy', 'ops', 80);"
)


def example(gold: str, *, ordered: bool = False) -> Text2SqlExample:
    rows, error = run_query(CONTEXT, gold)
    assert not error, error
    return Text2SqlExample(
        task_id="t",
        question="who?",
        context=CONTEXT,
        gold=gold,
        gold_rows=rows or (),
        ordered=ordered,
    )


# --- the metric's own failure mode --------------------------------------------------


def test_two_empty_result_sets_are_not_evidence_of_anything() -> None:
    """``execution_match`` says they are equal, which is why the *loader* must exclude them.

    This is not a bug to fix here: on an item whose gold genuinely returns nothing,
    returning nothing is the right answer. The problem is that it is also what
    ``SELECT 0 WHERE 0`` returns, and an evaluation set built from such items cannot
    tell a working model from a broken one. The defence lives one level up, in the
    load-time filter, and this test pins the reason it has to.

    Turns red when: someone "fixes" the comparison to treat empty as never-matching,
    which would silently change every score, or drops the loader's non-empty filter on
    the grounds that the comparison handles it.
    """
    assert execution_match((), (), ordered=False)

    degenerate, _ = run_query(CONTEXT, "SELECT id FROM staff WHERE 0")
    assert degenerate == ()
    truth, _ = run_query(CONTEXT, "SELECT id FROM staff WHERE dept = 'hr'")
    assert execution_match(truth or (), degenerate, ordered=False), (
        "an empty gold makes a nonsense query correct -- hence the loader's filter"
    )


def test_row_order_matters_only_when_the_gold_query_asked_for_one() -> None:
    """A query that did not specify an order has no order to get wrong.

    Turns red when: the comparison stops reading ``ordered``, or starts sorting the
    ``ORDER BY`` case too -- which would score a model that reversed the requested
    ranking as correct.
    """
    unordered = example("SELECT name FROM staff")
    reversed_rows, _ = run_query(CONTEXT, "SELECT name FROM staff ORDER BY id DESC")
    assert execution_match(unordered.gold_rows, reversed_rows, ordered=False)
    assert not execution_match(unordered.gold_rows, reversed_rows, ordered=True)


def test_duplicates_are_a_multiset_and_not_a_set() -> None:
    """``SELECT DISTINCT dept`` and ``SELECT dept`` are different questions.

    Comparing as sets would collapse them and credit either answer for the other.

    Turns red when: the unordered path starts deduplicating.
    """
    distinct = example("SELECT DISTINCT dept FROM staff")
    with_dupes, _ = run_query(CONTEXT, "SELECT dept FROM staff")
    assert not execution_match(distinct.gold_rows, with_dupes, ordered=False)


def test_a_float_answer_survives_a_different_summation_order() -> None:
    """``AVG`` computed two ways differs in the last bits and is the same answer.

    Turns red when: ``_normalise`` stops rounding, at which point a correct query that
    happens to aggregate in a different order is scored wrong -- the exact kind of
    false difference execution scoring exists to avoid.
    """
    direct, _ = run_query(CONTEXT, "SELECT AVG(salary) FROM staff")
    long_way, _ = run_query(CONTEXT, "SELECT SUM(salary) * 1.0 / COUNT(*) FROM staff")
    assert execution_match(direct or (), long_way, ordered=False)


# --- the sandbox --------------------------------------------------------------------


def test_the_query_cannot_reach_the_filesystem() -> None:
    """An in-memory database is not a sandbox if the query can attach a file.

    Unlike the Python tasks there is no subprocess boundary here to fall back on, so
    the authorizer is the only thing between a generated query and the disk. It is
    cheap and it is the whole defence, which is why it is asserted rather than assumed.

    Turns red when: ``_DENIED_ACTIONS`` is narrowed or the authorizer is dropped.
    """
    rows, error = run_query(CONTEXT, "ATTACH DATABASE 'evil.db' AS evil")
    assert rows is None
    assert "not authorized" in error


def test_a_query_that_would_not_terminate_is_stopped() -> None:
    """A recursive CTE with no base case must not hang the evaluation.

    Turns red when: both the step handler and the row cap are removed. Either alone is
    enough, and this asserts the pair's effect rather than which one fired -- the point
    is that the call returns.
    """
    rows, error = run_query(
        CONTEXT,
        "WITH RECURSIVE forever(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM forever) "
        "SELECT * FROM forever",
    )
    assert rows is not None or error, "the call has to come back one way or the other"


# --- reading an answer out of a generation ------------------------------------------


@pytest.mark.parametrize(
    ("generation", "expected"),
    [
        ("SELECT name FROM staff", "SELECT name FROM staff"),
        ("SELECT name FROM staff;", "SELECT name FROM staff"),
        ("```sql\nSELECT name FROM staff;\n```", "SELECT name FROM staff"),
        (
            "Here you go:\n```\nSELECT name FROM staff\n```\nHope that helps!",
            "SELECT name FROM staff",
        ),
        ("The query is: SELECT name FROM staff", "SELECT name FROM staff"),
        ("WITH t AS (SELECT 1) SELECT * FROM t", "WITH t AS (SELECT 1) SELECT * FROM t"),
        # A second statement is dropped rather than handed to sqlite, which would
        # refuse the whole thing over the tail and score a correct first query wrong.
        ("SELECT name FROM staff; SELECT 1", "SELECT name FROM staff"),
    ],
)
def test_the_query_is_recovered_from_the_shapes_a_model_writes(generation, expected) -> None:
    """An instruct model told to return only SQL still wraps it, prefixes it, or both.

    Turns red when: the fence pattern, the ``SELECT``/``WITH`` cut, or the
    one-statement rule changes. Each of these shapes was chosen because a model
    produces it, not because it is convenient to parse.
    """
    assert extract_sql(generation) == expected


def test_no_sql_at_all_is_unparseable_rather_than_wrong() -> None:
    """ "I cannot answer that" and a wrong join are the same score and different news.

    On a zero-floor task a damaged model usually keeps emitting *something*
    SQL-shaped, so a rising unparseable count is the first visible sign that
    quantization has broken format compliance rather than accuracy.

    Turns red when: the extractor starts returning a best-effort string for prose,
    which would move these into the ``errored`` bucket and hide the distinction.
    """
    assert extract_sql("I cannot answer that.") == ""
    assert extract_sql("") == ""


# --- prompts and accounting ---------------------------------------------------------


def test_the_completion_prompt_ends_where_the_model_must_continue() -> None:
    """The few-shot exemplars are separated by the stop sequence, and the last block is not.

    A prefix that ended after the exemplars would have the model answer a question it
    was not asked; a stop sequence that appeared inside an answer would truncate it.

    Turns red when: ``_COMPLETION_BLOCK`` or ``FEWSHOT_STOP`` changes without the other.
    """
    shot = example("SELECT 1")
    prompt = build_prompt(example("SELECT name FROM staff"), None, style="completion", shots=[shot])
    assert isinstance(prompt, str)
    assert prompt.endswith("SQL: ")
    assert prompt.count("Question:") == 2
    assert shot.gold in prompt


def test_accuracy_is_over_every_item_including_the_ones_with_no_query() -> None:
    """Unparseable generations are in the denominator.

    Dividing by the parseable count is how a model that answers a tenth of the set
    perfectly reports 100%.

    Turns red when: ``accuracy`` starts excluding ``unparseable`` or ``errored``.
    """
    result = Text2SqlResult(
        label="x", correct=3, total=10, unparseable=4, errored=2, hits=[True] * 3 + [False] * 7
    )
    assert result.accuracy == pytest.approx(0.3)
    assert len(result.hits) == result.total
    assert sum(result.hits) == result.correct


# --- what the fine-tune trains on ---------------------------------------------------


def test_the_training_pair_is_the_exemplar_the_completion_prompt_teaches() -> None:
    """``prompt + completion`` must be the few-shot exemplar, character for character.

    The two are built from the same ``_COMPLETION_BLOCK``, and the point of asserting it
    from the outside is that nothing else will notice if they drift. A model fine-tuned
    on one framing and evaluated on another is being measured on the gap between them,
    and the gap is invisible in the output because both halves look correct alone -- the
    training loss falls, the generations are well-formed SQL, and the accuracy is merely
    lower than it should be, uniformly across every arm that shares the fine-tune.

    Turns red when either framing changes without the other, including the trailing
    space: ``_COMPLETION_BLOCK`` ends in one, so the completion must not begin with one.
    """
    item = example("SELECT name FROM staff WHERE dept = 'eng'")
    prompt, completion = format_training_text(item)

    rendered = build_prompt(item, tokenizer=None, style="completion", shots=[item])
    # The exemplar the model is shown opens with exactly the pair it was trained on, and
    # the block it is asked to complete is exactly the training prompt.
    assert rendered.startswith(prompt + completion), "the training pair is not the exemplar"
    assert rendered.endswith(prompt), "the block to complete is not the training prompt"
    assert FEWSHOT_STOP in rendered, "no exemplar boundary -- the shot did not render"
    assert not completion.startswith(" "), "the prompt already ends in a space"
    assert prompt.endswith("SQL: ")
    assert completion == item.gold


def test_every_task_that_can_be_fine_tuned_returns_the_same_shape() -> None:
    """``format_training_text(example) -> tuple[str, str]``, in all four task modules.

    ``experiments/four_point/tasks.py`` types the slot as
    ``Callable[[Any], tuple[str, str]]``, so a task whose function takes a tokenizer, or
    returns a rendered string, cannot be registered there. ``text2sql`` shipped exactly
    that way: exported in ``__all__``, called by nothing, covered by no test, and unable
    to enter the one registry that wants it -- the same shape as the CLI carrying a
    hand-written copy of the task list while ``text2sql`` sat in ``TASKS``.

    Checked by signature rather than by calling, so a new task is admitted to this guard
    without needing a fixture here first.

    Turns red when a task's training pair drifts from the protocol -- which is the point
    at which it silently stops being fine-tunable through the shared driver.
    """
    import importlib

    modules = ["banking77", "casehold", "gsm8k", "text2sql"]
    for name in modules:
        module = importlib.import_module(f"dynquant.eval.{name}")
        assert "format_training_text" in module.__all__, name
        signature = inspect.signature(module.format_training_text)
        required = [
            p
            for p in signature.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        assert len(required) == 1, f"{name}: takes {len(required)} required arguments, not 1"
        assert signature.return_annotation == "tuple[str, str]", (
            f"{name}: returns {signature.return_annotation}, not a (prompt, completion) pair"
        )


# --- reasoning models -----------------------------------------------------------------


def test_a_query_inside_a_reasoning_trace_is_not_the_answer() -> None:
    """The trace is where the model argues against candidates, so its SQL is not a verdict.

    This is the exact generation shape ``LFM2.5-8B-A1B`` produces, abbreviated: a
    ``SELECT`` written mid-deliberation, immediately contradicted, and the real answer
    after the close tag. Reading the first ``SELECT`` in the whole text picks up the one
    the model went on to reject -- which runs, returns the wrong rows or no rows at all,
    and is counted as a wrong answer rather than as a harness that read the wrong region.

    Measured cost of getting this wrong: 6.2% against 40.6% on 32 items of the mixture.

    Turns red when ``extract_sql`` stops cutting the trace.
    """
    generation = (
        "<think>\nWe need the eng staff. Maybe SELECT name FROM employees WHERE dept = 'eng';\n"
        "But note that the table is called staff, not employees.\n</think>\n"
        "SELECT name FROM staff WHERE dept = 'eng';"
    )
    assert extract_sql(generation) == "SELECT name FROM staff WHERE dept = 'eng'"


def test_a_trace_that_never_closed_is_no_answer_rather_than_a_wrong_one() -> None:
    """A model that spent its whole budget thinking did not answer, and that is different.

    Counted unparseable, which is the truth and is separately tallied. Returning the
    trace instead would score a decode setting -- ``max_new_tokens`` too small for a
    model that reasons -- as though the model had produced an incorrect query, and a
    quantization comparison would read the resulting drop as damage.

    13 of the 32 probe generations were this case at 256 new tokens.

    Turns red when an unterminated trace starts being mined for a query.
    """
    truncated = "<think>\nWe need SELECT name FROM staff WHERE dept = 'eng' but the column may be"
    assert extract_sql(truncated) == ""


def test_stripping_reasoning_cannot_move_a_number_already_collected() -> None:
    """On output with no trace in it the cut is the identity, which is what makes it safe.

    All five task extractors call it, including the four whose models never emitted a
    trace. That is only defensible if it is provably inert on their generations, so the
    property is asserted rather than assumed.

    Turns red if the helper ever starts editing ordinary generations -- at which point
    every previously collected number in the campaign became unreproducible.
    """
    plain = "Here is the query:\n```sql\nSELECT name FROM staff WHERE dept = 'eng'\n```"
    assert strip_reasoning(plain) == plain
    assert extract_sql(plain) == "SELECT name FROM staff WHERE dept = 'eng'"


def test_the_chat_prompt_contains_the_shots_the_manifest_says_it_does() -> None:
    """``shots=2`` in the record has to mean two exemplars reached the model.

    The chat branch used to drop them while the CLI wrote ``"shots": 2`` into the result
    JSON. Nothing downstream could catch that: every arm shares the prompt, so the
    comparison stays clean and only the provenance is wrong -- and the record outlives
    the run that would have contradicted it.

    Asserted through a recording stub rather than a real tokenizer, because what is under
    test is the message list this builds, not how a template renders it.

    Turns red when the chat branch stops passing shots through, or renders an exemplar's
    answer as anything but bare SQL (which a reasoning model would read as a trace).
    """
    seen: list[list[dict[str, str]]] = []

    class _Tokenizer:
        def apply_chat_template(self, messages: object, **kwargs: object) -> list[int]:
            seen.append([dict(m) for m in messages])  # type: ignore[union-attr]
            return [0]

    item = example("SELECT name FROM staff WHERE dept = 'eng'")
    shot = example("SELECT dept FROM staff")
    build_prompt(item, _Tokenizer(), style="chat", shots=[shot, shot])

    assert seen, "render_chat never reached the tokenizer"
    roles = [m["role"] for m in seen[0]]
    assert roles == ["user", "assistant", "user", "assistant", "user"], roles
    assert seen[0][1]["content"] == shot.gold, "the exemplar answer is not bare SQL"
    assert seen[0][-1]["content"] == instruction(item), "the scored item is not the last turn"


def test_the_answer_region_starts_after_the_last_close_tag_not_the_first() -> None:
    """Which tag ends the reasoning is the model's convention, so follow the model.

    ``LFM2.5``'s chat template strips a previous turn with ``split("</think>")[-1]``, so a
    generation carrying more than one block has its answer after the *last* tag. Cutting
    at the first returns a region that is still reasoning, and it contains a ``SELECT``,
    so the failure is a plausible query rather than an empty extraction.

    Turns red on a switch to the first tag -- which no other test here distinguishes,
    because a single-block generation looks identical either way.
    """
    generation = (
        "<think>\nLet me re-read the schema.\n</think>\n"
        "<think>\nSELECT name FROM employees; is wrong -- the table is staff.\n</think>\n"
        "SELECT name FROM staff WHERE dept = 'eng';"
    )
    # The rejected query sits in the *second* block, so cutting at the first tag returns a
    # region whose leading SELECT is the wrong one. A discarded query in the first block
    # would leave both tag choices landing on the same answer, and this test would pass
    # without testing anything.
    assert extract_sql(generation) == "SELECT name FROM staff WHERE dept = 'eng'"
