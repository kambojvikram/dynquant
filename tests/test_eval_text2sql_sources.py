"""Where the items come from: admission, WikiSQL synthesis, and the balance of the mixture.

Its sibling ``test_eval_text2sql.py`` tests what happens once an item exists. This file
tests whether it should exist at all, which on this task is the part that decides whether
the numbers mean anything.

Three things are under test and each one has already failed once.

**Admission.** An item is usable only if its gold query, run against the schema in its own
prompt, returns something a wrong answer would not. That rule has three escape hatches and
all three are aggregates: an empty result set, ``COUNT(*)`` over an empty schema returning
``[(0,)]``, and ``AVG(x)`` over a non-matching condition returning ``[(None,)]``. Each one
admits an item whose correct answer is free to guess, and a set of them scores a broken
model near the ceiling.

**Synthesis.** WikiSQL ships tables and structured query fields, not SQL. The rebuild has
to survive Wikipedia's headers -- spaces, quotes, duplicates -- and the fact that the
annotator's condition values are typed in a different case from the cells they match.

**Balance.** A mixture whose headline is dominated by whichever source survives its filter
most often is a single-source number wearing three names.

No network. Every source read is faked; what is under test is this repository's code.
"""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from dynquant.errors import DynQuantError
from dynquant.eval.text2sql import (
    _is_degenerate,
    _quotas,
    admit,
    database_has_rows,
    load_text2sql,
    run_query,
)
from dynquant.eval.text2sql_sources import (
    DEFAULT_TRAIN,
    MAX_TABLE_ROWS,
    SOURCES,
    RawItem,
    Source,
    SourceTally,
    _read_create_context,
    _wikisql_item,
    evaluation_questions,
    is_readable_query,
    question_key,
    read_source,
    resolve_sources,
)

POPULATED = "CREATE TABLE t (a INT, b TEXT); INSERT INTO t VALUES (1, 'x'), (2, 'y');"
BARE = "CREATE TABLE t (a INT, b TEXT);"


def item(context: str, gold: str, *, source: str = "s") -> RawItem:
    return RawItem(task_id="i", question="q?", context=context, gold=gold, source=source)


# --- admission: the three ways an unanswerable item looks answerable ------------------


def test_a_schema_with_no_rows_is_refused_however_the_gold_is_written() -> None:
    """``SELECT COUNT(*)`` over an empty schema returns one row, and it is not evidence.

    This is the subtle half of the rule and the half that was missing. Checking only
    "did the gold return rows" admits every count and every aggregate over a schema with
    no ``INSERT``s, because those return ``[(0,)]`` rather than nothing -- and any model
    that writes any count over any table reproduces the gold exactly. The database is
    therefore checked as well as the query.

    Turns red when: ``database_has_rows`` stops being consulted, or the ``no_data``
    branch is reordered after the empty-result branch, which would relabel these items
    rather than admit them but would still hide the reason in the tally.
    """
    rows, error = run_query(BARE, "SELECT COUNT(*) FROM t")
    assert not error
    assert rows == ((0,),), "the trap: one row, from a schema holding nothing"

    tally = SourceTally()
    assert admit(item(BARE, "SELECT COUNT(*) FROM t"), require_rows=True, tally=tally) is None
    assert tally.no_data == 1
    assert tally.kept == 0


def test_an_aggregate_that_matched_nothing_is_refused_even_over_real_data() -> None:
    """``AVG`` over a condition nothing satisfies returns ``[(None,)]`` -- one row of nothing.

    The database is populated and the gold query is well-formed, so neither the
    ``no_data`` check nor the empty-result check fires; this is the third form and it
    needs its own test because it is reachable only through the one path the other two
    leave open.

    Turns red when: ``_is_degenerate`` is dropped, or narrowed to ``None`` alone -- the
    zero case is the same failure and is produced by ``COUNT``.
    """
    gold = "SELECT AVG(a) FROM t WHERE b = 'nobody'"
    rows, error = run_query(POPULATED, gold)
    assert not error
    assert rows == ((None,),)

    tally = SourceTally()
    assert admit(item(POPULATED, gold), require_rows=True, tally=tally) is None
    assert tally.degenerate == 1


@pytest.mark.parametrize(
    ("rows", "degenerate"),
    [
        (((None,),), True),
        (((0,),), True),
        (((0, None),), True),
        (((0,), (1,)), False),  # two rows: the model had to find them
        (((0, 5),), False),  # a zero beside a real value is an answer
        (((1,),), False),
    ],
)
def test_degenerate_is_one_row_of_nothing_and_not_merely_a_zero(rows, degenerate) -> None:
    """A zero is only meaningless when it is the *entire* result.

    ``SELECT wins, losses`` returning ``(0, 5)`` is a real answer that happens to contain
    a zero, and excluding it would delete correct items from the evaluation set -- the
    opposite failure, and one that shows up as a smaller N rather than as a wrong number.

    Turns red when: the predicate starts matching any row containing a zero, or stops
    requiring exactly one row.
    """
    assert _is_degenerate(rows) is degenerate


def test_a_populated_lookup_table_beside_an_empty_one_still_counts_as_data() -> None:
    """Every base table is checked, not the first.

    A schema that defines a populated fact table after an empty dimension table is
    normal in Gretel's contexts, and stopping at the first table would refuse the whole
    item over a table the question never touches.

    Turns red when: the scan stops at the first table or reads ``sqlite_master`` in a
    way that depends on declaration order.
    """
    assert not database_has_rows(BARE)
    assert database_has_rows(POPULATED)
    assert database_has_rows(
        "CREATE TABLE empty_dim (id INT);CREATE TABLE fact (id INT);INSERT INTO fact VALUES (1);"
    )


def test_training_keeps_items_evaluation_refuses() -> None:
    """A gold query returning nothing is broken scoring and perfectly good supervision.

    "List the orders from a customer who has none" has an empty answer and the SQL that
    produces it is the right SQL. Applying the evaluation filter to the training set
    would throw away most of two sources for a reason that only applies to scoring.

    Turns red when: ``require_rows`` stops being threaded through, or the ``train``
    default flips -- either of which silently changes what the fine-tune sees.
    """
    unanswerable = item(POPULATED, "SELECT a FROM t WHERE b = 'nobody'")
    assert admit(unanswerable, require_rows=True, tally=SourceTally()) is None

    tally = SourceTally()
    kept = admit(unanswerable, require_rows=False, tally=tally)
    assert kept is not None
    assert kept.gold_rows == ()
    assert tally.kept == 1


def test_a_refusal_records_which_kind_it_was() -> None:
    """A source that stops contributing must not look like a source nobody selected.

    Turns red when: a refusal path returns ``None`` without incrementing its counter,
    which makes the loader's log add up to less than ``seen`` and hides a source that
    quietly emptied.
    """
    # A real query over a table that is not there -- not a typo'd keyword, which would be
    # refused one step earlier as `not_a_query` and would test that check instead.
    tally = SourceTally()
    assert admit(item(POPULATED, "SELECT a FROM missing"), require_rows=True, tally=tally) is None
    assert tally.failed == 1
    assert sum(tally.errors.values()) == 1
    assert tally.seen == 1


@pytest.mark.parametrize(
    ("gold", "readable"),
    [
        ("SELECT a FROM t", True),
        ("  \n WITH x AS (SELECT 1) SELECT * FROM x", True),
        ("-- a comment\nSELECT a FROM t", True),
        ("select a from t", True),
        ("UPDATE t SET a = 1", False),
        ("INSERT INTO t VALUES (1)", False),
        ("DELETE FROM t WHERE a = 1", False),
        ("CREATE TABLE u (a INT)", False),
        # The anchoring case: a subquery does not make a statement a query.
        ("UPDATE t SET a = (SELECT max(b) FROM u)", False),
    ],
)
def test_only_a_statement_that_starts_as_a_query_can_be_asked_for(gold, readable) -> None:
    """9.8% of Gretel's golds are DML, and they teach an answer the scorer cannot read.

    ``extract_sql`` finds an answer by cutting a generation at ``SELECT`` or ``WITH``, so
    a model fine-tuned to reply ``UPDATE ...`` produces text that scores ``unparseable``
    -- zero, on a metric with no chance floor. The evaluation split already excluded
    these, but only as a side effect of the row filter, and filed under "gold matched
    nothing"; training has no row filter at all, so there they survived.

    Turns red when: the match is unanchored (the last case starts passing) or the check
    is moved behind ``require_rows``, which would restore the training leak.
    """
    assert is_readable_query(gold) is readable


def test_a_dml_gold_is_refused_in_training_too_and_says_so() -> None:
    """Both splits, and counted apart from ``empty_result``.

    An empty result set is a question about data that is not there; a DML statement is a
    row this task cannot pose. Merging them would report the first number wrong and hide
    the second entirely.

    Turns red when: the check moves inside the ``require_rows`` branch, or increments
    ``empty_result``.
    """
    dml = item(POPULATED, "UPDATE t SET a = 1 WHERE b = 'x'")
    for require_rows in (True, False):
        tally = SourceTally()
        assert admit(dml, require_rows=require_rows, tally=tally) is None
        assert tally.not_a_query == 1, require_rows
        assert tally.empty_result == 0


# --- WikiSQL synthesis ---------------------------------------------------------------


def wikisql_row(
    *,
    question="who?",
    header=("Home team", "Away team", "Year"),
    types=("text", "text", "real"),
    rows=(("Hawthorn FC", "Terrence Ross", 2011.0),),
    sel=0,
    agg=0,
    conds=None,
    table_id="2-1080",
):
    conds = conds or {"column_index": [1], "operator_index": [0], "condition": ["terrence ross"]}
    return {
        "question": question,
        "table": {
            "id": table_id,
            "header": list(header),
            "types": list(types),
            "rows": [list(r) for r in rows],
            "page_title": "p",
        },
        "sql": {"sel": sel, "agg": agg, "conds": conds},
    }


def test_the_annotators_lowercase_value_still_matches_the_wikipedia_cell() -> None:
    """``COLLATE NOCASE`` in the schema, and it is worth a third of the source.

    WikiSQL's condition values are what a crowdworker typed ("terrence ross"); the cells
    are what Wikipedia holds ("Terrence Ross"). sqlite's ``=`` on TEXT is case-sensitive,
    so without the collation a third of the gold queries matched nothing and the items
    were discarded as unanswerable when they were in fact correct. Declaring it in the
    schema also puts the rule where the model can see it.

    Turns red when: the type mapping drops the collation, or lowercases the data instead
    -- which would fix the match and change the answers the model is asked to produce.
    """
    built = _wikisql_item(wikisql_row(), 0)
    assert built is not None
    assert "COLLATE NOCASE" in built.context

    rows, error = run_query(built.context, built.gold)
    assert not error, error
    assert rows == (("Hawthorn FC",),)


def test_a_numeric_condition_is_quoted_and_still_matches_a_real_column() -> None:
    """Literals are quoted unconditionally, because affinity is applied to the comparison.

    ``"Year" = '2011'`` against a REAL column holding 2011.0 matches, so quoting costs
    nothing -- and quoting *conditionally* would let a value that merely looks numeric
    ("007", "1-2") become a different number on the way into the query.

    Turns red when: ``_literal`` starts emitting bare numerics for REAL columns.
    """
    built = _wikisql_item(
        wikisql_row(
            conds={"column_index": [2], "operator_index": [0], "condition": ["2011"]},
        ),
        0,
    )
    assert built is not None
    assert "\"Year\" = '2011'" in built.gold
    rows, error = run_query(built.context, built.gold)
    assert not error, error
    assert rows == (("Hawthorn FC",),)


def test_headers_with_spaces_quotes_and_twins_all_survive() -> None:
    """Wikipedia headers are captions, and two of them are often the same caption.

    Renaming to ``col0..colN`` -- what the reference evaluator does -- would delete the
    only thing that makes the question answerable from the schema alone. A duplicate
    would make ``CREATE TABLE`` fail outright, so the second occurrence is suffixed.

    Turns red when: identifier quoting stops doubling embedded quotes, or the dedup is
    removed and these items start failing at schema build (counted as ``failed``, which
    reads as a dialect problem rather than a header problem).
    """
    built = _wikisql_item(
        wikisql_row(
            header=('Score "final"', "Score", "Score"),
            types=("text", "text", "text"),
            rows=(("a", "b", "c"),),
            sel=2,
            conds={"column_index": [0], "operator_index": [0], "condition": ["a"]},
        ),
        0,
    )
    assert built is not None
    assert '"Score ""final"""' in built.context, "an embedded quote is doubled, not stripped"
    # `Score "final"` is a *different* header from `Score`, so only the latter pair are
    # twins: the first `Score` keeps its name and the second is suffixed. Numbering all
    # three would rename a column the question refers to by its real caption.
    assert '"Score" TEXT' in built.context
    assert '"Score (2)"' in built.context
    rows, error = run_query(built.context, built.gold)
    assert not error, error
    assert rows == (("c",),)


@pytest.mark.parametrize(
    ("row", "why"),
    [
        (
            wikisql_row(rows=tuple((f"a{i}", "b", 1.0) for i in range(MAX_TABLE_ROWS + 1))),
            "too big",
        ),
        (wikisql_row(header=("A", "", "C")), "blank header"),
        (wikisql_row(types=("text", "text")), "types and headers disagree"),
        (
            wikisql_row(conds={"column_index": [9], "operator_index": [0], "condition": ["x"]}),
            "condition column out of range",
        ),
        (
            wikisql_row(conds={"column_index": [1], "operator_index": [3], "condition": ["x"]}),
            "the OP placeholder, which is not an operator",
        ),
    ],
)
def test_a_table_that_cannot_be_rebuilt_faithfully_is_dropped_not_patched(row, why) -> None:
    """Refused rather than truncated, renamed, or guessed at.

    A table shortened to fit would be a question about data the model was not shown; a
    header invented to fill a blank would be a schema the question does not describe.
    Both produce items that are wrong for reasons unrelated to the model.

    Turns red when: any of these paths starts returning a best-effort item.
    """
    assert _wikisql_item(row, 0) is None, why


def test_the_gold_is_rebuilt_from_structure_and_not_from_the_display_string() -> None:
    """``sql.human_readable`` is display text that sqlite refuses.

    It emits ``WHERE Current slogan = SOUTH AUSTRALIA`` -- value unquoted, identifier
    unescaped -- which sqlite parses as a comparison between two column names and then
    errors on the missing column. Every such item would land in ``failed``, i.e. would
    be read as WikiSQL being dialect-incompatible rather than as us using the wrong
    field.

    Turns red when: the reader starts reading ``human_readable``.
    """
    built = _wikisql_item(wikisql_row(agg=3), 0)  # COUNT
    assert built is not None
    assert built.gold.startswith('SELECT COUNT("Home team") FROM table_2_1080 WHERE ')
    assert built.complexity == "COUNT"


# --- the mixture ---------------------------------------------------------------------


def test_a_source_without_rows_is_refused_for_a_scored_split_and_kept_for_training() -> None:
    """``sql-create-context`` trains the mapping and cannot score it.

    Naming it for ``test`` is an error rather than an empty contribution, because a
    source contributing zero items is indistinguishable in the output from a source
    scoring zero.

    Turns red when: ``has_data`` stops gating, at which point the evaluation quietly
    fills with items whose every answer is the empty result set.
    """
    with pytest.raises(DynQuantError, match=r"not scorable|without rows"):
        resolve_sources(["create-context"], split="test")

    assert [s.name for s in resolve_sources(["create-context"], split="train")] == [
        "create-context"
    ]
    assert all(s.has_data for s in resolve_sources(None, split="test"))
    assert len(resolve_sources(None, split="train")) == 3


def test_an_unknown_source_name_is_an_error_rather_than_an_empty_mixture() -> None:
    """Turns red when: unknown names are filtered out instead of raising, which turns a
    typo in a launch script into a run against a smaller set than the one it reports."""
    with pytest.raises(DynQuantError, match="unknown"):
        resolve_sources(["gretel", "spider"], split="test")


@pytest.mark.parametrize(
    ("total", "parts", "expected"),
    [(400, 2, [200, 200]), (401, 2, [201, 200]), (10, 3, [4, 3, 3]), (2, 3, [1, 1, 0])],
)
def test_the_limit_is_split_evenly_with_the_remainder_placed_deterministically(
    total, parts, expected
) -> None:
    """Two runs at the same limit must draw the same mixture, or a paired test is not paired.

    Turns red when: the remainder moves to the back or is distributed at random -- either
    of which makes McNemar on stored hits compare two different item sets.
    """
    assert _quotas(total, parts) == expected
    assert sum(_quotas(total, parts)) == total


def test_the_mixture_is_interleaved_so_a_truncated_run_still_sees_every_source(
    monkeypatch,
) -> None:
    """A prefix of a concatenation is an evaluation of whichever source came first.

    ``--limit`` applied downstream by something that does not know about sources, or a
    run that dies partway through, both take a prefix. Round-robin makes that prefix a
    proportional sample instead of one source.

    Turns red when: the round-robin is replaced by a concatenation, or the per-source
    quota stops being enforced -- this asserts both the balance and the order.
    """

    def fake_read_source(source, split, *, seed, cache_dir):
        for index in range(50):
            yield RawItem(
                task_id=f"{source.name}/{index}",
                question="q?",
                context=POPULATED,
                gold="SELECT a FROM t",
                source=source.name,
            )

    monkeypatch.setattr("dynquant.eval.text2sql.read_source", fake_read_source)

    examples = load_text2sql("test", sources=["gretel", "wikisql"], limit=10)
    assert len(examples) == 10
    assert [e.source for e in examples[:4]] == ["gretel", "wikisql", "gretel", "wikisql"]
    counts = {name: sum(e.source == name for e in examples) for name in ("gretel", "wikisql")}
    assert counts == {"gretel": 5, "wikisql": 5}


# --- decontamination ------------------------------------------------------------------
#
# The S2 driver has had a contamination check since phase 3 and it reported nothing on
# this mixture. It could not have reported anything: its markers are ``("gsm8k",
# "humaneval", "mbpp")`` and no SQL corpus name contains one, so the empty result was a
# check that could not fire rather than a mixture that passed. Measured properly, 189 of
# the 200 WikiSQL items this campaign scores are questions present in
# ``b-mc2/sql-create-context`` -- a community aggregate that ships one ``train`` split and
# never claimed to respect WikiSQL's boundary.
#
# So these tests are not about a hypothetical. They are about the ways the filter that
# replaced that check can be present and useless.


class _Split:
    """Enough of a ``datasets.Dataset`` for a source read: iterate, count, shuffle, slice.

    Shuffling is real rather than the identity, because one of the tests below turns on
    whether the caller shuffled before slicing, and a fake that returned itself would let
    an unshuffled prefix pass.
    """

    def __init__(self, rows: list[dict]) -> None:
        self.rows = list(rows)

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def shuffle(self, seed: int) -> _Split:
        order = list(range(len(self.rows)))
        random.Random(seed).shuffle(order)
        return _Split([self.rows[index] for index in order])

    def select(self, indices) -> _Split:
        return _Split([self.rows[index] for index in indices])


def fake_hub(monkeypatch, splits: dict[tuple[str, str], list[dict]]) -> None:
    """Serve ``{(repo, split): rows}`` in place of the Hub.

    Patched on the ``datasets`` module rather than on this repository's callers, because
    both :func:`read_source` and :func:`evaluation_questions` do their own
    ``from datasets import load_dataset`` at call time -- and a test that patched only one
    of them would be comparing the two against different data, which is the exact mistake
    the first version of ``experiments/phase4/leak_text2sql.py`` made.

    Skipped rather than faked when the module is absent, because the attach point *is* the
    module: with no ``datasets`` installed there is nothing for the callers to import and
    nothing for this to patch, so a stub would test the stub. The base CI matrix installs
    core and pytest only, which is where that state occurs; the ``transformers`` job
    installs ``datasets`` and asserts this file does not skip, so the skip cannot become
    the way these tests never run.
    """
    datasets = pytest.importorskip("datasets")

    def load_dataset(repo, config=None, **kwargs):
        return _Split(splits[(repo, kwargs["split"])])

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)


def gretel_row(question: str, gold: str = "SELECT a FROM t") -> dict:
    return {"id": question, "sql_prompt": question, "sql_context": POPULATED, "sql": gold}


def create_context_row(question: str) -> dict:
    return {"question": question, "context": BARE, "answer": "SELECT a FROM t"}


EVAL_QUESTION = "Which staff are in engineering?"
WIKISQL_EVAL_QUESTION = "Who was the home team?"


def three_sources(train: dict[str, list[dict]] | None = None) -> dict[tuple[str, str], list[dict]]:
    """A hub holding one question per evaluated source, plus whatever training rows.

    The ``test`` splits are the point: they are what the banned set is built from, and
    every training row below is contaminated or clean relative to *those two* strings.
    """
    train = train or {}
    return {
        ("gretelai/synthetic_text_to_sql", "test"): [gretel_row(EVAL_QUESTION)],
        ("gretelai/synthetic_text_to_sql", "train"): train.get("gretel", []),
        ("Salesforce/wikisql", "test"): [wikisql_row(question=WIKISQL_EVAL_QUESTION)],
        ("Salesforce/wikisql", "train"): train.get("wikisql", []),
        ("b-mc2/sql-create-context", "train"): train.get("create-context", []),
    }


@pytest.mark.parametrize(
    ("left", "right", "same"),
    [
        ("How many staff?", "how many staff", True),
        ("How  many\n staff ?", "How many staff?", True),
        ("Count the rows.", "Count the rows!", True),
        # And nothing beyond writing. A stemmer would fold these two, and folding them
        # deletes a legitimate training row for resembling a test item.
        ("List the orders", "List the order", False),
        ("How many staff in eng?", "How many staff in ops?", False),
        # Same words, different question. Found by mutation: a fold to a sorted token
        # set survived every case above it, and it is the shape a "make the match a bit
        # more robust" edit takes.
        ("Is a manager an employee?", "Is an employee a manager?", False),
    ],
)
def test_question_key_folds_writing_and_not_meaning(left, right, same) -> None:
    """The looser this gets, the more training data it silently deletes.

    Both directions cost something and only one of them is visible. Too strict and a
    leaked item survives, which shows up as an inflated accuracy. Too loose and legitimate
    rows are dropped for resembling the test set, which shows up as nothing at all -- a
    slightly smaller training set and a number nobody can attribute.

    Turns red when: the fold stops normalising case, punctuation or whitespace; or starts
    stemming, matching on a token set, or matching on a similarity threshold.
    """
    assert (question_key(left) == question_key(right)) is same


def test_question_columns_match_the_readers(monkeypatch) -> None:
    """The banned set has to name the same strings the readers put in ``RawItem.question``.

    ``evaluation_questions`` reads one raw column per source instead of running the
    reader, because synthesising 16k WikiSQL tables to collect 16k question strings is
    minutes of work for something that takes seconds. The cost of that shortcut is a
    second place the column name is written down, and if the two drift the filter compares
    training questions against a set of *something else* -- an empty intersection, a clean
    report, and a contaminated fine-tune.

    Turns red when: a reader changes which field it reads, or ``_QUESTION_COLUMN`` is
    edited without it -- either direction, because this asserts equality rather than
    containment.
    """
    fake_hub(monkeypatch, three_sources())

    through_the_readers = {
        question_key(item.question)
        for source in resolve_sources(None, split="test")
        for item in read_source(source, "test", seed=0, cache_dir=None)
    }
    assert through_the_readers == {question_key(EVAL_QUESTION), question_key(WIKISQL_EVAL_QUESTION)}
    assert evaluation_questions() == through_the_readers


def test_a_holdout_sources_banned_rows_are_the_shuffled_slice_not_the_prefix(monkeypatch) -> None:
    """A single-split source's test set is a slice of a *shuffle*, and the seed decides it.

    Reading an unshuffled prefix here would name 4000 rows that are not the test set, ban
    those, and leave the real ones in training -- a decontamination that runs, reports a
    healthy count, and protects nothing.

    Nothing in the registry reaches this branch today: the only source with a ``holdout``
    is ``create-context``, which has no rows and is therefore refused for ``test``. It is
    exercised through an injected source because the branch is one edit away from being
    live -- a data-bearing single-split corpus is the obvious next addition -- and an
    untested branch that is currently unreachable is how it would arrive already broken.

    Turns red when: the shuffle is dropped here, or ``read_source`` starts slicing at a
    different point in its pipeline than this does.
    """
    rows = [create_context_row(f"question number {index}") for index in range(20)]
    injected = Source(
        name="create-context",  # its ``_QUESTION_COLUMN`` entry, reused
        repo="fake/holdout",
        has_data=True,
        splits={"train": "train", "test": "train"},
        reader=_read_create_context,
        holdout=5,
    )
    monkeypatch.setitem(SOURCES, "held-out", injected)
    fake_hub(monkeypatch, {("fake/holdout", "train"): rows})

    banned = evaluation_questions(["held-out"], seed=0)
    through_read_source = {
        question_key(item.question)
        for item in read_source(injected, "test", seed=0, cache_dir=None)
    }

    assert len(banned) == 5
    assert banned == through_read_source
    assert banned != {question_key(f"question number {index}") for index in range(5)}, (
        "an unshuffled prefix would name five rows that are not the test set"
    )


def test_a_training_row_that_asks_an_evaluation_question_is_dropped_and_counted(
    monkeypatch,
) -> None:
    """The filter, doing the one thing it exists for, and saying that it did.

    Counted rather than only dropped, because a run whose manifest carries no number is
    indistinguishable from a run where the filter stopped working -- and on this mixture
    the expected count is 189 of 200, so "zero" has to be readable as suspicious rather
    than as clean.

    Turns red when: the comparison stops going through ``question_key`` (the two spellings
    below would stop matching), or the drop stops incrementing ``contaminated``.
    """
    train = {
        "gretel": [
            gretel_row("which staff are in engineering"),  # the eval question, rewritten
            gretel_row("How many departments are there?"),
        ]
    }
    fake_hub(monkeypatch, three_sources(train))

    tallies: dict[str, SourceTally] = {}
    kept = load_text2sql("train", sources=["gretel"], tallies=tallies)

    assert [item.question for item in kept] == ["How many departments are there?"]
    assert tallies["gretel"].contaminated == 1
    assert tallies["gretel"].kept == 1
    assert tallies["gretel"].seen == 2

    # And ``decontaminate=False`` still measures it. That is the seam
    # ``experiments/phase4/leak_text2sql.py`` reads the untreated mixture through; if the
    # default became unconditional, the scan would report the leak it had just removed.
    measuring: dict[str, SourceTally] = {}
    untreated = load_text2sql("train", sources=["gretel"], decontaminate=False, tallies=measuring)
    assert len(untreated) == 2
    assert measuring["gretel"].contaminated == 0


def test_a_contaminated_row_does_not_consume_a_quota_slot(monkeypatch) -> None:
    """Dropped before admission, not filtered out of the result afterwards.

    Filtering after the quota is met costs items: the mixture asks for N per source, one
    slot goes to a row that is then removed, and the run trains on N-1 while reporting N.
    Small on one source and not small across three at 50 000 examples -- and invisible,
    because the shortfall reads as an admission rate.

    Turns red when: the check moves below ``admit``, or the drop starts incrementing
    ``kept``.
    """
    train = {
        "gretel": [
            gretel_row(EVAL_QUESTION),
            gretel_row("first clean question"),
            gretel_row("second clean question"),
        ]
    }
    fake_hub(monkeypatch, three_sources(train))

    tallies: dict[str, SourceTally] = {}
    kept = load_text2sql("train", sources=["gretel"], limit=2, tallies=tallies)

    # Sorted, because ``read_source`` shuffles: what is asserted is that both clean rows
    # survived a quota of two, not the order they arrived in.
    assert sorted(item.question for item in kept) == [
        "first clean question",
        "second clean question",
    ]
    assert tallies["gretel"].kept == 2, "the dropped row was not admitted and then removed"
    assert tallies["gretel"].contaminated == 1


def test_the_evaluation_set_is_not_decontaminated_against_itself(monkeypatch) -> None:
    """Applied to ``test``, the filter removes all of it -- so it is refused, not ignored.

    Ignoring the argument would be the friendlier implementation and the worse one: a
    caller who passed it has misunderstood which side is being protected, and silently
    doing nothing leaves them believing something was done.

    Turns red when: the default stops being split-dependent (the first half empties the
    evaluation set), or the refusal degrades to a warning.
    """
    fake_hub(monkeypatch, three_sources())

    assert [item.question for item in load_text2sql("test", sources=["gretel"])] == [EVAL_QUESTION]

    with pytest.raises(DynQuantError, match="only meaningful on 'train'"):
        load_text2sql("test", sources=["gretel"], decontaminate=True)


def test_the_banned_set_comes_from_the_evaluated_sources_not_the_training_list(
    monkeypatch,
) -> None:
    """Two failures at once, and the first one raises rather than mismeasures.

    ``sources`` names what *trains*, and the training list legitimately contains
    ``create-context``, which ``resolve_sources`` refuses for ``test``. Threading it into
    the banned set turns "decontaminate this mixture" into an error about a source nobody
    is evaluating -- which is what the first version of this did.

    The second half is the reason the filter exists at all: ``create-context`` is where the
    WikiSQL evaluation questions actually are, so the row banned here is one whose question
    comes from a *different* source's test split. A banned set built per source would leave
    it in.

    Turns red when: ``sources`` is passed through to ``evaluation_questions``, or the
    banned set is rebuilt per source inside the loop.
    """
    train = {
        "gretel": [gretel_row("a clean gretel question")],
        "wikisql": [wikisql_row(question="a clean wikisql question")],
        "create-context": [
            create_context_row(WIKISQL_EVAL_QUESTION),
            create_context_row("a clean create-context question"),
        ],
    }
    # The real holdout is 4000 rows out of 78 577 and this fake corpus has two, so
    # ``train`` would be ``range(4000, 2)`` -- empty, and the source would contribute
    # nothing without saying so. Set aside here rather than worked around, because the
    # holdout is what the test above pins and this test is about something else.
    monkeypatch.setitem(SOURCES, "create-context", replace(SOURCES["create-context"], holdout=0))
    fake_hub(monkeypatch, three_sources(train))

    tallies: dict[str, SourceTally] = {}
    kept = load_text2sql("train", sources=list(DEFAULT_TRAIN), tallies=tallies)

    assert WIKISQL_EVAL_QUESTION not in [item.question for item in kept]
    assert tallies["create-context"].contaminated == 1
    assert tallies["gretel"].contaminated == 0
    assert tallies["wikisql"].contaminated == 0
