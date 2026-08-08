"""Text-to-SQL, scored by running the query rather than by comparing its text.

Why execution and not string match
----------------------------------
``SELECT a, b FROM t`` and ``SELECT t.a, t.b FROM t AS t`` are the same query and
differ in every character a string comparison looks at. Exact match on SQL text
therefore measures formatting agreement with one dataset's authors, and a quantized
model that answers correctly in a different shape is scored as broken. Since the whole
point here is to detect the accuracy a quantization method *costs*, a metric that fires
on formatting is a metric that manufactures differences.

So the schema is materialised in memory, both queries are run against it, and the
result sets are compared. That needs a database, and the dataset does not ship one --
it ships ``CREATE TABLE`` and ``INSERT`` statements as prompt context, which is enough
to build one per item.

Three sources, mixed, and why the evaluation set is a subset
------------------------------------------------------------
The mixture and the per-source admission rules live in
:mod:`dynquant.eval.text2sql_sources`; the short version is that a single synthetic
dataset measures one generator's idea of what SQL looks like, so training and scoring
both draw from Gretel (invented schemas), WikiSQL (real Wikipedia tables) and
sql-create-context (the community aggregate).

They are not interchangeable. ``sql-create-context`` ships bare ``CREATE TABLE``
statements with no rows, so every query over it runs against empty tables and almost
every result set is ``[]`` -- and then any prediction that also returns nothing scores
as correct. A model emitting ``SELECT 0`` would land near the ceiling. Screened on
2 000 shuffled rows it looked usable (99.6% of schemas build, 97.9% of golds execute)
and it is not: its 27.2% "returns rows" are aggregates over empty tables returning
``[(0,)]`` regardless of whether the aggregate was the right one. It trains, it does
not score.

So :func:`load_text2sql` admits an evaluation item only when **the database holds rows
and the gold query finds some**. Both halves are needed -- the second alone lets
``[(0,)]`` through. The filter depends only on the reference answer and the schema,
never on any model's output, so it is the same set of items for every arm of every
comparison, computed once before a model is loaded. The training split drops the filter,
because a gold query that returns nothing is still correct supervision.

Nothing is silently truncated: :attr:`Text2SqlResult.total` is the kept count,
:func:`load_text2sql` logs the per-source tally, and
:attr:`Text2SqlResult.by_source` decomposes every headline number into the sources that
produced it -- without which a mixture hides a model that learned one source and
none of the others.

What this does not claim
------------------------
Column *order* within a row is significant here: ``SELECT name, age`` and
``SELECT age, name`` compare unequal. Spider's official evaluator has the same
property. It is stricter than a human grader would be, and it is applied identically to
every arm, so it cannot favour one -- but a headline percentage from this module is not
comparable to a published Spider execution-accuracy number.

Row order is significant only when the gold query has an ``ORDER BY``, which is the
standard convention: a query that did not ask for an order has no order to get wrong.

The chance floor is zero. Unlike :mod:`dynquant.eval.casehold`, a destroyed model
scores 0% rather than returning to a guessing baseline, so the percentages here have
the same dynamic range as GSM8K's and :attr:`Text2SqlResult.hits` carries the paired
test as usual.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field, replace
from itertools import zip_longest
from typing import TYPE_CHECKING, Any, Literal

from dynquant._logging import get_logger

from .harness import (
    EvalConfig,
    Prompt,
    generate_batched,
    reasoning_state,
    render_chat,
    strip_reasoning,
)
from .text2sql_sources import (
    MAX_CONTEXT_CHARS,
    RawItem,
    SourceTally,
    is_ordered,
    is_readable_query,
    read_source,
    resolve_sources,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "DEFAULT_CHAT_CONFIG",
    "DEFAULT_COMPLETION_CONFIG",
    "FEWSHOT_STOP",
    "Text2SqlExample",
    "Text2SqlResult",
    "admit",
    "build_prompt",
    "database_has_rows",
    "execution_match",
    "extract_sql",
    "format_training_text",
    "instruction",
    "load_text2sql",
    "run_query",
]

_log = get_logger(__name__)

DATASET = "gretelai/synthetic_text_to_sql"

FEWSHOT_STOP = "\n\n"
"""Exemplars are separated by a blank line, so this is the model's own turn boundary.

Not ``";"``. A semicolon terminates the *first* statement, and the separator a few-shot
prefix teaches is the blank line -- a stop copied from inside the answer rather than
from between the exemplars is how a GSM8K arm silently lost 24 points once.
"""

DEFAULT_COMPLETION_CONFIG = EvalConfig(
    max_new_tokens=256,
    batch_size=32,
    max_prompt_tokens=3072,
    stop_sequences=(FEWSHOT_STOP,),
)

DEFAULT_CHAT_CONFIG = EvalConfig(
    max_new_tokens=384,
    batch_size=32,
    max_prompt_tokens=3072,
    add_special_tokens=False,
    stop_sequences=(),
)

_INSTRUCTION = (
    "Write a single SQL query that answers the question, using only the tables in the "
    "schema. Return just the query, with no explanation.\n\n"
    "Schema:\n{context}\n\nQuestion: {question}"
)

_COMPLETION_BLOCK = "Schema:\n{context}\n\nQuestion: {question}\nSQL: "

_FENCE = re.compile(r"```(?:sql|sqlite|postgresql|mysql)?\s*\n(.*?)(?:```|\Z)", re.DOTALL | re.I)
_SQL_START = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)
_TRAILING = re.compile(r"[\s;]*$")
_ORDER_BY = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)

#: Statements the model's query is not allowed to reach. ``ATTACH`` is the one that
#: matters: an in-memory database is not a sandbox if the query can attach a file on
#: disk, and unlike the Python tasks there is no subprocess boundary here to fall back
#: on. The rest are denied because nothing a ``SELECT`` needs is behind them.
_DENIED_ACTIONS = frozenset({sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH, sqlite3.SQLITE_PRAGMA})

_MAX_VM_STEPS = 2_000_000
"""Interpreter steps before a query is aborted, via ``set_progress_handler``.

A model that writes an accidental cross join over a handful of literal rows still
finishes; one that writes a recursive CTE with no base case does not, and a wall-clock
timeout would make the score depend on how loaded the box was. Steps are deterministic,
so the same generation scores the same way on every run and on every machine.
"""

_ROWS_CAP = 10_000
"""Rows fetched before a result is called too large to be a plausible answer."""

SHOT_SPLIT = "shots"
FEWSHOT_POOL = 64
"""Items loaded when the caller asks for the few-shot pool.

Small on purpose: the pool exists to be sampled from, and every item in it costs a gold
query execution at load time. Sixty-four is far more than the two or three exemplars any
framing uses, and it is drawn under the same seed, so which ones get picked is stable
across runs.
"""


@dataclass(frozen=True, slots=True)
class Text2SqlExample:
    task_id: str
    question: str
    context: str
    """``CREATE TABLE`` and ``INSERT`` statements, shown to the model *and* used to
    build the database it is scored against. Both arms of every comparison see the
    same string."""

    gold: str
    gold_rows: tuple[tuple[Any, ...], ...] = ()
    """The reference result set, computed once at load time.

    Cached rather than recomputed per arm because it is the same for all of them and
    because re-running it per measurement point is how two arms end up scored against
    two different databases.
    """

    ordered: bool = False
    """Whether row order is significant -- true iff the gold query has an ``ORDER BY``."""

    source: str = ""
    """Which dataset the item came from.

    Carried on the item rather than inferred from its position, because the mixture is
    interleaved and because :attr:`Text2SqlResult.by_source` has to survive any
    reordering or sub-setting a caller applies.
    """

    domain: str = ""
    complexity: str = ""
    """Gretel's own tags. Never used for scoring; they make a failure breakdown
    possible without re-reading the dataset."""


@dataclass(slots=True)
class Text2SqlResult:
    """One measurement point."""

    label: str
    correct: int
    total: int
    unparseable: int
    """Generations with no ``SELECT`` or ``WITH`` in them at all.

    Separated from wrong answers because it means something different: the model has
    stopped emitting SQL rather than emitting the wrong SQL. It is the earliest visible
    sign that quantization has broken format compliance, and on a task with a zero
    chance floor it is the difference between "answers badly" and "has stopped
    answering".
    """

    errored: int
    """Parseable SQL that sqlite refused to run. Wrong, but wrong in a recoverable way
    -- a syntax slip or a hallucinated column, rather than a collapse."""

    hits: list[bool] = field(default_factory=list)
    """Per-item correctness in dataset order, always recorded.

    Two arms scored on one fixed item set is a paired design, and the pairing is where
    the power is: most items are answered the same way by both, and only the flips
    carry information. Recovering this after the GPU-hours are spent is impossible, so
    it is never sampled. :mod:`dynquant.eval.compare` does the McNemar arithmetic.
    """

    exact: int = 0
    """Secondary: normalised-text equality with the gold query.

    Recorded, never headlined. It is here because a large gap between ``exact`` and
    ``correct`` is the evidence that execution scoring was worth its cost, and because
    a fine-tune that moves ``exact`` far more than ``correct`` has taught the model the
    dataset's formatting rather than the task.
    """

    unfinished_reasoning: int = 0
    """Generations that opened a reasoning trace and never closed it.

    A subset of :attr:`unparseable`, and the reason that counter stays readable on a
    model that thinks before it answers. These items have no answer region at all: the
    decode budget ran out mid-deliberation, so the model never got as far as a query.
    That is a decode setting, not a model failing the task, and without this count it
    arrives in the record wearing the costume of one.

    Zero for every model that does not emit a trace. Non-zero anywhere means the
    headline is bounded above by ``1 - unfinished_reasoning / total``, and the budget
    rather than the method is what the arm is measuring.
    """

    predictions: list[str] = field(default_factory=list)

    by_source: dict[str, tuple[int, int]] = field(default_factory=dict)
    """``source -> (correct, total)``, and the reason the mixture is safe to report.

    A single number over three datasets cannot say whether a model learned the task or
    learned Gretel. Worse for this campaign's purpose: a quantization method that
    damages one source's distribution and not another's shows up here as a couple of
    points on the headline, and as the whole story in this dict. Computed from
    :attr:`hits`, so it can never disagree with the headline.
    """

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def summary(self) -> str:
        parts = "  ".join(
            f"{name} {correct / total:.1%} ({total})" if total else f"{name} -"
            for name, (correct, total) in sorted(self.by_source.items())
        )
        headline = (
            f"{self.label:<28} {self.accuracy:6.2%}  "
            f"({self.correct}/{self.total} execution match, "
            f"{self.errored} would not run, {self.unparseable} no query)"
        )
        lines = headline
        if self.unfinished_reasoning:
            share = self.unfinished_reasoning / self.total if self.total else 0.0
            lines += (
                f"\n{'':<28} {self.unfinished_reasoning} never finished "
                f"reasoning ({share:.1%}) -- the headline is capped at {1 - share:.1%}"
            )
        return lines + (f"\n{'':<28} by source: {parts}" if parts else "")

    def as_dict(self) -> dict[str, Any]:
        """The three failure modes separately, because they mean different things.

        A single "wrong" count would conflate a query that ran and answered
        differently, a query sqlite refused, and a generation with no SQL in it. Only
        the third says the model stopped doing the task, and it is the one a
        quantization regression shows up in first. ``exact`` rides along so the gap
        between it and ``correct`` stays visible in the record rather than only in a
        rerun.
        """
        return {
            "label": self.label,
            "accuracy": self.accuracy,
            "correct": self.correct,
            "total": self.total,
            "unparseable": self.unparseable,
            "errored": self.errored,
            "exact": self.exact,
            "unfinished_reasoning": self.unfinished_reasoning,
            "by_source": {k: list(v) for k, v in sorted(self.by_source.items())},
        }


def _connect(context: str) -> sqlite3.Connection:
    """An in-memory database holding ``context``, with the escape hatches shut."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(_TRAILING.sub("", context) + ";")
    conn.set_authorizer(
        lambda action, *_: sqlite3.SQLITE_DENY if action in _DENIED_ACTIONS else sqlite3.SQLITE_OK
    )
    steps = [0]

    def _tick() -> int:
        steps[0] += 1000
        return 1 if steps[0] > _MAX_VM_STEPS else 0

    conn.set_progress_handler(_tick, 1000)
    return conn


def _normalise(value: Any) -> Any:
    """Make one cell comparable across two queries that computed it differently.

    ``COUNT(*)`` returns an int and ``SUM(1)`` returns an int, but ``AVG`` returns a
    float and ``ROUND(AVG(x), 2)`` returns a float that differs in the last bits
    depending on the order the rows were summed in. Rounding to 6 decimals is well
    inside any difference that reflects a genuinely different query and well outside
    float noise.
    """
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def run_query(context: str, sql: str) -> tuple[tuple[tuple[Any, ...], ...] | None, str]:
    """Run ``sql`` against a fresh database built from ``context``.

    Returns ``(rows, error)`` with exactly one of them meaningful. A fresh connection
    per call, because a query is allowed to be a ``SELECT`` and nothing else but the
    schema build is not free to be mutated by the previous item.
    """
    try:
        conn = _connect(context)
    except Exception as exc:  # noqa: BLE001 -- a broken context is data, not a bug
        return None, f"schema: {type(exc).__name__}: {exc}"
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(_ROWS_CAP)
        return tuple(tuple(_normalise(cell) for cell in row) for row in rows), ""
    except Exception as exc:  # noqa: BLE001 -- the model's SQL failing is the measurement
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()


def execution_match(
    gold_rows: Sequence[tuple[Any, ...]],
    pred_rows: Sequence[tuple[Any, ...]] | None,
    *,
    ordered: bool,
) -> bool:
    """Whether two result sets are the same answer.

    ``ordered`` comes from the gold query having an ``ORDER BY``: a query that did not
    ask for an order has no order to get wrong, so the rows are compared as a multiset.
    A multiset and not a set -- collapsing duplicates would score ``SELECT DISTINCT x``
    equal to ``SELECT x``, which is a different question.
    """
    if pred_rows is None:
        return False
    if ordered:
        return tuple(gold_rows) == tuple(pred_rows)
    return sorted(gold_rows, key=repr) == sorted(pred_rows, key=repr)


def extract_sql(text: str) -> str:
    """Pull one SQL statement out of a generation.

    A fenced block wins when there is one, since an instruct model told to return only
    the query still often wraps it. Otherwise the text is cut at the first ``SELECT``
    or ``WITH`` -- which drops a preamble like "Here is the query:" without needing to
    enumerate the ways a model can write that sentence.

    Returns ``""`` when neither is found, which the caller counts as unparseable rather
    than as wrong.

    A reasoning trace is cut before any of that -- see
    :func:`~dynquant.eval.harness.strip_reasoning`. Without it the first ``SELECT`` in the
    text is whatever the model wrote while still deciding, and on ``LFM2.5-8B-A1B`` that
    was worth 34 points of execution accuracy.
    """
    text = strip_reasoning(text)
    fenced = _FENCE.search(text)
    body = fenced.group(1) if fenced else text
    start = _SQL_START.search(body)
    if start is None:
        # The fence may have held prose while the SQL sits after it.
        start = _SQL_START.search(text)
        if start is None:
            return ""
        body = text
    statement = body[start.start() :]
    # One statement. A trailing "```" or a second sentence is cut here rather than
    # being handed to sqlite, which would refuse the whole thing over the tail.
    semicolon = statement.find(";")
    if semicolon != -1:
        statement = statement[:semicolon]
    return statement.split("```")[0].strip()


def _normalise_text(sql: str) -> str:
    """Whitespace- and case-folded SQL, for the secondary exact-match count only."""
    return " ".join(_TRAILING.sub("", sql).lower().split())


def database_has_rows(context: str) -> bool:
    """Whether the schema in ``context`` actually holds data.

    The other half of the admission rule, and the half that is easy to leave out.
    ``SELECT COUNT(*)`` over an empty schema returns ``[(0,)]`` -- one row, which passes
    a "gold returns rows" test, and which any model writing any count over any table
    reproduces exactly. Checking the query alone therefore admits precisely the items
    that cannot distinguish a working model from a broken one.

    Every base table is counted rather than the first one, because a schema can define
    a populated fact table beside empty lookup tables.
    """
    conn = _connect(context)
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        ]
        for table in tables:
            if conn.execute(f'SELECT EXISTS(SELECT 1 FROM "{table}")').fetchone()[0]:
                return True
        return False
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def _is_degenerate(rows: tuple[tuple[Any, ...], ...]) -> bool:
    """A single row of nothing but NULLs and zeros: an answer that answers nothing.

    ``SELECT AVG(x) FROM t WHERE <no match>`` gives ``(None,)`` and ``SELECT COUNT(*)``
    over the same gives ``(0,)``. Both are one row, so both survive a "did the gold
    return rows" check, and both are produced by any aggregate over any condition that
    happens to match nothing -- including one written by a model that did not read the
    question. Kept out of the evaluation set for the same reason the empty result set
    is: a correct answer that is free to guess cannot show a regression.
    """
    return len(rows) == 1 and all(cell is None or cell == 0 for cell in rows[0])


def admit(item: RawItem, *, require_rows: bool, tally: SourceTally) -> Text2SqlExample | None:
    """Decide one raw item, recording *why* when it is refused.

    The counts matter as much as the decision: a source that suddenly stops
    contributing looks identical to a source nobody selected unless the tally says
    which of "would not execute", "no rows in the database" and "gold matched nothing"
    it was.
    """
    tally.seen += 1
    if len(item.context) > MAX_CONTEXT_CHARS:
        tally.too_long += 1
        return None

    # Before execution, and in every split including `train`. A DML gold would run
    # cleanly and return no rows, so the row filter already keeps it out of the
    # evaluation -- but as `empty_result`, which describes a different problem. Training
    # has no row filter at all, so without this check a tenth of one source teaches the
    # model to answer with `UPDATE`, which `extract_sql` reads as no answer at all.
    if not is_readable_query(item.gold):
        tally.not_a_query += 1
        return None

    rows, error = run_query(item.context, item.gold)
    if error:
        tally.failed += 1
        kind = error.split(":", 1)[0][:60]
        tally.errors[kind] = tally.errors.get(kind, 0) + 1
        return None

    if require_rows:
        if not database_has_rows(item.context):
            tally.no_data += 1
            return None
        if not rows:
            tally.empty_result += 1
            return None
        if _is_degenerate(rows):
            tally.degenerate += 1
            return None

    tally.kept += 1
    return Text2SqlExample(
        task_id=item.task_id,
        question=item.question,
        context=item.context,
        gold=item.gold,
        gold_rows=rows or (),
        ordered=is_ordered(item.gold),
        source=item.source,
        domain=item.domain,
        complexity=item.complexity,
    )


def _quotas(total: int, parts: int) -> list[int]:
    """Split ``total`` across ``parts`` as evenly as possible, remainder to the front.

    Deterministic in source order, so two runs at the same limit draw the same mixture
    and a paired test compares the same items.
    """
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def load_text2sql(
    split: str = "test",
    *,
    sources: Sequence[str] | None = None,
    limit: int | None = None,
    seed: int = 0,
    require_rows: bool | None = None,
    cache_dir: str | None = None,
) -> list[Text2SqlExample]:
    """Load a balanced mixture of text-to-SQL items.

    Args:
        sources: Names from :data:`~dynquant.eval.text2sql_sources.SOURCES`. ``None``
            takes every source for ``train`` and only the ones carrying rows for
            ``test`` -- a source of bare schemas cannot be scored by execution, and
            asking for one here is refused rather than quietly contributing nothing.
        limit: Total items, **divided evenly across the sources and interleaved**. Two
            reasons it is not a plain prefix of the concatenation: the headline number
            would otherwise be an average weighted by whichever source survives its
            filter most often, and a run cut short would be an evaluation of whichever
            source happened to come first.
        seed: Fixed, and part of the identity of the evaluation set -- two arms
            compared across a change of seed are not a paired comparison. Shuffling
            happens before anything is taken, because all three sources arrive grouped.
        require_rows: The evaluation admission rule (see :func:`database_has_rows`).
            Defaults to on for every split except ``train``, where a gold query that
            returns nothing is still correct supervision and dropping it would throw
            away most of two sources.
    """
    if split == SHOT_SPLIT:
        # A pseudo-split, and the alternative was worse. `_pick_shots` loads a whole
        # pool to sample two exemplars from it, and loading the training mixture means
        # executing every gold query in ~230k rows -- hours, to choose two prompts.
        # Drawn from `train`, which is disjoint from `test` in both data-bearing
        # sources, so the exemplars cannot be items the run is scored on.
        split, limit = "train", limit or FEWSHOT_POOL
        require_rows = True if require_rows is None else require_rows

    chosen = resolve_sources(sources, split=split)
    if require_rows is None:
        require_rows = split != "train"
    quotas = _quotas(limit, len(chosen)) if limit is not None else [None] * len(chosen)

    per_source: list[list[Text2SqlExample]] = []
    tallies: dict[str, SourceTally] = {}
    for source, quota in zip(chosen, quotas, strict=True):
        tally = SourceTally()
        tallies[source.name] = tally
        kept: list[Text2SqlExample] = []
        for item in read_source(source, split, seed=seed, cache_dir=cache_dir):
            example = admit(item, require_rows=require_rows, tally=tally)
            if example is not None:
                kept.append(example)
                if quota is not None and len(kept) >= quota:
                    break
        per_source.append(kept)

    # Round-robin rather than concatenate: a run that dies partway through, or a
    # `--limit` applied downstream by something that does not know about sources, still
    # sees every source in proportion.
    examples: list[Text2SqlExample] = []
    for row in zip_longest(*per_source):
        examples.extend(item for item in row if item is not None)

    for name, tally in tallies.items():
        _log.info(
            "text2sql %s/%s: kept %d of %d seen (%d not a query, %d no rows in db, "
            "%d gold matched nothing, %d all-null/zero, %d would not run, %d over %d chars)",
            split,
            name,
            tally.kept,
            tally.seen,
            tally.not_a_query,
            tally.no_data,
            tally.empty_result,
            tally.degenerate,
            tally.failed,
            tally.too_long,
            MAX_CONTEXT_CHARS,
        )
    if limit is not None and len(examples) < limit:
        _log.warning(
            "text2sql %s: asked for %d, got %d -- a source ran out of admissible items",
            split,
            limit,
            len(examples),
        )
    return examples


def build_prompt(
    example: Text2SqlExample,
    tokenizer: Any,
    *,
    style: Literal["chat", "completion"],
    shots: Sequence[Text2SqlExample] = (),
) -> Prompt:
    """Render one item. The chat framing comes back as token ids -- see
    :func:`~dynquant.eval.harness.render_chat`.

    Both framings use ``shots``. The chat branch did not, and that was not merely a
    weaker prompt: the CLI resolves the shot pool, passes ``shots=2``, and writes
    ``"shots": 2`` into the result JSON and the run manifest, so a chat run recorded a
    two-shot prompt and sent a zero-shot one. A provenance error, not an accuracy one --
    every arm shared it, so no comparison in the campaign could have surfaced it, and the
    record would have outlived the run.

    In chat the exemplars are prior turns rather than text, which is the framing an
    instruct model was trained to read. The assistant turns are bare SQL, and that is
    also what a reasoning model needs to see: it demonstrates an answer with no
    deliberation in front of it. Templates strip a previous turn's reasoning block by
    default -- ``LFM2.5``'s splits on ``</think>`` and keeps the tail -- so a bare-SQL
    exemplar survives templating unchanged rather than being reinterpreted as a trace.
    """
    if style == "completion":
        blocks = [
            _COMPLETION_BLOCK.format(context=shot.context, question=shot.question)
            + f"{shot.gold}\n"
            for shot in shots
        ]
        blocks.append(_COMPLETION_BLOCK.format(context=example.context, question=example.question))
        return FEWSHOT_STOP.join(blocks)
    messages: list[dict[str, str]] = []
    for shot in shots:
        messages.append({"role": "user", "content": instruction(shot)})
        messages.append({"role": "assistant", "content": shot.gold})
    messages.append({"role": "user", "content": instruction(example)})
    return render_chat(tokenizer, messages)


def instruction(example: Text2SqlExample) -> str:
    """The user turn, rendered once and shared by everything that needs it.

    Three callers: the chat evaluation, the training-text builder, and the fine-tune
    driver's row assembler in ``scripts/run_s2_finetune.py``. They have to agree
    character for character -- a model trained on one phrasing and asked in another is
    being measured on the gap between them, and the gap is invisible in the output
    because both halves look correct on their own.
    """
    return _INSTRUCTION.format(context=example.context, question=example.question)


def format_training_text(example: Text2SqlExample) -> tuple[str, str]:
    """Return ``(prompt, completion)`` for supervised fine-tuning.

    Split rather than concatenated so the trainer masks the loss to the completion. The
    prompt here is a schema -- often several ``CREATE TABLE`` statements and the
    ``INSERT``\\ s behind them, up to the 3 072-character cap -- against a completion of
    one query. Training on the whole sequence spends the run modelling DDL the model is
    shown at evaluation time anyway.

    The prompt is ``_COMPLETION_BLOCK``, which is the same string
    :func:`build_prompt` ends its few-shot prefix with, and the completion carries no
    leading space because that block already ends in one. So ``prompt + completion`` is
    exactly the exemplar the completion-style evaluation teaches -- pinned by
    ``test_the_training_pair_is_the_exemplar_the_completion_prompt_teaches``, because a
    fine-tune that trains on a framing the evaluation does not use scores badly for a
    reason no arm of the comparison can reveal: every arm shares the fine-tune.

    ``(prompt, completion)`` and not a rendered chat string, which is what this returned
    first. Two things were wrong with that. It made this the only training path in the
    package going through ``apply_chat_template(..., tokenize=False)``, a round trip
    transformers itself calls unsafe and which cost 120 of 164 HumanEval problems on
    Ministral once -- see :func:`~dynquant.eval.harness.render_chat`. And the signature
    did not match the other three tasks', so the ``TaskSpec`` in
    ``experiments/four_point/tasks.py`` could not hold it: the function was exported,
    untested and unusable, the same shape as the CLI's hand-copied task list.

    The chat framing the campaign actually fine-tunes on is built by
    ``scripts/run_s2_finetune.py`` from :func:`instruction` and tokenized with
    ``tokenize=True``, which is why removing it here changes no collected number.
    """
    return (
        _COMPLETION_BLOCK.format(context=example.context, question=example.question),
        example.gold,
    )


def evaluate_text2sql(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Text2SqlExample],
    *,
    label: str,
    style: Literal["chat", "completion", "auto"] = "auto",
    shots: Sequence[Text2SqlExample] = (),
    config: EvalConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
    keep_predictions: int = 0,
) -> Text2SqlResult:
    """Score a model by running the SQL it writes.

    Args:
        shots: Few-shot exemplars for the completion framing, taken from the *train*
            split. Ignored under chat. Must be the same list at every measurement point.
        keep_predictions: How many raw generations to retain for inspection. The hit
            vector is always complete regardless.
    """
    from ._code_exec import prepare_decode, resolve_style

    resolved = resolve_style(tokenizer, style)
    if config is None:
        config = DEFAULT_CHAT_CONFIG if resolved == "chat" else DEFAULT_COMPLETION_CONFIG
    elif resolved == "completion" and FEWSHOT_STOP not in config.stop_sequences:
        config = replace(config, stop_sequences=(*config.stop_sequences, FEWSHOT_STOP))
    # Forces `add_special_tokens` off under a chat template, which emits BOS itself.
    # Two BOS tokens is worth a few points and no error message.
    config = prepare_decode(tokenizer, config, style=resolved, label=label)

    subset = list(examples[: config.limit] if config.limit else examples)
    prompts = [build_prompt(e, tokenizer, style=resolved, shots=shots) for e in subset]
    generations = generate_batched(model, tokenizer, prompts, config, progress=progress)

    hits: list[bool] = []
    correct = unparseable = errored = exact = unfinished = 0
    for example, generation in zip(subset, generations, strict=True):
        sql = extract_sql(generation)
        # Counted on every item rather than only the unparseable ones: the ratio worth
        # knowing is how much of the *whole* set ran out of budget mid-thought.
        unfinished += reasoning_state(generation) == "unclosed"
        if not sql:
            unparseable += 1
            hits.append(False)
            continue
        exact += _normalise_text(sql) == _normalise_text(example.gold)
        rows, error = run_query(example.context, sql)
        if error:
            errored += 1
            hits.append(False)
            continue
        hit = execution_match(example.gold_rows, rows, ordered=example.ordered)
        correct += hit
        hits.append(hit)

    # Derived from `hits` and the items, never accumulated alongside them: two
    # counters updated in the same loop are two things that can disagree, and the one
    # that would be wrong is the breakdown nobody checks.
    by_source: dict[str, tuple[int, int]] = {}
    for example, hit in zip(subset, hits, strict=True):
        was_correct, seen = by_source.get(example.source, (0, 0))
        by_source[example.source] = (was_correct + int(hit), seen + 1)

    result = Text2SqlResult(
        label=label,
        correct=correct,
        total=len(subset),
        unparseable=unparseable,
        errored=errored,
        hits=hits,
        exact=exact,
        unfinished_reasoning=unfinished,
        predictions=list(generations[:keep_predictions]),
        by_source=by_source,
    )
    _log.info(
        "%s: %.2f%% execution (%d/%d), exact %.2f%%, unparseable %d, sql errors %d",
        label,
        100 * result.accuracy,
        correct,
        result.total,
        100 * exact / result.total if result.total else 0.0,
        unparseable,
        errored,
    )
    return result
