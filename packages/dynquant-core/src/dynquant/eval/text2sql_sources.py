"""Where text-to-SQL items come from, and what each source can and cannot be used for.

Three datasets, mixed. The mixture is not for volume -- any one of them has more rows
than the campaign will use -- it is because a single synthetic source measures one
generator's idea of what SQL looks like. Gretel's schemas are invented, WikiSQL's are
scraped Wikipedia tables with the values people actually typed into them, and
sql-create-context is the community aggregate. A model that learns one is not the same
as a model that learns the task, and only a mixture can tell those apart.

**The admission rule differs between training and evaluation, and that is deliberate.**

An evaluation item is only usable if running the gold query against the schema in its
own prompt produces something. A source whose contexts are bare ``CREATE TABLE``
statements gives every query an empty result set, and then two queries that both return
nothing compare equal -- a model emitting ``SELECT 0`` scores near the ceiling and the
table looks normal. So :data:`SOURCES` records ``has_data`` per source and the evaluation
split takes only the sources that carry ``INSERT``s. A source without data can still
teach the mapping, so training takes all three.

That is also why the filter checks the *database* and not only the query. ``SELECT
COUNT(*)`` over an empty schema returns ``[(0,)]`` -- one row, passing a naive
"returns rows" test, and reproducible by any model that writes any count over any
table. Both conditions are required: the database must hold rows and the gold query
must find some.

Measured admission rates, 2000 shuffled rows per source, are in the module test and in
``experiments/phase4/screen_text2sql.py``, which imports this registry rather than
re-deriving it -- a screen that admitted a dataset under different rules from the loader
would be a verdict about a dataset nobody evaluates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dynquant._logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

__all__ = ["SOURCES", "RawItem", "Source", "is_readable_query", "resolve_sources"]

_log = get_logger(__name__)

MAX_CONTEXT_CHARS = 6000
"""Contexts longer than this are dropped, in every source alike.

The prompt budget is 3072 tokens and the context is most of it. A truncated schema is
worse than a dropped item: the model is asked about a table whose definition was cut,
gets it wrong for a reason that has nothing to do with quantization, and the item still
counts in the denominator. Applied uniformly so it is one rule rather than a per-source
allowance.
"""

MAX_TABLE_ROWS = 40
"""Rows inlined per synthesised table (WikiSQL only).

WikiSQL's tables run to 1950 rows and would not fit any prompt. The gold result is
computed against *the same truncated table the model is shown*, so the item stays
self-consistent -- it becomes a smaller version of the original question rather than a
question about data nobody can see. 94% of tables are under this cap; the rest are
dropped rather than silently shortened.
"""


@dataclass(frozen=True, slots=True)
class RawItem:
    """One (question, schema, gold) triple, before any execution check."""

    task_id: str
    question: str
    context: str
    gold: str
    source: str
    domain: str = ""
    complexity: str = ""


@dataclass(frozen=True, slots=True)
class Source:
    name: str
    repo: str
    has_data: bool
    """Whether the contexts carry ``INSERT``s, i.e. whether the source can be scored
    by execution at all. See the module docstring."""

    splits: dict[str, str]
    """Our split name to the dataset's. A source with one split maps both names onto
    it and is divided by :func:`_holdout`."""

    reader: Callable[[Any], Iterator[RawItem]]
    revision: str | None = None
    config: str | None = None
    holdout: int = 0
    """Items reserved for ``test`` when the source ships a single split. Zero means
    the dataset's own splits are used and nothing is held out."""

    notes: str = ""


_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


# --------------------------------------------------------------------------
# Gretel -- invented schemas, with INSERTs
# --------------------------------------------------------------------------


def _read_gretel(raw: Any) -> Iterator[RawItem]:
    for index, row in enumerate(raw):
        yield RawItem(
            task_id=f"gretel/{row.get('id', index)}",
            question=str(row["sql_prompt"]).strip(),
            context=str(row["sql_context"]).strip(),
            gold=str(row["sql"]).strip(),
            source="gretel",
            domain=str(row.get("domain", "")),
            complexity=str(row.get("sql_complexity", "")),
        )


# --------------------------------------------------------------------------
# sql-create-context -- schemas only, no rows
# --------------------------------------------------------------------------


def _read_create_context(raw: Any) -> Iterator[RawItem]:
    for index, row in enumerate(raw):
        yield RawItem(
            task_id=f"create-context/{index}",
            question=str(row["question"]).strip(),
            context=str(row["context"]).strip(),
            gold=str(row["answer"]).strip(),
            source="create-context",
        )


# --------------------------------------------------------------------------
# WikiSQL -- real Wikipedia tables, synthesised into a self-contained schema
# --------------------------------------------------------------------------

_AGG = ("", "MAX", "MIN", "COUNT", "SUM", "AVG")
_OPS = ("=", ">", "<", "OP")
_UNSAFE = re.compile(r"[^0-9A-Za-z_]+")


def _identifier(text: str) -> str:
    """A double-quoted SQL identifier holding arbitrary header text.

    Headers are scraped table captions: spaces, punctuation, parentheses, unicode.
    Renaming them ``col0..colN`` -- what the reference WikiSQL evaluator does -- would
    delete the only thing that makes the question answerable from the schema, so the
    text is kept and quoted instead. A literal double quote is doubled, which is how
    SQL escapes one inside a quoted identifier.
    """
    return '"' + text.replace('"', '""') + '"'


def _literal(text: str) -> str:
    """A single-quoted SQL string literal.

    Always quoted, including against ``REAL`` columns. SQLite applies the column's
    affinity to the *comparison*, so ``"Year" = '2011'`` against a REAL column holding
    2011.0 matches -- and quoting unconditionally means a condition value that merely
    looks numeric ("007", "1-2") cannot silently become a different number.
    """
    return "'" + str(text).replace("'", "''") + "'"


def _unique_headers(headers: Sequence[str]) -> list[str] | None:
    """Deduplicate repeated column names, or refuse the table.

    Wikipedia tables repeat headers -- two columns both called "Score". A duplicate
    would make ``CREATE TABLE`` fail outright, and disambiguating by position would
    make the gold's column index point at a name the model cannot tell apart from its
    twin. Suffixed on the second occurrence; a header that is empty after stripping
    means the table has no usable schema and the item is refused.
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for header in headers:
        name = str(header).strip()
        if not name:
            return None
        count = seen.get(name, 0)
        seen[name] = count + 1
        out.append(name if count == 0 else f"{name} ({count + 1})")
    return out


def _read_wikisql(raw: Any) -> Iterator[RawItem]:
    for index, row in enumerate(raw):
        item = _wikisql_item(row, index)
        if item is not None:
            yield item


def _wikisql_item(row: Any, index: int) -> RawItem | None:
    table, query = row["table"], row["sql"]
    body = table["rows"]
    if len(body) > MAX_TABLE_ROWS:
        return None
    headers = _unique_headers(table["header"])
    if headers is None:
        return None

    name = "table_" + _UNSAFE.sub("_", str(table.get("id", index))).strip("_")
    # `COLLATE NOCASE`, and it is not a nicety. WikiSQL's condition values are the
    # annotator's typing ("terrence ross") while the table cells are Wikipedia's
    # ("Terrence Ross"), and sqlite's `=` on TEXT is case-sensitive -- so a third of
    # the gold queries matched nothing and the items were being discarded as
    # unanswerable when in fact they were correct. The reference WikiSQL evaluator
    # lowercases both sides for the same reason; declaring the collation puts that
    # rule in the schema, where the model can see it too.
    types = ["REAL" if kind == "real" else "TEXT COLLATE NOCASE" for kind in table["types"]]
    if len(types) != len(headers):
        return None

    columns = ",\n  ".join(f"{_identifier(h)} {t}" for h, t in zip(headers, types, strict=True))
    create = f"CREATE TABLE {name} (\n  {columns}\n);"
    inserts = [
        f"INSERT INTO {name} VALUES ({', '.join(_literal(cell) for cell in cells)});"
        for cells in body
        if len(cells) == len(headers)
    ]
    if not inserts:
        return None

    select = _wikisql_select(query, headers, name)
    if select is None:
        return None

    return RawItem(
        task_id=f"wikisql/{table.get('id', index)}/{index}",
        question=str(row["question"]).strip(),
        context="\n".join([create, *inserts]),
        gold=select,
        source="wikisql",
        domain=str(table.get("page_title", "")),
        complexity=_AGG[query["agg"]] or "select",
    )


def _wikisql_select(query: Any, headers: Sequence[str], table: str) -> str | None:
    """Rebuild the query from WikiSQL's structured fields.

    Not from ``sql.human_readable``: that field is display text, not SQL -- it emits
    ``WHERE Current slogan = SOUTH AUSTRALIA``, with the value unquoted and the
    identifier unescaped, which sqlite reads as a comparison between two column names
    and refuses. The structured form is unambiguous.
    """
    conds = query["conds"]
    indices = [query["sel"], *conds["column_index"]]
    if any(i >= len(headers) or i < 0 for i in indices):
        return None
    if any(op >= len(_OPS) or _OPS[op] == "OP" for op in conds["operator_index"]):
        return None  # "OP" is WikiSQL's placeholder for an operator it never emits

    column = _identifier(headers[query["sel"]])
    agg = _AGG[query["agg"]]
    projection = f"{agg}({column})" if agg else column
    where = " AND ".join(
        f"{_identifier(headers[i])} {_OPS[op]} {_literal(value)}"
        for i, op, value in zip(
            conds["column_index"], conds["operator_index"], conds["condition"], strict=True
        )
    )
    return f"SELECT {projection} FROM {table}" + (f" WHERE {where}" if where else "")


# --------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------

SOURCES: dict[str, Source] = {
    "gretel": Source(
        name="gretel",
        repo="gretelai/synthetic_text_to_sql",
        has_data=True,
        splits={"train": "train", "test": "test"},
        reader=_read_gretel,
        notes="synthetic schemas across 100 verticals; contexts carry INSERTs",
    ),
    "wikisql": Source(
        name="wikisql",
        repo="Salesforce/wikisql",
        has_data=True,
        # The loading script was removed from the Hub; the auto-converted parquet
        # branch is the same data and needs no `trust_remote_code`.
        revision="refs/convert/parquet",
        splits={"train": "train", "test": "test"},
        reader=_read_wikisql,
        notes="real Wikipedia tables; schema and rows synthesised from table content",
    ),
    "create-context": Source(
        name="create-context",
        repo="b-mc2/sql-create-context",
        has_data=False,
        # One split, so `test` is a fixed holdout taken after a shuffle -- these rows
        # arrive grouped by upstream source, and a prefix would be one of them.
        splits={"train": "train", "test": "train"},
        reader=_read_create_context,
        holdout=4000,
        notes="schemas without rows: trains the mapping, cannot score it",
    ),
}

DEFAULT_TRAIN = ("gretel", "wikisql", "create-context")
DEFAULT_TEST = tuple(name for name, source in SOURCES.items() if source.has_data)


def resolve_sources(names: Sequence[str] | None, *, split: str) -> list[Source]:
    """Which sources a split uses, and why the two lists differ.

    ``None`` means the default for the split: everything for ``train``, and only the
    sources carrying rows for ``test``. Naming a source without data explicitly for
    ``test`` is an error rather than a silently empty contribution -- it would otherwise
    read as a source that scored zero.
    """
    from dynquant.errors import DynQuantError

    chosen = (
        list(names)
        if names is not None
        else list(DEFAULT_TRAIN if split == "train" else DEFAULT_TEST)
    )
    unknown = [name for name in chosen if name not in SOURCES]
    if unknown:
        raise DynQuantError(f"unknown text2sql source(s) {unknown}: expected {sorted(SOURCES)}")
    if split != "train":
        dataless = [name for name in chosen if not SOURCES[name].has_data]
        if dataless:
            raise DynQuantError(
                f"{dataless} ship schemas without rows, so every gold query over them "
                f"returns an empty result set and any query that also returns nothing "
                f"scores correct. They are trainable but not scorable; drop them from "
                f"--sources for the {split} split."
            )
    return [SOURCES[name] for name in chosen]


def read_source(
    source: Source, split: str, *, seed: int, cache_dir: str | None
) -> Iterator[RawItem]:
    """Stream a source's items for one split, shuffled, with the holdout applied."""
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": source.splits[split], "cache_dir": cache_dir}
    if source.revision:
        kwargs["revision"] = source.revision
    raw = load_dataset(source.repo, source.config, **kwargs)
    # Shuffled before anything is taken from it. Every one of these datasets arrives
    # grouped -- by vertical, by upstream corpus, by Wikipedia category -- so a prefix
    # is one group and reads as a model that collapsed on everything else.
    raw = raw.shuffle(seed=seed)
    if source.holdout:
        raw = raw.select(
            range(source.holdout) if split == "test" else range(source.holdout, len(raw))
        )
    yield from source.reader(raw)


def is_ordered(gold: str) -> bool:
    """Whether row order is significant, i.e. whether the gold asked for one."""
    return bool(_ORDER_BY.search(gold))


_LEADING_QUERY = re.compile(
    r"^\s*(?:--[^\n]*\n|/\*.*?\*/|\s)*(SELECT|WITH)\b", re.IGNORECASE | re.S
)


def is_readable_query(gold: str) -> bool:
    """Whether the gold is the kind of statement this task can ask for and read back.

    Gretel is a *SQL* corpus, not a *query* corpus: 9.8% of its golds are ``UPDATE``,
    ``INSERT``, ``DELETE`` or ``CREATE``. Those are excluded from the evaluation set
    already, but only as a side effect -- a DML statement returns no rows, so it lands
    in ``empty_result`` and is filed under "the gold matched nothing", which is a wrong
    diagnosis for a statement that was never going to match anything.

    They matter more in *training*, where the row filter is off and so they survive.
    :func:`~dynquant.eval.text2sql.extract_sql` reads an answer out of a generation by
    cutting at ``SELECT`` or ``WITH``, so a model taught to answer with ``UPDATE`` emits
    text the scorer cannot read at all -- scored ``unparseable``, on a metric whose floor
    is zero. Teaching a response format that is unscoreable by construction is worth
    9.8% of one source to avoid.

    Anchored at the start, past comments and whitespace: an unanchored search would
    accept ``UPDATE t SET x = (SELECT ...)`` on the strength of its subquery.
    """
    return bool(_LEADING_QUERY.match(gold))


#: Fields the mixture report carries per source, so a headline number can always be
#: decomposed into the sources that produced it.
BREAKDOWN_FIELDS = (
    "kept",
    "seen",
    "not_a_query",
    "empty_result",
    "degenerate",
    "no_data",
    "too_long",
    "failed",
)


@dataclass(slots=True)
class SourceTally:
    """Why items from one source were or were not admitted."""

    seen: int = 0
    kept: int = 0
    not_a_query: int = 0
    """The gold is a statement rather than a query -- ``UPDATE``, ``INSERT``, ``CREATE``.

    Counted separately from ``empty_result`` because the two call for opposite
    responses: an empty result is a question about data that is not there, and a DML
    statement is a row this task cannot pose or score. See
    :func:`is_readable_query`."""

    empty_result: int = 0
    """Gold ran against a populated database and matched nothing."""

    no_data: int = 0
    """The schema built but held no rows, so nothing could be scored against it."""

    degenerate: int = 0
    """Gold returned a single row that is entirely NULL or zero.

    The same failure as an empty result set, one step further along. ``SELECT
    AVG(x) WHERE <nothing matches>`` returns ``(None,)`` and ``SELECT COUNT(*) WHERE
    <nothing matches>`` returns ``(0,)`` -- both are one row, both pass a "returns
    rows" test, and both are what *any* aggregate over *any* non-matching condition
    produces. An item whose correct answer is reachable without understanding the
    question cannot show a quantization regression.
    """

    too_long: int = 0
    failed: int = 0
    """The schema or the gold query would not execute at all."""

    errors: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in BREAKDOWN_FIELDS}
