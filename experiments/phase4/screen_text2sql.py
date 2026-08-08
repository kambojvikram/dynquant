"""How much of each text-to-SQL source survives execution scoring? Measure before training.

Execution accuracy -- build the schema, run the gold query and the model's query, compare
the result sets -- is the only text-to-SQL metric that does not punish a correct answer
for its formatting. ``SELECT a, b FROM t`` and ``SELECT t.a, t.b FROM t AS t`` are the
same query and differ in every character that string matching looks at. But execution
needs a database and none of these datasets ships one: they ship ``CREATE TABLE``
statements as prompt context, so the database is built *from the context*, per item, in
memory. That works only to the extent the source's SQL is SQLite-compatible, and Gretel's
rows are synthesised across dialects -- ``SERIAL``, ``INTERVAL``, window frames,
``ILIKE``. This script measures what fraction of each source survives, because a metric
that quietly scores half its set as "both queries errored, call it a match" is worse than
string matching.

**It imports the loader's own admission rule rather than re-deriving one.** Every count
below comes from :func:`dynquant.eval.text2sql.admit` and
:func:`dynquant.eval.text2sql_sources.read_source` -- the same two functions the
evaluation harness calls. A screen with its own copy of the rule drifts from the loader
the first time either changes, and then it is a verdict about a dataset nobody evaluates.

**The headroom half deliberately lives elsewhere.** GSM8K cost a full six-arm campaign to
discover that Qwen3.5-2B was already at 66% before anyone fine-tuned it, so the base model
has to be measured on the mixture before the GPU-hours are spent. That measurement is::

    dynquant eval <model> --task text2sql --limit 400 --prompt-style chat

which is the same evaluator the six quantized arms will run through. Reproducing it here
would give the campaign two evaluators that could disagree, and the one that would be
wrong is the one nobody re-checks.

    python experiments/phase4/screen_text2sql.py --limit 2000
    python experiments/phase4/screen_text2sql.py --split train --limit 2000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dynquant.eval.text2sql import admit
from dynquant.eval.text2sql_sources import (
    SOURCES,
    SourceTally,
    read_source,
    resolve_sources,
)


def screen(
    name: str,
    *,
    split: str,
    limit: int,
    require_rows: bool,
    seed: int,
    cache_dir: str | None,
) -> tuple[SourceTally, Counter[str], list[str]]:
    """Run ``limit`` shuffled rows of one source through the real admission rule.

    Returns the tally, a histogram of gold complexity over the *kept* items, and a few
    kept examples. The last two are what tell a source that survives from a source that
    survives *interestingly*: a mixture where one member contributes only single-table
    equality lookups is measuring one thing under three names.
    """
    source = SOURCES[name]
    tally = SourceTally()
    complexity: Counter[str] = Counter()
    samples: list[str] = []

    for item in read_source(source, split, seed=seed, cache_dir=cache_dir):
        if tally.seen >= limit:
            break
        example = admit(item, require_rows=require_rows, tally=tally)
        if example is None:
            continue
        complexity[example.complexity or "-"] += 1
        if len(samples) < 3:
            samples.append(example.gold)
    return tally, complexity, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--limit", type=int, default=2000, help="rows read per source")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # `resolve_sources` is what refuses a dataless source for a scored split, so asking
    # it -- rather than iterating `args.sources` -- means the screen cannot report a
    # coverage number for a source the evaluation would never accept.
    chosen = resolve_sources(args.sources, split=args.split)
    # Training keeps items whose gold returns nothing: an empty result set is still
    # correct supervision, and only *scoring* is corrupted by one. The screen has to
    # apply the same asymmetry or its train numbers describe a filter that never runs.
    require_rows = args.split != "train"

    report: dict[str, dict[str, object]] = {}
    for source in chosen:
        tally, complexity, samples = screen(
            source.name,
            split=args.split,
            limit=args.limit,
            require_rows=require_rows,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )
        seen = max(tally.seen, 1)
        print(f"\n=== {source.name} ({source.repo}) {args.split}, {tally.seen} shuffled rows ===")
        print(f"  {source.notes}")
        print(f"  admitted            : {tally.kept:>5} ({tally.kept / seen:.1%})")
        print(f"  would not execute   : {tally.failed:>5} ({tally.failed / seen:.1%})")
        print(f"  not a query (DML)   : {tally.not_a_query:>5} ({tally.not_a_query / seen:.1%})")
        print(f"  over the char cap   : {tally.too_long:>5} ({tally.too_long / seen:.1%})")
        if require_rows:
            print(f"  schema holds no rows: {tally.no_data:>5} ({tally.no_data / seen:.1%})")
            print(
                f"  gold matched nothing: {tally.empty_result:>5} ({tally.empty_result / seen:.1%})"
            )
            print(f"  all-null/zero answer: {tally.degenerate:>5} ({tally.degenerate / seen:.1%})")
        if tally.errors:
            print("  top execution failures:")
            for reason, count in Counter(tally.errors).most_common(5):
                print(f"    {count:>5}  {reason}")
        if complexity:
            shape = "  ".join(f"{k} {v}" for k, v in complexity.most_common(6))
            print(f"  kept by shape       : {shape}")
        for gold in samples:
            print(f"    gold: {gold[:110]}")

        report[source.name] = {
            "repo": source.repo,
            "split": args.split,
            "require_rows": require_rows,
            **tally.as_dict(),
            "errors": dict(Counter(tally.errors).most_common(20)),
            "complexity": dict(complexity.most_common()),
        }

    kept = sum(int(r["kept"]) for r in report.values())
    print(f"\nadmitted across the mixture: {kept}")
    print(
        "headroom is not measured here -- run the registered task so the base model and "
        "the quantized arms go through one evaluator:\n"
        "  dynquant eval <model> --task text2sql --limit 400 --prompt-style chat"
    )

    if args.out:
        with Path(args.out).open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
