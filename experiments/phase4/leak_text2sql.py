"""Does the training mixture contain the questions the arms are scored on?

The S2 driver has a contamination check and it reported nothing. It could not have
reported anything: ``_CONTAMINATING`` is ``("gsm8k", "humaneval", "mbpp")``, the marker
list phase 3 needed, and no SQL corpus has any of those substrings in its name. An empty
``sources_overlapping_an_eval_task`` on this mixture is a check that did not run, not a
mixture that passed -- the same shape as a determinism guard whose stated reason for
firing is a hypothesis rather than a diagnosis.

The specific worry is not hypothetical. Evaluation draws ``test`` from ``gretel`` and
``wikisql``; training draws ``train`` from those two *and* from ``b-mc2/sql-create-context``,
which ships a single ``train`` split and is a community aggregate assembled from WikiSQL and
Spider. If that aggregate was built from all of WikiSQL rather than its training split, then
WikiSQL ``test`` questions are inside the training mixture, the fine-tuned model has seen the
answers to part of its own benchmark, and every arm in the six-arm panel inherits it.

What that would and would not invalidate is worth stating before the number arrives, so the
number does not get to decide it:

* **The A/B stays valid.** Every arm -- bf16 ceiling, GPTQ, AWQ, DynQuant -- quantizes the
  same fine-tuned model, so contamination inflates all seven equally and the paired test
  measures quantization damage regardless.
* **The absolute accuracy stops being a claim about text-to-SQL.** "The fine-tune reached
  X%" would partly be "the fine-tune memorised X%".
* **And the headroom argument weakens**, because the base-model 57.75% was measured without
  any of this and the fine-tuned number would be measured with it -- the gain between them
  would not be all learning.

So this is measured, and reported either way, rather than assumed disjoint on the strength of
the word "train" appearing in two different dataset cards.

Run::

    python experiments/phase4/leak_text2sql.py --limit 400
    python experiments/phase4/leak_text2sql.py --limit 400 --out runs/s4/leak.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")

#: Which raw column holds the question, per source. Read off
#: ``text2sql_sources``'s own readers rather than restated, so a source that renames a
#: column breaks this loudly instead of silently scanning an empty set of questions.
QUESTION_COLUMN = {
    "gretel": "sql_prompt",
    "wikisql": "question",
    "create-context": "question",
}

GOLD_COLUMN = {
    "gretel": "sql",
    "create-context": "answer",
    # WikiSQL's gold is a structured `sql` dict that only becomes a string after the
    # table is synthesised, so the gold arm of this scan skips it. The question arm --
    # which is the one that matters, because a leaked question carries its answer with
    # it -- covers WikiSQL fully.
}


def normalise(text: str) -> str:
    """Fold the differences that are not differences.

    Case, punctuation, and whitespace only. Deliberately *not* stemming or synonym
    folding: this has to answer "is this the same question", and a looser rule would
    start reporting two different questions about the same table as a leak, which turns
    a measurement into an argument.
    """
    return _WHITESPACE.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


def normalise_sql(sql: str) -> str:
    """Same, plus the trailing semicolon and backtick-vs-nothing identifier quoting."""
    return _WHITESPACE.sub(" ", sql.lower().replace("`", "").replace('"', "").rstrip(";")).strip()


def raw_split(repo: str, split: str, *, config: str | None, cache_dir: str | None) -> Any:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split, "cache_dir": cache_dir}
    if config is not None:
        kwargs["name"] = config
    return load_dataset(repo, **kwargs)


def train_questions(name: str, *, cache_dir: str | None) -> tuple[set[str], set[str], int]:
    """Every question, and every gold query, in one source's *whole* training pool.

    The whole pool, not the sampled 50 000. A sample is what this run trains on, but a
    pool overlap is what any run trains on -- change ``--seed`` and the sample changes
    while the exposure does not. Both get reported; the pool number is the one that
    describes the mixture.
    """
    from dynquant.eval.text2sql_sources import SOURCES

    source = SOURCES[name]
    rows = raw_split(source.repo, source.splits["train"], config=source.config, cache_dir=cache_dir)
    questions = {normalise(str(row[QUESTION_COLUMN[name]])) for row in rows}
    column = GOLD_COLUMN.get(name)
    golds = {normalise_sql(str(row[column])) for row in rows} if column else set()
    return questions, golds, len(rows)


def sampled_questions(*, examples: int, seed: int, cache_dir: str | None) -> Iterator[str]:
    """The questions in the mixture this campaign actually sampled.

    Through ``load_text2sql("train", ...)``, which is precisely what
    ``run_s2_finetune.load_text2sql_rows`` calls -- so this reads the mixture the
    fine-tune reads, at the same seed and the same limit.

    The first version of this walked the driver's rendered conversations and compared the
    *user turn* against the eval questions. It reported 0/200 while the pool scan directly
    above it reported 189/200, and the two are not in conflict: a user turn is
    ``instruction(item)``, which wraps the question in a schema and a directive, so the
    equality could never hold and a clean sample was being reported by a comparison that
    had no way to come out any other way. It is the same failure as the driver's
    ``_CONTAMINATING`` list, found in the tool written to expose it.
    """
    from dynquant.eval.text2sql import load_text2sql

    for item in load_text2sql(
        "train",
        limit=examples if examples > 0 else None,
        seed=seed,
        cache_dir=cache_dir,
        decontaminate=False,
    ):
        yield normalise(item.question)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=400, help="eval items, as the arms will use")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--examples", type=int, default=50_000, help="as the fine-tune will use")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--skip-sample", action="store_true", help="pool overlap only")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from dynquant.eval.text2sql import load_text2sql

    evaluated = load_text2sql("test", limit=args.limit, seed=args.seed, cache_dir=args.cache_dir)
    print(f"evaluation set: {len(evaluated)} items", flush=True)

    by_source: dict[str, list[Any]] = {}
    for item in evaluated:
        by_source.setdefault(item.task_id.split("/", 1)[0], []).append(item)

    report: dict[str, Any] = {
        "limit": args.limit,
        "seed": args.seed,
        "examples": args.examples,
        "eval_items": len(evaluated),
        "eval_by_source": {name: len(items) for name, items in by_source.items()},
        "pool": {},
    }

    # Against each training pool in full.
    for train_name in ("gretel", "wikisql", "create-context"):
        questions, golds, rows = train_questions(train_name, cache_dir=args.cache_dir)
        print(
            f"{train_name}: {rows} training rows, {len(questions)} distinct questions", flush=True
        )
        entry: dict[str, Any] = {"rows": rows, "distinct_questions": len(questions), "hits": {}}
        for eval_name, items in sorted(by_source.items()):
            hit = [i.task_id for i in items if normalise(i.question) in questions]
            gold_hit = (
                [i.task_id for i in items if normalise_sql(i.gold) in golds] if golds else None
            )
            entry["hits"][eval_name] = {
                "questions": len(hit),
                "of": len(items),
                "examples": hit[:5],
                "gold_queries": None if gold_hit is None else len(gold_hit),
            }
            mark = "LEAK" if hit else "clean"
            print(f"  {mark:5} {eval_name}: {len(hit)}/{len(items)} questions present", flush=True)
        report["pool"][train_name] = entry

    # And against the 50 000 the fine-tune will actually see.
    if not args.skip_sample:
        sampled = set(
            sampled_questions(examples=args.examples, seed=args.seed, cache_dir=args.cache_dir)
        )
        print(f"sampled mixture: {len(sampled)} distinct questions", flush=True)
        report["sampled"] = {"distinct_questions": len(sampled), "hits": {}}
        for eval_name, items in sorted(by_source.items()):
            hit = [i.task_id for i in items if normalise(i.question) in sampled]
            report["sampled"]["hits"][eval_name] = {
                "questions": len(hit),
                "of": len(items),
                "examples": hit[:5],
            }
            mark = "LEAK" if hit else "clean"
            print(f"  {mark:5} {eval_name}: {len(hit)}/{len(items)} in the sample", flush=True)

    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
