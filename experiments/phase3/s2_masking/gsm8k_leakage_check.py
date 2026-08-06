"""Does any GSM8K *test* question appear in the mixture S2 fine-tunes on?

The S2 census flags ``ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k`` as a GSM8K-derived
source, and :func:`run_s2_finetune.report_sources` reports it and keeps it, on the argument
that contamination inflates the fp16 and the quantized side of a comparison equally. That
argument is sound for arm-versus-arm, and it answers a different question than this asks.

If GSM8K *test* items are in the training mixture, the post-SFT GSM8K score is recall rather
than reasoning -- and a recalled answer is far more robust to weight perturbation than a
derived one. That compresses the range quantization damage has to show up in, which is the
exact property S1 was run to certify. Train-split overlap does not have that problem: it is
ordinary in-domain supervision and the test items stay unseen.

So the distinction that matters is not "is GSM8K in the mixture" but "is GSM8K's *test*
split in the mixture", and it is cheap to settle by looking.

Method: normalise to lowercase alphanumeric tokens, index every 13-gram of every test
question, then stream the rows the run actually selects and look up each of their 13-grams.
13 is long enough that a collision is a quotation rather than a coincidence, and the index
makes it one pass over the mixture instead of 1 319 substring searches per row.

**Measured 2026-08-05 on the 50 000 rows of ``--seed 0``: 2 of 1 319 test items flagged, and
neither is a usable duplicate.** Record: ``gsm8k_leakage.json``.

* row 13300 (the GSM8K-derived source) shares only the template phrasing *"at the same rate,
  how many additional hours would it take to travel an additional"* with test item 602. The
  quantities differ -- 360 mi / 2 h / +240 against 1200 mi / 3 h / +2000 -- so the test
  answer is not recoverable from it. A 13-gram is long enough to be a quotation and still
  short enough to be a word problem's skeleton.
* row 44416 is a WildChat conversation about extracting LoRA adapters by SVD that quotes
  test item 0, the "Janet's ducks" prompt -- the single most-quoted item in the benchmark.
  A genuine quotation, in a conversation about something else.

2 / 1 319 caps the effect at 0.15 points either way, against the +1.54 phase 2 was
separating. The GSM8K column is usable.

**SmolTalk (config ``all``), same 50 000 rows at ``--seed 0``, measured 2026-08-06: 0 of
1 319.** Record: ``gsm8k_leakage.smoltalk.json``. Nothing to discount there.

A null from this scan is only worth as much as the evidence that it looked. ``row_text``
went through ``row["messages"]`` directly until the SmolTalk run, which was right for two of
the three mixtures and silently wrong for OpenThoughts3, whose turns are ShareGPT
``{"from", "value"}`` under ``conversations`` -- every row would have extracted to the empty
string and the scan would have reported perfect cleanliness having read no characters at
all. It now normalises through :func:`run_s2_finetune.to_messages`, records
``chars_scanned`` and ``rows_without_text``, and refuses to write a result when extraction
fails on more than 1% of rows.

Run it against another mixture before trusting that conclusion there: this is a property of
one mixture at one seed, not of SFT mixtures.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

N = 13
WORD = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return WORD.findall(text.lower())


def shingles(toks: list[str], n: int = N) -> set[str]:
    if len(toks) < n:
        # Too short for an n-gram to exist; the whole thing is the fingerprint.
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def load_gsm8k_test() -> tuple[Any, str]:
    """Load GSM8K's test split from cache, reporting which repo id resolved.

    Tried in order rather than hardcoded: the cached name depends on which id the harness
    pulled under, and an offline load only resolves the one that is actually on disk.
    """
    from datasets import load_dataset

    errors = []
    for repo, config in (("openai/gsm8k", "main"), ("gsm8k", "main"), ("openai/gsm8k", None)):
        try:
            return load_dataset(repo, config, split="test"), f"{repo}:{config}"
        except Exception as exc:  # noqa: BLE001 - which candidate resolved is the point
            errors.append(f"{repo}/{config}: {type(exc).__name__}")
    raise SystemExit("could not load GSM8K test from cache: " + "; ".join(errors))


def row_text(row: dict, column: str, to_messages: Any) -> str:
    """The conversation as one string, via the same normaliser the fine-tune uses.

    Not ``row["messages"]``. Two of the phase-3 mixtures ship ShareGPT ``{"from", "value"}``
    turns under ``conversations``, and reading a column that is not there returns nothing,
    which this scan would report as *zero leakage* -- a clean bill of health issued by a
    scan that never looked at a single character. ``to_messages`` reads the shape off the
    row and is what S2 tokenizes, so what is scanned here is what the model is trained on.
    """
    messages = to_messages(row, column)
    if not messages:
        return ""
    return "\n".join(
        message["content"] for message in messages if isinstance(message.get("content"), str)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", default="tulu3")
    parser.add_argument("--out", type=Path, default=Path("gsm8k_leakage.json"))
    args = parser.parse_args(argv)

    import run_s2_finetune as s2

    test, resolved = load_gsm8k_test()
    print(f"gsm8k test: {len(test)} rows from {resolved}", flush=True)

    index: dict[str, set[int]] = {}
    for i, item in enumerate(test):
        for shingle in shingles(tokens(item["question"])):
            index.setdefault(shingle, set()).add(i)
    print(f"index: {len(index)} distinct {N}-grams over {len(test)} questions", flush=True)

    # The run's own selection, not a fresh sample: seed and count are the ones S2 trains on,
    # so a row this scan clears is a row the model actually saw cleared.
    spec = s2.DATASETS[args.dataset]
    rows = s2.load_rows(spec, examples=args.examples, seed=args.seed)
    print(f"mixture: {len(rows)} rows selected (seed {args.seed})", flush=True)

    hit_items: set[int] = set()
    hit_rows: list[dict] = []
    by_source: Counter = Counter()
    source_counts: Counter = Counter()
    empty = 0
    chars = 0

    for position, row in enumerate(rows):
        source = row.get("source") or "?"
        source_counts[source] += 1
        text = row_text(row, spec["column"], s2.to_messages)
        if not text:
            empty += 1
        chars += len(text)
        found: set[int] = set()
        for shingle in shingles(tokens(text)):
            owners = index.get(shingle)
            if owners:
                found |= owners
        if found:
            hit_items |= found
            by_source[source] += 1
            if len(hit_rows) < 25:
                hit_rows.append(
                    {"position": position, "source": source, "test_items": sorted(found)[:5]}
                )
        if position and position % 10_000 == 0:
            print(f"  {position} rows, {len(hit_items)} test items hit so far", flush=True)

    # A leakage scan reports absence, and absence is what a scan that read nothing also
    # reports. So the null has to carry evidence that it looked: how many rows yielded no
    # text at all, and how many characters were searched. Both go in the record, and an
    # extraction that failed wholesale refuses to write a clean bill of health.
    if empty > len(rows) // 100:
        raise SystemExit(
            f"{empty} of {len(rows)} rows yielded no text from column {spec['column']!r} "
            f"of {spec['repo']}: the scan did not read the conversations, so its zero is "
            f"an artefact of extraction and not a finding. Check to_messages against the "
            f"row shape before trusting any leakage number from this mixture."
        )

    result = {
        "n_test": len(test),
        "n_rows": len(rows),
        "ngram": N,
        "column": spec["column"],
        "rows_without_text": empty,
        "chars_scanned": chars,
        "test_items_hit": len(hit_items),
        "test_items_hit_rate": round(len(hit_items) / len(test), 6),
        "rows_hitting": sum(by_source.values()),
        "rows_hitting_by_source": dict(by_source.most_common()),
        "gsm8k_sourced_rows": source_counts.get("ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k", 0),
        "examples": hit_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, indent=2), flush=True)
    print(f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
