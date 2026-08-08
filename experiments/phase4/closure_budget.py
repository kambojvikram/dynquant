"""Read the decode budget for the quantized arms off a run instead of guessing it.

Take one of these arms used 256 new tokens. 13 of 32 probe generations never closed
``<think>``, and the closures that *did* land were still arriving at token 244 -- so the
distribution was censored by the cap, and the 5.50% headline was measuring the budget.

Guessing a bigger number would repeat the mistake with a different constant. This reads
the distribution off a run that kept all 400 generations, and it reads *both* halves of
the budget, because a trace that closes with no room left to write the query is as lost as
one that never closed:

    budget = tokens spent deliberating  +  tokens spent answering

Only the sum is actionable, and only the sum is what ``--max-new-tokens`` bounds.

The censoring test is the part that decides whether the answer is usable at all. If any
generation is still unclosed at the cap, the sample is truncated and its high percentiles
are lower bounds -- reporting a p99 off that would launder a censored distribution into a
confident number. The script says so rather than returning one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/dynquant-core/src"))

from dynquant.eval.harness import reasoning_state

#: What the six arms must all share with the ceiling. Compared rather than assumed: an arm
#: measured at a different budget or a different shot count is not comparable with this
#: one, and the record is the only place that fact is written down.
EXPECTED = {
    "shots": 2,
    "shot_seed": 0,
    "limit": 400,
    "task": "text2sql",
    "backend": "transformers",
}


def percentile(values: list[int], q: float) -> int:
    """Nearest-rank percentile. No interpolation -- these are token counts."""
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q / 100.0 * len(ordered) + 0.5) - 1))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--tokenizer", default="/workspace/models/LFM2.5-8B-A1B")
    parser.add_argument(
        "--headroom",
        type=float,
        default=1.25,
        help=(
            "multiple of the observed maximum to recommend. Above 1.0 because the arms are "
            "quantized and this measurement is not: perturbing the token distribution can "
            "make a model deliberate longer, and an arm that runs out of budget posts the "
            "loss as lost format compliance -- which is what a quantization comparison is "
            "looking for. Paying for tokens that are never generated costs nothing: decode "
            "stops at EOS"
        ),
    )
    args = parser.parse_args()

    record = json.loads(args.record.read_text(encoding="utf-8"))

    print(f"record   {args.record}")
    print(f"label    {record.get('label')}")
    print(f"accuracy {record.get('accuracy'):.4f}  ({record.get('correct')}/{record.get('total')})")
    decode = record.get("decode", {})
    print(
        f"decode   max_new_tokens={decode.get('max_new_tokens')} batch={decode.get('batch_size')}"
    )

    mismatched = {k: (record.get(k), v) for k, v in EXPECTED.items() if record.get(k) != v}
    if mismatched:
        print("\nthis record is not the one the arms will be compared against:")
        for key, (got, want) in mismatched.items():
            print(f"  {key}: got {got!r}, expected {want!r}")
        return 1

    generations = record.get("predictions") or []
    if len(generations) != record.get("total"):
        print(
            f"\nonly {len(generations)} of {record.get('total')} generations were kept, so the "
            "distribution below is a subsample of unknown selection. Re-run with "
            "--keep-predictions equal to --limit."
        )
        return 1

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    states = {"absent": 0, "closed": 0, "unclosed": 0}
    closure: list[int] = []
    answer: list[int] = []
    total: list[int] = []

    for text in generations:
        state = reasoning_state(text)
        states[state] += 1
        spent = len(tokenizer(text, add_special_tokens=False).input_ids)
        total.append(spent)
        if state != "closed":
            continue
        # The convention `strip_reasoning` uses: the *last* close tag, matching what the
        # model's own template does when it strips a prior turn.
        head, _, tail = text.rpartition("</think>")
        closure.append(len(tokenizer(head + "</think>", add_special_tokens=False).input_ids))
        answer.append(len(tokenizer(tail, add_special_tokens=False).input_ids))

    cap = decode.get("max_new_tokens")
    print(f"\ntrace state over {len(generations)} generations")
    for name, count in states.items():
        print(f"  {name:<9} {count:>4}  ({count / len(generations):.1%})")

    if states["unclosed"]:
        print(
            f"\nCENSORED: {states['unclosed']} generation(s) were still deliberating at the "
            f"{cap}-token cap, so every percentile below is a lower bound and the maximum is "
            "the cap itself. Raise --max-new-tokens and re-run before setting the arms' budget."
        )

    if not closure:
        print("\nno closed traces: nothing to measure.")
        return 1

    print(f"\ntokens, over the {len(closure)} closed generations")
    print(f"  {'':<12} {'p50':>6} {'p90':>6} {'p95':>6} {'p99':>6} {'max':>6}")
    for name, series in (("deliberating", closure), ("answering", answer), ("total", total)):
        row = "  ".join(f"{percentile(series, q):>6}" for q in (50, 90, 95, 99, 100))
        print(f"  {name:<12} {row}")

    observed = max(total)
    recommended = int(observed * args.headroom)
    print(f"\nlongest generation      {observed} tokens")
    print(f"recommended for the arms {recommended} tokens  ({args.headroom:g}x)")
    if states["unclosed"]:
        print("  -- but see CENSORED above: this is a lower bound, not a budget.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
