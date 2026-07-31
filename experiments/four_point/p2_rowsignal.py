"""Rank the vocabulary rows of the tied embedding/LM-head tensor by how much they matter.

The tensor does two jobs and the fine-tune left evidence about each:

``lm_head.output_grad_sq`` -- one gradient second moment per output row, already on disk
from the signal hook -- measures the **head** job. It answers "how much does this logit
move the loss", and on a task whose answers are single digits the answer is "almost none,
for almost every row".

Token frequency over the training sequences measures the **embedding** job. It answers
"how often did the run look this row up", which is what decides how much damage a badly
encoded row does to the input representation. This is not on disk, so it is counted here
from ``Task.training_row`` -- the same function the fine-tune tokenized with, so the count
is over the exact stream the run saw, not an approximation of it. In the shipped hook this
becomes one ``bincount`` per batch: device-resident, no sync, no calibration set.

The two are combined by **max of percentile ranks**, not by the product the module-level
scorer uses. The product is a soft AND and is right when two signals are two views of one
property -- an important module is usually both active and plastic. Here they are two
independent jobs sharing one tensor, and a row must keep its precision if *either* job
needs it. Multiplying would drive the answer rows to a middling rank (huge gradient, near
zero frequency) and the common-word rows to a middling rank (huge frequency, zero
gradient), and then quantize both away.

Writes the row order, most important first, plus the diagnostics that say whether the two
signals actually disagree -- because if they rank the same rows the combination rule does
not matter and the simpler story is the true one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from common import RUN_DIR, TASK, load_task, load_tokenizer


def percentile_rank(values: torch.Tensor) -> torch.Tensor:
    """Map to ``[0, 1]`` by rank, ties averaged.

    Ranks rather than raw values because the two signals are a squared gradient and a
    count -- different units, different scales, and different tail shapes. Ties are
    averaged because the frequency vector has a huge tie at zero (most of the vocabulary
    never appears) and breaking that tie by row index would encode token-id order as
    importance.
    """
    n = values.numel()
    _, inverse, counts = torch.unique(values, sorted=True, return_inverse=True, return_counts=True)
    ends = torch.cumsum(counts, 0)
    starts = ends - counts
    # Mean of the positions a tie group occupies: (first + last) / 2.
    average = (starts + ends - 1).to(torch.float64) / 2.0
    return average[inverse] / max(n - 1, 1)


def token_frequency(vocab: int) -> tuple[torch.Tensor, dict[str, float]]:
    """Count every token the fine-tune trained on. Train split only -- test is leakage."""
    train, _, _ = load_task()
    tokenizer = load_tokenizer()
    counts = torch.zeros(vocab, dtype=torch.float64)
    kept = 0
    # Buffered: a bincount per example would allocate a 248k-wide tensor tens of
    # thousands of times to add a few hundred ones to it.
    buffer: list[int] = []

    def flush() -> None:
        if buffer:
            ids = torch.tensor(buffer, dtype=torch.long)
            counts.add_(torch.bincount(ids, minlength=vocab).to(torch.float64))
            buffer.clear()

    for index, example in enumerate(train):
        row = TASK.training_row(example, tokenizer)
        if row is None:
            continue
        kept += 1
        buffer.extend(row["input_ids"])
        if len(buffer) > 1_000_000:
            flush()
        if (index + 1) % 2000 == 0:
            print(f"  [freq] {index + 1}/{len(train)} examples", flush=True)
    flush()
    seen = int((counts > 0).sum())
    stats = {
        "train_examples": len(train),
        "train_examples_tokenized": kept,
        "tokens_counted": float(counts.sum()),
        "distinct_rows_seen": seen,
        "distinct_rows_seen_pct": 100.0 * seen / vocab,
    }
    return counts, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moments", default=str(RUN_DIR / "dynquant_moments.safetensors"))
    parser.add_argument("--out", default=str(RUN_DIR / "p2_rowsignal.json"))
    parser.add_argument(
        "--recipe",
        default="max",
        choices=("max", "freq", "grad", "product"),
        help="how the two per-row signals combine; 'max' is the soft OR the tie needs",
    )
    args = parser.parse_args()

    from safetensors.torch import load_file

    moments = load_file(args.moments)
    key = next(
        (k for k in moments if k.endswith("output_grad_sq") and "head" in k),
        None,
    )
    if key is None:
        raise SystemExit(f"no lm_head.output_grad_sq in {args.moments}")
    grad = moments[key].to(torch.float64)
    vocab = grad.numel()
    print(f"  head signal {key} -> {vocab:,} rows", flush=True)

    counts, freq_stats = token_frequency(vocab)
    print("  " + json.dumps(freq_stats), flush=True)

    grad_rank = percentile_rank(grad)
    freq_rank = percentile_rank(counts)
    if args.recipe == "max":
        score = torch.maximum(grad_rank, freq_rank)
    elif args.recipe == "grad":
        score = grad_rank
    elif args.recipe == "freq":
        score = freq_rank
    else:
        score = grad_rank * freq_rank

    # Do the two signals actually disagree? If the frequent rows are the high-gradient
    # rows, the combination rule is decoration and the honest report says so.
    top_grad = set(torch.topk(grad, 512).indices.tolist())
    top_freq = set(torch.topk(counts, 512).indices.tolist())
    overlap = len(top_grad & top_freq)

    order = torch.argsort(score, descending=True)
    payload = {
        "recipe": args.recipe,
        "vocab": vocab,
        "moments": args.moments,
        "head_key": key,
        "frequency": freq_stats,
        "top512_overlap": overlap,
        "grad_rows_carrying_99pct": int(
            (torch.cumsum(torch.sort(grad, descending=True).values, 0) / grad.sum() < 0.99).sum()
        )
        + 1,
        "order": order.tolist(),
    }
    Path(args.out).write_text(json.dumps(payload), encoding="utf-8")

    print(
        f"\n  top-512 rows by gradient and by frequency overlap in {overlap}/512 rows"
        f" -- the two signals {'agree' if overlap > 256 else 'disagree'}",
        flush=True,
    )
    print(f"  -> wrote {args.out}", flush=True)

    # What the plan will actually protect, in tokens rather than rows.
    total = counts.sum()
    for k in (1024, 2048, 4096, 8192, 16384, 32768):
        covered = counts[order[:k]].sum() / total
        print(f"  top {k:6,} rows cover {100 * covered:6.2f}% of training tokens", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
