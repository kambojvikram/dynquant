"""Measurement point 1: the base model, before any fine-tuning.

This is the floor the other five numbers are read against. A base model that already
scores near the supervised ceiling would make the fine-tune look useless; one that
scores near chance would make any post-quantization drop unmeasurable. Either way the
number is needed before the rest of the run is worth doing -- and on GSM8K it was the
first of those, which is why :mod:`tasks` now records base-model headroom per task and
why the run moved to CaseHOLD.

The scoring is delegated to :func:`common.run_eval`, which every other measurement
point also calls. It used to be a copy of that function's body with the same
arguments filled in -- which is exactly the arrangement ``common``'s own docstring
warns about, and it had already drifted: the copy did not record the per-problem
correctness vector, so measurement 1 could not be compared to any other point as the
paired design it is. Two call sites agreeing today is not the same property as one
call site.

Usage::

    python stage1_eval_base.py            # full test split
    python stage1_eval_base.py --limit 64 # smoke run
"""

from __future__ import annotations

import argparse
import sys

from common import MODEL_ID, N_SHOTS, TASK, load_model, load_task, run_eval, set_seed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--name", default="stage1_base")
    args = parser.parse_args()

    set_seed()
    _, test, _ = load_task()
    print(
        f"{TASK.key} test: {len(test)} examples, {N_SHOTS}-shot, chance {TASK.chance:.1%}",
        flush=True,
    )

    model = load_model(MODEL_ID)
    print(f"loaded {MODEL_ID}: {sum(p.numel() for p in model.parameters()) / 1e9:.4f}B params")

    run_eval(
        model,
        label="base (no fine-tune)",
        name=args.name,
        limit=args.limit,
        extra={"model": MODEL_ID, "shots": N_SHOTS},
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
