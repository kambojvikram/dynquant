"""Where does the tracker's per-module cost actually go?

:mod:`tracker_overhead` measures the total and :mod:`tracker_dispatch_count` counts ATen
calls, and between them they have twice pointed at the wrong culprit -- once at device
syncs, once at the coherence cosine, which is off by default. Both diagnoses came from
reasoning about a *count* and assuming cost was proportional to it. It is not: a
``vector_norm`` over a 100 MB activation and a ``div`` on a scalar are one dispatch each
and differ by three orders of magnitude.

So this benchmark times the parts. It runs the real tracker on a real model and disables
one layer of work at a time, so each rung's difference from the previous one is that
layer's cost in microseconds per module per step:

===== =============================== =========================================
rung   what runs                       the delta measures
===== =============================== =========================================
A      nothing (tracker absent)        --
B      hooks fire, bodies return       torch's hook dispatch + the try/except
C      + saliency norm                 reading every activation, per channel
D      + input Gram, grad hook armed   the forward Gram and backward plumbing
E      everything (real outer_exact)   the grad Gram and the contraction
===== =============================== =========================================

``param`` is timed too, as the estimator that skips D and E entirely -- it is the floor
that no amount of work on the ``outer_exact`` contraction can get below.

**Why patch instead of adding config flags.** Every rung has to run the *same* code the
gate measures, so the flags would have to live in the shipped tracker and each one would
be a branch in a hot path that exists only for this benchmark. Monkeypatching the class
before the tracker is constructed keeps the production path branch-free, and the rungs
cannot silently diverge from it because they *are* it, minus one method.

The patches make the tracker compute wrong answers -- a rung that skips the saliency norm
records no saliency. That is the point, and it is why this file only ever reports times
and never inspects a signal.

Usage:
    python benchmarks/tracker_cost_ladder.py --model Qwen/Qwen3-0.6B --batch-size 8 \
        --seq-len 2048 --out ladder.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from _bench import achieved_tflops, compare, make_batch

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"

# Every rung is (label, what it adds, {method: replacement}). The replacements are applied
# to the class, so they must be plain functions taking self.
#
# The order is additive on purpose: each dict is a *superset* of the disabling done by the
# rung below it, so a delta is one layer and not a mix of two.
_NOOP_FORWARD = {"_observe_forward": lambda self, entry, args, output: None}
# Returning None from the shared-Gram helper makes _observe_forward return before it arms
# the grad hook (see its `if gram_x is None` guard), so saliency runs and nothing else.
_NO_GRAM = {"_shared_input_gram": lambda self, entry, inp: None}
# The forward side runs in full -- input Gram computed, grad hook registered, backward hook
# invoked -- and the hook returns before the grad Gram and the contraction.
_NOOP_BACKWARD = {"_grad_output_hook": lambda self, slot, grad: None}

LADDER: list[tuple[str, str, dict[str, Any]]] = [
    ("B hooks only", "hook dispatch", _NOOP_FORWARD),
    ("C + saliency", "activation norms", _NO_GRAM),
    ("D + input gram", "fwd gram + bwd plumbing", _NOOP_BACKWARD),
    ("E full outer_exact", "grad gram + contraction", {}),
]


@contextlib.contextmanager
def patched(replacements: dict[str, Any]) -> Iterator[None]:
    """Swap methods on :class:`SignalTracker` for the duration of the block.

    Patched on the class rather than an instance because the tracker under test is
    constructed inside ``track_signals``, out of reach -- and because the forward hook is
    bound at attach time, so a later instance patch would be looked straight past.
    """
    from dynquant.signals import SignalTracker

    original = {name: getattr(SignalTracker, name) for name in replacements}
    try:
        for name, replacement in replacements.items():
            setattr(SignalTracker, name, replacement)
        yield
    finally:
        for name, method in original.items():
            setattr(SignalTracker, name, method)


def build(model_name: str) -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = False
    model.gradient_checkpointing_disable()
    model.train()
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--subsample", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--budget", type=float, default=3.0, help="gate, in percent")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("no CUDA device: step-time attribution is not measurable here", file=sys.stderr)
        return 2

    model = build(args.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-9)
    batch = make_batch(model, args.batch_size, args.seq_len, seed=0)
    active_params = sum(p.numel() for p in model.parameters())

    from dynquant.signals import SignalTracker

    modules = len(SignalTracker(model))
    tokens = args.batch_size * args.seq_len

    print(
        f"{torch.cuda.get_device_name(0)}  batch {args.batch_size}x{args.seq_len}  "
        f"{modules} tracked modules  subsample {args.subsample}  pairs {args.pairs}",
        flush=True,
    )

    # (label, adds, replacements, overrides)
    arms: list[tuple[str, str, dict[str, Any], dict[str, Any]]] = [
        (label, adds, patch, {"grad_estimator": "outer_exact", "subsample_tokens": args.subsample})
        for label, adds, patch in LADDER
    ]
    # Not a rung: param shares B and C but replaces D and E with a gradient read, so its
    # cost is not on the same additive line. Reported because it is the achievable floor.
    arms.append(("param estimator", "(reference)", {}, {"grad_estimator": "param"}))

    rows: list[dict[str, Any]] = []
    for label, adds, replacements, overrides in arms:
        print(f"\n=== {label} ===", flush=True)
        with patched(replacements):
            result = compare(
                model,
                optimizer,
                batch,
                pairs=args.pairs,
                warmup=args.warmup,
                log_every=0,
                overrides=overrides,
            )
        rows.append(
            {
                "rung": label,
                "adds": adds,
                "per_module_us": (result.on - result.off) / modules * 1e6,
                **result.as_dict(),
            }
        )

    print()
    header = (
        f"{'rung':22s} {'adds':26s} {'untracked':>10s} {'tracked':>10s} "
        f"{'overhead':>9s} {'us/mod':>8s} {'delta':>8s}"
    )
    print(header)
    print("-" * len(header))
    previous = 0.0
    for row in rows:
        # The param row is not additive, so it gets no delta column.
        reference = row["rung"] == "param estimator"
        delta = float("nan") if reference else row["per_module_us"] - previous
        if not reference:
            previous = row["per_module_us"]
        print(
            f"{row['rung']:22s} {row['adds']:26s} {row['untracked_median'] * 1e3:9.2f}ms "
            f"{row['tracked_median'] * 1e3:9.2f}ms {row['overhead_percent']:+8.2f}% "
            f"{row['per_module_us']:8.1f} {'' if reference else f'{delta:+8.1f}'}"
        )

    baselines = [row["untracked_median"] for row in rows]
    drift = (max(baselines) / min(baselines) - 1.0) * 100.0
    print(f"\nuntracked baselines vary by {drift:.2f}% across rungs (drift floor)")
    # A delta smaller than the drift floor times the untracked step, expressed per module,
    # has not been resolved -- saying so is the difference between attribution and a guess.
    resolution_us = drift / 100.0 * min(baselines) / modules * 1e6
    print(f"a per-module delta under {resolution_us:.1f} us/mod is inside that floor")
    print(
        f"untracked step achieves "
        f"{achieved_tflops(min(baselines), active_params=active_params, tokens=tokens):.1f} "
        f"model TFLOP/s ({min(baselines) * 1e3:.1f} ms for {tokens} tokens, 6ND estimate)"
    )
    budget_us = args.budget / 100.0 * min(baselines) / modules * 1e6
    print(
        f"the {args.budget:.0f}% gate allows {budget_us:.1f} us/mod at this batch; "
        f"full outer_exact costs {rows[-2]['per_module_us']:.1f}, param {rows[-1]['per_module_us']:.1f}"
    )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "device": torch.cuda.get_device_name(0),
                    "model": args.model,
                    "batch_size": args.batch_size,
                    "seq_len": args.seq_len,
                    "subsample_tokens": args.subsample,
                    "tracked_modules": modules,
                    "active_params": active_params,
                    "budget_percent": args.budget,
                    "budget_us_per_module": budget_us,
                    "drift_percent": drift,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"-> wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
