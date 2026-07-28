"""How many ATen calls does the tracker add per module per step, and which ones?

This exists because :mod:`tracker_overhead` measures a cost without locating it. Twice
now a plausible-sounding cause has been wrong: the overhead was attributed to per-module
device syncs (there are none -- verified by a counter, not a grep), then to saliency
dispatch (real, and fixing it halved ``param`` mode but barely moved ``outer_exact``),
then to the coherence cosine (which is *off by default*, so batching it changed nothing
in a default-configuration benchmark). Each of those was a guess dressed as a diagnosis.

A ``TorchDispatchMode`` ends the guessing. It intercepts every ATen call, so the output is
the actual per-step operation census, attributed by name, with and without the tracker
installed. The difference is what the tracker costs in dispatches -- and since the
established finding is that the tracker's cost is CPU-side dispatch at a few microseconds
each, that census predicts the step-time overhead without needing a GPU at all.

**This measures counts, never time.** ``TorchDispatchMode`` adds a Python frame to every
ATen call, so a step under this mode runs many times slower than a real one. The counts
are exact; any duration printed alongside them would be an artifact. For timing, use
:mod:`tracker_overhead`.

Run on CPU with a tiny model: the operation *count* per module is a property of the code
path, not of the device or the tensor shapes, so a 4-layer random model on CPU gives the
same census per module as a 28-layer one on an A100 and gives it in seconds.

Usage:
    python benchmarks/tracker_dispatch_count.py
    python benchmarks/tracker_dispatch_count.py --estimator outer_exact --coherence 0.95
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class CountingMode(TorchDispatchMode):
    """Tally every ATen call by operator name."""

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def __torch_dispatch__(
        self,
        func: Any,
        types: Any,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))


def build_model(layers: int, hidden: int) -> Any:
    """A small dense causal LM, randomly initialised, on CPU."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.for_model(
        "qwen3",
        vocab_size=512,
        hidden_size=hidden,
        intermediate_size=hidden * 2,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.config.use_cache = False
    model.train()
    return model


def one_step(model: Any, optimizer: Any, tracker: Any, ids: torch.Tensor) -> None:
    optimizer.zero_grad(set_to_none=True)
    model(input_ids=ids, labels=ids).loss.backward()
    if tracker is not None:
        tracker.on_optimizer_step()
    optimizer.step()


def census(
    model: Any, ids: torch.Tensor, *, tracked: bool, overrides: dict[str, Any]
) -> Counter[str]:
    """ATen calls for one step, with the tracker either installed or entirely absent.

    A warmup step runs outside the counting mode so that one-time work -- lazy buffer
    creation, optimizer state allocation, the tracker's first-call bookkeeping -- is not
    billed to the census. The alternative is a first-step count that overstates every
    steady-state figure and cannot be compared between arms.
    """
    from dynquant.signals import SignalTracker, TrackerConfig

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-9)
    tracker = SignalTracker(model, TrackerConfig(**overrides)).attach() if tracked else None
    try:
        one_step(model, optimizer, tracker, ids)  # warmup, uncounted
        with CountingMode() as mode:
            one_step(model, optimizer, tracker, ids)
    finally:
        if tracker is not None:
            tracker.detach()
    return mode.counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--estimator", default="outer_exact")
    parser.add_argument("--subsample", type=int, default=256)
    parser.add_argument(
        "--coherence",
        type=float,
        default=None,
        help="coherence_ema_beta; off by default, matching TrackerConfig -- pass a value "
        "to price the signal a default-configuration benchmark never exercises",
    )
    parser.add_argument("--top", type=int, default=14, help="operators to list")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    torch.manual_seed(0)
    overrides: dict[str, Any] = {
        "grad_estimator": args.estimator,
        "subsample_tokens": args.subsample,
    }
    if args.coherence is not None:
        overrides["coherence_ema_beta"] = args.coherence

    ids = torch.randint(0, 512, (2, args.tokens))
    # A fresh model per arm: the tracker's discovery pass and the optimizer's state are
    # both stateful, and reusing one model would let the first arm's setup leak into the
    # second arm's steady-state count.
    off = census(build_model(args.layers, args.hidden), ids, tracked=False, overrides=overrides)
    on = census(build_model(args.layers, args.hidden), ids, tracked=True, overrides=overrides)

    tracked_modules = sum(
        1 for m in build_model(args.layers, args.hidden).modules() if isinstance(m, torch.nn.Linear)
    )
    total_off, total_on = sum(off.values()), sum(on.values())
    added = total_on - total_off

    print(f"model: {args.layers} layers, hidden {args.hidden}, {tracked_modules} Linear modules")
    print(f"estimator: {args.estimator}, coherence: {args.coherence}\n")
    print(f"{'ATen calls, untracked step':44s} {total_off:8d}")
    print(f"{'ATen calls, tracked step':44s} {total_on:8d}")
    print(f"{'added by the tracker':44s} {added:+8d}")
    print(f"{'added per Linear module':44s} {added / tracked_modules:8.2f}")
    print(
        "\nAt roughly 5-10 us of CPU dispatch per call, that is "
        f"{added * 5 / 1000:.1f}-{added * 10 / 1000:.1f} ms per step on a "
        f"{tracked_modules}-module model, and scales with module count."
    )

    print(f"\n{'operator':56s} {'off':>7s} {'on':>7s} {'delta':>7s}")
    print("-" * 80)
    delta = Counter({k: on[k] - off.get(k, 0) for k in on})
    for name, count in delta.most_common(args.top):
        if count <= 0:
            continue
        print(f"{name:56s} {off.get(name, 0):7d} {on[name]:7d} {count:+7d}")

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "layers": args.layers,
                    "hidden": args.hidden,
                    "linear_modules": tracked_modules,
                    "estimator": args.estimator,
                    "coherence_ema_beta": args.coherence,
                    "aten_calls_untracked": total_off,
                    "aten_calls_tracked": total_on,
                    "added_total": added,
                    "added_per_module": added / tracked_modules,
                    "delta_by_operator": dict(delta.most_common()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n-> wrote {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
