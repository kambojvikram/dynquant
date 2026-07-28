"""Allocate from measured sensitivity instead of from the rank-product score.

Stage 4 as it shipped ranks two signals within each role and multiplies the ranks. On
this model that ordering recovered 28.5% of the uniform-3-bit damage at a 3.125-bit
budget -- inside the band spanned by five random allocations of the same budget, and
below "upgrade the biggest tensors first" (52.0%). The quantity computed here,

    sens_i(b) = sum_rc  E[delta_r^2] E[x_c^2] (W - Q_b(W))_rc^2

recovered 85.5% against an oracle that had seen the answers at 84.9%. This script is
the end-to-end test of that: same checkpoint, same budget, same evaluation, allocation
driven by the estimate rather than by the ranks.

Two things about the calibration pass are worth stating rather than leaving to be
inferred.

**The moments come from CaseHOLD's `validation` split**, which the fine-tune never saw
and the evaluation never scores. Using `test` would be leakage -- the allocation would
be built from statistics of the split it is then graded on. (`marginal_sensitivity.py`
does use `test` for both, correctly: it measures whether a proxy predicts a loss, which
is a question about predictability rather than about generalisation.) Using `train` is
not leakage but is a worse measurement: two epochs of SFT put the model at a training
loss of 0.0000, so `E[delta_r^2]` comes out around 1e-9 and is being read off a regime
the quantized model will never be in. `--calib-split train` is kept so the difference
between the two can be shown rather than asserted.

**They are collected in one pass at the final weights, not streamed during training.**
The tracker computes the same two accumulators online under `DynQuantCallback`, and
running it that way is what the package ships; doing it post-hoc here reuses a
checkpoint that already exists instead of spending three hours re-running a fine-tune
to arrive at the same weights. The estimate is a property of the weights being
quantized, so the two differ in *when* the expectations were taken, and the record says
so.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from common import RUN_DIR, load_model, load_tokenizer, set_seed

BIT_OPTIONS = (2, 3, 4, 8)


def calibration_rows(tokenizer, device, *, split: str, n_rows: int, seed: int = 0):
    """Padded batch of calibration rows from a split the evaluation never scores."""
    import tasks as task_mod

    task = task_mod.get_task()
    if split == "train":
        source, _test, _shots = task_mod.split_task(task, seed=seed)
    else:
        source = task.load(split)

    rows = []
    for example in source:
        row = task.training_row(example, tokenizer)
        if row is not None:
            rows.append(row)
        if len(rows) >= n_rows:
            break

    width = max(len(r["input_ids"]) for r in rows)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    ids = torch.full((len(rows), width), pad, dtype=torch.long)
    labels = torch.full((len(rows), width), -100, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        n = len(r["input_ids"])
        ids[i, width - n :] = torch.tensor(r["input_ids"])
        labels[i, width - n :] = torch.tensor(r["labels"])
        mask[i, width - n :] = 1
    return ids.to(device), labels.to(device), mask.to(device)


def collect(model, ids, labels, mask, *, chunk: int):
    """Run the shipped tracker over the calibration batch and return its moments.

    Through `SignalTracker` rather than through a pair of hooks written here, so what
    the allocation is built from is the same code path a user gets from
    `DynQuantCallback` -- an estimate that only works when the moments are collected by
    a bespoke script is not a feature of the package.
    """
    from dynquant.signals import SignalTracker, TrackerConfig
    from dynquant.signals.moments import save_moments

    tracker = SignalTracker(
        model,
        config=TrackerConfig(collect_channel_moments=True, channel_moment_every=1),
    ).attach()

    started = time.time()
    for i in range(0, len(ids), chunk):
        out = model(
            input_ids=ids[i : i + chunk],
            attention_mask=mask[i : i + chunk],
            labels=labels[i : i + chunk],
        )
        out.loss.backward()
        model.zero_grad(set_to_none=True)
        tracker.on_optimizer_step()
        if (i // chunk) % 8 == 0:
            print(f"  calib step {i // chunk}, loss {float(out.loss.detach()):.4f}", flush=True)

    moments = tracker.channel_moments(reduce=False)
    tracker.detach()
    print(
        f"  collected in {time.time() - started:.0f}s: {len(moments.complete_names())} "
        f"complete of {len(moments.names)} touched",
        flush=True,
    )
    written = save_moments(moments, RUN_DIR)
    print(f"  -> {written}", flush=True)
    return moments


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(RUN_DIR / "finetuned"))
    parser.add_argument("--stats", default=str(RUN_DIR / "stats"))
    parser.add_argument("--targets", type=float, nargs="+", default=[3.25])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--calib-split", default="validation")
    parser.add_argument("--calib-rows", type=int, default=512)
    parser.add_argument("--calib-chunk", type=int, default=2)
    parser.add_argument("--out", default=str(RUN_DIR / "stage4_sensitivity.json"))
    args = parser.parse_args()

    set_seed()
    from dynquant.allocate.budget import Budget
    from dynquant.allocate.knapsack import allocate_bits
    from dynquant.allocate.policy import AllocationPolicy
    from dynquant.graph.classify import classify_model
    from dynquant.score.importance import score_modules
    from dynquant.score.sensitivity import estimate_sensitivity, module_weights
    from dynquant.signals.schema import load_stats

    tokenizer = load_tokenizer()
    model = load_model(args.model)
    model.config.use_cache = False
    model.eval()

    graph = classify_model(model)
    print(f"{len(list(graph.quantizable()))} quantizable modules", flush=True)

    ids, labels, mask = calibration_rows(
        tokenizer, "cuda", split=args.calib_split, n_rows=args.calib_rows
    )
    print(
        f"calibration batch {tuple(ids.shape)} from the {args.calib_split} split, "
        f"{int((labels != -100).sum())} scored tokens",
        flush=True,
    )
    moments = collect(model, ids, labels, mask, chunk=args.calib_chunk)

    print(f"pricing {len(BIT_OPTIONS)} widths per module ...", flush=True)
    started = time.time()
    weights = module_weights(model)
    table = estimate_sensitivity(
        graph,
        moments,
        weights,
        bit_options=BIT_OPTIONS,
        group_size=args.group_size,
    )
    print(f"  {time.time() - started:.0f}s\n{table.summary()}", flush=True)

    stats = load_stats(args.stats)
    report = score_modules(graph, stats)
    scores = report.scores()

    policy = AllocationPolicy(group_size=args.group_size)
    output: dict[str, object] = {
        "model": args.model,
        "stats": args.stats,
        "allocator": "sensitivity",
        "calibration": {
            "split": args.calib_split,
            "rows": int(ids.shape[0]),
            "tokens": int(mask.sum()),
            "collected": "post-hoc at the final weights, not streamed during training",
        },
        "sensitivity": {
            "bit_options": list(BIT_OPTIONS),
            "priced": len(table.values),
            "unestimable": list(table.unestimable),
        },
        "maps": {},
    }

    for target in args.targets:
        budget = Budget.from_target(graph, target_bits=target, group_size=args.group_size)
        real = allocate_bits(graph, scores, budget, policy, sensitivity=table)
        ranked = allocate_bits(graph, scores, budget, policy)
        moved = sum(1 for name, bits in real.bits.items() if ranked.bits[name] != bits)

        print(f"\n=== target {target:.2f} bits ===", flush=True)
        print(real.summary(), flush=True)
        print(
            f"  {moved}/{len(real.bits)} modules differ from the rank-product map "
            f"at the same budget",
            flush=True,
        )

        params_at: dict[int, int] = {}
        for name, bits in real.bits.items():
            params_at[bits] = params_at.get(bits, 0) + graph[name].num_params
        print(
            "  params per width: "
            + ", ".join(f"{b}b={p / 1e6:.1f}M" for b, p in sorted(params_at.items())),
            flush=True,
        )

        output["maps"][f"{target:.2f}"] = {  # type: ignore[index]
            "target_bits": target,
            "average_bits": real.average_bits,
            "nbytes": real.nbytes,
            "group_size": args.group_size,
            "modules_differing_from_rank_product": moved,
            "violations": [
                {
                    "name": v.name,
                    "role": v.role.value,
                    "floor_bits": v.floor_bits,
                    "assigned_bits": v.assigned_bits,
                    "num_params": v.num_params,
                }
                for v in real.violations
            ],
            "bits": real.bits,
        }

    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n-> wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
