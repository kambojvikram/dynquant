"""Capture a packed MoE decode step into a CUDA graph, which is a gate clause, not a claim.

P8 asks that graph replay remove measurable launch overhead. The packed-MoE report
established the *precondition* rather than the property: ``_segment_offsets`` returns a
device tensor, so the grouped forward contains no host read, and that was pinned by
counting ``.tolist()`` calls -- because removing a fence changes no output, counting the
fence is the only thing a correctness test can do. Counting a fence is not capturing a
graph. A forward can be free of host syncs and still refuse capture: an allocation
outside the graph pool, a stream the capture does not own, a kernel that touches
unified memory. The only way to know is to call ``torch.cuda.graph`` and see whether it
raises.

So this captures the real dispatch -- ``dynquant_experts_forward`` over two
``DynQuantExpertBank`` projections at LFM2.5-8B-A1B's own MoE geometry, one token routed
top-4, which is decode. Three questions, in the order that makes each answer
trustworthy:

1. does capture succeed at all,
2. does replay produce eager's values on *fresh* inputs -- a graph that captured stale
   pointers replays happily and returns the previous iteration's answer, which is the
   failure mode a timing-only probe reports as a speedup,
3. and what does replay save, measured against the same forward uncaptured, on the same
   tensors, timed after the capture so a difference cannot be warmup.

Two arms, because the clause is about *launch* overhead and the grouped kernel exists to
reduce launch count: ``fused`` is the one-launch-per-projection path and ``loop`` is the
per-expert fallback, which issues work proportional to the experts a token hit. If graph
replay is worth more to the loop than to the fused path, that is the honest reading and
it is what the numbers will say.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from dynquant.runtime import experts as experts_mod
from dynquant.runtime import ops
from dynquant.runtime.linear import DynQuantExpertBank


class _Experts(nn.Module):
    """The attributes ``use_experts_implementation`` injects, and nothing else.

    Deliberately not a transformers class, for the same reason the unit tests are not:
    the dispatch's contract is the ``ExpertsModule`` protocol, and a stand-in that
    satisfies exactly that cannot accidentally lean on something only a real
    ``Lfm2MoeExperts`` provides. What is real here is the geometry and the banks.
    """

    def __init__(self, gate_up: Any, down: Any, num_experts: int) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.gate_up_proj = gate_up
        self.down_proj = down

    def act_fn(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.silu(hidden)

    def _apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up_out.chunk(2, dim=-1)
        return self.act_fn(gate) * up


def _time(fn: Any, iters: int) -> float:
    """Median wall time of ``fn`` in ms, warmed and synchronized, via CUDA events."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return round(statistics.median(samples), 5)


def _bank(experts: int, out_features: int, in_features: int, bits: int) -> DynQuantExpertBank:
    dense = torch.randn(experts, out_features, in_features, dtype=torch.bfloat16, device="cuda")
    return DynQuantExpertBank.from_parameter(dense, bits, group_size=128).to("cuda")


def _try_capture(name: str, fn: Any) -> dict[str, Any]:
    """Whether ``fn`` captures, as a record. Warmed on a side stream first, as torch asks."""
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            fn()
    except Exception as exc:  # noqa: BLE001 -- whether it raises at all is the question
        return {"op": name, "captured": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"op": name, "captured": True}


def _ops_arm(experts: int) -> list[dict[str, Any]]:
    """Which primitive refuses capture, which is how ``bincount`` was identified.

    The forward-level probe says only that *something* in the step reads a value on the
    host, and the error names an op the traceback does not reach. This bisects it: same
    ids, one op at a time, and the two candidate ways of counting side by side. Kept in
    the file rather than deleted after the diagnosis because the conclusion it supports
    -- that ``minlength`` does not spare ``bincount`` its ``max()`` read -- is not
    documented anywhere in torch and is exactly the kind of thing a future reader will
    want to re-check against their own version rather than take on faith.
    """
    ids = torch.randint(0, experts, (4,), device="cuda", dtype=torch.int64)
    ones = torch.ones_like(ids)
    return [
        _try_capture("bincount(minlength=E)", lambda: torch.bincount(ids, minlength=experts)),
        _try_capture("bincount(minlength=2E)", lambda: torch.bincount(ids, minlength=2 * experts)),
        _try_capture("sort", lambda: torch.sort(ids)),
        _try_capture("cumsum", lambda: ids.cumsum(0)),
        _try_capture(
            "zeros.scatter_add_",
            lambda: torch.zeros(experts + 1, dtype=torch.long, device="cuda").scatter_add_(
                0, ids.clamp(max=experts), ones
            ),
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=("fused", "loop", "ops"), default="fused")
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--moe-intermediate", type=int, default=1792)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=1, help="1 is decode; raise it to see prefill")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    record: dict[str, Any] = {
        "arm": args.arm,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "has_grouped_gemv": bool(ops.has_grouped_gemv()),
        "experts": args.experts,
        "hidden": args.hidden,
        "moe_intermediate": args.moe_intermediate,
        "top_k": args.top_k,
        "bits": args.bits,
        "tokens": args.tokens,
    }

    if args.arm == "ops":
        record["ops"] = _ops_arm(args.experts)
        _emit(record, args.out)
        return 0

    if args.arm == "loop":
        # The loop is reached by telling `_fusable` the op is absent, which is the same
        # branch an older wheel takes. Patching that rather than un-packing the banks
        # keeps every other thing about the two arms identical -- same weights, same
        # values, same dtypes -- so a difference between them is the dispatch and
        # nothing else.
        experts_mod.ops.has_grouped_gemv = lambda: False  # type: ignore[assignment]
    elif not record["has_grouped_gemv"]:
        record["error"] = "grouped gemv op is absent; the fused arm would silently be the loop"
        print(json.dumps(record, indent=2))
        return 1

    module = _Experts(
        _bank(args.experts, 2 * args.moe_intermediate, args.hidden, args.bits),
        _bank(args.experts, args.hidden, args.moe_intermediate, args.bits),
        args.experts,
    )

    hidden_states = torch.randn(args.tokens, args.hidden, dtype=torch.bfloat16, device="cuda")
    top_k_index = torch.stack(
        [torch.randperm(args.experts, device="cuda")[: args.top_k] for _ in range(args.tokens)]
    )
    top_k_weights = torch.rand(args.tokens, args.top_k, dtype=torch.bfloat16, device="cuda")

    def forward() -> torch.Tensor:
        return experts_mod.dynquant_experts_forward(
            module, hidden_states, top_k_index, top_k_weights
        )

    forward()

    # Warm up on a side stream first. Capture on a cold allocator picks up the first
    # allocation of every intermediate, and torch documents the side-stream warmup as
    # the way to get those into the graph's private pool instead.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            forward()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            static_out = forward()
    except Exception as exc:  # noqa: BLE001 -- whether it raises at all is the question
        record["captured"] = False
        record["capture_error"] = f"{type(exc).__name__}: {exc}"
        _emit(record, args.out)
        # Not a failure of the probe. A path that cannot be captured is a result, and
        # the loop is expected to be one: it reads its segment bounds on the host.
        return 0

    record["captured"] = True

    # Fresh inputs, in place, because that is how a real decode step feeds a graph: the
    # captured tensors are the buffers, and the next token is copied into them. A graph
    # that captured stale pointers replays without complaint and returns the answer to
    # the previous token, so this is the check that distinguishes a working capture from
    # a fast one.
    hidden_states.copy_(torch.randn_like(hidden_states))
    top_k_index.copy_(
        torch.stack(
            [torch.randperm(args.experts, device="cuda")[: args.top_k] for _ in range(args.tokens)]
        )
    )
    top_k_weights.copy_(torch.rand_like(top_k_weights))

    reference = forward().clone()
    graph.replay()
    torch.cuda.synchronize()
    delta = (static_out.float() - reference.float()).abs().max().item()
    scale = reference.float().abs().max().item()
    record["replay_max_abs_delta"] = float(delta)
    record["replay_reference_max_abs"] = float(scale)
    record["replay_is_bit_identical"] = bool(delta == 0.0)

    record["eager_ms"] = _time(forward, args.iters)
    record["replay_ms"] = _time(graph.replay, args.iters)
    record["launch_overhead_removed_ms"] = round(record["eager_ms"] - record["replay_ms"], 5)
    record["speedup"] = round(record["eager_ms"] / record["replay_ms"], 3)
    _emit(record, args.out)
    return 0


def _emit(record: dict[str, Any], out: str | None) -> None:
    print(json.dumps(record, indent=2))
    if out:
        with Path(out).open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    sys.exit(main())
