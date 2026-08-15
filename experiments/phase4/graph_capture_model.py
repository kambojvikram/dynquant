"""The whole model under CUDA graphs, because one MoE block is not the model.

What section 13 left open
-------------------------
`graph_capture_probe.py` captured one packed MoE block and measured replay at
3.25x eager at one token. It then refused to turn that into a model-level claim,
and said so: 22 MoE layers x 0.371 ms is 8.2 ms against a measured 31.67 ms
decode step, so 26%, but that arithmetic is an **upper bound and not a
prediction**. In the real model the CPU is issuing other layers' kernels while
the GPU runs these, so some of that latency is already hidden -- exactly the way
it is hidden at 8 tokens in the block sweep, where 3.25x collapses to 1.15x.

The only way to know how much is already hidden is to capture the model.

Three arms, and the weakest one always runs
-------------------------------------------
Capturing a decode step needs a cache whose tensors do not move between steps,
and LFM2.5-8B-A1B is a hybrid: 18 of its 24 layers are short convolutions with a
conv cache, 6 are attention with a KV cache. Whether that composite is static is
a property of transformers, not of this kernel, so this probe asks rather than
assumes, and reports which arms it got:

* `compile` -- `torch.compile(mode="reduce-overhead")` over the packed model with
  a static cache. This is the production-relevant number: it is what a serving
  stack would actually turn on, and reduce-overhead is CUDA graphs underneath.
  Graph breaks are counted, because a compiled model with 40 breaks is not a
  captured model and its rate should not be read as one.
* `step`    -- one decode step captured by hand with `torch.cuda.graph`, cache
  prebuilt by a real prefill. Narrower than `compile` and harder to argue with:
  no compiler, no fusion, nothing changed except who issues the launches.
* `stack`   -- the full 24-layer forward at sequence length 1 with `use_cache=False`.
  This is **not a decode step** and is never quoted as one. It is a launch-count
  measurement: every layer, every MoE block, every packed projection runs, and
  attention runs over one token instead of over a history. It always captures, so
  it is the arm that still says something when the other two refuse.

An arm that refuses is a result and is recorded with the exception text, because
"the packed runtime cannot be captured at model level, and here is the op that
stops it" is the same kind of finding as the `bincount` one and is worth more
than a missing row.

Correctness, not just timing
----------------------------
Every arm re-checks its output against a freshly computed eager result on inputs
written into the static buffers *after* capture. A graph holding stale pointers
replays happily and returns the previous answer, which is fast and wrong, and a
probe that only timed would report it as a win.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

PROMPT = "Explain in two sentences why memory bandwidth rather than FLOPs limits decoding."


def _harness() -> Any:
    """The end-to-end harness, imported rather than copied.

    `_render` in particular is load-bearing: skipping the chat template made an
    earlier arm return fluent contentless loops that timed identically, and a
    second copy of it here would be a second thing to keep in step.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import moe_end_to_end

    return moe_end_to_end


def _time(fn: Any, iters: int) -> float:
    """Median of CUDA-event timings, warmed and synchronized.

    Median rather than mean: a single scheduler hiccup in 200 iterations moves a
    mean by more than the effect being measured.
    """
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(statistics.median(times))


def _build(path: str, bits: int, group_size: int, device: str) -> tuple[Any, Any, dict[str, Any]]:
    """Load bf16 on CPU, pack in place, then move -- so no dense weight reaches the GPU."""
    h = _harness()
    tok, model = h._load(path, torch.bfloat16)
    report = h._pack(model, bits, group_size)
    model.to(device)
    report.update(h._set_dispatch(model, "dynquant"))
    torch.cuda.synchronize()
    report["resident_mib"] = round(torch.cuda.memory_allocated() / 2**20, 1)
    return tok, model, report


def _cache_shapes(cache: Any) -> dict[str, list[int]]:
    """A census of the cache's own tensors, keyed by shape.

    The point is the *spread*: on a hybrid model the attention layers and the
    convolution layers hold different-shaped state, and a cache that reports only
    one of those shapes has been built as if every layer were the same kind.

    The dict branch is not decoration. A `LinearAttentionLayer` keeps its conv state
    in a `{0: tensor}` dict, and an earlier version of this walk covered lists,
    tuples and `__dict__` but not plain dicts -- so it reported twelve attention
    tensors and nothing else, and this file read that as "the 18 convolution layers
    contribute no cache state." They contribute all of it; the walk could not see
    inside a dict.
    """
    seen: dict[str, list[int]] = {}
    stack: list[Any] = [cache]
    depth = 0
    while stack and depth < 4096:
        depth += 1
        obj = stack.pop()
        if isinstance(obj, torch.Tensor):
            seen.setdefault(str(tuple(obj.shape)), [0])[0] += 1
            continue
        if isinstance(obj, (list, tuple)):
            stack.extend(obj)
            continue
        if isinstance(obj, dict):
            stack.extend(obj.values())
            continue
        inner = getattr(obj, "__dict__", None)
        if isinstance(inner, dict):
            stack.extend(inner.values())
    return seen


def _cache_capacity(shapes: dict[str, list[int]]) -> int:
    """Slots the cache actually has, read off its own tensors.

    Attention entries are 4-D `[batch, heads, slots, head_dim]` and the slot count is
    the third dimension. Scalars and anything else the object carries are ignored. The
    entries must agree: a cache whose layers disagree about capacity is one this file
    cannot pick a single decode position for, and guessing is what caused the defect
    this function exists to prevent.
    """
    slots = [
        int(dims.strip("()").split(",")[-2])
        for dims in shapes
        if len(dims.strip("()").rstrip(",").split(",")) == 4
    ]
    assert slots, f"no 4-D entries in the cache to read a capacity from: {shapes}"
    assert len(set(slots)) == 1, f"cache entries disagree about capacity: {shapes}"
    return slots[0]


def _cache_cursor(cache: Any, capacity: int) -> int:
    """How many slots are already written, which is where the next token goes.

    A static layer does not take a write position from its caller. It keeps a
    device-resident `cumulative_length` counter, writes at that counter, and
    increments it in place -- so `cache_position` is used for the mask and for RoPE
    but has no say in which slot the key lands in. A dynamic layer has no counter
    and no capacity: it is exactly as long as it is full, so its own length is the
    answer. Reading the cursor rather than assuming it is what distinguishes a cache
    with room left from one that is exactly full.
    """
    marks: list[int] = []
    for layer in getattr(cache, "layers", None) or []:
        val = getattr(layer, "cumulative_length", None)
        if isinstance(val, torch.Tensor) and val.numel() == 1:
            marks.append(int(val.item()))
        elif isinstance(val, int):
            marks.append(val)
    if not marks:
        return capacity
    assert len(set(marks)) == 1, f"layers disagree about how full the cache is: {marks}"
    return marks[0]


def _prefill(
    model: Any, tok: Any, device: str, max_len: int, cache: str, spare: int
) -> tuple[Any, torch.Tensor, int, int, str]:
    """Run a real prefill and hand back a cache, the next token, and the position.

    The cache class is transformers' choice, not this file's. LFM2.5-8B-A1B is a
    hybrid -- 18 of 24 layers are short convolutions with a conv cache, 6 are
    attention with a KV cache -- so the object a decode step writes into is a
    composite, and constructing one by hand would measure a model this checkpoint
    does not describe. Which kind is asked for is a command-line axis rather than a
    fallback inside this function: a kind that asserts on device poisons the CUDA
    context for the whole process, so a second attempt after a failed first one
    measures the poisoning. The class that comes back is recorded by name, because a
    dynamic cache growing by `torch.cat` is itself a reason a capture would refuse.

    `spare` is the headroom the cache is built with beyond the tokens `generate`
    puts in it, and without it there is no decode step to measure at all. See the
    comment on `max_cache_len` below.
    """
    from transformers import GenerationConfig

    h = _harness()
    enc = tok(h._render(tok, PROMPT), return_tensors="pt", add_special_tokens=False).to(device)
    ids = enc["input_ids"]
    # Decode from a realistic depth rather than from the prompt's own end. A step
    # taken at position 25 does less attention work than one at 256, and a graph's
    # share of a step is the share of whatever work that step does.
    warm = max(1, max_len - int(ids.shape[1]))
    kind = None if cache == "default" else cache
    if True:
        # `max_cache_len` is the whole reason a static decode step is measurable here.
        # transformers sizes a generation's static cache at `max_length - 1`
        # (generation/utils.py), one slot short of the sequence it returns, because
        # the last token emitted is never fed back. A prefill of N tokens therefore
        # leaves a static cache both N long and exactly N full, and the next step --
        # at any position, since a static layer writes at its own cursor rather than
        # where its caller asks -- runs off the end and asserts on device. Asking for
        # headroom explicitly is what turns that assert into an arm. A dynamic cache
        # has no capacity to run off, which is precisely why it hid this.
        #
        # `max_length` is set only because a bare `GenerationConfig` defaults it to
        # 20, shorter than the rendered prompt; transformers warns that
        # `max_new_tokens` takes precedence and that is fine.
        cfg = GenerationConfig(
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            num_beams=1,
            max_length=int(ids.shape[1]) + warm,
            max_new_tokens=warm,
            min_new_tokens=warm,
            cache_implementation=kind,
            max_cache_len=int(ids.shape[1]) + warm + spare if kind else None,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )
        try:
            with torch.no_grad():
                out = model.generate(**enc, generation_config=cfg, return_dict_in_generate=True)
                torch.cuda.synchronize()
        except Exception as exc:
            raise RuntimeError(f"prefill refused cache={cache!r}: {exc}") from exc
        built = out.past_key_values
        nxt = out.sequences[:, -1:].contiguous()
        name = type(built).__name__
        shapes = _cache_shapes(built)
        capacity = _cache_capacity(shapes)
        cursor = _cache_cursor(built, capacity)
        print(
            f"[prefill] prompt={int(ids.shape[1])} warm={warm} cache={name} "
            f"capacity={capacity} cursor={cursor} seq={int(out.sequences.shape[1])} {shapes}"
        )
        # The step runs at the cursor -- the number of tokens already cached -- which
        # is also the absolute position of the token being decoded, so the mask and
        # RoPE agree with the slot the key lands in. Three explanations of the static
        # assert were wrong before this one: the model's hybrid layer mix (the cache
        # holds all 24 layers, correctly), this file's arithmetic (the position was
        # right all along), and the packed runtime (a dense bf16 model asserts
        # identically). It was capacity, and the assertion below is what would have
        # said so on the first run.
        assert cursor == int(out.sequences.shape[1]) - 1, (
            f"cache holds {cursor} tokens but generate returned "
            f"{int(out.sequences.shape[1])}; the step would decode at the wrong position"
        )
        assert cursor < capacity, (
            f"cache is full at {cursor}/{capacity}: no slot for a decode step. "
            f"Raise --spare above {spare}."
        )
        return built, nxt, cursor, capacity, f"{kind or 'default'}:{name}"


def _capture(fn: Any) -> Any:
    """Warm on a side stream, then capture. Both halves are required by the runtime."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out


def _arm_stack(model: Any, device: str, iters: int) -> dict[str, Any]:
    """Every layer, one token, no cache. A launch-count measurement, not a decode step.

    Attention here runs over a single position rather than over a history, so the
    GPU work per layer is smaller than a real decode step's and the graph's share
    of it is correspondingly larger. That is why this arm is never quoted as a
    decode number: it bounds the launch structure, and the `step` arm is what
    prices it against real work.
    """
    ids = torch.randint(0, int(model.config.vocab_size), (1, 1), device=device)

    def run() -> torch.Tensor:
        with torch.no_grad():
            return model(input_ids=ids, use_cache=False).logits

    graph, static_out = _capture(run)
    eager_ms = _time(run, iters)

    # Fresh input written after capture: a graph holding a stale pointer replays
    # happily and returns the previous answer, which times beautifully.
    fresh = torch.randint(0, int(model.config.vocab_size), (1, 1), device=device)
    ids.copy_(fresh)
    graph.replay()
    torch.cuda.synchronize()
    replayed = static_out.clone()
    with torch.no_grad():
        expected = model(input_ids=ids, use_cache=False).logits
    replay_ms = _time(graph.replay, iters)
    delta = float((replayed.float() - expected.float()).abs().max())
    return {
        "captured": True,
        "eager_ms": round(eager_ms, 5),
        "replay_ms": round(replay_ms, 5),
        "removed_ms": round(eager_ms - replay_ms, 5),
        "speedup": round(eager_ms / replay_ms, 3),
        "max_abs_delta": delta,
        "argmax_agrees": bool(replayed.argmax(-1).eq(expected.argmax(-1)).all()),
    }


def _arm_step(
    model: Any, tok: Any, device: str, iters: int, max_len: int, cache: str, spare: int
) -> dict[str, Any]:
    """One decode step, cache prebuilt by a real prefill, captured by hand.

    This is the arm the section-13 caveat actually asks about: the same work a
    served token does, with nothing changed but who issues the launches.
    """
    kv, nxt, decode_at, capacity, cache_kind = _prefill(model, tok, device, max_len, cache, spare)
    ids = nxt.clone()
    pos = torch.tensor([decode_at], device=device)

    def run() -> torch.Tensor:
        with torch.no_grad():
            return model(
                input_ids=ids, past_key_values=kv, cache_position=pos, use_cache=True
            ).logits

    # Warmed and captured at a fixed cache position, so the cache tensors the graph
    # records are the ones every replay writes into.
    graph, static_out = _capture(run)
    eager_ms = _time(run, iters)
    replay_ms = _time(graph.replay, iters)
    ids.copy_(nxt)
    graph.replay()
    torch.cuda.synchronize()
    replayed = static_out.clone()
    with torch.no_grad():
        expected = model(
            input_ids=ids, past_key_values=kv, cache_position=pos, use_cache=True
        ).logits
    # The yardstick for the line above. A decode step mutates the cache it reads --
    # a static layer advances its cursor, a convolution layer shifts its window --
    # so two consecutive eager forwards on the same input are not required to agree
    # either. Without this control a nonzero `max_abs_delta` cannot be attributed to
    # the graph rather than to the container, and attributing it to the graph is the
    # exact mistake this file has already made once in the other direction.
    with torch.no_grad():
        again = model(input_ids=ids, past_key_values=kv, cache_position=pos, use_cache=True).logits
    # Every call above -- warmups, eager timings, replays -- advances a static
    # layer's write cursor by one, because the increment is a device op and a graph
    # replays it too. The slots past the decode position are masked out, so they do
    # not change the answer, but they do bound how many times a captured step can be
    # replayed before the cache is full. Recorded rather than reasoned about.
    cursor_after = _cache_cursor(kv, capacity)
    return {
        "captured": True,
        "decode_at_position": decode_at,
        "cache_kind": cache_kind,
        "cache_capacity": capacity,
        "cache_cursor_before": decode_at,
        "cache_cursor_after": cursor_after,
        "cache_writes": cursor_after - decode_at,
        "eager_ms": round(eager_ms, 5),
        "replay_ms": round(replay_ms, 5),
        "removed_ms": round(eager_ms - replay_ms, 5),
        "speedup": round(eager_ms / replay_ms, 3),
        "max_abs_delta": float((replayed.float() - expected.float()).abs().max()),
        "argmax_agrees": bool(replayed.argmax(-1).eq(expected.argmax(-1)).all()),
        "eager_vs_eager_delta": float((again.float() - expected.float()).abs().max()),
        "eager_vs_eager_argmax_agrees": bool(again.argmax(-1).eq(expected.argmax(-1)).all()),
    }


def _arm_compile(
    model: Any, tok: Any, device: str, iters: int, max_len: int, cache: str, spare: int
) -> dict[str, Any]:
    """`reduce-overhead` over the packed model: what a serving stack would turn on.

    Graph breaks are counted rather than assumed away. A compiled model that broke
    forty times is not a captured model, and its rate must not be read as one, so
    the break count travels with the number.
    """
    import torch._dynamo as dynamo

    kv, nxt, decode_at, capacity, cache_kind = _prefill(model, tok, device, max_len, cache, spare)
    ids = nxt.clone()
    pos = torch.tensor([decode_at], device=device)

    def run_eager() -> torch.Tensor:
        with torch.no_grad():
            return model(
                input_ids=ids, past_key_values=kv, cache_position=pos, use_cache=True
            ).logits

    eager_ms = _time(run_eager, iters)
    with torch.no_grad():
        expected = run_eager().clone()
        # Same yardstick as the step arm: the cache mutates under every forward, so
        # what "agrees" means has to be measured, not assumed.
        again = run_eager().clone()

    dynamo.reset()
    dynamo.utils.counters.clear()
    compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    def run_compiled() -> torch.Tensor:
        with torch.no_grad():
            return compiled(
                input_ids=ids, past_key_values=kv, cache_position=pos, use_cache=True
            ).logits

    t0 = time.perf_counter()
    got = run_compiled().clone()
    compile_seconds = round(time.perf_counter() - t0, 1)
    compiled_ms = _time(run_compiled, iters)
    breaks = dynamo.utils.counters.get("graph_break", {})
    return {
        "captured": True,
        "decode_at_position": decode_at,
        "cache_kind": cache_kind,
        "cache_capacity": capacity,
        "cache_cursor_before": decode_at,
        "cache_cursor_after": _cache_cursor(kv, capacity),
        "eager_ms": round(eager_ms, 5),
        "compiled_ms": round(compiled_ms, 5),
        "removed_ms": round(eager_ms - compiled_ms, 5),
        "speedup": round(eager_ms / compiled_ms, 3),
        "compile_seconds": compile_seconds,
        "graph_breaks": int(sum(breaks.values())),
        "graph_break_reasons": sorted(breaks)[:8],
        "max_abs_delta": float((got.float() - expected.float()).abs().max()),
        "argmax_agrees": bool(got.argmax(-1).eq(expected.argmax(-1)).all()),
        "eager_vs_eager_delta": float((again.float() - expected.float()).abs().max()),
        "eager_vs_eager_argmax_agrees": bool(again.argmax(-1).eq(expected.argmax(-1)).all()),
    }


def _emit(record: dict[str, Any], out: str | None) -> None:
    line = json.dumps(record, sort_keys=True)
    print(line, flush=True)
    if out:
        with Path(out).open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/dev/shm/lfm")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--max-cache-len", type=int, default=256)
    ap.add_argument("--arms", default="stack,step,compile")
    # An axis, not a fallback. A cache kind that asserts on device poisons the
    # context for the rest of the process, so trying a second kind after a failed
    # first one measures the poisoning, not the second kind.
    ap.add_argument("--cache", default="static", choices=("static", "default"))
    # Slots beyond what the prefill fills. A static cache that `generate` sized is
    # exactly full, so without headroom there is no step to capture; with it, one
    # slot per call is consumed by warmups, eager timings and replays alike.
    ap.add_argument("--spare", type=int, default=192)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda"
    tok, model, report = _build(args.model, args.bits, args.group_size, device)
    base = {
        "model": args.model,
        "bits": args.bits,
        "group_size": args.group_size,
        "iters": args.iters,
        "cache_arg": args.cache,
        "spare": args.spare,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
        **report,
    }

    runners = {
        "stack": lambda: _arm_stack(model, device, args.iters),
        "step": lambda: _arm_step(
            model, tok, device, args.iters, args.max_cache_len, args.cache, args.spare
        ),
        "compile": lambda: _arm_compile(
            model, tok, device, args.iters, args.max_cache_len, args.cache, args.spare
        ),
    }
    for name in args.arms.split(","):
        name = name.strip()
        if name not in runners:
            continue
        try:
            result = runners[name]()
        except Exception as exc:  # noqa: BLE001 -- a refusal is the finding, not a crash
            result = {
                "captured": False,
                "error": f"{type(exc).__name__}: {exc}".splitlines()[0][:400],
                # The frames, not the message. A CUDA error's last line is always
                # "Compile with TORCH_USE_CUDA_DSA", which names no code and cost this
                # file two rounds of guessing; the frames that mention a source file
                # are the part that localizes an asynchronous assert.
                "traceback_tail": [
                    line.strip()[:200]
                    for line in traceback.format_exc().splitlines()
                    if line.lstrip().startswith("File ")
                ][-8:],
            }
        _emit({**base, "arm": name, **result}, args.out)


if __name__ == "__main__":
    main()
