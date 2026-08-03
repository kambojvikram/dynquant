#!/usr/bin/env python3
"""S0/G4 gate: does a task score the same through vLLM as through ``transformers``?

The phase-3 campaign serves every arm through vLLM, because generative evaluation at
that volume through ``transformers`` is the difference between a week and a month.
That is only sound if the runtime is the *only* thing the switch changes, and the ways
it can quietly change something else all produce a plausible score rather than an
error: an engine that prepends its own BOS, one that truncates from the other end, one
that returns completions in an order the caller assumed. This script is the
measurement that says it did not happen, on a real engine, once per campaign.

``tests/test_eval_backends.py`` covers the same boundary against a fake engine, and
that fake is a claim about vLLM's API rather than a check of it. A vLLM release can
invalidate the claim without turning a single test red -- rename ``request_id``, stop
echoing ``prompt_token_ids``, change what ``TokensPrompt`` accepts -- so the unit tests
and this script are gating different things. Neither replaces the other.

**What "agree" has to mean here.** Not identical generations. The
`serving-parity report <../docs/reports/serving-parity.md>`_ already measured that and
found it false for a reason that is not a bug: on fp16, vLLM and ``transformers`` share
only 9 of 32 greedy tokens on some prompts, because ties near a decision boundary break
differently under different kernel orders, while top-1 agreement stays at 100 %. A gate
demanding equal strings would fail on a correct integration. So the claim is about the
*score*, tested as an equivalence: the paired 95 % interval on the difference must lie
entirely inside ``--max-delta`` points of zero.

That one condition is doing two jobs, and both are needed. An interval wider than the
bound fails whether it is wide because the arms really differ or because too few
problems were scored to tell -- a gate that accepted "not significantly different" on
40 examples would pass anything. The failure message says which case it was.

Two more things are checked because passing them is not the same as measuring
something. Both arms must score above the task's chance floor, since two arms that are
equally destroyed agree perfectly. And the ``transformers`` model must actually leave
the GPU before the engine is built, because vLLM preallocates most of the card at
construction and a lingering reference turns into either an OOM or a KV cache small
enough to change the schedule.

The tokenizer object is deliberately shared between the two arms. The harness does not
mutate it -- that is what ``test_the_tokenizer_is_never_mutated_at_all`` asserts -- so
sharing costs nothing and removes one more way the arms could differ.

Usage::

    python scripts/gate_runtime_parity.py --model out/qwen3.5-2b-dq-3.25bit \\
        --task gsm8k --quantization dynquant

    python scripts/gate_runtime_parity.py --model meta-llama/Llama-3.1-8B-Instruct \\
        --task ifeval --limit 300 --max-delta 2.0 --out reports/g4-parity.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dynquant.eval.compare import PairedComparison

# IFEval takes no few-shot exemplars, so it is not in the CLI's task registry; it is
# offered here anyway because it is the phase-3 headline and the only task whose
# prompts go through a chat template, which is where the double-BOS hazard that
# motivated passing ids rather than strings actually lives.
IFEVAL_CHANCE = 0.0

GIB = 1024**3


def main() -> int:
    args = _parse_args()

    from dynquant.commands import _shared
    from dynquant.eval.compare import compare_paired

    spec, chance = _task(args.task)
    examples, shots, config = _setup(spec, args)
    tokenizer = _shared.load_tokenizer(
        args.tokenizer or args.model, trust_remote_code=args.trust_remote_code
    )

    print(
        f"{args.task}: scoring {min(len(examples), config.limit or len(examples))} example(s) "
        f"twice, once per runtime, on {args.model}",
        flush=True,
    )

    direct = _score_direct(spec, args, examples, shots, config, tokenizer)
    _release_the_gpu(args)
    served = _score_vllm(spec, args, examples, shots, config, tokenizer)

    paired = compare_paired(
        direct["hits"],
        served["hits"],
        label_a="transformers",
        label_b="vllm",
    )
    print("\n" + paired.summary(), flush=True)

    failures = _judge(paired, chance=chance, max_delta=args.max_delta)
    _report(args, paired, direct, served, chance, failures)
    return 1 if failures else 0


# --------------------------------------------------------------------------
# The two arms
# --------------------------------------------------------------------------


def _score_direct(
    spec: Any,
    args: argparse.Namespace,
    examples: list[Any],
    shots: list[Any],
    config: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    """``model.generate``, the reference the engine is compared against."""
    from dynquant.commands import _shared

    print("\n[transformers] loading", flush=True)
    model = _shared.load_model(
        args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = True
    try:
        return _score(spec, args, model, tokenizer, examples, shots, config, arm="transformers")
    finally:
        # Named in the caller's scope too, so the reference this frame holds is not
        # the only one that has to go. See `_release_the_gpu`.
        del model


def _score_vllm(
    spec: Any,
    args: argparse.Namespace,
    examples: list[Any],
    shots: list[Any],
    config: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    """The same task, the same prompts, through an offline vLLM engine."""
    from dynquant.eval.backends import VllmBackend

    engine_kwargs: dict[str, Any] = {
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        # A context shorter than the harness's own limits would make the engine
        # truncate or refuse prompts the transformers arm scored in full, which is a
        # difference in the *prompt* reported as a difference in the runtime.
        "max_model_len": args.max_model_len or (config.max_prompt_tokens + config.max_new_tokens),
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": args.enforce_eager,
    }
    if args.quantization:
        engine_kwargs["quantization"] = args.quantization
    if args.tensor_parallel_size > 1:
        engine_kwargs["tensor_parallel_size"] = args.tensor_parallel_size

    print(f"\n[vllm] building engine with {engine_kwargs}", flush=True)
    backend = VllmBackend.from_pretrained(args.model, **engine_kwargs)
    return _score(spec, args, backend, tokenizer, examples, shots, config, arm="vllm")


def _score(
    spec: Any,
    args: argparse.Namespace,
    model: Any,
    tokenizer: Any,
    examples: list[Any],
    shots: list[Any],
    config: Any,
    *,
    arm: str,
) -> dict[str, Any]:
    """Run the task. ``model`` is a ``transformers`` model or an ``EvalBackend``.

    The single call site for both arms, on purpose: a helper per arm is how the two
    end up with different ``keep_predictions``, a different progress callback, or a
    different ``config``, and any of those would be attributed to the engine.
    """
    from dynquant.commands import _shared

    started = time.time()
    if args.task == "ifeval":
        from dynquant.eval.ifeval import evaluate_ifeval

        result = evaluate_ifeval(
            model,
            tokenizer,
            examples,
            label=f"{args.task}:{arm}",
            config=config,
            progress=_shared.progress_printer(arm, every=200),
            # A missing `langdetect` would otherwise refuse to produce a number at
            # all. Both arms drop the same keys, so the pairing survives.
            on_unverifiable="drop",
        )
    else:
        result = spec.evaluate(
            model,
            tokenizer,
            examples,
            label=f"{args.task}:{arm}",
            shots=shots,
            config=config,
            progress=_shared.progress_printer(arm, every=200),
        )
    elapsed = time.time() - started
    print(f"\n[{arm}] {result.summary()}", flush=True)
    print(f"[{arm}] {elapsed:.1f}s", flush=True)

    hits = list(result.hits)
    return {
        "arm": arm,
        "hits": hits,
        "accuracy": (sum(hits) / len(hits)) if hits else 0.0,
        "seconds": round(elapsed, 1),
    }


def _release_the_gpu(args: argparse.Namespace) -> None:
    """Give the card back before vLLM asks for 90 % of it.

    A model still resident here does not produce a clean error: vLLM sizes its KV
    cache from what is free at construction, so the engine either fails to start or
    starts with a cache small enough to change how requests are batched.
    """
    import torch

    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    held = torch.cuda.memory_allocated()
    print(f"\n[gpu] {held / GIB:.2f} GiB still allocated after releasing the model", flush=True)
    if held > args.residual_gib * GIB:
        print(
            f"  warning: expected under {args.residual_gib} GiB. Something still holds the "
            f"weights, and vLLM will size its KV cache around them.",
            flush=True,
        )


# --------------------------------------------------------------------------
# The verdict
# --------------------------------------------------------------------------


def _judge(paired: PairedComparison, *, chance: float, max_delta: float) -> list[str]:
    """Equivalence, not significance. See the module docstring."""
    failures: list[str] = []
    low, high = paired.interval_points

    if low < -max_delta or high > max_delta:
        half = (high - low) / 2.0
        if half > max_delta:
            failures.append(
                f"the interval is {2 * half:.2f} points wide against a +/-{max_delta:.2f} "
                f"bound, on {paired.total} problems with {paired.discordant} discordant. "
                f"This is not evidence the runtimes differ -- it is too few problems to "
                f"tell. Score more, or widen --max-delta and say so."
            )
        else:
            failures.append(
                f"the runtimes disagree: delta {paired.delta_points:+.2f} points, 95% CI "
                f"[{low:+.2f}, {high:+.2f}], outside the +/-{max_delta:.2f} bound "
                f"(p={paired.p_value:.4g}). The campaign cannot report vLLM-scored arms "
                f"alongside transformers-scored ones until this is explained."
            )

    floor = max(chance, 0.0)
    for label, accuracy in (
        (paired.label_a, paired.accuracy_a),
        (paired.label_b, paired.accuracy_b),
    ):
        if accuracy <= floor:
            failures.append(
                f"{label} scored {accuracy:.2%} against a chance floor of {floor:.2%}. "
                f"Two equally destroyed arms agree perfectly, so the equivalence above "
                f"measured nothing."
            )
    return failures


def _report(
    args: argparse.Namespace,
    paired: PairedComparison,
    direct: dict[str, Any],
    served: dict[str, Any],
    chance: float,
    failures: list[str],
) -> None:
    concordance = (paired.both_right + paired.both_wrong) / paired.total if paired.total else 0.0
    print(
        f"\nagreed on {paired.both_right + paired.both_wrong}/{paired.total} problems "
        f"({concordance:.2%}); {paired.a_only} transformers-only, {paired.b_only} vllm-only",
        flush=True,
    )
    print(
        f"wall clock: transformers {direct['seconds']:.1f}s, vllm {served['seconds']:.1f}s",
        flush=True,
    )

    if args.out:
        from dynquant._version import __version__

        payload = {
            "dynquant_core": __version__,
            "gate": "S0/G4 runtime parity",
            "model": args.model,
            "task": args.task,
            "limit": args.limit,
            "quantization": args.quantization,
            "dtype": args.dtype,
            "max_delta_points": args.max_delta,
            "chance": chance,
            "transformers": direct,
            "vllm": served,
            "paired": paired.as_dict(),
            "passed": not failures,
            "failures": failures,
        }
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"-> wrote {destination}", flush=True)

    print()
    for failure in failures:
        print(f"FAIL: {failure}", flush=True)
    if not failures:
        low, high = paired.interval_points
        print(
            f"PASS: {args.task} scores {paired.accuracy_a:.2%} through transformers and "
            f"{paired.accuracy_b:.2%} through vLLM on the same {paired.total} problems; "
            f"the difference is {paired.delta_points:+.2f} points, 95% CI [{low:+.2f}, "
            f"{high:+.2f}], inside +/-{args.max_delta:.2f} (p={paired.p_value:.4g}).",
            flush=True,
        )


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------


def _task(name: str) -> tuple[Any, float]:
    """The CLI's own task spec, so the gate cannot be configured differently."""
    from dynquant.commands.evaluate import TASKS

    if name == "ifeval":
        return None, IFEVAL_CHANCE
    spec = TASKS[name]
    return spec, spec.chance


def _setup(spec: Any, args: argparse.Namespace) -> tuple[list[Any], list[Any], Any]:
    """Examples, few-shot prefix and decode settings -- built once, used by both arms."""
    from dataclasses import replace

    from dynquant.eval.harness import EvalConfig

    if spec is None:
        from dynquant.eval.ifeval import DEFAULT_CONFIG, load_ifeval

        config = replace(DEFAULT_CONFIG, limit=args.limit)
        if args.max_new_tokens:
            config = replace(config, max_new_tokens=args.max_new_tokens)
        if args.batch_size:
            config = replace(config, batch_size=args.batch_size)
        return load_ifeval(args.split or "train"), [], config

    from dynquant.commands.evaluate import _pick_shots

    examples = spec.load(args.split or "test")
    shots = _pick_shots(spec, spec.shots, seed=args.shot_seed, split=args.shot_split)
    config = EvalConfig(
        max_new_tokens=args.max_new_tokens or spec.max_new_tokens,
        batch_size=args.batch_size or spec.batch_size,
        max_prompt_tokens=spec.max_prompt_tokens,
        limit=args.limit,
    )
    return examples, shots, config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="checkpoint both arms load")
    parser.add_argument(
        "--task",
        default="gsm8k",
        choices=["gsm8k", "casehold", "banking77", "ifeval"],
        help="gsm8k by default: long generations, where a runtime difference compounds",
    )
    parser.add_argument("--split", default=None, help="task default if unset")
    parser.add_argument("--limit", type=int, default=None, help="score only the first N")
    parser.add_argument("--shot-seed", type=int, default=0)
    parser.add_argument("--shot-split", default="train")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="transformers arm only")

    parser.add_argument("--tokenizer", default=None, help="defaults to --model")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", help="passed to both arms")
    parser.add_argument("--trust-remote-code", action="store_true")

    parser.add_argument(
        "--quantization",
        default=None,
        help="vLLM quantization method; 'dynquant' for a packed checkpoint, unset for "
        "fp16/bf16 and for GPTQ/AWQ baselines that declare it in their own config",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true", help="skip CUDA graph capture")
    parser.add_argument(
        "--residual-gib",
        type=float,
        default=1.0,
        help="warn if more than this is still allocated when the engine is built",
    )

    parser.add_argument(
        "--max-delta",
        type=float,
        default=1.0,
        help="equivalence bound in accuracy points; the paired 95%% CI must fit inside it",
    )
    parser.add_argument("--out", default=None, help="write the full record here as JSON")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
