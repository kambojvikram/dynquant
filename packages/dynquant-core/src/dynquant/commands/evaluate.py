"""``dynquant eval`` -- score a model on a task, and compare two scores honestly.

The number this prints is only worth anything if the *only* thing that differed
between two runs was the weights. That is a property of the harness, not of the
caller's discipline, so everything that could differ is fixed here and recorded in
the output: greedy decode, one prompt format, a fixed few-shot prefix chosen by
seed from the train split, and the same scorer whatever produced the text. The
few-shot prefix is kept for a fine-tuned model too -- dropping it would confound
"got better at the task" with "learned this output format".

Two things this deliberately does that a simpler evaluator would not:

**It writes the per-problem correctness vector.** ``--out`` records one boolean per
problem, because the comparison that matters -- allocated against uniform at the
same budget -- is *paired*: the same problems, in the same order, scored twice. An
unpaired test on this data is roughly twice as wide, and the first version of this
harness stored only the counts, which meant recovering the vector cost a re-run of
every arm on the GPU.

**It reports the chance floor.** A 5-way multiple choice bottoms out at 20%, so a
destroyed model returns to 20% rather than to zero, and a table without the floor
makes a collapsed arm look like a mildly damaged one.

``--map`` swaps the model onto the packed runtime before scoring, so the weights
stay quantized in VRAM. That is the configuration whose memory figure is real; a
directory written by ``dynquant quantize`` holds quantized *values* in the compute
dtype, which gives the right accuracy and the wrong size. So ``--map`` takes the
*unquantized* model plus a bit map, and the two paths agree on accuracy because
they run the same encoder over the same widths.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import _shared

if TYPE_CHECKING:
    import argparse

__all__ = ["TASKS", "run"]


class _TaskSpec:
    """Per-task defaults, chosen once so no two runs can pick different ones."""

    def __init__(
        self,
        key: str,
        *,
        shots: int,
        chance: float,
        max_new_tokens: int,
        max_prompt_tokens: int,
        batch_size: int,
    ) -> None:
        self.key = key
        self.shots = shots
        self.chance = chance
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.batch_size = batch_size

    def load(self, split: str) -> list[Any]:
        if self.key == "gsm8k":
            from dynquant.eval.gsm8k import load_gsm8k

            return load_gsm8k(split)
        from dynquant.eval.casehold import load_casehold

        return load_casehold(split)

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        if self.key == "gsm8k":
            from dynquant.eval.gsm8k import evaluate_gsm8k

            return evaluate_gsm8k(*args, **kwargs)
        from dynquant.eval.casehold import evaluate_casehold

        return evaluate_casehold(*args, **kwargs)


TASKS = {
    # Five shots, a full worked solution to generate, and a 0% floor.
    "gsm8k": _TaskSpec(
        "gsm8k",
        shots=5,
        chance=0.0,
        max_new_tokens=320,
        max_prompt_tokens=2048,
        batch_size=48,
    ),
    # Two shots, not five: a CaseHOLD prompt is ~400 tokens against GSM8K's ~90, so
    # five would push the prefix past 2k for no measured benefit. The answer is one
    # digit, so eight new tokens is room for " 3" plus punctuation.
    "casehold": _TaskSpec(
        "casehold",
        shots=2,
        chance=0.2,
        max_new_tokens=8,
        max_prompt_tokens=3072,
        batch_size=32,
    ),
}


def run(args: argparse.Namespace) -> int:
    from dynquant._version import __version__
    from dynquant.errors import DynQuantError
    from dynquant.eval.harness import EvalConfig

    spec = TASKS[args.task]
    n_shots = spec.shots if args.shots is None else args.shots

    examples = spec.load(args.split)
    shots = _pick_shots(spec, n_shots, seed=args.shot_seed, split=args.shot_split)
    print(
        f"{args.task}: {len(examples)} examples from the {args.split} split, "
        f"{len(shots)} shot(s) from {args.shot_split} at seed {args.shot_seed}",
        flush=True,
    )

    model = _shared.load_model(
        args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = True

    packed: dict[str, Any] | None = None
    if args.map is not None:
        packed = _pack(model, args)

    tokenizer = _shared.load_tokenizer(
        args.tokenizer or args.model, trust_remote_code=args.trust_remote_code
    )

    config = EvalConfig(
        max_new_tokens=args.max_new_tokens or spec.max_new_tokens,
        batch_size=args.batch_size or spec.batch_size,
        max_prompt_tokens=args.max_prompt_tokens or spec.max_prompt_tokens,
        limit=args.limit,
    )
    label = args.label or f"{args.task}:{Path(args.model).name}"

    started = time.time()
    result = spec.evaluate(
        model,
        tokenizer,
        examples,
        label=label,
        shots=shots,
        config=config,
        progress=None if args.quiet else _shared.progress_printer(args.task, every=200),
        keep_predictions=args.keep_predictions,
    )
    elapsed = time.time() - started
    print("\n" + result.summary(), flush=True)

    record: dict[str, Any] = {
        "dynquant_core": __version__,
        "task": args.task,
        "label": result.label,
        "model": args.model,
        "split": args.split,
        "shots": n_shots,
        "shot_seed": args.shot_seed,
        "limit": args.limit,
        "accuracy": result.accuracy,
        "correct": result.correct,
        "total": result.total,
        "unparseable": result.unparseable,
        "chance": spec.chance,
        "seconds": round(elapsed, 1),
        "decode": {
            "max_new_tokens": config.max_new_tokens,
            "batch_size": config.batch_size,
            "max_prompt_tokens": config.max_prompt_tokens,
            "greedy": True,
        },
        "packed": packed,
        "hits": result.hits,
        "predictions": result.predictions,
    }

    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
        print(f"-> wrote {destination}", flush=True)

    comparison: dict[str, Any] | None = None
    if args.compare:
        comparison = _compare(record, args.compare)

    if args.json:
        payload = dict(record)
        if comparison is not None:
            payload["comparison"] = comparison
        print(json.dumps(payload, indent=2, default=str))

    if args.min_accuracy is not None and result.accuracy < args.min_accuracy:
        raise DynQuantError(
            f"accuracy {result.accuracy:.4f} is below --min-accuracy {args.min_accuracy:.4f}"
        )
    return 0


def _pick_shots(spec: _TaskSpec, count: int, *, seed: int, split: str) -> list[Any]:
    """A fixed few-shot prefix, byte-identical across runs and restarts.

    Drawn from a split the evaluation does not score. Using the scored split would
    put the answer to a graded problem in its own prompt for at least one example,
    which is a leak that reads as a small unexplained gain.
    """
    if count == 0:
        return []
    pool = spec.load(split)
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(pool)), count))
    return [pool[i] for i in chosen]


def _pack(model: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Swap the named modules onto the packed runtime, in place."""
    import torch

    from dynquant.runtime.linear import pack_model
    from dynquant.runtime.ops import active_backend

    bits, metadata = _shared.read_bit_map(args.map, key=args.map_key)
    _shared.check_map_covers(model, bits)
    group_size = int(metadata.get("group_size", args.group_size))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    report = pack_model(
        model,
        bits,
        group_size=group_size,
        progress=None if args.quiet else _shared.progress_printer("pack"),
    )
    print("\n" + report.summary(), flush=True)

    backend = active_backend().value
    if backend != "cuda":
        # Not a failure -- the numbers stay correct -- but the fallback dequantizes
        # every weight on every call, so a speed figure from it describes the
        # fallback and not the kernel, and saying so is cheaper than a footnote.
        print(
            f"  note: backend is {backend!r}, not the compiled kernels. Accuracy is "
            f"unaffected; anything timed here is the fallback. Run `dynquant doctor`.",
            flush=True,
        )
    return {
        "map": args.map,
        "group_size": group_size,
        "backend": backend,
        "modules": len(report.modules),
        "tied": len(report.tied),
        "fp16_bytes": report.fp16_bytes,
        "packed_bytes": report.packed_bytes,
        "cuda_peak_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
    }


def _compare(record: dict[str, Any], path: str) -> dict[str, Any]:
    """Paired McNemar against a record written by an earlier run."""
    from dynquant.errors import DynQuantError
    from dynquant.eval.compare import compare_paired

    source = Path(path)
    if not source.is_file():
        raise DynQuantError(f"no evaluation record at {source}")
    other = json.loads(source.read_text(encoding="utf-8"))

    for field in ("task", "split", "shots", "shot_seed", "limit"):
        if other.get(field) != record.get(field):
            raise DynQuantError(
                f"{source} was measured with {field}={other.get(field)!r} but this run "
                f"used {record.get(field)!r}. A paired test needs the same problems in "
                f"the same order; comparing across settings would report a harness "
                f"difference as a quantization effect."
            )
    if len(other.get("hits", [])) != len(record["hits"]):
        raise DynQuantError(
            f"{source} scored {len(other.get('hits', []))} problems, this run scored "
            f"{len(record['hits'])}. Not the same problem set."
        )

    paired = compare_paired(
        record["hits"],
        other["hits"],
        label_a=str(record["label"]),
        label_b=str(other.get("label", source.name)),
    )
    print("\n" + paired.summary(), flush=True)
    return paired.as_dict()
