"""Three expert dispatches on one model, over one set of teacher-forced tokens.

This campaign has three ways of running LFM2.5-8B-A1B's expert banks and they make three
pairs, not one number:

  ``grouped_mm``  the whole bank handed to ``torch._grouped_mm`` -- what ``post_init``
                  chooses and what a plain ``from_pretrained`` therefore runs.
  ``eager``       the same 22 banks, indexed one expert at a time inside the bank tensor
                  -- the only dispatch a packed ``DynQuantExpertBank`` can serve.
  ``the loop``    no banks at all: ``llm-compressor``'s ``linearize_moe`` rewrites each
                  bank into 32 ``Linear`` modules, so 22 grouped matmuls become 704 module
                  calls. Every ``llm-compressor`` baseline in the panel runs this.

Only one of the three pairs has ever been measured on the real model: ``eager`` against
``grouped_mm`` disagree on 1.24% of teacher-forced tokens. The other two are asserted. The
loop-against-``eager`` pair is asserted *hardest* -- four places in the package call them
one class on the strength of a four-layer CPU fp32 model where they came out bitwise
identical -- and it is the claim the panel's clean post-re-score table rests on.

So: same weights, same items, same tokens, one dispatch at a time, all three pairs printed.
Each pass is derived from the previous one in place, so nothing here can be explained by
two loads of a checkpoint differing.

Teacher-forced argmax rather than generation, for the same reason the earlier probe used
it: a generated prefix diverges after the first differing token and then measures how far
apart two continuations wander, which is a quantity about decoding. Argmax over gold
positions asks the model the same question at every position regardless of what it
answered at the last one, so the disagreement rate is a property of the weights and the
dispatch and nothing else.

The seconds are secondary but come for free, and they are the only clock on this pair that
does not cost a 12,000-item re-score. Read them as a prefill-shaped forward, which is not
the decode regime the panel's ``seconds`` column measures -- section 8's 1.9-2.3x is a
generation number and this is not comparable to it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CORE_SRC = HERE.parents[1] / "packages" / "dynquant-core" / "src"
if str(CORE_SRC) not in sys.path:  # pragma: no cover - import bootstrap
    sys.path.insert(0, str(CORE_SRC))


@dataclass
class Pass:
    """One dispatch, its argmax over every scored position, and what it cost."""

    name: str
    argmax: list[int]
    seconds: float
    banks: int
    linears: int


def count_banks(model: Any) -> tuple[int, int]:
    """Expert banks still standing, and expert ``Linear``s standing in their place.

    The pair is the point. ``linearize_moe`` leaves ``config._experts_implementation``
    exactly as it found it -- a rewrite of modules does not touch the config -- so after
    the third pass the config still names whichever dispatch was set before, and believing
    it would label the loop as ``eager``. The modules are the only honest witness.
    """
    banks = linears = 0
    for name, module in model.named_modules():
        if ".experts" not in name:
            continue
        if name.endswith("_proj") or type(module).__name__ == "Linear":
            linears += 1
        elif any(p.dim() == 3 for p in module.parameters(recurse=False)):
            banks += 1
    return banks, linears


def dispatch_name(model: Any, banks: int) -> str:
    """What this model will actually run, not what its config remembers."""
    if banks == 0:
        return "the loop"
    return str(getattr(model.config, "_experts_implementation", None) or "unset")


def teacher_forced_argmax(
    model: Any, batches: list[dict[str, Any]], *, device: str
) -> tuple[list[int], float]:
    """Argmax at every gold position, flattened, plus wall clock.

    One forward per item. Batching would work and would be faster, but it puts padding
    between the dispatches and the number: two dispatches batched identically still see
    identical padding, so agreement survives it, while the *clock* would then be measuring
    how each dispatch handles a padded tail. At 24 items the saving is not worth the
    ambiguity.
    """
    import torch

    out: list[int] = []
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in batches:
            ids = batch["input_ids"].to(device)
            logits = model(input_ids=ids).logits
            # Predicting gold token j means reading the logits at position j-1, so the
            # window ends one before the last input position: the model is never asked to
            # predict past the end of what it was given.
            window = logits[0, batch["first"] - 1 : batch["last"] - 1, :]
            out.extend(int(t) for t in window.argmax(dim=-1).tolist())
    if device == "cuda":
        torch.cuda.synchronize()
    return out, time.perf_counter() - start


def build_batches(model_dir: str, *, items: int, shots: int, shot_seed: int, split: str) -> Any:
    """The panel's own items, prompts and framing -- not a re-implementation of them.

    Returns the tokenizer alongside the batches because the caller needs neither and the
    model load needs both to have agreed on a vocabulary.
    """
    import torch
    from transformers import AutoTokenizer

    from dynquant.eval.text2sql import build_prompt, load_text2sql

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    pool = load_text2sql("train", limit=max(shots, 1), seed=shot_seed) if shots else []
    examples = load_text2sql(split, limit=items, seed=shot_seed)

    batches = []
    for example in examples:
        prompt = build_prompt(example, tokenizer, style="chat", shots=pool[:shots])
        prompt_ids = (
            list(prompt)
            if isinstance(prompt, list)
            else list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        )
        gold_ids = list(tokenizer(example.gold, add_special_tokens=False)["input_ids"])
        if not gold_ids:
            continue
        ids = prompt_ids + gold_ids
        batches.append(
            {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "first": len(prompt_ids),
                "last": len(ids),
            }
        )
    return tokenizer, batches


def disagreement(left: Pass, right: Pass) -> tuple[int, int]:
    """Positions where the two argmaxes differ, and positions compared."""
    n = min(len(left.argmax), len(right.argmax))
    differing = sum(1 for a, b in zip(left.argmax[:n], right.argmax[:n], strict=True) if a != b)
    return differing, n


def report(passes: list[Pass]) -> dict[str, Any]:
    """Every pair, and a refusal for any pair whose two sides ran the same dispatch.

    A pair of identical dispatches yields exactly zero disagreement, which reads as the
    strongest possible confirmation and is in fact a statement that the probe failed to
    move the model. The same refusal ``dispatch_delta.py`` makes, for the same reason.
    """
    pairs: list[dict[str, Any]] = []
    for index, left in enumerate(passes):
        for right in passes[index + 1 :]:
            if left.name == right.name:
                pairs.append(
                    {
                        "pair": f"{left.name} vs {right.name}",
                        "refused": (
                            "both passes ran the same dispatch, so a zero here says the "
                            "probe did not move the model, not that the dispatches agree"
                        ),
                    }
                )
                continue
            differing, compared = disagreement(left, right)
            pairs.append(
                {
                    "pair": f"{left.name} vs {right.name}",
                    "tokens": compared,
                    "differing": differing,
                    "rate": differing / compared if compared else None,
                    "seconds": [round(left.seconds, 2), round(right.seconds, 2)],
                    "ratio": right.seconds / left.seconds if left.seconds else None,
                }
            )
    return {
        "passes": [
            {
                "dispatch": one.name,
                "seconds": round(one.seconds, 2),
                "tokens": len(one.argmax),
                "banks": one.banks,
                "expert_linears": one.linears,
            }
            for one in passes
        ],
        "pairs": pairs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="the merged bf16 checkpoint")
    parser.add_argument("--items", type=int, default=24)
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--shot-seed", type=int, default=0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="write the report here as well as printing it")
    parser.add_argument(
        "--skip-loop",
        action="store_true",
        help="stop after eager; use when llm-compressor is not installed",
    )
    args = parser.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM

    from dynquant.runtime.linear import use_eager_experts

    _, batches = build_batches(
        args.model, items=args.items, shots=args.shots, shot_seed=args.shot_seed, split=args.split
    )
    if not batches:
        raise SystemExit("no items with a non-empty gold, so there is nothing to teacher-force")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device)
    model.eval()

    passes: list[Pass] = []

    def take(label: str) -> None:
        banks, linears = count_banks(model)
        name = dispatch_name(model, banks)
        argmax, seconds = teacher_forced_argmax(model, batches, device=args.device)
        print(f"  {label:12s} ran {name!r}: {len(argmax)} tokens in {seconds:.1f}s", flush=True)
        passes.append(Pass(name, argmax, seconds, banks, linears))

    take("as loaded")
    use_eager_experts(model)
    take("eager")
    if not args.skip_loop:
        from llmcompressor.modeling.moe.linearize import get_non_linearized_moes, linearize_moe

        linearize_moe(model)
        assert not get_non_linearized_moes(model)
        take("linearized")

    payload = report(passes)
    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
