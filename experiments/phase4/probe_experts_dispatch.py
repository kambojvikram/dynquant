#!/usr/bin/env python3
"""Does the ``dynquant`` experts dispatch agree with ``grouped_mm``, where ``eager`` does not?

A packed :class:`~dynquant.runtime.linear.DynQuantExpertBank` can only be served by a
dispatch that indexes it one expert at a time, and until now that meant moving the model
to ``eager``. The move is not free: on LFM2.5-8B-A1B, ``eager`` and the default
``grouped_mm`` disagree on 1.24% of teacher-forced tokens, 0.29x what quantizing the model
does. So a packed artifact's accuracy was comparable to a bf16 number only if the bf16 side
was moved too, and every figure carried that condition.

:func:`dynquant.runtime.experts.dynquant_experts_forward` is meant to remove the condition
rather than restate it: it is ``grouped_mm_experts_forward`` with ``bank[e]`` substituted
for the segment read, keeping the sort, the offsets, the sentinel masking and -- the part
that matters -- the single ``view(tokens, k, dim).sum(1)`` reduction, where ``eager``
accumulates per expert in the model's own bf16.

This measures that claim the only way it can be measured cleanly: on an **unquantized**
model, where all three dispatches are computing the same function and any difference is
arithmetic ordering alone. Quantizing first would confound the comparison with the thing
the dispatch is supposed to be independent of.

Read the two numbers as a ratio, not as absolutes. What is being claimed is not that
``dynquant`` is bit-exact with ``grouped_mm`` -- a per-segment loop and a fused kernel need
not be -- but that it is *far* closer to it than ``eager`` is, so that moving a packed model
onto it costs a small fraction of what moving it to ``eager`` costs. The probe fails if that
ordering does not hold.

The probe also fails when it cannot see the effect at all, which is the outcome that
prompted the ``--scale`` switch. At ``tiny`` -- hidden 64, four experts, top-2 -- all three
dispatches produce **bit-identical** logits at bf16 and at fp32, with ``grouped_mm`` reaching
the real ``torch._grouped_mm`` and the other two verifiably running. There is nothing wrong
with that measurement except that it is not the one anybody wants: with no gap between
``eager`` and ``grouped_mm``, a ratio against zero says nothing about the 8B's 1.24%, and an
earlier draft of this file happily printed ``infx closer`` for exactly that reason. So
:func:`verdict` refuses on ``eager == grouped_mm`` before it compares anything, and ``wide``
exists to reproduce the 8B's MoE geometry -- hidden 2048, ``moe_intermediate`` 1792, 32
experts, top-4 -- at four layers instead of 24.

Usage::

    python experiments/phase4/probe_experts_dispatch.py --out dispatch.json

CPU-only and hermetic. Needs ``transformers`` new enough to carry ``Lfm2MoeConfig`` and
``ALL_EXPERTS_FUNCTIONS``; without the latter there is nothing here to measure and the
probe says so rather than passing vacuously.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/dynquant-core/src"))

SEED = 0

SCALES: dict[str, dict[str, Any]] = {
    "tiny": {
        "prompt": 64,
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 32,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_experts": 4,
        "num_experts_per_tok": 2,
    },
    "wide": {
        "prompt": 256,
        "hidden_size": 2048,
        "intermediate_size": 7168,
        "moe_intermediate_size": 1792,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_experts": 32,
        "num_experts_per_tok": 4,
    },
}
"""``wide`` is LFM2.5-8B-A1B's MoE geometry at four layers; only depth and vocab are cut.

Depth and width are not interchangeable here. The gap being chased is a per-layer rounding
that a top-k router turns discrete and 22 layers compound, so a model can be too *narrow*
to produce it even at full depth -- which is what ``tiny`` turned out to be. Width, expert
count and ``k`` are what a single layer's reduction sees, so those are the ones held equal
to the real model, and the residual caveat this probe carries is depth alone.
"""


def build(dtype: Any, scale: str) -> Any:
    """A four-layer ``lfm2_moe`` at a chosen dtype and geometry.

    ``dtype`` is a parameter because it is the variable under test as much as the
    dispatch is: the reduction ``eager`` and ``grouped_mm`` disagree about is an
    accumulation order, and an accumulation order is invisible at fp32 and loud at bf16.
    A run that only measured fp32 would report that the three dispatches agree and would
    be measuring the wrong model -- the campaign's is bf16.
    """
    import torch
    from transformers import AutoModelForCausalLM, Lfm2MoeConfig

    geometry = {k: v for k, v in SCALES[scale].items() if k != "prompt"}
    torch.manual_seed(SEED)
    config = Lfm2MoeConfig(
        vocab_size=256,
        num_hidden_layers=4,
        num_dense_layers=1,
        layer_types=["conv", "full_attention", "conv", "conv"],
        tie_word_embeddings=True,
        **geometry,
    )
    model = AutoModelForCausalLM.from_config(config).to(dtype)
    model.eval()
    return model


def logits_under(model: Any, implementation: str, prompt: int) -> Any:
    import torch

    model.set_experts_implementation(implementation)
    ids = torch.arange(prompt, dtype=torch.long).remainder(256).unsqueeze(0)
    with torch.no_grad():
        return model(ids).logits.float()


def _ran(model: Any, implementation: str) -> bool:
    """Did the config actually take the implementation we asked for?

    Cheap, and it is the difference between "the dispatches agree" and "one of them was
    never selected". A zero from this probe is a publishable finding only if every arm
    demonstrably ran, and ``set_experts_implementation`` is free to reject a name.
    """
    return str(getattr(model.config, "_experts_implementation", None)) == implementation


def measure(dtype_name: str, scale: str) -> dict[str, Any]:
    import torch

    from dynquant.runtime.experts import DISPATCH_NAME, register_experts_dispatch

    if not register_experts_dispatch():
        raise SystemExit("this transformers has no ALL_EXPERTS_FUNCTIONS; nothing to measure")

    prompt = int(SCALES[scale]["prompt"])
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    model = build(dtype, scale)
    reference = logits_under(model, "grouped_mm", prompt)
    magnitude = reference.abs().max().item()

    gaps: dict[str, float] = {}
    picks: dict[str, float] = {}
    selected: dict[str, bool] = {"grouped_mm": _ran(model, "grouped_mm")}
    for name in ("eager", DISPATCH_NAME):
        logits = logits_under(model, name, prompt)
        selected[name] = _ran(model, name)
        gaps[name] = (logits - reference).abs().max().item()
        # Argmax disagreement, because that is what an accuracy number is made of. A
        # logit delta that never changes a token is a delta nobody can observe.
        picks[name] = float((logits.argmax(-1) != reference.argmax(-1)).float().mean().item())

    return {
        "dtype": dtype_name,
        "scale": scale,
        "prompt_tokens": prompt,
        "geometry": {k: v for k, v in SCALES[scale].items() if k != "prompt"},
        "dispatch_selected": selected,
        "max_abs_logit": round(magnitude, 6),
        "logit_gap_vs_grouped_mm": gaps,
        "token_disagreement_vs_grouped_mm": picks,
    }


def verdict(runs: list[dict[str, Any]]) -> str:
    """Judged at bf16, controlled at fp32, and refused when the effect is not resolvable.

    Three gates, in order, because each one invalidates the next. A dispatch that was
    never selected makes every number meaningless. Dispatches that disagree at *fp32*
    are not disagreeing about an accumulation order, so the bf16 story would be a
    different phenomenon wearing its name. And ``eager == grouped_mm`` means there is no
    gap at this scale, so a ratio against it is division by the thing being claimed --
    the failure this function exists to catch, having once printed ``infx closer`` on a
    run where all three dispatches were bit-identical.
    """
    from dynquant.runtime.experts import DISPATCH_NAME

    at = {run["dtype"]: run for run in runs}
    unselected = [
        f"{run['dtype']}/{name}"
        for run in runs
        for name, ok in run["dispatch_selected"].items()
        if not ok
    ]
    if unselected:
        return f"NOT MEASURED: transformers did not select {', '.join(unselected)}"

    control = at["float32"]["logit_gap_vs_grouped_mm"]
    if max(control.values()) > 1e-3 * at["float32"]["max_abs_logit"]:
        return (
            f"CONTROL FAILED: at fp32 the dispatches should agree to rounding and do not "
            f"-- {control}. The bf16 numbers are not an accumulation-order story."
        )

    run = at["bfloat16"]
    gaps, picks = run["logit_gap_vs_grouped_mm"], run["token_disagreement_vs_grouped_mm"]
    if not gaps["eager"]:
        return (
            f"UNDERPOWERED at scale {run['scale']!r}: eager and grouped_mm produce "
            f"bit-identical logits over {run['prompt_tokens']} tokens, so there is no gap "
            f"for {DISPATCH_NAME!r} to close and no ratio to report. This says nothing "
            f"about the 8B's 1.24% -- it says four layers of "
            f"{run['geometry']['num_experts']} experts at hidden "
            f"{run['geometry']['hidden_size']} cannot resolve it."
        )
    if gaps[DISPATCH_NAME] >= gaps["eager"]:
        return (
            f"NO GAIN at scale {run['scale']!r}: {DISPATCH_NAME!r} sits "
            f"{gaps[DISPATCH_NAME]:.3e} from grouped_mm and eager sits {gaps['eager']:.3e} "
            f"-- moving a packed model onto this dispatch buys nothing over eager"
        )
    comparison = (
        "exactly, bit for bit, where eager does not"
        if not gaps[DISPATCH_NAME]
        else f"{gaps['eager'] / gaps[DISPATCH_NAME]:.1f}x more closely than eager does"
    )
    return (
        f"at scale {run['scale']!r} and bf16, {DISPATCH_NAME!r} tracks grouped_mm "
        f"{comparison} ({gaps[DISPATCH_NAME]:.3e} vs {gaps['eager']:.3e}), disagreeing on "
        f"{picks[DISPATCH_NAME]:.2%} of argmax tokens against eager's {picks['eager']:.2%}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=sorted(SCALES), default="wide")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    runs = [measure("bfloat16", args.scale), measure("float32", args.scale)]
    payload: dict[str, Any] = {"seed": SEED, "scale": args.scale, "runs": runs}
    payload["verdict"] = verdict(runs)

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        with args.out.open("w", encoding="utf-8") as handle:
            print(text, file=handle)
    return 0


if __name__ == "__main__":
    sys.exit(main())
