"""How far below its own floors each phase-3 model is asked to go, and what fusion costs.

Phase 2 ran on Qwen3.5-2B and Mistral-7B, both of which spell every projection separately.
Phi-4-mini does not: ``qkv_proj`` and ``gate_up_proj`` are **55.1% of its parameters**. A
fused tensor gets one width, and :attr:`ModuleInfo.floor_bits` takes the *strictest* of its
partitions' floors -- so the up-projection rows, which the role table prices at 3 bits,
are charged 4 because they share a tensor with the SwiGLU gate.

Two things follow, and both change how S3's table should be read rather than whether it can
be produced:

1. **Phi is a harsher regime than Ministral at the same nominal target.** Its floors cost
   4.43 average bits against Ministral's 3.82, so a 3.25-bit arm asks Phi to go 1.18 bits
   below its floors and Ministral 0.57. Phase 2 found the signal earns its keep precisely
   once the role floors stop being affordable, so the two models are not two samples of one
   condition.
2. **0.21 of Phi's 4.43 bits is fusion, not architecture.** Priced per row block the same
   floors would cost 4.22. This is a limitation of the allocator, which assigns one width
   per tensor and never reads ``partitions`` -- not of the checkpoint.

The allocator is nonetheless verified here to *descend* from an unaffordable floor cost on
both models rather than returning the floor map: that was the supplement's bug 4, and every
test guarding it runs on an unfused synthetic model.

Meta device throughout: only shapes and parameter counts are read, so neither model is
materialised and this runs on a laptop in seconds. Config values are the real ones, from
the two checkpoints cached on the phase-3 box.

Usage::

    python experiments/phase3/s3_allocation/floor_headroom.py --out floor_headroom.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
for _src in sorted((REPO / "packages").glob("*/src")):
    sys.path.insert(0, str(_src))

import torch  # noqa: E402
import transformers  # noqa: E402

from dynquant.allocate.budget import Budget  # noqa: E402
from dynquant.allocate.knapsack import allocate_bits  # noqa: E402
from dynquant.graph.classify import classify_model  # noqa: E402
from dynquant.graph.roles import DEFAULT_FLOOR_BITS, ModuleRole  # noqa: E402

_TOKENS = {"pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2}

#: The targets S3 runs. 3.25 is phase 2's headline; 4.0 is where GPTQ and AWQ sit.
TARGETS = (3.25, 4.0)


def build(kind: str) -> tuple[torch.nn.Module, object]:
    """A meta-device model at the checkpoint's real geometry.

    The concrete model class rather than ``AutoModelForCausalLM``: the auto mapping
    resolves lazily and, outside a pytest session that has already imported the
    modelling module, fails to find ``Phi3ForCausalLM`` at all.
    """
    from transformers.models.mistral.modeling_mistral import MistralForCausalLM
    from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM

    if kind == "phi4-mini":
        klass: type = Phi3ForCausalLM
        cfg = transformers.Phi3Config(
            hidden_size=3072,
            intermediate_size=8192,
            num_attention_heads=24,
            num_key_value_heads=8,
            num_hidden_layers=32,
            vocab_size=200064,
            tie_word_embeddings=True,
            partial_rotary_factor=0.75,
            **_TOKENS,
        )
    elif kind == "ministral-8b":
        klass = MistralForCausalLM
        cfg = transformers.MistralConfig(
            hidden_size=4096,
            intermediate_size=12288,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            num_hidden_layers=36,
            vocab_size=131072,
            tie_word_embeddings=False,
            **_TOKENS,
        )
    else:
        raise SystemExit(f"unknown model {kind!r}")

    with torch.device("meta"):
        model = klass(cfg)
    return model, cfg


def partitioned_floor_cost(graph) -> int:
    """What the same floors would cost if each row block paid its own.

    The counterfactual the allocator cannot currently reach. A tie still wins over a
    partition's own floor -- a tied embedding delivers the LM head's 8 bits however its
    rows are divided.
    """
    total = 0
    for info in graph.quantizable():
        if not info.partitions:
            total += info.num_params * info.floor_bits
            continue
        per_row = info.num_params // info.shape[0]
        blocks = sum(
            p.num_rows * per_row * DEFAULT_FLOOR_BITS.get(p.role, 4) for p in info.partitions
        )
        tied = max((DEFAULT_FLOOR_BITS.get(r, 4) for r in info.tied_roles), default=0)
        total += max(blocks, info.num_params * tied)
    return total + 16 * graph.unquantized_params()


def measure(kind: str) -> dict:
    model, cfg = build(kind)
    graph = classify_model(model, config=cfg)

    # `total_params()` already counts the unquantized tensors; it is the denominator
    # `Budget.from_target` divides by, so every average here is on one scale.
    denom = graph.total_params()
    floor = graph.floor_cost_bits()
    ideal = partitioned_floor_cost(graph)

    fused = [m for m in graph.quantizable() if m.partitions]
    up_rows = sum(
        p.num_rows * (m.num_params // m.shape[0])
        for m in graph.quantizable()
        if m.role is ModuleRole.MLP_GATE_UP
        for p in m.partitions
        if p.role is ModuleRole.MLP_UP
    )

    names = [m.name for m in graph.quantizable()]
    rng = random.Random(0)
    scores = {n: rng.random() for n in names}
    # The shuffled control from phase 2: identical score *distribution*, no correspondence
    # to the modules. If the allocation does not move, the allocator is not reading scores.
    shuffled = list(scores.values())
    rng.shuffle(shuffled)
    control = dict(zip(names, shuffled, strict=True))

    arms = {}
    for target in TARGETS:
        budget = Budget.from_target(graph, target_bits=target)
        real = allocate_bits(graph, scores, budget)
        ctl = allocate_bits(graph, control, budget)
        arms[f"{target}"] = {
            "achieved_bits": round(real.average_bits, 4),
            "miss": round(real.average_bits - target, 5),
            "violations": len(real.violations),
            "widths": {str(k): v for k, v in real.histogram().items()},
            "modules_moved_vs_shuffled": sum(1 for n in names if real.bits[n] != ctl.bits[n]),
            "modules": len(names),
            "fused_widths": {m.name.rsplit(".", 1)[-1]: real.bits[m.name] for m in fused[:2]},
        }

    return {
        "model": kind,
        "model_type": cfg.model_type,
        "params": denom,
        "quantizable_modules": len(names),
        "fused_params": sum(m.num_params for m in fused),
        "fused_fraction": round(sum(m.num_params for m in fused) / denom, 4),
        "floor_cost_bits": round(floor / denom, 4),
        "floor_cost_bits_if_partitioned": round(ideal / denom, 4),
        "fusion_surcharge_bits": round((floor - ideal) / denom, 4),
        "up_rows_overcharged_params": up_rows,
        "headroom": {f"{t}": round(t - floor / denom, 4) for t in TARGETS},
        "arms": arms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=["phi4-mini", "ministral-8b"])
    parser.add_argument("--out", type=Path, default=Path("floor_headroom.json"))
    args = parser.parse_args(argv)

    results = []
    for kind in args.models:
        result = measure(kind)
        results.append(result)
        print(
            f"{result['model']:<14} {result['params'] / 1e9:.3f}B  "
            f"fused {100 * result['fused_fraction']:>5.1f}%  "
            f"floors {result['floor_cost_bits']:.4f}b "
            f"(fusion +{result['fusion_surcharge_bits']:.4f})",
            flush=True,
        )
        for target, arm in result["arms"].items():
            print(
                f"    @{target:<5} -> {arm['achieved_bits']:.4f}b, "
                f"{arm['violations']} violations, "
                f"{arm['modules_moved_vs_shuffled']}/{arm['modules']} move vs shuffled",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
