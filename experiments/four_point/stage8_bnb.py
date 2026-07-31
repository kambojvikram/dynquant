"""The bitsandbytes NF4 arm.

Separate from :mod:`stage8_baselines` because it is not a llm-compressor recipe and
shares none of that file's machinery: there is no calibration pass, no Hessian and no
grid search. NF4 quantizes on load, from the weights alone, using a fixed non-uniform
4-bit data type whose levels are the quantiles of a unit normal -- the claim being that
transformer weights are approximately Gaussian, so a code book matched to that beats a
uniform grid at the same width without looking at any data.

It earns a place in the comparison for two reasons. It is a different *family* from the
other three arms -- GPTQ and AWQ both calibrate, RTN and DynQuant both use uniform
grids, and NF4 is the only one that changes the data type. And it is, by a wide margin,
the 4-bit quantization people actually run: it is what ``load_in_4bit=True`` gives you,
which makes it the honest answer to "what would I have used instead".

Accounting, which is the part that is easy to get wrong
-------------------------------------------------------
NF4's metadata is not an fp16 scale per group of 128. It is an absmax per block of 64,
and with double quantization that absmax is itself quantized to 8 bits with a second
fp32 scale per 256 blocks. So the overhead is ``8/64 + 32/(64*256)`` = 0.127 bits per
weight, against ~0.148 for the group-128 fp16-scale-plus-zero-point convention the other
arms use. Reusing the other arms' formula here would misreport the size of the one arm
whose metadata layout differs, in a table whose whole point is that size is measured
rather than assumed.
"""

from __future__ import annotations

import json
from typing import Any

from common import RUN_DIR, model_slug, run_eval, set_seed
from stage8_baselines import IGNORE, _unique_params

BLOCK = 64
DOUBLE_QUANT_BITS = 8 / BLOCK + 32 / (BLOCK * 256)
"""0.127 bits per weight. See the module docstring -- this is not the group-128
convention and must not be swapped for it."""


def accounted_bytes(source: str) -> dict[str, Any]:
    """NF4's stored cost, on the same total-bits-over-parameters convention as stage 5.

    Walks a meta-device copy of the architecture rather than the loaded model, for the
    same reason :func:`stage8_baselines.accounted_bytes` does: a bitsandbytes model's
    Linears have been replaced by ``Linear4bit`` holding a flat uint8 buffer, so a walk
    over the live module tree cannot see the original shapes at all.
    """
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(source)
    with torch.device("meta"):
        ref = AutoModelForCausalLM.from_config(config)

    quantized = 0
    counted: set[int] = set()
    for name, module in ref.named_modules():
        if not isinstance(module, nn.Linear) or any(name.endswith(p) for p in IGNORE):
            continue
        quantized += module.weight.numel() * (4 + DOUBLE_QUANT_BITS)
        counted.add(id(module.weight))

    unique = _unique_params(ref)
    fp16 = sum(p.numel() * 16 for p in unique if id(p) not in counted)
    total = sum(p.numel() for p in unique)
    return {
        "accounted_bits": round((quantized + fp16) / total, 4),
        "accounted_gib": round((quantized + fp16) / 8 / 2**30, 4),
        "fp16_bits_share": round(fp16 / (quantized + fp16), 4),
        "params": total,
    }


def main() -> None:
    import argparse

    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(RUN_DIR / "finetuned"))
    parser.add_argument("--name", default="stage8_nf4_4b")
    parser.add_argument("--label", default="bnb NF4 4-bit")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    set_seed()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            # On, because it is on in every recipe that ships this and off would be a
            # hand-tuned variant of the baseline rather than the baseline.
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    model.config.use_cache = True

    meta = {
        "method": "bnb-nf4",
        "bits": 4,
        "block_size": BLOCK,
        "double_quant": True,
        "ignore": IGNORE,
        "source": str(args.model),
        "model_id": model_slug(),
        **accounted_bytes(args.model),
    }
    print(json.dumps(meta, indent=2), flush=True)
    run_eval(model, label=args.label, name=args.name, limit=args.limit, extra=meta)


if __name__ == "__main__":
    main()
