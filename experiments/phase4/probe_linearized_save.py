#!/usr/bin/env python3
"""Does a linearized MoE checkpoint load back as the weights that were written?

The panel scores in memory, so nothing it reports depends on this. The release does.
``baselines_lfm2.py save`` writes a checkpoint by linearizing the expert banks, running
the recipe over the resulting ``nn.Linear`` modules, and calling ``save_pretrained``.
The names that reach disk are therefore the linearized ones, and putting them back is a
separate llm-compressor feature -- ``ARCH_TO_2D_MAPPINGS``, applied through
``set_save_conversion_mapping`` -- registered for ``deepseek_v4`` and ``qwen2_moe`` and
for nothing else. ``lfm2_moe`` linearizes through the generic protocol, so the surgery
runs and its inverse does not exist.

What that costs is not an exception. It is a load report:

    model.layers.{1,2,3}.feed_forward.experts.{0..3}.gate_proj.weight | UNEXPECTED
    model.layers.{1,2,3}.feed_forward.experts.gate_up_proj            | MISSING

Every expert tensor written is dropped; every bank the config describes is created fresh
from ``from_config``; ``from_pretrained`` returns a model and the logits are finite. On
LFM2.5-8B-A1B that is 91.5% of the weights randomly initialized behind a table nobody
reads. This probe turns the table into a number: a 4-bit group of 32 values can hold 16
distinct levels, and a freshly initialized one holds 32, so counting them says which
weights came back without trusting a warning.

The second half is the control. ``dynquant quantize`` never linearizes -- it encodes the
bank in place and ``save_pretrained`` writes it back through the model's own registered
conversion, which is the canonical ``experts.<i>.w{1,2,3}`` layout that both transformers
and vLLM read. Same instrument, same model, and the difference between the two rows is
the finding.

Run un-quantized on the llm-compressor side, because the naming failure is module
surgery and has nothing to do with the recipe -- which keeps this hermetic, with no
calibration set and no download. It was also measured once through the shipped path with
``--method rtn --bits 4``: 108 packed expert tensors written, all 108 ``UNEXPECTED``, the
banks back at 32 distinct values per 32-value group.

Usage::

    python experiments/phase4/probe_linearized_save.py --out probe_linsave.json

Needs one environment holding both ``llmcompressor`` and ``dynquant``; on the campaign
box that is ``venv-llmc`` with ``PYTHONPATH=packages/dynquant-core/src``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/dynquant-core/src"))

GROUP = 32
"""The tiny config's hidden size is 64, so the campaign's 128 does not divide a row. The
question is about names, and a group size does not have an opinion about names."""


def build_tiny(where: Path) -> Any:
    """A four-layer ``lfm2_moe`` with three MoE layers, built from config and saved.

    Small on purpose. Linearization is module surgery over weights it never reads, so
    every name this produces is the name the 8B produces, and the 8B costs a download
    and 17 GB to say the same thing.
    """
    from transformers import AutoModelForCausalLM, Lfm2MoeConfig

    config = Lfm2MoeConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_dense_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["conv", "full_attention", "conv", "conv"],
        tie_word_embeddings=True,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(str(where))
    return config


def distinct_per_group(path: Path, *, groups: int = 4) -> dict[str, Any]:
    """Load the directory and count distinct values in the first groups of a bank.

    The count is the whole instrument. ``n`` bits over a group of ``GROUP`` values admits
    at most ``2**n`` levels, so a group holding ``GROUP`` distinct values was never
    quantized -- and on a directory that was written quantized, that means the tensor on
    disk is not the tensor in the model.
    """
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(path), dtype=torch.float32)
    state = model.state_dict()
    name = next(k for k in state if k.endswith("experts.gate_up_proj"))
    bank = state[name]
    rows = bank[0, 0].reshape(-1, GROUP)[:groups]
    return {
        "bank": name,
        "shape": list(bank.shape),
        "distinct_per_group": [int(torch.unique(row).numel()) for row in rows],
        "group_size": GROUP,
    }


def written_expert_keys(path: Path) -> list[str]:
    from safetensors.torch import load_file

    index = path / "model.safetensors.index.json"
    keys = (
        json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        if index.exists()
        else load_file(str(path / "model.safetensors"))
    )
    return sorted(k for k in keys if "experts" in k)


def linearized_side(source: Path, out: Path) -> dict[str, Any]:
    """Save what ``baselines_lfm2.py save`` would save, and read it back."""
    import torch
    from llmcompressor.modeling.moe.conversion_mappings import has_linearize_load_mappings
    from llmcompressor.modeling.moe.linearize import get_non_linearized_moes, linearize_moe
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(source))
    if not hasattr(config, "hidden_act"):
        config.hidden_act = "silu"
    model = AutoModelForCausalLM.from_pretrained(str(source), config=config, dtype=torch.float32)

    before = len(get_non_linearized_moes(model))
    linearize_moe(model)
    model.save_pretrained(str(out))
    return {
        "path": str(out),
        "model_type": config.model_type,
        "banks_linearized": before,
        "has_linearize_load_mappings": has_linearize_load_mappings(config.model_type),
        "expert_keys_written": len(written_expert_keys(out)),
        "first_key_written": (written_expert_keys(out) or [None])[0],
        "reloaded": distinct_per_group(out),
    }


def dynquant_side(source: Path, out: Path, *, bits: int) -> dict[str, Any]:
    """Save what ``dynquant quantize`` saves, through the same instrument."""
    from dynquant.cli import main as dynquant_main

    code = dynquant_main(
        [
            "quantize",
            str(source),
            "-o",
            str(out),
            "--uniform",
            str(bits),
            "--group-size",
            str(GROUP),
            "--device",
            "cpu",
            "--quiet",
        ]
    )
    if code:
        raise SystemExit(f"`dynquant quantize` exited {code}")
    return {
        "path": str(out),
        "bits": bits,
        "expert_keys_written": len(written_expert_keys(out)),
        "first_key_written": (written_expert_keys(out) or [None])[0],
        "reloaded": distinct_per_group(out),
    }


def verdict(payload: dict[str, Any], *, bits: int) -> str:
    """One line, and it must be derived rather than asserted."""
    allowed = 2**bits
    left = max(payload["linearized"]["reloaded"]["distinct_per_group"])
    right = max(payload["dynquant"]["reloaded"]["distinct_per_group"])
    lost = left > min(allowed, GROUP) or left == GROUP
    kept = right <= allowed
    if lost and kept:
        return (
            f"the linearized directory came back at {left} distinct values per {GROUP} and "
            f"the dynquant one at {right}, against {allowed} for {bits}-bit: the expert "
            "weights written by the first are not the expert weights it loads"
        )
    if not lost and kept:
        return (
            f"both came back within {allowed} levels. Either llm-compressor registered a "
            "mapping for this architecture or transformers learned the names -- read "
            "`has_linearize_load_mappings` above before believing the second"
        )
    return f"unexpected: linearized {left}, dynquant {right}, {bits}-bit allows {allowed}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("probe_linsave_work"))
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--keep", action="store_true", help="leave the three directories behind for inspection"
    )
    args = parser.parse_args(argv)

    work = args.work
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    source = work / "source"
    build_tiny(source)

    payload: dict[str, Any] = {
        "group_size": GROUP,
        "bits": args.bits,
        "linearized": linearized_side(source, work / "linearized"),
        "dynquant": dynquant_side(source, work / "dynquant", bits=args.bits),
    }
    payload["verdict"] = verdict(payload, bits=args.bits)

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    if not args.keep:
        shutil.rmtree(work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
