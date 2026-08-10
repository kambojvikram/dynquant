#!/usr/bin/env python3
"""Is ``delinearize_state_dict`` the exact inverse of ``linearize_moe``?

``probe_linearize_mapping.py`` derives *where* each linearized ``nn.Linear`` came from and
asserts the slices tile the banks. That is necessary and not sufficient: a mapping can be
complete and still be applied wrongly -- parts concatenated in the wrong order, experts
stacked in the wrong order, a transpose applied on the way back that was not applied on the
way out. Each of those produces a checkpoint that loads, runs, and is wrong, which on this
architecture is the failure mode that already cost 91.5% of the weights once.

So this closes the loop on the only test that admits no interpretation. Snapshot the banked
state dict, linearize, rebuild it through the inverse, and require the two to be *identical*:
same keys, same shapes, same bits. Not close -- identical. Nothing here recomputes a float,
so any tolerance at all would be tolerance for a bug.

The second half loads the rebuilt state dict into a fresh ``from_config`` model with
``strict=True`` and reports what transformers says about it. ``probe_linearized_save.py``
established that a wrong key set on this architecture does not raise: unexpected keys are
dropped, missing ones are reinitialized, and ``from_pretrained`` returns a model with finite
logits. ``strict=True`` is what turns that silence into an exception, so it is the load this
probe performs and the one the publish path should use.

Usage::

    python experiments/phase4/probe_delinearize.py --out probe_delin.json

Needs ``llmcompressor`` and a transformers that knows ``lfm2_moe``. No GPU.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/dynquant-core/src"))

from baselines_lfm2 import delinearize_state_dict, expert_rules
from probe_linearized_save import build_tiny

GEOMETRY: dict[str, Any] = {"moe_intermediate_size": 24}
"""Rectangular, so a transposed rebuild is a shape error and not merely a value error.

With the builder's square default a wrongly-transposed bank would still be the right shape,
and this probe would have to catch it on values alone. It would -- but a test that fails two
ways is worth more than one that fails one way, and this costs nothing."""


def compare(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Key-set and bit-level agreement between the original and the rebuilt state dict."""
    import torch

    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    shape_mismatch = {
        key: {"before": list(before[key].shape), "after": list(after[key].shape)}
        for key in sorted(set(before) & set(after))
        if before[key].shape != after[key].shape
    }
    differing = sorted(
        key
        for key in set(before) & set(after)
        if key not in shape_mismatch and not torch.equal(before[key], after[key])
    )
    return {
        "keys_before": len(before),
        "keys_after": len(after),
        "missing": missing,
        "extra": extra,
        "shape_mismatch": shape_mismatch,
        "differing": differing,
        "identical": not (missing or extra or shape_mismatch or differing),
    }


def strict_load(source: Path, rebuilt: dict[str, Any]) -> dict[str, Any]:
    """Load the rebuilt state dict into the banked architecture with ``strict=True``."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(source))
    if not hasattr(config, "hidden_act"):
        config.hidden_act = "silu"
    model = AutoModelForCausalLM.from_config(config)
    model = model.to(torch.float32)
    incompatible = model.load_state_dict(rebuilt, strict=True)
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3, 4]])).logits
    return {
        "strict": True,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "logits_finite": bool(torch.isfinite(logits).all()),
    }


def run(source: Path) -> dict[str, Any]:
    import torch
    from llmcompressor.modeling.moe.linearize import get_non_linearized_moes, linearize_moe
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(source))
    if not hasattr(config, "hidden_act"):
        config.hidden_act = "silu"
    model = AutoModelForCausalLM.from_pretrained(str(source), config=config, dtype=torch.float32)

    before = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
    banks = len(get_non_linearized_moes(model))
    linearize_moe(model)

    rules = expert_rules()
    rebuilt = delinearize_state_dict(model, rules)

    payload: dict[str, Any] = {
        "banks_linearized": banks,
        "rules": rules,
        "round_trip": compare(before, rebuilt),
    }
    payload["reload"] = strict_load(source, rebuilt)
    return payload


def verdict(payload: dict[str, Any]) -> str:
    trip = payload["round_trip"]
    reload_ = payload["reload"]
    if not trip["identical"]:
        return (
            f"the rebuilt state dict is not the original: {len(trip['missing'])} missing, "
            f"{len(trip['extra'])} extra, {len(trip['shape_mismatch'])} reshaped, "
            f"{len(trip['differing'])} differing in value -- the inverse is wrong"
        )
    if reload_["missing_keys"] or reload_["unexpected_keys"]:
        return (
            "the rebuilt state dict round-trips but does not fit the banked architecture: "
            f"{len(reload_['missing_keys'])} missing, {len(reload_['unexpected_keys'])} "
            "unexpected. strict=True caught what from_pretrained would have swallowed"
        )
    return (
        f"all {trip['keys_before']} tensors survive linearize -> delinearize bit-identically, "
        f"and the result loads into the banked architecture under strict=True with finite "
        f"logits: the inverse is exact and the names are the ones the architecture expects"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("probe_delin_work"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)

    work = args.work
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    source = work / "source"
    build_tiny(source, **GEOMETRY)

    payload = run(source)
    payload["verdict"] = verdict(payload)

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    if not args.keep:
        shutil.rmtree(work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
