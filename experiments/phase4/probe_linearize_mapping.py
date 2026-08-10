#!/usr/bin/env python3
"""Which bank slice did each linearized ``nn.Linear`` come from? Measure it, don't read it.

``probe_linearized_save.py`` established the loss: ``linearize_moe`` renames every batched
expert bank into ``experts.<i>.{gate,up,down}_proj``, llm-compressor registers the inverse
for ``deepseek_v4`` and ``qwen2_moe`` only, and on ``lfm2_moe`` a saved checkpoint reloads
with all of it dropped and the banks reinitialized. This probe is the next question: if the
inverse is going to be written here, what exactly is it?

The obvious way to answer is to read ``llmcompressor/modeling/moe/linearize.py`` and
transcribe what it does. That is how this project has been wrong four times -- a second copy
of a registry agrees with the first until a release moves one of them, and the disagreement
surfaces as a silent transpose rather than an exception. Two of the three facts needed here
are exactly of that kind: whether a bank is stored ``[E, in, out]`` or ``[E, out, in]``, and
whether ``gate`` is the first half of the fused projection or the second. Reading the wrong
end of either produces a checkpoint that loads, runs, and is wrong.

So the mapping is derived by comparison instead. Build a tiny ``lfm2_moe`` with random
weights, snapshot every bank, linearize, and for each ``nn.Linear`` that appears, search the
snapshot for the slice whose values *are* its weight -- trying both orientations and both
halves of a fused bank. Random initialization makes the match unique: two distinct slices
agreeing element-for-element by chance is not a thing that happens at these sizes. The result
is a rule that describes the installed llm-compressor, not the one that was read about, and a
release that changes the layout changes this probe's output rather than corrupting a release.

The coverage assertion is the other half. Every element of every bank must be claimed by
exactly one Linear. A mapping that is merely self-consistent can still be missing an expert;
one that accounts for every element cannot.

Usage::

    python experiments/phase4/probe_linearize_mapping.py --out probe_linmap.json

Needs ``llmcompressor`` and a transformers that knows ``lfm2_moe``; on the campaign box that
is ``venv-llmc``. No GPU: this is module surgery over weights the surgery never reads, so it
runs under ``CUDA_VISIBLE_DEVICES=""`` and does not queue behind the panel.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_linearized_save import build_tiny

MAX_SPLITS = 2
"""A fused bank holds gate and up; nothing in this family fuses three. Searching splits we
have no reason to expect would only widen the chance of a coincidental match."""


def _bank_parameters(model: Any) -> dict[str, Any]:
    """Every batched expert parameter in the tree, by qualified name.

    Scoped to the modules llm-compressor's own ``get_non_linearized_moes`` returns, rather
    than to every 3-D parameter in the tree. The first draft did the latter, on the stated
    grounds that "nothing else in the model is a 3-D parameter", and that is false in a way
    worth keeping on the record: an ``lfm2_moe`` conv layer stores ``[channels, 1, kernel]``,
    which is 3-D, and 18 of the real model's 24 layers are conv. The probe matched every
    expert module correctly and then reported the conv weights as bank elements no Linear had
    claimed -- a true statement about a tensor that was never a bank, which read as a defect
    in the mapping.

    Using llm-compressor's detector keeps this from being a third opinion about what a bank
    is. It is the same predicate ``linearize_moe`` dispatches on, so a release that changes
    the answer changes both sides together.
    """
    from llmcompressor.modeling.moe.linearize import get_non_linearized_moes

    banks: dict[str, Any] = {}
    for moe in get_non_linearized_moes(model):
        module = moe[1] if isinstance(moe, tuple) else moe
        prefix = next(
            (name for name, candidate in model.named_modules() if candidate is module), None
        )
        assert prefix is not None, "a detected MoE module is not in the tree it came from"
        for suffix, param in module.named_parameters():
            if param.ndim == 3:
                banks[f"{prefix}.{suffix}"] = param.detach().clone()
    return banks


def _linear_weights(model: Any) -> dict[str, Any]:
    import torch

    return {
        name: module.weight.detach()
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }


def _views(slice_: Any) -> list[tuple[str, Any]]:
    """A bank slice as stored, and transposed. One of the two is the Linear's weight.

    ``F.linear`` computes ``x @ W.T`` for ``W = [out, in]``, and a batched bank that a
    module multiplies as ``x @ B[e]`` stores ``[in, out]``. Both conventions are in use in
    transformers -- which is the whole reason this is measured rather than assumed.
    """
    return [("as_stored", slice_), ("transposed", slice_.t())]


def _find_source(weight: Any, banks: dict[str, Any]) -> dict[str, Any] | None:
    """The (bank, expert, orientation, split) whose values equal ``weight``, or ``None``.

    Equality is exact. These are the same floats moved by module surgery, not recomputed,
    so anything short of exact would be admitting a match this probe cannot justify.
    """
    import torch

    out_features, in_features = weight.shape
    for bank_name, bank in banks.items():
        experts = bank.shape[0]
        for expert in range(experts):
            for orientation, view in _views(bank[expert]):
                rows, cols = view.shape
                for splits in range(1, MAX_SPLITS + 1):
                    if rows % splits or rows // splits != out_features or cols != in_features:
                        continue
                    width = rows // splits
                    for part in range(splits):
                        candidate = view[part * width : (part + 1) * width, :]
                        if torch.equal(candidate, weight):
                            return {
                                "bank": bank_name,
                                "expert": expert,
                                "orientation": orientation,
                                "splits": splits,
                                "part": part,
                            }
    return None


def _rule(linear_name: str, source: dict[str, Any]) -> dict[str, Any]:
    """Strip the layer index and the expert index, leaving the structural claim.

    A mapping that held only for layer 1 expert 0 would be a coincidence report. What the
    inverse needs is the rule, and the rule is what survives removing the indices -- which
    also gives :func:`main` something to assert every layer and expert agrees on.
    """
    import re

    generic_linear = re.sub(r"\.(\d+)\.", ".{}.", linear_name)
    return {
        "linear": generic_linear,
        "bank": re.sub(r"\.(\d+)\.", ".{}.", source["bank"]),
        "orientation": source["orientation"],
        "splits": source["splits"],
        "part": source["part"],
    }


def derive(source_dir: Path) -> dict[str, Any]:
    """Load, snapshot, linearize, and match every Linear the surgery produced."""
    import torch
    from llmcompressor.modeling.moe.linearize import get_non_linearized_moes, linearize_moe
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(source_dir))
    if not hasattr(config, "hidden_act"):
        config.hidden_act = "silu"
    model = AutoModelForCausalLM.from_pretrained(
        str(source_dir), config=config, dtype=torch.float32
    )

    banks = _bank_parameters(model)
    before_linears = set(_linear_weights(model))
    banks_before = len(get_non_linearized_moes(model))
    linearize_moe(model)
    banks_after = len(get_non_linearized_moes(model))

    produced = {
        name: weight
        for name, weight in _linear_weights(model).items()
        if name not in before_linears
    }

    matched: dict[str, dict[str, Any]] = {}
    unmatched: list[str] = []
    claimed = {name: torch.zeros_like(bank, dtype=torch.bool) for name, bank in banks.items()}
    for name, weight in sorted(produced.items()):
        found = _find_source(weight, banks)
        if found is None:
            unmatched.append(name)
            continue
        matched[name] = found
        view = claimed[found["bank"]][found["expert"]]
        if found["orientation"] == "transposed":
            view = view.t()
        width = view.shape[0] // found["splits"]
        view[found["part"] * width : (found["part"] + 1) * width, :] = True

    rules: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for name, found in matched.items():
        rule = _rule(name, found)
        seen = rules.setdefault(rule["linear"], rule)
        if seen != rule:
            conflicts.append({"linear": name, "expected": seen, "got": rule})

    unclaimed = {name: int((~mask).sum()) for name, mask in claimed.items() if not bool(mask.all())}
    return {
        "banks_before": banks_before,
        "banks_after": banks_after,
        "bank_parameters": {name: list(bank.shape) for name, bank in banks.items()},
        "linears_produced": len(produced),
        "linears_matched": len(matched),
        "unmatched": unmatched,
        "rules": sorted(rules.values(), key=lambda r: (r["bank"], r["part"], r["linear"])),
        "rule_conflicts": conflicts,
        "unclaimed_elements": unclaimed,
        "example": dict(sorted(matched.items())[0][1], linear=sorted(matched)[0])
        if matched
        else None,
    }


def verdict(payload: dict[str, Any]) -> str:
    """One line, derived. It has to be able to say the mapping is *not* recoverable."""
    if payload["banks_before"] == 0:
        return "nothing linearized, so there is no mapping to derive -- check the config"
    if payload["unmatched"]:
        return (
            f"{len(payload['unmatched'])} of {payload['linears_produced']} linearized modules "
            "hold values that are in no bank slice: linearization is not a pure permutation "
            "here and an inverse written from this probe would be wrong"
        )
    if payload["rule_conflicts"]:
        return (
            f"{len(payload['rule_conflicts'])} module(s) disagree with the rule their siblings "
            "follow, so the mapping is not structural and cannot be applied by name"
        )
    if payload["unclaimed_elements"]:
        return (
            f"every linearized module matched, but {payload['unclaimed_elements']} bank "
            "element(s) are claimed by none of them -- the inverse would leave them stale"
        )
    kinds = ", ".join(sorted({r["orientation"] for r in payload["rules"]}))
    return (
        f"all {payload['linears_matched']} linearized modules are exact slices of the banks "
        f"({kinds}), every bank element is claimed exactly once, and {len(payload['rules'])} "
        "structural rule(s) cover every layer and expert: the inverse is well defined"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("probe_linmap_work"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--keep", action="store_true", help="leave the source directory behind")
    args = parser.parse_args(argv)

    work = args.work
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    source = work / "source"
    # Rectangular on purpose: 2 * 24 != 64, so a fused bank is [E, 48, 64] and its two ends
    # are distinguishable by shape as well as by value. The builder's default geometry is
    # square, which would let a transposed reading survive with the right answer for the
    # wrong reason -- the same way a square test fixture let a transposed `_out_features`
    # mutation through earlier in this project.
    build_tiny(source, moe_intermediate_size=24)

    payload = derive(source)
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
