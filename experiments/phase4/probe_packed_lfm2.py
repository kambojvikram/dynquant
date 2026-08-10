#!/usr/bin/env python3
"""Does the packed DynQuant container round-trip a *real* LFM2 MoE, not a stand-in?

The report counts ``dq_4b`` and ``dq_3b`` as the two publishable variants, and the
evidence under that was a synthetic MoE whose ``forward`` is ``Lfm2MoeExperts``' loop
copied verbatim. A copy of a loop is not the class: it shares no config plumbing, no
registered checkpoint conversion, no ``from_pretrained`` path, and none of the names
transformers folds into banks at load. This closes that gap at the only scale available
while the GPU is held -- a four-layer ``Lfm2MoeForCausalLM`` built from ``Lfm2MoeConfig``,
the same class the 8B is, with the same seed and config as ``probe_linearized_save.py``.

Three questions, and the last two are the ones that fail quietly.

1. Is the bank held *packed* after load, or dequantized back into a dense parameter? A
   container that silently rehydrated would pass every accuracy check and save nothing,
   so this counts :class:`~dynquant.runtime.linear.DynQuantExpertBank` instances and any
   3-D expert parameter still sitting dense beside them.
2. Are the values the same values? ``dynquant quantize`` writes the canonical
   ``experts.<i>.w{1,2,3}`` layout that transformers folds back into banks by its own
   registered conversion, and is known to round-trip, so it is the oracle. If the packed
   bank mis-slices per expert -- the ``out_features``-of-a-flattened-bank failure -- this
   is where it shows, because every expert still produces finite numbers of the right
   shape either way.
3. Did the two commands quantize the same modules? They do not, and the difference is
   structural rather than a bug: in-place quantization rewrites a tensor and can reach
   anything with a weight, while the packed container has to put a module where the
   weight was and cannot reach a depthwise ``nn.Conv1d`` kernel. So the exported
   directory leaves those at full precision. Small -- 18 kernels of ``[2048, 1, 3]`` on
   LFM2.5-8B-A1B -- and in the safe direction, but it means the published artifact is
   not bit-for-bit the artifact the panel scored, and that is worth printing rather than
   discovering.

Why the comparison is in weight space and the logits are only context
---------------------------------------------------------------------
Both were logits first, and the answer was wrong twice. A top-2 router is discontinuous:
a perturbation far below the quantization error can flip which expert a token reaches
and move a logit by a hundred times the weight delta that caused it. Worse, the two
directories are written at different *precisions* -- ``quantize`` copies its fp32
reconstruction into a bf16 parameter and the packed container stores int32 words it
dequantizes at load -- so loading both at fp32 makes one of them pay a bf16 rounding the
other never does, and charges the container for it. Reading the weights, at the dtype
the directories were written at, asks the question the container is actually responsible
for. The logit gap is still reported, and still crosses the router, so it is context and
not the verdict.

**``register_hf_quantizer()`` is the load, not decoration.** ``import dynquant`` does not
register -- deliberately, to keep the CLI free of a 9.8 s transformers import -- and
without the call ``from_pretrained`` logs "Unknown quantization type", skips the
quantization and returns a randomly initialised model with no exception. The first run of
this probe did exactly that and reported a catastrophe that was its own.

Usage::

    python experiments/phase4/probe_packed_lfm2.py --out probe_packed.json

Needs ``transformers`` new enough to carry ``Lfm2MoeConfig``. CPU-only and hermetic: no
download, no calibration set, roughly 8 M parameters.
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
"""Hidden size is 64 in the tiny config, so the campaign's 128 does not divide a row."""

BITS = 4
SEED = 0
PROMPT = 24
"""Tokens, teacher-forced. Long enough to cross the conv cache and reach the attention
layer at index 1; short enough that the whole probe is seconds."""


def build_tiny(where: Path) -> None:
    """The four-layer ``lfm2_moe`` ``probe_linearized_save.py`` builds, same seed.

    Same config on purpose: the two probes answer opposite halves of one question --
    whether a directory's expert weights are the weights it loads -- and a reader
    comparing their numbers should not have to check that the models match.
    """
    import torch
    from transformers import AutoModelForCausalLM, Lfm2MoeConfig

    torch.manual_seed(SEED)
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
    AutoModelForCausalLM.from_config(config).save_pretrained(str(where))


def run_cli(argv: list[str]) -> None:
    from dynquant.cli import main as dynquant_main

    code = dynquant_main(argv)
    if code:
        raise SystemExit(f"`dynquant {argv[0]}` exited {code}")


def load(path: Path) -> Any:
    """``from_pretrained``, with the registration that makes it mean anything.

    ``dtype="auto"`` and not a literal: each directory records the precision it was
    written at, and forcing a common one here is what made an earlier version of this
    probe charge the container for a bf16 rounding that belongs to the other side.
    """
    from transformers import AutoModelForCausalLM

    import dynquant

    dynquant.register_hf_quantizer()
    return AutoModelForCausalLM.from_pretrained(str(path), dtype="auto")


def bank_census(model: Any) -> dict[str, Any]:
    """What is holding the expert weights: packed modules, or dense parameters.

    Asked of the *loaded module tree* rather than of the key names on disk. Comparing
    written keys against ``state_dict()`` names looks like the same question and is
    not: transformers converts ``experts.<i>.w{1,2,3}`` into batched banks at load, so
    every canonical checkpoint reports its own expert keys "missing" and a directory
    known to round-trip scores as a catastrophe. That mistake is why this function
    exists in this shape.
    """
    import torch

    from dynquant.runtime.linear import DynQuantExpertBank

    packed = [n for n, m in model.named_modules() if isinstance(m, DynQuantExpertBank)]
    dense = [
        n
        for n, p in model.named_parameters()
        if isinstance(p, torch.nn.Parameter) and p.ndim == 3 and "experts" in n
    ]
    return {"packed_banks": sorted(packed), "dense_expert_params": sorted(dense)}


def weight_agreement(packed_model: Any, inplace_model: Any, names: list[str]) -> dict[str, Any]:
    """Compare every module the exporter claims it packed against the in-place tensor.

    Driven by the exporter's own module list rather than by an ``isinstance`` walk of
    the tree, so the count reconciles by construction. The walk version compared 19 of
    23 and looked clean: it silently dropped the embedding, whose class is not
    ``DynQuantLinear``, and all three routers, which come back as stock
    ``Lfm2MoeTopKRouter`` because the loader restores them dense. A router restored to
    the wrong values is a live failure mode -- 22 of them on the 8B -- and no number in
    that version would have moved.

    Two kinds of module therefore land here and both are checked the same way: those
    still holding a ``weight_qt`` after load, and those the loader put back dense. The
    per-kind counts are reported rather than only the total, because a total that
    reconciles does not say *which* rows it reconciled.

    The lookup tries the module's own name before ``name.weight``: a bank replaced a
    bare ``nn.Parameter`` and the in-place model still holds one under exactly that
    name. Getting the order backwards skips all six banks -- the only modules this
    probe exists to check -- while reporting a clean result for everything else.
    """
    dense = dict(inplace_model.named_parameters())
    worst_name, worst, scale = None, 0.0, 0.0
    kinds: dict[str, int] = {}
    unmatched = []
    for name in names:
        module = packed_model.get_submodule(name)
        reference = dense.get(name, dense.get(f"{name}.weight"))
        if reference is None:
            unmatched.append(name)
            continue
        held = getattr(module, "weight_qt", None)
        kinds[type(module).__name__] = kinds.get(type(module).__name__, 0) + 1
        # At the reference's own dtype, and not at fp32. Both directories are written at
        # the model's storage precision: the in-place path copies an fp32 reconstruction
        # into a bf16 parameter, while the packed path keeps exact int words it
        # dequantizes at load. Asking for fp32 spares the packed side a rounding the
        # other one already paid to disk and then charges the container for it -- which
        # is exactly the false alarm the first version of this probe reported.
        got = (
            held.dequantize(dtype=reference.dtype).float()
            if held is not None
            else module.weight.detach().float()
        )
        want = reference.detach().float()
        delta = (got - want).abs().max().item()
        scale = max(scale, want.abs().max().item())
        if delta > worst:
            worst_name, worst = name, delta
    return {
        "modules_compared": len(names) - len(unmatched),
        "by_class": dict(sorted(kinds.items())),
        "worst_delta": worst,
        "worst_module": worst_name,
        "max_abs_weight": scale,
        "unmatched": sorted(unmatched),
    }


def module_sets(names: list[str], reference_model: Any, inplace_model: Any) -> dict[str, Any]:
    """What each command quantized -- read off the tensors, not off the console.

    ``export`` states its module list in ``config.json``; ``quantize`` states nothing, so
    the set it touched is recovered by diffing its weights against the source. Whatever
    the diff finds that the list does not name is a tensor the published directory
    carries at full precision while the scored one does not.
    """
    import torch

    packed = set(names)
    before = dict(reference_model.named_parameters())
    rewritten = {
        name
        for name, after in inplace_model.named_parameters()
        if name in before
        and before[name].shape == after.shape
        and not torch.equal(before[name].float(), after.float())
    }
    only_inplace = sorted(
        name
        for name in rewritten
        if name not in packed and name.removesuffix(".weight") not in packed
    )
    return {
        "packed_modules": len(packed),
        "inplace_tensors": len(rewritten),
        "quantized_in_place_but_left_dense_in_the_export": only_inplace,
    }


def logits(model: Any) -> Any:
    import torch

    model.eval()
    ids = torch.arange(PROMPT, dtype=torch.long).remainder(256).unsqueeze(0)
    with torch.no_grad():
        return model(ids).logits.float()


def verdict(payload: dict[str, Any]) -> str:
    """Derived, and it has to be able to say no.

    Three terms, because each is satisfiable on its own by a broken container. One that
    dequantized everything on load would score a *perfect* delta, so the census stands
    apart from the number. One that held its buffers packed but sliced them wrong would
    pass the census, so the delta is not dropped for it. And one that quietly skipped a
    module would report a clean delta over whatever it did compare, so the exporter's
    own count has to be the denominator.

    Where the agreement is not exact the delta is read against the error the encoding
    itself cost, rather than against a tolerance somebody picked, because the ratio is
    the claim: whatever the encoding did to the weight, re-reading it through the
    container costs a small fraction of that.
    """
    held = payload["packed"]["packed_banks"]
    loose = payload["packed"]["dense_expert_params"]
    agree = payload["weights"]
    delta, scale = agree["worst_delta"], agree["max_abs_weight"]
    claimed = payload["modules"]["packed_modules"]
    budget = payload["quantization_error"] * scale
    if not held or loose:
        return (
            f"NOT HELD PACKED: {len(held)} DynQuantExpertBank(s) and {len(loose)} dense "
            f"3-D expert parameter(s) after load -- the container gave the weights back"
        )
    if agree["unmatched"]:
        return (
            f"UNCHECKED: {len(agree['unmatched'])} of the {claimed} module(s) the export "
            f"claims to have packed have no in-place counterpart to compare against: "
            f"{', '.join(agree['unmatched'])}"
        )
    if delta and delta * 100 > budget:
        return (
            f"MISMATCH: the container moves a weight by {delta:.3e}, only "
            f"{budget / delta:.1f}x inside the {budget:.3e} the encoding itself moved "
            f"it, so the container is doing arithmetic of its own"
        )
    how_close = "bit for bit" if not delta else f"to {delta:.3e}, inside {budget:.3e}"
    return (
        f"{len(held)} bank(s) held packed after load, and across all {claimed} packed "
        f"module(s) -- {_kinds(agree['by_class'])} -- the container reproduces the "
        f"in-place encoding of a real {payload['model_class']} {how_close}"
    )


def _kinds(by_class: dict[str, int]) -> str:
    return ", ".join(f"{n} {name}" for name, n in by_class.items())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("probe_packed_work"))
    parser.add_argument("--bits", type=int, default=BITS)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)

    import torch

    work = args.work
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    source = work / "source"
    build_tiny(source)

    common = [
        "--uniform",
        str(args.bits),
        "--group-size",
        str(GROUP),
        "--device",
        "cpu",
        "--quiet",
    ]
    run_cli(["export", str(source), "-o", str(work / "packed"), *common])
    run_cli(["quantize", str(source), "-o", str(work / "inplace"), *common])

    config = json.loads((work / "packed" / "config.json").read_text(encoding="utf-8"))
    packed_names = sorted(config["quantization_config"]["modules"])

    packed_model = load(work / "packed")
    inplace_model = load(work / "inplace")
    reference_model = load(source)

    packed, inplace, reference = (
        logits(packed_model),
        logits(inplace_model),
        logits(reference_model),
    )
    payload: dict[str, Any] = {
        "bits": args.bits,
        "group_size": GROUP,
        "seed": SEED,
        "model_class": type(packed_model).__name__,
        "model_type": packed_model.config.model_type,
        "packed": bank_census(packed_model),
        "inplace": bank_census(inplace_model),
        "weights": weight_agreement(packed_model, inplace_model, packed_names),
        "modules": module_sets(packed_names, reference_model, inplace_model),
        # Relative Frobenius error of the encoding, so the weight delta has something
        # to be small against that is not an arbitrary tolerance.
        "quantization_error": _encoding_error(reference_model, inplace_model),
        "logits": {
            "max_abs": round(reference.abs().max().item(), 6),
            "container_gap": (packed - inplace).abs().max().item(),
            "quantization_gap": (inplace - reference).abs().max().item(),
            "note": "crosses a top-2 router, so it is context; the weights are the test",
        },
        "logits_finite": bool(torch.isfinite(packed).all()),
    }
    payload["verdict"] = verdict(payload)

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    if not args.keep:
        shutil.rmtree(work)
    return 0


def _encoding_error(reference_model: Any, inplace_model: Any) -> float:
    """Relative error the encoding cost, over the tensors it touched."""

    before = dict(reference_model.named_parameters())
    num, den = 0.0, 0.0
    for name, after in inplace_model.named_parameters():
        original = before.get(name)
        if original is None or original.shape != after.shape:
            continue
        a, b = original.detach().float(), after.detach().float()
        num += float(((a - b) ** 2).sum())
        den += float((a**2).sum())
    return (num / den) ** 0.5 if den else 0.0


if __name__ == "__main__":
    sys.exit(main())
