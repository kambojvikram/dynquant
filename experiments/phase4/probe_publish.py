#!/usr/bin/env python3
"""Does a published baseline directory hold the grid the arm was scored on?

``probe_delinearize.py`` proved the inverse of ``linearize_moe`` is exact on floats that
were only moved. This is the next and last question in that chain, and it is a different
one: the publish path does not move floats, it *re-encodes* them. Between the model the
recipe quantized and the directory a reader downloads sit four steps that each have a way
of being wrong while producing something that loads --

    carried_grids     read the integer codes off the quantizer instead of re-fitting
    banked_grids      rearrange per-expert codes back into banks, by the same rule
    delinearize       rebuild the banked float state dict for a fresh model
    export ... encoder pack the carried codes rather than quantizing the weights again

-- and the failure they share is silence. A wrong expert order, a swapped gate and up, a
band renormalized on the way through: each writes a directory that opens, reports no
missing keys, and produces finite logits from the wrong weights.

So the claim this probe tests is the strongest one available and the only one worth
publishing under: **the directory dequantizes to the same numbers the in-process arm was
scored on.** Not close to them, and not merely on the same grid -- the same numbers, up to
the one rounding the format cannot avoid, which is measured here in code steps rather than
asserted in a docstring.

The refit control is the other half
-----------------------------------
Carrying the codes is only worth its complexity if re-deriving them is wrong. That is an
empirical claim, so it is measured: the same fresh model is exported a second time with no
encoder, letting the exporter's own min/max fit the dequantized weights, and the distance
from *that* to the reference is reported beside the carried one. On an RTN arm the two
should agree -- RTN is min/max, so a refit recovers it -- and on GPTQ they should not,
because error compensation leaves groups that no longer occupy both ends of their range and
a fresh min/max then quantizes a grid that is already quantized. Both arms run for exactly
that contrast; an outcome where GPTQ's refit is also clean would say the carrying machinery
is unnecessary, and that is a result this probe is able to report.

The control uses ``candidates=[1.0]`` -- pure min/max, no clipping search. A refit *with*
the search would land further away and flatter the carried path for a reason that has
nothing to do with carrying.

Run at bfloat16, the precision the panel arms run at, because the tolerance being checked
is a bf16 rounding of the offset ``-scale * zero`` and fp32 would not exercise it.

Usage::

    python experiments/phase4/probe_publish.py --out probe_publish.json

Needs ``llmcompressor`` and a transformers that knows ``lfm2_moe``. CPU-only and hermetic:
roughly 9 M parameters, no download, synthetic calibration.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/dynquant-core/src"))

from _llmc import build_recipe, materialize_quantization
from baselines_lfm2 import (
    MAX_CARRY_DRIFT,
    banked_grids,
    carried_grids,
    carrying_encoder,
    delinearize_state_dict,
    expert_rules,
    recipe_weights,
    resolve_awq_mappings,
    tie_report,
    under_the_input_table,
)
from probe_linearized_save import build_tiny

GROUP_SIZE = 32
"""The smallest grouping ``QuantTensor`` accepts, and the reason for the geometry below.

``GROUP_SIZE_ALIGNMENT`` is 32, so every contracted dimension in the probe model has to be
a multiple of it or the pack refuses before any of this is tested."""

GEOMETRY: dict[str, Any] = {"moe_intermediate_size": 96}
"""Rectangular *and* 32-aligned, which the builder's default geometry is not.

The default makes ``2 * moe_intermediate_size == hidden_size``, so a fused bank comes out
square and a transposed reading of it survives on shape. 96 gives ``gate_up_proj`` as
``[E, 192, 64]`` and ``down_proj`` as ``[E, 64, 96]`` -- distinguishable by shape, and both
contracted dimensions (64 and 96) are multiples of 32."""

CALIB_ROWS = 16
CALIB_LEN = 64
"""Enough for GPTQ's Hessian to be non-degenerate at these widths, and no more.

The Hessian is ``in_features``-square, so the widest module here needs more than 96 tokens
before it is full rank. 16 x 64 gives 1024, which is an order of magnitude past that, and
the point of the arm is that compensation *moved* the weights -- not that it converged."""


def placeholder_tokenizer(vocab_size: int) -> Any:
    """One token per id, so ``oneshot`` has the processor it insists on.

    It insists whether or not the dataset needs one: ``pre_process`` builds a processor
    from the model path the moment a dataset is passed, and the tiny source directory has
    none, so the calibrated arms died before the recipe ran. Nothing tokenizes anything
    here -- the ids are already ids -- so the cheapest honest processor is an identity
    over the model's own vocabulary, which also keeps a real tokenizer's 128k vocabulary
    from being attached to a 256-token model and read later as the thing it was calibrated
    with.
    """
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    vocabulary = {str(index): index for index in range(vocab_size)}
    inner = Tokenizer(models.WordLevel(vocabulary, unk_token="0"))
    return PreTrainedTokenizerFast(tokenizer_object=inner, unk_token="0", pad_token="0")


def synthetic_calibration(vocab_size: int, *, seed: int) -> Any:
    """Random token ids, pre-tokenized.

    GPTQ reads activations, and activations of random ids through random weights are as
    good a Hessian as activations of English: what is being tested is whether the codes
    survive publication, not whether the arm is accurate. Pre-tokenized because the source
    directory carries no tokenizer and a download would put the network on this probe's
    critical path.
    """
    import torch
    from datasets import Dataset

    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab_size, (CALIB_ROWS, CALIB_LEN), generator=generator)
    return Dataset.from_dict(
        {"input_ids": ids.tolist(), "attention_mask": torch.ones_like(ids).tolist()}
    )


def quantized_reference(
    source: Path, rules: list[dict[str, Any]], method: str, bits: int, dtype: str
) -> tuple[Any, Any, Any]:
    """Run a real recipe on the linearized tiny model; return it and the banked floats.

    The returned state dict *is* the reference: the weights the recipe materialized,
    rearranged into the banked names the architecture uses, at the dtype they were
    computed in. Everything downstream is compared against this and nothing else, so a
    published directory that matches it matches the thing that was scored.
    """
    import torch
    from llmcompressor import oneshot
    from llmcompressor.modeling.moe.linearize import get_non_linearized_moes, linearize_moe
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(source))
    if not hasattr(config, "hidden_act"):
        config.hidden_act = "silu"
    model = AutoModelForCausalLM.from_pretrained(
        str(source), config=config, dtype=getattr(torch, dtype), device_map="cpu"
    )
    banks_before = len(get_non_linearized_moes(model))
    linearize_moe(model)
    if get_non_linearized_moes(model):
        raise SystemExit("linearize_moe left banks behind; the recipe would miss the experts")

    kwargs: dict[str, Any] = {}
    if method != "rtn":
        dataset = synthetic_calibration(config.vocab_size, seed=0)
        kwargs = {
            "dataset": dataset,
            "tokenizer": placeholder_tokenizer(config.vocab_size),
            "num_calibration_samples": len(dataset),
            "max_seq_length": CALIB_LEN,
            "pipeline": "basic",
        }
    # The stock AWQ mappings look for an `input_layernorm` before q/k/v. LFM2 names it
    # `operator_norm`, so `_set_resolved_mappings` finds q, k and v, finds no norm to smooth
    # them against, and raises on the incomplete set. The driver already carries the
    # architecture's own mappings and a resolver that checks them against the tree before
    # the calibration pass; using anything else here would calibrate a different arm from
    # the one the panel scored.
    smoothing = None
    recipe_mappings = None
    if method == "awq":
        recipe_mappings, smoothing = resolve_awq_mappings(model)
    oneshot(
        model=model,
        recipe=build_recipe(method, bits, GROUP_SIZE, ignore=[], mappings=recipe_mappings),
        **kwargs,
    )
    applied = materialize_quantization(model)
    applied["banks_linearized"] = banks_before
    if smoothing is not None:
        applied["smoothing"] = smoothing
    # Not the model's own state dict: a quantized module also holds the scales it was
    # fitted with, and those hang off linearized expert names the bank rules do not
    # describe. This probe is how that was found -- the first run refused here.
    return model, delinearize_state_dict(model, rules, recipe_weights(model)), applied


def fresh_banked(source: Path, weights: Any, dtype: str) -> Any:
    """A banked model holding exactly the recipe's weights, loaded the way publish does.

    ``strict=True`` because the whole point of the inverse is that the key set is right;
    ``assign=False`` because assigning would break the embedding's tie with the head and
    the exporter, which detects tying by storage identity, would then write it twice.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(source))
    if not hasattr(config, "hidden_act"):
        config.hidden_act = "silu"
    model = AutoModelForCausalLM.from_pretrained(
        str(source), config=config, dtype=getattr(torch, dtype), device_map="cpu"
    )
    model.load_state_dict(weights, strict=True)
    model.tie_weights()
    return model


def published(
    model: Any, banked: dict[str, dict[str, Any]], width: int, out: Path, *, carry: bool
) -> dict[str, Any]:
    """Export ``model`` to ``out``, either carrying the grid or re-fitting it."""
    import dataclasses

    from dynquant.quant.checkpoint import export_packed_checkpoint

    drift: dict[str, float] = {}
    report = export_packed_checkpoint(
        model,
        dict.fromkeys(banked, width),
        output_dir=out,
        group_size=GROUP_SIZE,
        compute_device="cpu",
        # Pure min/max for the control. The clipping search is a better quantizer of a
        # dense weight and a worse recoverer of one already on a grid, so leaving it on
        # would let the control lose for a reason the comparison is not about.
        candidates=[1.0],
        encoder=carrying_encoder(banked, group_size=GROUP_SIZE, drift=drift) if carry else None,
    )
    written = dataclasses.asdict(report)
    written["output_dir"] = str(written["output_dir"])
    written.pop("layers", None)
    written["carry_drift_steps"] = round(max(drift.values()), 6) if drift else None
    written["bytes_on_disk"] = sum(p.stat().st_size for p in out.rglob("*.safetensors"))
    return written


def agreement(directory: Path, names: list[str], reference: Any) -> dict[str, Any]:
    """Load the directory back and compare every claimed module against the reference.

    Through ``from_pretrained`` and ``register_hf_quantizer``, not by reading safetensors:
    the claim being published is that a reader gets these weights, and a reader is the
    loader. Reading the files directly would prove the bytes are right and skip the two
    places -- key naming and bank reassembly -- where a correct file set still loads wrong.

    The distance is reported in code steps, because that is the unit the tolerance is in
    and the unit in which a rearrangement error is unmistakable: a wrong expert moves a
    weight by whatever the weights happen to differ by, which at these scales is many
    steps, while the format's own offset rounding is a fraction of one.
    """
    import torch
    from transformers import AutoModelForCausalLM

    import dynquant
    from dynquant.runtime.linear import DynQuantExpertBank

    dynquant.register_hf_quantizer()
    model = AutoModelForCausalLM.from_pretrained(str(directory), dtype="auto")

    kinds: dict[str, int] = {}
    worst_name, worst_steps, unmatched = None, 0.0, []
    for name in names:
        want = reference.get(name, reference.get(f"{name}.weight"))
        if want is None:
            unmatched.append(name)
            continue
        module = model.get_submodule(name)
        kinds[type(module).__name__] = kinds.get(type(module).__name__, 0) + 1
        held = getattr(module, "weight_qt", None)
        if held is None:
            got, step = module.weight.detach().float(), None
        else:
            got = held.dequantize(dtype=want.dtype).float()
            step = held.scales.float().abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
        delta = (got - want.detach().float()).abs().amax()
        steps = float(delta if step is None else delta / step)
        if steps > worst_steps:
            worst_name, worst_steps = name, steps

    banks = [n for n, m in model.named_modules() if isinstance(m, DynQuantExpertBank)]
    dense = [n for n, p in model.named_parameters() if p.ndim == 3 and "experts" in n]
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3, 4]])).logits
    return {
        "modules_compared": len(names) - len(unmatched),
        "by_class": dict(sorted(kinds.items())),
        "worst_steps": round(worst_steps, 6),
        "worst_module": worst_name,
        "unmatched": sorted(unmatched),
        "packed_banks": len(banks),
        "dense_expert_params": sorted(dense),
        "logits_finite": bool(torch.isfinite(logits).all()),
    }


def arm(
    source: Path, work: Path, rules: list[dict[str, Any]], method: str, bits: int, dtype: str
) -> dict[str, Any]:
    """One (method, width): quantize, carry, publish, reload, and the refit control."""
    model, reference, applied = quantized_reference(source, rules, method, bits, dtype)
    grids = carried_grids(model, group_size=GROUP_SIZE)
    banked, width = banked_grids(model, grids, rules)
    tie = tie_report(model)
    banked = under_the_input_table(model, banked)
    del model, grids

    # The head as well as the table it was renamed onto. After the swap the head is a
    # `DynQuantLinear` holding no tensors of its own, so comparing it asks whether the
    # rename left it reading the right buffers -- which is the thing the rename is for
    # and the one part of it a directory listing cannot show.
    names = sorted(set(banked) | ({tie["output"]} if tie["output"] else set()))
    result: dict[str, Any] = {
        "method": method,
        "bits": bits,
        "dtype": dtype,
        "applied": applied,
        "tie": tie,
        "modules_carried": len(banked),
        "modules_compared": len(names),
    }
    for label, carry in (("carried", True), ("refit", False)):
        out = work / f"{method}_{bits}b_{label}"
        # Rebuilt per label rather than shared: `export_packed_checkpoint` does not modify
        # the model, but a probe whose second answer depends on the first not having
        # mutated anything is asserting the thing it is measuring.
        fresh = fresh_banked(source, reference, dtype)
        result[label] = {
            **published(fresh, banked, width, out, carry=carry),
            "reload": agreement(out, names, reference),
        }
        del fresh
        shutil.rmtree(out)
    return result


def verdict(arms: list[dict[str, Any]]) -> str:
    """One line, derived, and able to say the bridge is not exact."""
    carried = [(a, a["carried"]["reload"]) for a in arms]
    broken = [
        f"{a['method']}-{a['bits']}b"
        for a, reload_ in carried
        if reload_["unmatched"] or not reload_["logits_finite"] or reload_["dense_expert_params"]
    ]
    if broken:
        return (
            f"{', '.join(broken)} did not come back as a packed model with every claimed "
            "module accounted for -- the container is wrong before the numbers matter"
        )
    over = [
        f"{a['method']}-{a['bits']}b at {reload_['worst_steps']:.4f} steps"
        for a, reload_ in carried
        if reload_["worst_steps"] > MAX_CARRY_DRIFT
    ]
    if over:
        return (
            f"published directories disagree with the weights the arm was scored on: "
            f"{'; '.join(over)}, past the {MAX_CARRY_DRIFT}-step budget the format's offset "
            "rounding accounts for -- this is a rearrangement error, not a rounding"
        )
    worst = max(reload_["worst_steps"] for _, reload_ in carried)
    gained = [
        f"{a['method']}-{a['bits']}b {a['refit']['reload']['worst_steps']:.3f} vs "
        f"{a['carried']['reload']['worst_steps']:.3f}"
        for a in arms
        if a["refit"]["reload"]["worst_steps"] > 10 * a["carried"]["reload"]["worst_steps"]
    ]
    tail = (
        f"; re-fitting instead would have moved them ({', '.join(gained)} steps)"
        if gained
        else "; no arm here distinguishes carrying from re-fitting, so the control did not "
        "reproduce the case the encoder exists for"
    )
    return (
        f"all {len(arms)} arms publish to a directory that loads packed and dequantizes to "
        f"the weights the in-process arm was scored on, worst {worst:.4f} code steps against "
        f"a {MAX_CARRY_DRIFT} budget{tail}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("probe_publish_work"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--methods", nargs="+", default=["rtn", "gptq"])
    parser.add_argument("--bits", nargs="+", type=int, default=[4, 3])
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)

    work = args.work
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    source = work / "source"
    build_tiny(source, **GEOMETRY)

    # Derived once. `expert_rules` builds and linearizes its own tiny model to measure
    # the layout, and the answer cannot differ between arms of one process.
    rules = expert_rules()

    arms = []
    for method in args.methods:
        for bits in args.bits:
            print(f"--- {method} {bits}b ---", flush=True)
            arms.append(arm(source, work, rules, method, bits, args.dtype))
            print(json.dumps(arms[-1], indent=2), flush=True)

    payload = {"geometry": GEOMETRY, "group_size": GROUP_SIZE, "arms": arms}
    payload["verdict"] = verdict(arms)

    text = json.dumps(payload, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    if not args.keep:
        shutil.rmtree(work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
