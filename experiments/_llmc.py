"""What is true about ``llm-compressor`` regardless of which experiment is calling it.

Three facts live here, and none of them is about a task, a model or a run directory:

* how a weight-quantization scheme is spelled at an arbitrary bit width,
* which modifiers make up a GPTQ / AWQ / RTN recipe,
* and that ``oneshot`` does not round the weights, so an in-process arm has to.

That third one is why this module exists as a module. ``oneshot`` separates *calibration*
from *compression*: it fits scales and zero points, attaches them, sets
``quantization_status=FROZEN``, and for ``QuantizationModifier`` stops there. Rounding
happens later, inside ``save_pretrained(save_compressed=True)``. An arm that quantizes and
scores in the same process -- which is the only way to score a 3-bit arm without writing a
dequantized bf16 checkpoint four times the size -- therefore scores bf16 weights with an
unused set of scales bolted on.

That was found once, by RTN 4-bit returning byte-identical predictions to bf16 on all 3080
problems. The dangerous case is AWQ: its transform *does* rewrite weights, so an unrounded
AWQ arm produces plausibly-different numbers rather than suspiciously identical ones, and
enters a table as an unbeatable baseline that was never quantized. Two experiments now need
that guard, and a copy of it in each is how the finding gets un-found -- the copy that is
not being looked at is the one that drifts when llm-compressor next moves the boundary.

``ignore`` is a parameter, never a default. What stays in fp16 is the single largest
difference between a baseline's nominal width and its measured one, and on a model that
ties ``lm_head`` to ``embed_tokens`` the conventional ``["lm_head"]`` pins a quarter of the
checkpoint at full precision. That is the caller's decision about what claim it is making,
and it has to be made where the claim is.
"""

from __future__ import annotations

import json
from typing import Any

METHODS = ("gptq", "awq", "rtn")


def quant_args(bits: int, group_size: int, *, symmetric: bool) -> Any:
    """The weight-quantization contract shared by every recipe below.

    Built explicitly instead of naming the ``W4A16`` preset because the preset only exists
    at 4 and 8 bits, and a 3-bit arm has to be expressible in the same terms as its 4-bit
    sibling -- configured through a different code path it would not be comparable to it,
    let alone to DynQuant's.
    """
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    return QuantizationArgs(
        num_bits=bits,
        type=QuantizationType.INT,
        symmetric=symmetric,
        strategy=QuantizationStrategy.GROUP,
        group_size=group_size,
    )


def build_recipe(
    method: str,
    bits: int,
    group_size: int,
    *,
    ignore: list[str],
    mappings: Any = None,
) -> Any:
    """One modifier list per method, identical in every respect except the method.

    ``mappings`` is AWQ's only architecture-dependent input: which module produces the
    input of which other module, so a scale taken out of one can be folded into the other.
    ``None`` means llm-compressor's own table, which is right for every architecture in it.
    It is a parameter and not a constant here for the same reason ``ignore`` is -- this
    file is the one that knows nothing about the model.
    """
    from compressed_tensors.quantization import QuantizationScheme
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    # Asymmetric for AWQ, symmetric otherwise -- each method's own published default.
    # Forcing one convention on all three would make the arms differ from the checkpoints
    # a reader would download under those names.
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=quant_args(bits, group_size, symmetric=(method != "awq")),
    )
    groups = {"group_0": scheme}

    if method == "gptq":
        # dampening_frac is llm-compressor's default; named here only so a 3-bit run
        # cannot silently pick a different one if the default moves.
        return [GPTQModifier(config_groups=groups, ignore=ignore, dampening_frac=0.01)]
    if method == "awq":
        # Two modifiers, not one: as of 0.12 AWQModifier applies only the activation-aware
        # scaling transform, and the quantization itself is a separate step. The single
        # combined AWQModifier still importable from llmcompressor.modifiers.awq is a
        # deprecation shim.
        awq = AWQModifier(mappings=mappings) if mappings is not None else AWQModifier()
        return [awq, QuantizationModifier(config_groups=groups, ignore=ignore)]
    if method == "rtn":
        # Round-to-nearest: the same grouping and the same ignore list, with no
        # calibration-driven correction at all. It is the floor the other two have to beat
        # to have earned their calibration pass.
        return [QuantizationModifier(config_groups=groups, ignore=ignore)]
    raise SystemExit(f"unknown method {method!r}; choose from {', '.join(METHODS)}")


def materialize_quantization(model: Any, *, probes: int = 8) -> dict[str, Any]:
    """Write the frozen scales into the weights, and prove they landed.

    See the module docstring for why. Applied uniformly rather than per method -- it is
    idempotent on weights already on the grid, so GPTQ passes through unchanged, and a
    method-specific branch here would be one more thing to get wrong the next time
    llm-compressor moves the boundary between calibration and compression.

    The returned counts are recorded with the arm. ``weights_moved`` is expected to be zero
    for GPTQ and equal to ``materialized_modules`` for RTN and AWQ; an arm whose counts do
    not match that shape is reporting something other than what its label says.
    """
    import torch
    from compressed_tensors.quantization.lifecycle.forward import fake_quantize
    from compressed_tensors.utils import align_module_device, update_offload_parameter

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if getattr(getattr(module, "quantization_scheme", None), "weights", None) is not None
        and hasattr(module, "weight_scale")
    ]
    if not targets:
        raise SystemExit(
            "no module carries a weight quantization scheme -- the recipe did not apply, "
            "and scoring this model would measure the unquantized checkpoint"
        )

    moved, max_delta = 0, 0.0
    for _, module in targets:
        weights = module.quantization_scheme.weights
        with align_module_device(module):
            original = module.weight.data
            rounded = fake_quantize(
                original, module.weight_scale, getattr(module, "weight_zero_point", None), weights
            )
            delta = (rounded.float() - original.float()).abs().max().item()
        if delta > 0.0:
            moved += 1
            max_delta = max(max_delta, delta)
        update_offload_parameter(module, "weight", rounded.to(module.weight.dtype))

    # Fixed-point check on a spread of modules, re-reading the parameter rather than the
    # value just computed. Quantizing a quantized weight must be a no-op; a tensor that was
    # never rounded fails this, and so does one whose update silently did not stick --
    # assigning ``module.weight.data`` does not survive offloading, which is why the write
    # above goes through ``update_offload_parameter``.
    step = max(1, len(targets) // probes)
    off_grid, unique_per_row = [], []
    for name, module in targets[::step][:probes]:
        with align_module_device(module):
            stored = module.weight.data
            again = fake_quantize(
                stored,
                module.weight_scale,
                getattr(module, "weight_zero_point", None),
                module.quantization_scheme.weights,
            )
            if not torch.equal(again, stored):
                off_grid.append(name)
            unique_per_row.append(int(torch.unique(stored[0].float()).numel()))
    if off_grid:
        raise SystemExit(
            f"{len(off_grid)} of {len(targets[::step][:probes])} probed weights are not on "
            f"their own quantization grid after materialization ({', '.join(off_grid[:3])}); "
            "the model would be scored unquantized"
        )

    stats = {
        "materialized_modules": len(targets),
        "weights_moved": moved,
        "max_weight_delta": round(max_delta, 6),
        "probe_unique_values_per_row": unique_per_row,
    }
    print(json.dumps(stats), flush=True)
    return stats
