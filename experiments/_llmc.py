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


def default_symmetric(method: str) -> bool:
    """Whether ``method`` fits a symmetric grid when nothing asks it to do otherwise.

    GPTQ and RTN default to symmetric because ``QuantizationArgs()`` does -- verified against
    compressed_tensors 0.17.1, whose shipped ``W4A16`` preset is symmetric and whose
    ``W4A16_ASYM`` is the one that is not. AWQ's asymmetry is *this repository's* choice and
    not llm-compressor's: ``AWQModifier`` in 0.12 carries no scheme at all, only the
    activation-scaling transform, and the asymmetry here tracks AutoAWQ's ``zero_point=True``
    -- which is what a reader downloading a checkpoint named "AWQ" would get. The distinction
    is written down because a default inherited from a dependency and a default chosen here
    fail differently: the first moves when the dependency moves and takes every unpinned arm
    with it, and only the second is ours to defend.

    It lives here, at one definition, because three callers need it and each one that copied
    it would be a copy that stays green while the original changes: the recipe builder that
    resolves ``--symmetric auto``, the side file that records what the recipe resolved to,
    and `panel_table.scheme_of`, which has to say what an arm quantized *before* the flag
    existed was fitted on. That last one is a recovery, not a record, and it is labelled as
    one at the point of use.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {', '.join(METHODS)}")
    return method != "awq"


def stored_meta_bits(
    numel: int,
    in_features: int,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
    actorder: str | None = None,
) -> int:
    """Bits ``compressed-tensors`` stores *beside* one quantized weight, from its own rules.

    Read out of ``PackedQuantizationCompressor`` and ``initialize`` at 0.17.1 rather than
    reasoned about, because this is a dependency's arithmetic and nothing in this repository
    can contradict it:

    * ``weight_scale`` -- one per group, at the observed dtype, so 16 bits here.
    * ``weight_zero_point`` -- **only when asymmetric**. ``compression_param_names`` adds it
      to the state dict under ``if not weights.symmetric``, and ``_remove_symmetric_zp``
      strips it otherwise. It is packed by ``pack_to_int32(zp, num_bits)``, so it costs
      ``bits`` per group, not 16.
    * ``weight_g_idx`` -- only under ``actorder=group``: an ``int32`` per *input column*,
      which is per-tensor rather than per-group and is why this returns a total instead of a
      per-group rate.

    Why it is a function and not two constants
    ------------------------------------------
    Both baseline stages previously wrote ``meta_bits = 16 + bits`` and charged the zero
    point to every arm. The lfm2 copy defended it: charging only the asymmetric arm would
    supposedly make the two baselines "differ in the size column by a convention rather than
    by their weights". That inverts. The arms differ in the size column because they differ
    in what they store, which is the entire purpose of a size column; charging both the
    maximum does not remove a convention, it imposes one. GPTQ and RTN are symmetric by
    default and every arm this project has published under them was over-charged
    ``bits/group_size`` per weight -- 0.023 bits at 3-bit g128, 0.74% of the width -- in the
    direction that makes the baseline look more expensive than it is, which is the direction
    that flatters DynQuant.

    It was not found by review. It was found by the phase-2 asymmetric control refusing to
    print, because that control's premise is that the scheme costs something and the
    accounting said it cost nothing: two arms whose accuracies differed by 22 points
    accounted to the same 3.1522 bits. A size column blind to the very axis a control varies
    cannot measure that control.
    """
    if in_features % group_size:
        raise ValueError(
            f"a weight's contracted dimension is {in_features}, not a multiple of the group "
            f"size {group_size}. compressed-tensors ceil-divides and pads in that case "
            "(`strategy_cdiv`), and this accounting does not model the padding, so the "
            "number would come out low. Pick a group size that divides it, or extend this."
        )
    groups = numel // group_size
    total = groups * 16
    if not symmetric:
        total += groups * bits
    if actorder == "group":
        total += in_features * 32
    return total


def quant_args(bits: int, group_size: int, *, symmetric: bool, actorder: str | None = None) -> Any:
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
        actorder=actorder,
    )


def build_recipe(
    method: str,
    bits: int,
    group_size: int,
    *,
    ignore: list[str],
    mappings: Any = None,
    symmetric: bool | None = None,
    actorder: str | None = None,
) -> Any:
    """One modifier list per method, identical in every respect except the method.

    ``mappings`` is AWQ's only architecture-dependent input: which module produces the
    input of which other module, so a scale taken out of one can be folded into the other.
    ``None`` means llm-compressor's own table, which is right for every architecture in it.
    It is a parameter and not a constant here for the same reason ``ignore`` is -- this
    file is the one that knows nothing about the model.

    ``symmetric`` and ``actorder`` override the method's own default, and they exist
    for one purpose: running the control the defaults make impossible. A method's
    default is what a reader would download under that name, which is why it is the
    default here -- but a comparison between a symmetric arm and an asymmetric one
    spans the scheme as well as the allocation, and only an arm matching its
    opponent's scheme can say which of the two the delta belongs to. The size column
    does not move when they are used: the byte accounting charges a zero point per
    group for every arm regardless of symmetry.
    """
    # Activation ordering reorders columns by a Hessian diagonal, so it means something
    # only where a Hessian is estimated. Refused rather than ignored on the other two:
    # a flag that is silently dropped is a control a caller believes it ran.
    if actorder is not None and method != "gptq":
        raise SystemExit(
            f"actorder={actorder!r} was asked for on {method!r}, which estimates no "
            "Hessian and has no activation order to sort by"
        )

    # Before `default_symmetric`, which raises its own ValueError on an unknown method
    # and would otherwise pre-empt the SystemExit this function ends with.
    if method not in METHODS:
        raise SystemExit(f"unknown method {method!r}; choose from {', '.join(METHODS)}")

    from compressed_tensors.quantization import QuantizationScheme
    from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    # Each method's own published default unless the caller names one; forcing one
    # convention on all three would make the arms differ from the checkpoints a reader
    # would download under those names. Through `default_symmetric` rather than spelled
    # out, because the side file records the same decision a few frames up the stack and
    # two copies of it can disagree about what an arm was -- one deciding what runs and
    # one deciding what is written down is the worst possible pair to let drift.
    sym = default_symmetric(method) if symmetric is None else symmetric
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=quant_args(bits, group_size, symmetric=sym, actorder=actorder),
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
    raise SystemExit(  # pragma: no cover - unreachable until METHODS grows
        f"{method!r} is in METHODS but has no recipe here; a method was added to the "
        "registry without a modifier list to run it"
    )


def materialize_quantization(
    model: Any, *, method: str | None = None, probes: int = 8
) -> dict[str, Any]:
    """Write the frozen scales into the weights, and prove they landed.

    See the module docstring for why. Applied uniformly rather than per method -- it is
    idempotent on weights already on the grid, so GPTQ passes through unchanged, and a
    method-specific branch here would be one more thing to get wrong the next time
    llm-compressor moves the boundary between calibration and compression.

    The returned counts are recorded with the arm. ``weights_moved`` is expected to be zero
    for GPTQ and equal to ``materialized_modules`` for RTN and AWQ; an arm whose counts do
    not match that shape is reporting something other than what its label says. Pass
    ``method`` and that sentence is enforced instead of merely written down -- it was
    written down, and an arm violating it still ran for eleven hours and scored 0.00%.

    ``g_idx`` is the reason it has to be enforced. GPTQ with activation ordering sorts each
    row's columns by a Hessian diagonal, so column *i* belongs to group ``g_idx[i]`` and not
    to ``i // group_size``. ``fake_quantize`` takes that mapping and defaults it to ``None``,
    which silently means the contiguous one -- so omitting it does not fail, it applies every
    group's scale to the wrong columns and scrambles the weights. The fixed-point probe below
    cannot see this: it re-quantizes under the same wrong mapping and finds it idempotent.
    What does see it is ``weights_moved``, which goes from GPTQ's expected 0 to every module.

    Which is why ``weights_moved`` counts against one ULP of the storage dtype rather than
    against zero. Forwarding ``g_idx`` is exactly right -- on a float32 weight the
    round-trip is bit-identical -- but a bfloat16 checkpoint is not the float32 value GPTQ
    computed, it is that value cast down, and requantizing the cast lands up to a ULP away.
    Activation ordering is what exposes it, because permuting the columns changes which
    values share a group and so which sit near a rounding boundary. Bit-exactness would
    therefore have reported a correct act-ordered arm as 225 of 225 scrambled -- and did.
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

    moved, max_delta, max_ulps = 0, 0.0, 0.0
    for _, module in targets:
        weights = module.quantization_scheme.weights
        with align_module_device(module):
            original = module.weight.data
            rounded = fake_quantize(
                original,
                module.weight_scale,
                getattr(module, "weight_zero_point", None),
                weights,
                g_idx=getattr(module, "weight_g_idx", None),
            )
            delta = (rounded.float() - original.float()).abs().max().item()
            # One ULP of the *storage* dtype at this tensor's own scale, not zero. GPTQ
            # dequantizes in float32 and casts the result down to the checkpoint dtype;
            # requantizing that cast value in bfloat16 lands up to a ULP away from it. The
            # difference only shows up when activation ordering is on, because the
            # permutation changes which columns share a group and therefore which values
            # sit near a rounding boundary -- so a bit-exact test reads a correct
            # act-ordered checkpoint as scrambled. Measured on a 512x64 linear against the
            # real GPTQ path: the true mapping sits at 0.62 ULP and is exactly 0 when the
            # weight is float32, while omitting g_idx or shuffling it sits at 35 ULP. The
            # threshold separates those by a factor of 57 and still catches a wrong grid,
            # whose weights move by half a quantization step -- thousands of ULPs.
            ulp = torch.finfo(original.dtype).eps * original.abs().max().float().item()
        if delta > ulp:
            moved += 1
        max_delta = max(max_delta, delta)
        max_ulps = max(max_ulps, delta / ulp if ulp else 0.0)
        update_offload_parameter(module, "weight", rounded.to(module.weight.dtype))

    # Each method's own shape, refused rather than recorded. GPTQ rounds as it calibrates,
    # so materialization is a no-op on it and anything else means the scales are being
    # applied to columns they were not fitted on; RTN and AWQ leave every weight unrounded,
    # so a zero there means the recipe never reached the model. Checked before the probe
    # because the probe agrees with a wrong mapping and this does not.
    expected = {"gptq": 0, "awq": len(targets), "rtn": len(targets)}.get(method or "")
    if expected is not None and moved != expected:
        raise SystemExit(
            f"{method} moved {moved} of {len(targets)} weights, expected {expected}. "
            + (
                "GPTQ leaves its weights on the grid, so a non-zero count means "
                "materialization re-rounded them against a different grouping than the one "
                "they were fitted under -- check that every recipe artifact the scheme "
                "carries (weight_g_idx, above all) is being passed to fake_quantize"
                if expected == 0
                else "a count short of every module means the recipe did not reach them, "
                "and the arm would be scored on unquantized weights"
            )
        )

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
                g_idx=getattr(module, "weight_g_idx", None),
            )
            drift = (again.float() - stored.float()).abs().max().item()
            if drift > torch.finfo(stored.dtype).eps * stored.abs().max().float().item():
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
        "max_weight_ulps": round(max_ulps, 3),
        "probe_unique_values_per_row": unique_per_row,
    }
    print(json.dumps(stats), flush=True)
    return stats
