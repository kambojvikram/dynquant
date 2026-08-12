#!/usr/bin/env python3
"""GPTQ, AWQ and RTN on LFM2.5-8B-A1B -- quantizing all of it rather than 8.5% of it.

Pointed at this model unmodified, these recipes succeed and are wrong. LFM2.5-8B-A1B keeps
7.751 B of its 8.468 B parameters inside 22 batched expert banks -- one module per layer
holding ``gate_up_proj [32, 3584, 2048]`` and ``down_proj [32, 2048, 1792]`` as 3-D
tensors, not a ``ModuleList`` of ``nn.Linear``. ``targets=["Linear"]`` reaches 716 M of it:

    reachable as nn.Linear      716,570,624    8.5%
    held in expert banks      7,751,073,792   91.5%

So a run labelled "GPTQ 4-bit" would quantize 8.5% of the weights, leave 91.5% in bf16, and
emit a directory weighing ~15.5 GB against bf16's 16.9 GB. Nothing raises. The number in
the size column would be a *setting* the arm never honoured, and the error runs in
DynQuant's favour, which is the direction that does not get caught by disbelief.

This driver refuses to run without linearizing first, and it accounts the expert mass
explicitly rather than inferring it from a module walk -- see :func:`accounted_bytes`.

Linearization is llm-compressor's, and it is bit-exact
-----------------------------------------------------
``llmcompressor.modeling.moe.linearize_moe`` already does this, and ``oneshot`` already
calls it. The one blocker is that ``lfm2_moe``'s config has no ``hidden_act`` and
``MoEConfig.from_config`` requires one; ``silu`` is not a guess, it is what
``Lfm2MoeExperts.__init__`` hard-codes as ``self.act_fn``.

The swap had to be checked rather than argued, because the baselines are quantized through a
module layout the DynQuant arms never use: a changed reduction order would put a second
variable into every paired test against the shared bf16 ceiling. The reference forward
already loops per expert and calls ``nn.functional.linear(state, self.gate_up_proj[i])``, and
indexing the first dim of a contiguous tensor yields contiguous memory, so substituting a
``Linear``'s weight is the same op on the same layout. Measured on a full-size bank with all
32 experts routed, float32 and bfloat16 both came back bit-identical under ``torch.equal``
(max|delta| 0.0). ``allclose`` was not used; it would have hidden exactly the change being
looked for.

``ignore`` is empty, and on this model that is not a variant
-----------------------------------------------------------
``lfm2_moe`` sets ``tie_word_embeddings: true``. The conventional ``ignore=["lm_head"]``
therefore does not leave *a* tensor in fp16, it pins the shared embedding tensor -- and
this campaign has already measured what that does: on Qwen3.5-2B a "4-bit g128" checkpoint
accounted to 7.3605 bits, with 59% of the total bits sitting in a tensor nobody quantized.
DynQuant quantizes the embedding. So does every arm here. The default GPTQ/AWQ convention is
a real result about the convention and it is reported in §10 of the phase-4 report, but it is
not a budget these arms may be compared at.

Two properties of MoE that survive the fix, and belong in any table this feeds
-----------------------------------------------------------------------------
Routing is 4-of-32, so each expert sees roughly an eighth of the calibration tokens. GPTQ's
Hessian and AWQ's scales get an eighth of the per-expert statistics they would get on a dense
model of the same size. That handicap is theirs, not this script's -- but a DynQuant win
partly attributable to calibration sparsity must not be reported as a method advantage.
DynQuant is not exposed to it because its signal comes from a fine-tune-time hook that sees
every routed token, which is the premise of the method and also strictly more information
than either baseline gets.

Scored through ``dynquant eval``, not beside it
----------------------------------------------
:func:`score` builds a real ``dynquant eval`` namespace from
:func:`dynquant.cli.build_parser` and hands the quantized model to
``commands.evaluate.run``. Nothing about the prompt, the shots, the decode budget, the
scorer or the record schema is restated here, so these arms pair with the bf16 ceiling and
with the DynQuant arms under the same ``PAIRING_FIELDS`` guard a McNemar test checks.

Usage::

    # what the recipe can see, and what it would cost -- no GPU, no weights
    python experiments/phase4/baselines_lfm2.py plan --model /workspace/models/LFM2.5-8B-A1B

    # quantize and score in one process, writing the arm's record
    python experiments/phase4/baselines_lfm2.py run --method gptq --bits 4 \
        --model /workspace/runs/lfm25-t2s/finetuned --label gptq-4b \
        --out experiments/phase4/out/gptq_4b.json --max-new-tokens 512
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages/dynquant-core/src"))

from _llmc import METHODS, build_recipe, materialize_quantization

IGNORE: list[str] = []
"""Nothing is left in fp16. See the module docstring: on a tied model there is no such thing
as leaving only the LM head alone."""

EXPERT_ACT = "silu"
"""``Lfm2MoeExperts.__init__`` hard-codes ``self.act_fn = F.silu``. Written onto the config
because ``MoEConfig.from_config`` requires one of ``hidden_act`` / ``hidden_activation`` /
``mlp_hidden_act`` and ``lfm2_moe`` ships none of them -- so this is transcribing the model's
own activation into the field llm-compressor reads it from, not choosing one."""


def visibility(source: str) -> dict[str, Any]:
    """How much of this architecture ``targets=["Linear"]`` can reach, before linearizing.

    Measured on ``meta``: it needs the module tree and no weights, so it costs no GPU and no
    download, and it is the number that decides whether this driver is necessary at all. A
    model with no banks reports ``banked == 0`` and linearization is a no-op for it.
    """
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModelForCausalLM

    from dynquant.graph.experts import batched_expert_params

    config = AutoConfig.from_pretrained(source)
    with torch.device("meta"):
        ref = AutoModelForCausalLM.from_config(config)

    linear = 0
    for name, module in ref.named_modules():
        if isinstance(module, nn.Linear) and not any(name.endswith(p) for p in IGNORE):
            linear += module.weight.numel()
    banked = sum(
        param.numel() for module in ref.modules() for _, param in batched_expert_params(module)
    )
    total = sum(p.numel() for p in _unique_params(ref))
    return {
        "params": total,
        "reachable_as_linear": linear,
        "held_in_expert_banks": banked,
        "linear_share": round(linear / total, 4),
        "banked_share": round(banked / total, 4),
    }


def _unique_params(model: Any) -> list[Any]:
    """Parameters deduplicated by identity.

    ``model.parameters()`` already dedups, but only across distinct attributes; a tied
    embedding and LM head are one object and must be counted once. This model ties them, and
    a size table that double-counted 0.5 B parameters would be worse than no table.
    """
    seen: dict[int, Any] = {}
    for param in model.parameters():
        seen.setdefault(id(param), param)
    return list(seen.values())


def accounted_bytes(source: str, bits: int, group_size: int) -> dict[str, Any]:
    """What this checkpoint costs, counted the way the DynQuant arms are counted.

    On-disk size is not available for every arm: ``compressed-tensors`` packs 4 and 8 bits,
    so a 3-bit result is held as dequantized bf16 and a file listing would report 16 bits per
    weight for a checkpoint whose arithmetic is 3-bit. Both widths are therefore accounted
    from the format's own rules, so the 3-bit and 4-bit arms are comparable to each other
    rather than one measured and one computed.

    Counted against a **meta-device copy of the source architecture**, never against the
    model llm-compressor returns: that model's Linears carry observer state and
    scale/zero-point parameters and are no longer plain ``nn.Linear``, and walking one
    reported 14.34 bits per weight for a 3-bit checkpoint. The architecture is what the size
    question is about, it is free to instantiate, and the quantizer cannot perturb it.

    The expert banks are added explicitly rather than found by the ``nn.Linear`` walk, because
    the walk runs *before* linearization -- and a walk that ran after it would need real
    tensors on a GPU to produce a number this function must be able to give for free. Each
    bank param ``[E, out, in]`` becomes ``E`` Linears of ``[out, in]`` grouped along ``in``,
    so the group count is ``numel // group_size`` either way, provided ``in`` divides the
    group size. That is asserted rather than assumed: hidden 2048 and
    ``moe_intermediate_size`` 1792 are both multiples of 128 on this model, and an
    architecture where they are not would silently mis-account here.
    """
    import torch
    import torch.nn as nn
    from transformers import AutoConfig, AutoModelForCausalLM

    from dynquant.graph.experts import batched_expert_params

    config = AutoConfig.from_pretrained(source)
    with torch.device("meta"):
        ref = AutoModelForCausalLM.from_config(config)

    # An fp16 scale per group, plus a zero point at the weight's own width when asymmetric.
    # The zero point is counted for every arm: AWQ is asymmetric, and charging only the
    # asymmetric arm for it would make the two baselines differ in the size column by a
    # convention rather than by their weights.
    meta_bits = 16 + bits
    quantized = 0
    counted: set[int] = set()

    def charge(param: Any, tail: int) -> int:
        if tail % group_size:
            raise SystemExit(
                f"a weight's contracted dimension is {tail}, which is not a multiple of the "
                f"group size {group_size}. Group-wise quantization pads in that case and the "
                "accounting here does not model the padding, so the number would be low. "
                "Pick a group size that divides it, or extend this function."
            )
        counted.add(id(param))
        return param.numel() * bits + (param.numel() // group_size) * meta_bits

    for name, module in ref.named_modules():
        if isinstance(module, nn.Linear) and not any(name.endswith(p) for p in IGNORE):
            quantized += charge(module.weight, module.weight.shape[-1])

    banked_params = 0
    for module in ref.modules():
        for _, param in batched_expert_params(module):
            banked_params += param.numel()
            quantized += charge(param, param.shape[-1])

    unique = _unique_params(ref)
    fp16 = sum(p.numel() * 16 for p in unique if id(p) not in counted)
    total = sum(p.numel() for p in unique)
    quantized_params = sum(p.numel() for p in unique if id(p) in counted)
    return {
        "accounted_bits": round((quantized + fp16) / total, 4),
        "accounted_bytes": (quantized + fp16) // 8,
        "accounted_gib": round((quantized + fp16) / 8 / 2**30, 4),
        "fp16_bits_share": round(fp16 / (quantized + fp16), 4),
        "quantized_params": quantized_params,
        "quantized_share": round(quantized_params / total, 4),
        "banked_params_quantized": banked_params,
        "params": total,
    }


def load_linearized(
    source: str, *, dtype: str = "bfloat16", device: str = "cuda"
) -> tuple[Any, dict[str, Any]]:
    """Load the model with its expert banks turned into ``nn.Linear``, and say so in numbers.

    Loaded here rather than letting ``oneshot`` load it from a path, for one reason: the
    ``hidden_act`` field has to be on the config object before ``linearize_moe`` reads it, and
    writing it into the checkpoint's ``config.json`` would mutate a directory three other arms
    load from.

    The report is the guard. ``banks_before`` comes from llm-compressor's own detector, and
    ``banks_after`` must be zero -- a run where linearization silently did nothing is exactly
    the 8.5% run this driver exists to prevent, and it does not fail on its own.

    ``device`` exists so that guard can be run without a GPU. The conversion is module surgery
    over weights it does not read, so ``cpu`` answers the same question at the cost of host RAM
    -- which means the check does not have to queue behind whatever is holding the GPU, and a
    box with no GPU at all can still tell you the recipe would have seen 8.5%.
    """
    import torch
    from llmcompressor.modeling.moe.linearize import get_non_linearized_moes, linearize_moe
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(source)
    if not hasattr(config, "hidden_act"):
        config.hidden_act = EXPERT_ACT

    model = AutoModelForCausalLM.from_pretrained(
        source, config=config, dtype=getattr(torch, dtype), device_map=device
    )
    before = len(get_non_linearized_moes(model))
    if before:
        linearize_moe(model)
    after = len(get_non_linearized_moes(model))
    if after:
        raise SystemExit(
            f"{after} of {before} expert banks are still batched after linearize_moe. The "
            "recipe would reach only the projections outside them -- on LFM2.5-8B-A1B that "
            "is 8.5% of the weights, and the run would succeed anyway"
        )

    # Modules and parameters, because only the second one answers the question. A module
    # count says conversion happened; the share of the checkpoint now reachable as
    # ``nn.Linear`` is what the recipe will actually quantize, and it is the number the 8.5%
    # measurement has to be compared against. ``visibility`` computes the same quantity on
    # the unconverted tree, so the two are the before and after of one claim.
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    linear_params = sum(m.weight.numel() for m in linears)
    total = sum(p.numel() for p in model.parameters())
    report = {
        "banks_before": before,
        "banks_after": after,
        "linear_modules": len(linears),
        "linear_params": linear_params,
        "params": total,
        "linear_share": round(linear_params / total, 4),
        "device": device,
    }
    print(json.dumps(report), flush=True)
    return model, report


def _layer_alternation(indices: list[int]) -> str:
    """Layer indices as a regex alternation, widest first.

    ``re`` backtracks, so ``(2|21)`` matches ``layers.21.`` either way. Ordering by width
    means the expression is right by construction rather than by that.
    """
    return "|".join(str(i) for i in sorted(indices, reverse=True))


def upstream_mappings(model: Any) -> list[Any] | None:
    """llm-compressor's own smoothing pairs for this architecture, or ``None``.

    The mappings come from ``get_layer_mappings_from_model`` because that is the function
    ``AWQModifier`` calls when it is handed none, and the point of checking is to check
    what will run. It consults a dynamic registry first -- hybrid stacks get theirs built
    against the module tree -- then the static one, then falls back to the Llama defaults
    for anything it does not know.

    Whether this architecture is *known* is a separate question, and the tempting way to
    answer it is wrong. ``get_layer_mappings_from_model`` returns ``default_mappings`` for
    an unknown model, so "did it hand back the defaults?" looks like the test -- but eight
    entries in the static registry, ``MistralForCausalLM`` and ``LlamaForCausalLM`` among
    them, *are* that same list object. The Llama shape is the common case and the table
    stores one list for all of it. Identity would therefore report the architectures
    upstream supports best as unknown, and send them to a hand-written table that describes
    a different model. So the static registry is asked by name, and the dynamic one is
    asked by whether it actually produced something -- its builders return ``None`` when
    they decline, which falls through to exactly the same default.
    """
    from llmcompressor.modifiers.transform.awq import (
        AWQ_MAPPING_REGISTRY,
        default_mappings,
        get_layer_mappings_from_model,
    )

    mappings = get_layer_mappings_from_model(model)
    known = model.__class__.__name__ in AWQ_MAPPING_REGISTRY or mappings is not default_mappings
    return list(mappings) if known else None


def awq_mappings(config: Any) -> list[tuple[Any, int]]:
    """LFM2's activation-aware smoothing pairs, each with the number of sets it must resolve.

    AWQ divides a linear's input channels by a per-channel scale and folds the inverse into
    whatever produced that input. Which module that is comes from a per-architecture table,
    and ``Lfm2MoeForCausalLM`` is in neither of llm-compressor's two: not the static
    ``AWQ_MAPPING_REGISTRY``, and not the dynamic one, whose hybrid-stack builder is exactly
    the right shape but requires ``linear_attention`` in ``layer_types`` while this model
    says ``conv``. So the Llama defaults apply, and on this architecture they are wrong in
    both halves of every block: the pre-mixer norm is ``operator_norm``, not
    ``input_layernorm``, and the pre-FF norm is ``ffn_norm``, not
    ``post_attention_layernorm``. ``q/k/v_proj`` match, their smooth partner never does, and
    ``match_modules_set`` raises on the incomplete set -- after the calibration pass, which
    on the real model is 256 sequences through 8 B parameters.

    The count beside each mapping is the other half of the fix, and the half that is easy to
    leave out. A mapping that matches *nothing* does not raise: ``_set_resolved_mappings``
    logs it and moves on, so the arm runs to completion as round-to-nearest wearing an AWQ
    label and enters the table as a baseline that was never smoothed. That is the same
    failure ``materialize_quantization`` exists to prevent, in a different disguise, and it
    wants the same treatment -- predict the number from the config, check it against the
    tree, and fail before the calibration pass rather than after it.
    """
    from llmcompressor.modifiers.transform.awq import AWQMapping

    try:
        types = list(config.layer_types)
        experts = int(config.num_experts)
        dense = int(config.num_dense_layers)
        heads = int(config.num_attention_heads)
        kv_heads = int(config.num_key_value_heads)
    except AttributeError as exc:
        raise SystemExit(
            f"{type(config).__name__} does not describe an LFM2 MoE stack ({exc}). These "
            "mappings name operator_norm, ffn_norm and conv.in_proj; against another "
            "architecture they would match nothing and smooth nothing"
        ) from None

    attention = [i for i, kind in enumerate(types) if kind == "full_attention"]
    convolution = [i for i, kind in enumerate(types) if kind != "full_attention"]
    # The first ``num_dense_layers`` blocks hold an Lfm2MoeMLP -- w1/w3/w2, this model's
    # spelling of gate/up/down -- and every block after them holds a router and experts.
    dense_ff = list(range(dense))
    moe_ff = list(range(dense, len(types)))

    pairs: list[tuple[Any, int]] = []
    if attention:
        pairs += [
            (
                AWQMapping(
                    rf"re:.*layers\.({_layer_alternation(attention)})\.operator_norm$",
                    [
                        "re:.*self_attn.q_proj$",
                        "re:.*self_attn.k_proj$",
                        "re:.*self_attn.v_proj$",
                    ],
                ),
                len(attention),
            ),
        ]
    if attention and kv_heads == heads:
        # ``out_proj`` is this model's ``o_proj``, and this pair is conditional for the
        # reason upstream AWQ made it conditional: with grouped-query attention ``v_proj``
        # emits ``kv_heads * head_dim`` rows and ``out_proj`` consumes ``heads * head_dim``,
        # so there is no per-channel scale that divides one and multiplies the other.
        # llm-compressor drops the pair itself -- but its check reads
        # ``balance_name.endswith(".o_proj")``, which is never true here, so on this model
        # the guard is inert and ``_smooth`` reaches ``weight[-scales.size(0):]`` with 256
        # scales for 128 rows. The condition belongs in the mapping rather than in a
        # try/except, because ``kv_heads != heads`` is knowable from the config and the
        # consequence -- ``out_proj`` quantized unsmoothed -- is worth stating rather than
        # rescuing.
        pairs.append(
            (
                AWQMapping("re:.*self_attn.v_proj$", ["re:.*self_attn.out_proj$"]),
                len(attention),
            )
        )
    if convolution:
        # ``conv.in_proj`` only. The short convolution and the ``conv.out_proj`` that reads
        # it have no linear producer to fold an inverse scale into, so they are quantized
        # unsmoothed -- counted by the resolver below rather than left to be assumed.
        pairs.append(
            (
                AWQMapping(
                    rf"re:.*layers\.({_layer_alternation(convolution)})\.operator_norm$",
                    ["re:.*conv.in_proj$"],
                ),
                len(convolution),
            )
        )
    if dense_ff:
        pairs += [
            (
                AWQMapping(
                    rf"re:.*layers\.({_layer_alternation(dense_ff)})\.ffn_norm$",
                    ["re:.*feed_forward.w1$", "re:.*feed_forward.w3$"],
                ),
                len(dense_ff),
            ),
            (AWQMapping("re:.*feed_forward.w3$", ["re:.*feed_forward.w2$"]), len(dense_ff)),
        ]
    if moe_ff:
        pairs += [
            (
                AWQMapping(
                    rf"re:.*layers\.({_layer_alternation(moe_ff)})\.ffn_norm$",
                    [
                        # The router reads the same normalized hidden state its experts do,
                        # so it balances with them. Left out, it would be the one Linear in
                        # the block whose input no longer matches the weights that read it,
                        # and it is the module that decides which experts run at all.
                        "re:.*feed_forward.gate$",
                        "re:.*experts.*.gate_proj$",
                        "re:.*experts.*.up_proj$",
                    ],
                ),
                len(moe_ff),
            ),
            # One set per expert rather than per layer: ``match_modules_set`` yields when
            # the lowest common ancestor of the matched set changes, and for a pair that
            # sits inside one expert, that is the expert.
            (
                AWQMapping("re:.*experts.*.up_proj$", ["re:.*experts.*.down_proj$"]),
                len(moe_ff) * experts,
            ),
        ]
    return pairs


def resolve_awq_mappings(model: Any) -> tuple[list[Any] | None, dict[str, Any]]:
    """Resolve the mappings against the real tree before anything expensive happens.

    Runs llm-compressor's own matcher, so what is checked here is what the modifier will
    do -- a hand-rolled name scan could agree with the regexes and disagree with
    ``match_modules_set``, and the disagreement is the whole defect.

    Returns the mappings and a record of what they cover, and returns ``None`` for the
    mappings when upstream has a table for this architecture -- the recipe then passes no
    ``mappings`` at all and ``AWQModifier`` looks the same list up itself. Handing back
    what was just read would be a second copy of it, and the two would agree right until a
    release changed one of them.

    The unsmoothed suffixes are in the record on purpose: ``conv.out_proj``, ``lm_head``
    and, under grouped-query attention, ``o_proj`` are quantized without an AWQ scale. That
    is a property of the architecture rather than a miss, and a number nobody writes down
    is a number that becomes a surprise the first time it changes.
    """
    import torch
    from compressed_tensors.utils import match_modules_set

    # Private, and imported anyway. It decides whether a `v_proj -> o_proj` pair survives,
    # and the whole set is dropped when any balance layer fails it. Reimplementing the test
    # would put a second copy of a dependency's arithmetic in this file; importing it means
    # a move raises here rather than quietly overstating what was smoothed.
    from llmcompressor.modifiers.transform.awq.base import _check_layers_are_compatible

    upstream = upstream_mappings(model)
    table: list[tuple[Any, int | None]] = (
        [(mapping, None) for mapping in upstream]
        if upstream is not None
        else awq_mappings(model.config)
    )
    # Identity-keyed, which is what `named_modules` gives and what upstream builds for the
    # same purpose: `_check_layers_are_compatible` reads suffixes off *resolved* names, and
    # a mapping's `smooth_layer` is a regex -- `re:.*v_proj$` ends with `$`, so passing the
    # pattern would make every compatibility test pass and the filter inert.
    module_names = {module: name for name, module in model.named_modules()}

    smoothed: set[int] = set()
    incompatible = 0
    resolved, wrong = [], []
    for mapping, expected in table:
        targets = (mapping.smooth_layer, *mapping.balance_layers)
        try:
            sets = list(match_modules_set(model, targets))
        except ValueError as exc:
            raise SystemExit(
                f"AWQ mapping {mapping.smooth_layer} matched part of its target set and not "
                f"the rest, which means these names do not describe this model: {exc}"
            ) from None
        # `expected is None` on the upstream path, where no config arithmetic predicts a
        # set count. Coverage is still checked below; only the prediction is dropped,
        # because a guessed one would fail runs rather than catch anything.
        if expected is not None and len(sets) != expected:
            wrong.append(
                f"{mapping.smooth_layer} resolved {len(sets)} sets, config predicts {expected}"
            )
        resolved.append(mapping)
        for smooth_layers, *nested in sets:
            # Flattened and tested as one set, because that is how the modifier treats it:
            # one incompatible balance layer skips the whole mapping for that block rather
            # than just itself.
            balance_layers = [module for group in nested for module in group]
            if not smooth_layers or not balance_layers:
                # A set the matcher hands back with a side missing smooths nothing, and
                # there is no pair to ask the shape question about. Not tallied as
                # incompatible either: what catches a matcher returning nothing is the
                # coverage check below, and it counts the modules that actually came back.
                continue
            if not _check_layers_are_compatible(
                smooth_layers[0],
                module_names.get(smooth_layers[0]),
                balance_layers,
                [module_names.get(module) for module in balance_layers],
            ):
                incompatible += 1
                continue
            smoothed.update(id(module) for module in balance_layers)

    if wrong:
        raise SystemExit(
            "AWQ mappings do not resolve as this model's config predicts, and a mapping "
            "that matches too little is silently skipped rather than raised -- the arm "
            "would score as AWQ having smoothed less than its label claims:\n  "
            + "\n  ".join(wrong)
        )

    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    unsmoothed: dict[str, int] = {}
    for name, module in linears:
        if id(module) not in smoothed:
            suffix = ".".join(name.split(".")[-2:])
            unsmoothed[suffix] = unsmoothed.get(suffix, 0) + 1
    covered = len(linears) - sum(unsmoothed.values())
    if not covered:
        raise SystemExit(
            f"every AWQ mapping resolved and not one of {len(linears)} Linear modules is "
            "balanced by any of them; the arm would be round-to-nearest under an AWQ label"
        )

    report = {
        "mapping_source": "llm-compressor" if upstream is not None else "this file (LFM2)",
        "mappings": len(resolved),
        "linear_modules": len(linears),
        "smoothed_linears": covered,
        # Nonzero means the modifier will skip that many sets on shape. Not an error --
        # under GQA it is every `v_proj -> o_proj` pair in the model -- but it is the
        # difference between "AWQ smoothed everything" and what actually happened.
        "shape_incompatible_sets": incompatible,
        "unsmoothed_linears": dict(sorted(unsmoothed.items())),
    }
    print(json.dumps(report), flush=True)
    return (None if upstream is not None else resolved), report


def calibration_rows(tokenizer: Any, samples: int, seq_len: int, *, seed: int) -> Any:
    """Tokenized rows from the task's own training mixture, in the fine-tune's format.

    Calibrating on generic web text and evaluating on text-to-SQL would hand DynQuant a win
    that came from the calibration distribution rather than from the method. These are the
    same rows in the same shape the fine-tune saw, through
    :func:`~dynquant.eval.text2sql.format_training_text`.

    Pre-tokenized rather than handed over as text, so llm-compressor's preprocessing cannot
    quietly apply a chat template or a different truncation rule. Truncation is from the left
    because the trailing cue is the part the model acts on.

    The shot exemplars are not held out here, and cannot be: they come from a ``shots``
    pseudo-split that ``load_text2sql`` resolves separately, so asking for ``train`` already
    excludes them.

    ``limit`` asks the loader for exactly the number wanted, and the rows are used as they
    come. Drawing more and then sampling down would undo the loader's own balancing -- it
    divides ``limit`` evenly across the three sources and interleaves them, and a random
    subset of that is only approximately balanced. The seed is the loader's, so which rows
    calibrate an arm is a recorded setting rather than a property of this function.
    """
    from datasets import Dataset

    from dynquant.eval.text2sql import format_training_text, load_text2sql

    picked = load_text2sql("train", limit=samples, seed=seed)

    rows = []
    for example in picked:
        prompt, completion = format_training_text(example)
        ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
        ids = ids[-seq_len:]
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    return Dataset.from_list(rows)


def check_publishable(model: Any, linearization: dict[str, Any]) -> None:
    """Refuse to write a checkpoint whose expert weights nothing will read back.

    ``linearize_moe`` renames every batched bank into ``experts.<i>.gate_proj`` and its
    two siblings. Putting them back is a separate feature -- ``ARCH_TO_2D_MAPPINGS``,
    applied through ``set_save_conversion_mapping`` -- and it is registered for
    ``deepseek_v4`` and ``qwen2_moe`` only. ``lfm2_moe`` linearizes through the generic
    protocol instead, so the surgery happens and the inverse of it does not exist.

    What that costs was measured on a four-layer ``lfm2_moe``, through this subcommand.
    The directory is written and looks right: 108 packed expert tensors, their scales,
    and a ``quantization_config`` naming them. Reloading it prints a report marking all
    108 ``UNEXPECTED`` and the six bank tensors ``MISSING``, and then returns a model.
    No exception. The banks come back from ``from_config``, and a 32-value group holds
    32 distinct values where 4-bit allows 16. On LFM2.5-8B-A1B that is 91.5% of the
    weights randomly initialized behind finite logits -- the failure prints, passes, and
    is only visible to someone who reads a table transformers writes on every load.

    vLLM does not rescue it. ``lfm2_moe.py`` keys its expert loader on
    ``ckpt_names=("w1", "w2", "w3")``, the canonical layout, which is exactly the one the
    linearized names are not.

    The condition is llm-compressor's own predicate, so a release that adds the mapping
    opens this gate without anyone editing the reason it gives.
    """
    from llmcompressor.modeling.moe.conversion_mappings import has_linearize_load_mappings

    banks = linearization["banks_before"]
    model_type = model.config.model_type
    if not banks or has_linearize_load_mappings(model_type):
        return
    raise SystemExit(
        f"{banks} expert bank(s) were linearized and {model_type!r} has no entry in "
        "llm-compressor's ARCH_TO_2D_MAPPINGS, so nothing converts the names back on the "
        f"way out. The directory would hold every expert at {model_type}'s linearized "
        "names, transformers would mark them UNEXPECTED, drop them, and reinitialize the "
        "banks from the config without raising -- measured at 32 distinct values in a "
        "32-value group, against 16 for 4-bit. vLLM reads the same weights under "
        "('w1', 'w2', 'w3') and would not find them either. Score it in-process with "
        "`run`, which is what the panel does and is unaffected, or publish the widths "
        "through `dynquant quantize --map`, which never linearizes and round-trips the "
        "bank as a bank"
    )


def quantize(
    args: argparse.Namespace, *, for_publication: bool = False
) -> tuple[Any, dict[str, Any]]:
    """Run the recipe on the linearized model and return it with its provenance record."""
    from llmcompressor import oneshot
    from transformers import AutoTokenizer

    model, linearization = load_linearized(args.model, dtype=args.dtype, device=args.device)

    # First, and for the same reason the AWQ check below comes early -- more so, because
    # this one is answered by the model type alone and its failure is silent.
    if for_publication:
        check_publishable(model, linearization)

    # Before the tokenizer, the calibration rows and the forward passes, because a mapping
    # that does not fit this architecture is a fact about the model that is knowable from
    # the module tree alone -- and discovering it after the calibration pass costs 256
    # sequences through 8 B parameters to learn something the names already said.
    mappings, smoothing = (None, None)
    if args.method == "awq":
        mappings, smoothing = resolve_awq_mappings(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = calibration_rows(tokenizer, args.calib_samples, args.seq_len, seed=args.seed)
    print(f"calibration: {len(dataset)} rows from the text2sql train mixture", flush=True)

    started = time.time()
    # The model object, already linearized, rather than the path -- `oneshot` would reload
    # from disk and lose both the linearization and the hidden_act that made it possible.
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=build_recipe(
            args.method, args.bits, args.group_size, ignore=IGNORE, mappings=mappings
        ),
        num_calibration_samples=len(dataset),
        max_seq_length=args.seq_len,
        pipeline=args.pipeline,
    )
    applied = materialize_quantization(model)

    meta = {
        "method": args.method,
        "bits": args.bits,
        "group_size": args.group_size,
        "ignore": list(IGNORE),
        "calib_samples": len(dataset),
        "seq_len": args.seq_len,
        "source": str(args.model),
        "quantize_seconds": round(time.time() - started, 1),
        "linearization": linearization,
        **({"awq_smoothing": smoothing} if smoothing is not None else {}),
        **applied,
        **accounted_bytes(args.model, args.bits, args.group_size),
    }
    print(json.dumps(meta, indent=2), flush=True)
    return model, meta


def eval_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """A real ``dynquant eval`` namespace, parsed by the CLI's own parser.

    Parsed rather than constructed, so every default these arms are scored under is the
    default the ceiling was scored under. A hand-built namespace would be a second copy of
    the eval contract, and the first setting to drift in it -- a batch size, a prompt style,
    the shot seed -- would be a difference between arms that the record still describes as
    shared.
    """
    from dynquant.cli import build_parser

    argv = [
        "eval",
        args.model,
        "--task",
        "text2sql",
        "--backend",
        "transformers",
        "--dtype",
        args.dtype,
        "--split",
        args.split,
        "--shots",
        str(args.shots),
        "--shot-seed",
        str(args.shot_seed),
        "--prompt-style",
        args.prompt_style,
        # Forwarded rather than left to the eval default, for the same reason the panel
        # states it: the default is derived from the source registry and widens whenever a
        # dataset carrying rows is added to it, so an arm that stayed silent would be
        # scored on whatever mixture the registry holds on the day it ran.
        "--sources",
        *args.sources,
        # The arm this driver builds has no `*Experts` module left -- `llm-compressor`
        # rewrites the bank into per-expert `Linear`s, which is what `eager` computes -- so
        # pinning changes nothing here and records everything: it is what lets this record
        # pair against a dq arm that *was* moved.
        "--experts-impl",
        args.experts_impl,
        "--label",
        args.label,
        # Stated because the model field below stops being a path. `eval` defaults the
        # tokenizer to `--model`, and the qualification that makes the record readable --
        # `<path>#gptq-4b-g128` -- is not a directory, so leaving it to the default sends a
        # `#` into `from_pretrained` and the arm dies at the tokenizer having already paid
        # for the calibration pass.
        "--tokenizer",
        args.model,
    ]
    for flag, value in (
        ("--limit", args.limit),
        ("--max-new-tokens", args.max_new_tokens),
        ("--batch-size", args.batch_size),
        ("--keep-predictions", args.keep_predictions),
        ("--out", args.out),
    ):
        if value is not None:
            argv += [flag, str(value)]

    namespace = build_parser().parse_args(argv)
    # `run` reads args.model for the record's provenance field, and the weights it is about
    # to score are not what that path holds any more. Kept as the source and qualified, so
    # the record says which checkpoint the arm was built from and which recipe built it.
    namespace.model = f"{args.model}#{args.method}-{args.bits}b-g{args.group_size}"
    return namespace


def score(args: argparse.Namespace, model: Any, meta: dict[str, Any]) -> int:
    """Score the quantized model in-process, then write the arm record beside the eval one.

    Two files rather than one merged record: ``dynquant eval`` owns its schema, the paired
    test reads it, and folding quantizer provenance into it would make an arm's record
    disagree in shape with the ceiling's.
    """
    from dynquant.commands import evaluate

    model.config.use_cache = True
    status = evaluate.run(eval_namespace(args), model=model)
    if args.out:
        side = Path(args.out).with_suffix(".quant.json")
        side.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"-> wrote {side}", flush=True)
    return status


def do_plan(args: argparse.Namespace) -> int:
    """What the recipe can see and what each width would cost -- no GPU, no weights."""
    report = {"visibility": visibility(args.model)}
    for bits in (4, 3):
        report[f"{bits}bit"] = accounted_bytes(args.model, bits, args.group_size)
    print(json.dumps(report, indent=2))
    return 0


def do_linearize(args: argparse.Namespace) -> int:
    """Run the 22-banks-to-zero gate and nothing else.

    Separate from ``run`` because it is the one assertion in this driver that no CPU unit test
    can reach -- it depends on llm-compressor's detector agreeing with this checkpoint's module
    tree -- and finding out it fails should not cost a calibration pass first. On ``--device
    cpu`` it costs host RAM and a few minutes.
    """
    load_linearized(args.model, dtype=args.dtype, device=args.device)
    return 0


def do_run(args: argparse.Namespace) -> int:
    model, meta = quantize(args)
    return score(args, model, meta)


def do_save(args: argparse.Namespace) -> int:
    """Write the checkpoint, for the widths that survive the round trip to a runtime.

    The criterion is that the width divides 32, and it is computed rather than listed
    because the listed version was wrong. This refusal used to read "compressed-tensors
    packs 4 and 8 bits; 3 would be saved as dequantized bf16", and both halves are false:
    ``pack_to_int32`` accepts ``1 <= num_bits <= 8`` and round-trips 3-bit correctly. What
    it does at 3 bits is pack ``32 // 3 == 10`` values per word and leave the top 2 bits
    zero, so the directory stores 3.2 bits per weight -- 6.7% over its own label, about
    200 MiB on this model and 67x the panel's match tolerance. A row whose width is not a
    multiple of 10 pays slightly more again: 2048 values need 205 words, or 3.2031 bits.

    The second half matters more. vLLM sizes the same tensor as ``Fraction(32, num_bits)``,
    which is AutoGPTQ's exact layout of 32 values in 3 words, so for a 2048-wide row it
    reads 192 words where compressed-tensors wrote 205. The checkpoint would be writable,
    internally consistent, and unreadable by the runtime it was published for. Widths that
    divide 32 have no gap between the two conventions and need no special case.

    So the refusal was right and its reason was not, which is this project's recurring
    failure mode: a guard refuses, blames the format, and the blame is believed. Here it was
    believed hard enough to be pinned by a test and repeated in two other drivers.

    The width is the second thing checked, not the first. :func:`check_publishable` runs
    inside ``quantize`` and asks whether the expert *names* survive the round trip, which
    on this architecture they do not at any width. So the two widths that pass the test
    below were, until that check existed, the ones that failed silently rather than the
    ones that worked.
    """
    if 32 % args.bits:
        per_word = 32 // args.bits
        stored = 32 / per_word
        raise SystemExit(
            f"{args.bits} does not divide 32, so compressed-tensors packs {per_word} values "
            f"per int32 and the directory stores {stored:.4f} bits per weight, "
            f"{stored / args.bits - 1:.1%} over the label. vLLM sizes the same tensor as "
            f"Fraction(32, {args.bits}) -- AutoGPTQ's exact layout -- so for a 2048-wide row "
            f"it reads {2048 * args.bits // 32} words where {-(-2048 // per_word)} were "
            f"written. "
            "The directory would be writable, self-consistent and unreadable by the runtime "
            "it is published for. Score it in-process with `run`, or write it with "
            "`dynquant export`, which packs this width at exactly the label"
        )
    model, meta = quantize(args, for_publication=True)
    out = Path(args.save_to)
    model.save_pretrained(str(out), save_compressed=True)
    meta["bytes_on_disk"] = sum(p.stat().st_size for p in out.rglob("*.safetensors"))
    (out / "dq_baseline.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"-> {out}  {meta['bytes_on_disk'] / 2**30:.2f} GiB", flush=True)
    return 0


def expert_rules() -> list[dict[str, Any]]:
    """How ``linearize_moe`` slices a bank, measured on the installed llm-compressor.

    Derived rather than declared. The two facts an inverse needs -- whether a bank is
    stored ``[E, in, out]`` or ``[E, out, in]``, and whether ``gate`` is the first or the
    second half of the fused projection -- both fail as a silent transpose, and a constant
    written here would be a second copy of llm-compressor's layout that agrees with it right
    up until a release moves one of them. ``probe_linearize_mapping`` answers both by
    building a tiny ``lfm2_moe``, linearizing it, and finding the bank slice whose values
    *are* each produced ``Linear``'s weight. Seconds of CPU, and it describes the library
    that is actually imported.
    """
    import tempfile

    from probe_linearize_mapping import derive
    from probe_linearized_save import build_tiny

    with tempfile.TemporaryDirectory() as work:
        source = Path(work) / "source"
        # Rectangular, so orientation is pinned by shape as well as by value.
        build_tiny(source, moe_intermediate_size=24)
        payload = derive(source)

    if payload["unmatched"] or payload["rule_conflicts"] or payload["unclaimed_elements"]:
        raise SystemExit(
            "linearize_moe is not a pure permutation of the banks on this llm-compressor: "
            f"{len(payload['unmatched'])} unmatched module(s), "
            f"{len(payload['rule_conflicts'])} rule conflict(s), "
            f"{len(payload['unclaimed_elements'])} bank(s) with unclaimed elements. "
            "The inverse below would silently write a wrong checkpoint, so it refuses "
            "instead. Re-run probe_linearize_mapping.py to see which assumption moved"
        )
    rules: list[dict[str, Any]] = payload["rules"]
    return rules


def delinearize_state_dict(
    model: Any, rules: list[dict[str, Any]], tensors: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The banked state dict a linearized model would have had, by the derived rules.

    Reconstructing a *state dict* rather than performing the module surgery in reverse. The
    surgery replaced each bank ``Parameter`` with a ``ModuleList``, so undoing it in place
    would mean rebuilding module objects; loading a correctly-keyed state dict into a fresh
    ``from_config`` model reaches the same place through an operation that is checkable, and
    the check is that the key set and every shape match the model that was linearized.

    No geometry constants. Bank widths come from concatenating the parts in the order the
    rules give, and the expert count from the indices observed -- so a release that changed
    ``2 * moe_intermediate`` to a different fusion would change the derived rules and this
    would follow, rather than quietly padding.

    ``tensors`` overrides what is assembled. The default -- the model's own state dict --
    rebuilds the weights. The publish path passes the quantizer's per-module *codes*, and
    then its scales and offsets, through the same call: those are row-indexed exactly as
    the weights are, so a bank's codes are the concatenation of its parts and the stack of
    its experts by the identical rule. One assembler rather than two keeps the arrangement
    that was proven bit-identical from being re-derived, slightly differently, for the
    tensors nobody round-trip-tested.
    """
    import re

    import torch

    by_linear = {rule["linear"]: rule for rule in rules}
    passthrough: dict[str, Any] = {}
    parts: dict[tuple[str, int, int], Any] = {}
    orientation: dict[str, str] = {}
    experts_seen: dict[str, set[int]] = {}
    split_count: dict[str, int] = {}

    for key, tensor in (model.state_dict() if tensors is None else tensors).items():
        module, _, leaf = key.rpartition(".")
        numbers = [int(n) for n in re.findall(r"\.(\d+)\.", f".{module}.")]
        rule = by_linear.get(re.sub(r"\.(\d+)\.", ".{}.", f".{module}.")[1:-1])
        if rule is None:
            passthrough[key] = tensor
            continue
        if leaf != "weight":
            # Not a passthrough: a linearized-name key surviving into the output is exactly
            # the failure this whole exercise exists to prevent, and it would survive
            # silently. lfm2_moe's experts carry no bias, so reaching here means the
            # architecture grew a tensor the rules do not describe.
            raise SystemExit(
                f"{key!r} hangs off a linearized expert module but is not its weight, and "
                "the derived rules say nothing about where it belongs in a bank. Writing it "
                "through under its linearized name is what makes a checkpoint that loads "
                "and is wrong"
            )
        wanted = rule["bank"].count("{}")
        assert len(numbers) >= wanted + 1, f"{key} has too few indices for {rule['bank']}"
        bank = rule["bank"].format(*numbers[:wanted])
        expert = numbers[-1]
        parts[(bank, expert, rule["part"])] = tensor
        orientation[bank] = rule["orientation"]
        split_count[bank] = rule["splits"]
        experts_seen.setdefault(bank, set()).add(expert)

    for bank, experts in experts_seen.items():
        assert experts == set(range(len(experts))), f"{bank} has gaps in its expert indices"
        slices = []
        for expert in range(len(experts)):
            pieces = [parts[(bank, expert, part)] for part in range(split_count[bank])]
            merged = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
            if orientation[bank] == "transposed":
                # Never taken under the rules this llm-compressor produces, and left as a
                # refusal rather than a `.t()`: a transposed *weight* is meaningful, but the
                # scales and codes travelling through this same assembler are indexed by row
                # and group, so transposing them is not a rearrangement of the same numbers.
                # Silently doing it for one caller and not the other is how the arrangement
                # that was proven would stop being the arrangement that ships.
                raise SystemExit(
                    f"{bank} is stored transposed, which the assembler cannot apply to codes "
                    "and scales. Re-run probe_linearize_mapping.py -- the layout moved"
                )
            slices.append(merged)
        # The bank is a bare Parameter on the experts module, so its state-dict key is the
        # parameter name itself with no ".weight" leaf -- which is what the probe reported
        # and what `from_config` will be looking for.
        passthrough[bank] = torch.stack(slices)

    return passthrough


def recipe_scratch(model: Any) -> set[str]:
    """State-dict keys compressed-tensors added, found by asking the modules that hold them.

    A quantized module carries a ``weight_scale``, and depending on the recipe a zero point
    and a column permutation beside it. Those are the recipe's working state, not the
    model's: the published container writes its own scales, and the banked architecture
    these weights are loaded back into has no parameter to receive them.

    Left in, they are not merely extra. Every one that hangs off a *linearized expert* trips
    :func:`delinearize_state_dict`'s refusal -- correctly, since the derived rules describe
    where a bank's weight rows go and say nothing about where a per-group scale would --
    and every one that hangs off anything else survives into the output and makes
    ``load_state_dict(strict=True)`` reject the model. Which is how this was found: the
    publish path was written against an unquantized state dict and never ran against a real
    recipe until :mod:`probe_publish` ran one.

    Derived by subtraction rather than by listing ``weight_scale``, ``weight_zero_point``,
    ``weight_g_idx``: a list here would be a copy of compressed-tensors' own set of
    artifacts, and this project has now been wrong four times in exactly that way. What a
    ``Linear`` legitimately owns is a weight and a bias; anything else a quantized one is
    holding arrived with the recipe. The filter is allowed to be liberal because it is not
    the check -- ``do_publish`` loads the result with ``strict=True``, so dropping one key
    too many fails as loudly as keeping one too few.
    """
    architectural = {"weight", "bias"}
    scratch: set[str] = set()
    for name, module in model.named_modules():
        if getattr(getattr(module, "quantization_scheme", None), "weights", None) is None:
            continue
        for leaf in module.state_dict():
            if leaf.split(".")[0] not in architectural:
                scratch.add(f"{name}.{leaf}" if name else leaf)
    return scratch


def recipe_weights(model: Any) -> dict[str, Any]:
    """The model's state dict with the recipe's own tensors taken back out."""
    scratch = recipe_scratch(model)
    return {key: value for key, value in model.state_dict().items() if key not in scratch}


def pristine_config(source: str) -> Any:
    """The checkpoint's own config, re-read, with the field ``linearize_moe`` needs.

    Re-read rather than reused, because ``model.config`` is not the checkpoint's config
    after a recipe has run: ``oneshot`` attaches a ``quantization_config`` describing
    compressed-tensors' format and clears ``tie_word_embeddings``. Handing that object to
    the model the publish path exports would write a directory claiming two quantization
    formats and holding two embedding tables.

    ``hidden_act`` is the one field this adds, and for the same reason
    :func:`load_linearized` adds it: LFM2's config omits it and ``linearize_moe`` reads it.
    Writing it into the checkpoint would mutate a directory the other arms load from.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(source)
    if not hasattr(config, "hidden_act"):
        config.hidden_act = EXPERT_ACT
    return config


def tie_report(model: Any) -> dict[str, Any]:
    """Does this model still hold one table for two names, and do the two names agree?

    Answered from storage, with ``config.tie_word_embeddings`` reported beside it as a
    separate fact rather than used as the answer -- because on this path the two disagree.
    Running a recipe over a tied LFM2 leaves ``lm_head`` and ``model.embed_tokens`` sharing
    one storage and holding identical numbers, and sets the config flag to ``False``:
    compressed-tensors is describing the checkpoint it intends to write, which carries two
    tables, not the model in memory, which carries one. Believing the flag skips the rename
    below, publishes the table under the head's name, and produces a directory whose
    embedding transformers reports missing and re-initializes at random.

    Both halves of the storage question are kept because they fail differently. Two
    distinct storages holding identical numbers are as good as one for anything written
    from a state dict. Two holding *different* numbers mean the recipe replaced the head's
    parameter and left the embedding behind at full precision -- a scored arm larger than
    its own byte accounting says, with an unquantized table the label does not mention.
    """
    import torch

    output, table = model.get_output_embeddings(), model.get_input_embeddings()
    names = {id(module): name for name, module in model.named_modules()}
    report: dict[str, Any] = {
        "config_says": bool(getattr(model.config, "tie_word_embeddings", False)),
        "input": names.get(id(table)),
        "output": names.get(id(output)),
        "tied": False,
    }
    if output is None or table is None:
        return report
    head, entries = output.weight, table.weight
    report["shared_storage"] = head.data_ptr() == entries.data_ptr()
    report["values_equal"] = bool(
        head.shape == entries.shape and torch.equal(head.detach(), entries.detach())
    )
    report["tied"] = report["shared_storage"] or report["values_equal"]
    return report


def under_the_input_table(model: Any, banked: dict[str, Any]) -> dict[str, Any]:
    """Move a tied head's entry onto the input embedding, which is where the loader looks.

    A tied checkpoint holds one table, and this format stores it under the *input*
    embedding's name: ``_tie_output_embedding`` then replaces the head with a
    ``DynQuantLinear`` that registers no tensors of its own and reads the embedding's. A
    recipe cannot produce that name on its own -- it targets ``Linear``, so what it
    quantizes is ``lm_head``, and an export keyed by that name writes the table under a
    name the loader does not look for. transformers then reports the embedding missing,
    initializes it randomly, and dies in ``mark_tied_weights_as_initialized`` calling
    ``get_parameter`` on a packed head. The 27% is on disk either way; this decides whether
    the directory opens.

    A rename and not a second entry, for the reason ``_tie_output_embedding``'s docstring
    gives: writing the table twice costs a quarter of a tied model for bytes the loader
    discards. The grid moves with it unchanged, because the two names address one tensor.
    """
    pair = tie_report(model)
    head, table = pair["output"], pair["input"]
    if not pair["tied"] or head is None or table is None or head not in banked:
        return banked
    if table in banked:
        raise SystemExit(
            f"both {table!r} and {head!r} carry a grid, but they are two names for one tied "
            "table and the format has one entry for it. Publishing both would write the "
            "table twice and leave the loader to pick"
        )
    moved = dict(banked)
    moved[table] = moved.pop(head)
    return moved


SCORED_FLAGS = ("method", "bits", "group_size", "ignore", "seq_len", "source")
"""What a republished arm has to have *asked* for to be the arm the panel scored.

Readable from the namespace before the recipe runs, which is why they are separate from
:data:`SCORED_WEIGHTS` below. A calibration pass over 8 B parameters costs the better part
of an hour, and every one of these is knowable from the command line -- so a
``--bits 3`` typo against a 4-bit record should cost a second, not a pass.
"""

SCORED_WEIGHTS = (
    "calib_samples",
    "materialized_modules",
    "weights_moved",
    "max_weight_delta",
    "probe_unique_values_per_row",
    "accounted_bits",
    "accounted_bytes",
    "quantized_params",
    "banked_params_quantized",
    "params",
)
"""What a republished arm has to have *produced* to be the arm the panel scored.

``publish`` re-runs the recipe, because the panel never serialized one -- ``run`` scores in
process and writes a record, not a checkpoint. So the weights in the published directory
come from a second calibration pass, and nothing about the label on the directory makes
them the weights the table's row was measured on. These are what the first pass recorded
about the weights it produced: how many modules the recipe touched, how far it moved them,
the per-row distinct-value probe, and the byte accounting the model card would quote.

They are compared exactly. A GPTQ Hessian accumulated in a different order could in
principle shift one row's distinct-value count by one, and that is not a rounding detail to
be waived -- it is the finding that the published weights are not the scored weights, which
is the entire question this check exists to answer. There is deliberately no flag to
override it: the disagreement is printed field by field, and what to do about it is a
decision, not a default.
"""


def scored_flag_disagreements(args: Any, scored: dict[str, Any]) -> dict[str, Any]:
    """Which of :data:`SCORED_FLAGS` this invocation would not reproduce."""
    asked = {
        "method": args.method,
        "bits": int(args.bits),
        "group_size": int(args.group_size),
        "ignore": list(IGNORE),
        "seq_len": int(args.seq_len),
        "source": str(args.model),
    }
    return {
        field: {"scored": scored[field], "asked": asked[field]}
        for field in SCORED_FLAGS
        if field in scored and scored[field] != asked[field]
    }


def scored_weight_disagreements(meta: dict[str, Any], scored: dict[str, Any]) -> dict[str, Any]:
    """Which of :data:`SCORED_WEIGHTS` the second pass did not reproduce.

    A field the scored record does not carry is skipped rather than treated as a
    disagreement, and the count of those is what :func:`check_matches_scored` reports
    beside the result -- a record written before a field existed can still be matched on
    the fields it has, but nobody should read that as a match on the fields it does not.
    """
    return {
        field: {"scored": scored[field], "republished": meta.get(field)}
        for field in SCORED_WEIGHTS
        if field in scored and scored[field] != meta.get(field)
    }


def check_matches_scored(meta: dict[str, Any], scored: dict[str, Any], where: str) -> int:
    """Refuse a directory whose weights are not the ones the arm was scored on.

    Returns how many of :data:`SCORED_WEIGHTS` the record actually carried, so the caller
    can publish the coverage rather than only the verdict.
    """
    covered = sum(1 for field in SCORED_WEIGHTS if field in scored)
    moved = scored_weight_disagreements(meta, scored)
    if moved:
        raise SystemExit(
            f"this pass did not reproduce the arm recorded in {where}, so the directory it "
            f"would write is not the model that row was scored on:\n"
            f"{json.dumps(moved, indent=2)}"
        )
    return covered


MAX_CARRY_DRIFT = 0.125
"""How far a carried reconstruction may sit from the one that was scored, in code steps.

Not zero, and the reason is arithmetic rather than tolerance for error. compressed-tensors
reconstructs ``scale * (q - zero)`` with an integer zero point; this format stores
``scale * code + offset`` with a float offset, so the two agree as *grids* -- same levels,
same codes -- but the offset ``-scale * zero`` is itself rounded to the storage dtype, and
in bf16 that costs up to 2**-9 of its magnitude. The result is a constant shift per group,
predicted at a few percent of a step and measured per module by :func:`carrying_encoder`.

A mapping error -- experts stacked in the wrong order, gate and up swapped, a transposed
bank -- moves weights by whole steps or more, so this threshold is 8x below the smallest
failure it has to catch and well above the rounding it has to admit. The measured value is
recorded in the directory's own record either way, so the gap is a published number rather
than something this constant hides."""


def carried_grids(model: Any, *, group_size: int) -> dict[str, dict[str, Any]]:
    """Each quantized module's integer codes and affine terms, read off the quantizer.

    Not re-fitted from the weights. ``materialize_quantization`` has already written the
    rounded values back into ``module.weight``, so the floats sit on the quantizer's grid --
    but recovering the grid from them requires every group to still occupy both ends of its
    code range, and GPTQ's error compensation and AWQ's clipping search both leave groups
    that do not. Where a group does not, min/max fits a narrower step, the original levels
    fall between the new ones, and the published weights are not the ones that were scored.
    See :meth:`~dynquant.quant.tensor.QuantTensor.from_codes`.

    The codes are recomputed here rather than imported from compressed-tensors, and then
    *checked*: reconstructing ``scale * (q - zero)`` in the library's own order has to
    reproduce the materialized weight bit for bit. Since the weight already is
    ``fake_quantize(weight)``, that check has exactly one degree of freedom -- whether this
    file's idea of the convention matches the library's -- which is the disagreement that
    has cost this project four times, and it fails loudly here instead of silently on disk.
    It cost it a fifth time on the asymmetric AWQ arm, so the one piece of the convention
    that is not recomputed-and-checked -- the code range -- is now imported outright.
    """
    import torch
    from compressed_tensors.quantization.utils import calculate_range
    from compressed_tensors.utils import align_module_device

    grids: dict[str, dict[str, Any]] = {}
    for name, module in model.named_modules():
        scheme = getattr(getattr(module, "quantization_scheme", None), "weights", None)
        if scheme is None or not hasattr(module, "weight_scale"):
            continue
        with align_module_device(module):
            weight = module.weight.data
            scale = module.weight_scale.data
            zero = getattr(module, "weight_zero_point", None)
            zero = torch.zeros_like(scale) if zero is None else zero.data.to(scale.dtype)

            out_features, in_features = weight.shape
            groups = -(-in_features // group_size)
            if tuple(scale.shape) != (out_features, groups):
                raise SystemExit(
                    f"{name} carries a {tuple(scale.shape)} scale where a group-{group_size} "
                    f"recipe on a {tuple(weight.shape)} weight gives {(out_features, groups)}. "
                    "This reader only understands per-group scales along the input dimension"
                )

            bits = int(scheme.num_bits)
            # Asked of the library instead of derived here. compressed-tensors puts every
            # integer scheme on a *signed* band -- [-2^(b-1), 2^(b-1)-1] -- whether or not
            # it is symmetric; an asymmetric scheme rides that same band with a signed zero
            # point rather than moving to an unsigned code. Deriving it instead gave
            # [0, 2^b - 1] for the asymmetric case, which agrees with the library on every
            # symmetric arm and clamped 60% of an AWQ weight onto the bottom rail. The
            # format has no signed code, so the band is folded into the offset below and no
            # level moves.
            low, high = calculate_range(scheme, weight.device)
            qmin, qmax = int(low.item()), int(high.item())

            wide_scale = scale.repeat_interleave(group_size, dim=1)[:, :in_features]
            wide_zero = zero.repeat_interleave(group_size, dim=1)[:, :in_features]
            codes = torch.clamp(torch.round(weight / wide_scale) + wide_zero, qmin, qmax)
            recon = (wide_scale * (codes - wide_zero)).to(weight.dtype)
            if not torch.equal(recon, weight):
                worst = (recon.float() - weight.float()).abs().max().item()
                raise SystemExit(
                    f"{name}: scale * (q - zero) does not reproduce the materialized weight "
                    f"(max |delta| = {worst:.3e}). Either the weights were never rounded onto "
                    "the grid, or this reader's idea of compressed-tensors' convention has "
                    "diverged from the library's. Publishing from here would write a "
                    "directory that loads and is wrong"
                )

            grids[name] = {
                # scale * (q - zero) == scale * (q - qmin) + scale * (qmin - zero), so an
                # unsigned code and a float offset describe the identical set of levels.
                "codes": (codes - qmin).to(torch.uint8).cpu(),
                "scales": scale.detach().cpu().clone(),
                "offsets": (scale.float() * (qmin - zero.float())).to(scale.dtype).cpu(),
                "bits": bits,
            }

    if not grids:
        raise SystemExit(
            "no module carries a weight quantization scheme, so there is no grid to carry "
            "and the directory this would write is the unquantized checkpoint under another "
            "name. Check that the recipe ran and that materialize_quantization saw it"
        )
    return grids


def _module_of(key: str) -> str:
    """A state-dict key back to the module the bit map would name it by.

    A bank is a bare ``Parameter``, so its key *is* its name; a ``Linear`` contributes
    ``<name>.weight``. ``resolve_target`` accepts both, and this is the one place the
    difference between the assembler's output and the exporter's input is resolved.
    """
    return key[: -len(".weight")] if key.endswith(".weight") else key


def banked_grids(
    model: Any, grids: dict[str, dict[str, Any]], rules: list[dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], int]:
    """The per-module grids rearranged into banks, by the assembler the weights use.

    Codes, scales and offsets each go through :func:`delinearize_state_dict`, the same call
    that was proven bit-identical on the weights. They can, because all three are indexed by
    output row: a bank's codes are the concatenation of its parts and the stack of its
    experts under the identical rule its weights are, and a per-group scale row belongs to
    the same output row its weight row does. Writing a second arrangement here -- even a
    correct one -- would be a fifth copy of a mapping that has already gone wrong four times.

    Returns the banked grids and the single width they were all quantized at. A baseline
    recipe is uniform by construction, and the width has to be checked rather than assumed
    because a bank assembled from two widths has no single entry in the bit map.
    """
    assembled = {
        field: delinearize_state_dict(
            model, rules, {f"{name}.weight": grid[field] for name, grid in grids.items()}
        )
        for field in ("codes", "scales", "offsets")
    }

    widths = {grid["bits"] for grid in grids.values()}
    if len(widths) != 1:
        raise SystemExit(
            f"the recipe quantized at widths {sorted(widths)}. Experts at two widths land in "
            "one bank, which is one tensor with one entry in the bit map, so there is no "
            "width to record for it"
        )

    banked: dict[str, dict[str, Any]] = {}
    for key, codes in assembled["codes"].items():
        scales, offsets = assembled["scales"][key], assembled["offsets"][key]
        if scales.ndim == 3:
            # A bank: [E, out, groups] describes the same E * out rows the packer gets when
            # it folds the codes' leading dimensions, and in the same order.
            scales = scales.reshape(-1, scales.shape[-1])
            offsets = offsets.reshape(-1, offsets.shape[-1])
        banked[_module_of(key)] = {"codes": codes, "scales": scales, "offsets": offsets}
    return banked, widths.pop()


def carrying_encoder(
    banked: dict[str, dict[str, Any]], *, group_size: int, drift: dict[str, float]
) -> Any:
    """An ``export_packed_checkpoint`` encoder that adopts the baseline's grid.

    Every module the exporter asks about is answered from ``banked`` and then checked
    against the weight the exporter was going to quantize -- which is the *fresh banked
    model's* weight, assembled from the materialized floats by one path, against codes
    assembled by another. So this is not a self-consistency check: the two sides reach the
    same tensor through the float route and the integer route, and they only agree if the
    rules were applied identically to both. An expert stacked in the wrong order fails here,
    on the module, rather than in an accuracy number three days later.

    ``drift`` collects the measured gap in code steps, per module, for the record.
    """
    import torch

    from dynquant.quant.tensor import QuantTensor

    def encode(name: str, weight: torch.Tensor, width: int) -> QuantTensor:
        grid = banked.get(name)
        if grid is None:
            raise SystemExit(
                f"the exporter asked for {name!r}, which the recipe never quantized. The bit "
                "map and the carried grids came from the same dict, so this means one of "
                "them was rebuilt in between"
            )
        quantized = QuantTensor.from_codes(
            grid["codes"],
            grid["scales"],
            grid["offsets"],
            bits=width,
            group_size=group_size,
            symmetric=False,
            compute_dtype=weight.dtype,
            logical_shape=tuple(weight.shape),
        )
        got = quantized.dequantize(dtype=weight.dtype)
        step = grid["scales"].float().abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
        worst = ((got.float() - weight.float()).abs().amax() / step).item()
        drift[name] = worst
        if worst > MAX_CARRY_DRIFT:
            raise SystemExit(
                f"{name}: the carried grid reconstructs {worst:.3f} code steps away from the "
                f"weight the arm was scored on, past the {MAX_CARRY_DRIFT} this format's "
                "offset rounding accounts for. That is the size of a rearrangement error, "
                "not of a rounding -- re-run probe_delinearize.py"
            )
        return quantized

    return encode


def do_publish(args: argparse.Namespace) -> int:
    """Write the baseline's own grid into a DynQuant-format directory.

    :func:`do_save` cannot publish this model, for two reasons it states at length: the
    linearized expert names have no inverse registered for ``lfm2_moe``, and 3 bits does not
    divide 32 so compressed-tensors stores 3.2. Both are properties of that container, and
    neither is a property of the numbers. This subcommand keeps the numbers and changes the
    container -- the codes the recipe chose, in the format the DynQuant arms publish in,
    which packs every width at exactly its label and round-trips a bank as a bank.

    What it does *not* do is re-quantize. Handing the materialized weights to
    ``export_packed_checkpoint`` unaided would re-fit a grid from them, and where GPTQ or
    AWQ left a group not spanning its code range that fitted grid is narrower than the one
    the arm was scored on. The published checkpoint would be a different model than the
    table's row. The encoder is what keeps the two the same, and the drift it measures is
    what proves it.

    Not a vLLM artifact. Serving these through vLLM's compressed-tensors path would need a
    new ``lfm2_moe`` entry in llm-compressor's ``ARCH_TO_2D_MAPPINGS``, which composes with
    transformers' own checkpoint conversion mapping -- and ``lfm2_moe`` has none, because its
    published checkpoint is already banked. That is upstream work with an upstream release
    cycle. This directory loads through ``dynquant``'s own ``HfQuantizer``.

    And it re-quantizes, because the panel never serialized anything -- ``run`` scores in
    process and writes a record. So this is a *second* calibration pass, and the label on
    the directory is the only thing connecting it to the row in the table. ``--scored``
    replaces that with a measurement: the arm's own record, compared field by field against
    what this pass produced. Without it the directory is published on the strength of
    matching flags, which is a claim about the inputs and not about the weights.
    """
    import dataclasses

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from dynquant.quant.checkpoint import export_packed_checkpoint

    # Read before the recipe runs, because the recipe rewrites it. `oneshot` sets
    # `tie_word_embeddings` to False on the live config -- true of the compressed checkpoint
    # it would have written, false of the model in memory -- and it is the model in memory
    # that is being published.
    declared_tie = bool(getattr(pristine_config(args.model), "tie_word_embeddings", False))

    # Before the recipe, not after: every one of these is knowable from the namespace, and
    # learning that `--bits` disagrees with the record costs a second here and 32 minutes
    # of calibration below.
    scored = None
    if args.scored:
        scored = json.loads(Path(args.scored).read_text(encoding="utf-8"))
        asked = scored_flag_disagreements(args, scored)
        if asked:
            raise SystemExit(
                f"these flags would not reproduce the arm recorded in {args.scored}:\n"
                f"{json.dumps(asked, indent=2)}"
            )

    model, meta = quantize(args)
    if scored is not None:
        meta["scored"] = {
            "record": str(args.scored),
            "fields_compared": check_matches_scored(meta, scored, args.scored),
            "fields_available": len(SCORED_WEIGHTS),
        }
    rules = expert_rules()
    grids = carried_grids(model, group_size=args.group_size)
    banked, width = banked_grids(model, grids, rules)
    tie = tie_report(model)
    if declared_tie and not tie["tied"]:
        raise SystemExit(
            f"{args.model} declares its head tied to its embedding, and after the recipe "
            f"{tie['output']} and {tie['input']} hold different weights. This arm was scored "
            "with two tables and its byte accounting describes one, so either name published "
            "would be a model that was not evaluated"
        )
    banked = under_the_input_table(model, banked)
    meta["tie"] = tie
    # Not `model.state_dict()`: after a recipe runs, every quantized module also holds
    # the scales and zero points it was fitted with, and those belong to neither the
    # bank assembler nor the architecture being loaded. See :func:`recipe_scratch`.
    weights = delinearize_state_dict(model, rules, recipe_weights(model))
    source = args.model
    del model, grids

    # Reloaded from disk rather than built with `from_config`: the banked tree is what the
    # checkpoint already holds, so this costs a mmap instead of initializing 8.5 B
    # parameters, and every non-persistent buffer arrives real rather than on meta.
    # `assign=False` on purpose -- assigning would replace the tied embedding with two
    # distinct tensors and the exporter, which detects tying by identity, would write it
    # twice.
    fresh = AutoModelForCausalLM.from_pretrained(
        source, config=pristine_config(source), dtype=getattr(torch, args.dtype), device_map="cpu"
    )
    fresh.load_state_dict(weights, strict=True)
    fresh.tie_weights()
    del weights

    drift: dict[str, float] = {}
    out = Path(args.save_to)
    report = export_packed_checkpoint(
        fresh,
        dict.fromkeys(banked, width),
        output_dir=out,
        group_size=args.group_size,
        compute_device=args.pack_device,
        provenance={"baseline": meta, "carried_from": "compressed-tensors"},
        encoder=carrying_encoder(banked, group_size=args.group_size, drift=drift),
    )
    AutoTokenizer.from_pretrained(source).save_pretrained(str(out))

    written = dataclasses.asdict(report)
    written["output_dir"] = str(written["output_dir"])
    written.pop("layers", None)
    meta["published"] = written
    meta["carry_drift_steps"] = {
        "max": round(max(drift.values()), 6),
        "worst_module": max(drift, key=lambda k: drift[k]),
        "threshold": MAX_CARRY_DRIFT,
        "modules": len(drift),
    }
    meta["bytes_on_disk"] = sum(p.stat().st_size for p in out.rglob("*.safetensors"))
    (out / "dq_baseline.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta["carry_drift_steps"], indent=2), flush=True)
    print(f"-> {out}  {meta['bytes_on_disk'] / 2**30:.2f} GiB", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    def quant_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--method", choices=METHODS, required=True)
        p.add_argument("--bits", type=int, default=4)
        p.add_argument("--group-size", type=int, default=128)
        # 256 rows at 1024 tokens. GPTQ's Hessian is estimated from this and published
        # recipes use 128-512; on a 4-of-32 router each expert still sees only about an
        # eighth of it, which is the caveat in the module docstring rather than something
        # more rows here can fix.
        p.add_argument("--calib-samples", type=int, default=256)
        p.add_argument("--seq-len", type=int, default=1024)
        # Sequential holds one submodule's activations at a time, so an 8B calibrates
        # without the whole model plus its Hessians resident at once.
        p.add_argument("--pipeline", default="sequential")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--dtype", default="bfloat16")
        # Where the model loads for the recipe. `cuda` because that is where a calibration
        # pass belongs; exposed because otherwise `run` -- the only subcommand the panel
        # uses -- cannot be exercised at all on a box whose GPU is busy, and a driver that
        # can only be rehearsed on the hardware the real run needs is a driver that gets
        # rehearsed for the first time during the real run.
        p.add_argument("--device", default="cuda")

    def eval_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--label", required=True)
        p.add_argument("--out", default=None)
        p.add_argument("--split", default="test")
        p.add_argument("--shots", type=int, default=2)
        p.add_argument("--shot-seed", type=int, default=0)
        p.add_argument("--prompt-style", default="chat")
        p.add_argument("--sources", nargs="+", default=["gretel", "wikisql"], metavar="NAME")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--batch-size", type=int, default=None)
        # No default. The budget for these arms is read off the ceiling run's closure
        # distribution by `closure_budget.py`; a default here would be a second guess at the
        # number that script exists to measure.
        p.add_argument("--max-new-tokens", type=int, default=None)
        p.add_argument("--keep-predictions", type=int, default=None)
        # A second copy of a `dynquant eval` flag, which is a shape this campaign has been
        # bitten by four times. Kept anyway, for the reason `eval_namespace` gives: this
        # parser exists so the driver can *state* what it wants and let the CLI's own
        # parser supply everything it did not. The default here matches the CLI's, and the
        # value is threaded through rather than re-derived.
        p.add_argument("--experts-impl", default="eager", choices=("eager", "auto"))

    p = sub.add_parser("plan", help="what the recipe can see and what each width costs")
    p.add_argument("--model", required=True)
    p.add_argument("--group-size", type=int, default=128)
    p.set_defaults(func=do_plan)

    lin = sub.add_parser("linearize", help="check the banks convert, and nothing else")
    lin.add_argument("--model", required=True)
    lin.add_argument("--dtype", default="bfloat16")
    lin.add_argument("--device", default="cuda")
    lin.set_defaults(func=do_linearize)

    r = sub.add_parser("run", help="quantize and score in one process (no checkpoint)")
    r.add_argument("--model", required=True)
    quant_flags(r)
    eval_flags(r)
    r.set_defaults(func=do_run)

    pub = sub.add_parser(
        "publish", help="quantize and write the baseline's own grid in DynQuant format"
    )
    pub.add_argument("--model", required=True)
    pub.add_argument("--save-to", required=True)
    # Where the packing arithmetic runs, independent of where the recipe ran. Separate from
    # --device because the two happen at different times and the second one is cheap: it is
    # a pack of codes that already exist, not a calibration pass.
    pub.add_argument("--pack-device", default="auto")
    # The `<label>.quant.json` the scored arm wrote. Optional, because the probe publishes a
    # model no panel ever scored and has nothing to compare against; supplied for anything
    # that will carry a panel arm's name.
    pub.add_argument("--scored", default=None)
    quant_flags(pub)
    pub.set_defaults(func=do_publish)

    s = sub.add_parser("save", help="quantize and write a packed checkpoint")
    s.add_argument("--model", required=True)
    s.add_argument("--save-to", required=True)
    quant_flags(s)
    s.set_defaults(func=do_save)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    status: int = args.func(args)
    return status


if __name__ == "__main__":
    sys.exit(main())
