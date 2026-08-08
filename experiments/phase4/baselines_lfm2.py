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


def resolve_awq_mappings(model: Any) -> tuple[list[Any], dict[str, Any]]:
    """Resolve the mappings against the real tree before anything expensive happens.

    Runs llm-compressor's own matcher, so what is checked here is what the modifier will
    do -- a hand-rolled name scan could agree with the regexes and disagree with
    ``match_modules_set``, and the disagreement is the whole defect.

    Returns the mappings and a record of what they cover. The unsmoothed suffixes are in
    the record on purpose: ``conv.out_proj`` and ``lm_head`` are quantized without an AWQ
    scale, that is a property of the architecture rather than a miss, and a number nobody
    writes down is a number that becomes a surprise the first time it changes.
    """
    import torch
    from compressed_tensors.utils import match_modules_set

    smoothed: set[int] = set()
    resolved, wrong = [], []
    for mapping, expected in awq_mappings(model.config):
        targets = (mapping.smooth_layer, *mapping.balance_layers)
        try:
            sets = list(match_modules_set(model, targets))
        except ValueError as exc:
            raise SystemExit(
                f"AWQ mapping {mapping.smooth_layer} matched part of its target set and not "
                f"the rest, which means these names do not describe this model: {exc}"
            ) from None
        if len(sets) != expected:
            wrong.append(
                f"{mapping.smooth_layer} resolved {len(sets)} sets, config predicts {expected}"
            )
        resolved.append(mapping)
        for matched in sets:
            for balance in matched[1:]:
                smoothed.update(id(module) for module in balance)

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
        "mappings": len(resolved),
        "linear_modules": len(linears),
        "smoothed_linears": covered,
        "unsmoothed_linears": dict(sorted(unsmoothed.items())),
    }
    print(json.dumps(report), flush=True)
    return resolved, report


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


def quantize(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    """Run the recipe on the linearized model and return it with its provenance record."""
    from llmcompressor import oneshot
    from transformers import AutoTokenizer

    model, linearization = load_linearized(args.model, dtype=args.dtype, device=args.device)

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
    """Write the checkpoint, for the arms whose width ``compressed-tensors`` can pack.

    Refused at 3 bits rather than allowed to write a dequantized bf16 directory: that file
    would be four times the size its label implies, and a later reader taking its byte count
    for the arm's footprint is the same wrong-denominator error this driver exists to stop.
    """
    if args.bits not in (4, 8):
        raise SystemExit(
            f"compressed-tensors packs 4 and 8 bits; {args.bits} would be saved as "
            "dequantized bf16, so the directory's size would not be this arm's size. Score "
            "it in-process with `run` instead"
        )
    model, meta = quantize(args)
    out = Path(args.save_to)
    model.save_pretrained(str(out), save_compressed=True)
    meta["bytes_on_disk"] = sum(p.stat().st_size for p in out.rglob("*.safetensors"))
    (out / "dq_baseline.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
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
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--batch-size", type=int, default=None)
        # No default. The budget for these arms is read off the ceiling run's closure
        # distribution by `closure_budget.py`; a default here would be a second guess at the
        # number that script exists to measure.
        p.add_argument("--max-new-tokens", type=int, default=None)
        p.add_argument("--keep-predictions", type=int, default=None)

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
