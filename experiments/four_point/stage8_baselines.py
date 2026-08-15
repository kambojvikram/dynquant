"""Post-training baselines -- GPTQ, AWQ and RTN -- on the same checkpoint DynQuant quantizes.

Why this stage exists
---------------------
Stages 5 and 6 answer "does the allocated bit map beat a uniform one at the same
budget". That is an internal control: it holds the quantizer fixed and varies only the
allocation. It cannot say whether the *quantizer* is competitive, because both arms
are DynQuant's. This stage supplies the external comparison -- the methods a reader
would actually reach for instead.

Everything here is deliberately downstream of :mod:`common`. The baselines quantize
``RUN_DIR/finetuned``, the same merged fine-tune stage 5 quantizes, and they are scored
by :func:`common.run_eval`, the same function every other measurement point goes
through. So the arms differ in their weights and in nothing else -- not the prompt, not
the decode settings, not the test subset, not the few-shot prefix.

One toolchain, three methods
----------------------------
All three come from ``llm-compressor`` (vLLM project) rather than from ``gptqmodel``
plus ``autoawq`` plus something for RTN. Not convenience: three toolchains would mean
three loaders, three save formats and three sets of defaults about what stays in fp16,
and any accuracy gap between GPTQ and AWQ would then be partly a gap between two
codebases' conventions. Here the three recipes differ by the modifier and share
everything else, so the comparison between *them* is as controlled as the comparison
against DynQuant.

``autoawq`` was the alternative for the AWQ arm and was rejected: it prints its own
deprecation notice and reports its last tested configuration as torch 2.6 /
transformers 4.51, against the 2.11 / 5.x this box runs.

The calibration set is not generic web text
-------------------------------------------
It is drawn from the task's own training split and formatted with
``TASK.format_training_text`` -- the same rows, in the same shape, that the fine-tune
saw. Calibrating on wikitext and evaluating on Banking77 would hand DynQuant a win that
came from the calibration distribution rather than from the method. The shot exemplars
are held out here for the same reason they are held out of training: they sit in the
evaluation prompt, and a baseline that had calibrated on them would be scored partly on
memorisation.

The honest caveat, which belongs in any table this produces
-----------------------------------------------------------
DynQuant reads gradient and activation statistics collected *during* fine-tuning. GPTQ
and AWQ see only forward activations on a calibration set, and RTN sees nothing but the
weights. DynQuant is therefore using strictly more information than either baseline --
that is the method's premise, not an accident of this script, but it is not a
like-for-like input budget and a table that omits it overstates the result.

What is comparable, and what is not
-----------------------------------
Nominal bit width is not comparable and should not be the x-axis. DynQuant is
mixed-precision, so "4.25 bits" is an average over modules; GPTQ and AWQ carry an fp16
scale (and, asymmetric, a zero point) per group of 128, so "4 bits" is really ~4.19.
And the methods disagree about what stays in fp16: DynQuant quantizes the embedding and
the LM head, whereas the ``ignore`` list below leaves ``lm_head`` alone, which is the
convention every published GPTQ/AWQ checkpoint follows. Comparing on nominal width
would credit the baselines for bytes they still spend. So this stage records measured
on-disk bytes for every arm, and the table reads accuracy against that.

``oneshot`` does not round the weights, and that silently faked a result
-----------------------------------------------------------------------
llm-compressor separates *calibration* from *compression*. ``oneshot`` fits the scales
and zero points, attaches them to the module and sets ``quantization_status=FROZEN`` --
and for ``QuantizationModifier`` that is all it does. The weight tensor is untouched;
the rounding happens later, inside ``save_pretrained(save_compressed=True)``. The
in-process :func:`do_run` path deliberately never saves, so it was scoring bf16 weights
with an unused set of scales bolted on.

It was caught because RTN 4-bit returned *byte-identical* predictions to bf16 on all
3080 problems -- 47 seconds of calibration that changed nothing an evaluation can see.
The dangerous case is AWQ, not RTN: the AWQ transform *does* rewrite weights, so an
unrounded AWQ arm produces plausibly-different numbers rather than suspiciously
identical ones, and would have entered the table as an unbeatable baseline that was
never actually quantized. GPTQ is unaffected -- its algorithm writes corrected weights
back as it sweeps -- which is exactly why "the other arms look fine" was no evidence.

:func:`materialize_quantization` closes this by applying the frozen scales to the
weights itself, and it runs for every method: it is corrective for RTN and AWQ, and on
GPTQ it is a verified no-op (max residual 0.0, because those weights are already on the
grid). The guard that matters is the fixed-point check afterwards -- re-quantizing a
quantized weight must return it bit-identically, which is false for any tensor that was
not rounded, and false again if the write did not land. That second failure is real:
assigning ``module.weight.data`` directly does not survive, and the parameter has to be
updated through ``update_offload_parameter``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _llmc import METHODS, build_recipe, default_symmetric, stored_meta_bits
from common import RUN_DIR, SEED, TASK, load_task, model_slug, run_eval, set_seed

IGNORE = ["lm_head"]
"""Left in fp16 by every published GPTQ and AWQ checkpoint, and by llm-compressor's own
examples. Kept rather than "corrected" to match DynQuant: the point of a baseline is to
be the thing a reader would actually run, and a hand-tuned variant of it is a different
claim. The bytes it costs are counted -- see the module docstring.

``--include-head`` empties this, and on a model with a *tied* embedding that is not a
cosmetic variant -- it is the difference between a comparison and a category error.
Mistral-7B unties the two tensors, so the LM head is 5.5% of the checkpoint and leaving
it in fp16 costs the baselines 0.35 bits. Qwen3.5-2B shares one 508.6M tensor between
``embed_tokens`` and ``lm_head``, 27% of the model, and ``lm_head`` being on this list
pins that whole tensor at fp16: measured on the fine-tuned checkpoint, "4-bit g128"
accounts to **7.3605 bits** and "3-bit" to **6.6253**, with 59% and 65% of the total
bits sitting in tensors nobody quantized. Compared against DynQuant at 4.2486, the
accuracy column would then be reporting that a model keeping a quarter of itself in
full precision does better than one that does not, which is true and uninformative.

So both conventions get run on a tied model and reported as separate panels. The default
panel is what a reader would get today, and the size gap in it is a real result about the
convention. The ``--include-head`` panel is the one that isolates the allocator: with the
tie quantized the baselines land at 4.1597 and 3.1522 bits, 2.1% and 3.0% *below*
DynQuant's own measured width, so any accuracy difference there is method, not budget,
and the residual budget error runs against DynQuant rather than for it.
"""


def calibration_rows(tokenizer: Any, samples: int, seq_len: int) -> Any:
    """Tokenized rows from the task's training split, in the fine-tune's own format.

    Pre-tokenized rather than handed over as text, so llm-compressor's preprocessing
    cannot quietly apply a chat template or a different truncation rule than the one
    :meth:`tasks.Task.training_row` uses. Truncation is from the left for the same
    reason that class truncates left: the trailing cue is the part the model acts on.
    """
    from datasets import Dataset

    train, _, _ = load_task()
    rng = random.Random(SEED)
    picked = rng.sample(train, min(samples, len(train)))

    rows = []
    for example in picked:
        prompt, completion = TASK.format_training_text(example)
        ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
        ids = ids[-seq_len:]
        rows.append({"input_ids": ids, "attention_mask": [1] * len(ids)})

    return Dataset.from_list(rows)


def directory_bytes(path: Path) -> int:
    """Total weight bytes on disk -- the only size number that is method-neutral."""
    return sum(p.stat().st_size for p in path.rglob("*.safetensors"))


def _unique_params(model: Any) -> list[Any]:
    """Parameters deduplicated by identity -- ``model.parameters()`` already does this,
    but only for tensors reachable as distinct attributes; tied weights are one object
    and must be counted once."""
    seen: dict[int, Any] = {}
    for param in model.parameters():
        seen.setdefault(id(param), param)
    return list(seen.values())


def accounted_bytes(
    source: str,
    bits: int,
    group_size: int,
    *,
    symmetric: bool,
    actorder: str | None = None,
) -> dict[str, Any]:
    """What this checkpoint costs, counted the way stage 5 counts DynQuant's.

    Needed because on-disk size is only available for arms that can actually be packed,
    and the 3-bit arms cannot be -- though not for the reason written here originally,
    which was that compressed-tensors packs only 4- and 8-bit and would hold a 3-bit
    result as dequantized bf16. Measured later: it packs 1 to 8 bits and round-trips
    3-bit fine, but at ``32 // 3 == 10`` values per word, so the file stores 3.2 bits per
    weight rather than 3, and vLLM reads that tensor as ``Fraction(32, 3)`` and finds the
    wrong number of words. A 3-bit directory is therefore oversized against its label and
    unreadable by the runtime, which leaves on-disk size just as unusable a denominator as
    bf16 would have -- for a different reason, and by 6.7% rather than by 5x. That is the
    same situation stage 5 is in -- it writes dequantized values back in place too -- and ``RESULTS`` handles
    it the same way, by computing the size the packed checkpoint would occupy from the
    format's own accounting. Doing anything else here would compare a measured number
    against a computed one and call the difference a result.

    Measured against a **meta-device copy of the source architecture**, not against the
    model llm-compressor returns. The returned model is not a clean module tree: the
    quantized Linears carry observer state and scale/zero-point parameters, and the
    modules themselves are no longer plain ``nn.Linear``. Walking it reported 14.34 bits
    per weight for a 3-bit checkpoint -- most of the model misclassified as fp16 because
    the type test missed it, in the direction that flatters nobody but is wrong. The
    architecture is what the size question is actually about, it is free to instantiate
    on ``meta``, and the quantizer cannot perturb it.

    Counted per tensor rather than as ``params x bits``: what separates these arms from
    DynQuant on size is mostly *which* tensors stay in fp16, and a formula over the
    total parameter count cannot see that. Group metadata is an fp16 scale per group,
    plus a zero point when asymmetric, matching the convention in the module docstring.
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
        numel = module.weight.numel()
        quantized += numel * bits + stored_meta_bits(
            numel,
            module.weight.shape[-1],
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            actorder=actorder,
        )
        counted.add(id(module.weight))

    # Everything else at 16 bits, deduplicated by identity so a tied embedding/LM-head
    # pair is one tensor rather than two. Mistral unties them, but Qwen3.5-2B does not,
    # and a size table that silently double-counts 27% of the model on the second run
    # would be worse than no table.
    unique = _unique_params(ref)
    fp16 = sum(p.numel() * 16 for p in unique if id(p) not in counted)
    total = sum(p.numel() for p in unique)
    return {
        "accounted_bits": round((quantized + fp16) / total, 4),
        "accounted_gib": round((quantized + fp16) / 8 / 2**30, 4),
        "fp16_bits_share": round(fp16 / (quantized + fp16), 4),
        "quantized_params": sum(
            m.weight.numel()
            for n, m in ref.named_modules()
            if isinstance(m, nn.Linear) and not any(n.endswith(p) for p in IGNORE)
        ),
        "params": total,
    }


def materialize_quantization(model: Any, *, probes: int = 8) -> dict[str, Any]:
    """Write the frozen scales into the weights, and prove they landed.

    See the module docstring: ``oneshot`` fits scales but leaves the weight tensor alone
    for every recipe except GPTQ, so without this the in-process path scores an
    unquantized model. Applied uniformly rather than per method -- it is idempotent on
    weights that are already on the grid, so GPTQ passes through it unchanged, and a
    method-specific branch here would be one more thing to get wrong the next time
    llm-compressor moves the boundary between calibration and compression.

    The returned counts are recorded with the arm. ``weights_moved`` is expected to be
    zero for GPTQ and equal to ``modules`` for RTN and AWQ; an arm whose counts do not
    match that shape is reporting something other than what its label says.
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
                # Absent under stage 8's recipes, which name no activation ordering. Passed
                # regardless: `fake_quantize` reads a missing mapping as the contiguous one,
                # so the day a recipe here grows `actorder` this omission would not raise,
                # it would quietly quantize against the wrong columns. See `_llmc`.
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

    # Fixed-point check on a spread of modules, re-reading the parameter rather than the
    # value just computed. Quantizing a quantized weight must be a no-op; a tensor that
    # was never rounded fails this, and so does one whose update silently did not stick.
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


def quantize(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    """Run the recipe and return the quantized model plus its provenance record."""
    import time

    from llmcompressor import oneshot
    from transformers import AutoTokenizer

    # Mutated in place, not rebound: ``build_recipe`` hands this same list object to the
    # modifier, and the record below and ``accounted_bytes`` both read the global. One
    # object means the recipe that ran, the width that was accounted, and the ``ignore``
    # field in the record cannot disagree -- which is the failure this would invite if the
    # flag were threaded separately to each of the three.
    if getattr(args, "include_head", False):
        IGNORE.clear()

    symmetric = {"auto": None, "yes": True, "no": False}[getattr(args, "symmetric", "auto")]
    actorder = None if getattr(args, "actorder", "none") == "none" else args.actorder
    # Resolved once, here, because two things downstream need the answer and not the
    # request: the arm record, which has to say what ran, and `accounted_bytes`, which
    # charges a zero point only to the scheme that stores one. Resolving it twice is how
    # the record and the size column come to disagree about the same arm.
    resolved_symmetric = default_symmetric(args.method) if symmetric is None else symmetric

    set_seed()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = calibration_rows(tokenizer, args.calib_samples, args.seq_len)
    print(f"calibration: {len(dataset)} rows from {TASK.key} train", flush=True)

    started = time.time()
    model = oneshot(
        model=args.model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=build_recipe(
            args.method,
            args.bits,
            args.group_size,
            ignore=IGNORE,
            symmetric=symmetric,
            actorder=actorder,
        ),
        num_calibration_samples=len(dataset),
        max_seq_length=args.seq_len,
        precision="bfloat16",
        pipeline=args.pipeline,
    )
    # Before anything reads this model. Both callers need it: the run path because it
    # never saves, and the save path because materializing first makes the checkpoint it
    # writes and the model it held agree -- compression on already-rounded weights is a
    # no-op, so this costs the save path nothing but removes the divergence.
    applied = materialize_quantization(model)

    meta = {
        "method": args.method,
        "bits": args.bits,
        "group_size": args.group_size,
        # Resolved, not requested: `auto` is a different scheme per method, and an arm
        # that records the flag rather than the answer cannot say what it ran.
        "symmetric": resolved_symmetric,
        "actorder": actorder,
        "ignore": IGNORE,
        "calib_samples": len(dataset),
        "seq_len": args.seq_len,
        "source": str(args.model),
        "quantize_seconds": round(time.time() - started, 1),
        **applied,
        **accounted_bytes(
            args.model,
            args.bits,
            args.group_size,
            symmetric=resolved_symmetric,
            actorder=actorder,
        ),
    }
    print(json.dumps(meta, indent=2), flush=True)
    return model, meta


def do_quantize(args: argparse.Namespace) -> None:
    model, meta = quantize(args)
    out = Path(args.out)
    model.save_pretrained(str(out), save_compressed=True)
    meta["bytes_on_disk"] = directory_bytes(out)
    (out / "dq_baseline.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"-> {out}  {meta['bytes_on_disk'] / 2**30:.2f} GiB", flush=True)


def do_run(args: argparse.Namespace) -> None:
    """Quantize and score in one process, without writing a checkpoint.

    This is the default path for the comparison, and the reason given at the time was disk:
    a 3-bit result was believed to save as dequantized bf16, and a 7 B bf16 checkpoint is
    14.5 GB against ~23 GB free on this box with the fine-tune already occupying 14 GB of
    it. That belief was wrong -- a 3-bit save is packed, at about 2.9 GB -- so disk was
    never the binding constraint and the real one is that the directory would be 6.7% over
    its label and unreadable by vLLM. The path is unchanged because the second reason below
    was always the sufficient one. Holding the model in
    memory and scoring it there costs nothing that the table needs -- the size column
    comes from :func:`accounted_bytes`, not from the filesystem, for every arm including
    the packable ones, so the arms remain comparable to each other and to stage 5.
    """
    model, meta = quantize(args)
    model.config.use_cache = True
    run_eval(
        model,
        label=args.label or args.name,
        name=args.name,
        limit=args.limit,
        extra={**meta, "model_id": model_slug()},
    )


def do_eval(args: argparse.Namespace) -> None:
    """Score a saved checkpoint through the shared harness.

    Loads by path with no method-specific branch: a compressed-tensors checkpoint
    carries its own ``quantization_config``, so transformers reconstructs the right
    runtime from the directory. That is what lets the DynQuant and bf16 arms be
    re-scored by this same entry point -- which they must be, because a baseline
    measured under one transformers version and a DynQuant number measured under
    another are not the same measurement.
    """
    import torch
    from transformers import AutoModelForCausalLM

    set_seed()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    model.config.use_cache = True

    meta_path = Path(args.model) / "dq_baseline.json"
    extra: dict[str, Any] = {"checkpoint": str(args.model), "model_id": model_slug()}
    if meta_path.exists():
        extra.update(json.loads(meta_path.read_text(encoding="utf-8")))
    else:
        extra["bytes_on_disk"] = directory_bytes(Path(args.model))

    run_eval(model, label=args.label or args.name, name=args.name, limit=args.limit, extra=extra)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="stage", required=True)

    def quant_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--method", choices=METHODS, required=True)
        p.add_argument("--model", default=str(RUN_DIR / "finetuned"))
        p.add_argument("--bits", type=int, default=4)
        p.add_argument("--group-size", type=int, default=128)
        # 256 rows at 1024 tokens. GPTQ's Hessian is estimated from this, and the
        # published recipes use 128-512; the low end of that range is where the estimate
        # starts to be the thing limiting the arm rather than the method.
        p.add_argument("--calib-samples", type=int, default=256)
        p.add_argument("--seq-len", type=int, default=1024)
        # Sequential holds one submodule's activations at a time, so a 7B calibrates
        # without the whole model plus its Hessians resident at once.
        p.add_argument("--pipeline", default="sequential")
        # `auto` is each method's own published default -- symmetric for GPTQ and RTN,
        # asymmetric for AWQ -- which is what a reader downloading a checkpoint under
        # that name would get, and so is what the panel arms run. The override exists so
        # the control gets run instead: a delta between a symmetric arm and an
        # asymmetric one spans the scheme as well as the allocation, and only an arm
        # matching its opponent's scheme can say which of the two the delta belongs to.
        p.add_argument("--symmetric", choices=("auto", "yes", "no"), default="auto")
        # GPTQ only; `_llmc.build_recipe` refuses it on the other two rather than
        # dropping it, because a flag silently ignored is a control a caller believes
        # it ran.
        p.add_argument("--actorder", choices=("none", "group", "weight"), default="none")
        # See IGNORE. Off by default because the default has to stay the recipe a reader
        # would run; on for the second panel of a tied-embedding model, where the default
        # leaves 27% of the weights untouched and the size columns stop being comparable.
        p.add_argument(
            "--include-head",
            action="store_true",
            help="quantize lm_head too (required for a comparable footprint on a model "
            "that ties lm_head to embed_tokens, e.g. Qwen3.5)",
        )

    r = sub.add_parser("run", help="quantize and score in one process (no checkpoint)")
    quant_flags(r)
    r.add_argument("--name", required=True)
    r.add_argument("--label", default=None)
    r.add_argument("--limit", type=int, default=None)
    r.set_defaults(func=do_run)

    q = sub.add_parser("quantize", help="build and save a baseline checkpoint")
    quant_flags(q)
    q.add_argument("--out", required=True)
    q.set_defaults(func=do_quantize)

    e = sub.add_parser("eval", help="score a saved checkpoint through the shared harness")
    e.add_argument("--model", required=True)
    e.add_argument("--name", required=True)
    e.add_argument("--label", default=None)
    e.add_argument("--limit", type=int, default=None)
    e.set_defaults(func=do_eval)

    args = parser.parse_args()
    print(f"run dir: {RUN_DIR}  task: {TASK.key}  model: {os.environ.get('DQ_MODEL')}", flush=True)
    args.func(args)


if __name__ == "__main__":
    main()
