#!/usr/bin/env python3
"""S2, audio arm: QLoRA on Qwen3-Omni's Thinker, and the signal map it exists to produce.

Same contract as ``run_s2_finetune.py`` -- the fine-tuned checkpoint is the by-product
and ``stats/dynquant_stats.json`` is the deliverable -- against a model that the text
script cannot load, on a task whose examples are waveforms. Four things differ, and each
of them is the reason this is a second script rather than a flag on the first.

**The model is not an ``AutoModelForCausalLM``.** It is
``Qwen3OmniMoeForConditionalGeneration``, and only its ``.thinker`` submodule is trained
here: the Talker and code2wav are 3.2B of parameters that generate *speech*, which SLURP
does not score and no gradient in this run would reach. Dropping them is not a saving of
convenience -- 3.02B of those parameters are expert banks, and a bank that is measured
costs a gradient buffer whether or not anything trains it.

**``all-linear`` is banned.** On this checkpoint it attaches to the visual tower and the
lm_head and still misses 91.40% of the weight, because the Thinker's MLP is not made of
``nn.Linear`` at all -- it is 96 batched 3-D ``nn.Parameter`` banks holding 28.991B
parameters. The target list here is written out per tower and asserted non-empty per
pattern, so a rename upstream fails loudly instead of quietly training the attention and
calling the expert mass covered.

**bitsandbytes cannot 4-bit the banks either.** ``replace_with_bnb_linear`` replaces
``nn.Linear`` and ``Conv1D``; the banks are neither. So "QLoRA" on this model quantizes
about 2.1B of Linears and leaves 28.991B in bf16, and the memory plan has to be built
around that number rather than around the usual assumption that 4-bit halves the problem
twice over.

**The bank gradient buffer does not fit, and the fix is a rotation.**
``measure_expert_banks=True`` calls ``requires_grad_(True)`` on every bank so that
plasticity is measured on the tensor being quantized rather than on an adapter standing
in for it. Measuring all 96 at once costs a 54.0 GiB gradient buffer on top of 59.1 GiB
of weights -- 113 GiB against a 96 GiB card, and DDP replicates the model per rank, so
the second GPU does not relieve it. :class:`BankShardRotation` enables one shard of
layers per optimizer step, so the buffer is a fraction of that (9.0 GiB at the default
six shards) and each bank is sampled every sixth step instead of every step.

That rotation needs nothing from ``dynquant``: the tracker already skips an entry whose
``.grad`` is ``None`` and already masks the Welford update by whether an observation
arrived, so a bank outside the active shard contributes no sample rather than a zero.
Plasticity is a variance over optimizer steps and stays one; the shard only changes how
many steps each bank is sampled on, which the stats file records as its own count.

Usage::

    # cheap: build the data, report the prompt statistics, touch no GPU
    python scripts/run_s2_omni_finetune.py --dry-run

    # the load path alone, with a key census -- proves what actually came back
    python scripts/run_s2_omni_finetune.py --probe-load

    # the run, both GPUs, gradient accumulation on top
    torchrun --nproc_per_node 2 scripts/run_s2_omni_finetune.py \\
        --model /workspace/models/qwen3-omni-30b \\
        --out /workspace/runs/omni-slurp \\
        --examples 8000 --batch 1 --accum 8 --measure-expert-banks
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover -- import cost is the whole point of the guard
    pass

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

DEFAULT_MODEL = "/workspace/models/qwen3-omni-30b"

# Per tower, and written out rather than inferred. The suffixes collide across towers on
# purpose -- `q_proj` names both a text-stack and an audio-tower projection and both are
# wanted -- while `o_proj` (text) and `out_proj` (audio) differ, which is exactly the kind
# of asymmetry a hand-written list gets wrong silently. The counts are asserted at
# attach time against this table, so a rename upstream is a failure and not a smaller
# fine-tune.
LORA_TARGETS: dict[str, int] = {
    # Thinker text stack, 48 layers x 4 projections.
    "q_proj": 48 + 32,  # text self_attn + audio_tower self_attn
    "k_proj": 48 + 32,
    "v_proj": 48 + 32,
    "o_proj": 48,  # text only
    "out_proj": 32,  # audio tower only
    "fc1": 32,  # audio tower MLP
    "fc2": 32,
}

# Deliberately absent from the list above, and each for its own reason:
#   lm_head          -- 152064 x 2048 over a vocabulary the task uses about sixty tokens
#                       of. An adapter there is the largest in the model and trains the
#                       output distribution rather than the acoustics. (Not a tie: this
#                       checkpoint sets text_config.tie_word_embeddings=False and stores
#                       thinker.lm_head.weight separately from embed_tokens.)
#   mlp.gate         -- the router. It is a bare `nn.Parameter`, so `all-linear` and bnb
#                       both miss it structurally; an adapter cannot attach to it at all.
#   visual.*         -- SLURP is audio. The tower never runs, so it takes no gradient and
#                       would contribute an adapter of zeros.
#   mlp.experts.*    -- the banks. Not Linear, not adaptable; measured, not trained.

_BANK_LAYER = re.compile(r"\.layers\.(\d+)\.")


def bank_parameters(model: Any) -> dict[int, str]:
    """Every batched expert bank in ``model``, by ``id()``, named as the model names it.

    Read through ``dynquant.graph.experts.batched_expert_params`` rather than by matching
    names, because that is the registry the tracker itself uses to decide what a bank is.
    A second matcher here would agree with it right up until one of them learned about a
    new architecture, and the disagreement would show up as a gradient buffer for tensors
    nothing measures, or -- worse -- as banks left in the DDP bucket set.
    """
    from dynquant.graph.experts import batched_expert_params

    by_id: dict[int, str] = {}
    for module_name, module in model.named_modules():
        for param_name, param in batched_expert_params(module):
            by_id[id(param)] = f"{module_name}.{param_name}" if module_name else param_name
    return by_id


def bank_shards(model: Any, shards: int) -> tuple[dict[int, list[Any]], list[str]]:
    """Group the banks into ``shards`` rotation groups by layer index.

    By layer, not round-robin over tensors, so a shard is a contiguous slice of the
    backward pass: both banks of layer *i* are enabled together and their gradients are
    born and released at the same point. Sharding across a layer's own two banks would
    hold the layer's activations alive for two different steps to no benefit.
    """
    by_id = bank_parameters(model)
    groups: dict[int, list[Any]] = collections.defaultdict(list)
    names: list[str] = []
    unplaced: list[str] = []
    for _, param in model.named_parameters():
        name = by_id.get(id(param))
        if name is None:
            continue
        names.append(name)
        match = _BANK_LAYER.search(name)
        if match is None:
            # No layer index to rotate on. Put it in every shard rather than in none:
            # a bank that is never enabled is a bank that is never measured, which is
            # the failure this whole mechanism exists to avoid.
            unplaced.append(name)
            for shard in range(shards):
                groups[shard].append(param)
            continue
        groups[int(match.group(1)) % shards].append(param)
    if unplaced:
        print(
            f"WARNING: {len(unplaced)} expert bank(s) carry no layer index and are "
            f"enabled on every step: {unplaced[:3]}",
            flush=True,
        )
    return groups, names


def make_rotation_callback(model: Any, shards: int) -> tuple[Any, list[str]]:
    """A ``TrainerCallback`` that enables one shard of expert banks per optimizer step."""
    from transformers import TrainerCallback

    groups, names = bank_shards(model, shards)

    class BankShardRotation(TrainerCallback):
        def __init__(self) -> None:
            self.active: int | None = None

        def _apply(self, step: int) -> None:
            want = step % shards
            if want == self.active:
                return
            for shard, params in groups.items():
                on = shard == want
                for param in params:
                    param.requires_grad_(on)
                    if not on:
                        # Releasing here as well as in the tracker: a bank switched off
                        # mid-run keeps whatever `.grad` it last had, and that tensor
                        # would sit in memory for the rest of the run holding exactly the
                        # bytes this rotation exists to avoid holding.
                        param.grad = None
            self.active = want

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            # After DynQuantCallback in the callback list, so this runs after the tracker
            # has turned every bank on. Ordering is not incidental: attaching second is
            # what makes this a narrowing of the tracker's decision rather than a race
            # with it.
            self._apply(0)

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            # `global_step` has already been incremented, so this selects the shard for
            # the step about to begin. Every rank computes the same number from the same
            # counter, which is what keeps the ranks' `requires_grad` flags in agreement.
            self._apply(state.global_step)

    return BankShardRotation(), names


def load_thinker(
    path: str,
    *,
    four_bit: bool,
    device: Any,
    dtype: Any,
) -> tuple[Any, dict[str, int]]:
    """The Thinker, on ``device``, with the Talker and code2wav never left resident.

    The full model is loaded and the Thinker taken out of it, rather than the Thinker
    being loaded on its own. That is not the obvious way round, and the checkpoint is why:
    it stores 19,743 Thinker tensors against the 1,407 the Thinker module has, because the
    experts are written per-expert and unfused -- ``experts.{e}.{gate,up,down}_proj`` over
    128 experts and 48 layers -- and ``from_pretrained`` is what fuses gate with up and
    stacks the 128 into the batched 3-D banks. Asking the Thinker class for those keys
    directly matches 0 of 1,407 as-is and leaves exactly the 96 banks missing once the
    ``thinker.`` prefix is stripped. Loading the class transformers writes the conversion
    for, then dropping what is not trained, needs no assumption about that conversion.

    The transient cost is the Talker and code2wav being resident between the load and the
    ``del``: about 11 GiB above the 59.1 GiB that stays. That fits, and the alternative is
    a hand-written key remapping that would have to track the checkpoint format.
    """
    import torch
    import transformers

    kwargs: dict[str, Any] = {"dtype": dtype, "device_map": device}
    if four_bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            # Named explicitly because transformers did not skip it on its own here: a
            # bare 4-bit load turned all 504 Linears into `Linear4bit`, lm_head included.
            # The head is 0.62 GiB in bf16 and it is the last matmul before the loss, so
            # every gradient this run exists to measure is computed from its output.
            # Quantizing it would put NF4 rounding noise into the signal map itself.
            llm_int8_skip_modules=["lm_head"],
        )

    whole = transformers.Qwen3OmniMoeForConditionalGeneration.from_pretrained(path, **kwargs)
    census = {
        "loaded_params": sum(p.numel() for p in whole.parameters()),
        "loaded_modules": sum(1 for _ in whole.modules()),
    }
    thinker = whole.thinker
    for dropped in ("talker", "code2wav"):
        if getattr(whole, dropped, None) is not None:
            setattr(whole, dropped, None)
    del whole
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    census["thinker_params"] = sum(p.numel() for p in thinker.parameters())
    census["thinker_banked"] = sum(
        p.numel() for module in thinker.modules() for _, p in _banked(module)
    )
    return thinker, census


def _banked(module: Any) -> Any:
    from dynquant.graph.experts import batched_expert_params

    return batched_expert_params(module)


def probe_load(args: argparse.Namespace) -> int:
    """Load, drop, and count what came back -- per group, not as a total.

    A total that reconciles does not say which tensors are behind it, and
    ``from_pretrained`` reports a key mismatch as a printed table rather than an
    exception. So this counts the *distinct values* of a few things that would all be
    wrong together if the load had silently reinitialised anything: the bank dtypes (bf16
    if they came from the checkpoint), the Linear classes (bnb's ``Linear4bit`` if the
    quantization ran), and the fraction of parameters that are still on the meta device.
    """
    import torch

    started = time.time()
    thinker, census = load_thinker(
        args.model, four_bit=not args.no_four_bit, device={"": 0}, dtype=_dtype()
    )
    elapsed = time.time() - started

    classes = collections.Counter(type(m).__name__ for m in thinker.modules())
    banks = {name: p for name, p in thinker.named_parameters() if id(p) in bank_parameters(thinker)}
    meta = [n for n, p in thinker.named_parameters() if p.device.type == "meta"]

    print(f"\nloaded in {elapsed:.0f}s")
    print(f"  full model:    {census['loaded_params'] / 1e9:.3f}B params")
    print(f"  thinker kept:  {census['thinker_params'] / 1e9:.3f}B params")
    print(
        f"  of which banked: {census['thinker_banked'] / 1e9:.3f}B "
        f"({census['thinker_banked'] / census['thinker_params']:.2%})"
    )
    print(f"  bank tensors:  {len(banks)}")
    print(f"  bank dtypes:   {collections.Counter(str(p.dtype) for p in banks.values())}")
    print(f"  bank devices:  {collections.Counter(str(p.device) for p in banks.values())}")
    print(f"  on meta:       {len(meta)} parameters {meta[:3]}")
    for name in ("Linear", "Linear4bit", "Params4bit"):
        print(f"  {name:<12}   {classes.get(name, 0)}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(index) / 2**30
            print(f"  cuda:{index} allocated {allocated:.1f} GiB")

    groups, names = bank_shards(thinker, args.bank_shards)
    sizes = {shard: sum(p.numel() for p in params) * 2 / 2**30 for shard, params in groups.items()}
    print(f"\n{args.bank_shards} shards over {len(names)} banks:")
    for shard in sorted(sizes):
        print(f"  shard {shard}: {len(groups[shard])} tensors, grad buffer {sizes[shard]:.1f} GiB")

    if meta:
        print("\nPARAMETERS LEFT ON META: the load did not populate them", flush=True)
        return 7
    return 0


def _dtype() -> Any:
    import torch

    return torch.bfloat16


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def pick_shots(pool: list[Any], count: int, seed: int) -> tuple[list[Any], set[int]]:
    """The evaluation's own few-shot prefix, and the train indices it occupies.

    Reproduced rather than re-drawn. ``dynquant.commands.evaluate._pick_shots`` samples
    ``sorted(random.Random(seed).sample(range(len(pool)), count))`` from the *train*
    split, and the fine-tune has to (a) render the same prefix, so the model trains under
    the frame it is scored under, and (b) exclude those rows from its own training set,
    because an exemplar shown in every prompt and also trained on is an answer handed to
    the model for free. The second is the reason the indices come back as well as the
    examples.

    The pool is passed in rather than loaded here, so that this and the training rows are
    drawn from one load. Two ``load_slurp`` calls would pay the annotation join twice and,
    more to the point, would leave nothing in the code saying the two orderings are the
    same one -- and the exclusion below is only correct if they are.
    """
    if count == 0:
        return [], set()
    rng = random.Random(seed)
    chosen = sorted(rng.sample(range(len(pool)), count))
    return [pool[i] for i in chosen], set(chosen)


def build_examples(
    *, examples: int, shot_count: int, shot_seed: int, cache_dir: str | None
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """The training rows and the shot prefix, with the shots removed from the rows."""
    from dynquant.eval.slurp import load_slurp

    pool = load_slurp("train", cache_dir=cache_dir)
    shots, shot_indices = pick_shots(pool, shot_count, shot_seed)
    kept = [example for index, example in enumerate(pool) if index not in shot_indices]
    subset = kept if examples <= 0 else kept[:examples]

    intents = collections.Counter(example.intent for example in subset)
    census = {
        "train_pool": len(pool),
        "shots_excluded": len(shot_indices),
        "train_used": len(subset),
        "distinct_intents": len(intents),
        "menu_size": len(subset[0].intents) if subset else 0,
        "rarest_intent": min(intents.items(), key=lambda kv: kv[1]) if intents else None,
        "shot_answers": [shot.answer for shot in shots],
    }
    return subset, shots, census


class SlurpSft:
    """A map-style dataset that builds one prompt per ``__getitem__``.

    Lazy on purpose. The train split is 50,628 clips; materialising the waveforms to train
    on 8,000 of them costs gigabytes and minutes for rows that are never read. ``datasets``
    keeps the table memory-mapped and ``SlurpExample.audio`` is a thunk, so the decode
    happens in the dataloader worker that needs it, in parallel with the step before.
    """

    def __init__(
        self,
        processor: Any,
        examples: list[Any],
        shots: list[Any],
        *,
        max_prompt_tokens: int,
    ) -> None:
        self.processor = processor
        self.examples = examples
        self.shots = shots
        self.max_prompt_tokens = max_prompt_tokens
        tokenizer = processor.tokenizer
        end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        # `eos_token_id` on this checkpoint is not necessarily the token the chat template
        # closes an assistant turn with, and the target has to end with the one generation
        # will actually stop on -- otherwise the model is trained to run past its own
        # terminator and the evaluation reads the overrun as an unparseable answer.
        self.end_id = end if isinstance(end, int) and end >= 0 else tokenizer.eos_token_id
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.end_id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from dynquant.eval.slurp import build_prompt

        example = self.examples[index]
        prompt = build_prompt(self.processor, example, self.shots)
        target = self.processor.tokenizer.encode(example.answer, add_special_tokens=False)
        target = [*target, self.end_id]
        ids = [*prompt.ids, *target]
        if len(ids) > self.max_prompt_tokens:
            # Truncating would cut the audio placeholder run out of the middle of the
            # prompt and offset the spectrogram against the text. Refused instead, and
            # counted by the collator, because a silently shortened audio prompt scores
            # like a weak model rather than failing.
            return {"input_ids": None, "labels": None, "features": None, "dropped": True}
        labels = [-100] * len(prompt.ids) + target
        return {
            "input_ids": ids,
            "labels": labels,
            "features": dict(prompt.features),
            "dropped": False,
        }


def collate(rows: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    """Right-pad the ids and pad the encoder inputs by their own rule.

    Two paddings, because the two halves are padded for different reasons. The ids are
    padded to a common length with ``pad_id`` and masked out of the loss with ``-100``;
    the spectrograms are padded by :func:`dynquant.eval.omni.batch_features`, which pads
    with zeros because a zero mel frame is silence and a zero in
    ``feature_attention_mask`` is the flag that tells the encoder to ignore it.

    Right padding, not left: the audio placeholder run sits in the middle of the prompt
    and left padding would move it, while the loss mask already keeps the trailing pad out
    of the objective.
    """
    import torch

    from dynquant.eval.omni import batch_features

    live = [row for row in rows if not row["dropped"]]
    if not live:
        raise ValueError(
            "every example in this batch exceeded --max-prompt-tokens. An audio prompt is "
            "the menu plus a placeholder run whose length the encoder fixes; if all of "
            "them overflow, the limit is wrong rather than the data."
        )

    width = max(len(row["input_ids"]) for row in live)
    n = len(live)
    input_ids = torch.full((n, width), pad_id, dtype=torch.long)
    labels = torch.full((n, width), -100, dtype=torch.long)
    attention = torch.zeros((n, width), dtype=torch.long)
    for i, row in enumerate(live):
        length = len(row["input_ids"])
        input_ids[i, :length] = torch.tensor(row["input_ids"], dtype=torch.long)
        labels[i, :length] = torch.tensor(row["labels"], dtype=torch.long)
        attention[i, :length] = 1

    batch = batch_features([row["features"] for row in live])
    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention
    batch["labels"] = labels
    return batch


# --------------------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------------------


def freeze_base(model: Any) -> None:
    """What ``prepare_model_for_kbit_training`` does, minus the part that cannot run here.

    The peft helper is the standard QLoRA preamble, and it is wrong for this model in one
    line: it upcasts *every* fp16/bf16 parameter to fp32 for numerical headroom, skipping
    only bitsandbytes' ``Params4bit``. The expert banks are neither -- bnb replaces
    ``nn.Linear`` and ``Conv1D`` only, so 28.991 B of banks stay bf16 and are upcast, 116
    GiB of fp32 against a 96 GiB card. The first smoke run died inside that loop, at 82.7
    GiB allocated, having cast maybe two thirds of them.

    Of what the helper does at ``use_gradient_checkpointing=False``, only the freeze is
    load-bearing, and it is done here. The upcast is deliberately not reproduced, and not
    only because it does not fit:

    - The tensors the optimizer steps are fp32 already. ``get_peft_model`` defaults to
      ``autocast_adapter_dtype=True``, which upcasts every LoRA A/B after injection. The
      upcast the helper is *for* happens either way.
    - Upcasting the norms would be wrong here rather than merely unaffordable. An RMSNorm
      returns ``weight * hidden.to(input_dtype)``, so an fp32 weight makes the hidden
      state fp32, and the next thing the text stack does with it is a batched matmul
      against a bf16 expert bank -- which does not promote, it raises.

    The banks stay bf16, which is the dtype the tracker measures them in and the dtype the
    merged checkpoint is written in, so nothing downstream has to be told about this.
    """
    for param in model.parameters():
        param.requires_grad_(False)


def attach_lora(model: Any, args: argparse.Namespace) -> Any:
    """LoRA on the named projections, with the count of each asserted before training."""
    from peft import LoraConfig, get_peft_model

    targets = sorted(LORA_TARGETS)
    found: collections.Counter[str] = collections.Counter()
    import torch

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear) and type(module).__name__ != "Linear4bit":
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf in LORA_TARGETS and ".visual." not in name and not name.endswith("lm_head"):
            found[leaf] += 1

    wrong = {k: (found.get(k, 0), v) for k, v in LORA_TARGETS.items() if found.get(k, 0) != v}
    if wrong:
        raise SystemExit(
            f"LoRA target census disagrees with this checkpoint: {wrong} (found, expected). "
            f"The list is written out per tower precisely so a rename upstream fails here "
            f"rather than training a smaller model that looks like the intended one."
        )
    print(
        f"LoRA targets: {sum(found.values())} modules across {len(found)} patterns "
        f"{dict(sorted(found.items()))}",
        flush=True,
    )

    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha or 2 * args.lora_rank,
        lora_dropout=args.lora_dropout,
        target_modules=targets,
        # No `exclude_modules` for the visual tower. It was there as a statement of intent
        # and peft warned that it excluded nothing, which is correct and worth writing
        # down instead: the tower's 108 Linears are named `qkv`, `proj`, `linear_fc1` and
        # `linear_fc2`, and peft matches a target as the suffix `.<target>`, so
        # `...mlp.linear_fc1` does not match `fc1`. The census assert below is the real
        # guard and a stronger one -- it fails on a *count*, so a tower projection that
        # ever did match would break the run rather than quietly train.
        bias="none",
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    whole = sum(p.numel() for p in model.parameters())
    print(
        f"LoRA r={config.r} alpha={config.lora_alpha}: {trainable / 1e6:.1f}M trainable of "
        f"{whole / 1e9:.2f}B ({trainable / whole:.3%})",
        flush=True,
    )
    return model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default="/workspace/runs/omni-slurp", type=Path)
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    parser.add_argument("--examples", type=int, default=8000, help="0 for the whole split")
    parser.add_argument("--shots", type=int, default=4, help="must match the evaluation's")
    parser.add_argument("--shot-seed", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--estimator", default="outer_exact")
    parser.add_argument(
        "--bank-shards",
        type=int,
        default=6,
        help=(
            "expert banks are split into this many rotation groups; one group has "
            "requires_grad set per optimizer step. 1 measures every bank every step and "
            "needs the whole 54.0 GiB gradient buffer."
        ),
    )
    banks = parser.add_mutually_exclusive_group()
    banks.add_argument("--measure-expert-banks", dest="measure_expert_banks", action="store_true")
    banks.add_argument(
        "--no-measure-expert-banks", dest="measure_expert_banks", action="store_false"
    )
    parser.set_defaults(measure_expert_banks=None)
    parser.add_argument("--no-four-bit", action="store_true", help="skip the bnb NF4 quantization")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe-load", action="store_true")
    parser.add_argument("--repo", default=str(SCRIPTS.parent))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.probe_load:
        return probe_load(args)

    subset, shots, census = build_examples(
        examples=args.examples,
        shot_count=args.shots,
        shot_seed=args.shot_seed,
        cache_dir=args.cache_dir,
    )
    print(
        f"SLURP train: {census['train_used']} rows of {census['train_pool']} "
        f"({census['shots_excluded']} shot rows excluded), "
        f"{census['distinct_intents']} of {census['menu_size']} intents present, "
        f"rarest {census['rarest_intent']}",
        flush=True,
    )
    if args.dry_run:
        print(json.dumps(census, indent=2, default=str), flush=True)
        return 0

    return _train(args, subset, shots, census)


def _train(
    args: argparse.Namespace, subset: list[Any], shots: list[Any], census: dict[str, Any]
) -> int:
    """Everything from here down needs a GPU, which is why it is behind ``--dry-run``."""
    import torch
    from transformers import AutoProcessor, Trainer, TrainingArguments

    from dynquant import DynQuantCallback
    from dynquant.constants import STATS_FILENAME
    from dynquant.signals.reduce import is_main_rank, world_size

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    processor = AutoProcessor.from_pretrained(args.model)
    model, load_census = load_thinker(
        args.model, four_bit=not args.no_four_bit, device={"": local_rank}, dtype=_dtype()
    )
    model.config.use_cache = False  # incompatible with the backward hooks, unused in training

    freeze_base(model)
    model = attach_lora(model, args)

    measure_banks = args.measure_expert_banks
    if measure_banks is None:
        from run_s2_finetune import resolve_bank_measurement

        measure_banks = resolve_bank_measurement(model, None)

    rotation, bank_names = make_rotation_callback(model, args.bank_shards)
    # DDP builds its bucket set from the parameters that require a gradient when it is
    # constructed -- which is before `on_train_begin`, so the banks are off and excluded
    # already. Naming them anyway, because that exclusion is a consequence of ordering and
    # this is the statement of intent: a bank's gradient is a local measurement, and
    # all-reducing 54.0 GiB of it every step would cost more than the training does.
    model._ddp_params_and_buffers_to_ignore = [
        name for name, param in model.named_parameters() if id(param) in bank_parameters(model)
    ]

    destination = Path(args.out)
    stats_dir = destination / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_file = stats_dir / STATS_FILENAME
    callback = DynQuantCallback(
        stats_dir,
        grad_estimator=args.estimator,
        log_every=50,
        subsample_tokens=256,
        measure_expert_banks=bool(measure_banks),
    )

    dataset = SlurpSft(processor, subset, shots, max_prompt_tokens=args.max_prompt_tokens)

    # The optimizer-step count, needed for warmup and worth printing on its own: it is what
    # the bank rotation is divided over, so `total_steps // bank_shards` is how many
    # gradient samples each expert bank ends the run with.
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        # torchrun's `WORLD_SIZE`, not `world_size()`. The process group is initialized by
        # `TrainingArguments` further down, so `world_size()` still answers 1 up here and
        # the divisor comes out too small -- it printed 1000 steps against the 500 the
        # Trainer then ran, and would have set warmup to 6% of the schedule while calling
        # it 3%. The launcher exports `WORLD_SIZE` before the script starts, so it is the
        # only thing at this point in the program that knows.
        per_rank = args.batch * args.accum * int(os.environ.get("WORLD_SIZE", "1"))
        total_steps = math.ceil(math.ceil(len(dataset) / per_rank) * args.epochs)
    warmup = max(1, round(0.03 * total_steps))
    if is_main_rank():
        print(
            f"schedule: {total_steps} optimizer steps (warmup {warmup}), {args.bank_shards} "
            f"bank shards -> ~{total_steps // args.bank_shards} gradient samples per bank",
            flush=True,
        )

    training_args = TrainingArguments(
        output_dir=str(destination / "trainer"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        # `warmup_ratio` is gone in transformers v5; `warmup_steps` is what survived. The
        # ratio is kept and resolved here rather than dropped, because 3% of a schedule is
        # a different number of steps on the smoke than on the real run.
        warmup_steps=warmup,
        weight_decay=0.0,
        bf16=True,
        # Off deliberately, and not to buy speed: checkpointing replays the forward during
        # backward, so a module's forward hook would fire twice per step while its backward
        # hook fires once, and the saliency EMA would double-count the recomputed
        # activation. The memory problem here is the bank gradient buffer, and the shard
        # rotation is what pays for that.
        gradient_checkpointing=False,
        logging_steps=10,
        save_strategy="steps" if args.save_steps > 0 else "no",
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=3,
        report_to=[],
        seed=args.seed,
        # Each worker decodes waveforms and runs the processor, which is where an audio
        # task's data path actually costs something. Two per rank rather than more: the
        # workers hold decoded clips, and the memory headroom on this box belongs to the
        # bank gradient buffer.
        dataloader_num_workers=2,
        remove_unused_columns=False,
        # False because torch measured it, not because it is the cheap default. The DDP
        # smoke reported "did not find any unused parameters in the forward pass", and it
        # is structural rather than lucky: the only DDP-registered parameters are the 384
        # LoRA adapters, every one of them on the text stack or the audio tower, and every
        # SLURP item exercises both. The visual tower never runs -- which is exactly why
        # it carries no adapter, so it contributes no unused parameter either. The banks
        # are ignored by DDP and are not registered at all.
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,  # type: ignore[arg-type] -- a map-style dataset is enough
        data_collator=lambda rows: collate(rows, dataset.pad_id),
        callbacks=[callback, rotation],
    )

    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started

    if is_main_rank():
        from run_s2_finetune import _git_head, banked_entries_missing

        merged = destination / "adapter"
        model.save_pretrained(str(merged))
        processor.save_pretrained(str(merged))
        tracked = len(callback.tracker) if callback.tracker is not None else 0
        record = {
            **census,
            **load_census,
            "model": args.model,
            "task": "slurp",
            "regime": "qlora" if not args.no_four_bit else "lora",
            "lora_rank": args.lora_rank,
            "estimator": args.estimator,
            "bank_shards": args.bank_shards,
            "bank_tensors": len(bank_names),
            "measure_expert_banks": bool(measure_banks),
            "world_size": world_size(),
            "effective_batch": args.batch * args.accum * world_size(),
            "steps": result.global_step,
            "train_loss": result.training_loss,
            "seconds": round(elapsed, 1),
            "tracked_modules": tracked,
            "stats_file": str(stats_file),
            "output": str(merged),
            "commit": _git_head(Path(args.repo)),
        }
        (destination / "s2_omni_finetune.json").write_text(
            json.dumps(record, indent=2, default=str), encoding="utf-8"
        )
        print(f"-> wrote {destination / 's2_omni_finetune.json'}", flush=True)
        if tracked == 0:
            print("NO MODULES TRACKED: the signal map is empty", flush=True)
            return 4
        if measure_banks:
            missing = banked_entries_missing(model, stats_file)
            if missing:
                print(
                    f"EXPERT BANKS REQUESTED BUT NOT IN THE SIGNAL MAP: {len(missing)} "
                    f"tensors absent, first few {missing[:4]}. The allocator would score "
                    f"91.40% of this checkpoint neutrally and set its widths from role "
                    f"floors.",
                    flush=True,
                )
                return 6
        print(f"-> signal map: {tracked} modules at {stats_file}", flush=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # The stats reduce is the last collective, and it has already run inside the
        # callback's final flush. Tearing the group down explicitly so a rank that exits
        # early is a visible failure rather than a leaked NCCL communicator and a hang.
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
