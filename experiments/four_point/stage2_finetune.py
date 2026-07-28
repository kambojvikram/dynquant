"""Measurement point 2, and the source of the signal map: supervised fine-tuning.

Full fine-tuning or LoRA, chosen by ``--lora-rank`` and by what fits.

The plasticity signal is the variance of ``‖∇W‖`` over optimizer steps, where ``W`` is
the tensor that will later be quantized. Under LoRA those tensors are frozen, so the
signal has to come from the backward-pass gradient of the layer *output* -- which is
what ``outer_exact`` does: the forward hook stashes ``x``, the full backward hook gives
``δ = ∇_Y L``, and Eq. (1) ``∇W = δxᵀ`` recovers the base-weight gradient exactly. Not
an approximation of it: ``y = W x + s·BAx`` means ``∂L/∂y`` is the same tensor whether
or not an adapter is attached, so the reconstructed ``∇W`` is the gradient the base
weight would have received. The regime therefore changes what the model *learns*, not
how faithfully the signal is measured.

Which regime to use is a memory question. A 1.9B model full-tunes inside 80GB with room
to spare, so the Qwen run did that. A 7B does not: 14.5GB of bf16 parameters, another
14.5GB of gradients, and ~58GB of fp32 AdamW moments is ~87GB before a single
activation. LoRA replaces the last two terms with a few hundred MB.

The loss is masked to the completion. Training on the question tokens too would
spend capacity modelling the prompt distribution, and would put question text into
the very gradient signal that is supposed to be measuring which weights matter for
*solving*. On CaseHOLD that masking is not a refinement but the whole run: the
completion is a single digit after ~400 tokens of case-law prose, so unmasked it would
carry under 1% of the gradient. Banking77 is more lopsided still -- one index token
after a 77-line taxonomy. :meth:`tasks.Task.training_row` owns the details.

Usage::

    python stage2_finetune.py                     # full fine-tune
    python stage2_finetune.py --lora-rank 32      # LoRA, merged before saving
    python stage2_finetune.py --max-steps 20      # smoke run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from common import DTYPE, MODEL_ID, RUN_DIR, TASK, load_task, load_tokenizer, record, set_seed

# Full fine-tuning moves every weight, so it needs the smaller step; LoRA updates a
# low-rank residual through a fixed scaling and needs roughly an order of magnitude
# more to move the model at all. Resolved from the regime rather than defaulted to one
# value, because 1e-5 on a rank-32 adapter trains to a flat result that looks like the
# task having no headroom -- which is the exact conclusion this experiment exists to
# draw, and the one it must not draw by accident.
LR_BY_REGIME = {"full fine-tune": 1e-5, "lora": 1e-4}


class SftDataset(torch.utils.data.Dataset):
    """Tokenized (prompt, completion) pairs with the prompt masked out of the loss.

    How an overlong example is handled is the task's decision, not this class's --
    GSM8K drops it, CaseHOLD cuts the prompt at the front -- because getting it wrong
    is silent either way. Whatever the task chose, the count is printed: a run that
    quietly trained on 60% of its data would otherwise look like a run that trained on
    all of it and learned less.
    """

    def __init__(self, examples, tokenizer) -> None:
        self.rows = []
        dropped = 0
        for example in examples:
            row = TASK.training_row(example, tokenizer)
            if row is None:
                dropped += 1
                continue
            self.rows.append(row)
        if dropped:
            print(
                f"dropped {dropped}/{len(examples)} examples that did not fit "
                f"{TASK.train_max_len} tokens",
                flush=True,
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def collate(batch, pad_id: int):
    """Right-pad. Training reads the whole sequence at once, so unlike generation
    the padding side is free -- and right padding keeps label alignment obvious."""
    width = max(len(row["input_ids"]) for row in batch)
    input_ids, labels, mask = [], [], []
    for row in batch:
        pad = width - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_id] * pad)
        labels.append(row["labels"] + [-100] * pad)
        mask.append([1] * len(row["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }


def attach_lora(model, args) -> object:
    """Wrap the model in a LoRA adapter and report what is now trainable.

    ``all-linear`` rather than a hand-written list of projection names. The name lists
    differ per architecture -- ``gate_up_proj`` here, ``w1``/``w3`` there -- and a list
    that silently misses a family trains fewer modules than intended while looking like
    it worked. PEFT resolves the set from the module tree and excludes the output head,
    which is where a merged adapter would be least welcome anyway.
    """
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=args.lora_rank,
        # Twice the rank unless overridden: the update is scaled by alpha/r, so tying
        # them keeps the effective step size fixed as the rank is varied.
        lora_alpha=args.lora_alpha or 2 * args.lora_rank,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )
    wrapped = get_peft_model(model, config)
    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapped.parameters())
    print(
        f"LoRA r={config.r} alpha={config.lora_alpha}: "
        f"{trainable / 1e6:.1f}M trainable of {total / 1e9:.2f}B ({trainable / total:.2%})",
        flush=True,
    )
    return wrapped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=None, help="default depends on the regime")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--estimator", default="outer_exact")
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=0,
        help="0 fine-tunes every weight; >0 trains a LoRA adapter and merges it before saving",
    )
    parser.add_argument("--lora-alpha", type=int, default=0, help="default 2x rank")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--out", default=str(RUN_DIR / "finetuned"))
    args = parser.parse_args()

    regime = "lora" if args.lora_rank > 0 else "full fine-tune"
    lr = args.lr if args.lr is not None else LR_BY_REGIME[regime]

    set_seed()
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    from dynquant import DynQuantCallback
    from dynquant.integration.peft_utils import merge_adapters

    train, _, _ = load_task()
    tokenizer = load_tokenizer()
    dataset = SftDataset(train, tokenizer)
    print(f"training on {len(dataset)} {TASK.key} examples", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map="cuda")
    model.config.use_cache = False  # incompatible with the backward hooks and unused in training
    print(f"{MODEL_ID}: {regime}, lr {lr:g}", flush=True)
    if regime == "lora":
        model = attach_lora(model, args)

    stats_dir = RUN_DIR / "stats"
    callback = DynQuantCallback(
        stats_dir,
        grad_estimator=args.estimator,
        log_every=50,
        subsample_tokens=256,
    )

    training_args = TrainingArguments(
        output_dir=str(RUN_DIR / "trainer"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        # Off deliberately: checkpointing replays the forward pass during backward,
        # so a module's forward hook fires twice per step while its backward hook
        # fires once. The stashed activation would then be the recomputed one and
        # the saliency EMA would double-count. Memory is not the constraint here.
        gradient_checkpointing=False,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=0,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
        callbacks=[callback],
    )

    started = time.time()
    train_result = trainer.train()
    elapsed = time.time() - started

    out = Path(args.out)
    # Merged, not saved as an adapter. Every downstream stage loads this directory as a
    # plain causal LM: stage 3 evaluates it, stage 5 quantizes it. Saving adapter
    # weights would make stage 3 silently score the *base* model -- an arm that looks
    # measured, sits in the table as "fine-tuned", and is not.
    saved = merge_adapters(model) if regime == "lora" else model
    saved.config.use_cache = True
    saved.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"\nsaved fine-tuned model to {out}", flush=True)

    tracked = len(callback.tracker) if callback.tracker is not None else 0
    record(
        "stage2_finetune",
        {
            "model": MODEL_ID,
            "task": TASK.key,
            "regime": regime,
            "lora_rank": args.lora_rank,
            "estimator": args.estimator,
            "examples": len(dataset),
            "epochs": args.epochs,
            "lr": lr,
            "effective_batch": args.batch * args.accum,
            "steps": train_result.global_step,
            "train_loss": train_result.training_loss,
            "seconds": round(elapsed, 1),
            "tracked_modules": tracked,
            "stats_dir": str(stats_dir),
            "output": str(out),
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
