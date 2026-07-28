"""Measurement point 2, and the source of the signal map: supervised fine-tuning.

Full fine-tuning, not LoRA, for a reason specific to what this run is measuring.
The plasticity signal is the variance of ``‖∇W‖`` over optimizer steps, where ``W``
is the tensor that will later be quantized. Under LoRA those tensors are frozen, so
the signal has to be *estimated* from the backward-pass gradient of the layer
output -- which is what ``outer_exact`` exists to do, and it is the right answer for
users who can only afford LoRA. But a 1.9B model fits in an 80GB card with room to
spare, so this run can measure the real thing instead, and the bit map is then
derived from gradients of the actual weights rather than from a proxy for them.

The loss is masked to the completion. Training on the question tokens too would
spend capacity modelling the prompt distribution, and would put question text into
the very gradient signal that is supposed to be measuring which weights matter for
*solving*. On CaseHOLD that masking is not a refinement but the whole run: the
completion is a single digit after ~400 tokens of case-law prose, so unmasked it would
carry under 1% of the gradient. :meth:`tasks.Task.training_row` owns the details.

Usage::

    python stage2_finetune.py                  # full run
    python stage2_finetune.py --max-steps 20   # smoke run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from common import DTYPE, MODEL_ID, RUN_DIR, TASK, load_task, load_tokenizer, record, set_seed


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--estimator", default="outer_exact")
    parser.add_argument("--out", default=str(RUN_DIR / "finetuned"))
    args = parser.parse_args()

    set_seed()
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    from dynquant import DynQuantCallback

    train, _, _ = load_task()
    tokenizer = load_tokenizer()
    dataset = SftDataset(train, tokenizer)
    print(f"training on {len(dataset)} {TASK.key} examples", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE, device_map="cuda")
    model.config.use_cache = False  # incompatible with the backward hooks and unused in training

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
        learning_rate=args.lr,
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
    model.config.use_cache = True
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"\nsaved fine-tuned model to {out}", flush=True)

    tracked = len(callback.tracker) if callback.tracker is not None else 0
    record(
        "stage2_finetune",
        {
            "model": MODEL_ID,
            "task": TASK.key,
            "regime": "full fine-tune",
            "estimator": args.estimator,
            "examples": len(dataset),
            "epochs": args.epochs,
            "lr": args.lr,
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
