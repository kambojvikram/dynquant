#!/usr/bin/env python3
"""P2 gate: does a real LoRA fine-tune emit a stats file keyed by *base* weight names?

The unit tests cover ``canonical_name`` against handwritten PEFT strings and the tracker
against a stand-in wrapper shaped like ``peft.tuners.lora.Linear``. Neither can catch the
thing this script exists for: a real ``get_peft_model`` produces a module tree the stand-in
does not model. It inserts a ``LoraModel`` above the whole network, so every name gains a
``base_model.model.`` prefix; it demotes the original ``Linear`` to ``.base_layer``; it adds
``lora_dropout`` and ``lora_embedding_A/B`` ``ModuleDict``s alongside the factors. Bug 3 in
``docs/legacy-audit.md`` is precisely a read-time guess about those names, and bug 7 is a
gradient read that lands on ``lora_A.weight`` instead of the weight being quantized.

So the assertion is not "the keys look tidy". It is **every key resolves to a module of the
*unwrapped* model**, checked with ``get_submodule`` against a separately constructed base
model. That is what "canonical" has to mean for the stats file to be usable by the
quantizer, which sees a merged checkpoint with no adapters in it at all.

It runs through ``Trainer`` and ``DynQuantCallback`` rather than ``track_signals``, because
the callback is the advertised integration and the ordering it exists to fix -- Welford
folding on ``on_pre_optimizer_step``, not inside the gradient hook -- is only exercised by a
real trainer. A synthetic random-token dataset is enough; nothing here measures quality.

One module legitimately carries no plasticity signal, and the gate names it rather than
tolerating a count: the input embedding. Its weight is frozen under LoRA and its input is
integer, so its output has ``requires_grad=False``, and autograd never computes a gradient
with respect to it because nothing upstream of it needs one. That is not the tracker missing
something -- ``lm_head`` is frozen under LoRA too and keeps its signal, because its *input*
requires grad, which is exactly what bug 7's ``outer_exact`` fix exists to do. Downstream the
embedding lands in ``coverage().partial_signal`` and in ``score``'s ``unexercised``, where it
takes ``NEUTRAL_RANK`` on *both* axes -- so its measured saliency is discarded along with the
plasticity it never had. Whether that is the right scoring rule is a P4 question; that it
happens is a property of LoRA, and letting it pass as an unexplained "197 of 198" would hide
it.

Usage::

    python scripts/gate_lora_stats.py --model Qwen/Qwen3-0.6B --steps 50
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


class _RandomTokens:
    """``steps * batch`` sequences of uniform token ids, labelled for causal LM.

    A ``datasets.Dataset`` would pull in another dependency to no purpose: the gate is
    about names and counts, and uniform noise exercises every module exactly as real text
    would. Returns lists rather than tensors so the default data collator handles it.
    """

    def __init__(self, count: int, seq_len: int, vocab: int, seed: int = 0) -> None:
        import torch

        generator = torch.Generator().manual_seed(seed)
        self.rows = [
            torch.randint(0, vocab, (seq_len,), generator=generator).tolist() for _ in range(count)
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        ids = self.rows[index]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": ids}


def _base_module_index(model_name: str) -> tuple[set[str], str | None]:
    """Every module name in the *unwrapped* model, plus the input embedding's name.

    Meta init because this exists only to answer "is this name real?" -- no weights are
    read, so a 14B model costs nothing here. The embedding name comes back alongside
    because it is the one module allowed to carry no gradient signal, and spelling
    ``model.embed_tokens`` into the exemption would make it Qwen-specific.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_name)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    names = {name for name, _ in model.named_modules() if name}
    embedding = model.get_input_embeddings()
    found = next((name for name, mod in model.named_modules() if name and mod is embedding), None)
    return names, found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--estimator", default="outer_exact")
    parser.add_argument("--out", default=None, help="where to keep the stats file")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

    from dynquant import DynQuantCallback
    from dynquant.constants import STATS_FILENAME
    from dynquant.signals import load_stats

    out_dir = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="dynquant-gate-"))
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model.config.use_cache = False
    model = get_peft_model(
        model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, target_modules="all-linear")
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{args.model}  lora r={args.rank}  {trainable / 1e6:.1f}M trainable", flush=True)

    # Sanity-check the premise before spending the run: if the wrapper did not rename
    # anything, this script proves nothing and should say so rather than pass.
    wrapped = {name for name, _ in model.named_modules()}
    assert any(".base_layer" in name for name in wrapped), "get_peft_model did not wrap anything"

    dataset = _RandomTokens(
        args.steps * args.batch_size * args.grad_accum,
        args.seq_len,
        int(model.config.vocab_size),
    )
    callback = DynQuantCallback(out_dir, grad_estimator=args.estimator)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(out_dir / "trainer"),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.steps,
            learning_rate=1e-5,
            logging_steps=max(args.steps // 5, 1),
            report_to=[],
            save_strategy="no",
            bf16=torch.cuda.is_available(),
        ),
        train_dataset=dataset,  # type: ignore[arg-type]
        callbacks=[callback],
    )
    trainer.train()

    stats_path = out_dir / STATS_FILENAME
    if not stats_path.exists():
        print(f"FAIL: no stats file at {stats_path}", file=sys.stderr)
        return 1
    stats = load_stats(stats_path)
    print(f"\n{stats_path}  {len(stats.layers)} layers, {stats_path.stat().st_size / 1e3:.0f} kB")

    failures: list[str] = []

    # 1. Nothing from the adapter vocabulary may appear in a key.
    adapter_marks = ("base_model", "base_layer", "lora_A", "lora_B", "lora_embedding", ".default")
    leaked = sorted(n for n in stats.layers if any(m in n for m in adapter_marks))
    if leaked:
        failures.append(f"{len(leaked)} keys carry adapter names, first: {leaked[:3]}")

    # 2. The real test: every key names a module of the unwrapped model. A key that merely
    #    *looks* canonical but points nowhere is exactly the read-time guessing bug 3
    #    describes, moved to write time.
    real, embedding = _base_module_index(args.model)
    unresolvable = sorted(n for n in stats.layers if n not in real)
    if unresolvable:
        failures.append(
            f"{len(unresolvable)} keys do not name a module of the base model, "
            f"first: {unresolvable[:3]}"
        )

    # 3. Signals actually populated. An empty-but-well-named file passes 1 and 2.
    calls = {n: layer.forward_calls for n, layer in stats.layers.items()}
    micro = args.steps * args.grad_accum
    wrong_calls = sorted(n for n, c in calls.items() if c not in (0, micro))
    if wrong_calls:
        failures.append(
            f"{len(wrong_calls)} layers saw neither 0 nor {micro} forward calls, "
            f"first: {[(n, calls[n]) for n in wrong_calls[:3]]}"
        )
    exercised = [layer for layer in stats.layers.values() if layer.forward_calls]
    if not exercised:
        failures.append("no layer was exercised at all")
    flat = sorted(
        name
        for name, layer in stats.layers.items()
        if layer.forward_calls and not layer.activation_rms_ema
    )
    if flat:
        failures.append(f"{len(flat)} exercised layers have no saliency, first: {flat[:3]}")

    # 4. Plasticity is the half that bug 7 destroyed, and it is measured on the frozen base
    #    weight -- so grad_norm_count must track *optimizer* steps, not micro-batches, and
    #    the variance must be non-zero on a run where the weights actually moved.
    graded = [layer for layer in exercised if layer.grad_norm_count]
    if not graded:
        failures.append("no layer carries a gradient signal")
    # Which modules lack plasticity, not how many. Measured both ways on Qwen3-0.6B: full
    # fine-tuning gives the embedding grad_norm_count == steps and leaves this set empty; LoRA
    # gives 0 with the input embedding alone in it, for the requires_grad reason in the module
    # docstring. Everything else frozen under LoRA -- lm_head included -- keeps its signal, so
    # any other name here is outer_exact regressing rather than a property of LoRA, and a
    # "197 of 198 have gradients" count would not tell the two apart.
    ungraded = sorted(
        name
        for name, layer in stats.layers.items()
        if layer.forward_calls and not layer.grad_norm_count
    )
    unexpected = [name for name in ungraded if name != embedding]
    if unexpected:
        failures.append(
            f"{len(unexpected)} exercised layers carry no gradient signal and are not the "
            f"input embedding ({embedding}): {unexpected[:3]}"
        )
    wrong_steps = sorted(
        name
        for name, layer in stats.layers.items()
        if layer.grad_norm_count and layer.grad_norm_count != args.steps
    )
    if wrong_steps:
        failures.append(
            f"{len(wrong_steps)} layers folded a count other than {args.steps} optimizer "
            f"steps, first: {[(n, stats.layers[n].grad_norm_count) for n in wrong_steps[:3]]}"
        )
    if not any(layer.grad_norm_var > 0 for layer in graded):
        failures.append("every gradient variance is zero")

    print(f"  exercised {len(exercised)}/{len(stats.layers)}, with gradients {len(graded)}")
    if ungraded:
        print(
            f"  no plasticity: {', '.join(ungraded)} -- frozen weight, integer input, so the "
            f"output never requires grad (expected under LoRA; NEUTRAL_RANK downstream)"
        )
    print(
        f"  estimator {stats.provenance.grad_estimator}, world_size {stats.provenance.world_size}"
    )
    notes: dict[str, Any] = dict(stats.provenance.notes)
    print(f"  notes: {sorted(notes)}")
    sample = next(iter(sorted(n for n in stats.layers if n.endswith("self_attn.q_proj"))), None)
    if sample:
        layer = stats.layers[sample]
        print(
            f"  {sample}: forward_calls={layer.forward_calls} "
            f"grad_norm_count={layer.grad_norm_count} rms={layer.activation_rms_ema:.4g} "
            f"var={layer.grad_norm_var:.4g}"
        )

    print()
    if failures:
        for line in failures:
            print(f"FAIL: {line}")
        return 1
    print(
        f"PASS: {args.steps}-step LoRA r={args.rank} run on {args.model} emitted "
        f"{len(stats.layers)} canonical keys, every one a module of the unwrapped model"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
