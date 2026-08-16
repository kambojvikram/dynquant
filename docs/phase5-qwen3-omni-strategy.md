# Phase 5 — Qwen3-Omni-30B-A3B: build strategy

## What this campaign is for

Every DynQuant number to date comes from a **text** model whose quantizable mass is
`nn.Linear`. Phase 4 reached an MoE, but the question this phase asks is different and
sharper:

> When 91.4% of a checkpoint lives in batched 3-D expert banks that no LoRA updates, no
> `bitsandbytes` quantizes, and no `all-linear` target list sees — does the signal still
> get measured, and does the allocation still pay?

`Qwen/Qwen3-Omni-30B-A3B-Instruct` is the hardest available instance of that question, and
it adds a second one for free: it is multimodal, so the checkpoint contains towers the
task never runs. Those are the two things this phase measures. The task is **SLURP intent
classification from speech** — audio in, an index out, scored by exact match, which keeps
the metric identical in kind to phases 1–3 so the comparison is about the architecture
rather than about the scoring.

**Scope, as set:** DynQuant at 3 and 4 bits. No GPTQ / AWQ / RTN arms — descoped
deliberately, and the consequence is stated in §7.

**Status: complete.** The signal does get measured on the banks — all 654 modules tracked,
the 96 batched banks included, via the masked per-entry Welford of §2. The allocation holds
at 4 bits and does not at 3: `dq4` is 86.20% against the bf16 fine-tune's 86.80% at 4.00x
fewer bytes and the pair does not separate (p=0.70), while `dq3` falls to 25.00% at 5.33x
(p=1.6e-88). That 3-bit budget sits **below this architecture's own floor budget** — the
floors alone cost 3.418 average bits — so it is not a test of the ordering but of what
happens when 400 modules are forced through their floors at once. Whether the ordering beats
a uniform allocator at matched bytes is a question no arm here answers, because the arms that
would answer it were descoped. Full record:
[`docs/reports/phase5-omni-slurp.md`](reports/phase5-omni-slurp.md). The plan below is kept
as written, with each stage's outcome recorded against it.

---

## 1. What the architecture is, measured

All figures below are measured on the checkpoint, not read off a model card.

| | params | share |
|---|---|---|
| Thinker, total | 31.719 B | — |
| ├─ 96 batched expert banks (48 layers × `gate_up_proj`, `down_proj`) | 28.991 B | **91.40%** |
| ├─ visual tower (27 blocks + mergers) | 0.538 B | 1.70% |
| └─ everything else (attention, routers, audio tower, embed, lm_head) | 2.190 B | 6.90% |
| Talker + code2wav — **not written to the S3 checkpoint** | 3.541 B | — |

Three facts here drive every decision in this document:

1. **The banks are the model.** A pipeline that handles them approximately is a pipeline
   that handles Qwen3-Omni approximately.
2. **The banks are not `nn.Linear`.** They are bare `nn.Parameter` of rank 3, so every
   tool that enumerates Linears — `bitsandbytes`, `all-linear`, most role classifiers —
   walks straight past them.
3. **The checkpoint's layout is not the module graph.** The index lists the experts
   per-expert and unfused (19,743 Thinker keys against the Thinker module's 1,407);
   `from_pretrained` is what fuses gate with up and stacks 128 experts into each bank.
   Reading the index instead of the live model inverts every count above.

---

## 2. Build strategy: what changes, and where

The governing constraint was *"without breaking anything in dynquant"*. The result:

**No change to `dynquant-core` was needed, and none was made.** Everything model-specific
lives in two scripts under `scripts/`. That is the headline of this section, and it was
not a foregone conclusion — it survived exactly because of two pieces of core behaviour
that were written for other reasons:

| core behaviour | written for | what it bought here |
|---|---|---|
| `_collect_bank_grads` skips entries whose `.grad is None` | robustness | lets a bank be *absent* from a step rather than wrong |
| the step-end Welford update masks by `active = pending > 0` | correctness under uneven routing | an unobserved entry gets **no sample**, not a zero |

Those two together are what make §3's rotation free.

### The two scripts

| script | role |
|---|---|
| `scripts/run_s2_omni_finetune.py` | QLoRA SFT + signal capture. Loads the Thinker, attaches LoRA to an asserted target census, rotates the bank gradient shards, writes the signal map. |
| `scripts/run_s3_omni_merge.py` | Folds the adapter into a fresh **bf16** Thinker and writes a Thinker-only checkpoint, verifying that exactly the targeted weights moved. |

Downstream — `quantize`, `inspect`, `eval`, `export` — is used unmodified. All four already
accept `--model-class`, which is the entire reason a Thinker-only checkpoint is viable:

```
--model-class Qwen3OmniMoeThinkerForConditionalGeneration
```

### Why the checkpoint is Thinker-only

The Talker and code2wav are 3.541 B parameters that SLURP never executes, so they receive
no gradient and carry no signal. Writing them out would hand the allocator 3.5 B of
parameters priced by role floors alone **and put them in the denominator of every
average-bits figure the campaign reports** — a 10% dilution of the headline number, earned
by quantizing weights nothing measured and nothing evaluates.

---

## 3. The blocker, and why the fix cost nothing

`--measure-expert-banks` reads as a boolean and behaves as a second copy of the model.
`_collect_bank_grads` runs at `on_pre_optimizer_step`, so every bank's `.grad` is live from
the moment autograd writes it until the collector releases it — all 96 coexist at the end
of every backward:

| | bf16 |
|---|---|
| Thinker weights | 59.1 GiB |
| **gradient buffer, all 96 banks at once** | **54.0 GiB** |
| total | 113 GiB against a 96 GiB card |

QLoRA does not rescue this — `bitsandbytes` shrinks ~2.1 B of projections and leaves all
28.991 B of banks in bf16. Nor does a second GPU: DDP replicates the buffer per rank.

**Fix: rotate the banks over steps.** Shard by layer index, enable `requires_grad` on one
shard per optimizer step (`global_step % N`, identical on every rank). At N=6 the peak
buffer is 9.0 GiB and each bank is sampled every 6th step. Plasticity stays a variance
over optimizer steps — Appendix H's definition is untouched; only the sample count per
bank changes, and the stats file records that count itself.

At the campaign's 500 steps that is **~83 gradient samples per bank**, against 500 for
everything else. This is the one place where the omni pipeline measures something less
finely than a text run does, and it is recorded rather than hidden.

---

## 4. What had to be replaced, and what it cost to find out

Both of these were found by running, not by reading.

### `prepare_model_for_kbit_training` cannot be used

The standard QLoRA preamble upcasts *every* fp16/bf16 parameter to fp32, skipping only
`Params4bit`. The banks are neither — 28.991 B of them, 108 GiB of fp32 against a 96 GiB
card. The first smoke died inside that loop at 82.7 GiB allocated.

It is replaced by `freeze_base()`, which does the one load-bearing thing (freeze) and
deliberately omits the upcast, because:

- the tensors the optimizer steps are fp32 anyway — `get_peft_model` defaults to
  `autocast_adapter_dtype=True`;
- upcasting the norms would be *wrong* here, not merely unaffordable: an RMSNorm returns
  `weight * hidden.to(input_dtype)`, so an fp32 weight makes the hidden state fp32, and
  the next thing the text stack does with it is a batched matmul against a bf16 bank —
  which does not promote, it raises.

### `warmup_ratio` is gone in transformers v5

Only `warmup_steps` survived. Worth flagging beyond this script: **`scripts/run_s2_finetune.py`
still passes `warmup_ratio` and therefore cannot run on a transformers-5 box.** Not fixed
here — it is out of this campaign's scope — but it is a live incompatibility, not a
cosmetic one.

Resolving the ratio into steps then exposed a second, quieter bug: `world_size()` reads the
process group, which `TrainingArguments` has not yet initialized at that point in the
script, so it answered 1 and the run announced 1000 steps against the 500 the Trainer
actually ran. `WORLD_SIZE` from the launcher is the only thing that knows at that point.
The number was wrong in the log, not in the training — but a log is what gets quoted.

---

## 5. Verification gates, and what each one caught

| gate | what it asserts | status |
|---|---|---|
| LoRA target census | exactly 384 modules over 7 patterns, per-pattern counts asserted (`q_proj` = 48 text + 32 audio) | green |
| load probe | 0 parameters left on meta, banks bf16 on one device, per-shard buffer 9.0 GiB | green |
| 1-GPU smoke | collator, tracker, rotation, finite loss | green — 654 modules |
| 2-GPU DDP smoke | cross-rank Welford merge | green — counts exactly double (12 vs 6; banks 2 vs 1) |
| `banked_entries_missing` | every bank present in the signal map | green |
| role classification, on meta | 654 modules; all 96 banks typed `moe.expert.*`, none refused; stats keys == graph keys exactly, 0 role disagreements | green |
| merge verification | exactly the targeted weights moved; all 96 banks unchanged | green — 384/384 moved, 96 banks bit-identical |

### The gate that failed a correct merge

Worth recording, because the gate was the thing that was wrong. S3 first aborted with *"MERGE
LEFT 9 OF 384 TARGETED WEIGHTS UNCHANGED"*, all nine in the audio tower. They had not been
left unchanged. The fingerprint compared each parameter's **L2 norm**, and a norm difference
is not a distance: `||W+D|| - ||W||` is `<W,D>/||W||` to first order, so a delta near-
orthogonal to the base weight — which a LoRA delta fitted against an NF4 base and folded into
the bf16 one is, `cos(W,D) ≈ 1e-3` — shows up only at second order.

Measured offline in float64 over all 384 targets, before changing anything:

| quantity | range over the 384 |
|---|---|
| `‖D‖/‖W‖` | 2.7e-3 .. 1.2e-2 — every target had a real delta |
| share of stored bf16 elements changed | 48% .. 91% |
| relative **norm** change | 7.7e-7 .. 1.2e-4 — one continuous band |

The 1e-6 cut fell inside that band: passing minimum 1.073e-6 against failing maximum
9.888e-7. There was no gap to threshold, and nothing in the output said so. Compounding it,
the norms were reduced in fp32, whose error on a 6.5 M-element tensor is ~1e-4 relative —
a hundred times the signal, and enough that fp32 reported `-1.40e-04` where float64 says
`+7.66e-07`.

The fix is to compare values rather than a summary of them: a 4096-element sample at a
name-seeded index, tested with `!=`. No threshold, no reduction, kilobytes of state, and the
separation becomes categorical — **48.0%..90.9% of the sample changed on every merged weight,
and exactly 0 elsewhere**. That second number is the one that matters most, because the
assertion a tolerance quietly hollows out is "no expert bank moved". Seven guards in
`tests/test_run_s3_omni_merge.py` pin the geometry.

One operational note from the same run: the merge is done on **CPU**. It needs no GPU — 384
matmuls of a rank-16 factor pair — but `save_pretrained` under transformers v5 calls
`revert_weight_conversion` to un-fuse what it fused at load, allocating `torch.chunk(...)
.contiguous()` copies beside the 63.4 GiB model. On a 95 GiB card that OOMs at 93.6 GiB
allocated; with 488 GiB of host RAM it is unremarkable, and the whole write takes 48 s.

### Signal coverage

**97.28% of parameters are priced on measured gradient moments** — 531 of 654 modules,
30.854 B of 31.718 B, read off the live run's step-200 checkpoint. The residue is accounted
for rather than assumed:

| not gradient-measured | params | why |
|---|---|---|
| visual tower | 0.538 B | never executed on an audio task — recorded in `provenance.notes.unexercised_modules` |
| `embed_tokens` | 0.311 B | an `Embedding`, not a Linear; no outer-product estimate |
| audio `conv2d1..3`, `conv_out` | 0.014 B | convolutions, role `other` |

That the visual tower is *listed by name* in the stats provenance is the property that
matters. Its zero saliency is absence of evidence, and the allocator is told so.

**That percentage has to be computed against the model, not against the stats file.** Under
4-bit QLoRA the two disagree: `bitsandbytes` stores a `Linear4bit` weight packed two values
to a byte and *flat*, so a 4096x2048 projection has `weight.shape == (4194304, 1)`, and the
tracker's `param_count` — `weight.numel()` — came back at exactly half. Measured here: 503 of
654 modules at ratio 0.5, 151 at 1.0, splitting perfectly along "did bnb replace this
module", and a stats-file total of 30.677 B against the model's real 31.718 B.

The split is what makes it worth a fix rather than a footnote. The halved modules are
attention, the dense MLPs and the vision MLPs; the intact ones are the embedding, the LM
head, the 48 routers, the 96 expert banks and — the five that complete the 151 —
`visual.pos_embed` and the four 4-D convolutions, none of which is a `Linear` for
`bitsandbytes` to replace. Anything sizing modules from that field sees
attention at half price against experts at full price. Nothing on this campaign's path does
— `allocate_bits` prices from `ModelGraph`, off the real shapes — and
`dynquant._legacy.allocator`, which *does* allocate from `param_counts`, is fed shape-derived
counts by its golden test. The exposure is that handing that allocator this file's counts is
the obvious thing to do. `_logical_numel` now prefers a module's declared
`out_features × in_features` over its storage, with the coverage figure above recomputed on
the corrected denominator: it was 98.07% on the biased one.

---

## 6. What the allocator sees, measured before the checkpoint exists

Role inference reads shapes and config, never values, so a **meta-device** Thinker answers
it exactly and for free. Running it before S3 is what turns "the allocator should handle
batched banks" into a fact:

| role | modules | params | share | floor |
|---|---|---|---|---|
| `moe.expert.gate_up` | 48 | 19.327 B | 60.93% | 4 |
| `moe.expert.down` | 48 | 9.664 B | 30.47% | 2 |
| `attn.q` / `attn.o` | 80 / 80 | 0.455 B each | 1.43% each | 4 |
| `lm_head` | 1 | 0.311 B | 0.98% | 8 |
| `embedding` | 1 | 0.311 B | 0.98% | 4 |
| `other` | 70 | 0.289 B | 0.91% | 4 |
| `vision.mlp` | 54 | 0.268 B | 0.84% | 4 |
| `mlp.up` / `mlp.down` | 32 / 32 | 0.210 B each | 0.66% each | 3 |
| `attn.k` / `attn.v` | 80 / 80 | 0.103 B each | 0.32% each | 4 |
| `moe.router` | 48 | 0.013 B | 0.04% | **8, structural** |

All 96 batched banks are classified, none refused. And the signal map's 654 keys are
**exactly** the graph's 654 keys, with zero role disagreements between the map written
during training and a fresh classification — so nothing downstream is matching by luck.

### The two arms are two different regimes

**The floors alone cost 3.418 average bits.** That single number decides what each arm is
actually testing, and they are not the same experiment:

- **dq4** — floors fit inside the budget with 0.582 bits of headroom. The score spends that
  headroom: it decides *what gets upgraded*.
- **dq3** — floors are unreachable. Soft floors bind, the allocator downgrades by lowest ROI
  and reports every breached role. The score decides *what gets cut*.

Both are score-driven, which is the property that matters (a hard-floor allocator would
have returned the floor map at 3.0 and the score would have had no effect at all). But a
result at 3 bits and a result at 4 bits are answering different questions here, and the
write-up has to say which.

### The allocation itself

`classify_model` -> `score_modules` -> `allocate_bits` needs shapes, a stats file and a
budget — no weights. So the whole S4 path was first run on the meta model against the
**smoke** stats (6 steps, so the *scores* were noise) purely to see what the allocator does
with this architecture, and then re-run for real against the final 500-step signal file.
bf16 for reference is 59.1 GiB. The numbers below are the **real** run:

| | 3.0 bits | 4.0 bits |
|---|---|---|
| achieved | 2.999976 | 3.999945 |
| size | 11.08 GiB (5.33x) | 14.77 GiB (4.00x) |
| modules per width | 125 @ 2, 368 @ 3, 109 @ 4, 48 @ 8 | 9 @ 2, 67 @ 3, 524 @ 4, 50 @ 8 |
| params per width | 13.596 B @ 2, 12.596 @ 3, 5.507 @ 4, 0.013 @ 8 | 1.812 B @ 2, 5.698 @ 3, 23.877 @ 4, 0.325 @ 8 |
| floor violations | **400** | **0** |
| within-role width/score concordance | 1.0000 (8791 pairs) | 0.9943 (1234 pairs) |

The **byte totals reproduce the smoke-stats preview exactly** — 11 894 607 776 and
15 859 386 016 — and the histograms do not. That is the right way round: the budget is a
property of the architecture and the floors, which the preview had exactly, while *which*
module gets which width is a property of the signal, which the preview did not have. A
preview that had also reproduced the histogram would have meant the scores were not being
read.

That confirms the two regimes as a measurement rather than an argument. At 4.0 nothing is
breached and the score spends 0.582 bits of headroom — visibly, by lifting `moe.expert.down`
off its 2-bit floor while `gate_up` sits at its own floor of 4. At 3.0 the allocator
breaches **400 of 650** modules and puts **42.9% of the parameters at 2 bits**, and *which*
roles it breaches is the interesting part:

| role | breached | params | moves |
|---|---|---|---|
| `moe.expert.gate_up` | 35 of 48 | 14.093 B | 4->2 x9, 4->3 x26 |
| `attn.o`/`attn.q`/`attn.k`/`attn.v` | 58 of 80 each | 0.419/0.331/0.074/0.067 B | 4->2 x14, 4->3 x44 each |
| `lm_head` | 1 of 1 | 0.311 B | **8->3** |
| `embedding` | 1 of 1 | 0.311 B | 4->3 |
| `other` | 65 of 70 | 0.280 B | 4->3 |
| `vision.mlp` | 54 of 54 | 0.268 B | 4->3 |
| `mlp.up`/`mlp.down` | 6 of 32 each | 0.039 B each | 3->2 |

**The 48 routers are never breached** — `MOE_ROUTER` is in `STRUCTURAL_ROLES`, so the one
tensor that decides which 500 M parameters a token goes through keeps its 8 bits while
everything around it is cut. That is the floor policy doing exactly the job it was written
for, and S5 confirms it from the other side: the 3-bit collapse is not a routing collapse.
`moe.expert.down` is absent from the table for the opposite reason — it is already at its
own 2-bit floor and there is nothing below it to cut to.

The single largest breach is `lm_head`, the only 5-bit drop in the map, and
[the results record](reports/phase5-omni-slurp.md) measures what that width does to that
tensor: top-1 agreement 99.8% at its 8-bit floor, 77.7% at 4, **58.6% at 3**.

Two smaller facts fell out. `score_modules` covers **650 of 654** modules; the four it does
not are `audio_tower.conv2d1..3` and `visual.patch_embed.proj`, all 4-D convolutions, which
therefore score zero and are cut first — correct, and 5.92 M parameters. And 119 modules
(0.858 B) are reported `unexercised`, which is the visual tower plus `embed_tokens` plus
`audio_tower.conv_out`, matching §5's coverage table from the other direction.

`embedding` and `lm_head` are **untied** here — `tie_word_embeddings: False`, and the
checkpoint index lists `thinker.lm_head.weight` and `thinker.model.embed_tokens.weight`
separately — so the two 0.311 B tensors are genuinely two tensors and the denominator is
right. Worth checking rather than assuming: on a tied model the allocator would be handing
two widths to one piece of storage.

### Channel moments cover 1.02% of this model

`--moments` switches the allocator from rank-product ordering to measured Gauss-Newton
sensitivity, and it is the stronger signal where it exists. On this run it exists for **49
of 654 modules — 0.324 B of 31.719 B**: the 48 routers and `lm_head`.

The cause is mechanical and was verified rather than inferred. The moment hook guards on
`inp.shape[-1] == entry.weight.shape[1]`, and a `bitsandbytes` 4-bit weight is stored
flat — a 4096x2048 projection has `weight.shape == (4194304, 1)`. So the check rejects
**every module bnb replaced**, which under QLoRA is every Linear except the ones it skips.
The 49 that survive are exactly the ones left in bf16. The expert banks are excluded for a
separate and deliberate reason: the hook takes `kind == "linear"` only, and a bank is 3-D.

`_apply_fallback_scale` handles a partial table honestly — it rescales the proxy price by a
ratio of rung-normalised medians and records the split in the saved map. But with 1.02%
coverage the calibration anchor is 48 routers and an `lm_head`, all of them 8-bit-floor
roles that resemble nothing in the 91.4% being priced. That is a thin and biased anchor for
a global multiplier, so **the primary arms allocate without `--moments`** — one formula
pricing the whole heap, which is the case the scale factor is explicitly not needed for.
The comparison was still worth running as a dry-run allocation diff, since it costs no
weights, and it was: **at 4.00 bits `--moments` changes 0 of 650 widths, and at 3.00 it
changes 23.** The byte totals are identical to the digit at both budgets, because the
budget binds either way and only the ordering moves.

So on this model the thin anchor is close to harmless — but the 23 moves at 3.00 argue for
the choice that was made rather than against it. One of them cuts `lm_head` from 3 bits to
**2**, the tensor the S5 probe measures at 28.5% top-1 agreement at that width. A
fallback scale calibrated on 48 routers and an `lm_head` is being asked to price 91.4% of
the model it has never seen, and the one module it *has* seen is the one it moves furthest
in the wrong direction. The proxy-priced map the primary arms used is the conservative
choice here, and the diff is banked at
[`experiments/phase5/omni-slurp/s5/map_diff_3.00.json`](../experiments/phase5/omni-slurp/s5/map_diff_3.00.json).

---

## 7. Design consequences to state plainly

**The S1 headroom screen returned 79.40% (397/500), above the pre-registered `[30%, 70%]`
band.** Recorded as a breach rather than silently re-set. The fine-tune remains mandatory
regardless of headroom, because the signal map only exists during training — but the
narrow band means the primary framing must be *damage against the bf16 fine-tuned
ceiling*, not *percentage of a fine-tuning gain retained*.

**No baseline arms.** With GPTQ/AWQ/RTN descoped, this phase cannot make a
matched-bytes competitive claim. What it can make is an internal one: dq3 and dq4 against
the bf16 ceiling, paired per item. That is a smaller claim, and it should be written as
one.

**The adapter is trained against NF4 and merged into bf16.** Standard for QLoRA and shared
with every prior phase, but it means the merged checkpoint is not bit-for-bit the model
that produced the signal map.

**Roles that fall to `other`: 70 modules, 288.7 M parameters, 0.91%.** Measured, not
estimated — the breakdown is 143 M of visual attention (`qkv` and `proj`, 27 blocks), 122 M
of visual mergers (`merger` plus the three `merger_list` deepstack heads), 18 M of audio
convolutions and projections, and 4.5 M of position/patch embeddings. They take the
conservative 4-bit default.

The mergers are the interesting entry: a merger *is* the multimodal projector, and
`MM_PROJECTOR` carries an 8-bit floor precisely because a projector has no redundancy. It
is not being detected as one. That costs this campaign nothing — SLURP never runs the
visual tower, so those modules carry no signal and the allocator correctly sends them to
the bottom — but on an image task it would be a real defect, and it is written down here
rather than left to be rediscovered.

---

## 8. Sequence

| stage | command | outcome |
|---|---|---|
| S1 headroom | `dynquant eval --task slurp` on the base | done — 79.40% (397/500), above the pre-registered band |
| S2 fine-tune | `torchrun --nproc_per_node 2 scripts/run_s2_omni_finetune.py --measure-expert-banks` | done — 500 steps, effective batch 16, signal file written |
| S3 merge | `scripts/run_s3_omni_merge.py` | done — 63 440 876 184 B bf16 Thinker, all 96 expert banks live |
| S4 allocate | `dynquant inspect --target 3.0 4.0 --save-map maps.json` | done — 400 violations at 3.0, 0 at 4.0, §6 |
| S5 evaluate | `dynquant eval --map --map-apply encode` × {bf16, dq3, dq4}, per-item `hits` stored | done — see below |

| arm | accuracy | vs comparator | p |
|---|---|---|---|
| `omni-base` (whole Omni, bf16) | 79.40% | — | — |
| `omni-sft` (Thinker-only, merged) | 86.80% | **+7.40** vs base | 7.5e-07 separated |
| `omni-dq4` (4.00x fewer bytes) | 86.20% | −0.60 vs SFT | 0.70 **not separated** |
| `omni-dq3` (5.33x fewer bytes) | 25.00% | **−61.80** vs SFT | 1.6e-88 separated |

The full record, including the floor-breach tables, the `lm_head` probe ladder and what this
campaign cannot claim, is [`docs/reports/phase5-omni-slurp.md`](reports/phase5-omni-slurp.md).

S4 **allocates without writing a checkpoint** — `inspect` takes both targets at once and
writes both maps into one file — and S5 applies a saved map in VRAM. This is the path the
CLI's own help recommends, and here it is also the only affordable one: a `quantize
--output` writes the decoded weights in the compute dtype, so each arm would cost another
~63 GiB of disk beside a 70 GiB base and a 63 GiB merge.

**S5 must pass `--map-apply encode`, not the default `pack`** — and the reason is the
router, not the banks. It is worth being precise about which, because the first version of
this section gave the wrong one and the wrong one is the more alarming of the two.

`pack` *does* reach a batched expert bank. `resolve_target` answers a rank-3 tensor whose
owner is not a torch builtin with an `ExpertBank`, which `_wrap` turns into a
`DynQuantExpertBank` the parent indexes one expert at a time
([`runtime/linear.py`](../packages/dynquant-core/src/dynquant/runtime/linear.py)); that
landed 154 commits before the one this campaign ran. All 96 banks are packable, and there
was never a run in which 91.40% of the parameters were silently left in bf16.

What `pack` cannot reach is `mlp.gate`. A meta-device instantiation of the Thinker config —
the same architecture the merge produces, at no VRAM and no download — types every one of
the map's 650 names: **504 `nn.Linear`, 96 bare rank-3 parameters on
`Qwen3OmniMoeThinkerTextExperts`, 48 `Qwen3OmniMoeThinkerTextTopKRouter`, and 2
`nn.Embedding`** (`visual.pos_embed` is an Embedding, not the bare parameter its shape
suggests). The router owns a 2-D weight and calls `F.linear` on it itself, so it is neither
a `Linear` the packed runtime can replace nor a bank it can index — and `resolve_target`
**raises** on it rather than skipping it. `pack_model` walks the map in sorted order, and
`model.layers.0.mlp.gate` is name **200 of 650** — so a `pack` run would have packed the
whole audio tower, the head, the embedding and layer 0's two banks, then died, with the
error naming the remedy. Loud, not silent.

`encode` writes the same encoder's output back in the compute dtype: the same values, at
fp16 footprint. Accuracy is therefore the quantized model's accuracy, and the size claim
comes from the map rather than from `nvidia-smi` — which is the correct division of labour
here, and the reason the map file is the artifact the campaign reports against.

The corollary is that a **packed checkpoint on disk is still producible**, and by a
different command. `dynquant export` resolves the same names with `restore=True`, which
answers the router with a `RestoredWeight` instead of refusing it, so all 650 names pass.
`dynquant eval --map-apply pack` never writes a checkpoint at all — it is in-VRAM surgery
for the duration of one eval. See §9.

One asymmetry in that pairing is worth stating rather than leaving to be noticed. The
S1 base screen ran the **whole** Omni checkpoint; every arm after the merge runs the
Thinker alone. So `base vs SFT` differs in two things at once — the fine-tune, and the
absence of the Talker. It is still a fair comparison, because the Talker is strictly
downstream of the text the Thinker emits and SLURP scores that text: nothing the Talker
does can reach the scored tokens. The arms that carry the campaign's actual claim —
`SFT bf16 vs dq4 vs dq3` — are all Thinker-only and differ in exactly one thing.

**S5 stores per-item `hits`, so every A/B is a McNemar test on paired outcomes** rather
than two independent proportions. `--compare` refuses to pair records that differ in task,
split, shots, seed or limit, so the bf16 arm has to be run at the S1 screen's own settings
(500 items, 4 shots) for the comparison to exist at all. On this project the paired test
has repeatedly been the difference between a promotable result and an unpromotable one, in
both directions.

## 9. Producing a packed checkpoint, and what it proved

The campaign reported sizes off the map, not off a file. That was sound — the map's byte
total *is* the encoder's arithmetic — but it left a reasonable question unanswered: can
this model actually be written to disk packed, at 3 and 4 bits, and does the packed file
compute what the map said it would? Both arms have now been exported, loaded back and
scored. It can, it does, and nothing in the run before it had to be different for that to
be true. Two facts settle the first half, both checked rather than reasoned about.

**The export path resolves every name the map contains.** `dynquant export` calls
`resolve_target(model, name, restore=True)`
([`quant/checkpoint.py`](../packages/dynquant-core/src/dynquant/quant/checkpoint.py)),
and `restore=True` is exactly the flag that turns the router refusal into a
`RestoredWeight`. Against the meta-device type census in §8 that gives 504 Linears, 2
Embeddings and 96 banks resolved as they are under `pack`, plus 48 routers resolved as
restored weights — 650 of 650. There is no name to drop and no `--ignore` to write.

**The map is already the artifact export wants.** `maps.json` holds both targets, so
`--map-key` is mandatory rather than optional, and `--group-size 128` has to be repeated
because it is a property of the encoding, not of the map. The arms ran asymmetric, so
`--symmetric` must *not* be passed — passing it would encode a different model than the one
that was scored, which is the failure
that cost phase 2 a whole replicate ([question 4](reports/README.md)).

```bash
dynquant export /workspace/runs/omni-slurp/merged \
  --model-class Qwen3OmniMoeThinkerForConditionalGeneration \
  --map /workspace/runs/omni-slurp/maps.json --map-key 3.00 \
  --group-size 128 --device cpu --dtype bfloat16 --compute-device auto \
  -o omni-dq3-packed --json
# and again with --map-key 4.00 -o omni-dq4-packed
```

**Both arms ran, and both reconcile to the byte.** Each wrote 650 of 650 modules —
*including all 48 routers*, which is the claim above turned from an argument into an
artifact — of which **96 are batched expert banks**, written packed and dequantized one
expert per routing hit until the grouped kernel lands.

| arm | seconds | packed | + dense | = total | map predicted | avg bits | directory |
|---|---|---|---|---|---|---|---|
| dq4 | 78 | 15 845 134 784 | 14 251 232 | **15 859 386 016** | 15 859 386 016 | 3.9972 | 15 871 835 299 |
| dq3 | 74 | 11 880 356 544 | 14 251 232 | **11 894 607 776** | 11 894 607 776 | 2.9971 | 11 907 056 499 |

The dense residue is *inside* the map's total, not beside it: the norms and biases the map
does not name are already priced at 16 bits in its denominator, which is why packed + dense
lands on the prediction to the byte on both arms rather than near it. Both residues are the
same 14 251 232 B, as they must be — the map changes what is quantized, not what is left.

The directory column is the shipped directory. It is 2 356 B larger on both arms than the
`directory_nbytes` recorded in
[`experiments/phase5/omni-slurp/s6/export.log`](../experiments/phase5/omni-slurp/s6/export.log)
(15 871 832 943 and 11 907 054 143), because `processor_config.json` — 2 356 B, and the file
`AutoProcessor` actually resolves from — was written after the export ran. The log is correct
for the moment it was written; the column is correct for the artifact that shipped.

The one figure that moves is average bits, 3.9972 against 3.9999 and 2.9971 against 3.0000,
because the map prices a module at `bits × params` while the file also carries scales and
offsets per group and some tensors round to fewer words than predicted. Three thousandths
of a bit is the size of that gap on this architecture. A *larger* one is the signal that the
export encoded something the allocator did not price, and is the first thing to check.

Four things the box time taught. The first two are properties of the format; the last two
were defects, both found by running and both since fixed in the package:

- **The dispatch question lands at read-back, not at export.** `export` has no
  `--experts-impl` — it writes tensors, it does not run the model. But
  `use_dynquant_experts` reads `_experts_implementation` off the *outer* config
  ([`runtime/experts.py`](../packages/dynquant-core/src/dynquant/runtime/experts.py)), and
  on a Thinker-only checkpoint that is not the config the banks were written under, so
  whatever loads this directory has to pin its own dispatch and record which ran.
- **Read-back needs the quantizer registered, and nothing on the eval path did it.** This
  is not a caution written in advance; it cost two 8-minute arms that both scored **0.0%
  with 499 and 500 of 500 answers unparseable**, which reads exactly like a destroyed
  model. It is not. `from_pretrained` sets `pre_quantized = hasattr(config,
  "quantization_config")`, then asks `AutoHfQuantizer.supports_quant_method`, and for a
  method nobody registered that helper **logs a warning and returns False** — so
  `pre_quantized` flips back to False, transformers builds a **randomly initialised** model
  from the config, and nothing raises. The literal line is `Unknown quantization type, got
  dynquant ... Hence, we will skip the quantization.` The sharpest part: the fix already
  existed. `integration/hf_quantizer.py` documents this exact failure mode at length under
  *"Why this exists, measured rather than assumed"* — it was written, documented, and
  never **called**. `load_model` in
  [`commands/_shared.py`](../packages/dynquant-core/src/dynquant/commands/_shared.py) now
  registers before `from_pretrained` and counts packed modules after it, raising if a
  checkpoint that declares `quant_method: dynquant` comes back with none. Pinned by
  [`tests/test_shared_packed_load.py`](../tests/test_shared_packed_load.py), whose negative
  control asserts the downgrade behaviour against transformers itself rather than a
  fixture.
- **`vllm serve` is hard-refused for fused MoE**, in
  [`integration/vllm_plugin/config.py`](../packages/dynquant-core/src/dynquant/integration/vllm_plugin/config.py),
  even though export prints a `vllm serve` line in its summary. There is no packed grouped
  kernel yet; that is phase 8.
- **`_save_tokenizer` writes no `processor_config.json`**, so `AutoProcessor` refuses the
  packed directory and every eval against it exits before loading a single weight. The
  error names `preprocessor_config.json`, which is a red herring: the merged bf16 directory
  has no `preprocessor_config.json` either and loads fine. `processor_config.json` is the
  file that decides -- with it present `AutoProcessor` resolves `Qwen3OmniMoeProcessor`
  directly, and without it the resolution falls through to `image_processing_auto`, which
  is what raises. `_save_tokenizer` now copies the five processor sidecars named in
  `constants.HF_PROCESSOR_SIDECARS`; the negative control in
  [`tests/test_processor_sidecars.py`](../tests/test_processor_sidecars.py) reads those
  names back off `transformers.processing_utils` rather than restating them.

**Read-back, on the box, with the quantizer registered.** Both directories load, and what
comes back is not a summary but a census: `{DynQuantLinear: 504, DynQuantEmbedding: 2,
DynQuantExpertBank: 96}` on both arms, with **48 of 48 routers dense** — the export restored
them rather than packing them, which is what `restore=True` is for.

The value lattice is the check that separates "the class says packed" from "the values are
packed", and it has to be counted **per quantization group**: a 4096-element slab spans 32
groups of 128, each with its own scale and zero point, so its distinct-value ceiling is
`32 x 2**bits` and it proves nothing. The first probe made exactly that mistake and printed
371 distinct against a claimed ceiling of 16. Counted per group of 128 the picture is exact,
and it shows more than the width — it shows the **map's heterogeneity surviving into the
file**:

| module | 4.00 arm | 3.00 arm |
|---|---|---|
| `audio_tower.layers.0.self_attn.{q,k,v}_proj` | 4b, 13-16 distinct of 128 | 2b, 4-4 of 128 |
| `audio_tower.layers.0.self_attn.out_proj` | 4b, 14-16 | 3b, 8-8 |
| `audio_tower.layers.0.fc1` | 3b, 8-8 | 2b, 4-4 |
| `audio_tower.layers.0.fc2` | 3b, 6-8 | 3b, 6-8 |
| `model.layers.{0,1,2}.mlp.experts.{gate_up,down}_proj` (banks) | not separately probed | — |
| `model.layers.0.mlp.gate` (router, never quantized) | 65-95 of 128 | 65-95 of 128 |

Every count sits at or just under its own `2**bits` ceiling, never above it, and the dense
router sits an order of magnitude above all of them. The 3.00 column is the floor-breach
story made physical: the same six modules that the 4.00 map places at 4/4/4/4/3/3 bits are
at 2/2/2/3/2/3 in the 3.00 file.

The banks are the one row this table cannot fill. The per-group probe covered the audio
tower and the router only; the sole bank measurement anywhere is the discredited whole-slab
count above, so 91.4% of the model is attested by its module class and its manifest widths
and not by a counted lattice. Nothing here contradicts it — 371 distinct over 32 groups of a
16-value ceiling is exactly what a correct 4-bit bank would print — but it is not the same
evidence as the rows above, and the section exists to keep those two apart.

**And VRAM is the packed size, which is the one thing `encode` could never show.** Measured
as `torch.cuda.memory_allocated` across the load, on an otherwise idle card:

| arm | resident VRAM | manifest total | over | vs the bf16 merged Thinker (63 440 876 184 B) |
|---|---|---|---|---|
| dq4 | 15 892 454 912 | 15 859 386 016 | **+0.21%** | **3.99x** |
| dq3 | 11 927 683 584 | 11 894 607 776 | **+0.28%** | **5.32x** |

A fifth of a percent over the payload is **mostly the routers**, and the split is exact. The
48 MoE routers are written packed at 8 bits (270 336 B each) but restored to a dense bfloat16
weight at load, because nothing in the packed runtime can stand where a router stands. Each
therefore costs 524 288 B resident against 270 336 B on disk, and 48 x 253 952 B =
**12 189 696 B** of the overage is that restoration alone — 36.9% of it on both arms
(dq4 33 068 896 B, dq3 33 075 808 B). It is *not* scales and zeros: those are already inside
`packed_nbytes`, and therefore already inside the manifest total being compared against. The
~20.9 MB remainder is the runtime's own bookkeeping. Either way the number says nothing
dequantized to fp16 on the way in — had any module fallen back, the residue would be measured
in gigabytes rather than tens of megabytes.

**And both packed arms score exactly what their `encode` twins scored, item for item.**
86.20% (431/500, 0 unparseable, 497.2 s) and 25.00% (125/500, 3 unparseable, 732.5 s), against
86.20% and 25.00% from S5 — with **zero discordant pairs on either width**, 40/40 identical kept
predictions, and even the unparseable counts matching. That is the *predicted* result and it is
what makes it a test rather than a replication: `encode` writes the encoder's dequantized output
into the dense tensor at compute dtype, `pack` stores the codes and dequantizes at compute time,
so identical codes, scales and dtype must give identical greedy tokens. One discordant item out
of 1 000 would have located a divergence in the packed path. `experts` reports `{found:
"dynquant", ran: "eager"}` on both, against `{found: "grouped_mm", ran: "eager"}` on the encode
arms — the value that decides the arithmetic is `ran`, and it is `eager` on all four, so no
comparison straddles two dispatch paths. Full numbers in
[the campaign report](reports/phase5-omni-slurp.md#the-packed-checkpoint-computes-the-same-thing).

And the thing worth saying loudest: **a packed 3-bit artifact is a 25.00% model.** §6
explains why — 3.00 is below this architecture's 3.418-bit floor budget — and the artifact
does not become more defensible for being on disk instead of in VRAM. If it is exported, it
is exported as a reproduction of a measured collapse, labelled as one.
