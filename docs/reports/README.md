# The experimental record

Every experiment run on DynQuant since the package was built, what it measured, and where the
full record lives. Nothing here is taken from the paper; everything is measured on this
repository's own code, all of it on a single NVIDIA A100 80GB PCIe.

There are seven campaigns. They answer seven different questions, in this order:

| # | question | verdict | full record |
|---|---|---|---|
| 1 | Does a signal-driven allocator beat a same-size uniform one? | **Only after the score was replaced.** The published rank-product score *lost* by 2.03 pts; a measured Gauss–Newton sensitivity wins by **+10.29** | [`RESULTS.md`](../../experiments/four_point/RESULTS.md) |
| 2 | Does that hold on a different model, scale, architecture and training regime? | **Yes, qualitatively; not in magnitude** | [`RESULTS-mistral7b-banking77.md`](../../experiments/four_point/RESULTS-mistral7b-banking77.md) |
| 3 | Does it beat what people actually ship — GPTQ, AWQ, RTN, bnb-NF4? | **Wins at 2.42×, ties at 3.8×, lost at 4.9×** | [phase 1 PDF](dynquant-phase1-external-comparison.pdf) · [record](../../experiments/four_point/RESULTS-external-comparison.md) |
| 4 | Can the 3-bit loss be reversed without adopting GPTQ's mechanism? | **Yes. +1.54 over GPTQ at 7.4 % fewer bytes, p < 0.0001** | [phase 2 PDF](dynquant-phase2-beating-gptq-3bit.pdf) |
| 5 | Do inference servers hold the same quantized weights the direct run holds? | **Yes, on both vLLM and SGLang** — after a real defect was found and fixed | [`serving-parity.md`](serving-parity.md) |
| 6 | Do the phase-3 benchmarks have room for quantization damage to show? | **Yes, all four** — 54.5–83.2 %, no arm near ceiling; two harness defects caught | [`phase3-s1-headroom-screen.md`](phase3-s1-headroom-screen.md) |
| 7 | Does the S2 fine-tune know which tokens are the assistant's? | **Yes on both tokenizers, ≤0.07 % unmaskable** — after the obvious method dropped 100 % of the data on one of them | [`phase3-s2-loss-masking.md`](phase3-s2-loss-masking.md) |

The method itself — signals, sensitivity estimator, allocator, encoder, format, packed
runtime, kernels — is documented end to end in the
[**whitepaper**](../whitepaper/dynquant-whitepaper.pdf), which also carries the kernel and
VRAM measurements described in §6 below.

---

## 1. The allocator campaign — Qwen3.5-2B-Base, GSM8K and CaseHOLD

[`experiments/four_point/RESULTS.md`](../../experiments/four_point/RESULTS.md) · 1 302 lines

The experiment the package was built to make possible: collect a signal map during a real
fine-tune, let the scorer and allocator choose per-module bit widths from it, quantize, and
measure the same benchmark at every stage.

It resolved in three stages, and the first one was a negative result against the method as
published.

**Stage one.** At a ~4.25-bit budget, where the score changes only 14–16 of 187 module
widths, score-driven allocation is indistinguishable from a same-size uniform control
(GSM8K +1.36, p = 0.20; CaseHOLD −0.13, p = 0.56). At ~3.25 bits, where it changes 70–79 of
them, allocation **loses**: −7.88 on GSM8K (p = 1.1e−10) and −2.03 on CaseHOLD (p = 5.8e−05).
The more the rank-product score drove the widths, the more it cost.

**Stage two — the diagnosis and the fix.** The published score
`Rank(plasticity) × Rank(saliency)` multiplies a factor that correlates with measured damage
by one that correlates **against** it (Spearman ρ = −0.21 over 187 modules), and then compares
modules by ordinal rank when the allocator's decision is cardinal. Replacing it with a
measured per-module sensitivity, `Σ E[δ_r²] E[x_c²] (W − Q_b(W))²`, collected by the same
tracker and priced per **move** rather than per module, turns the same budget on the same
checkpoint from −2.03 into **+10.29 points** (84.08 % vs 73.79 %, p = 1.3e−81). Quantization
damage at 3.25 bits falls from 16.73 points to **4.40**.

**Stage three — packed on real kernels.** Accuracy exactly unchanged (the same 4 468 of 5 314
problems), which retroactively licenses every simulated-quantization number in the document.
See §6.

**Task selection is part of the result.** GSM8K is the paper's own task and the fine-tune did
not move it (−0.99, p = 0.48) — not because the fine-tune failed but because the base model
was already at the supervised ceiling, so its table cannot read quantization damage against a
fine-tuning gain. CaseHOLD, chosen by measuring base-model headroom *first*
([`experiments/screen/`](../../experiments/screen/)), gained **+53.35 points** from the
identical recipe. Against that gain the quantized arms become legible: ~4.25 bits keeps 96 %
of it, ~3.25 bits keeps 69 %.

Four defects were found and fixed by this campaign, each written up where it occurred: every
role's least-important module was free to destroy; the paired test was unrecoverable after
the GPU-hours were spent; fallback calibration double-counted the width span; a tied embedding
table arrived on the GPU twice.

## 2. Transfer — Mistral-7B-Instruct-v0.3, Banking77

[`experiments/four_point/RESULTS-mistral7b-banking77.md`](../../experiments/four_point/RESULTS-mistral7b-banking77.md) · 388 lines

Four conditions changed at once against campaign 1: 1.88 B → 7.25 B, hybrid linear/full
attention with a **tied** embedding → dense GQA with an **untied** one, CaseHOLD → Banking77,
full fine-tune → **LoRA r=32**. Changing four things at once is normally how you learn
nothing; it is right here because the question is not "which of these mattered" but "does the
method transfer at all", and a method that only works on one 2 B model with a tied embedding
under a full fine-tune is not a method.

What replicated:

1. The allocator helps where damage exists and not otherwise — null at 4.25 bits on both
   models, separated at 3.25 bits on both.
2. The **reallocation direction** — bits out of MLP `up_proj`/`down_proj`, into attention
   `k_proj`/`v_proj`. Two runs sharing no model, task, architecture family or training regime
   agreeing on which roles are worth paying for is evidence the signal measures transformer
   structure rather than one checkpoint.
3. The current default scorer is on the right side of zero end to end, which had never been
   shown before this run.

What did not transfer, and could not: the magnitude. A 7 B model at 3 bits loses 2.53 points
to uniform quantization where the 2 B lost 14.69. The method's value scales with the damage —
which means it matters most exactly where compression is most aggressive, and that a bigger
model buys robustness that reduces how much any allocator can add.

## 3. Phase 1 — against GPTQ, AWQ, RTN and bitsandbytes NF4

[**PDF**](dynquant-phase1-external-comparison.pdf) · [LaTeX](dynquant-phase1-external-comparison.tex) ·
records: [`REPORT-quantization-comparison.md`](../../experiments/four_point/REPORT-quantization-comparison.md) (934 lines),
[`RESULTS-external-comparison.md`](../../experiments/four_point/RESULTS-external-comparison.md) (1 540 lines)

Two models, two tasks, one harness, one process. 45 arms across three panels. Each model is
fine-tuned on the task its own headroom screen selected, then quantized six ways from that one
checkpoint, and every arm scores the same held-out split in the same order. Baselines are
produced with `llm-compressor` (GPTQ, AWQ, RTN) and `bitsandbytes` (NF4).

| model | task | test rows | chance | base | fine-tuned |
|---|---|---:|---:|---:|---:|
| Mistral-7B-Instruct-v0.3 | Banking77, 77-way | 3 080 | 1.30 % | 33.7 % | 94.38 % |
| Qwen3.5-2B-Base | CaseHOLD, 5-way | 5 314 | 20.0 % | 32.33 % | 89.74 % |

Seven findings, in descending order of how much they should change your mind:

1. **At 2.42× compression DynQuant is lossless and GPTQ is not.** At byte-identical size
   (1.4512 vs 1.4514 GiB) DynQuant scores 89.71 % against GPTQ's 88.97 % — **+0.73, p = 0.017** —
   and is statistically indistinguishable from bf16 (−0.04, p = 0.905) where GPTQ is not
   (−0.77, p = 0.012). One tier, one task, one model.
2. **At 3.8× compression DynQuant ties everything.** 11 of 12 comparisons fail to separate;
   the twelfth favours DynQuant. On the tied-embedding model the tie is achieved at **1.73×
   fewer bytes** than the baselines' shipped configuration. The tie is the result; the size is
   the win.
3. **At 4.9× compression GPTQ wins, on all three panels** — −1.34 (p = 4.2e−04), −1.04
   (p = 0.011), −0.78 (p = 7.1e−03) — simultaneously 3.1 % smaller and 1.34 points better.
   This became the brief for phase 2.
4. **DynQuant beats AWQ at 3 bits by 2.5–3.3 points on both Qwen tasks**, at half the bytes.
5. **The margin over naive rounding is 88 % architectural prior and 12 % training signal.**
   At 3.25 bits DynQuant beats byte-matched RTN by 25.78 points; holding allocator, graph,
   floors and budget fixed and replacing only the signal scores with a constant splits that
   into **22.62 points of allocation and 3.16 of signal** (p = 1.9e−15).
6. **Above ~4 bits the training signal is worth nothing measurable** — 12 of 187 widths moved,
   +0.19 points, p = 0.15. The operating rule: DynQuant earns its keep only when the budget is
   tight enough that the role floors cannot all be paid for.
7. **The accuracies survive real kernels and the bytes are exact; the speed is not there yet.**

The single most important methodological point in this report is §2, the accounting problem:
GPTQ and AWQ's conventional `ignore=["lm_head"]` makes a nominally 4-bit arm measure 7.36
bits on a tied-embedding model, so any comparison that does not match bytes is comparing
different sizes.

## 4. Phase 2 — reversing the 3-bit loss without copying GPTQ

[**PDF**](dynquant-phase2-beating-gptq-3bit.pdf) · [LaTeX](dynquant-phase2-beating-gptq-3bit.tex) ·
record: [`RESULTS-external-comparison.md` §Phase 2](../../experiments/four_point/RESULTS-external-comparison.md)

Measured 2026-07-30. One panel: Qwen3.5-2B-Base / CaseHOLD, 5 314 items, every comparison a
paired exact McNemar test on stored per-item hits.

| | arm | acc % | bytes | avg bits | vs `gptq_3b_head` | p |
|---|---|---:|---:|---:|---:|---:|
| before | `dq_3p25` (shipped) | 86.70 | 764 290 013 | 3.2494 | −1.34 | 0.0004 |
| tie | `p2_wc_agg` (per-module body) | 88.54 | 708 087 808 | 3.0104 | +0.51 | 0.16 (ns) |
| **after** | **`p2_rb_agg`** (per-row body) | **89.57** | **708 087 808** | 3.0104 | **+1.54** | **< 0.0001** |

+2.87 points end to end at 7.4 % fewer bytes than the shipped arm. The fp16 ceiling on this
panel is 89.74 %, so the final arm gives up **0.17 points against full precision at 3.01
stored bits**.

**The constraint.** The obvious fix for "rounds naively" is GPTQ's own — propagate each
column's quantization error into the not-yet-quantized columns via an inverse Hessian. It was
rejected for two reasons: it would make the method a reimplementation of GPTQ with a
preprocessing step, and it requires a calibration dataset, which is the exact dependency the
fine-tune-time hook exists to remove. Every lever below acts on a component DynQuant already
owns, and every input is a moment the `transformers` hook already accumulates online.

Four levers, each isolated at byte-identical budgets: a Gauss–Newton sensitivity model, a
row-partitioned tied embedding, an `E[x²]`-weighted clip objective, and per-row body
allocation.

**The central finding is not the headline.** Allocation granularity is a *multiplier on the
signal, not a gain of its own*: with the row widths shuffled, per-row allocation **loses 1.28
points** at identical bytes, and wins 2.31 against that control when given the signal's
ordering. Any granularity change must ship with its shuffled control.

## 5. Serving parity — vLLM and SGLang

[`serving-parity.md`](serving-parity.md)

The only experiment in this project that runs the model through something other than
`transformers`. A serving integration can load a checkpoint, start, answer, and be wrong: the
failure mode is not a crash but a model that silently dropped its packed tensors.

Both servers reproduce the direct GPU run to **100 % teacher-forced top-1 agreement** at all
108 scored positions, with a mean absolute logprob difference of 0.009147 (vLLM) and 0.006822
(SGLang) against fp16 controls of 0.006304 and 0.006896 — so the serving gap is 1.45× and
**0.99×** the gap each runtime already has on an unquantized model, and 269× / 361× smaller
than quantization's own effect on the same numbers (2.461067). Neither run passes a
`--quantization` flag or patches the server.

It found a real defect. SGLang reads `getattr(model_class, "packed_modules_mapping", {})`,
and on 0.5.16 that attribute is absent from **172 of 210** model files — including
`Qwen2ForCausalLM`, which fuses q/k/v inside `load_weights` anyway. The result was a server
that started, answered, and computed with uninitialised buffers. Fixed in `9e4c6ed` with a
bounded fallback plus a guard that makes the failure loud forever after.

One number is not yet explained and is flagged as open: on fp16 the two servers generate
identically for 12/12 prompts, on the quantized checkpoint only 10/12.

## 6. The packed runtime, VRAM and the kernels

Not a separate document — measured inside campaigns 1–3 and written up in the
[whitepaper](../whitepaper/dynquant-whitepaper.pdf) §"The packed runtime" and in
[`RESULTS.md` §"Running it packed"](../../experiments/four_point/RESULTS.md). Collected here
because it answers its own question: *do the accounted bytes exist on the GPU, and what do
they cost?*

**Accuracy through the kernels: exactly unchanged.** The same 4 468 of 5 314 problems, packed
and simulated. Not "within noise" — identical, because both paths quantize through the same
search over the same grid, so anything but an exact match is a kernel bug rather than a
rounding difference. On Mistral the two arms differ by 5 and 1 predictions out of 3 080.

**VRAM: the manifest was right to 0.03 %.** 0.7237 GiB resident against 3.5052 GiB at bf16
(**4.84×**); the allocator predicted 764 317 696 bytes and the live model holds 764 530 560.
This is the claim the paper's own Appendix F concedes it cannot make. At batch 32 the same
model is only 1.52× smaller as a process, because the KV cache dominates — quantization buys
headroom in proportion to how much of your memory is weights.

**Decode is slightly slower, and the reason is measured rather than guessed.** 0.90× at
batch 1, 0.60× at batch 32 (above `gemv_max_rows() == 8` the runtime falls back to
dequantize-then-`F.linear`). Profiling both arms through one script puts in-model matmul time
at **1.42× faster** packed, with the entire drop in GPU-busy time accounted for by that one
change — but a decode step on this model issues ~2 000 kernel launches, leaves the GPU idle
~70 % of the time, and spends 12 % of its wall clock in matmuls, so the 1.04 ms won is
outweighed by the host-side cost of dispatching 187 packed modules from Python. CUDA Graphs
(P8) are what remove that.

**The kernel in isolation.** Three rounds of optimization took it from 0.64–1.83× bf16 to
**1.09–2.56×**, and from 26–41 % of achievable HBM bandwidth to **10–98 %**. The ≥70 % gate is
met at 8 bits (98 %), three points short at 4 bits (67 %), and missed at 3 and 2 bits (52 %,
36 %). What remains is issue-bound rather than bandwidth-bound — measured against a computed
instruction-issue floor of ~141 µs against 237 µs achieved — and needs the tensor-core path
(P7) rather than more of the same. Round 2 is the most useful measurement in that section: a
−11.7 % instruction count bought only −2.5 % time, which is how the ceiling was located.

**Verification.** 417 kernel-parity tests across every geometry × width × M × dtype, and all
four `compute-sanitizer` tools — memcheck, initcheck, racecheck, synccheck — report 0 errors
and 0 hazards over that suite.

## 7. Phase 3 — S2, locating the assistant turns

[`phase3-s2-loss-masking.md`](phase3-s2-loss-masking.md)

The second pre-flight check of phase 3, and the same rule as §"Task selection precedes
fine-tuning" applied one level down: verify the loss mask before spending 30 GPU-hours on it.
A wrong mask does not fail — it trains a slightly worse model and reports success, which is
fatal to a campaign whose purpose is to measure small differences *on top of* that model.

The obvious method — locate turn `i` by the length of the render of `messages[:i]` — assumes
each rendered prefix is a token prefix of the next. Nothing promises that. `mistral_common`,
behind Ministral-8B, refuses to render *any* assistant-final conversation at all (it is
validating a serving request), so the walk drops **3 000 of 3 000** rows; Phi-4-mini appends a
document terminator to a conversation it is not asked to continue, so its "prefix" isn't one.
The mode that works renders each assistant turn *open* with `continue_final_message=True` and
closes it with a terminator measured from three synthetic renders — `<|end|>` (200020) on Phi,
which is **not** its `eos_token_id`. Which mode a tokenizer needs is not an attribute anywhere,
so `--mask-mode auto` measures both on 32 real rows and takes the winner.

Result on 3 000 Tulu-3 rows each: **0.00 %** unmaskable on Phi, **0.07 %** on Ministral (2 rows
of empty assistant content), 70.8 % / 70.5 % of tokens supervised. Phi accepts both modes, and
on 500 rows the two agree on every id and every span start, differing only by that one document
terminator.

Three designs died on real tokenizers after passing a stub suite; the most dangerous **passed a
3 000-row dry run at 95 %**, because `mistral_common` merges adjacent same-role turns and the
merge separator and turn opener are each one token, so two errors cancelled. That is why the
fourth design was prototyped against the real tokenizers before being written into the module.

The same run's census flags a GSM8K-derived source at **5.2 % of the mixture**, against a
headline that includes GSM8K. Reporting it and keeping it is defensible for arm-versus-arm —
both sides train on the same mixture — but that argument does not cover *test* leakage, which
would make the score recall rather than reasoning and compress the range quantization damage
has to show up in. Settled by looking: a 13-gram index over all 1 319 test questions, scanned
against the run's own 50 000 rows, flags **2 items and 0 usable duplicates** — one shared word
problem skeleton with different quantities, one WildChat conversation quoting "Janet's ducks".
0.15 points of headroom either way, against a +1.54 effect size. GSM8K stays; the claim that
its post-SFT number measures *general* math gain does not.

---

## 8. Phase 3 — S3, what fusion costs the model that has it

Full report: [phase3-s3-fused-floors.md](phase3-s3-fused-floors.md). Measured before S3's arms
were built, because phase 2 ran only on models that spell every projection separately and
Phi-4-mini does not: `qkv_proj` and `gate_up_proj` are **55.1 % of its parameters**.

What holds: the row partitions land correctly at the checkpoint's real geometry — 24:8 GQA
and an unset `head_dim`, neither of which the hidden-64 fixture produces — and the allocator
descends from floors it cannot afford on a fused model, hitting 3.2492 against a 3.25 target
with 90 breaches named and 69 of 129 modules moving against a shuffled-score control. That is
bug 4's guard, which until now had only ever run on an unfused synthetic model.

What does not: **the two panel models are in different regimes at the same nominal target.**
Phi's floors cost **4.43 average bits** against Ministral's **3.82**, so a 3.25-bit arm asks
Phi to go 1.18 bits below its floors and Ministral 0.57 — and phase 2 found the signal only
earns its keep once floors stop being affordable. Of Phi's 4.43, **0.21 bits is fusion rather
than architecture**: a fused tensor takes the strictest of its partitions' floors, so 0.805 B
up-projection parameters are charged the SwiGLU gate's 4 bits instead of their own 3. The
allocator never reads `partitions`, though `QuantTensor` already carries them — the format is
ready and the allocator is not. Left open deliberately; closing it would break comparability
with phase 2.

---

## 9. Phase 3 — S2 arm 1, verifying the signal map before spending it

Full report: [phase3-s2-phi-signal-map.md](phase3-s2-phi-signal-map.md). Phi-4-mini × Tulu-3
finished in **12.68 h**, 1501 optimizer steps, final train loss **0.6747**. A run that completes
is not a run that produced something, so the map is checked before S3 reads it.

Structure is sound: 130 modules (2 + 32×4) under canonical names, the tie recorded,
`grad_norm_count` equal to the optimizer-step count on every module that has one (bug 10 stays
fixed), and `forward_calls` **uniformly 12 004** — the check that gradient checkpointing stayed
off, without which one part of the model's saliency would sit on a squared EMA decay and rank
against the rest on different terms. Saliency spans 428.9×.

One module scores on nothing: `model.embed_tokens`, for **three** independent reasons — no
`δxᵀ` to form under `outer_exact` (an embedding's gradient is a scatter-add), a guard that
discards its measured saliency along with the missing plasticity, and a role group of size one,
where a percentile rank is 0.5 by construction. It is 16% of the model and **pays 67.8% of the
3.25-bit floor shortfall by itself**. Forcing its score across the full range moves it 2 → 4
bits, so the neutrality is not structurally harmless; substituting its tied partner `lm_head`'s
row — the same tensor, measured — scores 0.9264 and lands on the same 3 bits. Costs this
checkpoint nothing, stated as a caveat rather than fixed. **Ministral is untied and must be
re-checked**: two singleton role groups, and an `lm_head` whose real gradient signal per-role
ranking would then discard.

---

## Conventions that apply to every campaign

**Paired tests on stored per-item hits.** Every arm stores which items it got right, so every
A/B is an exact McNemar test rather than a comparison of two independent proportions. The
standard error of the difference roughly halves — which promoted a +1.07 to p = 0.0002 and
refused to promote a +0.51 over GPTQ.

**Stored bits, not payload bits.** Every "average bits" figure in every table is total stored
bytes divided by parameter count — scales, offsets and packing overhead included — so the
manifest number is the on-disk number. Comparisons against baselines are made at matched
*bytes*, never at matched nominal width.

**Task selection precedes fine-tuning.** Base-model accuracy is screened before a fine-tune is
spent, because a task the model is already at ceiling on cannot read quantization damage
against a fine-tuning gain. GSM8K's flat arms cost a full six-arm run to diagnose.

**Negative results stay in.** The rank-product score losing by 2.03 points, the +0.51 that
never became significant, the shuffled-row control that loses 1.28, the missed bandwidth gate,
the decode slowdown, and the unexplained 10/12 in §5 are all in the record at the same weight
as the wins.

## Where the raw records live

| path | contents |
|---|---|
| [`experiments/four_point/`](../../experiments/four_point/) | the four campaign records and every stage script (`stage1`–`stage8`, `p2_*`, `run_*.sh`) |
| [`experiments/screen/`](../../experiments/screen/) | the base-model headroom screen that selects a task |
| [`stats/`](../../stats/) | the signal maps collected by the fine-tune hooks |
| [`docs/format-spec.md`](../format-spec.md) | the checkpoint format contract these experiments write and read |
| [`docs/legacy-audit.md`](../legacy-audit.md) | what was wrong with the supplementary code, defect by defect |
| [`decode-neutrality.md`](decode-neutrality.md) | the checkpoint's own `generation_config` reaching a "greedy" decode: how the phase-3 G4 gate found it, which campaigns it does and does not touch, and why the fix took two attempts — the first was correct on transformers 4.x and inert on the 5.x the campaign runs. Ends with what the fixed gate measures: −0.83 points, and a ±1.00 bound GSM8K is too small to resolve |
| [`runtime-parity-gap.md`](runtime-parity-gap.md) | the other half: a GSM8K stop sequence the model never wrote back, generations running on into invented problems, and the two explanations that fitted the data and were wrong (padded batching, different inputs) |
| [`docs/sglang-integration-plan.md`](../sglang-integration-plan.md) | the SGLang plugin design and its S0–S8 staging |
| [`CHANGELOG.md`](../../CHANGELOG.md) | every change, in order, with the reasoning |
