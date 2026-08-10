# The experimental record

Every experiment run on DynQuant since the package was built, what it measured, and where the
full record lives. Nothing here is taken from the paper; everything is measured on this
repository's own code, all of it on a single NVIDIA A100 80GB PCIe.

There are fourteen campaigns. They answer eighteen questions, in this order — phase 4
answers five of them, because whether a benchmark can read damage, whether the model has
already seen its answers, whose bytes “matched bytes” means, which of two prices chose the
widths, and whether the driver that runs the arms runs at all are five separate failures:

| # | question | verdict | full record |
|---|---|---|---|
| 1 | Does a signal-driven allocator beat a same-size uniform one? | **Only after the score was replaced.** The published rank-product score *lost* by 2.03 pts; a measured Gauss–Newton sensitivity wins by **+10.29** | [`RESULTS.md`](../../experiments/four_point/RESULTS.md) |
| 2 | Does that hold on a different model, scale, architecture and training regime? | **Yes, qualitatively; not in magnitude** | [`RESULTS-mistral7b-banking77.md`](../../experiments/four_point/RESULTS-mistral7b-banking77.md) |
| 3 | Does it beat what people actually ship — GPTQ, AWQ, RTN, bnb-NF4? | **Wins at 2.42×, ties at 3.8×, lost at 4.9×** | [phase 1 PDF](dynquant-phase1-external-comparison.pdf) · [record](../../experiments/four_point/RESULTS-external-comparison.md) |
| 4 | Can the 3-bit loss be reversed without adopting GPTQ's mechanism? | **Yes. +1.54 over GPTQ at 7.4 % fewer bytes, p < 0.0001** | [phase 2 PDF](dynquant-phase2-beating-gptq-3bit.pdf) |
| 5 | Do inference servers hold the same quantized weights the direct run holds? | **Yes, on both vLLM and SGLang** — after a real defect was found and fixed | [`serving-parity.md`](serving-parity.md) |
| 6 | Do the phase-3 benchmarks have room for quantization damage to show? | **Yes, all four** — 54.5–83.2 %, no arm near ceiling; two harness defects caught | [`phase3-s1-headroom-screen.md`](phase3-s1-headroom-screen.md) |
| 7 | Does the S2 fine-tune know which tokens are the assistant's? | **Yes on both tokenizers, ≤0.07 % unmaskable** — after the obvious method dropped 100 % of the data on one of them | [`phase3-s2-loss-masking.md`](phase3-s2-loss-masking.md) |
| 8 | What does fusion cost the one panel model that has it? | **0.21 of 4.43 floor bits** — and the two panel models are in different regimes at the same target | [`phase3-s3-fused-floors.md`](phase3-s3-fused-floors.md) |
| 9 | Did the S2 fine-tune produce a signal map worth spending? | **Yes, with one caveat** — `embed_tokens` scores on nothing and pays 67.8 % of the floor shortfall | [`phase3-s2-phi-signal-map.md`](phase3-s2-phi-signal-map.md) |
| 10 | Is S3's shuffled control actually ablating the signal? | **No — it was a null by construction.** 0 of 129 modules differed from the treatment; it permuted a file the allocator had stopped reading | [`phase3-s3-null-control.md`](phase3-s3-null-control.md) |
| 11 | Did the second S2 fine-tune produce a signal map worth spending? | **Yes, and it exposed a ranker defect** — a singleton role group scores 0.5 against itself, costing the baseline a whole bit on 6.7 % of the model | [`phase3-s2-ministral-signal-map.md`](phase3-s2-ministral-signal-map.md) |
| 12 | Does S3's resume guard know a fresh map from a stale one? | **No, twice over** — it compared against a file it regenerates, and called a metadata reordering a content change | [`phase3-s3-reuse-guard.md`](phase3-s3-reuse-guard.md) |
| 13 | At matched bytes, what are the allocator and the signal each worth? | **+4.31 over uniform at 3.25 bits, split 1.91 allocator / 2.41 signal — and nothing at 4.25, where the maps diverge just as far and buy nothing** | [`phase3-s3-allocation.md`](phase3-s3-allocation.md) |
| 14 | Can the phase-4 text-to-SQL benchmark tell a correct answer from a broken one? | **Only after four defects were removed** — two SQLite-semantics bugs discarding a third of one corpus, a DML block teaching an unscoreable answer format, and a registered task argparse refused | [`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) |
| 15 | Does the training mixture contain the questions the arms are scored on? | **Yes — 189 of the 200 WikiSQL evaluation items are questions in `b-mc2/sql-create-context`**, and the guard that reported the mixture clean could not have fired | [`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §11 |
| 16 | “At matched bytes” — whose bytes? | **The baselines’** — DynQuant’s own format costs 4.25 bits/param at 4 bits against `compressed-tensors`’ 4.15625, so anchoring on its uniform arm would hand it **2.3% more bytes inside the arm whose accuracy is the claim** | [`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §12 |
| 17 | Which of DynQuant’s two prices chose the widths, and on how much of the model? | **44 of 133 modules — 91.54% of parameters — by the rank-product proxy rescaled by 1.807e-17**, because a batched expert bank has no boundary where the Gauss–Newton form exists; the concordance guard reads 1.000 over the other 8.46% | [`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §12 |
| 18 | Does the seven-arm panel driver run? | **Not until six defects were removed** — three of them fail *after* the calibration pass, one would have finished and entered the table as round-to-nearest wearing an AWQ label, the fifth would have run all seven arms over the whole 16 143-item test split because `--limit` is forwarded only when set, and the sixth would have let `--resume` staple in a record scored on another checkpoint | [`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §12 |
| 19 | How many of the six quantized variants can actually be published? | **Four can be written; two can be loaded -- and not the two that were named** — `dynquant export` refused a batched expert bank and said the packed *format* could not hold one; the format always could, and the refusal came from a second copy of the name resolver -- which then refused a second time, on a router, because the copy had been narrowed twice. Both DynQuant arms now write packed at their manifest bytes, size-honest and not yet loadable -- but *not loadable* turned out to mean transformers skipping the unknown `quant_method` and **returning a randomly initialised model with no exception**, measured identically on 4.53.2, 5.10.1 and 5.14.1, none of which has entry-point discovery. An `HfQuantizer` now makes that a hard error, round-trips a dense model to 4.9e-4 of the encoder, and stops a tied `lm_head` crashing on a packed embedding that has no `.weight`. And *needs the grouped path* was two blockers read as one: the kernel that makes batched experts fast, and the object that lets them be held at all. The second is Python -- the parent reaches an expert by **indexing**, so a module registered under the parameter's own name intercepts it and dequantizes 10.5 MiB of a 336 MiB bank per hit. That is 91.5% of this model, and it was also 91.5% missing from the byte denominator, which walked modules and so never saw a tensor no module owns. The two 3-bit baselines were refused for a false reason as well: `compressed-tensors` packs 1-8 bits and round-trips 3-bit fine, but at `32 // 3` values per word it stores **3.2 bits against a label of 3**, and vLLM sizes the same tensor as `Fraction(32, 3)` -- 192 words per 2048-wide row where 205 were written. The count held and the identity flipped. `gptq_4b` and `awq_4b` were the row's "yes -- vLLM and transformers", and they are the two that cannot be published: the recipe reaches 91.5% of this model by **renaming** the banks, `ARCH_TO_2D_MAPPINGS` registers the inverse for `deepseek_v4` and `qwen2_moe` and nothing else, and `lfm2_moe` linearizes through the generic protocol -- so the surgery runs and its inverse does not exist. Measured through the shipped `save` on a four-layer model: 108 packed expert tensors written, all 108 `UNEXPECTED` on reload, both banks `MISSING`, **no exception**, finite logits, and the reloaded bank at **32 distinct values in a 32-value group** where 4-bit allows 16 -- against 13 for the same instrument on a `dynquant quantize` directory. vLLM keys its expert loader on `("w1", "w2", "w3")`, so it does not find them either. `do_save` now refuses on llm-compressor's own predicate, before the calibration pass, and the panel is untouched because `run` never serializes | [`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §12 |
| 20 | Is the model the panel scores the model a person would download? | **It was five defects away, and the fifth was in this report** — a rank-3 test that could not tell a batched expert bank from a `Conv1d` kernel, on a model where **18 of 24 layers are conv**; a packed bank that installed cleanly and then died *inside transformers* at `weight.transpose(-2, -1)`, because 5.14.1 replaces every `*Experts.forward` with a dispatcher whose **default is `grouped_mm`**, so the indexing loop the whole design rests on is not what runs; and the packer and the in-place encoder disagreeing by **0.0082** on the same fp32 bank — 16% of what quantization itself moves — because the scale dtype had **three definitions** that agree on fp16 and bf16 and therefore on every model anyone ships. All three are now zero, or exactly 0 where exactness is the honest bar, verified against the genuine `Lfm2MoeExperts` — which then found a fourth, an export whose own reader refused it, and a fifth that was this report's own claim: it called the dispatch move **free** on the strength of that one-layer 1.79e-07, and at 8B the two dispatches disagree on **1.24% of teacher-forced tokens**, **0.29x** the quantization effect, on the same axis as the panel's margins. Corrected in five places and pinned in code. §11 corrects §8's mechanism: a linearised baseline keeps `_experts_implementation` and its `*Experts` modules, so "no dispatch left" was reasoning about modules while the code read the config — and it corrects itself in turn: the four baselines are not unrecoverable, their own `banks_after: 0` was counted in the process that scored them, so the genuinely unrecorded arms are `bf16`/`dq_4b`/`dq_3b`, which is exactly the re-score set. Then twice more, from asking whether the re-score would *finish* rather than whether it was justified: the driver's guard would have killed it on arm two, and the table's would have blanked **every row in every block** — the re-score clearing every caveat and deleting the numbers they annotate. Comparability is a property of a pair, and `experts.ran` is not part of it: *paired* means the hit vectors index the same items, which a dispatch difference does not break. It is priced on the row instead, and the resulting caveat-free table now prints the one claim holding it up. §8 also finally prices its own title: the linearised arms cost **1.9—2.3x** the banked ones over the same 12,000 items, which re-brackets the re-score at **8.5—17 hours** and makes it the first `eager`-against-`grouped_mm` clock this campaign will have taken. That paragraph then corrected itself within the day: it had offered the 18% between GPTQ and AWQ as a bound on the dequantization confound, calling it "kernel choice alone" — but the two manifests are the same 4 bits at the same group size for the same 4,399,629,312 bytes, so there is no second kernel, only asymmetric-against-symmetric dequant and the length difference two sets of weights produce. **A difference inside dequantization does not bound dequantization**, so the split stays unmeasured, and the two probes that can split it are the re-score and 24 teacher-forced items on bf16 — both with identical weights on either side, hence no dequant in either. The 3-bit half then answered itself from a file written for another reason. The box's progress sampler stamps every 800-item line, so the panel's own log is an interval profile over the same items in the same order — which the four records confirm to the item, 3,063 gretel and 8,937 wikisql each. Aligned block-for-block, linearised-against-banked swings **1.82—3.78x** where a fixed per-forward cost would be flat, capping that cost at 1.82 against an aggregate of 2.55 and leaving **at least 1.40x as decode steps**; and `gptq_3b` against `awq_4b` — both linearised, so the dispatch is held fixed and only the width and weights move — swings **1.12—2.95x** over five shared blocks. A fixed unpack cost is the same work every block, so a 2.6x swing falsifies it outright and **1.12 is the ceiling worth defending** — at most 12% of a forward against an aggregate of 1.62x, leaving at least 1.45x as decode steps. There is a block reading 0.96, which would settle it more strongly still, but it divides by `awq_4b`'s own anomaly — 2.70 where that arm's median is 1.50 and `dq_4b` is flat — and a quotient inherits the reason its denominator was slow, so it is not the number quoted. Either way the 3-bit slowdown is generation length, settled for free where this report had priced it at a re-quantization. The 1.82 also puts an argument under the top of the re-score bracket, which had none: that re-score holds the weights fixed on both sides, so length cannot move and the fixed multiplier is the only one in play — at most 1.82x the banked arms' own seconds and for the *loop* at that, with `eager` under it, which lands the ceiling near **15 hours** rather than 17. §8 then answers its own title: the move it priced is **not required**. `dynquant_experts_forward` indexes a packed bank from inside the grouped path and measures **bit-identical** to `grouped_mm` at bf16 and fp32 — 0.00% of argmax tokens against `eager`'s **1.95%**, at the 8B's own MoE geometry — so the condition every packed figure carried is retired rather than restated, and the pin survives only for linearised baselines, which have nothing left to dispatch. Two false starts are kept because they were the informative part: a tiny geometry where all three dispatches were bit-identical and an earlier draft divided by that zero to print `infx closer` — the zero was real, proven by confirming each dispatch ran, and the fix was to scale width and `k` rather than depth — and a sentinel-offset bug the probe was structurally blind to and a two-expert unit test caught immediately, since bands index a *sorted* array and one over-wide bin displaces every band after it. Six mutations of the forward, six named tests. Then the banked records were finally read for their dispatch, confirming what §11 predicted and widening it: the `experts` key is **absent**, not null, so `_comparability` exempts all five, and the straddle is five of five including `bf16` — no arm in the banked panel chose its dispatch. §10 closes the last of §7's three named byte gaps: the tensors classification *refuses* were recorded with a reason and left out of the denominator, which is **205 KB of 4.4 GB here** and **91.5% of the model** on a MoE whose banks are refused for orientation. §12 answers row 19 in a different container. A foreign grid is a grid: `scale * (q - zero)` *is* `scale * code + offset` with `code = q` and `offset = -scale * zero`, so the four variants `compressed-tensors` cannot hold can be written as DynQuant with the codes carried **unchanged** rather than re-fitted. All six baseline arms of a four-layer `lfm2_moe` published and reloaded through the real `HfQuantizer`: rtn and gptq at **0.0000 code steps** from the weights the in-process arm was scored on, awq at **0.0681** and **0.0240** against a 0.125 budget, where re-fitting the same weights -- the same code path with the encoder swapped -- moves them **0.399--0.518**. Three defects nothing smaller than a real recipe could reach: the recipe's own `weight_scale` / `weight_zero_point` / `weight_g_idx` published as model weights; the tied table written under `lm_head` where the loader reads `model.embed_tokens`, because the first fix gated on `config.tie_word_embeddings` and **`oneshot` sets that to `False` while the storage stays shared** -- a config true of the compressed checkpoint it never wrote; and the first *asymmetric* arm refusing at 6.976e-02 with the weight on its own lattice to **0.0377** of a step, because the reader had worked the integer range out for itself as unsigned `[0, 2^b-1]` where `compressed-tensors` puts *every* integer scheme on the signed band and lets asymmetric ride it with a signed zero point (**-4..3** measured), clamping **7,443 of 12,288** elements onto a rail that does not exist. Imported now, not derived -- the same shape as the six duplicated-registry cases below, one step further out, since a second copy of a *dependency's arithmetic* is a copy nothing in this repository can contradict. And the container is not free: **4.25 bits at 4 and 3.25 at 3** against `compressed-tensors`' 4.15625, so a republished `gptq_4b` costs **+99 MB** for identical codes while a republished 3-bit is *smaller* than the container that could not hold it honestly at all. And the carry check answers only half of it: it proves the export is faithful to the model in memory, but that model is a *second* calibration pass, because `run` scores in process and writes a record rather than a checkpoint. `publish --scored <arm>.quant.json` compares the second pass against the arm's own record -- six flags before the recipe, ten weight fingerprints after -- and the byte accounting alone could not have done it: `gptq_4b` and `awq_4b` agree to the byte on **4.1565** bits, **4,399,629,312** bytes and every parameter count, and differ only in what the recipe did to the weights (**0** modules moved against **2,201**, by **0.0** against **2.890625**). The last document is the one anyone actually reads, so it is generated too: `model_cards.py` builds each arm's README from `panel_table --json-out` and the fine-tune's own record, typing no number, and refuses the ceiling, an unscored arm and an unknown label rather than describing them. Writing it found the table dropping `question` and `same_arithmetic` on the way into json -- survivable while the only reader was a terminal with the dispatch census two blocks up, and not once six Hub READMEs are built from that payload, since the card would publish `separated` with its **0.29x** confound stripped off. Three caveats are emitted from the row rather than from a checklist -- scored-in-bf16 for a map arm, **+2.3%** on disk for a recipe arm, `[^1]` for a flagged pair -- because carrying all three on every card would tell a DynQuant reader their directory is oversized and a GPTQ reader their accuracy came from somewhere else, and both are false. | [`phase4-packed-moe-runtime.md`](phase4-packed-moe-runtime.md) |

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
row — the same tensor, measured — scores 0.9264 and lands on the same 3 bits.

**Corrected 2026-08-06.** That was originally read as *costs this checkpoint nothing*, off one
tensor's width. The budget is shared: at the same substitution the map moves **12 / 5 / 11 / 8**
other modules at 3.25 / 4.0 / 4.25 / 4.5, at byte totals equal to within 0.02%. The tied tensor
holds its width and the tail is rewritten around it. The verifier now reports
`other_modules_moved` beside `changes_width`.

---

## 10. Phase 3 — S3, the control that was ablating nothing

Full report: [phase3-s3-null-control.md](phase3-s3-null-control.md). Found by running the S3
driver's CPU path on real weights, before any arm was quantized.

S3's `shuf` arm permutes the signal within role and re-allocates at the same byte count; the
`dq` − `shuf` gap is what the campaign calls the signal's contribution. The driver permuted
the **stats** file and then passed `--moments` the real, unpermuted sidecar. Since the
Gauss-Newton estimator was added, `move_value` prices every width change from the measured
sensitivity table whenever the module has one and only falls back to the stats-derived score
when it does not — and Phi's sidecar covers **all 129** quantizable modules. The permuted
score was therefore never read. Confirmed on the pre-fix run rather than argued: `shuf3` and
`dq3` differ on **0 of 129 modules**, hash identically, and land on the same byte count and
the same 3.249487442337263 average bits — while `rank3`, which allocates from the score with
no moments at all, moves 65. The control would have reported *the signal does not matter*,
silently, with both arms exiting 0.

Fixed by deriving one permutation and applying it to both artifacts, grouped by
`(role, in-channels, out-channels)` — a no-op on a dense model, and there because a channel
vector of the wrong length does not raise; it makes the module drop out of the sensitivity
table, so the control would ablate coverage as well as correspondence. Nothing downstream had run, so no
reported number changes; §8's 69-of-129 figure is unaffected, its allocation never loading
moments at all. The transferable rule: **a control is defined by what the treatment reads,
not by what it is named after**, and an ablation arm whose map is identical to its treatment's
has not run.

---

## 11. Phase 3 — S2 arm 2, the tensor ranked against nobody

Full report: [phase3-s2-ministral-signal-map.md](phase3-s2-ministral-signal-map.md).
Ministral-8B × Tulu-3 finished in **11.23 h**, 1492 optimizer steps, final train loss
**0.6335**. Structure is sound on all four properties: 254 modules (2 + 36×7) under canonical
names, `grad_norm_count` = 1492 everywhere it exists, `forward_calls` uniformly 23 865,
saliency spanning 1221×.

§9 predicted what would be different: Ministral is untied, so `lm_head` is its own quantizable
tensor **in a role group of one**. It has a complete measurement — 1492 gradient observations,
**rank 1 of 254 on saliency** (1.78× the runner-up) and **rank 6 of 254 on plasticity**, so both
signals agree and this is not a tensor that merely has large activations for scale reasons — and
the shipped per-role ranker computes its percentile against a set containing only itself and
returns **0.5**. Ranked globally, the same measurements score **0.9783**.

At the headline 3.25-bit target that is a whole bit on 6.7% of an 8B model: **3 bits shipped
against 4 bits measured**, paid for by 24 projections dropping 4 b → 3 b at an *identical* byte
total and identical 3.2500 average bits. At 4.0 b the head's own width does not move and **23
other modules do**. With `model.embed_tokens` — same 536.9 M parameters, no gradient signal at
all because the `outer_exact` hook never registers on a frozen embedding fed integer ids —
**13.4% of the model is allocated on a number reflecting no comparison**.

Controlled, because a greedy knapsack near its ceiling could just be chaotic: forcing the same
0.9783 onto 24 ordinary projections moves a **median of 0** modules (max 8 at 3.25, max 2 at
4.0). Nor is it a size effect — `embed_tokens` is the same size and moves **0** at 4.0, because
it already sits at its floor and is not a knapsack candidate, while `lm_head` sits four bits
below its 8-bit floor and its ROI position decides the tail. **The neutral score bites hardest
where the tensor is large and its floor is breached**, which is the regime the headline targets
are defined by.

**Corrected 2026-08-07 — which allocator reads the score.** Those numbers come from
`verify_signal_map.py`, which allocates without a sensitivity table: the **rank-product** path,
which is the `rank` *baseline* arm. `allocate_bits` prices a module from measured sensitivity
wherever the channel moments cover it and from the score only where they do not. Ministral's
sidecar covers 253 of 254 modules. So at the 3.25-bit anchor, all four arms landing on exactly
3 257 925 632 bytes: `lm_head` gets **3 b under rank-product and 4 b under `dq`** — the headline
arm reaches the width the measurement implies on its own, and the singleton handicap costs the
baseline, not `dq`. The tensor that *does* carry a neutral score into the headline arm is
`model.embed_tokens`: no moments, untied, no partner to borrow from, and `dq` puts it at **2 b**
where rank-product left it at 3. Phi's tie is the contrast — it routes `lm_head`'s moments onto
the same tensor, so Phi's `dq3` gives it **4 b**. Untying removes the coverage, not just the
check. And `shuf` cannot ablate either tensor: within-role permutation is a fixed point on a
group of one, so both are assigned identically to `dq` by construction and neither is among the
39 of 254 modules the arms differ on.

**The 4.25-bit anchor corroborates it from the other side.** Both anchors are now complete and
byte-matched (4 260 364 288 B, +0 B widest drift). At 4.25 every role floor is affordable — all
three allocating arms hit **zero violations** — so `rank4` hands `lm_head` its full **8 b** floor
and `model.embed_tokens` its **4 b**, the same widths the sensitivity arms pick. The neutral 0.5
costs the baseline nothing there. The defect is only visible where the allocator is *forced to
choose*, which is §7's finding restated in the negative: the signal earns its keep once the role
floors stop being affordable, and so does the flaw in how it is ranked. Where the allocators part
at 4.25 is instead the **8-bit tail** — `rank4` widens 2 modules, `dq4` widens 37, paid for by
dropping 62 to 3 b against rank-product's 43. The control ablates at both anchors (39 of 254 at
3.25, 28 at 4.25) and is blind to both singletons at both. All of that is asserted against the
committed maps in `tests/test_s3_allocation_arms.py`.

Which map is better is an eval question and an S3 arm. The scorer was **not** changed —
that would move every model's allocation including phase 2's published comparisons, and it is
a decision, not a fix. The *verifier* was: it now enumerates singleton role groups, reports
`informed` apart from `scored`, and measures whole-map movement rather than one width, which
is what forced §9's correction above.

---

## 12. Phase 3 — S3, a resume guard that measured the wrong thing

Full report: [phase3-s3-reuse-guard.md](phase3-s3-reuse-guard.md). Found while clearing the
way for the S3 quantize-and-eval sweep, before any of it was launched.

Six maps are allocated before a single weight is written, three of them priced from the
channel moments at about 1 h 45 m of CPU each — seven hours before the GPU has anything to do.
`--reuse-maps` skips that, and its whole burden is telling a current map from a stale one,
because existence on disk says nothing about *when* a file was written. It got that wrong in
both directions at once.

**It stamped freshness against a file the same run regenerates.** The maps' mtimes were
compared to the stats file each map *names* — but for the three signal arms that is a derived
variant the driver writes at the top of every invocation. The comparison asked *is this map
older than a file I wrote thirty seconds ago*, answered yes, and rebuilt five of six. The fix
stamps against the run's sources — S2's stats and moments and the merged checkpoint — and
leaves the variants their real job, which is the **identity** witness separating `shuf3` from
`dq3`, two arms that name the same model, allocator and group size.

**And it called a metadata reordering a content change.** With the first defect fixed,
`arms.json`'s `rewritten: false` is the only record that a reused map was priced from the
numbers now on disk — and it was a constant `true`. Root-caused, not waved off:
`safetensors.torch.save_file` serialises `__metadata__` from a Rust hash map, and Rust seeds
each map instance separately, so **two writes of the same object in one process differ**.
Measured on the Ministral moments: both 11 519 008 B, payload byte-identical, headers equal as
parsed mappings, 506 tensors with **0 differing values**, 254 metadata entries equal as a
mapping — and the first differing byte at offset **43**, where the key order splits. So
`sha256(file)` is not a content identity for a safetensors artifact, and every hash-based
cache, dedup or provenance check built on one reports a change every time while distinguishing
nothing.

Verified end to end before the sweep launched: 7 of 7 maps reused, both variants
`rewritten: false`, widest byte drift **+0 B** at both anchors. Independently, the seven maps
on the box are sha256-identical to the seven committed — including `map.rank3.json`, which the
aborted dry run had rebuilt from scratch, which is a determinism proof nobody designed.
Nothing downstream had run, so no reported number changes. The transferable rule: **a resume
guard must compare against what it reads, not against what it writes** — and a comparison that
is too strict does not fail safe, it just stops distinguishing anything.

## 13. Phase 3 — S3, what the allocator and the signal are each worth

Full report: [phase3-s3-allocation.md](phase3-s3-allocation.md). Ministral-8B × Tulu-3, eight
quantized arms over two budgets and four benchmarks, plus a bf16 ceiling — **36 eval cells**,
each arm at an anchor byte-identical to its siblings (3 257 925 632 B at 3.25, 4 260 364 288 B
at 4.25, widest drift +0 B).

**At 3.25 bits the method works. At 4.25 nothing does.** `dq3` beats uniform by **+4.31** mean
points and has the best mean at both budgets, but at 4.25 all sixteen comparisons sit between
p = 0.22 and p = 1.00 — uniform is already within 6.15 points of bf16 and the floors stop
binding (182 breaches at 3.25 against 1 at 4.25). §11 predicted exactly this from the
allocation side: **the allocator earns its keep only where the floors stop being affordable.**

**But not for the reason the first draft gave.** It said the 4.25 maps converge; they do not.
`dq4` differs from uniform on **99 of 254 modules** — more than `dq3`'s 96 — and promotes 37 to
8 bits where uniform has none. The null is not a degenerate allocation scoring like uniform
because it *is* uniform; it is a heavy reallocation that buys nothing measurable. The exception
is `rank4`, 82 % identical to uniform with 45 modules moved, which really did stop allocating —
a different null with a different cause, hiding under the same p-values. The correction came
from reading the committed maps instead of inferring their shape from floor counts and eval
discordance.

**The signal's whole footprint is 39 modules of 254.** `dq3` and `shuf3` share allocator,
budget and byte total, so their disagreements are exactly what the measured signal moved: 15 %
of modules over 24 of 36 layers, concentrated in `o_proj` (17), `q_proj` (11) and `gate_proj`
(9), two-way at matched bytes — 14 promoted 2→3 paid for by 11 demoted 3→2. That is what
+2.41 mean points and the +4.78 on GSM8K are bought with.

**The signal is 56 % of the 3.25-bit margin — where on Qwen3.5-2B it was 12 %.** The
decomposition is `dq − rtn = +4.31 = allocator +1.91 + signal +2.41`. Same anchor, same module
granularity, opposite split. Both are real: on Qwen the uniform baseline was catastrophic and
almost any reallocation recovered most of a 25.78-point gap, so knapsack mechanics dominated;
here uniform is merely mediocre and *which* modules to protect is proportionally more of a
smaller answer. **The signal's share grows as the allocator's structural advantage shrinks**,
so the 12 % figure is scoped to its campaign, not retracted. The 4.25 row is deliberately left
undecomposed — dividing a +0.74 margin that is inside noise prints "the signal is 139 % of it",
and `s3_table.py` now refuses to compute the share unless something clears p < 0.05.

**Two of forty-eight paired tests survive Bonferroni**, and the important one is the control
comparison: `dq3` − `shuf3` on GSM8K, **+4.78, p = 4.8 × 10⁻⁵**. At identical bytes with the
same allocator, pricing from the measured signal beats pricing from a within-role permutation
of it — and the control is not a strawman, `shuf3` beats uniform by +1.91 on its own. The other
survivor is `rank3` − `rtn3` on HumanEval at +14.02. `dq3` − `rtn3` on HumanEval (+11.59,
p = 0.0034) does **not** clear correction and is reported as record, not result.

**Negatives, stated as such.** The published rank-product score is a coin flip — it wins
HumanEval by +14.02 and loses GSM8K and IFEval at 3.25, then does nothing at 4.25 — the fourth
campaign in which it fails to reliably beat doing nothing. MBPP separates no arm anywhere (all
four 3.25-bit arms within 1.0 point, discordant counts split evenly); it reads damage, not
allocation, and belongs in no headline that claims an allocator difference. And **no arm wins
all four tasks at 3.25**, so the mean is never quoted without the per-task table beside it.

## 14. Phase 4 — the text-to-SQL mixture, before any of it is scored

[`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) · screened 2026-08-08

Phase 4 quantizes `LiquidAI/LFM2.5-8B-A1B` six ways against a bf16 ceiling. This entry is the
benchmark those seven arms will be compared on, and it is here as its own campaign because
building it found four things, none of which would have shown up as a number that looked wrong.

**Execution accuracy has one failure mode with three shapes.** Two queries that both return
nothing compare equal, so a set of items whose gold returns nothing scores `SELECT 0` near the
ceiling. `COUNT(*)` over an empty schema returns `[(0,)]` and an unmatched `AVG` returns
`[(None,)]` — both pass a naive "did it return rows?" test. Admission requires the database to
hold rows, the gold to find some, and the answer not to be a single all-NULL/all-zero row.

**A third of WikiSQL was being discarded for the wrong reason.** Its condition values are the
annotator's typing, its cells are Wikipedia's, and SQLite `=` on `TEXT` is case-sensitive.
33 % of golds matched nothing and were refused as "gold finds no rows" — a correct refusal with
an incorrect cause. Declaring `COLLATE NOCASE` took it to 0.4 %.

**Gretel is a SQL corpus, not a query corpus.** 10.2 % of test and 11.3 % of train golds are
`UPDATE`/`INSERT`/`DELETE`/`CREATE`. The evaluation already excluded them, but as
`empty_result` — the wrong diagnosis. Training has no row filter, so they survived, and the
scorer reads an answer by cutting at `SELECT`: an 11 % `UPDATE` diet teaches a response format
scored `unparseable` on a zero-floor metric, identically across all seven arms. Closed before
the fine-tune; Gretel's training admission fell from ~78 % to 68.8 %, which is the leak.

**`text2sql` was fully implemented and unreachable.** `dynquant eval --task` carried a
hand-written copy of the registry, so argparse refused the task with a usage error naming the
other six. Choices now derive from the registry. Fixing it exposed a test asserting the
`style` capability against `executes_code` — passing on a coincidence until the first task that
takes a framing and runs nothing.

Measured admission: 2 796 items on the evaluation split, 5 326 on the training split, balanced
per source and round-robin interleaved so a truncated run still sees every corpus.

The same report carries three later screens. The base model reaches **57.75 %** on the mixture
once it is given room to finish reasoning, so there is headroom for damage to show — the first
attempt read 5.50 %, and 34 of the missing points were a truncated reasoning trace rather than a
model that cannot do the task. And the architecture is only partly visible: DynQuant's signal
collection reached **11.6 %** of its parameters and `llmcompressor`'s GPTQ and AWQ reach
**8.5 %**, because 91.5 % of the weights live in batched expert banks that are not `nn.Linear`
modules. Linearizing the banks takes both to 100 %, verified bit-exact against the original
forward.

## 15. Phase 4 — the benchmark's answers were already in the training set

[`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §11 · measured 2026-08-08

The S2 driver has carried a contamination check since phase 3 and it reported nothing on this
mixture. It could not have reported anything: its markers are `("gsm8k", "humaneval", "mbpp")`,
and no SQL corpus name contains one. **An empty result from a check that cannot fire reads
exactly like a mixture that passed.**

Measured against each training pool in full: Gretel is clean in both directions, WikiSQL's own
train split contributes 1 of 200, and `b-mc2/sql-create-context` — a community aggregate
assembled from WikiSQL and Spider, shipping a single `train` split — holds **189 of the 200
WikiSQL evaluation questions**. In the 50 000-row mixture this run would have sampled at seed 0,
38 of those 200 are present.

What that does and does not invalidate was written down before the number arrived, so the number
did not get to decide it. **The A/B stays valid**: all seven arms quantize the same fine-tuned
model, so contamination inflates them equally and the paired test still measures quantization
damage. What it would have cost is the *absolute* accuracy as a claim about text-to-SQL, and the
base→fine-tuned difference as a claim about learning.

**The scan that found it first reported clean.** Its sampled arm compared the rendered *user
turn* — the question wrapped in a schema and a directive — against the bare evaluation question,
so `0 / 200` printed directly beneath the pool scan's `189 / 200`. That is the same failure as
`_CONTAMINATING`, committed inside the tool written to expose it; it is recorded in that
function's docstring rather than quietly fixed.

The filter matches on the **question**, not provenance — the aggregate does not record which
upstream corpus a row came from, and a question match catches an item arriving by any route.
It matches against Gretel's and WikiSQL's **whole** test splits (21 729 rows, 21 681 distinct
keys) rather than the 400 sampled items, so changing `--limit` or the seed cannot undo it
silently. And it drops **before admission**, so a contaminated row never consumes a quota slot
and is never counted as kept. It removed 5 Gretel, 21 WikiSQL and 3 990 create-context rows with
every quota still met.

Both checks are now in the census, and it says what each is worth: the empty
`sources_overlapping_an_eval_task` is kept, the marker list is written beside it, and
`decontaminated` reports per source rather than as a boolean — because on this mixture the
expected number is four thousand, so zero has to be readable as suspicious rather than as clean.

## 16. Phase 4 — the matched-byte anchor has a direction, and it is worth 2.3%

[`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §12 · decided 2026-08-08

GPTQ and AWQ take a width and the bytes fall out of the format; DynQuant takes a size. So "at
matched bytes" has a direction, and the two formats do not cost the same: `compressed-tensors`
writes an fp16 scale plus a zero point at the weight's own width, `16 + bits` per group of 128,
while DynQuant writes an fp16 scale **and** an fp16 offset, a flat 32. At 4 bits that is 4.25
bits per parameter against **4.15625**; at 3 bits, 3.25 against **3.1484**.

Anchoring the panel on DynQuant's uniform arm would therefore have given it **2.3% more bytes at
4 bits and 3.2% more at 3, inside the arm whose accuracy is the claim** — and nothing would have
reported it, because each accounting is correct about the format it describes and both would have
printed a matched line.

DynQuant is pinned to the baselines' counts: **4 399 629 312 B** and **3 332 904 576 B**. The
overhead comes out of its own payload, which is the point rather than a concession — a format that
stores more metadata has fewer bits left for weights at the same footprint, and that is a real
cost. Charging both sides by one set of rules would be wrong in either direction: by DynQuant's,
the baselines are billed for an offset they never write; by the baselines', DynQuant writes
metadata it never paid for.

Two smaller decisions fall out of it. Each DynQuant arm's realised size is read back from the map
the allocator wrote rather than taken from the request, because `--target-size` is a *ceiling* and
the drift that actually happens is downward — so the tolerance check is on the absolute value,
where a signed one would wave through the only failure that occurs. And every arm is a subprocess
of one interpreter, refusing to start if llm-compressor is not importable from it: per-arm
environments would score the baselines and the DynQuant arms under two transformers versions,
which is a difference in the measuring instrument reported as a difference between methods.

Wiring the panel's own guard through the eval command's `_comparability` then exposed a hole in the
contract. Every field in `PAIRING_FIELDS` comes off the command line, so seven arms from one driver
cannot differ in any of them — but **`prompt_style` does not**. `--prompt-style auto` is answered by
the tokenizer, so a quantized checkpoint whose saved tokenizer lost its chat template is asked
bare-text questions while the ceiling is asked chat questions, with byte-identical commands on both.
That failure is already measured here: it put Ministral-8B-Instruct at 24.77% on IFEval against
Phi-4-mini's 68.76%. IFEval and the code tasks record their resolved style; `text2sql` did not.
`DETAIL_PAIRING_FIELDS` now reads it out of the record's `detail` block, where a task's own metrics
already go, and absence is exempted from the "this run always writes it" guard but still refuses
against a record that has one — "unknown" is not "the same".

The same pass caught the panel about to run at a budget nobody chose. `--max-new-tokens` was left
unset on the reasoning that an inherited default is inherited identically by all seven arms; there
are two defaults, the CLI's task spec says **320** and the in-process chat config says **384**, and
the panel routes through the CLI. All seven would have been consistent, pairable, and 704 tokens
under the 1024 the ceiling was meant to establish — and a truncated query does not near-miss, it
fails to parse, so a binding budget is a floor under accuracy that binds hardest on the most
damaged arm. The panel now states 1024 on every command, refuses a *ceiling* that was still
deliberating at the cap, and pairs after each arm rather than after all seven. A censored
quantized arm is left alone: that one is the finding, not a defect in the run.

Then the DynQuant arm was rehearsed on a 38 M-parameter `lfm2_moe` built from the real
`config.json` with every dimension shrunk — same module tree, same fourteen 3-D tensors, random
weights, four problems, CPU, four minutes. Allocation was healthy (−0.039% off its anchor, all ten
bank tensors priced) and then `eval --map` refused the map `inspect --save-map` had just written,
as "10 module(s) this model does not have". Two defects, both on the arms scheduled fourth and
seventh, both of which would have surfaced roughly four hours into the real seven-hour panel. The
first is **"named_modules misses raw parameters"** in its third location: the
pre-flight guard `check_map_covers` — which also fronts `quantize` and `export` — resolved names
with `get_submodule` while the quantizer behind it resolved them correctly, so it refused maps the
next stage would have applied, for 91.5% of this checkpoint. It now calls the quantizer's own
resolver, because a guard that predicts what the next stage will do should call that stage's
resolver rather than reimplement it. The second is not a bug: the **packed runtime cannot hold a
weight that is not a module**, and the grouped path is P8. So `dynquant eval` gained
`--map-apply {pack,encode}` — `pack` unchanged and still the default, `encode` running the identical
encoder and writing the reconstruction back in the compute dtype, pinned bit-identical to `pack` on
a bf16 `Linear` where both apply. The MoE arms encode, the record says which mode ran, and the byte
figures still come from the allocator's priced map rather than from what the scored model holds.
The general lesson is the rehearsal, not either fix: a structural double costs minutes and issues
every command the expensive run will, and the only thing it cannot check is the answer.

The step after the panel is a table, so the table was written before the panel — nine tests and an
eight-mutation harness against a synthetic seven-arm run, no GPU and no model load. Its size column
reads the manifest rather than the scored model, because the DynQuant arms are resident at fp16
under `--map-apply encode` and a measured column would print 16 bits for the arm whose compression
is the claim. It re-checks every arm's drift from its anchor **in both directions** and, past 0.1%,
refuses to print the comparisons at all rather than footnote them — a panel that is not byte-matched
cannot support a table, and the signed version of that test was the one mutation that initially
survived, because under-budget is the failure that actually happens. And its twelve comparisons are
Holm-corrected inside two blocks — six head-to-head at matched bytes, six against the ceiling — with
the block sizes printed and the verdict following the adjusted p: in the fixture, two 4-bit
comparisons are significant raw (0.0309, 0.0243) and neither survives (0.0972), which is the case
the correction exists for. Writing it first is the cheap half of the same lesson as the rehearsal:
the formatting decisions get made against a panel that can be re-run in seconds, not against the one
that cost seven hours.

## 17. Phase 4 — the concordance reads 1.000, over 8.5% of the model

[`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §12 · measured 2026-08-08

The allocator prices a module two ways. Where the fine-tune left channel moments it uses measured
Gauss–Newton sensitivity; where it did not, the rank-product proxy, rescaled onto the sensitivity
scale by a ratio of medians. On LFM2.5-8B-A1B that split is **89 modules measured against 44
proxied — 8.46% of parameters against 91.54%**, because a batched expert bank's forward spans two
matmuls and a non-linearity, so no module boundary yields the `dY = dW x` pairing the Kronecker
form needs. The banks are not unmeasured by omission; they are unmeasurable by construction, and
178 moment tensors covering exactly 89 modules with zero bank keys is the confirmation.

That makes the rescale multiplier — **1.807e-17** here, identical at both anchors, as a property of
the two price populations rather than of the budget — the constant that decides where 91.5% of the
parameters sit in a heap ordered against the other 8.5%. The maps show how completely: at the
4-bit anchor every module above 4 bits is a measured one and every module below 4 bits is a
proxied one. Until this campaign that number appeared in no artifact — not the saved map, not the
panel record, not the manifest. It was a `logger.debug` line in a run whose driver captures stdout
at WARNING.

The within-role concordance that exists to catch the supplement's headline defect — an allocator
that produced a plausible bit map while never reading the scores — reports **138 of 138 pairs
agreeing, 1.000** at the 4-bit anchor, and every one of those pairs comes from `attn.o`, `ssm.in`,
`mlp.gate` or `mlp.up`. None comes from an expert bank. The guard is true and it is silent about
the mass of the model, including every module whose floor the 3-bit anchor breaches.

Two `inspect` paths were worse than silent. Per-width score statistics were computed over all
members of a group with unmeasured modules padded to zero, so a group of nothing but expert banks
printed `min = median = max = 0.0` — indistinguishable from a group the signal measured and found
worthless, on 2.11 G parameters at the 4-bit anchor and 5.89 G at the 3-bit one. `narrowest` had
the same error in a different shape, showing the widest tensors in the model at the bottom of a
ranked list with a score of zero. `BitMap` now carries a `Pricing` record that `_map_payload`
writes into the saved map and `panel_table` prints beside the width histogram; width statistics
cover only the members the quantity covers; and the allocation is bit-identical before and after,
which is what an observability fix should be. Four mutations added, 17 of 17 caught — including
the one that reports the proxied share as a module count, since 44 of 133 reads like an edge case
and is 91.5% of the checkpoint.


## 18. Phase 4 — seven arms on 38 M parameters, and the six they stopped

[`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §12 · measured 2026-08-08

The panel is seven arms over one 8 B checkpoint: a bf16 ceiling, GPTQ and AWQ at two widths each,
and DynQuant at two anchors. Nothing had ever run the *driver* — arms in sequence, records into the
manifest, manifest into the pairing guard and the table. Rehearsing it on a structurally identical
38 M double (the same `lfm2_moe` config with every dimension shrunk, random weights, four problems,
32 new tokens, CPU) costs about four minutes a pass. It failed four times, and reading the driver against the real split and the real resume path afterwards found two more the rehearsal could not reach. Reading the
driver against the real split afterwards found a fifth the rehearsal could not have reached.

**`run` could only load on the GPU.** `--device` existed on `linearize` alone; the one subcommand
the driver invokes hardcoded `device_map="cuda"`. The box's GPU is held for hours by the fine-tune
that produces the signal being scored, so the only machine the panel is scheduled on was the one
machine it could not be rehearsed on — which is why the rehearsal had been deferred rather than
run.

**The provenance qualification reached the tokenizer.** Baseline arms set `--model` to
`<merge>#gptq-4b-g128` so six records from one checkpoint do not all claim to be the same weights;
`--tokenizer` defaults to `--model`, and `from_pretrained` rejected it as a Hub repo id. It fails
after the calibration pass — on the real panel, four arms × a 256-sample pass over 8 B parameters,
each dying with the quantized weights already in memory.

**AWQ had no mappings for this architecture, and would not have said so.** `Lfm2MoeForCausalLM` is
in neither of llm-compressor's two registries — the dynamic hybrid-stack builder is the right shape
but requires `linear_attention` in `layer_types`, and this model says `conv` — so the Llama defaults
applied, and they are wrong in both halves of every block (`operator_norm` not `input_layernorm`,
`ffn_norm` not `post_attention_layernorm`, `out_proj` not `o_proj`). The visible failure is a raise
on a partially matched set. The invisible one is the reason the fix is a *count* and not a regex: a
mapping that matches **nothing** does not raise. It is logged at `debug` and skipped, and the arm
finishes as round-to-nearest under an AWQ label. Predicting the set count from `layer_types`,
`num_dense_layers` and `num_experts` — `[2, 2, 6, 18, 22, 704]` over six mappings — turns that
silence into a pre-calibration abort. The 704 is the one that would have been quietly wrong:
`match_modules_set` groups by lowest common ancestor, so an expert-local pair yields one set per
expert, not one per layer.

**Upstream's grouped-query guard is inert here.** `v_proj → out_proj` cannot be smoothed under
GQA — `v_proj` emits `kv_heads × head_dim` rows and `out_proj` consumes `heads × head_dim`.
llm-compressor drops the pair, but its check reads `balance_name.endswith(".o_proj")`, so on this
model `_smooth` reached `weight[-scales.size(0):]` with 256 scales for 128 rows. The pair is now
conditional on `kv_heads == heads` from the config, and the consequence — `self_attn.out_proj`
quantized unsmoothed — is written into the record rather than assumed.

Both AWQ arms now report **6 mappings, 138 of 145 Linear modules smoothed**, the seven that are not
named by suffix, and 145/145 weights rounded. Nineteen tests and sixteen mutations across the four
defects, all caught — after the first mutation pass caught nine of ten. The miss is the instructive
one: the selection test checked that every balance layer sits under a selected block but not the
converse, so deleting a mapping's layer scope, which selects all 24 blocks where 6 belong, passed.
Set equality in both directions closes it.

All seven arms then ran into the table. Every accuracy is 0.0%, which is correct and meaningless —
38 M random weights answering four text-to-SQL problems. The rehearsal measures plumbing, and the
plumbing is what four of the real panel's GPU-hours would otherwise have been spent discovering,
one arm at a time, each time after the expensive part.

**And the fifth, which the rehearsal supplied rather than tested.** Every setting in `eval_flags` is
stated on every arm's command line, because a setting left to a default is one two arms can disagree
about while their commands read identically. `--limit` is the exception: it is forwarded only when
set, and unset means the whole test split — **16 143 items**, on each of seven arms. The rehearsal
ran four problems at 32 new tokens, so it *passed the flag whose absence is the defect*. That is
general, and worth stating once: a rehearsal is only worth running if it is cheap, and what it
passes on the command line to make itself cheap is exactly what it cannot test. The launcher now
refuses `--go` without an item count, and a 128-item timed probe prices an arm before seven are
spent — which also catches a merge that landed below the 57.75% the base model scored, from five
minutes rather than from a finished bf16 ceiling.

**And the sixth, which no rehearsal could reach, because it needs a second run.** The panel launches
with `--resume`, and the whole of what resume checks is that the record file exists. `check_pairable`
looks like the guard against a foreign one and is not: it compares records through the fields a
paired test needs — task, backend, split, shots, shot seed, limit — every one of which describes the
problem set, and none of which names the model or can be older than anything. Two records scored on
two different merges at identical settings pair perfectly. `check_resumable` now refuses a directory
whose manifest names different inputs, and any record older than the inputs it claims to have
scored, charging the signal file against the two DynQuant arms and not the four baselines that never
open it — because condemning all six would price the cheapest correct fix at a whole new panel,
which is how a guard teaches people to pass `--resume` less carefully rather than more.

## 19. Phase 4 — the refusal that blamed the format

[`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §12 · fixed 2026-08-09

Six variants were to be published and only two could be written. The two 3-bit baselines have no
container — `compressed-tensors` has no 3-bit packed form, so the only thing `save_pretrained`
could write is a full-size bf16 folder wearing a 3-bit label. The two DynQuant arms were blocked
by something else: `dynquant export` raised on every batched MoE expert bank, saying the packed
format could not represent one. That was written into the report as scheduled kernel work, and it
was wrong.

`QuantTensor` had carried `logical_shape` for exactly this case all along. An `[E, out, in]` bank
flattens to `[E x out, in]` rows, packs, accounts for its bytes and restores rank three on
dequantize. What refused it was a **second copy of the name resolver**: the export path had built
its own on `get_submodule`, which cannot reach a bare 3-D `nn.Parameter`, while `quantize_model`
had been resolving both cases correctly through `target_tensor`. The blind spot arrived wearing
the costume of a capability limit, because whoever wrote the refusal was reasoning from the same
wrong copy — the third instance of this shape in the package, after the eval task argparse
refused and the test that asserted `style` against `executes_code`.

Verified before the claim changed: a `[8, 256, 128]` bank exported at 4 bits writes
`.qweight (2048, 16)`, `.scales (2048, 1)` and `.offsets (2048, 1)` — 139 264 bytes for
262 144 parameters, **4.25 bits per weight** — with `logical_shape` preserved and the raw
parameter key absent from the shards. The fix had its own trap, mutation-checked by removing the
line that avoids it: a bank belongs to no tie-alias group, so unless the export loop marks the
bank's own state-dict key consumed, the dense pass writes every value again at fp16 and the
quantized directory comes out larger than the checkpoint it compressed.

Half the original claim survives and is the half worth keeping separate. The packed *runtime*
still cannot load a bank — it swaps `Linear` and `Embedding` modules and there is none to swap
— so a DynQuant directory is **size-honest today and loadable after P8**. Two facts about one
artifact, and both now travel with the directory as `ExportReport.banks` and as `expert_banks` in
the manifest, because whoever uploads the folder will have the folder and not the report.

Run against the real architecture rather than a grafted Llama, the same duplicate refused a second
time one line lower: an `Lfm2MoeTopKRouter` owns a plain `[num_experts, hidden]` weight and is not
a `Linear`, so a whitelist `target_tensor` never had rejected five routers the pre-flight guard
accepted and the quantizer encodes. **A copy narrows, and this one narrowed twice** — so the
duplicate is gone rather than widened again. The full 38 M map then exports whole at
**4.1576 average bits against a predicted 4.1585**, and the packed prediction reconciles to the
byte. Only rank-1 tensors go unpriced: 205 056 B on the real model, 0.0047% of the 4-bit
anchor and 21x inside the panel's match tolerance, and the same set `llm-compressor` counts as its
211 712 fp16 parameters.

## 20. Phase 4 — the artifact and the measurement were five defects apart

[`phase4-packed-moe-runtime.md`](phase4-packed-moe-runtime.md) · fixed 2026-08-09, extended 2026-08-10

The panel scores the **encoder**, `quantize_model(in_place=True)`, because a matched-byte panel
needs one GPU pass per arm and nothing on disk. What a person downloads is the **packer**. Those
are two implementations of one format, and the panel’s evidentiary value is entirely the claim
that they are the same object. Run against the genuine `Lfm2MoeExperts` rather than the synthetic
copy the unit tests use, they were not, in three ways that hid behind each other.

**Rank 3 is not a bank.** `resolve_target` accepted any rank-3 `nn.Parameter` as a batched expert
bank. LFM2.5-8B-A1B is a hybrid whose **18 of 24 layers are short convolutions**, and an
`nn.Conv1d` kernel is `[channels, 1, width]` — rank 3, and a `[1, 3]` strip of it per "expert".

**The indexing loop is not what runs.** The design rests on a counted property: 49 of the 52
`*Experts*` classes in transformers 5.14.1 reach an expert by indexing their bank. That is a count
of what those classes’ `forward` methods contain, and `integrations/moe.py` **replaces** each one
with a dispatcher reading `config._experts_implementation` against `ALL_EXPERTS_FUNCTIONS`. The
loop is only the `eager` entry; the default is `grouped_mm`, which hands the bank whole to
`torch._grouped_mm`. So the bank installed correctly, passed every test, and raised
`'DynQuantExpertBank' object has no attribute 'transpose'` at its first real forward. Both the pack
path and the load path now move the model through its own `set_experts_implementation`.

**And the move is not free, which this report said it was.** The line above used to read *"the
move is free: eager and `grouped_mm` differ by 1.79e-07 against a quantization effect of 0.0101"*
— a true measurement of a 4-layer 6-expert fp32 model, copied into five places and used
to license a claim about an 8B one. On LFM2.5-8B-A1B, teacher-forced over 24 real text-to-SQL
items, the two dispatches disagree on **1.24% of tokens** against a quantization effect of 4.33%:
**0.29x**, confirmed independently by peak-logit deltas at 0.28x. The mechanism is that a top-k
router turns a last-bit numeric difference into a *different set of experts* — layer 2
routes bit-identically, layer 23 agrees on 7% of its slots. Four arms of the panel had already
landed straddling it, because `llm-compressor` linearises an expert bank into per-expert `Linear`
modules and a baseline therefore computes eager while an encoded dq arm computes `grouped_mm`.
`dynquant eval` now pins every arm to `eager` — the only dispatch a linearised baseline
and a packed artifact can both run — and records the dispatch in the record, where
`EXPERTS_PAIRING_FIELDS` refuses to pair two arms that ran different arithmetic.

**Two corrections to that, one of them to the sentence above.** The pairing field refuses to pair
two arms that *recorded* different arithmetic, and the banked arms recorded nothing: the panel clone
is pinned at `4109dcc`, which predates the pin, the flag and `use_eager_experts` alike, so the
`experts` key is **absent** and `_comparability` exempts it. Checked against the five records rather
than assumed, the straddle is also wider than "four arms" — it is five of five, `bf16` included, so
no arm in the banked panel had its dispatch chosen rather than inherited. The bf16-to-DynQuant
margin survives that intact, both sides being on `grouped_mm`; the DynQuant-to-baseline margins do
not, and the re-score is what settles them.

**And the pin is no longer what a packed artifact needs.** The observation that the right home for
the P8 kernel is *inside* `ALL_EXPERTS_FUNCTIONS` turned out to be the fix rather than a plan for
one: `dynquant_experts_forward` is `grouped_mm`'s own forward — same sort, same offsets, same
sentinel mask, same single reduction over the k axis — with `bank[e]` substituted for the whole-bank
read. At the 8B's MoE geometry it is **bit-identical** to `grouped_mm` at bf16 and fp32 where
`eager` disagrees on **1.95% of argmax tokens**, so a downloaded checkpoint no longer trades
arithmetic for a packed bank. The pin stays on `eager` for panels only, and only because a
linearised baseline has nothing left in it to dispatch. What is still owed is the fast path.

**One format, two encodings.** The packer and the encoder encoded the same fp32 bank 0.0082 apart.
Three copies of the scale-dtype rule: the packer used the weight’s own dtype, the exporter used
fp16 unless already half, the encoder passed nothing and inherited the exporter’s. All three agree
on fp16 and bf16 — so on every model anyone ships — and the disagreement waited for an fp32
test model. The **fourth** instance of *a second copy of a registry agrees until it doesn’t*, and
the most expensive, because the two copies were the two things being compared. Settled on 16-bit
metadata in one shared `storage_dtype()`, not on accuracy grounds but because `budget.py` prices
every quoted bit-width against `metadata_bits: int = 16` and fp32 scales would put a model 0.25
bits/weight above its own manifest; the first attempt, on the weight’s own dtype, turned 166 tests
red and that is what identified the budget as the fourth copy. The one caller that genuinely needs
the parent’s dtype — `DynQuantExpertBank.__getitem__`, which is asked for a weight before any
activation exists — is told at construction and holds it in a non-persistent buffer so `.half()`
carries it.

Afterwards, on the genuine class: `max|bank[e] - encoder[e]| = 0`, packed-vs-encoder logits `= 0`,
and a bank exported and reloaded through `from_pretrained` returns `0`. Eleven mutations across the
two fixes, all eleven red. The limit is scale, not fidelity: this is the real *class* at 4 layers
and 6 experts in fp32, and the 8B’s own export-and-reload is running beside the panel.

**And the same writer, pointed the other way.** Four of the six promised variants cannot be written
in the container their own recipe produced — `lfm2_moe` has no registered inverse for the expert
rename, and 3 bits does not divide 32. Both are properties of that container, neither is a property
of the numbers, and DynQuant's own format holds the numbers exactly: `scale * (q - zero)` is
`scale * code + offset` with `code = q`, so a baseline's grid is *carried* rather than re-fitted.
Re-fitting is the control and it is not a strawman — the same code path with the encoder swapped —
and it moves the weights 0.399—0.518 code steps where carrying moves them 0.0000 on every symmetric
arm. The residual on the two AWQ arms, 0.0681 and 0.0240, is the bf16 offset dtype rather than the
carry: a symmetric arm's `scale * -8` is exactly representable and an asymmetric arm's
`scale * -11` is not. Getting there cost three defects that only a real `llm-compressor` run could
reach, and the third is the family this record keeps enumerating — a range worked out here instead
of asked of the library, green against every symmetric arm and wrong about the first asymmetric one.
What is still owed is the 8B: this is the real architecture at four layers, and the publish path has
not been run at scale because the panel has the GPU.

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
| [`experiments/phase3/`](../../experiments/phase3/) | the phase-3 campaign: the headroom screen, the S2 signal maps, and S3's allocation maps and 36 eval cells with per-item hits |
| [`experiments/phase4/`](../../experiments/phase4/) | the phase-4 campaign: the text-to-SQL admission screen for both splits, per source, with the refusal broken down by cause. Panel artifacts pulled off the non-volume box while it ran live in [`s4_panel/`](../../experiments/phase4/s4_panel/) (records with per-item hits, quant manifests, decode probes, leakage scans) and [`s4_runs/`](../../experiments/phase4/s4_runs/) (the signal file and the measured expert-bank moments). |
| [`docs/format-spec.md`](../format-spec.md) | the checkpoint format contract these experiments write and read |
| [`docs/legacy-audit.md`](../legacy-audit.md) | what was wrong with the supplementary code, defect by defect |
| [`decode-neutrality.md`](decode-neutrality.md) | the checkpoint's own `generation_config` reaching a "greedy" decode: how the phase-3 G4 gate found it, which campaigns it does and does not touch, and why the fix took two attempts — the first was correct on transformers 4.x and inert on the 5.x the campaign runs. Ends with what the fixed gate measures: −0.83 points, and a ±1.00 bound GSM8K is too small to resolve |
| [`runtime-parity-gap.md`](runtime-parity-gap.md) | the other half: a GSM8K stop sequence the model never wrote back, generations running on into invented problems, and the two explanations that fitted the data and were wrong (padded batching, different inputs) |
| [`docs/sglang-integration-plan.md`](../sglang-integration-plan.md) | the SGLang plugin design and its S0–S8 staging |
| [`CHANGELOG.md`](../../CHANGELOG.md) | every change, in order, with the reasoning |
