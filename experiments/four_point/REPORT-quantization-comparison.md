# DynQuant against GPTQ, AWQ, RTN and bitsandbytes NF4

**Complete report. 45 comparison arms plus 2 determinism re-runs and 5 packed-path records, two
models, two tasks, one harness.**

Measured 2026-07-26 → 2026-07-30 on an A100 80GB. Every number in this report was read back off
disk from the run records at write time — with exactly one exception, flagged where it appears in
§6.7 — and none is quoted from memory or from an earlier draft. The per-arm records are the source
of truth and live in `/workspace/runs/{model}_{task}/stage8_*.json` on the measurement box, with the
packed-path records under `stage6_*`.

This report is self-contained. The working document with the full derivations, the failed attempts
and the intermediate reasoning is [RESULTS-external-comparison.md](RESULTS-external-comparison.md);
this is the settled version.

---

## Bottom line

Seven findings, in descending order of how much they should change your mind about the method.

1. **At 2.42× compression DynQuant is lossless and GPTQ is not.** At byte-identical size
   (1.4512 GiB vs 1.4514 GiB) DynQuant scores 89.71% against GPTQ's 88.97% — **+0.73, p = 0.017** —
   and is statistically indistinguishable from bf16 (−0.04, p = 0.905) where GPTQ is not
   (−0.77, p = 0.012). This is the campaign's single clean win. It is one tier, one task, one model.

2. **At 3.8× compression DynQuant ties everything.** Across three panels and four baselines, 11 of
   12 comparisons against DynQuant's 4.25-bit arm fail to separate; the twelfth (RTN on
   Qwen/Banking77, +1.14) favours DynQuant. On the tied-embedding model the tie is achieved at
   **1.73× fewer bytes** than the baselines' shipped configuration. The tie is the result; the size
   is the win.

3. **At 4.9× compression GPTQ wins, on all three panels.** −1.34 (p = 4.2e-04) on Qwen/CaseHOLD,
   −1.04 (p = 0.011) on Qwen/Banking77, −0.78 (p = 7.1e-03) on Mistral/Banking77. At matched bytes
   GPTQ is simultaneously **3.1% smaller and 1.34 points better**. This is the method's real
   weakness and it replicates across model, task and scale.

4. **DynQuant beats AWQ at 3 bits by 2.5–3.3 points on both Qwen tasks** — +3.33 (p = 7.5e-14) and
   +2.47 (p = 3.3e-07) — at half the bytes, and by +3.39 (p = 1.5e-13) against AWQ's matched-byte
   arm. Its clearest advantage over a calibration-based method, though on Mistral at matched bytes
   the two are a dead tie (+0.03, p = 1.00).

5. **The margin over naive rounding is 88% architectural prior and 12% training signal.** At
   3.25 bits DynQuant beats byte-matched RTN by 25.78 points. Holding the allocator, the graph, the
   floors and the budget fixed and replacing only the signal scores with a constant splits that into
   **22.62 points of allocation and 3.16 points of signal** (p = 1.9e-15). Both halves are real and
   separated; the sizes are very different.

6. **Above ~4 bits the training signal is worth nothing measurable.** At 4.25 bits the same
   ablation moves 12 of 187 widths and buys +0.19 points (p = 0.15). The operating rule that follows:
   DynQuant earns its keep only when the budget is tight enough that the role floors cannot all be
   paid for. Above that point it is an expensive round-to-nearest.

7. **The accuracies above survive real kernels, and the bytes are exact; the speed is not there
   yet.** Run packed on CUDA, the two Mistral arms differ from their simulated counterparts by
   **5 and 1 predictions out of 3,080** (Qwen: exact, 4468 = 4468 of 5,314) — so §3–§6 measure what
   the kernels actually compute. On-disk size matches the accounted figure **to the byte** and peak
   VRAM is **3.55× / 4.55×** below bf16. But decode is 10–15% slower than bf16 at batches 1–8 and
   3.1–5.7× slower at batch 32, from per-launch overhead and a missing large-M path — not from the
   bit allocation (§6.7). No baseline was run packed, so this report contains no speed comparison
   against GPTQ or AWQ.

---

## 1. Setup

### Models and tasks

Each model is paired with the task it was originally validated on in this project, so no pairing is
a fresh choice made after seeing results.

| | Qwen3.5-2B-Base | Mistral-7B-Instruct-v0.3 |
|---|---|---|
| parameters | 1,881,825,088 | 7,248,023,552 |
| bf16 size | 3.5052 GiB | 13.5005 GiB |
| `embed_tokens` / `lm_head` | **tied**, 509,108,032 params = **27.054%** of the model | **untied**, 268,701,696 params = **3.707%** |
| task | CaseHOLD (legal holding selection, 5-way) | Banking77 (intent classification, 77-way) |
| eval set | 5,314 examples | 3,080 examples |
| also run on | Banking77, 3,080 examples | — |

Three panels result: **Qwen/CaseHOLD** (primary, 21 arms), **Qwen/Banking77** (task transfer, 10
arms), **Mistral/Banking77** (scale and untied embedding, 14 arms).

Every arm in a panel starts from the same fine-tuned checkpoint and is scored through the same
`common.run_eval` — same prompt, same exemplars, same decode settings, same scorer. The comparison
is shared by construction, not by having been copied carefully.

### Baselines

GPTQ, AWQ and RTN come from **llm-compressor** (the vLLM project's quantization library), driven by
[stage8_baselines.py](stage8_baselines.py). NF4 comes from **bitsandbytes** via
[stage8_bnb.py](stage8_bnb.py).

| | setting |
|---|---|
| calibration | 256 rows from the task's own train split, `seq_len` 1024 |
| grouping | `group_size` 128, `targets=["Linear"]` |
| symmetry | symmetric for GPTQ and RTN, asymmetric for AWQ |
| GPTQ | `dampening_frac=0.01` (llm-compressor's default, named explicitly) |
| `ignore` | `["lm_head"]` for the shipped-convention arms, `[]` for the `_head` arms |

Two things about this harness are worth stating because they are load-bearing.

**Every baseline arm is verified to be actually rounded.** `llm-compressor`'s `oneshot` fits scales
but does *not* write rounded weights back for any recipe except GPTQ — so an in-process eval scores
the *original* checkpoint and reports a suspiciously good number. The harness materializes the
rounding itself, counts how many modules moved, and then runs a fixed-point check on a spread of
modules: re-quantizing an already-quantized weight must be a no-op, and a tensor that was never
rounded fails that. An arm that fails raises `SystemExit` rather than reporting a number
([stage8_baselines.py:330-352](stage8_baselines.py#L330-L352)).

`weights_moved` is recorded with every arm and is expected to be zero for GPTQ and equal to the
module count for RTN and AWQ. **Every arm matches that shape exactly**: across all 24 llm-compressor
arms in the three panels, GPTQ reports `0/186`, `0/187` or `0/224` with `max_weight_delta = 0.0`
(already rounded), and every RTN and AWQ arm reports `186/186`, `187/187` or `224/224` (rounded by
the harness). No arm scored an unquantized checkpoint.

**Bytes are accounted on one convention for all methods.** `accounted_bytes` counts
`numel × bits + (numel / group_size) × meta_bits` for every quantized tensor and 16 bits per
parameter for everything left alone, including the embedding. This is the number quoted throughout;
it is not each library's self-report.

### How accuracy is compared

Paired McNemar on per-example correctness, with a 95% confidence interval on the paired difference.
`flips a/b` counts examples the first arm got right and the second wrong, and vice versa. Two arms
can have identical accuracy and still differ: on Qwen/CaseHOLD, GPTQ 4-bit and bf16 both score
89.74% with **58/58 flips** — equal accuracy, 116 different predictions. Accuracy ties are not
behavioural ties, and nothing in this report claims otherwise.

---

## 2. The accounting problem that has to be settled first

**A baseline labelled "4-bit" is not 4 bits, and on a tied-embedding model it is not close.**

`llm-compressor` targets `Linear` modules. `embed_tokens` is an `Embedding`, so it is never touched;
`lm_head` is excluded by the default `ignore=["lm_head"]`. On Qwen3.5-2B those are the *same tensor*
— `[248320, 2048]`, **508,559,360 parameters, 27.025%** of the model's 1,881,825,088 — and it stays
at 16 bits. Adding the 1-D norm weights (106,304) brings the directly counted unquantized residue to
508,665,664, or **27.030%**.

The `f` in the table below is the residue **back-solved from the measured effective bits** rather
than counted, which is why it reads 27.054%: `f = (7.3605 − 4.15625) / (16 − 4.15625)`. The two
differ by ~442k parameters — further small modules `llm-compressor` declines to quantize — and the
gap is immaterial to every conclusion here, but the counted and the fitted figure are not the same
number and this report should not imply they are.

The effective-bit arithmetic recovers this exactly. With `group_size` 128, a 16-bit scale and a
4-bit zero point, a quantized parameter costs `b + (16+4)/128 = b + 0.156` bits. With an fp16
residue fraction `f`:

| model | `f` measured | b=4 predicted | b=4 measured | b=3 predicted | b=3 measured |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-2B (tied) | 27.054% | 7.360 | **7.3605** | 6.625 | **6.6253** |
| Mistral-7B (untied) | 3.707% | 4.595 | **4.5953** | 3.625 | **3.6249** |

The `_head` arms (`ignore=[]`) drive the residue to **0.029%** — the LayerNorm weights, which no
method quantizes.

So the phrase "4-bit GPTQ" means 7.36 bits on the small tied model and 4.60 bits on the large
untied one. DynQuant quantizes every quantizable tensor including the embedding, so its 4.25-bit
map really stores 4.2486 bits.

**This is the single largest confound in the whole comparison and it is architecture-dependent.**
Every byte-ratio claim below therefore comes in two versions:

- **shipped convention** — what a user actually gets from `llm-compressor` today, `ignore=["lm_head"]`.
- **matched bytes** — the `_head` arms, `ignore=[]`, where the baselines quantize the tie too and
  DynQuant's byte advantage disappears by construction.

Both are reported. Only the matched-byte comparison answers "is the allocation better"; only the
shipped-convention one answers "what happens if I use these tools as they ship".

---

## 3. Panel A — Qwen3.5-2B-Base / CaseHOLD (primary, 21 arms)

bf16 reference: **89.74%** (4769/5314), 3.505 GiB.

| arm | method | eff. bits | GiB | ratio | acc % | correct | Δ vs bf16 | p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gptq_4b_head | GPTQ, tie quantized | 4.1597 | 0.9113 | 3.85× | **89.76** | 4770 | +0.02 | 1.00 |
| gptq_4b | GPTQ, shipped | 7.3605 | 1.6125 | 2.17× | 89.74 | 4769 | +0.00 | 1.00 |
| fp16 | — | 16 | 3.505 | 1.00× | 89.74 | 4769 | — | — |
| **dq_iso6p63** | **DynQuant** | 6.6247 | 1.4512 | 2.42× | **89.71** | 4767 | −0.04 | 0.905 |
| **dq_iso7p36** | **DynQuant** | 7.3602 | 1.6123 | 2.17× | 89.63 | 4763 | −0.11 | 0.327 |
| **dq_4p25** | **DynQuant** | 4.2486 | 0.9307 | 3.77× | **89.25** | 4743 | −0.49 | 0.099 |
| awq_4b | AWQ, shipped | 7.3605 | 1.6125 | 2.17× | 89.18 | 4739 | −0.56 | 0.027 |
| dq_ctl4p25 | *control* (no signal) | 4.2486 | 0.9307 | 3.77× | 89.07 | 4733 | −0.68 | 0.021 |
| nf4_4b | bnb NF4 | 7.3391 | 1.6078 | 2.18× | 89.01 | 4730 | −0.73 | 4.2e-03 |
| gptq_3b | GPTQ, shipped | 6.6253 | 1.4514 | 2.41× | 88.97 | 4728 | −0.77 | 0.012 |
| rtn_4b | RTN, shipped | 7.3605 | 1.6125 | 2.17× | 88.93 | 4726 | −0.81 | 5.6e-03 |
| awq_4b_head | AWQ, tie quantized | 4.1597 | 0.9113 | 3.85× | 88.93 | 4726 | −0.81 | 1.2e-03 |
| rtn_4b_head | RTN, tie quantized | 4.1597 | 0.9113 | 3.85× | 88.80 | 4719 | −0.94 | 1.8e-03 |
| gptq_3b_head | GPTQ, tie quantized | 3.1522 | 0.6906 | 5.08× | 88.03 | 4678 | −1.71 | 1.1e-07 |
| **dq_3p25** | **DynQuant** | 3.2494 | 0.7118 | 4.92× | **86.70** | 4607 | −3.05 | 4.9e-17 |
| dq_ctl3p25 | *control* (no signal) | 3.2494 | 0.7118 | 4.92× | 83.53 | 4439 | −6.21 | 1.2e-44 |
| awq_3b | AWQ, shipped | 6.6253 | 1.4514 | 2.41× | 83.36 | 4430 | −6.38 | 3.7e-44 |
| awq_3b_head | AWQ, tie quantized | 3.1522 | 0.6906 | 5.08× | 83.31 | 4427 | −6.44 | 1.6e-45 |
| dq_ctl3p25_emb3 | *control* + 3-bit embed pin | 3.2494 | 0.7118 | 4.92× | 80.56 | 4281 | −9.18 | 1.1e-73 |
| rtn_3b | RTN, shipped | 6.6253 | 1.4514 | 2.41× | 65.53 | 3482 | −24.22 | 2.1e-259 |
| rtn_3b_head | RTN, tie quantized | 3.1522 | 0.6906 | 5.08× | 60.91 | 3237 | −28.83 | ~0 |

Read the ordering with care: **it is not a size ordering.** `dq_3p25` at 0.7118 GiB sits between
arms that are twice its size in both directions. The size column is what makes this table a
comparison rather than a leaderboard.

Three observations.

**The 4-bit tier is flat.** Not one adjacent gap between `gptq_4b_head` (89.76) and `rtn_4b_head`
(88.80) separates statistically. Six methods spanning rounding-only to inverse-Hessian error
feedback land inside one point of each other. Whatever distinguishes these algorithms is not
visible at 4 bits on this task.

**The 3-bit tier is where methods separate, and it separates by a lot.** From 88.03 (GPTQ) to 60.91
(RTN) is 27 points at identical bytes. This is the regime the comparison is actually about.

**RTN's collapse is the scale of the prize.** RTN at 3 bits loses 24–29 points. Everything any of
these methods does — Hessian error feedback, activation smoothing, signal-driven allocation — is
competing for that 27-point gap.

---

## 4. Panel B — Qwen3.5-2B-Base / Banking77 (task transfer, 10 arms)

Same model, same allocator, same fine-tuning recipe, different task. bf16: **93.41%** (2877/3080).

| arm | eff. bits | GiB | acc % | correct | Δ vs bf16 | p |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 16 | 3.505 | 93.41 | 2877 | — | — |
| gptq_4b | 7.3605 | 1.6125 | 93.08 | 2867 | −0.32 | 0.076 |
| nf4_4b | 7.3391 | 1.6078 | 92.89 | 2861 | −0.52 | 0.037 |
| **dq_4p25** | 4.2486 | 0.9307 | **92.66** | 2854 | −0.75 | 2.7e-03 |
| awq_4b | 7.3605 | 1.6125 | 92.50 | 2849 | −0.91 | 2.3e-04 |
| gptq_3b | 6.6253 | 1.4514 | 92.14 | 2838 | −1.27 | 2.2e-05 |
| rtn_4b | 7.3605 | 1.6125 | 91.53 | 2819 | −1.88 | 4.3e-09 |
| **dq_3p25** | 3.2494 | 0.7118 | **91.10** | 2806 | −2.31 | 6.8e-10 |
| awq_3b | 6.6253 | 1.4514 | 88.64 | 2730 | −4.77 | 2.0e-25 |
| rtn_3b | 6.6253 | 1.4514 | 61.62 | 1898 | −31.79 | 2.7e-255 |

The shape of Panel A survives: the 4-bit tier is tight, RTN collapses at 3 bits, DynQuant's 4.25-bit
arm sits inside the 4-bit cluster at 1.73× fewer bytes, and its 3.25-bit arm sits between GPTQ and
AWQ. The `_head` arms were not re-run here — the accounting lesson from Panel A carries over
unchanged and re-measuring it would have bought nothing.

---

## 5. Panel C — Mistral-7B-Instruct-v0.3 / Banking77 (scale, untied, 14 arms)

bf16: **94.38%** (2907/3080), 13.50 GiB.

| arm | eff. bits | GiB | ratio | acc % | correct | Δ vs bf16 | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| **dq_iso4p60** | 4.5949 | 3.8770 | 3.48× | **94.42** | 2908 | +0.03 | 1.00 |
| awq_4b | 4.5953 | 3.8774 | 3.48× | **94.42** | 2908 | +0.03 | 1.00 |
| gptq_3b | 3.6249 | 3.0586 | 4.41× | 94.38 | 2907 | +0.00 | 1.00 |
| fp16 | 16 | 13.50 | 1.00× | 94.38 | 2907 | — | — |
| rtn_4b | 4.5953 | 3.8774 | 3.48× | 94.25 | 2903 | −0.13 | 0.503 |
| nf4_4b | 4.5671 | 3.8536 | 3.50× | 94.25 | 2903 | −0.13 | 0.481 |
| gptq_4b | 4.5953 | 3.8774 | 3.48× | 94.25 | 2903 | −0.13 | 0.424 |
| **dq_4p25** | 4.2500 | 3.5859 | 3.76× | **94.19** | 2901 | −0.19 | 0.263 |
| dq_4p25_ft1map | 4.2500 | 3.5859 | 3.76× | 94.19 | 2901 | −0.19 | 0.263 |
| **dq_iso3p62** | 3.6244 | 3.0581 | 4.41× | 94.06 | 2897 | −0.32 | 0.154 |
| awq_3b | 3.6249 | 3.0586 | 4.41× | 94.03 | 2896 | −0.36 | 0.090 |
| **dq_3p25** | 3.2500 | 2.7422 | 4.92× | **93.60** | 2883 | −0.78 | 3.2e-03 |
| dq_3p25_ft1map | 3.2500 | 2.7422 | 4.92× | 93.57 | 2882 | −0.81 | 2.6e-03 |
| rtn_3b | 3.6249 | 3.0586 | 4.41× | 93.34 | 2875 | −1.04 | 5.8e-05 |

**This panel has little discriminating power, and that is its finding.** The whole spread from
best to worst is **1.08 points** on 3,080 examples. Even RTN at 3 bits — which loses 24 and 32 points
on the two Qwen panels — loses only 1.04 here. Two explanations, and the panel cannot separate them:

- **Scale.** 7.25 B parameters have redundancy that 1.88 B do not, so the same relative weight
  perturbation costs less accuracy.
- **The untied embedding.** The tensor that dominates the tied model's error budget is 3.7% of this
  one, and the baselines leave it in fp16 either way.

The panel is not *entirely* flat, and the distinction matters. Six comparisons do separate here, all
of them in the 3-bit tier: `rtn_3b` vs bf16 (−1.04, p = 5.8e-05), `dq_3p25` vs bf16 (−0.78,
p = 3.2e-03), `dq_3p25` vs `gptq_3b` (−0.78, p = 7.1e-03), `dq_iso3p62` vs `rtn_3b` (+0.71,
p = 7.2e-03), `dq_3p25` vs `dq_4p25` (−0.58, p = 0.027) and `dq_iso3p62` vs `dq_3p25` (+0.45,
p = 0.039). So the panel has enough power to resolve a 0.5-point effect; what it lacks is **dynamic
range**. Every 4-bit-class comparison fails to separate, and a 25-point effect and a 0.7-point effect
cannot look different in kind when the ceiling is one point away. The panel can rank the 3-bit arms;
it cannot tell you how much the ranking is worth.

The practical consequence is that a 7B model on a 77-way classification task is a poor discriminating
benchmark for weight quantization. Any method that rounds competently passes. **A comparison that
had only been run here would have concluded that all five methods are equivalent.**

Note also the byte picture inverts relative to Panel A: DynQuant's advantage over the shipped
convention drops from 1.73× to **1.08×**, because the fp16 residue the baselines carry drops from
27.0% to 3.7%. DynQuant's storage advantage is largely an artifact of *what the baselines decline to
quantize*, and that is an architectural property, not a property of either method.

---

## 6. Matched bytes — the six budgets that answer the actual question

Every comparison below is between arms of the same size, so allocation is the only difference left.

### 6.1 — 1.61 GiB / 7.36 bits / 2.17× (Qwen/CaseHOLD)

| pair | delta | CI95 | flips | p |
|---|---:|---|---:|---:|
| dq_iso7p36 vs gptq_4b | −0.11 | [−0.51, +0.28] | 54/60 | 0.64 |
| dq_iso7p36 vs awq_4b | +0.45 | [−0.02, +0.92] | 93/69 | 0.070 |
| dq_iso7p36 vs rtn_4b | **+0.70** | [+0.13, +1.26] | 137/100 | **0.019** |
| dq_iso7p36 vs nf4_4b | **+0.62** | [+0.13, +1.11] | 105/72 | **0.016** |
| dq_iso7p36 vs fp16 | −0.11 | [−0.30, +0.08] | 10/16 | 0.327 |

DynQuant ties GPTQ, edges AWQ without separating, and beats both RTN and NF4. Lossless relative to
bf16.

### 6.2 — 1.45 GiB / 6.63 bits / 2.42× (Qwen/CaseHOLD) — the one outright win

| pair | delta | CI95 | flips | p |
|---|---:|---|---:|---:|
| **dq_iso6p63 vs gptq_3b** | **+0.73** | [+0.14, +1.32] | 147/108 | **0.017** |
| dq_iso6p63 vs awq_3b | +6.34 | [+5.43, +7.25] | 474/137 | 2.1e-44 |
| dq_iso6p63 vs rtn_3b | +24.18 | [+22.71, +25.65] | 1434/149 | 6.9e-264 |
| dq_iso6p63 vs fp16 | −0.04 | [−0.35, +0.27] | 34/36 | 0.905 |

**This is the result the method should be presented on.** At 1.4512 GiB DynQuant is
indistinguishable from bf16; GPTQ at 1.4514 GiB is 0.77 points below it (p = 0.012). Same bytes,
same eval, and one of the two is lossless.

The honest qualification: 6.63 bits is a mild budget. `dq_iso6p63` and `dq_iso7p36` do not differ
from each other (+0.08, p = 0.72), so the win is not "DynQuant degrades gracefully" — it is
"DynQuant has not started degrading yet at a budget where GPTQ has."

### 6.3 — 0.91–0.93 GiB / 4.16–4.25 bits / 3.8× (Qwen/CaseHOLD)

| pair | delta | CI95 | flips | p |
|---|---:|---|---:|---:|
| dq_4p25 vs gptq_4b_head | −0.51 | [−1.08, +0.06] | 106/133 | 0.092 |
| dq_4p25 vs awq_4b_head | +0.32 | [−0.25, +0.89] | 127/110 | 0.299 |
| dq_4p25 vs rtn_4b_head | +0.45 | [−0.16, +1.06] | 150/126 | 0.166 |

Nothing separates. DynQuant is also **2.1% larger** than the GPTQ arm here, so on bytes it loses
slightly.

### 6.4 — 0.69–0.71 GiB / 3.15–3.25 bits / 5.0× (Qwen/CaseHOLD) — the weakness

| pair | delta | CI95 | flips | p |
|---|---:|---|---:|---:|
| **dq_3p25 vs gptq_3b_head** | **−1.34** | [−2.07, −0.60] | 162/233 | **4.2e-04** |
| dq_3p25 vs awq_3b_head | +3.39 | [+2.49, +4.29] | 388/208 | 1.5e-13 |
| dq_3p25 vs rtn_3b_head | +25.78 | [+24.26, +27.30] | 1531/161 | 3.0e-280 |

**GPTQ is 3.1% smaller and 1.34 points better.** There is no reading of this budget in which
DynQuant wins it. The mechanism is not mysterious: GPTQ propagates and compensates quantization
error through an inverse-Hessian update, and at the point where every module is badly quantized,
compensating the error beats choosing which modules to damage. DynQuant chooses well and then rounds
naively; GPTQ chooses nothing and then rounds cleverly. At 3 bits the second matters more.

### 6.5 — 3.88 GiB and 3.06 GiB (Mistral/Banking77)

Two more matched-byte budgets, on the untied 7B model. `dq_iso4p60` sits at 3.8770 GiB against the
baselines' 3.8774; `dq_iso3p62` at 3.0581 against 3.0586.

| pair | delta | CI95 | flips | p |
|---|---:|---|---:|---:|
| dq_iso4p60 vs gptq_4b | +0.16 | [−0.12, +0.44] | 12/7 | 0.36 |
| dq_iso4p60 vs awq_4b | +0.00 | [−0.24, +0.24] | 7/7 | 1.00 |
| dq_iso4p60 vs rtn_4b | +0.16 | [−0.10, +0.42] | 11/6 | 0.33 |
| dq_iso4p60 vs nf4_4b | +0.16 | [−0.10, +0.42] | 11/6 | 0.33 |
| dq_iso4p60 vs bf16 | +0.03 | [−0.20, +0.26] | 7/6 | 1.00 |
| dq_iso3p62 vs gptq_3b | −0.32 | [−0.81, +0.16] | 24/34 | 0.24 |
| dq_iso3p62 vs awq_3b | +0.03 | [−0.45, +0.51] | 29/28 | 1.00 |
| **dq_iso3p62 vs rtn_3b** | **+0.71** | [+0.21, +1.22] | 42/20 | **7.2e-03** |
| dq_iso3p62 vs bf16 | −0.32 | [−0.73, +0.08] | 15/25 | 0.15 |

At 4.60 bits `dq_iso4p60` is the joint-best arm in the panel (94.42%, tied with AWQ at 2908/3080 and
indistinguishable from bf16) — which on a 1.08-point spread means only that it is not worse.

**The 3.62-bit row is the one that matters, and it changes a verdict.** At matched bytes DynQuant
beats RTN by **+0.71 (p = 7.2e-03)** on Mistral. The shipped-convention comparison had *failed* to
separate (+0.26, p = 0.42) — but that pitted a 3.25-bit DynQuant against a 3.62-bit RTN, giving away
10% of the size. Match the bytes and the separation appears. It also confirms GPTQ's edge is not
scale-specific: −0.32 here, in the same direction as both Qwen panels, though this panel is too
compressed to separate it.

### 6.6 — summary across all four Qwen budgets

| budget | ratio | DynQuant | GPTQ | AWQ | RTN | NF4 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| 7.36 b | 2.17× | 89.63 | 89.74 | 89.18 | 88.93 | 89.01 | tie GPTQ, beat RTN/NF4 |
| 6.63 b | 2.42× | **89.71** | 88.97 | 83.36 | 65.53 | — | **DynQuant wins** |
| 4.16 b | 3.85× | 89.25 | 89.76 | 88.93 | 88.80 | — | tie |
| 3.15 b | 5.08× | 86.70 | **88.03** | 83.31 | 60.91 | — | **GPTQ wins** |

DynQuant's advantage is a band, not a trend. It is absent at mild budgets, present at 6.63 bits,
absent again at 4.16, and negative at 3.15.

### 6.7 — do the accounted bytes exist? The packed path, measured

Every accuracy number in §3–§6 comes from the **simulated** path: the quantizer writes dequantized
values back in place, so the arithmetic is quantized but the tensors are still bf16. Every byte
number is *accounted* — computed from the width map. Both need checking against a device, and on
Mistral they have been: `stage6_packed_{4p25,3p25}.json` (accuracy, `backend=cuda`, weights held as
packed `int32` words with the compiled `gemv_nbit` kernels doing the arithmetic) and the three
`stage6_*_runtime.json` records. The Mistral-specific write-up is
[RESULTS-mistral7b-banking77.md](RESULTS-mistral7b-banking77.md); the Qwen packed path was measured
in the earlier campaign and is in [RESULTS.md](RESULTS.md). What follows is the part that bears on
this report's claims.

**Provenance first, because these records predate the corrected bit map.** They were written
2026-07-29 03:21–04:27; the map was refit 2026-07-30 11:52 (defect 2). Diffing the two maps module
by module:

| target | width histogram | `nbytes` | placement vs shipped map |
|---|---|---:|---|
| 4.25 b | `{3: 10, 4: 212, 8: 4}` — identical | 3,850,371,072 — identical | **0 of 226 modules differ** |
| 3.25 b | `{2: 35, 3: 137, 4: 54}` — identical | 2,944,401,408 — identical | 24 of 226 differ |

The 4.25-bit map is bit-identical to the shipped one. The 3.25-bit one has the same histogram and
the same byte total with 24 modules placed differently — so its size and VRAM figures transfer
unchanged. More usefully, both packed arms are **map-matched to the `_ft1map` arms in Panel C**,
which makes the kernel comparison below clean.

**The accounting is exact, not approximate.**

| arm | accounted `nbytes` | `packed_bytes` on disk | resident VRAM | over manifest |
|---|---:|---:|---:|---:|
| dense bf16 | 14,496,047,616 (13.5005 GiB) | 14,495,514,624 (13.5000 GiB) | 13.5005 GiB | — |
| packed 4.25 | 3,850,371,072 (3.5859 GiB) | **3,850,371,072** | 3.5864 GiB | +0.014% |
| packed 3.25 | 2,944,401,408 (2.7422 GiB) | **2,944,401,408** | 2.7427 GiB | +0.018% |

Both packed arms match the accounted figure **to the byte** on disk, `manifest_bytes` agrees, and
resident VRAM exceeds it by 0.0005 GiB — the packing metadata and the fp16 norms the format keeps
unquantized. §2's arithmetic is not a model of the checkpoint; it is the checkpoint, and it holds on
a device.

**The kernels reproduce the simulated path, which is what licenses §3–§5.** Comparing each packed
arm against the simulated arm built from the same map, at the prediction level rather than by
net accuracy:

| pair | packed | simulated | flips | p |
|---|---:|---:|---:|---:|
| packed 4.25 vs `dq_4p25_ft1map` | 2900/3080 | 2901/3080 | **2 / 3** | 1.00 |
| packed 3.25 vs `dq_3p25_ft1map` | 2881/3080 | 2882/3080 | **0 / 1** | 1.00 |

Five differing predictions out of 3,080 at 4.25 bits and **one** at 3.25 bits. On Qwen/CaseHOLD the
earlier campaign got exact agreement — 4468 = 4468 on 5,314 problems — alongside 417 kernel-parity
tests and a clean four-tool `compute-sanitizer` sweep. So the simulated accuracies this report is
built on are what the real kernels produce, and that is measured rather than assumed.

**Peak VRAM tracks the manifest — the claim the paper explicitly conceded it could not make.**

| arm | peak, batch 1 | vs bf16 | batch 4 | batch 8 | batch 32 | peak over the full 3080-row eval |
|---|---:|---:|---:|---:|---:|---:|
| dense bf16 | 13.6766 GiB | — | 14.1810 | 14.8536 | 18.8891 | 19.49 GiB † |
| packed 4.25 | 3.8526 GiB | **3.55×** | 4.2993 | 4.9395 | 8.9750 | 9.5759 GiB |
| packed 3.25 | 3.0089 GiB | **4.55×** | 3.4556 | 4.0958 | 8.1313 | 8.7322 GiB |

† The two packed eval peaks are read off the stage-6 records; the dense one is carried from
[RESULTS-mistral7b-banking77.md](RESULTS-mistral7b-banking77.md), because the dense record on the box
holds `peak_decode_gib` but no full-eval peak. It is the only number in this report not read back off
disk at write time.

At batch 1 the overhead above resident weights is **0.2662 GiB in both packed arms** — the same
figure to four decimals, so it is the KV cache and activations for a 721-token prompt, not per-arm
slack. But the batch-32 and full-eval columns are the ones a serving decision turns on: **3.55× on
weights is 2.04× at eval peak**, because activations and KV cache are not quantized. Quantization
buys headroom in proportion to how much of your memory is weights, and that fraction is not 100%.

**Decode is slower than bf16, and the batch-32 case is severe.**

| arm | batch 1 | batch 4 | batch 8 | batch 32 |
|---|---:|---:|---:|---:|
| dense bf16 | 42.41 tok/s | 165.50 | 328.01 | **1244.41** |
| packed 4.25 | 36.90 (0.87×) | 145.15 (0.88×) | 294.03 (0.90×) | 400.48 (**0.32×**) |
| packed 3.25 | 36.93 (0.87×) | 149.30 (0.90×) | 280.01 (0.85×) | 220.30 (**0.18×**) |

Two cautions on the batch 1–8 rows. These are single runs per arm, measured about half an hour
apart, and the Qwen campaign established that batch-1 decode has a **12% spread across repeated
bf16 runs in one session** — wide enough that a 0.87× ratio from unrepeated cells is not separable
from host jitter. An earlier version of that Qwen table reported 0.98× from exactly this mistake and
the repeated measurement made it 0.90×. So read 0.85–0.90× as "slower, magnitude not pinned."

What does not depend on the bf16 baseline is the comparison *between* the two packed arms, and it is
the informative one: at batch 1 they decode in **27.099 ms and 27.078 ms — 0.08% apart — while
carrying 3.5864 and 2.7427 GiB of weights.** A quarter less weight traffic produced no measurable
change. This kernel is not bandwidth-bound; packed moves 3.85 GB in 27.1 ms = 142 GB/s against
roughly 1935 GB/s of A100 HBM. The deficit at batches 1–8 is a *fixed additive* 2.6–4.2 ms/step,
flat while throughput varies eightfold — per-launch cost, ~15 µs across 226 quantized modules,
which is what CUDA Graphs (P8) exists to remove.

Batch 32 is a different failure with a known cause, not a worse version of the same one:
`GEMV_MAX_ROWS` is **8**, so batches 1–8 use the packed GEMV and 32 falls back to
dequantize-then-`F.linear`, materialising a bf16 copy per call and therefore reading *more* memory
than the dense arm. 25.7 ms/step becomes 79.9 ms and 145.3 ms. That is a missing tensor-core GEMM
path (P7), and it is why the narrower arm is the slower one there.

Prefill is unaffected: at batch 32 (23,072 tokens) it is 1.7924 s dense against 1.8013 s and
1.8573 s packed, within 4%, because prefill already goes through dequantize-then-GEMM. Packing cost
**1685 s and 1746 s** — 28–29 minutes for a 7 B model against the plan's P5 gate of under 6 minutes
for a 14 B one, roughly an order of magnitude off per parameter.

The summary: **the size and VRAM claims are real, exact, and confirmed on hardware, and the
accuracy claims survive the move to real kernels. The speed claim does not exist yet** — decode is
slower at every batch size tested, and the reason is per-launch overhead and a missing large-M path,
neither of which is a property of the bit allocation this report evaluates.

---

## 7. What is the training signal actually worth?

DynQuant has two components that could explain its results: an **architectural prior** (role-based
floors, structural protection of the recurrence coefficients, honest byte accounting) and a
**training-time signal** (activation saliency × gradient plasticity, ranked and fed to the
allocator). These are separable, and separating them is the most informative thing in the campaign.

### The control

[stage4_allocate.py](stage4_allocate.py) runs the knapsack twice at every target: once with the real
scores, once with `uniform = dict.fromkeys(scores, 0.5)` — graph, floors, group size and budget held
fixed. This is the only comparison in the work where **exactly one input changes**.

It is not a straw man. With equal scores the ROI metric `score / (params × Δbits)` degenerates to
pure size, so the control is a real strategy: *spend bits where they are cheapest per parameter.*
All arms are byte-exact — net `params × bits` between the maps is exactly **+0**, because the budget
binds. Different width histograms at identical size is expected, not a bug.

### The result

| target | widths the signal moves | signal worth | p |
|---|---:|---:|---:|
| 4.25 b | 12 / 187 (6%) | +0.19 | 0.15 |
| 3.25 b | 87 / 187 (47%) | **+3.16** | **1.9e-15** |

At 3.25 bits the full decomposition, all three arms at exactly 0.7118 GiB:

```
RTN 3-bit + head        60.91%
      ↓ +22.62  (p = 2.8e-229)   ← the allocator: role floors, structural protection, ROI knapsack
uniform-score control   83.53%
      ↓  +3.16  (p = 1.9e-15)    ← the training signal
DynQuant                86.70%
```

**≈88% architectural prior, 12% training signal.** Both halves separate overwhelmingly. The sharpest
way to say it: *without the signal, DynQuant at 3.25 bits is AWQ* — 83.53% against
`awq_3b_head`'s 83.31% (p = 0.66), and AWQ's arm is 3% smaller.

At 4.25 bits the signal is worth nothing measurable, and the reason is mechanical rather than
mysterious: at that budget the role floors are nearly all affordable, so the allocator has almost no
decisions left to make. The signal's value tracks how many decisions it is allowed to make.

**Operating rule.** Run DynQuant when the target is tight enough that the role floors cannot all be
satisfied. Above that point it is an expensive round-to-nearest.

### The decision that looked like it should carry the 3.16, and didn't

The control breaches the tied embedding's inherited 8-bit floor all the way down to 2 bits; the real
map stops at 3. That one difference is 27% of the model and was the obvious candidate for the whole
margin. It was tested directly by pinning the control's embedding to 3 bits at the same budget:

| 3.25 b, all at 0.7118 GiB | embedding | other 186 modules | acc | correct |
|---|---:|---|---:|---:|
| DynQuant (signal) | 3 b | signal-ROI | **86.70%** | 4607/5314 |
| uniform-score control | 2 b | size-ROI | 83.53% | 4439 |
| uniform-score control + 3-bit pin | 3 b | size-ROI | **80.56%** | 4281 |

**It landed below both.** Pinning the embedding to 3 bits made the control 2.97 points *worse* —
CI [−3.78, −2.17], flips 100/258, p = 2.4e-12 — and 6.13 points behind the real map (p = 4.5e-45).

The rescue has to be paid for, and *how* it is paid dominates it. Under size-ROI, buying the
embedding's third bit spreads 55 modules from 4 to 3 bits, and that loses more than the embedding
gains. Under signal-ROI the same purchase is financed by holding 32 modules at 4 bits and driving 24
down to 2 — and that earns it back. **What the signal supplies at this budget is the payment
schedule, not the embedding decision.**

Two caveats, both real:
- **It measures a pair, not the pin.** Changing the embedding's width at a fixed budget necessarily
  changes other widths. The arm measures (embedding 3 b + its financing), not the embedding bit.
- **The 2×2 has three cells, not four.** The fourth — signal scores with the embedding *forced* to
  2 — cannot be built: `AllocationPolicy` exposes floors and `structural_roles`, never a per-module
  ceiling.

### An untested lever this exposed

Lowering the embedding floor from 8 to 2 under *real* scores produces `{2:8, 3:65, 4:78, 8:36}` at
3.2489 average bits, where the downgrade-from-floor path produces `{2:25, 3:94, 4:32, 8:36}` at the
same budget. Same scores, same budget, radically different map: **the allocator is path-dependent.**
Start-low-and-upgrade is a distinct strategy from start-at-floor-and-downgrade, and it has not been
run.

---

## 8. Verdicts across all three panels

DynQuant on the left. Shipped-convention baselines, so DynQuant is the smaller arm in every row.

### 4.25 bits (3.77×)

| vs | Qwen/CaseHOLD | Qwen/Banking77 | Mistral/Banking77 | replicated? |
|---|---:|---:|---:|---|
| GPTQ 4b | −0.49 (p = 0.10) | −0.42 (p = 0.14) | −0.06 (p = 0.85) | yes — never separated |
| AWQ 4b | +0.08 (p = 0.85) | +0.16 (p = 0.64) | −0.23 (p = 0.21) | yes — never separated |
| RTN 4b | +0.32 (p = 0.34) | **+1.14 (p = 7.3e-04)** | −0.06 (p = 0.83) | **no** — one panel only |
| NF4 | +0.24 (p = 0.45) | −0.23 (p = 0.48) | −0.06 (p = 0.85) | yes — never separated |
| bf16 | −0.49 (p = 0.099) | **−0.75 (p = 2.7e-03)** | −0.19 (p = 0.26) | **no** — one panel only |

### 3.25 bits (4.92×)

| vs | Qwen/CaseHOLD | Qwen/Banking77 | Mistral/Banking77 | replicated? |
|---|---:|---:|---:|---|
| GPTQ 3b | **−2.28 (p = 7.0e-10)** | **−1.04 (p = 0.011)** | **−0.78 (p = 7.1e-03)** | **yes — GPTQ ahead on all three** |
| AWQ 3b | **+3.33 (p = 7.5e-14)** | **+2.47 (p = 3.3e-07)** | −0.42 (p = 0.14) | partly — two of three |
| RTN 3b | **+21.17** | **+29.48** | +0.26 (p = 0.42) † | direction on all three |
| bf16 | **−3.05 (p = 4.9e-17)** | **−2.31 (p = 6.8e-10)** | **−0.78 (p = 3.2e-03)** | yes — separated on all three |

† The Mistral cell compares a 3.25-bit DynQuant against a 3.62-bit RTN, so DynQuant is giving away
10% of the size. At matched bytes (§6.5) it separates: **+0.71, p = 7.2e-03**.

### What replicated, and what did not

**Replicated across all three panels:**

1. **GPTQ wins the 3-bit tier.** −2.28 / −1.04 / −0.78, separated every time. The single most robust
   conclusion in the campaign, and it is the one unfavourable to DynQuant. **Superseded on
   Qwen/CaseHOLD by a later rebuild** — sensitivity allocation, an `E[x²]`-weighted clip objective,
   a row-partitioned tied embedding and per-row body widths turn that −2.28 into **+1.54 at 4.5%
   fewer bytes, p < 0.0001**. The other two panels have not been re-run and stand as written. See
   *Phase 2* in [RESULTS-external-comparison.md](RESULTS-external-comparison.md#phase-2--the-3-bit-gap-reversed-on-qwencasehold).
2. **The 4-bit tier is a tie.** Across three panels and four baselines, 12 comparisons at 4-bit-class
   budgets produce exactly one separation (RTN on Qwen/Banking77). DynQuant achieves that tie at
   1.73× / 1.73× / 1.08× fewer bytes.
3. **Quantization to 3.25 bits costs real accuracy.** Separated from bf16 on all three panels. No
   arm of DynQuant below 4 bits is lossless.

4. **DynQuant beats RTN at 3 bits** — and this one needed matched bytes to see. On the two Qwen
   panels the shipped-convention comparison is overwhelming (+21.17, +29.48). On Mistral it *fails*
   (+0.26, p = 0.42) — but that comparison gives away 10% of the size, pitting a 3.25-bit DynQuant
   against a 3.62-bit RTN. At matched bytes it separates: **+0.71, p = 7.2e-03**. The direction holds
   on all three panels; only the magnitude collapses, because RTN does not collapse on Mistral.

**Replicated on both Qwen panels, absent on Mistral:**

5. **DynQuant beats AWQ at 3 bits** (+3.33, +2.47) at roughly half the bytes — and +3.39 against
   AWQ's matched-byte `_head` arm on CaseHOLD, so the win is not an artefact of the size mismatch.
   On Mistral at matched bytes the two are a dead tie: +0.03, 29/28 flips, p = 1.00. This one
   genuinely does not carry over.

The pattern in both cases is the same and it is not a DynQuant property: **RTN and AWQ do not
collapse on Mistral**, so there is no collapse for DynQuant to avoid. RTN at 3 bits loses 1.04 points
there against 24–32 on Qwen. The Mistral panel is not underpowered — it resolves that 1.04 at
p = 5.8e-05, and it resolves DynQuant's +0.71 over RTN at p = 7.2e-03 — it is *compressed*: the
entire span from best method to worst is 1.08 points, so a 25-point advantage and a 0.7-point
advantage cannot look different in kind. It lacks dynamic range, not sample size.

**Did not replicate across tasks on the same model:**

6. **DynQuant 4.25 vs RTN 4b** separated on Banking77 (+1.14) and not on CaseHOLD (+0.32). The
   mechanism is worth stating precisely, because it is not the flattering one: DynQuant's 4.25-bit
   arm did not get *better* on Banking77 — it got worse, by 0.26 points. It separated from RTN there
   because RTN got worse *faster*, losing 1.07 where DynQuant lost 0.26. **The gap opens when the
   task punishes uniform rounding, not when the signal finds something extra.**

**One negative result worth recording as a result:** the bit map is mostly a property of the
architecture, not the task. At 4.25 bits the CaseHOLD and Banking77 maps agree on 177 of 187 widths;
86–89% of off-uniform decisions are task-invariant. A method sold on task-adaptive allocation
produces a nearly task-invariant map at moderate budgets.

---

## 9. Defects found and fixed during this campaign

Five, all found by guards or by measurement rather than by reading. Recorded because the fixes are
now the harness and because three of them would each have produced a wrong published number.

**1. The tied-embedding accounting defect.** `ignore=["lm_head"]` makes a "4-bit" baseline measure
7.3605 bits on a tied model. Found by computing bytes independently instead of trusting the label.
Fixed by adding the seven `_head` arms so every claim has a matched-byte version. *Consequence if
missed: every byte-ratio claim in the report would have been inflated by 1.7×.*

**2. A stale bit map on the Mistral panel.** Both Mistral DynQuant arms were originally quantized
through a map built from a **superseded** fine-tune's statistics. The fine-tune had been re-run; the
map had not. Every skip-if-output-exists resume guard stayed true because every output still
existed — existence cannot detect staleness, ordering can. Repaired by rebuilding the map and
re-running both arms.

*The defect turned out to be empty in consequence, and this is measured, not assumed:* at 4.25 bits
the corrected map differs from the stale one on **0 of 226 widths**, and the re-run arm is
**bit-identical** — +0.00, 0/0 flips, p = 1.00, across 3,080 examples. At 3.25 bits the maps differ
on 24 of 226 widths and the accuracy difference is **+0.03, CI [−0.31, +0.38], 15/14 flips,
p = 1.00**. Both stale rows are retained in Panel C as `_ft1map` so the claim is checkable.

**3. Records written into another model's directory.** `RUN_DIR` is derived from `DQ_MODEL` and
`DQ_TASK`, not from `--model`. A hand-written driver pinned the run directory in a shell variable
and passed `--model` explicitly — pinning only the *input*. With `DQ_MODEL` unset, four Mistral arms
wrote into the **Qwen** directory, and the same defaulted `MODEL_ID` handed a **Qwen tokenizer to
Mistral weights**, producing out-of-vocabulary token ids and a `device-side assert` on every arm —
*after* each 7B quantize had already been paid for. Three files were written under Qwen names, two
of them overwriting real Qwen records (`dq_4p25_quant`, `dq_3p25_quant`; the third,
`dq_iso4p60_quant`, is a Mistral-only arm name and collided with nothing). All accuracy records
survived, because `record()` runs after eval and every one of these arms crashed *during* eval.

The misplaced files are in `/workspace/quarantine/` with a README, deliberately **not** moved into
the directory they were meant for: a `_quant.json` there with no matching accuracy record reads as a
partially-completed arm, which is the same stale-artifact trap that produced defect 2. The lesson,
generalized: **passing the input explicitly does not pin the output.**

**4. A guard that fired for the wrong reason.** The first repair attempt aborted with "the map was
never rebuilt", inferred from the rebuilt map being byte-identical to the old one. It was wrong —
the map *had* been rebuilt, and at 4.25 bits rebuilding it correctly produces the same file. **A
content diff cannot separate "never rebuilt" from "rebuilt identically"; only mtimes can.** This is
the second time in this project that a guard's stated reason for firing turned out to be a
hypothesis rather than a diagnosis, and the general form is worth keeping: a guard reports that its
condition tripped, never why.

**5. A silent container restart.** The repair chain died at 12:58:19 with no traceback, no OOM
record and the GPU freed. `memory.events` showed `oom_kill 0` and a peak of 4.15 GB against a 208 GB
limit; no XID or ECC errors. The cause was the host restarting the container at 13:00:31 —
`.log.old` files from 08:32 show it had happened earlier the same day. It was invisible for 44
minutes because the log still *said* `992/3080` while the box had come back underneath it.
**Silence is not success**; liveness is now checked by log age plus process presence, not by content.

### The guards these bought

Both defects 2 and 3 are now checked inside
[stage5_quantize.py](stage5_quantize.py#L88-L145) rather than in a runner, so they apply to every
arm regardless of which driver launches it. To be exact about the timeline: **the arms in this report
were not produced under these guards.** They were produced under a runner-level `assert_rundir.py`
check plus a hand-rebuilt map, and the in-script guards were written afterwards so the *next*
campaign cannot repeat either defect. What protects the numbers here is the measurement in defect 2
above, not the guard.

- **Output-directory check** — `--model` must resolve inside the `RUN_DIR` that `common` itself
  computed. Catches defect 3 before a single weight is loaded.
- **Map-is-youngest check** — the bit map must be newer than both the weights and the stats it
  describes, with `newest_mtime` reaching inside directories because a directory's own mtime does
  not move when a file inside it is overwritten in place. Catches defect 2.
- `--allow-stale` downgrades both to warnings, for the one legitimate case: deliberately measuring a
  stale pairing to quantify what the staleness cost. That is how the `_ft1map` rows exist.

### The determinism check on the two clobbered records

The two overwritten `_quant.json` files were regenerated by re-running both Qwen/Banking77 DynQuant
arms end to end, which doubles as a determinism test: the accuracy records were never damaged, so
the re-run has a published number to reproduce. **Both reproduce exactly.**

| arm | published | re-run | delta | CI95 | flips | p |
|---|---:|---:|---:|---|---:|---:|
| `dq_4p25` | 92.6623% | 92.6623% | +0.00 | [+0.00, +0.00] | 0/0 | 1.00 |
| `dq_3p25` | 91.1039% | 91.1039% | +0.00 | [+0.00, +0.00] | 0/0 | 1.00 |

Not merely equal accuracy — **0/0 flips across 3,080 examples each**, so every individual
prediction is identical. Allocation, MSE clipping and evaluation are all reproducible from the
seed, and the numbers in Panel B were not a lucky draw. (Compare `gptq_4b vs fp16` on
Qwen/CaseHOLD: +0.00 delta with **58/58 flips** — equal accuracy, 116 different predictions. Equal
accuracy is not equal behaviour, and only the flip counts tell them apart.)

One trap encountered while writing the freshness check is recorded in the source, because it would
have been a plausible-looking guard that broke every clean run: an earlier draft also asserted
`mtime(stats) ≥ mtime(weights)`. But stats are written *throughout* training and the weights are
saved at the *end*, so stats-older-than-weights is the normal case — on the Mistral run, by eleven
seconds. Only "the map is the youngest artifact" is a valid assertion.

### The guards are tested, and the test found a hole in them

[test_guards.sh](test_guards.sh) exercises all four paths. Every case is designed to exit *before*
`load_model`, so validating a guard costs seconds rather than a 7B quantize:

| case | expected | result |
|---|---|---|
| map backdated 1 h before the weights | abort rc=3, no model load | **pass** — rc=3, and no load line in the output |
| `DQ_MODEL` unset, `--model` pointing at Mistral | abort rc=3 naming the Qwen dir | **pass** |
| correct map + correct `DQ_MODEL` | `provenance: ok`, proceed | **pass** (rc=2 on the deliberately bogus target) |
| stale map + `--allow-stale` | warn, proceed | **pass** |

6 of 6 assertions pass and no test writes a record. Note the timeline honestly: **this suite ran
after the campaign**, so it certifies the guards, not the arms.

**The first version of the test failed, and its failure is the more useful result.** It used the
retained `stage4_bitmaps.ft1.json` as the stale map — the actual stale artifact from defect 2. The
guard passed it, and the run went on to quantize and evaluate for real before it was killed. The
guard was right and the test was wrong: that file is a *copy*, made on 2026-07-30 11:50, so its mtime
is a day **younger** than the 2026-07-29 03:08 weights it misdescribes.

So the guard's actual scope is narrower than "detects stale maps":

> **An mtime guard cannot see through a copy, a `git checkout`, an `rsync` without `-t`, a container
> image rebuild, or a download. Each of those refreshes the mtime while preserving stale content.**
> It catches the failure mode that actually occurred — a map left untouched while its inputs were
> regenerated — and it is blind to a stale map that has since been moved.

Closing that would need the map to record a content hash of the stats it was built from, and the
check to compare hashes rather than times. That is the correct fix and it is not implemented. Until
it is, the guard is a tripwire for one specific accident, not a provenance system — and defect 4
already showed that a content diff alone cannot substitute, because it cannot separate "never
rebuilt" from "rebuilt identically". Times and hashes each catch what the other misses; the guard
currently has only one of them.

---

## 10. Limitations

Stated plainly, because most of them bound the conclusions above more tightly than the p-values do.

1. **Two models.** One 1.88 B tied and one 7.25 B untied. Every claim that differs between them is
   confounded by scale *and* by the tie, and no arm separates the two.
2. **Two tasks, both classification.** Accuracy on a fixed answer set is a coarse metric. No
   generative task, no perplexity, no long-context measurement. A method could preserve
   classification accuracy and still damage generation.
3. **One clean win, on one tier.** The 6.63-bit result is a single (model, task, budget) point. It
   has not been reproduced on Banking77 or on Mistral, and the equal-accuracy arm at 7.36 bits
   suggests the win sits at the edge of where degradation begins rather than in the middle of a
   trend.
4. **The runtime evidence covers DynQuant only, and its decode numbers are not tightly measured.**
   §6.7 confirms the bytes, the VRAM and the kernel-vs-simulated equivalence on Mistral, so the
   accuracy comparisons are not undermined by the packed path. What it does not do is measure any
   *baseline* packed: GPTQ, AWQ, RTN and NF4 were scored through `llm-compressor`'s own simulated
   output and never run through a serving kernel here, so **no speed or VRAM comparison against a
   baseline exists in this report** — the 0.87× decode figure is DynQuant against bf16, not
   DynQuant against GPTQ, and Marlin-class baseline kernels would very likely beat it. Beyond that:
   the batch 1–8 decode cells are single runs against a baseline whose run-to-run spread is 12%, so
   their magnitude is not pinned (the direction is, and the batch-32 collapse is far outside noise);
   the 3.25-bit runtime checkpoint has the shipped histogram and byte total but 24 of 226 modules
   placed differently; and no perplexity or logit-KL check accompanies the packed path, so
   equivalence is established on classification predictions only.
5. **The ablation is one model, one task, two budgets.** The 88/12 split is measured at 3.25 bits on
   Qwen/CaseHOLD. It is not known whether the signal's share grows at 3 bits or below, and the
   path-dependence lever suggests the allocator's own share is not even fixed.
6. **Baselines are one library's implementation at one configuration.** 256 calibration rows,
   `group_size` 128, `dampening_frac` 0.01. A better-tuned GPTQ would widen its lead at 3 bits; a
   more aggressive AWQ configuration might close its gap. No baseline was tuned.
7. **The Mistral panel is compressed, not powerful.** Its whole best-to-worst span is 1.08 points.
   It does separate six comparisons, all in the 3-bit tier, so it is not underpowered — but no
   4-bit-class row there can distinguish a large effect from a negligible one, and those rows should
   not be read as evidence either way.

---

## 11. What this means for the method

The comparison is not flattering everywhere, and the shape it does have is specific.

**DynQuant's real contribution, as measured, is the allocation prior — not the training signal.**
88% of its 25.78-point margin over byte-matched RTN comes from role floors, structural protection
and honest byte accounting; those are architectural facts that require no training run to obtain.
The signal contributes a separated but small 3.16 points, and only where the budget is tight enough
to force many decisions. A user who wants most of DynQuant's benefit does not need to instrument a
fine-tune.

**The 3-bit tier is the method's frontier and it currently loses there.** GPTQ is smaller and better
at 3 bits on all three panels. The mechanism is legible: DynQuant chooses widths well and then
rounds naively, while GPTQ chooses nothing and compensates error. **These are orthogonal.** Nothing
about signal-driven width allocation precludes inverse-Hessian error compensation inside each
module, and the obvious next experiment is to stop treating them as competitors.

**The compression is real and the speed is the next problem, in that order.** The packed checkpoints
match the accounted bytes exactly, cut peak VRAM 3.55–4.55×, and reproduce the simulated accuracies
to within a handful of predictions — which retires the paper's own concession that the savings are
storage-only, and retires the worry that the simulated evaluation was measuring something the
kernels do not do. What replaces it is a narrower complaint: decode is 10–15% slower than bf16 at
batches 1–8 and falls apart at 32. Both causes are diagnosed rather than guessed — a fixed ~15 µs
per-module launch tax that CUDA Graphs removes, and a `GEMV_MAX_ROWS` of 8 that sends batch 32 down
a dequantize-then-`F.linear` fallback reading more memory than bf16. Neither is a property of
signal-driven bit allocation, which is what makes them P7/P8 work rather than a verdict on the
method. But the honest position today is that DynQuant trades throughput for footprint, and no
baseline was benchmarked packed, so its speed against a Marlin-class GPTQ kernel is unknown and
should not be assumed favourable.

**The clean claim the method can make today** is the 6.63-bit one: lossless at 2.42× compression
where GPTQ at identical bytes is not. That is a narrower claim than the paper's headline and it is
the one that survives a matched-byte comparison.

---

## 12. Reproduction

Everything runs from [experiments/four_point/](.) on a single 80 GB GPU.

```bash
# 1. environment for the baselines (llm-compressor + bitsandbytes)
python -m venv venv-cmp && venv-cmp/bin/pip install llmcompressor bitsandbytes

# 2. pin BOTH variables -- RUN_DIR is derived from them, not from --model
export DQ_MODEL=Qwen/Qwen3.5-2B-Base
export DQ_TASK=casehold
RUN=/workspace/runs/qwen3_5_2b_base_casehold

# 3. verify python and shell agree on the output directory before anything runs
python assert_rundir.py "$RUN"

# 4. baselines, shipped convention and matched-byte
bash run_baselines.sh          # gptq/awq/rtn at 4 and 3 bits
bash run_tiedhead.sh           # the same with ignore=[]  -> the _head arms
python stage8_bnb.py --name stage8_nf4_4b

# 5. DynQuant arms
python stage4_allocate.py --targets 4.25,3.25
python stage5_quantize.py --target 4.25 --name stage8_dq_4p25
bash run_isosize.sh            # the equal-byte arms at 7.36 and 6.63 bits

# 6. the signal ablation
python make_control_map.py     # uniform scores, everything else fixed
python stage5_quantize.py --bitmaps "$RUN/stage4_bitmaps.control.json" \
                          --target 3.25 --name stage8_dq_ctl3p25

# 7. read the results
python dump_all.py                                   # every arm, every panel
python pairs.py "$RUN" dq_3p25:gptq_3b_head          # McNemar on any pair
```

`pairs.py` takes the run directory **first** and prefixes `stage8_` itself, so pass bare arm names.

### File map

| file | what it is |
|---|---|
| [stage4_allocate.py](stage4_allocate.py) | the knapsack; runs twice per target (real scores + uniform control) |
| [stage5_quantize.py](stage5_quantize.py) | applies a map and scores it; holds both provenance guards |
| [stage8_baselines.py](stage8_baselines.py) | llm-compressor GPTQ/AWQ/RTN, with the rounding verification |
| [stage8_bnb.py](stage8_bnb.py) | bitsandbytes NF4 |
| [test_guards.sh](test_guards.sh) | validates both provenance guards; every case exits before the model load |
| [bitmap_diff.py](bitmap_diff.py) | width-level diff between two maps |
| [baselines_table.py](baselines_table.py) | regenerates the panel tables from the records |
| [RESULTS-external-comparison.md](RESULTS-external-comparison.md) | the working document, with derivations and failed attempts |
| [RESULTS-mistral7b-banking77.md](RESULTS-mistral7b-banking77.md) | the Mistral campaign, including the packed VRAM/speed section behind §6.7 |
| [RESULTS.md](RESULTS.md) | the Qwen campaign: kernel parity, sanitizer sweep, step profile, bandwidth gates |

---

*All 45 comparison arms are measured and every verification job has landed. The last one — a re-run of the two
Qwen/Banking77 arms whose `_quant.json` was overwritten by defect 3 — reproduced both published
numbers with 0/0 prediction flips; its verdict is in §9. §6.7 was added last, after diffing its bit
maps against the shipped ones — the 4.25-bit map is identical, the 3.25-bit map differs on 24 of 226
placements at an identical histogram and byte total — and it replaced an earlier limitation that
wrongly claimed no runtime measurement existed. Nothing in this report is pending.*
