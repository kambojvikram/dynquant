# Phase 3 — Generalization: four architectures, generative tasks, served

## What this campaign is for

Phases 1 and 2 measured two models, two **classification** tasks, scored in-process through
`transformers`. Every headline number this project owns — the +0.73 at 2.42×, the tie at 3.8×,
the +1.54 over GPTQ at 3 bits — sits inside that box.

This campaign changes three things at once and asks whether any of it survives:

| | phases 1–2 | phase 3 |
|---|---|---|
| architecture | Qwen3.5-2B (hybrid linear attn), Mistral-7B (dense GQA) | 4 families, **none Qwen** |
| task | CaseHOLD, Banking77 — pick-one-of-N, scored by exact match on a label | free generation, scored by execution / programmatic verification |
| execution | `transformers`, in-process | **vLLM server**, the path we just proved at parity |

Changing three things at once is the right design for a transfer question and the wrong one for
an attribution question. This campaign can say *whether* the result generalizes. It cannot say
which of the three changes broke it if it doesn't — that is what §7's fallback arms are for.

---

## Three problems with the design as stated

Each of these is fixable, and each one costs a full run to discover late.

### P1 — All four models are already instruction-tuned, so the fine-tuning gain may not exist

This is the GSM8K failure repeating. `Qwen3.5-2B-Base` on GSM8K produced six flat arms because
the base model was already at the supervised ceiling, and it cost a full run to diagnose. All
four models here are `-Instruct` / `-it` checkpoints, and Tulu-3 / SmolTalk are general SFT
mixtures. Fine-tuning `Llama-3.1-8B-Instruct` on Tulu-3 will very plausibly *cost* a point on
IFEval rather than gain one.

Two consequences, one harmless and one fatal:

**Harmless.** The signal hook does not care. It needs gradients and activations, not
improvement — DynQuant's inputs are produced by any fine-tune, including one that makes the
model slightly worse.

**Fatal, if unaddressed.** "Percentage of the fine-tuning gain retained" — the framing that made
`RESULTS.md` legible — has no denominator when the gain is zero or negative.

**Fix, and it is free:** make the primary metric phase 1/2's framing, which never needed a gain.
Matched-byte comparison against GPTQ / AWQ / RTN / NF4, paired McNemar on stored per-item hits,
with the bf16 fine-tuned checkpoint as the ceiling. That measures damage against a ceiling and
requires only that the ceiling not be 100 %.

**Second fix:** run the headroom screen anyway (S1). It is pure inference, no training, and it
tells you *before* anything is spent which (model, task) pairs have room. This is established
practice here for exactly this reason.

**The alternative worth keeping in reserve:** `meta-llama/Llama-3.1-8B` and
`google/gemma-3-4b-pt` exist as base checkpoints, and on those Tulu-3 SFT produces a real, large
gain, restoring both framings. Phi-4-mini and Ministral-8B have no public base release, so this
cannot be applied uniformly — which is why I would not lead with it. Keep instruct for all four
so the four are comparable, and pull in a base arm only if S1 says the instruct models are at
ceiling on everything.

### P2 — 4 models × 3 datasets is twelve fine-tunes, and it buys less than it costs

Twelve LoRA fine-tunes on 8B-class models, each followed by ~8 quantization arms and five
generative evals, is not a campaign — it is a quarter. And the third factor does not need to be
crossed with the first, because they answer different questions.

Cross them once instead:

| sweep | arms | question |
|---|---|---|
| **model** | 4 models × Tulu-3 | does the result hold across attention structure |
| **dataset** | 1 model × {SmolTalk, OpenThoughts3} | are the conclusions data-specific |

Six fine-tunes, not twelve. Pick the dataset-sweep model by what S1 shows most damage on —
almost certainly Phi-4-mini, which is also the cheapest to train three times.

That reduction is not only a saving. The three-dataset arm on one model is a direct test of a
finding this project already has and has never stressed: 86–89 % of off-uniform bit decisions
were task-invariant across CaseHOLD and Banking77, i.e. the bit map is mostly architecture. This
tests the same claim on a *third* axis — training data rather than evaluation task — which is
the axis most likely to break it, since the signal is collected during training.

### P3 — `gemma-3-4b-it` is multimodal, and nothing here has ever seen a vision tower

`google/gemma-3-4b-it` is `Gemma3ForConditionalGeneration` with a SigLIP tower. (Only the 1B is
text-only.) Three implications:

1. Role classification has never been run on a VLM. The plan's arch matrix listed `llava` and
   `qwen2_vl`; no test covers either, and `graph/arch/` contains exactly one plugin.
2. Quantizing the tower means compressing something none of the five evals exercises.
3. Leaving it dense means the compression ratio is diluted by a component we chose not to
   compress, and the byte accounting has to say so out loud.

**Recommendation: keep `gemma-3-4b-it`, quantize the language model only, put the tower in
`modules_to_not_convert`, and report both ratios — over the whole checkpoint and over the
language model.** That is the honest version, and it exercises the "fused layer in a region the
exporter left alone" path that the SGLang guard was written for two days ago and that has only
ever been tested against a synthetic checkpoint.

---

## What must exist before any GPU time is spent

Four gates. All are cheap, all are CPU-only or near it, and each one can invalidate the campaign
if it fails after the fine-tunes are paid for.

### G1 — Role classification on all four architectures ✅ **done 2026-08-03**

`tests/test_graph_arch_matrix.py`, 23 tests, real `transformers` classes at 1/100th scale via
`from_config` — no downloads, no weights. The gate found four defects; all four are fixed and
each is guarded by tests verified to go red when the fix is reverted.

**1. Phi's fused projections got no row partitions — 60 % of the model.** `qkv_proj` and
`gate_up_proj` classified correctly as fused roles but `_partitions` returns `()` without an
architecture plugin, and there was no `phi3` plugin. So on Phi-4-mini the 40 % of quantizable
parameters in `gate_up_proj` sat at the SwiGLU gate's 4-bit floor with the `up` half paying for
the gate's sensitivity, and `qkv_proj`'s 20 % could not price Q apart from K and V. This is the
model where phase 2's row-partitioning lever should pay most, and it was switched off.

Fixed by `graph/arch/phi_fused.py`. Row orders read off the modelling source rather than assumed:
`Phi3MLP.forward` does `gate, up = gate_up_proj(x).chunk(2, dim=-1)`, so gate is the **first**
`intermediate_size` rows; `Phi3Attention.forward` slices contiguous `[q; k; v]` sized by
`num_attention_heads` / `num_key_value_heads`, so the split follows the GQA ratio (8:1 on the
real model) and not an even third. Both partitions are guarded by an arithmetic check against
`out_features` and decline rather than guess when a checkpoint disagrees with its config.

**2. Gemma-3 shipped one module in `OTHER`.** SigLIP's `embeddings.position_embedding` shares no
substring with `embed_tokens`, so it fell through every rule. Fixed by a `position_embedding`
leaf rule, remapped inside a vision tower to the patch embedding's floor — the two tables are
summed, so they share a fate.

**3. `nn.MultiheadAttention.in_proj_weight` read as Mamba's `ssm.in`.** The substring pass
contains `in_proj`. A fused QKV was landing on the SSM input floor — confidently, and wrongly.
Fixed with an exact-leaf rule; exact matching runs before the substring pass.

**4. Three tensors per Gemma-3 were invisible to the graph entirely.** `named_modules` finds a
weight only if its owner spells it `self.weight`, and both `nn.MultiheadAttention` and
`Gemma3MultiModalProjector` keep theirs in bare `nn.Parameter` attributes. The multimodal
projector — which `DEFAULT_FLOOR_BITS` rates among the least compressible tensors in a VLM — was
in neither the graph nor the byte accounting. **This is the tied-embedding error running
backwards:** a tensor on disk that the denominator does not divide by, so the reported average
bits describes a subset of the file. Fixed by a raw-parameter sweep in `classify_model`, which
also records vectors-wearing-matrix-shapes (SigLIP's `[1, 1, hidden]` pooling probe) as skipped
by non-singleton rank rather than quantizing them. Gemma-3's floor cost moved 4.2240 → 4.2899
average bits once the missing tensors entered the count.

Also asserted and passing: Phi-4-mini's tie detected with the floor escalating to `lm_head`'s
8 bits (with an untied control, so the test cannot pass by raising the embedding floor
globally); Gemma-3's vision tower separable by role rather than by name glob, which is what makes
"quantize the language model only" implementable; and Ministral's `sliding_window` provably a
no-op for role assignment — same role map with the window set and unset.

Suite: 1138 passed, 13 skipped (was 1115). Ruff clean.

### G2 — Pack / unpack and kernel parity on each model's real shapes

The 417 kernel-parity tests sweep geometries, not these geometries. Gemma-3 uses `head_dim=256`
(not `hidden/heads`), Phi-4-mini has a partial rotary factor, and the four models' GQA ratios
differ. One pack → unpack → compare-against-the-torch-oracle run per model's actual layer shapes,
at 2/3/4/8 bits. Cheap, and it is the only thing standing between a shape assumption and a
silently wrong checkpoint.

### G3 — Three of the five evals do not exist

`dynquant.eval` ships `casehold`, `banking77`, `gsm8k`, plus `compare` (paired McNemar with CIs).
So GSM8K is free; **IFEval, HumanEval and MBPP have to be written**, and the judge evals more so.

Order by value per unit of work:

1. **IFEval** — programmatically verifiable, no judge, no execution sandbox, and it measures the
   thing SFT actually changes. Write this first.
2. **HumanEval + MBPP** — execution-based pass@1. Needs a sandboxed subprocess runner with a
   timeout; the harness does not have one. Non-trivial but bounded.
3. **MT-Bench / AlpacaEval 2** — see §5. Not on the critical path.

Hard requirement on all of them: **store per-item outcomes**. Every task must emit a `hits` array
so every A/B is an exact McNemar test rather than two independent proportions. This is what
halved the standard error in phase 2, and it is what refused to promote a +0.51 that looked like
a win.

### G4 — Evaluate through vLLM, and prove it first

Generative eval at this volume through `transformers` is prohibitive — this is the difference
between a campaign that takes a week and one that takes a month. Serve every arm through vLLM,
including the baselines (vLLM serves GPTQ and AWQ natively), and score against the server.

That is free leverage in both directions: it makes the campaign affordable, *and* it turns the
whole thing into a second serving-parity test three orders of magnitude larger than the
12-prompt sweep. But it has to be gated: one arm, scored both ways — direct `transformers` and
through vLLM — must agree within the bound the
[serving-parity report](reports/serving-parity.md) established. Precedent exists: Qwen3.5-2B at
3.25 bits scored 86.96 % both ways, p = 1.0000.

---

## Stages

| | stage | GPU | output |
|---|---|---|---|
| **S0** | G1–G4 gates | none | arch matrix test, IFEval task, pass@1 runner, vLLM eval path |
| **S1** | headroom screen — 4 models × 5 tasks, no training | ~6 h | which (model, task) pairs have room; the decision to proceed |
| **S2** | 4 fine-tunes on Tulu-3, signal hook attached | ~60 h | 4 bf16 checkpoints + 4 stats maps |
| **S3** | quantization arms, matched bytes | ~25 h | ~32 checkpoints |
| **S4** | eval every arm through vLLM, per-item hits stored | ~40 h | the results matrix |
| **S5** | dataset sensitivity — Phi-4-mini × {SmolTalk, OpenThoughts3} | ~30 h | 2 more checkpoints, same arms |
| **S6** | bit-map cross-analysis | ~0 | architecture vs task vs training-data decomposition |
| **S7** | report | none | `docs/reports/phase3-generalization.md` |

Hour figures are order-of-magnitude on one A100 80GB, from this project's own measured rates
(Mistral-7B LoRA on Banking77, and the phase-1 quantize+eval sweep). They are not quotes.

### S1 — the screen, in detail

Pure inference, no training, 4 models × {IFEval, GSM8K, HumanEval, MBPP}. Two things come out:

- **the ceiling per pair**, which is the denominator every later comparison divides by;
- **the go/no-go.** A pair scoring above ~90 % has no room for a quantization regression to show
  and should be dropped from the headline, exactly as GSM8K was for Qwen3.5-2B.

If S1 shows all four models at ceiling on everything, stop and switch to the base-checkpoint
variant of P1 before spending S2. That decision costs six hours here and a week later.

### S3 — the arms

Per checkpoint, at matched **bytes** rather than matched nominal width — the accounting point
from phase 1 §2, where a nominally 4-bit GPTQ arm measured 7.36 stored bits on a tied model:

| arm | note |
|---|---|
| bf16 fine-tuned | the ceiling |
| DynQuant @ ~4.25 b | where the signal is expected to be worth nothing (phase 1 finding 6) |
| DynQuant @ ~3.25 b | the contested tier |
| DynQuant @ ~3.0 b | phase 2's winning recipe: GN sensitivity + row-partitioned tie + E[x²] clip + per-row body |
| GPTQ, AWQ, RTN, NF4 | `llm-compressor` / `bitsandbytes`, each at the byte budget of the DynQuant arm it is compared to |

Two arms are mandatory and easy to forget:

- **the shuffled-row control** on every per-row arm. Phase 2's central finding is that
  granularity is a multiplier on the signal, not a gain of its own: with row widths shuffled,
  per-row allocation *loses* 1.28 points at identical bytes. An arm shipped without its control
  cannot tell those apart.
- **the signal ablation** — same allocator, same graph, same floors, same budget, constant
  scores — on at least the Phi-4-mini and Llama arms. That is the only measurement that
  separates "architectural prior" from "training signal", and phase 1 put the split at 88/12.

### S6 — the analysis this campaign uniquely enables

Three bit maps now vary along three axes that have never been separated:

| axis | held fixed | varies |
|---|---|---|
| architecture | task, training data | 4 model families |
| task | model, training data | IFEval / GSM8K / HumanEval / MBPP |
| **training data** | model, task | Tulu-3 / SmolTalk / OpenThoughts3 |

The existing finding is that the map is 86–89 % architecture-determined across *tasks*. The
training-data axis is the one that should break it if anything does, because the signal is
collected during training and nowhere else. If the maps are also stable across training data,
that is a substantially stronger claim than the one currently in the record — and if they are
not, it bounds how much a signal map can be reused, which is a result either way.

---

## §5 — On the judge evals

MT-Bench and AlpacaEval 2 are the odd ones out and should not carry claims:

- not per-item binary, so not McNemar-testable — they need a paired bootstrap over per-question
  score differences instead;
- they need a judge model, which adds API cost, nondeterminism, and a dependency on a third
  party's model version;
- LLM judges are known to be sensitive to response length and formatting, and both are exactly
  what quantization perturbs first, so a judge delta is ambiguous between "worse answer" and
  "shorter answer".

**Run them last, on the winning arm and the bf16 ceiling only, reported as a qualitative sanity
check with a paired bootstrap.** IFEval, GSM8K, HumanEval and MBPP carry the claims.

## §6 — On OpenThoughts3

"Long generations are where quantization error compounds" is the most interesting untested claim
in your set, and it is also the most expensive to test:

- training sequence length goes to 8–16k, which multiplies the fine-tune cost superlinearly;
- eval generation length multiplies the eval cost linearly;
- at long context the KV cache dominates and the memory win shrinks — this is measured, not
  speculative: a 4.92× smaller weight footprint became a 1.52× smaller *process* at batch 32.

Keep it, on the smallest model, with the sequence length capped explicitly and stated. And note
the honest framing: at long context, weight quantization is buying progressively less of the
total, so the interesting number is accuracy compounding, not memory.

## §7 — Fallback arms, if the result does not generalize

Three things changed at once, so a negative result needs a way to be attributed. Two cheap arms
recover it:

- **the same generative eval on Qwen3.5-2B** — isolates architecture, since everything else
  matches phase 2;
- **CaseHOLD on one phase-3 model** — isolates task type, since everything else is new.

Neither needs a new fine-tune if the phase-2 checkpoint is still on disk.

---

## Licensing — research-only as specified

| | license | gated |
|---|---|---|
| Llama-3.1-8B-Instruct | Llama 3.1 Community License | yes |
| gemma-3-4b-it | Gemma Terms of Use | yes |
| Phi-4-mini-instruct | MIT | no |
| Ministral-8B-Instruct-2410 | **Mistral Research License — non-commercial** | yes |
| Tulu-3-SFT-mixture | ODC-BY-1.0, some constituent subsets non-commercial | no |

Three of four models need token-gated access accepted on the account before S1 can run, and
Ministral's MRL makes any resulting checkpoint non-commercial regardless of what the other three
allow. Fine for a paper. It should be stated in the report rather than discovered by someone
downstream.

---

## What I would cut first, if the budget is half this

In order:

1. **MT-Bench / AlpacaEval** — §5.
2. **MBPP** — HumanEval already covers execution-based pass@1; MBPP adds breadth, not a new
   failure mode.
3. **OpenThoughts3** — the most expensive arm, and its finding is the least load-bearing.
4. **Ministral-8B** — of the four models it is the one whose distinguishing feature
   (sliding-window attention) touches the KV cache rather than the weights, so it is the least
   likely of the four to move a weight-quantization result. Also the only non-commercial one.

That leaves 3 models × Tulu-3, IFEval + GSM8K + HumanEval, one dataset-sensitivity arm — which
still answers the campaign's actual question.

---

## Decisions — locked 2026-08-03

1. **Instruct checkpoints for all four.** Base checkpoints stay as the fallback if and only if S1
   shows every model at ceiling on every task; Phi-4-mini and Ministral have no public base, so
   that fallback cannot be applied uniformly and would cost comparability.
2. **Six fine-tunes** — 4 models × Tulu-3, plus Phi-4-mini × {SmolTalk, OpenThoughts3}. The
   dataset-sweep model is provisional on S1; it changes only if the screen shows Phi-4-mini has
   no headroom where another model does.
3. **Judge evals out of the headline.** MT-Bench / AlpacaEval run last, on the winning arm and
   the bf16 ceiling only, reported as a qualitative check with a paired bootstrap. IFEval,
   GSM8K, HumanEval and MBPP carry every claim.
