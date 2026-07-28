# Audit of the research code

Every defect below was **confirmed by reading or running the supplement's own
code**, not inferred. Line numbers refer to the supplementary material as
submitted — the `dynquant_paper/`, `inference/` and `fine-tuning_and_stats_hook/`
trees.

### Where that code is, and is not

**The research tree is not committed.** `.gitignore` excludes all four of its
directories. It is part of the same confidential submission as the PDF; it
hardcodes a Hugging Face token placeholder and the author's VM paths (§10); and
nothing the package ships depends on it. Clone this repository and those
directories will not be there — which is why nothing here should be read as a
pointer to a file you can open. It is a record of what was found, with enough
quoted evidence to stand on its own.

**Three modules are vendored, verbatim**, into
`packages/dynquant-core/src/dynquant/_legacy/`: `scorer.py`, `allocator.py` and
`quantizer.py`. That is what `--preset paper-3.15` needs to reproduce published
numbers, and what the golden tests run against. All three import only the standard
library and torch — no paths, no credentials, no file reads — which is what makes
shipping them safe. `tests/test_legacy_provenance.py` pins each by SHA-256, checks
them against the originals on any machine that still has the supplement, and fails
if anything else is ever copied into that directory.

The rest — `supervised_finetuning.py`, `run_quantization.py`, `unified_tracker.py`,
the inference scripts, the pipeline drivers — is **not** vendored. It is not needed
for compat mode, it is drivers rather than computations, and it carries the
literals that must never reach an installed package.

So §4's measurements are reproducible and §10's are not, and the difference is
deliberate: `pytest tests/test_legacy_allocator.py` runs the allocator claim
anywhere, while the credential and path findings can only be re-checked against a
copy of the supplement.

This is not a criticism of the research. Every one of these is the ordinary
consequence of code that was run once, by its author, on one machine, to produce
one set of numbers. They matter here only because the same code is now becoming a
package other people install, where "works if you run it exactly as I did" is not
a property anyone can rely on.

Two things follow from the list and are worth stating before it:

- **Items 1–3 mean the pipeline cannot execute end to end as shipped.** Training
  raises `NameError` at startup, no checkpoint the quantizer writes can be opened
  by any reader, and three modules import paths that do not exist.
- **Item 4 means the published 3-bit configuration was not produced by the
  allocator in the supplement.** This is measured, not argued, in §4.

| # | Defect | Severity | v2 status |
|---|---|---|---|
| 1 | Training crashes at startup (`NameError`) | blocker | fixed |
| 2 | Checkpoints can never be read back (filename case) | blocker | fixed, lint-guarded |
| 3 | Three modules import non-existent paths | blocker | fixed |
| 4 | Importance scores have **zero** effect at the headline target | critical | fixed (soft floors) |
| 5 | Fused projections mis-assigned, contradicting the stated policy | critical | fixed (row partitions) |
| 6 | MoE routers / MLA / SSM projections fall to the 2–3-bit floor | critical | fixed (structural roles) |
| 7 | The gradient signal measures the wrong tensor under LoRA | critical | fixed (`outer_exact`) |
| 8 | One GPU sync per module per step | major | fixed (device-resident) |
| 9 | `group_size` guessed at load; layout hostile to kernels | critical | fixed (explicit metadata) |
| 10 | Secrets, absolute paths, import-time side effects, no DDP reduction | major | contained (§10); DDP reduction lands in P2 |

---

## 1. Training crashes at startup

**Evidence.** `fine-tuning_and_stats_hook/supervised_finetuning.py:32` imports the
class under its post-rename name:

```python
from dynquant_paper.unified_tracker import UnifiedDynQuantTracker
```

and lines 565 and 574 use the pre-rename one:

```python
tracker = UnifiedGasqTracker(model, track_modules=tuple(track_types))
def __init__(self, tracker: UnifiedGasqTracker, log_every_n_steps: int = 50):
```

`unified_tracker.py:33` defines only `UnifiedDynQuantTracker`. `UnifiedGasqTracker`
is not bound anywhere.

**Consequence.** `NameError` the moment signal collection is set up. The training
half of the pipeline cannot run. A comment at line 94 still refers to
`UnifiedGasqTracker` too, so the rename was applied to the definition and the
import but not to the uses.

**v2.** One name, one module, and `tests/test_packaging.py` imports every public
symbol so an unbound name fails CI rather than a user's first run.

---

## 2. Checkpoints can never be read back

**Evidence.** The writer emits a capital Q
(`dynquant_paper/run_quantization.py:372`):

```
dynQuant_quantized_weights.safetensors
```

Every reader opens lowercase (`inference/inference_4bits.py:198`):

```
dynquant_quantized_weights.safetensors
```

**Consequence.** On Linux — where the experiments ran, and where CI would run —
these are two different files. No checkpoint the quantizer produced could be
loaded by the inference code in the same supplement. On Windows and macOS the
case-insensitive filesystem hides it, which is why it could survive local testing.
The names appeared as string literals in five separate files.

**v2.** Every filename lives in
[`constants.py`](../packages/dynquant-core/src/dynquant/constants.py) and nowhere
else. `tests/test_constants_are_the_only_filenames.py` greps the source tree for
filename literals outside that module, and separately asserts that no two
DynQuant-owned names differ only by case — the exact failure mode, caught on a
case-insensitive dev machine because it is a string comparison rather than a file
operation. Both historical spellings are retained in
`LEGACY_CHECKPOINT_FILENAMES` so `dynquant migrate` can still find an old
directory.

---

## 3. Three modules import paths that do not exist

**Evidence.**

| Import | Actual | File |
|---|---|---|
| `from fine_tuning_and_stats_hook import ...` | directory is `fine-tuning_and_stats_hook` | `run_sft_with_dynquant.py:85`, `run_sft_with_gasq.py:85` |
| `from dynquant_paper1.quantizer import ...` | package is `dynquant_paper` | `inference_4bits.py:110, 222, 277` |
| `from gasq_paper1.quantizer import ...` | package is `dynquant_paper` | `inference_3bits.py:111, 225, 280` |

A hyphen is not legal in a Python identifier, so the first is unimportable under
any `sys.path`.

**Consequence.** Neither entry point can be imported. Note also that
`run_sft_with_dynquant.py` and `run_sft_with_gasq.py` are the same file under two
names, as are `inference_4bits.py` and `inference_3bits.py` — the rename was done
by copying, so every defect below exists in two copies that have since drifted
apart.

**v2.** One installed package, `import dynquant`, with importability asserted in
CI on every supported Python.

---

## 4. Importance scores have zero effect at the headline target

This is the finding that most affects how the method should be described, so it
is measured rather than asserted. It is also the only item here that is *pinned by
a test*: `tests/test_legacy_allocator.py` runs the supplement's own
`allocate_bits_pareto` against its own shipped stats and asserts each number in the
table below. It is not a regression test — nothing will fix it — it is an
executable record, so the claim stays checkable and so `--preset paper-3.15` has a
definition of the behaviour it must reproduce.

```bash
python -m pytest tests/test_legacy_allocator.py -q
```

**Evidence.** `allocator.py:137`:

```python
remaining = total_budget - base_cost - spent
if remaining < 0:
    # Floors exceed the requested budget; return the forced allocation.
    return allocation
```

Running the supplement's `allocate_bits_pareto` on
`stats/qwen3_14b/unified_gasq_stats_collapsed.json` (282 modules, Qwen3-14B
geometry: hidden 5120, intermediate 17408, vocab 151936, 40 layers, 40 heads / 8
KV heads):

| target | `remaining` | greedy loop | modules upgraded | achieved avg |
|---|---|---|---|---|
| 3.00 | −7.662 Gbit | **skipped** | **0** | 3.5477 |
| 3.15 | −5.563 Gbit | **skipped** | **0** | 3.5477 |
| 3.50 | −0.667 Gbit | **skipped** | **0** | 3.5477 |
| 4.00 | +6.328 Gbit | runs | 71 | 4.0000 |

The floors alone cost **3.5477 bits** averaged over the 13.990 B variable
parameters, so the greedy ROI loop cannot run at any target below that — including
`allocate_bits_pareto`'s own default of `target_avg_bits=3.5`. The decisive check:

> **Inverting every importance score changes the allocation of 0 of 282 modules at
> targets 3.0, 3.15 and 3.5.**

The window in which the score influences anything at all is narrow, and bounded at
*both* ends. Negating every score and counting how many modules change width:

| target | 3.0 | 3.5 | 3.6 | 3.7 | **3.8** | 3.9 | 3.95 | 4.0 |
|---|---|---|---|---|---|---|---|---|
| modules changed | 0 | 0 | 16 | 46 | **64** | 32 | 16 | 0 |

Zero at the bottom because the floors early-return. Zero at the **top** for a
different reason: by 4.0 the budget lifts every remaining module to 4-bit, so
there is no choice left for a ranking to influence. The score therefore affects
the outcome only between roughly 3.55 and 3.95 bits — and the paper's headline
setting, 3.0, is outside it.

The score `S_i = Rank(plasticity) × Rank(saliency)` — the paper's central
contribution — provably does not influence the shipped 3-bit configuration. What
the allocator returns is the hand-written floor map from `_min_bits_floor`, whose
shape (embeddings 4, attention 4, MLP gate 4, MLP up/down 3, LM head 8) is exactly
the configuration the paper reports. Also note that a run targeting 3.0 silently
returns 3.5477 bits over variable layers — 3.7822 bits counting the 8-bit head over
all 14.768 B — and reports no error; it misses its own target by half a bit and
says nothing.

The 3.5477 rather than a rounder number comes from a second rule: `activation_rms >
2.5 → 4 bits` fires on 42 of 282 modules, of which 9 are `up_proj`/`down_proj` that
would otherwise have been 3-bit.

**Consequence.** The reported accuracy at 3-bit is real — those numbers came out of
a real quantized model. But the ablation in the paper's Table 8, which varies the
scoring function at the 3-bit setting, cannot have been produced by this allocator
as configured, because at that setting the scoring function is not read. Anyone
reimplementing from the paper would find scores mattering only inside a ~0.4-bit
window that contains none of the settings the paper reports.

**v2.** Floors become **soft**. When floors exceed the budget the allocator
downgrades by lowest ROI and emits a `floor_violations` report naming every
breached role (§7 of [format-spec.md](format-spec.md)) instead of silently
returning the floor map. Scores therefore drive allocation at every target. The
guard is a regression test asserting that inverting the scores changes the
allocation at a 3.0 target — the assertion that fails on the code as shipped.
`--preset paper-3.15` keeps the hard-floor behaviour, pinned by a golden test, so
published numbers stay reproducible.

---

## 5. Fused projections mis-assigned, contradicting the stated policy

**Evidence.** `_min_bits_floor` (`allocator.py:23`) documents its own intent:

```
- MLP Gate: 4 bits (Crucial for SwiGLU stability).
- MLP Up/Down: 3 bits (Can tolerate noise better than Gate).
```

and matches on name substrings:

```python
if "gate_proj" in name:
    return 4
if any(x in name for x in ["down_proj", "up_proj", "mlp"]):
    return 3
```

Phi-4 has no `gate_proj`. It has `gate_up_proj` — one tensor whose first half *is*
the SwiGLU gate. `"gate_proj" in "…mlp.gate_up_proj"` is `False`; the `"mlp"`
catch-all matches, and the whole fused tensor is assigned **3 bits**. The
supplement's own `stats/phi-4/…json` contains `…mlp.gate_up_proj.base_layer`, so
this is the configuration the Phi-4 experiments ran under.

The attention side is handled — `qkv_proj` is in the attention list — so this is
specifically the gate that the docstring calls "crucial" being quantized at the
width the docstring says destroys it.

**Consequence.** The paper's headline Phi-4 result was obtained with its SwiGLU
gate at 3-bit under a policy stating gates need 4. Whether that helped or hurt is
untested; what is certain is that the model was not quantized according to the
described policy.

**v2.** `fusion.py` splits fused projections into **row partitions**, so
`gate_up_proj` carries 4-bit gate rows and 3-bit up rows in one tensor — the
policy applied exactly rather than by rounding the whole tensor up (wasteful) or
down (the shipped behaviour). This needs no new quantization math, because
group-wise quantization already stores one scale per `(row, group)`; rows are
independent. The row partition and each shard's role are recorded in the manifest
(§7 of [format-spec.md](format-spec.md)), so a reader recovers which rows are the
gate without re-deriving it from config.

---

## 6. Routers, MLA and SSM projections fall to the floor

**Evidence.** `_min_bits_floor` matches substrings and ends with `return 2`.
Tracing real module names through it:

| Architecture | Module | Matches | Assigned |
|---|---|---|---|
| Mixtral / Qwen3-MoE | `…mlp.gate` (the **router**) | `"mlp"` catch-all | **3-bit** |
| DeepSeek-V2/V3 (MLA) | `…self_attn.kv_a_proj_with_mqa` | nothing | **2-bit** |
| DeepSeek-V2/V3 (MLA) | `…self_attn.q_a_proj` | nothing | **2-bit** |
| Mamba | `…mixer.in_proj`, `x_proj`, `dt_proj` | nothing | **2-bit** |
| Jamba | mixer projections | nothing | **2-bit** |

A router is a `[hidden, num_experts]` Linear whose output is an argmax over
experts. At 3-bit its logits are perturbed enough to change which expert is
selected — not a small numerical error but a different computation. MLA's
`kv_a_proj_with_mqa` produces the compressed latent every K and V in the layer is
reconstructed from; at 2-bit the attention layer is destroyed.

Note `"v_proj"` is not a substring of `"kv_a_proj_with_mqa"` (the string contains
`v_a_proj`), which is why the MLA case falls through rather than accidentally
matching.

**Consequence.** The supplement covers dense Llama-family models only, and its
allocator silently produces catastrophic assignments on any MoE, MLA or SSM model
— silently because nothing reports an unmatched module.

**v2.** Roles are resolved structurally, not by substring: user override →
architecture plugin for `config.model_type` → **structural inference from the
module tree** → name substrings only as a last resort. The router test is
`out_features == config.num_experts` with an `experts` sibling `ModuleList`, which
identifies routers in MoE families that do not exist yet. `ModuleRole.OTHER` maps
to a conservative 4-bit default and `dynquant inspect` lists every unclassified
module — nothing reaches 2-bit by falling off the end of an `if` chain.

---

## 7. The gradient signal measures the wrong tensor

**Evidence.** Under QLoRA the base weights are frozen, so
`unified_tracker.py:129-137` falls back to hooking whichever parameter requires
grad — the LoRA factors — and records the result under the *module's* name.

`lora_A.weight` is `[r, in_features]` and `lora_B.weight` is `[out_features, r]`.
The coherence EMA (`:213-216`) stores a per-channel vector between steps and takes
a dot product with the previous one:

```python
dot = float(torch.dot(channel_rms, prev).item())
```

Consecutive steps therefore alternate between vectors of length `r` and length
`out_features`. `torch.dot` raises on the size mismatch, and a bare `except
Exception` (lines 145, 150, 188, 234) discards it.

**Consequence.** The plasticity signal describes the LoRA adapters' gradients, not
the base weight's — the tensor actually being quantized. Coherence is silently
near-dead. And the shipped stats files declare `"collapsed_lora_into_base": true`,
a post-processing step whose script is **not in the supplement**, so the artifacts
cannot be regenerated from the code provided.

**v2.** The default estimator is `outer_exact`: a full backward hook gives
`δ = ∇_Y L`, a forward hook stashes `x`, and the paper's own Eq. (1) `∇W = δxᵀ`
yields the exact base-weight gradient norm on a bounded token subsample. It
measures the tensor being quantized and is shape-stable regardless of LoRA rank.
`lowrank` (compose from the factors) and `param` (legacy, for `paper-3.15`) remain
available, and the mode is recorded per layer so mixing them is detectable (§8 of
[format-spec.md](format-spec.md)). Coherence uses per-output-channel `‖δ‖`, always
length `d_out`, so the mismatch cannot recur. No bare excepts — a hook that fails
raises.

**One module the fix cannot reach, and it is not a bug.** `outer_exact` needs
`δ = ∇_Y L`, so it needs autograd to compute a gradient with respect to the module's
*output*. The input embedding under LoRA is the one place that never happens: the
weight is frozen and the input is integer, so the output tensor has
`requires_grad=False` and nothing upstream of it needs a gradient. Measured on
Qwen3-0.6B, 3 steps, `outer_exact`:

| arm | embedding output `requires_grad` | `grad_norm_count` | exercised-but-ungraded set |
|---|---|---|---|
| full fine-tune | `True` | 3 | ∅ |
| LoRA r=8, `all-linear` | `False` | 0 | `{model.embed_tokens}` |

The contrast that matters is `lm_head`: frozen under LoRA too, and it keeps
`grad_norm_count == 3`, because its *input* requires grad. So this is the
frozen-weight case working everywhere it can, and the embedding is excluded by the
autograd graph rather than by the estimator. `scripts/gate_lora_stats.py` asserts the
ungraded set is exactly `{input embedding}` and fails on any other name, since a
second entry would mean `outer_exact` regressing.

Downstream, that module lands in `coverage().partial_signal` and in `score`'s
`unexercised`, where it takes `NEUTRAL_RANK` on **both** axes — so its measured
activation RMS is discarded along with the plasticity it never had, and its role floor
decides its bits. For the embedding specifically the practical effect is small (it is
the largest tensor in the model, so its ROI is near the bottom of the knapsack either
way), but "we measured saliency and then threw it away" is a scoring rule, not a
consequence of the missing gradient. Whether a partially-measured module should keep
its measured axis and take neutral only on the absent one is an open **P4** question.
The remedies that restore the signal are `modules_to_save=["embed_tokens"]` in the
`LoraConfig`, or full fine-tuning.

---

## 8. One GPU sync per module per step

**Evidence.** Both hooks force a device→host transfer inside the training step:

```python
activation_rms = float(rms_per_channel.mean().cpu().item())  # :180
grad_norm = float(g.float().norm().cpu().item())  # :206
```

**Consequence.** `.item()` blocks until the GPU drains. On Qwen3-14B that is ~280
tracked modules × 2 hooks ≈ 560 full pipeline stalls per step; on a 128-expert MoE
it is tens of thousands. The paper's claim that signal collection is essentially
free does not survive this implementation — the *method* is nearly free, the
implementation is not.

**v2.** Accumulators are device-resident tensors indexed by module id, updated
without leaving the GPU. One transfer every `log_every` steps. Welford updates
happen in `on_pre_optimizer_step`, so variance is over optimizer steps as the
paper's Appendix H specifies, rather than over micro-batches as the code does —
under gradient accumulation those differ by the accumulation factor.

Overhead measured on an A100 80GB PCIe, Qwen3-0.6B (198 tracked modules, the worst case
for a per-module cost — ~0.6 GFLOP/token) at 16×2048 with gradient checkpointing:
**+2.30%** of step time for the default `outer_exact` estimator, **+1.52%** for `param`,
and +1.68% / +1.15% for the same configuration under LoRA `r=16`. Removing the syncs was
necessary and nowhere near sufficient; what the number cost is recorded in
`benchmarks/tracker_overhead.py`. Read it as a ratio whose denominator matters: the same
tracker reads **+8.96%** on the same model at 2×2048, and per-module microseconds, not
percent, is what compares two runs.

---

## 9. `group_size` guessed at load; layout hostile to kernels

**Evidence.** Nothing stores `group_size`. `quantizer.py:587-602` reconstructs it:

```python
num_groups = scale.numel() // out_features
if in_features % num_groups == 0:
    group_size = in_features // num_groups
else:
    # Padding was likely used. Assume 128 as standard, or infer.
    if (in_features + 127) // 128 == num_groups:
        group_size = 128
    else:
        # Fallback to integer division (risky if padded)
        group_size = in_features // num_groups
```

The comment "risky if padded" is the supplement's own.

**Consequence.** Two distinct failures. The guess breaks for `ndim != 2` (Mamba
`conv1d.weight` is `[channels, 1, 4]`; stacked MoE experts are `[E, out, in]`) and
for padded `in_features`, and when it is wrong it does not raise — it returns a
tensor of the right shape containing wrong numbers, which looks exactly like
quantization being lossy.

The second failure is structural and is why the paper ships no kernels: 3-bit
packing flattens the **entire tensor** before packing, so group `g` of row `r`
begins at a bit offset that depends on every preceding row. There is no coalesced
load, no compile-time shift, no unrolled inner loop. No efficient GEMV can be
written against that layout at all, which is why inference dequantizes back to
fp16 and the method saves disk but not VRAM.

**v2.** `QuantTensor` stores `bits, group_size, symmetric, layout, in_features,
row_offset, logical_shape, compute_dtype` explicitly; nothing is inferred from a
shape, ever. Groups are word-aligned (`group_size % 32 == 0`), so every group
occupies a whole number of `uint32` words and a thread block reads a
compile-time-constant word count with constant shifts. The full contract is
[format-spec.md](format-spec.md) §4; it is what P6–P8's kernels are written
against, and it is the difference between storage-only compression and a model
that is smaller in VRAM.

---

## 10. Secondary defects

| Defect | Evidence | v2 |
|---|---|---|
| Hardcoded credential placeholder | `token="hf_token_here"` at `supervised_finetuning.py:530, 544` (and commented at `:499`) | env var / `huggingface_hub` auth. Three layers: the file is gitignored, `scripts/check_no_confidential.py` refuses it as a pre-commit hook if it is force-added, and `tests/test_confidential_guard.py` proves the guard fires on this exact file. That last layer is not redundant — the guard's first version matched `hf_[A-Za-z0-9]{8,}`, which cannot match `hf_token_here`, and it passed the file it was written to stop |
| Real absolute paths from the author's VM | `/home/azureuser/cloudfiles/code/gasq_research_qwen14/...` at `inference_3bits.py:53, 56` | CLI arguments; the same three layers, plus a pattern for POSIX and Windows home paths that the first version of the guard did not have at all |
| Model loaded at **import** time | `inference_4bits.py:413` runs `load_quantized_model_once(...)` at module scope, printing `=== Loading Model Globally ===` | nothing loads on import; `dynquant.__init__` imports no torch modules eagerly |
| No DDP/FSDP reduction | stats written per-rank, last writer wins | Chan parallel merge before write; a 2-process run must match a 1-process run within tolerance |
| Welford over micro-batches | `unified_tracker.py` updates per backward, not per optimizer step | updated in `on_pre_optimizer_step`, per Appendix H |
| `count < 2` indistinguishable from "no movement" | `variance()` returns `0.0` for both; `qwen3_14b`'s `lm_head` really does have `grad_norm_count: 0` and survives only because of its 8-bit floor | `has_gradient_signal` requires `count >= 2`; missing signal is reported, not scored as maximally compressible |
| GSM8K hardcoded into the trainer | `supervised_finetuning.py` | dataset is a parameter |
| No packaging, tests, or CI | no `pyproject.toml` anywhere in the supplement | three wheels, 497 tests, ruff + mypy strict + CPU/GPU CI |
| Whole pipeline duplicated per naming era | `run_sft_with_{dynquant,gasq}.py`, `inference_{4,3}bits.py` | one implementation, bit-width is an argument |

---

## 11. Gradient checkpointing double-counts the saliency EMA

Found by measurement during P2, not by reading — which is why it is numbered after
the secondary defects rather than beside the other hook findings. Nothing in the
source looks wrong; the defect only exists in combination with a training flag.

**Evidence.** `supervised_finetuning.py:378` calls
`model.gradient_checkpointing_enable()` unconditionally, and `:774` passes
`gradient_checkpointing=True` to `TrainingArguments` as well. `torch.utils.checkpoint`
rebuilds a checkpointed block's activations by re-running its forward during the
backward pass, and module forward hooks are **not** suppressed on that replay. So the
saliency hook fires twice per micro-batch for modules inside a checkpointed block, on
identical data.

Measured on Qwen3-0.6B, 4 optimizer steps, `use_reentrant=False`:

| module | checkpointing off | on |
|---|---|---|
| `model.layers.0.self_attn.q_proj` | `forward_calls=4` | **`forward_calls=8`** |
| `lm_head` (outside the checkpointed layers) | 4 | 4 |

Nothing cheap distinguishes the two calls: `torch.is_grad_enabled()`,
`out.requires_grad` and even `type(out.grad_fn)` are identical. Only the autograd
graph-task id differs — `-1` on the real forward, `0` on the replay.

**Which** modules double depends on the checkpointing implementation, and that is worse
than it sounds. Over 3 steps on Qwen3-0.6B (198 tracked modules, `observe_recompute=True`
so nothing is suppressed):

| `use_reentrant` | `forward_calls == 3` | `== 6` |
|---|---|---|
| `False` (the transformers 5.x default) | 30 — `embed_tokens`, `lm_head`, **and all 28 `mlp.down_proj`** | 168 |
| `True` (the 4.3x default) | 2 — `embed_tokens`, `lm_head` | 196 |

Non-reentrant checkpointing stops recomputing as soon as the last *needed* saved tensor
has been produced, and a decoder layer's `down_proj` output is not one of them — so it
never replays. The horizon therefore splits three ways under the modern default, and the
split runs *inside* a single MLP: `up_proj` averages over ~50 optimizer steps while
`down_proj`, ranked against it, averages over ~100. The supplement pins no transformers
version, so which of these two shapes produced the shipped stats files cannot be
determined from the code.

**Consequence.** Two EMA updates with the same value leave the fixed point alone but
square the decay, so at β = 0.99 the replayed modules average activation RMS over a
~50-step horizon while the ones that never replay average over ~100. The absolute
horizon hardly matters. The *inconsistency* does: `dynquant.score` percentile-ranks
every module against every other, so this ranks a 50-step average against a 100-step
one — two different statistics — and the comparison silently decides bit widths.
Both shipped stats files were collected this way.

It is also expensive. The duplicated activation read cost 150.6 → 221.1 µs per module
per step on Qwen3-0.6B at 8×2048, ~30% of the tracker's whole bill.

**v2.** `_in_backward()` tests the graph-task id, and `_observe_forward` skips the
saliency update and the `forward_calls` increment when a hook fires inside a backward
pass. The predicate is deliberately "during a backward" rather than "during a
checkpoint recompute": saliency averages over forward passes on training data, and a
forward that happens inside a backward is a replay or a higher-order derivative, not a
new observation. Gradient accumulation is unaffected — its micro-batches each run a
real forward outside any backward, and each should count.

The guard stops at the saliency read and deliberately leaves the gradient path alone.
Under `use_reentrant=True` — what `gradient_checkpointing_enable()` with no arguments
used to select — the real forward runs under `no_grad`, so the tracker's existing
`out.requires_grad` test skips gradient-hook registration there and the **replay's**
registration is the live one. Returning early from the hook instead would have deleted
the plasticity signal for every checkpointed module while every `forward_calls`
assertion still passed. Both `use_reentrant` modes are parametrized in
`tests/test_signals_tracker.py` for exactly that reason.

`TrackerConfig(observe_recompute=True)` restores the old behaviour, which
`--preset paper-3.15` needs in order to reproduce a stats file collected this way.

The stats file now also states which of the two it was. `snapshot()` emits
`provenance.notes["recompute_forward_calls"] = {"count": n, "observed": bool}` whenever a
forward hook fired inside a backward pass — the count of replays, and whether they were
folded in. What happened is recorded rather than what was configured, because the flag alone
does not say whether it mattered: `observe_recompute=True` on a run without checkpointing
produces identical statistics to the default. A non-zero count with `"observed": true` is the
one combination that means the saliency EMA has two horizons. The two shipped v1 files were
collected in exactly that state and carry no such note, which is the argument for emitting
one.

After the fix, per-module cost under checkpointing is back at parity with the
uncheckpointed path — two paired runs on Qwen3-0.6B at 8×2048 read 152.1 / 157.0 and
154.4 / 154.2 µs, so what is left is run-to-run variation, not a residue of the replay —
and the `forward_calls` histogram is identical in both arms (`{3: 198}` over 3 steps).
The P2 overhead gate on Qwen3-0.6B at 16×2048 reads **+2.30%** for the default
`outer_exact` estimator and **+1.52%** for `param`.

---

## What is worth keeping

The audit is long, but the method is sound and several pieces of the
implementation are good enough to carry over unchanged:

- **3-bit vectorized pack/unpack** (`quantizer.py:451-522`) is correct. It becomes
  the torch reference oracle that CUDA kernels are tested against.
- **The MSE clipping grid search** (`:167-210`) is sound — it is ported to CUDA in
  P5 and kept in torch as the oracle.
- **The tie-averaging percentile ranker** (`scorer.py:16-45`) carries over as-is.
- **The symmetric per-row 4-bit embedding path** (`:38-74`) becomes the
  `EMBEDDING` role policy, and is why `PER_ROW_GROUP_SIZE` exists in the format.
- **`_normalize_stats_layer_name`** (`run_quantization.py:17-36`) has the right
  idea in the wrong place; it becomes `canonical_name()`, applied at **write**
  time so keys are canonical in the file rather than guessed at read time.
- **Streaming safetensors application** (`inference_4bits.py:231-320`) is the
  correct load strategy and becomes the `HfQuantizer` load path.

And the central claim survives all of this. Item 4 shows the *allocator* did not
consult the scores at 3-bit, but the signals themselves — activation saliency and
gradient plasticity — are cheap to collect, and the floor map they were meant to
produce is a reasonable one. v2's job is to make the score actually reach the
allocation, and to make the resulting checkpoint something a kernel can execute.
