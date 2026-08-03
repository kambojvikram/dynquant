# Serving parity: does vLLM — and does SGLang — hold the same quantized weights?

Every accuracy number this project reports was measured through `transformers`, either with
the weights reconstructed into bf16 (stages 1–5) or through the packed CUDA kernels
(stage 6). None of it says anything about an inference server. A serving integration can
load a checkpoint, start, answer, and be wrong: the failure mode is not a crash but a model
that has silently dropped the packed tensors and is computing with whatever the parameter
buffers were initialised to.

This report is the measurement that closes that gap for both servers the package integrates
with. It is the one experiment in this project that runs the model through something other
than `transformers`, and it found a real defect in the SGLang path that no unit test on
either side could have caught.

**Result.** Both servers reproduce the direct GPU run to within the noise those runtimes
already have against `transformers` on an unquantized model. On the packed checkpoint the
mean absolute logprob difference is **0.009147** for vLLM and **0.006822** for SGLang, against
an fp16 control of 0.006304 and 0.006896 respectively — so vLLM's serving gap is 1.45× its
own fp16 gap and SGLang's is **0.99×**, indistinguishable from no gap at all. Quantization
itself moves the same logprobs by **2.461067**. The serving gap is 269× (vLLM) and 361×
(SGLang) smaller than the thing being served, and top-1 agreement is **100.0 %** at every
scored position on every arm.

| | date | commit under test |
|---|---|---|
| vLLM arms | 2026-07-31 | `dynquant-core` 0.1.2 |
| SGLang arms | 2026-08-02 | `dynquant-core` 0.2.0, fix `9e4c6ed` |

---

## 1. Hardware and software

One machine, one card, for every arm — including the direct reference the servers are
compared against, so no cross-machine difference can enter the comparison.

| | |
|---|---|
| GPU | NVIDIA A100 80GB PCIe |
| driver | 595.71.05 |
| host | vast.ai instance 45884068 (interruptible bid instance) |

Two deliberately separate virtual environments. They are not merged because vLLM and SGLang
pin incompatible `transformers` ranges, and forcing one environment to satisfy both would
mean at least one server running against a `transformers` it does not support — which is a
different experiment from the one intended.

| | `venv-vllm` | `venv-sglang` |
|---|---|---|
| Python | 3.12.13 | 3.12.13 |
| torch | 2.11.0+cu130 | 2.11.0+cu130 |
| server | vllm 0.26.0 | sglang 0.5.16 |
| transformers | 5.14.1 | 5.12.1 |
| numpy | 2.3.5 | 2.3.5 |
| safetensors | 0.8.0 | 0.8.0 |
| `dynquant-core` | 0.1.2 (editable) | 0.2.0 (editable) |
| `dynquant-kernels` | 0.1.2 | 0.1.2 |

The two `dynquant-core` versions are worth stating rather than hiding. The checkpoint was
exported by 0.1.2 and read back by 0.2.0 without conversion, and the SGLang arms score within
noise of the vLLM arms against the *same* stored reference — which is a small, free piece of
evidence that `dynquant_checkpoint_v2` survived a minor version bump, and the only such
evidence this project has.

## 2. The checkpoint

Qwen2.5-0.5B-Instruct, packed by [`01_export.py`](#6-the-scripts). 0.5B rather than the 2B
this project evaluates accuracy on: this experiment measures whether two encodings of the
same weights agree, and a small model makes the sweep cheap without changing what is being
tested.

| module | bits |
|---|---|
| `q_proj` | 4 |
| `k_proj` | 3 |
| `v_proj` | 3 |
| `o_proj` | 4 |
| `gate_proj` | 4 |
| `up_proj` | 3 |
| `down_proj` | 3 |

168 quantized modules, group size 128, asymmetric, `lm_head_quantized: false`.

| | bytes |
|---|---|
| packed weights | 163 258 368 |
| the same weights dense | 715 653 120 |
| dense remainder (embedding, norms, biases) | 272 412 416 |
| total, manifest | 435 670 784 |
| `model.safetensors` on disk | 435 739 568 |
| average bits over quantized elements | 3.65 |

626 tensors, **unfused names** on disk (`q_proj`, `k_proj`, `v_proj` separately) — the servers
fuse them at load time, which is the whole subject of §4.

Two properties of this bit map are deliberate and both earn their place:

**The fused shards disagree.** q at 4 bits with k and v at 3 is the configuration no other
quantization method produces, and the one the flat per-shard buffer layout exists for. A
uniform map would pass even if the layout arithmetic were wrong, because every shard would
have the same words-per-row and any offset error would land on a boundary that happens to be
correct.

**The embedding stays dense.** vLLM's plain `VocabParallelEmbedding` gets its unquantized
method from the plugin, so a checkpoint carrying `embed_tokens.qweight` would have no loader
on the vLLM side. Qwen2.5-0.5B ties its head to that table, so leaving it dense keeps both
ends consistent.

## 3. Protocol

Five stages. The design constraint throughout is that every number in the final table must be
attributable to exactly one difference.

### 3.1 Export, and the identity check that licenses everything after it

`01_export.py` writes the packed directory. `02_direct.py --packed` then rebuilds the
quantized model the way `dynquant quantize` does — load the dense weights, run `pack_model` —
and **asserts the resulting buffers are bit-identical to the exported checkpoint's tensors**
before measuring anything. **504 buffers compared** across 168 modules (`qweight`, `scales`,
`offsets` each), zero differences.

That assertion is what makes the later comparison mean anything. If the two encodings
differed, a server-vs-direct gap would be measuring the exporter, and it would look exactly
like a kernel bug.

### 3.2 One prompt set, tokenized once

Twelve prompts, `MAX_TOKENS = 32`, 108 scored logprob positions in total.

```
"The capital of France is"
"In one sentence, explain what a transformer neural network does."
"def fibonacci(n):\n    "
"The three primary colors are"
"Q: What is 17 multiplied by 24?\nA:"
"Once upon a time in a small village by the sea,"
"The chemical symbol for gold is"
"List three causes of the French Revolution:\n1."
"Translate to French: 'Where is the train station?'\n"
"The derivative of x^2 with respect to x is"
"A SQL query that selects all rows from a table named users:"
"Summarize: The mitochondria is the powerhouse of the cell."
```

Both sides are handed **token ids, not text**, tokenized once and cached to `ids.json`. vLLM,
SGLang and `transformers` can disagree on `add_special_tokens` and on whether a chat template
is applied, and either difference would show up in the comparison as a serving-path
discrepancy when it is nothing of the kind.

### 3.3 What is measured, and what each quantity is worth

Three quantities per arm, and they are not equally trustworthy.

**Teacher-forced logprobs** — at each prompt position, the logprob of the token actually
there. Every position is conditioned on the real prefix, so a disagreement is one
disagreement rather than the first one plus all its consequences. This is the primary
statistic.

**Teacher-forced argmax** — the token the model *would* emit at each position. Also
non-compounding. Reported as `top1_agreement`.

**Greedy continuations** — 32 tokens, `temperature=0.0`. Reported, but the weakest evidence
here: one divergence at token 9 changes every token after it, so the statistic measures chaos
rather than correctness. It is in the table because leaving it out would look like hiding it.

The direct reference measures one prompt at a time rather than batched: padding changes
attention masks and the reduction order inside the GEMM, and a batched reference would differ
from a server that schedules the same prompts differently for reasons that have nothing to do
with quantization. Both servers are likewise run with `max_num_seqs`/`max_running_requests`
of 1, eager execution, no CUDA graphs, no radix cache. This is a correctness comparison, not
a throughput one.

### 3.4 The three pairs, and why the third one is load-bearing

vLLM and `transformers` never agree bit-for-bit even on an unquantized model — different
attention kernel, different RMSNorm, different GEMM reduction order. Neither do SGLang and
`transformers`. So "the DynQuant logprobs differ by 9e-3" is not interpretable on its own.
Two reference quantities put it in scale:

* **the fp16 pair** — the same measurement with quantization removed. Whatever the two
  runtimes disagree about for reasons of their own shows up here.
* **the quantization effect** — direct-fp16 against direct-packed, both in the same process.
  This is how far quantization moves the model at all, and it is the yardstick. A serving gap
  far below it means the server is reconstructing the same weights, whatever the last decimal
  place says.

Both servers are compared against the **same stored `direct_fp16.json` and `direct_dq.json`**,
so the two gaps are distances from the same point and are directly comparable to each other.

### 3.5 Nothing DynQuant-specific is passed to either server

Neither server run passes a `--quantization` flag, and neither runs a patched server. The
plugins register through the servers' own plugin entry points and `config.json` in the
exported directory names the method:

```
vLLM   resolved quantization='dynquant'
SGLang resolved quantization='dynquant'
```

If either script needed an argument the checkpoint did not already carry, the integration
would not be finished.

## 4. The defect this experiment found

The first real SGLang serve started cleanly, answered prompts, and was wrong.

### 4.1 Symptom

The load emitted a long run of warnings naming parameters that could not be found, including
mangled names like `qkqkv_proj`, and then started the server anyway. Generation produced
text. Nothing in the exit status or the API response indicated a problem.

### 4.2 The misleading intermediate

The `qkqkv_proj` name is real but is downstream noise. `srt/models/qwen2.py:614-639` loops
over `stacked_params_mapping`, rebinds `name` in place, and `continue`s carrying the rewrite
after a failed lookup — so `"qkv_proj".replace("v_proj", "qkv_proj")` fires a second time. Per
buffer per layer the loop generates `q_proj` → [`qkv_proj`, `qkqkv_proj`], `k_proj` →
[`qkv_proj`, `qkqkv_proj`], `v_proj` → [`qkv_proj`]. Chasing this would have produced a fix
to a symptom.

### 4.3 Root cause

SGLang's `_get_quantization_config` reads `getattr(model_class, "packed_modules_mapping", {})`
(`model_loader/loader.py:204`) and injects it into the dict handed to `from_config`
(`model_loader/weight_utils.py:278`). On SGLang 0.5.16 that attribute is **absent from 172 of
the 210 files** in `srt/models/` — including `Qwen2ForCausalLM`, which fuses q/k/v inside
`load_weights` all the same.

An AST survey of all 210 model files in that release:

| | files |
|---|---|
| fuse `qkv_proj ← q/k/v_proj` inside `load_weights` | 70 |
| fuse `gate_up_proj ← gate_proj/up_proj` inside `load_weights` | 71 |
| declare a class-level `packed_modules_mapping` | 38 |
| of those 38, declare exactly the two-entry modal literal | 25 |

So an empty mapping reached the plugin, `resolve_shards` returned `None` for every fused
prefix, `get_quant_method` fell through to `UnquantizedLinearMethod`, `params_dict` came to
hold `qkv_proj.weight` rather than `qkv_proj.qweight`, and every packed tensor was dropped
with a `logger.warning`. The server then served uninitialised buffers.

vLLM never showed this because vLLM declares `packed_modules_mapping` on essentially every
model.

### 4.4 Fix

Two changes, one on each side of the boundary, both bounded.

**A fallback in the schema.** `serving_common/schema.py` gained
`CONVENTIONAL_FUSED_MODULES = {"qkv_proj": ("q_proj","k_proj","v_proj"), "gate_up_proj":
("gate_proj","up_proj")}`, consulted by `resolve_shards` only through an explicit
`fusion_defaults=` keyword, and only when the framework declared nothing *and* the checkpoint
has no tensor at the fused prefix. The two entries are the modal declaration copied verbatim
(25 of the 38 files that do declare one). Nothing rarer belongs there: `W_pack`, `c_attn`,
`fused_qkv_a_proj_with_mqa` and `in_proj_qkvz` appear only in files that declare their own
mapping. The guess is bounded on both sides — it applies only when the prefix is absent from
the checkpoint, and `resolve_shards` is all-or-none, so a partial match raises rather than
silently quantizing half a layer.

**A guard in the plugin.** `sglang_plugin/config.py` gained
`_refuse_an_empty_fused_layer`, which raises `DynQuantError` when the layer being built is a
`QKVParallelLinear` or `MergedColumnParallelLinear`, resolves to no shards, and has quantized
siblings in the checkpoint. The class test is what makes it precise. An earlier version of
this guard tested only for quantized siblings and lived in `schema.py`; it would have raised
on an `o_proj` deliberately left dense beside quantized q/k/v, which is a legitimate
configuration the test suite already pins. Only the layer's *class* separates "a fused layer
whose shards all vanished" from "a layer someone chose not to quantize", and that class is
visible in the plugin, not in the schema.

The complementary false positive is also guarded: a fused layer inside a region the exporter
left alone entirely — a dense vision tower — has no quantized siblings and is not refused.

### 4.5 Verification

| | |
|---|---|
| dropped-parameter warnings on the re-run | **0** |
| plugin + schema tests, Windows | 115 targeted, 1115 passed / 13 skipped full suite |
| tests against real SGLang 0.5.16 on the box | 126 passed |
| `ruff check` / `ruff format` | clean |

Eleven new tests were added, and each one is written against a specific future regression: a
fused prefix that resolves only via the default; a declared mapping never being overridden by
a default; a checkpoint already fused on disk not being second-guessed; the `W_pack`
checkpoint that must raise; the `o_proj` that must not; the dense-tower fused layer that must
not. Three of them are conformance tests that read SGLang's own source with `inspect.getsource`
and fail if `Qwen2ForCausalLM` ever starts declaring a mapping, or if the `continue`-carrying
rewrite in `qwen2.py` is ever fixed upstream — so the stub cannot drift away from the real
thing without the suite noticing.

Commit `9e4c6ed`, "Resolve fused layers on SGLang models that declare no fusion mapping".

## 5. Results

108 scored logprob positions, 12 prompts, 32 generated tokens per prompt. Every arm compared
against the same `direct_*.json` references, all backends CUDA.

| arm | mean \|Δ logprob\| | max | argmax agree | greedy prefix | identical seqs |
|---|---|---|---|---|---|
| vLLM / fp16 | 0.006304 | 0.064087 | 100.0 % | 9.00 / 32 | 0 / 12 |
| vLLM / dynquant | 0.009147 | 0.062078 | 100.0 % | 7.50 / 32 | 0 / 12 |
| vLLM / quantization-effect | 2.461067 | 19.890288 | 44.4 % | 0.42 / 32 | 0 / 12 |
| SGLang / fp16 | 0.006896 | 0.072972 | 100.0 % | 9.00 / 32 | 0 / 12 |
| **SGLang / dynquant** | **0.006822** | 0.046448 | 100.0 % | 7.08 / 32 | 0 / 12 |
| SGLang / quantization-effect | 2.461067 | 19.890288 | 44.4 % | 0.42 / 32 | 0 / 12 |
| cross: dq, vLLM vs SGLang | 0.008205 | 0.048009 | 100.0 % | 29.33 / 32 | 10 / 12 |
| cross: fp16, vLLM vs SGLang | 0.005805 | 0.047184 | 100.0 % | 32.00 / 32 | 12 / 12 |

Derived:

| | vLLM | SGLang |
|---|---|---|
| serving gap ÷ own fp16 gap | 1.45× | **0.99×** |
| quantization effect ÷ serving gap | 269× | **361×** |

### 5.1 Reading these

**The primary claim.** Top-1 agreement is 100 % at all 108 positions on both servers on the
quantized model, and the mean logprob difference sits two and a half orders of magnitude
below what quantization itself does to the same numbers. Both servers are holding the same
quantized weights the direct run holds.

**SGLang's 0.99× is not better engineering.** It means the quantized gap and the fp16 gap are
the same size, which is what "no additional discrepancy from the quantized path" looks like.
vLLM's 1.45× is a larger ratio of a very small number; both are far below the yardstick and
the difference between them is not something 108 positions can resolve.

**The quantization-effect row is identical across both blocks** — 2.461067 in both — because
it is the same comparison of the same two stored files, included in each block so neither
table has to be read against the other to be interpreted.

**The greedy-prefix column falls from 9.0 to ~7 on the quantized arms in both servers.** This
is the compounding statistic and it is doing what compounding statistics do. Note that the
fp16 arms only agree for 9 of 32 tokens either — a server and `transformers` diverge on greedy
decoding within a few tokens regardless of quantization, which is precisely why §3.3 ranks
this evidence last.

### 5.2 One number that is not yet explained

The cross-runtime rows are the anomaly worth flagging.

On fp16 the two servers generate **identically for all 12 prompts** (mean prefix 32.00/32 —
i.e. every token of every continuation). On the quantized checkpoint that falls to **10 / 12
identical** with a mean prefix of 29.33 / 32.

Argmax agreement between them is still 100 % at every scored position, so this reads as ties
breaking differently very near a decision boundary rather than as a numerical disagreement —
two logits within float noise of each other, and each server's reduction order picking a
different one. It is consistent with the logprob column, where the two servers differ by
0.008205 on the quantized model against 0.005805 on fp16.

But it is the one measurement where the quantized path is *measurably* looser than fp16, and
it has not been chased to ground. Until it is, this report claims teacher-forced parity, which
is what it measured, and does not claim generation-identical parity on the quantized path,
which it did not.

## 6. The scripts

`/workspace/vllm_parity/parity/` on the box; the tree is not under version control there, and
this is the record of it.

| script | what it does |
|---|---|
| `_common.py` | the 12 prompts, `MAX_TOKENS`, `token_ids()` — tokenize once, cache, hand ids to everything |
| `01_export.py` | packs Qwen2.5-0.5B-Instruct with the per-leaf width map, writes `parity_bits.json` alongside |
| `02_direct.py` | the reference: `transformers` + `dynquant.runtime`, `--packed` re-packs and asserts bit-identity against the checkpoint first |
| `03_vllm.py` | vLLM arm — `prompt_logprobs=1`, `enforce_eager`, `max_num_seqs=1` |
| `03_sglang.py` | SGLang arm — same five JSON keys; asserts SGLang's `input_token_logprobs` line up with the prompt and that it skipped at most one leading position, so the two servers' arrays are the same 108 positions |
| `04_compare.py` | the pair table and the two ratios |

`parity_bits.json` is written by the exporter and read by the direct run rather than being
read back out of `config.json` — `config.json` is the loader's copy, and reading it back would
make a round-trip bug invisible to the comparison.

Driver scripts: `/workspace/run_parity.sh` (vLLM), `/workspace/run_parity_sglang.sh` (SGLang
plus the two cross-runtime pairs). The SGLang driver reuses `direct_fp16.json` and
`direct_dq.json` verbatim rather than regenerating them.

## 7. What this establishes, and what it does not

**Established.** Two independent inference servers load a `dynquant-packed` checkpoint with
no server-side patch and no command-line flag, fuse its unfused per-shard tensors into their
own `QKVParallelLinear`/`MergedColumnParallelLinear` layers with mixed widths across the
shards (4/3/3 and 4/3), and reproduce the direct GPU run to 100 % teacher-forced top-1
agreement and a mean logprob difference two and a half orders of magnitude below the effect
of quantizing at all. The packed buffers were verified bit-identical to the exported
checkpoint before any of it was measured. A checkpoint written by `dynquant-core` 0.1.2 was
read by 0.2.0 without conversion.

**Not established.** Throughput or latency — every arm runs eager, single-sequence, with
CUDA graphs and prefix caching disabled, because those change the numerical path. Tensor
parallelism beyond one rank: the plugin's TP placement logic is unit-tested but a real
two-rank engine needs a two-GPU box and has not been run. Any model other than Qwen2.5-0.5B,
any architecture other than dense-fused, and in particular MoE, MLA and SSM layers, none of
which this checkpoint contains. A quantized embedding or LM head — deliberately dense here.
And, per §5.2, generation-identical parity between the two servers on the quantized path.

---

*Related: [`docs/sglang-integration-plan.md`](../sglang-integration-plan.md) for the plugin
design and the S0–S8 staging; [`docs/format-spec.md`](../format-spec.md) for the checkpoint
format these servers read; [`docs/reports/README.md`](README.md) for the rest of the
experimental record.*
