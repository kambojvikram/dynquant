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
| **model** | 2 models × Tulu-3 | does the result hold across attention structure |
| **dataset** | 1 model × {SmolTalk, OpenThoughts3} | are the conclusions data-specific |

Six fine-tunes, not twelve — and **four in the end**, once the panel was settled at two models
(see *Licensing*). Pick the dataset-sweep model by what S1 shows most damage on — almost
certainly Phi-4-mini, which is also the cheapest to train three times.

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

### G2 — Pack / unpack and kernel parity on each model's real shapes ✅ **done 2026-08-03**

The 417 kernel-parity tests sweep geometries, not these geometries. Gemma-3 uses `head_dim=256`
(not `hidden/heads`), Phi-4-mini has a partial rotary factor, and the four models' GQA ratios
differ. One pack → unpack → compare-against-the-torch-oracle run per model's actual layer shapes,
at 2/3/4/8 bits. Cheap, and it is the only thing standing between a shape assumption and a
silently wrong checkpoint.

#### What it found

`tests/test_quant_phase3_shapes.py` — 302 tests over 40 pinned weight shapes covering all four
models, plus three phase-3 reduction widths appended to `tests/test_kernels_parity.py`. Suite
1229 → **1531 passed, 14 skipped**; ruff clean; **8 of 8 mutations killed**, including one per
defect below.

The shapes are **pinned config literals, not fetched** — two of the four repos are gated (HTTP
401 without a token), so a fetching test would be red on any machine without HF credentials. The
literals are cross-checked two ways: a `needs_hf` test builds each model on the meta device and
asserts the built shape set equals the pinned set, and each model's total parameter count must
land within 0.5% of its model card (llama 8,030,261,248; ministral 8,019,808,256; phi4
3,836,021,760; gemma3 4,300,079,472). A transcription error would have to be made identically in
two places to survive. The gated configs were themselves checked against the NousResearch and
unsloth mirrors, which agree exactly on every field used.

The suite was green before any of this, which is the point: **none of the three defects was
covered.**

**1. The budget was blind to group zero-padding.** `allocate.budget.module_stored_bits` priced the
payload as `params * bits` while the packer stores `rows * padded_in_features * bits` —
`padded_in_features` zero-fills the tail group *before* quantization and those pad values are
packed like any other. Undercount measured at real shapes: **1.1%** for Gemma-3's vision `fc2`
(4304 → 4352) and **4.8× (2-bit) to 7.3× (8-bit)** for its `patch_embedding`, whose
`[1152, 3, 14, 14]` folds to 48384 rows of 14 columns and pads to 128. The plan's own P4 gate
requires the manifest number to be the on-disk number; a budget that targets a size the writer
cannot hit fails it silently, and only at the shapes we had never tested.

Fixed by adding `quant.pack.stored_bits()` — the arithmetic counterpart of `QuantTensor.nbytes`,
for callers that must price a tensor before quantizing it — and routing `module_stored_bits`
through it. Both sides now resolve through `row_geometry`, so **a disagreement is not
expressible**, the same argument `RowGeometry`'s docstring already records for the encoder and
the validator. A test asserts predicted == actual *exactly* (`abs=0.0`) at all 40 shapes × 4
widths. Resulting bpw: 2.250/3.250/4.250/8.250 for the ordinary shapes, 2.275/3.286/4.297/8.342
for `fc2`, 20.571/29.714/38.857/75.429 for `patch_embedding`.

**2. Nothing stopped us quantizing a tensor that costs more than leaving it dense.** That last row
of numbers is the defect: at group 128 a row costs one padded group — 128·2 payload bits plus an
fp16 scale and offset, so **288 bits no matter how few columns it holds** — against `16 ·
in_features` for fp16. They cross at 18 columns; 19 make quantizing cheaper. `patch_embedding`
has 14, so every width on offer is worse than not quantizing, and the *narrowest* is the worst
by 1.3×.

`UNQUANTIZED_FLOOR = 16` exists for exactly this case but was keyed on `ModuleRole.LIN_ATTN_CONV`
and `ModuleRole.SSM_CONV` — **a role cannot see a shape**, and a short trailing dimension is a
property of the tensor. Every architecture grows its own; Gemma-3's is a patch embedding, and it
is only the first one we happened to look at. Fixed with `ModuleInfo.pays_for_itself`, wired into
`is_quantizable`, so the module is reported unquantized, lands in `Budget.fixed_bits` at compute
dtype, and never reaches the knapsack. The threshold is asserted at exactly 18/19. Every other
real phase-3 width clears it; the next-narrowest, 1152, clears it by 6×.

**3. The test oracle over-predicted on padded shapes**, by `sqrt(padded / in_features)`.
`_oracle.expected_rel_fro` reasoned that padded positions are exact zeros which dilute the mean —
they do not: the reported error is a ratio and the pad cancels out of numerator and denominator
together. Removed, and the group mean is now weighted by *real* element counts so a tail group
holding 14 of 128 counts for 14.

This one had been sitting there in the green. At the only padded shapes previously under test —
100-of-128 and 320-of-384 — the spurious factor is 1.13× and 1.10×, inside the ±25% band and
pointing opposite to the known `_UNIFORM_ERROR_EFFICIENCY = 0.90` bias. **Two errors cancelling is
the failure mode a test oracle exists to prevent.** It took 14-of-128, where the factor is 3.02×
and the prediction misses by two thirds, to make it visible.

#### Also verified at full scale

Phi-4-mini's fused row partitions land on real boundaries (`qkv_proj` → Q 0:3072, K 3072:4096,
V 4096:5120 — GQA 24:8, not an even third; `gate_up_proj` → GATE 0:8192, UP 8192:16384) and its
tied `lm_head` resolves to `model.embed_tokens` with `floor_bits == 8`. Rows are independent under
group-wise packing — groups run along `in_features` and every row carries its own scales — which
is what licenses probing a 262208-row embedding with 6 rows; it is now asserted rather than
assumed, by packing 37 rows and comparing three ragged slices against the full pack.

### G3 — Three of the five evals do not exist

`dynquant.eval` ships `casehold`, `banking77`, `gsm8k`, plus `compare` (paired McNemar with CIs).
So GSM8K is free; **IFEval, HumanEval and MBPP have to be written**, and the judge evals more so.

Order by value per unit of work:

1. **IFEval** ✅ **done 2026-08-03** — programmatically verifiable, no judge, no execution
   sandbox, and it measures the thing SFT actually changes. Written first.
2. **HumanEval + MBPP** ✅ **done 2026-08-03** — execution-based pass@1. Needs a sandboxed
   subprocess runner with a timeout; the harness does not have one. Non-trivial but bounded.
3. **MT-Bench / AlpacaEval 2** — see §5. Not on the critical path.

Hard requirement on all of them: **store per-item outcomes**. Every task must emit a `hits` array
so every A/B is an exact McNemar test rather than two independent proportions. This is what
halved the standard error in phase 2, and it is what refused to promote a +0.51 that looked like
a win.

#### G3a — IFEval ✅ **done 2026-08-03**

`eval/_ifeval_instructions.py` (the 25 verifiers, ported literally from
`google-research/instruction_following_eval`) and `eval/ifeval.py` (prompting, decode, and all
four official metrics), with `tests/test_eval_ifeval.py` — 51 tests, 9 of them verified to go red
when their fix is reverted. Suite 1138 → **1189 passed, 13 skipped**; ruff clean.

**IFEval has no gold answers.** A response is correct iff a Python function says so, which makes
the scorer the benchmark. That framing is what turned up the four defect classes below; each
would have produced a *number*, not an error.

**1. Double BOS on every chat-templated prompt.** `apply_chat_template` emits BOS, and the
tokenizer's `add_special_tokens=True` emits a second. Llama-3 and Gemma-3 prompts would have
carried two. It reports nothing, costs a few points of instruction following, and is **the same
magnitude as the effect the campaign exists to measure** — arriving from the harness instead of
from the weights. Fixed structurally: `EvalConfig.add_special_tokens`, threaded through both
tokenizer calls in `generate_batched`, defaulting off in `ifeval.DEFAULT_CONFIG` and force-overridden
(with a warning) whenever a tokenizer carries a chat template. Guarded by a spy test on the
config the decode loop actually receives.

**2. Three of the 25 instruction types cannot be scored without `langdetect`** — `language:response_language`
and the two `change_case:english_*` rules. This is not an approximation but an absence, and both
available guesses produce ordinary-looking results: counting them followed inflates, counting them
violated deflates. `evaluate_ifeval` therefore refuses by default (`on_unverifiable="raise"`, with
an actionable message) and under `"drop"` records exactly which prompt keys were dropped.
`langdetect` is now in the `eval` extra.

**3. Scorer drift between machines.** NLTK present on one box and absent on another silently
changes sentence-splitting rules — same code, same checkpoint, different number. Every result now
carries `scorer_fingerprint()` (e.g. `ifeval/regex-sentences+regex-words+langdetect`). Worth being
precise about the exposure: the *paired difference* every claim rests on is immune, because a
splitter that miscounts "Dr. Smith arrived." miscounts it identically for both arms; it is the
absolute number that stops being leaderboard-comparable. And `length_constraints:number_words`,
the commonest constraint, is bit-identical either way — upstream's `RegexpTokenizer(r"\w+")` *is*
`re.findall(r"\w+", …)`.

**4. `EvalConfig` was being rebuilt field-by-field in three task modules.** Adding the few-shot
stop sequence meant re-listing every field, so the rebuild silently reverted any field it forgot —
and the fields it forgets are the ones added after it was written. Already live: `gsm8k`'s copy
omitted `early_stop`, so a caller passing `early_stop=False` got `True`. All three now use
`dataclasses.replace`. Found only because `add_special_tokens` was the next field to be forgotten.

Ported infidelities are preserved deliberately, since fixing them would produce numbers that are
not IFEval numbers: `keywords:existence` matches inside words while `keywords:forbidden_words`
does not; `number_paragraphs` tolerates an empty run at either end but not an interior one;
`nth_paragraph_first_word` counts non-blank paragraphs but indexes the unfiltered list. Each has
a test that pins it.

Two structural choices worth carrying to the remaining evals. **Every checker is built before a
single token is generated** — a malformed kwarg or an uncompilable dataset-interpolated regex
costs two seconds, not a full GPU generation pass (also spy-tested). And per-item vectors are
stored at **two** granularities, `hits` per prompt and `instruction_hits` per constraint, because
prompt-level strict and instruction-level loose run ~10 points apart and a single "IFEval" number
does not say which one it is. All four official metrics are reported.

#### G3b — HumanEval + MBPP ✅ **done 2026-08-03**

`eval/_code_exec.py` (the sandbox, the extractor, and the shared tally), `eval/humaneval.py` and
`eval/mbpp.py`, with `tests/test_eval_code.py` — 41 tests, **all 16 mutations verified to turn
their test red**. Suite 1189 → **1229 passed, 14 skipped**; ruff clean. Subprocesses are real in
the tests; only generation is stubbed.

**These are the only two tasks whose scorer cannot be argued with** — the output is executed and
either passes the assertions or does not. The cost is that the *sandbox* now sits in the
measurement path, and a sandbox's failures produce numbers rather than errors. Six defect classes,
each closed structurally and each with a test:

**1. `exit(0)` reads as a pass.** A candidate calling `sys.exit(0)` or `os._exit(0)` before its
assertions run leaves a zero exit code, and a harness that trusts the exit code scores it correct.
Not hypothetical: models emit `if __name__ == "__main__": sys.exit(main())` unprompted. Closed by
requiring a **sentinel file** alongside the zero exit — written from a guard script's namespace
after `runpy.run_path` returns normally, which `sys.exit(0)` cannot reach (SystemExit propagates
out as non-zero) and `os._exit(0)` never arrives at. The candidate never sees the guard.

**2. A timeout is a property of the machine, not only of the model.** A 5.9 s solution passes
alone and fails under eight-way contention, so the pass rate becomes a function of how busy the
box was — indistinguishable, in the results table, from quantization damage. Every timeout is
**re-run once serially** before it is counted, and `sandbox_fingerprint()`
(`exec/linux/py3.11/rlimits/t=8s/m=4096MB`) records the bounds, the platform and the interpreter,
because a solution using `itertools.batched` passes on one minor version and fails on another.

**3. Environment leakage.** The parent process is a fine-tuning run: it holds the HF token, the
W&B key and the CUDA configuration. None of that belongs in a process executing text a model
wrote. The child environment is an **allow-list**, plus `PYTHONHASHSEED=0` (a solution whose
output depends on set iteration order would otherwise flip between runs and read as quantization
noise) and four `*_NUM_THREADS=1` caps, since a candidate importing numpy inside eight concurrent
workers otherwise starts eight thread pools and the box spends the evaluation context-switching.

**4. Unbounded child output, and `input()` burning the timeout.** stdout to `DEVNULL`, stderr to a
file read **from the end** at 2 KB — a candidate printing in a loop would otherwise fill the
parent's pipe buffer and deadlock the run it was supposed to score. `stdin=DEVNULL` so a candidate
calling `input()` raises `EOFError` immediately instead of being scored as an infinite loop.

**5. An instruct checkpoint scored under the completion framing.** The largest artefact available
on this benchmark, and all four phase-3 models are instruct checkpoints. Concatenating prompt and
generation appends "Sure! Here's the function:" to a function signature; every problem then fails
for a syntax error and the arm reads as catastrophically damaged, with nothing in the output
saying the harness did it. `style="auto"` follows `tokenizer.chat_template`, and the framing
actually used is recorded on the result, because two arms prompted differently are not comparable
and the pass rates alone do not say so.

**6. A truncated fence must still parse.** `max_new_tokens` can cut a block mid-way. A regex
requiring the closing fence finds nothing and scores a nearly complete answer as "produced no
code" — and does so most often on the longest problems, exactly where a quantized model is already
weakest, so the artefact points the same direction as the effect being measured. `\Z` is an
accepted alternative to the close. Two smaller extraction rules matter for the same reason: the
block that *defines the entry point* wins over the first block (models routinely open with a
fenced restatement of the tests), and the prompt's imports are carried into a standalone block, or
every HumanEval problem annotated `List[int]` dies with a `NameError` the model did not cause.

MBPP is paired with HumanEval rather than replacing it: 164 problems cannot resolve a two-point
difference, and MBPP's 500-problem test split roughly triples the sample with independently
written problems. Its asserts are shown to the model because the task statement never names the
function, and the entry point is recovered from those asserts by AST walk, skipping wrappers
(`set`, `len`) and Attribute calls (`math.isclose`). Its prompt is deliberately **empty** at the
extractor — MBPP asks for a whole function, so prepending the English statement would be a
`SyntaxError` on every problem.

Scoring is greedy pass@1, not the unbiased pass@k estimator: sampling turns a two-point effect
into noise, and n=1 collapses the estimator to the mean while yielding the paired vector every
claim rests on. **Execution is opt-in** — both entry points refuse until the caller passes
`allow_execution=True`, because importing an evaluation module should never be enough to run code
a language model wrote. The isolation is against *accidents*, not an adversary; the threat model
is a model you fine-tuned yourself producing a wrong program.

The POSIX-only paths cannot be exercised on the development box, and the campaign runs on Linux,
so they were verified under WSL directly: `RLIMIT_AS` caps a 3 GiB allocation at `memory_mb=256`,
`RLIMIT_FSIZE` kills a 200 MB write, and `killpg` reaps a detached grandchild that a timed-out
candidate had spawned. On Windows there are no rlimits and the wall clock is the only bound, which
is why the fingerprint says so.

### G4 — Evaluate through vLLM, and prove it first ⚠️ **run 2026-08-04: decode fixed, bound unreachable**

Generative eval at this volume through `transformers` is prohibitive — this is the difference
between a campaign that takes a week and one that takes a month. Serve every arm through vLLM,
including the baselines (vLLM serves GPTQ and AWQ natively), and score against the server.

That is free leverage in both directions: it makes the campaign affordable, *and* it turns the
whole thing into a second serving-parity test three orders of magnitude larger than the
12-prompt sweep. But it has to be gated: one arm, scored both ways — direct `transformers` and
through vLLM — must agree within the bound the
[serving-parity report](reports/serving-parity.md) established. Precedent exists: Qwen3.5-2B at
3.25 bits scored 86.96 % both ways, p = 1.0000.

#### How it was built

`eval/backends.py` (new, `VllmBackend`) and a rewritten `eval/harness.py`, with
`tests/test_eval_backends.py` — 14 new tests plus 2 in `tests/test_eval_harness.py`, **all 10
mutations verified to turn their test red**. Suite 1531 → **1547 passed, 14 skipped**; ruff
clean. **No task file was touched.**

That last point is the design. The alternative — a `generate=` injection point on every task's
signature — lets a task be scorable through one path and not the other, and the two would drift.
Instead `generate_batched` dispatches internally on `isinstance(model, EvalBackend)`, so every
task's call site is unchanged and `test_a_task_cannot_tell_which_backend_it_was_given` runs
`evaluate_casehold` through both engines and asserts identical `hits`.

**The harness owns the tokenize/detokenize boundary; a backend's only job is prompt ids →
continuation ids.** This is the same "one resolver means a disagreement is not expressible"
move as `row_geometry` in G2. Every way two engines can quietly disagree about a *prompt*
produces a score a few points off rather than an error, and none of them is expressible when
there is one implementation of each:

- **Ids, not strings, cross the boundary.** vLLM tokenizes with `add_special_tokens=True`, so
  handing it a chat-templated *string* double-BOSes every Llama-3 and Gemma-3 prompt while the
  `transformers` arm gets one. Worth a few points of instruction following, raises nothing, and
  lands in the results table as a "runtime difference" — i.e. as the thing being measured.
- **Truncation is a slice, not a tokenizer setting.** The harness no longer mutates
  `padding_side` / `truncation_side` at all, where it previously set both and restored them in a
  `finally`. A tokenizer shared with a training pipeline can no longer pick up an evaluation's
  settings, and truncation no longer depends on what the caller had configured.
- **`_in_request_order` sorts by `int(request_id)`.** `LLM.generate`'s request counter persists
  across calls, so a second call's ids start mid-stream rather than at zero. The mutation pass
  is what showed this matters: the plausible-looking rule — check for a permutation of
  `range(n)`, else leave the order alone — **passes** the out-of-order test and is caught only by
  the non-zero-start one. It would sort correctly on the first task of a run and silently stop
  on every task after it.
- **`_assert_aligned` compares the echoed `prompt_token_ids`, not their length.** Weakening it
  to a length check also survives the misalignment test unless the fake sends a same-length
  wrong prompt, which is why it does.

Two scope calls, both stated in the code rather than left implicit. **Offline `vllm.LLM`, not an
HTTP server**: same scheduler, same paged attention, same kernels, but an OpenAI-compatible
endpoint adds a process boundary and a *server-side* detokenization step, which gives away the
"only the engine differs" guarantee. And **`EvalConfig.batch_size` is a `transformers`-path knob
that `VllmBackend` ignores** — throttling continuous batching to 32-request waves gives back most
of why the engine was chosen — documented on both classes rather than silently dropped.

#### The gate itself

`scripts/gate_runtime_parity.py` scores one checkpoint twice on one task — once through
`AutoModelForCausalLM`, once through `VllmBackend.from_pretrained` — and runs `compare_paired`
over the two hit vectors. It needs a GPU with vLLM installed, so it is a campaign-time gate, not
a CI one; the unit tests fake the engine at its documented surface, and that fake is a claim
about vLLM's API which a vLLM release can invalidate without turning a single test red.

**The pass condition is equivalence, not agreement.** Identical generations is the wrong bar and
the serving-parity report already measured why: on fp16 the two runtimes share only 9 of 32
greedy tokens on some prompts, because ties near a decision boundary break differently under
different kernel orders, while top-1 agreement stays at 100 %. So the gate requires the paired
95 % interval on the score difference to lie entirely inside `--max-delta` (default 1.0 point).

That leaves three outcomes, not two, and they are told apart by whether the interval **excludes
zero** — not by how wide it is. Inside the bound is a pass, even when the delta is
statistically significant, because equivalence is the claim. An interval that excludes zero
means the arms really differ and more data will not change that. An interval that contains zero
but exceeds the bound means too few problems were scored to tell — and a gate that accepted
"not significantly different" on 40 examples would pass anything. The first real run got the
second and third of those confused; see below.

Two further checks, because passing is not the same as measuring. Both arms must beat the task's
chance floor, since two equally destroyed arms agree perfectly. And the `transformers` model
must actually leave the GPU before the engine is built — vLLM sizes its KV cache from what is
free at construction, so a lingering reference is either an OOM or a quietly smaller cache.

#### What the first two runs found — three defects

The smoke run on `Qwen/Qwen2.5-1.5B-Instruct` (GSM8K, 100 problems) came back 23 points apart:
vLLM 60.00 %, transformers 37.00 %. Qwen publish ≈ 60 % for that checkpoint, so the arm that
was wrong was the *direct* one — the opposite of what a serving gate is usually built to catch.

**Two decode defects, in [`reports/decode-neutrality.md`](reports/decode-neutrality.md).**
`model.generate` merges the checkpoint's own `generation_config` **underneath** the caller's
keyword arguments, and this one sets `repetition_penalty: 1.1` for chat, so it was applied on
every greedy generation through `transformers` and on none through vLLM.
`greedy_generation_config` now pins every field of `NEUTRAL_DECODE` **by name**; the vLLM arm's
mirror-image hole is pinned the same way. Separately, the gate's verdict branched on interval
*width* before asking whether the interval excluded zero, and so reported a real disagreement as
"too few problems to tell" — which sends the operator to score more problems, the one action
that cannot help.

**A third defect, found by dumping the generations.** `FEWSHOT_STOP` was `"\n\nQuestion:"`, the
separator `build_prompt` uses, and the model writes `"the answer is 366. Question: There are 12
more green apples…"` — same line, one space. Nothing matched, generation ran to
`max_new_tokens` through invented problems, and the fallback extractor answered one of those.
The stop is now the bare `"Question:"`, worth 24 points on the smoke block. Full diagnosis, plus
the two explanations that fitted the data and were wrong (padded batching, different inputs), in
[`reports/runtime-parity-gap.md`](reports/runtime-parity-gap.md).

**All three were real, and the middle one was retracted by mistake.** The decode fix appeared to
buy nothing — the `transformers` arm scored 37.00 % again, unchanged to the problem — and was
written up as a refuted cause. It had not run: transformers 5.x refills a passed config's unset
fields from `model.generation_config`, so building a fresh one stopped replacing the
checkpoint's somewhere in the 4.x → 5.x transition, and every test stayed green because they
assert on the config the harness builds rather than the one `generate` uses. The full
1319-problem gate then failed at **−9.17 points** where the 100-problem smoke had tied, because
block 0 is the only block of GSM8K where these arms agree. Isolated field by field on problems
100–199: 42/100 as shipped, **61/100** with `repetition_penalty` alone pinned. A
`transformers-lines` CI job now runs the suite on 4.56.2 and 5.14.1, since on 4.x the broken and
fixed programs are indistinguishable.

The residual matters for how this gate is read: with identical token ids, identical weights and
greedy decode, the two arms still diverge at the *first token* — `sdpa` and FlashAttention-2
reduce a 1111-token prefill differently in bf16 and the argmax is close to tied after
`Answer:`. So G4 can never certify that two engines produce the same text, only that they
produce the same score within a bound. The second is what the campaign needs.

#### What the fixed gate measures, and why it changes the campaign's shape

Re-run on all 1319 problems with the decode pinned: transformers **61.49 %** against vLLM's
62.32 % (unchanged from the failing run, which is the check that only one arm moved). Delta
−0.83, 95 % CI [−2.19, +0.52], *p* = 0.27, agreement 93.71 %, and the 83 remaining disagreements
split 36 transformers-only to 47 vLLM-only — the shape of first-token divergence, not of a bias.

**The gate still fails, and cannot be made to pass on this task.** The interval is 2.71 points
wide against the ±1.00 bound. At 6.29 % discordance a ±1.00 half-width needs about 2418
problems; GSM8K's test split *is* 1319. Holding n fixed instead, discordance would have to fall
to 45 — below the floor two independent bf16 attention implementations put under it. The gate
now says so rather than advising "score more", which is the one action the operator cannot take.

So the number to plan around is that **cross-engine parity here is certifiable to roughly ±1.4
points**, and phase 2's headline margin over GPTQ was +1.54. Mixing engines inside one
comparison would spend most of a real margin on runtime noise. **Every arm that is compared to
another arm is scored through the same engine — vLLM throughout, which is also 33× faster (20.7 s
against 682.5 s here).** G4 stays as a standing check that the reference implementation still
agrees to within what the task can resolve; it is not a licence to mix.

Scope of the decode defects, measured rather than assumed: `Qwen3.5-2B-Base` ships no
`generation_config.json` at all and `Mistral-7B-Instruct-v0.3` ships only token ids, so **no
phase-1 or phase-2 number changes**. Phi-4-mini and Ministral-8B are likewise clean; the two
gated models could not be checked without an accepted licence.

The stop-sequence defect is GSM8K-only, and the one earlier GSM8K measurement — the headroom
screen's `Qwen3.5-2B-Base` at 66.11 % — is not obviously damaged: a *base* model given
few-shot exemplars imitates their format, separator included, which is precisely the case
where the old stop matched. The defect also only ever costs points, barring a coincidence
where a run-on's last number happens to equal the gold. So 66.11 % is a floor on that number,
not a ceiling, and the conclusion drawn from it — that GSM8K sat too near its ceiling to show
a fine-tuning gain — is if anything strengthened. Worth a cheap re-check when the box is
otherwise idle; not worth blocking S1 on.

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

**S1 is done — go, on two of the four models.**
[`docs/reports/phase3-s1-headroom-screen.md`](reports/phase3-s1-headroom-screen.md).

| model | IFEval | GSM8K | HumanEval | MBPP |
|---|---:|---:|---:|---:|
| Phi-4-mini-instruct | 68.76 % | 83.17 % | 77.44 % | 60.00 % |
| Ministral-8B-Instruct | 54.53 % | 80.89 % | 79.27 % | 55.80 % |

No pair is above ~90 %, so all four benchmarks stay in the headline; the highest arm leaves 16.8
points of room. Llama-3.1-8B-Instruct and gemma-3-4b-it are **not** screened, and as of
2026-08-05 are **out of the campaign** rather than pending — both are `gated=manual`, licence
acceptance is per-HF-account through the web UI, and no token is being supplied. **Phase 3 is a
two-model panel by decision**; see *Licensing* for what that panel does and does not cover.

The screen also caught two harness defects, each of which produced a stable wrong number rather
than an error, together worth up to 56 points on a single arm: `tokenizer.chat_template` is not
a capability test for a tekken-backed tokenizer (`8336aab`), and a chat frame rendered to text
does not survive re-tokenization back into control tokens (`2b62907`). Both are fixed, both
carry regression tests. The provisional dataset-sweep choice (Phi-4-mini) is unchanged: the
screen gives no reason to move it.

### S2 — the fine-tunes, and the loss mask they depend on

Driver: [`scripts/run_s2_finetune.py`](../scripts/run_s2_finetune.py). Before any of the ~30
GPU-hours is spent, one thing has to be verified rather than assumed: **which token positions
the loss is applied to.** A wrong mask does not raise — it trains a slightly worse model and
reports success, and every arm in S3 is measured on top of that model.

**The mask is verified on both models, and the fine-tunes launched 2026-08-05** — Phi-4-mini
then Ministral-8B, sequentially on the one GPU, 50 000 conversations each, LoRA r=32, effective
batch 32, one epoch. [`docs/reports/phase3-s2-loss-masking.md`](reports/phase3-s2-loss-masking.md).

| model | mask mode (auto-selected) | unmaskable | over `--max-len` | supervised |
|---|---|---:|---:|---:|
| Phi-4-mini-instruct | `template` (probe 30/30 tie) | **0.00 %** | 3.67 % | 70.8 % |
| Ministral-8B-Instruct | `assemble` (probe 0 vs 30) | **0.07 %** | 4.10 % | 70.5 % |

3 000 Tulu-3 rows each, `--max-len 2048`. The two drop classes carry separate ceilings on
purpose: over-length is a budget set by a flag (`--max-length-drop-rate`, 0.15), unmaskable is a
broken assumption about the tokenizer (`--max-drop-rate`, 0.05). Both models are under both.

The per-turn walk that works on Phi drops **3 000 of 3 000** rows on Ministral —
`mistral_common` refuses to render any assistant-final conversation, because it is validating a
serving request. `assemble` mode renders each assistant turn open with
`continue_final_message=True` and closes it with a terminator measured from three synthetic
renders. On Phi, which accepts both modes, they agree on every id and every span start over 495
of 500 rows. Three earlier designs passed a stub test suite and failed on real tokenizers, one
of them at a 95 % success rate on 3 000 rows — the report is mostly about that.

**Launch configuration**, and the one place the two arms differ:

| | Phi-4-mini | Ministral-8B |
|---|---:|---:|
| micro-batch × accumulation | 4 × 8 | **2 × 16** |
| effective batch / steps / LR | 32 / 1 501 / 1e-4 | 32 / 1 501 / 1e-4 |

Phi holds 86.7 GB of a 97 GB card at micro-batch 4, and Ministral's activations are ~1.7×
larger (36 × 4096 × 12288 against 32 × 3072 × 8192), with gradient checkpointing off because
recomputation would fire each forward hook twice and double-count the saliency EMA. Halving the
micro-batch and doubling accumulation keeps everything the optimizer sees identical. What it
does not keep identical is the activation-EMA horizon, which is per micro-batch — so the two
signal maps are comparable in their *decisions*, not their magnitudes, which is what S6 reads.

**GSM8K test leakage: checked, not assumed.** Tulu-3 contributes
`tulu_v3.9_open_math_2_gsm8k_50k` at 5.2 % of the 50 000 selected rows. A 13-gram index over
all 1 319 GSM8K test questions, scanned against the run's own selection, flags **2 items and 0
usable duplicates** (one shared word-problem skeleton with different quantities; one WildChat
conversation quoting the "Janet's ducks" prompt). That caps the effect at 0.15 points against a
+1.54 effect size, so GSM8K stays in the headline — but its post-SFT number is in-domain by
construction and is not evidence of general math gain. S5's mixtures get their own scan:
[`experiments/phase3/s2_masking/gsm8k_leakage_check.py`](../experiments/phase3/s2_masking/gsm8k_leakage_check.py).

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

Gating re-checked against the Hub API rather than read off the model cards, because the two
disagree: a restrictive licence is not the same fact as a gated download.

| | license | `gated` (Hub API) |
|---|---|---|
| Llama-3.1-8B-Instruct | Llama 3.1 Community License | `manual` |
| gemma-3-4b-it | Gemma Terms of Use | `manual` |
| Phi-4-mini-instruct | MIT | `False` |
| Ministral-8B-Instruct-2410 | **Mistral Research License — non-commercial** | `False` |
| Tulu-3-SFT-mixture | ODC-BY-1.0, some constituent subsets non-commercial | `False` |

**Two** of four models need access accepted on the account before they can be pulled — Llama
and Gemma, both `manual`, so approval is a human step with unbounded latency and has to be
requested now rather than at S1. Ministral is *not* gated despite the MRL, which is why this
table separates the two columns: its licence still makes any resulting checkpoint
non-commercial regardless of what the other three allow. Fine for a paper. It should be stated
in the report rather than discovered by someone downstream.

This is why S1 starts with the two ungated models: Phi-4-mini and Ministral can be screened
today, and neither is blocked on an approval that may not arrive.

### Decided 2026-08-05: the panel is two models

Llama-3.1-8B-Instruct and gemma-3-4b-it are **out of phase 3**, not pending. No token for an
account that has accepted both licences is being supplied, so the approval that "may not
arrive" is not arriving, and a campaign does not wait on it.

Everything downstream is stated as a two-model panel from here on. What that panel still
covers, and what it does not:

| | covered by | |
|---|---|---|
| fused projections | Phi-4-mini — `qkv_proj`, `gate_up_proj` | ✅ the row-partition path phase 1 finding 5 turns on |
| dense GQA, unfused | Ministral-8B — separate `q/k/v`, `gate`/`up` | ✅ |
| two tokenizer backends | `TokenizersBackend` and `MistralCommonBackend` | ✅ and they behave differently enough to matter |
| 3.8 B vs 8.0 B | both | ✅ a 2.1× scale span |
| the Llama family | — | ❌ |
| a sliding-window / full alternating stack | — | ❌ (gemma-3 was the only one) |

The two missing rows are stated as scope, not hidden. The generalization claim phase 3 can make
is therefore "across attention structure, projection fusion, tokenizer backend and 2.1× of
scale", which is narrower than the four-model version and is the claim the report will make.
Adding a model later costs one fine-tune plus its arms and invalidates nothing already run,
because every arm is compared against its own model's bf16 ceiling.

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
