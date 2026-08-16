# Changelog

All notable changes to DynQuant are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Three version numbers move independently, and a release note has to say which one
it is talking about:

| Number | Meaning | Breaks what |
|---|---|---|
| `dynquant` / `dynquant-core` | the Python package | imports, CLI, config |
| `KERNEL_ABI_VERSION` | the contract between core and the compiled wheel | a mismatched kernels wheel refuses to load |
| `CHECKPOINT_FORMAT_VERSION` | the on-disk quantized checkpoint | reading older or newer checkpoints |
| `STATS_SCHEMA_VERSION` | the on-disk collected-signals file | reading older or newer stats |

A bump to any of the last three is called out explicitly, because those are the
ones that invalidate artifacts a user has already produced.

## [Unreleased]

### Fixed — phase 2's held claim: the asymmetric GPTQ control has been run

Phase 2 reported DynQuant beating `gptq_3b_head` by +1.54 points at 3 bits, then held the claim:
the GPTQ arm was fitted symmetric while every DynQuant arm was asymmetric, a knob worth 69.4
points on a different panel. The original checkpoint no longer exists, so the control was run as
a replicate — same recipe, fresh fine-tune. DynQuant beats the real asymmetric arm by **+1.13 at
4.5% fewer bytes** (exact McNemar *p* = 0.000178), beats the symmetric arm by +1.90 at 3.79%
fewer bytes, and statistically ties the fp16 ceiling at 5.32× fewer bytes.

It also checks a prediction the byte-accounting fix below made: the original panel's
`gptq_3b_head` byte figure was itself computed under the zero-point bug, and the fix predicted the
replicate's true symmetric arm would land "~0.7% cheaper" — it lands at −0.741%.

Getting here also surfaced an unrelated driver bug: `p2replicate.sh`'s freshness guard used
bash's sub-second `-ot`/`-nt`, which read the training callback's moments file as older than the
checkpoint it describes because the two land a fraction of a second apart within the same save
sequence — both in the same whole second, wrong order at nanosecond resolution. The guard now
compares whole seconds.

Full writeup: [`docs/reports/phase2-asymmetric-control-replicate.md`](docs/reports/phase2-asymmetric-control-replicate.md).

### Fixed — every symmetric baseline was charged for a zero point it never stores

Both baseline stages priced group metadata with `meta_bits = 16 + bits`, charged to every arm.
`compressed-tensors` writes `weight_zero_point` **only when the grid is asymmetric**, so every
GPTQ and RTN arm this project has published was over-charged `bits / group_size` per weight
— 0.7% of a 3-bit checkpoint, in the direction that makes the baseline look more expensive.

Two copies of the line, and the second carried a comment defending it: charging only the
asymmetric arm would make the baselines differ by a *convention* rather than by their weights.
The argument runs backwards. Arms differ in a size column because they **store** different
things; charging both the maximum is what imposes a convention.

[`_llmc.stored_meta_bits`](experiments/_llmc.py) is now the single definition, taking
`symmetric` and `actorder` and charging `weight_g_idx` as an `int32` per input column under
`actorder=group` — which exactly one shipped arm used and none accounted for. Both stages call
it; neither keeps a local rate. `--symmetric auto` resolves **once** per run into
`resolved_symmetric`, read by both the arm record and the size column, because the record
saying `symmetric: false` while the accounting priced `true` is the split that produced this.

[`arms_lfm2.anchor_bytes`](experiments/phase4/arms_lfm2.py) returned one budget per width, so
every DynQuant arm in phase 4 was sized on the asymmetric figure and scored against a symmetric
GPTQ arm — 0.65% to 0.71% more bytes, 6.5–7.1× the panels' own tolerance, one way. It now
returns the cheaper scheme, so DynQuant is pinned under every baseline rather than between
them, and `check_matched` holds each baseline to the size its own scheme predicts while
printing its distance from the anchor.

Ground truth is measured rather than read: [`probe_zero_point_storage.py`](experiments/probe_zero_point_storage.py)
quantizes one tiny model twice with RTN and enumerates the safetensors keys — 0 zero-point
tensors symmetric, 186 asymmetric at exactly `groups × bits`. The arithmetic being corrected is
a dependency's, so its source establishes intent and only the file establishes fact.

No accuracy figure moves and nothing was re-quantized. The restatements, the byte-match
consequence and the corrected sentences are in
[`docs/reports/byte-accounting-zero-point.md`](docs/reports/byte-accounting-zero-point.md).

### Added — `table`, the null that isolates the score channel

`--score-null flat` was reached for as "the real pricing with a constant score" and is not
that. [`apply_null`](packages/dynquant-core/src/dynquant/score/null.py) draws one permutation
per seed and hands the *same* one to `shuffle` and to `flat`, so `flat`'s sensitivity table is
permuted too. The rung beneath it therefore prices a permuted table against no table, which is
a real measurement and not the one the reports were reading off it: on LFM2.5-8B-A1B it came
to **+9.88 points**, more than the whole fine-tune signal is worth, because the rung above it
gives 1.18 back.

`--score-null table` sets every score to 1.0 and passes the measured table through **by
identity** — not rebuilt, not filtered, not permuted. It is the only mode that changes exactly
one of the allocator's two inputs, so `dq_3b − dq_3b_tabl` is the score channel and nothing
else.

It is not a fourth rung on the existing ladder, and it is not off the ladder either: the
nesting over these modes is a partial order, not a line. `shuffle` and `table` are the
incomparable pair — one keeps every magnitude and destroys both correspondences, the other
keeps one correspondence exactly and destroys every magnitude — so no chain holds both. Both
sit above `flat`, which sits above `uniform`. `NULL_CHAINS` now names both chains
(`shuffle → flat → uniform` and `table → flat → uniform`) and `NULL_MODES` names every mode
the CLI accepts. Separate registries because the two facts go stale independently: adding a
mode must extend the second without silently extending the first, and a mode that joins a
chain without earning it turns a partition into a sum of overlapping differences that still
adds up.

The `table` chain is the one to reach for when the question is what each channel is worth.
Against the real arm its rungs are single-channel contrasts — real → `table` moves the score
and nothing else, `table` → `uniform` moves the measured table and nothing else, and the two
sum to the whole signal in two steps. Every middle rung of the `shuffle` chain moves two
things at once, which is why the published four-rung decomposition has a negative middle.

Two guards came with it. The seed test now derives "does this mode draw?" from the code's
behaviour at two seeds instead of pinning a literal list — the old form's stated reason for
being safe was that the deterministic side gains no members, and it just gained one. And
`null_label` names arms by `mode[:4]`, so a mode sharing four characters with an existing one
would silently overwrite its record and map; the naming is now asserted injective over
(mode, seed) arms.

## [0.4.0] — 2026-08-11

`KERNEL_ABI_VERSION` moves 2 → 3, additively: `MIN_KERNEL_ABI_VERSION` stays at 2, so a
0.1.x–0.3.0 kernels wheel already installed keeps working — core feature-detects the new
grouped op and falls back to the per-expert loop, which is slower and not wrong.
`CHECKPOINT_FORMAT_VERSION` stays 2 and `STATS_SCHEMA_VERSION` stays 2, so checkpoints and
stats files written by any earlier release are read unchanged. The kernels wheel version moves
to 0.4.0 regardless — PyPI refuses a filename it already holds — so the meta package's ceiling
moves from `<0.4` to `<0.5`.

This release is what it took to quantize a batched-expert MoE end to end. Every entry below was
found by running LFM2.5-8B-A1B through the pipeline, not by reading the code.

### Added — a batched expert bank stays packed, and its parent indexes it

91.5 % of LFM2.5-8B-A1B's quantizable parameters are two 3-D `nn.Parameter` banks per layer, and
the packed runtime had nothing to put in their place: `pack_model` swaps modules, and a bank is
a parameter. That refusal was the last thing standing between the two DynQuant variants and a
checkpoint that loads.

`DynQuantExpertBank` registers under the parameter's own name and intercepts the one access
every batched-MoE forward makes — `self.gate_up_proj[expert_idx]`, on LFM2, Qwen3-Next and
GPT-OSS alike — dequantizing 11 MiB of a 352 MiB bank per expert. No forward is rewritten and
nothing is dequantized whole, which were the two shortcuts on offer. `nn.Module.__setattr__`
refuses to overwrite a registered parameter with a module, so the swap has to deregister first;
that is now `replace_module`, one copy, used by the packer and the loader alike.
`QuantTensor.rows()` addresses a band of rows as a view — the same addressing the grouped kernel
below needs, so that kernel landed against this interface instead of replacing it.

Two things fell out. `packed_bytes` gained a pass over bare parameters, without which an
unpacked bank and every MoE router sat outside the denominator and each ratio computed from it
flattered us by 91.5 % of the model. And `_shell` now reads the bank's geometry off the model
rather than `spec.out_features`, which is the flattened `E*out` row count and cannot rebuild
rank 3.

### Added — `moe_grouped_gemv`, one launch per bank and no host read of the segment table

The arithmetic is unchanged: each expert's band is the same GEMV either way. What moves is where
the segment offsets live. The loop path calls `.tolist()` on the table once per bank per layer,
which is a device synchronization — 44 per token on the 22-layer, two-bank model this campaign
quantized. The fused kernel derives its launch geometry from `seg_offsets.shape` alone and reads
the values on device, so the caller never syncs and the forward becomes capturable.

The CUDA side clamps a malformed segment table rather than checking it: monotonicity needs a
reduction it would then have to synchronize on, which is the one thing this path exists to
avoid. Validation therefore lives in the CPU reference, which has the whole table in hand. GPU
parity asserts *exact* equality against `dynquant::gemv` band for band — a tolerance would let a
band-addressing bug through whenever the wrong expert happened to produce a nearby number.

### Added — `DynQuantHfQuantizer`, so an unloadable checkpoint says so

`AutoModelForCausalLM.from_pretrained` on an exported DynQuant directory did not fail. With no
quantizer registered, transformers logs *"Unknown quantization type, got dynquant … Hence, we
will skip the quantization"*, reports every packed tensor as an unused key and every real weight
as newly initialised, and returns a **randomly initialised model**. No exception, no non-zero
exit. Publishing a directory with that failure mode invites the conclusion that the quantizer is
bad rather than absent.

`integration/hf_quantizer.py` registers a `DynQuantConfig` and a `DynQuantHfQuantizer`, and both
outcomes are now loud. The load itself copies nothing: the packed modules already register
`qweight`/`scales`/`offsets` under the checkpoint's own keys, so the hook swaps each named module
for a correctly shaped but uninitialised shell and transformers fills the buffers by name. Shapes
come from a new `QuantTensor.empty`, derived through the same `row_geometry` resolver the encoder
used, and hold `torch.empty` rather than zeros — so an unfilled buffer decodes to garbage instead
of to a plausible-looking model. Tied heads survive `tie_weights()`, which matters because
LFM2.5-8B-A1B and Qwen3.5-2B are both tied.

### Added — text-to-SQL over a three-corpus mixture, scored by execution accuracy

Gretel, WikiSQL and sql-create-context, balanced per source and round-robin interleaved so a
truncated run still sees every corpus. Execution accuracy has one failure mode worth naming: two
queries that both return nothing compare equal, so admission requires the database to hold rows,
the gold to find some, and the answer not to be a single all-null or all-zero row.

Three defects closed, all found by screening rather than by a number that looked wrong. WikiSQL
declares its text columns `COLLATE NOCASE`; its condition values are the annotator's typing and
its cells are Wikipedia's, so under SQLite's case-sensitive `TEXT` comparison a third of golds
matched nothing and were refused as *"the gold finds no rows"* — a correct refusal for an
incorrect reason, discarding a third of the corpus. 33 % to 0.4 %. DML golds now carry their own
tally: 10.2 % of Gretel's test golds and 11.3 % of its train golds are
`UPDATE`/`INSERT`/`DELETE`/`CREATE`, already excluded from evaluation but as `empty_result`,
which is the wrong diagnosis for a statement that was never going to match anything — and
training had no row filter at all, so they survived there, teaching a response format
`extract_sql` reads as no answer, scored unparseable on a metric whose floor is zero,
identically across every arm of the comparison. And `eval --task` now derives its choices from
the registry: `text2sql` shipped with a loader, a scorer, a registry entry and two test files and
could not be run, because argparse carried a hand-written copy of the other six task names and
refused it with a usage error, which reads as a typo rather than as the omission it was.

The mixture is also decontaminated against its own benchmark before training.

### Added — `--score-null`, the control that says how much of a win is the signal

A panel showing DynQuant beating a uniform recipe at matched bytes cannot say how much of the
margin is the training signal and how much is mixed-width structure. `apply_null` returns scores
the allocator can still consume but cannot learn from, in three modes. `uniform` sets every score
to 1.0 and drops the sensitivity table, so allocation falls back to pure ROI over params and
floors. `shuffle` permutes scores *within role* — attention keeps attention's score distribution,
routers keep routers' — so the marginal distribution the allocator sees is unchanged and only the
assignment of score to module is destroyed. `flat` is both at once, the same permutation at the
same seed with every score set to 1.0 and the table kept, which is what separates the ranking
from the measured pricing.

The sensitivity row is permuted alongside the score with the same donor map, because on this
checkpoint 8.5 % of parameters are priced by measured `dL` and the rest by a proxy built from the
score; a null that moved one and not the other would leave the measured modules priced by their
own moments and report a partial control as a whole one. `NullReport` records the mode, the seed
and how many modules moved, and composes an allocator label (`sensitivity+null:shuffle(seed=0)`)
that lands in the map, so a control cannot be mistaken for a real arm downstream. The
seed-to-permutation map is pinned by a golden test: arms banked weeks apart are one sample only
if the draw never moved, and nothing else raises when it does.

What it measured, on LFM2.5-8B-A1B at 3.15 bits over 12 000 text-to-SQL items: the three rungs
partition a +19.13-point margin over GPTQ exactly, 92 + 1045 + 1159 = 2296 flipped items —
within-role placement **+0.77**, role assignment plus measured pricing **+8.71**, and
floors-plus-knapsack with no signal at all **+9.66**.

### Added — `QuantTensor.from_codes`, for a checkpoint that is already on a grid

`from_dense` fits the affine map from float weights. That is the wrong operation for a GPTQ or
AWQ checkpoint being re-housed, because re-deriving the map is not the identity: min/max recovers
the original scale only when a group still occupies both ends of its code range, and neither
GPTQ's error compensation nor AWQ's clipping search leaves that true everywhere. Where it is
false the re-fitted step is narrower, the original levels fall between the new ones, and the
weights move.

`from_codes` takes the codes and the map as given and only packs. The format needs no adapter for
this: DynQuant stores `scale * code + offset` with a float offset and no integer zero-point, so
the ecosystem's `scale * (q - zero)` lands on it directly with `offset = -scale * zero`. The
translation is left at the call site, so the only code that knows a foreign convention is the
code reading a foreign checkpoint.

### Fixed — the allocator priced a smaller model than it was writing

`classify_model` records every tensor it declines to classify in `ModelGraph.skipped`,
with a reason. Nothing read that dictionary. `total_params()` summed the quantizable
modules and the ones floored at compute dtype, and `Budget.from_target` divided by that
sum — so refused tensors were documented and unpriced: present in the artifact,
absent from the arithmetic describing it.

The magnitude spans four orders of magnitude and depends on *what* got refused. On
LFM2.5-8B-A1B the refused tensors are 1-D norms and biases, 205,056 bytes against
4.4 GB (0.005%, against a 0.1% match tolerance) — which is why this sat on the list
as "price the rank-1 tensors". But `_expert_bank` also refuses a whole batched bank
whose input axis is not last, and a refused bank is the entire MLP of every layer:
**91.5% of the quantizable parameters** on an LFM2-class MoE. A budget that leaves that
out of the denominator is not off by a header, it is sizing a different model.

`skipped` becomes `dict[str, SkippedTensor]` carrying `num_params` and `tied_to`.
`total_params()` gains a third term, `floor_cost_bits()` and `Budget.from_target` charge
the refused parameters at the 16-bit floor, and `dynquant inspect` writes the count into
the manifest beside `unquantized` rather than inside it. All three refusal sites file
through one recorder that gives the parameters to the first name and records a `tied_to`
on the rest, so two modules sharing one norm cannot pay for it twice — the
tied-embedding error running in the opposite direction.

The safetensors container, 58,880 bytes on the same model, is still not priced and is now
named in the module docstring: its size is a function of the tensor names and offsets,
which do not exist until the allocation being budgeted has been made. A target is hit to
within a header.

No existing bit map is invalidated. The phase-4 panel's arms were allocated under the old
accounting and are reused as-is through `--rescore`, so the correction does not ride along
with the dispatch re-score it would otherwise have been confounded with. Full reasoning in
[`docs/reports/phase4-packed-moe-runtime.md`](docs/reports/phase4-packed-moe-runtime.md)
§10.

### Fixed — CI, red since `a428231`, on two rules the repository set against itself

Nothing here reaches the wheel: `scripts/` and `tests/` are not packaged, so the
v0.3.0 artifacts are what they were. Both failures were in the guard rails.

**The confidential-material guard refused two files `.gitignore` explicitly invites.**
`.gitignore` carries `!**/stats/dynquant_moments.safetensors` as a deliberate negation
of its own `*.safetensors` line — the channel-moment sidecars are measurements rather
than weights, they are the cardinal sensitivity estimator's only input, and the box that
produces them is not a volume, so this repo is where they survive. The guard matched on
extension alone and knew nothing about it, so one file invited what the other refused.
`ALLOWED_BINARY_PATHS` now names the same two exceptions, and is deliberately narrower
than the ignore rule: the directory *and* the filename are pinned, so a checkpoint
copied into `stats/` is still refused.

The reason nothing caught it locally is worth more than the fix. The repository-wide
test scanned `REPO_ROOT.rglob("*")` filtered to `TEXT_SUFFIXES` — so a rule about
`.safetensors` was verified against a file list that could not contain one. It now
scans `git ls-files`, which is the exact list the CI job pipes into the guard, so the
test and the job cannot reach different verdicts.

**Two test modules imported `transformers` at collection time.** The core-only `test`
matrix job installs no transformers on purpose, and both new modules reach it
indirectly — `test_run_s2_finetune` through `run_s1_headroom` to the eval command,
`test_verify_signal_map` through `floor_headroom`, which builds the two real configs.
An unguarded import there is a collection error, not a skip, so six matrix cells went
red over a dependency they were never meant to have. Both now declare
`pytest.importorskip("transformers")` at module scope, matching what the eval and
export tests already do; the two pinned-transformers jobs are where they actually run.

## [0.3.0] — 2026-08-08

No version contract moves: `KERNEL_ABI_VERSION` stays 2, `CHECKPOINT_FORMAT_VERSION`
stays 2, `STATS_SCHEMA_VERSION` stays 2. Checkpoints and stats files written by 0.1.x
or 0.2.0 are read unchanged, and a 0.1.x kernels wheel already installed keeps working.
The kernels wheel version moves to 0.3.0 regardless — PyPI refuses a filename it
already holds — so the meta package's ceiling moves from `<0.3` to `<0.4`.

### Added — `dynquant.integration.sglang_plugin`, reached through SGLang's own entry point

Serving a DynQuant checkpoint on SGLang, with no patch to SGLang. Registration, config,
parameter and linear method, wired to the `sglang.srt.plugins` entry point that
`load_plugins()` calls in the spawned scheduler process.

SGLang's quantization layer is a fork of vLLM v0.5.5/v0.6.4, so most of this is a port.
The parts that are not: SGLang predates registration decorators, so `register()` writes
`QUANTIZATION_METHODS`, `QUANTIZATION_CHOICES` and `WEIGHT_LOADER_V2_SUPPORTED` directly
with each write guarded by a check that names the installed version; `tp_rank` is an
argument to the placement hooks rather than cached state, and the column/row hooks give
it no default so `weight_loader_v2`'s `except TypeError` fallback re-raises instead of
loading rank 0's rows on every rank; `packed_modules_mapping` arrives inside the dict
handed to `from_config` and nothing copies it onto the instance, so the config lifts it
itself.

**And a defect that only a real serve could find.** The first packed checkpoint served on
SGLang loaded every unfused layer, silently dropped every fused one, printed
`SGLang resolved quantization='dynquant'`, started, and answered requests. SGLang injects
`getattr(model_class, "packed_modules_mapping", {})`, and on 0.5.16 that attribute is
absent from **172 of the 210 files** in `srt/models/` — including `Qwen2ForCausalLM`,
which fuses q/k/v inside `load_weights` all the same. `CONVENTIONAL_FUSED_MODULES` now
supplies the modal declaration, applied only where the leaf declares nothing and the
checkpoint holds no tensor at the prefix itself, so a mapping SGLang does declare always
wins. `resolve_shards` is all-or-none, so a wrong guess lands on the same unquantized
path as before rather than mislabelling a shard.

### Added — `integration/serving_common/`, the half of the plugin that fails quietly

`geometry.py`, `schema.py` and `fuse.py` moved out of `vllm_plugin/`, since SGLang's
quantization layer is a fork of vLLM's and both plugins need identical arithmetic. A pure
move, no behaviour change.

`fuse.py` belongs here for a reason stronger than sharing: it registers
`dynquant::fused_shard_concat` in the process-global `torch.library` namespace, so a
per-plugin copy raises on whichever import came second — which a harness comparing both
backends in one process hits immediately.

The split is along how the two halves fail. A framework half fails loudly: a renamed base
class is an `ImportError` at registration. A shard offset wrong by one group loads
plausible weights into the wrong rows and serves a model that is merely slightly worse,
which is indistinguishable from quantization loss. This is also the only half testable off
Linux, since neither vLLM nor SGLang ships a wheel that installs elsewhere.

### Added — IFEval, HumanEval and MBPP, and a vLLM backend for every task

Three tasks join `casehold`, `banking77` and `gsm8k`, and `eval/backends.py` lets any of
them score through vLLM.

**IFEval** ports the 25 verifiers literally from google-research's
`instruction_following_eval`, including where the original is arguably wrong —
`keywords:existence` matches inside words while `keywords:forbidden_words` does not;
`nth_paragraph_first_word` counts non-blank paragraphs but indexes the unfiltered list.
Fixing those would produce numbers that are not IFEval numbers. Each is pinned by a test.

**HumanEval and MBPP** execute the model's output, so the sandbox now sits in the
measurement path and its failure modes produce numbers rather than errors. Six are closed
structurally: `sys.exit(0)` before the assertions no longer reads as a pass (a sentinel
file written from a guard script the candidate cannot reach is required alongside the exit
code); every timeout is re-run once serially before being counted; the child gets an
env allow-list rather than a copy of a parent holding an HF token and a W&B key; stdout to
`DEVNULL` with a 2 KB stderr tail so a candidate printing in a loop cannot deadlock the
run; stdin to `DEVNULL` so `input()` raises instead of burning the timeout.

**The backend boundary carries ids, not strings.** vLLM tokenizes with
`add_special_tokens=True`, so handing it a chat-templated string double-BOSes every
Llama-3 and Gemma-3 prompt while the transformers arm gets one. The harness owns encoding,
decoding and stop truncation; a backend's only job is prompt ids to continuation ids.
`generate_batched` dispatches on `isinstance(model, EvalBackend)` rather than taking a
`generate=` injection point, so no task can be scorable through one path and not the other.


### Added — the phase-3 S2 fine-tune driver, and a loss mask that is measured rather than assumed

`scripts/run_s2_finetune.py`. Everything in an SFT driver is standard except one thing no
dataset ships and no tokenizer API reliably provides: **which token positions are the
assistant's**. A wrong loss mask does not raise — it trains a slightly worse model and reports
success, which is fatal to a campaign built to measure small differences on top of that model.

The obvious method locates turn `i` by the length of the render of `messages[:i]`, which assumes
each rendered prefix is a *token* prefix of the next. Nothing promises that, and neither
tokenizer on the phase-3 panel simply obeys it:

- **`mistral_common` (Ministral-8B) refuses to render any assistant-final conversation** —
  `InvalidMessageStructureException`, because it is validating a serving request. Every prefix
  the walk needs is exactly the shape it rejects: **3 000 of 3 000 rows dropped**.
- **Phi-4-mini closes a conversation it is not asked to continue**, appending `<|endoftext|>`,
  so that render is not a prefix of the training sequence either.

So there are two modes. `template` walks the renders and *checks* prefix-stability per turn;
`assemble` renders each assistant turn open with `continue_final_message=True` — accepted by
both backends, and what the mistral refusal's own error message asks for — then closes it with a
terminator measured once from three synthetic renders. That measurement matters: Phi's turns end
in `<|end|>` (200020) while its `eos_token_id` is `<|endoftext|>` (199999), so a design that
reached for `eos_token_id` would have taught the model to stop with the wrong token. Which mode a
tokenizer needs is not an attribute anywhere, so `--mask-mode auto` masks 32 real rows both ways
and takes the winner.

`return_assistant_tokens_mask=True` is not used and is a trap here: it needs `{% generation %}`
markers, which Phi-4-mini's template lacks, so it returns a full-length **all-zero** mask and
reports success.

Dry runs, 3 000 Tulu-3 rows, `--max-len 2048`: Phi `template`, **0.00 %** unmaskable, 70.8 %
of tokens supervised; Ministral `assemble`, **0.07 %** (2 rows of empty assistant content),
70.5 %. Phi accepts both modes, and over 495 of 500 rows they agree on every id and every span
start, differing only by that one document terminator. The driver refuses to train past
`--max-drop-rate` (0.05); over-length rows are counted separately under
`--max-length-drop-rate` (0.15), because a budget set by a flag and a broken assumption about a
tokenizer are different facts and must not hide behind each other.

28 tests in `tests/test_run_s2_finetune.py`; full write-up, including three designs that passed
a stub suite and failed on real tokenizers — one at a 95 % success rate over 3 000 rows, because
`mistral_common` merges adjacent same-role turns and two one-token errors cancelled — in
[`docs/reports/phase3-s2-loss-masking.md`](docs/reports/phase3-s2-loss-masking.md).

### Fixed — the chat frame was detected and then thrown away on the way to the model

The sequel to the entry below, and the more dangerous half. Having correctly decided that
Ministral-8B-Instruct is an instruct checkpoint, the harness rendered its chat turn with
`apply_chat_template(tokenize=False)` and then re-tokenized that string. Rendering is
lossy for `MistralCommonBackend`: it emits the frame as the *characters*
`<s>[INST]…[/INST]`, and `tekken` never parses control tokens back out of user text —
a deliberate injection guard, not a bug. So the frame survived rendering and died on
encoding. `<s>[INST]Write a function add(a,b).[/INST]` went to the model as **17 tokens
of literal punctuation with no BOS and no `[INST]`**, where the model's own tokenizer
produces 10 beginning `[1, 3, …]`. transformers warns about exactly this
(`apply_chat_template(..., tokenize=False)` "is unsafe … don't encode the output
manually") in a line that a batch evaluation buries.

Handed that, the model mostly returned nothing at all: **120 of 164** HumanEval problems
and **84 of 541** IFEval prompts came back empty, for 23.17% and 37.52%. Both stable,
neither an error, and `prompt_style` now read `chat-template` throughout — so the one
field that caught the previous bug could not catch this one.

`harness.render_chat` replaces the round trip rather than repairing it: it asks for
`tokenize=True` and returns token ids, and `encode_prompts` accepts a prompt that is
already ids and truncates it identically. "Re-tokenizing rendered text reproduces the
render" is an assumption no tokenizer promises to keep, so the harness stops making it.
`ifeval`, `humaneval` and `mbpp` build chat prompts through it.

**No already-collected number changes for a Jinja-backed tokenizer.** The round trip is
lossless there — Phi-4-mini gives the same ten ids either way, verified on the box — and
a test pins that, which is what lets the Phi-4-mini arms stand rather than be re-run.

### Fixed — an instruct checkpoint was measured as a base checkpoint

The harness decided whether to frame a prompt as a chat turn by reading
`tokenizer.chat_template`. That attribute belongs to the Jinja-backed tokenizer, not to
the capability. `AutoTokenizer` returns `MistralCommonBackend` for any Mistral
checkpoint shipping a `tekken.json` — Ministral-8B-Instruct, one of the four phase-3
models, among them — and that class leaves the attribute `None` while
`apply_chat_template` works and renders `<s>[INST]…[/INST]`.

So an instruct model was handed bare text, and an instruct model handed bare text
continues it rather than answering it. On IFEval that put Ministral-8B at **24.77% with
195 of 541 generations empty**, against Phi-4-mini's 68.76% with none. HumanEval and
MBPP took the same misclassification through `resolve_style`, scoring under the
completion framing at 70.73% and 43.60%. GSM8K is untouched — it is 5-shot completion
framing for every model and never consults a template.

None of those is an error, a warning, or a crash. A 24.77% is low enough to read as a
destroyed checkpoint and stable enough to read as a real measurement, which is what
makes this the expensive kind of bug: had it gone unnoticed, every Ministral arm in the
campaign would have been compared against a baseline that was already at the floor, and
quantization would have looked harmless on it.

`harness.chat_prompt_style` now decides by *asking* the tokenizer to render a turn and
seeing whether it can, which distinguishes "will not" from "cannot" — a base checkpoint
still gets the raw prompt, because it genuinely has no turn structure. The four call
sites (three in `ifeval`, one in `_code_exec.resolve_style`) go through it.

The thing that made this visible at all was `IfevalResult.prompt_style`, which records
which framing was used. It is why that field exists, and the argument for keeping this
class of provenance on every result.

Two test doubles had to be corrected to expose the bug: both rendered a template while
reporting `chat_template = None`, a shape no real tokenizer has in either direction.
They now raise, as transformers does for a checkpoint that was never taught a turn
structure.

### Fixed — GSM8K generations ran on into problems the model invented

`gsm8k.FEWSHOT_STOP` was `"\n\nQuestion:"`, taken from how `build_prompt` separates its
few-shot exemplars. Models do not write the separator back.
`Qwen/Qwen2.5-1.5B-Instruct` ends an answer and starts a new problem on the same line —
`"the answer is 366. Question: There are 12 more green apples…"` — so nothing matched:
the stopping criterion never fired, generation ran to `max_new_tokens` through two or
three invented problems, and `extract_answer`'s "last number in the text" fallback
returned an answer to one of *those*. Not an unparseable generation, which is counted
separately and would have shown: a confident wrong number that reads like bad
arithmetic. On the runtime-parity smoke it turned three solved problems out of six into
misses, one-directionally, on the arm that ran on.

The stop is now the bare `"Question:"`, which is also what `lm-eval-harness` uses for
this task. A continuation that has written "Question:" has stopped answering either way.
Diagnosis, the generations it was read from, and the two explanations that fitted the
data and were wrong, in
[`docs/reports/runtime-parity-gap.md`](docs/reports/runtime-parity-gap.md).

### Fixed — a "greedy" decode that was partly the checkpoint's decode

`model.generate` merges the checkpoint's own `generation_config` **underneath** the
keyword arguments it is given, so every field the call site does not name is whatever
the checkpoint author picked for chat. `dynquant eval`'s `transformers` path named
`do_sample`, `num_beams`, `temperature`, `top_p` and `top_k` — and not
`repetition_penalty`, which `Qwen/Qwen2.5-1.5B-Instruct` sets to 1.1. That penalty was
applied to every greedy generation on the direct path and to none on the vLLM path. The
runtime-parity gate is what found it.

A middle version of this entry retracted that, on the grounds that replacing the decode
left the `transformers` arm at 37.00 %, unchanged to the problem. **The retraction was
wrong and is withdrawn.** The replacement did not reach the decode: transformers 5.x
refills a passed config's *unset* fields from `model.generation_config`, so a fresh
`GenerationConfig` no longer replaces the checkpoint's the way it does on 4.x. Isolated
on GSM8K problems 100–199 — 42/100 as shipped, **61/100** with `repetition_penalty`
pinned to 1.0 — the penalty is worth **19 points**, and pinning all nineteen neutral
fields gives the same 61 with per-problem identical hits.

`eval.harness.greedy_generation_config` therefore pins every field of
`eval.harness.NEUTRAL_DECODE` **by name** rather than inheriting any: nineteen fields
whose non-neutral value would move a greedy score, so neutrality is a property of that
dict and not of a library default that can change between majors. The tokenizer's
`eos`/`bos`/`pad` ids are the deliberate exception, and a checkpoint that names no `eos`
is now warned about rather than left to decode silently to `max_new_tokens`.
`VllmBackend` pins the mirror-image four — `repetition_penalty`, `presence_penalty`,
`frequency_penalty`, `min_tokens` — which had been left unpassed because vLLM's own
defaults happen to be neutral.

The suite already carried a tripwire asserting
`GenerationConfig().repetition_penalty == 1.0`. It never fired: the `test` job installs
no transformers, so `pytest.importorskip` skipped the whole decode module while the
campaign ran on 5.14.1. A new `transformers-lines` CI job runs the suite on **4.56.2 and
5.14.1** with transformers installed, verifies the pinned version is the one imported,
and fails if any transformers-gated file skips. On 4.x the broken and fixed programs are
indistinguishable — reverting the fix fails four tests on 5.14.1 and zero on 4.56.2 —
so testing on the version the campaign runs is the guard, not an optional extra.

No published DynQuant number changes. Measured, not assumed: `Qwen3.5-2B-Base` ships no
`generation_config.json` at all and `Mistral-7B-Instruct-v0.3` ships only token ids, so
the phase-1 and phase-2 campaigns never carried a penalty. Full scope, including what a
contaminated path would and would not have invalidated, in
[`docs/reports/decode-neutrality.md`](docs/reports/decode-neutrality.md).

On all 1319 GSM8K test problems the fix moves the `transformers` arm from 53.15 % to
**61.49 %** against vLLM's 62.32 %, unchanged — delta −9.17 → **−0.83**, agreement
73.24 % → **93.71 %**, *p* 1.1e−10 → 0.27, with the 83 remaining disagreements split 36
to 47. The gate still fails, now as the underpowered case rather than as a disagreement.

### Changed — the gate no longer tells you to score problems that do not exist

At 6.29 % discordance a ±1.00-point half-width needs about 2418 problems and GSM8K's
test split is 1319, so "score more" is unfollowable on a run that already covered it.
`_judge` takes `exhausted` (set when `--limit` is absent) and reports the two actions
that remain: make the runtimes agree more closely, or set the bound to what the task can
resolve. `_problems_needed` prints the count the current bound would require, because
the difference between advice that can be acted on and advice that cannot is the number.

The practical consequence, for the campaign rather than for the script: cross-engine
parity on GSM8K is certifiable to roughly ±1.4 points, and DynQuant's phase-2 margin over
GPTQ is +1.54. Arms that are compared to each other should therefore be scored through
the same engine.

### Fixed — CI, red on `main` for three pushes

Two jobs, both introduced by the decode work above and neither caught before the push.

Twenty-three tests across three files gated on `torch` alone, but every path that
*drives* the decode stub reaches `greedy_generation_config`, which imports
`GenerationConfig` when it is called rather than at module import. The `test` job
installs no transformers — deliberately, to prove `dynquant-core` installs on a box that
will never fine-tune — so they failed there with `ModuleNotFoundError` raised from
inside the harness, several frames below anything the tests mention. The gate now sits
in `tests/_decode_stub.py`, the one import all of them share, and the three files are
listed in the `transformers-lines` job's "no gated file may skip" step, so the coverage
moves to the runners that have transformers instead of evaporating.

macOS was red for an unrelated and more interesting reason:
`test_a_memory_bomb_is_bounded_rather_than_taking_the_box_with_it` allocates 3 GiB under
a 256 MB `RLIMIT_AS` and expects the child to die. Darwin accepts the `setrlimit` call
and does not enforce it, so the allocation succeeds. The test now skips there — with the
reason spelled out, because it is a real gap and not a missing feature: **on macOS the
sandbox's memory bound is nominal and the wall-clock timeout is the only enforced one.**
`sandbox_fingerprint` already records `platform.system()`, which is what keeps that from
being invisible in a results table, and the campaign scores HumanEval and MBPP on Linux.

`mypy --strict` reported 34 errors, and four of them were configuration rather than
code. `transformers` ships `py.typed` while annotating almost none of its constructors
and re-exporting its public names through a `_LazyModule`, which under strict mode makes
`from transformers import GenerationConfig` an unexported attribute and
`GenerationConfig(...)` an untyped call. Both are now settled once in `pyproject.toml`
(`implicit_reexport` for `transformers.*`, `untyped_calls_exclude = ["transformers"]`)
rather than by a `type: ignore` per call site, and three such ignores were deleted.
The rest were real: the 25 IFEval instruction builders had no return annotation (now
`_Predicate`, with `_Builder` for the registry), `_preferred_block` returned `Any`,
`phi_fused` read `num_key_value_heads` before the guard that makes its fallback
non-`None`, and `_isolation_kwargs`/`_terminate` branched on `os.name`, which no type
checker narrows — under `os.name` each platform's mypy run reports the *other*
platform's branch, so both are now `sys.platform`.

### Fixed — `scripts/gate_runtime_parity.py` named the wrong failure

The verdict branched on how *wide* the confidence interval was before asking whether it
excluded zero, so the 23-point disagreement above was reported as "too few problems to
tell". That sends the operator to score more problems — the one action that cannot help,
and the expensive one. Equivalence testing has three outcomes and the gate now
distinguishes them: inside the bound is a pass even when the delta is significant;
excluding zero is a real disagreement; containing zero while exceeding the bound is too
little data.

## [0.2.0] — 2026-08-02

A DynQuant checkpoint serves on vLLM. No fork, no patch, no `--quantization` flag,
no code in the user's project:

```bash
pip install dynquant
vllm serve my-org/qwen3-2b-dynquant-3bit
```

and it is measurably the same model. Qwen3.5-2B at 3.25 bits, CaseHOLD, 5,314 items
with per-item hits stored so every comparison is paired (exact two-sided McNemar):

| arm | accuracy | vs the direct run | 95 % CI |
|---|---|---|---|
| direct run | 86.96 % (4621) | — | — |
| vLLM, eager | 86.96 % (4621) | +0.00, *p* = 1.0000 | [−0.13, +0.13] |
| vLLM, inductor + FULL CUDA graphs | 87.00 % (4623) | +0.04, *p* = 0.7905 | [−0.10, +0.18] |
| fine-tuned bf16 | 89.74 % (4769) | −2.75, *p* < 0.0001 | [−3.45, −2.04] |

The bound only means something beside the last row: the serving gap is under a fifth
of a point where quantization itself costs 2.75. An equivalence claim with no
quantization-effect arm beside it is unfalsifiable, which is why that row is in the
table rather than in a footnote.

Python-side only. `KERNEL_ABI_VERSION` is still 2, `CHECKPOINT_FORMAT_VERSION` and
`STATS_SCHEMA_VERSION` are untouched, and no kernel source changed — a 0.1.x
checkpoint serves unchanged and a 0.1.x kernels wheel still loads against this core.
`dynquant-kernels` moves to 0.2.0 with the others only because PyPI will not accept a
second upload under a filename it already holds, and the meta package's kernels
ceiling widens from `<0.2` to `<0.3` so this release can resolve its own wheels.

### Added — `dynquant.integration.vllm_plugin`, reached through vLLM's own entry point

`register()` is wired to the `vllm.general_plugins` entry-point group, so vLLM calls
it once per process during startup — engine, workers and API server alike — and
`quant_method: "dynquant"` in a checkpoint's `config.json` then resolves. Everything
this package touches inside vLLM is a supported extension point or a documented base
class. The one thing that looked like it would force a patch,
`WEIGHT_LOADER_V2_SUPPORTED` being a hardcoded list, has a public decorator for
exactly this case.

The hard part is that **bit width varies per module and vLLM fuses modules**. `q_proj`
at 4 bits and `k_proj` at 3 have 512- and 384-word rows over the same 4096 inputs, so
the fused `qkv_proj` parameter is not a rectangle and no `narrow` expresses it — which
is what every other quantization method relies on. `geometry.py` lays each shard's
packed rows end to end in one flat buffer and hands out views; `parameter.py` overrides
the four placement hooks to translate vLLM's *output row ranges* into flat spans.
Neither imports vLLM, so the arithmetic is unit-tested on a laptop.

Tensor parallelism is covered at the placement level: `tests/test_vllm_tp_placement.py`
builds both ranks of a `tp_size=2` layer and requires them to reassemble the checkpoint
exactly — fused QKV at three different widths, replicated KV heads, `gate_up_proj`, and
row-parallel word-axis splits at all four bit widths, plus the two refusals. A
row-parallel split off a group boundary and a fused row-parallel layer both fail at
`create_weights`, naming the module, before a weight is read. What is **not** covered is
a real two-rank engine, which needs a second GPU; the uncovered part is vLLM's
all-reduce rather than DynQuant's arithmetic, but it is uncovered.

### Added — `fuse.py`, because inductor cancels `split(cat(...))`

A fused layer computes one matmul per shard and joins them. Written as `torch.cat`,
that is correct eagerly and an illegal memory access under `torch.compile`: the caller
splits the result straight back apart, inductor cancels the pair, and vLLM records its
piecewise-boundary strides from fake-tensor propagation which runs *before* the
cancellation. Qwen3.5's gated delta net traced `z` at stride 8192 and got contiguous
stride 2048.

`dynquant::fused_shard_concat` is an opaque custom op with a `register_fake` meta
implementation, so the join survives as a real tensor across the boundary. This is not
a vLLM-specific bug and any port inherits it — anywhere a serving framework splits a
compiled graph and records strides across the split.

### Fixed — the MoE refusal was not firing on vLLM 0.26

DynQuant does not serve a fused MoE layer (the packed grouped GEMM is phase 8), and the
guard that says so by name had stopped matching. Through vLLM 0.25 `FusedMoE` was the
class owning expert weights; in 0.26 it is a factory returning a `MoERunner`, and the
object handed to `get_quant_method` is a `RoutedExperts`. A probe keyed on the class
name matched nothing.

That failure is silent rather than loud, which is why it is worth a release note:
returning `None` from `get_quant_method` means "use vLLM's unquantized method", and
`RoutedExperts._get_quant_method` substitutes `UnquantizedFusedMoEMethod` on `None` —
so a probe that stops matching does not fall through to an error, it falls through to
fp16 experts built for a checkpoint that holds packed words. The probe now keys on the
defining module, and `tests/test_vllm_moe_guard.py` sweeps every `nn.Module` subclass in
the installed vLLM's `fused_moe` package rather than naming classes one by one.

### Fixed — lint and mypy strict pass on the new package

`ruff` at the pinned 0.16.0 and `mypy --strict` are both green again. Every finding was
fixed at the source rather than silenced; the one `noqa` added is a `BLE001` on a
deliberately blind `except` in a test sweep, which is the escape hatch the rule's config
comment describes. `vllm.*` joins the `ignore_missing_imports` list — the `types` job
runs on a CPU runner that deliberately does not install a serving stack — and
`disallow_subclassing_any` is switched off for this package alone, because subclassing
`QuantizationConfig`, `LinearMethodBase` and `BasevLLMParameter` *is* the extension
contract.

## [0.1.2] — 2026-07-30

DynQuant beats GPTQ at 3 bits. On Qwen3.5-2B / CaseHOLD: **89.57 % at
708,087,808 B against GPTQ's 88.03 % at 741,475,927 B** — +1.54 points on 203/121
discordant pairs, exact McNemar *p* < 0.0001, CI [+0.88, +2.21], at 4.5 % fewer
bytes and 0.17 points under fp16. No error feedback, no inverse Hessian, no
sequential column compensation, and no calibration set: everything the allocation
uses was collected by the fine-tune's own hook.

Two of the four levers that produced it ship in this release as opt-in encoder
arguments. The other two — row-partitioning the tied embedding, and per-row body
allocation — are experiment code here and are the next thing to land in core.

Python-side only. `KERNEL_ABI_VERSION` is still 2, `CHECKPOINT_FORMAT_VERSION` and
`STATS_SCHEMA_VERSION` are untouched, and no kernel source changed — a 0.1.1
checkpoint reads unchanged and a 0.1.1 kernels wheel still loads against this core.
`dynquant-kernels` moves to 0.1.2 with the others only because PyPI will not accept
a second upload under a filename it already holds.

### Added — `DEEP_CLIP_CANDIDATES`, a clip grid that reaches the answer at 2 bits

`CLIP_CANDIDATES` stops at 0.80. That is a defensible floor at 4 bits and cannot be
one at 2: with four levels, the MSE-optimal shrink for a roughly Gaussian group sits
well below 0.80, so the search returns the floor on essentially every 2-bit group and
reports it as a win. `DEEP_CLIP_CANDIDATES` continues the same spacing rule down to
0.40. Measured mean chosen ratio: **0.52–0.59 at 2 bits, 0.73–0.86 at 3, 0.88–0.93 at
4** — and at 4 bits and above the extension is *inert*, returning ratios and errors
bit-identical to the shipped grid. The first eight entries are unchanged, so any
difference is attributable to the extension rather than to a re-tuning of what was
already there.

Opt in with `candidates=DEEP_CLIP_CANDIDATES` on `search_clip_ratios`,
`quantize_with_search`, `quantize_tensor` and `quantize_model`. The default is
unchanged, so an existing call reproduces its previous checkpoint exactly.

### Added — a channel-weighted clip objective

`channel_weight` on the search minimises E[x²]-weighted reconstruction error instead
of plain MSE, so a group containing an input channel the network actually drives hard
is allowed to keep its outliers. The vector is **padded to the group boundary, not
broadcast**: a tensor whose input width is not a multiple of `group_size` would
otherwise weight the padding lanes of its last group. A vector of ones reproduces the
unweighted search exactly — bit-identical, asserted in the tests.

`quantize_model(channel_weights=...)` is **all-or-nothing** and raises on a partial
map naming the missing modules. Half a checkpoint encoded against one objective and
half against another is not something anyone can price afterwards, and it is the kind
of mistake that yields a plausible number rather than an error. Pass ones for any
module that should keep the unweighted objective.

### Added — the allocator can price against the grid the quantizer will run

`estimate_sensitivity` and `weight_only_sensitivity` take `candidates=`, and
`estimate_sensitivity` also takes `weighted_clip=`. Pricing widths on the shipped grid
and then encoding on the deep one is a silent mismatch — the estimator reports a cost
the encoder never pays.

This matters more than it sounds, because **the two changes pay through different
stages**, and which one is which is not guessable from the code:

| Change | Re-encode at a fixed map | Re-price, then encode |
|---|---:|---:|
| Deeper clip grid | +0.09 pts | **+0.58 pts** |
| E[x²]-weighted objective | **+0.41 pts** | −0.09 pts |

The grid pays through the allocator: it changes what each width *costs*, so the map
moves. The objective pays through the encoder: it changes what a given width *stores*,
and only 4 of 187 widths move. A change that pays through the allocator and is only
plumbed into the encoder delivers roughly a sixth of its value, silently.

### Changed — the reference experiments

`experiments/four_point/` gains the phase-2 arms (`p2*.py`), the external-baseline
drivers (`stage8_baselines.py`, `stage8_bnb.py`, `run_baselines.sh`) and the full
result ladder in `RESULTS-external-comparison.md` and
`REPORT-quantization-comparison.md`: 17 arms, 2 negative controls, 6 frontier rungs.

`stage5_quantize.py` gains a provenance check, from two failures that each produced a
wrong published number without producing an error. `RUN_DIR` derives from `DQ_MODEL`
and `DQ_TASK`, not from `--model`, so pinning the input pins nothing — four Mistral
arms landed in the Qwen directory, and were evaluated with the Qwen tokenizer, before
anything complained. And a bit map older than the weights it describes is invisible to
every skip-if-output-exists resume guard, because the output does exist; existence
cannot detect a stale map, but ordering can. `--allow-stale` downgrades the check to a
warning for the one legitimate case: deliberately measuring what the staleness cost.

### Added — `docs/reports/`

Phase 1 (the external comparison against GPTQ, AWQ, RTN and bitsandbytes on two
models) and phase 2 (beating GPTQ at 3 bits) as xelatex sources. Built PDFs are
attached to the release rather than committed — the repository-wide `*.pdf` rule
exists to keep a confidential document out of history and is not worth a hole.

### Notes

- The **negative control ships with the feature.** Per-row allocation with a *shuffled*
  row order loses 1.28 points at identical bytes (125/193, *p* = 0.0002) and wins 1.04
  with the real signal. Finer granularity is a multiplier on the signal, not a gain in
  itself, and a granularity change without its shuffled control does not distinguish
  the two.
- **A claim that did not survive its own test is not in here.** `rb_agg` against
  `gptq_3b` is +0.60 with *p* = 0.0685 and is reported as a tie, not a win.
- The largest single lever is not in this release: row-partitioning the tied
  embedding/LM-head tensor is worth **+2.90 points for 7.00 MiB** — 0.41 pts/MiB
  against 0.011 for the model body, 37×. It needs a row-partitioned checkpoint path in
  core, and it needs the allocator to be told the tie's cost up front or the result
  lands ~24 MB over target.

## [0.1.1] — 2026-07-29

A point release for one reason above the others: **install this instead of 0.1.0**,
which shipped a kernels wheel pip would happily pair with a torch it cannot load.
0.1.0 needed `pip install dynquant 'torch==2.7.*'` to work; 0.1.1 carries the bound
in the wheel metadata, so the wrong pairing is now unresolvable rather than silently
degraded.

Everything else here is Python-side. No kernel source changed, `KERNEL_ABI_VERSION`
is still 2, and the checkpoint and stats formats are untouched — a 0.1.0 checkpoint
reads unchanged, and a 0.1.0 kernels wheel still loads against this core.
`dynquant-kernels` moves to 0.1.1 with the others only because PyPI will not accept
a second upload under a filename it already holds.

### Changed — encoding runs on the accelerator, wherever the model lives

`quantize_model` and `pack_model` grew a `compute_device` argument, defaulting to
`"auto"`, exposed as `--compute-device` on `dynquant quantize` and `dynquant eval`
and as `$DYNQUANT_QUANTIZE_DEVICE`. New module: `dynquant.quant.device`.

Encoding a weight reads one tensor and writes one tensor, so it gives the same
answer wherever it is evaluated. The search followed its input's device anyway,
which meant a model held in host RAM did eight candidate encodes, eight decodes and
a grouped error reduction per module on the CPU with an idle GPU beside it. That is
not a corner case — it is every model too large for VRAM, and it is *deliberately*
the case when the point of the run is to measure packed VRAM without a dense copy
ever reaching the device, which is exactly the configuration that most wants the
accelerator and least wants the model on it. Measured on Mistral-7B-Instruct-v0.3 at
4.25 bits, 226 modules: **1685 s on the CPU against 17.6 s on an A100 — 96× faster**,
or 7.5 s per module against 0.08 s.

Weights move one at a time and each packed result is returned to the model's own
device before its module is replaced, so the model does not move, mixed-device models
are not created, and the extra VRAM is one tensor's working set rather than a second
copy of the model: 3.77 GiB of GPU peak to pack a model with 13.5 GiB of bf16
weights. That bound scales with the *largest* tensor, not the model, and at roughly
an order of magnitude over its bf16 size — so on a large-vocabulary model with an
untied `lm_head` it can still exceed a small card. A tensor that will not fit falls
back to its own device for that tensor alone, with a warning; falling back wholesale
would surrender the entire speedup to the single largest weight.

Because it is a performance change it must not be a change in *quality*, and
`tests/test_quant_device.py` pins that at 2/3/4/8 bits. It pins a tolerance rather
than byte-equality, which is worth stating plainly because byte-equality is what an
earlier draft of this entry claimed and measurement refuted. CPU and CUDA encodings
differ in two ways, both negligible and neither fixable by asserting harder: one
group scale in ~10^5 differs by a single fp16 ulp (floating-point contraction in the
clip arithmetic — group min/max and chosen ratios agree exactly, so it is not
reduction order), and the eight-candidate clip search tie-breaks differently on
groups whose top two candidates sit within float noise — 2 groups in 131,072 at 4
bits, none at 2, 3 or 8. Relative reconstruction error differs by at most 1e-6.

The consequence is documented rather than papered over: **a packed checkpoint is
bit-reproducible on a given device, not across devices.** Anything comparing two
encodings for identity — a parity check between a simulated arm and a packed one
most of all — has to encode both on the same device to be measuring what it thinks
it is measuring.

`dynquant eval --map` now records `cuda_pack_peak_bytes` and `cuda_resident_bytes`
in place of `cuda_peak_bytes`. The old key spanned the encode working set, which was
harmless while encoding happened on whatever device the model already sat on and
would have quietly become an overstatement now that it does not. What a reader wants
from a packed model is what it holds; that is measured after the transients are
released, and separately from the peak.

### Added — Mistral-7B-Instruct-v0.3 on Banking77, the second end-to-end run

`experiments/four_point/RESULTS-mistral7b-banking77.md`. All six stages on a model,
task, architecture family and training regime that share nothing with the Qwen3.5-2B
runs: 7.25 B dense GQA with untied embeddings, LoRA r=32 instead of a full fine-tune,
77-way intent classification instead of 5-way multiple choice or grade-school maths.

The fine-tune moved Banking77 **+58.25 points** (36.27% → 94.51%, above the ~93%
BERT-base supervised reference), which is the headroom the dataset screen predicted and
what makes the quantization arms readable. At a 4.25-bit budget quantization costs
nothing measurable (−0.19 pts, p = 0.345) and allocation has nothing to recover (+0.10,
p = 0.375, five discordant pairs out of 3 080). At 3.25 bits allocation beats a
same-size uniform control by **+1.36 points (p = 3.2e−05, 71 problems fixed against
29)**, recovering 53.8% of the damage uniform quantization does, at 4.92× smaller than
bf16.

This is the first **end-to-end** measurement of the `combine="plasticity"` default. That
default was chosen on an in-batch loss proxy after the paper's rank-product score lost
to uniform by 2.03 points on CaseHOLD; it had never been confirmed on accuracy, on any
model. It is *not* a replication of the Gauss-Newton `fisher_diff` result (+10.29 pts on
CaseHOLD) — that allocator was not run here, and the in-batch screen puts it well above
plasticity, so +1.36 is a floor rather than the method's ceiling on this model.

### Fixed — the fp16 size column described whichever model the script was written for

`results_table.py --params` defaulted to a literal `1.8821e9`, Qwen3.5-2B's parameter
count. No caller ever passed the flag, so the Mistral-7B run printed **3.506 GiB** for
its fp16 rows — not merely wrong but *smaller* than the 4.25-bit arm beneath it, which
inverts the one comparison a size column exists to support.

The count is now inferred from any quantized row, since `quantized_gib` is
`params × average_bits / 8` and inverts exactly. That also makes the column internally
consistent — fp16 and quantized rows count the same tensors, so the ratio between them
is a real ratio — and it removes the class of bug rather than the instance: a size
column that has to be *told* which model it is describing will eventually describe the
wrong one. With no quantized arm run yet the cell prints `--` instead of a number
derived from a constant. The published Qwen table is unaffected: its own 4.249-bit /
0.931 GiB row infers 1.8821 B and reproduces 3.506.

### Changed — the experiment harness is no longer tied to one model

`experiments/qwen35_2b/` is now `experiments/four_point/`, and the model is read from
`DQ_MODEL` instead of being a constant in `common.py`. Six of the eight stage scripts
never mentioned a dataset and — after this — none of them mentions a model either, so
the alternative was copying ~700 lines per model and letting the copies drift. Drift in
the *shared* machinery is exactly what makes two runs' tables incomparable, which is the
same argument `tasks.py` already makes for keeping every task in one file.

`RUN_DIR` now defaults to `/workspace/runs/<model-slug>_<task>`. A Mistral record read
into a Qwen table is the same failure as a GSM8K record read into a CaseHOLD one, and
less obvious. The committed Qwen records predate the slug and live under
`qwen35_2b_<task>`; point `DQ_RUN_DIR` at them to re-read them.

`stage2_finetune.py` gained `--lora-rank`. It still full fine-tunes by default, which is
what the Qwen runs did, but a 7B needs ~87 GB for parameters, gradients and fp32 AdamW
moments before a single activation and does not fit an 80 GB card. Signal fidelity is
unaffected: `outer_exact` reconstructs `∇W = δxᵀ` from the layer *output* gradient, and
`y = Wx + s·BAx` means `∂L/∂y` is the same tensor whether or not an adapter is attached.
The adapter is merged before the checkpoint is written — saving adapter weights would
leave stage 3 scoring the base model under the label "fine-tuned".

`Task.stop_sequence` became a field fed from each eval module's `FEWSHOT_STOP` rather
than a method each truncation subclass overrode, which had tied *where a prompt is cut*
to *what generation stops on* — two unrelated decisions that a third task had no way to
combine.

### Added — Banking77 in the experiment task registry

Selected by re-running the headroom screen against Mistral-7B-Instruct-v0.3, since
headroom is a property of the model/dataset pair and not of the dataset. ~57 points of
it, the widest of four candidates, and a 1.3% chance floor — the most sensitive of the
three tasks to quantization damage, because there is no cushion for a small regression
to hide under. `dynquant.eval.banking77` holds the screen table.

`load_banking77` shuffles under a fixed seed. Upstream ships both splits in *label
order* — the test split is 77 contiguous blocks of 40 — so any prefix of it is a
single-intent sample and `--limit` silently stops meaning what it says. Measured, not
supposed: a 32-row smoke run reported 3.12% against a real base figure of 41%, which on
a quantized arm would have read as a destroyed model. The seed is fixed rather than
absent because the paired analysis compares per-problem hit vectors position by
position, and an order that varied between arms would pair each problem with a
different one.

### Fixed — the kernels wheel let pip pair it with a torch it cannot load

`pip install dynquant` on Linux x86_64 — the one command the meta-package exists to
make work — installed the CUDA kernels and then did not use them. Verified on Ubuntu
22.04, glibc 2.35, CPython 3.10, against the published 0.1.0:

```
dynquant-kernels 0.1.0     # built against torch 2.7.1+cu126
torch            2.13.0    # + the whole CUDA 13 runtime stack
```

Those cannot load together. The extension links libtorch's C++ ABI, so `import`
raises `undefined symbol`, `_loader.py` does its job and falls back to the torch
backend, and the user gets a correct-but-slow install with no VRAM saving — the
opposite of the reason to install kernels at all. `pip` reports success throughout.

The cause is that the variant machinery only ever *described* the build. The version
label said `+cu126torch27` and the wheel's own `_build_info.py` recorded
`TORCH_VERSION = 2.7.1+cu126`, but the dependency metadata said `torch>=2.4` with no
upper bound, so nothing in it constrained the resolver. Provenance is not a
constraint; only the specifier is.

`scripts/stamp_kernel_version.py` now stamps the runtime pin alongside the version,
derived from the same `--torch` that produces the label — `torch>=2.7,<2.8` for the
torch 2.7 cell. `--torch` became required for **every** cell including `--plain`,
which is the one that reaches PyPI and the one whose invocation was missing it.
Bounds are minor-wide because that is where libtorch's ABI actually moves; pinning
patches would reject 2.7.2, which loads fine.

Guarded by three tests rather than one, because each covers a different way to
regress it: that the pin rejects the exact 2.7-wheel/2.13-torch pairing that shipped,
that the rewrite hits the runtime dependency and leaves `[build-system] requires`
alone, and that no workflow cell invokes the stamper without `--torch`.

### Fixed — `dynquant doctor` prescribed an install that cannot work

On any platform with no published kernels wheel, the remedy for a missing CUDA
backend was "pip install dynquant (the meta-package pulls in the kernels wheel for
your platform)" — printed, on Windows, to a user who had just run exactly that. The
meta-package's marker is Linux-x86_64-only *on purpose*, so there was nothing to pull
and the fallback advice pointed at a 20-minute CUDA source build. The doctor now
tests the platform, names it, and says the torch backend is the expected path there,
keeping the source build as an explicit opt-in. `_prebuilt_wheel_exists()` and the
marker are pinned to each other by a test that evaluates the real marker against
four synthetic environments.

## [0.1.0] — 2026-07-28

First release. The phases below are the build order from the project plan, and the
status table in [README.md](README.md) is authoritative for what actually works
today — this is P0 through P5 plus the P6 decode kernels, not a finished package.

### Known issues in 0.1.0

**Pin torch when installing this version.** On Linux x86_64:

```bash
pip install dynquant 'torch==2.7.*'
```

Without the pin the kernels install but never load: the wheel is built against torch
2.7.1 and declares only `torch>=2.4`, so pip resolves torch 2.13 beside it and the
extension fails its import. DynQuant falls back to the torch backend — correct
results, no VRAM saving, no speedup — and `pip` reports success. `dynquant doctor`
is what tells you which backend you actually got; run it. Fixed at source in
[0.1.1] above, where every wheel carries the bound of the minor it was built
against — so the simplest response to this entry is to install 0.1.1.

Two more things to know before installing:

- **PyPI carries one binary variant; the GitHub Release carries the rest.** On Linux
  x86_64 with CPython 3.10–3.13, `pip install dynquant` gets a real wheel from PyPI —
  the cu126 / torch 2.7 build — and compiles nothing.

  Every *other* combination is published with a PEP 440 local version
  (`0.1.0+cu126torch26`, `0.1.0+cu128torch28`) naming the CUDA and torch build it
  matches, and PyPI rejects local versions outright, so those twelve wheels are
  attached to the [GitHub Release][0.1.0] instead. It serves as the `--find-links`
  variant index. On a different torch, either point pip at it or build the sdist,
  which needs a CUDA toolkit. Without one, `dynquant-core` still installs and runs on
  the torch backend.
- The CUDA wheels are `manylinux_2_34_x86_64`, so they need **glibc 2.34 or newer** —
  Ubuntu 22.04, Debian 12, RHEL 9 and up. Older hosts fall back to the sdist, which
  needs a CUDA toolkit. This is forced rather than chosen: the repair step excludes
  `libcudart.so.*` so the wheel uses torch's own runtime, which means the build
  toolkit's CUDA version has to match torch's exactly, and of the published CUDA
  manylinux images only the `2_34` family has versions torch actually ships against.
  It is also the floor the one hand-verified wheel already had.
- `KERNEL_ABI_VERSION` is 2. A kernels wheel refuses to load against a core that
  expects a different number, with the remedy in the error rather than an import
  traceback.

### Added — P0, foundations

- Monorepo with three independently releasable distributions: `dynquant-core`
  (pure Python), `dynquant-kernels` (compiled CUDA), and `dynquant` (the
  meta-distribution users install). `dynquant_kernels` is a top-level import name,
  deliberately not `dynquant.kernels`, so the two wheels never share an import
  namespace.
- CMake + `scikit-build-core` build for the extension, with a CPU-only path that
  compiles and registers every operator without a CUDA toolkit. That path is what
  lets a free CI runner catch packaging, schema and registration mistakes in
  minutes rather than at the end of a GPU matrix.
- Probe kernels (`probe_axpy`, `probe_reduce`, `probe_gemm`) covering the three
  link-time dependencies — a raw launch, Thrust, and cuBLASLt — so a broken build
  names which one failed instead of reporting "kernels don't work".
- Operators registered through `TORCH_LIBRARY` rather than pybind, so they carry a
  schema `torch.compile` can trace instead of forcing a graph break at every
  quantized `Linear`.
- ABI handshake between core, the Python shell and the compiled binary, with a
  source-text lint (`tests/test_abi.py`) that runs on a machine where no extension
  is installed.
- `dynquant doctor`: environment report, backend selection with a reason for every
  rejected backend, and a numerical self-check. It verifies pack/unpack bijection
  at every bit width and checks measured quantization error against theory,
  because a wrong-ABI kernel returns tensors of the right shape and dtype full of
  wrong numbers — indistinguishable from quantization simply being lossy, and so
  never reported as a bug.
- `dynquant version`, exit-code discipline in the CLI (`0` ok, `1` diagnosed
  failure, `2` usage, `130` interrupt).
- CI: ruff, strict mypy, a test matrix across Linux/Windows/macOS and CPython
  3.10–3.13, a CPU-only extension build, distribution-content checks, and a
  dormant GPU parity job. Release workflow for the CUDA wheel matrix, with the
  binary variant recorded in the PEP 440 local version segment
  (`0.1.0+cu126torch27`), the same scheme torch uses.
- Repository hygiene: Apache-2.0, pre-commit, and a `check_no_confidential.py`
  guard that runs *before* a commit exists, because a commit cannot be un-made.
- [`docs/images/cuda-kernel-architecture.png`](docs/images/cuda-kernel-architecture.png):
  the binary pipeline in one page — the five files under `csrc/`, the four-way CUDA
  fork at configure time, the wheel matrix, the seven load gates and their
  remedies, and backend dispatch. Generated by
  [`docs/diagrams/kernel_architecture.py`](docs/diagrams/kernel_architecture.py)
  rather than drawn, so it is regenerated from a readable diff when the build
  changes. Nothing in it depicts a kernel that does not exist: P5–P8 appear only as
  what each probe de-risks, and the page says on its face that the P0 gate — `pip
  install dynquant` on a clean Linux GPU box — has not been run.

### Added — P0, the four commands

The library was complete and tested before any of this existed, and `README.md`
documented a CLI that did not. These four commands are thin shells over the same
functions the tests already drive — no new quantization, allocation or evaluation
logic landed with them — so what is new is the seam between them, and the seam is
where the mistakes live.

- `dynquant inspect` — role, parameters, sensitivity and assigned width per module,
  plus the three things a width histogram cannot show: **within-role concordance**
  of width with score, every **floor the budget forced it to breach**, and every
  module **the signal never saw**. The first of those is the direct answer to the
  research allocator's failure mode: it produced a complete, plausible bit map at
  its own headline 3-bit target while never reading the importance scores, and
  inverting every score changed 0 of Qwen3-14B's 282 assigned widths with nothing
  in the output saying so. Concordance near 0.5 says exactly that, in one number.
  `--uniform 3 4` puts the control arms in the same table, priced by the same
  accounting, because two budgets computed two ways are not a comparison. Defaults
  to CPU: only names, shapes and the module tree are read, and a second copy of the
  weights on the GPU is how an analysis becomes an OOM in the run that mattered.
- `dynquant quantize` — allocate (or read a map) and encode. It prints the packed
  size beside the directory size on **every** run, because what it writes today is
  quantized values in the compute dtype (P9 is the packed writer) and the research
  supplement reported storage savings from exactly this path while the model it had
  loaded was fp16 all along.
- `dynquant eval` — one task, one prompt format, greedy decode, a fixed few-shot
  prefix drawn by seed from a split the run does not score. `--out` writes the
  **per-problem correctness vector**, not just the count, because the comparison
  that matters is paired and the first version of this harness made recovering the
  vector cost a re-run of every arm on the GPU. `--compare` runs McNemar and
  refuses across a differing task, split, shot count, seed or limit — a harness
  difference reported as a quantization effect is the failure it exists to prevent.
  `--map` swaps onto the packed runtime first, which is the only configuration in
  this repository from which a memory figure means anything.
- `dynquant bench` — the packed GEMV as a fraction of the card's **measured**
  achievable bandwidth, not its datasheet peak (an A100 80GB PCIe is quoted at 1935
  GB/s and delivers about 1630). Timing is `self_device_time_total` over
  kernel-level profiler events, because an 8 us kernel behind a ~10 us Python
  dispatch reads as 13 us on the wall clock and every optimization looks worthless.
  Rows where the weight fits in L2 and the bf16 baseline therefore posts over 100%
  are flagged rather than left to be noticed.
- `ALLOCATION_FILENAME` / `ALLOCATION_SCHEMA` in `constants.py`, and
  `commands/_shared.py` as the one place a bit map is written and read. `inspect
  --save-map` then `quantize --map` is one seam with one format, so the reviewed map
  is the map that gets applied — no second allocation in between that could differ
  from the one that was looked at. Reading is deliberately permissive about shape
  (three accepted layouts, so the `stage*.py` outputs that predate the format still
  load) and deliberately strict about content: an unsupported width, a
  non-integer, an empty file or several maps with no `--map-key` are each a
  diagnosed error rather than a guess, and a map whose names do not resolve against
  the model fails before a weight is touched — the damage worth catching early is
  that the names which *do* resolve get quantized at widths chosen for a different
  checkpoint, and the result runs.
- `tests/test_cli.py` (72 tests, CPU, no checkpoint and no download — the synthetic
  Qwen3.5-shaped fixture drives the parts that matter). Two invariants are pinned
  outright because both are one innocuous import away from breaking: **parser
  construction imports nothing heavy**, checked in a subprocess since this process
  has already imported torch, so `dynquant doctor` still runs on an install where
  torch is the broken thing; and **`inspect` and `quantize` allocate through the
  same code**.

### Fixed — P0, the concordance check measured the wrong signal

- **`dynquant inspect` reported the concordance of a signal the allocator had not
  used.** Concordance exists to answer one question — did the ordering reach the
  allocator, or is size alone choosing the widths — and it was computed against the
  rank score unconditionally. With `--moments` the widths come from measured `ΔL`
  instead, and the two orderings disagree by design: that is the whole finding
  behind shipping sensitivity as the default, ρ +0.521 against +0.231 within role.
  So the number under the heading "concordance of width with score" answered a
  question nobody asked, and it could fail in both directions — a working
  sensitivity allocator reading as "the score is not reaching the allocator", or a
  broken one flattered by a rank score that happened to agree.

  Found by running the command against a real fine-tuned Qwen3.5-2B rather than a
  fixture: it printed 699/707 = 0.989 for an allocation that scores 706/707 = 0.999
  against what actually ordered it. Near-miss, but the *reason* the numbers were
  close was a property of that model — the shipped `combine="plasticity"` score
  correlates strongly with `ΔL` — not of the code.

  Fixed by measuring against the driving quantity: `(ΔL(2b) − ΔL(8b)) / num_params`
  when sensitivity is present, the score when it is not. Not a new metric — in the
  proxy path that expression *is* `score × (err(2) − err(8))`, a positive constant
  times the score, and concordance is scale-invariant, so the two definitions
  coincide exactly wherever sensitivity is absent and the rank-product path did not
  change meaning. Dividing by `num_params` is what keeps the diagnostic pointed at
  its own failure mode, since an undivided `ΔL` grows with the tensor and would read
  as healthy in precisely the case where size was choosing the widths. Every line
  that prints the quantity now names it, because a rank score and a loss delta per
  parameter differ by fourteen orders of magnitude and by meaning. A module with no
  measured value is left out of the ratio and listed, rather than entered as zero,
  which is an ordering claim about something nothing was measured for.

- The parenthetical read `(0.5 means ordered)` on a healthy allocation, which is the
  opposite of what 0.5 means. The verdict and the reference are now separate:
  `(ordered; 0.5 is what no ordering at all looks like)`.

### Fixed — P0, `quantize --map` withheld the one number it exists to print

- **On the `--map` path, `dynquant quantize` printed no packed size** — the run ended
  `its size is not the quantized size.` and stopped there, while an allocating run
  continued `— that is 1.207 GiB, and \`dynquant eval --map\` measures it in VRAM.`
  So the path the README recommends, and the only one where the map that was
  reviewed is the map that gets applied, was the path that left the reader with a
  compute-dtype directory size and no figure to compare it against. Both the README
  ("prints the packed size beside it on every run") and the command's own docstring
  ("printed next to the directory size") claimed otherwise.

  The cause was a shortcut with a defensible half: `_resolve_widths` returns no
  `BitMap` for a file-sourced map, because recomputing that accounting here would
  need the graph its writer used, and quietly re-deriving it is how two numbers that
  should agree start to drift. That part stays. What was wrong was concluding that
  an unrecomputable figure is an unavailable one — `--save-map` records `nbytes` and
  `average_bits` in the file, and `read_bit_map` was already returning both in
  `metadata` and dropping them on the floor.

  Now a `_Stored` carries the figure and *where it came from*, printed either way:
  `(this allocation)` or `(maps/dynquant_allocation.json [3.25], as recorded)`. A
  bare hand-written `{name: bits}` map genuinely has no accounting, and that case
  says so and names the command that would produce one, rather than letting the
  directory size stand in — which is the single number the research supplement got
  wrong, and not a mistake worth reproducing in the tool built to correct it. Floor
  violations recorded in a map file are now printed on application too, in the same
  shape `BitMap.summary()` uses on the allocating path: a breached floor is the one
  thing in a bit map that is a risk to the model rather than an accounting detail,
  and the run that allocated it may be one nobody still has the output of.

### Fixed — P0, `bench` ran its clip search on the CPU

- **`dynquant bench --model` looked hung.** It builds a random weight per distinct
  shape and quantizes it, and the weight was allocated on the CPU — so the MSE clip
  search, eight candidate reconstructions over the whole tensor, ran there too. The
  shapes are measured largest first, which on Qwen3.5-2B means the 248320×2048
  embedding is the first row: ten minutes in, the GPU was at 0% utilization, 552 MiB
  resident, and the only output was `model.embed_tokens 248320x2048 ...`. It was not
  hung, it was single-threaded numerics on 508M parameters, four times over.

  The weight is now allocated on the device it is benchmarked on. The whole table
  takes about two minutes, and the search runs on the same code path a real
  quantization does rather than a slower one that exists only in this command.

### Changed — P0, the kernels wheel now carries a publishable platform tag

- **`auditwheel repair` to `manylinux_2_34_x86_64`.** A wheel built in place is tagged
  `linux_x86_64`, and PyPI rejects that tag outright — it says nothing about which
  glibc the extension needs, so the index cannot know whether it will run. Repairing
  it is now part of the release path. `auditwheel show` puts the floor at
  `manylinux_2_34` (the symbol that binds it is `GLIBC_2.34`), which is Ubuntu 22.04
  and newer, RHEL 9, Debian 12.

  The repair excludes the nine torch and CUDA libraries — `libtorch`, `libtorch_cpu`,
  `libtorch_cuda`, `libtorch_python`, `libc10`, `libc10_cuda`, `libcudart.so.13`,
  `libcublas.so.13`, `libcublasLt.so.13`. Vendoring any of them would put a second
  copy of libtorch in the process next to the one torch already loaded, which is not
  a size problem but a correctness one: two libtorch images mean two copies of the
  dispatcher's global registries. The extension has no `RPATH` before or after the
  repair and resolves those symbols from the already-loaded process image, which is
  what `dynquant_kernels/_loader.py` imports torch first to guarantee.

  So the repair is a **pure retag**: the output wheel is 443 bytes larger than the
  input, entirely metadata, and `_C.cpython-312-x86_64-linux-gnu.so` is byte-identical
  (same sha256). Verified by installing the repaired wheel into a clean venv on the
  A100, with torch supplied on `PYTHONPATH` from a symlink farm that excludes every
  `dynquant*` entry — otherwise the editable install in the build venv shadows the
  wheel under test and the gate proves nothing. There, `dynquant doctor` passes all
  five checks including `cubin-coverage` for `sm_80`, and `quantized_matmul` agrees
  with the dequantized oracle at every width on both dispatch paths (worst relative
  error 7.0e-03 at M ≤ 8, which is bf16 output rounding; exactly 0.0 above
  `gemv_max_rows()`, where `torch.ops.dynquant.dequant` proves bit-identical to
  `QuantTensor.dequantize`).

- **The release workflow's CUDA excludes are wildcards now.** They named
  `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12` explicitly, which is
  correct for every cell of the current matrix and fails *open* the moment a CUDA 13
  cell is added: a soname matching no `--exclude` is grafted rather than reported, so
  the failure mode is a silently 400MB wheel with a duplicate cuBLAS, not a red build.
  `'libcudart.so.*'` and friends were checked against the real cu130 wheel — auditwheel
  logs `Excluding libcudart.so.13`, `Excluding libcublasLt.so.13`, and the output is
  identical in size and file list to the run with pinned sonames.

  Also worth recording: the manual repair reached `manylinux_2_34` while the workflow
  asked for `manylinux_2_28`. Those did not disagree — the floor comes from the build
  host's glibc, and the repair ran on the box's native Ubuntu rather than in a
  container. The workflow now asks for `manylinux_2_34` as well, for the reason in the
  release-blocker note below, so CI and the hand-verified wheel land on the same floor.
  If a wheel ever needs a higher glibc than the requested `--plat`, auditwheel refuses
  rather than mislabelling it.

  **Still not publishable, for a different reason.** The wheel's version is
  `0.1.0+cu130torch213`, and PEP 440 local versions cannot be uploaded to PyPI at
  all — not under a different tag, not with a flag. Distribution therefore needs the
  torch-style variant index that the plan calls for: one wheel per CUDA × torch ×
  Python combination behind a `find-links` URL, with PyPI holding only the pure-Python
  core plus whichever single combination is chosen as the default.

### Fixed — P0, release blocker: the CUDA wheel matrix had never been run

`wheels.yml` is triggered by `push: tags: ["v*"]` and nothing else, so from the day it
was written until the `v0.1.0` tag it had executed exactly zero times. Tagging ran it
for the first time and it failed on all three CUDA arms — then failed three times more,
on four further causes, each one hidden behind the one before it. All five are the same
mistake wearing different faces: **assuming what a container we do not build contains,
or which containers would be used at all.**

1. **The images did not exist.** `pull access denied for
   sameli/manylinux_2_28_x86_64_cuda_12.4, repository does not exist`, before compiling
   anything. That namespace publishes exactly one `manylinux_2_28` CUDA image, 12.3, and
   no torch wheel is built against CUDA 12.3 — so the `2_28` family could never have
   been paired at all and the matrix was unbuildable as written.

   The pairing has to be *exact*, not same-major: the repair step excludes
   `libcudart.so.*` so the wheel uses torch's bundled runtime, and a `_C.so` built by
   nvcc 12.8 against a torch carrying cudart 12.6 can want a symbol that runtime does
   not have — a failure that lands on the user at import, not on us at build. That
   requirement is what selects the images, and only the `manylinux_2_34` family has CUDA
   versions torch actually ships (12.6, 12.8). Hence the glibc 2.34 floor recorded
   above: forced, not chosen. All twelve (torch, CUDA, CPython) cells were checked
   against `download.pytorch.org` before pushing this time rather than after.

   Images are now pinned **by digest**. These repositories publish only a mutable
   `latest`, which means a tag reference lets whoever controls it choose the compiler
   that produces the binaries our users run.

2. **cibuildwheel probed for a CPython the image had dropped.** `Command
   ['/opt/python/cp38-cp38/bin/python', ...] failed with code 127` on the CUDA 12.8 arm.
   2.21.3 hardcodes cp38 as the interpreter it uses to read the container environment,
   and the newer image no longer ships one; `CIBW_BUILD` never asked for 3.8, so the
   restriction did not help. 2.22.0 moved that probe to cp39, and the pin is now the
   last 2.x (2.23.4) — a version bump rather than a 3.x config migration.

3. **nvcc rejected the image's default host compiler.** `error: #error -- unsupported
   GNU version! gcc versions later than 13 are not supported!` on both CUDA 12.6 arms.
   nvcc compiles the host half of every `.cu` with a host C++ compiler and refuses any
   version newer than it knows; the `manylinux_2_34` base ships GCC 14.2.1. It surfaced
   inside `enable_language(CUDA)` in torch's own `cuda.cmake`, reached through our
   `find_package(Torch)`.

   `-allow-unsupported-compiler` was available and was not used: nvcc's own message says
   it "may cause compilation failure or incorrect run time execution," which is not a
   bet worth taking on numerics kernels — a miscompiled dequant is a wrong answer, not a
   crash. Instead `CIBW_BEFORE_ALL` installs gcc-toolset-13 and `CUDAHOSTCXX` pins it
   (GCC 13 satisfies both 12.6 and 12.8, so one pin covers the matrix). That variable
   has to arrive as an environment variable and not in `CMAKE_ARGS`, because it is
   consumed at `enable_language(CUDA)` — reached while `find_package(Torch)` is still
   running, so a `-D` on our own command line lands too late to be read.

   Worth having regardless of the failure: the host compiler was previously whatever the
   image defaulted to, so an image rebuild could change code generation under every
   wheel we ship without a line of our own changing.

4. **It was also building musllinux wheels, which cannot exist.** With the compiler
   fixed, all four manylinux wheels built and repaired — and the job then failed on
   `exit 127` anyway, 26 minutes in, because "Linux" means manylinux *and* musllinux to
   cibuildwheel and `cp310-*` matches `cp310-musllinux_x86_64` too. The `dnf` line added
   for cause 3 had met an Alpine image, whose package manager is `apk`.

   Skipping musl is correct rather than expedient: torch publishes `manylinux_2_28` and
   `win_amd64` wheels and nothing else, so an Alpine build has no torch to compile
   against or link to, and the musllinux image has no CUDA toolkit either. `CIBW_SKIP:
   "*musllinux*"` now says so. Note the shape of this one — the leg that cannot work
   ran *last*, so it discarded a pile of successful work rather than failing fast.

5. **The image also chose our auditwheel, and one chose a version too old.** With musl
   skipped, the CUDA 12.8 arm went green end to end and both CUDA 12.6 arms failed in
   the verify step:

   ```
   vendored libraries that must come from the user's torch:
     libcublasLt-…so.12.6.4.1, libcudart-…so.12.6.77, libcublas-…so.12.6.4.1
   ```

   auditwheel comes from the manylinux image, and the two images disagree: the CUDA 12.6
   image (built 2024-12) carries **6.1.0**, the CUDA 12.8 one (2026-05) carries
   **6.6.0**. Wildcard `--exclude` arrived in 6.2.0 ([pypa/auditwheel#508]), so on 12.6
   the `libcudart.so.*` patterns matched nothing at all and auditwheel grafted the CUDA
   runtime into the wheel. Its log tells the story plainly — 12.8 prints `Excluding
   libcudart.so.12`, 12.6 prints no such line and never mentions them again.

   This is precisely the "fails open" outcome those wildcards were introduced to
   prevent, arriving by a route the note above did not anticipate: not a CUDA 13 soname
   slipping past a pinned pattern, but a *correct* pattern silently inert because the
   tool reading it was one release too old. It also corrects a claim made when the
   wildcards landed — they need auditwheel ≥ 6.2.0, not ≥ 5.4.

   `CIBW_BEFORE_ALL` now pins `auditwheel==6.6.0` into the image's pipx venv, so every
   arm repairs wheels identically regardless of how old its image is. Pinned exactly
   rather than floored: this is a release pipeline, and 6.6.0 is the version that
   produced the arm which passed.

   Worth stating what actually held here. The wildcard patterns failed, and the wheel was
   wrong, and *nothing shipped* — because the verify step checks the wheel it is about to
   publish rather than trusting the tool that produced it. A 400MB wheel carrying a
   duplicate cuBLAS would have been a bad day for whoever installed it, and it never got
   past the build.

[pypa/auditwheel#508]: https://github.com/pypa/auditwheel/pull/508

Two process changes came out of this, both cheap and both aimed at the *class* of bug
rather than the five instances:

- **The build now records its own toolchain.** `CIBW_BEFORE_ALL` prints the host GCC,
  nvcc, `auditwheel` and `/opt/python` inventory. Along the way this corrected a claim
  in the workflow's own comments: the wildcard excludes need auditwheel ≥ 5.4, and
  auditwheel comes from the *manylinux image*, not from cibuildwheel as the comment
  said — a third version the image picks for us, so it is now printed too.
- **Validate with `workflow_dispatch` on `main` before moving a tag.** Every publish and
  release job in this workflow is gated on `startsWith(github.ref, 'refs/tags/v')` and
  on `needs: [sdist, linux-wheels]`, so a dispatch run builds all twelve wheels while
  publishing nothing. Doing that found causes 2 and 3 at zero cost. A tag-triggered
  workflow that has never run is not a tested workflow.

Nothing was published during any of this — `release` needs both build jobs, so the failed
runs produced no GitHub Release and all three publish jobs skipped. The only artifact was
the tag itself, so `0.1.0` stayed unclaimed and reusable.

### Fixed — P0, the CPU-only kernels build never worked

- `CMakeLists.txt` read the torch probe's third line to get the CUDA version, but
  on a CPU-only torch `torch.version.cuda` is `None`, so that line was empty and
  `OUTPUT_STRIP_TRAILING_WHITESPACE` removed it. The list had two elements and
  configuration died with `list index: 2 out of range`. The field now carries a
  `cuda=` prefix so the line is never empty, the line count is asserted before
  indexing, and `\r` is stripped so Windows cannot produce a variant of the same
  failure.

  This broke exactly the configuration the cheap CI runner exists to exercise, and
  it went unnoticed because every manual build so far ran on a CUDA box, where the
  third line is non-empty. The CPU-only path is not a lesser build — it is the one
  that catches CMake and operator-registration mistakes before a GPU runner is
  needed — so it had been silently absent from the gate it was written for.

- With the build fixed, the same job failed one step later on
  `'_OpNamespace' 'dynquant' object has no attribute 'probe_axpy'`. The step
  imported `dynquant_kernels` under a comment claiming the import registers the
  ops. It does not: loading is lazy by design, and `load_result()` — not the
  import — is what imports `_C` and runs the static initializers that declare the
  schemas. The caller was fixed rather than the laziness, since the laziness is
  the documented contract (importing `dynquant` must not pay for a large shared
  object, and a machine with a broken driver still has to run the CPU-side
  quantizer). Every other caller in the tree already went through
  `is_available()`, which loads; this step was the only one that did not.

- The `types` job could not reproduce a local mypy run. `strict = true` enables
  `warn_unused_ignores`, and the job installed neither `transformers` (an optional
  extra) nor a pinned `safetensors`. Absent transformers, `ignore_missing_imports`
  made every transformers symbol `Any` — so the callback and eval harness were not
  type-checked at all, and the ignores they need reported as unused; meanwhile
  safetensors 0.8 types `safe_open`, which retired an ignore calibrated against
  0.5.3. Both are now pinned in `env` as the reference surfaces the tree's
  `# type: ignore` comments are written against, and transformers is installed so
  that code is actually checked. Runtime floors stay `>=` in package metadata.

### Changed — P0, PyPI publication is one job per distribution

- `publish-pypi` is replaced by three jobs — `publish-core`, `publish-kernels`,
  `publish-meta` — each running in its own GitHub environment (`pypi-core`,
  `pypi-kernels`, `pypi`).

  This is forced by PyPI, not a preference. A *pending* trusted publisher is keyed
  on `(owner, repository, workflow, environment)` and deliberately **not** on the
  project name, so a single environment can hold exactly one pending publisher.
  Registering the second of three names against one environment fails with *"a
  pending trusted publisher matching this configuration has already been registered
  for a different project name"*. Confirmed against the constraint in
  `warehouse/accounts/views.py`, which says so in a comment; it is not documented on
  docs.pypi.org, which is why the first attempt at this configuration was wrong.

- The three jobs are chained with `needs:` rather than run in parallel, so a
  dependency always reaches PyPI before the distribution that pins it. `dynquant`
  requires `dynquant-core==<the same version>` exactly, and with required reviewers
  on each environment the approvals can be minutes or hours apart — publishing the
  meta package first would leave `pip install dynquant` unresolvable for that whole
  window.

- Each job selects only its own artifacts by filename prefix and fails if the
  selection is empty. `dynquant-` cannot collide with `dynquant_core-` or
  `dynquant_kernels-` because distribution filenames normalise the project name with
  underscores, so the hyphen is only ever the name/version separator. An empty
  upload directory is a no-op that reports success, which is a worse outcome than a
  red build.

- The artifact-selection step is inlined rather than factored into a checked-in
  script, because that would require `actions/checkout` inside a job holding
  `id-token: write`. Executing a script from the same tag being published means
  anyone who can push a tag can rewrite what gets uploaded. These jobs now touch
  build artifacts only, never repository source.

### Added — P1, formats and data model

- `constants.py` as the single owner of every filename in the format.
- Group-aligned uint32 packing for 2/3/4/8-bit. The invariant
  `group_size % 32 == 0` makes every group start on a word boundary at every bit
  width, which is the precondition for a coalesced load with compile-time shifts —
  and the reason the research layout admitted no efficient kernel at all.
- `QuantTensor` with all dequantization metadata explicit: bits, group size,
  symmetry, layout, row offset, logical shape. Nothing is inferred from tensor
  sizes at load time.
- Affine convention `w ≈ q·scale + offset` with an unconstrained float offset and
  **no integer zero-point**. GPTQ and AWQ widen every group's range to include
  zero; for a group spanning `[0.50, 0.52]` that inflates the range 26× and throws
  away more than four bits of resolution. The cost is that exact zero is not a
  guaranteed grid point, so all-zero groups fold through an explicit constant path
  instead.
- `StatsFile` v2 schema with lossless v1 migration, including recovery of Welford
  state: storing the sample count makes `m2 = var·(count−1)` recoverable, so
  Chan's parallel merge stays exact.
- `RowGeometry` / `row_geometry()` as the single resolver for every size derived
  from `(bits, group_size, in_features)`. Both the encoder and the validator call
  it, so a disagreement between them is not expressible.
- [`docs/format-spec.md`](docs/format-spec.md): the versioned on-disk contract.
  Every number in it — word counts, the 3-bit worked example, effective bits per
  weight, the zero-point widening figure — is generated from the implementation
  rather than written by hand. This is what P6–P8's kernels are written against.
- [`docs/legacy-audit.md`](docs/legacy-audit.md): every confirmed defect in the
  research code, with evidence and the v2 replacement.
- `dynquant/_legacy/`: the paper's `scorer.py`, `allocator.py` and `quantizer.py`,
  vendored verbatim as the reference oracle for `--preset paper-3.15` and the
  golden tests. All three import only the standard library and torch, which is
  what makes shipping them safe. Excluded from ruff, mypy and coverage on purpose:
  satisfying a linter means editing the file, and editing the file destroys the
  only property that makes it useful.
- `tests/test_legacy_allocator.py`: an executable record, not a regression test.
  It runs the supplement's own allocator against its own shipped stats and pins
  audit item 4 — that at the paper's headline 3-bit target the stability floors
  already exceed the budget, so the greedy loop early-returns and the importance
  score is never read. Inverting every score changes 0 of 282 modules at targets
  3.0, 3.15 and 3.5. Also gives `--preset paper-3.15` an executable definition of
  the behaviour it must reproduce.
- `tests/test_legacy_provenance.py`: pins each vendored file by SHA-256 over
  LF-normalized bytes, re-checks them against the originals wherever the
  supplement is present, asserts nothing else is ever copied into that directory,
  and runs a subprocess to confirm importing `dynquant` does not pull `_legacy` in.
- `.gitattributes` normalizing the tree to LF. Load-bearing rather than cosmetic:
  a Windows checkout with `core.autocrlf=true` would rewrite the vendored files on
  disk.
- The theoretical error oracle (`step/√12` residual, efficiency 0.90, ±25% band),
  which would reject an encoder that widens group ranges.

### Fixed — P1

- **Per-row `group_size` sentinel was resolved away at construction, making the
  format's one alignment exemption unenforceable.** `checked_group_size` exempted
  `group_size = -1` from the `% 32` rule and then returned `in_features`, so the
  resolved width was what got stored. Every later check re-ran the same function on
  the resolved value, took the alignment branch, and rejected a tensor the encoder
  had just produced. Consequences, both reproduced: `validate()` raised
  `group_size=100 must be a multiple of 32` on a freshly built tensor, and
  `words_per_group` raised at 2-, 3- and 4-bit for the `[channels, 1, 4]` Mamba
  `conv1d` shape that `checked_group_size`'s own docstring names as the reason
  per-row exists. No per-row tensor whose `in_features` was not a multiple of 32
  could be loaded back. Nothing in the 409-test suite caught it because no test
  called `validate()` on one.

  The sentinel is now stored verbatim and per-row rows round up to whole *words*
  rather than padding values — at most 31 unused high bits in the final word, and
  no accuracy cost. Guarded by five tests, including a round-trip through
  `state_dict` at every bit width for `in_features ∈ {4, 31, 100, 1024}`.

### Security — P1

- **`scripts/check_no_confidential.py` passed both files it was written to
  block.** The token pattern was `hf_[A-Za-z0-9]{8,}`, which cannot match
  `hf_token_here` — the underscore ends the character class after five characters —
  and there was no absolute-path pattern at all. A secret scanner that never fires
  is worse than no scanner, because the green tick is taken as evidence. Fixed to
  `hf_[A-Za-z0-9_-]{4,}` (the *placeholder* is the thing to catch: whoever fills it
  in ships a live credential) and given POSIX/Windows home-path patterns applied to
  code but not to prose. It now names all five real occurrences.
  `tests/test_confidential_guard.py` uses the actual research files as fixtures,
  not synthetic strings — a made-up token of the well-formed shape would have
  passed the broken pattern too, and the bug would have survived its own test.
  (Writing that sentence with a literal example in it tripped the fixed guard on
  this very file, which is the shortest available demonstration that it works.)
- **The research tree is no longer committed.** `dynquant_paper/`, `inference/`,
  `fine-tuning_and_stats_hook/` and `pipeline/` are gitignored. They are part of
  the same confidential submission as the PDF, they carry the credential
  placeholder and the author's VM paths, and nothing in the wheel needs them —
  `dynquant/_legacy/` holds the three modules that are actually depended on.
  `stats/` stays committed: it holds measurements, and the golden tests read it.
- `.claude/settings.local.json` gitignored. It records absolute paths from the
  machine it was written on, which is the same defect audited as item 10; the
  fixed guard caught it on its first full-repository run.

### Added — P2, the training-time signal hook

- `DynQuantCallback` for `Trainer` and TRL's `SFTTrainer`, plus
  `track_signals(model, out=...)` as a context manager for hand-written loops. Three
  lines into an existing fine-tune.
- Device-resident accumulators indexed by module, with one host sync every
  `log_every` steps instead of one per module per step. The research tracker called
  `float(...cpu().item())` inside both hooks — roughly 850 stalls per step on a 14B
  dense model and about 18k on a 128-expert MoE, which is what the "zero additional
  cost" claim was measured against.
- Welford updates moved to `on_pre_optimizer_step`, so gradient-norm variance is
  taken over optimizer steps as the paper's Appendix H specifies rather than over
  micro-batches. Under gradient accumulation those differ by the accumulation
  factor, and the variance is the signal.
- `GradNormEstimator` in three modes. `outer_exact` (default) reconstructs the
  *base-weight* gradient norm from `δ = ∇_Y L` and the stashed activation via the
  paper's own `∇W = δxᵀ`, on a bounded token subsample. `lowrank` composes it from
  the LoRA factors; `param` is the legacy path, kept for `paper-3.15`. The research
  code hooked `lora_A`/`lora_B` under the base module's name, so it measured the
  adapter rather than the tensor being quantized, and its coherence term alternated
  between vectors of length `r` and length `d_out` — `torch.dot` raised on the
  mismatch every time and a bare `except` swallowed it.
- `forward_calls` per module and a `CoverageReport` naming every module the training
  run never exercised. On a sparse MoE this is the difference between "this expert is
  unimportant" and "this expert never ran", which the research scorer could not
  express: both arrived as zero.
- Names canonicalized at write time through `canonical_name()`
  (`base_model.model.model.…base_layer` → `model.…`), so a stats file is keyed the
  way the graph is keyed and the reader does not have to guess.
- DDP / FSDP / DeepSpeed reduction before the file is written. Welford state combines
  through Chan's parallel formula, which is algebraically equal to a single-stream pass
  over the union; the EMAs cannot combine exactly and use an observation-count-weighted
  mean, documented in the schema rather than papered over. `all_gather_object` rather
  than a tensor all-reduce, because ranks can legitimately track different module sets —
  an expert that received no tokens on one rank — and a positional collective
  mismatches silently when they do. The research code had no reduction step at all, so
  it wrote rank 0's evidence and discarded `world_size - 1` of it invisibly.
- `tests/test_signals_reduce.py`, a real two-process gloo run rather than a direct call
  to the merge function: it asserts that two ranks over disjoint, deliberately
  unequal-sized observation sets reproduce a single-stream pass to floating-point
  equality, that every rank returns the same reduction, that a module only one rank saw
  survives, and that `provenance.world_size` records the reduction happened. This
  closes P2's DDP-parity gate item. `reduce_stats` previously had no test coverage.

### Added — P3, architecture-generic role classification

- Resolution order with first match winning: user override → architecture plugin for
  `config.model_type` → generic structural inference → name substrings as a last
  resort. The research code had only the last of those.
- Structural inference reads the module tree rather than the name: ancestor class,
  membership in an `experts` `ModuleList`, and a router test on
  `out_features == config.num_experts` with an `experts` sibling. That covers MoE
  families not yet released, where a substring list cannot.
- `RowPartition` for fused projections, so `qkv_proj` and `gate_up_proj` carry
  different widths per row block instead of being rounded to the most conservative
  one. Needs no new encoder math — scales already live per row.
- `OTHER` maps to a conservative 4-bit default and `dynquant inspect` lists every
  module that reached it. Nothing silently falls to the 2-bit floor, which is what
  happened to MoE routers (`mlp.gate` matched neither `gate_proj` nor the attention
  list), MLA `kv_a_proj_with_mqa`, and Mamba's `in_proj`/`x_proj`/`dt_proj`.
- Verified against the real `Qwen3_5ForCausalLM`: 187 quantizable modules, 1.8817B
  unique parameters, none unclassified. The tied embedding/LM-head pair is listed
  once — counting it twice would inflate the parameter total by 508.6M, 27% of the
  model, and every budget computed from it.

### Added — P4, scoring and allocation

- `score/importance.py`: percentile-rank each signal, multiply. Ranking within role
  by default, because activation RMS differs systematically between roles and a
  global ranking sorts mostly by role, squeezing the within-role question — the one
  the allocator actually asks — into the noise.
- Soft quality floors. When floors exceed the budget the allocator downgrades by
  lowest ROI and reports every breach, instead of returning the floor map untouched.
  `tests/test_legacy_allocator.py` pins what the old behaviour cost: at the paper's
  headline 3-bit target, inverting every importance score changed 0 of 282 modules.
- Structural floors kept hard and separated from quality floors in
  `allocate/policy.py`. A 4-bit MLP up-projection is a worse model; a 4-bit router
  is a different one. `LM_HEAD` is deliberately a *quality* floor: on a tied model it
  is 27% of the parameters, and pinning it at 8 bits makes every target below about
  3.9 bits arithmetically unreachable.
- Budget accounting in *stored* bits — scales and offsets included — so the number
  the CLI prints is the number the filesystem will report. At group 128 asymmetric
  the metadata is 0.25 bits per weight, which means a 4.00-bit target is a 3.75-bit
  payload and is *not* what the field calls a "4-bit g128" checkpoint.

### Added — P5, quantizer driver

- `quant/quantizer.py`: walks a bit map, encodes each tensor, and reports per-layer
  RMSE, relative error, clipped fraction and clip improvement. Those per-layer errors
  are the only diagnostic that separates "three bits is simply lossy" from "one
  tensor was destroyed and dragged the model down", and they are unrecoverable once
  the originals are gone.
- MSE-optimal clipping grid search over α ∈ {1.00 … 0.80}, per row and per group,
  ported from the research code and pinned to its grid verbatim. On Qwen3.5-2B it
  improved SSE on 146 of 187 layers, 9.4% on average.
- `in_place=True` writes the dequantized values back, which is how a model is scored
  without a kernel. The reconstruction is exact to the last bit — pinned by a test —
  so accuracy measured this way is the accuracy of the packed checkpoint. Memory and
  speed measured this way are not, and the docstring says so.

### Added — P6, decode kernels

- `csrc/gemv/gemv.cu`: `y = x @ Wᵀ` with the weight never leaving its packed form,
  templated on `<BITS, MROWS, scalar_t>` for BITS ∈ {2, 3, 4, 8} and M ∈ {1, 2, 4, 8}.
  This is the kernel the whole on-disk format exists for, and the only place where the
  VRAM claim becomes true: `dequant → cublasLt` is the right answer for prefill but
  materialises the fp16 weight, so a model run entirely that way peaks at fp16 size.
- Two kernels behind one host predicate. `gemv_vec_kernel` is the fast path and covers
  every weight a transformer actually contains; `gemv_kernel` accepts any geometry the
  format permits — ragged per-row grouping, groups too small to vectorize, unaligned
  storage — and is what Mamba's 4-tap `conv1d` runs through. `vec_geometry_ok` chooses,
  and `DYNQUANT_GEMV_SCALAR=1` forces the general path, so both stay measurable and
  both stay under test rather than the fallback rotting unexercised.
- `csrc/quantize/dequant.cu` for the M > 8 prefill path, selected at
  `gemv_max_rows() == 8`.
- Accumulation is fp32 throughout and the cross-lane reduction is a fixed-shape
  `__shfl_down` tree — no atomics, no split-K — so the result is bit-identical run to
  run.
- Verified on an A100 80GB PCIe: 417 kernel-parity tests across every
  geometry × width × M × dtype against the torch oracle, and 0 errors from all four
  `compute-sanitizer` tools (memcheck, initcheck, racecheck, synccheck).

### Changed — P6, three rounds of GEMV optimization

Measured on `embed/lm_head` 248320×2048 at M = 1 against 1665 GB/s achievable read
bandwidth. Net effect across the five real shapes of Qwen3.5-2B: **0.64–1.83× → 1.09–2.56×**
versus bf16 `F.linear`, and **26–41 % → 10–98 %** of achievable bandwidth.

- **Vectorized loads (the one that mattered).** The original kernel read the activation
  one scalar at a time: 32 lanes reading two-byte values spaced `kVals·2` bytes apart
  touched sixteen 32-byte sectors and used two bytes of each, once per *value* rather
  than once per word. Since a row has `K` values regardless of bit width, that cost did
  not shrink when the weights did — which is why 2/3/4-bit originally landed within 40 %
  of each other in absolute time while reading 1.4–2× different amounts of weight.
  Making a lane's values consecutive turns its activations into one 128-bit load and the
  sixteen-sector access into four. Chunk geometry was reworked to keep
  `kValues · BITS == kWords · 32` at every width, which makes chunks value-aligned and
  guarantees a chunk's last value ends on the last bit of its last word, so
  `decode_value`'s straddle path needs no bounds guard. 3-bit reads `uint2` (6 words,
  64 values) rather than `uint4`, because a `uint4` chunk would have to be lcm(4,3) = 12
  words and leave half of every warp idle at K = 2048.
- **Full-rate integer→float conversion.** `nbit::decoded_to_float` narrows through
  `unsigned short` before converting. The narrowing is lossless for a ≤8-bit value and
  is not cosmetic: it emits `I2F.F32.U16` instead of `I2F.F32.U32`, and on sm_80 the
  programming guide rates conversions *from* 8- and 16-bit integers at 64 results per SM
  per clock and every other conversion at 16. The SASS showed 257 quarter-rate `I2F`
  against 512 `FFMA` — one instruction per weight costing more issue slots than both
  FFMAs consuming it. Also speeds up `dequant.cu`, which shares `decode_block`.
- **Dequant hoisted out of the per-value loop**, using
  `Σ(qᵢ·s + o)·xᵢ == s·Σqᵢxᵢ + o·Σxᵢ`. Legal because a chunk lies inside one group, so
  `s` and `o` are constant across it. Costs one FFMA per value instead of two, shares one
  activation sum across the rows a warp owns, and is *more* accurate — the reconstructed
  weight is never materialised, so there is one rounding per value where there were two.
  Inner-loop instruction count 1568 → 1384.

  Together these two bought only 2.5 % at 2-bit and 7 % at 4-bit against a predicted ~2×.
  That miss is recorded because it is the diagnostic: an 11.7 % instruction-count cut
  buying 2.5 % of time means the kernel is issue- and memory-latency bound, not FP-pipe
  bound.
- **`kRowsPerWarp` pinned at 4 by measurement, not argument.** A warp re-reads all of `x`
  per row-group, so more rows per warp cuts activation traffic and costs registers. Both
  neighbours lose: rows = 2 gives 34/50/65/98 % and rows = 8 gives 36/49/64/97 % (with
  register spills at M ≥ 4) against rows = 4's 36/52/67/98 % at 2/3/4/8-bit. A maximum
  that is flat in both directions says neither activation traffic nor occupancy is
  binding. The constant now carries the table so the sweep is not repeated.

The remaining gap is quantified rather than guessed: ~5.4 instructions per (value, row)
puts a pure issue-rate floor at ~141 µs against 237 µs measured at 2-bit, and at K = 2048
each warp executes exactly one iteration of the main loop, so there is no intra-warp
memory-level parallelism to hide latency with. Closing it needs `LOP3`/`PRMT` into `half2`
and `mma.sync` accumulation — the AWQ/Marlin route — which is P7.

### Changed — P6, the model-level decode figure was wrong and is now retracted

- **Batch-1 decode is 0.90×, not the 0.98× previously reported.** The old figure paired a
  packed run against a bf16 run measured in a *different session*. Repeated within one
  session, bf16 batch-1 decode came out at 33.1, 29.5 and 33.1 tok/s — a 12 % spread
  between best-of-three runs, wide enough that the slow bf16 run is slower than either
  packed run. A step that is ~2000 launches and ~70 % idle is paced by the host, so host
  jitter exceeds the entire effect being measured; the arms must be sampled repeatedly and
  in one session. Both arms re-measured (3 bf16 runs, 2 packed): 0.90× / 0.92× / 0.94× /
  0.60× at batch 1 / 4 / 8 / 32.
- **`experiments/qwen35_2b/stage7_profile_step.py`** — decode-step attribution for both
  arms through one script. The "71 % launch gaps" figure previously in `RESULTS.md` came
  from a one-off that no longer existed, which is an unreproducible number in a document
  whose argument is that numbers should be reproducible. It encodes the two traps: only
  `DeviceType.CUDA` events may be summed (op-level entries such as `aten::mm` double-count
  the kernels they launch, which is how an earlier pass reported 58 % and 4551 launches),
  and fractions must divide by an *untraced* wall — tracing 2000 launches per step inflates
  the step from 28 ms to 106 ms and makes a launch-bound step look GPU-starved by 3×.
- **What that profile settles:** in-model matmul time is **1.42× faster** packed (3.519 →
  2.486 ms/step), and the whole 1.038 ms drop in GPU-busy time is accounted for by the
  1.033 ms drop in matmul — the kernel does exactly what it was built to do, and nothing
  else moved. It is outweighed on the host: the packed path issues *fewer* kernels than
  bf16 (1980 vs 2013) and still takes longer, putting the cost of `DynQuantLinear.forward`
  and the `torch.ops.dynquant.gemv` dispatch at roughly 20–45 µs per module per step over
  187 modules. So the model-level shortfall is host dispatch, not the kernel, and P8's CUDA
  Graphs are what remove it.

### Fixed — P6

- **The DDP signal-reduction tests had never executed, on any machine.**
  `tests/test_signals_reduce.py` hands `_worker` to `torch.multiprocessing.spawn`, and
  the spawn start method pickles it by `(module, qualname)` — so the child performs a
  real `import tests.test_signals_reduce`. Under `--import-mode=importlib` pytest
  registers test modules in `sys.modules` under that dotted name *without* requiring a
  package, so the parent resolved it and the child could not. All five tests errored with
  a bare `ProcessExitedException`, the real `ModuleNotFoundError` buried in the child's
  captured stderr, which reads like an environment problem and had been taken for one.
  Fixed with `tests/__init__.py` plus `pythonpath = ["."]`; the six tests now run and
  pass, and with them the P2 gate that two-rank reduction equals one-rank reduction over
  the union.

### Fixed — P4

- **Percentile ranks could be exactly zero, which made a module free to destroy.**
  `percentile_ranks` mapped onto the closed interval `[0, 1]` via
  `(rank − 1) / (n − 1)`, so the lowest-ranked member of every group scored exactly
  0. The importance score is a *product* of two ranks, and the allocator prices
  damage as `score × params × Δerror` — so one zero zeroed the score and the module
  went to the hard 2-bit minimum at no cost, for no measurable gain.

  On Qwen3.5-2B at a **4-bit** target this put exactly 20 of 187 modules at 2 bits:
  two per role, being the minimum of each role in each of the two signals. Relative
  reconstruction error reached 0.61 on tensors no evidence called unimportant, and
  the model scored 20.8% on GSM8K against 65.1% unquantized. Nothing in the bit map
  looked wrong — the width histogram was plausible, the target was hit to four
  decimals, and the violation report listed the breaches without hinting any of them
  were unmotivated. Being the least important `v_proj` in a model is not the same as
  being worthless.

  Fixed with the Hazen plotting position `(rank − 0.5) / n`, which maps onto the open
  interval. After the fix, no module is below 3 bits at a 4-bit target. The same
  change removes the mirror-image artifact at the top: a single-member group now gets
  `0.5` rather than `1.0`, so `EMBEDDING` — one instance on most architectures, and
  the largest tensor in the model — is no longer handed the maximum score by
  construction. Non-finite values still rank 0, the one place a hard zero is right,
  because it is the only case where nothing was measured.

  Guarded by `tests/test_score_ranks.py` (no finite value ranks 0 or 1 at
  n ∈ {2, 3, 5, 20, 187}), `tests/test_score_importance.py` (the scorer had no test
  file at all before this, despite being where both artifacts lived), and
  `tests/test_allocate.py`, which pins the *consequence* so the two stay connected:
  a zero score is cut to the minimum regardless of how loose the budget is.

### Notes

- The research code this package is derived from is a confidential NeurIPS
  reviewer copy. The supplementary PDF must not be committed; `.gitignore` and the
  pre-commit guard both block it by pattern.
- P6 has landed, so on a machine with the compiled wheel `dynquant doctor` now reports
  `backend=cuda` and decode runs on packed weights. P7 (tensor-core prefill) and P8
  (MoE grouped GEMM, CUDA Graphs) have not: prefill still goes through
  dequantize-then-GEMM, and above `gemv_max_rows() == 8` so does decode.
- The decode kernel is faster than bf16 both in isolation (1.09–2.56×) and inside the model
  (1.42× on matmul time), but this particular model still decodes **0.90×** at batch 1, and
  the reason is not the kernel: a decode step on Qwen3.5-2B issues ~2000 kernel launches,
  leaves the GPU idle ~70 % of the time, and spends 12 % of its wall clock in matmuls, so
  the 1.42× is worth 3.5 % of the step — less than what dispatching 187 packed modules from
  Python costs. CUDA Graphs (P8) and a `flash-linear-attention` fast path are what address
  it. `experiments/qwen35_2b/RESULTS.md` has the measurement.

[Unreleased]: https://github.com/kambojvikram/dynquant/compare/v0.4.0...main
[0.4.0]: https://github.com/kambojvikram/dynquant/releases/tag/v0.4.0
[0.3.0]: https://github.com/kambojvikram/dynquant/releases/tag/v0.3.0
[0.2.0]: https://github.com/kambojvikram/dynquant/releases/tag/v0.2.0
[0.1.2]: https://github.com/kambojvikram/dynquant/releases/tag/v0.1.2
[0.1.1]: https://github.com/kambojvikram/dynquant/releases/tag/v0.1.1
[0.1.0]: https://github.com/kambojvikram/dynquant/releases/tag/v0.1.0
