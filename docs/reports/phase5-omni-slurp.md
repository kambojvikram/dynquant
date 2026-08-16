# Phase 5: DynQuant on a 30 B audio MoE — 4 bits holds, 3 bits does not

**Run 2026-08-16** on `Qwen/Qwen3-Omni-30B-A3B-Instruct`, QLoRA fine-tuned on SLURP intent
classification and quantized with DynQuant at two budgets. Box: vast.ai instance 47855285,
**2× RTX PRO 6000 Blackwell Max-Q** (94.97 GiB each, sm_120), torch 2.11+cu128,
transformers 5.15.0, `dynquant-core` 0.4.0. This is the first campaign on this machine and
the first on a model with batched MoE expert banks carrying the *majority* of the
parameters — 28.991 B of 31.719 B, **91.40%**.

The design, the architecture census and every pre-registered gate are in
[`docs/phase5-qwen3-omni-strategy.md`](../phase5-qwen3-omni-strategy.md). This file is what
the arms returned.

## The result

Every arm is SLURP `test`, 4 shots from `train` at seed 0, 500 items, greedy, 8 new tokens,
60 intents (`intents_sha` `d04b663b…`, identical across all four). Per-item `hits` are stored,
so every comparison below is an exact **McNemar** test on paired outcomes.

| arm | what it is | accuracy | vs its comparator | discordant | *p* |
|---|---|---|---|---|---|
| `omni-base` | the whole Omni checkpoint, bf16, no fine-tune | 79.40% (397/500) | — | — | — |
| `omni-sft` | Thinker-only, adapter merged, bf16 | **86.80%** (434/500) | **+7.40** vs base | 47/10 | 7.51e-07 **separated** |
| `omni-dq4` | the same weights at a 4.00-bit map | **86.20%** (431/500) | **−0.60** vs sft | 12/15 | 0.7011 not separated |
| `omni-dq3` | the same weights at a 3.00-bit map | **25.00%** (125/500) | **−61.80** vs sft | 3/312 | 1.56e-88 **separated** |

95% CIs on the paired differences: base→sft `[+4.51, +10.29]`, sft→dq4 `[−2.64, +1.44]`,
sft→dq3 `[−66.17, −57.43]`.

Read plainly:

- **The fine-tune worked.** +7.40 points over the base checkpoint, separated at *p* = 7.5e-07.
- **4 bits is free, to the resolution 500 items buys.** dq4 loses 0.60 points and the
  comparison does not separate. The honest form of that claim is the interval: at
  **4.00× fewer bytes** this excludes damage worse than 2.64 points; it does not establish
  that damage is zero.
- **3 bits destroys the model.** 25.00% against a 1.67% chance floor — above chance, and
  three items came back unparseable at all, against zero in every other arm. The discordance
  is 3/312: dq3 recovered three items the bf16 ceiling missed and lost 312 it had.
- **Both budgets also exist as packed checkpoints on disk**, at **3.99x and 5.32x** less
  resident VRAM than the bf16 merge, and each reproduces the arm above it on **all 500 items**.
  [Below](#the-packed-checkpoint-computes-the-same-thing).

`experts` reports `{found: grouped_mm, ran: eager}` in all three post-merge arms, so every arm
ran the same expert arithmetic and no comparison here straddles two dispatch paths.

## The two budgets are two different experiments

This was pre-registered before any arm ran, and it is the reason the two results differ in
kind and not only in degree. **The role floors alone cost 3.418 average bits on this
architecture.** So:

- At **4.00** the floors fit with 0.582 bits to spare. Every floor is respected —
  `violations: 0` — and the score decides only *what gets upgraded*.
- At **3.00** the floors are unreachable. Soft floors bind, the allocator downgrades by
  lowest ROI, and the score decides *what gets cut*. It reports **400 of 650 modules
  breached**.

A hard-floor allocator would simply have refused 3.00. That the arm ran at all is the soft
floor policy working as designed; that it returned 25% is the answer to what those cuts cost.

### What the allocator actually did

Run against the final 500-step signal file (not the step-200 snapshot the strategy doc's
offline preview used — the byte totals reproduce exactly, the histograms do not):

| | 3.00 bits | 4.00 bits |
|---|---|---|
| achieved average bits | 2.999976 | 3.999945 |
| stored bytes | 11 894 607 776 (11.08 GiB) | 15 859 386 016 (14.77 GiB) |
| vs the 59.1 GiB bf16 merge | **5.33×** | **4.00×** |
| 2-bit | 125 modules, 13.596 B params | 9 modules, 1.812 B |
| 3-bit | 368 modules, 12.596 B | 67 modules, 5.698 B |
| 4-bit | 109 modules, 5.507 B | 524 modules, 23.877 B |
| 8-bit | 48 modules, 0.013 B | 50 modules, 0.325 B |
| floor violations | **400** | **0** |
| within-role width/score concordance | 1.0000 (8791 pairs) | 0.9943 (1234 pairs) |

At 3.00, **42.9% of the parameters sit at 2 bits.** The breaches by role:

| role | modules breached | params | moves |
|---|---|---|---|
| `moe.expert.gate_up` | 35 of 48 | 14.093 B | 4→2 ×9, 4→3 ×26 |
| `attn.o` / `attn.q` / `attn.k` / `attn.v` | 58 of 80 each | 0.419 / 0.331 / 0.074 / 0.067 B | 4→2 ×14, 4→3 ×44 (each) |
| `lm_head` | 1 of 1 | 0.311 B | **8→3** |
| `embedding` | 1 of 1 | 0.311 B | 4→3 |
| `other` | 65 of 70 | 0.280 B | 4→3 |
| `vision.mlp` | 54 of 54 | 0.268 B | 4→3 |
| `mlp.up` / `mlp.down` | 6 of 32 each | 0.039 B each | 3→2 |

**The 48 routers are never breached, at either budget.** `MOE_ROUTER` is in
`STRUCTURAL_ROLES`, so the tensor that decides which 500 M parameters a token passes through
keeps its 8 bits while everything around it is cut. That is the one part of the floor policy
this campaign confirms works: the collapse at 3 bits is not a routing collapse.

`moe.expert.down` is also absent from the breach list — it is already at its own 2-bit floor
and there is nothing below it to cut to.

## Where the 3-bit collapse comes from

400 modules were breached, so no single one *is* the cause. But one breach is much larger
than the rest — `lm_head` fell from a floor of **8** to **3**, the only 5-bit drop in the
map — and it is the tensor that emits the token SLURP scores. That is cheap to measure
directly, so it was measured rather than argued: load that one weight out of the merged
checkpoint, encode it at each width with the same encoder the arms used, and ask both how far
the weights moved and how often the **decision** changes.

| width | relative error | top-1 agreement | groups clipped | MSE removed by the clip search |
|---|---|---|---|---|
| 8 (its floor, and dq4's assignment) | 0.0059 | **99.8%** | 0.0% | 0.0% |
| 4 | 0.0951 | 77.7% | 95.4% | 11.0% |
| **3 (dq3's assignment)** | **0.1873** | **58.6%** | 100.0% | 25.0% |
| 2 | 0.4070 | 28.5% | 100.0% | 35.3% |

Top-1 agreement is measured against unit-norm gaussian probes in the head's input space,
because the hidden states the head really sees are not available offline. Real hidden states
concentrate where the top-1 margin is wider, so each figure is an **upper bound on the damage**
that width does, not an estimate of it. What survives that caveat is the ordering, which the
probe distribution does not change: the arm that kept `lm_head` at 8 bits (99.8% agreement)
tied its bf16 ceiling, and the arm that cut it to 3 (≤58.6%) collapsed.

This is consistent with the whole-model encode error, which the arms report directly:
**median relative error 0.0973 at dq4 against 0.1912 at dq3**, worst module 0.4134 against
0.6309. dq3's six worst are `audio_tower.layers.{1,2}.fc{1,2}` and
`model.layers.{42,43}.self_attn.v_proj`, all at 2 bits with 95.7–99.9% of groups clipped.

**What this does not establish.** It does not isolate `lm_head` as *the* cause — 42.9% of the
parameters were simultaneously moved to 2 bits, and either injury alone might be sufficient.
Separating them needs an arm that pins `lm_head` at its floor and spends the difference
elsewhere, which is a change to the floor policy (`LM_HEAD` is not currently in
`STRUCTURAL_ROLES`, `MOE_ROUTER` is) rather than another run of the existing one. That
experiment is named here and was not run.

## Measured sensitivity covers 1.02% of this model, and moves 3.5% of the widths

`--moments` switches the allocator from rank-product ordering to measured Gauss–Newton
sensitivity. On this run the moments exist for **49 of 654 modules — 0.324 B of 31.719 B** —
because the moment hook guards on `inp.shape[-1] == entry.weight.shape[1]` and a
`bitsandbytes` 4-bit weight is stored flat, so the guard rejects every module bnb replaced.
The 49 survivors are the 48 routers plus `lm_head`. Expert banks are excluded separately and
deliberately: the hook takes `kind == "linear"` and a bank is 3-D.

Both maps were allocated and diffed rather than argued about:

| | 3.00 | 4.00 |
|---|---|---|
| modules whose width changes | **23 of 650 (3.54%)** | **0 of 650 (0.00%)** |
| stored bytes, proxy → moments | 11 894 607 776 → 11 894 599 584 | identical |
| floor violations, proxy → moments | 400 → 392 | 0 → 0 |

At 4.00 the two allocations are **identical**: measured sensitivity on 1% of the parameters
did not reorder anything the budget could act on. At 3.00 it moves 23 modules — 7 `mlp.up`
and 6 `mlp.down` 3→4, two of each attention projection 3→4, one `moe.expert.down` 2→3, and
**`lm_head` 3→2**, i.e. the measured arm cuts the output head *further*. Given the table
above, that is a reason to prefer the proxy-priced map here, and the primary arms used it.

One mechanical note for whoever reads the saved maps: the sensitivity allocator records
`pricing.scale: null`, because it prices measured modules directly and has no single global
fallback multiplier, where the rank-product path rescales every proxied module by one number.
A reader that formats that field unconditionally crashes on the moments map; the
throwaway diff script used here prints `none` for it. That script is not shipped —
`s5/map_diff_*.json` are its output, kept as artifacts rather than as a tool.

## The packed checkpoint computes the same thing

Both budgets have been exported with `dynquant export`, loaded back, and scored. This is the
section the `encode` caveat below points at, and it changes exactly one claim — the VRAM one.

**Export.** 650 of 650 modules written on each arm, in **78 s** and **74 s**, *including all
48 routers*, which `restore=True` answers with a `RestoredWeight` rather than refusing. Both
totals land on the map's byte prediction **exactly**:

| arm | packed | + dense | = total | map predicted | avg bits |
|---|---|---|---|---|---|
| dq4 | 15 845 134 784 | 14 251 232 | **15 859 386 016** | 15 859 386 016 | 3.9972 |
| dq3 | 11 880 356 544 | 14 251 232 | **11 894 607 776** | 11 894 607 776 | 2.9971 |

The dense residue is *inside* the map's total rather than beside it — the norms and biases the
map does not name are already priced at 16 bits in its denominator — which is why the sum lands
on the prediction to the byte instead of near it. Both residues are the same 14 251 232 B, as
they must be: the map changes what is quantized, not what is left.

**Read-back.** Both directories load to `{DynQuantLinear: 504, DynQuantEmbedding: 2,
DynQuantExpertBank: 96}` with **48 of 48 routers dense**. The check that separates "the class
says packed" from "the values are packed" is the value lattice, and it has to be counted **per
quantization group**: a 4096-element slab spans 32 groups of 128, each with its own scale and
zero, so its ceiling is `32 x 2**bits` and it proves nothing. The first probe made exactly that
mistake and printed 371 distinct against a claimed ceiling of 16. Counted per group of 128 it is
exact, and it shows more than the width — it shows the **map's heterogeneity surviving into the
file**:

| module | 4.00 arm | 3.00 arm |
|---|---|---|
| `audio_tower.layers.0.self_attn.{q,k,v}_proj` | 4b, 13-16 distinct of 128 | 2b, 4-4 of 128 |
| `audio_tower.layers.0.self_attn.out_proj` | 4b, 14-16 | 3b, 8-8 |
| `audio_tower.layers.0.fc1` | 3b, 8-8 | 2b, 4-4 |
| `audio_tower.layers.0.fc2` | 3b, 6-8 | 3b, 6-8 |
| `model.layers.{0,1,2}.mlp.experts.{gate_up,down}_proj` (banks) | not separately probed | — |
| `model.layers.0.mlp.gate` (router, never quantized) | 65-95 of 128 | 65-95 of 128 |

Every count sits at or just under its own ceiling, never above, and the dense router sits an
order of magnitude above all of them. The 3.00 column is the floor-breach story made physical:
the six modules the 4.00 map places at 4/4/4/4/3/3 bits are at 2/2/2/3/2/3 in the 3.00 file.

The banks are the one row this table cannot fill. The per-group probe covered the audio
tower and the router only; the sole bank measurement anywhere is the discredited whole-slab
count above, so 91.4% of the model is attested by its module class and its manifest widths
and not by a counted lattice. Nothing here contradicts it — 371 distinct over 32 groups of a
16-value ceiling is exactly what a correct 4-bit bank would print — but it is not the same
evidence as the rows above, and the section exists to keep those two apart.

**VRAM is the packed size.** `torch.cuda.memory_allocated` across the load, on an idle card:

| arm | resident VRAM | manifest total | over | vs the bf16 merged Thinker (63 440 876 184 B) |
|---|---|---|---|---|
| dq4 | 15 892 454 912 | 15 859 386 016 | **+0.21%** | **3.99x** |
| dq3 | 11 927 683 584 | 11 894 607 776 | **+0.28%** | **5.32x** |

A fifth of a percent over the payload is **mostly the routers, not metadata**, and the split is
exact. The 48 MoE routers are written packed at 8 bits (270 336 B each) but restored to a dense
bfloat16 weight at load, because nothing in the packed runtime can stand where a router stands.
Each one therefore costs 524 288 B resident against 270 336 B on disk, and 48 x 253 952 B =
**12 189 696 B** of the overage is that restoration alone — 36.9% of it in both arms
(dq4 33 068 896 B, dq3 33 075 808 B). The ~20.9 MB remainder is the runtime's own metadata. The
router figure is a design decision and is identical across the two arms because `MOE_ROUTER` is
a structural floor at 8 bits in both maps; the metadata remainder is what actually scales.
Had any module fallen back to fp16 the residue would be gigabytes, not tens of megabytes.
**This is the claim `encode` could not make**, and it is now made off
`torch.cuda.memory_allocated`, not off the map.

**Both packed arms then scored, and both reproduce their `encode` counterpart item for item.**

| arm | accuracy | unparseable | seconds | vs its `encode` twin | discordant | identical kept predictions |
|---|---|---|---|---|---|---|
| `omni-packed4` | **86.20%** (431/500) | 0 | 497.2 | `omni-dq4` 86.20% | **0 of 500** | 40/40 |
| `omni-packed3` | **25.00%** (125/500) | 3 | 732.5 | `omni-dq3` 25.00% | **0 of 500** | 40/40 |

Zero discordant pairs is not a lucky tie and it is not an independent replication — **it is the
predicted result, and that is what makes it a test.** `encode` writes the encoder's dequantized
output back into the dense tensor at compute dtype; `pack` stores the codes and dequantizes at
compute time. Same codes, same scales, same dtype means bit-identical weights, so greedy decoding
must emit identical tokens. A single discordant item would have meant the packed path diverges
from the encoder somewhere. None does, on 1 000 paired items across two widths, down to the
unparseable counts (0 and 3) matching too. There is no McNemar test to run: `b01 = b10 = 0`.
The eval's own `--compare` says the same thing without going through this page's arithmetic —
`{both_right: 431, a_only: 0, b_only: 0, both_wrong: 69}` and `{125, 0, 0, 375}` — and both are
in [`s6/packed_eval.log`](../../experiments/phase5/omni-slurp/s6/packed_eval.log).

Both packed arms record `experts: {found: "dynquant", ran: "eager"}` against the `encode` arms'
`{found: "grouped_mm", ran: "eager"}`. The *found* value differs because the packed config
declares DynQuant experts and the merged one declares torch's grouped matmul; the value that
decides the arithmetic is `ran`, and it is `eager` on all four. **No comparison on this page
straddles two dispatch paths.**

The 3-bit artifact is a **25.00% model**, and being on disk does not make it more defensible.
It is published here as a reproduction of a measured collapse, labelled as one.

## What this campaign cannot claim

**No baseline arms.** GPTQ, AWQ and RTN were descoped for this phase. There is therefore no
matched-bytes competitive claim here, and specifically **no evidence about whether the 3-bit
collapse is DynQuant's allocation or simply 3 bits on this architecture.** A uniform-width
control at the same byte anchor would answer that in one run and was not run.

**The four scored arms above ran `--map-apply encode`, not `pack`.** `pack` swaps modules onto
the packed runtime and is the mode that makes VRAM equal the packed size. `encode` writes the
same encoder's output back in the compute dtype: the same values at fp16 footprint. So **for
those four arms accuracy is the quantized model's accuracy and the size claim is the map's, not
`nvidia-smi`'s.** The reports say so in their own `packed.holds` field. That limit no longer
stands on its own: both budgets have since been exported and scored packed, and the VRAM claim
is made in [the section above](#the-packed-checkpoint-computes-the-same-thing).

An earlier revision of this section gave the wrong reason for that choice, and the wrong reason
was worse than the right one, so it is corrected here rather than quietly replaced. It said
`pack` "cannot reach a batched MoE expert bank … so a `pack` run would have scored a model whose
experts were still bf16 and printed a packed size anyway." **That is false, and it was already
false when this campaign ran.** `resolve_target` answers a rank-3 tensor with an `ExpertBank`
and `_wrap` turns it into a `DynQuantExpertBank`, which landed 154 commits before the commit
these arms ran on; all 96 banks are packable. What `pack` refuses is `mlp.gate`: a meta-device
type census of the map's 650 names returns **504 `nn.Linear`, 96 bare rank-3 parameters, 48
`Qwen3OmniMoeThinkerTextTopKRouter`, 2 `nn.Embedding`**, and the router owns a weight while
being neither a Linear nor a bank, so `resolve_target` **raises** on it. A `pack` run would have
died at `model.layers.0.mlp.gate` with the error naming the remedy — loud, not silent. No arm in
this campaign was ever scored on bf16 experts; every one records `packed.apply = "encode"`.

The correction had a consequence worth stating: **a packed checkpoint is producible.**
`dynquant export` resolves the same names with `restore=True`, which answers the router with a
`RestoredWeight`, so all 650 names pass — see §9 of
[the strategy doc](../phase5-qwen3-omni-strategy.md). It has now been run, at both budgets, and
the results are below. The 3-bit artifact is a 25.00% model and is labelled one.

**Base→SFT differs in two things.** The S1 screen ran the whole Omni checkpoint; every arm
after the merge runs the Thinker alone. It is still a fair comparison — the Talker is strictly
downstream of the text the Thinker emits, and SLURP scores that text — but the arms carrying
the campaign's actual claim (`sft` vs `dq4` vs `dq3`) are all Thinker-only and differ in
exactly one thing.

**The adapter was trained against NF4 and merged into bf16.** Standard for QLoRA, shared with
every prior phase, and it means the merged checkpoint is not bit-for-bit the model that
produced the signal map.

## Defects found on the way

**`dynquant eval` scored a randomly initialised model and reported it as an accuracy.** The
first two packed arms returned **0.0%, with 499 and 500 of 500 answers unparseable** — the
signature of a destroyed checkpoint. The checkpoint was fine. Nothing on the eval path called
`register_hf_quantizer`, so `from_pretrained` asked
`AutoHfQuantizer.supports_quant_method({"quant_method": "dynquant"})`, got **False with a
`logger.warning` and no exception**, set `pre_quantized = False`, and built a model from the
config with **random weights**. The literal line, recovered from a CPU probe: `Unknown
quantization type, got dynquant ... Hence, we will skip the quantization.` It cost two 8-minute
arms. The sharpest part is that the fix already existed and was already documented:
`integration/hf_quantizer.py` explains this exact failure under a heading reading *"Why this
exists, measured rather than assumed"*. It had never been **called**. `load_model` in
[`commands/_shared.py`](../../packages/dynquant-core/src/dynquant/commands/_shared.py) — the
single loading seam for every subcommand — now registers first and, for any checkpoint whose
config declares `quant_method: dynquant`, counts packed modules afterwards and raises if the
count is zero. The re-run's proof is a count: **0 occurrences of that warning against 2 in the
random-init run.**

**`export` wrote a directory `AutoProcessor` refuses, and the error named the wrong file.**
Both packed evals exited before loading a weight on `OSError: Can't load image processor ...
preprocessor_config.json`. That filename is a red herring: the merged bf16 directory has no
`preprocessor_config.json` either and loads fine. The file that decides is
**`processor_config.json`** — with it present `AutoProcessor` resolves `Qwen3OmniMoeProcessor`
directly, and without it resolution falls through to `image_processing_auto`, which is what
raises. `_save_tokenizer` calls `tokenizer.save_pretrained`, which writes none of the five
processor sidecars. It now copies them, and the negative control in
[`tests/test_processor_sidecars.py`](../../tests/test_processor_sidecars.py) reads the names
back off `transformers.processing_utils` rather than restating them in a fixture.

**A `pkill -f` killed the shell that ran it.** Relaunching the eval as
`ssh box 'pkill -f probe_evalload.py; ... nohup bash run_packed_eval.sh &'` matched the ssh
command line itself, so the kill succeeded, the relaunch never happened, and ssh returned exit
1 with no output. The previous run's log was still on disk, so the monitor's completion grep
matched a **stale** artifact and reported the new run finished in seconds. Same shape as the
`pgrep -f` deadlock below, one step worse. The rerun writes to a new log path so a stale one
cannot be mistaken for it.

**A norm difference is not a distance.** The merge verifier fingerprinted every parameter by
its L2 norm and aborted a *correct* merge, reporting 9 of 384 LoRA targets as unmoved. Three
independent flaws: `‖W+D‖−‖W‖` is `⟨W,D⟩/‖W‖` to first order, so a near-orthogonal delta
registers only at second order; the 7.7e-7 … 1.2e-4 band it produced is continuous, with
passing minimum 1.073e-6 against failing maximum 9.888e-7, so the threshold fell *inside* the
distribution; and the norms were reduced in fp32, whose error on a 6.5 M-element tensor is
~1e-4 — **a hundred times the signal**. Replaced by comparing 4096 name-seeded sampled
elements for inequality: no threshold, no reduction, and a categorical gap — **48.0%–90.9% of
the sample changed on every merged weight against exactly 0.0% everywhere else**. Seven
regression tests pin the geometry in
[`tests/test_run_s3_omni_merge.py`](../../tests/test_run_s3_omni_merge.py).

**A `pgrep -f` watcher matched itself.** The S3 chain polled
`pgrep -f "run_s2_omni_finetune.py --model"`, which matched an older watcher whose own
command line contained that literal string — and that watcher's `pgrep` matched *itself*.
Training had finished 40 minutes earlier with both GPUs at 2 MiB and nothing errored. The
only symptom of the deadlock was a stage that had not started, which reads identically to
"still running".

**`save_pretrained` OOMs beside its own model.** transformers v5 calls
`revert_weight_conversion`, which un-fuses weights with `torch.chunk(...).contiguous()` on
whatever device holds them — 93.60 GiB of 94.97 GiB in use when it fired. The merge needs no
GPU; only the write does. Running the whole stage with `--device cpu` costs 48.1 s.

## The artifacts

All of these are banked under
[`experiments/phase5/omni-slurp/`](../../experiments/phase5/omni-slurp/), at the paths in the
first column.

| file | what it holds |
|---|---|
| `s1/omni.slurp.json` | base screen, 500 per-item hits |
| `s2_omni_finetune.json` | the fine-tune record: 8 000 of 50 628 train items, 59 distinct intents, 654 tracked modules, 96 banks over 6 shards, `outer_exact`, 2 ranks × batch 16 × 500 steps in 4 188 s |
| `s2/stats/dynquant_stats.json` | **the signal itself** — the saliency and plasticity moments every allocation on this page was made from |
| `s2/stats/dynquant_moments.safetensors` | the channel-moment sidecar, the 49 modules of it that exist |
| `s2/trainer_state.json` | the loss curve, 5.479 → 3.080 over 500 steps, mean 4.032 |
| `s5/omni.{sft,dq4,dq3}.slurp.json` | the three post-merge arms, per-item hits, 40 kept predictions, encode error summary |
| `s5/inspect{,_moments}.json` | both allocations in full — per-width params and score ranges, every violation, concordance, the 119 unexercised modules |
| `maps{,_moments}.json` | the saved bit maps both budgets were evaluated from |
| `s5/map_diff_{3.00,4.00}.json` | the proxy-vs-moments diff, rolled up by role |
| `s3_omni_merge.json` | the merge record: 384 moved, 96 banks bit-identical, sampled-change bounds |
| `s6/manifest_dq{4,3}.json` | the packed export manifests, per-tensor: 650 modules, bits, group size, bytes |
| `s6/export.log` | both export runs, 78 s and 74 s, with the byte reconciliation |
| `s6/omni.packed{4,3}.slurp.json` | the two packed arms, per-item hits and 40 kept predictions |
| `s6/readback{4,3}.log`, `s6/lattice4b.log` | module-type census, per-group value lattice, resident VRAM |

`/workspace` on a vast.ai box is not a persistent volume, so these were pulled off-box as
produced. Three things are deliberately not in the repo: the two packed checkpoints (15.86 GB
and 11.89 GB), which are reproducible from the merge and the maps in one `dynquant export` each;
the 59.1 GiB merged checkpoint, which is reproducible from the adapter and the base weights; and
the 96 MB LoRA adapter itself,
which is staged in the untracked `_offbox_artifacts/` rather than committed. The adapter is
the only artifact here that cost GPU hours and cannot be recomputed from anything in this
repository, which is the reason it is copied off the box at all.
