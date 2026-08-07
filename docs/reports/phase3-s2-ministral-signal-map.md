# S2, arm 2: Ministral-8B's map, and the tensor ranked against nobody

**Measured 2026-08-06**, on the map produced by the second phase-3 fine-tune.
Script: [`experiments/phase3/s3_allocation/verify_signal_map.py`](../../experiments/phase3/s3_allocation/verify_signal_map.py).
Record: `signal_map.ministral-8b.json` beside it. Artifacts:
[`experiments/phase3/s2_runs/ministral-8b.tulu3/`](../../experiments/phase3/s2_runs/ministral-8b.tulu3/).
Companion: [`phase3-s2-phi-signal-map.md`](phase3-s2-phi-signal-map.md), whose verdict
this run's measurement forced a correction to.

Phi's report closed by naming what would be different here: *"Ministral-8B is untied,
so `embed_tokens` and `lm_head` are separate tensors in two singleton role groups —
and `lm_head` will have its own gradient signal that per-role ranking then discards.
Run this script on that arm before reading anything from it."* That prediction was
right, and the size of what it discards is the finding.

## The run

| | |
|---|---|
| model / data | `mistralai/Ministral-8B-Instruct-2410` × `allenai/tulu-3-sft-mixture` |
| steps | 1492 optimizer steps, LoRA rank 32, effective batch 32, lr 1e-4 |
| final train loss | 0.6335 |
| wall clock | 40 434 s = **11.23 h** |
| conversations | 50 000 seen, 47 730 kept (4.54% dropped) |
| tokens | 26 762 820 total, 19 144 448 supervised |
| masking | `assemble` mode, probe tie 30/30 against `template` 0/30, 0.14% unmaskable |
| estimator | `outer_exact` |

The masking mode differs from Phi's, which used `template`. The probe settles it per
model rather than per campaign — see [`phase3-s2-loss-masking.md`](phase3-s2-loss-masking.md).

## Structure: all four properties hold

- **254 modules tracked** = `2 + 36 × 7` — every `q_proj`, `k_proj`, `v_proj`,
  `o_proj`, `gate_proj`, `up_proj`, `down_proj`, plus `embed_tokens` and `lm_head`.
  Unfused, unlike Phi, so no row-partition surcharge applies here.
- **No tie.** `tied_parameters` is empty. This is the structural difference from Phi
  and it removes a check rather than adding a problem, below.
- **`grad_norm_count` is 1492** on every module that has one, equal to the
  optimizer-step count — the guard against the supplement's per-micro-batch bug 10.
- **`forward_calls` is uniformly 23 865**, one distinct value across all 254 modules,
  so gradient checkpointing stayed off and every module's EMA decayed on the same
  footing.
- **`channel_moment_modules: 253`** of 254. The absentee is `embed_tokens`, for the
  documented reason that an embedding's Gram axes are transposed relative to how its
  weight is stored. On Phi the tie let the head's moments cover the same tensor;
  here nothing covers it.

## Spread: the signals rank

| signal | median | max/min | zeros |
|---|---|---|---|
| `activation_rms_ema` | 0.3983 | **1221** | 0 |
| `grad_norm_var` | 8.992e-08 | — | 1 |
| `grad_norm_mean` | 1.135e-03 | — | 1 |

Saliency spans three orders of magnitude, wider than Phi's 429×. Both signals
discriminate. The single zero is `embed_tokens`.

## Two tensors, both scored 0.5, for different reasons

| | `lm_head` | `model.embed_tokens` |
|---|---|---|
| parameters | 536 870 912 (6.69%) | 536 870 912 (6.69%) |
| role floor | 8 b | 4 b |
| role group size | **1** | **1** |
| `forward_calls` | 23 865 | 23 865 |
| `grad_norm_count` | **1492** | **0** |
| `activation_rms_ema` | **2.8528** | 0.00305 |
| `grad_norm_var` | 2.238e-06 | 0 |
| shipped score | **0.5** | **0.5** |
| score on its own measurements, ranked globally | **0.9783** | 0.5 |

13.4% of the model allocated on a number that reflects no comparison. The two cases
are not the same failure:

- **`model.embed_tokens` has nothing to rank.** Under `outer_exact` the base-weight
  gradient is reconstructed as `∇W = δxᵀ`, and `signals/tracker.py:705` returns early
  unless the module's *output* requires grad. Under LoRA the embedding weight is
  frozen and its input is integer token ids, so nothing upstream requires grad,
  autograd never materialises `dL/dY` at that node, and the output-gradient hook is
  never registered. Ranking it globally returns the same 0.5, which is the honest
  answer: no measurement exists. (This is not a missing estimator —
  `estimators.py:161 embedding_gram` implements the token-equality Gram matrix for
  exactly this case. It is the hook that never fires.)
- **`lm_head` has everything to rank and no one to rank against.** 1492 gradient
  observations, **rank 1 of 254 on saliency** (2.8528, 1.78× the runner-up
  `layers.16.self_attn.k_proj` at 1.5984) and **rank 6 of 254 on plasticity** — so
  this is not a tensor that merely has large activations for scale reasons, which is
  the obvious way an LM head could top the saliency table without being fragile. Both
  signals agree. Per-role ranking then computes its percentile against a set
  containing only itself and returns 0.5. Phi never showed this because Phi's head is
  tied and does not appear as a separate quantizable tensor.

`score_modules` bins the first case into `unexercised`
([`score/importance.py:188`](../../packages/dynquant-core/src/dynquant/score/importance.py#L188))
and the second nowhere at all. The verifier's coverage line counted `lm_head` as
*scored*, because it is. It now reports `informed` separately.

## What the 0.5 costs

Bracket (force the score to 0.0 and to 1.0, spanning every width any signal could
buy) against the realistic counterfactual (the module's own measurements ranked
globally):

### `lm_head`

| target | score 0.0 | score 1.0 | shipped | on its own signal | other modules moved |
|---|---|---|---|---|---|
| **3.25** | 2 b | 4 b | **3 b** | **4 b** | **24** |
| 4.0 | 4 b | 4 b | 4 b | 4 b | **23** |
| 4.25 | 8 b | 8 b | 8 b | 8 b | 0 |
| 4.5 | 8 b | 8 b | 8 b | 8 b | 0 |

At the headline 3.25-bit target the shipped map gives the model's highest-saliency
tensor **3 bits where its own measurement buys 4** — a whole bit on 6.7% of an 8B
model, paid for by 24 attention and MLP projections dropping 4 b → 3 b. Byte totals
are **identical** (3 007 315 968 both ways) and average bits identical at 3.2500, so
this is purely where the bits went, not how many.

At 4.0 the head does not move and **23 other modules do**. That is the general shape:
the budget is shared, so a large tensor's position in the ROI order decides how much
budget reaches everything below it even when its own width is pinned.

### `model.embed_tokens`

| target | score 0.0 | score 1.0 | shipped | on its own signal | other modules moved |
|---|---|---|---|---|---|
| 3.25 | 2 b | 4 b | 3 b | 3 b | 0 |
| 4.0 | 3 b | 4 b | 4 b | 4 b | 0 |
| 4.25 | 4 b | 4 b | 4 b | 4 b | 0 |
| 4.5 | 4 b | 4 b | 4 b | 4 b | 0 |

The bracket is open at both S3 targets — a signal *would* have moved this tensor —
and there is no signal to supply. On Phi the tie provided one and it came back
clean. **Untying removes the only check available**, which is the part worth carrying
forward: the tied case looked like the harder one and was the one that could be
verified.

## The control: is this the allocator being chaotic?

A greedy ROI knapsack near its ceiling could reshuffle its tail under any nudge, in
which case "24 modules moved" would be a fact about the allocator and not about the
singleton. Forcing the *same* score (0.9783) onto 24 ordinary projections sampled
across depth and role:

| arm | modules moved at 3.25 | at 4.0 |
|---|---|---|
| `lm_head` (536.9 M, singleton) | **25** | **23** |
| `model.embed_tokens` (536.9 M, singleton) | 25 | **0** |
| 24 ordinary projections (4.2 M – 50.3 M) | median 0, max 8 | median 0, max 2 |

The map is not chaotic: the median ordinary module moves nothing. And the 23 at 4.0
is not a size effect either — `embed_tokens` is the *same* 536 870 912 parameters and
moves nothing at that target, because it already sits at its 4-bit floor and is not a
knapsack candidate. `lm_head` sits four bits *below* its 8-bit floor, so it is a live
candidate with an enormous byte cost, and its ROI position decides the tail.

**The neutral score bites hardest exactly where the module is large and its floor is
breached — which is the regime DynQuant's headline targets are defined by.**

## What this does and does not settle

Settled: the shipped map is not the map the measurements imply, at the headline
target, on this checkpoint, by one bit on 6.7% of the parameters.

Not settled: which map is better. The role floors already encode a prior that the LM
head is sensitive (8 b, the highest floor in the table) and the measurement agrees
with that prior, which is suggestive and is not evidence. Only an eval decides it,
and that is an S3 arm — the same shape as the `shuf3` control in
[`phase3-s3-null-control.md`](phase3-s3-null-control.md).

## Verdict for S3

The map is usable. Two things carry forward:

1. **Ministral's arms must be read knowing 6.7% of the model is placed on 0.5 at
   3.25 b.** Phi's equivalent caveat is 16.0% and was checkable; this one is 13.4%
   across two tensors and half of it is not.
2. **Do not change the scorer mid-campaign.** Making `score_modules` keep a measured
   saliency when plasticity is absent, or ranking singleton groups globally, would
   change every model's allocation including phase 2's published comparisons. Both
   are defensible and both are a decision, not a fix. Recorded here; not taken.

The verifier was extended rather than the scorer: it now enumerates singleton role
groups, reports `informed` apart from `scored`, and measures whole-map movement
instead of one width. Pinned by
[`tests/test_verify_signal_map.py`](../../tests/test_verify_signal_map.py).
