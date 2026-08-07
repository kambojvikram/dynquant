# S2, arm 1: is Phi-4-mini's signal map usable?

**Measured 2026-08-05**, on the map produced by the first phase-3 fine-tune.
Script: [`experiments/phase3/s3_allocation/verify_signal_map.py`](../../experiments/phase3/s3_allocation/verify_signal_map.py).
Record: `signal_map.phi4-mini.json` beside it. Artifacts:
[`experiments/phase3/s2_runs/phi4-mini.tulu3/`](../../experiments/phase3/s2_runs/phi4-mini.tulu3/).

The run finished; that is not the same as the run having produced something. A
fine-tune that completes and writes a well-formed stats file can still emit a
signal that is degenerate — identical across modules, or absent on a module large
enough to decide the allocation by itself. Both look like success in the trainer
log, and both would be discovered in S3 as an inexplicable result rather than as a
measurement failure. So the map is checked before it is spent.

## The run

| | |
|---|---|
| model / data | `microsoft/Phi-4-mini-instruct` × `allenai/tulu-3-sft-mixture` |
| steps | 1501 optimizer steps, LoRA rank 32, effective batch 32, lr 1e-4 |
| final train loss | 0.6747 |
| wall clock | 45 632 s = **12.68 h** |
| conversations | 50 000 seen, 48 015 kept (3.97% dropped, all over `max_len` 2048) |
| tokens | 25 975 349 total, 18 583 548 supervised |
| masking | `template` mode, probe tie 30/30, 0.0% unmaskable |
| estimator | `outer_exact` |

`sources_overlapping_an_eval_task` names one Tulu subset,
`tulu_v3.9_open_math_2_gsm8k_50k` (2615 rows) — already scanned against GSM8K
test, 2/1319 flagged, 0 usable duplicates.

## Structure: what a healthy map has to show

All four properties hold.

- **130 modules tracked**, which is `2 + 32 × 4` — every `qkv_proj`, `o_proj`,
  `gate_up_proj`, `down_proj`, plus `embed_tokens` and `lm_head`. Canonical names,
  so nothing needs read-time guessing.
- **The tie is recorded**: `tied_parameters: {"model.embed_tokens": ["lm_head"]}`.
- **`grad_norm_count` is 1501**, equal to the optimizer-step count, on every module
  that has one. The supplement's bug 10 updated Welford per *micro-batch*, which
  inflates the count 16-fold here and changes the variance it reports; this is the
  guard that it stayed fixed.
- **`forward_calls` is uniformly 12 004** — one distinct value across all 130
  modules. This is the check that matters most and reads as the least. A module
  inside a gradient-checkpointed block fires its forward hook twice per micro-batch
  on identical data; two EMA updates with the same value leave the fixed point alone
  but square the decay, so the replayed modules end up on a different footing from
  the rest and the saliency ranking silently compares two populations. A single
  distinct value means checkpointing stayed off and the comparison is sound.

## Spread: the signals rank

| signal | min | median | max | max/min |
|---|---|---|---|---|
| `activation_rms_ema` | 0.03718 | 0.3541 | 15.94 | **428.9** |
| `grad_norm_var` | 0 | 1.218e-08 | 1.041e-05 | — |
| `grad_norm_mean` | 0 | 3.1e-04 | 8.219e-03 | — |

Saliency spans nearly three orders of magnitude and plasticity about three across
the modules that have it, so both discriminate. The zeros are one module, below.

## The one module with no signal, and what it costs

`model.embed_tokens` has `grad_norm_count: 0`. Its forward hook fired — saliency is
0.169 — but nothing ever updated its gradient statistics, and it is also the one
module missing from the channel-moment sidecar (`modules: 129` against 130 tracked).

**Both absences are by design, and for the same reason.** Under `outer_exact` the
base-weight gradient is reconstructed as `∇W = δxᵀ` from a forward/backward hook
pair. An `nn.Embedding` has no such outer product: its input is a token id, not a
feature vector, and its gradient is a scatter-add into rows. The channel moments are
skipped for exactly the same transposition, and `signals/tracker.py` says so
directly — *"the honest thing is to leave it unestimated. When the embedding is tied
to an `lm_head` … the head's moments describe the same tensor in the right
orientation and the estimator uses those."* So the cardinal sensitivity path is
already covered: it reads `lm_head`'s moments for this tensor.

The ranking scorer is not covered, and there are **three independent reasons** this
one tensor scores neutrally:

1. It has no plasticity signal of its own, so `score_modules` routes it to
   `unexercised` and both its ranks become `NEUTRAL_RANK`.
2. That guard discards its *measured* saliency along with the missing plasticity.
3. Even if it had both, it is the **only member of its role group** — role sizes are
   `{embedding: 1, attn.qkv: 32, attn.o: 32, mlp.gate_up: 32, mlp.down: 32}` — and a
   percentile rank over one element is 0.5 by construction. Per-role ranking exists
   to stop 18k experts drowning attention (plan P4); a singleton group is its blind
   spot.

This is worth chasing rather than noting because of the size of what it decides.
The tensor is **614.6 M parameters, 16.0% of the model**, it carries `lm_head`'s
8-bit floor through the tie, and at a 3.25-bit target the allocator cuts it to
3 bits — **paying 67.8% of the model's entire 4.53 Gbit floor shortfall with this one
tensor**.

### Does the neutral score change that width?

Two measurements, because one is not enough.

**Bracket** — force the score to 0.0 and to 1.0, which spans every width any signal
could buy it:

| target | score 0.0 | score 1.0 | open? |
|---|---|---|---|
| 3.25 | 2 b | 4 b | **yes** |
| 4.0 | 3 b | 4 b | **yes** |
| 4.25 | 4 b | 4 b | no |
| 4.5 | 4 b | 4 b | no |

So the neutral score is *not* structurally harmless: at S3's two targets this
tensor's width is score-responsive across a 2-bit range.

**Realistic counterfactual** — substitute `lm_head`'s stats row, which describes the
same tensor, and rank globally (the only comparison a group of one admits):

> `model.embed_tokens` borrowing `lm_head` scores **0.9264** — near the top of the
> map, on the highest saliency in the model — and **changes the width at no target**.

The step in the response sits above 0.93. A real measurement lands on 3 bits, which
is where the neutral 0.5 already put it.

### Correction, 2026-08-06: that was read off one width

The line above is true and the conclusion drawn from it was not. This report
concluded *"the gap costs this checkpoint nothing"*, on the strength of one tensor's
width holding at four targets. The budget is shared. A score that changes where a
614 M-parameter tensor sits in the ROI order changes how much budget reaches
everything ranked below it, whether or not that tensor itself moves — and it did:

| target | `model.embed_tokens` | other modules moved | bytes |
|---|---|---|---|
| 3.25 | 3 b → 3 b | **12** | 1 438 433 280 → 1 438 433 280 |
| 4.0 | 4 b → 4 b | **5** | 1 797 586 944 → 1 797 586 944 |
| 4.25 | 4 b → 4 b | **11** | 1 917 517 824 → 1 917 911 040 |
| 4.5 | 4 b → 4 b | **8** | 2 037 448 704 → 2 037 055 488 |

At matched bytes to within 0.02%, so this is a reallocation and not a budget change:
`down_proj` and `qkv_proj` layers trade bits among themselves because the neutral 0.5
put the embedding in the wrong place in the queue. Nothing was free; one width was
watched while five to twelve others moved.

The verifier now reports `other_modules_moved` alongside `changes_width` and
[`tests/test_verify_signal_map.py`](../../tests/test_verify_signal_map.py) pins this
exact case, so the narrow reading cannot be repeated. Whether the moved allocation is
*better* is an eval question, not one the allocator can answer about itself.

## Verdict for S3

The map is usable and nothing here blocks the campaign.

1. 128 of 129 quantizable tensors are scored on real measurements with signals that
   discriminate; the 129th is neutral for documented reasons and takes the same width
   with or without them.
2. **State the caveat rather than fixing it.** The neutral score does not cost this
   checkpoint the tied tensor's own width, and it does move 5–12 other tensors at
   matched bytes. It is luck of where 0.9264 falls, not a property of the design: a
   checkpoint whose tied tensor ranked at the very top would be handed a bit it did
   not get.
3. **Ministral-8B is a different case and must be re-checked when its map lands.**
   It is untied, so `embed_tokens` and `lm_head` are separate tensors in two
   singleton role groups — two neutral scores by reason 3 above, and `lm_head` will
   have its own gradient signal that per-role ranking then discards. Run this script
   on that arm before reading anything from it.
   → It landed, it is worse, and it is written up in
   [`phase3-s2-ministral-signal-map.md`](phase3-s2-ministral-signal-map.md): the head
   scores 0.9783 on its own measurements against the 0.5 its group of one hands it,
   and that is a whole bit on 6.7% of an 8B model at the headline target.
4. Read alongside [`phase3-s3-fused-floors.md`](phase3-s3-fused-floors.md): that the
   tied tensor absorbs two-thirds of the shortfall is *why* Phi's floors are
   unaffordable in the first place, and it compounds with the +0.21 b fusion
   surcharge measured there.
