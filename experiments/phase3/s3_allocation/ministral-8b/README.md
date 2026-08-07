# S3 allocation maps — Ministral-8B × Tulu-3

Built by [`scripts/run_s3_allocate.py`](../../../../scripts/run_s3_allocate.py) with
`--allocate-only`, from the S2 artifacts in
[`../../s2_runs/ministral-8b.tulu3/`](../../s2_runs/ministral-8b.tulu3/). Kept here
because the box they were built on is not a volume and a moments-priced map costs
about 1 h 45 m of CPU to rebuild — the stats file they derive from is cheap to keep,
the maps are not.

Both anchors are complete. The run's own record is [`arms.json`](arms.json); the two
preconditions the four-arm reading depends on are asserted in
[`tests/test_s3_allocation_arms.py`](../../../../tests/test_s3_allocation_arms.py)
against these files, so they cannot quietly stop holding.

## The 3.25-bit anchor

All four arms land on **exactly 3 257 925 632 bytes** and 3.2500 average bits, so every
comparison between them is assignment and not size.

| arm | allocator | `lm_head` | `model.embed_tokens` | floors breached | histogram |
|---|---|---|---|---|---|
| `rtn` | uniform | 3 b | 3 b | 182 | all 254 at 3 b |
| `rank3` | rank-product | 3 b | 3 b | 128 | 42 / 146 / 66 at 2 / 3 / 4 b |
| `shuf3` | sensitivity | 4 b | 2 b | 98 | 18 / 151 / 84 / 1 at 2 / 3 / 4 / 8 b |
| `dq3` | sensitivity | 4 b | 2 b | 104 | 16 / 158 / 80 at 2 / 3 / 4 b |

**The control ablates.** `shuf3` and `dq3` differ on **39 of 254** modules. That is the
property [`phase3-s3-null-control.md`](../../../../docs/reports/phase3-s3-null-control.md)
was written about: on the pre-fix run the two arms differed on 0 of 129 and the control
was a null by construction.

**It cannot ablate everything.** `permutation_within_role` is a fixed point on a role
group of one, so `lm_head` and `model.embed_tokens` — 13.4% of the model — are assigned
identically in both arms and are not among the 39. Whatever an S4 `shuf`-vs-`dq` gap
turns out to be, it is a statement about the other 252 modules.

**Which allocator reads the score.** `rank3` is the only arm here whose widths come from
the percentile score for every module. `shuf3` and `dq3` price from measured sensitivity
wherever the channel moments cover a module, which on this checkpoint is 253 of 254 —
everything but `model.embed_tokens`. So that one tensor is where a neutral score reaches
the headline arm, and it is the reason `dq3` puts it at 2 b.

## The 4.25-bit anchor

All four arms land on **exactly 4 260 364 288 bytes** and 4.2500 average bits. The
allocator reports a widest drift of **+0 B**, so this is the same exact match the 3.25
anchor got and not a tolerance being spent.

| arm | allocator | `lm_head` | `model.embed_tokens` | floors breached | histogram |
|---|---|---|---|---|---|
| `rtn` | uniform | 4 b | 4 b | 1 | all 254 at 4 b |
| `rank4` | rank-product | 8 b | 4 b | 0 | 43 / 209 / 2 at 3 / 4 / 8 b |
| `shuf4` | sensitivity | 8 b | 4 b | 0 | 62 / 158 / 34 at 3 / 4 / 8 b |
| `dq4` | sensitivity | 8 b | 4 b | 0 | 62 / 155 / 37 at 3 / 4 / 8 b |

**The control ablates here too**, on **28 of 254** modules — fewer than the 39 at 3.25,
which is what a looser budget should do to the size of any assignment difference. `rank4`
and `dq4` differ on 56.

**The budget affords every floor here**, demonstrated by all three allocating arms
reaching 0 violations. That is the condition phase 2 found the signal does *not* pay
under, and it is the reason this anchor is a control on the regime rather than a second
headline: with the floors bought, an allocator's only remaining job is to spend the
surplus.

`rtn4`'s one breach is the exception that shows what the allocation is worth. Uniform
means uniform, so RTN hands `lm_head` 4 bits against its 8-bit floor � not because the
budget could not afford the other 4, but because RTN has no mechanism for spending
unevenly. Every allocating arm buys that floor back inside the same 4 260 364 288 bytes.

**Neither singleton is handicapped here.** `lm_head` takes its 8-bit floor on all three
signal arms and `model.embed_tokens` its 4-bit floor, so the neutral 0.5 that costs
`rank3` a bit at 3.25 costs nothing at 4.25 — it has nothing to lose to, because the
floors are affordable. The same fact read from the other side: the singleton defect in
`score_modules` is only visible in a regime where the allocator is forced to choose.

**Where the two allocators actually part.** `rank4` sends 2 modules to 8 b; `dq4` sends
**37**. Measured sensitivity finds a tail of modules worth widening that the percentile
score prices as ordinary, and it pays for them by dropping 62 modules to 3 b where the
rank-product map drops 43. Whether that trade is worth anything is an S4 question — this
file only establishes that the two arms are different maps at identical bytes.
