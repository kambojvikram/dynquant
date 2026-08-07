# S3 allocation maps — Ministral-8B × Tulu-3

Built by [`scripts/run_s3_allocate.py`](../../../../scripts/run_s3_allocate.py) with
`--allocate-only`, from the S2 artifacts in
[`../../s2_runs/ministral-8b.tulu3/`](../../s2_runs/ministral-8b.tulu3/). Kept here
because the box they were built on is not a volume and a moments-priced map costs
about 1 h 45 m of CPU to rebuild — the stats file they derive from is cheap to keep,
the maps are not.

**The 4-bit anchor is incomplete in this commit**: `shuf4` and `dq4` were still
building when these were pulled. `rtn`, `rank3`, `shuf3`, `dq3` and `rank4` are final.

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
