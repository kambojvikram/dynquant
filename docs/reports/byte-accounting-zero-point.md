# Every symmetric baseline was charged for a zero point it never stores

**Found 2026-08-15**, by the phase-2 asymmetric control refusing to print. Fixed in
[`experiments/_llmc.py`](../../experiments/_llmc.py) (`stored_meta_bits`), wired into both
baseline stages, measured against the format by
[`experiments/probe_zero_point_storage.py`](../../experiments/probe_zero_point_storage.py),
and guarded by six tests in
[`tests/test_stage8_baselines.py`](../../tests/test_stage8_baselines.py) and
[`tests/test_baselines_lfm2.py`](../../tests/test_baselines_lfm2.py).

## The defect

Both baseline stages priced group metadata with the same line:

```python
meta_bits = 16 + bits  # an fp16 scale, plus a zero point, per group of 128
```

charged to every arm. `compressed-tensors` stores a zero point **only when the grid is
asymmetric**. GPTQ and RTN are symmetric by default — `default_symmetric` has said so, in
this repository, for the whole campaign — so every GPTQ and RTN arm this project has
published was charged `bits / group_size` per weight for a tensor that is not in its
checkpoint: 0.023 bits at 3-bit g128, 0.031 at 4-bit, about 0.7% of the arm's width.

The direction matters more than the size. It makes the baseline look **more expensive than
it is**, on the axis every headline in this campaign is read against, which is the direction
that flatters DynQuant.

A second term was missing entirely: under `actorder=group` the format also writes
`weight_g_idx`, an `int32` per *input column*. That one is under-counted, and exactly one
shipped arm used it.

## How it was found, which is the part worth keeping

Not by review. The phase-2 replicate runs a control pair — `gptq_3b_head` fitted symmetric
against the same recipe fitted asymmetric, one flag apart — and the driver refuses to
print a table unless the two arms actually straddle the flag. It refused:

```
smoke_gptq_sym:  symmetric=True  actorder=None bits=3.1522 acc=0.28125
smoke_gptq_asym: symmetric=False actorder=None bits=3.1522 acc=0.0625
AssertionError: identical accounted bits: an asymmetric grid stores a zero point per group
and must cost more, so equal widths mean the flag did not reach the recipe
```

The assertion's stated hypothesis was wrong — the flag *had* reached the recipe, which is
why the accuracies differ by 22 points — and it fired anyway, because the thing it actually
tested was that the size column can see the scheme. It could not. Two arms 22 points apart
accounted to an identical 3.1522 bits, and without that guard the control would have
printed a clean-looking table in which the variable under control was free.

The general form: **a control that varies an axis needs every column to be able to see that
axis.** A blind column does not produce an obviously broken table. It produces a table whose
numbers are individually plausible and whose comparison measures nothing.

## Ground truth, measured rather than read

Reading `PackedQuantizationCompressor` establishes intent. The correction restates published
byte counts, so it is held to the file instead. `probe_zero_point_storage.py` quantizes one
tiny model twice — symmetric and asymmetric, everything else held — saves both with
`save_compressed=True` and lists what landed:

```
--- SYMMETRIC:  0 zero-point tensors, 186 scale tensors
--- ASYMMETRIC: 186 zero-point tensors, 186 scale tensors
      ...weight_zero_point  dtype=I32 shape=[2, 16]
      ...weight_scale       dtype=BF16 shape=[16, 16]
keys only in symmetric: []
```

Zero, not a smaller one. And the asymmetric zero point is `pack_to_int32(zp, num_bits)`
along dim 0, so its 256 groups occupy `2 x 16 x 32 = 1024` bits — exactly `groups x bits`,
not `groups x 16`. Both halves of the corrected arithmetic are measurements.

This is the ninth instance of the duplicated-registry failure this campaign has recorded,
and the second where the duplicated thing is a *dependency's* arithmetic — a copy nothing in
this repository can contradict, which is why the probe exists and why `stored_meta_bits` is
one function rather than two constants.

## Corrected numbers

Every affected record reconciles exactly with the old formula before correction, so these
are restatements of the same measurement under the right rule, not re-runs.

| panel | arm | scheme | recorded | corrected | delta |
|---|---|---|---|---|---|
| LFM2.5-8B-A1B | `gptq_4b` | symmetric | 4.1565 b / 4 399 629 312 B | **4.1253 b / 4 366 552 576 B** | −33 076 736 B (−0.752%) |
| LFM2.5-8B-A1B | `gptq_3b` | symmetric | 3.1488 b / 3 332 904 576 B | **3.1253 b / 3 308 097 024 B** | −24 807 552 B (−0.744%) |
| Mistral-7B-v0.3 | `gptq_4b` | symmetric | 4.3760 b / 3 964 674 048 B | **4.3453 b / 3 936 886 784 B** | −27 787 264 B (−0.701%) |
| Mistral-7B-v0.3 | `gptq_3b` | symmetric | 3.3869 b / 3 068 534 784 B | **3.3639 b / 3 047 694 336 B** | −20 840 448 B (−0.679%) |
| Mistral-7B-v0.3 | `gptq_3b_asym` | asym + actorder | 3.3869 b / 3 068 534 784 B | **3.3924 b / 3 073 531 904 B** | +4 997 120 B (**+0.163%**) |

Unchanged, because they were already right: every `awq_*` arm (AWQ is asymmetric here by
this repository's choice, and the zero point it was charged is one it stores), and
`gptq_3b_asym_noao` on both panels.

The act-order row is the only one that moves *up*. Its `weight_g_idx` is `32 x 1 249 280`
bits on this architecture — derived from a per-tensor shape list that reproduces the
record's own `quantized_params` of 7 113 539 584 to the unit, which is what makes the
in-features sum trustworthy.

## What this does to the byte-matched claims

`anchor_bytes` computed **one** budget per width and matched all three kinds to it. Under
the correction there is no single budget per width: a symmetric arm and an asymmetric arm at
the same nominal width are genuinely different sizes. Every DynQuant arm in phase 4 was
therefore sized against the asymmetric figure and compared against a symmetric GPTQ arm:

| panel | arm | DynQuant bytes | true symmetric GPTQ | gap | panel tolerance |
|---|---|---|---|---|---|
| LFM2.5-8B-A1B | `dq_4b` | 4 397 666 304 | 4 366 552 576 | **+0.713%** | 0.1% |
| LFM2.5-8B-A1B | `dq_3b` | 3 331 526 656 | 3 308 097 024 | **+0.708%** | 0.1% |
| Mistral-7B-v0.3 | `dq_4b` | 3 964 149 760 | 3 936 886 784 | **+0.693%** | 0.1% |
| Mistral-7B-v0.3 | `dq_3b` | 3 067 617 280 | 3 047 694 336 | **+0.654%** | 0.1% |

So, stated plainly: **DynQuant-against-GPTQ was not byte-matched on any phase-4 panel.** The
gap is 6.5x to 7.1x the panel's own stated tolerance, and it runs in DynQuant's favour every
time. It is small in absolute terms — 0.7% of a checkpoint — and "small" is not the standard
the panel set for itself; 0.1% is, and these fail it.

**DynQuant-against-AWQ is unaffected and remains byte-matched**, which is worth stating
because it means every phase-4 panel still carries one external baseline the comparison is
honest against. Where the report's claim is against AWQ, nothing here touches it.

Two specific sentences are now false and are corrected at their source:

* `phase4-text2sql-mixture.md` said the asymmetric control landed "**byte-identical to the
  symmetric arm**, so the arm prices the scheme and nothing else". It is not byte-identical
  to it; it is 24 807 552 B larger. The arm still prices the scheme — that was the point —
  but the *size column could not see the price*, so "and nothing else" was measuring a
  weight difference while silently holding a byte difference constant that was not constant.
* `phase4-packed-moe-runtime.md` offered `gptq_4b` and `awq_4b` agreeing "to the byte on
  4.1565 bits, 4 399 629 312 bytes" as evidence that the two arms differ only in what the
  recipe did to the weights. They do not agree to the byte: the symmetric arm is
  4 366 552 576 B. The conclusion that argument reached — that there is no second kernel
  involved, only asymmetric-against-symmetric dequantization — survives, because both arms
  really are 4-bit g128 through the same `compressed-tensors` path. The stated evidence for
  it does not.

## What this does not change

* **No accuracy figure moves.** Nothing was re-quantized and nothing was re-scored; the
  weights in every arm are the weights that were measured. This is a denominator correction.
* **No DynQuant arm's own size moves.** DynQuant's accounting is its own and never had this
  term; `p2_rowbody` landing on 5 664 702 464 stored bits is unaffected.
* **The AWQ comparisons stand**, as above.
* **The phase-2 headline is still held.** It was already held pending a control that its
  checkpoint no longer permits; this finding neither settles nor worsens it, though it does
  mean the replicate's GPTQ arm will come in ~0.7% cheaper than the destroyed panel's did.

## Fixed

`stored_meta_bits` in `_llmc.py` is now the single definition, taking `symmetric` and
`actorder` and returning a per-tensor total. Both stages call it; neither keeps a local
rate. `--symmetric auto` is resolved **once** per run into `resolved_symmetric` and read by
both the arm record and the size column, because the record saying `symmetric: false` while
the accounting priced `true` is precisely the split that produced this. `do_plan` now
reports both schemes at each width instead of one number for whichever method it silently
assumed.

Six tests. The load-bearing one asserts that the two schemes **cannot** account to the same
width at any of 2/3/4/8 bits — a property, not a value, because the broken version satisfied
every equality anyone would have thought to write. One asserts the literal `meta_bits` is
absent from both stages' source, anchored so `stored_meta_bits` does not satisfy its own
prohibition (the first draft passed for exactly that reason). One prices `weight_g_idx`
before an arm is priced by it rather than after.

## Still open

* The phase-4 panels are not re-run. Re-running them at the corrected symmetric anchor is
  the remedy for the byte-match failure; until then the GPTQ comparison on those panels
  should be cited with the gap stated, and the AWQ comparison preferred where one is needed.
* `stage8_bnb.py` has a third copy of `accounted_bytes` for NF4. NF4 is not a
  `compressed-tensors` scheme and does not take this term, so it is untouched here — but it
  is a third copy, and it has not been audited against bitsandbytes' own storage.
