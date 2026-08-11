# S3 methods: the shuffled control was ablating a signal the allocator no longer reads

**Found 2026-08-06**, before S3's arms were quantized and before any GPU time was spent on
them. Driver: [`scripts/run_s3_allocate.py`](../../scripts/run_s3_allocate.py). Fixed in
`6bf6713`; guarded by five tests in
[`tests/test_run_s3_allocate.py`](../../tests/test_run_s3_allocate.py).

## What the arm was for

S3 exists to separate three explanations for a DynQuant win: a bigger file, the allocator's
structure, or the signal. Four arms at one byte count do that:

| arm | allocated from | isolates |
|---|---|---|
| `rtn` | nothing — uniform width | the file size |
| `rank` | the real stats, no channel moments | the score, without the cardinal estimate |
| `shuf` | the stats **permuted within role**, plus moments | — |
| `dq` | the real stats plus the real moments | the full method |

`dq` − `shuf` is the number the campaign is built around: same score *distribution*, same
role structure, same byte count, correspondence to modules destroyed. Whatever that gap is,
it is the signal and nothing else.

## Why it measured nothing

`_Candidate.move_value`
([`knapsack.py:169-183`](../../packages/dynquant-core/src/dynquant/allocate/knapsack.py#L169))
prices a width change from the *measured* sensitivity table whenever the module has one, and
falls back to the stats-derived score only when it does not:

```python
if self.sens is not None:
    lo, hi = self.sens.get(lower), self.sens.get(upper)
    if lo is not None and hi is not None:
        return max(lo - hi, 0.0)
proxy = self.score * self.num_params * (_error_scale(lower) - _error_scale(upper))
```

`sens` comes from the channel moments. Phi-4-mini's sidecar holds 258 tensors — 129
`input_sq` and 129 `output_grad_sq` — against a graph with exactly **129 quantizable
modules**. Coverage is total, so `self.sens is not None` for every candidate the knapsack
ever considers, and the permuted score is never read.

The driver was passing `--moments` the real, unpermuted sidecar to both arms. The command it
actually emitted, taken from the validation log rather than from the source:

```
python -m dynquant inspect .../merged --group-size 128
  --save-map .../map.shuf3.json --json
  --stats .../dynquant_stats.shuffled-0.json          # permuted
  --target-size 1558302720
  --moments .../s2/phi4-mini.tulu3/stats/dynquant_moments.safetensors   # not permuted
```

So `shuf` was allocating from the same table as `dq`, arriving at the same map, and would
have reported `dq − shuf ≈ 0` — read as *the signal does not matter* — on the strength of an
ablation that never happened. The failure is silent in both directions: nothing errors, both
arms complete, and the two maps agree to the bit.

### Confirmed on the pre-fix run

The four-arm `--allocate-only` validation was already in flight when this was found, so it
was left to finish and the maps compared. Phi-4-mini × Tulu-3, 129 quantizable modules, all
three arms held to the 3-bit uniform anchor's 1 558 302 720-byte ceiling:

| pair | modules at a different width | bytes | average bits | floors breached |
|---|---|---|---|---|
| `shuf3` vs `dq3` | **0 / 129** | identical, 1 558 056 960 | identical, 3.249487442337263 | 88 vs 88 |
| `rank3` vs `dq3` | 65 / 129 | — | — | 79 vs 88 |

The two maps hash identically. Not "close" — the same allocation, because the same table
produced it. `rank3` is the contrast that shows the machinery works: it allocates from the
stats score with no moments at all, and it moves half the model.

`_fallback_scale` makes the prediction exact rather than approximate. It assigns
`fallback_scale` only to modules *without* a measured sensitivity, and with all 129 covered
that set is empty, so no permuted quantity reaches the objective by any path.

### Confirmed again after the fix

The same four arms re-allocated at `d74059a`, same model, same anchor, same ceiling:

| pair | modules at a different width | bytes | average bits | floors breached |
|---|---|---|---|---|
| `shuf3` vs `dq3` — **before** | 0 / 129 | identical | identical | 88 vs 88 |
| `shuf3` vs `dq3` — **after** | **55 / 129** | 1 557 270 528 vs 1 558 056 960 | 3.2478 vs 3.2495 | 88 vs 88 |
| `rank3` vs `dq3` | 65 / 129 | 1 558 302 720 | 3.25 | 79 vs 88 |

The permutation moved 122 of 130 entries and the allocation moved 55 of 129 modules — the
arm is now ablating something. Two properties of the after-row are worth more than the
count:

* `dq3` hashes to `1665658612e607a6` in **both** runs. The treatment is byte-identical
  before and after, so the fix moved the control and nothing else. Had `dq3` moved, the fix
  would have been a change to the method wearing a control's name.
* Both arms breach **88** floors, against `rank3`'s 79. Floor pressure is a property of the
  budget and the role structure, which the permutation preserves by construction; that the
  two agree here and disagree with the moments-free arm is the grouping doing its job.

The width histograms differ by one module — `dq3` {2: 41, 3: 57, 4: 31} against `shuf3`
{2: 42, 3: 56, 4: 31}. So the two arms spend the budget on nearly the same *shape* of
allocation and disagree about *which* modules get which width, which is exactly the
difference a correspondence ablation should produce and not, say, the difference a
distribution ablation would.

What this does not yet say is what the gap is *worth*. 55 modules moving is a statement
about the map, not about accuracy; the quantize-and-eval arms are what turn it into points,
and they need the GPU that Ministral currently holds.

This is the same failure mode as -- but not the same control as -- the phase-2 finding that the
25.78-point margin at 3.25 b splits 22.62 allocator / 3.16 signal. Phase 2 did not permute
anything: `stage4_allocate.py` replaces every score with the constant 0.5, over an allocator that
predates the moments path and therefore reads a score and nothing else. That is why it was
trustworthy -- its control emptied the allocator's only input, not because it permuted -- and it
is why it measures a different quantity from `shuf` here, which is worth stating wherever the two
percentages are compared. What the two share is the dependency: a control is only an ablation of
what the allocator reads *at the time*, and adding the Gauss-Newton estimator moved what the
allocator reads without moving the control.

## The fix

One permutation, applied to every artifact the allocator consumes.

* `permutation_within_role(stats, seed, *, moments=None)` returns `target -> donor` and is the
  single source of correspondence for both files. Permuting stats and moments independently
  would give a module one donor's scalars and another's channel vectors — a third
  distribution belonging to no module at all, which is exactly the failure the existing
  multiset test rules out *within* the stats file and could not see across two.
* Grouping is by `(role, len(input_sq), len(output_grad_sq))` rather than by role alone. On a
  dense model this is a no-op — every member of a role has the same geometry — and it exists
  because nothing guarantees that. A channel vector of the wrong length does not raise where
  it is consumed: `_shapes_agree` rejects the pair, `_moments_for` returns `None`, and the
  module quietly leaves the sensitivity table to be priced from the proxy. The control would
  then be ablating *which modules can be measured at all*, which is structural, on top of the
  correspondence it exists to ablate.
* `write_variants` now emits four files, a stats/moments pair per variant, and re-saves the
  *real* moments through the same writer so treatment and control differ in the permutation
  and in nothing else, not even in which writer produced the file they were parsed from.
* `allocate_arm` passes `variant["moments"]`, never `args.moments`. That was the wrong line,
  so that is the line
  `test_each_arm_is_allocated_from_its_own_moments` pins, at the argv the subprocess would
  have received.

## What this does not invalidate

The "69 of 129 modules move against a shuffled control at 3.25 b" figure in
[`phase3-s3-fused-floors.md`](phase3-s3-fused-floors.md) stands.
[`floor_headroom.py`](../../experiments/phase3/s3_allocation/floor_headroom.py) allocates from
a score dict directly and never loads moments, so its control permuted the only input its
allocation had. Phase 2's signal-share decomposition likewise predates the moments path.

Nothing downstream of S3 had run, so no reported number changes.

## The transferable part

A control is defined by what the treatment *reads*, not by what the treatment is *named
after*. The stats file is the artifact with "signal" in its name and the one every earlier
control permuted; the moments sidecar arrived later, quietly became the primary input, and
inherited none of that scrutiny. When an estimator is added, the ablation has to be re-derived
from the consumer, not carried forward from the previous design.

The cheap check that would have caught it at any point: **an ablation arm that produces a
bit-identical map to its treatment has not run.** The driver now has that comparison in the
four-arm `--allocate-only` validation, which is CPU-only and costs minutes.
