# The grouped MoE kernel, timed and sanitized on one box

`docs/reports/kernel-first-compile.md` closed with two things it would not claim: that the
grouped kernel had never been timed against the Python loop it replaces, and that
`compute-sanitizer` had not been run on any build containing it. This directory is what retires
both, from a single uniform run of `../../bench_grouped_gemv.py` on an idle L4.

Every number below is read out of `bench.json`, which is self-describing -- device, capability,
torch build, dtype, iteration count, whether `torch._grouped_mm` was available, and the launch
and tolerance totals sit beside the rows rather than in prose that can drift from them.

## The box

`NVIDIA L4`, capability `8.9`, torch `2.11.0+cu126`, `dynquant-kernels 0.4.0` compiled on the
same box. `torch.float16`. Median of 50 timed iterations after 3 warmups, CUDA events around
each. Sole tenant: nothing else held the device for the length of the run.

## Provenance

Three earlier sweeps were killed and thrown away rather than merged, so that every row here comes
from one file at one revision. The file that ran is the file that is committed:

```
c6792913251e97acb01f530cdd0156412c27cdde5991287d81d5a69df1f23eb9  bench_grouped_gemv.py
```

-- the same sha256 for `/workspace/bench_grouped_gemv.py` on the L4 and for
`../../bench_grouped_gemv.py` here. The clone the benchmark imports from carries the
`runtime/linear.py` fix described below, so `loop_ms` is the loop as it ships today and not as it
shipped when the sweep was written.

## Correctness first, over the whole sweep

3 bank geometries x 3 widths x 6 row counts x 2 segment shapes = **108 configurations, 108
grouped launches, worst tolerance spent 0.0362** against the parity suite's own `atol=rtol=2e-2`.
The two segment shapes are `even` (tokens spread across every expert) and `skewed` (all tokens to
one expert), which is the load-imbalance case P8's gate names. Worst by band: `even` 0.0323,
`skewed` 0.0362 -- the imbalanced case is no worse than the balanced one in any width.

Timing runs on `even` bands only, so 54 of the 108 configurations carry times.

## Four denominators, because a speedup is a ratio

| column | what it is | why it is here |
|---|---|---|
| `loop_ms` | the shipped per-expert loop | the thing the kernel actually replaces |
| `loop_ref_ms` | the same loop forced onto `QuantTensor.dequantize` | what the loop cost before the fix below |
| `dense_ms` | the same loop over an already-dense bank | separates "we skipped dequantizing" from "the kernel is faster" |
| `grouped_mm_ms` | `torch._grouped_mm` on dense fp16 | the reference that is not our code at all |

`dequant_ms` is timed once per bank -- the cost of making `dense_ms` possible -- so the prefill
question is `fused_ms` against `dequant_ms + dense_ms` rather than against a bank somebody
dequantized for free.

`grouped_mm` reports **available** on this L4 at fp16, so the fp16 column is a measurement and
not a skipped row.

## Decode: one to four tokens per expert

`rows` is tokens entering the bank. `rows=4` on a 32-expert bank is the regime the kernel
exists for -- most experts receive nothing, the ones that fire receive one or two tokens.

| bank | bits | fused ms | vs loop | vs pre-fix loop | vs dense | vs fp16 `_grouped_mm` |
|---|---:|---:|---:|---:|---:|---:|
| `32x1024x2048` | 3 | 0.0481 | 9.9x | 31x | 4.8x | 3.5x |
| `32x1024x2048` | 4 | 0.0532 | 8.6x | 31x | 4.4x | 3.2x |
| `32x1024x2048` | 8 | 0.0594 | 7.7x | 29x | 4.4x | 3.2x |
| `64x512x2048` | 3 | 0.0408 | 11.2x | 32x | 5.7x | 6.3x |
| `64x512x2048` | 4 | 0.0410 | 11.0x | 31x | 5.7x | 6.2x |
| `64x512x2048` | 8 | 0.0450 | 10.2x | 29x | 5.2x | 5.7x |
| `8x2048x4096` | 3 | 0.0911 | 15.9x | 209x | 4.0x | 3.6x |
| `8x2048x4096` | 4 | 0.1034 | 5.6x | 185x | 3.5x | 3.2x |
| `8x2048x4096` | 8 | 0.1178 | 4.0x | 167x | 3.1x | 2.8x |

P8's gate reads *grouped path beats the per-expert loop by >=3x at decode*. Against the loop
as it ships today that is **4.0x-15.9x**, so the gate is met on the denominator that is
hardest to argue with. Against the loop as it shipped last week it is 29x-209x, and the
difference between those two columns is a defect in the baseline, not a property of grouping.

Two further readings, which are different claims and worth separating:

- **`vs dense` is 3.1x-5.7x.** The kernel beats the same loop when the bank arrives already
  dense and the dequantization costs nothing. So the win is not "we skipped dequantizing" --
  one launch over segments beats E launches over slices on its own.
- **The quantized kernel is 2.8x-6.3x faster than dense fp16 `torch._grouped_mm`.** Decode
  is bandwidth-bound, a 3-bit weight is a smaller read, and a tensor-core GEMM has no use for
  its tensor cores at four rows.

## Where it stops winning, which is the argument for P7

`vs loop` crosses 1.0 as `rows` grows, and where it crosses is a property of the geometry:

| bank | 3-bit | 4-bit | 8-bit |
|---|---:|---:|---:|
| `32x1024x2048` | 512 | 2048 | 512 |
| `64x512x2048` | 2048 | 2048 | 2048 |
| `8x2048x4096` | 128 | 128 | 128 |

Read as: ahead up to that many rows, behind past it. On the widest bank the fused kernel is
already behind at 512 rows and loses by **8.3x** at 2048 (27.6 ms against 3.4) -- at which
point the right move is to dequantize the bank once and call a dense grouped matmul. That is
not a defect. A GEMV at 2048 rows is the wrong kernel, and this table is the prefill split
stated in measured milliseconds instead of in principle: it is what P7's tensor-core GEMM is
for, and the crossover row is where a dispatcher should switch between them.

## The 3-bit path costs about twice the 4-bit path once `rows` is large

| bank | 3-bit ms | 4-bit ms | ratio |
|---|---:|---:|---:|
| `32x1024x2048` | 5.5460 | 2.8620 | 1.94x |
| `64x512x2048` | 2.5342 | 1.3005 | 1.95x |
| `8x2048x4096` | 27.6439 | 11.3725 | 2.43x |

Reading *fewer* bytes and taking *more* time is the unpacking, not the memory. Three bits is
the one width that does not divide 32, so 32 values come out of 3 words with cross-word
shifts, while 4 and 8 bits are constant-shift extractions from a single word. It is invisible
at decode, where the read dominates, and it dominates once the read is amortised. This is what
a vectorized grouped variant -- still unwritten -- would go after.

## Sanitizers, in `sanitizer/`

The same script hands the same benchmark to each tool in turn, so what is inspected is the
workload timed above. The sanitizer passes omit `--time` and run the correctness sweep only,
which is the `grouped_mm=not probed without --time` line in each log.

| tool | over the 108-launch workload | elapsed | over the grouped parity suite | elapsed |
|---|---|---:|---|---:|
| `memcheck` | `ERROR SUMMARY: 0 errors` | 104 s | 23 passed, `0 errors` | 13 s |
| `initcheck` | `ERROR SUMMARY: 0 errors` | 110 s | -- | -- |
| `synccheck` | `ERROR SUMMARY: 0 errors` | 33 s | -- | -- |
| `racecheck` | `0 hazards displayed (0 errors, 0 warnings)` | 2348 s | 23 passed, `0 hazards` | 14 s |

Every workload row carries `configurations=108 launches=108 worst_ratio=0.0362` in its own log,
so each tool saw the whole sweep and instrumentation did not change the numerical result.

`racecheck` cost 22.6x what `memcheck` did over identical work. An earlier attempt was killed at
fifteen minutes on the assumption that it had hung; it had not, and it needed thirty-nine.

### Why `--target-processes all`, demonstrated rather than asserted

`compute-sanitizer` defaults to `--target-processes application-only`, which instruments the
process it launches and nothing that process spawns. The parity suite spawns a subprocess, so
under the default the number of grouped launches that had ever been instrumented was **one**, not
the suite's twenty-three. `sanpos.py` is a deliberately out-of-bounds `torch.index_select` and
`sanpos_sub.py` is the same thing one `subprocess.run` away:

| | invocation | what the tool said |
|---|---|---|
| A | direct, `--target-processes all` | `ERROR SUMMARY: 3 errors` |
| B | one subprocess away, `--target-processes application-only` | **no `ERROR SUMMARY` at all** |
| C | one subprocess away, `--target-processes all` | `ERROR SUMMARY: 3 errors` |

A is the control on the control: a definitely-wrong kernel has to be caught, or every clean report
here is worth nothing. B is the failure mode, verbatim in `sanpos-B-verbatim.txt` -- it does not
print a reassuring zero, it prints

```
========= Error: Target application terminated before first instrumented API call
========= Tracking kernels launched by child processes requires the --target-processes all option.
```

which reads as tooling noise beside a green pytest line, and the missing `ERROR SUMMARY` reads as
nothing rather than as a hole. The workload benchmark launches its 108 grouped kernels in the
process the sanitizer started, which is why it, and not the suite, is the vehicle for the table.

## The defect the benchmark found before it could overstate anything

The first version of this sweep timed the fused kernel against the loop and reported a number in
the 29x-209x range -- `loop_ref_ms` is that same denominator, re-measured in this record rather
than quoted from the run that was thrown away. It is arithmetically true and it was not the
number to publish, because most of it was not about grouping at all.

`DynQuantExpertBank.__getitem__` called `self.weight_qt.rows(...).dequantize(...)`.
`QuantTensor.dequantize` says in its own docstring that it is the *reference* implementation: it
unpacks, casts to fp32, multiplies, adds, reshapes and slices -- five elementwise passes moving
about 1.3 GB for a 67M-element bank. Meanwhile `runtime/ops.dequantize` dispatches exactly that
arithmetic to one CUDA kernel when the extension is loaded, and falls through to the same
reference when it is not. The bank never called it, so on a box with the kernels installed the
loop paid the pure-torch price anyway.

Measured across the 54 timed configurations, that gap is:

| | min | median | max |
|---|---:|---:|---:|
| `loop_ref_ms / loop_ms` | 2.81x | 4.01x | 46.13x |

It composes exactly as a product: on `8x2048x4096` at 3 bits and 4 rows the fused kernel is
15.9x the repaired loop and the repaired loop is 13.1x the reference one, and 15.9 x 13.1 is the
209x the first draft would have published as a kernel result.

`loop_ref_ms` is not a rewrite of the loop. It is the shipped loop with `__getitem__`
monkeypatched to `_bank_getitem_via_reference`, so the only thing differing between the two
columns is which dequantizer `bank[e]` reaches. And the swap is checked rather than assumed:
`loop_ref_spent` records how far the patched loop's output moves from the shipped loop's, as a
fraction of the same `atol + rtol*|expected|` the parity suite uses. It is **0.0000 in all 54** --
not "within tolerance", identical, because both paths accumulate in fp32 and round once.

The fix is two lines in `runtime/linear.py`: index through `ops.dequantize`. It costs nothing on
a backend that had no kernel to begin with, since that is the fallback. It is guarded by
`test_indexing_a_bank_goes_through_the_dispatcher_not_the_reference`, which asserts the *call*
and not the output -- on CPU the two paths are numerically indistinguishable, so a test comparing
numbers would have stayed green through a revert.

The column stayed in the sweep after the fix landed, with its sense inverted. It is now the price
of the `ops` hop rather than a historical note, so a revert shows up here as `loop_ms` climbing
to meet `loop_ref_ms` rather than as a speedup quietly improving.

The general shape is worth keeping: **a benchmark's denominator is a claim about your own code,
and it deserves to be read as carefully as the numerator.**
