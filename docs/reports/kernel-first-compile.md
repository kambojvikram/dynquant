# The first compile: what the CUDA sources did when they finally ran

Two of this repository's four `.cu` files had been compiled and measured before — `gemv.cu` and
`dequant.cu` are where the packed runtime's bandwidth and VRAM numbers come from. One had not.
The closing sentence of the packed-MoE report is explicit about it:

> What is still not claimed: **`grouped_gemv.cu` has never been compiled**, there is no
> vectorized variant so a busy expert decodes at the general path's bandwidth, and no speedup
> exists yet — only a counter.

This report closes the first of those three. It builds the whole extension from source on a card
nothing in this project has used before, runs the grouped-MoE parity suite for the first time,
and then runs the rest of the kernel surface.

**Result.** The extension builds clean for `sm_89` and the grouped kernel is correct: **23 of 23**
grouped parity cases pass on real silicon at first run. The full surface came back **653 passed,
1 failed** — and the one failure was in the CPU *reference*, not in any kernel. It was a real
defect, present in two copies, which put an fp16 rounding on every weight element in front of a
sum over `in_features` of them. Measured as the fraction of the parity tolerance each path
actually spends, over every geometry × width × row count:

| | CPU worst | CUDA worst |
|---|---|---|
| before | **1.041** | 0.043 |
| after | **0.000** | 0.043 |

The CUDA column is byte-identical on both sides of the fix, which is the direct evidence that the
defect was never in the kernel. After the fix the surface is **656 passed, 0 failed**.

| | date | commit |
|---|---|---|
| build and first run | 2026-08-15 | `034dae2` + two working-tree files |
| after the fix | 2026-08-15 | `8d24f55` |

---

## 1. Hardware and software

One card, one environment, for the build and every arm below.

| | |
|---|---|
| GPU | NVIDIA L4 (compute capability 8.9) |
| host | vast.ai instance 46418347 |
| toolkit | system `nvcc` 12.4.131 (`cuda_12.4.r12.4`) |
| torch | 2.11.0+cu126 |
| python | 3.12 |
| built architectures | `89-real` |

The toolkit is deliberately the *system* one. A pip-installed `nvidia-cuda-nvcc` reaches its own
headers through `-isystem`, ahead of the system toolkit, and those headers can be stamped with a
different version than the compiler reports — which produces a mismatch that looks like a source
problem and is not. Using the system `nvcc` removes the question rather than answering it.

Two environment guards fired before anything compiled, and both are worth recording because
neither is a property of this repository:

- **`/venv/omni` has no `pip`.** `python -m pip` is `No module named pip`. Installed through
  `uv pip install --python /venv/omni/bin/python` instead.
- **System `cmake` is 3.22.1, and `CMakeLists.txt` requires 3.26.** The build's own
  `cmake.version = ">=3.26"` constraint is what caught it. Installed cmake 4.4.2 into the venv.

## 2. What actually compiled

Seven targets, four of them CUDA, and the third one in the list is the one that had never been
built anywhere:

```
[1/7] Building CXX object  csrc/probe/probe_cublaslt.cpp.o
[2/7] Building CXX object  csrc/bindings.cpp.o
[3/7] Building CUDA object csrc/quant/dequant.cu.o
[4/7] Building CUDA object csrc/probe/probe_cuda.cu.o
[5/7] Building CUDA object csrc/moe/grouped_gemv.cu.o
[6/7] Building CUDA object csrc/gemv/gemv.cu.o
[7/7] Linking CXX shared module _C.cpython-312-x86_64-linux-gnu.so
```

`-- DynQuant kernel ABI version 3` at configure time, and the installed module reports
`ABI_VERSION == 3`, which is what `dynquant-core` requires. Output:
`dynquant_kernels-0.4.0-cp312-cp312-linux_x86_64.whl`. No warnings from the DynQuant sources; the
only CMake warnings are torch's own (`kineto not found`, and torch objecting to
`CMAKE_CUDA_ARCHITECTURES` in favour of `TORCH_CUDA_ARCH_LIST` — it still emitted
`-gencode arch=compute_89,code=sm_89`, so the request was honoured).

## 3. The grouped MoE kernel, first run

`moe_grouped_gemv` is present in the op table and **23 passed, 586 deselected in 10.13s**. Every
case compares the CUDA kernel against `QuantTensor.dequantize()` per expert band, so a kernel that
read the wrong band, or read the right band at the wrong offset, fails rather than returns
plausible numbers — which is the failure mode the sentinel-offset bug in the Python loop had, and
the reason that suite exists in the shape it does.

That is a correctness result and nothing else. There is still **no speedup measurement** for this
kernel, no vectorized variant, and no comparison against the Python indexing loop it is meant to
replace. Compiling it was the blocker; it is no longer the blocker.

## 4. The one failure, and why it was not a kernel

```
FAILED tests/test_kernels_parity.py::test_gemv_matches_dequant_then_matmul[65-3072-128-8-4-cpu]
  Mismatched elements: 1 / 260 (0.4%)
  Greatest absolute difference: 0.0255126953125 at index (1, 58) (up to 0.02 allowed)
  Greatest relative difference: 0.11334228515625 (up to 0.02 allowed)
```

CPU, not CUDA. The widest geometry in the list (`65×3072`), at the widest bit width (8), with four
rows. One element in 260.

`gemv_cpu` built its answer by materialising the dequantized weight through the `dequant` op and
multiplying. `dequant` dispatches on the **scales'** dtype and rounds its store to that dtype, so
with fp16 scales the reference rounded every weight element to fp16, upcast, and then summed
`in_features` of them. That error accumulates like `sqrt(K)` against a tolerance that is flat, so
the reference gets less accurate as the matrix gets wider.

The kernel has no such error. `gemv_kernel` never materialises the weight: it accumulates in
`float acc[...]` and applies scale and offset in fp32 via
`fmaf(scale[r], qacc[r][m], fmaf(offset[r], xsum[m], acc[r][m]))`. Its own header already says
so — *"It is also the more accurate form: the reconstructed weight is never materialised."* So
the test was holding a less accurate reference against a more accurate kernel and calling the
difference the kernel's error.

`moe_grouped_gemv_cpu` had the same line. Both copies are fixed in the same change.

## 5. The measurement that settles it

`torch.testing.assert_close` passes when `|d| <= atol + rtol*|expected|`, so counting how many
elements exceed `atol` alone cannot tell a large deviation on a large output element from a real
breach. (A first pass at this probe did exactly that and reported 20 of 20 CPU seeds failing where
pytest reported one failure in 654. That figure was wrong and is not quoted anywhere.) The right
instrument is the **ratio to the combined tolerance**,
`max |got - expected| / (atol + rtol*|expected|)`, which is `1.0` exactly at the pass/fail
boundary.

Swept over 10 geometries × 4 bit widths × 6 row counts × 2 devices, taking the worst ratio per
geometry over all four widths:

| `in_features` | geometry | group | CPU worst, before | CPU worst, after | CUDA worst |
|---|---|---|---|---|---|
| 4 | 7×4 | per-row | 0.032 | 0.000 | 0.000 |
| 100 | 48×100 | per-row | 0.273 | 0.000 | 0.021 |
| 128 | 16×128 | 32 | 0.177 | 0.000 | 0.000 |
| 256 | 33×256 | 128 | 0.301 | 0.000 | 0.032 |
| 320 | 64×320 | 128 | 0.379 | 0.000 | 0.038 |
| 512 | 128×512 | 128 | 0.521 | 0.000 | 0.032 |
| 2048 | 70×2048, 80×2048 | 128 | 0.825 | 0.000 | 0.043 |
| 3072 | 65×3072 | 128 | **1.041** | 0.000 | 0.040 |
| 4304 | 72×4304 | 128 | 0.978 | 0.000 | 0.038 |

The CPU column rises from 0.032 to about 1.0 across three orders of magnitude in `in_features`;
the CUDA column is flat at 0.02–0.04 above `K = 100` and does not move at all. That is the whole
argument in one table: **the margin was a function of the geometry on one device and not on the
other**, which is what an accumulation error in the reference looks like and is not what a wrong
kernel looks like.

Two entries are out of order and both are worth naming rather than smoothing. `16×128` at
`K = 128` reads *below* `48×100`, but it is the only case grouped at 32 rather than per-row or
128, so it is not the same experiment. And `72×4304` reads slightly below `65×3072` — these are
maxima over six row counts, so the ordering of any adjacent pair is luck. The growth is consistent
with `sqrt(K)` (`K = 256 → 3072` grows 3.46× against `sqrt(12) = 3.46`; `512 → 2048` grows 1.58×
against 2.00), but no single pair proves it. What is not luck is that it grows monotonically at
all on one device while the other stays flat.

## 6. The fix, and what was deliberately not changed

One helper, `dequant_reference_fp32`, used by both call sites: it widens the scales and offsets to
fp32 before calling `dequant_cpu`, which rounds its store to the scales' dtype and therefore now
stores fp32.

`dequant` itself still returns the storage dtype. Its own test asserts **bit-exact** equality
against `QuantTensor.dequantize()` in fp16, and that contract is the reason the format is
trustworthy; widening the op would break it. This change widens the *use* of the op inside one
reference, not the op.

The result is pinned by a new test rather than by the fixed number:
`test_gemv_margin_is_the_kernels_not_the_references` runs the geometry that spent the most budget
and asserts the spend is at most **0.25** of it. The reverted route measures 1.04 there, so the
test fails on the diff that reintroduces the defect — which is the only kind of regression guard
worth the runtime.

## 7. What the fix costs the test, stated plainly

After the fix the CPU ratio is `0.000` everywhere — not small, zero. That deserves suspicion, and
the reason is real but narrowing: `gemv_cpu` now computes `x.float() @ dequant_fp32(...).t()` and
the test's expectation is `x.float() @ QuantTensor.dequantize(float32).t()`, so the two share the
same `at::matmul` and differ only in which unpacker produced the weight. A zero therefore says
that **the hand-written C++ scalar unpacker in `dequant_cpu_rows` and the vectorized torch
unpacker in `QuantTensor.dequantize` agree bit-for-bit in fp32** — which is a genuine and fairly
strong statement about the packing layout, and a *weaker* statement than the test's name suggests
about the CPU gemv path itself.

So: on CPU, `test_gemv_matches_dequant_then_matmul` is now an unpacker-agreement test with the
matmul held in common. The kernel-against-reference question it is named for is answered on
**CUDA only**, where it reads 0.043 of the budget. That was already true before the fix — the CPU
"kernel" is an `at::matmul` either way — but the pre-fix number hid it behind a margin that looked
like measurement noise.

## 8. Why the four gates never caught it

`tests/test_kernels_parity.py` opens with `pytest.importorskip("dynquant_kernels")` and a
module-level `skipif` on `kernels.is_available()`. There is no compiled extension on the Windows
development machine, so the entire module is skipped at collection — and has been on every run of
the four gates this project has ever done. The local gate is still green after the fix (**2238
passed, 14 skipped**) and it was green before, on both versions of the defect.

That is the general shape and it is worth stating once: a test that skips is not a test that
passes, and a suite whose skip count is stable is not a suite that is covering more.

The parity test's docstring also claimed the tolerance margin was independent of geometry. That
was true of the CUDA path and false of the CPU one, and it is the sentence a future reader would
have used to dismiss exactly this failure as flakiness. It is corrected in place rather than
deleted, with the reason it was wrong kept next to it.

## 9. What is still not claimed

- **No performance number for the grouped kernel.** It is correct; it has not been timed, and the
  Python indexing loop it replaces has not been timed against it on the same box.
  **Superseded 2026-08-15.** Both are timed, on this box, in the same process, against four
  denominators rather than one -- see [section 10](#10-the-timing-that-closes-the-first-bullet)
  and `experiments/phase4/results/l4-sm89-kernels/`. At decode the grouped kernel is
  **4.0x-15.9x** the per-expert loop as it ships today, so P8's `>=3x` gate is met on the
  denominator that is hardest to argue with. It is 29x-209x against the loop as it stood when this
  bullet was written, and the gap between those two columns is the second thing the timing
  produced: a defect in the *baseline*. `DynQuantExpertBank.__getitem__` was reaching the pure
  torch reference dequantizer even with the kernels loaded, worth a median 4.0x, now fixed in
  `runtime/linear.py` and kept as a standing column of the sweep so a revert is visible. What
  still stands from the original bullet: nothing here is an end-to-end model measurement, and the
  numbers are one card.
- **No vectorized grouped variant.** A busy expert still decodes at the general path's bandwidth.
- **One architecture.** Built `89-real` only, on one L4. Nothing here says anything about `sm_80`,
  `sm_90a` or Blackwell, and no wheel from this build has been published.
- **No end-to-end model ran through the grouped kernel here.** That is the packed-MoE report's
  measurement, and it used the Python loop.
  **Superseded 2026-08-15.** `LiquidAI/LFM2.5-8B-A1B` decodes through `grouped_gemv.cu` at 3 and
  at 4 bits, coherently, in `experiments/phase4/results/l4-moe-end-to-end/` -- see
  [section 12](#12-the-end-to-end-run-that-closes-the-third-bullet). The model-level speedup over
  the per-expert loop is **1.95x at 4 bits and 2.58x at 3**, not the 8.55x the nearest swept
  geometry gives, and the difference is Amdahl rather than a shortfall: the expert banks are
  roughly half a decode step. Two things the run produced that the benchmark could not. The
  memory figure it nearly published was **67% pack-time workspace** -- peak over load-and-pack is
  7167.5 MiB where resident is 4295.7, and only the second is what a server pays; measured
  properly it lands **6.4 MiB** from the byte accounting, which is P6's VRAM gate. And
  `dynquant eval --map-apply pack` **cannot reach this path on this model family at all**: the
  router is in every map, by floor or by uniform width, and `pack_model` refuses it by class.
- **`compute-sanitizer` was not re-run on this build.** The 0-error, 0-hazard result recorded for
  the packed runtime predates `grouped_gemv.cu` ever being compiled, so it does not cover it.
  **Superseded 2026-08-15.** All four tools ran on this build over a 108-launch grouped workload
  and over the grouped parity suite -- see
  [section 11](#11-the-sanitizer-run-that-closes-the-last-bullet).
  Establishing that required a correction the first framing got wrong twice: the default
  `--target-processes application-only` does not follow the parity suite into the `subprocess` it
  spawns, so the number of grouped launches that had ever run under the tool was **one**, not the
  suite's twenty-three. The workload benchmark launches in-process, which is why it is the vehicle
  for this rather than the suite.

---

*Sections 10 and 11 were added on 2026-08-15, after the two bullets above were closed. They are
appended rather than folded in so the report still reads in the order the work happened: the
compile and its one defect first, then the measurement.*

## 10. The timing that closes the first bullet

The vehicle is `experiments/phase4/bench_grouped_gemv.py`, run on the same L4, sole tenant, fp16,
median of 50 iterations after 3 warmups with CUDA events around each. It sweeps 3 bank geometries
× 3 widths × 6 row counts × 2 segment shapes = **108 configurations**, checks every one against
the shipped per-expert loop first (**worst tolerance spent 0.0362** of the parity suite's own
`atol = rtol = 2e-2`), and times the 54 that use even segments.

### It took four denominators to get one honest number

The first version of this sweep timed the fused kernel against the loop and got a number in the
29×–209× range — `loop_ref_ms` below is that same denominator, re-measured here rather than quoted
from the discarded run. It is arithmetically true and it was not the number to publish, because
most of it was not about grouping. `DynQuantExpertBank.__getitem__` was reaching the pure-torch
*reference* dequantizer even on a box with the kernels loaded, which cost the loop a median 4.01×
it did not have to pay. That is fixed in `runtime/linear.py`, `loop_ms` below is the repaired
loop, and the old cost is kept as its own column rather than deleted:

| column | what it is |
|---|---|
| `loop_ms` | the shipped per-expert loop, as it stands after the fix |
| `loop_ref_ms` | the same loop with `__getitem__` forced back onto the torch reference — what it cost before |
| `dense_ms` | the same loop over a bank that is already dense — dequantization made free |
| `grouped_mm_ms` | `torch._grouped_mm` on dense fp16, which is not our code at all |

`torch._grouped_mm` reports **available** on this card at fp16, so the fp16 column is a
measurement rather than a skipped row.

### Decode, which is what a GEMV is for

`rows` is tokens entering the bank; `rows=4` on a 32-expert bank is the regime where most experts
receive nothing.

| bank | bits | fused ms | vs loop | vs pre-fix loop | vs dense | vs fp16 `_grouped_mm` |
|---|---:|---:|---:|---:|---:|---:|
| `32×1024×2048` | 3 | 0.0481 | 9.9× | 31.3× | 4.8× | 3.5× |
| `32×1024×2048` | 4 | 0.0532 | 8.6× | 31.0× | 4.4× | 3.2× |
| `32×1024×2048` | 8 | 0.0594 | 7.7× | 29.1× | 4.4× | 3.2× |
| `64×512×2048` | 3 | 0.0408 | 11.2× | 31.6× | 5.7× | 6.3× |
| `64×512×2048` | 4 | 0.0410 | 11.0× | 31.3× | 5.7× | 6.2× |
| `64×512×2048` | 8 | 0.0450 | 10.2× | 28.6× | 5.2× | 5.7× |
| `8×2048×4096` | 3 | 0.0911 | 15.9× | 208.8× | 4.0× | 3.6× |
| `8×2048×4096` | 4 | 0.1034 | 5.6× | 185.2× | 3.5× | 3.2× |
| `8×2048×4096` | 8 | 0.1178 | 4.0× | 167.1× | 3.1× | 2.8× |

P8's gate reads *grouped path beats the per-expert loop by ≥3× at decode*. Against the loop as it
ships today that is **4.0×–15.9×**, so the gate is met on the denominator that is hardest to argue
with. Against the loop as it shipped last week it is 29×–209×, and the difference between those
two columns is a defect in the baseline, not a property of grouping.

Two more things in that table deserve separating, because they are different claims:

- **`vs dense` is 3.1×–5.7×.** The kernel beats the same loop when the bank arrives already
  dense and the dequantization costs nothing. So the win is not "we skipped dequantizing" — one
  launch over segments beats E launches over slices on its own.
- **The quantized kernel is 2.8×–6.3× faster than dense fp16 `torch._grouped_mm`.** Decode is
  bandwidth-bound, a 3-bit weight is a smaller read, and a tensor-core GEMM has no use for its
  tensor cores at four rows.

### Where it stops winning, which is the argument for P7

`vs loop` crosses 1.0 as `rows` grows, and where it crosses depends on the geometry — this is the
last row count at which the fused kernel is still ahead:

| bank | 3-bit | 4-bit | 8-bit |
|---|---:|---:|---:|
| `32×1024×2048` | 512 | 2048 | 512 |
| `64×512×2048` | 2048 | 2048 | 2048 |
| `8×2048×4096` | 128 | 128 | 128 |

Read as: ahead up to that many rows, behind past it. On the widest bank the fused kernel is
already behind at 512 rows and loses by **8.3×** at 2048 (27.6439 ms against 3.3802) — there,
dequantize the bank once and call a dense grouped matmul. That is not a defect. A GEMV at 2048
rows is the wrong kernel, and this is the prefill split stated in measured milliseconds instead of
in principle: it is exactly what P7's tensor-core GEMM is for, and the crossover row is where the
dispatcher should switch.

### The 3-bit path costs about twice the 4-bit path once `rows` is large

| bank | 3-bit ms | 4-bit ms | ratio |
|---|---:|---:|---:|
| `32×1024×2048` | 5.5460 | 2.8620 | 1.94× |
| `64×512×2048` | 2.5342 | 1.3005 | 1.95× |
| `8×2048×4096` | 27.6439 | 11.3725 | 2.43× |

Reading *fewer* bytes and taking *more* time is the unpacking, not the memory. Three bits is the
one width that does not divide 32, so 32 values come out of 3 words with cross-word shifts, while
4 and 8 bits are constant-shift extractions from a single word. It is invisible at decode, where
the read dominates, and it dominates once the read is amortised. This is what a vectorized
grouped variant — still unwritten, still bullet 2 of section 9 — would go after.

### The baseline defect, and why its column stayed

`loop_ref_ms` is not a rewritten loop. It is the shipped loop with `__getitem__` monkeypatched
back onto `QuantTensor.dequantize`, so the only thing differing between the two columns is which
dequantizer `bank[e]` reaches. Across the 54 timed configurations that is worth a factor of
**2.81× minimum, 4.01× median, 46.13× maximum**, and it composes: on `8×2048×4096` at 3 bits and
4 rows, 15.9 × 13.1 is the 209× the first draft would have published as a kernel result.

The swap is checked rather than assumed. `loop_ref_spent` records how far the patched loop's
output moves from the shipped loop's, as a fraction of the same `atol + rtol·|expected|` the
parity suite uses, and it is **0.0000 in all 54** — not "within tolerance", identical, because
both paths accumulate in fp32 and round once.

The column stayed in the sweep after the fix landed, with its sense inverted, so it is now the
standing price of the `ops` hop rather than a historical note: a revert shows up here as `loop_ms`
climbing to meet `loop_ref_ms`, rather than as a speedup quietly getting better. The fix itself is
guarded by `test_indexing_a_bank_goes_through_the_dispatcher_not_the_reference`, which asserts the
*call* and not the output — on CPU the two paths are numerically indistinguishable, so a test
comparing numbers would have stayed green through a revert.

## 11. The sanitizer run that closes the last bullet

Same box, same build, same script. `l4_final.sh` runs the timing pass and then hands the *same*
benchmark to each tool in turn, so what the sanitizer inspects is the workload section 10
measured and not a smaller stand-in. The sanitizer passes omit `--time`, so they run the
correctness sweep only — that is the `grouped_mm=not probed without --time` line in each log.

| tool | over the 108-launch workload | elapsed | over the grouped parity suite | elapsed |
|---|---|---:|---|---:|
| `memcheck` | `ERROR SUMMARY: 0 errors` | 104 s | 23 passed, `0 errors` | 13 s |
| `initcheck` | `ERROR SUMMARY: 0 errors` | 110 s | — | — |
| `synccheck` | `ERROR SUMMARY: 0 errors` | 33 s | — | — |
| `racecheck` | `0 hazards displayed (0 errors, 0 warnings)` | 2348 s | 23 passed, `0 hazards` | 14 s |

Every workload row carries `configurations=108 launches=108 worst_ratio=0.0362` in the same log,
so each tool saw the full sweep and the numerical result was unchanged under instrumentation. The
raw logs are in `experiments/phase4/results/l4-sm89-kernels/sanitizer/`.

### The correction that had to come first, demonstrated rather than asserted

The first framing of this got the coverage wrong twice, and the thing that exposed it was a cost
that did not add up: the parity suite finished in about ten seconds under every tool, the same as
bare. Instrumented kernels do not usually cost nothing. Either the GPU work was a small share of
a mostly import-bound session, or the tool had attached to a process that launched no kernels at
all.

It was the second. `compute-sanitizer` defaults to `--target-processes application-only`, which
instruments the process it launches and nothing that process spawns — and the parity suite spawns
a subprocess. So the number of grouped launches that had ever run under the tool was **one**, not
the suite's twenty-three.

That claim is now a measurement rather than a story. The positive control is a deliberately
out-of-bounds `torch.index_select`, run three ways
(`sanitizer/sanpos.py`, `sanitizer/sanpos_sub.py`, `sanitizer/sanpos-targetprocesses.txt`):

| | invocation | what the tool said |
|---|---|---|
| A | direct, `--target-processes all` | `ERROR SUMMARY: 3 errors` |
| B | one subprocess away, `--target-processes application-only` | **no `ERROR SUMMARY` at all** |
| C | one subprocess away, `--target-processes all` | `ERROR SUMMARY: 3 errors` |

A is the control on the control: if a definitely-wrong kernel reports clean, every clean report
above is worthless. It does not. C shows the same fault is caught once the tool is told to follow
children. B is the failure mode, and it is worth quoting verbatim, because it does *not* print a
reassuring zero:

```
========= Error: Target application terminated before first instrumented API call
========= Tracking kernels launched by child processes requires the --target-processes all option.
```

The tool says so plainly. The trap is that the line reads as tooling noise next to a green pytest
summary, and the absence of an `ERROR SUMMARY` reads as nothing rather than as a hole. Which is
also why the workload benchmark, not the suite, is the vehicle for the table above: it launches
its 108 grouped kernels in the process the sanitizer started, so the coverage question does not
arise for it at all.

### What racecheck cost, and why that is the reassuring part

`racecheck` took **2348 s against memcheck's 104 s** — 22.6× — over identical work. That ratio is
what an honestly instrumented shared-memory tool looks like on a kernel that stages scales,
offsets and segment boundaries through shared memory: it serializes and re-checks every access.
The GPU sat at 100% utilization and 36 W throughout, which is the signature of fully serialized
instrumented execution rather than a hang.

The earlier attempt at this was killed after fifteen minutes on the assumption that it had hung.
It had not; it needed thirty-nine.

### What this still does not claim

One card, one architecture, `89-real` only. Four tools is not all of them, and a clean
`compute-sanitizer` is evidence about memory safety, initialization, synchronization and shared
memory races — not about numerical correctness, which is what section 10's `worst_ratio` and the
parity suite are for.

---

*Section 12 was added on 2026-08-15, closing the third bullet of section 9. It is the first time
a whole model has decoded through `grouped_gemv.cu`.*

## 12. The end-to-end run that closes the third bullet

Sections 10 and 11 timed and sanitized the kernel against synthetic banks. Section 9's third
bullet asked for something the benchmark cannot answer: whether a real model, loaded from a real
checkpoint, decodes through this kernel and comes back coherent — and what the speedup is worth
once the rest of a transformer is in the loop.

The vehicle is `LiquidAI/LFM2.5-8B-A1B`: 8,467,856,128 parameters, 24 layers of which **22 are
MoE**, 32 experts routed top-4, hidden 2048, expert intermediate 1792. It is chosen because it is
the model whose genuine `Lfm2MoeExperts` the packed runtime was verified against in the packed-MoE
report, and because its expert bank is a bare three-dimensional `nn.Parameter` — the exact case
`DynQuantExpertBank` exists to stand in front of.

`experiments/phase4/moe_end_to_end.py` runs one arm per process, so peak VRAM is a property of
that arm and not of whatever ran before it. The decode rate is a **slope** between two budgets, 32
and 96 new tokens, rather than a division of one generation by its token count: prefill, tokenizer
and sampling setup are identical at both budgets and cancel exactly in the difference. Both are
reported, and they differ by under 2%, which is itself the evidence that prefill was never the
thing being measured.

### The three arms

| arm | dispatch | decode tok/s | slope-free tok/s | resident MiB | peak MiB, load+pack | peak MiB, generate |
|---|---|---:|---:|---:|---:|---:|
| `bf16` | `grouped_mm` | **33.14** | 32.70 | 16151.2 | 16151.2 | 16246.7 |
| `eager`, 4-bit | `eager` | **16.20** | 15.66 | 4295.7 | 7167.5 | 4332.9 |
| `dynquant`, 4-bit | `dynquant` | **31.64** | 31.47 | 4295.7 | 7167.5 | 4332.9 |

The `dispatch` column is recorded, not assumed. An arm named `dynquant` on a build where the op is
missing silently becomes the loop and reports a speedup of 1.0x — which reads as *the kernel not
helping* rather than as *the kernel not running*, and those are opposite conclusions from the same
number. `has_grouped_gemv` is `true` in all three records and `experts_impl_after` is `dynquant`
on the last arm, `eager` on the arm that exists to be beaten.

The whole run was repeated end to end after the memory accounting was corrected (below). The two
independent runs agree to within **0.4%** on every rate: 33.14 / 16.13 / 31.66 the first time,
33.14 / 16.20 / 31.64 the second.

### The grouped kernel is 1.95x the loop end-to-end, not 8.55x

Against the per-expert loop over the same packed weights, the grouped path is
**31.64 / 16.20 = 1.95x**. The nearest geometry in section 10's sweep — 32 experts, 4 bits, 4
rows, which is precisely this model's decode regime of one token routed to four of thirty-two —
measures **8.55x**. Both numbers are right and the gap between them is the finding.

A decode step is not only expert GEMMs. Attention, the 18 convolutional layers, the norms, the
router itself, sampling and the Python driving all of it are identical across the two packed arms,
so the kernel can only take back the part of the step it owns. Solving
`1 / ((1 - f) + f/8.55) = 1.95` for that share puts the expert banks at roughly **55%** of the
eager-packed decode step. That figure is an inference, not a measurement — it assumes the swept
geometry's ratio transfers to banks of `[32, 3584, 2048]` and `[32, 2048, 1792]`, which are wider
than the `[32, 1024, 2048]` that was swept — and it is written here as an inference because the
alternative is to quote 8.55x next to a model and let a reader assume it is the model's number.

The claim P8's gate makes is about the path, and the sweep meets it. The claim a person deploying
this cares about is the model's, and it is 1.95x. Neither substitutes for the other.

### What it costs against bf16

The grouped 4-bit arm decodes at **0.955x** the bf16 model's rate — 4.5% slower — while holding
**3.76x** less weight memory. The per-expert loop at the same memory decodes at 0.49x, so on this
model the choice the kernel actually removes is *not* between fast and small: without it, packing
this model costs half the decode rate.

### The memory number this section nearly published was 67% workspace

The first run recorded one memory figure per arm, `torch.cuda.max_memory_allocated` sampled after
load: **7167.5 MiB** for the packed arms against bf16's 16151.2. Written up as it stood, that is a
2.25x reduction and it is wrong — not as arithmetic but as a description of what a served model
costs.

Packing runs the clipping search on the GPU -- `compute_device` defaults to the accelerator
independently of where the model sits, which is what makes packing a CPU-resident model cheap.
While a module's packed form is being built, a dense copy of that module and the search's
candidate reconstructions are resident, so the peak over load-and-pack carries a workspace that a
server loading an exported checkpoint never pays. The size is consistent with that reading: the
largest expert bank here is `[32, 3584, 2048]` at bf16, **447.9 MiB**, and the excess over
resident is 2871.8 MiB, or about six working copies of it. The peak is a real number about this
harness; it is not the model's footprint, and only the harness knows the difference.

Recording `torch.cuda.memory_allocated` separately — currently allocated rather than
high-water — splits them:

| quantity | bf16 | packed 4-bit |
|---|---:|---:|
| resident after load | 16151.2 MiB | **4295.7 MiB** |
| peak over load and pack | 16151.2 MiB | 7167.5 MiB |
| peak during generation | 16246.7 MiB | 4332.9 MiB |
| `packed_bytes`, from the tensors that exist | — | 4289.4 MiB |

Resident sits **6.3 MiB above** the byte accounting on a 4.3 GB model, and the residue is
accounted for: the 22 dense routers are 2.75 MiB and the norms are most of the rest. That is P6's
gate — *peak VRAM ≈ manifest size, not fp16 size* — measured against the allocator rather than
predicted from the bit map. The two are separate claims and this report has been careful elsewhere
about the difference; the honest one is the smaller one, and it is the one that agrees.

The reduction is **3.76x** resident, against `fp16_bytes / packed_bytes = 3.765x` predicted. The
prediction was right; the first measurement of it was not.

### The router is left dense, and the shipped path cannot do that

`pack_model` refuses `Lfm2MoeTopKRouter` by class: it owns a weight and calls `F.linear` on it
without being an `nn.Linear`, so the packed runtime has no forward to stand in for. The refusal is
correct and its message names the alternatives. It is also what the allocator's own floors want —
a top-k decision is discrete, and a router that rounds to a different argmax has not lost
precision, it has lost the token.

So this harness drops routers from the map before packing and reports what that costs rather than
asserting it is small: **22 routers, 1,441,792 parameters, 0.017%** of the model, 2.75 MiB at
bf16.

That is a caller-side fix, and the shipped path does not have it.
[`evaluate.py::_pack`](../../packages/dynquant-core/src/dynquant/commands/evaluate.py#L925)
reads the bit map from file and hands it to `pack_model` unfiltered. The router is in that map
whichever way the map was made: `--uniform` gives it the uniform width, and the role-aware
allocator gives it 8 bits, because `MOE_ROUTER` is a `STRUCTURAL_ROLE` with an 8-bit floor rather
than an exclusion. **`dynquant eval --map --map-apply pack` therefore refuses on any
LFM2.5-family MoE**, and the packed runtime — which is to say every VRAM figure in this
section — is reachable on this family only through `--map-apply encode`, `dynquant export`, or a
caller that filters the map itself. This was found by running the thing rather than by reading it,
which is the argument for having run it.

### Two mistakes the harness made first

**A missing chat template.** The first bf16 arm returned fluent, grammatical, entirely contentless
loops — *"the problem is that the issue is that the problem is that"* — because a bare instruction
went to an instruct-tuned model. Timing was unaffected and the arm would have passed any check
this harness makes. A coherence claim read off that output would have been a claim about the
harness. `_render()` applies the checkpoint's own template now, and the reasoning is recorded in
its docstring rather than here, because the next person to add an arm needs it there.

**A figure written before it was measured.** The comment explaining why routers stay dense
originally asserted that they are *0.05% of this model*. Nothing had measured that. It says
`router_params` now and the record carries the number, which is 0.017%.

### At 3 bits the loop gets worse and the kernel does not

P8's gate names 3-bit coherence, and the run above is 4-bit, so the pair was repeated at 3.

| bits | `eager` loop | `dynquant` grouped | grouped / loop | grouped / bf16 | resident | vs bf16 |
|---|---:|---:|---:|---:|---:|---:|
| 4 | 16.20 tok/s | 31.64 tok/s | **1.95x** | 0.955x | 4295.7 MiB | 3.76x smaller |
| 3 | 12.23 tok/s | **31.55 tok/s** | **2.58x** | 0.952x | 3286.7 MiB | 4.91x smaller |

The grouped path is effectively **width-independent** at decode — 31.64 against 31.55, three parts
in a thousand — while the per-expert loop loses **24%** going from 4 bits to 3. Narrower codes are
more expensive to unpack, 3-bit most of all because thirty-two values span three words and the
shifts cross word boundaries, and the loop pays that per expert per call where the grouped kernel
pays it once inside a launch that was already memory-bound. So the kernel's advantage *grows* as
the width narrows, which is the direction that matters: 3-bit is where this project's margins are,
and it is where the loop is worst.

> **Corrected 2026-08-15.** *Width-independent* was the fence talking. Every number in the table
> above was taken with a `torch.bincount` host read live inside `_segment_offsets`, one per bank
> per layer — a fixed per-step cost paid identically at both widths, which is precisely what
> flattens a width difference. With it removed the same harness on the same box reads
> `31.58 -> 32.52`: **3 bits is 3.0% faster than 4**, the direction a memory-bound decode should
> move. The rest of this subsection survives — the loop still loses 24% going 4 to 3, and the
> grouped path's advantage still grows as the width narrows, from 1.95x to **2.66x**. See
> [section 13](#13-the-capture-and-the-fence-counting-could-not-see) for the fix, the per-op
> bisect that found it, and the re-run.

Both 3-bit arms generate coherently. Asked for a SQL query over departments by average salary,
the grouped 3-bit arm opens *"I need to write a SQL query that returns the three departments with
the highest average salary... to get averages and order..."* and proceeds correctly; asked why
memory bandwidth rather than FLOPs limits decoding, it answers on topic. Byte accounting holds at
the narrower width too: resident **3286.7 MiB** against `packed_bytes` of 3280.1 MiB, a gap of
**6.6 MiB**, matching the 6.4 MiB gap at 4 bits — the same dense routers and norms, not a
width-dependent error.

### What section 12 does not claim

- **One model family.** P8's gate names Mixtral-8x7B and Qwen3-MoE; this is LFM2.5-8B-A1B. The
  bank layout `DynQuantExpertBank` stands in front of is shared across the 49 transformers
  `*Experts` classes that index a single parameter, but shared layout is not a measurement, and
  no Mixtral or Qwen3-MoE has run through this kernel.
- **One card.** Still `sm_89`, still one L4, as in sections 10 and 11.
- **Coherence is read, not scored.** Two prompts, eyeballed against the bf16 arm's own answers to
  the same prompts. That is enough to retire *"no end-to-end model ran through the grouped
  kernel"*; it is not an accuracy result, and the panels elsewhere in `docs/reports/` are what
  accuracy claims rest on.
- **CUDA Graphs are not in this.** *(Superseded 2026-08-15 by section 13.)* P8's gate also asks
  that graph replay remove measurable launch overhead. Nothing here is captured, and the decode
  rates above are eager-mode launches. The device-tensor `_segment_offsets` work in the packed-MoE
  report removed the host reads that made capture impossible, but capture itself remains
  unmeasured. Both sentences turned out to be wrong in the same place: that work removed *a* host
  read, not *the* host reads, and the first capture attempt refused on the fused path.
- **The load-imbalance stress case is the sweep's, not this run's.** Routing here is whatever the
  model does on two prompts; the all-tokens-to-one-expert case is section 10's `band` column.


## 13. The capture, and the fence counting could not see

Section 12 closed with *"CUDA Graphs are not in this"* and an argument for why they would be easy:
the packed-MoE report had removed `_segment_offsets`' `.tolist()`, a host read is the thing that
makes capture impossible, therefore the forward was capturable. That is a syllogism, and it was
built on a claim nothing had tested — the docstring's *"`bincount` and `cumsum` are both
shape-determined, so the whole function traces."*

The first capture attempt refused. Not the loop — **the fused path**, the one the removal was
about:

```
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph capture
unless the CPU tensor is pinned.
```

### `torch.bincount` reads its input on the host, and `minlength` does not spare it

`bincount` sizes its output from `input.max()`, which it takes as a scalar on the host. Supplying
`minlength` raises the floor on that size; it does not remove the read, because the op still has
to know whether the data exceeds the floor. Bisected one primitive at a time, same ids, at
`--arm ops`:

| op | captures |
|---|---|
| `bincount(minlength=E)` | **no** |
| `bincount(minlength=2E)` | **no** |
| `sort` | yes |
| `cumsum` | yes |
| `zeros(E+1).scatter_add_` | yes |

So the count is a `scatter_add_` into a fixed `[E + 1]` buffer now, and the extra bin is where the
clamp sends the expert-parallel sentinels — dropped by the same `[:num_experts]` slice as before,
so the semantics did not move, only the arithmetic that produced them.

The interesting part is not the fix. It is that **this survived the section that was about it**.
The packed-MoE report found the `.tolist()`, removed it, and asserted the absence by counting
`.tolist()` calls — the only assertion available, since removing a fence changes no output. That
counter is exact, and it answered a narrower question than its section claimed: `bincount` does
not call `.tolist()`, so a second fence per bank per layer sat in the same function, on the same
44-per-token budget, and passed every test in the file. Capture is what found it, because capture
does not need to be told in advance what to look for.

### What replay is worth, and it is a decode-shaped number

One MoE block — `dynquant_experts_forward` over two `DynQuantExpertBank` projections at
LFM2.5-8B-A1B's geometry (E=32, hidden 2048, moe-intermediate 1792, top-4). Eager is timed
*after* the capture, on the same buffers, so nothing about the ordering favours replay. Every
arm re-checks the graph's output against a fresh eager result on fresh inputs, because a graph
holding stale pointers replays happily and returns the previous answer:

| arm | bits | tokens | captured | eager ms | replay ms | removed ms | speedup |
|---|---|---|---|---|---|---|---|
| fused | 4 | 1 | yes | 0.53555 | 0.16486 | 0.37069 | **3.25x** |
| fused | 4 | 8 | yes | 1.40074 | 1.22163 | 0.17911 | 1.15x |
| fused | 4 | 64 | yes | 3.22605 | 3.20973 | 0.01632 | 1.00x |
| fused | 3 | 1 | yes | 0.53851 | 0.14746 | 0.39105 | **3.65x** |
| fused | 3 | 8 | yes | 1.28965 | 1.09349 | 0.19616 | 1.18x |
| fused | 3 | 64 | yes | 4.12979 | 4.06830 | 0.06149 | 1.01x |
| **loop** | 4 | 1 | **NO** | — | — | — | — |

Read the *removed* column, not the ratio. It falls too — 0.371 ms at one token, 0.179 at eight,
0.016 at sixty-four. What replay removes is therefore not the launch cost but **the launch cost
not already hidden behind GPU work**. At batch 64 the CPU has finished issuing before the GPU
has finished the first kernel, and a graph has nothing left to save. That is exactly the shape a
launch-bound claim should have, and it is why this number belongs to decode and to nothing else.

### The loop cannot be captured at any width

The per-expert loop reads `.tolist()` to decide how many iterations to run. That is not a fence
that tuning removes; it is the trip count. So the grouped path is not merely 1.95x faster than
the loop at 4 bits — the two are on opposite sides of a capturability line, and no amount of work
on the loop crosses it.

### Re-running end to end, because the fence was on the fused path too

Section 12's decode numbers were taken with the `bincount` fence live in every packed step. So
they were re-taken, same harness, same box, same checkpoint, after the fix:

| arm | bits | section 12 | re-run | rep 2 | change |
|---|---|---|---|---|---|
| `bf16` | — | 33.14 | 33.25 | — | +0.3% |
| `eager` | 4 | 16.20 | 16.18 | — | -0.1% |
| `dynquant` | 4 | 31.64 | 31.58 | — | -0.2% |
| `eager` | 3 | 12.23 | 12.19 | 12.27 | +0.1% |
| `dynquant` | 3 | 31.55 | **32.78** | **32.25** | **+3.1%** |

`eager` is the control and it is a control the harness supplies rather than one this section
constructed: it is the built-in expert loop, it never enters `_segment_offsets`, and it did not
move — 0.7% spread across both widths. `bf16` did not move either. Only the packed 3-bit arm did.

Two post-fix readings 1.6% apart, both above the single pre-fix reading. That is **suggestive,
not settled**: with one measurement on the old side there is no paired test to run, and the
honest statement is that the 3-bit arm moved by about the size of its own run-to-run spread plus
a bit. Nothing else in this report depends on which end of that interval is right.

**A section 12 claim is wrong and is corrected here.** Section 12 read `31.64 -> 31.55` across
widths and called the grouped path *width-independent at decode*. Post-fix it reads
`31.58 -> 32.52` (mean of the two 3-bit readings): **3 bits is 3.0% faster than 4**, which is
what a memory-bound decode should do when it reads 25% fewer weight bytes. The fence was
flattening the difference — a fixed per-step cost paid identically at both widths will do that.
The corrected picture, against `bf16` at 33.25 tok/s:

| arm | bits | tok/s | vs loop | vs bf16 | resident MiB | vs bf16 |
|---|---|---|---|---|---|---|
| `dynquant` | 4 | 31.58 | 1.95x | 0.950x | 4295.7 | 3.76x less |
| `dynquant` | 3 | 32.52 | 2.66x | 0.978x | 3286.7 | 4.91x less |

Resident memory is identical to section 12 to the tenth of a MiB in both re-runs, as it must be:
the fix changes how a count is computed, not what is stored.

### What section 13 does not claim

- **One MoE block is not the model.** The 3.25x is a block-level replay number. The end-to-end
  effect of capturing a *whole decode step* is not measured here, and the arithmetic that
  suggests it — 22 MoE layers x 0.371 ms = 8.2 ms against a measured 31.67 ms step, so 26% — is
  an **upper bound, not a prediction**. In the real model the CPU is issuing other layers' work
  while the GPU runs these, so some of that latency is already hidden, exactly as it is at 8
  tokens in the sweep above.
- **The capture probe uses synthetic weights.** It measures launch structure, not accuracy;
  accuracy lives in section 12's generations and in the parity suite.
- **One card, one family.** L4, sm_89, torch 2.11.0+cu126, LFM2.5's MoE geometry. A
  capturability result is architectural and should port; a 3.25x is not.
- **The re-run does not re-open section 12's other findings.** Resident memory, the coherence
  checks, and the load-imbalance stress case are unaffected by a fence.

## 14. The whole model, and what one block could not predict

Section 13's first disclaimer was arithmetic dressed as a caveat: 22 MoE layers x 0.371 ms of
removed launch latency is 8.2 ms, against a measured 31.67 ms decode step, so 26% — *an upper
bound, not a prediction*. An upper bound stated that plainly is an invitation to test it, and the
L4 was already rented and otherwise idle, so `experiments/phase4/graph_capture_model.py` tests it
on the real packed checkpoint rather than on one synthetic block.

Three arms, each capturing something different, on LFM2.5-8B-A1B packed to 4 and to 3 bits:

- `stack` — the full 24-layer forward at sequence length 1 with `use_cache=False`. Every module
  the model owns, in the order the model calls them, and nothing else. It is **not a decode
  step** and is never quoted as one; it has no cache to read and no attention history to attend
  over. What it measures is the *launch structure* of the whole model.
- `step` — one real decode step, cache built by an actual prefill, captured by hand.
- `compile` — the same step under `torch.compile(mode="reduce-overhead")`, which is the way a
  user would reach for graphs without writing capture code.

### The whole-model launch overhead is 75% of a sequence-length-1 forward

| bits | eager ms | replay ms | removed ms | speedup | max abs delta | argmax agrees |
|---|---|---|---|---|---|---|
| 4 | 29.633 | 7.438 | 22.196 | **3.98x** | 0.0 | yes |
| 3 | 29.522 | 6.798 | 22.724 | **4.34x** | 0.0 | yes |

Both captured, both bit-identical to a fresh eager forward after a new token was written into the
captured input buffer — so the graph is reading the token it was given, not replaying a memorized
answer. Resident memory was 4295.7 MiB at 4 bits, the same figure to the tenth of a MiB that
sections 12 and 13 report.

The internal consistency check is the interesting part, and it is the reason these two rows are
worth more than either alone. **The removed time is width-invariant and the remaining time is
not.** 22.196 against 22.724 ms is a 2.3% spread; 7.438 against 6.798 ms is 8.6%. That is exactly
the pattern a launch-overhead story predicts and no other story does: issuing 111 packed modules
plus norms, routers and convolutions costs the CPU the same regardless of how many bits the
weights are stored in, while the work the GPU then does is cheaper at 3 bits than at 4. Had the
graph been removing *work* rather than *launches*, the removed column would have tracked width
and the remaining column would have been flat. It is the other way around.

It also puts a number on something section 12 could only see through a fence. At 4 bits the
end-to-end decode advantage of 3 bits over 4 measured 3.0%; with launches removed, the same
comparison on the same checkpoint is 8.6%. Both are real: the first is what a user gets today,
the second is what the arithmetic underneath is worth once the launch tax is paid off.

### A real decode step on a `DynamicCache` captures, replays four times faster, and is wrong

`stack` is a proxy. The `step` arm is not: it runs a real prefill of a real prompt, takes the
cache `generate` hands back, and captures one decode step at position 255 with
`cache_position` pinned. It captured. It replayed at **4.034x** — 30.769 ms eager against 7.628
ms replayed, 23.141 ms removed. Every number in that sentence is a number a serving stack would
want.

The step it replayed produces the wrong token. `max_abs_delta` is **15.33** and
`argmax_agrees` is **false**.

Nothing raised. Capture returned cleanly, replay ran, and the timings are excellent, because a
CUDA graph is a recording of *addresses* and the cache transformers gives this model is a
`DynamicCache`, which grows by `torch.cat` and therefore hands back a newly allocated pair of
tensors on every step. The graph faithfully replays reads of the buffers that were live during
capture; by replay time the cache has abandoned them. The failure is invisible to every check
except the one that compares the replayed output against a fresh eager forward — which is why
that comparison is in the record next to the speedup and not in a follow-up.

The cache census printed at prefill says the same thing structurally. It was also, for two
drafts of this section, evidence for a claim it could not support:

```
[prefill] prompt=25 warm=231 cache=DynamicCache {'(1, 8, 255, 64)': [12]}
```

Twelve tensors, all the same shape: 6 attention layers x (K, V), 8 KV heads, head dim 64, and
nothing else. That was read here as "the other 18 layers contribute nothing to
`past_key_values`," and then used to argue that a capture of this model reads a container
describing a quarter of it. **They contribute all of it.** A `LinearAttentionLayer` keeps its
convolution state in a `{0: tensor}` dict, and the walk that produced this census descended
into lists, tuples and `__dict__` but not into plain dicts, so it could not see inside one. A
later diagnostic that enumerated `cache.layers` directly found all 24 layers present and
initialized, each convolution layer holding its `conv_states`. The walk is now dict-aware and
the claim is withdrawn — the census above is left exactly as printed, because what it shows
is what a partial walk shows.

Two things follow, and they should not be run together. The **timing** is representative: the
replay issues the same kernels over the same shapes in the same order as a correct one would, so
4.034x bounds what a correct capture is worth on this model and this card. The **capture** is
not deliverable *on this container*: a hand-captured decode step cannot be made correct
without a cache whose storage does not move. Getting one turned out to take three more
subsections than expected.

It is also a second, independent reading of the same launch structure, and it agrees with the
first. `step` eager is 30.769 ms against `stack` eager 29.633 ms, 3.8% apart, the difference
being attention over 255 positions that `stack` does not do; removed is 23.141 against 22.196
ms, 4.3% apart. Two arms built differently, measuring the same tax.
### The static cache was full, and four explanations of that were wrong

The static arm asserted on device at both widths, in both the hand-captured and the compiled
arm. A device-side assert is asynchronous, so the traceback it produces names the next
synchronization point rather than the op that failed; `CUDA_LAUNCH_BLOCKING=1` and one process
per configuration are what make it localizable at all, because a poisoned context turns every
later arm in the same process into a measurement of the poisoning.

Four explanations were written down before the right one. Each is worth recording, because each
was refuted by a specific observation and not by argument:

* **The model's hybrid layer mix.** LFM2.5-8B-A1B is 18 short-convolution layers to 6 attention
  layers, and the failing `index_copy_` ran in a single block of about 96 threads, which reads
  like a per-layer structure that a uniform static cache would not have. Refuted by enumerating
  `cache.layers`: all 24 are present and initialized, each convolution layer holding its
  `conv_states`. The composite is built correctly.
* **This file's own arithmetic.** The decode position came from `sequences.shape[1] - 1`, and
  the cache is one slot shorter than that sequence, so 255 looked like one past the end of 255
  slots. Refuted by fixing it: at position 254, in bounds by inspection, the assert survived
  unchanged at both widths.
* **The graph.** Refuted by taking one **eager** step at the same position with no capture
  anywhere in the process. It asserts identically, which means `captured: false` had been
  labelling the wrong stage.
* **The packed runtime.** One of the failures came back as `RuntimeError: DynQuant CUDA
  error: ...`, which puts our own kernels in the call path. Refuted by running the same step on
  a dense bf16 model with nothing packed: byte-identical traceback, same two frames in
  `cache_utils.py`.

What it actually is takes two lines of `transformers` to state. `generation/utils.py` sizes a
generation's static cache at `max_length - 1` -- 255 slots for the 256-token sequence it
returns, because the last token emitted is never fed back. And `StaticLayer.update` does not
take a write position from its caller at all:

```python
cache_position = torch.arange(kv_length, device=self.device) + self.cumulative_length
self.cumulative_length.add_(kv_length)
self.keys.index_copy_(2, cache_position, key_states)
```

The `cache_position` a caller passes is used for the causal mask and for RoPE; the slot the key
lands in comes from a device-resident cursor that the layer advances itself. So a prefill of N
tokens leaves a static cache both N long and exactly N full, and the next step runs off the end
**at any position**. There was never a position that worked. A `DynamicCache` has no capacity to
run off, which is exactly why it hid this for three rounds and offered a wrong answer instead.

The fix is one argument: ask `generate` for `max_cache_len` above what the prefill will fill.
With 192 spare slots the prefill reports `capacity=448 cursor=255 seq=256` and the step has
somewhere to go.

That cursor is not free, and the arm records what it costs. `cache_writes` is **115** for a run
of 50 timed replays: every call advances it -- three capture warmups, five timing warmups, fifty
eager timings, fifty replays, and the correctness forwards. A replayed graph increments the
cursor because the increment is a device op inside the recording. **A captured decode step on a
static cache is replayable only as many times as the cache has spare slots**, which is a real
constraint on shipping one and is not visible from the speedup.

### With headroom, the supported path captures the whole model

`torch.compile(mode="reduce-overhead")` is how a serving stack reaches CUDA Graphs without
writing capture code, so it is the arm that decides whether any of this is reachable in
practice. Against a `DynamicCache` it did not produce a number at either width. Both raised the
same thing:

```
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten
by a subsequent run.
```

Inductor's `cudagraph_trees` has a guard for precisely the failure the hand capture produced
silently, and the guard fired. Against a static cache with spare slots, the same call goes
through:

| bits | eager ms | compiled ms | removed ms | speedup | graph breaks | compile s |
|---|---|---|---|---|---|---|
| 4 | 32.463 | 6.577 | 25.886 | **4.936x** | **0** | 6.3 |
| 3 | 32.382 | 5.940 | 26.442 | **5.452x** | **0** | 6.1 |

**Zero graph breaks** across the whole packed model: 111 DynQuant modules, the grouped MoE
kernel, 18 short-convolution layers and 6 attention layers, all inside one graph. Nothing in the
packed runtime forces a fallback to eager -- which is the property `torch.library.custom_op`
plus `register_fake` was built for, and the first evidence that it holds at model scale rather
than at block scale. Compilation costs about six seconds, once.

This retires the previous draft of this section, which read the `DynamicCache` refusal as a
statement about the model and concluded that the straightforward way to turn graphs on does not
work here. It works. What did not work was handing it a container whose storage moves, and the
refusal was the guard doing its job on an input this file had chosen badly.

### The delta needed a yardstick before it could be read

Every static arm captured, replayed, and agreed with eager on the argmax. None of them agreed
to zero. `max_abs_delta` came back between 0.375 and 2.906 -- small against logits of order 20,
but not the exact 0.0 the cacheless `stack` arm produces, and this report has already once read
a nonzero number as a statement about capture when it was a statement about the container.

So the arms were re-run with a control: after taking the graph's output and the eager reference,
take **a second eager forward** on the same input and compare the two eager runs to each other.
A decode step mutates the cache it reads -- a static layer advances its write cursor, a
convolution layer shifts its window -- so two consecutive eager forwards are not required to
agree either, and until it is known by how much they disagree, a graph-vs-eager delta cannot be
attributed to the graph.

| bits | arm | graph vs eager | **eager vs eager** | argmax, both comparisons |
|---|---|---|---|---|
| 4 | `step` | 1.6875 | **2.3125** | agrees |
| 4 | `compile` | 0.3750 | **0.2500** | agrees |
| 3 | `step` | 1.1875 | **2.0000** | agrees |
| 3 | `compile` | 2.9062 | **2.2773** | agrees |

On three of the four arms **the control is at least as large as the quantity it controls**: the
graph is no further from eager than eager is from itself. On the fourth, 3-bit compiled, the
graph delta is 2.906 against a control of 2.277 -- 28% larger, the same order of magnitude, and
not a separation this measurement can resolve. In every arm both comparisons select the same
token.

That is what a correct capture looks like on a model whose cache is mutable state, and it is the
strongest statement the measurement supports. It is not bit-exactness -- nothing here is
bit-exact, including eager against itself -- and it does not extend past the one step captured.
What it rules out is the failure the `DynamicCache` arm exhibits, where the delta is 15.33 and
the argmax flips. That is two orders of magnitude away and needs no control to see.

### What captures, what does not, and what it costs

Six configurations, one model, one card. `stack` and the `DynamicCache` rows come from the
earlier sweep; the four static rows come from the re-run that carries the control.

| arm | cache | captured | correct | 4-bit | 3-bit |
|---|---|---|---|---|---|
| `stack` | none (`use_cache=False`) | yes | delta 0.0, argmax agrees | 3.984x | 4.343x |
| `step` | `DynamicCache` | yes | **no** -- delta 15.33 / 18.38, argmax flips | 4.030x | 4.338x |
| `compile` | `DynamicCache` | **no** -- guard raises | n/a | n/a | n/a |
| `step` | `static`, no headroom | **no** -- device assert | n/a | n/a | n/a |
| `step` | `static`, 192 spare | yes | within the eager-vs-eager control | **4.207x** | **4.556x** |
| `compile` | `static`, 192 spare | yes | within the control at 4-bit | **4.936x** | **5.452x** |

Two things fall out of the column pair that did not fall out of section 13.

**The removed time is width-invariant; the remaining time is not.** Across all four static arms
removed is 24.76, 25.89, 25.22, 26.44 ms -- a 6.8% spread with no ordering by width. The
remaining time is 7.72, 6.58, 7.09, 5.94 ms, and there the 3-bit arms are consistently the
faster. That is the signature of removing *launches* rather than *work*: launch cost does not
know how many bits a weight is stored in, and the arithmetic left running after the launches are
gone does. Section 13 measured this on one block and put an upper bound of 26% on it. At model
scale it is 76-82%, because a block has a handful of launches around real arithmetic and a model
has 24 layers of them around the same arithmetic.

**The supported path beats the hand-rolled one at both widths** -- 4.936x against 4.207x, 5.452x
against 4.556x. The hand capture wraps one `model(...)` call in a graph and leaves everything
around it in Python; Inductor fuses inside the graph as well as capturing it, so it removes
launches the hand capture never had a way to reach. The arm written to be the reference is the
one that loses, which is the outcome that makes it worth shipping the compiled path rather than
capture code.

One comparison that is **not** available: static eager against dynamic eager. Static eager is
32.3-32.5 ms and dynamic eager is 30.7, because a static cache runs attention over all 448
allocated slots while the dynamic one runs over the 255 it holds. The speedup ratios are each
internally consistent -- eager and replay measured on the same container -- but the two
containers are not measuring the same step, and the 4.207x should not be read as an improvement
on the 4.030x.

### What section 14 does not claim

* **Not a tokens-per-second number.** One decode step is timed in isolation, 50 iterations, on a
  cache that already holds 255 tokens. A generation loop pays prefill, sampling, detokenization
  and a cursor that keeps advancing; none of that is here. 4.94x on a step is not 4.94x on a
  request.
* **Not a claim about long context.** Capacity is 448 slots. Attention cost grows with the
  cursor and launch cost does not, so the fraction of a step that graphs can remove **falls** as
  context grows. This measurement is at the short end, where the ratio is most favourable.
* **Not a replayable-forever step.** `cache_writes` is 116 for a 50-iteration run: every call
  advances the static cursor, replays included, because the increment is a device op inside the
  recording. A shipped decode loop must re-capture or re-allocate before the cache fills.
* **Not bit-exactness.** The control establishes that the graph is no further from eager than
  eager is from itself, on this step, at this position. It does not establish agreement to zero,
  which does not hold for eager either.
* **Not a statement about other cache implementations.** `StaticCache` was the container tested.
  The hybrid `Mamba`-style and quantized caches transformers also ships were not.
* **Not generalized past this model and this card.** LFM2.5-8B-A1B on one L4. The launch-bound
  regime that makes the number large is a property of a small-batch decode on a card with modest
  SM count; a larger card or a batched server sits somewhere else on that curve.

## 15. A second MoE family, and the loop gets worse exactly where the kernel does not

Every number in sections 12 through 14 came off one checkpoint. P8's gate names *Mixtral-8x7B /
Qwen3-MoE*, and every one of those sections closes by saying so: **one model family**. Neither
named model fits an L4 -- Mixtral-8x7B is 47B parameters and Qwen3-30B-A3B is 30B, against 23 GB
of card -- so the clause was answered with the nearest thing that is genuinely a different test
rather than the nearest thing that shares a name.

### The geometry that makes it a different test

`allenai/OLMoE-1B-7B-0125-Instruct`, 6,919,161,856 parameters, against LFM2.5-8B-A1B's
8,467,856,128:

| | LFM2.5-8B-A1B | OLMoE-1B-7B |
|---|---|---|
| experts / top-k | 32 / 4 | **64 / 8** |
| MoE layers | 22 | 16 |
| expert intermediate | 1792 | **1024** |
| attention | 6 full + 18 short-convolution | **16 full, no convolution** |
| router | `Lfm2MoeTopKRouter` (not an `nn.Linear`) | `mlp.gate`, a plain `nn.Linear` |
| embeddings | tied | **untied** |
| expert bank share of params | 91.5% | 93.1% |

Different expert count, different top-k, different expert width, a different layer mix, a
different router class, and a different embedding arrangement. What it shares is the batched
`[E, out, in]` bank layout, which is what the grouped kernel actually consumes -- so it tests
the dispatch against a second router and a second routing density without changing the thing
under test.

The packer needed nothing added for it: **98 modules packed, 16 routers left dense** (2,097,152
parameters, 0.030% of the model), 0 tied, 0 skipped, `accounted_bits` **3.2515**, and the expert
banks moved from `grouped_mm` to `dynquant` exactly as they do on LFM2.5. The generic structural
classifier reached `mlp.gate` through `out_features == num_experts` with an `experts` sibling,
which is the test P3 was written around and the first time a family it was not developed against
has exercised it end to end.

### 3.18x the per-expert loop, on a family the kernel was not tuned against

Same harness as section 12 (`experiments/phase4/moe_end_to_end.py`), same L4, same two prompts,
128 new tokens, greedy, chat template applied, three arms in one process:

| arm | decode tok/s | vs loop | vs bf16 | resident MiB | peak loaded MiB | experts run by |
|---|---|---|---|---|---|---|
| `bf16` | 41.48 | 3.58x | 1.000x | 13197.3 | 13197.3 | `grouped_mm` |
| `eager` | 11.59 | 1.00x | 0.279x | 2685.4 | 7254.3 | per-expert loop |
| `dynquant` | **36.86** | **3.180x** | 0.889x | 2685.4 | 7254.6 | grouped kernel |

**3.180x clears P8's >=3x gate on a second family**, and a first run before the cache probe was
added measured 3.167x, so the figure reproduces across processes. Both packed arms hold the
same weights -- the only difference between them is one call swapping `_experts_implementation`
after the pack, which is why they are directly comparable in a way two separate runs would not be.

Memory lands where P6 says it should. Resident after load is **2685.4 MiB** against a manifest
`packed_bytes` of 2,810,003,456 B = **2679.8 MiB**: **5.6 MiB apart, 0.21%**. The ratio
against bf16 is **4.914x**, the same ratio LFM2.5 gets at 3 bits to four significant figures --
not a coincidence, since both land at `accounted_bits` 3.2515 and 16/3.2515 = 4.921. The
`peak_mib_loaded` of 7254.3 is again the clipping search's transient dense GPU copy during
packing, not a steady state.

Coherent at 3 bits on both prompts in all three arms -- the SQL prompt produces a correct
`GROUP BY ... ORDER BY AVG(salary) DESC LIMIT 3` in every arm. The `eager` and `dynquant`
generations are **byte-identical on the first prompt** and diverge partway through the second,
which is the expected signature of substituting one kernel at the dispatch: identical inputs,
identical weights, floating-point differences below the argmax margin until greedy decoding
amplifies one of them.

### Where the two families disagree, and why the disagreement is the point

Reading the two grouped-vs-loop numbers side by side is more informative than either alone:

| | LFM2.5, 3 bits | OLMoE, 3 bits |
|---|---|---|
| `bf16` tok/s | 33.25 | 41.48 |
| loop tok/s | 12.23 | **11.59** |
| grouped tok/s | 32.52 | 36.86 |
| **grouped / loop** | 2.66x | **3.18x** |
| grouped / bf16 | 0.978x | 0.889x |
| expert matmuls launched per token | 22 x 4 x 3 = **264** | 16 x 8 x 3 = **384** |
| parameters per expert matmul | 3,670,016 | **2,097,152** |
| active expert parameters per token | 968,884,224 | 805,306,368 |

OLMoE is the smaller model, and its bf16 arm is correspondingly faster -- 41.48 against 33.25.
Its per-expert loop is nevertheless **slower in absolute terms**, 11.59 against 12.23, while
reading 17% fewer expert parameters per token. A loop that does less work and takes more time is
launch-bound, and the two rows underneath say by how much: 45% more launches, each doing 43% less
arithmetic, so roughly 1.75x the launch overhead per unit of work.

That is the same effect section 14 measured directly, arriving from the other side. There,
freezing a whole decode step into one graph removed 76-82% of it. Here, a geometry that issues
more and smaller expert calls pays more of that overhead in the loop -- and the grouped kernel,
which issues **one** launch per layer regardless of expert count, does not pay it at all. The gap
between the two paths therefore widens exactly where the launches get smaller, which is what a
launch-bound explanation predicts and a bandwidth-bound one does not.

The other column moves the opposite way, and honestly. Against `bf16`, OLMoE's grouped path
recovers 0.889x where LFM2.5 recovers 0.978x. Smaller expert matrices give the quantized GEMV
less arithmetic to hide the dequantization behind, so the fraction of bf16 it can reach is lower
-- the cost of 3-bit weights is more visible on a 2048x1024 matrix than on a 2048x1792 one.
Both statements are about the same geometry, and both are what the memory-bound decode model
says should happen.

### The checkpoint said `use_cache: false`, and the run used one anyway

OLMoE-1B-7B ships `"use_cache": false` in its `config.json`. LFM2.5 ships `true`. The harness had
been building an explicit `GenerationConfig` -- `do_sample=False`, `temperature=None`,
`top_p=None`, `top_k=None` -- for the reasons section 12 gives, and had simply not named
`use_cache`, so on this checkpoint it was inherited. Decoding 128 tokens without a KV cache is
quadratic in the budget. All three arms would have paid it equally, so the **ratio** would have
survived and the **rate** would not: 3.18x would still have been 3.18x, and 36.86 tok/s would
have been a number about a configuration nobody runs. That is precisely the kind of error a
comparison hides, which is why it is worth naming rather than quietly fixing.

The fix was one line, `use_cache=True` in the config. The instrumentation added alongside it was
wrong, and the second attempt is the part worth recording:

```python
"config_use_cache": bool(getattr(model.config, "use_cache", True)),   # measures nothing
```

That field reports the checkpoint's static declaration. It read `False` on all three OLMoE arms
*after* the fix, while all three demonstrably decoded with a cache -- because `generate` is
governed by the per-call `GenerationConfig`, not by `model.config`. A field that reads `False`
next to a run that is manifestly `True` is worse than no field: it invites the reader to conclude
the fix did not take. It was replaced with an observation instead of a declaration --

```python
cfg = GenerationConfig(..., use_cache=True, max_new_tokens=4, return_dict_in_generate=True)
out = model.generate(**enc, generation_config=cfg)
return getattr(out, "past_key_values", None)  # None means it decoded without one
```

-- a four-token probe run once per arm outside the timed region, whose returned cache length goes
into the record as `decoded_cache_len`. It reads **30** on all three arms: 26 prompt tokens plus
4 generated. The static field is still recorded next to it, now correctly labelled as the
checkpoint's declaration rather than the run's behaviour, and the two disagreeing is exactly the
hazard this pair exists to make visible.

### What section 15 does not claim

- **OLMoE is not Mixtral-8x7B or Qwen3-30B-A3B.** P8's gate names those two. Neither fits on an
  L4 at bf16 for the baseline arm, so the second-family evidence is a family that fits, not the
  family the gate names. What generalises is the mechanism -- batched `[E, out, in]` banks, a
  generically classified router, one launch per layer -- not the constant.
- **Two families are not a trend.** 2.66x and 3.18x are two points, and the launch-count
  explanation drawn through them is a reading consistent with both plus section 14's direct
  measurement, not a fitted model. A third geometry could move it either way.
- **The cross-family arithmetic is a sanity check, not a controlled comparison.** LFM2.5 and
  OLMoE differ in attention (18 short-convolution layers against none), vocabulary (128k against
  50k), depth, and embedding tying. Only the within-model ratios -- grouped against loop, on the
  same weights in the same process -- are controlled.
- **No accuracy claim.** Three arms generating coherently on two prompts is a smoke test for
  3-bit viability on a second family. It is not an evaluation, and nothing here revises the
  panel numbers in the phase 4 reports.
- **`--map-apply pack` still cannot reach this path.** Routers carry an 8-bit structural floor
  rather than an exclusion, so a DynQuant bit map naming them cannot be packed; the arms here
  quantize uniformly at 3 bits. That limitation is unchanged from section 12 and is still open.
