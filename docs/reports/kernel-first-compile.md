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
- **CUDA Graphs are not in this.** P8's gate also asks that graph replay remove measurable launch
  overhead. Nothing here is captured, and the decode rates above are eager-mode launches. The
  device-tensor `_segment_offsets` work in the packed-MoE report removed the host reads that made
  capture impossible, but capture itself remains unmeasured.
- **The load-imbalance stress case is the sweep's, not this run's.** Routing here is whatever the
  model does on two prompts; the all-tokens-to-one-expert case is section 10's `band` column.

