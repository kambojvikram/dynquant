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
- **No vectorized grouped variant.** A busy expert still decodes at the general path's bandwidth.
- **One architecture.** Built `89-real` only, on one L4. Nothing here says anything about `sm_80`,
  `sm_90a` or Blackwell, and no wheel from this build has been published.
- **No end-to-end model ran through the grouped kernel here.** That is the packed-MoE report's
  measurement, and it used the Python loop.
- **`compute-sanitizer` was not re-run on this build.** The 0-error, 0-hazard result recorded for
  the packed runtime predates `grouped_gemv.cu` ever being compiled, so it does not cover it.
