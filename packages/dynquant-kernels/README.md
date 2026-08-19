# dynquant-kernels

Compiled CUDA kernels for [DynQuant](https://github.com/kambojvikram/dynquant).

You almost certainly want `pip install dynquant`, which resolves this wheel for
your platform. Install it directly only to force a specific variant.

## What is in here

One shared object, exposed through the PyTorch dispatcher under the `dynquant`
namespace (not pybind, so `torch.compile` can trace through it):

* n-bit tile dequantization
* decode GEMV, templated on bit-width — the memory-bound case that dominates serving
* tensor-core prefill GEMM (cuBLASLt, then fused CUTLASS)
* MoE grouped GEMM — one launch for all experts, not one per expert

Built with Thrust, CUB, cuBLASLt and CUTLASS. `-use_fast_math` is off by design:
it implies denormals-to-zero, and a dequantized weight near the bottom of a 2-bit
group's range is exactly the denormal case, so accuracy measured with it on is not
reproducible.

## Variants

The wheel version carries a PEP 440 local segment identifying what it was built
against, the same scheme torch uses:

```
dynquant-kernels==0.5.3+cu128torch28
```

Compatibility with `dynquant-core` is enforced by an ABI number baked into the
binary, checked at import. A mismatch refuses to load with an actionable message
rather than returning wrong numbers — see `csrc/include/dynquant/abi.h`.

## Diagnosing an install

This package never raises on import, so a broken binary degrades to the reference
backend rather than taking down `import dynquant`. To find out what happened:

```python
import dynquant_kernels

print(dynquant_kernels.report())
```

That works with `dynquant-core` absent, which is the point — it makes this wheel
diagnosable on its own. `dynquant doctor` gives the same information in context.

## Building from source

```bash
pip install --no-build-isolation -e packages/dynquant-kernels
```

Needs CMake ≥ 3.26, nvcc, and a C++17 compiler. `--no-build-isolation` is
required: an isolated build resolves the newest torch, and an extension linked
against a different libtorch than the one you run fails to import.

With no CUDA toolkit present the build still succeeds, producing a CPU-only
extension — deliberately, so a cheap CI runner catches CMake and registration
errors before a GPU runner is needed. Release builds set
`DYNQUANT_REQUIRE_CUDA=ON` so that a toolkit which failed detection is a build
failure, not a wheel advertising kernels it does not contain.

## License

Apache-2.0.
