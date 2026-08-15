// Operator schemas, CPU reference implementations, and the module entry point.
//
// Ops are registered through TORCH_LIBRARY rather than exposed as pybind
// functions. That is not a style preference: an op in the dispatcher gets a
// schema the compiler can reason about, so `torch.compile` traces through it
// (given the Python-side `register_fake`), autograd and AMP see a real node, and
// CUDA Graph capture works. A pybind function is opaque to all of it and forces a
// graph break at every quantized Linear -- i.e. at every layer.
//
// Every op has a CPU implementation. Not for speed: it is the reference the CUDA
// path is tested against, and it means the extension's registration, schemas and
// fake impls are all exercised by CI on a runner with no GPU. A packaging bug
// therefore surfaces on a cheap CPU job, minutes after the push, instead of on a
// scarce GPU runner at the end of the matrix.

#include <ATen/ATen.h>
#include <Python.h>
#include <torch/library.h>

#include <cstdint>
#include <optional>
#include <string>

#include "dynquant/abi.h"
#include "dynquant/geometry.h"

namespace dynquant {

// Defined in probe_cuda.cu when CUDA is available.
#ifdef DYNQUANT_WITH_CUDA
std::string compiled_architectures();
#else
std::string compiled_architectures() { return ""; }
#endif

namespace {

int64_t abi_version_op() { return static_cast<int64_t>(::dynquant::abi_version()); }

bool built_with_cuda() {
#ifdef DYNQUANT_WITH_CUDA
  return true;
#else
  return false;
#endif
}

std::string compiled_architectures_op() { return compiled_architectures(); }

int64_t gemv_max_rows_op() { return DYNQUANT_GEMV_MAX_ROWS; }

// ---------------------------------------------------------------------------
// CPU reference implementations
// ---------------------------------------------------------------------------

at::Tensor probe_axpy_cpu(const at::Tensor& x, const at::Tensor& y, double alpha) {
  TORCH_CHECK(x.sizes() == y.sizes(), "probe_axpy: shape mismatch ", x.sizes(), " vs ", y.sizes());
  TORCH_CHECK(x.scalar_type() == y.scalar_type(), "probe_axpy: dtype mismatch");
  // Accumulate in fp32 then cast back, matching the CUDA kernel exactly. Doing
  // `x * alpha + y` in fp16 here instead would make the CPU oracle disagree with
  // the GPU by one rounding, and every parity test would need a fudge factor.
  return (x.to(at::kFloat) * alpha + y.to(at::kFloat)).to(x.scalar_type());
}

at::Tensor probe_reduce_cpu(const at::Tensor& x) {
  return at::full({}, x.to(at::kFloat).sum().item<double>(), x.options().dtype(at::kFloat));
}

at::Tensor probe_gemm_cpu(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "probe_gemm: expected 2-D tensors");
  TORCH_CHECK(a.size(1) == b.size(0), "probe_gemm: inner dimensions disagree: ", a.sizes(), " @ ",
              b.sizes());
  return at::matmul(a.to(at::kFloat), b.to(at::kFloat)).to(a.scalar_type());
}

// --- dequant ---------------------------------------------------------------
//
// Scalar, row-major, no vectorisation. This is the oracle, so it is written to be
// obviously the same arithmetic as `dynquant.quant.pack.unpack_nbit` -- one value
// at a time, the shift spelled out -- rather than to be fast. Anything clever here
// would have to be re-verified against the Python before it could be trusted to
// verify the CUDA.
template <typename scalar_t>
void dequant_cpu_rows(const uint32_t* packed, const scalar_t* scales, const scalar_t* offsets,
                      scalar_t* out, int64_t num_rows, int64_t bits,
                      const nbit::RowGeometry& geom) {
  const uint32_t mask = (uint32_t{1} << bits) - 1u;
  for (int64_t r = 0; r < num_rows; ++r) {
    const uint32_t* row = packed + r * geom.words_per_row;
    for (int64_t g = 0; g < geom.num_groups; ++g) {
      const uint32_t* group = row + g * geom.words_per_group;
      const float scale = static_cast<float>(scales[r * geom.num_groups + g]);
      const float offset =
          offsets == nullptr ? 0.0f : static_cast<float>(offsets[r * geom.num_groups + g]);
      for (int64_t j = 0; j < geom.group_values; ++j) {
        const int64_t v = g * geom.group_values + j;
        // Past in_features is the pad that rounded the row up to a whole group.
        if (v >= geom.in_features) {
          break;
        }
        const int64_t bit = j * bits;
        const int64_t w = bit >> 5;
        const int shift = static_cast<int>(bit & 31);
        uint32_t value = group[w] >> shift;
        if (32 - shift < bits) {
          // Only reachable at 3-bit. The straddled-into word is always still
          // inside this group: the value's last bit is below group_values * bits,
          // which is at most words_per_group * 32.
          value |= group[w + 1] << (32 - shift);
        }
        value &= mask;
        out[r * geom.in_features + v] =
            static_cast<scalar_t>(static_cast<float>(value) * scale + offset);
      }
    }
  }
}

at::Tensor dequant_cpu(const at::Tensor& packed, const at::Tensor& scales,
                       const std::optional<at::Tensor>& offsets, int64_t bits,
                       int64_t group_values, int64_t in_features) {
  const auto geom =
      resolve_geometry("dequant", packed, scales, offsets, bits, group_values, in_features);
  auto out = at::empty({packed.size(0), in_features}, scales.options());
  if (out.numel() == 0) {
    return out;
  }
  const auto* words = reinterpret_cast<const uint32_t*>(packed.const_data_ptr<int32_t>());
  const int64_t num_rows = packed.size(0);
  const bool has_offsets = offsets.has_value();

  AT_DISPATCH_SWITCH(
      scales.scalar_type(), "dynquant::dequant",
      AT_DISPATCH_CASE(at::kHalf,
                       [&] {
                         dequant_cpu_rows<scalar_t>(
                             words, scales.const_data_ptr<scalar_t>(),
                             has_offsets ? offsets->const_data_ptr<scalar_t>() : nullptr,
                             out.data_ptr<scalar_t>(), num_rows, bits, geom);
                       })
          AT_DISPATCH_CASE(at::kBFloat16,
                           [&] {
                             dequant_cpu_rows<scalar_t>(
                                 words, scales.const_data_ptr<scalar_t>(),
                                 has_offsets ? offsets->const_data_ptr<scalar_t>() : nullptr,
                                 out.data_ptr<scalar_t>(), num_rows, bits, geom);
                           })
              AT_DISPATCH_CASE(at::kFloat, [&] {
                dequant_cpu_rows<scalar_t>(
                    words, scales.const_data_ptr<scalar_t>(),
                    has_offsets ? offsets->const_data_ptr<scalar_t>() : nullptr,
                    out.data_ptr<scalar_t>(), num_rows, bits, geom);
              }));
  return out;
}

// The reference dequantizes in fp32, and both callers below go through this rather
// than through the `dequant` op itself.
//
// `dequant_cpu` dispatches on the scales' dtype and rounds its *store* to that dtype,
// so calling it with fp16 scales and upcasting afterwards puts one fp16 rounding on
// every weight element in front of a sum over `in_features` of them. That error grows
// like sqrt(in_features) while the parity test's tolerance is flat -- and it is not an
// error the CUDA kernel has, because `gemv_kernel` never materialises the weight and
// applies scale and offset in fp32 (`fmaf(scale[r], qacc[r][m], ...)`). Measured on an
// L4 across the parametrized geometries, the fp16 route runs at 0.03 to 0.98 of the
// allowed tolerance and crosses it at 65x3072 8-bit, while the CUDA path stays under
// 0.043 everywhere. A reference less accurate than the kernel it defines is the wrong
// way round, and a tolerance that holds on one CPU and not another is a test that gets
// deleted rather than fixed.
//
// `dequant` itself keeps returning the storage dtype: its own test asserts bit-exact
// equality against `QuantTensor.dequantize()` in fp16, so widening the op would break
// the contract this only widens the *use* of.
at::Tensor dequant_reference_fp32(const at::Tensor& packed, const at::Tensor& scales,
                                  const std::optional<at::Tensor>& offsets, int64_t bits,
                                  int64_t group_values, int64_t in_features) {
  const std::optional<at::Tensor> wide =
      offsets.has_value() ? std::optional<at::Tensor>(offsets->to(at::kFloat)) : std::nullopt;
  return dequant_cpu(packed, scales.to(at::kFloat), wide, bits, group_values, in_features);
}


// --- gemv ------------------------------------------------------------------
//
// Dequantize, then matmul. That materialises the dense weight, which is precisely
// what the CUDA kernel exists to avoid -- but on CPU there is no bandwidth claim to
// make and the composition is the definition the GPU kernel is checked against.
//
// The row limit is enforced here too even though nothing on CPU requires it. An op
// whose contract differs by device is an op whose tests pass on the CI runner and
// fail on the GPU.
at::Tensor gemv_cpu(const at::Tensor& x, const at::Tensor& packed, const at::Tensor& scales,
                    const std::optional<at::Tensor>& offsets, int64_t bits, int64_t group_values,
                    int64_t in_features) {
  TORCH_CHECK(x.dim() == 2, "gemv: x must be 2-D [rows, in_features], got ", x.sizes());
  TORCH_CHECK(x.size(1) == in_features, "gemv: x has ", x.size(1), " columns but in_features is ",
              in_features);
  TORCH_CHECK(x.scalar_type() == scales.scalar_type(), "gemv: x dtype ", x.scalar_type(),
              " != scales dtype ", scales.scalar_type());
  TORCH_CHECK(x.size(0) <= DYNQUANT_GEMV_MAX_ROWS, "gemv: ", x.size(0),
              " activation rows exceeds the kernel's limit of ", DYNQUANT_GEMV_MAX_ROWS,
              ". Above this a quantized matmul is compute-bound and the dequant + cuBLASLt "
              "path is faster; dispatch there instead.");
  const auto weight =
      dequant_reference_fp32(packed, scales, offsets, bits, group_values, in_features);
  return at::matmul(x.to(at::kFloat), weight.t()).to(x.scalar_type());
}


// --- moe_grouped_gemv ------------------------------------------------------
//
// Per expert: take that expert's band of the bank, dequantize it, matmul its
// segment of the activation. Which is exactly what
// `dynquant.runtime.experts._grouped_linear_packed` does in Python, and that is
// the point -- the CUDA kernel's claim is that it computes the same thing without
// the host round-trip, so the reference has to be the thing it replaces rather
// than a re-derivation of it.
//
// This side also *validates the segment table*, which the CUDA side structurally
// cannot: the table is device data, and a kernel that read it back would reinstate
// the synchronization the kernel exists to remove. So the contract is that a
// malformed table raises here and is merely clamped there, and the tests exercise
// the CPU path to pin it.
at::Tensor moe_grouped_gemv_cpu(const at::Tensor& x, const at::Tensor& packed,
                                const at::Tensor& scales, const std::optional<at::Tensor>& offsets,
                                const at::Tensor& seg_offsets, int64_t bits, int64_t group_values,
                                int64_t in_features, int64_t out_features) {
  resolve_geometry("moe_grouped_gemv", packed, scales, offsets, bits, group_values, in_features);
  check_grouped(x, packed, scales, seg_offsets, in_features, out_features);

  const int64_t total_rows = x.size(0);
  const int64_t num_experts = seg_offsets.size(0) - 1;
  // Zeros, not empty: rows past the last segment are the expert-parallel sentinels
  // that no expert owns, and handing `torch.empty`'s contents to the next
  // projection is how one NaN becomes the whole layer.
  auto out = at::zeros({total_rows, out_features}, x.options());
  if (total_rows == 0) {
    return out;
  }

  const auto table = seg_offsets.to(at::kCPU).contiguous();
  const auto* bound = table.const_data_ptr<int32_t>();
  TORCH_CHECK(bound[0] == 0, "moe_grouped_gemv: seg_offsets must start at 0, got ", bound[0]);
  for (int64_t e = 0; e < num_experts; ++e) {
    const int64_t start = bound[e];
    const int64_t stop = bound[e + 1];
    TORCH_CHECK(start <= stop, "moe_grouped_gemv: expert ", e, " spans [", start, ", ", stop,
                "), which runs backwards; seg_offsets must be non-decreasing");
    TORCH_CHECK(stop <= total_rows, "moe_grouped_gemv: expert ", e, " ends at ", stop,
                " but x has only ", total_rows, " rows");
    if (start == stop) {
      continue;
    }
    const int64_t band = e * out_features;
    const auto band_packed = packed.narrow(0, band, out_features);
    const auto band_scales = scales.narrow(0, band, out_features);
    const std::optional<at::Tensor> band_offsets =
        offsets.has_value() ? std::optional<at::Tensor>(offsets->narrow(0, band, out_features))
                            : std::nullopt;
    const auto weight = dequant_reference_fp32(band_packed, band_scales, band_offsets, bits,
                                              group_values, in_features);
    const auto rows = x.narrow(0, start, stop - start);
    out.narrow(0, start, stop - start)
        .copy_(at::matmul(rows.to(at::kFloat), weight.t()).to(x.scalar_type()));
  }
  return out;
}

}  // namespace

TORCH_LIBRARY(dynquant, m) {
  // Metadata ops. The two-argument `def` registers a catch-all implementation,
  // which is what these want: they take no tensors, so there is no device to
  // dispatch on.
  m.def("abi_version() -> int", TORCH_FN(abi_version_op));
  m.def("built_with_cuda() -> bool", TORCH_FN(built_with_cuda));
  m.def("compiled_architectures() -> str", TORCH_FN(compiled_architectures_op));
  // The runtime's dispatch threshold, read from the binary rather than assumed.
  // Registered here and not in gemv.cu so a CPU-only build answers it too --
  // otherwise the dispatch logic would have to special-case having no CUDA.
  m.def("gemv_max_rows() -> int", TORCH_FN(gemv_max_rows_op));

  // Probes. Declared without implementations here so each backend registers its
  // own below; the fake/meta impls live in dynquant_kernels/ops.py.
  m.def("probe_axpy(Tensor x, Tensor y, float alpha) -> Tensor");
  m.def("probe_reduce(Tensor x) -> Tensor");
  m.def("probe_gemm(Tensor a, Tensor b) -> Tensor");

  // Quantized kernels.
  //
  // `group_values` is the *resolved* values-per-group, never the -1 sentinel the
  // Python format uses for per-row grouping, and `in_features` is the unpadded
  // width. Both are explicit arguments rather than derived from shapes because a
  // fake tensor has shapes but no data, so a meta kernel could not recover them --
  // and because deriving the group size by division is the bug (see geometry.h).
  m.def(
      "dequant(Tensor packed, Tensor scales, Tensor? offsets, int bits, int group_values, "
      "int in_features) -> Tensor");
  m.def(
      "gemv(Tensor x, Tensor packed, Tensor scales, Tensor? offsets, int bits, "
      "int group_values, int in_features) -> Tensor");

  // The grouped MoE path. `seg_offsets` is [num_experts + 1] int32 on the same
  // device as the activation: it stays a tensor precisely so the caller never has
  // to read it, which is the difference between this and the Python loop it
  // replaces. `out_features` is explicit for the same reason `group_values` is --
  // the bank arrives flattened to [num_experts * out_features, ...], and a meta
  // kernel has that product but no way to factor it.
  m.def(
      "moe_grouped_gemv(Tensor x, Tensor packed, Tensor scales, Tensor? offsets, "
      "Tensor seg_offsets, int bits, int group_values, int in_features, "
      "int out_features) -> Tensor");
}

TORCH_LIBRARY_IMPL(dynquant, CPU, m) {
  m.impl("probe_axpy", TORCH_FN(probe_axpy_cpu));
  m.impl("probe_reduce", TORCH_FN(probe_reduce_cpu));
  m.impl("probe_gemm", TORCH_FN(probe_gemm_cpu));
  m.impl("dequant", TORCH_FN(dequant_cpu));
  m.impl("gemv", TORCH_FN(gemv_cpu));
  m.impl("moe_grouped_gemv", TORCH_FN(moe_grouped_gemv_cpu));
}

}  // namespace dynquant

// ---------------------------------------------------------------------------
// Module entry point
// ---------------------------------------------------------------------------
//
// A bare CPython module with no methods. The ops above are registered by static
// initializers that run when the shared object is loaded, so importing this
// module is the whole mechanism -- `torch.ops.dynquant.*` exists afterwards.
//
// Deliberately not PYBIND11_MODULE: pybind11 would be an extra ~1MB of template
// instantiation and a second ABI to keep in step with torch's, in exchange for
// nothing, since no function here crosses the Python boundary directly.

// PyMODINIT_FUNC, not a hand-written `extern "C" PyObject*`. The macro expands to
// `extern "C"` *plus* the platform's export attribute, and this extension is built
// with -fvisibility=hidden -- so the hand-written form compiles and links fine and
// then fails at import with "dynamic module does not define module export function
// (PyInit__C)", because the symbol is in the .so but not in its dynamic table.
PyMODINIT_FUNC PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT, "_C", "DynQuant compiled kernels.", -1, nullptr,
      nullptr,               nullptr, nullptr,                     nullptr,
  };
  return PyModule_Create(&module_def);
}
