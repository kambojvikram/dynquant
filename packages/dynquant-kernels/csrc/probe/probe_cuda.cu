// P0 probe kernels: prove the binary pipeline before any real kernel is written.
//
// Each probe exercises exactly one thing that, if it is broken, breaks a later
// phase in a way that is expensive to diagnose from inside that phase:
//
//   probe_axpy     nvcc compiled device code, the fatbin contains a cubin the
//                  running GPU can execute, and a <<<>>> launch reaches it.
//                  Fails if CMAKE_CUDA_ARCHITECTURES misses the device.
//   probe_reduce   Thrust/CUB headers resolve and their device code links. This
//                  is what P5's fused reduce_by_key quantizer and P8's
//                  stable_sort_by_key expert permutation are built on.
//   compiled_archs the arch list baked into this binary, so `dynquant doctor`
//                  can say "no cubin for your sm_120, launches will JIT from
//                  PTX and the first one will stall for seconds" instead of the
//                  user discovering it as a mysterious warm-up cost.
//
// These stay in the shipped wheel permanently. `dynquant doctor` runs them as a
// numerical self-check, so a broken install reports itself at diagnosis time
// rather than as a wrong logit somewhere in a 40-layer model.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform_reduce.h>

#include <string>

#include "dynquant/common.cuh"

namespace dynquant {
namespace {

// Grid-stride rather than one-thread-one-element: the same kernel shape the
// decode GEMV uses, so the probe validates the launch geometry helper too.
template <typename scalar_t>
__global__ void axpy_kernel(const scalar_t* __restrict__ x, const scalar_t* __restrict__ y,
                            scalar_t* __restrict__ out, float alpha, int n) {
  const int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    // Accumulate in fp32 even for half inputs. This is the project-wide rule
    // (see docs/format-spec.md): storage precision is fp16, arithmetic is not.
    const float v = alpha * static_cast<float>(x[i]) + static_cast<float>(y[i]);
    out[i] = static_cast<scalar_t>(v);
  }
}

template <typename scalar_t>
struct ToFloat {
  __device__ float operator()(scalar_t v) const { return static_cast<float>(v); }
};

}  // namespace

at::Tensor probe_axpy_cuda(const at::Tensor& x, const at::Tensor& y, double alpha) {
  TORCH_CHECK(x.is_cuda() && y.is_cuda(), "probe_axpy: expected CUDA tensors");
  TORCH_CHECK(x.sizes() == y.sizes(), "probe_axpy: shape mismatch ", x.sizes(), " vs ", y.sizes());
  TORCH_CHECK(x.scalar_type() == y.scalar_type(), "probe_axpy: dtype mismatch");

  const at::cuda::CUDAGuard guard(x.device());
  auto xc = x.contiguous();
  auto yc = y.contiguous();
  auto out = at::empty_like(xc);

  const int n = static_cast<int>(xc.numel());
  if (n == 0) {
    return out;
  }
  const auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, xc.scalar_type(), "probe_axpy_cuda", [&] {
        axpy_kernel<scalar_t><<<grid_for(n), kDefaultBlock, 0, stream>>>(
            xc.const_data_ptr<scalar_t>(), yc.const_data_ptr<scalar_t>(),
            out.mutable_data_ptr<scalar_t>(), static_cast<float>(alpha), n);
      });
  DYNQUANT_CHECK_LAUNCH();
  return out;
}

at::Tensor probe_reduce_cuda(const at::Tensor& x) {
  TORCH_CHECK(x.is_cuda(), "probe_reduce: expected a CUDA tensor");
  const at::cuda::CUDAGuard guard(x.device());
  auto xc = x.contiguous();

  // Thrust dispatched onto torch's current stream, not the default stream.
  // Getting this wrong is silent: the reduction races with whatever produced `x`
  // and reads partially-written memory only under load.
  const auto policy = thrust::cuda::par.on(at::cuda::getCurrentCUDAStream());

  double total = 0.0;
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, xc.scalar_type(), "probe_reduce_cuda", [&] {
        auto begin = thrust::device_pointer_cast(xc.const_data_ptr<scalar_t>());
        // transform_reduce to fp32, not reduce in scalar_t: summing 10^8 halves
        // in half precision saturates the mantissa long before the end.
        total = static_cast<double>(thrust::transform_reduce(
            policy, begin, begin + xc.numel(), ToFloat<scalar_t>{}, 0.0f, thrust::plus<float>()));
      });
  return at::full({}, total, xc.options().dtype(at::kFloat));
}

// Stringize a macro that expands to a comma-separated list. The variadic form is
// required: __CUDA_ARCH_LIST__ expands to `750,800,860`, which a single-parameter
// macro would reject as three arguments.
#define DQ_STRINGIFY_(...) #__VA_ARGS__
#define DQ_STRINGIFY(...) DQ_STRINGIFY_(__VA_ARGS__)

std::string compiled_architectures() {
#if defined(__CUDA_ARCH_LIST__)
  return DQ_STRINGIFY(__CUDA_ARCH_LIST__);
#else
  // CUDA < 11.5 does not define the list. Reporting "unknown" is honest; doctor
  // then skips the cubin-coverage check rather than claiming coverage it cannot
  // verify.
  return "unknown";
#endif
}

TORCH_LIBRARY_IMPL(dynquant, CUDA, m) {
  m.impl("probe_axpy", TORCH_FN(probe_axpy_cuda));
  m.impl("probe_reduce", TORCH_FN(probe_reduce_cuda));
}

}  // namespace dynquant
