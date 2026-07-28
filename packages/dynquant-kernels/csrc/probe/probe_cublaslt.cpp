// cuBLASLt probe -- and the skeleton of P7's Phase A prefill path.
//
// Why this is in P0 rather than P7: cuBLASLt is a separate library from cuBLAS
// with its own import library, and whether it resolves depends on the CUDA
// toolkit version, whether the wheel found CUDA::cublasLt, and on Windows
// whether the DLL is on the search path next to torch's. Discovering a linking
// problem in P7 means redesigning the build with three phases of work stacked on
// top of it; discovering it here costs an afternoon.
//
// Nothing here is throwaway. The descriptor setup, the row-major mapping and the
// heuristic query are exactly what `dequant_tile -> cublasLtMatmul` needs, and
// that path is the correctness-guaranteed prefill fast path that ships before the
// fused CUTLASS kernel exists.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublasLt.h>
#include <torch/library.h>

#include <mutex>
#include <string>

#include "dynquant/abi.h"

namespace dynquant {
namespace {

#define DQ_LT_CHECK(expr)                                                            \
  do {                                                                               \
    cublasStatus_t _dq_st = (expr);                                                   \
    TORCH_CHECK(_dq_st == CUBLAS_STATUS_SUCCESS, "cuBLASLt call failed (", #expr,      \
                ") with status ", static_cast<int>(_dq_st));                           \
  } while (0)

// One handle for the process. cuBLASLt handles are stateless and thread-safe --
// unlike a cuBLAS handle, they carry no stream or pointer-mode -- so there is
// nothing to be gained from per-thread handles and creating one per call costs
// milliseconds.
cublasLtHandle_t lt_handle() {
  static cublasLtHandle_t handle = nullptr;
  static std::once_flag once;
  std::call_once(once, [] { DQ_LT_CHECK(cublasLtCreate(&handle)); });
  return handle;
}

// RAII so an early TORCH_CHECK cannot leak a descriptor.
template <typename T, cublasStatus_t (*Destroy)(T)>
struct LtHandleGuard {
  T value{};
  ~LtHandleGuard() {
    if (value != nullptr) {
      Destroy(value);
    }
  }
};

}  // namespace

// Row-major C[M,N] = A[M,K] @ B[K,N], fp16 in/out with fp32 accumulate.
//
// cuBLASLt is column-major, and rather than transposing data we reinterpret it.
// A row-major matrix [r, c] with row stride c is bit-identical to a column-major
// matrix [c, r] with leading dimension c -- i.e. its own transpose. So:
//
//     C^T = (A @ B)^T = B^T @ A^T
//
// and every one of those transposes is free, being just the other reading of the
// same bytes. We therefore ask cuBLASLt for the column-major product
//
//     D[N,M] = Bview[N,K] * Aview[K,M]      (both ops CUBLAS_OP_N)
//
// where Bview is B's bytes read column-major with ld=N, Aview is A's bytes read
// column-major with ld=K, and D's bytes read row-major are exactly C[M,N].
// No copies, no transposes, and it is the same mapping the fused kernel will use.
at::Tensor probe_gemm_cublaslt(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "probe_gemm: expected CUDA tensors");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "probe_gemm: expected 2-D tensors, got ", a.dim(),
              "-D and ", b.dim(), "-D");
  TORCH_CHECK(a.size(1) == b.size(0), "probe_gemm: inner dimensions disagree: ", a.sizes(), " @ ",
              b.sizes());
  TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf,
              "probe_gemm: fp16 only (this probe pins the descriptor types that the "
              "prefill path uses); got ",
              a.scalar_type(), " and ", b.scalar_type());

  const at::cuda::CUDAGuard guard(a.device());
  auto ac = a.contiguous();
  auto bc = b.contiguous();

  const int64_t m = ac.size(0);
  const int64_t k = ac.size(1);
  const int64_t n = bc.size(1);
  auto c = at::empty({m, n}, ac.options());
  if (m == 0 || n == 0 || k == 0) {
    return c.zero_();
  }

  LtHandleGuard<cublasLtMatmulDesc_t, &cublasLtMatmulDescDestroy> op_desc;
  LtHandleGuard<cublasLtMatrixLayout_t, &cublasLtMatrixLayoutDestroy> layout_b, layout_a, layout_d;
  LtHandleGuard<cublasLtMatmulPreference_t, &cublasLtMatmulPreferenceDestroy> preference;

  // CUBLAS_COMPUTE_32F, not 16F: fp16 accumulation loses ~3 mantissa bits per
  // doubling of K and would put the quantization error budget in the noise of the
  // GEMM itself.
  DQ_LT_CHECK(cublasLtMatmulDescCreate(&op_desc.value, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t op_n = CUBLAS_OP_N;
  DQ_LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc.value, CUBLASLT_MATMUL_DESC_TRANSA, &op_n,
                                             sizeof(op_n)));
  DQ_LT_CHECK(cublasLtMatmulDescSetAttribute(op_desc.value, CUBLASLT_MATMUL_DESC_TRANSB, &op_n,
                                             sizeof(op_n)));

  // Argument order follows the reinterpretation above: cuBLASLt's "A" is our B.
  DQ_LT_CHECK(cublasLtMatrixLayoutCreate(&layout_b.value, CUDA_R_16F, n, k, n));
  DQ_LT_CHECK(cublasLtMatrixLayoutCreate(&layout_a.value, CUDA_R_16F, k, m, k));
  DQ_LT_CHECK(cublasLtMatrixLayoutCreate(&layout_d.value, CUDA_R_16F, n, m, n));

  // Workspace from torch's caching allocator rather than cudaMalloc: a raw
  // cudaMalloc here would fragment the allocator's arena and, worse, is illegal
  // during CUDA Graph capture, which P8 depends on.
  constexpr size_t kWorkspaceBytes = 32u * 1024u * 1024u;
  auto workspace = at::empty({static_cast<int64_t>(kWorkspaceBytes)},
                             ac.options().dtype(at::kByte));

  DQ_LT_CHECK(cublasLtMatmulPreferenceCreate(&preference.value));
  DQ_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(preference.value,
                                                   CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                   &kWorkspaceBytes, sizeof(kWorkspaceBytes)));

  cublasLtMatmulHeuristicResult_t heuristic{};
  int returned = 0;
  DQ_LT_CHECK(cublasLtMatmulAlgoGetHeuristic(lt_handle(), op_desc.value, layout_b.value,
                                             layout_a.value, layout_d.value, layout_d.value,
                                             preference.value, 1, &heuristic, &returned));
  TORCH_CHECK(returned > 0,
              "cuBLASLt found no algorithm for ", m, "x", k, " @ ", k, "x", n,
              " fp16. This usually means the GPU predates tensor cores or the "
              "shapes exceed a dimension limit.");

  const float alpha = 1.0f;
  const float beta = 0.0f;
  DQ_LT_CHECK(cublasLtMatmul(lt_handle(), op_desc.value, &alpha, bc.const_data_ptr(),
                             layout_b.value, ac.const_data_ptr(), layout_a.value, &beta,
                             c.const_data_ptr(), layout_d.value, c.mutable_data_ptr(),
                             layout_d.value, &heuristic.algo, workspace.mutable_data_ptr(),
                             kWorkspaceBytes, at::cuda::getCurrentCUDAStream()));
  return c;
}

TORCH_LIBRARY_IMPL(dynquant, CUDA, m) { m.impl("probe_gemm", TORCH_FN(probe_gemm_cublaslt)); }

}  // namespace dynquant
