// Shared device-side utilities. Header-only; included by every .cu.
#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "dynquant/abi.h"

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
//
// Launch errors are asynchronous: a bad launch config or an out-of-bounds write
// surfaces at the *next* synchronising call, which is usually somewhere unrelated
// in user code. Checking after every launch turns "loss became NaN three steps
// later" into a stack trace at the offending kernel.
//
// A `std::runtime_error` rather than TORCH_CHECK so this header stays compilable
// without libtorch, and rather than a bespoke struct so that pybind and torch
// translate it to a Python RuntimeError carrying the message. Throwing a type
// that does not derive from std::exception surfaces as "unknown exception" with
// the cause discarded, which is the opposite of the point.

#define DYNQUANT_CUDA_CHECK(expr)                                                        \
  do {                                                                                   \
    cudaError_t _dq_err = (expr);                                                         \
    if (_dq_err != cudaSuccess) {                                                         \
      ::dynquant::throw_cuda_error(#expr, cudaGetErrorString(_dq_err), __FILE__,           \
                                   __LINE__);                                              \
    }                                                                                     \
  } while (0)

// Use immediately after a <<<>>> launch. Catches invalid configuration
// (too many threads, too much shared memory) at the launch site rather than
// wherever the next sync happens to be.
#define DYNQUANT_CHECK_LAUNCH() DYNQUANT_CUDA_CHECK(cudaGetLastError())

namespace dynquant {

[[noreturn]] inline void throw_cuda_error(const char* expression, const char* message,
                                          const char* file, int line) {
  throw std::runtime_error(std::string("DynQuant CUDA error: ") + message + " (" + expression +
                           ") at " + file + ":" + std::to_string(line));
}

// ---------------------------------------------------------------------------
// Launch geometry
// ---------------------------------------------------------------------------

constexpr int kWarpSize = 32;

// Threads per block for elementwise/streaming kernels. 256 is the usual sweet
// spot: enough warps to hide global latency, small enough that a block's
// register budget does not cap occupancy on sm_75 (64K registers/SM).
constexpr int kDefaultBlock = 256;

__host__ __device__ inline constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Grid size for a grid-stride loop over `n` items. Capped so the grid stays
// resident: a persistent grid amortises launch cost and keeps the L2 warm, and
// past ~2^16 blocks the scheduler is just queueing work anyway.
__host__ inline int grid_for(int n, int block = kDefaultBlock, int max_blocks = 65535) {
  int blocks = ceil_div(n, block);
  return blocks < 1 ? 1 : (blocks > max_blocks ? max_blocks : blocks);
}

}  // namespace dynquant
