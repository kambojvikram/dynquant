// Grouped decode GEMV: every expert of a packed MoE bank in one launch.
//
// What this replaces is not a kernel. It is a Python `for` loop over experts in
// `dynquant.runtime.experts._grouped_linear_packed`, and the loop's cost is not
// the arithmetic -- each expert's GEMM is the same work either way -- it is that
// the loop has to *know its own bounds on the host*. Segment offsets come from a
// `bincount` over the router's output, so reading them costs one device-to-host
// sync per bank per layer. On a 22-layer model with two banks each that is 44
// syncs per decoded token, and it makes `torch.compile(fullgraph=True)` and CUDA
// Graph capture impossible: the trip count is a tensor value.
//
// So the substitution here is narrow and specific. The offsets stay on device,
// the grid is computed from shapes alone, and each block reads its own segment
// bounds from global memory. Nothing about the launch depends on where the router
// sent anything.
//
//     grid = (ceil_div(out_features, rows per CTA), num_experts)
//
// Both dimensions are shapes. A block for an expert no token reached loads two
// int32s and returns; a block for a busy one walks its band of activation rows in
// tiles of `MROWS`. The trip count varies per block and the *launch* does not,
// which is the whole trade: an empty expert costs a block launch instead of
// costing a synchronization to discover it is empty.
//
// What it is not
// --------------
// Not a tensor-core GEMM. This is the decode regime -- one token routed to `k`
// experts is `k` segments of one row each -- where a matmul is bound by the time
// to stream the weights and tensor cores have nothing to do. Prefill has a
// different answer (`dequant` into a workspace, then cuBLASLt) and the runtime
// dispatches there on activation count, exactly as `quantized_matmul` does.
//
// Numerically it is `dynquant::gemv` restricted to a band of rows: same fp32
// accumulate, same fixed-shape warp butterfly, same decode helpers. A segment run
// through this kernel and the same segment run through `gemv` agree bit for bit,
// which is the property that lets the runtime pick either. Neither agrees bit for
// bit with `dequantize -> F.linear`, because cuBLAS reduces in a different order;
// that difference is the ordinary one between two matmul implementations and is
// bounded by the fp32 accumulation both ends do.

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cstdint>
#include <optional>

#include "dynquant/abi.h"
#include "dynquant/common.cuh"
#include "dynquant/geometry.h"
#include "dynquant/nbit.cuh"

namespace dynquant {
namespace {

// The same 4-rows-per-warp, 4-warps-per-CTA shape `gemv.cu` settled on, for the
// same reason: a warp reads the whole activation to produce its output rows, so
// covering several rows per warp amortises that read. Written out here rather
// than shared, because the two kernels do not have the same register story --
// this one holds an accumulator tile across a segment loop -- and a constant
// shared between them would couple two independent tuning decisions.
constexpr int kRowsPerWarp = 4;
constexpr int kWarpsPerCta = 4;
constexpr int kMoeBlock = kWarpSize * kWarpsPerCta;
constexpr int kRowsPerCta = kRowsPerWarp * kWarpsPerCta;

template <int BITS, int MROWS, typename scalar_t>
__global__ void __launch_bounds__(kMoeBlock)
    moe_grouped_gemv_kernel(const scalar_t* __restrict__ x,         // [total_rows, K], sorted
                            const uint32_t* __restrict__ packed,    // [E * out, words_per_row]
                            const scalar_t* __restrict__ scales,    // [E * out, num_groups]
                            const scalar_t* __restrict__ offsets,   // same shape, or null
                            const int32_t* __restrict__ seg,        // [E + 1], device-resident
                            scalar_t* __restrict__ out,             // [total_rows, out], pre-zeroed
                            int total_rows, int out_features, nbit::RowGeometry geom) {
  constexpr int kVals = nbit::Traits<BITS>::kValuesPerBlock;
  constexpr int kWords = nbit::Traits<BITS>::kWordsPerBlock;

  const int expert = static_cast<int>(blockIdx.y);
  // Clamped, and this is the only bounds check the device side can make. The
  // segment table is device data, so a kernel cannot validate it the way the CPU
  // reference does -- it would have to read it back. Two instructions here turn a
  // malformed table from an out-of-bounds read into wrong numbers, and the CPU
  // reference still refuses such a table outright, so a test catches it there.
  const int start = max(0, __ldg(seg + expert));
  const int stop = min(total_rows, __ldg(seg + expert + 1));
  // No token reached this expert. Two loads and out -- and, importantly, its
  // weights were never touched, so an unrouted expert costs no bandwidth.
  if (start >= stop) {
    return;
  }

  const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int row0 = (static_cast<int>(blockIdx.x) * kWarpsPerCta + warp) * kRowsPerWarp;
  if (row0 >= out_features) {
    return;
  }
  const int rows_here = min(kRowsPerWarp, out_features - row0);

  // Expert `e` owns bank rows [e * out_features, (e + 1) * out_features). That is
  // `QuantTensor.rows()`'s addressing, arithmetic rather than a lookup, which is
  // why a packed bank needed no new storage layout to be servable here.
  const int64_t bank_row0 = static_cast<int64_t>(expert) * out_features + row0;
  const int64_t stride_words = geom.words_per_row;
  const int64_t stride_meta = geom.num_groups;

  for (int base = start; base < stop; base += MROWS) {
    float acc[kRowsPerWarp][MROWS];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
      for (int m = 0; m < MROWS; ++m) {
        acc[r][m] = 0.0f;
      }
    }

    for (int block = lane; block < geom.blocks_per_row; block += kWarpSize) {
      const int group = block / geom.blocks_per_group;
      const int block_in_group = block - group * geom.blocks_per_group;
      const int word_base = group * geom.words_per_group + (block_in_group * kVals * BITS) / 32;
      const int value_base = group * geom.group_values + block_in_group * kVals;
      const int words_left = geom.words_per_row - word_base;

      uint32_t words[kRowsPerWarp][kWords];
      float scale[kRowsPerWarp];
      float offset[kRowsPerWarp];
#pragma unroll
      for (int r = 0; r < kRowsPerWarp; ++r) {
        if (r < rows_here) {
          const int64_t row = bank_row0 + r;
          nbit::load_block<BITS>(packed + row * stride_words + word_base, words[r], words_left);
          const int64_t meta = row * stride_meta + group;
          scale[r] = static_cast<float>(scales[meta]);
          offset[r] = offsets == nullptr ? 0.0f : static_cast<float>(offsets[meta]);
        } else {
          // Zeroed for the reason `gemv.cu` zeroes them: the accumulate below is
          // unpredicated on `r` so it unrolls, and a zero scale and offset make
          // the surplus rows contribute nothing to accumulators nobody reads.
#pragma unroll
          for (int w = 0; w < kWords; ++w) {
            words[r][w] = 0u;
          }
          scale[r] = 0.0f;
          offset[r] = 0.0f;
        }
      }

#pragma unroll
      for (int j = 0; j < kVals; ++j) {
        const int v = value_base + j;
        // Past `in_features` is the pad that rounded the row up to a whole group:
        // real stored codes with no activation to pair them with.
        if (v >= geom.in_features) {
          continue;
        }
        float xv[MROWS];
#pragma unroll
        for (int m = 0; m < MROWS; ++m) {
          // The activation tile is masked here rather than padded on the host, as
          // `gemv` pads. There is one segment per expert and they have different
          // lengths, so a host-side pad would mean a copy of the whole activation
          // per bank; a zero here costs one predicate and contributes nothing.
          const int token = base + m;
          xv[m] = token < stop
                      ? static_cast<float>(
                            __ldg(x + static_cast<int64_t>(token) * geom.in_features + v))
                      : 0.0f;
        }
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
          const float w =
              fmaf(static_cast<float>(nbit::decode_value<BITS>(words[r], j)), scale[r], offset[r]);
#pragma unroll
          for (int m = 0; m < MROWS; ++m) {
            acc[r][m] = fmaf(w, xv[m], acc[r][m]);
          }
        }
      }
    }

    // Fixed-shape butterfly over the full warp, identical to `gemv`'s: the same
    // tree every launch, so the result does not depend on scheduling and a segment
    // gives the same bits here as it would there.
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
      for (int m = 0; m < MROWS; ++m) {
#pragma unroll
        for (int delta = kWarpSize / 2; delta > 0; delta >>= 1) {
          acc[r][m] += __shfl_down_sync(0xffffffffu, acc[r][m], delta, kWarpSize);
        }
      }
    }

    if (lane == 0) {
      for (int r = 0; r < rows_here; ++r) {
#pragma unroll
        for (int m = 0; m < MROWS; ++m) {
          const int token = base + m;
          if (token < stop) {
            out[static_cast<int64_t>(token) * out_features + row0 + r] =
                static_cast<scalar_t>(acc[r][m]);
          }
        }
      }
    }
  }
}

template <int MROWS, typename scalar_t>
void launch_grouped_m(const scalar_t* x, const uint32_t* packed, const scalar_t* scales,
                      const scalar_t* offsets, const int32_t* seg, scalar_t* out, int bits,
                      const nbit::RowGeometry& geom, int num_experts, int total_rows,
                      int out_features) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(static_cast<unsigned>(ceil_div(out_features, kRowsPerCta)),
                  static_cast<unsigned>(num_experts));

#define DYNQUANT_LAUNCH_GROUPED(BITS_VALUE)                                             \
  moe_grouped_gemv_kernel<BITS_VALUE, MROWS, scalar_t><<<grid, kMoeBlock, 0, stream>>>( \
      x, packed, scales, offsets, seg, out, total_rows, out_features, geom);            \
  break

  switch (bits) {
    case 2:
      DYNQUANT_LAUNCH_GROUPED(2);
    case 3:
      DYNQUANT_LAUNCH_GROUPED(3);
    case 4:
      DYNQUANT_LAUNCH_GROUPED(4);
    case 8:
      DYNQUANT_LAUNCH_GROUPED(8);
    default:
      TORCH_CHECK(false, "moe_grouped_gemv: unsupported bit width ", bits);
  }
#undef DYNQUANT_LAUNCH_GROUPED
  DYNQUANT_CHECK_LAUNCH();
}

// Smallest instantiated tile that can cover any segment. `total_rows` bounds every
// segment length -- the segments partition the rows -- and it is a *shape*, so this
// choice needs no knowledge of the routing and costs no synchronization. A model
// decoding one token to `k` experts arrives here with `total_rows == k`.
int row_tile(int64_t total_rows) {
  if (total_rows <= 1) return 1;
  if (total_rows <= 2) return 2;
  if (total_rows <= 4) return 4;
  return DYNQUANT_GEMV_MAX_ROWS;
}

template <typename scalar_t>
void launch_grouped(const at::Tensor& x, const at::Tensor& packed, const at::Tensor& scales,
                    const at::Tensor& offsets_or_empty, const at::Tensor& seg, at::Tensor& out,
                    int bits, const nbit::RowGeometry& geom, int num_experts, int total_rows,
                    int out_features, int tile, bool has_offsets) {
  const auto* x_ptr = x.data_ptr<scalar_t>();
  const auto* packed_ptr = reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>());
  const auto* scales_ptr = scales.data_ptr<scalar_t>();
  const scalar_t* offsets_ptr = has_offsets ? offsets_or_empty.data_ptr<scalar_t>() : nullptr;
  const auto* seg_ptr = seg.data_ptr<int32_t>();
  auto* out_ptr = out.data_ptr<scalar_t>();

  switch (tile) {
    case 1:
      launch_grouped_m<1, scalar_t>(x_ptr, packed_ptr, scales_ptr, offsets_ptr, seg_ptr, out_ptr,
                                    bits, geom, num_experts, total_rows, out_features);
      break;
    case 2:
      launch_grouped_m<2, scalar_t>(x_ptr, packed_ptr, scales_ptr, offsets_ptr, seg_ptr, out_ptr,
                                    bits, geom, num_experts, total_rows, out_features);
      break;
    case 4:
      launch_grouped_m<4, scalar_t>(x_ptr, packed_ptr, scales_ptr, offsets_ptr, seg_ptr, out_ptr,
                                    bits, geom, num_experts, total_rows, out_features);
      break;
    case 8:
      launch_grouped_m<8, scalar_t>(x_ptr, packed_ptr, scales_ptr, offsets_ptr, seg_ptr, out_ptr,
                                    bits, geom, num_experts, total_rows, out_features);
      break;
    default:
      TORCH_CHECK(false, "moe_grouped_gemv: no instantiation for a tile of ", tile);
  }
}

at::Tensor moe_grouped_gemv_cuda(const at::Tensor& x, const at::Tensor& packed,
                                 const at::Tensor& scales, const std::optional<at::Tensor>& offsets,
                                 const at::Tensor& seg_offsets, int64_t bits, int64_t group_values,
                                 int64_t in_features, int64_t out_features) {
  const at::cuda::CUDAGuard guard(packed.device());
  const auto geom = resolve_geometry("moe_grouped_gemv", packed, scales, offsets, bits,
                                     group_values, in_features);
  check_grouped(x, packed, scales, seg_offsets, in_features, out_features);

  const int64_t total_rows = x.size(0);
  const int64_t num_experts = seg_offsets.size(0) - 1;
  // Zeros and not `empty`: rows past the last segment are the expert-parallel
  // sentinels, no block writes them, and `torch.empty`'s contents reaching the next
  // projection is how one NaN becomes the whole layer.
  auto out = at::zeros({total_rows, out_features}, x.options());
  if (total_rows == 0 || out_features == 0 || num_experts == 0) {
    return out;
  }

  const at::Tensor offsets_tensor = offsets.has_value() ? *offsets : at::Tensor();
  const bool has_offsets = offsets.has_value();
  const int bits_i = static_cast<int>(bits);
  const int tile = row_tile(total_rows);
  const int experts_i = static_cast<int>(num_experts);
  const int rows_i = static_cast<int>(total_rows);
  const int out_i = static_cast<int>(out_features);
  const at::Tensor seg = seg_offsets.contiguous();

  AT_DISPATCH_SWITCH(
      x.scalar_type(), "dynquant::moe_grouped_gemv",
      AT_DISPATCH_CASE(at::kHalf,
                       [&] {
                         launch_grouped<scalar_t>(x, packed, scales, offsets_tensor, seg, out,
                                                  bits_i, geom, experts_i, rows_i, out_i, tile,
                                                  has_offsets);
                       })
          AT_DISPATCH_CASE(at::kBFloat16,
                           [&] {
                             launch_grouped<scalar_t>(x, packed, scales, offsets_tensor, seg, out,
                                                      bits_i, geom, experts_i, rows_i, out_i, tile,
                                                      has_offsets);
                           })
              AT_DISPATCH_CASE(at::kFloat, [&] {
                launch_grouped<scalar_t>(x, packed, scales, offsets_tensor, seg, out, bits_i, geom,
                                         experts_i, rows_i, out_i, tile,
                                         has_offsets);
              }));
  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(dynquant, CUDA, m) {
  m.impl("moe_grouped_gemv", TORCH_FN(moe_grouped_gemv_cuda));
}

}  // namespace dynquant
