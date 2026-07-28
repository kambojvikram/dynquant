// Tile dequantization: packed words -> a dense fp16/bf16 matrix.
//
// What this kernel is for
// -----------------------
// Two things, and they pull in the same direction.
//
// 1. **Prefill.** At large M a quantized GEMM is compute-bound, so the weight read
//    is amortised over hundreds of rows of activations and there is nothing to win
//    by keeping the weight packed *inside* the matmul. Dequantizing a weight into a
//    bounded workspace and handing it to cuBLASLt gets fp16 tensor-core throughput
//    immediately and, being one kernel plus one library call, is the version whose
//    correctness is not in question. The fused CUTLASS path (P7) has to beat this
//    to earn its place; until it does, this is the fast path, not a placeholder.
//
// 2. **Embedding lookup.** `nn.Embedding` gathers a few thousand rows out of a
//    151k-row table. Dequantizing just those rows is ~0.01% of the work of
//    dequantizing the table, so the packed table stays in VRAM for the whole run
//    and the gather costs almost nothing. That is the difference between a 2B model
//    whose 508M-parameter tied embedding is quantized on disk only, and one where
//    it is quantized in memory too.
//
// Parallel decomposition
// ----------------------
// One thread per *block* of values (see nbit.cuh), grid-stride over
// `num_rows * blocks_per_row`. Each thread loads its block's words once, decodes in
// registers, and writes `values_per_block` contiguous outputs -- so a warp writes
// 32 * values_per_block contiguous elements and both the read and the write are
// fully coalesced.
//
// Scales and offsets are read once per block rather than once per value. The block
// cannot straddle a group boundary, so this is exact, not an approximation.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <algorithm>

#include "dynquant/common.cuh"
#include "dynquant/geometry.h"
#include "dynquant/nbit.cuh"

namespace dynquant {
namespace {

template <int BITS, typename scalar_t>
__global__ void dequant_kernel(const uint32_t* __restrict__ packed,
                               const scalar_t* __restrict__ scales,
                               const scalar_t* __restrict__ offsets,  // may be null
                               scalar_t* __restrict__ out, int num_rows,
                               nbit::RowGeometry geom) {
  constexpr int kVals = nbit::Traits<BITS>::kValuesPerBlock;

  const int64_t total = static_cast<int64_t>(num_rows) * geom.blocks_per_row;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

  for (int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; idx < total;
       idx += stride) {
    const int row = static_cast<int>(idx / geom.blocks_per_row);
    const int block = static_cast<int>(idx - static_cast<int64_t>(row) * geom.blocks_per_row);

    const int group = block / geom.blocks_per_group;
    const int block_in_group = block - group * geom.blocks_per_group;

    // Where this block's words start, and how many of the row's words are left
    // after that point -- the second number only bites on a ragged per-row tail.
    const int word_base = group * geom.words_per_group + (block_in_group * kVals * BITS) / 32;
    const int64_t row_words = static_cast<int64_t>(row) * geom.words_per_row;

    uint32_t words[nbit::Traits<BITS>::kWordsPerBlock];
    nbit::load_block<BITS>(packed + row_words + word_base, words, geom.words_per_row - word_base);

    float decoded[kVals];
    nbit::decode_block<BITS>(words, decoded);

    const int64_t meta = static_cast<int64_t>(row) * geom.num_groups + group;
    const float scale = static_cast<float>(scales[meta]);
    const float offset = offsets == nullptr ? 0.0f : static_cast<float>(offsets[meta]);

    const int value_base = group * geom.group_values + block_in_group * kVals;
    scalar_t* dst = out + static_cast<int64_t>(row) * geom.in_features + value_base;

#pragma unroll
    for (int j = 0; j < kVals; ++j) {
      // The pad region of the last group has no home in the output, and neither
      // does the ragged tail of a per-row row. Both are the same test.
      if (value_base + j < geom.in_features) {
        dst[j] = static_cast<scalar_t>(fmaf(decoded[j], scale, offset));
      }
    }
  }
}

template <typename scalar_t>
void launch_dequant(const at::Tensor& packed, const at::Tensor& scales,
                    const at::Tensor& offsets_or_empty, at::Tensor& out, int bits,
                    const nbit::RowGeometry& geom, int num_rows, bool has_offsets) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* packed_ptr = reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>());
  const auto* scales_ptr = scales.data_ptr<scalar_t>();
  const scalar_t* offsets_ptr = has_offsets ? offsets_or_empty.data_ptr<scalar_t>() : nullptr;
  auto* out_ptr = out.data_ptr<scalar_t>();

  const int64_t total = static_cast<int64_t>(num_rows) * geom.blocks_per_row;
  const int block_dim = kDefaultBlock;
  const int grid = grid_for(static_cast<int>(std::min<int64_t>(total, 1 << 30)), block_dim);

#define DYNQUANT_LAUNCH(BITS_VALUE)                                                       \
  dequant_kernel<BITS_VALUE, scalar_t><<<grid, block_dim, 0, stream>>>(                   \
      packed_ptr, scales_ptr, offsets_ptr, out_ptr, num_rows, geom);                      \
  break

  switch (bits) {
    case 2:
      DYNQUANT_LAUNCH(2);
    case 3:
      DYNQUANT_LAUNCH(3);
    case 4:
      DYNQUANT_LAUNCH(4);
    case 8:
      DYNQUANT_LAUNCH(8);
    default:
      TORCH_CHECK(false, "dequant: unsupported bit width ", bits);
  }
#undef DYNQUANT_LAUNCH
  DYNQUANT_CHECK_LAUNCH();
}

at::Tensor dequant_cuda(const at::Tensor& packed, const at::Tensor& scales,
                        const std::optional<at::Tensor>& offsets, int64_t bits,
                        int64_t group_values, int64_t in_features) {
  const at::cuda::CUDAGuard guard(packed.device());
  const auto geom = resolve_geometry("dequant", packed, scales, offsets, bits, group_values,
                                     in_features);
  const int num_rows = static_cast<int>(packed.size(0));

  auto out = at::empty({packed.size(0), in_features}, scales.options());
  if (out.numel() == 0) {
    return out;
  }

  const at::Tensor offsets_tensor = offsets.has_value() ? *offsets : at::Tensor();
  const bool has_offsets = offsets.has_value();
  const int bits_i = static_cast<int>(bits);

  AT_DISPATCH_SWITCH(scales.scalar_type(), "dynquant::dequant",
                     AT_DISPATCH_CASE(at::kHalf,
                                      [&] {
                                        launch_dequant<scalar_t>(packed, scales, offsets_tensor,
                                                                 out, bits_i, geom, num_rows,
                                                                 has_offsets);
                                      })
                         AT_DISPATCH_CASE(at::kBFloat16,
                                          [&] {
                                            launch_dequant<scalar_t>(packed, scales, offsets_tensor,
                                                                     out, bits_i, geom, num_rows,
                                                                     has_offsets);
                                          })
                             AT_DISPATCH_CASE(at::kFloat, [&] {
                               launch_dequant<scalar_t>(packed, scales, offsets_tensor, out, bits_i,
                                                        geom, num_rows, has_offsets);
                             }));
  return out;
}

}  // namespace

TORCH_LIBRARY_IMPL(dynquant, CUDA, m) { m.impl("dequant", TORCH_FN(dequant_cuda)); }

}  // namespace dynquant
