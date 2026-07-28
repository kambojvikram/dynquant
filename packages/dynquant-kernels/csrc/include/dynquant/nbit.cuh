// Device-side n-bit decoding: the one place that knows the packed layout.
//
// Every kernel in this extension reads weights through this header, so there is a
// single definition of "where is value j" rather than one per kernel that drifts.
// It mirrors `dynquant/quant/pack.py` exactly, and `tests/test_kernels_parity.py`
// compares the two on real tensors -- a divergence here does not raise, it
// produces a weight matrix of the right shape full of wrong numbers.
//
// The layout, restated so this file is readable on its own
// --------------------------------------------------------
// A row of `padded_in_features` values is split into `num_groups` groups of
// `group_values` each. Group g of row r occupies `words_per_group` consecutive
// uint32 words starting at `packed[r * words_per_row + g * words_per_group]`, and
// within a group value j sits at bit offset `j * BITS`, LSB-first.
//
// `group_values % 32 == 0` is what makes this work: `group_values * BITS` is then
// a multiple of 32 for every BITS in {2,3,4,8}, so a group is a whole number of
// words and the next group starts on a word boundary. The one exemption is per-row
// grouping (`num_groups == 1`), where the row simply rounds up to a whole number of
// words and the final word carries unused high bits -- legal because there is no
// next group whose start could be knocked off alignment.
//
// The block abstraction
// ---------------------
// Rather than have each thread decode one value -- which would make every thread
// re-load the same word and turn a 4-byte access into a 32-way broadcast -- threads
// work in *blocks*: the smallest run of values that occupies a whole number of
// words.
//
//     BITS   values/block   words/block
//        2             16             1
//        3             32             3
//        4              8             1
//        8              4             1
//
// One thread owns one block, loads its words once, and unpacks them in registers
// with compile-time shifts. A warp therefore reads 32 consecutive words (128 bytes,
// one fully coalesced transaction) for the 1-word widths.
//
// Blocks never straddle a group boundary: values/block divides 32, and
// `group_values` is a multiple of 32, so `group_values % values_per_block == 0`.
// That is what lets scale and offset be loaded once per block rather than per value.
// The exemption is again per-row grouping, where the single group is the whole row
// and there is no boundary to straddle; its last block may be ragged, which
// `load_block` handles.

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

#include "dynquant/abi.h"
// `RowGeometry` lives in the pure-C++ header so the CPU reference implementations
// in bindings.cpp share it. One struct, one resolver, both backends.
#include "dynquant/geometry.h"

namespace dynquant {
namespace nbit {

// --------------------------------------------------------------------------
// Per-width block geometry
// --------------------------------------------------------------------------

template <int BITS>
struct Traits;

template <>
struct Traits<2> {
  static constexpr int kBits = 2;
  static constexpr int kValuesPerBlock = 16;
  static constexpr int kWordsPerBlock = 1;
};

template <>
struct Traits<3> {
  static constexpr int kBits = 3;
  // 32 values in 96 bits. Exactly two of them straddle a word boundary (indices
  // 10 and 21); `decode_block` handles that uniformly rather than special-casing,
  // so there is no separate 3-bit path to keep in step with the others.
  static constexpr int kValuesPerBlock = 32;
  static constexpr int kWordsPerBlock = 3;
};

template <>
struct Traits<4> {
  static constexpr int kBits = 4;
  static constexpr int kValuesPerBlock = 8;
  static constexpr int kWordsPerBlock = 1;
};

template <>
struct Traits<8> {
  static constexpr int kBits = 8;
  static constexpr int kValuesPerBlock = 4;
  static constexpr int kWordsPerBlock = 1;
};

// --------------------------------------------------------------------------
// Decoding
// --------------------------------------------------------------------------

// One value out of a block's words. `j` is the index within the block, and it is
// always a compile-time constant at the call sites below, so the word index, the
// shift and the straddle test all fold away and the emitted code is a shift, an
// and, and (for the two 3-bit stragglers) a shift-or.
template <int BITS>
__device__ __forceinline__ uint32_t decode_value(const uint32_t* __restrict__ words, int j) {
  constexpr uint32_t kMask = (1u << BITS) - 1u;
  const int bit = j * BITS;
  const int w = bit >> 5;
  const int s = bit & 31;
  uint32_t v = words[w] >> s;
  // Only reachable when `s + BITS > 32`, i.e. never for 2/4/8-bit (32 % BITS == 0)
  // and for exactly two j at 3-bit. `32 - s >= BITS` is compile-time false only in
  // those cases, so nvcc deletes the branch everywhere else.
  const int fits = 32 - s;
  if (fits < BITS) {
    v |= words[w + 1] << fits;
  }
  return v & kMask;
}

// A decoded value as a float, converted the cheap way.
//
// The narrowing looks like a no-op and is not. A decoded value is at most 8 bits,
// so `unsigned short` holds it exactly and the two casts are together lossless --
// but they emit `I2F.F32.U16` instead of `I2F.F32.U32`, and on sm_80 those are not
// the same instruction. The programming guide rates conversions *from* 8- and
// 16-bit integers at 64 results per SM per clock and every other conversion,
// including from the `uint32_t` this decodes into, at 16. A quarter-rate
// instruction executed once per weight is more issue slots than the two FFMAs
// that consume it, which is exactly what the SASS showed: 257 `I2F` against 512
// `FFMA` in the 2-bit inner loop, and a kernel whose runtime barely moved when the
// weights got smaller.
__device__ __forceinline__ float decoded_to_float(uint32_t v) {
  return static_cast<float>(static_cast<unsigned short>(v));
}

// Decode a whole block into `out[0 .. kValuesPerBlock)` as floats.
//
// Fully unrolled, so every shift above is a literal. Returns floats rather than
// integers because every consumer immediately multiplies by a float scale, and
// converting here lets the compiler schedule the conversion alongside the next
// load.
template <int BITS>
__device__ __forceinline__ void decode_block(const uint32_t* __restrict__ words, float* out) {
#pragma unroll
  for (int j = 0; j < Traits<BITS>::kValuesPerBlock; ++j) {
    out[j] = decoded_to_float(decode_value<BITS>(words, j));
  }
}

// Load a block's words through the read-only data path.
//
// `__ldg` maps to `ld.global.nc`, which routes through the texture/L1 cache and,
// more importantly, tells the compiler the data cannot alias the output -- without
// it nvcc must assume a store to `out` invalidates the weights and re-issues the
// loads on every iteration of the caller's loop.
//
// `words_available` counts the words remaining in the row from `base`. It is
// unbounded in the aligned case -- `group_values * BITS` is a whole number of words
// so every block is fully backed -- and matters only for the last block of a
// per-row tensor, where the row rounds up to a whole number of words and a block
// can reach past the end. Mamba's `conv1d.weight` is the real instance: 4 values at
// 3-bit occupy one word, while a 3-bit block spans three. Reading the two words
// past the end would be an out-of-bounds global access, so they read as zero, and
// the values they would have carried are discarded by the caller's `v <
// in_features` test anyway.
//
// The comparison is not free but it is perfectly predicted: for every aligned
// tensor it is true on every block of every row.
template <int BITS>
__device__ __forceinline__ void load_block(const uint32_t* __restrict__ base, uint32_t* words,
                                           int words_available) {
  if (words_available >= Traits<BITS>::kWordsPerBlock) {
#pragma unroll
    for (int w = 0; w < Traits<BITS>::kWordsPerBlock; ++w) {
      words[w] = __ldg(base + w);
    }
  } else {
#pragma unroll
    for (int w = 0; w < Traits<BITS>::kWordsPerBlock; ++w) {
      words[w] = (w < words_available) ? __ldg(base + w) : 0u;
    }
  }
}

// `RowGeometry` is defined in geometry.h and resolved on the host by
// `dynquant::resolve_geometry`, so no kernel contains a branch on grouping mode:
// the per-row sentinel is gone by launch time and `num_groups == 1` is not a
// special case at all -- it is a group that happens to span the row.

}  // namespace nbit
}  // namespace dynquant
