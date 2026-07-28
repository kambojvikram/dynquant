// Decode GEMV: y = x @ W^T with W never leaving its packed form.
//
// This is the kernel the whole format exists for. At decode time M is 1 (or a
// handful, with speculative decoding or a small batch), so a matmul reads
// `out_features * in_features` weights to do `M * out_features * in_features`
// multiply-adds -- arithmetic intensity below 1 FLOP/byte, three orders of
// magnitude under what an A100 needs to be compute-bound. The run time is the time
// to stream the weights, and nothing else. Storing them at 3 bits instead of 16
// therefore makes decode roughly five times faster, and it is the only place in
// DynQuant where the compression ratio turns directly into a speedup.
//
// It is also the only place where the VRAM claim becomes true. `dequant ->
// cublasLt` (see dequant.cu) is the right answer for prefill, but it materialises
// the fp16 weight, so a model run entirely that way peaks at fp16 size. Here the
// packed buffer is read straight out of global memory into registers and the dense
// weight never exists.
//
// Two kernels, one shape of work
// ------------------------------
// `gemv_vec_kernel` is the fast path and runs for every weight a transformer
// actually contains. `gemv_kernel` is the general path: it accepts any geometry the
// format allows -- ragged per-row rows, groups too small to vectorize, unaligned
// storage -- and is what Mamba's 4-tap `conv1d` and the odd-sized tensors in the
// test matrix run through. Both decompose the problem identically (one warp owns
// `kRowsPerWarp` output rows and its lanes stride over that row's payload), so the
// launch geometry, the reduction and the output write are shared; they differ only
// in how much payload one lane takes at a time and how wide its loads are.
//
// Decomposition
// -------------
// One warp owns `kRowsPerWarp` consecutive output rows and its 32 lanes stride over
// the row's payload.
//
// The reason a warp takes several output rows rather than one is the activation.
// Every output row needs the same `x`, and reading it once per row would issue
// `M * K * N` loads in total. Holding a chunk's `x` values in registers across the
// warp's rows divides that by the row count outright. How many rows that should be
// is a measured number and not an argued one; see `kRowsPerWarp`.
//
// Why M is a template parameter
// -----------------------------
// `acc` and the per-value activation vector have to live in registers -- indexing
// them with a runtime bound spills the array to local memory and the kernel loses
// several times its speed. So M is compile-time, instantiated at {1, 2, 4, 8}, and
// a request for 3 or 5 rows is zero-padded up to the next instantiation. The
// padding costs one small copy of an `[M, K]` activation and happens essentially
// never: decode is M=1.
//
// Accumulation is fp32 throughout and the cross-lane reduction is a fixed-shape
// `__shfl_down` tree, so the result is bit-identical run to run. No atomics, no
// split-K.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cstdlib>

#include "dynquant/common.cuh"
#include "dynquant/geometry.h"
#include "dynquant/nbit.cuh"

namespace dynquant {
namespace {

// Four, and both neighbours were measured rather than assumed.
//
// A warp reads the whole of `x` to produce `kRowsPerWarp` outputs, so the
// activation is re-read `num_rows / kRowsPerWarp` times over a launch -- on the
// vocabulary projection at 2-bit that is 254 MB of activation traffic against
// 127 MB of weight, and the ratio halves with each doubling of the bit width, which
// is suggestively the order the measured shortfall falls in. Taking more rows per
// warp cuts that traffic; it also costs registers, and registers are warps.
//
// Eight loses and two loses, on `embed/lm_head` at M=1, percent of achievable read
// bandwidth at 2/3/4/8-bit:
//
//   rows=2  (48-56 reg)   34 / 50 / 65 / 98
//   rows=4  (64-72 reg)   36 / 52 / 67 / 99
//   rows=8  (96-128 reg)  36 / 49 / 64 / 97   + spills at MROWS >= 4
//
// So the activation re-read is not the binding constraint and neither is occupancy:
// the maximum is flat and in the middle, which is what an instruction-issue-bound
// kernel looks like. Do not re-tune this without a reason the SASS supports.
constexpr int kRowsPerWarp = 4;

constexpr int kWarpsPerCta = 4;
constexpr int kGemvBlock = kWarpSize * kWarpsPerCta;

// ---------------------------------------------------------------------------
// Vectorized path
// ---------------------------------------------------------------------------
//
// The general kernel below was correct and roughly four times slower than it had
// any business being, for a reason that is invisible in the source: it reads the
// activation one scalar at a time. Lane L owns the values `[kVals*L, kVals*L +
// kVals)`, so at any fixed position within the block the 32 lanes of a warp read 32
// two-byte activations spaced `kVals * 2` bytes apart. That single instruction
// touches sixteen 32-byte sectors and uses two bytes of each. The values are in L1
// -- the whole activation is 4 KB -- but L1 is charged per sector, not per useful
// byte, so the warp pays 8x the transactions it needs, and it pays them once per
// *value*, not once per word. Since the number of values is K regardless of bit
// width, the cost does not shrink when the weights do, which is exactly what the
// first benchmark showed: 2/3/4-bit all landed within 40 % of each other in
// absolute time while reading 1.4x-2x different amounts of weight.
//
// The fix is that a lane's values are *consecutive*, so its activations are a
// contiguous run and can be read with one 128-bit load. Across the warp those loads
// are then contiguous too, and the sixteen-sector access becomes four. That is the
// whole idea; the rest of this section is making the chunk sizes line up so it is
// legal.
//
// Chunk geometry
// --------------
// A chunk is the run of payload one lane takes per iteration. It has to satisfy two
// things at once:
//
//   * a whole number of *values*, so no value straddles the boundary between one
//     lane's chunk and the next (a straddling value would need a word neither lane
//     owns), and
//   * a whole number of *vector loads*, so the words can be read 128 or 64 bits at
//     a time rather than 32.
//
// For 2/4/8-bit a value never crosses a word, so 4 words -- one `uint4` -- already
// satisfies both. 3-bit is the interesting one: 32 bits is not a multiple of 3, so
// values straddle words, and the smallest run of words that is also a whole number
// of values is 3 (32 values in 96 bits). A `uint4` chunk would have to be 12 words
// -- lcm(4, 3) -- which is legal but coarse: at K=2048 with 128-value groups that
// is only 16 chunks in a row, so half of every warp would sit idle. 6 words is the
// next size down that is still value-aligned and still vectorizable, at 64 bits per
// load instead of 128. That is the trade this table makes, and it is why 3-bit is
// the one width that reads `uint2`.
//
//        BITS   words/chunk   values/chunk   load width
//           2             4             64      uint4
//           3             6             64      uint2
//           4             4             32      uint4
//           8             4             16      uint4
//
// `kValues * BITS == kWords * 32` for every row of that table, which is what makes
// the chunk value-aligned, and it also means the last value of a chunk ends exactly
// on the last bit of its last word -- so `decode_value` never reaches past the
// chunk and the straddle handling inside it stays in bounds without a guard.
//
// The host decides whether a tensor may use this path (`vec_geometry_ok`); anything
// that fails drops to the general kernel rather than to a slower special case here.
//
// What is left after the loads are fixed
// --------------------------------------
// Fixing the access pattern moved the 8-bit path to 98 % of achievable read
// bandwidth and left 2-bit at 35 %, which is the signature of a kernel that has
// stopped being memory-bound: from 8 bits to 2 the bytes fall 3.6x and the time
// falls 1.31x, so what remains costs per *value*, not per byte. The SASS said which
// instruction. Per value the loop ran one `I2F`, two `FFMA`, one `SHF` and one
// `LOP3`; sm_80 issues `FFMA` at 64 per SM per clock and a 32-bit integer
// conversion at 16, so that one conversion outweighed both multiply-adds. Hence the
// two things the inner loop does differently from the obvious version: it converts
// through `unsigned short` (`decoded_to_float`, full rate, and lossless because a
// value is at most 8 bits), and it applies `scale`/`offset` once per chunk instead
// of once per value.
//
// Those two are worth keeping and they are not worth much: together they cut
// inner-loop instructions from 1568 to 1384 (`FFMA` 512 -> 264) and bought **2.5 %
// at 2-bit, 7 % at 4-bit.** An 11.7 % instruction cut buying 2.5 % of time is the
// measurement that says what this kernel is actually bound by, and it is not the FP
// pipe. Per (value, row) the loop now issues about 5.4 instructions, which at 108 SMs
// x 4 schedulers x 1.41 GHz puts a pure issue-rate floor of ~141 us against the
// 237 us measured on the 2-bit vocabulary projection -- about 60 % issue efficiency.
// The other half is latency that cannot be hidden here: at K = 2048 with 64 values
// per chunk there are exactly 32 chunks in a row, one per lane, so **a warp executes
// exactly one iteration of the loop below** and there is no intra-warp
// memory-level parallelism at all. Latency is covered only by having other warps
// resident, which is why the register count in `kRowsPerWarp` matters more than the
// arithmetic does.
//
// So the next real multiple is not another peephole. It is to stop turning weights
// into floats one at a time: `LOP3`/`PRMT` two values straight into a `half2` and
// accumulate through `mma.sync`, which is the AWQ/Marlin route and is P7.
//
// One shortcut that does *not* work here, so it is not tried again: the magic-number
// trick, `__int_as_float(0x4B000000 | q)`, which produces a float with no conversion
// instruction at all. It needs a -2^23 bias folded into the group offset, and in fp32
// that makes `offset' = -(2^23 + z) * scale` -- magnitude ~8.4e6 * scale, storable to
// only 24 mantissa bits, so it carries 0.5 * scale of absolute error onto a result
// whose magnitude at 2-bit is at most 3 * scale. That is ~17 % error. Folding the
// correction into a per-chunk `xsum` term fails identically, since `2^23 * xsum`
// swamps `sum q*x`. Marlin and AWQ get away with it because they are in fp16, where
// the constant is 1024 and every step stays exact. An explicit per-value `FADD` would
// be exact but costs the same issue slot as the `I2F.U16` it replaces.

template <int BITS>
struct ChunkTraits;

template <>
struct ChunkTraits<2> {
  static constexpr int kWords = 4;
  static constexpr int kValues = 64;
  static constexpr int kLoadWords = 4;  // uint4
};

template <>
struct ChunkTraits<3> {
  static constexpr int kWords = 6;
  static constexpr int kValues = 64;
  static constexpr int kLoadWords = 2;  // uint2
};

template <>
struct ChunkTraits<4> {
  static constexpr int kWords = 4;
  static constexpr int kValues = 32;
  static constexpr int kLoadWords = 4;  // uint4
};

template <>
struct ChunkTraits<8> {
  static constexpr int kWords = 4;
  static constexpr int kValues = 16;
  static constexpr int kLoadWords = 4;  // uint4
};

// Host-side mirror of the table, so `vec_geometry_ok` and the launcher agree
// without either duplicating the numbers.
int chunk_words(int bits) {
  switch (bits) {
    case 3:
      return ChunkTraits<3>::kWords;
    case 2:
    case 4:
    case 8:
      return 4;
    default:
      return 0;
  }
}

int chunk_load_bytes(int bits) { return bits == 3 ? 8 : 16; }

// A 128-bit load reinterpreted as the element type. The union is what keeps it in
// registers: taking a pointer into a local `uint4` and casting it would put the
// variable in local memory, and the loop below reads every lane of it.
template <typename scalar_t>
union XPack {
  uint4 raw;
  scalar_t val[16 / sizeof(scalar_t)];
};

// One vector load, unpacked field by field into a plain `uint32_t[]`.
//
// The destination is written component-wise rather than through a `uint4*` cast:
// `decode_value` indexes the array with compile-time constants and so wants it in
// registers, and a store through a pointer into a local array is the classic way to
// force nvcc to give up and put the whole thing in local memory instead. Assigning
// `.x/.y/.z/.w` emits the same single `LDG.E.128` and keeps it in registers.
template <int LOAD_WORDS>
__device__ __forceinline__ void load_words(const uint32_t* __restrict__ src, uint32_t* dst);

template <>
__device__ __forceinline__ void load_words<4>(const uint32_t* __restrict__ src, uint32_t* dst) {
  const uint4 q = __ldg(reinterpret_cast<const uint4*>(src));
  dst[0] = q.x;
  dst[1] = q.y;
  dst[2] = q.z;
  dst[3] = q.w;
}

template <>
__device__ __forceinline__ void load_words<2>(const uint32_t* __restrict__ src, uint32_t* dst) {
  const uint2 q = __ldg(reinterpret_cast<const uint2*>(src));
  dst[0] = q.x;
  dst[1] = q.y;
}

template <int BITS, int MROWS, typename scalar_t>
__global__ void __launch_bounds__(kGemvBlock)
    gemv_vec_kernel(const scalar_t* __restrict__ x,        // [MROWS, K]
                    const uint32_t* __restrict__ packed,   // [N, words_per_row]
                    const scalar_t* __restrict__ scales,   // [N, num_groups]
                    const scalar_t* __restrict__ offsets,  // [N, num_groups] or null
                    scalar_t* __restrict__ out,            // [MROWS, N]
                    int num_rows, nbit::RowGeometry geom) {
  constexpr int kWords = ChunkTraits<BITS>::kWords;
  constexpr int kValues = ChunkTraits<BITS>::kValues;
  constexpr int kLoadWords = ChunkTraits<BITS>::kLoadWords;
  constexpr int kLoads = kWords / kLoadWords;
  constexpr int kXPerVec = 16 / sizeof(scalar_t);  // activations per 128-bit load
  constexpr int kSubs = kValues / kXPerVec;

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int row0 = (static_cast<int>(blockIdx.x) * kWarpsPerCta + warp) * kRowsPerWarp;
  if (row0 >= num_rows) {
    return;
  }
  const int rows_here = min(kRowsPerWarp, num_rows - row0);

  float acc[kRowsPerWarp][MROWS];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
    for (int m = 0; m < MROWS; ++m) {
      acc[r][m] = 0.0f;
    }
  }

  const int64_t stride_words = geom.words_per_row;
  const int64_t stride_meta = geom.num_groups;
  const int chunks_per_group = geom.words_per_group / kWords;
  const int chunks_per_row = geom.num_groups * chunks_per_group;

  for (int chunk = lane; chunk < chunks_per_row; chunk += kWarpSize) {
    const int group = chunk / chunks_per_group;
    const int chunk_in_group = chunk - group * chunks_per_group;
    const int word_base = group * geom.words_per_group + chunk_in_group * kWords;
    const int value_base = group * geom.group_values + chunk_in_group * kValues;

    uint32_t words[kRowsPerWarp][kWords];
    float scale[kRowsPerWarp];
    float offset[kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
      if (r < rows_here) {
        const uint32_t* src = packed + (row0 + r) * stride_words + word_base;
#pragma unroll
        for (int q = 0; q < kLoads; ++q) {
          load_words<kLoadWords>(src + q * kLoadWords, &words[r][q * kLoadWords]);
        }
        const int64_t meta = (row0 + r) * stride_meta + group;
        scale[r] = static_cast<float>(scales[meta]);
        offset[r] = offsets == nullptr ? 0.0f : static_cast<float>(offsets[meta]);
      } else {
        // Zeroed rather than left undefined: the accumulate below is unpredicated
        // on `r` so it can unroll, and a zero scale and offset make the surplus
        // rows contribute nothing to accumulators that are never read back.
#pragma unroll
        for (int w = 0; w < kWords; ++w) {
          words[r][w] = 0u;
        }
        scale[r] = 0.0f;
        offset[r] = 0.0f;
      }
    }

    // Dequant lifted out of the per-value loop.
    //
    // A chunk lies inside one group -- `words_per_group % kWords == 0` is one of
    // the conditions the host checks before choosing this kernel -- so `scale` and
    // `offset` are constant across it, and
    //
    //   sum_i (q_i * scale + offset) * x_i == scale * sum_i q_i x_i + offset * sum_i x_i
    //
    // The right-hand side costs one FFMA per value where the left costs two, and
    // pays the two remaining multiply-adds once per *chunk*. The activation sum
    // does not depend on the output row either, so the rows a warp owns share one
    // copy of it rather than each accumulating its own. It is also the more
    // accurate form: the reconstructed weight is never materialised, so it is
    // never rounded -- there is one rounding per value here against two before.
    float qacc[kRowsPerWarp][MROWS];
    float xsum[MROWS];
#pragma unroll
    for (int m = 0; m < MROWS; ++m) {
      xsum[m] = 0.0f;
#pragma unroll
      for (int r = 0; r < kRowsPerWarp; ++r) {
        qacc[r][m] = 0.0f;
      }
    }

    // No `v < in_features` test anywhere in here. The host only takes this path
    // when the groups tile `in_features` exactly, so there is no pad tail to skip
    // and every value a chunk covers is a real one.
#pragma unroll
    for (int s = 0; s < kSubs; ++s) {
      XPack<scalar_t> xv[MROWS];
#pragma unroll
      for (int m = 0; m < MROWS; ++m) {
        const int64_t off = static_cast<int64_t>(m) * geom.in_features + value_base + s * kXPerVec;
        xv[m].raw = __ldg(reinterpret_cast<const uint4*>(x + off));
      }
#pragma unroll
      for (int m = 0; m < MROWS; ++m) {
#pragma unroll
        for (int t = 0; t < kXPerVec; ++t) {
          xsum[m] += static_cast<float>(xv[m].val[t]);
        }
      }
#pragma unroll
      for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
        for (int t = 0; t < kXPerVec; ++t) {
          const float q =
              nbit::decoded_to_float(nbit::decode_value<BITS>(words[r], s * kXPerVec + t));
#pragma unroll
          for (int m = 0; m < MROWS; ++m) {
            qacc[r][m] = fmaf(q, static_cast<float>(xv[m].val[t]), qacc[r][m]);
          }
        }
      }
    }

    // Surplus rows carry a zero scale and a zero offset, so they fold in as zero
    // and the `r` loop stays unpredicated.
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
      for (int m = 0; m < MROWS; ++m) {
        acc[r][m] = fmaf(scale[r], qacc[r][m], fmaf(offset[r], xsum[m], acc[r][m]));
      }
    }
  }

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
        out[static_cast<int64_t>(m) * num_rows + row0 + r] = static_cast<scalar_t>(acc[r][m]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// General path
// ---------------------------------------------------------------------------
//
// Every geometry the format admits, including the ones the vectorized kernel
// refuses: rows whose groups do not tile `in_features`, groups too small to hold a
// vectorizable chunk, and per-row tensors whose final word is partly padding. One
// lane owns one *block* -- the smallest run of values occupying a whole number of
// words -- and reads it a word at a time.

template <int BITS, int MROWS, typename scalar_t>
__global__ void gemv_kernel(const scalar_t* __restrict__ x,        // [MROWS, K]
                            const uint32_t* __restrict__ packed,   // [N, words_per_row]
                            const scalar_t* __restrict__ scales,   // [N, num_groups]
                            const scalar_t* __restrict__ offsets,  // [N, num_groups] or null
                            scalar_t* __restrict__ out,            // [MROWS, N]
                            int num_rows, nbit::RowGeometry geom) {
  constexpr int kVals = nbit::Traits<BITS>::kValuesPerBlock;
  constexpr int kWords = nbit::Traits<BITS>::kWordsPerBlock;

  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = static_cast<int>(threadIdx.x) >> 5;
  const int row0 = (static_cast<int>(blockIdx.x) * kWarpsPerCta + warp) * kRowsPerWarp;
  if (row0 >= num_rows) {
    return;
  }
  // The tail warp can own fewer than kRowsPerWarp rows.
  const int rows_here = min(kRowsPerWarp, num_rows - row0);

  float acc[kRowsPerWarp][MROWS];
#pragma unroll
  for (int r = 0; r < kRowsPerWarp; ++r) {
#pragma unroll
    for (int m = 0; m < MROWS; ++m) {
      acc[r][m] = 0.0f;
    }
  }

  const int64_t stride_words = geom.words_per_row;
  const int64_t stride_meta = geom.num_groups;

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
        const int64_t row = row0 + r;
        nbit::load_block<BITS>(packed + row * stride_words + word_base, words[r], words_left);
        const int64_t meta = row * stride_meta + group;
        scale[r] = static_cast<float>(scales[meta]);
        offset[r] = offsets == nullptr ? 0.0f : static_cast<float>(offsets[meta]);
      } else {
        // Zeroed rather than left undefined: the accumulate below is unpredicated on
        // `r` so it can unroll, and a zero scale and offset make the surplus rows
        // contribute nothing to accumulators that are never read back.
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
      // Past `in_features` is the zero pad that rounded the row up to a whole
      // group. Its weights are real stored codes, but there is no activation to
      // pair them with, so the term does not merely round to nothing -- it does
      // not exist.
      if (v >= geom.in_features) {
        continue;
      }
      float xv[MROWS];
#pragma unroll
      for (int m = 0; m < MROWS; ++m) {
        xv[m] = static_cast<float>(__ldg(x + static_cast<int64_t>(m) * geom.in_features + v));
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

  // Fixed-shape butterfly over the full warp: the same tree every launch, so the
  // summation order -- and therefore the result, bit for bit -- does not depend on
  // scheduling.
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
        out[static_cast<int64_t>(m) * num_rows + row0 + r] = static_cast<scalar_t>(acc[r][m]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

// May this tensor use the vectorized kernel? Everything the fast path assumes is
// checked here once on the host, so the kernel itself carries no guards.
//
// `elem_size` is the activation's, because the activation is loaded 16 bytes at a
// time and where those 16 bytes start depends on how big an element is.
bool vec_geometry_ok(int bits, const nbit::RowGeometry& geom, int64_t elem_size,
                     const void* packed_ptr, const void* x_ptr) {
  const int words = chunk_words(bits);
  if (words == 0) {
    return false;
  }
  // Groups must tile `in_features` exactly: the fast path has no pad-tail test, so
  // a row that rounds up to a whole group would multiply real weights by
  // activations that do not exist.
  if (static_cast<int64_t>(geom.num_groups) * geom.group_values != geom.in_features) {
    return false;
  }
  // ...and chunks must tile a group exactly, or a chunk would straddle a group
  // boundary and need two scales.
  if (geom.words_per_group % words != 0) {
    return false;
  }
  const int64_t x_per_vec = 16 / elem_size;
  // Every activation load is 16 bytes starting at a multiple of `kXPerVec`
  // elements from the row base, and row m starts at `m * in_features`.
  if (geom.group_values % x_per_vec != 0 || geom.in_features % x_per_vec != 0) {
    return false;
  }
  // Base pointers. Torch's allocator hands out 512-byte alignment, but a view or a
  // user-supplied external tensor need not, and a misaligned vector load is a
  // fault, not a slow path.
  const auto packed_addr = reinterpret_cast<uintptr_t>(packed_ptr);
  const auto x_addr = reinterpret_cast<uintptr_t>(x_ptr);
  if (packed_addr % chunk_load_bytes(bits) != 0 || x_addr % 16 != 0) {
    return false;
  }
  return true;
}

// Escape hatch for A/B measurement: `DYNQUANT_GEMV_SCALAR=1` forces the general
// kernel. Read once. The two paths agree to fp32 accumulation noise, not bit for
// bit -- they sum a row in a different order -- so this is a benchmarking control,
// not a correctness switch.
bool vectorized_enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("DYNQUANT_GEMV_SCALAR");
    return !(v != nullptr && v[0] != '\0' && v[0] != '0');
  }();
  return enabled;
}

template <int MROWS, typename scalar_t>
void launch_gemv_m(const scalar_t* x, const uint32_t* packed, const scalar_t* scales,
                   const scalar_t* offsets, scalar_t* out, int bits, const nbit::RowGeometry& geom,
                   int num_rows, bool use_vec) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  // Both kernels cover `kRowsPerWarp` rows per warp, so one grid serves either. If
  // that ever stops being true the grid has to become path-dependent, and the failure
  // is silent: too small a grid leaves the tail of the output untouched rather than
  // faulting.
  const int warps_needed = ceil_div(num_rows, kRowsPerWarp);
  const int grid = ceil_div(warps_needed, kWarpsPerCta);

#define DYNQUANT_LAUNCH(BITS_VALUE)                                                          \
  if (use_vec) {                                                                             \
    gemv_vec_kernel<BITS_VALUE, MROWS, scalar_t>                                             \
        <<<grid, kGemvBlock, 0, stream>>>(x, packed, scales, offsets, out, num_rows, geom);  \
  } else {                                                                                   \
    gemv_kernel<BITS_VALUE, MROWS, scalar_t>                                                 \
        <<<grid, kGemvBlock, 0, stream>>>(x, packed, scales, offsets, out, num_rows, geom);  \
  }                                                                                          \
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
      TORCH_CHECK(false, "gemv: unsupported bit width ", bits);
  }
#undef DYNQUANT_LAUNCH
  DYNQUANT_CHECK_LAUNCH();
}

template <typename scalar_t>
void launch_gemv(const at::Tensor& x, const at::Tensor& packed, const at::Tensor& scales,
                 const at::Tensor& offsets_or_empty, at::Tensor& out, int bits,
                 const nbit::RowGeometry& geom, int num_rows, int m_pad, bool has_offsets) {
  const auto* x_ptr = x.data_ptr<scalar_t>();
  const auto* packed_ptr = reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>());
  const auto* scales_ptr = scales.data_ptr<scalar_t>();
  const scalar_t* offsets_ptr = has_offsets ? offsets_or_empty.data_ptr<scalar_t>() : nullptr;
  auto* out_ptr = out.data_ptr<scalar_t>();

  const bool use_vec =
      vectorized_enabled() &&
      vec_geometry_ok(bits, geom, static_cast<int64_t>(sizeof(scalar_t)), packed_ptr, x_ptr);

  switch (m_pad) {
    case 1:
      launch_gemv_m<1, scalar_t>(x_ptr, packed_ptr, scales_ptr, offsets_ptr, out_ptr, bits, geom,
                                 num_rows, use_vec);
      break;
    case 2:
      launch_gemv_m<2, scalar_t>(x_ptr, packed_ptr, scales_ptr, offsets_ptr, out_ptr, bits, geom,
                                 num_rows, use_vec);
      break;
    case 4:
      launch_gemv_m<4, scalar_t>(x_ptr, packed_ptr, scales_ptr, offsets_ptr, out_ptr, bits, geom,
                                 num_rows, use_vec);
      break;
    case 8:
      launch_gemv_m<8, scalar_t>(x_ptr, packed_ptr, scales_ptr, offsets_ptr, out_ptr, bits, geom,
                                 num_rows, use_vec);
      break;
    default:
      TORCH_CHECK(false, "gemv: no instantiation for ", m_pad, " activation rows");
  }
}

// Smallest instantiated M that fits `m`. Padding up costs one [M, K] copy; the
// alternative -- a runtime M -- costs the register residency of `acc`.
int padded_rows(int64_t m) {
  if (m <= 1) return 1;
  if (m <= 2) return 2;
  if (m <= 4) return 4;
  return 8;
}

at::Tensor gemv_cuda(const at::Tensor& x, const at::Tensor& packed, const at::Tensor& scales,
                     const std::optional<at::Tensor>& offsets, int64_t bits, int64_t group_values,
                     int64_t in_features) {
  const at::cuda::CUDAGuard guard(packed.device());
  const auto geom =
      resolve_geometry("gemv", packed, scales, offsets, bits, group_values, in_features);

  TORCH_CHECK(x.dim() == 2, "gemv: x must be 2-D [rows, in_features], got ", x.sizes());
  TORCH_CHECK(x.size(1) == in_features, "gemv: x has ", x.size(1), " columns but in_features is ",
              in_features);
  TORCH_CHECK(x.is_contiguous(), "gemv: x must be contiguous");
  TORCH_CHECK(x.scalar_type() == scales.scalar_type(), "gemv: x dtype ", x.scalar_type(),
              " != scales dtype ", scales.scalar_type());
  TORCH_CHECK(x.device() == packed.device(), "gemv: x is on ", x.device(), " but the weight is on ",
              packed.device());
  TORCH_CHECK(x.size(0) <= DYNQUANT_GEMV_MAX_ROWS, "gemv: ", x.size(0),
              " activation rows exceeds the kernel's limit of ", DYNQUANT_GEMV_MAX_ROWS,
              ". Above this a quantized matmul is compute-bound and the dequant + cuBLASLt "
              "path is faster; dispatch there instead.");

  const int64_t m = x.size(0);
  const int num_rows = static_cast<int>(packed.size(0));
  if (m == 0 || num_rows == 0) {
    return at::empty({m, packed.size(0)}, x.options());
  }

  const int m_pad = padded_rows(m);
  at::Tensor x_in = x;
  if (m_pad != m) {
    x_in = at::zeros({m_pad, in_features}, x.options());
    x_in.narrow(0, 0, m).copy_(x);
  }
  auto out = at::empty({m_pad, packed.size(0)}, x.options());

  const at::Tensor offsets_tensor = offsets.has_value() ? *offsets : at::Tensor();
  const bool has_offsets = offsets.has_value();
  const int bits_i = static_cast<int>(bits);

  AT_DISPATCH_SWITCH(x.scalar_type(), "dynquant::gemv",
                     AT_DISPATCH_CASE(at::kHalf,
                                      [&] {
                                        launch_gemv<scalar_t>(x_in, packed, scales, offsets_tensor,
                                                              out, bits_i, geom, num_rows, m_pad,
                                                              has_offsets);
                                      })
                         AT_DISPATCH_CASE(at::kBFloat16,
                                          [&] {
                                            launch_gemv<scalar_t>(x_in, packed, scales,
                                                                  offsets_tensor, out, bits_i, geom,
                                                                  num_rows, m_pad, has_offsets);
                                          })
                             AT_DISPATCH_CASE(at::kFloat, [&] {
                               launch_gemv<scalar_t>(x_in, packed, scales, offsets_tensor, out,
                                                     bits_i, geom, num_rows, m_pad, has_offsets);
                             }));

  // `contiguous()` rather than returning the narrow view: an op that returns a view
  // of a larger allocation keeps the padding alive for as long as the result is
  // referenced, and makes the output's storage size depend on M in a way the meta
  // kernel does not model.
  return m_pad == m ? out : out.narrow(0, 0, m).contiguous();
}

}  // namespace

TORCH_LIBRARY_IMPL(dynquant, CUDA, m) { m.impl("gemv", TORCH_FN(gemv_cuda)); }

}  // namespace dynquant
