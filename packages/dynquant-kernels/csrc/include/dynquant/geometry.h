// Host-side validation of a packed tensor's geometry. Shared by CPU and CUDA.
//
// Every kernel entry point starts here, and it is the only place that turns the
// (packed, scales, offsets, bits, group_values, in_features) argument tuple into
// sizes. That matters more than it looks: the research code re-derived the group
// size from `scale.numel() // out_features` at load time, guessed 128 when the
// division did not come out, and dequantized at the wrong stride when the guess was
// wrong -- producing a tensor of the right shape full of wrong numbers, with
// nothing raised. Nothing is derived by guesswork here. Everything is either passed
// in or implied by a shape, and every implication is checked.
//
// Why `group_values` and not `group_size`
// ---------------------------------------
// The Python format carries `group_size == -1` to mean "one group spanning the
// row", the exemption that lets embeddings and Mamba's 4-tap `conv1d` be packed at
// all. Resolving that sentinel is the caller's job, so `group_values` arriving here
// is always a real count. No kernel then contains a branch on grouping mode:
// per-row is simply `num_groups == 1`.

#pragma once

#include <ATen/ATen.h>

#include <optional>

#include "dynquant/abi.h"

namespace dynquant {
namespace nbit {

// Values packed into one block, per width. Mirrors `dynquant.constants.VALUES_PER_WORD`.
inline int values_per_block(int64_t bits) {
  switch (bits) {
    case 2:
      return 16;
    case 3:
      return 32;
    case 4:
      return 8;
    case 8:
      return 4;
    default:
      return 0;
  }
}

struct RowGeometry {
  int group_values;      ///< Values per group (== in_features when per-row).
  int num_groups;        ///< Groups per row.
  int words_per_group;   ///< Words one group occupies. Rounded up when per-row.
  int words_per_row;     ///< num_groups * words_per_group.
  int in_features;       ///< Real values per row, before padding. The only bound
                         ///< that decides what contributes to a result.
  int blocks_per_group;  ///< ceil(group_values / values_per_block); exact unless per-row.
  int blocks_per_row;    ///< num_groups * blocks_per_group.
};

}  // namespace nbit

// Validate and resolve. Raises rather than returning a status: there is no
// sensible partial answer, and a caller that ignored one would be dequantizing at
// the wrong stride.
inline nbit::RowGeometry resolve_geometry(const char* op, const at::Tensor& packed,
                                          const at::Tensor& scales,
                                          const std::optional<at::Tensor>& offsets, int64_t bits,
                                          int64_t group_values, int64_t in_features) {
  const int vpb = nbit::values_per_block(bits);
  TORCH_CHECK(vpb > 0, op, ": unsupported bit width ", bits, "; expected one of {",
              DYNQUANT_BITS_LIST, "}");
  TORCH_CHECK(packed.dim() == 2, op, ": packed must be 2-D [rows, words], got ", packed.sizes());
  TORCH_CHECK(packed.scalar_type() == at::kInt, op,
              ": packed must be int32 (unsigned bit patterns stored as int32), got ",
              packed.scalar_type());
  TORCH_CHECK(scales.dim() == 2, op, ": scales must be 2-D [rows, groups], got ", scales.sizes());
  TORCH_CHECK(packed.size(0) == scales.size(0), op, ": packed has ", packed.size(0),
              " rows but scales has ", scales.size(0));
  TORCH_CHECK(packed.is_contiguous() && scales.is_contiguous(), op,
              ": packed and scales must be contiguous");
  if (offsets.has_value()) {
    TORCH_CHECK(offsets->sizes() == scales.sizes(), op, ": offsets shape ", offsets->sizes(),
                " != scales shape ", scales.sizes());
    TORCH_CHECK(offsets->scalar_type() == scales.scalar_type(), op,
                ": offsets dtype ", offsets->scalar_type(), " != scales dtype ",
                scales.scalar_type());
    TORCH_CHECK(offsets->is_contiguous(), op, ": offsets must be contiguous");
  }
  TORCH_CHECK(in_features > 0, op, ": in_features must be positive, got ", in_features);
  TORCH_CHECK(group_values > 0, op,
              ": group_values must be a resolved positive count, got ", group_values,
              ". The per-row sentinel (-1) is the caller's to resolve.");

  const int64_t num_groups = scales.size(1);
  const int64_t words_per_row = packed.size(1);
  TORCH_CHECK(num_groups > 0, op, ": scales has zero groups");
  TORCH_CHECK(words_per_row % num_groups == 0, op, ": ", words_per_row,
              " words per row is not divisible by ", num_groups,
              " groups; the checkpoint is inconsistent");
  const int64_t words_per_group = words_per_row / num_groups;

  // A group must fit in its words. Equality for aligned groups; per-row (one group)
  // may round up and leave unused high bits in the final word.
  const int64_t group_bits = group_values * bits;
  TORCH_CHECK(words_per_group * 32 >= group_bits, op, ": a group of ", group_values, " values at ",
              bits, "-bit needs ", (group_bits + 31) / 32, " words but only ", words_per_group,
              " are stored");
  if (num_groups > 1) {
    TORCH_CHECK(group_bits % 32 == 0, op, ": group_values=", group_values, " at ", bits,
                "-bit occupies ", group_bits,
                " bits, not a whole number of words, so group 1 would begin mid-word. "
                "group_values must be a multiple of ",
                DYNQUANT_GROUP_SIZE_ALIGNMENT, " unless the whole row is one group.");
    TORCH_CHECK(words_per_group * 32 == group_bits, op, ": ", words_per_group,
                " words per group does not match ", group_bits, " bits of payload");
    TORCH_CHECK(num_groups * group_values >= in_features, op, ": ", num_groups, " groups of ",
                group_values, " cover ", num_groups * group_values, " values, short of in_features=",
                in_features);
  } else {
    TORCH_CHECK(group_values == in_features, op,
                ": a single group must span the row exactly; got group_values=", group_values,
                " for in_features=", in_features);
  }

  nbit::RowGeometry geom{};
  geom.group_values = static_cast<int>(group_values);
  geom.num_groups = static_cast<int>(num_groups);
  geom.words_per_group = static_cast<int>(words_per_group);
  geom.words_per_row = static_cast<int>(words_per_row);
  geom.in_features = static_cast<int>(in_features);
  geom.blocks_per_group = static_cast<int>((group_values + vpb - 1) / vpb);
  geom.blocks_per_row = geom.num_groups * geom.blocks_per_group;
  return geom;
}

// The extra contract a grouped MoE call has on top of one packed tensor.
//
// Separate from `resolve_geometry` because it is about a *bank*: the relation
// between the segment table, the number of experts, and the fact that the packed
// rows are E banks of `out_features` stacked, not one matrix. Shared by the CPU
// reference and the CUDA kernel so that the two cannot come to disagree about what
// they accept -- the failure this campaign has now found in nine places, where a
// second copy of a rule agrees with the first until the case that separates them.
inline void check_grouped(const at::Tensor& x, const at::Tensor& packed, const at::Tensor& scales,
                          const at::Tensor& seg_offsets, int64_t in_features,
                          int64_t out_features) {
  const char* op = "moe_grouped_gemv";
  TORCH_CHECK(x.dim() == 2, op, ": x must be 2-D [total_rows, in_features], got ", x.sizes());
  TORCH_CHECK(x.size(1) == in_features, op, ": x has ", x.size(1),
              " columns but in_features is ", in_features);
  TORCH_CHECK(x.is_contiguous(), op, ": x must be contiguous");
  TORCH_CHECK(x.scalar_type() == scales.scalar_type(), op, ": x dtype ", x.scalar_type(),
              " != scales dtype ", scales.scalar_type());
  TORCH_CHECK(out_features > 0, op, ": out_features must be positive, got ", out_features);
  TORCH_CHECK(seg_offsets.dim() == 1, op, ": seg_offsets must be 1-D [num_experts + 1], got ",
              seg_offsets.sizes());
  TORCH_CHECK(seg_offsets.size(0) >= 2, op, ": seg_offsets needs at least [0, n]; got ",
              seg_offsets.size(0), " entries");
  // int32 and not int64. The kernel reads two of these per block and the values are
  // token counts, which cannot exceed a batch; carrying them as int64 would double
  // the loads for a range no model reaches.
  TORCH_CHECK(seg_offsets.scalar_type() == at::kInt, op, ": seg_offsets must be int32, got ",
              seg_offsets.scalar_type());
  const int64_t num_experts = seg_offsets.size(0) - 1;
  TORCH_CHECK(packed.size(0) == num_experts * out_features, op, ": the bank has ", packed.size(0),
              " rows, which is not ", num_experts, " experts of ", out_features,
              ". The segment table and the packed bank describe different models.");
  TORCH_CHECK(x.device() == packed.device() && seg_offsets.device() == packed.device(), op,
              ": x is on ", x.device(), ", seg_offsets on ", seg_offsets.device(),
              ", and the bank on ", packed.device());
}

}  // namespace dynquant
