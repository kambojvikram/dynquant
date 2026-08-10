// Contract between the compiled extension and the Python package.
//
// DYNQUANT_ABI_VERSION must equal `dynquant._version.KERNEL_ABI_VERSION`.
// tests/test_abi.py parses this header and asserts it, because an ABI mismatch is
// the one failure mode that yields plausible wrong numbers instead of an
// exception: an old kernel reading a new packed layout still produces a tensor of
// the right shape and dtype.
//
// Bump on any change to:
//   * a kernel's C++ signature or its registered schema string,
//   * the packed word layout (see docs/format-spec.md),
//   * the affine convention `w ~= q * scale + offset`.
//
// Do NOT bump for internal kernel optimisations that keep all three fixed --
// those are exactly the changes that should ship without forcing a reinstall.

#pragma once

// 1 -> 2: added the `dequant` and `gemv` ops. A wheel at ABI 1 has neither, so
// core would reach for `torch.ops.dynquant.gemv` and get an AttributeError deep
// inside a forward pass; MIN_KERNEL_ABI_VERSION was raised to 2 alongside so the
// mismatch is caught at import with a message naming the wheel to install.
// 2 -> 3: added `moe_grouped_gemv`. Unlike the 1 -> 2 step this does NOT raise
// MIN_KERNEL_ABI_VERSION. An ABI-2 wheel is missing the op and nothing else, and
// the grouped MoE path is an optimisation over a Python loop that still works --
// so core feature-detects the op and falls back, rather than refusing to load a
// wheel that can serve every model it could serve yesterday.
#define DYNQUANT_ABI_VERSION 3

// Bit-widths every kernel is templated over. Kept in sync with
// `dynquant.constants.BIT_OPTIONS` by the same test.
#define DYNQUANT_BITS_LIST 2, 3, 4, 8

// Alignment invariant the layout depends on: group_size % 32 == 0 makes
// group_size * bits a whole number of 32-bit words for every width above, so a
// group never begins mid-word and every value's shift is a compile-time constant.
#define DYNQUANT_GROUP_SIZE_ALIGNMENT 32

// Most activation rows `dynquant::gemv` accepts.
//
// Part of the ABI because the runtime dispatches on it: at or below this many
// rows a matmul goes to the packed GEMV, above it to `dequant` + cuBLASLt. The
// kernel holds its accumulators in registers indexed by a compile-time row count,
// so the bound is a real property of the binary and not a policy knob -- Python
// reads it back through `dynquant::gemv_max_rows()` rather than assuming.
#define DYNQUANT_GEMV_MAX_ROWS 8

namespace dynquant {

// Returned to Python by the `dynquant::abi_version` op. A function rather than
// only a macro so the value is baked into the shared object and cannot be faked
// by a stale header on the reader's side.
inline constexpr int abi_version() { return DYNQUANT_ABI_VERSION; }

}  // namespace dynquant
