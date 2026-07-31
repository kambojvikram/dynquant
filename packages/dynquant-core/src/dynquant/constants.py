"""Every filename, schema tag and magic constant DynQuant writes or reads.

Why this module exists
----------------------
The research supplement had the writer emit ``dynQuant_quantized_weights.safetensors``
(capital Q, ``dynquant_paper/run_quantization.py:372``) while every reader opened
``dynquant_quantized_weights.safetensors`` (``inference/inference_4bits.py:198``).
No checkpoint produced by that code could ever be loaded back. The names were
spelled out as string literals in five different files.

Nothing outside this module may contain a checkpoint filename literal. There is a
test that greps the source tree to enforce it
(``tests/test_constants_are_the_only_filenames.py``).
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "ALLOCATION_FILENAME",
    "ALLOCATION_SCHEMA",
    "BIT_OPTIONS",
    "CHECKPOINT_FILES",
    "COMPUTE_DTYPES",
    "DEFAULT_GROUP_SIZE",
    "FP16_WEIGHTS_FILENAME",
    "GEMV_MAX_ROWS",
    "HF_CONFIG_FILENAME",
    "HF_QUANT_METHOD",
    "HF_SHARD_PATTERN",
    "HF_WEIGHTS_FILENAME",
    "HF_WEIGHTS_INDEX_FILENAME",
    "KERNELS_DISTRIBUTION",
    "KERNELS_IMPORT_NAME",
    "LEGACY_STATS_FILENAMES",
    "LEGACY_STATS_SCHEMAS",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA",
    "MOMENTS_FILENAME",
    "MOMENTS_SCHEMA",
    "MOMENT_TENSOR_SUFFIXES",
    "PACKED_WEIGHTS_FILENAME",
    "PACKED_WEIGHTS_INDEX_FILENAME",
    "PER_ROW_GROUP_SIZE",
    "QUANT_TENSOR_SUFFIXES",
    "SHARD_KEY_TEMPLATE",
    "SHARD_PATTERN",
    "STATS_FILENAME",
    "STATS_SCHEMA",
    "STATS_SCHEMA_V1",
]

# --------------------------------------------------------------------------
# Checkpoint layout
# --------------------------------------------------------------------------

PACKED_WEIGHTS_FILENAME: Final = "dynquant_packed_weights.safetensors"
"""Single-shard packed weights. Lowercase, always."""

SHARD_PATTERN: Final = "dynquant_packed_weights-{index:05d}-of-{total:05d}.safetensors"
"""Sharded packed weights, HF-style numbering (1-based ``index``)."""

PACKED_WEIGHTS_INDEX_FILENAME: Final = "dynquant_packed_weights.safetensors.index.json"
"""Shard index: ``{"metadata": {...}, "weight_map": {tensor_key: shard_filename}}``."""

FP16_WEIGHTS_FILENAME: Final = "dynquant_unquantized.safetensors"
"""Tensors deliberately left in fp16/bf16: norms, biases, rotary caches."""

MANIFEST_FILENAME: Final = "dynquant_manifest.json"
"""Allocation map, per-layer metadata, achieved budget, provenance."""

MANIFEST_SCHEMA: Final = "dynquant_checkpoint_v2"
"""``schema`` field inside the manifest. v1 == the supplement's format."""

ALLOCATION_FILENAME: Final = "dynquant_allocation.json"
"""A bit map on its own: what ``dynquant inspect --save-map`` writes and
``dynquant quantize --map`` reads.

Deliberately *not* :data:`MANIFEST_FILENAME`. A manifest is part of a packed
checkpoint and a loader that finds one expects
:data:`PACKED_WEIGHTS_FILENAME` beside it (see :data:`CHECKPOINT_FILES`). An
allocation is an answer to "what width should each tensor get", which is
produced before any weight is touched and is equally valid next to a directory
whose weights are stored in the compute dtype. Writing it under the manifest
name would make a values-quantized folder claim to be a packed checkpoint.
"""

ALLOCATION_SCHEMA: Final = "dynquant_allocation_v1"

LEGACY_MANIFEST_SCHEMA: Final = "dynquant_packed_checkpoint_v1"

CHECKPOINT_FILES: Final = (
    MANIFEST_FILENAME,
    PACKED_WEIGHTS_FILENAME,
    PACKED_WEIGHTS_INDEX_FILENAME,
    FP16_WEIGHTS_FILENAME,
)
"""All names a checkpoint directory may contain. Used by the loader's probe."""

# Names the supplement's writer/readers used, kept only so that
# `dynquant migrate` can pick up an old directory and rewrite it.
LEGACY_CHECKPOINT_FILENAMES: Final = (
    "dynQuant_quantized_weights.safetensors",  # writer (capital Q)
    "dynquant_quantized_weights.safetensors",  # readers
    "dynquant_fp16_remaining.safetensors",
    "gasq_quantized_weights.safetensors",  # pre-rename era
)

# --------------------------------------------------------------------------
# Per-tensor key suffixes inside the safetensors file
# --------------------------------------------------------------------------

QUANT_TENSOR_SUFFIXES: Final = {
    "packed": "qweight",  # int32 words holding unsigned bit patterns
    "scale": "scales",  # fp16/bf16, one per (row, group)
    "offset": "offsets",  # fp16/bf16, additive term: w ~= q * scale + offset
    "bias": "bias",  # untouched, fp16/bf16
}
"""Suffix appended to a canonical module name to form a safetensors key.

``qweight``/``scales`` deliberately match the GPTQ/AWQ names so Hub viewers and
existing tooling show something recognisable.

``offsets`` deliberately does *not* reuse GPTQ's ``qzeros`` name, because it is a
different object: an unconstrained float additive term, not a packed integer
zero-point, and it is *added* rather than subtracted. Reusing the name would
invite a reader to apply ``(q - zeros) * scale`` and get silently wrong weights --
and because a DynQuant offset is generally not a multiple of its scale, no integer
zero-point exists that would make that formula agree. See
:class:`dynquant.quant.tensor.QuantTensor` for why the constraint is dropped.
"""

SHARD_KEY_TEMPLATE: Final = "{name}.s{index}"
"""Key prefix for one shard of a row-partitioned fused projection.

A fused ``qkv_proj`` whose Q/K/V halves earned different bit-widths is stored as
several :class:`~dynquant.quant.tensor.QuantTensor` shards over disjoint row
ranges. Single-shard tensors (the common case) use the bare module name with no
suffix, so the overwhelming majority of keys stay identical to what an
unfused-model reader would expect.
"""

# --------------------------------------------------------------------------
# Signal statistics
# --------------------------------------------------------------------------

STATS_FILENAME: Final = "dynquant_stats.json"
STATS_SCHEMA: Final = "dynquant_stats_v2"
STATS_SCHEMA_V1: Final = "unified_dynquant_stats_v1"

LEGACY_STATS_SCHEMAS: Final = (
    STATS_SCHEMA_V1,
    "unified_gasq_stats_v1",  # what the two shipped stats files actually declare
    "unified_gasq_stats",
    "gasq_stats_v1",
)
"""Every ``schema`` tag that means "v1 field layout".

The rename from ``gasq`` to ``dynquant`` was never applied to the schema tag in
the artifacts: both ``stats/phi-4`` and ``stats/qwen3_14b`` declare
``unified_gasq_stats_v1`` even though the code that reads stats was updated to
the ``dynquant`` spelling. A reader that trusts one spelling rejects the only two
real stats files in existence. They are all the same format, so they are all
accepted -- and only here."""

LEGACY_STATS_FILENAMES: Final = (
    "unified_dynquant_stats.json",
    "unified_dynquant_stats_collapsed.json",
    "unified_gasq_stats.json",
    "unified_gasq_stats_collapsed.json",
)
"""Filenames the supplement's trainer wrote. Auto-detected by the CLI."""

STATS_SIDECAR_FILENAME: Final = "dynquant_stats_meta.json"
"""Run provenance kept next to the stats: model id, steps, estimator mode, git sha."""

MOMENTS_FILENAME: Final = "dynquant_moments.safetensors"
"""Per-channel second moments, written beside the stats file.

Separate from :data:`STATS_FILENAME` because of size, not taste. The stats file
holds four scalars per module; this holds two *vectors* -- one entry per input
channel and one per output channel. On a 2B model that is about a million floats,
which is 20 MB of JSON and 4 MB of safetensors, and the JSON would have to be
parsed in full by every consumer including the ones that only want the scalars.

A stats file without this sidecar is complete and usable; only the cardinal
sensitivity estimator needs it, and it says so rather than assuming.
"""

MOMENTS_SCHEMA: Final = "dynquant_moments_v1"

MOMENT_TENSOR_SUFFIXES: Final = {
    "input_sq": "input_sq",  # E[x_c^2] over tokens, length in_features
    "output_grad_sq": "output_grad_sq",  # E[delta_r^2] over tokens, length out_features
}
"""Suffix appended to a canonical module name to form a moments key."""

# --------------------------------------------------------------------------
# Quantization defaults
# --------------------------------------------------------------------------

BIT_OPTIONS: Final = (2, 3, 4, 8)
"""Supported weight bit-widths. Every kernel is templated over exactly these."""

COMPUTE_DTYPES: Final = ("bfloat16", "float16", "float32")
"""Dtypes a model may be loaded in from the command line.

No ``auto``: a checkpoint's ``dtype`` field is advisory and is frequently
``float32`` on a model that was trained in bf16, and loading a 2B model in fp32
because a config said so is how an evaluation OOMs with nothing in the log
pointing at the dtype. The default is stated, printed, and recorded in every
artifact instead.
"""

DEFAULT_GROUP_SIZE: Final = 128
"""Quantization group size along the input dimension.

Must satisfy ``group_size % 32 == 0`` so that every group starts on a uint32
word boundary for all widths in :data:`BIT_OPTIONS` (see docs/format-spec.md).
"""

GROUP_SIZE_ALIGNMENT: Final = 32
"""LCM of the values-per-word counts across BIT_OPTIONS: 16, 32, 8, 4 -> 32."""

PER_ROW_GROUP_SIZE: Final = -1
"""``group_size`` sentinel meaning "one group spanning the whole row".

Stored in the format *as the sentinel*, never resolved to ``in_features``. The
distinction is load-bearing: per-row grouping is exempt from the
:data:`GROUP_SIZE_ALIGNMENT` rule, because with a single group there is no
following group that could begin mid-word. Writing the resolved value instead
throws that exemption away -- a reader then sees ``group_size = 4`` for a Mamba
``conv1d`` and cannot tell it apart from a badly chosen explicit group size, so it
rejects a tensor that is perfectly well formed.
"""

VALUES_PER_WORD: Final = {2: 16, 3: 32, 4: 8, 8: 4}
"""Quantized values encoded per uint32 *group* of words, per bit-width.

For 2/4/8-bit this is values per single word. For 3-bit it is 32 values spanning
exactly 3 words (96 bits), which is why 3-bit is special-cased everywhere.
"""

WORDS_PER_BLOCK: Final = {2: 1, 3: 3, 4: 1, 8: 1}
"""Words consumed by one :data:`VALUES_PER_WORD` block."""

GEMV_MAX_ROWS: Final = 8
"""Most activation rows the packed GEMV kernel accepts; above it, dequant + GEMM.

This is the compiled kernel's limit, mirrored here the same way
:data:`GROUP_SIZE_ALIGNMENT` is, and held to the ``#define`` in
``csrc/include/dynquant/abi.h`` by ``tests/test_abi.py``. It is a real property of
the binary rather than a tuning knob: the kernel indexes its accumulator registers
by a compile-time row count, so exceeding it is not slow, it is unimplemented.

Used only as the fallback when there is no extension to ask. When one is loaded,
:mod:`dynquant.runtime.linear` reads ``torch.ops.dynquant.gemv_max_rows()``, so a
wheel built with a different bound is obeyed rather than second-guessed.
"""

# --------------------------------------------------------------------------
# Compiled kernels
# --------------------------------------------------------------------------

KERNELS_DISTRIBUTION: Final = "dynquant-kernels"
"""PyPI distribution holding the compiled extension."""

KERNELS_IMPORT_NAME: Final = "dynquant_kernels"
"""Top-level import name of the compiled extension package.

Deliberately *not* ``dynquant.kernels``. A submodule of ``dynquant`` shipped by a
different wheel makes ``dynquant`` a namespace package split across two
distributions, and then an interrupted upgrade leaves ``dynquant/kernels/`` from
one version next to ``dynquant/`` from another with nothing to detect it. A
separate top-level name means pip can reason about the two wheels independently,
and the version handshake below is explicit rather than implied by file layout.
"""

#: The ABI number itself lives in :mod:`dynquant._version` next to the other
#: version contracts, and is *not* repeated here -- it is declared in three places
#: already (there, ``dynquant_kernels.ABI_VERSION``, and the ``#define`` in
#: ``csrc/include/dynquant/abi.h``), which is two more than anyone can keep in
#: step by hand. ``tests/test_abi.py`` is the lint that holds them together; a
#: fourth copy would be a fourth thing for it to have to know about.

TORCH_MINIMUM: Final = (2, 4)
"""Oldest torch supported. ``torch.library.custom_op`` and ``register_fake``
arrived in 2.4, and they are how kernels stay visible to ``torch.compile``."""

# --------------------------------------------------------------------------
# transformers integration
# --------------------------------------------------------------------------

HF_QUANT_METHOD: Final = "dynquant"
"""``config.json -> quantization_config.quant_method``.

Registered into ``transformers.quantizers.auto`` so that
``AutoModelForCausalLM.from_pretrained`` dispatches to DynQuant natively.
"""

HF_CONFIG_FILENAME: Final = "config.json"
"""Owned by transformers, named here because the exporter writes one."""

HF_WEIGHTS_FILENAME: Final = "model.safetensors"
HF_SHARD_PATTERN: Final = "model-{index:05d}-of-{total:05d}.safetensors"
HF_WEIGHTS_INDEX_FILENAME: Final = "model.safetensors.index.json"
"""Standard HF weight-file layout, which ``dynquant export`` writes rather than
:data:`PACKED_WEIGHTS_FILENAME`.

Deliberate, and the reason is interoperability rather than taste. vLLM's default
loader globs ``*.safetensors`` and, when it finds more than one file, keeps only
those listed in ``model.safetensors.index.json``; a directory whose weights are
called ``dynquant_packed_weights-*.safetensors`` with a differently-named index
loads every shard including ones the index would have excluded. Writing the
standard names means no special case in vLLM, in transformers, or in any Hub
tool -- the tensors inside are what identify the checkpoint as DynQuant's, along
with ``quantization_config`` in ``config.json``.

:data:`PACKED_WEIGHTS_FILENAME` remains the name for DynQuant's own
self-contained artifacts, which are not HF model directories.
"""

# --------------------------------------------------------------------------
# Environment variables
# --------------------------------------------------------------------------

ENV_BACKEND: Final = "DYNQUANT_BACKEND"
"""``cuda`` | ``triton`` | ``torch``. Overrides automatic backend selection."""

ENV_CACHE_DIR: Final = "DYNQUANT_CACHE"
"""Autotuner cache location. Defaults to platform user-cache/dynquant."""

ENV_DISABLE_AUTOTUNE: Final = "DYNQUANT_DISABLE_AUTOTUNE"
ENV_LOG_LEVEL: Final = "DYNQUANT_LOG_LEVEL"
