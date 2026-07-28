# DynQuant format specification

**Checkpoint format version 2 · Stats schema version 2 · Kernel ABI 1**

This is a versioned contract, not documentation of the current code. Where the two
disagree the code is wrong. Everything a reader needs in order to invert a
DynQuant weight is written down here, and nothing may be inferred from tensor
shapes at load time — that inference is the single defect that made every
checkpoint the research supplement produced unreadable
([legacy-audit.md](legacy-audit.md), items 2 and 9).

## Status

Not all of this is implemented yet. The distinction matters, because the parts
marked *specified* are what the CUDA kernels of P6–P8 will be written against, and
changing them later is a format break.

| Section | Status |
|---|---|
| Version numbers and compatibility | implemented |
| Packed weight encoding, grouping, word layout | implemented, fully tested |
| Affine convention, degenerate groups, padding | implemented, fully tested |
| Per-tensor metadata | implemented |
| safetensors key naming | implemented (constants only; no writer yet) |
| `dynquant_manifest.json` | **specified, not implemented** — written in P5 |
| Shard index, fp16 sidecar | **specified, not implemented** — written in P5 |
| `tc_shuffled` layout byte order | **reserved**; the field exists, the permutation is defined in P7 |
| Stats file v2 + v1 migration | implemented, fully tested |
| Kernel ABI | handshake implemented; op signatures land with the ops |

Every filename and schema tag in this document is defined in exactly one place,
[`dynquant/constants.py`](../packages/dynquant-core/src/dynquant/constants.py).
`tests/test_constants_are_the_only_filenames.py` greps the source tree to keep it
that way. Quote the constant, never the literal.

### Optional fields

Throughout §7 and §8: an optional field is **omitted when unknown**, never written
as `null`. `absent` and `null` therefore mean the same thing to a reader and it may
treat them interchangeably — but a writer must omit, so that a file does not grow a
column of nulls describing what nobody measured. The exceptions are stated where
they occur (`hyperparameters.coherence_ema_beta`, `budget.floor_violations`), and
they are exceptions because absent and empty carry different meanings there.

No file in this format may contain a JSON `NaN` or `Infinity`. Those are not valid
JSON, several parsers accept them anyway, and a `NaN` importance score silently
sorts to an arbitrary position in the allocator. Writers pass `allow_nan=False`;
this is enforced, not advised.

---

## 1. Version numbers

Four numbers move independently. Conflating them is how a user ends up with a
kernel that loads happily and returns wrong numbers.

| Number | Where | Governs | On mismatch |
|---|---|---|---|
| `dynquant-core` version | `dynquant._version.__version__` | Python API, CLI, config | ordinary semver |
| `CHECKPOINT_FORMAT_VERSION` | manifest `schema_version` | this document, §2–§7 | reader migrates older, **refuses** newer |
| `STATS_SCHEMA_VERSION` | stats `schema_version` | §8 | reader migrates older, **refuses** newer |
| `KERNEL_ABI_VERSION` | `dynquant_kernels.ABI_VERSION` | §9 | extension **refuses to import** |

Rules:

- A reader that encounters a **newer major** must refuse, with the version it
  found and the version it speaks. It must not attempt a best-effort read. A
  format change that a reader could safely ignore would not have needed a version
  bump.
- A reader that encounters an **older** version must migrate transparently, or say
  precisely what is missing. Silently defaulting a field that used to be absent is
  how `group_size` came to be guessed.
- `KERNEL_ABI_VERSION` bumps on *any* change to a kernel signature, a packed
  memory layout, or the meaning of any field in §4. It is checked at import, not
  at first call, because by first call there is already a model in VRAM.

---

## 2. Checkpoint directory layout

```
my-model-dynquant-3bit/
├── config.json                                     # HF config + quantization_config
├── dynquant_manifest.json                          # MANIFEST_FILENAME       (§7)
├── dynquant_packed_weights.safetensors             # PACKED_WEIGHTS_FILENAME (§6)
├── dynquant_unquantized.safetensors                # FP16_WEIGHTS_FILENAME
├── tokenizer.json, tokenizer_config.json, ...      # untouched
└── (sharded instead, above ~40 GiB:)
    ├── dynquant_packed_weights-00001-of-00003.safetensors
    ├── ...
    └── dynquant_packed_weights.safetensors.index.json
```

`dynquant_unquantized.safetensors` holds every tensor deliberately left in the
compute dtype: norms, biases, rotary caches — the roles in
`dynquant.graph.roles.NEVER_QUANTIZE`. Together these are well under 0.1% of a
model's parameters. They live in a separate file rather than alongside the packed
tensors so that a loader can map the packed file read-only and stream it, while
the small file is loaded eagerly.

`CHECKPOINT_FILES` is the exact tuple of names a checkpoint directory may contain,
and is what the loader's probe iterates. **The stats file (§8) is not among them**:
it is an *input* to quantization, produced weeks earlier by a training run, and a
checkpoint is complete without it. Its SHA-256 is recorded in the manifest instead,
so a checkpoint can be traced back to the signals that shaped it without carrying
them.

Sharding follows the HF convention exactly: 1-based `index`, zero-padded to 5
digits, `total` in every filename, and an index file whose `weight_map` sends each
tensor key to its shard. A DynQuant tensor's several keys (§6) **must all land in
the same shard** — a reader that has to open two files to reconstruct one weight
cannot stream.

### Names that are not this

The supplement wrote `dynQuant_quantized_weights.safetensors` (capital Q) and read
`dynquant_quantized_weights.safetensors`, so no checkpoint it produced could ever
be loaded. Those names, and the `gasq_*` spellings from before the rename, are
listed in `LEGACY_CHECKPOINT_FILENAMES` and are recognised **only** by
`dynquant migrate`. They are never written.

---

## 3. The affine convention

Dequantization is one fused multiply-add per value:

```
w  ≈  q · scale + offset          q ∈ [0, 2^bits)  unsigned integer
```

`scale` and `offset` are stored in the compute dtype (fp16 or bf16), one pair per
`(row, group)`. `offset` is an **unconstrained float**. There is no integer
zero-point anywhere in this format, and no reader should look for one.

This is a deliberate departure from GPTQ and AWQ, which store a packed integer
zero-point and therefore must widen every group's range to include zero — an
integer `zero_point` is only invertible if it lands inside `[0, qmax]`. For a
group whose values all share a sign that widening is pure loss:

| | range | width | resolution at 4-bit |
|---|---|---|---|
| group as measured | `[0.50, 0.52]` | 0.02 | step 0.00133 |
| widened to include zero | `[0.00, 0.52]` | 0.52 | step 0.03467 |

A **26.0×** range inflation, discarding **4.70 bits** of resolution on that group.
A float offset needs no widening, so DynQuant is the textbook min/max affine
quantizer with nothing given away. Three further consequences:

- The kernel inner loop is one `__hfma2` per value pair — no integer subtract, no
  second unpacking pass over a packed zero-point table.
- Packed integer zero-points need their own word alignment along the *group* axis,
  and `num_groups` is frequently not a multiple of 32 — 5120/128 = 40 groups, and
  40 × 3 bits = 120 bits is not a whole number of words. An entire class of
  alignment bug does not arise here.
- **Exact zero is not a guaranteed grid point.** A weight of `0.0` reconstructs
  within half a step like any other value. Nothing in the format depends on it
  being exact: an all-zero group folds through the constant path (§3.2), and the
  zero-filled pad region contributes nothing because kernels predicate activation
  loads past `in_features` to zero rather than trusting the weights there.

Readers **must not** reuse GPTQ's `qzeros` key name for `offsets`, or apply
`(q − zeros) · scale`. Because a DynQuant offset is generally not a multiple of
its scale, there is no integer zero-point that would make that formula agree.

### 3.1 Symmetric mode

`symmetric = true` is the constrained case, not a separate code path:

```
offset = −2^(bits−1) · scale
scale  = max|w| / (2^(bits−1) − 1)
```

The encoder writes the derived `offset` out explicitly rather than leaving readers
to reconstruct it, so a reader that ignores the `symmetric` flag entirely still
produces correct numbers. The flag is informational — it lets a tool verify the
constraint, and it records the encoder's intent.

`offsets` may be **absent**, in which case it is zero. `has_offsets` in the
metadata says which. Absent-means-zero is not the same as symmetric.

### 3.2 Degenerate and constant groups

A group whose raw range is zero — every value identical, including all-zero —
is representable exactly, and is stored as:

```
scale  = 0
offset = the constant value
q      = 0 for every value in the group
```

Two details are load-bearing:

- Constancy is detected on the **raw** range, before any clipping narrows it and
  before either branch derives a scale. Detecting it afterwards would let a
  clipped near-constant group fall into the general path and spend its whole code
  space on one value.
- The constant is folded into the affine map rather than carried in a separate
  `is_constant` / `constant_value` pair the way the supplement did. That removes a
  branch from every kernel: `q · 0 + c = c` needs no special case.

A reader must therefore tolerate `scale == 0` and must not divide by it.

### 3.3 Rounding order

`scale` and `offset` are rounded to the storage dtype **before** the codes are
derived, so the encoder solves against exactly the two numbers the kernel will
read. Rounding afterwards leaves a systematic per-group bias of up to one
storage-dtype ulp of the group range — around 12% of a quantization step at
8-bit, where steps are small enough for that to be measurable.

---

## 4. Grouping and word layout

### 4.1 The alignment invariant

Weights are `[out_features, in_features]`, grouped along `in_features`. The
invariant every kernel depends on:

> **`group_size % 32 == 0`**

With that, `group_size · bits` is a whole number of 32-bit words for every
`bits ∈ {2, 3, 4, 8}`, so each group occupies a whole number of words and the next
group starts on a word boundary. A thread block loading a group reads a
compile-time-constant number of consecutive words as `uint4` vector loads, and
every value's shift is a compile-time constant.

At the default `group_size = 128`:

| bits | words/group | bytes/group |
|---|---|---|
| 2 | 8 | 32 |
| 3 | 12 | 48 |
| 4 | 16 | 64 |
| 8 | 32 | 128 |

This is the invariant the research layout lacked. It flattened the entire matrix
before packing, so group `g` of row `r` began at a bit offset that depended on
every preceding row — no coalesced load, no constant shift, no unrolled inner
loop. That is why the paper ships no kernels, and it is why this section comes
before anything about performance.

### 4.2 Per-row grouping

`group_size = −1` (`PER_ROW_GROUP_SIZE`) means **one group spanning the whole
row**. It is the path for embeddings (symmetric per-row, as in the paper's own
4-bit embedding code) and for tiny trailing dimensions such as Mamba's
`conv1d.weight` of shape `[channels, 1, 4]`, where demanding a multiple of 32
would reject a legitimate tensor outright.

Per-row is the **only** exemption from §4.1, and it is sound for a specific
reason: the alignment rule exists to keep the *next* group on a word boundary, and
per-row there is no next group. So:

- `num_groups == 1`, `scales` and `offsets` have shape `[num_rows, 1]`.
- **No value padding.** The group is exactly `in_features` wide.
- The row rounds up to a whole number of *words*:
  `words_per_row = ceil(in_features · bits / 32)`. The final word may carry up to
  31 unused high bits, which must be written as zero and must be ignored on read.

| shape | bits | words/row | slack bits |
|---|---|---|---|
| `[·, 4]` (Mamba conv1d) | 2 | 1 | 24 |
| `[·, 4]` | 4 | 1 | 16 |
| `[·, 4]` | 8 | 1 | 0 |
| `[·, 100]` | 4 | 13 | 16 |
| `[·, 5120]` (embedding) | 3 | 480 | 0 |

The sentinel is stored **verbatim** in the metadata as `-1`, never resolved to
`in_features`. This is not cosmetic. The sentinel is the only record that the
tensor is exempt from §4.1; a reader that sees `group_size: 100` cannot tell a
legitimate per-row tensor from a misaligned explicit group size, and correctly
rejects it. (This exact bug existed: the encoder resolved the sentinel, and
`validate()` then refused every per-row tensor whose `in_features` was not a
multiple of 32 — including the conv1d case the feature was written for.)

### 4.3 Padding, aligned mode

When `in_features` is not a multiple of `group_size`, the reduction dimension is
zero-padded up to `padded_in_features = ceil(in_features / group_size) ·
group_size`. The pad is applied to the weight **before** quantization, so the
zeros participate in their group's min/max like any other value.

`in_features` — the true, unpadded length — is stored, and is what a kernel uses
to know where real data stops. Kernels predicate activation loads past
`in_features` to zero; the tail block pays one comparison. Correctness therefore
does not depend on the padded weights being zero.

### 4.4 Word layout

Values are packed per row, per group, into `uint32` words, **LSB-first**:

```
packed[r, g · words_per_group + w]     = group g's w-th word of row r
value i within a group occupies bits   [i · bits, i · bits + bits)
```

counting from the LSB of word `⌊i · bits / 32⌋`. A value whose bit range crosses a
word boundary is split: the low `32 − (i · bits mod 32)` bits go in the lower word
at that shift, the remainder in the low bits of the next word.

Words are stored as **int32**, not `torch.uint32`. The bit pattern is identical
under two's-complement reinterpretation and kernels read the pointer as `uint32*`,
but int32 has full operator coverage across supported torch versions, safetensors
and Hub tooling, whereas `torch.uint32` still lacks bitwise ops on several
supported builds. GPTQ, AWQ and Marlin made the same choice. Byte order is
little-endian, as safetensors mandates.

### 4.5 3-bit, worked

3-bit is not a special case in the format — it falls out of §4.1 — but it is the
only width whose values straddle word boundaries, so it is worth writing out. 32
values × 3 bits = 96 bits = exactly 3 words, and within each 3-word block exactly
two values straddle: **indices 10 and 21**, at bits 30–32 and 63–65.

Encoding `q[i] = i mod 8` for `i` in 0..31:

| word | value |
|---|---|
| 0 | `0x88FAC688` |
| 1 | `0xC688FAC6` |
| 2 | `0xFAC688FA` |

Word 0 accumulates `v1<<3 | v2<<6 | … | v9<<27`, then `v10 = 2` contributes its
low two bits at shift 30 (`0x80000000`) and its high bit — which is 0 here — to
bit 0 of word 1. Any other straddle position means the layout has drifted from
this spec, which would break every 3-bit kernel while leaving a round-trip test
green. There is a test asserting exactly `[10, 21]`.

### 4.6 Layouts

`layout` is one of:

| value | meaning |
|---|---|
| `linear` | §4.4 as written. Row-major, groups consecutive within a row. What the decode GEMV kernels want. |
| `tc_shuffled` | **Reserved.** Values permuted at quantization time into the register order `mma.sync` fragments expect, so the tensor-core prefill kernel loads straight from global memory into fragments with no shared-memory transpose. Semantically identical to `linear`; byte order different. The permutation is defined in P7. |

A reader **must refuse** an unrecognised `layout` rather than assuming `linear`.
The field exists in v2 from the start specifically so that adding the tensor-core
path is not a format break, and so that no reader ever has to guess a layout from
a shape.

### 4.7 Uniform bit-width, and row partitions

A single packed tensor has **one** bit-width. This is what lets every CUDA kernel
be templated on a compile-time `BITS` with no per-row branching, which is the
whole reason the inner loop can be unrolled.

Fused projections that deserve different precision per projection — Phi-4's
`qkv_proj`, `gate_up_proj` — are stored as **several tensors over disjoint row
ranges**, each with its own `bits` and its own `row_offset`. The runtime
concatenates their outputs. This costs no extra machinery in the format because
group-wise quantization already stores one scale per `(row, group)`: rows are
independent.

`row_offset` is where a shard's rows begin in the parent module's output space,
and `num_rows` is its extent. Shards of one module must tile its output space
exactly — no gaps, no overlap — and must be listed in ascending `row_offset`.
Row order carries meaning: `gate_up_proj` is `[gate; up]`, so the first block is
the SwiGLU gate.

---

## 5. Dtypes

| Buffer | dtype | Notes |
|---|---|---|
| `qweight` | `int32` | unsigned bit patterns, §4.4 |
| `scales` | `float16` or `bfloat16` | the tensor's compute dtype |
| `offsets` | same as `scales` | optional; absent means zero |
| `bias` | same as `scales` | never quantized, copied verbatim |

`scales` and `offsets` must share a dtype, and it must match `compute_dtype` in
the metadata. Dequantization accumulates in **fp32** regardless of storage dtype;
this is what the reference implementation does and what kernels must do, so parity
tests compare like with like.

---

## 6. safetensors keys

A quantized module contributes several tensors, keyed by suffix:

```
<canonical_module_name>.qweight     int32   [num_rows, num_groups · words_per_group]
<canonical_module_name>.scales      fp16    [num_rows, num_groups]
<canonical_module_name>.offsets     fp16    [num_rows, num_groups]   (optional)
<canonical_module_name>.bias        fp16    [num_rows]               (optional)
```

`qweight` and `scales` deliberately match the GPTQ/AWQ names so that Hub viewers
and existing tooling show something recognisable. `offsets` deliberately does
**not** reuse `qzeros`, for the reason in §3.

Module names are canonical: wrapper prefixes and suffixes introduced by PEFT,
DDP, FSDP and `torch.compile` are stripped at write time by
`dynquant.graph.naming.canonical_name`, so a key does not record which training
stack produced it. `base_model.model.model.layers.0.self_attn.q_proj.base_layer`
is written as `model.layers.0.self_attn.q_proj`.

### 6.1 Row-partitioned modules

When a module is stored as several shards (§4.7), each shard's keys are prefixed
`SHARD_KEY_TEMPLATE` = `{name}.s{index}`, `index` from 0 in ascending row order:

```
model.layers.0.self_attn.qkv_proj.s0.qweight     rows [0, 4096)     4-bit
model.layers.0.self_attn.qkv_proj.s1.qweight     rows [4096, 5120)  3-bit
```

A module with a **single** shard uses the bare name with no `.s0`, so the
overwhelming majority of keys are identical to what an unfused-model reader would
expect. `bias` belongs to the module, not a shard, and is never prefixed.

---

## 7. The manifest

> **Specified, not implemented.** P5 writes this. The shape below is normative for
> that implementation.

`dynquant_manifest.json`, schema tag `MANIFEST_SCHEMA` = `dynquant_checkpoint_v2`.
UTF-8, sorted keys, indent 2, written atomically (write-then-rename). The v1 tag
`dynquant_packed_checkpoint_v1` is recognised by `dynquant migrate` only.

```json
{
  "schema": "dynquant_checkpoint_v2",
  "schema_version": 2,
  "provenance": {
    "dynquant_version": "0.1.0",
    "model_name_or_path": "Qwen/Qwen3-14B",
    "model_type": "qwen3",
    "stats_file_sha256": "…",
    "preset": "default",
    "created_at_unix": 1750000000.0,
    "kernel_abi_version": 1
  },
  "budget": {
    "target_bits": 3.0,
    "achieved_bits_per_weight": 3.24,
    "packed_bytes": 6100000000,
    "unquantized_bytes": 41000000,
    "dense_elements": 14800000000,
    "floor_violations": []
  },
  "compute_dtype": "float16",
  "default_group_size": 128,
  "tensors": {
    "model.layers.0.self_attn.qkv_proj": {
      "role": "attn.qkv",
      "shards": [
        {"bits": 4, "group_size": 128, "in_features": 5120, "logical_shape": [4096, 5120],
         "num_rows": 4096, "num_groups": 40, "words_per_group": 16, "row_offset": 0,
         "layout": "linear", "symmetric": false, "has_offsets": true,
         "compute_dtype": "float16", "nbytes": 11141120, "bits_per_weight": 4.25,
         "role": "attn.q", "error": {"rel_fro": 0.0421, "max_abs": 0.0134}}
      ]
    }
  },
  "unquantized": ["model.norm.weight", "model.layers.0.input_layernorm.weight"]
}
```

- Every key of a `shards` entry except `role` and `error` is exactly what
  `QuantTensor.metadata()` returns; that method is the normative producer.
- `shards` is always a list, length ≥ 1, ascending `row_offset`, even for the
  single-shard case. A reader that special-cases length 1 will mishandle fused
  projections.
- **`role` appears at both levels and the two differ deliberately.** The outer
  `role` is what the module *is* — `attn.qkv`, a fused projection. A shard's `role`
  is which constituent those rows are — `attn.q`. For a single-shard module the two
  are equal. This is what lets a reader recover, from the checkpoint alone, that
  rows 4096–5119 of `qkv_proj` are K and V and were therefore allowed a lower
  width than Q; recomputing it from `num_key_value_heads` and `head_dim` at load
  time is exactly the shape-inference this format exists to avoid.
- `budget` accounts **honestly**: `packed_bytes` includes scales and offsets, so
  `achieved_bits_per_weight` is the number on the filesystem (§10). The supplement
  reported packed weights only, understating a group-128 "3-bit" checkpoint by
  0.25 bits per weight.
- `floor_violations` lists roles the allocator had to take below their default
  floor to meet the budget, with the role, the floor and what it got. An empty
  list means no floor was breached. It is never omitted — absent and empty must
  not be the same thing, or a reader cannot tell "nothing breached" from "written
  by a version that did not record breaches".
- `error` per shard is the measured reconstruction error, so a bad allocation is
  visible in the checkpoint itself rather than only in a downstream eval.

Unrecognised keys must be **preserved** on rewrite, not dropped. A tool that
re-shards a checkpoint should not silently discard fields it does not understand.

---

## 8. The stats file

`dynquant_stats.json`, schema tag `dynquant_stats_v2`. This is the only artifact
that travels from the training run to the quantization run, often across machines
and weeks, so it is self-describing and honest about what it does not know.

```json
{
  "schema": "dynquant_stats_v2",
  "schema_version": 2,
  "provenance": {
    "dynquant_version": "0.1.0",
    "canonical_names": true,
    "model_name_or_path": "Qwen/Qwen3-14B",
    "model_type": "qwen3",
    "num_optimizer_steps": 500,
    "grad_estimator": "outer_exact",
    "world_size": 8,
    "created_at_unix": 1750000000.0
  },
  "hyperparameters": {"activation_ema_beta": 0.99, "coherence_ema_beta": 0.95, "eps": 1e-12},
  "layers": {
    "model.layers.0.self_attn.q_proj": {
      "activation_rms_ema": 0.8134,
      "grad_norm_count": 500,
      "grad_norm_mean": 0.0231,
      "grad_norm_var": 1.44e-05,
      "coherence_ema": 0.71,
      "forward_calls": 500,
      "param_count": 26214400,
      "role": "attn.q",
      "grad_estimator": "outer_exact"
    }
  }
}
```

`dynquant_version` and `canonical_names` are always present. Everything else in
`provenance` and everything after `grad_norm_var` in a layer record is optional and
omitted when unknown — the example above is a dense-model file, so it carries no
`routing_hits`, and it was collected rather than migrated, so no `migrated_from`.
The four fields `activation_rms_ema`, `grad_norm_count`, `grad_norm_mean`,
`grad_norm_var` are always written, defaulting to `0` / `0.0`, and keep their v1
spelling exactly so that a v1 file round-trips through v2 and back with no renaming
and the `paper-3.15` compat path can feed the vendored scorer untouched.

`hyperparameters.coherence_ema_beta` is the one field written as `null` rather than
omitted, because `null` there means "coherence was deliberately not collected" —
distinct from a missing key in a file written before the field existed.

Field meanings that are not obvious from the name:

- **`grad_norm_var` is a *sample* variance**, `m2 / (count − 1)`. Therefore
  `m2 = var · (count − 1)` is recoverable exactly, which is what makes two
  accumulators combinable with no loss via Chan's parallel formula. That single
  fact supplies both the DDP/FSDP reduction and the "collapse LoRA into base"
  step that the shipped stats files claim (`"collapsed_lora_into_base": true`) but
  whose script was never included in the supplement.
- **`grad_norm_count < 2` means no variance was observed**, and must be treated as
  *missing signal*, not as zero variance. A layer hooked once and a layer that
  never moved both produce `var == 0.0`; since low variance maps to low
  importance, an unmeasured layer is otherwise indistinguishable from a maximally
  compressible one. `qwen3_14b`'s shipped stats contain exactly this case —
  `lm_head` has `grad_norm_count: 0` — and it survived only because the LM head
  has an 8-bit floor.
- **`coherence_ema`** is in the research code's score product but absent from the
  paper's Eq. (4): the published formula is a two-way product, the shipped code
  computes a three-way one. Retained because reproducing the paper's numbers
  requires it; optional for the default scorer.
- **`forward_calls`** is how many forward passes the module was seen in. It answers
  a question `activation_rms_ema` cannot: a module that never ran has an EMA of
  exactly `0.0`, and so does a module that ran constantly with zero-magnitude
  activations. Low RMS maps to low importance, so without this field the first case
  is silently scored as the second and the tensor is compressed on evidence that was
  never collected. `forward_calls == 0` means *unmeasured* and the scorer must
  substitute a neutral rank rather than a low one. Distinct from `routing_hits`,
  which counts token assignments within a call; on a dense model the two would agree,
  on an MoE they do not.
- **`routing_hits`** (MoE only) is the denominator that stops a rarely-routed
  expert from being scored as stable when it is really unmeasured.
- **`grad_estimator`** — `outer_exact`, `lowrank` or `param`. Different modes are
  not comparable, so mixing them in one file is a detectable bug.
- **`canonical_names`** — whether keys have been through
  `canonical_name()`. v2 writers always set `true`.

### 8.1 v1 migration

v1 is the supplement's format. Recognised tags: `unified_dynquant_stats_v1`,
`unified_gasq_stats_v1`, `unified_gasq_stats`, `gasq_stats_v1`, and a file with a
`layers` key and no tag at all (hand-edited files, same field layout).

Both shipped stats files declare `unified_gasq_stats_v1` even though the code that
read them had been renamed to the `dynquant` spelling — a reader that trusts one
spelling rejects the only two real stats files in existence.

Migration is lossless: every numeric quantity carries over unchanged,
unrecognised top-level keys are preserved under `provenance.notes`, keys are
canonicalised, and collisions created by canonicalisation are **merged with
Chan's formula** rather than overwritten. `provenance.grad_estimator` is set to
`param`, because the research code's only estimator hooked whatever parameter
happened to require grad — LoRA factors under QLoRA, not the base weight.

EMAs are the one approximation. An EMA is path-dependent, so no algebra recovers
the joint history from two summaries; merging combines them by
observation-count-weighted mean. That is the right first-order estimate and it is
documented rather than hidden.

Writes are atomic: write-then-rename, so a crash mid-write cannot leave a
truncated stats file that a later run silently under-covers from.

---

## 9. Kernel ABI

`KERNEL_ABI_VERSION` is the contract between `dynquant-core` and the compiled
`dynquant-kernels` wheel: tensor layouts, argument order, operator names. It is
declared in three places that must agree —
`dynquant._version.KERNEL_ABI_VERSION`, `dynquant_kernels.ABI_VERSION`, and the
`#define` in `csrc/include/dynquant/abi.h` — and `tests/test_abi.py` is the lint
that holds them together, by reading source text so that it runs on a machine
where no extension is installed.

- The extension refuses to import on mismatch, with the wheel to install instead.
  Checked at import, not at first call: by first call there is a model in VRAM.
- `dynquant-core` accepts ABI in `[MIN_KERNEL_ABI_VERSION, KERNEL_ABI_VERSION]`.
- Operators register through `TORCH_LIBRARY`, not pybind, so each carries a schema
  `torch.compile` can trace instead of forcing a graph break at every quantized
  `Linear`. Every op has a `register_fake` meta implementation.
- A wheel records the torch version and CUDA version it was built against, and the
  binary variant is carried in the PEP 440 local version segment
  (`0.1.0+cu126torch27`) — the same scheme torch itself uses.

Why this is checked so aggressively: a wrong-ABI kernel does not crash. It returns
tensors of the right shape and dtype full of wrong numbers, which is
indistinguishable from quantization simply being lossy, and so never gets reported
as a bug. `dynquant doctor` therefore runs a numerical self-check — pack/unpack
bijection at every width, and measured quantization error against theory — rather
than only reporting versions.

---

## 10. Size accounting

Effective bits per weight counts **everything on disk**, including scales and
offsets, over the number of *dense* elements:

```
bits_per_weight = (packed + scales + offsets bytes) · 8 / dense_elements
```

Metadata overhead is `2 · sizeof(compute_dtype) · 8 / group_size` bits per weight:

| group_size | overhead (bpw) |
|---|---|
| 32 | 1.0000 |
| 64 | 0.5000 |
| 128 | 0.2500 |
| 256 | 0.1250 |

So at the default group size, measured on a `[256, 512]` tensor:

| nominal | effective bpw | bytes |
|---|---|---|
| 2-bit | 2.2500 | 36 864 |
| 3-bit | 3.2500 | 53 248 |
| 4-bit | 4.2500 | 69 632 |
| 8-bit | 8.2500 | 135 168 |

A "3-bit" group-128 checkpoint occupies 3.25 bits per weight, and the gap widens
at smaller group sizes. Budgeting against this — not against the packed weights
alone — is what makes the number in the manifest the number on the filesystem.

Per-row grouping is nearly free on a wide tensor and ruinous on a narrow one,
because the cost is per row rather than per 128 values:

| tensor | mode | effective bpw |
|---|---|---|
| `[152064, 5120]` embedding, 4-bit | per-row | **4.006** |
| `[512, 1, 4]` conv1d, 4-bit | per-row | **16.000** |

The second row is not a bug in the accounting, it is the accounting doing its job:
one fp16 scale and one fp16 offset per 4 values is 8 bits per value on top of the
4, so a per-row 4-bit conv1d is *four times larger than fp16*. Tensors like this
belong in the unquantized sidecar, and the honest number is what makes that
obvious. An allocator that counted packed bytes only would have shipped it.

---

## 11. Extension and compatibility rules

For anyone changing this format:

1. **Never infer metadata from a shape.** Every parameter needed to invert a
   tensor is stored. The failure mode of inference is not an exception, it is a
   tensor of the right shape full of wrong numbers.
2. **Add fields, don't repurpose them.** Fields are optional-with-a-documented-
   default or required; a field whose meaning changes needs a version bump even if
   its type does not.
3. **A new `layout`, `bits`, or grouping mode bumps `KERNEL_ABI_VERSION`.** Kernels
   are templated on these; an unrecognised value must fail loudly at load, not be
   coerced to the nearest known one.
4. **Absent and default are not the same thing** where the difference is
   observable. `floor_violations: []` and a missing `floor_violations` mean
   different things and must be distinguishable.
5. **Preserve unknown keys on rewrite.**
6. **Every new filename goes in `constants.py`** and nowhere else.

## 12. Conformance

An implementation claiming to read this format must pass, at minimum:

- pack/unpack bijection at 2/3/4/8-bit for value counts ≢ 0 mod {2, 4, 8, 32} —
  the suite uses `[1, 2, 3, 7, 8, 31, 32, 33, 37, 96, 127, 128, 129, 1023]`;
- 3-bit straddlers at exactly indices 10 and 21 per 32-value block;
- group independence: rewriting one group must not alter any other group's words;
- per-row round-trip through `state_dict` at every width for `in_features` not a
  multiple of 32, **including 4**;
- reconstruction error within ±25% of `step/√12 · 0.90`, computed from each
  group's own min/max with **no** clamping around zero — a tolerance band
  computed from a zero-widened range would rubber-stamp an encoder that had
  reintroduced the widening;
- `group_size` read from metadata, never derived from `scales.numel()`;
- refusal of a newer `schema_version`, and of an unrecognised `layout`;
- an unmeasured layer scored **neutrally, never low**. A reader must distinguish
  three cases that all arrive as a small number or a missing one: no key in `layers`,
  `forward_calls == 0`, and `grad_norm_count < 2`. Mapping any of them onto "low
  importance" compresses a tensor on evidence that was never collected, and the output
  is indistinguishable from a measurement.

The corresponding tests are `tests/test_pack.py`, `tests/test_quant_tensor.py`,
`tests/test_stats_schema.py`, `tests/test_score_importance.py` and
`tests/test_abi.py`.
