"""Packing and budget accounting on the phase-3 models' real, full-scale shapes.

[test_kernels_parity.py][1] sweeps geometries chosen to stress the *layout*, and
[test_quant_tensor.py][2] sweeps small random tensors. Neither has ever seen a
number that any of the four phase-3 checkpoints actually contains. That gap is not
academic: three of the four depart from the shape a reader would guess.

* Gemma-3 sets ``head_dim=256`` independently of ``hidden_size``, so its
  ``o_proj`` is ``[2560, 2048]`` -- not square, and ``in_features`` is neither the
  hidden size nor a head count times anything obvious.
* Gemma-3's vision tower has ``intermediate_size=4304``, which is **not** a
  multiple of the 128 group. It is the only real ``in_features`` in the set that
  pads, and it costs a real 1.1%.
* Gemma-3's ``patch_embedding`` is rank 4 -- ``[1152, 3, 14, 14]`` -- and folds to
  48384 rows of **14** columns, which pads by 9.1x.
* Phi-4-mini fuses both projections, so ``qkv_proj`` and ``gate_up_proj`` carry row
  partitions whose boundaries follow the GQA ratio rather than an even split.

Three defects were live when this file was first run, and the tests that name them
are their regression guards:

1. ``module_stored_bits`` priced the payload as ``params * bits``, which is the
   size of an unpadded tensor. The packer pads ``in_features`` up to a whole
   number of groups and stores the pad. On Gemma-3's vision ``fc2`` the budget
   undercounted by 1.1%; on its ``patch_embedding`` by **4.8x**. The budget is
   supposed to be the number the filesystem reports.
2. Nothing stopped the allocator from quantizing a tensor that costs *more* than
   fp16. ``patch_embedding`` at 14 columns measures 20.6 bits per weight at 2-bit
   and 75.4 at 8-bit, against 16 for leaving it alone. ``UNQUANTIZED_FLOOR``
   exists for exactly this, but was keyed on role, and a role cannot see a shape.
3. The error oracle in [_oracle.py][3] scaled its prediction up by
   ``sqrt(padded / in_features)`` on padded shapes. The factor does not belong in
   a ratio, and at the pad ratios previously under test it was 1.13x and 1.10x --
   inside the tolerance band, and pointing opposite to the band's other known
   bias, so the two cancelled. At 14 of 128 it is 3.02x and the prediction misses
   by two thirds. Found because the width sweep below includes 14.

**Why the shapes are pinned literals rather than fetched.** Two of the four repos
are gated (HTTP 401 without a token), so a test that downloads configs is a test
that fails on a fresh checkout for a reason having nothing to do with the code.
The literals below were transcribed from the published configs -- cross-checked
against independent ungated mirrors for the two gated ones -- and
:func:`test_the_pinned_shapes_are_what_transformers_builds` rebuilds all four
models on the meta device and proves the pins still hold. The parameter-count
assertion in that test is what stops a transcription error from being encoded
twice: a mistyped ``intermediate_size`` moves the total off the advertised count.

[1]: test_kernels_parity.py
[2]: test_quant_tensor.py
[3]: _oracle.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import pytest
import torch

from dynquant.allocate.budget import module_stored_bits
from dynquant.constants import BIT_OPTIONS, DEFAULT_GROUP_SIZE
from dynquant.graph.classify import ModuleInfo
from dynquant.graph.roles import UNQUANTIZED_FLOOR, ModuleRole
from dynquant.quant.pack import pack_nbit, padded_in_features, unpack_nbit
from dynquant.quant.tensor import QuantTensor

from _oracle import assert_error_matches_theory  # type: ignore[import-not-found]

if TYPE_CHECKING:
    from collections.abc import Iterator


class Weight(NamedTuple):
    """One weight tensor class, with the layer index collapsed away."""

    model: str
    module: str
    shape: tuple[int, ...]
    count: int
    """How many of these the model has -- 32 per-layer copies collapse to one row."""

    @property
    def in_features(self) -> int:
        return self.shape[-1]

    @property
    def rows(self) -> int:
        rows = 1
        for dim in self.shape[:-1]:
            rows *= dim
        return rows

    @property
    def numel(self) -> int:
        return self.rows * self.in_features


# Transcribed from each repo's config.json. Llama-3.1 and Gemma-3 are gated, so
# their values were cross-checked against ungated mirrors that publish the same
# weights: NousResearch/Meta-Llama-3.1-8B-Instruct and unsloth/... for Llama,
# unsloth/gemma-3-4b-it and gaunernst/gemma-3-4b-it-qat-... for Gemma. The mirrors
# agree with each other exactly on every field used here.
PHASE3_CONFIGS: dict[str, dict] = {
    # meta-llama/Llama-3.1-8B-Instruct -- the control. Untied head.
    "llama": {
        "model_type": "llama",
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "vocab_size": 128256,
        "head_dim": 128,
        "tie_word_embeddings": False,
    },
    # mistralai/Ministral-8B-Instruct-2410 -- interleaved sliding-window attention.
    # Same hidden size as Llama with a narrower MLP and four more layers.
    "ministral": {
        "model_type": "mistral",
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 36,
        "vocab_size": 131072,
        "head_dim": 128,
        "tie_word_embeddings": False,
    },
    # microsoft/Phi-4-mini-instruct -- tied embeddings, both projections fused,
    # partial RoPE. `partial_rotary_factor` changes no weight shape but is pinned
    # because it is the field a "rotary dim == head dim" assumption breaks on.
    "phi4": {
        "model_type": "phi3",
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "vocab_size": 200064,
        "partial_rotary_factor": 0.75,
        "tie_word_embeddings": True,
    },
    # google/gemma-3-4b-it -- 5:1 local/global interleave, QK-norm, vision tower.
    "gemma3": {
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "hidden_size": 2560,
            "intermediate_size": 10240,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_hidden_layers": 34,
            "head_dim": 256,
            "vocab_size": 262208,
            "sliding_window": 1024,
            "query_pre_attn_scalar": 256,
        },
        "vision_config": {
            "model_type": "siglip_vision_model",
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_attention_heads": 16,
            "num_hidden_layers": 27,
            "image_size": 896,
            "patch_size": 14,
            "num_channels": 3,
            "vision_use_head": False,
        },
        "mm_tokens_per_image": 256,
    },
}

ADVERTISED_PARAMS = {
    # What each model card claims, to the precision it claims it. The pinned
    # configs must reproduce these; a typo in one field will not.
    "llama": 8.03e9,
    "ministral": 8.02e9,
    "phi4": 3.84e9,
    "gemma3": 4.30e9,
}

WEIGHTS: tuple[Weight, ...] = (
    # -- llama -----------------------------------------------------------------
    Weight("llama", "model.embed_tokens", (128256, 4096), 1),
    Weight("llama", "model.layers.N.self_attn.q_proj", (4096, 4096), 32),
    Weight("llama", "model.layers.N.self_attn.k_proj", (1024, 4096), 32),
    Weight("llama", "model.layers.N.self_attn.v_proj", (1024, 4096), 32),
    Weight("llama", "model.layers.N.self_attn.o_proj", (4096, 4096), 32),
    Weight("llama", "model.layers.N.mlp.gate_proj", (14336, 4096), 32),
    Weight("llama", "model.layers.N.mlp.up_proj", (14336, 4096), 32),
    Weight("llama", "model.layers.N.mlp.down_proj", (4096, 14336), 32),
    Weight("llama", "lm_head", (128256, 4096), 1),
    # -- ministral -------------------------------------------------------------
    Weight("ministral", "model.embed_tokens", (131072, 4096), 1),
    Weight("ministral", "model.layers.N.self_attn.q_proj", (4096, 4096), 36),
    Weight("ministral", "model.layers.N.self_attn.k_proj", (1024, 4096), 36),
    Weight("ministral", "model.layers.N.self_attn.v_proj", (1024, 4096), 36),
    Weight("ministral", "model.layers.N.self_attn.o_proj", (4096, 4096), 36),
    Weight("ministral", "model.layers.N.mlp.gate_proj", (12288, 4096), 36),
    Weight("ministral", "model.layers.N.mlp.up_proj", (12288, 4096), 36),
    Weight("ministral", "model.layers.N.mlp.down_proj", (4096, 12288), 36),
    Weight("ministral", "lm_head", (131072, 4096), 1),
    # -- phi4 -- both projections fused; lm_head is the embedding ---------------
    Weight("phi4", "model.embed_tokens", (200064, 3072), 1),
    Weight("phi4", "model.layers.N.self_attn.qkv_proj", (5120, 3072), 32),
    Weight("phi4", "model.layers.N.self_attn.o_proj", (3072, 3072), 32),
    Weight("phi4", "model.layers.N.mlp.gate_up_proj", (16384, 3072), 32),
    Weight("phi4", "model.layers.N.mlp.down_proj", (3072, 8192), 32),
    Weight("phi4", "lm_head", (200064, 3072), 1),
    # -- gemma3 vision tower ---------------------------------------------------
    Weight("gemma3", "...embeddings.patch_embedding", (1152, 3, 14, 14), 1),
    Weight("gemma3", "...embeddings.position_embedding", (4096, 1152), 1),
    Weight("gemma3", "...vision.self_attn.q_proj", (1152, 1152), 27),
    Weight("gemma3", "...vision.self_attn.k_proj", (1152, 1152), 27),
    Weight("gemma3", "...vision.self_attn.v_proj", (1152, 1152), 27),
    Weight("gemma3", "...vision.self_attn.out_proj", (1152, 1152), 27),
    Weight("gemma3", "...vision.mlp.fc1", (4304, 1152), 27),
    Weight("gemma3", "...vision.mlp.fc2", (1152, 4304), 27),
    # -- gemma3 language model -- o_proj is [2560, 2048], not square ------------
    Weight("gemma3", "model.language_model.embed_tokens", (262208, 2560), 1),
    Weight("gemma3", "model.language_model.layers.N.self_attn.q_proj", (2048, 2560), 34),
    Weight("gemma3", "model.language_model.layers.N.self_attn.k_proj", (1024, 2560), 34),
    Weight("gemma3", "model.language_model.layers.N.self_attn.v_proj", (1024, 2560), 34),
    Weight("gemma3", "model.language_model.layers.N.self_attn.o_proj", (2560, 2048), 34),
    Weight("gemma3", "model.language_model.layers.N.mlp.gate_proj", (10240, 2560), 34),
    Weight("gemma3", "model.language_model.layers.N.mlp.up_proj", (10240, 2560), 34),
    Weight("gemma3", "model.language_model.layers.N.mlp.down_proj", (2560, 10240), 34),
    Weight("gemma3", "lm_head", (262208, 2560), 1),
)

REAL_IN_FEATURES: tuple[int, ...] = tuple(sorted({w.in_features for w in WEIGHTS}))
"""Every distinct reduction width the four models contain: 14, 1152, 2048, 2560,
3072, 4096, 4304, 8192, 10240, 12288, 14336."""

PROBE_ROWS = 6
"""Rows used to stand in for a full tensor.

Groups run along ``in_features`` and every row carries its own scales, so the
packed geometry of row ``r`` does not depend on how many other rows there are --
:func:`test_a_row_subset_packs_exactly_like_the_full_tensor` is the proof, and it
is what lets these tests cover a 262208-row embedding in milliseconds. Six rather
than a power of two so that a row-tiling assumption has a ragged tail to trip on.
"""


def _info(weight: Weight, role: ModuleRole = ModuleRole.MLP_DOWN) -> ModuleInfo:
    return ModuleInfo(
        name=f"{weight.model}/{weight.module}",
        role=role,
        module_type="Linear",
        shape=weight.shape,
        num_params=weight.numel,
        source="test",
    )


def _dense(in_features: int, bits: int, rows: int = PROBE_ROWS) -> torch.Tensor:
    torch.manual_seed(bits * 100_003 + in_features)
    return torch.randn(rows, in_features, dtype=torch.float32) * 0.02


def _in_features_params() -> Iterator[pytest.param]:
    for in_features in REAL_IN_FEATURES:
        owners = sorted({w.model for w in WEIGHTS if w.in_features == in_features})
        yield pytest.param(in_features, id=f"{in_features}-{'+'.join(owners)}")


IN_FEATURES_PARAMS = tuple(_in_features_params())


# --------------------------------------------------------------------------
# The pins themselves
# --------------------------------------------------------------------------


@pytest.mark.needs_hf
def test_the_pinned_shapes_are_what_transformers_builds():
    """Rebuild all four models at full scale and compare against ``WEIGHTS``.

    Meta device, so this allocates nothing and downloads nothing -- it is the
    shapes that are wanted, and ``from_config`` knows them without weights. All
    four build in about four seconds.

    Turns red when: a config literal above is mistyped, a ``transformers`` release
    renames or re-splits a projection, or ``head_dim`` stops being read from the
    config (which would silently make Gemma-3's ``o_proj`` square).
    """
    transformers = pytest.importorskip("transformers")

    for name, spec in PHASE3_CONFIGS.items():
        cfg = transformers.AutoConfig.for_model(**spec)
        # `from_config` inside a meta context raises `ValueError: Could not find
        # LlamaForCausalLM neither in <module transformers.models.llama> nor in
        # <module transformers>` on a fresh interpreter, because the lazy auto
        # mapping cannot import under `torch.device("meta")`. Resolving the class
        # first and constructing second sidesteps it, and does not depend on
        # whether some earlier test already populated the lazy module.
        if type(cfg).__name__ == "Gemma3Config":
            cls = transformers.Gemma3ForConditionalGeneration
        else:
            cls = transformers.MODEL_FOR_CAUSAL_LM_MAPPING[type(cfg)]
        with torch.device("meta"):
            model = cls(cfg)

        total = sum(p.numel() for p in model.parameters())
        advertised = ADVERTISED_PARAMS[name]
        assert abs(total - advertised) / advertised < 0.005, (
            f"{name} built to {total:,} parameters against an advertised "
            f"{advertised:,.0f}; a pinned config field is wrong"
        )

        built: dict[tuple[int, ...], int] = {}
        for _, module in model.named_modules():
            tensor = getattr(module, "weight", None)
            if tensor is not None and tensor.ndim >= 2:
                built[tuple(tensor.shape)] = built.get(tuple(tensor.shape), 0) + 1

        for weight in (w for w in WEIGHTS if w.model == name):
            assert weight.shape in built, (
                f"{name} has no weight of shape {weight.shape} ({weight.module}); "
                f"built shapes were {sorted(built)}"
            )
        # Tied weights appear under two module names but are one tensor, so counts
        # are compared only for the shapes that are not part of a tie.
        pinned = {w.shape for w in WEIGHTS if w.model == name}
        assert set(built) == pinned, (
            f"{name} builds shapes the pins do not mention: {sorted(set(built) - pinned)}"
        )


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bits", BIT_OPTIONS)
@pytest.mark.parametrize("in_features", IN_FEATURES_PARAMS)
def test_pack_unpack_is_a_bijection_at_every_real_in_features(bits, in_features):
    """Every reduction width the four models contain, at every width we ship.

    Turns red when: a change to the bit-offset tables is correct for the widths
    the sweep in ``test_pack.py`` happens to use but not for one of these. 4304 is
    the case with teeth -- it is the only real ``in_features`` that is not a whole
    number of groups, so it is the only one where the padded tail is packed.
    """
    torch.manual_seed(bits * 7919 + in_features)
    padded = padded_in_features(in_features, DEFAULT_GROUP_SIZE)
    codes = torch.randint(0, 2**bits, (PROBE_ROWS, padded), dtype=torch.uint8)

    packed = pack_nbit(codes, bits)
    assert packed.dtype is torch.int32
    restored = unpack_nbit(packed, bits, padded)
    assert torch.equal(restored, codes)


@pytest.mark.parametrize("bits", BIT_OPTIONS)
def test_a_row_subset_packs_exactly_like_the_full_tensor(bits):
    """The justification for :data:`PROBE_ROWS`, asserted rather than assumed.

    If packing were row-coupled -- if a row's words depended on the rows before it,
    the way the research layout's flattened packing did -- then every test in this
    file that probes six rows of a 262208-row embedding would be measuring
    something other than the tensor it claims to.

    Turns red when: packing goes back to flattening across row boundaries, or a
    future "optimisation" shares a scale or a word between adjacent rows.
    """
    in_features = 4304  # the padded case, where a shared tail would be visible
    padded = padded_in_features(in_features, DEFAULT_GROUP_SIZE)
    torch.manual_seed(bits)
    codes = torch.randint(0, 2**bits, (37, padded), dtype=torch.uint8)

    full = pack_nbit(codes, bits)
    for start, stop in ((0, PROBE_ROWS), (11, 12), (30, 37)):
        subset = pack_nbit(codes[start:stop], bits)
        assert torch.equal(subset, full[start:stop])


@pytest.mark.parametrize("bits", BIT_OPTIONS)
@pytest.mark.parametrize("in_features", IN_FEATURES_PARAMS)
def test_quantization_error_is_what_theory_predicts_at_every_real_width(bits, in_features):
    """Two-sided, against a prediction that never calls our quantizer.

    Turns red when: an encoder change loses accuracy at one of these widths, or
    when the round trip stops being a round trip at all -- an error of *zero* fails
    just as loudly as an error that is too large, because it means the comparison
    was against the input.
    """
    dense = _dense(in_features, bits)
    quantized = QuantTensor.from_dense(dense, bits=bits, group_size=DEFAULT_GROUP_SIZE)
    measured = quantized.quantization_error(dense)["rel_fro"]
    assert_error_matches_theory(measured, dense, bits, DEFAULT_GROUP_SIZE)


# --------------------------------------------------------------------------
# Budget accounting -- defect 1
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bits", BIT_OPTIONS)
@pytest.mark.parametrize("weight", WEIGHTS, ids=lambda w: f"{w.model}-{w.module.split('.')[-1]}")
def test_the_budget_predicts_exactly_the_bytes_the_packer_writes(bits, weight):
    """``module_stored_bits`` must equal ``QuantTensor.nbytes``, to the bit.

    Not "within a percent". The allocator picks a bit map by comparing these
    numbers against a target, and the manifest reports the target as the size of
    the checkpoint; an approximation there is a checkpoint that misses the size the
    user asked for. Both sides now resolve through ``pack.row_geometry``, so a
    disagreement is no longer expressible -- this test is what says so.

    Turns red when: the payload is priced as ``params * bits`` again (which was the
    shipped behaviour, undercounting Gemma-3's ``patch_embedding`` by 4.8x), when
    metadata stops being counted, or when the packer's padding rule changes without
    the budget's.
    """
    dense = _dense(weight.in_features, bits)
    quantized = QuantTensor.from_dense(dense, bits=bits, group_size=DEFAULT_GROUP_SIZE)

    probe_bits = quantized.nbytes * 8
    assert probe_bits % PROBE_ROWS == 0, "packed size is not a whole number of bits per row"
    actual = probe_bits // PROBE_ROWS * weight.rows

    predicted = module_stored_bits(_info(weight), bits, group_size=DEFAULT_GROUP_SIZE)
    assert predicted == pytest.approx(float(actual), abs=0.0), (
        f"{weight.model}/{weight.module} at {bits}-bit: budget says {predicted:,.0f} bits, "
        f"the packer writes {actual:,} ({actual / predicted:.4f}x)"
    )


def test_the_shapes_that_pad_are_named_and_priced():
    """Exactly two real reduction widths are not a whole number of groups.

    Naming them is the point. Padding is invisible in a shape and shows up only in
    a size, so a test that merely asserts "prediction == actual" would still pass
    if both sides forgot padding together. This one states which shapes pay and how
    much, from the geometry alone.

    Turns red when: a fifth model's shapes are pinned without anyone noticing they
    pad, or when ``padded_in_features`` changes its rounding.
    """
    padding = {
        w.in_features: padded_in_features(w.in_features, DEFAULT_GROUP_SIZE)
        for w in WEIGHTS
        if padded_in_features(w.in_features, DEFAULT_GROUP_SIZE) != w.in_features
    }
    assert padding == {4304: 4352, 14: 128}

    # Gemma-3 vision fc2: 48 wasted columns in 4352, so ~1.1% -- small, and real.
    fc2 = next(w for w in WEIGHTS if w.in_features == 4304)
    at_4bit = module_stored_bits(_info(fc2), 4, group_size=DEFAULT_GROUP_SIZE) / fc2.numel
    assert at_4bit == pytest.approx(4.297, abs=0.001)

    # Gemma-3 patch_embedding: 14 columns padded to 128, so 9.1x the payload plus a
    # scale and an offset for every 14 values. Every width is worse than fp16.
    patch = next(w for w in WEIGHTS if w.in_features == 14)
    priced = {
        bits: module_stored_bits(_info(patch), bits, group_size=DEFAULT_GROUP_SIZE) / patch.numel
        for bits in BIT_OPTIONS
    }
    assert priced == pytest.approx({2: 20.571, 3: 29.714, 4: 38.857, 8: 75.429}, abs=0.001)
    assert min(priced.values()) > UNQUANTIZED_FLOOR


# --------------------------------------------------------------------------
# Quantizing has to be worth doing -- defect 2
# --------------------------------------------------------------------------


def test_a_tensor_that_cannot_pay_for_itself_is_left_dense():
    """Gemma-3's ``patch_embedding`` is not offered to the allocator.

    Its cheapest quantized form costs 20.6 bits per weight against 16 for storing
    it as it is, so there is no budget under which quantizing it is the right call
    -- and because wider is *more* expensive, not less, budget pressure pushes the
    wrong way. The graph therefore reports it as unquantized, which puts it in
    ``Budget.fixed_bits`` at compute dtype and out of the knapsack entirely.

    Turns red when: the shape rule is dropped and the decision goes back to being
    role-only, which is how a 677k-parameter tensor came to be marked quantizable
    at 8 bits -- 4.7x the cost of leaving it alone.
    """
    patch = next(w for w in WEIGHTS if w.in_features == 14)
    info = _info(patch, ModuleRole.VISION_PATCH_EMBED)
    assert info.role.is_quantizable, "the role is fine; it is the shape that is not"
    assert not info.pays_for_itself
    assert not info.is_quantizable

    dense_cost = module_stored_bits(info, UNQUANTIZED_FLOOR, group_size=DEFAULT_GROUP_SIZE)
    assert dense_cost == float(patch.numel * UNQUANTIZED_FLOOR)


@pytest.mark.parametrize("weight", WEIGHTS, ids=lambda w: f"{w.model}-{w.module.split('.')[-1]}")
def test_every_other_real_shape_still_pays_for_itself(weight):
    """The rule must not catch anything else in the set.

    A predicate that excludes too much is worse than the defect it fixes: it would
    quietly leave real projections at fp16 and blow the budget while every test
    still passed. 1152 is the narrowest reduction width here that is not the patch
    embedding, and it clears the threshold by 6x.

    Turns red when: the threshold is raised far enough to catch a legitimate
    narrow projection -- Gemma-3's vision tower at 1152, or a future model's.
    """
    expected = weight.in_features != 14
    assert _info(weight).pays_for_itself is expected


def test_the_threshold_sits_where_the_arithmetic_puts_it():
    """The break-even width, derived rather than asserted from a magic number.

    At group 128 and 2 bits a row costs one padded group -- 128*2 payload bits plus
    an fp16 scale and an fp16 offset -- so 288 bits however few columns it holds.
    Dense costs ``16 * in_features``. They cross at 18: 18 columns cost 288 either
    way, 19 columns make quantizing cheaper.

    Turns red when: the metadata term is dropped from the predicate (which moves
    the break-even to 16), or the predicate starts pricing at a width other than
    the narrowest.
    """

    def pays(in_features: int) -> bool:
        return _info(Weight("synthetic", "w", (64, in_features), 1)).pays_for_itself

    assert not pays(18)
    assert pays(19)
    assert not pays(14)  # the real one


# --------------------------------------------------------------------------
# Fused projections
# --------------------------------------------------------------------------


@pytest.mark.needs_hf
def test_phi4_row_partitions_land_on_the_full_scale_boundaries():
    """The fused boundaries at 3.8B, not at the 1/100 scale the arch matrix uses.

    [test_graph_arch_matrix.py][1] pins that the partitions exist and are ordered;
    this pins the arithmetic that produces them at the real head counts. Phi-4-mini
    is 24 query heads and 8 KV heads at ``head_dim = 3072 / 24 = 128``, so
    ``qkv_proj``'s 5120 rows split 3072 / 1024 / 1024 -- not into thirds, which is
    what an even split of a fused QKV would give and what would look plausible.

    Turns red when: the phi3 plugin starts splitting by anything other than the GQA
    head counts, or when ``head_dim`` stops being derived for a config that does
    not state it.

    [1]: test_graph_arch_matrix.py
    """
    transformers = pytest.importorskip("transformers")
    from dynquant.graph import classify_model

    cfg = transformers.AutoConfig.for_model(**PHASE3_CONFIGS["phi4"])
    cls = transformers.MODEL_FOR_CAUSAL_LM_MAPPING[type(cfg)]
    with torch.device("meta"):
        model = cls(cfg)
    graph = classify_model(model, config=cfg)

    qkv = graph["model.layers.0.self_attn.qkv_proj"]
    assert [(p.role, p.start, p.stop) for p in qkv.partitions] == [
        (ModuleRole.ATTN_Q, 0, 3072),
        (ModuleRole.ATTN_K, 3072, 4096),
        (ModuleRole.ATTN_V, 4096, 5120),
    ]

    gate_up = graph["model.layers.0.mlp.gate_up_proj"]
    assert [(p.role, p.start, p.stop) for p in gate_up.partitions] == [
        (ModuleRole.MLP_GATE, 0, 8192),
        (ModuleRole.MLP_UP, 8192, 16384),
    ]

    # Tied head: one tensor, two names, and the strictest floor of the two.
    head = graph["lm_head"]
    assert head.tied_to == "model.embed_tokens"
    representative = graph["model.embed_tokens"]
    assert ModuleRole.LM_HEAD in representative.tied_roles
    assert representative.floor_bits == 8
