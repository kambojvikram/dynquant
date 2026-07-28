"""Role classification and safety floors.

The four cases the module docstring calls out -- MoE router, fused ``gate_up_proj``,
MLA ``kv_a_proj``, Mamba ``in_proj`` -- get their own named tests, because each was
a silent quality collapse in the research code and a regression on any of them
would be equally silent.

These test the *name-based fallback*, the last resort in the resolution chain. The
structural classifier (phase 3) is what production paths use; this is what
``dynquant inspect`` has when handed nothing but a stats JSON.
"""

from __future__ import annotations

import pytest

from dynquant.constants import BIT_OPTIONS
from dynquant.graph.roles import (
    DEFAULT_FLOOR_BITS,
    NEVER_QUANTIZE,
    UNQUANTIZED_FLOOR,
    ModuleRole,
    RowPartition,
    role_of_name,
)

# --------------------------------------------------------------------------
# The four regressions
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.gate",  # Qwen3-MoE, OLMoE, DeepSeek
        "model.layers.0.block_sparse_moe.gate",  # Mixtral
        "model.layers.0.mlp.router",  # GPT-OSS
        "model.layers.0.mlp.gating",
        "model.layers.0.mlp.shared_expert_gate",  # Qwen2-MoE
    ],
)
def test_moe_router_is_never_confused_with_a_swiglu_gate(name):
    """The supplement's substring matcher gave routers 3 bits via the ``mlp``
    catch-all. A perturbed router sends tokens through the wrong experts, which
    nothing downstream can recover from."""
    assert role_of_name(name) is ModuleRole.MOE_ROUTER
    assert DEFAULT_FLOOR_BITS[ModuleRole.MOE_ROUTER] == 8


def test_swiglu_gate_is_not_confused_with_a_router():
    """The converse -- and the reason the router test is a leaf check, not a
    substring check."""
    assert role_of_name("model.layers.0.mlp.gate_proj") is ModuleRole.MLP_GATE


def test_fused_gate_up_splits_into_parts():
    """Phi-4 / Gemma fuse gate and up. The supplement's ``mlp`` catch-all gave the
    whole tensor 3 bits, including the gate half the paper says needs 4."""
    role = role_of_name("model.layers.0.mlp.gate_up_proj")
    assert role is ModuleRole.MLP_GATE_UP
    assert role.is_fused
    assert role.fused_parts == (ModuleRole.MLP_GATE, ModuleRole.MLP_UP)
    assert DEFAULT_FLOOR_BITS[ModuleRole.MLP_GATE] == 4


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.layers.0.self_attn.kv_a_proj_with_mqa", ModuleRole.MLA_KV_A),
        ("model.layers.0.self_attn.kv_b_proj", ModuleRole.MLA_KV_B),
        ("model.layers.0.self_attn.q_a_proj", ModuleRole.MLA_Q_A),
        ("model.layers.0.self_attn.q_b_proj", ModuleRole.MLA_Q_B),
    ],
)
def test_mla_projections_are_recognised(name, expected):
    """DeepSeek-V2/V3. The supplement matched none of these, so they hit the
    2-bit floor -- destroying the low-rank bottleneck every head reads through."""
    assert role_of_name(name) is expected


def test_mla_bottlenecks_get_eight_bits():
    assert DEFAULT_FLOOR_BITS[ModuleRole.MLA_KV_A] == 8
    assert DEFAULT_FLOOR_BITS[ModuleRole.MLA_Q_A] == 8


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("backbone.layers.0.mixer.in_proj", ModuleRole.SSM_IN),
        ("backbone.layers.0.mixer.x_proj", ModuleRole.SSM_X),
        ("backbone.layers.0.mixer.dt_proj", ModuleRole.SSM_DT),
        ("backbone.layers.0.mixer.out_proj", ModuleRole.ATTN_O),
        ("backbone.layers.0.mixer.conv1d", ModuleRole.SSM_CONV),
    ],
)
def test_ssm_projections_are_recognised(name, expected):
    assert role_of_name(name) is expected


def test_ssm_floors_protect_the_exponentiated_paths():
    """``dt`` is exponentiated and ``x_proj`` emits B/C/dt -- both tiny, both
    catastrophic to quantize hard."""
    assert DEFAULT_FLOOR_BITS[ModuleRole.SSM_X] == 8
    assert DEFAULT_FLOOR_BITS[ModuleRole.SSM_DT] == 8


# --------------------------------------------------------------------------
# Architecture coverage
# --------------------------------------------------------------------------

ARCHITECTURE_NAMES = {
    "llama/qwen dense": [
        ("model.layers.0.self_attn.q_proj", ModuleRole.ATTN_Q),
        ("model.layers.0.self_attn.k_proj", ModuleRole.ATTN_K),
        ("model.layers.0.self_attn.v_proj", ModuleRole.ATTN_V),
        ("model.layers.0.self_attn.o_proj", ModuleRole.ATTN_O),
        ("model.layers.0.mlp.gate_proj", ModuleRole.MLP_GATE),
        ("model.layers.0.mlp.up_proj", ModuleRole.MLP_UP),
        ("model.layers.0.mlp.down_proj", ModuleRole.MLP_DOWN),
        ("model.embed_tokens", ModuleRole.EMBEDDING),
        ("lm_head", ModuleRole.LM_HEAD),
    ],
    "phi-4 fused": [
        ("model.layers.0.self_attn.qkv_proj", ModuleRole.ATTN_QKV),
        ("model.layers.0.mlp.gate_up_proj", ModuleRole.MLP_GATE_UP),
    ],
    "mixtral experts": [
        ("model.layers.0.block_sparse_moe.experts.0.w1", ModuleRole.MOE_EXPERT_GATE),
        ("model.layers.0.block_sparse_moe.experts.0.w3", ModuleRole.MOE_EXPERT_UP),
        ("model.layers.0.block_sparse_moe.experts.0.w2", ModuleRole.MOE_EXPERT_DOWN),
    ],
    "qwen3-moe experts": [
        ("model.layers.0.mlp.experts.5.gate_proj", ModuleRole.MOE_EXPERT_GATE),
        ("model.layers.0.mlp.experts.5.up_proj", ModuleRole.MOE_EXPERT_UP),
        ("model.layers.0.mlp.experts.5.down_proj", ModuleRole.MOE_EXPERT_DOWN),
    ],
    "deepseek shared expert": [
        ("model.layers.0.mlp.shared_experts.gate_proj", ModuleRole.MOE_SHARED_GATE),
        ("model.layers.0.mlp.shared_experts.up_proj", ModuleRole.MOE_SHARED_UP),
        ("model.layers.0.mlp.shared_experts.down_proj", ModuleRole.MOE_SHARED_DOWN),
    ],
    "gpt-oss experts": [
        ("model.layers.0.mlp.experts.gate_up_proj", ModuleRole.MOE_EXPERT_GATE_UP),
        ("model.layers.0.mlp.experts.down_proj", ModuleRole.MOE_EXPERT_DOWN),
    ],
    "falcon / gpt-neox": [
        ("transformer.h.0.self_attention.query_key_value", ModuleRole.ATTN_QKV),
        ("transformer.h.0.mlp.dense_h_to_4h", ModuleRole.MLP_UP),
        ("transformer.h.0.mlp.dense_4h_to_h", ModuleRole.MLP_DOWN),
    ],
    "gpt-2 style": [
        ("transformer.h.0.mlp.c_fc", ModuleRole.MLP_UP),
        ("transformer.h.0.mlp.c_proj", ModuleRole.MLP_DOWN),
        ("transformer.wte", ModuleRole.EMBEDDING),
    ],
    "vlm": [
        ("vision_tower.vision_model.encoder.layers.0.self_attn.q_proj", ModuleRole.VISION_ATTN),
        ("vision_tower.vision_model.encoder.layers.0.mlp.fc1", ModuleRole.VISION_MLP),
        ("multi_modal_projector.linear_1", ModuleRole.MM_PROJECTOR),
    ],
    "norms": [
        ("model.layers.0.input_layernorm", ModuleRole.NORM),
        ("model.norm", ModuleRole.NORM),
        ("model.layers.0.post_attention_layernorm", ModuleRole.NORM),
        ("model.rotary_emb", ModuleRole.ROTARY),
    ],
}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param(name, expected, id=f"{family}:{name.split('.')[-1]}")
        for family, cases in ARCHITECTURE_NAMES.items()
        for name, expected in cases
    ],
)
def test_architecture_matrix(name, expected):
    assert role_of_name(name) is expected


def test_expert_weights_are_marked_as_experts():
    role = role_of_name("model.layers.0.block_sparse_moe.experts.0.w2")
    assert role.is_expert
    assert role.is_moe


def test_shared_experts_are_moe_but_not_experts():
    """Shared experts run for every token, so they are priced like a dense FFN
    rather than like a rarely-routed expert."""
    role = role_of_name("model.layers.0.mlp.shared_experts.down_proj")
    assert role.is_moe
    assert not role.is_expert
    assert DEFAULT_FLOOR_BITS[role] == DEFAULT_FLOOR_BITS[ModuleRole.MLP_DOWN]


# --------------------------------------------------------------------------
# Invariants over the role table itself
# --------------------------------------------------------------------------


def test_every_quantizable_role_has_a_floor():
    """A role with no floor entry would be allocated blind."""
    missing = [r.value for r in ModuleRole if r.is_quantizable and r not in DEFAULT_FLOOR_BITS]
    assert not missing, f"roles without a floor: {missing}"


def test_never_quantize_roles_have_no_floor():
    for role in NEVER_QUANTIZE:
        assert role not in DEFAULT_FLOOR_BITS
        assert not role.is_quantizable


def test_every_floor_is_a_supported_bit_width_or_the_compute_dtype():
    """Floors are packable widths, plus one sentinel meaning "do not pack".

    ``UNQUANTIZED_FLOOR`` is deliberately *outside* ``BIT_OPTIONS`` so that no
    packing can satisfy it and the allocator must route the tensor around
    quantization rather than rounding it up to the widest available option.
    """
    allowed = {*BIT_OPTIONS, UNQUANTIZED_FLOOR}
    bad = {r.value: b for r, b in DEFAULT_FLOOR_BITS.items() if b not in allowed}
    assert not bad, f"floors outside {sorted(allowed)}: {bad}"
    assert UNQUANTIZED_FLOOR not in BIT_OPTIONS


def test_conv1d_roles_agree_across_architectures():
    """The same tensor shape gets the same answer in a DeltaNet and in a Mamba.

    Both are ``[channels, 1, kernel]`` depthwise convolutions with a handful of
    taps. Group-wise quantization stores an fp16 scale per group, so at 4 taps the
    scale outweighs the payload -- there is nothing to win, and the output feeds a
    recurrence where the error would not be averaged away. Letting the two roles
    disagree would make the floor depend on which block the tensor sits in.
    """
    assert DEFAULT_FLOOR_BITS[ModuleRole.SSM_CONV] == UNQUANTIZED_FLOOR
    assert DEFAULT_FLOOR_BITS[ModuleRole.LIN_ATTN_CONV] == UNQUANTIZED_FLOOR


def test_unknown_modules_get_a_cautious_floor_not_the_minimum():
    """An architecture we have not seen must degrade to cautious, not destroyed."""
    assert role_of_name("model.layers.0.some_novel_projection") is ModuleRole.OTHER
    assert DEFAULT_FLOOR_BITS[ModuleRole.OTHER] > min(BIT_OPTIONS)
    assert DEFAULT_FLOOR_BITS[ModuleRole.OTHER] == 4


def test_fused_roles_have_parts_and_others_do_not():
    for role in ModuleRole:
        if role.is_fused:
            assert len(role.fused_parts) >= 2
            assert all(p in DEFAULT_FLOOR_BITS for p in role.fused_parts)
        else:
            assert role.fused_parts == ()


def test_fused_floor_is_at_least_the_max_of_its_parts():
    """A fused tensor quantized as one unit must not be cheaper than its most
    demanding half, or fusing a model would silently lower its floor."""
    for role in ModuleRole:
        if not role.is_fused:
            continue
        assert DEFAULT_FLOOR_BITS[role] >= max(DEFAULT_FLOOR_BITS[p] for p in role.fused_parts)


def test_roles_serialise_as_their_string_values():
    assert ModuleRole.MOE_ROUTER == "moe.router"
    import json

    assert json.dumps({"role": ModuleRole.MOE_ROUTER}) == '{"role": "moe.router"}'


def test_role_values_are_unique():
    values = [r.value for r in ModuleRole]
    assert len(values) == len(set(values))


# --------------------------------------------------------------------------
# RowPartition
# --------------------------------------------------------------------------


def test_row_partition_geometry():
    part = RowPartition(ModuleRole.MLP_GATE, 0, 8192)
    assert part.num_rows == 8192
    assert part.role is ModuleRole.MLP_GATE


def test_row_partitions_of_a_fused_projection_tile_exactly():
    """A gate_up_proj of 2*I rows splits into two I-row halves with no gap or
    overlap -- the property the shard writer depends on."""
    intermediate = 8192
    parts = [
        RowPartition(ModuleRole.MLP_GATE, 0, intermediate),
        RowPartition(ModuleRole.MLP_UP, intermediate, 2 * intermediate),
    ]
    assert parts[0].stop == parts[1].start
    assert sum(p.num_rows for p in parts) == 2 * intermediate
