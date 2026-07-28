"""Canonical naming: one spelling per weight, whatever wrapped it.

Every entry here is a spelling that really occurs. When canonicalisation misses
one, the layer's signal reads as absent and the allocator scores it as
unimportant -- a silent quality loss with no error message anywhere.
"""

from __future__ import annotations

import pytest

from dynquant.graph.naming import canonical_name, is_adapter_name, strip_wrappers

CASES = [
    # bare
    ("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.q_proj"),
    # PEFT LoRA
    (
        "base_model.model.model.layers.0.self_attn.q_proj.base_layer",
        "model.layers.0.self_attn.q_proj",
    ),
    ("base_model.model.lm_head", "lm_head"),
    ("base_model.model.lm_head.base_layer", "lm_head"),
    # DDP
    ("module.model.layers.3.mlp.gate_proj", "model.layers.3.mlp.gate_proj"),
    # torch.compile
    ("_orig_mod.model.layers.3.mlp.gate_proj", "model.layers.3.mlp.gate_proj"),
    # stacked wrappers, plus a parameter suffix
    ("module._orig_mod.model.layers.3.mlp.gate_proj.weight", "model.layers.3.mlp.gate_proj"),
    # FSDP, which interleaves wrappers mid-path
    (
        "_fsdp_wrapped_module.model.layers.1._fsdp_wrapped_module.self_attn.o_proj",
        "model.layers.1.self_attn.o_proj",
    ),
    # activation checkpointing
    (
        "model.layers.2._checkpoint_wrapped_module.mlp.down_proj",
        "model.layers.2.mlp.down_proj",
    ),
    # adapter tensors attribute to the base weight they will be merged into
    (
        "base_model.model.model.layers.0.mlp.up_proj.lora_A.default.weight",
        "model.layers.0.mlp.up_proj",
    ),
    (
        "base_model.model.model.layers.0.mlp.up_proj.lora_B.default.weight",
        "model.layers.0.mlp.up_proj",
    ),
    # DoRA magnitude vector
    (
        "base_model.model.model.layers.0.mlp.up_proj.lora_magnitude_vector.default",
        "model.layers.0.mlp.up_proj",
    ),
    # fused projections keep their fused name; splitting happens later, by role
    (
        "base_model.model.model.layers.0.self_attn.qkv_proj.base_layer",
        "model.layers.0.self_attn.qkv_proj",
    ),
    (
        "base_model.model.model.layers.0.mlp.gate_up_proj.base_layer",
        "model.layers.0.mlp.gate_up_proj",
    ),
    # MoE experts
    (
        "base_model.model.model.layers.0.mlp.experts.7.w2.base_layer",
        "model.layers.0.mlp.experts.7.w2",
    ),
    ("model.layers.0.mlp.gate", "model.layers.0.mlp.gate"),
    # MLA
    ("model.layers.0.self_attn.kv_a_proj_with_mqa", "model.layers.0.self_attn.kv_a_proj_with_mqa"),
    # vision tower
    (
        "vision_tower.vision_model.encoder.layers.0.self_attn.k_proj",
        "vision_tower.vision_model.encoder.layers.0.self_attn.k_proj",
    ),
    ("multi_modal_projector.linear_1", "multi_modal_projector.linear_1"),
]


@pytest.mark.parametrize(("raw", "expected"), CASES)
def test_canonical_name(raw, expected):
    assert canonical_name(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), CASES)
def test_canonical_name_is_idempotent(raw, expected):
    assert canonical_name(canonical_name(raw)) == expected


def test_lora_a_and_b_collapse_to_the_same_key():
    """The bug-7 mechanism, turned into a guarantee.

    ``lora_A.weight`` is ``[r, in]`` and ``lora_B.weight`` is ``[out, r]``. The
    research tracker recorded both under the parent's name without reconciling
    them, so the per-channel coherence buffer alternated length between ``r`` and
    ``out`` and ``torch.dot`` raised into a bare ``except``. Collapsing them to
    one key deliberately -- and merging the accumulators -- is what makes the
    attribution correct instead of accidental.
    """
    base = "base_model.model.model.layers.0.mlp.up_proj"
    assert canonical_name(f"{base}.lora_A.default.weight") == "model.layers.0.mlp.up_proj"
    assert canonical_name(f"{base}.lora_B.default.weight") == "model.layers.0.mlp.up_proj"
    assert canonical_name(f"{base}.base_layer.weight") == "model.layers.0.mlp.up_proj"


def test_weight_suffix_can_be_kept():
    assert (
        canonical_name("model.layers.0.mlp.up_proj.weight", strip_param_suffix=False)
        == "model.layers.0.mlp.up_proj.weight"
    )


def test_bias_suffix_is_not_stripped():
    """Only ``.weight`` is stripped; a bias is a different tensor."""
    assert canonical_name("model.layers.0.mlp.up_proj.bias").endswith(".bias")


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.up_proj.lora_A.default.weight",
        "model.layers.0.mlp.up_proj.lora_B.default.weight",
        "model.layers.0.mlp.up_proj.lora_magnitude_vector.default",
        "base_model.model.model.layers.0.vera_A",
    ],
)
def test_is_adapter_name_detects_adapters(name):
    assert is_adapter_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.up_proj.base_layer.weight",
        "model.layers.0.mlp.gate",
        "model.embed_tokens",
    ],
)
def test_is_adapter_name_leaves_base_weights_alone(name):
    assert not is_adapter_name(name)


def test_module_is_only_stripped_at_the_front():
    """``module`` is too common a word to remove from the middle of a path.

    A model with a submodule genuinely named ``module`` must not lose it.
    """
    assert strip_wrappers("module.model.module.layers.0") == "model.module.layers.0"


def test_bare_model_prefix_is_preserved():
    """Only ``base_model.model`` (the PEFT pair) collapses -- a plain ``model.``
    prefix is the real model attribute and must stay."""
    assert canonical_name("model.model.layers.0.q_proj") == "model.model.layers.0.q_proj"
