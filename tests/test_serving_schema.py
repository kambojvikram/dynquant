"""The ``quantization_config`` block: what the exporter writes and a server reads back.

This is the whole contract between the two halves. ``from_config`` receives the
parsed dict and nothing else -- no model path, no files -- so anything the loader
needs has to survive this round trip or it does not exist. That signature is the same
in vLLM and in SGLang, so this contract is one contract, not two.
"""

from __future__ import annotations

import json

import pytest

from dynquant.constants import DEFAULT_GROUP_SIZE, HF_QUANT_METHOD
from dynquant.errors import DynQuantError, FormatVersionError, PackingError
from dynquant.integration.serving_common.schema import (
    CHECKPOINT_FORMAT,
    SCHEMA_VERSION,
    ModuleQuantSpec,
    QuantizationConfigSchema,
    expand_fused_prefix,
)

LLAMA_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def schema(**overrides) -> QuantizationConfigSchema:
    modules = overrides.pop(
        "modules",
        {
            "model.layers.0.self_attn.q_proj": ModuleQuantSpec(4, 128, False),
            "model.layers.0.self_attn.k_proj": ModuleQuantSpec(3, 128, False),
            "model.layers.0.self_attn.v_proj": ModuleQuantSpec(3, 128, False),
            "model.layers.0.mlp.gate_proj": ModuleQuantSpec(4, 128, False),
            "model.layers.0.mlp.up_proj": ModuleQuantSpec(2, 64, False),
        },
    )
    return QuantizationConfigSchema(modules=modules, **overrides)


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------


def test_round_trip_through_json_preserves_every_width():
    original = schema(version="0.1.2", lm_head_quantized=True)
    restored = QuantizationConfigSchema.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored.modules == dict(original.modules)
    assert restored.group_size == original.group_size
    assert restored.symmetric == original.symmetric
    assert restored.lm_head_quantized is True
    assert restored.version == "0.1.2"


def test_serialised_block_names_the_method_vllm_dispatches_on():
    payload = schema().to_dict()
    assert payload["quant_method"] == HF_QUANT_METHOD
    assert payload["checkpoint_format"] == CHECKPOINT_FORMAT
    assert payload["schema_version"] == SCHEMA_VERSION


def test_per_module_entries_omit_what_matches_the_file_defaults():
    """A 200-module map that repeats the defaults on every line is 3x the size."""
    payload = schema().to_dict()
    assert payload["modules"]["model.layers.0.self_attn.q_proj"] == {"bits": 4}
    # ...and states what differs.
    assert payload["modules"]["model.layers.0.mlp.up_proj"] == {"bits": 2, "group_size": 64}


def test_bare_int_shorthand_is_accepted():
    restored = QuantizationConfigSchema.from_dict(
        {
            "quant_method": HF_QUANT_METHOD,
            "checkpoint_format": CHECKPOINT_FORMAT,
            "modules": {"model.layers.0.mlp.up_proj": 3},
        }
    )
    assert restored.get("model.layers.0.mlp.up_proj") == ModuleQuantSpec(
        3, DEFAULT_GROUP_SIZE, False
    )


def test_defaults_propagate_into_modules_that_do_not_override_them():
    restored = QuantizationConfigSchema.from_dict(
        {
            "quant_method": HF_QUANT_METHOD,
            "checkpoint_format": CHECKPOINT_FORMAT,
            "group_size": 64,
            "symmetric": True,
            "modules": {"a": {"bits": 4}, "b": {"bits": 4, "symmetric": False}},
        }
    )
    assert restored.get("a") == ModuleQuantSpec(4, 64, True)
    assert restored.get("b") == ModuleQuantSpec(4, 64, False)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_a_values_quantized_directory_is_refused_by_format_not_by_crash():
    """`dynquant quantize` output has the same quant_method and dense weights."""
    with pytest.raises(DynQuantError, match="dynquant export"):
        QuantizationConfigSchema.from_dict(
            {
                "quant_method": HF_QUANT_METHOD,
                "checkpoint_format": "dense-values",
                "modules": {"a": 4},
            }
        )


def test_another_methods_config_is_refused():
    with pytest.raises(DynQuantError, match="gptq"):
        QuantizationConfigSchema.from_dict({"quant_method": "gptq", "modules": {"a": 4}})


def test_a_newer_schema_says_to_upgrade():
    with pytest.raises(FormatVersionError, match="pip install --upgrade"):
        QuantizationConfigSchema.from_dict(
            {
                "quant_method": HF_QUANT_METHOD,
                "checkpoint_format": CHECKPOINT_FORMAT,
                "schema_version": SCHEMA_VERSION + 1,
                "modules": {"a": 4},
            }
        )


def test_an_empty_map_is_an_error_because_there_is_no_fallback_width():
    with pytest.raises(DynQuantError, match="width per module"):
        QuantizationConfigSchema.from_dict(
            {"quant_method": HF_QUANT_METHOD, "checkpoint_format": CHECKPOINT_FORMAT, "modules": {}}
        )


def test_an_unsupported_width_names_the_ones_that_exist():
    with pytest.raises(PackingError, match=r"\[2, 3, 4, 8\]"):
        ModuleQuantSpec(5, 128, False)


def test_a_module_entry_without_bits_is_named():
    with pytest.raises(DynQuantError, match="has no 'bits'"):
        QuantizationConfigSchema.from_dict(
            {
                "quant_method": HF_QUANT_METHOD,
                "checkpoint_format": CHECKPOINT_FORMAT,
                "modules": {"model.layers.0.mlp.up_proj": {"group_size": 128}},
            }
        )


# --------------------------------------------------------------------------
# Fused-layer resolution
# --------------------------------------------------------------------------


def test_expand_fused_prefix_reaches_the_checkpoint_names():
    assert expand_fused_prefix("model.layers.0.self_attn.qkv_proj", LLAMA_MAPPING) == [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    ]


def test_an_unfused_prefix_is_itself():
    assert expand_fused_prefix("model.layers.0.self_attn.o_proj", LLAMA_MAPPING) == [
        "model.layers.0.self_attn.o_proj"
    ]


def test_a_self_referential_mapping_is_the_prefix_itself():
    """Phi-style checkpoints ship already fused; vLLM maps the name to itself."""
    assert expand_fused_prefix("m.qkv_proj", {"qkv_proj": ["qkv_proj"]}) == ["m.qkv_proj"]


def test_resolve_shards_returns_widths_in_vllm_shard_order():
    resolved = schema().resolve_shards("model.layers.0.self_attn.qkv_proj", LLAMA_MAPPING)
    assert [spec.bits for _, spec in resolved] == [4, 3, 3]


def test_resolve_shards_is_none_for_a_layer_left_dense():
    assert schema().resolve_shards("model.layers.0.self_attn.o_proj", LLAMA_MAPPING) is None
    assert schema().resolve_shards("lm_head", LLAMA_MAPPING) is None


def test_a_half_quantized_fused_layer_is_refused():
    """One packed buffer per fused layer, so a mixed layer is not representable."""
    half = schema(
        modules={
            "model.layers.0.mlp.gate_proj": ModuleQuantSpec(4, 128, False),
            # up_proj left dense -- nothing to concatenate its rows onto.
        }
    )
    with pytest.raises(DynQuantError, match="up_proj"):
        half.resolve_shards("model.layers.0.mlp.gate_up_proj", LLAMA_MAPPING)


def test_modules_to_not_convert_wins_over_the_map():
    partial = schema(modules_to_not_convert=("model.layers.0.self_attn.q_proj",))
    assert partial.get("model.layers.0.self_attn.q_proj") is None
    with pytest.raises(DynQuantError, match="q_proj"):
        partial.resolve_shards("model.layers.0.self_attn.qkv_proj", LLAMA_MAPPING)


# --------------------------------------------------------------------------
# vLLM's name mapper
# --------------------------------------------------------------------------


def test_remap_rewrites_names_without_mutating_the_original():
    original = schema()
    renamed = original.remap(
        lambda names: [n.replace("model.layers", "transformer.h") for n in names]
    )

    assert "transformer.h.0.self_attn.q_proj" in renamed.modules
    assert "model.layers.0.self_attn.q_proj" in original.modules
    assert renamed.get("transformer.h.0.self_attn.k_proj").bits == 3


def test_remap_carries_the_dense_list_through_too():
    original = schema(modules_to_not_convert=("model.layers.0.mlp.up_proj",))
    renamed = original.remap(lambda names: [n.upper() for n in names])
    assert renamed.modules_to_not_convert == ("MODEL.LAYERS.0.MLP.UP_PROJ",)


def test_a_mapper_that_loses_a_name_is_caught_rather_than_silently_dropping_a_width():
    with pytest.raises(DynQuantError, match="returned"):
        schema().remap(lambda names: names[:-1])
