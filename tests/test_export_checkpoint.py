"""``dynquant export`` writes a directory a server can actually load.

The checks that matter are structural, and they are the ones that were wrong in
the research code: the weight files have to be named what a loader globs for, the
widths have to survive into ``config.json`` where ``from_config`` can reach them,
a tied embedding must be written once and not twice, and the values must be the
same ones the in-process packed runtime produces -- otherwise a vLLM parity check
is comparing two different quantizations and cannot fail informatively.

Runs on CPU. vLLM is not imported anywhere in this file.
"""

from __future__ import annotations

import json
import re

import pytest

from dynquant.constants import (
    HF_CONFIG_FILENAME,
    HF_QUANT_METHOD,
    HF_WEIGHTS_FILENAME,
    HF_WEIGHTS_INDEX_FILENAME,
    MANIFEST_FILENAME,
)
from dynquant.errors import DynQuantError
from dynquant.integration.serving_common.schema import (
    CHECKPOINT_FORMAT,
    QuantizationConfigSchema,
)

transformers = pytest.importorskip("transformers")
torch = pytest.importorskip("torch")

from dynquant.quant.checkpoint import export_packed_checkpoint  # noqa: E402


def tiny_model(*, tie_word_embeddings: bool = False):
    """A two-layer Llama, small enough to export in under a second."""
    config = transformers.LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=tie_word_embeddings,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).to(torch.float16)
    model.eval()
    return model


def mixed_widths(model) -> dict[str, int]:
    """Per-module widths, deliberately unequal across a fused layer's shards.

    q at 4 bits and k/v at 3 is the case no other quantization method produces and
    the one the flat-buffer layout exists for, so it is the case the export path
    is exercised with rather than a uniform map.
    """
    widths: dict[str, int] = {}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        widths[name] = {
            "q_proj": 4,
            "k_proj": 3,
            "v_proj": 3,
            "o_proj": 4,
            "gate_proj": 4,
            "up_proj": 3,
            "down_proj": 3,
        }.get(leaf, 8)
    return widths


@pytest.fixture
def exported(tmp_path):
    model = tiny_model()
    bits = mixed_widths(model)
    report = export_packed_checkpoint(
        model, bits, output_dir=tmp_path / "ckpt", compute_device=None
    )
    return model, bits, report


# --------------------------------------------------------------------------
# Directory layout
# --------------------------------------------------------------------------


def test_weights_land_where_a_loader_globs_for_them(exported):
    _, _, report = exported
    assert report.files == (HF_WEIGHTS_FILENAME,)
    assert (report.output_dir / HF_WEIGHTS_FILENAME).is_file()
    # One file, so no index: vLLM only consults the index when the glob returns
    # more than one, and an index listing a single file is a second thing to keep
    # consistent for no gain.
    assert not (report.output_dir / HF_WEIGHTS_INDEX_FILENAME).exists()
    assert (report.output_dir / HF_CONFIG_FILENAME).is_file()
    assert (report.output_dir / MANIFEST_FILENAME).is_file()


def test_sharding_writes_an_index_that_names_every_tensor(tmp_path):
    model = tiny_model()
    report = export_packed_checkpoint(
        model,
        mixed_widths(model),
        output_dir=tmp_path / "ckpt",
        compute_device=None,
        max_shard_bytes=16 * 1024,
    )
    assert len(report.files) > 2
    assert HF_WEIGHTS_INDEX_FILENAME in report.files

    index = json.loads((report.output_dir / HF_WEIGHTS_INDEX_FILENAME).read_text())
    from safetensors.torch import load_file

    # Completeness both ways: every tensor named in the index exists in the shard
    # the index names, and no shard holds a tensor the index does not list. A
    # loader that filters by the index would silently drop the second kind.
    on_disk: dict[str, str] = {}
    for shard in set(index["weight_map"].values()):
        on_disk.update(dict.fromkeys(load_file(report.output_dir / shard), shard))
    assert on_disk == index["weight_map"]
    assert index["metadata"]["total_size"] == report.total_bytes


# --------------------------------------------------------------------------
# What the loader reads
# --------------------------------------------------------------------------


def test_config_carries_every_width_where_from_config_can_reach_it(exported):
    _, bits, report = exported
    config = json.loads((report.output_dir / HF_CONFIG_FILENAME).read_text())

    block = config["quantization_config"]
    assert block["quant_method"] == HF_QUANT_METHOD
    assert block["checkpoint_format"] == CHECKPOINT_FORMAT

    schema = QuantizationConfigSchema.from_dict(block)
    assert {name: spec.bits for name, spec in schema.modules.items()} == bits


def test_config_still_names_the_architecture(exported):
    """Without `architectures` no loader can pick a model class."""
    _, _, report = exported
    config = json.loads((report.output_dir / HF_CONFIG_FILENAME).read_text())
    assert config["architectures"] == ["LlamaForCausalLM"]
    assert config["model_type"] == "llama"


def test_fused_shards_resolve_to_their_separate_widths(exported):
    """The case the flat buffer exists for, read back the way vLLM will read it."""
    _, _, report = exported
    block = json.loads((report.output_dir / HF_CONFIG_FILENAME).read_text())["quantization_config"]
    schema = QuantizationConfigSchema.from_dict(block)

    shards = schema.resolve_shards(
        "model.layers.0.self_attn.qkv_proj",
        {"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
    )
    assert [spec.bits for _, spec in shards] == [4, 3, 3]


def test_each_quantized_module_becomes_a_triple(exported):
    from safetensors.torch import load_file

    _, bits, report = exported
    tensors = load_file(report.output_dir / HF_WEIGHTS_FILENAME)

    for name in bits:
        assert f"{name}.qweight" in tensors
        assert f"{name}.scales" in tensors
        assert f"{name}.offsets" in tensors
        assert f"{name}.weight" not in tensors
    assert tensors["model.layers.0.self_attn.q_proj.qweight"].dtype == torch.int32


def test_unquantized_tensors_pass_through_under_their_original_names(exported):
    from safetensors.torch import load_file

    model, _bits, report = exported
    tensors = load_file(report.output_dir / HF_WEIGHTS_FILENAME)

    # Norms are what a model's own load_weights routes without knowing anything
    # about DynQuant, so they must keep both their name and their values.
    assert "model.layers.0.input_layernorm.weight" in tensors
    assert torch.equal(
        tensors["model.layers.0.input_layernorm.weight"],
        model.model.layers[0].input_layernorm.weight.detach(),
    )
    assert "model.norm.weight" in tensors


# --------------------------------------------------------------------------
# Values
# --------------------------------------------------------------------------


def test_dequantized_weights_match_the_in_process_packed_runtime(exported):
    """The parity claim, checked without a GPU or a server.

    `export` and `pack_model` must call the same encoder with the same arguments.
    If they ever drift, a vLLM-vs-direct comparison starts measuring encoder
    disagreement instead of the thing it is meant to measure -- and it would look
    like a kernel bug.
    """
    from safetensors.torch import load_file

    from dynquant.quant.tensor import QuantTensor
    from dynquant.runtime import ops
    from dynquant.runtime.linear import pack_model

    _, bits, report = exported
    tensors = load_file(report.output_dir / HF_WEIGHTS_FILENAME)

    reference = tiny_model()
    pack_model(reference, bits, compute_device=None)

    for name in ("model.layers.1.mlp.down_proj", "model.layers.0.self_attn.q_proj"):
        in_process = reference.get_submodule(name).weight_qt
        from_disk = QuantTensor(
            packed=tensors[f"{name}.qweight"],
            scales=tensors[f"{name}.scales"],
            offsets=tensors[f"{name}.offsets"],
            bits=bits[name],
            group_size=in_process.group_size,
            in_features=in_process.in_features,
            logical_shape=in_process.logical_shape,
        )
        assert torch.equal(from_disk.packed, in_process.packed)
        assert torch.equal(from_disk.scales, in_process.scales)
        assert torch.equal(ops.dequantize(from_disk), ops.dequantize(in_process))


def test_accounting_is_measured_not_predicted(exported):
    from safetensors.torch import load_file

    _, _, report = exported
    tensors = load_file(report.output_dir / HF_WEIGHTS_FILENAME)

    packed = sum(
        t.numel() * t.element_size()
        for name, t in tensors.items()
        if name.rsplit(".", 1)[-1] in ("qweight", "scales", "offsets")
    )
    assert report.packed_bytes == packed
    assert 2.0 < report.average_bits < 8.0

    manifest = json.loads((report.output_dir / MANIFEST_FILENAME).read_text())
    assert manifest["accounting"]["packed_nbytes"] == packed
    assert set(manifest["layers"]) == set(report.layers)


def test_manifest_records_the_reconstruction_error_no_loader_keeps(exported):
    _, _, report = exported
    manifest = json.loads((report.output_dir / MANIFEST_FILENAME).read_text())
    layer = manifest["layers"]["model.layers.0.mlp.up_proj"]
    assert layer["bits"] == 3
    assert 0.0 <= layer["clipped_fraction"] <= 1.0
    assert layer["clip_improvement"] >= 0.0


# --------------------------------------------------------------------------
# Tied embeddings
# --------------------------------------------------------------------------


def test_a_tied_table_is_written_once(tmp_path):
    """27% of a tied 2B model, and safetensors refuses the duplicate anyway."""
    from safetensors.torch import load_file

    model = tiny_model(tie_word_embeddings=True)
    bits = {"model.embed_tokens": 4}
    report = export_packed_checkpoint(
        model, bits, output_dir=tmp_path / "ckpt", compute_device=None
    )

    tensors = load_file(report.output_dir / HF_WEIGHTS_FILENAME)
    assert "model.embed_tokens.qweight" in tensors
    assert "lm_head.weight" not in tensors
    assert "lm_head.qweight" not in tensors
    assert report.tied == {"lm_head": "model.embed_tokens"}


def test_an_untied_head_is_written_separately(tmp_path):
    from safetensors.torch import load_file

    model = tiny_model(tie_word_embeddings=False)
    report = export_packed_checkpoint(
        model,
        {"model.embed_tokens": 4, "lm_head": 8},
        output_dir=tmp_path / "ckpt",
        compute_device=None,
    )
    tensors = load_file(report.output_dir / HF_WEIGHTS_FILENAME)
    assert "model.embed_tokens.qweight" in tensors
    assert "lm_head.qweight" in tensors
    assert report.tied == {}


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_a_map_for_another_model_names_the_module_it_could_not_find(tmp_path):
    model = tiny_model()
    with pytest.raises(DynQuantError, match=re.escape("transformer.h.0.attn.c_attn")):
        export_packed_checkpoint(
            model,
            {"transformer.h.0.attn.c_attn": 4},
            output_dir=tmp_path / "ckpt",
            compute_device=None,
        )


def test_a_non_linear_module_is_refused_rather_than_skipped(tmp_path):
    model = tiny_model()
    with pytest.raises(DynQuantError, match="LlamaMLP"):
        export_packed_checkpoint(
            model,
            {"model.layers.0.mlp": 4},
            output_dir=tmp_path / "ckpt",
            compute_device=None,
        )


def test_skipped_modules_are_reported_not_dropped(exported):
    from safetensors.torch import load_file

    model, bits, report = exported
    assert set(report.skipped_modules) == {
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear | torch.nn.Embedding) and name not in bits
    }
    # The embedding is the one this map declines, and it stays on disk in its
    # original form rather than disappearing because nothing claimed it.
    assert "model.embed_tokens" in report.skipped_modules
    tensors = load_file(report.output_dir / HF_WEIGHTS_FILENAME)
    for name in report.skipped_modules:
        assert f"{name}.weight" in tensors
