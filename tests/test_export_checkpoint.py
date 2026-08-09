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

from dynquant.quant.checkpoint import (  # noqa: E402
    _resolve,
    export_packed_checkpoint,
)


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


def test_a_module_with_no_weight_is_refused_rather_than_skipped(tmp_path):
    # A container, not a leaf: `LlamaMLP` owns three Linears and no tensor of its
    # own. The refusal is about there being nothing to pack, which is the packer's
    # own criterion -- not about the class not appearing on a whitelist.
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


# --------------------------------------------------------------------------
# Modules that are not Linear but own a weight
# --------------------------------------------------------------------------

ROUTER = "model.layers.0.mlp.gate"


class BareWeightRouter(torch.nn.Module):
    """A router shaped the way transformers ships one.

    `Lfm2MoeTopKRouter`, Qwen3-Next's router and GPT-OSS's all hold
    `[num_experts, hidden]` as their own parameter and call `F.linear` on it rather
    than owning a `Linear`. So `get_submodule` reaches the module, the module owns a
    weight, and the module is not a `Linear` -- the one combination the export path's
    whitelist refused while `quantize_model` encoded it without complaint.
    """

    def __init__(self, num_experts: int, hidden: int) -> None:
        super().__init__()
        torch.manual_seed(2)
        self.weight = torch.nn.Parameter(torch.randn(num_experts, hidden, dtype=torch.float16))


@pytest.fixture
def exported_router(tmp_path):
    model = tiny_model()
    model.model.layers[0].mlp.gate = BareWeightRouter(8, 128)
    report = export_packed_checkpoint(
        model, {ROUTER: 8}, output_dir=tmp_path / "ckpt", compute_device=None
    )
    return model, report


def test_a_router_that_is_not_a_linear_packs_at_the_width_it_was_given(exported_router):
    from safetensors.torch import load_file

    _, report = exported_router
    tensors = load_file(report.output_dir / HF_WEIGHTS_FILENAME)
    assert tensors[f"{ROUTER}.qweight"].shape == (8, 32)
    assert tensors[f"{ROUTER}.scales"].shape == (8, 1)
    meta = report.layers[ROUTER]
    assert meta["bits"] == 8
    # 8 bits of payload plus one fp16 scale and one fp16 offset over a 128-wide group.
    assert meta["bits_per_weight"] == pytest.approx(8.25)


def test_a_packed_router_is_not_also_written_dense(exported_router):
    from safetensors.torch import load_file

    model, report = exported_router
    tensors = load_file(report.output_dir / HF_WEIGHTS_FILENAME)
    assert f"{ROUTER}.weight" not in tensors
    fp16 = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    assert report.total_bytes < fp16


def test_a_router_is_not_reported_as_an_expert_bank(exported_router):
    # It resolves through the module branch, so its key is `f"{name}.weight"` and the
    # not-yet-loadable caveat -- which is about bare parameters -- must not attach to it.
    _, report = exported_router
    assert report.banks == ()
    assert "expert bank" not in report.summary()


def test_a_map_naming_something_no_loader_could_read_is_refused_before_any_bytes(
    tmp_path,
) -> None:
    """The writer asks the reader's question first, and asks it before ``mkdir``.

    The packer's rule is "a tensor exists to pack" and the runtime's is "something can
    stand where this stands". Those were allowed to differ, and the difference shipped:
    the LFM2.5-8B-A1B export wrote 22 routers over 23 minutes and ``from_pretrained``
    refused the 4.4 GB artifact it had just produced. A router is now *restored* rather
    than refused, so the case that broke is fixed -- but the two rules can still drift,
    and this is what stops the drift from reaching disk.

    Two names that genuinely have no reader: a bare 2-D parameter, whose owner passes it
    whole to ``F.linear`` so no module can stand at its index, and a ``Conv1d`` kernel,
    which is rank 3 like a bank and indexed by nothing -- on the campaign's own model 18
    of 24 layers are conv, so that is the common case rather than an exotic one.

    Turns red when: the pre-flight is dropped, or moved after ``out.mkdir`` -- a refusal
    that leaves a directory behind is a refusal someone will mistake for a checkpoint.
    """
    model = tiny_model()
    model.model.layers[0].mlp.gate = BareWeightRouter(8, 128)
    model.model.layers[0].mlp.conv = torch.nn.Conv1d(64, 64, 3, groups=64, bias=False)

    for name, reason in (
        (f"{ROUTER}.weight", "bare 2-D parameter"),
        ("model.layers.0.mlp.conv.weight", "3-D weight of a Conv1d"),
    ):
        out = tmp_path / name
        with pytest.raises(DynQuantError, match=reason):
            export_packed_checkpoint(model, {name: 8}, output_dir=out, compute_device=None)
        assert not out.exists(), out

    # And the width the encoder *can* place is still written: the guard rejects the two
    # shapes above and nothing else, or every published checkpoint stops being writable.
    kept = tmp_path / "kept"
    report = export_packed_checkpoint(model, {ROUTER: 8}, output_dir=kept, compute_device=None)
    assert set(report.layers) == {ROUTER}


def test_the_export_resolver_answers_what_the_quantizer_answers(exported_router):
    # The property the whitelist broke, stated directly: pre-flight said yes, the
    # quantizer said yes, and export said no. One resolver, so one answer.
    from dynquant.quant.quantizer import resolves_to_weight, target_tensor

    model, report = exported_router
    for name in (ROUTER, "model.layers.0.self_attn.q_proj", "model.embed_tokens"):
        assert resolves_to_weight(model, name)
        assert target_tensor(model, name) is _resolve(model, name)[0]
    assert set(report.layers) == {ROUTER}


# --------------------------------------------------------------------------
# Batched expert banks
# --------------------------------------------------------------------------


BANK = "model.layers.0.mlp.experts.gate_up_proj"


def tiny_model_with_a_bank():
    """A Llama with an MoE-style expert bank grafted onto its first layer.

    Batched-MoE checkpoints keep every expert's weight in one 3-D ``nn.Parameter``
    -- ``[num_experts, out, in]`` -- and have no ``nn.Linear`` for them anywhere.
    Grafting one on reproduces the property that actually matters, a quantization
    target ``get_submodule`` cannot reach, without downloading an 8B model, and
    leaves the rest of the fixture identical to every other test in this file.
    """
    model = tiny_model()
    experts = torch.nn.Module()
    torch.manual_seed(1)
    experts.gate_up_proj = torch.nn.Parameter(torch.randn(8, 256, 128, dtype=torch.float16))
    model.model.layers[0].mlp.experts = experts
    return model


@pytest.fixture
def exported_bank(tmp_path):
    model = tiny_model_with_a_bank()
    report = export_packed_checkpoint(
        model, {BANK: 4}, output_dir=tmp_path / "ckpt", compute_device=None
    )
    return model, report


def test_a_batched_expert_bank_packs_at_the_width_it_was_given(exported_bank):
    """Banks are 91.5% of an LFM2.5-8B-A1B; refusing them refuses the model.

    This path used to raise, saying the packed *format* could not represent a
    bank. The format could all along -- ``logical_shape`` exists for exactly this
    -- and the refusal came from a second copy of the name resolver that only knew
    ``get_submodule``. What future diff turns this red: reintroducing a
    module-only lookup in the export path, or dropping ``logical_shape`` from the
    flattening.
    """
    from safetensors.torch import load_file

    _model, report = exported_bank
    tensors = load_file(report.output_dir / HF_WEIGHTS_FILENAME)

    assert set(report.banks) == {BANK}
    # [8, 256, 128] flattens to 2048 rows of 128 values; 4 bits each is 16 words.
    assert tuple(tensors[f"{BANK}.qweight"].shape) == (2048, 16)
    assert tuple(tensors[f"{BANK}.scales"].shape) == (2048, 1)

    layer = report.layers[BANK]
    assert layer["logical_shape"] == [8, 256, 128]
    assert layer["num_rows"] == 2048
    # 4 bits per weight plus an fp16 scale and offset for each 128-value group.
    assert layer["nbytes"] == 139_264
    assert report.average_bits == pytest.approx(4.25)


def test_a_packed_bank_is_not_also_written_dense(exported_bank):
    """What the state-dict key returned beside the tensor is for.

    A bank belongs to no tie-alias group -- ``_tied_aliases`` walks modules and a
    bank is not one -- so unless the loop marks the bank's own key consumed, the
    dense pass writes all 262 144 values again at fp16 and the "quantized"
    directory comes out *larger* than the checkpoint it compressed.
    """
    from safetensors.torch import load_file

    model, report = exported_bank
    tensors = load_file(report.output_dir / HF_WEIGHTS_FILENAME)

    assert BANK not in tensors
    assert f"{BANK}.weight" not in tensors
    fp16 = sum(t.numel() * t.element_size() for t in model.state_dict().values())
    assert report.total_bytes < fp16


def test_a_packed_bank_dequantizes_back_to_rank_three(exported_bank):
    """``logical_shape`` is what makes the flattening reversible.

    Without it a bank reads back as a 2048x128 matrix and every consumer that
    slices per expert is silently off, which is worse than the refusal was.
    """
    from safetensors.torch import load_file

    from dynquant.quant.tensor import QuantTensor
    from dynquant.runtime import ops

    model, report = exported_bank
    tensors = load_file(report.output_dir / HF_WEIGHTS_FILENAME)
    meta = report.layers[BANK]

    restored = ops.dequantize(
        QuantTensor(
            packed=tensors[f"{BANK}.qweight"],
            scales=tensors[f"{BANK}.scales"],
            offsets=tensors[f"{BANK}.offsets"],
            bits=4,
            group_size=meta["group_size"],
            in_features=meta["in_features"],
            logical_shape=tuple(meta["logical_shape"]),
        )
    )
    original = model.get_parameter(BANK).detach()
    assert restored.shape == original.shape
    error = torch.linalg.vector_norm(restored.float() - original.float())
    assert error / torch.linalg.vector_norm(original.float()) < 0.1


def test_the_bank_caveat_travels_with_the_directory(exported_bank):
    """Size-honest and loadable are two facts, and the folder outlives the report.

    Whoever uploads this to a Hub next week has the directory and not the
    ``ExportReport``, so the manifest has to carry the caveat too.
    """
    _model, report = exported_bank
    summary = report.summary()
    assert "batched expert banks" in summary
    assert "cannot load them back" in summary

    manifest = json.loads((report.output_dir / MANIFEST_FILENAME).read_text())
    assert manifest["expert_banks"] == [BANK]


def test_a_directory_without_banks_makes_no_claim_about_them(exported):
    """The caveat is conditional: a dense model's report must not carry it."""
    _model, _bits, report = exported
    assert report.banks == ()
    assert "expert bank" not in report.summary()
    manifest = json.loads((report.output_dir / MANIFEST_FILENAME).read_text())
    assert manifest["expert_banks"] == []


def test_what_this_writer_wrote_is_now_what_the_runtime_holds(exported_bank):
    """The export half and the runtime half have to agree, tensor for tensor.

    This once asserted that ``pack_model`` refused a bank -- true when the writer
    landed and the reader had not, and the reason the ``ExportReport`` carries a
    caveat at all. Now both exist, so the property worth pinning is that they meet:
    the keys the exporter writes are the buffer names the packed module registers,
    and the values decode to the same numbers.

    Turns red when: either half changes its geometry or its key names without the
    other, which would produce a directory that loads and is wrong rather than one
    that refuses.
    """
    from safetensors.torch import load_file

    from dynquant.runtime.linear import DynQuantExpertBank, pack_model

    _model, report = exported_bank
    exported = load_file(report.output_dir / HF_WEIGHTS_FILENAME)
    written = {key.rpartition(".")[2] for key in exported if key.startswith(f"{BANK}.")}

    model = tiny_model_with_a_bank()
    pack_model(model, {BANK: 4}, compute_device=None)
    bank = model.get_submodule(BANK)
    assert isinstance(bank, DynQuantExpertBank)
    held = bank.state_dict()
    assert set(held) == written

    for suffix in sorted(written):
        assert torch.equal(exported[f"{BANK}.{suffix}"], held[suffix]), suffix
