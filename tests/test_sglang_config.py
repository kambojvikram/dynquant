"""``DynQuantConfig`` on SGLang: the four places the fork differs from vLLM.

The class is a port, and a port's tests should be about the deltas rather than
re-proving the shared parts -- those are covered once, in
``test_serving_schema.py``, because the schema is genuinely one contract for both
frameworks. What is *not* shared:

1. ``packed_modules_mapping`` arrives inside the config dict and nothing lifts it out.
2. ``get_scaled_act_names`` is still abstract here, so omitting it makes the class
   unbuildable -- and only at instantiation, in a spawned subprocess.
3. ``UnquantizedLinearMethod`` lives in ``quantization/unquant.py``.
4. ``override_quantization_method`` is polled for every checkpoint, so answering
   anything but ``None`` hijacks other people's models.

Runs against :mod:`_sglang_stub`; see that module for why not a real SGLang, and
:mod:`test_sglang_stub_conformance` for what keeps the stub honest.
"""

from __future__ import annotations

import pytest

from dynquant.constants import DEFAULT_GROUP_SIZE, HF_QUANT_METHOD
from dynquant.errors import DynQuantError

from _sglang_stub import fake_sglang

LLAMA_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}

BLOCK = {
    "quant_method": HF_QUANT_METHOD,
    "group_size": DEFAULT_GROUP_SIZE,
    "modules": {
        "model.layers.0.self_attn.q_proj": {"bits": 4},
        "model.layers.0.self_attn.k_proj": {"bits": 3},
        "model.layers.0.self_attn.v_proj": {"bits": 3},
        "model.layers.0.mlp.gate_proj": {"bits": 4},
        "model.layers.0.mlp.up_proj": {"bits": 2},
    },
}


@pytest.fixture
def sglang():
    """The stub tree, yielding the plugin's config module built against it."""
    with fake_sglang():
        import dynquant.integration.sglang_plugin.config as config

        yield config


# --------------------------------------------------------------------------
# 1. The mapping SGLang injects but never installs
# --------------------------------------------------------------------------


def test_from_config_lifts_the_injected_mapping(sglang):
    """``weight_utils.py:278`` puts it in the dict; the base class never reads it.

    ``QuantizationConfig.__init__`` sets ``packed_modules_mapping`` to ``{}`` and the
    only caller of ``update_packed_modules_mapping`` in SGLang's tree is one model
    file. GPTQ and AWQ do not notice because neither needs fusion structure to build
    a layer; DynQuant does, since a fused ``qkv_proj`` is three modules at three
    widths and ``resolve_shards`` is where that is worked out.
    """
    config = sglang.DynQuantConfig.from_config({**BLOCK, "packed_modules_mapping": LLAMA_MAPPING})
    assert config.packed_modules_mapping == LLAMA_MAPPING


def test_the_lifted_mapping_actually_resolves_a_fused_layer(sglang):
    """The consequence, not just the attribute: without it, shards come back empty.

    This is the assertion that would have caught shipping the vLLM ``from_config``
    unchanged. The attribute test above passes for a config that lifted the mapping
    into the wrong name; this one does not.
    """
    config = sglang.DynQuantConfig.from_config({**BLOCK, "packed_modules_mapping": LLAMA_MAPPING})
    shards = config.schema.resolve_shards(
        "model.layers.0.self_attn.qkv_proj", config.packed_modules_mapping
    )
    assert shards is not None
    assert [name.rsplit(".", 1)[-1] for name, _ in shards] == ["q_proj", "k_proj", "v_proj"]
    assert [spec.bits for _, spec in shards] == [4, 3, 3]

    without = sglang.DynQuantConfig.from_config(BLOCK)
    assert (
        without.schema.resolve_shards(
            "model.layers.0.self_attn.qkv_proj", without.packed_modules_mapping
        )
        is None
    )


@pytest.mark.parametrize("injected", [{}, None])
def test_an_absent_mapping_leaves_the_base_class_default(sglang, injected):
    """SGLang injects unconditionally, but a model with no fusion injects ``{}``,
    and ``.get`` on a key written as JSON ``null`` returns ``None``. Neither may
    replace the dict with something a later ``update_packed_modules_mapping`` or a
    membership test would choke on."""
    config = sglang.DynQuantConfig.from_config({**BLOCK, "packed_modules_mapping": injected})
    assert config.packed_modules_mapping == {}


def test_the_injected_key_does_not_reach_the_schema(sglang):
    """``from_dict`` tolerates unknown keys; that is what makes the injection safe.

    Worth pinning rather than assuming: if the schema ever grew strict validation,
    SGLang's own mutation of the dict would start rejecting every checkpoint, and
    the traceback would point at the schema rather than at the caller that added a
    key to somebody else's config.
    """
    config = sglang.DynQuantConfig.from_config({**BLOCK, "packed_modules_mapping": LLAMA_MAPPING})
    assert "packed_modules_mapping" not in config.schema.to_dict()


# --------------------------------------------------------------------------
# 1b. ...and the model classes that never inject one
# --------------------------------------------------------------------------


def test_a_fused_layer_resolves_even_when_the_model_class_declares_nothing(sglang):
    """Lifting the injected mapping is necessary and not sufficient.

    What SGLang injects is ``getattr(model_class, "packed_modules_mapping", {})``
    (``model_loader/loader.py:204``), and on 0.5.16 that attribute is absent from 172
    of the 210 files in ``srt/models/``. ``Qwen2ForCausalLM`` is one of them: it fuses
    q/k/v inside ``load_weights`` and declares nothing, so the injected dict is empty
    for one of the most-served architectures there is.

    The test above pins that an empty mapping resolves to nothing at the *schema*
    layer, which is still true and still correct. This one pins that the plugin does
    not stop there.
    """
    config = sglang.DynQuantConfig.from_config(BLOCK)
    assert config.packed_modules_mapping == {}

    shards = config._shards("model.layers.0.self_attn.qkv_proj")
    assert [name.rsplit(".", 1)[-1] for name, _ in shards] == ["q_proj", "k_proj", "v_proj"]
    assert [spec.bits for _, spec in shards] == [4, 3, 3]


def test_the_fallback_reaches_get_quant_method_on_both_branches(sglang):
    """The fallback is only worth anything at the point layers are built.

    ``get_quant_method`` asks twice, once per layer kind, and a fallback wired into
    one branch and not the other is a bug that no schema-level test can see.
    """
    config = sglang.DynQuantConfig.from_config(BLOCK)
    from sglang.srt.layers.linear import QKVParallelLinear

    method = config.get_quant_method(QKVParallelLinear(), "model.layers.0.self_attn.qkv_proj")
    assert type(method).__name__ == "DynQuantLinearMethod"


def test_a_fused_layer_with_no_shards_is_refused_rather_than_served_dense(sglang):
    """The failure this whole path exists to stop, asserted as a raise.

    Serving a fused layer unquantized against a packed checkpoint is not a
    degradation, it is uninitialised memory: SGLang's loader rewrites
    ``q_proj.qweight`` to ``qkv_proj.qweight``, finds no such parameter on the
    unquantized layer, and drops it with ``logger.warning``
    (``models/qwen2.py:639``). Nothing raises and the server answers requests. It did
    exactly that on the first real serve of this integration.

    A checkpoint whose fused shards are named something neither SGLang nor
    :data:`CONVENTIONAL_FUSED_MODULES` knows -- ``W_pack`` here -- is the reachable
    version of that state.
    """
    exotic = {
        **BLOCK,
        "modules": {
            "model.layers.0.self_attn.W_pack": {"bits": 4},
            "model.layers.0.self_attn.o_proj": {"bits": 4},
        },
    }
    config = sglang.DynQuantConfig.from_config(exotic)
    from sglang.srt.layers.linear import QKVParallelLinear

    with pytest.raises(DynQuantError, match="QKVParallelLinear"):
        config.get_quant_method(QKVParallelLinear(), "model.layers.0.self_attn.qkv_proj")


def test_an_unfused_layer_left_dense_is_not_refused(sglang):
    """The false positive the class test exists to avoid.

    ``o_proj`` beside a quantized ``q_proj`` resolves to nothing for a perfectly
    good reason -- the exporter left it dense -- and its fp16 weight is on disk under
    its own name for the loader to find. Only a *fused* layer has no such fallback,
    which is why the guard tests the layer class and not the emptiness.
    """
    config = sglang.DynQuantConfig.from_config(BLOCK)
    from sglang.srt.layers.linear import LinearBase

    method = config.get_quant_method(LinearBase(), "model.layers.0.self_attn.o_proj")
    assert type(method).__name__ == "UnquantizedLinearMethod"


def test_a_fused_layer_in_an_untouched_region_is_not_refused(sglang):
    """The other false positive: a dense vision tower beside a quantized LM.

    Its ``qkv_proj`` resolves to nothing and should, and its dense weights load
    normally. The guard's second half -- are any siblings quantized -- is what tells
    this apart from a checkpoint whose shard names did not match.
    """
    config = sglang.DynQuantConfig.from_config(BLOCK)
    from sglang.srt.layers.linear import QKVParallelLinear

    method = config.get_quant_method(QKVParallelLinear(), "visual.blocks.0.attn.qkv_proj")
    assert type(method).__name__ == "UnquantizedLinearMethod"


# --------------------------------------------------------------------------
# 2. The abstract method vLLM deleted and SGLang kept
# --------------------------------------------------------------------------


def test_the_config_is_concrete(sglang):
    """Instantiation is the only thing that reveals a missed abstract method.

    Registration puts the *class* in a dict and succeeds either way; the
    ``TypeError: Can't instantiate abstract class`` arrives later, inside
    ``run_scheduler_process``, where it reads as a worker that died during startup.
    Asserted by building one, so adding an abstract method upstream turns this red
    rather than turning a serve red.
    """
    config = sglang.DynQuantConfig.from_config(BLOCK)
    assert config.get_scaled_act_names() == []
    assert config.get_name() == HF_QUANT_METHOD
    assert config.get_config_filenames() == []
    assert config.get_min_capability() == 75


def test_repr_says_what_the_checkpoint_is(sglang):
    """It goes in the startup log, where it is the only place a reader learns that
    this checkpoint has more than one width."""
    text = repr(sglang.DynQuantConfig.from_config(BLOCK))
    assert "modules=5" in text
    assert "[2, 3, 4]" in text


# --------------------------------------------------------------------------
# 3. `override_quantization_method` is asked about everyone else's checkpoints
# --------------------------------------------------------------------------

FOREIGN = [
    {"quant_method": "gptq", "bits": 4, "group_size": 128, "desc_act": False},
    {"quant_method": "awq", "bits": 4, "group_size": 128, "zero_point": True},
    {"quant_method": "bitsandbytes", "load_in_4bit": True},
    {"quant_method": "compressed-tensors", "config_groups": {}},
    {"quant_algo": "FP8", "quant_method": "modelopt_fp8"},
]


@pytest.mark.parametrize("hf_quant_cfg", FOREIGN)
@pytest.mark.parametrize("user_quant", [None, "gptq", "awq", HF_QUANT_METHOD])
def test_we_never_claim_a_checkpoint_that_is_not_ours(sglang, hf_quant_cfg, user_quant):
    """``ModelConfig._verify_quantization`` polls *every* registered config.

    It loops over the whole registry for every checkpoint and takes the first truthy
    answer, then breaks (``configs/model_config.py:1409``). So a config that
    volunteered a name for a foreign checkpoint would silently divert somebody's GPTQ
    model into DynQuant's loader -- and whether it got the chance would depend on
    dict iteration order, which is decided by the order plugins happened to load.

    The right answer is the base class's ``None``, always. This test exists because
    the tempting shape -- "detect our format here" -- is wrong, and nothing else in
    the plugin would notice.
    """
    assert sglang.DynQuantConfig.override_quantization_method(hf_quant_cfg, user_quant) is None


def test_we_do_not_claim_our_own_checkpoint_either(sglang):
    """Even for a DynQuant block: ``quant_method`` in ``config.json`` already
    resolves it through ``QUANTIZATION_METHODS``. Overriding would be a second,
    order-dependent path to the same answer, and the two could disagree."""
    assert sglang.DynQuantConfig.override_quantization_method(BLOCK, None) is None
    assert sglang.DynQuantConfig.override_quantization_method(None, None) is None


# --------------------------------------------------------------------------
# 4. MoE fails closed, by name
# --------------------------------------------------------------------------


def test_a_moe_layer_is_refused_with_a_reason(sglang):
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    config = sglang.DynQuantConfig.from_config(BLOCK)
    with pytest.raises(DynQuantError) as excinfo:
        config.get_quant_method(FusedMoE(), "model.layers.0.mlp.experts")

    message = str(excinfo.value)
    assert "model.layers.0.mlp.experts" in message
    assert "phase 8" in message


def test_a_model_files_moe_subclass_is_refused_too(sglang):
    """``DeepEPMoE(FusedMoE)`` and every model file's own subclass. The probe walks
    ``__mro__``, so a check on the type itself would let these through."""
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    class SomeModelsExperts(FusedMoE):
        pass

    assert sglang._class_is_fused_moe(SomeModelsExperts)


def test_linear_and_embedding_layers_are_not_caught(sglang):
    """The refusal has to be narrow, or it takes down every dense model."""
    import torch
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

    for cls in (LinearBase, ParallelLMHead, torch.nn.Linear):
        assert not sglang._class_is_fused_moe(cls), cls.__name__
