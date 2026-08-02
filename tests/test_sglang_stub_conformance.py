"""Does :mod:`_sglang_stub` still describe the SGLang that is installed?

``test_sglang_register.py`` and ``test_sglang_config.py`` run against a stub, because
SGLang ships manylinux-only wheels from 0.5.11 and would otherwise skip everywhere
except one Linux box. That trade buys coverage on every machine and takes on the
standard risk of a fake: it keeps passing after the thing it imitates has moved.

This file is the other half. It skips where SGLang is absent -- so on a laptop the
suite is honest about having checked the plugin and not the framework -- and where
SGLang is present it compares the two, symbol by symbol. A failure here does not mean
the plugin is broken; it means the stub's account of SGLang is out of date, and the
tests written against it have been measuring the wrong thing since.

Assertions are directional where direction matters. For an abstract-method set the
requirement is *real ⊆ stub*: a stub missing one of SGLang's abstract methods accepts
a config SGLang would refuse to instantiate, which is the exact failure the stub was
built to catch.
"""

from __future__ import annotations

import inspect

import pytest

import _sglang_stub as stub

sglang = pytest.importorskip("sglang", reason="the SGLang plugin needs SGLang to check against")

from dynquant.constants import HF_QUANT_METHOD  # noqa: E402


def test_every_module_the_stub_fakes_exists():
    """The coarsest drift, and the one the stub is least able to notice."""
    from importlib import import_module

    missing = []
    for name in stub.MODULE_NAMES:
        try:
            import_module(name)
        except ImportError:
            missing.append(name)
    assert not missing, f"the stub fakes modules SGLang no longer has: {missing}"


def test_the_config_base_class_has_not_grown_an_abstract_method():
    """The one that fails late and reads as an unrelated crash.

    An abstract method added upstream does not break registration -- that stores the
    class. It breaks the first ``from_config``, as ``TypeError: Can't instantiate
    abstract class DynQuantConfig``, raised inside ``run_scheduler_process`` where it
    reaches the user as a worker that died during startup.
    """
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

    real = set(QuantizationConfig.__abstractmethods__)
    faked = set(stub.QuantizationConfig.__abstractmethods__)
    assert real <= faked, (
        f"SGLang's QuantizationConfig now requires {sorted(real - faked)}, which the "
        f"stub does not -- add it to _sglang_stub and implement it in the plugin"
    )
    assert not faked - real, (
        f"the stub requires {sorted(faked - real)} that SGLang no longer does; "
        f"harmless at runtime, but the plugin is carrying a method for nobody"
    )


def test_get_scaled_act_names_is_still_required():
    """Named separately because it is the delta from vLLM, which deleted it.

    If SGLang drops it too, the plugin's implementation becomes dead weight and its
    docstring becomes wrong -- and the generic test above says only "some set
    changed", which is not the same sentence.
    """
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

    assert "get_scaled_act_names" in QuantizationConfig.__abstractmethods__


def test_the_base_class_still_leaves_packed_modules_mapping_empty():
    """The premise of ``from_config`` lifting it out of the dict by hand.

    Should SGLang start populating this itself -- by calling
    ``update_packed_modules_mapping`` from the loader, say -- the lift becomes
    redundant rather than wrong, but the docstring explaining why it exists becomes
    false, and false comments outlive the code they describe.
    """
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

    signature = inspect.signature(QuantizationConfig.__init__)
    assert list(signature.parameters) == ["self"]
    assert hasattr(QuantizationConfig, "update_packed_modules_mapping")


def test_override_quantization_method_still_defaults_to_none():
    """We inherit it. If the base ever returned something truthy, every DynQuant
    config would start claiming checkpoints during ``_verify_quantization``."""
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

    assert QuantizationConfig.override_quantization_method({"quant_method": "gptq"}, None) is None


def test_the_mapper_hook_is_still_spelled_sglangs_way():
    """``apply_weight_name_mapper``, not vLLM's ``apply_vllm_mapper``.

    An override under the wrong name is not an error -- it is a method nobody calls,
    and the schema silently keeps HF-side module names while the layers ask for
    SGLang-side ones. Every fused layer then resolves to no shards.
    """
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

    assert hasattr(QuantizationConfig, "apply_weight_name_mapper")
    assert not hasattr(QuantizationConfig, "apply_vllm_mapper")

    parameters = list(inspect.signature(QuantizationConfig.apply_weight_name_mapper).parameters)
    assert parameters == ["self", "hf_to_sglang_mapper"]


def test_the_unquantized_methods_are_where_the_plugin_imports_them_from():
    """``quantization/unquant.py``, which is where SGLang moved them; vLLM keeps
    them in ``linear.py`` and ``vocab_parallel_embedding.py``.

    The two are siblings here. vLLM's ``UnquantizedEmbeddingMethod`` derives from
    ``UnquantizedLinearMethod``; SGLang's derives straight from
    ``QuantizeMethodBase`` and so is not a ``LinearMethodBase`` at all. Asserted
    rather than assumed because ``get_quant_method`` returns one of these two by
    layer kind, and on the vLLM hierarchy returning the embedding method where a
    linear method was wanted is still an ``isinstance(..., LinearMethodBase)`` --
    a confusion that types out correctly there and not here.
    """
    from sglang.srt.layers.quantization.base_config import (
        LinearMethodBase,
        QuantizeMethodBase,
    )
    from sglang.srt.layers.quantization.unquant import (
        UnquantizedEmbeddingMethod,
        UnquantizedLinearMethod,
    )

    assert issubclass(UnquantizedLinearMethod, LinearMethodBase)
    assert issubclass(UnquantizedEmbeddingMethod, QuantizeMethodBase)
    assert not issubclass(UnquantizedEmbeddingMethod, LinearMethodBase)
    # `method_has_implemented_embedding` tests for this by identity against the base,
    # and `VocabParallelEmbedding.__init__` refuses a method that lacks it.
    assert "embedding" in vars(UnquantizedEmbeddingMethod)


def test_the_layer_classes_the_plugin_dispatches_on_still_exist():
    from sglang.srt.layers.linear import LinearBase
    from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding

    assert issubclass(ParallelLMHead, VocabParallelEmbedding)
    assert not issubclass(ParallelLMHead, LinearBase), (
        "ParallelLMHead has become a LinearBase; get_quant_method tests ParallelLMHead "
        "first, so this ordering is now load-bearing rather than incidental"
    )


def test_the_fused_linear_classes_are_still_subclasses_of_the_one_we_dispatch_on():
    """``get_quant_method`` tests ``LinearBase`` first, then narrows to these two.

    If either stopped deriving from ``LinearBase`` it would never reach the narrowing
    at all, and the fused-layer guard would go quiet -- returning to exactly the
    silent-uninitialised-weights behaviour it was added to stop.
    """
    from sglang.srt.layers.linear import (
        LinearBase,
        MergedColumnParallelLinear,
        QKVParallelLinear,
    )

    assert issubclass(QKVParallelLinear, LinearBase)
    assert issubclass(MergedColumnParallelLinear, LinearBase)


def test_qwen2_still_fuses_qkv_without_declaring_how():
    """The premise of :data:`CONVENTIONAL_FUSED_MODULES`, checked against SGLang.

    Two facts, and the fallback is only justified while both hold: the model fuses
    q/k/v into one layer, and it tells a quantization config nothing about it. If
    SGLang ever adds the declaration this test turns red, and the right response is
    to delete the fallback for that model rather than to keep guessing beside a
    source of truth.

    ``Qwen2ForCausalLM`` specifically, because it is what the parity checkpoint is
    and where this was found -- but it is not a special case: 172 of the 210 files in
    ``srt/models/`` declare no ``packed_modules_mapping``.
    """
    import inspect

    from sglang.srt.models.qwen2 import Qwen2ForCausalLM

    assert getattr(Qwen2ForCausalLM, "packed_modules_mapping", {}) == {}

    source = inspect.getsource(Qwen2ForCausalLM.load_weights)
    for shard in ("q_proj", "k_proj", "v_proj"):
        assert f'"qkv_proj", "{shard}"' in source, (
            "Qwen2 no longer fuses qkv in load_weights; the fallback expansion "
            "in CONVENTIONAL_FUSED_MODULES describes a fusion that stopped happening"
        )


def test_a_failed_stacked_mapping_lookup_keeps_the_rewritten_name():
    """Why the symptom was ``qkqkv_proj`` and not a clean "qkv_proj not found".

    The loop rebinds ``name`` in place and, when the lookup misses, ``continue``\\ s
    to the next entry carrying the rewrite with it -- so ``v_proj`` matches inside
    the ``qkv_proj`` it just produced and ``.replace`` fires a second time. Pinned
    because that mangled name is the string a user will search for, and it points at
    a substring bug that is not the actual fault: the fault is that the parameter was
    missing, one layer up.
    """
    import inspect

    from sglang.srt.models.qwen2 import Qwen2ForCausalLM

    source = inspect.getsource(Qwen2ForCausalLM.load_weights)
    assert "name = name.replace(weight_name, param_name)" in source
    assert "if name not in params_dict:\n                    continue" in source


def test_the_moe_package_is_where_the_guard_looks():
    """``_MOE_PACKAGE`` is matched against ``__module__``, so a move is invisible.

    That is not hypothetical -- the vLLM twin of this guard stopped matching when
    ``FusedMoE`` became a factory, silently, and the symptom was fp16 experts built
    for a packed checkpoint rather than an error.
    """
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    from dynquant.integration.sglang_plugin.config import _MOE_PACKAGE, _class_is_fused_moe

    assert FusedMoE.__module__.startswith(_MOE_PACKAGE)
    assert _class_is_fused_moe(FusedMoE)


def test_the_registry_split_is_still_a_copy():
    """We write to ``QUANTIZATION_METHODS``; ``BASE_`` is copied at import time."""
    import sglang.srt.layers.quantization as quantization

    assert isinstance(quantization.QUANTIZATION_METHODS, dict)
    assert quantization.QUANTIZATION_METHODS is not quantization.BASE_QUANTIZATION_METHODS


def test_the_choices_setter_still_does_not_deduplicate():
    """Our idempotence guard exists because SGLang's setter is a bare ``extend``.

    Asserted by using it, and restoring the list afterwards -- a signature check
    would pass for an implementation that had started deduplicating, which would
    make our guard redundant rather than necessary.
    """
    import sglang.srt.server_args as server_args

    original = list(server_args.QUANTIZATION_CHOICES)
    try:
        server_args.add_quantization_method_choices(["__dynquant_probe__"])
        server_args.add_quantization_method_choices(["__dynquant_probe__"])
        assert server_args.QUANTIZATION_CHOICES.count("__dynquant_probe__") == 2
    finally:
        server_args.QUANTIZATION_CHOICES[:] = original


def test_the_weight_loader_v2_list_is_still_class_names_as_strings():
    """A list of ``str``, compared against ``type(method).__name__``.

    If it ever became a set of classes, appending our *name* would still succeed
    and still never match, so registration would go on looking correct while every
    DynQuant layer silently fell back to the v1 loader -- which cannot place a flat
    buffer at all, and says so with a bare ``AssertionError`` from inside a spawned
    scheduler process. This is the selection mechanism itself, checked against a
    real SGLang; ``test_sglang_linear.py`` checks what happens downstream of it.
    """
    import sglang.srt.layers.linear as linear

    assert isinstance(linear.WEIGHT_LOADER_V2_SUPPORTED, list)
    assert all(isinstance(entry, str) for entry in linear.WEIGHT_LOADER_V2_SUPPORTED)
    assert "GPTQMarlinLinearMethod" in linear.WEIGHT_LOADER_V2_SUPPORTED


def test_registration_works_against_the_real_thing():
    """The whole point, in one test: everything above is why this one is meaningful
    when it fails on somebody else's machine."""
    import sglang.srt.layers.linear as linear
    import sglang.srt.layers.quantization as quantization
    import sglang.srt.server_args as server_args

    from dynquant.integration.sglang_plugin import register
    from dynquant.integration.sglang_plugin.config import DynQuantConfig

    register()

    assert quantization.get_quantization_config(HF_QUANT_METHOD) is DynQuantConfig
    assert server_args.QUANTIZATION_CHOICES.count(HF_QUANT_METHOD) == 1
    assert linear.WEIGHT_LOADER_V2_SUPPORTED.count("DynQuantLinearMethod") == 1


def test_the_entry_point_names_something_importable():
    """SGLang finds ``register`` through ``sglang.srt.plugins``, not through an
    import anybody writes. A typo in ``pyproject.toml`` is therefore invisible until
    a server starts and silently does not know what ``dynquant`` means."""
    from importlib.metadata import entry_points

    found = [
        entry
        for entry in entry_points(group="sglang.srt.plugins")
        if entry.value.startswith("dynquant.")
    ]
    if not found:
        pytest.skip("dynquant is not installed as a distribution, so no entry points are declared")
    assert [entry.load() for entry in found] == [
        __import__("dynquant.integration.sglang_plugin", fromlist=["register"]).register
    ]
