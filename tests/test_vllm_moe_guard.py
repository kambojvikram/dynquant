"""MoE must fail closed, and keep failing closed as vLLM moves.

DynQuant does not serve a fused MoE layer through vLLM: the packed grouped GEMM is
phase 8 of the kernel work, and until it exists there is nothing for the expert
weights to run on. The plugin says so, by name, while the layer is being built.

That guard needs a test of its own because of how it fails. ``get_quant_method``
returning ``None`` means "use vLLM's unquantized method", and
``RoutedExperts._get_quant_method`` substitutes ``UnquantizedFusedMoEMethod`` when
it gets one -- so a probe that stops matching does not fall through to an error, it
falls through to a *different model*: fp16 experts built for a checkpoint that
holds packed words.

It has already stopped matching once. Through vLLM 0.25 ``FusedMoE`` was the class
that owned expert weights; in 0.26 it is a factory function returning a
``MoERunner``, and the object handed to ``get_quant_method`` is a
``RoutedExperts``. A probe keyed on the class name ``FusedMoE`` matched nothing at
all, silently, which is exactly the failure described above. The probe now keys on
the defining module, and these tests are what would have caught it.

Skips where vLLM is absent rather than pretending to have checked.
"""

from __future__ import annotations

import pytest

pytest.importorskip("vllm", reason="the vLLM plugin needs vLLM to import at all")

import torch  # noqa: E402

from dynquant.errors import DynQuantError  # noqa: E402
from dynquant.integration.vllm_plugin.config import (  # noqa: E402
    DynQuantConfig,
    _class_is_fused_moe,
    _is_fused_moe,
)


def moe_owner_classes() -> list[type]:
    """Every class in the installed vLLM that a quantization method could be asked for.

    Collected from the module rather than named one by one, so a rename upstream
    shows up as this list changing rather than as a test that still passes while
    checking something that no longer exists.
    """
    import inspect
    import pkgutil

    import vllm.model_executor.layers.fused_moe as package

    found: list[type] = []
    for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            module = __import__(info.name, fromlist=["_"])
        except Exception:  # pragma: no cover - optional backends (deep_gemm, pplx)
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__.startswith(package.__name__) and issubclass(
                obj, torch.nn.Module
            ):
                found.append(obj)
    return found


def test_the_layer_vllm_actually_hands_us_is_recognised():
    """``RoutedExperts`` is the one that calls ``get_quant_method`` in vLLM 0.26.

    Not instantiated -- the probe reads the type, and its ``__init__`` wants a
    parallel state, a device and a real expert count to build against.
    """
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    assert _is_fused_moe(RoutedExperts.__new__(RoutedExperts))


def test_every_moe_layer_class_in_this_vllm_is_recognised():
    """Asked of the classes, not instances: several are abstract, and
    ``PluggableLayer.__new__`` refuses to build those."""
    classes = moe_owner_classes()
    assert classes, "found no MoE layer classes at all -- the package moved"
    missed = sorted({cls.__name__ for cls in classes if not _class_is_fused_moe(cls)})
    assert not missed, f"unrecognised MoE layer classes: {missed}"


def test_a_model_file_subclass_is_recognised():
    """Model files subclass these, so a test on the type itself would miss them."""
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    class SomeModelsExperts(RoutedExperts):
        def __init__(self):  # never called
            raise AssertionError

    assert _is_fused_moe(SomeModelsExperts.__new__(SomeModelsExperts))


def test_linear_layers_are_not_caught():
    """They share ``PluggableLayer``, so the probe has to be narrower than that."""
    from vllm.model_executor.layers.linear import (
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    for cls in (
        RowParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        ParallelLMHead,
    ):
        assert not _is_fused_moe(cls.__new__(cls)), cls.__name__
    assert not _is_fused_moe(torch.nn.Linear(4, 4))


def test_the_refusal_names_the_layer_and_the_reason():
    """The error a user actually sees, raised through ``get_quant_method``.

    Reached on a config built with ``__new__``: the refusal has to come before
    anything that reads the schema, because an MoE checkpoint DynQuant cannot
    serve is one it should decline to start on rather than decline halfway
    through loading.
    """
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    config = DynQuantConfig.__new__(DynQuantConfig)
    layer = RoutedExperts.__new__(RoutedExperts)
    with pytest.raises(DynQuantError) as excinfo:
        DynQuantConfig.get_quant_method(config, layer, "model.layers.0.mlp.experts")
    message = str(excinfo.value)
    assert "model.layers.0.mlp.experts" in message
    assert "phase 8" in message
