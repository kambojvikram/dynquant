"""A minimal stand-in for the SGLang surface the plugin writes to and subclasses.

Why this exists at all
----------------------
SGLang stopped publishing ``py3-none-any`` wheels after 0.5.10.post1; 0.5.16 ships
manylinux_2_34 only. So ``pip install sglang`` on Windows or macOS silently resolves
to a release that predates the plugin system entirely, and an
``importorskip("sglang")`` -- the pattern ``test_vllm_*.py`` uses -- would skip the
registration and config tests on every developer machine and every CPU CI runner.
Skipping is the honest answer when the alternative is pretending; it is the wrong
answer when a stand-in can exercise the logic under test for real.

What this is and is not a test of
---------------------------------
Under this stub, :func:`dynquant.integration.sglang_plugin.register` performs its
actual writes, and ``DynQuantConfig`` is built against a base class carrying SGLang's
actual abstract-method set. So the stub tests **the plugin**: idempotence, guard
messages, which container is written, whether a required override was forgotten.

It cannot test **SGLang**. A stub that drifts from the framework passes while the
plugin is broken, which is the failure mode every fake has. That is what
``test_sglang_stub_conformance.py`` is for: where a real SGLang is importable it
compares this module against it, symbol by symbol, and fails if they have parted.
Read the two files as one test split across two machines.

Fidelity notes -- each of these mirrors something verified in the 0.5.16 wheel:

* ``QuantizationConfig`` is an ABC with the same abstract set, *including*
  ``get_scaled_act_names``, which vLLM deleted and SGLang kept. Omitting it in the
  plugin makes ``DynQuantConfig`` abstract, and the resulting ``TypeError`` surfaces
  from inside a spawned scheduler process where it reads as a worker that died for
  no reason. Here it surfaces as a red unit test.
* ``__init__`` sets ``packed_modules_mapping`` to an empty dict and nothing else ever
  populates it -- the reason ``from_config`` has to lift it out of the config dict.
* ``override_quantization_method`` returns ``None`` on the base, and is polled for
  *every* checkpoint by ``ModelConfig._verify_quantization``.
* ``add_quantization_method_choices`` is ``QUANTIZATION_CHOICES.extend(choices)``
  with no dedup, so idempotence is the caller's problem.
* ``UnquantizedEmbeddingMethod`` derives from ``QuantizeMethodBase``, *not* from
  ``UnquantizedLinearMethod``. The vLLM twin does derive, and this stub asserted the
  vLLM relationship until a real SGLang contradicted it on the Linux box. What both
  frameworks agree on is the part that is load-bearing: ``embedding()`` is defined in
  the class body, and ``method_has_implemented_embedding`` tests for it by identity
  against the base.
* The parameter base classes take ``tp_rank`` as an *argument* rather than caching
  it in ``__init__`` the way vLLM does, and ``_ColumnvLLMParameter`` /
  ``RowvLLMParameter`` cooperate through ``**kwargs`` so a subclass of both can pass
  ``output_dim`` and ``input_dim`` up one ``super().__init__``. Both are copied here
  because :class:`~dynquant.integration.sglang_plugin.parameter.DynQuantPackedParameter`
  derives from them for real -- the stub is not standing in for the base class, it is
  standing in for the framework around it.
* ``copy_with_check`` is SGLang's own addition, absent from vLLM, and it refuses
  silent dtype downcasts. Our ``_place`` ends in it, so the stub has to as well or
  the fp32-scales case would only be exercised on the Linux box.
"""

from __future__ import annotations

import contextlib
import sys
import types
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import torch

#: The exact module paths the plugin imports. Listed once, because the teardown has
#: to remove precisely what the setup added -- see :func:`fake_sglang`.
MODULE_NAMES = (
    "sglang",
    "sglang.srt",
    "sglang.srt.layers",
    "sglang.srt.layers.linear",
    "sglang.srt.layers.moe",
    "sglang.srt.layers.moe.fused_moe_triton",
    "sglang.srt.layers.moe.fused_moe_triton.layer",
    "sglang.srt.layers.parameter",
    "sglang.srt.layers.quantization",
    "sglang.srt.layers.quantization.base_config",
    "sglang.srt.layers.quantization.unquant",
    "sglang.srt.layers.vocab_parallel_embedding",
    "sglang.srt.server_args",
    "sglang.srt.utils",
)

#: Everything the plugin's own modules cache, so a stubbed run cannot leave a
#: ``DynQuantConfig`` bound to a stub base class behind for a later real-SGLang test.
PLUGIN_MODULE_PREFIX = "dynquant.integration.sglang_plugin"


class QuantizeMethodBase(ABC):
    """``layers/quantization/base_config.py``. Abstract set as in 0.5.16."""

    @abstractmethod
    def create_weights(self, layer: torch.nn.Module, *args: Any, **kwargs: Any) -> None: ...

    @abstractmethod
    def apply(self, layer: torch.nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor: ...

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return None


class LinearMethodBase(QuantizeMethodBase):
    """``layers/quantization/base_config.py:46`` -- *not* ``linear.py``, as in vLLM.

    Empty because the plugin overrides both halves; what it is standing in for is
    the import path, which is one of the four documented deltas from the vLLM twin
    and the kind of thing that only fails on a machine with SGLang installed.
    """


class QuantizationConfig(ABC):
    """``layers/quantization/base_config.py``, reduced to what the plugin touches."""

    def __init__(self) -> None:
        super().__init__()
        self.packed_modules_mapping: dict[str, list[str]] = {}

    def update_packed_modules_mapping(self, mapping: dict[str, list[str]]) -> None:
        self.packed_modules_mapping = mapping

    @abstractmethod
    def get_name(self) -> str: ...

    @abstractmethod
    def get_supported_act_dtypes(self) -> list[torch.dtype]: ...

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int: ...

    @staticmethod
    @abstractmethod
    def get_config_filenames() -> list[str]: ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> QuantizationConfig: ...

    @abstractmethod
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None: ...

    @abstractmethod
    def get_scaled_act_names(self) -> list[str]: ...

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg: Any, user_quant: Any) -> str | None:
        return None

    def apply_weight_name_mapper(self, hf_to_sglang_mapper: Any) -> None:  # noqa: B027
        # Empty and *not* abstract, which is what B027 objects to -- and the noqa is
        # copied from SGLang, which carries the same one on the same method. Most
        # configs have no module names to remap, so the base has to be a no-op; the
        # cost is that a plugin misspelling the override gets silence rather than a
        # TypeError, which `test_sglang_stub_conformance` checks for by name.
        pass


class LinearBase(torch.nn.Module):
    """``layers/linear.py``. The plugin only ever asks ``isinstance`` of this."""


class ColumnParallelLinear(LinearBase):
    """``layers/linear.py:292``. Here only to carry the two fused classes below.

    Their real base, and worth reproducing rather than flattening: the plugin's
    fused-layer guard tests ``isinstance`` against the *subclasses*, so a stub that
    made them siblings of ``ColumnParallelLinear`` would let an over-broad guard --
    one that matched the parent and so caught every column-parallel layer -- pass.
    """


class MergedColumnParallelLinear(ColumnParallelLinear):
    """``layers/linear.py:491``. ``gate_up_proj``: two checkpoint modules, one layer."""


class QKVParallelLinear(ColumnParallelLinear):
    """``layers/linear.py:920``. ``qkv_proj``: three checkpoint modules, one layer."""


class VocabParallelEmbedding(torch.nn.Module):
    """``layers/vocab_parallel_embedding.py``."""


class ParallelLMHead(VocabParallelEmbedding):
    """``layers/vocab_parallel_embedding.py``."""


class FusedMoE(torch.nn.Module):
    """``layers/moe/fused_moe_triton/layer.py:154``.

    Present only so the MoE guard has something with the right ``__module__`` to
    walk. Its real ``__init__`` assigns
    ``self.quant_method = quant_config.get_quant_method(self, prefix)`` with no
    substitution on ``None`` -- which is why the plugin raises instead of returning
    ``None`` here, and why that raise is worth a test.
    """

    # The probe reads `__module__`, and a class defined here would otherwise carry
    # this file's name. Assigned rather than left to the import system because that
    # attribute *is* the thing under test: `_class_is_fused_moe` matches on the
    # defining package, not on the class name.
    __module__ = "sglang.srt.layers.moe.fused_moe_triton.layer"


class UnquantizedLinearMethod(LinearMethodBase):
    """``layers/quantization/unquant.py:172`` -- *not* ``linear.py``, as in vLLM."""

    def create_weights(self, layer: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
        return None

    def apply(self, layer: torch.nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError


class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    """``layers/quantization/unquant.py:134``, and a sibling of the linear method.

    vLLM derives this from ``UnquantizedLinearMethod``; SGLang does not, so here it
    is not a ``LinearMethodBase`` at all. Written out rather than inherited so the
    stub cannot make a plugin that confuses the two look correct.
    """

    def create_weights(self, layer: torch.nn.Module, *args: Any, **kwargs: Any) -> None:
        return None

    def apply(self, layer: torch.nn.Module, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# --------------------------------------------------------------------------
# layers/parameter.py -- copied rather than summarised, because the plugin's
# parameter subclasses these for real. Everything below is 0.5.16's behaviour
# minus the CPU branch (`narrow_padded_param_and_loaded_weight`) and the
# `pad_or_narrow_weight` special case for non-8-aligned Qwen2.5-VL MLPs, neither
# of which a packed layer can reach: both pad the *loaded* tensor to a shape the
# param expects, and DynQuant's row geometry is derived from the checkpoint
# rather than from the layer.
# --------------------------------------------------------------------------

_DTYPE_RANK = {
    torch.float8_e4m3fn: 0,
    torch.float8_e5m2: 0,
    torch.float16: 1,
    torch.bfloat16: 1,
    torch.float32: 2,
    torch.float64: 3,
}


def copy_with_check(target: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """``layers/parameter.py:51``. SGLang's own; vLLM has no equivalent.

    The real one has an escape hatch -- ``SGLANG_QUANT_ALLOW_DOWNCASTING`` -- left
    out here deliberately. Reading it would drag in ``sglang.srt.environ``, and no
    DynQuant path sets it: a checkpoint whose scales are wider than the layer's
    compute dtype is a packing bug, and the point of routing our copy through this
    function is that it says so.
    """
    assert target.shape == loaded_weight.shape, f"{target.shape=}, {loaded_weight.shape=}"
    if target.dtype == loaded_weight.dtype:
        target.copy_(loaded_weight)
        return

    target_rank = _DTYPE_RANK.get(target.dtype)
    loaded_rank = _DTYPE_RANK.get(loaded_weight.dtype)
    if target_rank is None or loaded_rank is None:
        raise ValueError(
            f"Unsupported copy between dtypes: {target.dtype=}, {loaded_weight.dtype=}"
        )
    if target_rank < loaded_rank:
        raise ValueError(f"Downcasting not allowed: {target.dtype=}, {loaded_weight.dtype=}")
    target.copy_(loaded_weight)


class BasevLLMParameter(torch.nn.Parameter):
    """``layers/parameter.py:82``. Still named for vLLM inside ``sglang``.

    Note what ``__init__`` does *not* do: vLLM's caches
    ``get_tensor_model_parallel_rank()`` on the instance, and SGLang's fork dropped
    that when it started passing ``tp_rank`` into each hook. A port that kept
    reading ``self.tp_rank`` would raise ``AttributeError`` here -- which is the
    good case; the bad one is a port that falls back to a global and is wrong on
    every rank but zero.
    """

    def __new__(cls, data: torch.Tensor, **kwargs: Any) -> BasevLLMParameter:
        del kwargs
        instance = super().__new__(cls, data=data, requires_grad=False)
        return instance  # type: ignore[return-value]

    def __init__(self, data: torch.Tensor, weight_loader: Any) -> None:
        del data
        self._weight_loader = weight_loader

    @property
    def weight_loader(self) -> Any:
        return self._weight_loader

    def _assert_and_load(self, loaded_weight: torch.Tensor) -> None:
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
        self._assert_and_load(loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
        self._assert_and_load(loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs: Any) -> None:
        del kwargs
        self._assert_and_load(loaded_weight)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs: Any) -> None:
        del kwargs
        self._assert_and_load(loaded_weight)


class _ColumnvLLMParameter(BasevLLMParameter):
    """``layers/parameter.py:126``. Private upstream, and subclassed here anyway.

    Not a liberty: ``ColumnParallelLinear.weight_loader_v2`` gates its call
    convention on ``isinstance(param, _ColumnvLLMParameter)``, so there is no
    public way to be on the right side of that test. The underscore is why
    :mod:`test_sglang_stub_conformance` checks the name still exists.
    """

    def __init__(self, output_dim: int, **kwargs: Any) -> None:
        self._output_dim = output_dim
        super().__init__(**kwargs)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def load_column_parallel_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
    ) -> None:
        if not use_presharded_weights:
            shard_size = self.data.shape[self.output_dim]
            loaded_weight = loaded_weight.narrow(self.output_dim, tp_rank * shard_size, shard_size)
        copy_with_check(self.data, loaded_weight)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs: Any) -> None:
        shard_offset = kwargs.get("shard_offset")
        shard_size = kwargs.get("shard_size")
        tp_rank = kwargs.get("tp_rank")
        param_data = self.data.narrow(self.output_dim, shard_offset, shard_size)
        if not kwargs.get("use_presharded_weights"):
            loaded_weight = loaded_weight.narrow(self.output_dim, tp_rank * shard_size, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def load_qkv_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
        **kwargs: Any,
    ) -> None:
        shard_offset = kwargs.get("shard_offset")
        shard_size = kwargs.get("shard_size")
        shard_id = kwargs.get("shard_id")
        num_heads = kwargs.get("num_heads")
        source = tp_rank if shard_id == "q" else tp_rank // num_heads
        param_data = self.data.narrow(self.output_dim, shard_offset, shard_size)
        if not use_presharded_weights:
            loaded_weight = loaded_weight.narrow(self.output_dim, source * shard_size, shard_size)
        assert param_data.shape == loaded_weight.shape, (
            f"{param_data.shape=}, {loaded_weight.shape=}"
        )
        param_data.copy_(loaded_weight)


class RowvLLMParameter(BasevLLMParameter):
    """``layers/parameter.py:277``."""

    def __init__(self, input_dim: int, **kwargs: Any) -> None:
        self._input_dim = input_dim
        super().__init__(**kwargs)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    def load_row_parallel_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
    ) -> None:
        if not use_presharded_weights:
            shard_size = self.data.shape[self.input_dim]
            loaded_weight = loaded_weight.narrow(self.input_dim, tp_rank * shard_size, shard_size)
        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)


class ModelWeightParameter(_ColumnvLLMParameter, RowvLLMParameter):
    """``layers/parameter.py``. SGLang's own both-parents parameter.

    Here as the precedent rather than for use: ``DynQuantPackedParameter`` has the
    same two bases in the same order, and the cooperative ``**kwargs`` chain that
    makes one ``super().__init__`` reach all three levels is a property of *these*
    classes, so it belongs in the copy.
    """


def set_weight_attrs(weight: torch.Tensor, weight_attrs: dict[str, Any] | None) -> None:
    """``utils/common.py:2192``, re-exported through ``sglang.srt.utils``.

    Byte-for-byte vLLM's, assert included. The assert is the reason
    ``create_weights`` strips ``weight_loader`` out of ``extra_weight_attrs``
    before forwarding: ``BasevLLMParameter`` has already installed one, and this
    refuses to overwrite rather than winning silently.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"
        setattr(weight, key, value)


def _build_modules() -> dict[str, types.ModuleType]:
    """The tree, assembled fresh per entry so no test can mutate another's registry."""
    modules = {name: types.ModuleType(name) for name in MODULE_NAMES}

    # Packages need `__path__` or `from x.y import z` on a submodule fails even when
    # the submodule is already in sys.modules.
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.layers",
        "sglang.srt.layers.moe",
        "sglang.srt.layers.moe.fused_moe_triton",
        "sglang.srt.layers.quantization",
        # A package upstream too (`sglang/srt/utils/__init__.py`, which is
        # `from .common import *`). Faked flat, since the plugin imports the
        # re-export rather than the module it comes from.
        "sglang.srt.utils",
    ):
        modules[name].__path__ = []  # type: ignore[attr-defined]

    modules["sglang.srt.layers.moe.fused_moe_triton.layer"].FusedMoE = FusedMoE  # type: ignore[attr-defined]

    linear = modules["sglang.srt.layers.linear"]
    linear.LinearBase = LinearBase  # type: ignore[attr-defined]
    linear.ColumnParallelLinear = ColumnParallelLinear  # type: ignore[attr-defined]
    linear.MergedColumnParallelLinear = MergedColumnParallelLinear  # type: ignore[attr-defined]
    linear.QKVParallelLinear = QKVParallelLinear  # type: ignore[attr-defined]
    # A copy of the real 0.5.16 contents, not an empty list: `register()` appends to
    # whatever is here, and a test that asserted on a list it had also emptied would
    # not notice the plugin clobbering the entries SGLang ships with.
    linear.WEIGHT_LOADER_V2_SUPPORTED = [  # type: ignore[attr-defined]
        "CompressedTensorsLinearMethod",
        "AWQLinearMethod",
        "GPTQMarlinLinearMethod",
        "Fp8LinearMethod",
        "MarlinLinearMethod",
        "GPTQLinearMethod",
    ]

    quantization = modules["sglang.srt.layers.quantization"]
    # Both halves of the 0.5.16 split, so a plugin that wrote to the wrong one is
    # visibly wrong here rather than only on a real server.
    quantization.BASE_QUANTIZATION_METHODS = {"awq": object, "gptq": object}  # type: ignore[attr-defined]
    quantization.QUANTIZATION_METHODS = dict(  # type: ignore[attr-defined]
        quantization.BASE_QUANTIZATION_METHODS  # type: ignore[attr-defined]
    )

    parameter = modules["sglang.srt.layers.parameter"]
    parameter.BasevLLMParameter = BasevLLMParameter  # type: ignore[attr-defined]
    parameter._ColumnvLLMParameter = _ColumnvLLMParameter  # type: ignore[attr-defined]
    parameter.RowvLLMParameter = RowvLLMParameter  # type: ignore[attr-defined]
    parameter.ModelWeightParameter = ModelWeightParameter  # type: ignore[attr-defined]
    parameter.copy_with_check = copy_with_check  # type: ignore[attr-defined]

    utils = modules["sglang.srt.utils"]
    utils.set_weight_attrs = set_weight_attrs  # type: ignore[attr-defined]

    base_config = modules["sglang.srt.layers.quantization.base_config"]
    base_config.QuantizationConfig = QuantizationConfig  # type: ignore[attr-defined]
    base_config.QuantizeMethodBase = QuantizeMethodBase  # type: ignore[attr-defined]
    base_config.LinearMethodBase = LinearMethodBase  # type: ignore[attr-defined]

    unquant = modules["sglang.srt.layers.quantization.unquant"]
    unquant.UnquantizedLinearMethod = UnquantizedLinearMethod  # type: ignore[attr-defined]
    unquant.UnquantizedEmbeddingMethod = UnquantizedEmbeddingMethod  # type: ignore[attr-defined]

    embedding = modules["sglang.srt.layers.vocab_parallel_embedding"]
    embedding.VocabParallelEmbedding = VocabParallelEmbedding  # type: ignore[attr-defined]
    embedding.ParallelLMHead = ParallelLMHead  # type: ignore[attr-defined]

    server_args = modules["sglang.srt.server_args"]
    server_args.QUANTIZATION_CHOICES = ["awq", "fp8", "gptq", "marlin"]  # type: ignore[attr-defined]

    def add_quantization_method_choices(choices: list[str]) -> None:
        # Verbatim from server_args.py:359 -- extend, no dedup. Idempotence is the
        # plugin's job, and a stub that deduplicated would hide it failing to do so.
        server_args.QUANTIZATION_CHOICES.extend(choices)  # type: ignore[attr-defined]

    server_args.add_quantization_method_choices = add_quantization_method_choices  # type: ignore[attr-defined]

    return modules


@contextlib.contextmanager
def fake_sglang(**overrides: Any) -> Iterator[types.ModuleType]:
    """Install the stub tree, yield ``sglang.srt.layers.quantization``, restore.

    ``overrides`` sets attributes on the modules after construction, addressed as
    ``"sglang.srt.server_args:QUANTIZATION_CHOICES"``. That is how the guard tests
    make a container the wrong type; passing :data:`REMOVED` deletes it instead.

    Restoration is exact rather than "delete the sglang keys". Two things have to be
    put back: the stub modules, and every plugin module imported while they were in
    place. ``DynQuantConfig`` closes over whichever ``QuantizationConfig`` was
    visible at class-creation time, so a cached stub-based one would poison a
    real-SGLang test later in the same session -- and on the Linux box, both run.
    """
    saved = {name: sys.modules[name] for name in list(sys.modules) if _is_ours(name)}
    for name in saved:
        del sys.modules[name]

    modules = _build_modules()
    for target, value in overrides.items():
        module_name, _, attribute = target.partition(":")
        if value is REMOVED:
            delattr(modules[module_name], attribute)
        else:
            setattr(modules[module_name], attribute, value)

    sys.modules.update(modules)
    try:
        yield modules["sglang.srt.layers.quantization"]
    finally:
        touched = {name for name in sys.modules if _is_ours(name)} | set(saved)
        for name in touched:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        for name in sorted(touched):
            _resync_parent_attribute(name)


def _is_ours(name: str) -> bool:
    return name in MODULE_NAMES or name.split(".")[0] == "sglang" or _is_plugin(name)


def _is_plugin(name: str) -> bool:
    return name == PLUGIN_MODULE_PREFIX or name.startswith(PLUGIN_MODULE_PREFIX + ".")


def _resync_parent_attribute(name: str) -> None:
    """Point ``parent.child`` at whatever ``sys.modules`` now says, or unbind it.

    ``sys.modules`` is not the only place an imported submodule is recorded -- the
    import system also sets it as an attribute of its parent package, and that copy
    survives a ``del sys.modules[...]``. Leaving it stale is how a stub leaks: after
    the context exits, ``dynquant.integration.sglang_plugin.config`` would still
    *attribute*-resolve to the class built against the stub base, while an
    ``import`` of the same path builds a fresh one against the real SGLang. Two
    ``DynQuantConfig`` classes, and which one a test gets depends on how it spelled
    the import.
    """
    parent_name, _, child = name.rpartition(".")
    parent = sys.modules.get(parent_name) if parent_name else None
    if parent is None:
        return
    restored = sys.modules.get(name)
    if restored is None:
        with contextlib.suppress(AttributeError):
            delattr(parent, child)
    else:
        setattr(parent, child, restored)


class _Removed:
    """Sentinel: ``None`` and ``0`` are both plausible wrong *values* for a
    container, so "absent" needs a spelling that cannot collide with one."""

    def __repr__(self) -> str:
        return "REMOVED"


REMOVED = _Removed()
