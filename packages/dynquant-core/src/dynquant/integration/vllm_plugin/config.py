"""``DynQuantConfig`` -- what vLLM asks when it wants to know how to build a layer.

Registered into vLLM's quantization registry by
:func:`dynquant.integration.vllm_plugin.register`, which is why nothing imports
this module at ``dynquant`` import time: it needs vLLM, and DynQuant does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)

from dynquant.constants import HF_QUANT_METHOD
from dynquant.errors import DynQuantError
from dynquant.integration.vllm_plugin.schema import QuantizationConfigSchema

if TYPE_CHECKING:  # pragma: no cover - typing only
    from vllm.model_executor.models.utils import WeightsMapper

__all__ = ["DynQuantConfig"]

MIN_CAPABILITY = 75
"""Turing. The packed GEMV needs ``__hfma2`` and ``lop3.b32``, both of which are
sm_75; below that the fallback would be dequantize-then-GEMM, which uses the fp16
weight's worth of VRAM and so gives up the only thing quantization was for."""


class DynQuantConfig(QuantizationConfig):
    """Per-module bit widths, which is the part vLLM has no other example of.

    Every other config in vLLM's registry answers "what precision is this
    checkpoint?" with one number. DynQuant's answer is a map, because allocating
    width by measured sensitivity is the method rather than a tuning option. The
    consequences are contained to two places: this class hands each layer its own
    ``bits`` at construction time, and
    :class:`~dynquant.integration.vllm_plugin.parameter.DynQuantPackedParameter`
    lays out a fused layer whose shards disagree.
    """

    def __init__(self, schema: QuantizationConfigSchema) -> None:
        super().__init__()
        self.schema = schema

    def __repr__(self) -> str:
        widths = sorted({spec.bits for spec in self.schema.modules.values()})
        return (
            f"DynQuantConfig(modules={len(self.schema.modules)}, bits={widths}, "
            f"group_size={self.schema.group_size}, "
            f"lm_head_quantized={self.schema.lm_head_quantized})"
        )

    # -- registry contract -------------------------------------------------

    def get_name(self) -> str:
        return HF_QUANT_METHOD

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return MIN_CAPABILITY

    @staticmethod
    def get_config_filenames() -> list[str]:
        """None: the whole configuration is in ``config.json``.

        vLLM uses this list to look for a sidecar next to the weights, but
        ``from_config`` is handed only the parsed dict, with no path to resolve a
        sidecar against for a Hub model. See
        :mod:`dynquant.integration.vllm_plugin.schema`.
        """
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DynQuantConfig":
        return cls(QuantizationConfigSchema.from_dict(config))

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper") -> None:
        self.schema = self.schema.remap(hf_to_vllm_mapper.apply_list)

    # -- per-layer dispatch ------------------------------------------------

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # Imported here, not at module scope: linear.py imports this module for
        # the config type, and importing it back at the top would be a cycle.
        from dynquant.integration.vllm_plugin.linear import (
            DynQuantEmbeddingMethod,
            DynQuantLinearMethod,
        )

        # Asked before the linear cases, not after. The layer that owns expert
        # weights has changed class twice in vLLM's history and could plausibly
        # acquire a linear base; being wrong in that direction quantizes an MoE
        # with a method that cannot run it, and being wrong in this one costs a
        # `__mro__` walk per layer.
        if _is_fused_moe(layer):
            raise DynQuantError(
                f"{prefix or type(layer).__name__} is a fused MoE layer, which DynQuant "
                f"does not serve through vLLM yet: the packed grouped GEMM is phase 8. "
                f"Serve an MoE model unquantized, or quantize a dense model."
            )

        if isinstance(layer, ParallelLMHead):
            if not self.schema.lm_head_quantized:
                return UnquantizedEmbeddingMethod()
            shards = self.schema.resolve_shards(prefix, self.packed_modules_mapping)
            if shards is None:
                return UnquantizedEmbeddingMethod()
            return DynQuantEmbeddingMethod(self, shards)

        if isinstance(layer, LinearBase):
            shards = self.schema.resolve_shards(prefix, self.packed_modules_mapping)
            if shards is None:
                return UnquantizedLinearMethod()
            return DynQuantLinearMethod(self, shards)

        # Everything else -- plain VocabParallelEmbedding, norms, rotary caches --
        # keeps vLLM's own unquantized method. Returning None is how the caller
        # is told that, and it is the correct answer rather than an omission: the
        # exporter leaves those tensors in the compute dtype.
        return None


#: Every class vLLM has used to own expert weights lives under this package --
#: ``FusedMoE`` when it was a class, ``RoutedExperts`` and ``MoERunner`` since it
#: became a factory. Matching the defining module rather than the class name is
#: what makes the guard survive that kind of reorganisation, and it is specific:
#: the base these layers share with every linear layer, ``PluggableLayer``, is
#: defined in ``vllm.model_executor.custom_op`` and so does not match.
_MOE_PACKAGE = "vllm.model_executor.layers.fused_moe"


def _is_fused_moe(layer: torch.nn.Module) -> bool:
    """True for vLLM's MoE layer, without importing it.

    The MoE module tree drags in the whole kernel stack on import, which is a
    real cost to pay in every worker process just to answer a type question for
    models that do not have one -- hence a walk over ``__mro__`` rather than
    ``isinstance``.

    Getting this wrong is not a crash, which is why it is worth stating what the
    check is for. ``get_quant_method`` returning ``None`` means "vLLM's own
    unquantized method", and ``RoutedExperts._get_quant_method`` substitutes
    ``UnquantizedFusedMoEMethod`` on ``None``. A probe that failed to match would
    therefore not fall through to the error below -- it would fall through to
    fp16 experts being built for a checkpoint that holds packed words.
    """
    return _class_is_fused_moe(type(layer))


def _class_is_fused_moe(cls: type) -> bool:
    """The question :func:`_is_fused_moe` actually asks, on the type itself.

    Separate because several of vLLM's MoE classes are abstract, and
    ``PluggableLayer.__new__`` refuses to build those -- so a test that wants to
    sweep every class in the package cannot get an instance to ask about.
    """
    return any(
        base.__module__ == _MOE_PACKAGE or base.__module__.startswith(_MOE_PACKAGE + ".")
        for base in cls.__mro__
    )
