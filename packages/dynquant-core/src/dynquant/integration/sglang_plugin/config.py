"""``DynQuantConfig`` -- what SGLang asks when it wants to know how to build a layer.

The vLLM twin of this module is
:mod:`dynquant.integration.vllm_plugin.config`, and the resemblance is not
coincidence: SGLang's ``base_config.py`` still carries
``Adapted from .../vllm/v0.5.5/...`` in its header. The differences below are the
places where the fork has since drifted, and each one is load-bearing.

Registered into SGLang's registry by
:func:`dynquant.integration.sglang_plugin.register`, which is why nothing imports
this module at ``dynquant`` import time: it needs SGLang, and DynQuant does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)

# Not `sglang.srt.layers.linear` / `...vocab_parallel_embedding`, which is where vLLM
# keeps these two. SGLang moved both into the quantization package, and importing
# them from the vLLM locations raises an ImportError that reads like the layer module
# is missing rather than like the class moved.
from sglang.srt.layers.quantization.unquant import (
    UnquantizedEmbeddingMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead

from dynquant.constants import HF_QUANT_METHOD
from dynquant.errors import DynQuantError
from dynquant.integration.serving_common.schema import QuantizationConfigSchema

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sglang.srt.models.utils import WeightsMapper

__all__ = ["DynQuantConfig"]

MIN_CAPABILITY = 75
"""Turing. The packed GEMV needs ``__hfma2`` and ``lop3.b32``, both of which are
sm_75; below that the fallback would be dequantize-then-GEMM, which uses the fp16
weight's worth of VRAM and so gives up the only thing quantization was for."""


class DynQuantConfig(QuantizationConfig):
    """Per-module bit widths, which is the part SGLang has no other example of.

    Every other config in SGLang's registry answers "what precision is this
    checkpoint?" with one number. DynQuant's answer is a map, because allocating
    width by measured sensitivity is the method rather than a tuning option.
    """

    def __init__(
        self,
        schema: QuantizationConfigSchema,
        packed_modules_mapping: dict[str, list[str]] | None = None,
    ) -> None:
        super().__init__()
        self.schema = schema
        # See `from_config` for why this is a constructor argument at all.
        if packed_modules_mapping:
            self.packed_modules_mapping = dict(packed_modules_mapping)

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

        See :mod:`dynquant.integration.serving_common.schema` -- ``from_config`` is
        handed only the parsed dict, with no path to resolve a sidecar against.
        """
        return []

    def get_scaled_act_names(self) -> list[str]:
        """Empty, and present only because SGLang still declares it abstract.

        vLLM deleted this from its base class; SGLang's fork kept it
        (``base_config.py``), and an abstract method left unimplemented does not
        fail at registration -- it fails at the first instantiation, as a
        ``TypeError: Can't instantiate abstract class`` raised inside a spawned
        scheduler process, where it reads as a worker that died for no reason.

        It is AWQ's hook for post-scaling activations of specific ops. DynQuant
        rescales inside the packed GEMV, so there is nothing for the caller to do.
        """
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> DynQuantConfig:
        """Build from the ``quantization_config`` block, plus what SGLang slipped in.

        The signature is vLLM's, unchanged. The *caller* is not: SGLang mutates the
        dict on the way in, at ``model_loader/weight_utils.py``::

            hf_quant_config["packed_modules_mapping"] = packed_modules_mapping
            return quant_cls.from_config(hf_quant_config)

        and then nothing copies that key onto the instance. The base
        ``QuantizationConfig.__init__`` sets ``self.packed_modules_mapping`` to an
        empty dict, and the only caller of ``update_packed_modules_mapping()`` in the
        whole tree is one model file. ``GPTQConfig`` and ``AWQConfig`` never notice,
        because neither needs fusion structure to build a layer.

        DynQuant does need it. ``resolve_shards`` is how a fused ``qkv_proj`` learns
        it is three modules at three different widths, so without the mapping every
        fused layer resolves to no shards and takes the unquantized path against a
        checkpoint holding packed words. Hence the lift here.
        """
        mapping = config.get("packed_modules_mapping") or {}
        return cls(QuantizationConfigSchema.from_dict(config), packed_modules_mapping=mapping)

    def apply_weight_name_mapper(self, hf_to_sglang_mapper: WeightsMapper) -> None:
        """SGLang's spelling of vLLM's ``apply_vllm_mapper``. Same ``WeightsMapper``."""
        self.schema = self.schema.remap(hf_to_sglang_mapper.apply_list)

    # -- per-layer dispatch ------------------------------------------------

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        # Asked before anything else. Partly for the reason the vLLM plugin gives --
        # being wrong in this direction costs an `__mro__` walk, and being wrong in
        # the other quantizes an MoE with a method that cannot run it -- and partly
        # so the refusal does not first pay for the linear import below.
        if _is_fused_moe(layer):
            raise DynQuantError(
                f"{prefix or type(layer).__name__} is a fused MoE layer, which DynQuant "
                f"does not serve through SGLang yet: the packed grouped GEMM is phase 8. "
                f"Serve an MoE model unquantized, or quantize a dense model."
            )

        # Imported here, not at module scope: linear.py imports this module for the
        # config type, and importing it back at the top would be a cycle.
        from dynquant.integration.sglang_plugin.linear import (
            DynQuantEmbeddingMethod,
            DynQuantLinearMethod,
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
                # An explicit method, not `None`, and this is where SGLang differs
                # from vLLM in a way that bites. vLLM's `LinearBase.__init__`
                # substitutes `UnquantizedLinearMethod()` when `get_quant_method`
                # returns None; SGLang's does that only when `quant_config` is
                # *absent* (`linear.py:176-181`). With a config present the None is
                # simply assigned, and every subclass then trips a bare
                # `assert self.quant_method is not None` (`linear.py:228`, `:280`,
                # `:346`, `:472`, `:1438`, `:1570`) -- an AssertionError carrying no
                # message at all, on a line that does not mention quantization.
                return UnquantizedLinearMethod()
            return DynQuantLinearMethod(self, shards)

        # Everything else -- plain VocabParallelEmbedding, norms, RadixAttention --
        # keeps SGLang's own handling. `None` is the correct answer rather than an
        # omission: `VocabParallelEmbedding` substitutes `UnquantizedEmbeddingMethod`
        # on None (`vocab_parallel_embedding.py:297`), and `RadixAttention` skips
        # `create_weights` entirely when the method is None
        # (`radix_attention.py:135`). The exporter leaves those tensors in the
        # compute dtype, so that is what we want in both cases.
        return None


#: Every class SGLang uses to own expert weights lives under this package --
#: ``FusedMoE`` in ``fused_moe_triton.layer``, ``DeepEPMoE`` deriving from it in
#: ``ep_moe.layer``, and the ``moe_runner`` backends. Matching the defining module
#: rather than the class name is what makes the guard survive a reorganisation, and
#: the package has already been reorganised once -- ``moe_runner/`` did not exist
#: when the vLLM twin of this constant was written.
_MOE_PACKAGE = "sglang.srt.layers.moe"


def _is_fused_moe(layer: torch.nn.Module) -> bool:
    """True for SGLang's MoE layer, without importing it.

    A walk over ``__mro__`` rather than ``isinstance`` because the MoE module tree
    drags in the whole kernel stack on import -- a real cost in every worker process
    just to answer a type question for models that do not have one.

    Unlike the vLLM twin, a missed match here would *not* silently build fp16
    experts: SGLang assigns ``self.quant_method = quant_config.get_quant_method(...)``
    with no substitution on None (``fused_moe_triton/layer.py:322-325``), so a None
    fails on the next line. The explicit guard is still worth having, because
    ``'NoneType' object has no attribute 'create_weights'`` does not tell anyone that
    MoE support is a later phase.
    """
    return _class_is_fused_moe(type(layer))


def _class_is_fused_moe(cls: type) -> bool:
    """The question :func:`_is_fused_moe` actually asks, on the type itself.

    Separate so that a test can sweep every class in SGLang's MoE package without
    needing to construct one -- several take a configured distributed environment.
    """
    return any(
        base.__module__ == _MOE_PACKAGE or base.__module__.startswith(_MOE_PACKAGE + ".")
        for base in cls.__mro__
    )
