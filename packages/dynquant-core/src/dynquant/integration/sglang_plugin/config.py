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
from sglang.srt.layers.linear import (
    LinearBase,
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
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
from dynquant.integration.serving_common.schema import (
    CONVENTIONAL_FUSED_MODULES,
    ModuleQuantSpec,
    QuantizationConfigSchema,
)

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

        The lift is necessary and not sufficient: what SGLang injects is
        ``getattr(model_class, "packed_modules_mapping", {})``
        (``model_loader/loader.py:204``), and on 0.5.16 that attribute is absent from
        172 of the 210 files in ``srt/models/`` -- including ``Qwen2ForCausalLM``,
        which fuses q/k/v inside ``load_weights`` all the same. So an empty dict here
        is the common case rather than the broken one, and
        :data:`CONVENTIONAL_FUSED_MODULES` covers it at the point of use.
        """
        mapping = config.get("packed_modules_mapping") or {}
        return cls(QuantizationConfigSchema.from_dict(config), packed_modules_mapping=mapping)

    def _shards(self, prefix: str) -> list[tuple[str, ModuleQuantSpec]] | None:
        """:meth:`QuantizationConfigSchema.resolve_shards`, with SGLang's gap filled.

        Every call goes through here so the fallback cannot be applied on one branch
        of :meth:`get_quant_method` and forgotten on the other.
        """
        return self.schema.resolve_shards(
            prefix,
            self.packed_modules_mapping,
            fusion_defaults=CONVENTIONAL_FUSED_MODULES,
        )

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
            shards = self._shards(prefix)
            if shards is None:
                return UnquantizedEmbeddingMethod()
            return DynQuantEmbeddingMethod(self, shards)

        if isinstance(layer, LinearBase):
            shards = self._shards(prefix)
            if shards is None:
                _refuse_an_empty_fused_layer(layer, prefix, self)
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


#: The layer classes whose prefix names no tensor on disk, so falling back to an
#: unquantized method against a packed checkpoint leaves them with nothing to load.
#: ``MergedColumnParallelRepeatedLinear`` is a third fused class and is deliberately
#: absent: its own docstring says "quantization is not supported yet", so it is
#: never handed a quant_config and never reaches this code.
_FUSED_LINEAR_CLASSES = (MergedColumnParallelLinear, QKVParallelLinear)


def _refuse_an_empty_fused_layer(
    layer: torch.nn.Module, prefix: str, config: DynQuantConfig
) -> None:
    """Raise before a fused layer is built with nothing to fill it.

    An empty resolution has two meanings that the return value cannot tell apart,
    and they are as far apart in consequence as two outcomes get. For an ordinary
    module it means the checkpoint left this one dense, and the dense weight is
    there on disk to load -- correct, and common. For a *fused* layer it means the
    server is about to build a ``qkv_proj`` that no tensor is named after: the
    checkpoint stores ``q_proj``/``k_proj``/``v_proj`` packed, SGLang's loader
    rewrites their names to ``qkv_proj.qweight`` and friends, finds no such
    parameter on an unquantized layer, and drops every one of them with
    ``logger.warning`` (``models/qwen2.py:639``). Nothing raises. The layer keeps
    the uninitialised buffer it was constructed with and the server answers
    requests.

    That is not hypothetical: it is what this integration did on its first real
    serve, and the only outward sign was a wall of "Parameter ... not found in
    params_dict" among the startup logs, half of them naming ``qkqkv_proj`` --
    SGLang's ``stacked_params_mapping`` loop reuses the already-rewritten ``name``
    after a failed lookup, so ``"qkv_proj".replace("v_proj", "qkv_proj")`` mangles
    it a second time. Both the mangling and the missing parameters were downstream
    of this one decision.

    The class test is what makes the check precise, and a name test would not be:
    ``o_proj`` in a quantized attention block also resolves to nothing when the
    exporter leaves it dense, and that is fine. Only a fused layer has no on-disk
    counterpart to fall back to.

    The sibling test is the second half. A fused layer inside a region the exporter
    never touched -- a vision tower beside a quantized language model -- resolves
    empty for a good reason and loads its dense weights normally.
    """
    if not isinstance(layer, _FUSED_LINEAR_CLASSES):
        return
    siblings = config.schema.quantized_siblings(prefix)
    if not siblings:
        return
    raise DynQuantError(
        f"{prefix} is {type(layer).__name__}, a fused layer, and none of the "
        f"checkpoint modules it fuses are in this checkpoint's quantization map -- "
        f"while {siblings} beside it are. Serving it unquantized would build a layer "
        f"with no weights to load, because the checkpoint stores these packed and "
        f"keeps no dense copy.\n"
        f"The prefix is expanded through the model class's `packed_modules_mapping`, "
        f"which SGLang declares on 38 of its 210 model files; where it is missing "
        f"DynQuant falls back to {dict(CONVENTIONAL_FUSED_MODULES)}. Neither named a "
        f"module in this checkpoint.\n"
        f"Declare the fusion on the model class, or list every shard in "
        f"quantization_config.modules_to_not_convert to serve this layer dense."
    )


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
