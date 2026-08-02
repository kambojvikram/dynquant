"""The two ``QuantizeMethodBase`` implementations SGLang builds layers from.

Both go through :mod:`dynquant.runtime.ops` rather than calling
``torch.ops.dynquant.*`` directly, so ``DYNQUANT_BACKEND=torch`` still selects the
reference implementation *inside SGLang*. That is what makes "does the served
model match the packed weights" a question anyone can answer on the machine that
is serving, instead of one that needs a second harness.

Four differences from the vLLM twin
-----------------------------------
1. ``LinearMethodBase`` is imported from ``layers/quantization/base_config.py``.
   vLLM keeps it in ``layers/linear.py``, and importing it from there here fails
   with an ImportError that reads as though the layer module were missing.
2. There is no ``@register_weight_loader_v2_supported_method``. SGLang's fork
   predates that decorator; opting in is an append of this class's *name* to
   ``WEIGHT_LOADER_V2_SUPPORTED``, done in
   :func:`dynquant.integration.sglang_plugin.register`. Same effect, different
   spelling, and the name is checked against this class by the tests.
3. ``create_weights`` takes ``skip_block_quant_check``. SGLang passes it to every
   column-parallel layer (``linear.py:366``, ``:1014``); left to fall into
   ``**extra_weight_attrs`` it would be stapled onto all three tensors as a weight
   attribute -- silently, since ``set_weight_attrs`` only objects to *overwriting*.
4. No ``tie_weights``. SGLang's ``ParallelLMHead.tie_weights`` assigns
   ``self.weight = embed_tokens.weight`` directly and never consults the quant
   method, so the vLLM override would be dead code that looked live. See
   :class:`DynQuantEmbeddingMethod`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from sglang.srt.layers.quantization.base_config import LinearMethodBase
from sglang.srt.utils import set_weight_attrs

from dynquant.errors import DynQuantError
from dynquant.integration.serving_common.fuse import fused_shard_concat
from dynquant.integration.serving_common.geometry import (
    FusedPackedGeometry,
    match_shards_to_partitions,
    row_parallel_split,
)
from dynquant.integration.serving_common.schema import ModuleQuantSpec
from dynquant.integration.sglang_plugin.parameter import DynQuantPackedParameter
from dynquant.quant.pack import row_geometry
from dynquant.quant.tensor import QuantLayout, QuantTensor
from dynquant.runtime import ops

if TYPE_CHECKING:  # pragma: no cover - typing only
    from dynquant.integration.serving_common.geometry import TensorParallelSplit
    from dynquant.integration.sglang_plugin.config import DynQuantConfig

__all__ = ["DynQuantEmbeddingMethod", "DynQuantLinearMethod"]


class _Stash(Protocol):
    """What ``create_weights`` leaves on the layer for ``apply`` to find.

    A quantization method owns no state per layer -- SGLang builds one method per
    layer but ``apply`` is handed the module anyway -- so everything
    shape-dependent is stashed on the layer itself. That is the framework's own
    convention, but ``nn.Module.__getattr__`` is typed ``Tensor | Module``, which
    is right for submodules and parameters and wrong for a geometry object or an
    int. Declaring the stash once is cheaper than casting at each of a dozen
    reads, and it is also the only written record of the contract between the two
    halves of this class.
    """

    dynquant_geometry: FusedPackedGeometry
    dynquant_row_split: TensorParallelSplit | None
    dynquant_in_features: int
    dynquant_spec: ModuleQuantSpec
    qweight: torch.Tensor
    scales: torch.Tensor
    offsets: torch.Tensor


def _stash(layer: torch.nn.Module) -> _Stash:
    return cast(_Stash, layer)


class DynQuantLinearMethod(LinearMethodBase):
    """A linear layer whose fused shards may each have their own bit width.

    The class *name* is load-bearing. ``ColumnParallelLinear.__init__`` picks
    between two weight loaders by testing ``self.quant_method.__class__.__name__``
    against ``WEIGHT_LOADER_V2_SUPPORTED`` (``linear.py:369``, ``:1454``), so a
    rename that missed the literal in the registration shim would leave the v1
    loader in place. That does not corrupt anything -- every v1 path ends in
    ``assert param_data.shape == loaded_weight.shape`` before the copy, and our
    flat 1-D buffers fail it -- but the failure is a bare ``AssertionError``
    inside a spawned scheduler subprocess, naming neither the module nor
    quantization. The real point of the registry entry is upstream of that: the
    v1 loader has no way to *express* this layout, because it places shards with
    ``narrow(output_dim, ...)`` on the assumption that all rows are the same
    width, and here they are not.
    """

    def __init__(
        self, quant_config: DynQuantConfig | None, shards: list[tuple[str, ModuleQuantSpec]]
    ) -> None:
        self.quant_config = quant_config
        self.shards = shards

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        skip_block_quant_check: bool = False,
        **extra_weight_attrs: Any,
    ) -> None:
        """
        Args:
            skip_block_quant_check: Declared, defaulted and then ignored. It is
                FP8's flag for a layer whose shape does not divide the block quant
                tile, and DynQuant has no block quantization to check. It is named
                here rather than left to ``**extra_weight_attrs`` for two reasons:
                the attrs dict is forwarded to ``set_weight_attrs``, which would
                attach a stray bool to all three tensors; and a reader looking for
                where SGLang's kwarg goes should find an answer, not an absence.
                Positioned after ``params_dtype`` because ``ReplicatedLinear``
                (``linear.py:231``) passes the first six arguments positionally.
        """
        del output_size, skip_block_quant_check  # see the docstring
        # SGLang compiles the model with `fullgraph`, so the backend probe has to
        # have run before the first forward is traced -- see `runtime.ops`. Layer
        # construction is the last point at which it is an ordinary call.
        ops.warm_dispatch()
        specs = match_shards_to_partitions(
            self.shards, output_partition_sizes, input_size_per_partition
        )
        geometry = FusedPackedGeometry(specs)

        # Row-parallel layers split the reduction dimension, which for packed
        # weights means splitting the word axis -- only legal on group
        # boundaries. Checking here, while the layer is being built, means an
        # illegal --tp-size fails at startup with the module named, rather than
        # after the weights are on the GPU.
        split = None
        tp_in = input_size // input_size_per_partition if input_size_per_partition else 1
        if tp_in > 1:
            if len(specs) != 1:
                raise DynQuantError(
                    f"row-parallel layer {[s.name for s in specs]} has more than one "
                    f"fused shard. Each width implies its own words-per-rank, so the "
                    f"word axis has no single split. SGLang does not build such a "
                    f"layer, so this means the quantization map does not describe "
                    f"this model."
                )
            split = row_parallel_split(
                bits=specs[0].bits,
                group_size=specs[0].group_size,
                in_features=input_size,
                tp_size=tp_in,
                name=specs[0].name,
            )

        stash = _stash(layer)
        stash.dynquant_geometry = geometry
        stash.dynquant_row_split = split
        # Read back in apply(). Not `layer.input_size_per_partition`, which
        # ReplicatedLinear does not define.
        stash.dynquant_in_features = input_size_per_partition

        def make(numel: int, dtype: torch.dtype, kind: str) -> DynQuantPackedParameter:
            return DynQuantPackedParameter(
                data=torch.empty(numel, dtype=dtype),
                weight_loader=extra_weight_attrs.get("weight_loader"),
                geometry=geometry,
                elements_per_row=kind,
                row_split=split,
            )

        qweight = make(geometry.qweight_numel, torch.int32, "qweight")
        scales = make(geometry.scale_numel, params_dtype, "scale")
        offsets = make(geometry.scale_numel, params_dtype, "scale")

        # `weight_loader` is already installed: `BasevLLMParameter.__init__` takes
        # it, and `make` above passes it. SGLang's `set_weight_attrs` is vLLM's
        # verbatim, including the `assert not hasattr` that refuses to overwrite,
        # so handing it the whole dict raises `Overwriting existing tensor
        # attribute: weight_loader` while the layer is being built -- before a
        # single weight has been read.
        rest = {key: value for key, value in extra_weight_attrs.items() if key != "weight_loader"}
        for name, param in (("qweight", qweight), ("scales", scales), ("offsets", offsets)):
            layer.register_parameter(name, param)
            set_weight_attrs(param, rest)

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        stash = _stash(layer)
        geometry = stash.dynquant_geometry
        in_features = stash.dynquant_in_features

        # One shard is the overwhelmingly common case (every non-fused layer), and
        # concatenating a single tensor still copies it.
        if len(geometry) == 1:
            out = ops.quantized_matmul(x, _shard_tensor(layer, geometry, 0, in_features))
        else:
            # Not `torch.cat`: the caller splits this straight back apart, and
            # inductor cancels that pair in a way the compiler's graph boundaries
            # do not survive. See `fuse.py` -- the docstring is the whole argument.
            out = fused_shard_concat(
                [
                    ops.quantized_matmul(x, _shard_tensor(layer, geometry, i, in_features))
                    for i in range(len(geometry))
                ]
            )
        if bias is not None:
            out = out + bias
        return out


class DynQuantEmbeddingMethod(DynQuantLinearMethod):
    """Packed vocabulary table, for ``embed_tokens`` and a quantized LM head.

    Deliberately a different parameter *shape* from the linear method: 2-D
    ``[rows, words]`` rather than flat. ``VocabParallelEmbedding.weight_loader``
    is the layer's own -- it never goes through the v2 parameter hooks -- and it
    does ``param[:n].data.copy_(loaded)`` and ``param[n:].data.fill_(0)``
    (``vocab_parallel_embedding.py:501-502``), which only mean the right thing if
    dim 0 is the vocabulary. An embedding is never fused, so the flat layout buys
    nothing here and would cost correctness.

    The zero-fill on the padded tail is why ``offsets`` must exist even for a
    symmetric table: a padded row is filled with zero *quantized codes*, and only
    ``q * scale + offset`` with a zero offset reconstructs to the zero vector. A
    ``None`` offsets buffer would leave those rows holding whatever the symmetric
    decode maps code 0 to.

    :meth:`embedding` must be defined in this class body and not inherited from
    somewhere convenient: ``method_has_implemented_embedding``
    (``base_config.py:260``) compares ``inspect.getattr_static`` of the method
    class against the base, and ``VocabParallelEmbedding.__init__`` raises
    ``NotImplementedError`` when they match.

    Unlike the vLLM twin there is no ``tie_weights`` override, because SGLang
    never calls one. Its ``ParallelLMHead.tie_weights`` (``:654``) is
    ``self.weight = embed_tokens.weight; return self`` -- no ``quant_method`` in
    sight, where vLLM's delegates to the method. That is survivable only because
    the two SGLang models that call it (``granite``, ``zaya``) are not ones we
    serve, and the dense models we do serve tie by assigning the whole module
    (``qwen3.py:488``: ``self.lm_head = self.model.embed_tokens``), which carries
    all three buffers and the stash with it. A tied *quantized* head on a model
    that goes through ``ParallelLMHead.tie_weights`` would silently keep its own
    uninitialised scales; S5's first real serve is on a tied Qwen, so the
    assumption gets exercised rather than assumed.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        skip_block_quant_check: bool = False,
        **extra_weight_attrs: Any,
    ) -> None:
        del input_size, output_size, skip_block_quant_check
        ops.warm_dispatch()
        if len(self.shards) != 1:
            raise DynQuantError(
                f"an embedding table cannot be fused, but {len(self.shards)} shards "
                f"were resolved for it: {[name for name, _ in self.shards]}"
            )
        name, spec = self.shards[0]
        num_rows = sum(output_partition_sizes)
        geom = row_geometry(spec.bits, spec.group_size, input_size_per_partition)

        stash = _stash(layer)
        stash.dynquant_spec = spec
        stash.dynquant_in_features = input_size_per_partition

        qweight = torch.nn.Parameter(
            torch.zeros(num_rows, geom.words_per_row, dtype=torch.int32),
            requires_grad=False,
        )
        scales = torch.nn.Parameter(
            torch.zeros(num_rows, geom.num_groups, dtype=params_dtype), requires_grad=False
        )
        offsets = torch.nn.Parameter(
            torch.zeros(num_rows, geom.num_groups, dtype=params_dtype), requires_grad=False
        )

        for attr, param in (
            ("qweight", qweight),
            ("scales", scales),
            ("offsets", offsets),
        ):
            layer.register_parameter(attr, param)
            # output_dim tells the layer's loader which axis the vocabulary runs
            # along. packed_dim is deliberately omitted: setting it equal to
            # output_dim would make the loader divide vocab offsets by a pack
            # factor, and DynQuant packs along the *input* axis.
            set_weight_attrs(param, {"output_dim": 0, "input_dim": 1})
            set_weight_attrs(param, extra_weight_attrs)
        del name

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return ops.embedding_lookup(_embedding_tensor(layer), input_)

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        return ops.quantized_matmul(x, _embedding_tensor(layer), bias)


# --------------------------------------------------------------------------
# Reconstruction
# --------------------------------------------------------------------------


def _shard_tensor(
    layer: torch.nn.Module, geometry: FusedPackedGeometry, shard_id: int, in_features: int
) -> QuantTensor:
    """A :class:`QuantTensor` viewing one shard of the flat buffers.

    Rebuilt on every forward rather than cached on the layer. The parts that cost
    anything -- the views -- are metadata operations on tensors that already
    exist, so this is a few hundred nanoseconds of Python against a kernel launch,
    and a cache would have to be invalidated on every ``.to()`` and every weight
    update SGLang performs.
    """
    plan = geometry[shard_id]
    stash = _stash(layer)
    return QuantTensor(
        packed=geometry.view_qweight(stash.qweight, shard_id),
        scales=geometry.view_scale(stash.scales, shard_id),
        offsets=geometry.view_scale(stash.offsets, shard_id),
        bits=plan.spec.bits,
        group_size=plan.spec.group_size,
        in_features=in_features,
        logical_shape=(plan.spec.out_features, in_features),
        row_offset=plan.row_offset,
        layout=QuantLayout.LINEAR,
        symmetric=plan.spec.symmetric,
    )


def _embedding_tensor(layer: torch.nn.Module) -> QuantTensor:
    stash = _stash(layer)
    spec = stash.dynquant_spec
    in_features = stash.dynquant_in_features
    return QuantTensor(
        packed=stash.qweight,
        scales=stash.scales,
        offsets=stash.offsets,
        bits=spec.bits,
        group_size=spec.group_size,
        in_features=in_features,
        logical_shape=(stash.qweight.shape[0], in_features),
        layout=QuantLayout.LINEAR,
        symmetric=spec.symmetric,
    )
