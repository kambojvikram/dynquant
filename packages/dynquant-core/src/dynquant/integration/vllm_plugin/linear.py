"""The two ``QuantizeMethodBase`` implementations vLLM builds layers from.

Both go through :mod:`dynquant.runtime.ops` rather than calling
``torch.ops.dynquant.*`` directly, so ``DYNQUANT_BACKEND=torch`` still selects
the reference implementation *inside vLLM*. That is what makes "does the served
model match the packed weights" a question anyone can answer on the machine that
is serving, instead of one that needs a second harness.
"""

from __future__ import annotations

import torch
from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    register_weight_loader_v2_supported_method,
)
from vllm.model_executor.utils import set_weight_attrs

from dynquant.errors import DynQuantError
from dynquant.integration.vllm_plugin.fuse import fused_shard_concat
from dynquant.integration.vllm_plugin.geometry import (
    FusedPackedGeometry,
    match_shards_to_partitions,
    row_parallel_split,
)
from dynquant.integration.vllm_plugin.parameter import DynQuantPackedParameter
from dynquant.integration.vllm_plugin.schema import ModuleQuantSpec
from dynquant.quant.pack import row_geometry
from dynquant.quant.tensor import QuantLayout, QuantTensor
from dynquant.runtime import ops

__all__ = ["DynQuantEmbeddingMethod", "DynQuantLinearMethod"]


@register_weight_loader_v2_supported_method
class DynQuantLinearMethod(LinearMethodBase):
    """A linear layer whose fused shards may each have their own bit width.

    The decorator is not optional. ``ColumnParallelLinear.__init__`` picks between
    the v1 and v2 weight loaders by testing the method class *name* against a
    module-level list, and v1 places a shard with ``param.data.narrow(output_dim,
    ...)`` -- which on a flat buffer narrows into the wrong rows and copies
    silently, because the shapes happen to be compatible. Registering here is the
    supported way for an out-of-tree method to opt in, and it is the difference
    between a correct model and a plausible one.
    """

    def __init__(self, quant_config, shards: list[tuple[str, ModuleQuantSpec]]) -> None:
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
        **extra_weight_attrs,
    ) -> None:
        del output_size  # implied by output_partition_sizes and the tp size
        # vLLM compiles the model with `fullgraph`, so the backend probe has to
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
        # illegal --tensor-parallel-size fails at startup with the module named,
        # rather than after the weights are on the GPU.
        split = None
        tp_in = input_size // input_size_per_partition if input_size_per_partition else 1
        if tp_in > 1:
            if len(specs) != 1:
                raise DynQuantError(
                    f"row-parallel layer {[s.name for s in specs]} has more than one "
                    f"fused shard. Each width implies its own words-per-rank, so the "
                    f"word axis has no single split. vLLM does not build such a layer, "
                    f"so this means the quantization map does not describe this model."
                )
            split = row_parallel_split(
                bits=specs[0].bits,
                group_size=specs[0].group_size,
                in_features=input_size,
                tp_size=tp_in,
                name=specs[0].name,
            )

        layer.dynquant_geometry = geometry
        layer.dynquant_row_split = split
        # Read back in apply(). Not `layer.input_size_per_partition`, which
        # ReplicatedLinear does not define.
        layer.dynquant_in_features = input_size_per_partition

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
        # it, and `make` above passes it. vLLM's `set_weight_attrs` asserts
        # `not hasattr` rather than overwriting, so handing it the whole dict
        # raises `Overwriting existing tensor attribute: weight_loader` while the
        # layer is being built -- before a single weight has been read.
        rest = {key: value for key, value in extra_weight_attrs.items() if key != "weight_loader"}
        for name, param in (("qweight", qweight), ("scales", scales), ("offsets", offsets)):
            layer.register_parameter(name, param)
            set_weight_attrs(param, rest)

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        geometry: FusedPackedGeometry = layer.dynquant_geometry
        in_features: int = layer.dynquant_in_features

        # One shard is the overwhelmingly common case (every non-fused layer), and
        # concatenating a single tensor still copies it.
        if len(geometry) == 1:
            out = ops.quantized_matmul(x, _shard_tensor(layer, geometry, 0, in_features))
        else:
            # Not `torch.cat`: the caller splits this straight back apart, and
            # inductor cancels that pair in a way vLLM's piecewise boundaries do
            # not survive. See `fuse.py` -- the docstring is the whole argument.
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
    does ``param[:n].copy_(loaded)`` and ``param[n:].fill_(0)``, which only mean
    the right thing if dim 0 is the vocabulary. An embedding is never fused, so
    the flat layout buys nothing here and would cost correctness.

    The zero-fill on the padded tail is why ``offsets`` must exist even for a
    symmetric table: a padded row is filled with zero *quantized codes*, and only
    ``q * scale + offset`` with a zero offset reconstructs to the zero vector. A
    ``None`` offsets buffer would leave those rows holding whatever the symmetric
    decode maps code 0 to.
    """

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        ops.warm_dispatch()
        if len(self.shards) != 1:
            raise DynQuantError(
                f"an embedding table cannot be fused, but {len(self.shards)} shards "
                f"were resolved for it: {[name for name, _ in self.shards]}"
            )
        name, spec = self.shards[0]
        num_rows = sum(output_partition_sizes)
        geom = row_geometry(spec.bits, spec.group_size, input_size_per_partition)

        layer.dynquant_spec = spec
        layer.dynquant_in_features = input_size_per_partition

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

    def tie_weights(self, layer: torch.nn.Module, embed_tokens: torch.nn.Module):
        """Share the packed table with ``embed_tokens`` -- all three buffers.

        The base implementation assigns ``layer.weight`` and stops, which for a
        packed table would leave the LM head with its own uninitialised scales
        and offsets while appearing to be tied. On a tied model the table is
        typically the single largest tensor -- 27% of a 2B Qwen -- so getting
        this wrong doubles it as well as corrupting it.
        """
        for attr in ("qweight", "scales", "offsets"):
            setattr(layer, attr, getattr(embed_tokens, attr))
        layer.dynquant_spec = embed_tokens.dynquant_spec
        layer.dynquant_in_features = embed_tokens.dynquant_in_features
        return layer


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
    and a cache would have to be invalidated on every ``.to()``, every sleep/wake
    cycle and every weight refit that vLLM performs.
    """
    plan = geometry[shard_id]
    return QuantTensor(
        packed=geometry.view_qweight(layer.qweight, shard_id),
        scales=geometry.view_scale(layer.scales, shard_id),
        offsets=geometry.view_scale(layer.offsets, shard_id),
        bits=plan.spec.bits,
        group_size=plan.spec.group_size,
        in_features=in_features,
        logical_shape=(plan.spec.out_features, in_features),
        row_offset=plan.row_offset,
        layout=QuantLayout.LINEAR,
        symmetric=plan.spec.symmetric,
    )


def _embedding_tensor(layer: torch.nn.Module) -> QuantTensor:
    spec: ModuleQuantSpec = layer.dynquant_spec
    in_features: int = layer.dynquant_in_features
    return QuantTensor(
        packed=layer.qweight,
        scales=layer.scales,
        offsets=layer.offsets,
        bits=spec.bits,
        group_size=spec.group_size,
        in_features=in_features,
        logical_shape=(layer.qweight.shape[0], in_features),
        layout=QuantLayout.LINEAR,
        symmetric=spec.symmetric,
    )


