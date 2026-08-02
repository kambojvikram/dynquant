"""A vLLM parameter whose rows do not all have the same width.

vLLM's stock parameter classes place a loaded shard with
``param.data.narrow(output_dim, offset, size)``. That works because for every
other quantization method a fused layer is a rectangle: ``q``, ``k`` and ``v``
have the same bits, so the same number of packed words per row, so
``[out_total, words]`` is well defined and row ``i`` starts at word ``i * words``.

Under per-module allocation it is not. ``q_proj`` at 4 bits and ``k_proj`` at 3
bits have 512- and 384-word rows over the same 4096 inputs, and no ``narrow``
expresses that. So the buffer is flat (see
:mod:`~dynquant.integration.vllm_plugin.geometry`) and this class overrides the
four placement hooks to translate *output row ranges* -- which is what vLLM
actually hands us, in every one of the four -- into flat spans.

Everything here is placement. There is no reshaping, no repacking and no dtype
conversion: the exporter already wrote the exact bytes this buffer holds, so a
load is a strided copy and a shape assertion. If the assertion fires, the
checkpoint and the config disagree about a module's width, and that is worth a
crash rather than a silent 512-vs-384 mismatch producing noise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from vllm.model_executor.parameter import BasevLLMParameter

from dynquant.errors import DynQuantError
from dynquant.integration.vllm_plugin.geometry import FusedPackedGeometry, TensorParallelSplit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from dynquant.integration.vllm_plugin.geometry import ShardPlan

__all__ = ["DynQuantPackedParameter"]


class DynQuantPackedParameter(BasevLLMParameter):
    """Flat buffer holding one fused layer's packed words, scales or offsets.

    ``output_dim`` and ``input_dim`` are declared even though nothing here uses
    them to index this parameter: vLLM reads them off the parameter to slice the
    *loaded* tensor, which is a normal 2-D ``[rows, words]`` array, and
    ``MergedColumnParallelLinear._load_fused_module_from_checkpoint`` needs
    ``output_dim`` to split a checkpoint that was already fused on disk. Notably
    absent is ``packed_dim``: vLLM would then divide row offsets by a
    ``packed_factor``, which is right when packing runs along the output
    dimension and wrong here, where it runs along the input dimension. Leaving it
    unset is what keeps those adjustment branches switched off.
    """

    def __new__(cls, data: torch.Tensor, **kwargs: Any) -> DynQuantPackedParameter:
        return cast(DynQuantPackedParameter, super().__new__(cls, data=data, **kwargs))

    def __init__(
        self,
        *,
        data: torch.Tensor,
        weight_loader: Callable[..., None] | None,
        geometry: FusedPackedGeometry,
        elements_per_row: str,
        row_split: TensorParallelSplit | None = None,
    ) -> None:
        """
        Args:
            geometry: The layer's flat layout. Shared by reference with the
                layer's other parameters, so words and scales can never be placed
                against different views of the same checkpoint.
            elements_per_row: ``"qweight"`` or ``"scale"`` -- which of the two
                per-row strides in the geometry this buffer uses.
            row_split: Set only for a row-parallel layer, where the checkpoint
                tensor spans all ranks and this rank owns a slice of the word
                axis. ``None`` means the loaded tensor's rows are already this
                rank's rows.
        """
        super().__init__(data=data, weight_loader=weight_loader)
        if elements_per_row not in ("qweight", "scale"):
            raise DynQuantError(f"unknown buffer kind {elements_per_row!r}")
        self._geometry = geometry
        self._kind = elements_per_row
        self._row_split = row_split

    # -- what vLLM reads off us -------------------------------------------

    @property
    def output_dim(self) -> int:
        return 0

    @property
    def input_dim(self) -> int:
        return 1

    @property
    def geometry(self) -> FusedPackedGeometry:
        return self._geometry

    # -- the four placement hooks -----------------------------------------

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
        """A plain column-parallel layer: one shard, this rank's slice of rows."""
        rows = self._geometry.total_out_features
        self._place(loaded_weight, dest_row=0, num_rows=rows, src_row=self.tp_rank * rows)

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs: Any) -> None:
        """One shard of a ``gate_up_proj``-style merge.

        ``shard_offset`` and ``shard_size`` arrive already divided by ``tp_size``,
        so they are this rank's row range within the fused output; the loaded
        tensor still spans all ranks, hence the ``tp_rank * shard_size`` source
        offset. Both facts are vLLM's convention, not ours -- see
        ``MergedColumnParallelLinear.weight_loader_v2``.
        """
        shard_offset = int(kwargs["shard_offset"])
        shard_size = int(kwargs["shard_size"])
        self._place(
            loaded_weight,
            dest_row=shard_offset,
            num_rows=shard_size,
            src_row=self.tp_rank * shard_size,
        )

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs: Any) -> None:
        """One of q/k/v.

        Differs from the merged case only in the source offset: with fewer KV
        heads than query heads, several ranks read the *same* K and V rows, so the
        source index is ``tp_rank // num_kv_head_replicas`` rather than
        ``tp_rank``.
        """
        shard_offset = int(kwargs["shard_offset"])
        shard_size = int(kwargs["shard_size"])
        shard_id = kwargs["shard_id"]
        num_heads = int(kwargs.get("num_heads", 1)) or 1
        src_index = self.tp_rank if shard_id == "q" else self.tp_rank // num_heads
        self._place(
            loaded_weight,
            dest_row=shard_offset,
            num_rows=shard_size,
            src_row=src_index * shard_size,
        )

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor) -> None:
        """A row-parallel layer: all rows, this rank's slice of the input.

        The input dimension is what got packed, so "this rank's slice of the
        input" is a slice of the *word* axis -- and it is only a clean slice
        because :func:`~dynquant.integration.vllm_plugin.geometry.row_parallel_split`
        refused the layer at ``create_weights`` time otherwise.
        """
        rows = self._geometry.total_out_features
        self._place(loaded_weight, dest_row=0, num_rows=rows, src_row=0)

    # -- placement ---------------------------------------------------------

    def _place(
        self, loaded_weight: torch.Tensor, *, dest_row: int, num_rows: int, src_row: int
    ) -> None:
        plan = self._plan_for_rows(dest_row, num_rows)
        per_row = plan.words_per_row if self._kind == "qweight" else plan.num_groups
        base = plan.qweight_offset if self._kind == "qweight" else plan.scale_offset
        start = base + (dest_row - plan.row_offset) * per_row

        dest = self.data[start : start + num_rows * per_row].view(num_rows, per_row)
        source = loaded_weight.narrow(0, src_row, num_rows)
        if self._row_split is not None:
            column = (
                self._row_split.word_slice(self.tp_rank)
                if self._kind == "qweight"
                else self._row_split.group_slice(self.tp_rank)
            )
            source = source[:, column]

        if source.shape != dest.shape:
            raise DynQuantError(
                f"{plan.spec.name}: checkpoint tensor gives {tuple(source.shape)} for this "
                f"rank but the config says {plan.spec.bits}-bit at group_size="
                f"{plan.spec.group_size}, which needs {tuple(dest.shape)}. The "
                f"quantization_config and the weights were written by different runs."
            )
        dest.copy_(source)

    def _plan_for_rows(self, dest_row: int, num_rows: int) -> ShardPlan:
        """The shard owning ``[dest_row, dest_row + num_rows)``.

        Delegated to the geometry, which is the vLLM-free half and so the half
        whose arithmetic gets checked without a GPU. The lookup is not a
        formality: under the run mapping one shard backs several of vLLM's output
        partitions, so most placements name a sub-range of a shard rather than
        the whole of one.
        """
        return self._geometry.plan_for_rows(dest_row, num_rows)
