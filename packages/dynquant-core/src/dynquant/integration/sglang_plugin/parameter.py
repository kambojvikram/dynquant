"""An SGLang parameter whose rows do not all have the same width.

The vLLM twin is :mod:`dynquant.integration.vllm_plugin.parameter`, and the two
are close because SGLang's ``layers/parameter.py`` is still a fork of vLLM's --
its classes are literally named ``BasevLLMParameter`` and ``_ColumnvLLMParameter``
inside an ``sglang`` module. *What* gets placed is identical and lives in
:mod:`dynquant.integration.serving_common.geometry`, which imports neither
framework. What differs is entirely about where the *source* rows are, and there
are three of them.

**1. ``tp_rank`` is an argument here, not ``self.tp_rank``.** vLLM's
``BasevLLMParameter.__init__`` calls ``get_tensor_model_parallel_rank()`` and
caches the answer; SGLang passes the rank into each placement hook, because its
linear layers accept an explicit ``tp_rank``/``tp_size`` pair
(``linear.py:340-345``, ``:1432-1436``) and a layer's group is not always the
global one -- data-parallel attention is the case that makes this matter. A port
that kept reading a global would be right on rank 0 and wrong everywhere else,
which is the failure mode that looks like a quality regression rather than a bug.

**2. ``use_presharded_weights``.** SGLang can hand a rank a tensor that has
already been sliced for it. Every source offset below then has to be zero, and
the row-parallel word slice has to be skipped -- otherwise a rank slices an
already-sliced tensor and loads a quarter of its own weights.

**3. The base classes are load-bearing.** ``weight_loader_v2`` picks its call
convention by ``isinstance``::

    if isinstance(param, _ColumnvLLMParameter):
        param.load_column_parallel_weight(loaded_weight, tp_rank=..., ...)
    else:
        try:    param.load_column_parallel_weight(loaded_weight, tp_rank=..., ...)
        except TypeError:
            param.load_column_parallel_weight(loaded_weight)   # <- no rank

That ``except TypeError`` (``linear.py:456-466`` and ``:1547-1558``) exists for
foreign parameter classes, and it cannot tell "this parameter does not accept a
rank" from "a TypeError escaped while placing with one". Inheriting from both
``_ColumnvLLMParameter`` and ``RowvLLMParameter`` -- exactly what SGLang's own
``ModelWeightParameter`` does -- takes both ``isinstance`` gates and keeps us out
of that branch. It also supplies the ``output_dim``/``input_dim`` properties,
which is why they are constructor arguments here and hand-written properties in
the vLLM twin.

Notably absent, in both ports: ``packed_dim``. Setting it would make the loaders
divide row offsets by a ``packed_factor``, which is right when packing runs along
the output dimension and wrong here, where it runs along the input dimension.

Everything else is placement. No reshaping, no repacking, no dtype conversion:
the exporter already wrote the exact bytes this buffer holds, so a load is a
strided copy and a shape assertion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from sglang.srt.layers.parameter import (
    RowvLLMParameter,
    _ColumnvLLMParameter,
    copy_with_check,
)

from dynquant.errors import DynQuantError
from dynquant.integration.serving_common.geometry import FusedPackedGeometry, TensorParallelSplit

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from dynquant.integration.serving_common.geometry import ShardPlan

__all__ = ["DynQuantPackedParameter"]


class DynQuantPackedParameter(_ColumnvLLMParameter, RowvLLMParameter):
    """Flat buffer holding one fused layer's packed words, scales or offsets.

    A fused layer is a rectangle for every other quantization method: ``q``, ``k``
    and ``v`` share a width, so they share a words-per-row, so ``[out_total,
    words]`` is well defined and row ``i`` starts at word ``i * words``. Under
    per-module allocation it is not -- ``q_proj`` at 4 bits and ``k_proj`` at 3
    bits have 512- and 384-word rows over the same 4096 inputs, and no ``narrow``
    expresses that. Hence a flat buffer, and hence four overridden hooks that
    translate the *output row ranges* SGLang hands us into flat spans.
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
        # `output_dim` and `input_dim` go to the two base classes, which turn them
        # into properties. Nothing here indexes *this* parameter by them -- the
        # buffer is flat. They describe the tensor coming off disk, which is a
        # normal `[rows, words]` array, and `_load_fused_module_from_checkpoint`
        # (`linear.py:738`, `:1034`) needs `output_dim` to split a checkpoint that
        # was already fused on disk.
        super().__init__(data=data, weight_loader=weight_loader, output_dim=0, input_dim=1)
        if elements_per_row not in ("qweight", "scale"):
            raise DynQuantError(f"unknown buffer kind {elements_per_row!r}")
        self._geometry = geometry
        self._kind = elements_per_row
        self._row_split = row_split

    @property
    def geometry(self) -> FusedPackedGeometry:
        return self._geometry

    # -- the four placement hooks -----------------------------------------

    def load_column_parallel_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
    ) -> None:
        """A plain column-parallel layer: one shard, this rank's slice of rows.

        The signature is copied from ``_ColumnvLLMParameter`` verbatim, including
        ``tp_rank`` having no default. That is deliberate: it is the one thing
        that makes the ``except TypeError`` fallback described in the module
        docstring re-raise instead of quietly loading rank 0's rows.
        """
        rows = self._geometry.total_out_features
        self._place(
            loaded_weight,
            dest_row=0,
            num_rows=rows,
            src_row=0 if use_presharded_weights else tp_rank * rows,
            word_rank=None if use_presharded_weights else tp_rank,
        )

    def load_merged_column_weight(self, loaded_weight: torch.Tensor, **kwargs: Any) -> None:
        """One shard of a ``gate_up_proj``-style merge.

        ``shard_offset`` and ``shard_size`` arrive already divided by ``tp_size``,
        so they are this rank's row range within the fused output; the loaded
        tensor still spans all ranks, hence the ``tp_rank * shard_size`` source
        offset. Both facts are SGLang's convention, not ours -- see
        ``MergedColumnParallelLinear.weight_loader_v2``.

        Subscripted rather than ``.get``-ed, unlike SGLang's own implementation of
        this method, which reaches for ``kwargs.get("tp_rank")`` and then
        multiplies the ``None``. Every call site that can reach *this* class
        passes all four (``linear.py:829``, ``:874``, ``:909``); the two that omit
        them are guarded by ``type(param) in (RowvLLMParameter,
        BasevLLMParameter)``, an exact-type test we do not satisfy. So a missing
        key means the convention moved, and ``KeyError: 'tp_rank'`` says which.
        """
        shard_offset = int(kwargs["shard_offset"])
        shard_size = int(kwargs["shard_size"])
        tp_rank = int(kwargs["tp_rank"])
        presharded = bool(kwargs.get("use_presharded_weights", False))
        self._place(
            loaded_weight,
            dest_row=shard_offset,
            num_rows=shard_size,
            src_row=0 if presharded else tp_rank * shard_size,
            word_rank=None if presharded else tp_rank,
        )

    def load_qkv_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
        **kwargs: Any,
    ) -> None:
        """One of q/k/v.

        Differs from the merged case only in the source offset: with fewer KV
        heads than query heads, several ranks read the *same* K and V rows, so the
        source index is ``tp_rank // num_kv_head_replicas`` rather than
        ``tp_rank``.

        ``num_heads`` is SGLang's name for ``num_kv_head_replicas``, which is a
        ``divide()`` result and so never zero (``linear.py:979-984``) -- the
        division below needs no guard.
        """
        shard_offset = int(kwargs["shard_offset"])
        shard_size = int(kwargs["shard_size"])
        shard_id = kwargs["shard_id"]
        num_heads = int(kwargs["num_heads"])
        src_index = tp_rank if shard_id == "q" else tp_rank // num_heads
        self._place(
            loaded_weight,
            dest_row=shard_offset,
            num_rows=shard_size,
            src_row=0 if use_presharded_weights else src_index * shard_size,
            word_rank=None if use_presharded_weights else tp_rank,
        )

    def load_row_parallel_weight(
        self,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        use_presharded_weights: bool = False,
    ) -> None:
        """A row-parallel layer: all rows, this rank's slice of the input.

        The input dimension is what got packed, so "this rank's slice of the
        input" is a slice of the *word* axis -- and it is only a clean slice
        because :func:`~dynquant.integration.serving_common.geometry.row_parallel_split`
        refused the layer at ``create_weights`` time otherwise.
        """
        rows = self._geometry.total_out_features
        self._place(
            loaded_weight,
            dest_row=0,
            num_rows=rows,
            src_row=0,
            word_rank=None if use_presharded_weights else tp_rank,
        )

    # -- placement ---------------------------------------------------------

    def _place(
        self,
        loaded_weight: torch.Tensor,
        *,
        dest_row: int,
        num_rows: int,
        src_row: int,
        word_rank: int | None,
    ) -> None:
        """Copy ``num_rows`` rows into the flat span that backs them.

        ``word_rank`` is the rank whose word slice to take off a row-parallel
        checkpoint tensor, or ``None`` for "take the whole row" -- which covers
        both the column-parallel layers, where there is no row split at all, and
        the presharded case, where the slice was taken before we saw the tensor.
        Spelling it as one nullable argument rather than passing ``tp_rank`` and a
        flag keeps the two ways of meaning "do not slice" from disagreeing.
        """
        plan = self._plan_for_rows(dest_row, num_rows)
        per_row = plan.words_per_row if self._kind == "qweight" else plan.num_groups
        base = plan.qweight_offset if self._kind == "qweight" else plan.scale_offset
        start = base + (dest_row - plan.row_offset) * per_row

        dest = self.data[start : start + num_rows * per_row].view(num_rows, per_row)
        source = loaded_weight.narrow(0, src_row, num_rows)
        if self._row_split is not None and word_rank is not None:
            column = (
                self._row_split.word_slice(word_rank)
                if self._kind == "qweight"
                else self._row_split.group_slice(word_rank)
            )
            source = source[:, column]

        if source.shape != dest.shape:
            raise DynQuantError(
                f"{plan.spec.name}: checkpoint tensor gives {tuple(source.shape)} for this "
                f"rank but the config says {plan.spec.bits}-bit at group_size="
                f"{plan.spec.group_size}, which needs {tuple(dest.shape)}. The "
                f"quantization_config and the weights were written by different runs."
            )
        # SGLang's own copy, not `dest.copy_(source)`, for the dtype half of it:
        # `copy_with_check` refuses a silent downcast unless
        # SGLANG_QUANT_ALLOW_DOWNCASTING is set, so fp32 scales landing in an fp16
        # layer raise instead of quietly losing the exponent range that made a
        # 2-bit group's scale meaningful. Its shape assertion is redundant after
        # the check above, which is the point -- ours names the module.
        copy_with_check(dest, source)

    def _plan_for_rows(self, dest_row: int, num_rows: int) -> ShardPlan:
        """The shard owning ``[dest_row, dest_row + num_rows)``.

        Delegated to the geometry, which is the framework-free half and so the
        half whose arithmetic gets checked without a GPU. The lookup is not a
        formality: under the run mapping one shard backs several of SGLang's
        output partitions, so most placements name a sub-range of a shard rather
        than the whole of one.
        """
        return self._geometry.plan_for_rows(dest_row, num_rows)
