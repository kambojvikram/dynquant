"""Where each shard's packed words live inside one vLLM parameter.

The problem this module exists to solve
---------------------------------------
vLLM fuses projections that the checkpoint stores separately: ``q_proj``,
``k_proj`` and ``v_proj`` become one ``qkv_proj`` layer holding one parameter,
and ``gate_proj``/``up_proj`` become one ``gate_up_proj``. Every other
quantization method gets away with this because its bit width is a property of
the *checkpoint*, so all three shards have identical rows and the fused parameter
is a plain concatenation along the output dimension.

DynQuant allocates width per module. ``q_proj`` at 4 bits and ``k_proj`` at 3
bits produce rows of *different word counts* -- 4-bit rows of 4096 inputs are 512
words, 3-bit rows are 384 -- so there is no rectangle that holds both. That is
the whole difficulty, and it is not incidental: per-module width is the method.

The layout
----------
One flat 1-D ``qweight`` buffer holding each shard's packed rows end to end, in
the layer's own shard order::

    [ shard 0: out_0 * words_0 words ][ shard 1: out_1 * words_1 ] ...

with ``scales`` and ``offsets`` laid out the same way over ``out_i * groups_i``.
Each shard's span is contiguous, so ``flat[a:b].view(out_i, words_i)`` is a
*view* -- no copy, no extra VRAM -- and reconstructing a
:class:`~dynquant.quant.tensor.QuantTensor` per shard at ``apply`` time is free.

Flat rather than padded-to-a-common-width because padding is not free: padding
``k_proj``'s 384-word rows out to ``q_proj``'s 512 wastes a quarter of the
projection's VRAM, and the kernel would still need a per-shard launch to know
which shift table to use. The launches are unavoidable; the waste is not.

Nothing here imports vLLM. Every size is resolved through
:func:`dynquant.quant.pack.row_geometry` -- the same resolver the encoder and
:meth:`QuantTensor.validate` use -- so a disagreement between what the exporter
wrote and what the loader expects is not expressible. This module is therefore
importable and testable on a laptop with no GPU and no vLLM, which is where its
arithmetic actually gets checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from dynquant.errors import DynQuantError
from dynquant.integration.vllm_plugin.schema import ModuleQuantSpec
from dynquant.quant.pack import RowGeometry, row_geometry

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

__all__ = [
    "FusedPackedGeometry",
    "ShardPlan",
    "ShardSpec",
    "TensorParallelSplit",
    "match_shards_to_partitions",
    "row_parallel_split",
]


@dataclass(frozen=True, slots=True)
class ShardSpec:
    """One checkpoint module that vLLM has folded into a fused layer.

    A layer that is not fused has exactly one of these, and everything below
    degenerates to "the whole buffer is shard 0" -- deliberately, so there is no
    second code path whose divergence from this one has to be maintained.

    Attributes:
        name: The checkpoint module name, e.g. ``model.layers.0.self_attn.k_proj``.
            Carried so error messages can name the tensor a user has to go fix,
            and so per-shard widths can be looked up by the same key the exporter
            wrote.
        bits: This shard's width. The reason this class exists.
        group_size: As stored, so possibly
            :data:`~dynquant.constants.PER_ROW_GROUP_SIZE`.
        out_features: Rows **on this tensor-parallel rank**, i.e. already divided
            by ``tp_size``. vLLM hands ``create_weights`` the per-partition
            output sizes, so taking them pre-divided keeps the division in one
            place instead of two.
        in_features: Reduction length **on this rank**, likewise already divided
            for a row-parallel layer.
        symmetric: Whether this module was encoded without offsets. Nothing in
            the layout depends on it -- the offsets buffer is allocated either
            way -- but the reconstruction at ``apply`` time does, and this class
            is the only per-shard record the layer keeps. Required rather than
            defaulted so that a call site building a spec from a checkpoint
            cannot quietly drop it and get asymmetric decoding of a symmetric
            table.
    """

    name: str
    bits: int
    group_size: int
    out_features: int
    in_features: int
    symmetric: bool


def match_shards_to_partitions(
    shards: Sequence[tuple[str, ModuleQuantSpec]],
    output_partition_sizes: Sequence[int],
    in_features: int,
) -> list[ShardSpec]:
    """Line the checkpoint's modules up against vLLM's output partitions.

    Two cases, and the difference is whether the checkpoint stores the shards
    separately:

    * ``len(shards) == len(output_partition_sizes)`` -- the ordinary case. vLLM's
      ``packed_modules_mapping`` lists the sub-modules in the same order it
      concatenates their outputs, so the pairing is positional.
    * ``len(shards) == 1`` with several partitions -- the checkpoint was already
      fused on disk (Phi-3's ``qkv_proj``). There is one width for the whole
      matrix, so the partitions collapse into one shard and vLLM's
      ``_load_fused_module_from_checkpoint`` splits the loaded tensor by rows,
      which the flat layout handles because a uniform-width shard has a constant
      words-per-row.

    Anything else means the config and the model disagree about the layer's
    structure, and continuing would place weights at plausible wrong offsets.
    """
    if len(shards) == len(output_partition_sizes):
        return [
            ShardSpec(
                name=name,
                bits=spec.bits,
                group_size=spec.group_size,
                out_features=out,
                in_features=in_features,
                symmetric=spec.symmetric,
            )
            for (name, spec), out in zip(shards, output_partition_sizes)
        ]

    if len(shards) == 1:
        name, spec = shards[0]
        return [
            ShardSpec(
                name=name,
                bits=spec.bits,
                group_size=spec.group_size,
                out_features=sum(output_partition_sizes),
                in_features=in_features,
                symmetric=spec.symmetric,
            )
        ]

    raise DynQuantError(
        f"this layer has {len(output_partition_sizes)} output partitions "
        f"{list(output_partition_sizes)} but the quantization map resolved "
        f"{len(shards)} modules {[name for name, _ in shards]}. The checkpoint's "
        f"quantization_config does not describe this model."
    )


@dataclass(frozen=True, slots=True)
class ShardPlan:
    """Where one shard's words, scales and offsets live in the flat buffers.

    Offsets are in elements of the buffer they index, not bytes: ``qweight``
    offsets count int32 words, ``scale`` offsets count scale entries.
    """

    spec: ShardSpec
    row: RowGeometry
    shard_id: int
    row_offset: int
    """First row of this shard in the fused layer's output space, before
    tensor-parallel narrowing. Recorded on the reconstructed ``QuantTensor`` so a
    kernel that cares which slice of the output it is writing can ask."""

    qweight_offset: int
    scale_offset: int

    @property
    def words_per_row(self) -> int:
        return self.row.words_per_row

    @property
    def num_groups(self) -> int:
        return self.row.num_groups

    @property
    def qweight_numel(self) -> int:
        return self.spec.out_features * self.row.words_per_row

    @property
    def scale_numel(self) -> int:
        return self.spec.out_features * self.row.num_groups

    @property
    def qweight_slice(self) -> slice:
        return slice(self.qweight_offset, self.qweight_offset + self.qweight_numel)

    @property
    def scale_slice(self) -> slice:
        return slice(self.scale_offset, self.scale_offset + self.scale_numel)

    @property
    def qweight_shape(self) -> tuple[int, int]:
        return (self.spec.out_features, self.row.words_per_row)

    @property
    def scale_shape(self) -> tuple[int, int]:
        return (self.spec.out_features, self.row.num_groups)


class FusedPackedGeometry:
    """The flat layout of every shard in one vLLM linear layer.

    Built once in ``create_weights`` and kept on the layer, because every later
    operation -- placing a loaded shard, viewing a shard at ``apply`` time,
    reporting a size -- has to agree about the same offsets, and recomputing them
    at each site is how they stop agreeing.
    """

    __slots__ = ("_by_name", "_shards", "qweight_numel", "scale_numel")

    def __init__(self, specs: Sequence[ShardSpec]) -> None:
        if not specs:
            raise DynQuantError("a fused layer needs at least one shard")

        shards: list[ShardPlan] = []
        qweight_offset = 0
        scale_offset = 0
        row_offset = 0
        for shard_id, spec in enumerate(specs):
            if spec.out_features <= 0:
                raise DynQuantError(
                    f"{spec.name}: out_features must be positive, got {spec.out_features}"
                )
            # Validates bits, group size and alignment as a side effect, and does
            # it here rather than at first use so a malformed checkpoint fails
            # while the layer is being built and the error can still name it.
            row = row_geometry(spec.bits, spec.group_size, spec.in_features)
            plan = ShardPlan(
                spec=spec,
                row=row,
                shard_id=shard_id,
                row_offset=row_offset,
                qweight_offset=qweight_offset,
                scale_offset=scale_offset,
            )
            shards.append(plan)
            qweight_offset += plan.qweight_numel
            scale_offset += plan.scale_numel
            row_offset += spec.out_features

        self._shards: tuple[ShardPlan, ...] = tuple(shards)
        self._by_name = {plan.spec.name: plan for plan in shards}
        self.qweight_numel = qweight_offset
        self.scale_numel = scale_offset

    # -- access ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._shards)

    def __iter__(self):
        return iter(self._shards)

    def __getitem__(self, shard_id: int) -> ShardPlan:
        return self._shards[shard_id]

    @property
    def shards(self) -> tuple[ShardPlan, ...]:
        return self._shards

    def by_name(self, name: str) -> ShardPlan:
        try:
            return self._by_name[name]
        except KeyError:
            raise DynQuantError(
                f"{name} is not a shard of this layer; it holds "
                f"{[s.spec.name for s in self._shards]}"
            ) from None

    @property
    def total_out_features(self) -> int:
        return sum(plan.spec.out_features for plan in self._shards)

    @property
    def is_uniform(self) -> bool:
        """Every shard at the same width and grouping.

        True for the overwhelming majority of layers even under per-module
        allocation -- the allocator usually gives ``q``/``k``/``v`` the same
        width -- and a caller that wants to skip the per-shard loop and issue one
        kernel over the whole buffer can check this. Nothing here requires it.
        """
        first = self._shards[0].spec
        return all(
            plan.spec.bits == first.bits and plan.spec.group_size == first.group_size
            for plan in self._shards
        )

    # -- views -------------------------------------------------------------

    def view_qweight(self, flat: "torch.Tensor", shard_id: int) -> "torch.Tensor":
        """``[out_i, words_i]`` view into the flat packed buffer. No copy."""
        plan = self._shards[shard_id]
        self._check_flat(flat, self.qweight_numel, "qweight")
        return flat[plan.qweight_slice].view(plan.qweight_shape)

    def view_scale(self, flat: "torch.Tensor", shard_id: int) -> "torch.Tensor":
        """``[out_i, groups_i]`` view into a flat scales/offsets buffer."""
        plan = self._shards[shard_id]
        self._check_flat(flat, self.scale_numel, "scales/offsets")
        return flat[plan.scale_slice].view(plan.scale_shape)

    def _check_flat(self, flat: "torch.Tensor", expected: int, what: str) -> None:
        if flat.dim() != 1 or flat.numel() != expected:
            raise DynQuantError(
                f"{what} buffer should be 1-D of {expected} elements for this layer, "
                f"got shape {tuple(flat.shape)}"
            )


@dataclass(frozen=True, slots=True)
class TensorParallelSplit:
    """How one row-parallel shard divides across ranks, in packed units.

    Row-parallel layers (``o_proj``, ``down_proj``) split the *reduction*
    dimension, which for a packed tensor means splitting the word axis. That is
    only expressible if each rank's slice starts on a group boundary -- otherwise
    a rank would own a fraction of a group, whose scale belongs to a neighbour
    and whose first value starts mid-word. Rather than let that produce quietly
    wrong numbers, :func:`row_parallel_split` refuses.
    """

    in_features_per_partition: int
    groups_per_partition: int
    words_per_partition: int

    def word_slice(self, tp_rank: int) -> slice:
        start = tp_rank * self.words_per_partition
        return slice(start, start + self.words_per_partition)

    def group_slice(self, tp_rank: int) -> slice:
        start = tp_rank * self.groups_per_partition
        return slice(start, start + self.groups_per_partition)


def row_parallel_split(
    *,
    bits: int,
    group_size: int,
    in_features: int,
    tp_size: int,
    name: str = "<layer>",
) -> TensorParallelSplit:
    """Divide a packed row across tensor-parallel ranks, or explain why not.

    Args:
        in_features: The **full** reduction length as stored in the checkpoint.
        tp_size: Number of tensor-parallel ranks.
        name: Used only in the error message.

    Raises:
        DynQuantError: If the split would cut a quantization group. The message
            names the constraint in terms the user can act on -- a different
            ``--group-size`` at quantization time, or a different
            ``--tensor-parallel-size`` -- because the alternative is a model that
            loads and produces plausible garbage.
    """
    full = row_geometry(bits, group_size, in_features)
    if tp_size <= 0:
        raise DynQuantError(f"tp_size must be positive, got {tp_size}")
    if tp_size == 1:
        return TensorParallelSplit(
            in_features_per_partition=in_features,
            groups_per_partition=full.num_groups,
            words_per_partition=full.words_per_row,
        )

    if full.is_per_row:
        raise DynQuantError(
            f"{name} is packed per-row (one group spanning all {in_features} inputs) "
            f"and cannot be split across {tp_size} tensor-parallel ranks: each rank "
            f"would own part of a group whose single scale it does not have. Quantize "
            f"this module with an explicit --group-size, or serve it at "
            f"--tensor-parallel-size 1."
        )

    if in_features % tp_size != 0:
        raise DynQuantError(
            f"{name}: in_features={in_features} is not divisible by "
            f"tensor-parallel size {tp_size}"
        )
    per_partition = in_features // tp_size
    if per_partition % full.effective_group != 0:
        raise DynQuantError(
            f"{name}: at --tensor-parallel-size {tp_size} each rank gets "
            f"{per_partition} of {in_features} inputs, which is not a whole number "
            f"of {full.effective_group}-value quantization groups. A rank would own "
            f"a fraction of a group and none of its scale. Re-quantize with a "
            f"group_size dividing {per_partition}, or use a tensor-parallel size "
            f"dividing {in_features // full.effective_group}."
        )

    part = row_geometry(bits, group_size, per_partition)
    # Padding at the tail of the full row would make the parts not tile it. It
    # cannot happen here -- divisibility above rules it out -- but the packed
    # offsets below are only meaningful if the parts tile exactly, so it is
    # asserted rather than assumed.
    if part.words_per_row * tp_size != full.words_per_row:
        raise DynQuantError(
            f"{name}: {tp_size} partitions of {part.words_per_row} words do not tile "
            f"the row's {full.words_per_row} words"
        )
    return TensorParallelSplit(
        in_features_per_partition=per_partition,
        groups_per_partition=part.num_groups,
        words_per_partition=part.words_per_row,
    )
