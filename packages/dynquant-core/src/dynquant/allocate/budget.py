"""Honest budget accounting.

The supplement budgeted against packed weights only. At group size 128 with
asymmetric quantization that omits an fp16 scale and an fp16 offset per 128
weights -- 0.25 bits per weight -- so its "3-bit" checkpoint was 3.25 bits on disk,
and the gap grows as the group shrinks. Users compare a quantization method by the
size of the folder it produced, so the number the allocator targets has to be the
number the filesystem reports.

Everything here is therefore counted in *stored* bits: packed payload, scales,
offsets, and the tensors that stay at compute dtype -- including the ones the graph
refused, which stay at compute dtype without ever having been a decision. On
LFM2.5-8B-A1B those are 205,056 bytes of norms and biases against 4.4 GB, and on a MoE
whose expert banks are refused they are most of the model.

One term is named and deliberately not priced: the safetensors container itself, 58,880
bytes on that same model. Its size is a function of the tensor *names* and offsets, which
do not exist until the allocation this budget is being computed for has been made. A
target is therefore hit to within a header, and the header is the only thing between the
number asked for and the number on disk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dynquant.constants import DEFAULT_GROUP_SIZE
from dynquant.graph.roles import UNQUANTIZED_FLOOR
from dynquant.quant.pack import stored_bits

if TYPE_CHECKING:
    from dynquant.graph.classify import ModelGraph, ModuleInfo

__all__ = ["Budget", "module_stored_bits", "parse_size"]

_SIZE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([KMGT]?)(i?)B?\s*$", re.IGNORECASE)
_UNIT = {"": 1, "K": 1_000, "M": 1_000**2, "G": 1_000**3, "T": 1_000**4}
_UNIT_BINARY = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def parse_size(text: str) -> int:
    """Parse ``6.5GiB`` / ``6.5GB`` / ``700MB`` into bytes.

    ``GiB`` and ``GB`` are kept distinct because the 7% between them is larger
    than the accuracy difference between adjacent bit maps -- rounding them
    together would make a target the user can hit look like one they cannot.
    """
    match = _SIZE.match(text)
    if match is None:
        raise ValueError(f"cannot parse size {text!r}; expected something like '6.5GiB' or '700MB'")
    amount, unit, binary = match.groups()
    table = _UNIT_BINARY if binary else _UNIT
    return int(float(amount) * table[unit.upper()])


def module_stored_bits(
    info: ModuleInfo,
    bits: int,
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    symmetric: bool = False,
    metadata_bits: int = 16,
) -> float:
    """Bits this module occupies on disk at ``bits``, metadata included.

    ``bits == UNQUANTIZED_FLOOR`` means the tensor is not quantized at all and is
    stored at compute dtype -- no groups, no scales, just the weights.

    Delegates the geometry to :func:`~dynquant.quant.pack.stored_bits` rather than
    recomputing it, because this function and the packer disagreeing is not a
    theoretical risk: it computed ``params * bits`` for the payload, which silently
    drops the zero-padded tail group that the packer does store.
    """
    params = info.num_params
    if bits >= UNQUANTIZED_FLOOR:
        return float(params * UNQUANTIZED_FLOOR)

    shape = info.shape
    if len(shape) < 2:  # pragma: no cover -- rank-1 tensors are never quantizable
        return float(params * bits)

    columns = shape[-1]
    rows = params // columns if columns else params
    return float(
        stored_bits(
            rows,
            columns,
            bits,
            group_size=group_size,
            symmetric=symmetric,
            metadata_bits=metadata_bits,
        )
    )


@dataclass(frozen=True, slots=True)
class Budget:
    """A resolved bit budget over one model.

    Attributes:
        total_bits: What the whole checkpoint may occupy, metadata included.
        fixed_bits: What is already spoken for -- tensors that stay at compute
            dtype and cannot be traded, whether they were floored there by their role
            or refused by the graph before they got one. Subtracted before the
            allocator sees a budget, so it never "spends" bits it was never able to
            move.
        denominator: Params the reported average is divided by. This is *every*
            parameter, including the unquantized ones, because that is what a user
            means by "the model at 3 bits".
    """

    total_bits: float
    fixed_bits: float
    denominator: int
    label: str

    @property
    def variable_bits(self) -> float:
        """What the allocator actually has to distribute."""
        return self.total_bits - self.fixed_bits

    @property
    def average_bits(self) -> float:
        return self.total_bits / max(self.denominator, 1)

    @classmethod
    def from_target(
        cls,
        graph: ModelGraph,
        *,
        target_bits: float | None = None,
        target_size: str | int | None = None,
        target_ratio: float | None = None,
        group_size: int = DEFAULT_GROUP_SIZE,
    ) -> Budget:
        """Resolve one of the three ways to ask for a budget into stored bits."""
        given = [t is not None for t in (target_bits, target_size, target_ratio)]
        if sum(given) != 1:
            raise ValueError(
                "specify exactly one of target_bits, target_size or target_ratio "
                f"(got {sum(given)})"
            )

        denominator = graph.total_params()
        # Refused tensors are added as a count rather than through `module_stored_bits`,
        # because there is no `ModuleInfo` to hand it: they never entered the graph. At
        # `UNQUANTIZED_FLOOR` that function is `params * 16` anyway, so the two paths
        # agree; what differs is only that one of them has an object to pass.
        fixed = sum(
            module_stored_bits(info, UNQUANTIZED_FLOOR, group_size=group_size)
            for info in graph.unquantized()
        ) + float(UNQUANTIZED_FLOOR * graph.skipped_params())

        if target_bits is not None:
            total = float(denominator) * target_bits
            label = f"{target_bits:.2f} avg bits"
        elif target_ratio is not None:
            total = float(denominator) * UNQUANTIZED_FLOOR * target_ratio
            label = f"{target_ratio:.1%} of fp16"
        else:
            assert target_size is not None
            nbytes = parse_size(target_size) if isinstance(target_size, str) else target_size
            total = float(nbytes) * 8.0
            label = f"{nbytes / 1024**3:.2f} GiB"

        return cls(total_bits=total, fixed_bits=fixed, denominator=denominator, label=label)
