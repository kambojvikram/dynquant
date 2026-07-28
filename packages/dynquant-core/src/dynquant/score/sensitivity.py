"""Cardinal sensitivity: an estimate of ``dLoss`` per module per candidate width.

What this replaces
------------------
The rank-product score answers "which module is more important" on an ordinal
scale, and the allocator then multiplies that rank by a *universal* error curve --
``4**-bits``, identical for every module in the model -- to guess what a width
change is worth. Two approximations stacked, and neither is checkable against
anything.

This computes the quantity directly. For a weight ``W`` and its quantized
reconstruction ``Q_b(W)``, the second-order expansion of the loss around the
trained weights, with the Gauss-Newton approximation to the Hessian and its
diagonal taken in the Kronecker factorisation, is::

    dL(b)  ~=  sum_rc  E[delta_r^2] * E[x_c^2] * (W - Q_b(W))_rc^2

``E[x_c^2]`` and ``E[delta_r^2]`` come from the fine-tune
(:mod:`dynquant.signals.moments`); the error term is the *actual* error of the
*actual* candidate width, computed with the same clipping search the quantizer
will use. So the number is in loss units, per module, per width -- which makes the
knapsack's move pricing a subtraction (``dL(3) - dL(4)``: what this bit buys)
rather than a rank times a constant.

Three things follow from that, and they are the reason to prefer it:

* **Range survives.** Percentile-ranking compressed plasticity's measured 5.9
  decades into a 47x spread before the allocator ever saw it. Nothing here ranks.
* **The error curve stops being universal.** A module whose weights are nearly
  uniform within each group loses little to 3-bit and gains little from a 4th bit;
  one with heavy tails loses a lot. ``4**-bits`` cannot express the difference, and
  the difference is exactly what a bit-allocator exists to exploit.
* **It is falsifiable.** ``dL`` is measurable by quantizing a module and running
  the model. The rank product is not measurable at all.

What was measured
-----------------
See :mod:`dynquant.signals.moments` for the table. Within role, against the true
loss disturbance of 187 modules of a fine-tuned Qwen3.5-2B: this estimator +0.521
mean Spearman (right sign in 11 of 12 roles), the shipped rank product +0.231
(9/12), and the same formula with ``E[delta^2]`` dropped **-0.338** (1/12).

The last figure is why :func:`weight_only_sensitivity` exists but is not the
default. It is the honest control -- what a calibration-free method can compute --
and it points the wrong way.

Known limits, stated rather than hidden
---------------------------------------
* The estimate is **first-order in the module**: it assumes the other modules are
  unquantized. Measured on the same model, the sum of 187 individually-measured
  disturbances was +0.089 against an actual all-3-bit cost of +0.047, so errors
  partially cancel and this over-counts by roughly 2x. It over-counts *everything*,
  which a ranking is insensitive to and a claimed absolute ``dL`` is not -- hence
  :attr:`SensitivityTable.absolute`, which says so.
* It predicts the **magnitude** of disturbance. Of those 187 single-module
  quantizations, 71 lowered the loss; sign is decided by interactions this
  approximation does not model.
* Modules whose input is not a feature axis (embeddings, whose moment is a token
  histogram, and batched expert banks) are reported in
  :attr:`SensitivityTable.unestimable` rather than given a fabricated number.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dynquant._logging import get_logger
from dynquant.constants import BIT_OPTIONS, DEFAULT_GROUP_SIZE

if TYPE_CHECKING:
    from torch import Tensor, nn

    from dynquant.graph.classify import ModelGraph
    from dynquant.signals.moments import ChannelMoments

__all__ = [
    "SensitivityTable",
    "estimate_sensitivity",
    "module_weights",
    "weight_only_sensitivity",
]

_log = get_logger(__name__)

_ROW_CHUNK_BYTES = 64 << 20
"""Working-set cap for the per-module error computation.

The error tensor is fp32 and the same shape as the weight, so a tied 248,320 x 2048
embedding would need 2 GB in one go -- on the same device the model is on. Rows are
independent (groups run along the *input* axis, so no group straddles a row
boundary) which makes row chunking exact rather than approximate."""


@dataclass(slots=True)
class SensitivityTable:
    """Estimated ``dLoss`` for every (module, width) pair the allocator may consider."""

    values: dict[str, dict[int, float]] = field(default_factory=dict)
    """``values[name][bits]`` -- estimated loss increase from quantizing to ``bits``.

    Monotonically non-increasing in ``bits`` by construction, since more bits means
    less error, but not guaranteed exactly so: the clipping search optimises MSE per
    group and can occasionally find a better clip at a lower width."""

    unestimable: tuple[str, ...] = ()
    """Quantizable modules with no usable moments. The caller must fall back to a
    score for these rather than treat a missing entry as zero sensitivity, which
    would make them the first thing downgraded."""

    absolute: bool = False
    """Whether :attr:`values` may be read as a real loss delta.

    Always ``False`` today, and the field exists to keep that visible. The estimator
    is first-order in the module and measured ~2x high in aggregate on Qwen3.5-2B,
    so the numbers are a calibrated *ordering with proportions*, which is what the
    knapsack needs, and not a prediction of the loss a checkpoint will have."""

    group_size: int = DEFAULT_GROUP_SIZE
    bit_options: tuple[int, ...] = BIT_OPTIONS

    def gain(self, name: str, from_bits: int, to_bits: int) -> float | None:
        """What upgrading ``from_bits -> to_bits`` buys, in estimated loss units.

        ``None`` when either width was not estimated, which the caller must treat as
        "no information" rather than as zero -- see :attr:`unestimable`.
        """
        widths = self.values.get(name)
        if widths is None:
            return None
        lo, hi = widths.get(from_bits), widths.get(to_bits)
        if lo is None or hi is None:
            return None
        return lo - hi

    def summary(self) -> str:
        lines = [f"sensitivity for {len(self.values)} modules at widths {self.bit_options}"]
        if self.unestimable:
            lines.append(
                f"  {len(self.unestimable)} unestimable, falling back to score "
                f"(e.g. {', '.join(self.unestimable[:3])})"
            )
        return "\n".join(lines)


def module_weights(model: nn.Module) -> dict[str, Tensor]:
    """Canonical name -> weight tensor, for every module carrying one.

    Keyed the way :class:`~dynquant.graph.classify.ModelGraph` and the stats file
    are keyed, so a PEFT-wrapped model and the merged checkpoint it becomes produce
    the same dictionary.
    """
    from dynquant.graph.naming import canonical_name, is_adapter_name

    out: dict[str, Tensor] = {}
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or is_adapter_name(name):
            continue
        out[canonical_name(name)] = weight.detach()
    return out


def estimate_sensitivity(
    graph: ModelGraph,
    moments: ChannelMoments,
    weights: Mapping[str, Tensor],
    *,
    bit_options: Sequence[int] = BIT_OPTIONS,
    group_size: int = DEFAULT_GROUP_SIZE,
    symmetric: bool = False,
) -> SensitivityTable:
    """Estimate ``dLoss`` for every module at every candidate width.

    Args:
        weights: Canonical name -> weight, from :func:`module_weights` or from a
            safetensors checkpoint opened lazily. Passed rather than read off the
            graph because :class:`~dynquant.graph.classify.ModelGraph` holds no live
            module references -- deliberately, so it can be built, saved, and
            reasoned about without the model resident -- and because the fine-tune
            and the quantization commonly happen on different machines.
    """
    import torch

    widths = tuple(sorted({int(b) for b in bit_options}))
    table = SensitivityTable(group_size=group_size, bit_options=widths)
    unestimable: list[str] = []
    aliases = _tie_aliases(graph)

    for info in graph.quantizable():
        weight = weights.get(info.name)
        pair = _moments_for(info.name, aliases, moments, weight)
        if weight is None or pair is None:
            unestimable.append(info.name)
            continue
        x2, d2 = pair
        with torch.no_grad():
            table.values[info.name] = {
                bits: _module_sensitivity(
                    weight,
                    x2,
                    d2,
                    bits=bits,
                    group_size=group_size,
                    symmetric=symmetric,
                )
                for bits in widths
            }

    table.unestimable = tuple(unestimable)
    if unestimable:
        _log.info(
            "%d of %d quantizable modules have no usable channel moments and will be "
            "allocated from their score instead. First few: %s",
            len(unestimable),
            len(unestimable) + len(table.values),
            ", ".join(unestimable[:5]),
        )
    return table


def weight_only_sensitivity(
    graph: ModelGraph,
    weights: Mapping[str, Tensor],
    *,
    bit_options: Sequence[int] = BIT_OPTIONS,
    group_size: int = DEFAULT_GROUP_SIZE,
    symmetric: bool = False,
) -> SensitivityTable:
    """``sum_rc (W - Q_b(W))^2`` -- the same shape with no data in it at all.

    The control, not an option. It is what a calibration-free method can compute,
    and on the one model where both have been checked against measured loss
    disturbance it scored **-0.227** mean within-role Spearman: not merely weaker
    than the Gauss-Newton form's +0.521, but consistently the wrong sign. Kept
    because a claim that data-driven sensitivity beats weight-only sensitivity is
    worth being able to re-run rather than cite.
    """
    import torch

    widths = tuple(sorted({int(b) for b in bit_options}))
    table = SensitivityTable(group_size=group_size, bit_options=widths)
    unestimable: list[str] = []

    for info in graph.quantizable():
        weight = weights.get(info.name)
        if weight is None or weight.ndim != 2:
            unestimable.append(info.name)
            continue
        with torch.no_grad():
            table.values[info.name] = {
                bits: _module_sensitivity(
                    weight,
                    None,
                    None,
                    bits=bits,
                    group_size=group_size,
                    symmetric=symmetric,
                )
                for bits in widths
            }
    table.unestimable = tuple(unestimable)
    return table


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------


def _tie_aliases(graph: ModelGraph) -> dict[str, tuple[str, ...]]:
    """Representative -> every other name the same tensor answers to.

    Tied weights are the case where looking a module up by its own name is not
    enough. Qwen3.5-2B ties ``embed_tokens`` to ``lm_head``: one 248,320 x 2048
    tensor, 27% of the model, and only the *head* has moments, because an
    embedding's input is a token id rather than a feature vector so the pairing its
    forward pass produces is transposed relative to how the tensor is stored. The
    representative is whichever name ``named_modules`` yielded first, which is the
    embedding -- so without this, the single largest tensor in the model is
    unestimable for a reason that is an accident of iteration order.
    """
    aliases: dict[str, list[str]] = {}
    for info in graph.modules.values():
        if info.tied_to is not None:
            aliases.setdefault(info.tied_to, []).append(info.name)
    return {name: tuple(sorted(others)) for name, others in aliases.items()}


def _moments_for(
    name: str,
    aliases: Mapping[str, tuple[str, ...]],
    moments: ChannelMoments,
    weight: Tensor | None,
) -> tuple[Tensor, Tensor] | None:
    """The first shape-compatible ``(E[x^2], E[delta^2])`` pair for this tensor.

    First rather than summed. A tied tensor is used at two points in the network and
    the true disturbance is the sum of both uses, but only one of the two produces a
    pairing in the stored tensor's orientation, so there is nothing to add -- and
    fabricating the other by transposing would be inventing a measurement.
    """
    if weight is None:
        return None
    for candidate in (name, *aliases.get(name, ())):
        x2 = moments.input_sq.get(candidate)
        d2 = moments.output_grad_sq.get(candidate)
        if _shapes_agree(weight, x2, d2):
            assert x2 is not None and d2 is not None  # _shapes_agree rejects None
            return x2, d2
    return None


def _shapes_agree(weight: Tensor, x2: Tensor | None, d2: Tensor | None) -> bool:
    """Reject anything the Kronecker form does not apply to, explicitly.

    A wrong axis broadcasts silently on a square weight and only raises on a
    rectangular one -- which is how a first attempt at this measurement ended up
    treating a sequence length as an input-channel count and produced numbers for
    two thirds of the model before anything complained.
    """
    if x2 is None or d2 is None or weight.ndim != 2:
        return False
    return x2.shape[0] == weight.shape[1] and d2.shape[0] == weight.shape[0]


def _module_sensitivity(
    weight: Tensor,
    x2: Tensor | None,
    d2: Tensor | None,
    *,
    bits: int,
    group_size: int,
    symmetric: bool,
) -> float:
    """One module, one width. Row-chunked so peak memory is bounded by the chunk."""
    import torch

    from dynquant.quant.grid import quantize_with_search

    rows, cols = weight.shape
    per_row = max(cols * 4, 1)
    chunk = max(1, min(rows, _ROW_CHUNK_BYTES // per_row))

    x2f = x2.to(weight.device, torch.float32) if x2 is not None else None
    d2f = d2.to(weight.device, torch.float32) if d2 is not None else None

    total = 0.0
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        block = weight[start:stop]
        quantized, _ = quantize_with_search(
            block,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            row_offset=start,
        )
        error = block.float() - quantized.dequantize(dtype=torch.float32)
        error.mul_(error)
        if x2f is None or d2f is None:
            total += float(error.sum())
            continue
        # (E * x2) summed over input channels gives one number per output row; the
        # dot with d2 then applies the output weighting. Never forms the [out, in]
        # weighted product, so the working set stays the chunk.
        per_output = error @ x2f
        total += float(per_output @ d2f[start:stop])
    return total
