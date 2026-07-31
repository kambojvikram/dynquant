"""MSE-optimal clipping search.

Pure min/max quantization lets a single outlier in a group set the range for all
128 weights in it. Shrinking the range clips that outlier badly but gives every
other weight a finer step, and below about 4 bits that trade is usually worth
taking. The search is a small grid over shrink factors, scored by reconstruction
error, decided independently per group.

Two things here differ from the supplement's version, both deliberate.

**The objective is the real encoder's error, not a model of it.** The supplement
scored candidates with an inline copy of the affine math that quantized against
full-precision scales. The scales it then *stored* were rounded to fp16, so the
alpha it picked was optimal for an encoder that was never run. Here each candidate
goes through :meth:`QuantTensor.from_dense` and is scored after
:meth:`~QuantTensor.dequantize`, which costs one pack per candidate and in exchange
makes the winner optimal for the bytes that actually reach the disk. It also means
the search cannot drift away from the encoder as the encoder changes, because it
has no independent copy of it.

**Selection is per group, and the winner is returned rather than applied.**
Returning the ratios lets the caller quantize once more with them, keeps this
module free of packing concerns, and makes the chosen ratios inspectable -- how far
a tensor wanted to clip is a useful signal about how outlier-dominated it is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from dynquant.constants import DEFAULT_GROUP_SIZE

from .pack import row_geometry
from .tensor import QuantTensor

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "CLIP_CANDIDATES",
    "DEEP_CLIP_CANDIDATES",
    "ClipSearchResult",
    "quantize_with_search",
    "search_clip_ratios",
]

CLIP_CANDIDATES: tuple[float, ...] = (1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.85, 0.80)
"""The supplement's grid, kept verbatim so ``paper-3.15`` can reproduce its numbers.

The spacing is deliberate and worth keeping regardless: fine near 1.0 where most
groups land, coarse below 0.9 where only genuinely outlier-dominated groups go.
"""

DEEP_CLIP_CANDIDATES: tuple[float, ...] = (
    *CLIP_CANDIDATES,
    0.75,
    0.70,
    0.65,
    0.60,
    0.55,
    0.50,
    0.45,
    0.40,
)
"""The same grid, continued to where a 2-bit group's optimum actually lives.

0.80 is a defensible floor at 4 bits and cannot be one at 2. With four levels the
MSE-optimal shrink is far tighter than 0.80 for anything resembling a Gaussian
group, so every 2-bit group whose optimum lies below the floor returns the floor
and reports it as the winner. Measured on Qwen3.5-2B, the mean chosen ratio once
the grid can reach it is **0.52-0.59** at 2 bits, 0.73-0.86 at 3, and 0.88-0.93 at
4 -- so the floor bound at 2 bits, only at 2 bits, and this extension is inert at
the widths where the original was already right.

What it moves, per-module Gauss-Newton sensitivity at 2 bits / group 128:

===========================  ==========  ==========  =======
module                       shipped     deep        change
===========================  ==========  ==========  =======
``embed_tokens`` (508M)      2.2188e+04  1.5624e+04  -29.6%
``mlp.up_proj``              2.0242e-09  1.3330e-09  -34.1%
``mlp.gate_proj``            1.6606e-09  1.1596e-09  -30.2%
``mlp.down_proj``            2.5438e-09  1.8271e-09  -28.2%
``linear_attn.out_proj``     2.9094e-08  1.3734e-08  -52.8%
``linear_attn.in_proj_qkv``  1.9475e-09  1.4916e-09  -23.4%
===========================  ==========  ==========  =======

**Do not read that table as the payoff.** It is the payoff's cause, and the two are
not the same size or even the same mechanism. Measured end to end on Qwen3.5-2B /
CaseHOLD at a byte-identical 740,724,736 B, splitting the change into the grid the
allocator prices with and the grid the encoder runs:

===============================================  ======
arm                                              acc %
===============================================  ======
priced shipped, encoded shipped                  85.42
priced shipped, **encoded deep**                 85.51
**priced deep**, encoded deep                    86.09
===============================================  ======

Encoding better is worth +0.09 points, which on 5,314 items is about 0.2 sigma --
nothing. The +0.67 is almost entirely re-pricing. A grid clamped at 0.80 reports a
2-bit error that 2 bits does not have to incur, which inflates the 2b->3b
sensitivity ratio (4.69x on ``mlp.gate`` layer 0 shipped, 3.47x deep); the allocator
reads that cliff as *2 bits is catastrophic here* and buys 3 bits it did not need.
Correcting the encoder cannot recover that, because by then the budget is already
committed elsewhere. Correcting the price moves the budget.

The practical consequence is the argument for
:func:`~dynquant.score.sensitivity.estimate_sensitivity` taking the grid at all: a
30% sensitivity improvement that the allocator never sees is worth roughly zero, and
a 30% improvement it does see is worth seven times more than the encoding that
produced it.

It is not a uniform win: on the 16x2048 ``in_proj_a``/``in_proj_b`` projections and
on ``k_proj`` the deeper clip lowers reconstruction error while *raising* weighted
sensitivity, because the search minimises unweighted MSE per group and a tighter
clip trades away exactly the large-magnitude weights the channel moments say
matter. Those modules carry almost no parameter mass and sit at 8 bits, so the
aggregate is strongly positive -- but the disagreement is real, and it is why
:func:`~dynquant.score.sensitivity.estimate_sensitivity` takes the grid as an
argument: the allocator should be priced with whatever grid the quantizer will
actually run, never with a different one.
"""


class ClipSearchResult:
    """The chosen ratios and what they bought, for ``dynquant inspect``."""

    __slots__ = ("baseline_mse", "best_mse", "ratios")

    def __init__(self, ratios: torch.Tensor, best_mse: torch.Tensor, baseline_mse: torch.Tensor):
        self.ratios = ratios
        self.best_mse = best_mse
        self.baseline_mse = baseline_mse

    @property
    def improvement(self) -> float:
        """Fraction of squared error removed relative to pure min/max."""
        baseline = float(self.baseline_mse.sum())
        if baseline <= 0.0:
            return 0.0
        return float((baseline - float(self.best_mse.sum())) / baseline)

    @property
    def clipped_fraction(self) -> float:
        """Fraction of groups that preferred something tighter than min/max."""
        return float((self.ratios < 1.0).float().mean())


def _group_errors(
    weight: torch.Tensor,
    recon: torch.Tensor,
    in_features: int,
    groups: int,
    group: int,
    channel_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Squared error summed within each group, shaped ``[rows, groups]``."""
    rows = weight.reshape(-1, in_features)
    padded_width = groups * group
    if padded_width != in_features:
        rows = torch.nn.functional.pad(rows, (0, padded_width - in_features))
        recon = torch.nn.functional.pad(
            recon.reshape(-1, in_features), (0, padded_width - in_features)
        )
    else:
        recon = recon.reshape(-1, in_features)
    diff = (rows.to(torch.float32) - recon.to(torch.float32)) ** 2
    if channel_weight is not None:
        # Both operands are zero-padded, so the pad columns contribute nothing
        # whatever value they carry here.
        w = channel_weight.to(diff.device, torch.float32).reshape(-1)
        if w.numel() != padded_width:
            w = torch.nn.functional.pad(w, (0, padded_width - w.numel()))
        diff = diff * w
    return diff.reshape(rows.shape[0], groups, group).sum(dim=-1)


def search_clip_ratios(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int = DEFAULT_GROUP_SIZE,
    symmetric: bool = False,
    candidates: Sequence[float] = CLIP_CANDIDATES,
    compute_dtype: torch.dtype | None = None,
    channel_weight: torch.Tensor | None = None,
) -> ClipSearchResult:
    """Pick the error-minimising clip ratio for every group of ``weight``.

    Each candidate is encoded and decoded by the real encoder, so the reported
    error is the error the checkpoint will have, not an estimate of it.

    Args:
        channel_weight: Optional per-input-channel weight of length ``in_features``,
            normally ``E[x_c^2]`` from the fine-tune's own moments. Without it the
            objective is plain reconstruction MSE, which treats every input channel
            as equally consequential -- and the network does not. A group's error
            reaches the loss through ``sum_c E[x_c^2] (W - Q)^2_rc``, so clipping
            chosen on unweighted MSE can lower total error while *raising* the part
            of it that matters: measured on Qwen3.5-2B, extending the clip grid cut
            ``k_proj``'s 2-bit reconstruction error by 16% while its Gauss-Newton
            sensitivity rose 6.7%.

            The output weight ``E[delta_r^2]`` is deliberately absent. Selection is
            per ``(row, group)`` and that factor is constant along a row, so it
            scales every candidate identically and cannot change the argmin. Passing
            it would cost a broadcast and buy nothing.
    """
    if not candidates:
        raise ValueError("need at least one candidate clip ratio")

    in_features = weight.shape[-1]
    geom = row_geometry(bits, group_size, in_features)
    rows = weight.reshape(-1, in_features).shape[0]
    groups = geom.num_groups

    best_mse = torch.full((rows, groups), float("inf"), device=weight.device, dtype=torch.float32)
    best_ratio = torch.ones((rows, groups), device=weight.device, dtype=torch.float32)
    baseline = torch.zeros((rows, groups), device=weight.device, dtype=torch.float32)

    for index, ratio in enumerate(candidates):
        quantized = QuantTensor.from_dense(
            weight,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            compute_dtype=compute_dtype,
            clip_ratio=ratio,
        )
        mse = _group_errors(
            weight,
            quantized.dequantize(dtype=torch.float32),
            in_features,
            groups,
            geom.effective_group,
            channel_weight,
        )
        if index == 0:
            baseline = mse.clone()
        better = mse < best_mse
        best_mse = torch.where(better, mse, best_mse)
        best_ratio = torch.where(better, torch.full_like(best_ratio, ratio), best_ratio)

    return ClipSearchResult(ratios=best_ratio, best_mse=best_mse, baseline_mse=baseline)


def quantize_with_search(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int = DEFAULT_GROUP_SIZE,
    symmetric: bool = False,
    candidates: Sequence[float] = CLIP_CANDIDATES,
    compute_dtype: torch.dtype | None = None,
    row_offset: int = 0,
    device: torch.device | str | None = None,
    channel_weight: torch.Tensor | None = None,
) -> tuple[QuantTensor, ClipSearchResult]:
    """Search, then encode once with the per-group winners.

    ``device`` runs the arithmetic somewhere other than where ``weight`` lives --
    see :mod:`dynquant.quant.device` for why that is worth separating. The weight is
    moved once and the result comes back on the device that did the work, because
    the caller knows where it belongs and this function does not. ``None`` leaves
    everything where it is.

    Prefer :func:`dynquant.quant.device.quantize_tensor` over passing ``device``
    here: it is this call plus the out-of-memory fallback, and a bare fallback-less
    move turns a tensor that does not fit into a failed run rather than a slow one.
    """
    work = weight if device is None else weight.to(device)
    result = search_clip_ratios(
        work,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        candidates=candidates,
        compute_dtype=compute_dtype,
        channel_weight=channel_weight,
    )
    quantized = QuantTensor.from_dense(
        work,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        compute_dtype=compute_dtype,
        row_offset=row_offset,
        clip_ratio=result.ratios,
    )
    return quantized, result
