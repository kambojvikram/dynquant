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

__all__ = ["CLIP_CANDIDATES", "ClipSearchResult", "quantize_with_search", "search_clip_ratios"]

CLIP_CANDIDATES: tuple[float, ...] = (1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.85, 0.80)
"""The supplement's grid, kept verbatim so ``paper-3.15`` can reproduce its numbers.

The spacing is deliberate and worth keeping regardless: fine near 1.0 where most
groups land, coarse below 0.9 where only genuinely outlier-dominated groups go.
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
    weight: torch.Tensor, recon: torch.Tensor, in_features: int, groups: int, group: int
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
    return diff.reshape(rows.shape[0], groups, group).sum(dim=-1)


def search_clip_ratios(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int = DEFAULT_GROUP_SIZE,
    symmetric: bool = False,
    candidates: Sequence[float] = CLIP_CANDIDATES,
    compute_dtype: torch.dtype | None = None,
) -> ClipSearchResult:
    """Pick the error-minimising clip ratio for every group of ``weight``.

    Each candidate is encoded and decoded by the real encoder, so the reported
    error is the error the checkpoint will have, not an estimate of it.
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
) -> tuple[QuantTensor, ClipSearchResult]:
    """Search, then encode once with the per-group winners."""
    result = search_clip_ratios(
        weight,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        candidates=candidates,
        compute_dtype=compute_dtype,
    )
    quantized = QuantTensor.from_dense(
        weight,
        bits=bits,
        group_size=group_size,
        symmetric=symmetric,
        compute_dtype=compute_dtype,
        row_offset=row_offset,
        clip_ratio=result.ratios,
    )
    return quantized, result
