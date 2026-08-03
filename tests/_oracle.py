"""Independent predictions to test measurements against.

The point of this module is that nothing in it calls DynQuant's quantizer. A
quantizer test that compares against the quantizer's own scales proves only
self-consistency -- it passes just as happily if the whole grid is computed wrong.
Everything here is derived from the dense tensor with plain torch, so the
comparison has something to say.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_UNIFORM_ERROR_EFFICIENCY = 0.90
"""Ratio of measured RMS error to the ``step/sqrt(12)`` ideal.

``step/sqrt(12)`` assumes the residual is uniform across one quantization step,
which slightly overestimates for min/max grids: the two values that define a
group's range land exactly on grid points and contribute no error at all.
Measured across 2/3/4/8-bit on Gaussian weights the ratio is a stable ~0.90, so
the prediction is scaled by it and the tolerance band can stay tight enough to be
worth having. If a future change makes this drift, that is a real finding about
the quantizer, not a number to retune.
"""


def expected_rel_fro(weight, bits: int, group_size: int) -> float:
    """Predict ``||w - dequant(quant(w))||_F / ||w||_F`` from ``w`` alone.

    For asymmetric min/max quantization with ``2**bits`` levels, a group spanning
    ``R`` has step ``R / (2**bits - 1)`` and RMS residual ``step / sqrt(12)``.
    Squared residuals are averaged across groups -- errors add in quadrature, not
    linearly -- and normalised by the weight's RMS.

    Works for any distribution and any group size, because the group ranges are
    measured rather than assumed.

    **Padding weights the tail group down, it does not scale the answer up.** This
    function used to finish with ``err_rms *= sqrt(padded / in_features)``, on the
    reasoning that padded positions are exact zeros which dilute the mean. They do
    not: the reported error is a ratio, and the pad drops out of the numerator and
    the denominator together. The factor was invisible while the only padded shapes
    under test were 100-of-128 and 320-of-384, where a 13% and a 10% inflation sat
    inside a 25% band pointing the opposite way from
    :data:`_UNIFORM_ERROR_EFFICIENCY`. Gemma-3's ``patch_embedding`` is 14 of 128,
    where the same factor is 3.02x and the prediction misses by two thirds. Two
    errors cancelling is exactly what this module exists to avoid.
    """
    import torch

    w = weight.reshape(-1, weight.shape[-1]).to(torch.float32)
    in_features = w.shape[-1]
    effective_group = in_features if group_size == -1 else group_size

    pad = (-in_features) % effective_group
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    groups = w.reshape(w.shape[0], -1, effective_group)

    # The group's own min/max, *not* clamped around zero. DynQuant stores a float
    # offset, so unlike GPTQ/AWQ it does not widen a group's range to make an
    # integer zero-point representable. Clamping here would predict a step up to
    # 26x too large for a one-signed group -- and would therefore rubber-stamp an
    # encoder that had reintroduced the widening.
    lo = groups.amin(dim=-1)
    hi = groups.amax(dim=-1)
    step = (hi - lo) / (2**bits - 1)

    # Averaged over *real* elements, so a tail group holding 14 values of 128
    # counts for 14 and not for a whole group. Without the weighting a short tail
    # would pull the mean toward its own step, which for `patch_embedding` -- one
    # group per row, 14 real values in it -- is the entire answer.
    counts = torch.full((groups.shape[1],), float(effective_group))
    if pad:
        counts[-1] = effective_group - pad
    mean_sq = ((step.pow(2) / 12.0) * counts).sum().item() / (w.shape[0] * counts.sum().item())
    err_rms = math.sqrt(mean_sq) * _UNIFORM_ERROR_EFFICIENCY

    w_rms = weight.to(torch.float32).pow(2).mean().sqrt().item()
    return err_rms / w_rms


def assert_error_matches_theory(
    measured: float, weight, bits: int, group_size: int, *, band: float = 0.25
) -> None:
    """Two-sided check that a measured error is what theory predicts.

    Two-sided matters: too high means the quantizer lost accuracy, too low means
    the measurement is not measuring quantization (a comparison against itself, a
    dequantize that returned the original tensor).
    """
    predicted = expected_rel_fro(weight, bits, group_size)
    lo, hi = predicted * (1 - band), predicted * (1 + band)
    assert lo <= measured <= hi, (
        f"{bits}-bit g={group_size}: measured rel_fro {measured:.6f} outside "
        f"[{lo:.6f}, {hi:.6f}] predicted from the dense tensor ({predicted:.6f})."
    )


SHIPPED_STATS = {
    "phi-4": REPO_ROOT / "stats" / "phi-4" / "unified_gasq_stats_collapsed.json",
    "qwen3_14b": REPO_ROOT / "stats" / "qwen3_14b" / "unified_gasq_stats_collapsed.json",
}
"""The only two real DynQuant stats files in existence -- the paper's own runs.

They are the golden inputs for migration and, later, for reproducing the
published allocation. Tests skip rather than fail when they are absent, so the
suite still works in a checkout that does not carry the paper artifacts.
"""


def shipped_stats_params():
    """Parametrise over whichever shipped stats files are present."""
    return [
        pytest.param(
            path,
            id=name,
            marks=[] if path.is_file() else pytest.mark.skip(reason=f"{path} not present"),
        )
        for name, path in SHIPPED_STATS.items()
    ]
