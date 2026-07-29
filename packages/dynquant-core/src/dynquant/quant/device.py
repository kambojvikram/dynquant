"""Where the quantization arithmetic runs, which is not the same question as where
the weights live.

Encoding a weight reads exactly one tensor and writes exactly one tensor. Eight
candidate encodes, eight decodes, a grouped error reduction, then a final encode
with the per-group winners -- all of it a pure function of that tensor, so it gives
the same answer wherever it is evaluated. Where the *model* sits is a different
question, decided by whether it fits.

Those two questions were conflated: :func:`~dynquant.quant.grid.quantize_with_search`
followed its input's device, so a model held in host RAM did all of that arithmetic
on the CPU with an idle accelerator beside it. That is not a rare configuration.
It is the normal one for anything too large for VRAM, and the deliberate one when
the point of the run is to measure packed VRAM without a dense copy ever reaching
the device -- which is exactly the case that most wants the GPU and least wants the
model on it. Measured on a 7 B: about 10 s per module on CPU against a fraction of
a second on an A100, or roughly forty minutes against roughly one.

So the compute device is chosen separately, and defaults to the accelerator when
there is one. Weights move one at a time and the packed result is handed back to
the caller to place, which bounds the extra device memory at a single tensor's
working set rather than the model's: a 70 B living in host RAM can be quantized
using a 24 GB card. Nothing dense stays resident, because nothing dense is kept.

Reproducibility, honestly
-------------------------
Same answer, not same bytes. Encoding the same weights on CPU and on CUDA produces
two encodings of *equal quality* that are not byte-identical, for two measured
reasons:

* One group scale in roughly 10^5 differs by a single fp16 ulp. Group min and max
  agree exactly and so do the chosen clip ratios, so this is not reduction order --
  it is floating-point contraction, the ``centre +/- half`` clip arithmetic
  compiling to a fused multiply-add on one device and not the other.
* The clip search is an ``argmin`` over eight candidates, so a group whose top two
  candidates sit within float noise of each other can tie-break either way: 2 groups
  in 131,072 at 4 bits, none at 2, 3 or 8 on the tensor this was measured against.

The effect on quality is nil -- relative reconstruction error differs by at most
1e-6 -- and ``tests/test_quant_device.py`` pins that as a tolerance rather than
pinning byte-equality, which would only be a test that no compiler ever changed its
mind about an FMA. What it does mean is that a packed checkpoint is bit-reproducible
on a given device and not across devices. Anything comparing two encodings for
*identity* -- a parity check between a simulated arm and a packed one, most
obviously -- has to encode both on the same device to be measuring what it thinks.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from dynquant._logging import get_logger
from dynquant.constants import DEFAULT_GROUP_SIZE
from dynquant.errors import DynQuantError

from .grid import CLIP_CANDIDATES, quantize_with_search

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .grid import ClipSearchResult
    from .tensor import QuantTensor

__all__ = ["COMPUTE_DEVICE_ENV", "quantize_tensor", "resolve_compute_device"]

_log = get_logger(__name__)

COMPUTE_DEVICE_ENV = "DYNQUANT_QUANTIZE_DEVICE"
"""Overrides ``compute_device="auto"`` process-wide.

For the case where the caller is a script you do not own -- a fine-tuning harness,
someone else's eval driver -- and the machine's answer differs from the default.
"""

_STAY = frozenset({"none", "same", "keep"})


def resolve_compute_device(
    spec: str | torch.device | None = "auto",
) -> torch.device | None:
    """Decide where encoding runs. ``None`` means "wherever the weight already is".

    ``"auto"`` picks CUDA when it is available and stays put otherwise, consulting
    ``$DYNQUANT_QUANTIZE_DEVICE`` first. Only CUDA is auto-selected: MPS and other
    backends are reachable by naming them explicitly, but they are not silently
    chosen, because a wrong answer from an unexercised backend is worse than a slow
    correct one.

    Raises:
        DynQuantError: A device was named explicitly and is not available. Named
            explicitly means the caller wanted that device specifically, so falling
            back to CPU would silently deliver the forty minutes they were trying to
            avoid.
    """
    if spec is None:
        return None

    if isinstance(spec, torch.device):
        return _verify(spec)

    text = str(spec).strip().lower()
    if text in _STAY:
        return None
    if text == "auto":
        override = os.environ.get(COMPUTE_DEVICE_ENV, "").strip()
        if override and override.lower() != "auto":
            return resolve_compute_device(override)
        return torch.device("cuda") if torch.cuda.is_available() else None

    try:
        device = torch.device(text)
    except (RuntimeError, ValueError) as exc:
        raise DynQuantError(
            f"{spec!r} is not a device. Expected 'auto', 'none', or something "
            f"torch.device accepts such as 'cuda', 'cuda:1' or 'cpu'."
        ) from exc
    return _verify(device)


def _verify(device: torch.device) -> torch.device:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DynQuantError(
            f"compute device {str(device)!r} was requested but torch reports no CUDA. "
            f"Use --compute-device none to encode on whichever device holds the "
            f"weights, or `dynquant doctor` to see why CUDA is missing."
        )
    return device


def quantize_tensor(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int = DEFAULT_GROUP_SIZE,
    symmetric: bool = False,
    candidates: Sequence[float] = CLIP_CANDIDATES,
    compute_dtype: torch.dtype | None = None,
    row_offset: int = 0,
    device: torch.device | None = None,
) -> tuple[QuantTensor, ClipSearchResult]:
    """Encode one weight on ``device``, falling back to its own device if it will not fit.

    The result is returned on whichever device did the work. Callers place it:
    :func:`~dynquant.runtime.linear.pack_model` sends it back beside the model,
    :func:`~dynquant.quant.quantizer.quantize_model` has one more pass to do on it
    first and would rather do that pass on the accelerator too.

    The out-of-memory fallback is per tensor, not per model. One oversized tensor --
    typically an untied ``lm_head`` against a small card -- costs its own encode on
    the CPU and nothing else; every other module still gets the GPU. Falling back
    wholesale on the first failure would surrender the entire speedup to the single
    largest weight in the model.
    """
    if device is None or weight.device == device:
        return quantize_with_search(
            weight,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            candidates=candidates,
            compute_dtype=compute_dtype,
            row_offset=row_offset,
        )

    try:
        return quantize_with_search(
            weight,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            candidates=candidates,
            compute_dtype=compute_dtype,
            row_offset=row_offset,
            device=device,
        )
    except RuntimeError as exc:
        if not _is_out_of_memory(exc):
            raise
        # Warning rather than debug: the run is now slower than the user asked for,
        # in a way that shows up as wall-clock and nothing else.
        _log.warning(
            "%s ran out of memory encoding a %s tensor at %d bits; falling back to %s "
            "for this weight only",
            device,
            tuple(weight.shape),
            bits,
            weight.device,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return quantize_with_search(
            weight,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            candidates=candidates,
            compute_dtype=compute_dtype,
            row_offset=row_offset,
        )


def _is_out_of_memory(exc: BaseException) -> bool:
    """Whether ``exc`` is an allocation failure rather than a real error.

    ``torch.cuda.OutOfMemoryError`` subclasses ``RuntimeError`` and does not exist on
    every torch this package supports, so the type is checked when present and the
    message when not. Matching on the message alone would swallow unrelated
    ``RuntimeError``s that happen to mention memory.
    """
    oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return "out of memory" in str(exc).lower()
