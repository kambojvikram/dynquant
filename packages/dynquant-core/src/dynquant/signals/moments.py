"""Per-channel second moments -- the ingredients of a cardinal sensitivity estimate.

Why a second artifact exists
----------------------------
The four scalars in :class:`~dynquant.signals.schema.LayerStats` answer "how active
was this module, and was it still moving". They cannot answer the question the
allocator actually asks, which is **how much does the loss rise if I quantize this
weight** -- because that quantity depends on the quantization error itself, which is
not known until quantization time, and on *which* channels the error lands in.

The Gauss-Newton diagonal gives it::

    dL  ~=  sum_rc  E[delta_r^2] * E[x_c^2] * (W - Q(W))_rc^2

Two vectors per module -- one over input channels, one over output channels --
combined at allocation time with the actual error of the actual candidate width.
That is what this module stores.

The measurement that produced this design
-----------------------------------------
187 modules of a fine-tuned Qwen3.5-2B were quantized one at a time to 3 bits and
the task loss measured against a fixed held-out batch. Ranked against the resulting
disturbance, *within role* so that role floors could not do the work (12 roles,
mean Spearman, and the count of roles where the correlation had the right sign):

===========================================  =======  =======
signal                                          rho    sign
===========================================  =======  =======
Gauss-Newton, the formula above                +0.521   11/12
plasticity alone (``Var_t||grad W||``)         +0.491   11/12
the shipped score, ``rank(sal) x rank(pl)``    +0.231    9/12
``||W - Q(W)||^2``, no data at all             -0.227    4/12
saliency alone (activation RMS)                -0.301    2/12
the same formula *without* ``E[delta^2]``      -0.338    1/12
===========================================  =======  =======

The last row is the load-bearing one. Drop the output-gradient factor and the
estimate is not merely weaker, it is **anti**-correlated -- the version of this
quantity that a calibration-only method can compute points the wrong way. The label
term is what makes it work, and collecting it requires a backward pass over labelled
data, which is exactly what a fine-tune already runs. That is the strongest available
argument for signals-during-training as a method rather than a convenience.

What it costs
-------------
``E[delta_r^2]`` is one reduction over a gradient the backward pass already
materialised; ``E[x_c^2]`` is one reduction over an activation the forward pass
already has. Neither allocates a copy. But *per step* they would roughly double the
tracker's per-module hook work, which measured at +2.30% of step time against a 3%
budget -- so they are sampled every ``channel_moment_every`` steps instead. These are
distributional statistics of the activations, not per-step events like the gradient
variance, and they move on the timescale of training rather than of a batch; sampling
16 steps apart changes the estimate by far less than the batch-to-batch noise already
in it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from dynquant._logging import get_logger
from dynquant.constants import MOMENT_TENSOR_SUFFIXES, MOMENTS_FILENAME, MOMENTS_SCHEMA

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["ChannelMoments", "load_moments", "save_moments"]

_log = get_logger(__name__)

_INPUT = MOMENT_TENSOR_SUFFIXES["input_sq"]
_OUTPUT = MOMENT_TENSOR_SUFFIXES["output_grad_sq"]


@dataclass(slots=True)
class ChannelMoments:
    """``E[x_c^2]`` and ``E[delta_r^2]`` per module, keyed by canonical name.

    Both are optional per module and independently so. A module can have an input
    moment and no output moment -- a frozen embedding under LoRA produces no output
    gradient at all (see ``docs/legacy-audit.md`` finding 7) -- and the consumer
    needs to distinguish "measured as small" from "not measured", so absence is
    represented by absence rather than by a vector of zeros.
    """

    input_sq: dict[str, Tensor] = field(default_factory=dict)
    """Mean over observed tokens of ``x_c^2``. Length ``in_features``."""

    output_grad_sq: dict[str, Tensor] = field(default_factory=dict)
    """Mean over observed tokens of ``(dL/dy_r)^2``. Length ``out_features``."""

    observations: dict[str, int] = field(default_factory=dict)
    """How many sampled steps contributed, per module. A reader deciding whether to
    trust a moment needs the denominator, and a module routed twice in a 500-step
    run (a rare MoE expert) should not look as authoritative as one routed always."""

    def __len__(self) -> int:
        return len(set(self.input_sq) | set(self.output_grad_sq))

    def __contains__(self, name: str) -> bool:
        return name in self.input_sq or name in self.output_grad_sq

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.input_sq) | set(self.output_grad_sq)))

    def complete_names(self) -> tuple[str, ...]:
        """Modules carrying *both* moments -- the only ones the estimator can use."""
        return tuple(sorted(set(self.input_sq) & set(self.output_grad_sq)))

    def summary(self) -> str:
        both = len(self.complete_names())
        return (
            f"{len(self)} modules with channel moments, {both} complete "
            f"(input only: {len(set(self.input_sq) - set(self.output_grad_sq))}, "
            f"output only: {len(set(self.output_grad_sq) - set(self.input_sq))})"
        )

    # -- io ----------------------------------------------------------------

    def to_tensors(self) -> dict[str, Tensor]:
        """Flatten to safetensors keys: ``<module>.<suffix>``.

        Names carry dots already, and so do the suffixes' separators, which is
        harmless: the split below takes the *last* dot, and no canonical module
        name ends in one of the two reserved suffixes.
        """
        out: dict[str, Tensor] = {}
        for name, tensor in self.input_sq.items():
            out[f"{name}.{_INPUT}"] = tensor
        for name, tensor in self.output_grad_sq.items():
            out[f"{name}.{_OUTPUT}"] = tensor
        return out

    @classmethod
    def from_tensors(cls, tensors: Mapping[str, Tensor]) -> ChannelMoments:
        moments = cls()
        for key, tensor in tensors.items():
            name, _, suffix = key.rpartition(".")
            if suffix == _INPUT:
                moments.input_sq[name] = tensor
            elif suffix == _OUTPUT:
                moments.output_grad_sq[name] = tensor
            else:
                _log.warning("ignoring unrecognised moments key %r", key)
        return moments

    def save(self, path: str | Path, *, observations: Mapping[str, int] | None = None) -> Path:
        return save_moments(self, path, observations=observations)


def save_moments(
    moments: ChannelMoments,
    path: str | Path,
    *,
    observations: Mapping[str, int] | None = None,
) -> Path:
    """Write moments to safetensors. Returns the path written.

    Observation counts go in the file's metadata rather than as tensors: they are
    per-module scalars, and safetensors metadata is a flat ``str -> str`` map, which
    is exactly enough and keeps them out of the tensor namespace where a stray key
    would be parsed as a channel vector.
    """
    from safetensors.torch import save_file

    target = Path(path)
    if target.is_dir():
        target = target / MOMENTS_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)

    counts = observations if observations is not None else moments.observations
    metadata = {f"obs:{name}": str(count) for name, count in sorted(counts.items())}
    metadata["schema"] = MOMENTS_SCHEMA

    tensors = {key: t.detach().to("cpu").contiguous() for key, t in moments.to_tensors().items()}
    # Write-then-rename, matching StatsFile.save: a crash mid-write must not leave a
    # truncated sidecar that a later run would silently allocate from.
    tmp = target.with_suffix(target.suffix + ".tmp")
    save_file(tensors, str(tmp), metadata=metadata)
    tmp.replace(target)
    return target


def load_moments(path: str | Path) -> ChannelMoments:
    """Read a moments sidecar. Accepts the file or the directory holding it."""
    from safetensors import safe_open

    source = Path(path)
    if source.is_dir():
        source = source / MOMENTS_FILENAME
    if not source.is_file():
        raise FileNotFoundError(
            f"no channel-moments sidecar at {source}. It is written next to the stats "
            f"file by a tracker with collect_channel_moments enabled; a run that "
            f"predates it, or that turned it off, produces stats without one."
        )

    tensors: dict[str, Tensor] = {}
    with safe_open(str(source), framework="pt") as handle:  # type: ignore[no-untyped-call]
        raw_metadata = handle.metadata() or {}
        for key in handle.keys():  # noqa: SIM118 -- safe_open is not a Mapping
            tensors[key] = handle.get_tensor(key)

    moments = ChannelMoments.from_tensors(tensors)
    for key, value in raw_metadata.items():
        if key.startswith("obs:"):
            moments.observations[key[4:]] = int(value)
    return moments
