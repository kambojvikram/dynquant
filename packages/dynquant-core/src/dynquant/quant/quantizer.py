"""Apply a bit map to a model.

This is the step that turns an allocation into weights. For every module the
allocator assigned a width, the weight is encoded with the MSE-optimal clipping
search and the result is recorded, so the per-layer reconstruction error is
available before anything is evaluated -- a layer whose error is orders of
magnitude worse than its peers is visible here rather than as an unexplained
accuracy drop later.

``in_place=True`` writes the dequantized weights back over the originals. That is
*simulated storage*: the numbers are exactly the ones a packed checkpoint would
reconstruct, so accuracy measured this way is exactly the accuracy of the real
thing, but the tensor still occupies fp16 in memory and nothing gets faster. It
exists because accuracy and speed are separable questions and the first one does
not have to wait for the kernels. What it must never be is *reported* as the
second: a benchmark run against an in-place model measures fp16.

Tied weights are handled by the caller's bit map rather than here. The graph lists
one representative per tied group, so the shared storage is encoded once and
``copy_`` propagates it to every alias. Listing both names is not caught here and
does not raise: at one width the second pass is a no-op, because the values already
sit exactly on the grid and re-encoding recovers the same codes. At *different*
widths it is order-dependent -- the later name in the sorted walk wins -- and the
report would price the tensor at one width while the weights carry another. Keeping
one representative per tied group is what prevents that; the driver cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from dynquant._logging import get_logger
from dynquant.constants import DEFAULT_GROUP_SIZE
from dynquant.errors import DynQuantError

from .device import quantize_tensor, resolve_compute_device
from .grid import CLIP_CANDIDATES

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from torch import Tensor, nn

__all__ = [
    "LayerQuantResult",
    "QuantizationReport",
    "quantize_model",
    "resolves_to_weight",
    "target_tensor",
]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LayerQuantResult:
    """What one module cost."""

    name: str
    bits: int
    num_params: int
    rmse: float
    relative_error: float
    """RMSE divided by the weight's own RMS. Comparable across layers; raw RMSE is
    not, because weight scales differ by an order of magnitude between roles."""
    clipped_fraction: float
    clip_improvement: float


@dataclass(slots=True)
class QuantizationReport:
    layers: dict[str, LayerQuantResult] = field(default_factory=dict)
    skipped: tuple[str, ...] = ()

    def worst(self, count: int = 5) -> list[LayerQuantResult]:
        return sorted(self.layers.values(), key=lambda r: -r.relative_error)[:count]

    def summary(self) -> str:
        if not self.layers:
            return "quantized nothing"
        errors = sorted(r.relative_error for r in self.layers.values())
        median = errors[len(errors) // 2]
        lines = [
            f"quantized {len(self.layers)} modules; "
            f"relative error median {median:.4f}, max {errors[-1]:.4f}",
            "  worst layers:",
        ]
        lines.extend(
            f"    {r.name} @{r.bits}b  rel_err {r.relative_error:.4f}  "
            f"clipped {r.clipped_fraction:.1%}"
            for r in self.worst()
        )
        if self.skipped:
            lines.append(f"  {len(self.skipped)} skipped: {', '.join(self.skipped[:5])}")
        return "\n".join(lines)


def quantize_model(
    model: torch.nn.Module,
    bits: Mapping[str, int],
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    symmetric: bool = False,
    candidates: Sequence[float] = CLIP_CANDIDATES,
    channel_weights: Mapping[str, torch.Tensor] | None = None,
    in_place: bool = True,
    compute_device: str | torch.device | None = "auto",
    progress: Callable[[int, int], None] | None = None,
) -> QuantizationReport:
    """Encode every module named in ``bits`` and write the result back.

    ``compute_device`` decides where the encoding arithmetic runs, independently of
    where the model sits; it defaults to the accelerator when there is one. A model
    on the CPU is still a model on the CPU afterwards -- weights move one at a time
    and the reconstruction is copied back across the device boundary -- so this
    changes how long the call takes and nothing about what it produces. Pass
    ``None`` to keep the arithmetic on the weights' own device. See
    :mod:`dynquant.quant.device`.

    ``channel_weights`` maps a module name to a per-input-channel weight of length
    ``in_features`` -- in practice ``E[x_c^2]`` from the training-time moments -- and
    switches that module's clip search from plain MSE to the error the layer output
    actually sees. It is opt-in and all-or-nothing: when supplied it must cover every
    name in ``bits``. That is deliberate rather than convenient. The allocator prices
    a bit map against one clip objective, and a module encoded on a different
    objective than it was priced with is mispriced in a way nothing downstream can
    detect -- the checkpoint is the right size and simply worse. A module with no
    usable moments is not an exception to feed through the gap: pass it a vector of
    ones, which is the unweighted objective written explicitly.

    Raises:
        DynQuantError: A name in the bit map does not resolve to a module with a
            weight. Raised rather than skipped: the bit map came from the same
            graph as the model, so a miss means the two have diverged, and
            silently leaving that tensor at fp16 would produce a checkpoint
            larger than the target with no indication why. Also raised when
            ``channel_weights`` is supplied but does not cover ``bits``, or when a
            weight vector's length does not match the module's ``in_features``.
    """
    report = QuantizationReport()
    total = len(bits)
    device = resolve_compute_device(compute_device)

    if channel_weights is not None:
        missing = sorted(set(bits) - set(channel_weights))
        if missing:
            raise DynQuantError(
                f"channel_weights is missing {len(missing)} of {len(bits)} modules in the "
                f"bit map, starting with {missing[:5]}. Pass a vector of ones for any "
                f"module that should keep the unweighted clip objective."
            )

    for index, (name, width) in enumerate(sorted(bits.items())):
        weight = target_tensor(model, name)
        cweight = None if channel_weights is None else channel_weights[name]
        if cweight is not None and cweight.numel() != weight.shape[-1]:
            raise DynQuantError(
                f"{name}: channel weight has {cweight.numel()} entries but the weight has "
                f"{weight.shape[-1]} input channels"
            )

        with torch.no_grad():
            original = weight.detach()
            quantized, search = quantize_tensor(
                original,
                bits=width,
                group_size=group_size,
                symmetric=symmetric,
                candidates=candidates,
                compute_dtype=original.dtype
                if original.dtype in (torch.float16, torch.bfloat16)
                else None,
                channel_weight=cweight,
                device=device,
            )
            # Scored where the encoding happened rather than where the weight
            # lives: the error metrics are a reduction over the same tensors the
            # search just built, and dragging them back first would spend the
            # transfer to compute two scalars.
            recon = quantized.dequantize(dtype=torch.float32)
            reference = original.to(device=recon.device, dtype=torch.float32)
            rmse = float(torch.sqrt(torch.mean((reference - recon) ** 2)))
            rms = float(torch.sqrt(torch.mean(reference**2)))

            if in_place:
                # copy_ rather than assignment: it preserves both the parameter
                # object and, critically, any tying -- an lm_head sharing storage
                # with the embedding sees the change without being quantized again.
                # It also crosses devices and casts, which is what carries the
                # reconstruction back to a model the accelerator never held.
                weight.copy_(recon)

            del recon, reference

        report.layers[name] = LayerQuantResult(
            name=name,
            bits=width,
            num_params=original.numel(),
            rmse=rmse,
            relative_error=rmse / rms if rms > 0 else 0.0,
            clipped_fraction=search.clipped_fraction,
            clip_improvement=search.improvement,
        )
        if progress is not None:
            progress(index + 1, total)

    _log.info("%s", report.summary())
    return report


def resolves_to_weight(model: nn.Module, name: str) -> bool:
    """Whether ``name`` addresses a tensor :func:`quantize_model` would encode.

    The pre-flight check every command runs before it starts quantizing needs to answer
    exactly the question the quantizer will answer later, and the only way to be sure of
    that is to ask the quantizer. A separate lookup drifts: this one did, using
    ``get_submodule`` alone, which is right for every module in the map and wrong for
    every batched MoE expert bank -- so on LFM2.5-8B-A1B the guard refused a map that
    :func:`quantize_model` would have applied without complaint, and the refusal named the
    banks as modules the model does not have.

    Cheap enough to run per name: resolution is attribute lookups, no arithmetic and no
    copies.
    """
    try:
        target_tensor(model, name)
    except DynQuantError:
        return False
    return True


def target_tensor(model: nn.Module, name: str) -> Tensor:
    """The tensor ``name`` addresses -- a module's ``weight``, or a bare parameter.

    Almost every name in a bit map resolves to a module that owns a ``weight``. A
    batched MoE expert bank does not: it keeps its experts as 3-D parameters on the
    bank itself, so ``model.layers.0.mlp.experts.gate_up_proj`` names a
    :class:`~torch.nn.Parameter` and ``get_submodule`` raises for it.

    That name is not a workaround -- it is what the state dict, the graph and the
    stats file all call the tensor. So the lookup widens to match the name rather
    than the name narrowing to match the lookup.

    Raises:
        DynQuantError: The name resolves to nothing, or to something with no
            quantizable tensor. Never skipped. The bit map came from the same graph
            as the model, so a miss means the two have diverged, and leaving that
            tensor at fp16 would produce a checkpoint over its target with nothing
            saying why.
    """
    try:
        module: nn.Module | None = model.get_submodule(name)
    except AttributeError:
        module = None

    if module is not None:
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            return weight
        raise DynQuantError(
            f"{name!r} is a {type(module).__name__} with no weight to quantize. If this "
            f"is a batched MoE expert bank, the bit map should name its parameters "
            f"individually (e.g. {name}.gate_up_proj), not the bank."
        )

    parent_name, _, leaf = name.rpartition(".")
    try:
        parent = model.get_submodule(parent_name)
    except AttributeError as exc:
        raise DynQuantError(
            f"bit map names {name!r}, which is not a module or a parameter of this model"
        ) from exc

    param = getattr(parent, leaf, None)
    if param is None:
        # Nothing of that name exists. Distinct from the case below -- there the
        # attribute is present and merely unquantizable -- and the two want
        # different answers from a reader: one is a stale bit map, the other a
        # bit map naming something that was never a weight.
        raise DynQuantError(
            f"bit map names {name!r}, which is not a module or a parameter of this model"
        )
    if isinstance(param, torch.Tensor) and param.ndim >= 2:
        return param
    raise DynQuantError(
        f"bit map names {name!r}; {type(parent).__name__} has no quantizable tensor "
        f"called {leaf!r} (found {type(param).__name__})"
    )
