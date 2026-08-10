"""The MoE dispatch a packed bank can serve without leaving the grouped path.

Transformers 5.14 stopped calling an ``*Experts`` class's own ``forward``. It wraps it
and picks from ``config._experts_implementation`` -- ``grouped_mm`` (the default),
``batched_mm``, ``deepgemm``, ``sonicmoe``, or ``eager`` -- and only ``eager`` is the
per-expert indexing loop that :class:`~dynquant.runtime.linear.DynQuantExpertBank`
answers. ``grouped_mm`` hands the whole bank to ``torch._grouped_mm``, asks it to
``transpose``, and gets an ``AttributeError`` from ``nn.Module``.

So until now loading a packed LFM2 moved the model to ``eager``, and that move is not
free. Measured on LFM2.5-8B-A1B, ``eager`` and ``grouped_mm`` disagree on **1.24% of
teacher-forced tokens** -- 0.29x the effect of quantizing the model. Which means an
accuracy number for a packed artifact was not comparable to a bf16 number for the same
model unless the bf16 side was also moved, and every published figure had to carry that
condition.

This closes it from the other side. ``dynquant_experts_forward`` is
``grouped_mm_experts_forward`` line for line -- the same sort, the same offsets, the same
sentinel masking, the same inverse permutation, the same ``view(tokens, k, dim).sum(1)``
reduction -- with exactly one substitution: where the grouped kernel reads a segment of a
dense ``[E, out, in]`` tensor, this reads ``bank[e]``. A packed model therefore stays on
the reduction order the default dispatch uses, and the only thing separating it from bf16
is the quantization.

Why the reduction order is the thing worth copying
--------------------------------------------------
It would have been easier to keep ``eager`` and call the difference noise. The
disagreement is not in the GEMMs, which are per-expert either way; it is that ``eager``
accumulates each expert's contribution into the output as it goes, in the model's own
bf16, while the grouped path collects all ``k`` contributions per token and sums them in
one reduction. A top-k router then turns a small numeric difference into a discrete one
and 22 layers compound it. Copying the reduction is most of what closes the gap.

What this is not
----------------
Not the CUDA kernel. Each active expert still costs one dequantization of its slice,
and the segment loop is data-dependent -- it reads its own offsets -- so this breaks
``torch.compile(fullgraph=True)`` and CUDA-graph capture, which is precisely what
``grouped_mm``'s comment about ``torch.unique`` is avoiding. The fused kernel replaces
:func:`_grouped_linear_packed` and nothing above it: ``QuantTensor.rows()`` is already
the segment addressing it needs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import nn

__all__ = [
    "DISPATCH_NAME",
    "ExpertsModule",
    "dynquant_experts_forward",
    "register_experts_dispatch",
    "use_dynquant_experts",
]

_log = logging.getLogger(__name__)

DISPATCH_NAME = "dynquant"
"""The key registered into ``ALL_EXPERTS_FUNCTIONS`` and written to the config."""


class ExpertsModule(Protocol):
    """What transformers passes as ``self`` to an experts implementation.

    Written down rather than typed as ``nn.Module`` because ``nn.Module`` says nothing
    about any of it -- every attribute below is added by the ``use_experts_implementation``
    decorator or by the model class, and typing ``self`` loosely turns each read into an
    unchecked ``Any``. This is also the contract a new architecture has to meet for the
    dispatch to serve it, which is worth being able to point at.
    """

    num_experts: int
    has_gate: bool
    has_bias: bool
    is_transposed: bool
    gate_up_proj: Any
    up_proj: Any
    down_proj: Any
    gate_up_proj_bias: torch.Tensor
    up_proj_bias: torch.Tensor
    down_proj_bias: torch.Tensor

    def act_fn(self, hidden: torch.Tensor) -> torch.Tensor: ...

    def _apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor: ...


def _expert_weight(bank: Any, expert: int, dtype: torch.dtype) -> torch.Tensor:
    """One expert's matrix, from a packed bank or from a plain tensor.

    Both cases are live in the same model. A ``--map`` can leave one layer's bank dense
    while packing another's, and a bank whose width the allocator set to 16 is never
    packed at all, so this dispatch has to serve a mixed model or it cannot be the
    model-wide default.
    """
    weight: torch.Tensor = bank[expert]
    return weight.to(dtype)


def _grouped_linear_packed(
    x: torch.Tensor,
    bank: Any,
    offsets: list[int],
    *,
    bias: torch.Tensor | None,
    is_transposed: bool,
    out_features: int,
) -> torch.Tensor:
    """``_grouped_linear`` over segments, taking each segment's weight from ``bank[e]``.

    ``offsets[e]`` to ``offsets[e + 1]`` is expert ``e``'s band of the sorted rows, so an
    expert no token reached costs nothing: it is skipped, never dequantized. That is the
    difference from ``grouped_mm``, which passes the whole bank and lets the kernel's
    offsets select -- it can afford to, having no per-expert decode to skip.

    Rows past ``offsets[-1]`` are the EP sentinels. The grouped kernel leaves them
    uninitialised and relies on a post-mask; this zeroes them, because a Python loop that
    skipped them would otherwise hand ``torch.empty``'s contents to the next projection,
    and one NaN there survives the mask that only covers the last one.
    """
    out = x.new_zeros((x.size(0), out_features))
    for expert in range(len(offsets) - 1):
        start, stop = offsets[expert], offsets[expert + 1]
        if start == stop:
            continue
        weight = _expert_weight(bank, expert, x.dtype)
        rows = x[start:stop]
        out[start:stop] = rows @ weight if is_transposed else F.linear(rows, weight)
    if bias is not None:
        out.add_(bias)
    return out


def _segment_offsets(sorted_expert_ids: torch.Tensor, num_experts: int) -> list[int]:
    """Where each expert's band starts, as Python ints, plus the end of the last one.

    Takes the ids *unclamped*, and the ``[:num_experts]`` slice is what makes that safe:
    an expert-parallel sentinel lands in a bin past the end and is dropped, so the last
    offset is the count of pairs this rank actually holds and every sentinel row sits
    beyond it. Passing clamped ids here instead would count sentinels as real members of
    whichever expert they were clamped to, and since the bands index a sorted array, one
    over-wide band displaces all the ones after it.

    Python ints and not a tensor: the loop above indexes with them, and keeping them on
    device would sync once per expert instead of once per layer. The single
    ``.tolist()`` is the data-dependent step this implementation cannot avoid and the
    fused kernel does not have to take.
    """
    counts = torch.bincount(sorted_expert_ids, minlength=num_experts)[:num_experts]
    return [0, *torch.cumsum(counts, dim=0).tolist()]


def dynquant_experts_forward(
    self: ExpertsModule,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """``grouped_mm_experts_forward`` with ``bank[e]`` where the segment read used to be.

    Kept structurally parallel to the transformers original on purpose, down to the
    variable names, because the whole value of it is that the two agree -- a reader
    checking that claim should be able to diff them, and a future transformers release
    that changes the reduction should be visible as a diff rather than as a drift in an
    accuracy table.
    """
    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens = hidden_states.size(0)
    hidden_dim = hidden_states.size(-1)
    num_experts = self.num_experts

    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)

    expert_ids_g, perm = torch.sort(expert_ids)
    selected_hidden_states_g = hidden_states[perm // num_top_k]
    sample_weights_g = sample_weights[perm]

    # Sentinels sort to the tail, so the count of real pairs is where the bands end --
    # and the bands must be counted *before* the clamp. Folding a sentinel into a real
    # expert's bin, which an earlier version did, does not merely misattribute it: the
    # bands are offsets into a sorted array, so widening bin 0 by rows that physically
    # sit at the tail shifts every later band off its own rows. A two-expert test with
    # one sentinel per token returned expert 0's answer for both.
    sentinel_mask = (expert_ids_g >= num_experts).unsqueeze(-1)
    offsets = _segment_offsets(expert_ids_g, num_experts)
    # Clamped only so the bias gather below stays in range; the bands are already fixed.
    expert_ids_g = expert_ids_g.clamp(max=num_experts - 1)

    selected_hidden_states_g = selected_hidden_states_g.masked_fill(sentinel_mask, 0.0)

    if self.has_gate:
        weights, biases = self.gate_up_proj, self.gate_up_proj_bias if self.has_bias else None
    else:
        weights, biases = self.up_proj, self.up_proj_bias if self.has_bias else None

    proj_out = _grouped_linear_packed(
        selected_hidden_states_g,
        weights,
        offsets,
        bias=None if biases is None else biases[expert_ids_g],
        is_transposed=self.is_transposed,
        out_features=_out_features(weights, is_transposed=self.is_transposed),
    )

    proj_out = self._apply_gate(proj_out) if self.has_gate else self.act_fn(proj_out)

    down = self.down_proj
    proj_out = _grouped_linear_packed(
        proj_out,
        down,
        offsets,
        bias=None if not self.has_bias else self.down_proj_bias[expert_ids_g],
        is_transposed=self.is_transposed,
        out_features=hidden_dim,
    )

    weighted_out = proj_out * sample_weights_g.unsqueeze(-1)
    weighted_out = weighted_out.masked_fill(sentinel_mask, 0.0)

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)
    weighted_out = weighted_out[inv_perm]

    final_hidden_states = weighted_out.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
    return final_hidden_states.to(hidden_states.dtype)


def _out_features(bank: Any, *, is_transposed: bool) -> int:
    """The projection's output width, from a packed bank or a dense parameter alike.

    One expression for both because ``_PackedModule.shape`` is defined to return the
    *logical* ``[E, out, in]``, not the flattened word count it stores. An earlier version
    branched on ``hasattr(bank, "logical_shape")`` to say the same thing twice; the branch
    could not be made to fail, which is the tell that it was not a distinction.

    ``is_transposed`` is the real content here: transformers stores a bank as
    ``[E, in, out]`` when the module multiplies ``x @ w`` and ``[E, out, in]`` when it
    calls ``F.linear``, and reading the wrong end of that is a silent transpose.
    """
    shape = tuple(bank.shape)
    return int(shape[-1] if is_transposed else shape[-2])


def register_experts_dispatch() -> bool:
    """Put ``dynquant`` in ``ALL_EXPERTS_FUNCTIONS``. Idempotent; False if unavailable.

    False rather than an exception because the caller is a load path with a working
    fallback: an older transformers has no such interface, and moving the model to
    ``eager`` is still correct there -- just not comparable. Whether registration
    happened is therefore something the report has to say out loud, which is why
    :func:`use_dynquant_experts` returns which dispatch it landed on.
    """
    try:
        from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
    except ImportError:
        return False
    if DISPATCH_NAME not in ALL_EXPERTS_FUNCTIONS:
        ALL_EXPERTS_FUNCTIONS.register(DISPATCH_NAME, dynquant_experts_forward)
    return True


def use_dynquant_experts(model: nn.Module) -> str | None:
    """Move a transformers MoE onto the dispatch a packed bank can serve.

    Prefers ``dynquant``, which keeps the grouped path's reduction order, and falls back
    to ``eager`` where the interface does not exist. Returns the implementation moved
    *away from*, or ``None`` if nothing moved -- a model with no transformers config, one
    already on a dispatch this serves, and one on an older transformers with no dispatch
    at all are the same answer to the caller, which is "no action of mine to report".
    """
    from dynquant.runtime.linear import use_eager_experts

    config = getattr(model, "config", None)
    if config is None:
        return None
    current = getattr(config, "_experts_implementation", None)
    if current is None or current == DISPATCH_NAME:
        return None
    if not register_experts_dispatch():
        return use_eager_experts(model)

    setter = getattr(model, "set_experts_implementation", None)
    if callable(setter):
        setter(DISPATCH_NAME)
    else:  # pragma: no cover - every model with the attribute also has the setter
        config._experts_implementation = DISPATCH_NAME
    _log.info(
        "moved the experts dispatch from %r to %r: a packed bank is indexed one expert "
        "at a time, and %r passes the whole bank to a grouped matmul",
        current,
        DISPATCH_NAME,
        current,
    )
    return str(current)
