"""One opaque op for the concatenation at the output of a fused layer.

A fused serving layer -- ``qkv_proj``, ``gate_up_proj``, ``in_proj_qkvz`` -- is always
consumed by a split. Every other quantization method holds one weight matrix and
does one matmul, so that split is a genuine view into one wide buffer. DynQuant
holds a separate width per shard, so it does one matmul per shard and joins them,
and ``torch.cat`` immediately followed by ``torch.split`` is a pattern inductor
cancels: the split is rewritten to return the pre-cat buffers directly.

That rewrite is arithmetically correct and still breaks vLLM. ``@support_torch_compile``
splits the model into pieces and records each boundary tensor's shape *and stride*
from fake-tensor propagation, which runs before the cancellation. On Qwen3.5 the
gated delta net's ``z`` is recorded as a stride-8192 view into the fused qkvz output
and arrives as a contiguous stride-2048 buffer, so the consumer's index math --
``128*(x0 % 16) + 8192*(x0 // 16)`` -- reads four times past the end of the
allocation. With ``VLLM_LOGGING_LEVEL=DEBUG`` that is an ``assert_size_stride``
failure naming both strides; without it, an illegal memory access, or silently
garbage logits if the read lands inside another live allocation.

Making the join opaque removes the pattern to cancel. Inductor must materialise one
contiguous buffer, so the recorded strides describe what the consumer actually gets.
The cost is one fusion boundary per fused layer, against a copy that was already
being paid.

The diagnosis above is vLLM's because vLLM is where it was caught, but nothing in it
is vLLM-specific: the rewrite is inductor's, and any server that traces a piecewise
graph and records boundary strides ahead of it inherits the same bug. SGLang does.
Hence this module lives in ``serving_common`` -- and it *has* to, for a second and
harder reason: ``torch.library.custom_op`` registers ``dynquant::fused_shard_concat``
in a process-global namespace, so a per-plugin copy would raise on whichever import
came second.
"""

from __future__ import annotations

import torch

__all__ = ["fused_shard_concat"]


@torch.library.custom_op("dynquant::fused_shard_concat", mutates_args=())
def fused_shard_concat(parts: list[torch.Tensor]) -> torch.Tensor:
    """``torch.cat(parts, dim=-1)``, but not visible to the inductor rewriter."""
    return torch.cat(parts, dim=-1)


@fused_shard_concat.register_fake
def _fused_shard_concat_fake(parts: list[torch.Tensor]) -> torch.Tensor:
    # Deliberately `new_empty` rather than anything view-like: the point of this op
    # is that the traced layout is a single contiguous allocation, and the fake is
    # what the piece boundary records.
    total = sum(part.shape[-1] for part in parts)
    return parts[0].new_empty((*parts[0].shape[:-1], total))
