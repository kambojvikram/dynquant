"""The join at the output of a fused layer must stay opaque to the compiler.

This is the CPU half of a bug that only bites on a GPU. A fused vLLM layer is
always consumed by a split, and ``split(cat(...))`` is a pair inductor cancels --
correctly, arithmetically, and fatally, because vLLM records each piece boundary's
stride from fake-tensor propagation, which runs *before* the cancellation. See
:mod:`dynquant.integration.vllm_plugin.fuse` for the full account.

What is checkable without a GPU is the contract the fix rests on: the op computes a
concatenation, its fake declares a single contiguous buffer, and it survives tracing
as one call rather than being decomposed back into the pattern that gets cancelled.
The end-to-end guard -- vLLM under ``VLLM_LOGGING_LEVEL=DEBUG``, which is what turns
inductor's ``assert_size_stride`` on -- lives with the GPU scripts.
"""

from __future__ import annotations

import pytest
import torch

from dynquant.integration.vllm_plugin.fuse import fused_shard_concat


def shards(rows: int = 7) -> list[torch.Tensor]:
    """Qwen3.5's ``in_proj_qkvz`` widths -- the layer the bug was found on."""
    torch.manual_seed(0)
    return [torch.randn(rows, width) for width in (2048, 2048, 2048, 2048)]


def test_matches_cat():
    parts = shards()
    torch.testing.assert_close(fused_shard_concat(parts), torch.cat(parts, dim=-1))


def test_output_is_one_contiguous_buffer():
    out = fused_shard_concat(shards())
    assert out.shape == (7, 8192)
    assert out.is_contiguous()
    # The property the consumer depends on: a narrow of the join is a *view*, so
    # its row stride is the full fused width and not the shard's own.
    assert out[:, 6144:].stride() == (8192, 1)


def test_fake_declares_the_same_layout():
    """The traced strides are the fake's, and vLLM bakes them into a boundary.

    A fake that returned anything view-like, or that inferred the width from
    ``parts[0]`` alone, would trace a layout the real op does not produce -- which
    is the failure mode being fixed, reintroduced one level down.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        parts = [torch.empty(7, 2048), torch.empty(7, 1024)]
        out = torch.ops.dynquant.fused_shard_concat(parts)
    assert out.shape == (7, 3072)
    assert out.stride() == (3072, 1)


def test_uneven_widths_and_leading_dims():
    parts = [torch.randn(2, 5, w) for w in (128, 384, 64)]
    out = fused_shard_concat(parts)
    assert out.shape == (2, 5, 576)
    torch.testing.assert_close(out, torch.cat(parts, dim=-1))


@pytest.mark.parametrize("dynamic", [True, False])
def test_survives_tracing_as_a_single_opaque_call(dynamic):
    """The regression guard: one custom-op node, and no ``aten.cat`` to cancel.

    Captured off dynamo's own graph rather than inductor's output, so this asserts
    the property the fix needs -- the pattern is not *present* to be rewritten --
    without depending on which rewrites a given inductor version performs.
    """
    graphs: list[torch.fx.GraphModule] = []

    def capture(gm: torch.fx.GraphModule, example_inputs):
        graphs.append(gm)
        return gm.forward

    def join_then_split(a, b, c, d):
        return torch.split(fused_shard_concat([a, b, c, d]), 2048, dim=-1)

    compiled = torch.compile(join_then_split, backend=capture, fullgraph=True, dynamic=dynamic)
    got = compiled(*shards())
    torch.testing.assert_close(torch.cat(got, dim=-1), torch.cat(shards(), dim=-1))

    targets = [node.target for graph in graphs for node in graph.graph.nodes]
    assert torch.ops.dynquant.fused_shard_concat.default in targets
    assert not any(target is torch.ops.aten.cat.default for target in targets)
