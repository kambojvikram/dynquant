"""Tensor parallelism: do the ranks between them hold the checkpoint exactly once?

Requires vLLM, so it skips where vLLM is not installed -- but not a GPU, and not
two of them. What tensor parallelism can break in DynQuant is *placement*: which
rows of a fused column-parallel layer, and which packed words of a row-parallel
one, each rank copies out of a checkpoint tensor that spans every rank. A wrong
answer there loads plausible weights into the wrong rank and the model degrades
without erroring, which is the failure worth a test.

Both ranks are built in one process. ``tp_rank`` is an ordinary attribute that
``BasevLLMParameter.__init__`` reads off the process group, and the *size* of the
split comes from ``create_weights``'s arguments rather than the group at all, so
setting the rank by hand supplies exactly the value a second process would have
computed. Spawning two would test ``torch.distributed``, which is not the part
that can be wrong here.

The assertion is always the same shape: reassemble what the ranks hold and require
it to equal the checkpoint. That catches an off-by-one in either direction, since
a gap and an overlap both fail it.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

pytest.importorskip("vllm", reason="the vLLM plugin needs vLLM to import at all")

from dynquant.errors import DynQuantError
from dynquant.integration.vllm_plugin.linear import DynQuantLinearMethod
from dynquant.integration.vllm_plugin.schema import ModuleQuantSpec
from dynquant.quant.pack import row_geometry

TP = 2
IN_FEATURES = 2048
GROUP_SIZE = 128


@pytest.fixture(scope="module", autouse=True)
def _parallel_state():
    """A world of one, purely so vLLM's parameter base class can construct.

    The rank it reads here is overwritten per assertion below. What must be real
    is the *config* context: vLLM's custom ops read it when they are instantiated,
    which happens while a layer is being built.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    with contextlib.ExitStack() as stack:
        stack.enter_context(set_current_vllm_config(VllmConfig()))
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:29531",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(1, 1)
        yield


def spec(bits: int, out_features: int, *, group_size: int = GROUP_SIZE) -> ModuleQuantSpec:
    return ModuleQuantSpec(
        bits=bits, group_size=group_size, symmetric=False, out_features=out_features
    )


def checkpoint_tensors(bits: int, out_features: int, in_features: int = IN_FEATURES):
    """One module's packed words and scales, spanning every rank.

    Values are an ascending ramp rather than a real quantization: this asks where
    bytes land, and a ramp makes a misplacement a mismatch at a nameable index
    instead of two plausible-looking tensors.
    """
    geom = row_geometry(bits, GROUP_SIZE, in_features)
    words = torch.arange(out_features * geom.words_per_row, dtype=torch.int32)
    scales = torch.arange(out_features * geom.num_groups, dtype=torch.bfloat16)
    return words.view(out_features, geom.words_per_row), scales.view(out_features, geom.num_groups)


def build(shards, output_partition_sizes, *, in_per_partition=IN_FEATURES):
    layer = torch.nn.Module()
    method = DynQuantLinearMethod(quant_config=None, shards=shards)
    method.create_weights(
        layer,
        input_size_per_partition=in_per_partition,
        output_partition_sizes=list(output_partition_sizes),
        input_size=IN_FEATURES,
        output_size=sum(output_partition_sizes),
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )
    return layer


def set_rank(layer: torch.nn.Module, rank: int) -> None:
    for name in ("qweight", "scales", "offsets"):
        param = getattr(layer, name)
        param.tp_rank = rank
        param.tp_size = TP


# --------------------------------------------------------------------------
# Column parallel: the rows split
# --------------------------------------------------------------------------


def test_qkv_ranks_tile_the_checkpoint_rows():
    """Three modules of three different widths, split across two ranks.

    Different bits per shard is the case no other quantization method has: with
    one width the flat buffer is a rectangle and vLLM's own ``narrow`` would do.
    Here q, k and v have 256-, 192- and 512-word rows over the same input, so
    every offset in the placement is shard-dependent.
    """
    widths = {"q": (4, 2048), "k": (3, 512), "v": (8, 512)}
    full = {name: checkpoint_tensors(bits, rows) for name, (bits, rows) in widths.items()}
    shards = [(f"l.{name}_proj", spec(bits, rows)) for name, (bits, rows) in widths.items()]
    partitions = [rows // TP for _, rows in widths.values()]

    for rank in range(TP):
        layer = build(shards, partitions)
        set_rank(layer, rank)
        geometry = layer.dynquant_geometry

        offset = 0
        for shard_id, (name, (_, rows)) in enumerate(widths.items()):
            size = rows // TP
            for attr, source in (("qweight", full[name][0]), ("scales", full[name][1])):
                getattr(layer, attr).load_qkv_weight(
                    loaded_weight=source,
                    shard_offset=offset,
                    shard_size=size,
                    shard_id=name,
                    num_heads=1,
                )
            offset += size

            got = geometry.view_qweight(layer.qweight.data, shard_id)
            want = full[name][0][rank * size : (rank + 1) * size]
            assert torch.equal(got, want), f"{name} rows on rank {rank}"

            got_scales = geometry.view_scale(layer.scales.data, shard_id)
            want_scales = full[name][1][rank * size : (rank + 1) * size]
            assert torch.equal(got_scales, want_scales), f"{name} scales on rank {rank}"


def test_kv_heads_replicated_across_ranks_read_the_same_rows():
    """Fewer KV heads than ranks: k and v are copied, not split.

    vLLM signals this with ``num_heads=num_kv_head_replicas``, and getting it
    wrong is silent -- rank 1 would read rows past the end of a tensor that has
    exactly one head's worth, or wrap onto rank 0's. Both ranks must end up with
    byte-identical k.
    """
    k_rows = 128
    k_words, k_scales = checkpoint_tensors(4, k_rows)
    shards = [("l.q_proj", spec(4, 2048)), ("l.k_proj", spec(4, k_rows))]

    held = []
    for rank in range(TP):
        layer = build(shards, [1024, k_rows])
        set_rank(layer, rank)
        layer.qweight.load_qkv_weight(
            loaded_weight=k_words,
            shard_offset=1024,
            shard_size=k_rows,
            shard_id="k",
            num_heads=TP,  # replicated: src index is tp_rank // num_heads == 0
        )
        layer.scales.load_qkv_weight(
            loaded_weight=k_scales,
            shard_offset=1024,
            shard_size=k_rows,
            shard_id="k",
            num_heads=TP,
        )
        held.append(layer.dynquant_geometry.view_qweight(layer.qweight.data, 1).clone())

    assert torch.equal(held[0], k_words)
    assert torch.equal(held[0], held[1])


def test_merged_column_ranks_tile_the_checkpoint_rows():
    """``gate_up_proj``: the same split, reached through vLLM's other hook."""
    gate = checkpoint_tensors(4, 4096)
    up = checkpoint_tensors(3, 4096)
    shards = [("l.gate_proj", spec(4, 4096)), ("l.up_proj", spec(3, 4096))]

    for rank in range(TP):
        layer = build(shards, [2048, 2048])
        set_rank(layer, rank)
        for shard_id, (source, offset) in enumerate(((gate, 0), (up, 2048))):
            layer.qweight.load_merged_column_weight(
                loaded_weight=source[0], shard_offset=offset, shard_size=2048
            )
            got = layer.dynquant_geometry.view_qweight(layer.qweight.data, shard_id)
            assert torch.equal(got, source[0][rank * 2048 : (rank + 1) * 2048])


# --------------------------------------------------------------------------
# Row parallel: the packed word axis splits
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_row_parallel_ranks_tile_the_word_axis(bits):
    """``o_proj``/``down_proj``: every row, this rank's columns.

    Parametrised over all four widths because this is the split that is only
    legal on a group boundary, and the words-per-group differs per width -- 3-bit
    in particular packs 32 values into 3 words, so a split that is clean for 4-bit
    tells you nothing about it.
    """
    rows = 1024
    words, scales = checkpoint_tensors(bits, rows)
    shards = [("l.o_proj", spec(bits, rows))]

    halves, scale_halves = [], []
    for rank in range(TP):
        layer = build(shards, [rows], in_per_partition=IN_FEATURES // TP)
        assert layer.dynquant_row_split is not None
        set_rank(layer, rank)
        layer.qweight.load_row_parallel_weight(words)
        layer.scales.load_row_parallel_weight(scales)
        geometry = layer.dynquant_geometry
        halves.append(geometry.view_qweight(layer.qweight.data, 0).clone())
        scale_halves.append(geometry.view_scale(layer.scales.data, 0).clone())

    assert torch.equal(torch.cat(halves, dim=1), words)
    assert torch.equal(torch.cat(scale_halves, dim=1), scales)


def test_row_parallel_split_off_a_group_boundary_is_refused_at_build_time():
    """Three groups over two ranks. Named module, before any weight is read."""
    in_features = 3 * GROUP_SIZE
    shards = [("l.down_proj", spec(4, 512))]
    method = DynQuantLinearMethod(quant_config=None, shards=shards)
    with pytest.raises(DynQuantError, match="down_proj"):
        method.create_weights(
            torch.nn.Module(),
            input_size_per_partition=in_features // TP,
            output_partition_sizes=[512],
            input_size=in_features,
            output_size=512,
            params_dtype=torch.bfloat16,
            weight_loader=None,
        )


def test_row_parallel_fused_layer_is_refused():
    """More than one width on the reduction axis has no single word split."""
    shards = [("l.a", spec(4, 512)), ("l.b", spec(3, 512))]
    method = DynQuantLinearMethod(quant_config=None, shards=shards)
    with pytest.raises(DynQuantError, match="more than one"):
        method.create_weights(
            torch.nn.Module(),
            input_size_per_partition=IN_FEATURES // TP,
            output_partition_sizes=[256, 256],
            input_size=IN_FEATURES,
            output_size=512,
            params_dtype=torch.bfloat16,
            weight_loader=None,
        )
