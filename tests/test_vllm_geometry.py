"""The flat-buffer and tensor-parallel arithmetic behind the vLLM plugin.

Everything here runs on CPU with no vLLM installed, which is the point of keeping
:mod:`dynquant.integration.vllm_plugin.geometry` free of vLLM imports: the layout
maths is what breaks silently -- an off-by-one offset loads plausible weights into
the wrong rows -- and it is exactly the part that does not need a GPU to check.
"""

from __future__ import annotations

import pytest

from dynquant.constants import PER_ROW_GROUP_SIZE
from dynquant.errors import DynQuantError
from dynquant.integration.vllm_plugin.geometry import (
    FusedPackedGeometry,
    ShardSpec,
    match_shards_to_partitions,
    row_parallel_split,
)
from dynquant.integration.vllm_plugin.schema import ModuleQuantSpec
from dynquant.quant.pack import row_geometry


def qkv(*, q_bits=4, k_bits=4, v_bits=4, hidden=1024, kv=256, group_size=128):
    """A Llama-shaped attention projection, per-shard widths configurable."""
    return [
        ShardSpec("l.q_proj", q_bits, group_size, hidden, hidden, False),
        ShardSpec("l.k_proj", k_bits, group_size, kv, hidden, False),
        ShardSpec("l.v_proj", v_bits, group_size, kv, hidden, False),
    ]


# --------------------------------------------------------------------------
# Flat layout
# --------------------------------------------------------------------------


def test_shards_tile_the_buffer_without_gap_or_overlap():
    geometry = FusedPackedGeometry(qkv())
    cursor = 0
    for plan in geometry:
        assert plan.qweight_offset == cursor
        cursor += plan.qweight_numel
    assert cursor == geometry.qweight_numel


def test_mixed_widths_are_the_case_that_needs_a_flat_buffer():
    """q at 4 bits and k at 3 have different words per row -- no rectangle holds both."""
    geometry = FusedPackedGeometry(qkv(q_bits=4, k_bits=3, v_bits=3))
    words = {plan.spec.name: plan.words_per_row for plan in geometry}
    assert words["l.q_proj"] != words["l.k_proj"]
    assert not geometry.is_uniform

    expected = sum(
        spec.out_features * row_geometry(spec.bits, spec.group_size, spec.in_features).words_per_row
        for spec in qkv(q_bits=4, k_bits=3, v_bits=3)
    )
    assert geometry.qweight_numel == expected


def test_uniform_widths_are_still_flat_but_report_uniform():
    geometry = FusedPackedGeometry(qkv())
    assert geometry.is_uniform
    assert geometry.total_out_features == 1024 + 256 + 256


def test_row_offsets_are_the_layer_output_space():
    geometry = FusedPackedGeometry(qkv())
    assert [plan.row_offset for plan in geometry] == [0, 1024, 1280]


def test_single_shard_layer_degenerates_to_the_whole_buffer():
    geometry = FusedPackedGeometry([ShardSpec("l.o_proj", 4, 128, 1024, 1024, False)])
    assert len(geometry) == 1
    plan = geometry[0]
    assert plan.qweight_offset == 0
    assert plan.qweight_numel == geometry.qweight_numel


def test_views_are_views_not_copies():
    torch = pytest.importorskip("torch")
    geometry = FusedPackedGeometry(qkv(q_bits=4, k_bits=3, v_bits=3))
    flat = torch.zeros(geometry.qweight_numel, dtype=torch.int32)

    view = geometry.view_qweight(flat, 1)
    assert view.shape == geometry[1].qweight_shape
    view.fill_(7)
    # Writing through the view must land in the flat buffer, and only in k's span.
    assert bool((flat[geometry[1].qweight_slice] == 7).all())
    assert bool((flat[geometry[0].qweight_slice] == 0).all())
    assert bool((flat[geometry[2].qweight_slice] == 0).all())


def test_scale_views_use_group_counts_not_word_counts():
    torch = pytest.importorskip("torch")
    geometry = FusedPackedGeometry(qkv(q_bits=4, k_bits=3, v_bits=3))
    flat = torch.zeros(geometry.scale_numel, dtype=torch.float16)
    for shard_id, plan in enumerate(geometry):
        view = geometry.view_scale(flat, shard_id)
        assert view.shape == (plan.spec.out_features, plan.num_groups)
    # Every shard has 1024 inputs at group 128, so scales are shape-identical even
    # though the packed words are not. That asymmetry is why they are two buffers.
    assert len({plan.num_groups for plan in geometry}) == 1


def test_wrong_sized_buffer_is_refused_rather_than_reinterpreted():
    torch = pytest.importorskip("torch")
    geometry = FusedPackedGeometry(qkv())
    with pytest.raises(DynQuantError, match="1-D"):
        geometry.view_qweight(torch.zeros(geometry.qweight_numel + 1, dtype=torch.int32), 0)
    with pytest.raises(DynQuantError, match="1-D"):
        geometry.view_qweight(torch.zeros(4, geometry.qweight_numel // 4, dtype=torch.int32), 0)


def test_by_name_names_the_alternatives_when_it_misses():
    geometry = FusedPackedGeometry(qkv())
    assert geometry.by_name("l.k_proj").shard_id == 1
    with pytest.raises(DynQuantError, match="q_proj"):
        geometry.by_name("l.gate_proj")


def test_empty_and_degenerate_shards_are_rejected():
    with pytest.raises(DynQuantError, match="at least one shard"):
        FusedPackedGeometry([])
    with pytest.raises(DynQuantError, match="out_features must be positive"):
        FusedPackedGeometry([ShardSpec("l.q_proj", 4, 128, 0, 1024, False)])


# --------------------------------------------------------------------------
# Pairing the checkpoint's modules with vLLM's partitions
# --------------------------------------------------------------------------


def spec(bits=4, group_size=128, symmetric=False):
    return ModuleQuantSpec(bits=bits, group_size=group_size, symmetric=symmetric)


def test_separate_modules_pair_positionally_with_the_partitions():
    shards = [("l.q_proj", spec(4)), ("l.k_proj", spec(3)), ("l.v_proj", spec(3))]
    matched = match_shards_to_partitions(shards, [1024, 256, 256], 1024)
    assert [s.name for s in matched] == ["l.q_proj", "l.k_proj", "l.v_proj"]
    assert [s.bits for s in matched] == [4, 3, 3]
    assert [s.out_features for s in matched] == [1024, 256, 256]


def test_a_checkpoint_fused_on_disk_collapses_to_one_shard():
    """Phi-3 writes ``qkv_proj`` as a single tensor, so one width covers all rows."""
    matched = match_shards_to_partitions([("l.qkv_proj", spec(4))], [1024, 256, 256], 1024)
    assert len(matched) == 1
    assert matched[0].out_features == 1536


def test_symmetry_survives_the_pairing():
    """The reconstruction reads ``symmetric`` off the shard, not off the config.

    Dropping it here is not a crash and not obviously wrong output: an asymmetric
    decode of a symmetric table adds an offset buffer that is all zeros, so the
    model still runs and still produces text. It went unnoticed until a forward
    pass asked for the attribute that had never been carried.
    """
    for symmetric in (True, False):
        shards = [("l.q_proj", spec(symmetric=symmetric)), ("l.k_proj", spec(symmetric=symmetric))]
        matched = match_shards_to_partitions(shards, [1024, 256], 1024)
        assert [s.symmetric for s in matched] == [symmetric, symmetric]
        # And through the fused-on-disk branch, which is a separate construction.
        (collapsed,) = match_shards_to_partitions(shards[:1], [1024, 256], 1024)
        assert collapsed.symmetric is symmetric


def test_a_partition_count_the_map_cannot_explain_is_an_error():
    shards = [("l.q_proj", spec()), ("l.k_proj", spec())]
    with pytest.raises(DynQuantError, match="does not describe this model"):
        match_shards_to_partitions(shards, [1024, 256, 256], 1024)


# --------------------------------------------------------------------------
# Tensor parallelism
# --------------------------------------------------------------------------


def test_row_parallel_partitions_tile_the_full_row():
    full = row_geometry(4, 128, 4096)
    split = row_parallel_split(bits=4, group_size=128, in_features=4096, tp_size=4)
    assert split.in_features_per_partition == 1024
    assert split.words_per_partition * 4 == full.words_per_row
    assert split.groups_per_partition * 4 == full.num_groups

    covered = [split.word_slice(rank) for rank in range(4)]
    assert covered[0].start == 0
    assert covered[-1].stop == full.words_per_row
    assert all(a.stop == b.start for a, b in zip(covered, covered[1:]))


def test_tp_one_is_the_whole_row():
    full = row_geometry(3, 128, 4096)
    split = row_parallel_split(bits=3, group_size=128, in_features=4096, tp_size=1)
    assert split.in_features_per_partition == 4096
    assert split.words_per_partition == full.words_per_row
    assert split.word_slice(0) == slice(0, full.words_per_row)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_every_width_splits_the_same_way(bits):
    split = row_parallel_split(bits=bits, group_size=128, in_features=4096, tp_size=2)
    full = row_geometry(bits, 128, 4096)
    assert split.words_per_partition * 2 == full.words_per_row


def test_split_that_would_cut_a_group_is_refused_with_both_ways_out():
    # 4096 inputs, group 128, tp 64 -> 64 inputs per rank: half a group.
    with pytest.raises(DynQuantError) as exc:
        row_parallel_split(
            bits=4, group_size=128, in_features=4096, tp_size=64, name="l.down_proj"
        )
    message = str(exc.value)
    assert "l.down_proj" in message
    assert "group_size" in message and "tensor-parallel size" in message


def test_per_row_grouping_cannot_be_split_at_all():
    with pytest.raises(DynQuantError, match="per-row"):
        row_parallel_split(
            bits=4,
            group_size=PER_ROW_GROUP_SIZE,
            in_features=4096,
            tp_size=2,
            name="l.embed",
        )
    # ...but is fine on one rank, which is how a per-row embedding still serves.
    split = row_parallel_split(
        bits=4, group_size=PER_ROW_GROUP_SIZE, in_features=4096, tp_size=1
    )
    assert split.groups_per_partition == 1


def test_indivisible_in_features_is_named_as_such():
    with pytest.raises(DynQuantError, match="not divisible"):
        row_parallel_split(bits=4, group_size=128, in_features=1536, tp_size=5)


def test_non_positive_tp_size_is_rejected():
    with pytest.raises(DynQuantError, match="tp_size must be positive"):
        row_parallel_split(bits=4, group_size=128, in_features=4096, tp_size=0)
