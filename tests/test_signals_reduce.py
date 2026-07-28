"""Cross-rank reduction: two ranks must equal one rank over the union of the data.

This closes P2's DDP-parity gate item, and it is written as a real two-process gloo run
rather than as a direct call to ``LayerStats.merged_with``. Calling the merge function
directly tests the algebra, which was already the easy part; what breaks in practice is
the plumbing around it -- gathering with the wrong collective, gathering on some ranks
and not others, merging a rank into itself, or writing rank 0's file and never reducing
at all. The research code did the last of those, and the resulting file looked complete.

The parity assertion is exact-to-floating-point rather than approximate. Chan's parallel
formula is algebraically equal to a single-stream Welford pass, so a loose tolerance here
would hide a genuine wrong-weighting bug -- the difference between weighting by count and
weighting by rank would pass at 1e-3 on data that happens to be balanced.
"""

from __future__ import annotations

import json
import socket
import statistics
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch

from dynquant.signals.schema import LayerStats, StatsFile

# Two ranks, disjoint observation sets, deliberately different sizes: equal-sized halves
# would pass even if the merge weighted the ranks equally instead of by count.
RANK_OBSERVATIONS: tuple[tuple[float, ...], ...] = (
    (0.5, 1.5, 2.0, 3.25, 4.0, 7.5, 0.125),
    (2.5, 2.75, 9.0),
)

SHARED = "model.layers.0.mlp.down_proj"
RANK1_ONLY = "model.layers.0.mlp.experts.7.down_proj"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def layer_from(name: str, observations: tuple[float, ...], *, rms: float) -> LayerStats:
    """A LayerStats holding exactly the Welford summary of ``observations``."""
    count = len(observations)
    return LayerStats(
        name=name,
        activation_rms_ema=rms,
        grad_norm_count=count,
        grad_norm_mean=statistics.fmean(observations),
        grad_norm_var=statistics.variance(observations) if count >= 2 else 0.0,
        param_count=4096,
        role="mlp.down",
        forward_calls=count,
    )


def _worker(rank: int, world_size: int, port: int, out_dir: str) -> None:
    """One rank: build local stats, reduce, write the result for the parent to check."""
    # Re-import inside the spawned process: on spawn start methods nothing is inherited.
    from dynquant.signals.reduce import reduce_stats

    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        observations = RANK_OBSERVATIONS[rank]
        layers = {SHARED: layer_from(SHARED, observations, rms=1.0 + rank)}
        if rank == 1:
            # A module this rank saw and rank 0 did not -- an expert that never received a
            # token on rank 0. A positional tensor all-reduce would drop or misalign it.
            layers[RANK1_ONLY] = layer_from(RANK1_ONLY, observations, rms=0.5)

        reduced = reduce_stats(StatsFile(layers=layers))
        Path(out_dir, f"rank{rank}.json").write_text(
            json.dumps(reduced.to_dict()), encoding="utf-8"
        )
    finally:
        torch.distributed.destroy_process_group()


@pytest.fixture(scope="module")
def reduced_ranks(tmp_path_factory) -> tuple[StatsFile, StatsFile]:
    """Run the two-process reduction once and hand both ranks' results to the tests."""
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed is not available in this build")
    if sys.platform == "darwin":
        pytest.skip("gloo spawn is unreliable under the macOS sandbox")

    out_dir = tmp_path_factory.mktemp("reduce")
    torch.multiprocessing.spawn(
        _worker,
        args=(2, _free_port(), str(out_dir)),
        nprocs=2,
        join=True,
    )
    return tuple(  # type: ignore[return-value]
        StatsFile.from_dict(json.loads((out_dir / f"rank{rank}.json").read_text("utf-8")))
        for rank in (0, 1)
    )


def test_two_ranks_equal_one_rank_over_the_union(reduced_ranks) -> None:
    """The gate: reduced Welford state matches a single pass over both ranks' data."""
    union = RANK_OBSERVATIONS[0] + RANK_OBSERVATIONS[1]
    merged = reduced_ranks[0][SHARED]

    assert merged.grad_norm_count == len(union)
    assert merged.grad_norm_mean == pytest.approx(statistics.fmean(union), rel=1e-12)
    assert merged.grad_norm_var == pytest.approx(statistics.variance(union), rel=1e-12)


def test_every_rank_returns_the_same_reduction(reduced_ranks) -> None:
    """Both ranks return the merged file, so which rank writes cannot change the answer.

    ``reduce_stats`` merges other ranks into ``self`` in rank order, and floating-point
    addition is not associative, so this is a real constraint on the implementation
    rather than a restatement of the previous test.
    """
    rank0, rank1 = reduced_ranks
    assert rank0.names == rank1.names
    for name in rank0.names:
        assert rank0[name].grad_norm_count == rank1[name].grad_norm_count
        assert rank0[name].grad_norm_mean == pytest.approx(rank1[name].grad_norm_mean)
        assert rank0[name].grad_norm_var == pytest.approx(rank1[name].grad_norm_var)


def test_a_module_only_one_rank_saw_survives(reduced_ranks) -> None:
    """Keyed, not positional: a module absent on rank 0 still reaches rank 0's file."""
    for reduced in reduced_ranks:
        assert RANK1_ONLY in reduced
        only = reduced[RANK1_ONLY]
        assert only.grad_norm_count == len(RANK_OBSERVATIONS[1])
        assert only.grad_norm_mean == pytest.approx(
            statistics.fmean(RANK_OBSERVATIONS[1]), rel=1e-12
        )


def test_world_size_is_recorded(reduced_ranks) -> None:
    """Provenance says how many ranks contributed.

    Without it a reduced file and an unreduced one are indistinguishable, which is
    exactly the failure this module exists to prevent.
    """
    for reduced in reduced_ranks:
        assert reduced.provenance.world_size == 2


def test_activation_ema_is_count_weighted(reduced_ranks) -> None:
    """The EMA blend follows observation counts, not rank count.

    An EMA cannot be combined exactly, and the documented approximation is a
    count-weighted mean. Rank 0 carries 7 observations at 1.0 and rank 1 carries 3 at
    2.0, so an equal-weight blend would give 1.5 and a count-weighted one 1.3.
    """
    n_a, n_b = (len(observations) for observations in RANK_OBSERVATIONS)
    expected = (1.0 * n_a + 2.0 * n_b) / (n_a + n_b)
    assert reduced_ranks[0][SHARED].activation_rms_ema == pytest.approx(expected, rel=1e-12)


def test_single_process_is_a_passthrough() -> None:
    """Undistributed runs return the input untouched, so the call is free to leave in."""
    from dynquant.signals.reduce import is_main_rank, reduce_stats, world_size

    assert world_size() == 1
    assert is_main_rank()

    stats = StatsFile(layers={SHARED: layer_from(SHARED, RANK_OBSERVATIONS[0], rms=1.0)})
    assert reduce_stats(stats) is stats
