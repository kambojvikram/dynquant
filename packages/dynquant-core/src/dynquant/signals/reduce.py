"""Cross-rank reduction of collected signals.

Under any data-parallel scheme each rank sees different micro-batches, so each
rank's accumulators summarise a different sample. Writing rank 0's file discards
``world_size - 1`` of the evidence -- and worse, does so invisibly: the file looks
complete. The research code had no reduction step at all.

Reduction is exact for the quantities that admit it. Welford state combines
through Chan's parallel formula (:meth:`LayerStats.merged_with`), which returns
bit-comparable results to a single-stream pass over the union of observations. The
EMAs cannot: an EMA is path-dependent, and no algebra recovers the joint history
from two summaries, so they are combined by observation-count-weighted mean. That
approximation is documented in the schema rather than papered over.

Object collectives rather than tensor collectives, deliberately. Ranks can in
principle track different module sets -- an expert-parallel MoE, a pipeline-parallel
split, or simply a rank whose data never routed to some expert -- and a tensor
all-reduce over a fixed layout silently mismatches when that happens.
``all_gather_object`` carries the names along with the numbers, so the merge is
keyed rather than positional. It runs once per training job, so its cost does not
matter.
"""

from __future__ import annotations

from typing import Any

import torch

from dynquant._logging import get_logger

from .schema import StatsFile

__all__ = ["is_main_rank", "reduce_stats", "world_size"]

_log = get_logger(__name__)


def _distributed_active() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def world_size() -> int:
    return int(torch.distributed.get_world_size()) if _distributed_active() else 1


def rank() -> int:
    return int(torch.distributed.get_rank()) if _distributed_active() else 0


def is_main_rank() -> bool:
    """True on rank 0, and on any non-distributed run."""
    return rank() == 0


def reduce_stats(stats: StatsFile) -> StatsFile:
    """Merge one rank's stats with every other rank's.

    Returns the merged file on every rank -- callers decide who writes. Returns
    ``stats`` unchanged when the run is single-process, so the distributed path
    costs nothing to leave in place.
    """
    size = world_size()
    if size == 1:
        return stats

    payloads: list[Any] = [None] * size
    torch.distributed.all_gather_object(payloads, stats.to_dict())

    merged = stats
    mine = rank()
    for index, payload in enumerate(payloads):
        if index == mine or payload is None:
            continue
        merged = merged.merged_with(StatsFile.from_dict(payload))

    merged.provenance.world_size = size
    _log.info("reduced signal stats across %d ranks (%d layers)", size, len(merged))
    return merged
