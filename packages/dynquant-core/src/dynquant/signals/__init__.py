"""Signal collection: what training reveals about which weights matter.

DynQuant's premise is that a fine-tune already contains the information needed to
allocate bits, and you only have to write it down:

* **Activation saliency** -- an EMA of activation RMS per module. Measures how
  much signal actually flows through a weight on real data.
* **Gradient plasticity** -- the online variance of gradient norms. Measures
  whether the weight had settled by the time training stopped. A weight still
  being pushed around is one the task cares about.

Their product (after percentile-ranking, since RMS and variance-of-norms share no
units) is a soft AND: a weight must be *both* load-bearing *and* unsettled to earn
bits.

Collect them with the ``transformers`` callback::

    from dynquant import DynQuantCallback

    trainer = Trainer(model=model, ..., callbacks=[DynQuantCallback("stats/")])

or, in a hand-written loop, with the context manager::

    from dynquant.signals import track_signals

    with track_signals(model, "stats/", optimizer=optimizer):
        ...

:mod:`dynquant.signals.schema` is the on-disk contract,
:mod:`dynquant.signals.tracker` the collector, and
:mod:`dynquant.signals.estimators` the three ways to measure a gradient norm --
including the one that measures the frozen base weight under LoRA rather than the
adapter standing in front of it.

Imports here are lazy: :class:`DynQuantCallback` needs ``transformers``, which
``dynquant-core`` treats as optional, and the schema has to stay importable
without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .schema import CoverageReport, LayerStats, Provenance, StatsFile, load_stats, save_stats

_LAZY: dict[str, str] = {
    "ChannelMoments": "dynquant.signals.moments",
    "DynQuantCallback": "dynquant.signals.callback",
    "GradEstimatorMode": "dynquant.signals.estimators",
    "SignalTracker": "dynquant.signals.tracker",
    "TrackerConfig": "dynquant.signals.tracker",
    "load_moments": "dynquant.signals.moments",
    "reduce_stats": "dynquant.signals.reduce",
    "save_moments": "dynquant.signals.moments",
    "track_signals": "dynquant.signals.context",
}

if TYPE_CHECKING:  # pragma: no cover
    from .callback import DynQuantCallback
    from .context import track_signals
    from .estimators import GradEstimatorMode
    from .moments import ChannelMoments, load_moments, save_moments
    from .reduce import reduce_stats
    from .tracker import SignalTracker, TrackerConfig

__all__ = [
    "ChannelMoments",
    "CoverageReport",
    "DynQuantCallback",
    "GradEstimatorMode",
    "LayerStats",
    "Provenance",
    "SignalTracker",
    "StatsFile",
    "TrackerConfig",
    "load_moments",
    "load_stats",
    "reduce_stats",
    "save_moments",
    "save_stats",
    "track_signals",
]


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'dynquant.signals' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache so subsequent access skips this path
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
