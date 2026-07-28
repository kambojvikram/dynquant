"""``with track_signals(model, ...)`` -- signal collection for hand-written loops.

The ``transformers`` callback in :mod:`dynquant.signals.callback` is the path most
people want. This is the path for everyone else: a custom loop, an RL trainer, a
distillation script, a research harness that predates ``Trainer``.

The one thing a raw loop must get right is telling the tracker when an optimizer
step happened, since that -- not the micro-batch -- is the unit the plasticity
variance is defined over. Passing ``optimizer=`` wraps ``step()`` so the boundary
is detected rather than remembered::

    with track_signals(model, "stats/", optimizer=optimizer):
        for batch in loader:
            model(**batch).loss.backward()
            optimizer.step()          # tracker folds Welford, then steps
            optimizer.zero_grad()

Without it, call :meth:`SignalTracker.on_optimizer_step` yourself, before
gradients are zeroed. Forgetting leaves ``grad_norm_count`` at zero -- which the
schema reports as "no gradient signal" rather than as a plausible-looking
variance, so the mistake surfaces in the coverage summary instead of silently
producing a saliency-only allocation.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dynquant._logging import get_logger

from .tracker import SignalTracker, TrackerConfig

if TYPE_CHECKING:
    from torch import nn
    from torch.optim import Optimizer

__all__ = ["track_signals"]

_log = get_logger(__name__)


@contextmanager
def track_signals(
    model: nn.Module,
    out: str | Path | None = None,
    *,
    config: TrackerConfig | None = None,
    optimizer: Optimizer | None = None,
    save_on_error: bool = True,
    **overrides: Any,
) -> Iterator[SignalTracker]:
    """Collect signals for the duration of the block.

    Args:
        model: The model being trained. Wrappers (DDP, ``torch.compile``, FSDP)
            are fine -- names are canonicalised, so the stats keys match the bare
            model either way.
        out: Where to write on exit. A directory gets the standard filename. Pass
            ``None`` to skip writing and take the :class:`StatsFile` from
            :meth:`SignalTracker.snapshot` instead.
        config: Full configuration. Mutually exclusive with ``**overrides``.
        optimizer: If given, ``step()`` is wrapped to fold Welford state at the
            correct moment. Restored on exit, including on exception.
        save_on_error: Write the partial stats even if the block raises. A run that
            OOMs at step 900 still yields usable signals, and the alternative --
            discarding them -- helps nobody.
        **overrides: Convenience keyword arguments forwarded to
            :class:`TrackerConfig`, e.g. ``grad_estimator="param"``.

    Yields:
        The live :class:`SignalTracker`.
    """
    if config is not None and overrides:
        raise TypeError(
            "pass either config= or keyword overrides, not both -- otherwise which "
            "one wins is a coin flip that shows up as wrong numbers much later"
        )
    tracker = SignalTracker(model, config or TrackerConfig(**overrides))
    restore = _wrap_optimizer(tracker, optimizer)
    tracker.attach()
    try:
        yield tracker
    except BaseException:
        if out is not None and save_on_error:
            try:
                tracker.save(out)
                _log.warning("training raised; wrote partial signal stats to %s", out)
            except Exception as exc:  # noqa: BLE001 -- a failed rescue must never
                # mask the exception the caller actually needs to see.
                _log.warning("could not write partial signal stats: %s", exc)
        raise
    else:
        if out is not None:
            tracker.save(out)
    finally:
        restore()
        tracker.detach()


def _wrap_optimizer(tracker: SignalTracker, optimizer: Optimizer | None) -> Any:
    """Patch ``optimizer.step`` to fold Welford first. Returns the undo callable.

    Bound-method patching on the instance, so other optimizers of the same class
    are untouched and the class itself is never mutated.
    """
    if optimizer is None:
        return lambda: None

    original = optimizer.step

    def step(*args: Any, **kwargs: Any) -> Any:
        # Before, not after: `param` and `lowrank` read .grad, and step() may
        # modify gradients in place (fused optimizers, gradient centralisation).
        tracker.on_optimizer_step()
        return original(*args, **kwargs)

    optimizer.step = step  # type: ignore[method-assign]

    def restore() -> None:
        optimizer.step = original  # type: ignore[method-assign]

    return restore
