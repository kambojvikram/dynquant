"""``DynQuantCallback`` -- the three-line integration with a ``transformers`` fine-tune.

::

    from dynquant import DynQuantCallback

    trainer = Trainer(model=model, ..., callbacks=[DynQuantCallback("stats/")])
    trainer.train()

The same object works with TRL's ``SFTTrainer``, ``DPOTrainer`` and anything else
built on ``Trainer``, because it only uses the documented ``TrainerCallback``
surface. When training finishes, ``stats/dynquant_stats.json`` holds the signal map
that scoring, allocation and quantization consume.

Where the callback earns its keep
---------------------------------
``on_pre_optimizer_step`` is the whole reason this is a callback rather than a pair
of hooks. Plasticity is defined as the variance of the gradient norm over
*optimizer steps* (Appendix H), and only the trainer knows where those boundaries
fall -- gradient accumulation, and any step skipped by a gradient-scaler overflow
under mixed precision, both decouple "a backward pass happened" from "the weights
moved". The research code updated Welford from inside the gradient hook, so at 8-
step accumulation it recorded eight observations per step and measured within-batch
noise rather than step-to-step movement.

Older ``transformers`` releases do not emit ``on_pre_optimizer_step``. Rather than
require a floor version, the callback detects the silence and folds from
``on_step_end`` instead, warning once. That fallback is exact for the default
``outer_exact`` estimator, which stages its observations during backward and needs
nothing from the optimizer -- but it lands after ``zero_grad``, so ``param`` and
``lowrank`` would find no gradients and the warning is escalated accordingly.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from dynquant._logging import get_logger
from dynquant.errors import MissingDependencyError

from .estimators import GradEstimatorMode
from .schema import StatsFile
from .tracker import SignalTracker, TrackerConfig

if TYPE_CHECKING:
    from torch import nn

__all__ = ["DynQuantCallback"]

_log = get_logger(__name__)

_TrainerCallbackBase: type = object
_TRANSFORMERS_ERROR: Exception | None = None
try:  # transformers is an optional dependency of dynquant-core
    from transformers import (  # type: ignore[attr-defined]
        TrainerCallback as _ImportedTrainerCallback,
    )

    _TrainerCallbackBase = _ImportedTrainerCallback
except Exception as exc:  # noqa: BLE001 -- a broken transformers install must
    # surface as MissingDependencyError from __init__, not as an import-time crash
    # in code that never touches the callback.
    _TRANSFORMERS_ERROR = exc


class DynQuantCallback(_TrainerCallbackBase):  # type: ignore[misc]
    """Collects DynQuant signals over a ``transformers`` training run.

    Args:
        output_dir: Directory (or file path) for the stats file. Written at the end
            of training, at every checkpoint, and every ``log_every`` steps.
        config: A fully-specified :class:`TrackerConfig`. Mutually exclusive with
            the keyword overrides.
        save_on_checkpoint: Also write stats next to each ``Trainer`` checkpoint, so
            a resumed or preempted run carries its signals with it.
        **overrides: Forwarded to :class:`TrackerConfig` --
            ``grad_estimator="param"``, ``coherence_ema_beta=0.95``,
            ``log_every=100``, ``exclude=("*.vision_tower.*",)`` and so on.

    Attributes:
        tracker: The live :class:`SignalTracker`, or ``None`` before training
            begins. Useful for inspecting ``tracked_names`` mid-run.
    """

    def __init__(
        self,
        output_dir: str | Path = "stats",
        *,
        config: TrackerConfig | None = None,
        save_on_checkpoint: bool = True,
        **overrides: Any,
    ) -> None:
        if _TRANSFORMERS_ERROR is not None:
            raise MissingDependencyError(
                "transformers",
                feature="DynQuantCallback",
                extra="hf",
            ) from _TRANSFORMERS_ERROR
        if config is not None and overrides:
            raise TypeError(
                "pass either config= or keyword overrides to DynQuantCallback, not both"
            )

        self.output_dir = Path(output_dir)
        self.config = config or TrackerConfig(**overrides)
        self.config.validate()
        self.save_on_checkpoint = save_on_checkpoint
        self.tracker: SignalTracker | None = None
        self._saw_pre_optimizer_step = False
        self._warned_fallback = False
        self._written: Path | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        model = kwargs.get("model")
        if model is None:
            raise RuntimeError(
                "DynQuantCallback.on_train_begin received no model. This happens when "
                "the callback is invoked outside a Trainer; use "
                "dynquant.signals.track_signals(model, ...) for hand-written loops."
            )
        self.tracker = SignalTracker(model, self.config).attach()
        _log.info(
            "DynQuant signal collection attached to %d modules; stats -> %s",
            len(self.tracker),
            self.output_dir,
        )
        self._describe_blind_spots(model)

    def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._saw_pre_optimizer_step = True
        if self.tracker is not None:
            self.tracker.on_optimizer_step()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.tracker is None:
            return
        if not self._saw_pre_optimizer_step:
            self._warn_fallback()
            self.tracker.on_optimizer_step()
        self.tracker.maybe_flush(self.output_dir)

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.tracker is None or not self.save_on_checkpoint:
            return
        checkpoint = _checkpoint_dir(args, state)
        if checkpoint is not None:
            self.tracker.save(checkpoint)

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.tracker is None:
            return
        self._written = self.tracker.save(self.output_dir)
        self.tracker.detach()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    @property
    def stats_path(self) -> Path | None:
        """Where the final stats landed, once training has ended."""
        return self._written

    def stats(self) -> StatsFile:
        """The collected signals. Callable mid-run; costs one host sync."""
        if self.tracker is None:
            raise RuntimeError("no signals collected yet -- call this during or after training")
        return self.tracker.snapshot()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _warn_fallback(self) -> None:
        if self._warned_fallback:
            return
        self._warned_fallback = True
        estimator = self.config.resolved_estimator()
        if estimator is GradEstimatorMode.OUTER_EXACT:
            _log.warning(
                "this transformers release does not emit on_pre_optimizer_step; "
                "folding plasticity from on_step_end instead. Step boundaries are "
                "still correct, so the outer_exact estimator is unaffected."
            )
        else:
            _log.warning(
                "this transformers release does not emit on_pre_optimizer_step, so "
                "the %s estimator will find gradients already zeroed and will "
                "record no plasticity signal. Upgrade transformers, or switch to "
                "grad_estimator='outer_exact', which stages its observations during "
                "the backward pass and does not depend on optimizer timing.",
                estimator.value,
            )

    def _describe_blind_spots(self, model: nn.Module) -> None:
        """Say up front which modules this dataset cannot possibly exercise.

        A text-only corpus never routes through a vision tower, so its saliency and
        plasticity are structurally zero -- not small, absent. Naming the towers at
        attach time means the operator learns it before spending the GPU hours,
        rather than from a bit map that sent 15% of the model to the floor.
        """
        config = getattr(model, "config", None)
        towers = [
            name
            for name in ("vision_config", "audio_config", "video_config")
            if getattr(config, name, None) is not None
        ]
        if towers:
            _log.warning(
                "model declares %s: any modality absent from the training data will "
                "report zero saliency and zero plasticity. forward_calls==0 records "
                "that as 'never exercised' rather than 'unimportant'; check the "
                "coverage report before allocating.",
                ", ".join(towers),
            )


def _checkpoint_dir(args: Any, state: Any) -> Path | None:
    """Locate the checkpoint directory ``Trainer`` is currently writing."""
    root = getattr(args, "output_dir", None)
    step = getattr(state, "global_step", None)
    if root is None or step is None:
        return None
    candidate = Path(root) / f"checkpoint-{step}"
    return candidate if candidate.is_dir() else None
