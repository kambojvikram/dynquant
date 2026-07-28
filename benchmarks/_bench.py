"""Step-time measurement primitives, shared by every tracker benchmark.

One module rather than a copy per benchmark, because two benchmarks that each implement
their own timing will eventually disagree about whether to synchronize, whether to warm
up, and whether to interleave -- and the disagreement shows up as a difference in
overhead that gets attributed to whatever was under test.

The design choices are all about not measuring the wrong thing:

**Interleave, never block.** Running every untracked step and then every tracked step
attributes GPU clock drift to the tracker. An A100 under sustained load drops its SM
clock as it heats, so whichever arm runs second looks a few percent slower -- the same
magnitude as the effect. Arms alternate step by step and are compared by median.

**Synchronize at both ends of every step.** CUDA launches are asynchronous, so an unsynced
timer measures how fast Python queued work, not how long the GPU took. The leading
synchronize matters as much as the trailing one: on an interleaved A/B, work still in
flight from the previous step would otherwise be billed to the other arm.

**Install and remove the tracker, don't toggle a flag.** A disabled hook still pays the
hook-invocation cost. The question is what a user who never passes the callback pays,
which is nothing.

**Report the spread next to the overhead.** A 1% effect measured against a 4% step-to-step
spread has not been resolved, and saying so is the difference between a measurement and a
number.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class Comparison:
    """The outcome of one interleaved A/B, in seconds and percent."""

    untracked: tuple[float, ...]
    tracked: tuple[float, ...]

    @property
    def off(self) -> float:
        return statistics.median(self.untracked)

    @property
    def on(self) -> float:
        return statistics.median(self.tracked)

    @property
    def overhead_percent(self) -> float:
        return (self.on / self.off - 1.0) * 100.0

    @property
    def spread_percent(self) -> float:
        """One standard deviation of the untracked arm, relative to its median.

        The noise floor the overhead has to clear to mean anything.
        """
        if len(self.untracked) < 2:
            return float("nan")
        return statistics.pstdev(self.untracked) / self.off * 100.0

    @property
    def resolved(self) -> bool:
        return abs(self.overhead_percent) > 2 * self.spread_percent

    def as_dict(self) -> dict[str, Any]:
        return {
            "untracked_seconds": list(self.untracked),
            "tracked_seconds": list(self.tracked),
            "untracked_median": self.off,
            "tracked_median": self.on,
            "overhead_percent": self.overhead_percent,
            "untracked_spread_percent": self.spread_percent,
            "resolved": self.resolved,
        }


def synchronized_step(model: Any, optimizer: Any, batch: dict[str, torch.Tensor]) -> float:
    """One full training step, timed with the device idle at both ends. Returns seconds."""
    torch.cuda.synchronize()
    started = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch, labels=batch["input_ids"]).loss
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize()
    return time.perf_counter() - started


def make_batch(model: Any, batch_size: int, seq_len: int, *, seed: int) -> dict[str, torch.Tensor]:
    """A fixed random batch, identical across arms.

    Random token ids are fine: step time depends on shapes, not on whether the text means
    anything, and a fixed batch removes data order as a way for the arms to differ.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    vocab = int(model.config.get_text_config().vocab_size)
    ids = torch.randint(0, vocab, (batch_size, seq_len), generator=generator)
    device = next(model.parameters()).device
    return {
        "input_ids": ids.to(device),
        "attention_mask": torch.ones_like(ids).to(device),
    }


def achieved_tflops(
    step_seconds: float, *, active_params: int, tokens: int, flops_per_param_token: float = 6.0
) -> float:
    """Model TFLOP/s implied by a step time, via the standard ``6ND`` estimate.

    Forward plus backward costs roughly 6 FLOPs per *active* parameter per token -- 2
    forward, 4 backward -- ignoring attention's quadratic term, which is a few percent at
    these sequence lengths and would only make the figure more generous.

    ``flops_per_param_token`` exists for the frozen-base case: under LoRA the backward
    computes an input gradient but no weight gradient for the base layers, so the factor
    is nearer 4 than 6. Leaving it at 6 there would report a throughput a third higher
    than the hardware actually delivered, which is the direction that hides a slow step.

    This exists because an overhead *ratio* is only as meaningful as its denominator, and
    nothing in a percentage reveals a denominator that was inflated. Every way of getting a
    slow step -- a kernel falling back to a Python-level implementation, gradient
    checkpointing left enabled, a throttled or shared GPU, an offloaded optimizer -- makes a
    fixed per-module cost look proportionally smaller and can turn a failing gate into a
    passing one. Rather than enumerate those causes, report the throughput they all depress
    and let the reader see it. ``active_params`` is a parameter and not ``sum(p.numel())``
    because on an MoE model only the routed experts run, and counting all of them would
    overstate the work by the expert count.
    """
    return flops_per_param_token * active_params * tokens / step_seconds / 1e12


def run_arm(
    model: Any,
    optimizer: Any,
    batch: dict[str, torch.Tensor],
    *,
    steps: int,
    tracked: bool,
    log_every: int,
    overrides: dict[str, Any] | None = None,
) -> list[float]:
    """Time ``steps`` steps with the tracker either installed or entirely absent.

    ``overrides`` goes straight to :class:`TrackerConfig` -- ``grad_estimator``,
    ``subsample_tokens`` and so on. It exists because "the tracker costs 9%" is not an
    actionable measurement: the three estimator modes do substantially different work per
    module, so without a way to vary them the number names a symptom and no cause.
    """
    from dynquant.signals.context import track_signals

    times: list[float] = []
    manager: Any = (
        track_signals(
            model, out=None, optimizer=optimizer, log_every=log_every, **(overrides or {})
        )
        if tracked
        else contextlib.nullcontext()
    )
    with manager:
        for _ in range(steps):
            times.append(synchronized_step(model, optimizer, batch))
    return times


def compare(
    model: Any,
    optimizer: Any,
    batch: dict[str, torch.Tensor],
    *,
    pairs: int,
    warmup: int,
    log_every: int,
    overrides: dict[str, Any] | None = None,
    on_pair: Any = None,
) -> Comparison:
    """Warm up both arms, then time ``pairs`` interleaved (untracked, tracked) steps.

    Warmup is outside the measurement because the first steps pay cuBLAS autotuning,
    allocator growth and optimizer-state allocation -- all one-time costs that would
    otherwise land on whichever arm ran first.

    The untracked arm is re-timed for every ``overrides`` setting rather than measured once
    and reused. That costs half the run, and it buys the only available check on the
    measurement itself: the untracked medians come from identical work, so if they disagree
    across settings the machine drifted and none of the overheads mean what they say.
    """
    for tracked in (False, True):
        run_arm(
            model,
            optimizer,
            batch,
            steps=warmup,
            tracked=tracked,
            log_every=log_every,
            overrides=overrides,
        )

    untracked: list[float] = []
    tracked_times: list[float] = []
    for index in range(pairs):
        untracked += run_arm(model, optimizer, batch, steps=1, tracked=False, log_every=log_every)
        tracked_times += run_arm(
            model, optimizer, batch, steps=1, tracked=True, log_every=log_every, overrides=overrides
        )
        if on_pair is not None:
            on_pair(index, untracked[-1], tracked_times[-1])

    return Comparison(untracked=tuple(untracked), tracked=tuple(tracked_times))
