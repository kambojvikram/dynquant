"""The signal tracker: what a fine-tune reveals, written down as it happens.

Two signals, both collected from hooks that ride along with training:

* **Saliency** -- an EMA of activation RMS per module. How much signal actually
  flows through a weight on real data.
* **Plasticity** -- the online variance of the gradient norm across optimizer
  steps. Whether the weight had settled by the time training stopped.

The research supplement collected both, and its implementation is the reason this
file exists rather than a thin wrapper around it. Three properties had to change.

**No host syncs.** The original called ``float(t.cpu().item())`` inside both
hooks. That is one full pipeline stall per module per hook per step -- roughly 850
on a dense 14B model, about 18,000 on a 128-expert MoE. The paper's "negligible
overhead" claim cannot survive that implementation. Here every accumulator lives
in a preallocated device tensor indexed by slot, updated in place, and the host
sees numbers exactly once per :meth:`SignalTracker.snapshot` -- one ``.cpu()`` on
a single stacked ``[5, n_modules]`` tensor, so one sync for the whole model
regardless of size. Counters that do not need to be on the device (call counts,
observation counts) are kept as Python ints, because the hook body is Python
anyway and a host-side increment costs nothing and launches no kernel.

**Variance over optimizer steps, not micro-batches.** Appendix H specifies the
variance is over optimizer steps; the original updated Welford inside the gradient
hook, which fires once per micro-batch. At 8-step gradient accumulation that
inflates the observation count 8x and measures within-batch noise instead of the
step-to-step movement the signal is supposed to capture. Here micro-batch
observations are *staged* into a buffer and folded into Welford exactly once, from
:meth:`on_optimizer_step`. The fold is vectorized across every module at once --
one set of kernels per step, not one per module.

**The right tensor.** See :mod:`dynquant.signals.estimators`. Under LoRA the base
weight is frozen, and the original hooked the adapter factors instead and filed
the result under the base module's name. The default ``outer_exact`` estimator
measures the base weight itself.

Debiased EMAs
-------------
An EMA seeded at zero is heavily biased low for its first few updates -- after one
update it holds ``(1 - beta) * x``, which at ``beta = 0.99`` is 1% of the true
value. The original avoided this by branching on "is this the first call?", which
requires knowing a device-resident count on the host: a sync. Instead the biased
accumulator is kept as-is and divided by ``1 - beta**k`` at snapshot time, with
``k`` the host-side call count. This is Adam's bias correction, it needs no
branch, and after one update it returns ``(1 - beta) x / (1 - beta) = x``
exactly -- identical to the original's first-call assignment, with none of the
sync.

Hook failures are counted, not hidden
-------------------------------------
A hook that raises must not take training down with it -- but a bare ``except:
return`` is precisely what kept the supplement's coherence signal silently dead
through every experiment in the paper. Exceptions here are caught per module,
counted, logged once with the traceback, and reported in
``provenance.notes["hook_errors"]``, so a degraded collection run is visible in
the artifact rather than discovered later by whoever wonders why the numbers look
odd.
"""

from __future__ import annotations

import fnmatch
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

from dynquant._logging import get_logger
from dynquant._version import __version__
from dynquant.errors import SignalCollectionError
from dynquant.graph.experts import batched_expert_params
from dynquant.graph.naming import canonical_name, is_adapter_name
from dynquant.graph.roles import ModuleRole, role_of_name

from .estimators import (
    GradEstimatorMode,
    channel_norm,
    embedding_gram,
    gram,
    lowrank_grad_sq,
    outer_grad_sq_batched,
)
from .schema import LayerStats, Provenance, StatsFile

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

    from .moments import ChannelMoments

__all__ = ["SignalTracker", "TrackerConfig"]

_log = get_logger(__name__)

_MOE_EXPERT_ROLES = frozenset(
    {
        ModuleRole.MOE_EXPERT_GATE,
        ModuleRole.MOE_EXPERT_UP,
        ModuleRole.MOE_EXPERT_DOWN,
        ModuleRole.MOE_EXPERT_GATE_UP,
        ModuleRole.MOE_EXPERT_FC,
    }
)
"""Roles for which ``forward_calls`` *is* the routing-hit count.

An expert's forward runs only when the router selects it, so the call count the
tracker keeps anyway is exactly the denominator the scorer needs to tell "stable"
apart from "never asked". Shared/always-on experts are deliberately excluded --
they fire every token, so reporting a routing-hit count for them would imply a
selection that never happened."""

# How many accumulators the single snapshot transfer carries. Kept as a named
# constant because the stack order below and the unpack in snapshot() must agree.
_N_DEVICE_ACCUMULATORS = 5

_CONTRACTION_QUEUE_BYTES = 64 << 20
"""Cap on the Gram pairs awaiting a batched contraction, in bytes.

Batching the contraction means holding two ``T x T`` fp32 Grams per module until the
flush -- 512 KB at the default ``T = 256``, so 100 MB on a dense 2B model but tens of
gigabytes on a 128-expert MoE, where nearly every one of thousands of experts is routed
at least once per step. The cap makes peak residency a constant instead of a function
of module count, and costs nothing: a flush covering a hundred modules already captures
essentially all of the batching win, so splitting a step into a few flushes is not
measurably worse than one."""

_GRAPH_TASK_ID = getattr(torch._C, "_current_graph_task_id", None)
"""``-1`` outside a backward pass, the running task's id inside one.

Private, hence the ``getattr``: on a torch that does not expose it, ``_in_backward``
reports ``False`` and the tracker keeps its pre-fix behaviour rather than raising. Present
on every version this package supports (checked on 2.7 and 2.13)."""


def _in_backward() -> bool:
    """Is this hook firing inside a backward pass rather than on a real forward?

    Gradient checkpointing replays a block's forward during backward to rebuild the
    activations it chose not to save, and module forward hooks are not suppressed on the
    replay. Measured on a checkpointed Qwen3-0.6B: ``self_attn.q_proj`` fires **twice** per
    micro-batch, ``lm_head`` -- outside the checkpointed layers -- once. Nothing else
    distinguishes the two calls; ``torch.is_grad_enabled()``, ``out.requires_grad`` and even
    ``type(out.grad_fn)`` are identical, and only the graph-task id differs (``-1`` then
    ``0``).

    So the predicate is deliberately "during a backward" and not "during a checkpoint
    recompute", because that is both what is detectable and what is *meant*: saliency
    averages over forward passes on training data, and a forward that happens inside a
    backward is a replay of data already counted or a higher-order derivative, not a new
    observation. Gradient accumulation is unaffected -- its micro-batches each run a real
    forward outside any backward, and each should count.
    """
    return _GRAPH_TASK_ID is not None and _GRAPH_TASK_ID() != -1


_MomentHalf = tuple[dict[int, Tensor], dict[int, float]]
"""One side of the channel-moment state: running sums of squares by slot, and the
token counts that turn them into means. Paired rather than merged because the two
sides are collected by different hooks and either can be missing."""


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """Collection hyperparameters.

    Defaults reproduce the research code's *semantics* where they were sound
    (``activation_ema_beta``, saliency measured on the module output) and depart
    from it only where it was measurably wrong (the estimator, the Welford
    cadence, the syncs).
    """

    activation_ema_beta: float = 0.99
    """Paper's beta for the saliency EMA."""

    coherence_ema_beta: float | None = None
    """``None`` disables the coherence signal.

    Off by default, which is a deliberate departure. The paper's Eq. (4) is a
    two-way product of saliency and plasticity; the shipped research code computes
    a three-way product including coherence. Since the published formula does not
    use it, collecting it by default would mean paying for a signal the documented
    method ignores. ``--preset paper-3.15`` turns it on, because reproducing the
    published numbers requires the code's behaviour rather than the paper's.

    Only meaningful with the ``outer_exact`` estimator, which is the only one that
    produces a shape-stable per-output-channel vector cheaply."""

    grad_estimator: str = GradEstimatorMode.OUTER_EXACT.value
    saliency_source: str = "output"
    """``"output"`` or ``"input"``.

    ``"output"`` matches the research code, and therefore the two shipped stats
    files that the golden tests replay. There is a reasonable argument for
    ``"input"`` -- quantization error enters as ``dY = dW x``, so the input's scale
    is what multiplies the weight error -- but it is an argument, not a defect, and
    the method was validated end to end on output RMS. Changing the default would
    trade a known-good signal for an untested one."""

    subsample_tokens: int = 256
    """Token budget per module per micro-batch for ``outer_exact``.

    Drives both Gram matrices, so cost is quadratic in this and workspace is
    ``T^2`` floats per tracked module -- 256 KB each at the default, about 220 MB
    across a dense 14B model, transient and released as the backward pass walks
    down. Raising it to 1024 costs 16x the arithmetic and 4 MB per module."""

    collect_channel_moments: bool = True
    """Also accumulate ``E[x_c^2]`` and ``E[delta_r^2]`` per channel.

    These are the two vectors the cardinal sensitivity estimator needs
    (:mod:`dynquant.score.sensitivity`); without them the allocator falls back to
    ranking scores and multiplying by a universal ``4**-bits`` error curve. On the
    one model where both have been checked against measured loss disturbance, the
    cardinal estimate scored +0.521 mean within-role Spearman against the score's
    +0.231, so this is on by default and turning it off costs allocation quality
    rather than correctness.

    Written to a separate safetensors sidecar, not into the stats JSON -- see
    :data:`~dynquant.constants.MOMENTS_FILENAME`. Linear modules only: an
    embedding's input is a token id rather than a feature vector, so the axes of
    its Gram-identity pairing are transposed relative to how its weight is stored,
    and the honest thing is to leave it unestimated. When the embedding is tied to
    an ``lm_head`` -- 27% of Qwen3.5-2B's parameters -- the head's moments describe
    the same tensor in the right orientation and the estimator uses those."""

    channel_moment_every: int = 16
    """Optimizer steps between channel-moment samples. ``1`` measures every step.

    Not a corner cut, a cost decision with a stated basis. The saliency read alone
    is ~76 us of the ~101 us of forward-side per-module work, and the tracker
    measured at +2.30% of step time against a 3% budget; a second per-step
    reduction on each side would put it over. Sampling makes the added cost ~1/16th
    of that.

    Sixteen is safe because of what these are. Plasticity is a variance *of* a
    per-step quantity, so it needs every step. Channel second moments are
    distributional statistics of the activations, which drift on the timescale of
    training rather than of a batch -- and the estimate is a mean over every token
    of every sampled step, so a 500-step run still averages over ~30 steps and
    millions of tokens. The batch-to-batch variation already in the quantity is
    larger than what the skipped steps would have changed."""

    observe_recompute: bool = False
    """Count a gradient-checkpointing replay as a saliency observation.

    ``False`` -- the default -- is a fix, not a preference. With checkpointing on, a module
    inside a checkpointed block fires its forward hook twice per micro-batch on *identical*
    data while modules outside those blocks fire once. Two EMA updates with the same value
    leave the fixed point alone but square the decay, so the replayed modules end up
    averaging over a ~50-step horizon at ``beta = 0.99`` while the rest average over ~100.
    The absolute horizon barely matters; the *inconsistency* does, because
    :mod:`dynquant.score` percentile-ranks every module against every other, and ranking a
    50-step average against a 100-step one compares two different statistics.

    "Inside a checkpointed block" is not even the whole set. Non-reentrant checkpointing
    stops recomputing once the last saved tensor it needs has been produced, so on
    Qwen3-0.6B at the transformers 5.x default 168 of 198 modules doubled and every
    ``mlp.down_proj`` -- last op in its block -- did not; under ``use_reentrant=True`` all
    196 in-block modules did. So the horizon can differ between ``up_proj`` and
    ``down_proj`` of the same MLP, which no consumer of the stats file could guess.

    Measured cost of the double read, Qwen3-0.6B at 8x2048: 150.6 -> 221.1 us per module
    per step. Skipping the replay is therefore also where the checkpointed step gets its
    overhead back.

    ``True`` restores the research code's behaviour, which had no such guard. Reproducing a
    stats file collected under checkpointing needs it; nothing else should."""

    eps: float = 1e-12
    log_every: int = 0
    """Steps between intermediate flushes to disk. ``0`` writes only at the end.

    Non-zero costs one sync per flush and buys a usable stats file if the run dies
    -- which on an interruptible spot instance is not a hypothetical."""

    include: tuple[str, ...] = ()
    """Glob patterns against canonical names. Empty means everything eligible."""

    exclude: tuple[str, ...] = ()
    track_embeddings: bool = True
    max_hook_errors: int = 64
    """Hook exceptions tolerated before :meth:`snapshot` refuses to pretend the
    collection succeeded."""

    def resolved_estimator(self) -> GradEstimatorMode:
        return GradEstimatorMode.parse(self.grad_estimator)

    def validate(self) -> None:
        if not 0.0 < self.activation_ema_beta < 1.0:
            raise ValueError(
                f"activation_ema_beta must be in (0, 1), got {self.activation_ema_beta}"
            )
        if self.coherence_ema_beta is not None and not 0.0 < self.coherence_ema_beta < 1.0:
            raise ValueError(
                f"coherence_ema_beta must be in (0, 1) or None, got {self.coherence_ema_beta}"
            )
        if self.saliency_source not in ("output", "input"):
            raise ValueError(
                f"saliency_source must be 'output' or 'input', got {self.saliency_source!r}"
            )
        if self.subsample_tokens <= 0:
            raise ValueError(f"subsample_tokens must be positive, got {self.subsample_tokens}")
        if self.channel_moment_every < 1:
            raise ValueError(f"channel_moment_every must be >= 1, got {self.channel_moment_every}")
        if self.log_every < 0:
            raise ValueError(f"log_every must be >= 0, got {self.log_every}")
        self.resolved_estimator()


@dataclass(slots=True)
class _Tracked:
    """One module the tracker watches."""

    slot: int
    name: str
    """Canonical name -- the key this module's weight will have in the merged
    model, and therefore the key the quantizer will look it up by."""

    module: nn.Module
    weight: nn.Parameter
    role: ModuleRole
    kind: str
    """``"linear"``, ``"embedding"`` or ``"other"``. Decides whether the Gram
    identity applies, and how the input Gram is built."""

    forward_calls: int = 0
    coherence_calls: int = 0
    pending_observations: int = 0
    """Micro-batch gradient observations staged since the last optimizer step."""

    hook_errors: int = 0


class SignalTracker:
    """Collects DynQuant signals from a live training run.

    Usually driven by :class:`~dynquant.signals.callback.DynQuantCallback` or
    :func:`~dynquant.signals.context.track_signals` rather than directly. Direct
    use looks like::

        tracker = SignalTracker(model)
        tracker.attach()
        for batch in loader:
            loss = model(**batch).loss
            loss.backward()
            tracker.on_optimizer_step()   # before optimizer.step() zeroes grads
            optimizer.step()
            optimizer.zero_grad()
        tracker.save("stats/")
        tracker.detach()

    :meth:`on_optimizer_step` must be called while gradients are still live --
    ``param`` and ``lowrank`` read them there. ``outer_exact`` does not, so it is
    insensitive to the ordering, which is one more reason it is the default.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrackerConfig | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config or TrackerConfig()
        self.config.validate()
        self.model = model
        self._estimator = self.config.resolved_estimator()

        self._tracked: list[_Tracked] = []
        self._by_module: dict[int, _Tracked] = {}
        self._skipped: dict[str, str] = {}
        self._handles: list[RemovableHandle] = []
        self._attached = False
        self._optimizer_steps = 0
        self._flushes = 0
        self._logged_errors: set[str] = set()
        # Forward-hook calls that landed inside a backward pass -- gradient-checkpointing
        # replays, near enough. Counted whether or not they were observed, because it is
        # the count that tells a reader of the stats file whether the saliency EMA has one
        # horizon or two. See snapshot()'s recompute_forward_calls note.
        self._recompute_calls = 0

        self._discover()
        self._device = torch.device(device) if device is not None else self._infer_device()
        self._allocate()

        # Transient per-backward state. Keyed by slot, popped as consumed, so peak
        # residency falls monotonically once the backward pass starts.
        self._gram_x: dict[int, Tensor] = {}
        self._prev_channel: dict[int, Tensor] = {}
        # The one-entry input-Gram cache: (input tensor, kind, its Gram or None). See
        # _shared_input_gram for why it holds the tensor and why it is not a dict.
        self._last_gram: tuple[Tensor, str, Tensor | None] | None = None

        # Saliency observations awaiting a batched fold. Parallel lists rather than a
        # dict because the flush feeds them straight to multi-tensor ops, which take
        # lists; the set is the duplicate test that keeps per-forward-call EMA
        # ordering exact. Holding one length-``d_out`` vector per module is under 2 MB
        # on a 2B model, and each is fresh storage, so no activation stays alive.
        self._sal_slots: list[int] = []
        self._sal_norms: list[Tensor] = []
        self._sal_counts: list[float] = []
        self._sal_pending: set[int] = set()

        # The same deferral on the backward side: staged gradient norms, and the
        # coherence pairs whose cosine is computed in one batch at the flush.
        self._bwd_slots: list[int] = []
        self._bwd_values: list[Tensor] = []
        # outer_exact queues Gram pairs instead of scalars: the contraction that turns
        # them into a norm is the batchable part. Each pair is 2 x T^2 fp32 -- 512 KB
        # per module at the default T=256, and released at the flush.
        self._ctr_slots: list[int] = []
        self._ctr_gram_x: list[Tensor] = []
        self._ctr_gram_d: list[Tensor] = []
        self._ctr_bytes = 0
        self._coh_slots: list[int] = []
        self._coh_current: list[Tensor] = []
        self._coh_previous: list[Tensor] = []
        self._bwd_pending: set[int] = set()

        # Channel moments: running sums of squares and the token counts that
        # normalise them. Dicts rather than the stacked accumulators above because
        # every module's vectors have a different length, so there is nothing to
        # stack; they stay on whatever device produced them and move once, at
        # snapshot. About 4 MB in total across a 2B model.
        self._moment_x: dict[int, Tensor] = {}
        self._moment_d: dict[int, Tensor] = {}
        self._moment_x_tokens: dict[int, float] = {}
        self._moment_d_tokens: dict[int, float] = {}
        self._moment_obs: dict[int, int] = {}
        # Whole optimizer steps are sampled, never individual micro-batches: the
        # forward that records E[x^2] and the backward that records E[delta^2] must
        # agree about whether this step counts, or a module accumulates one moment
        # without the other and the pair is unusable.
        self._moment_active = self.config.collect_channel_moments

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _resolve_roles(self) -> dict[str, ModuleRole]:
        """Roles from the full classifier, keyed by canonical name.

        The tracker writes a role into every stats entry, and that role has to be
        the same one the allocator will later act on. Falling back to
        :func:`role_of_name` -- which matches substrings and nothing else -- would
        put a different answer in the file than the graph produces: Qwen3.5's
        ``in_proj_a`` reads as an SSM input projection by name while it is
        structurally a linear-attention decay projection, and a reader of the
        stats file alone would have no way to know which was meant.

        Classification is best-effort here. It needs a config and an architecture
        the registry knows, and a tracker that refused to start because it could
        not name a module's role would be trading a working signal collection for
        a cosmetic field. On failure each module falls back to the name-based
        guess, which is what this method exists to improve on rather than depend
        on.
        """
        from dynquant.graph.classify import classify_model

        try:
            graph = classify_model(self.model)
        # BLE001: deliberately blind. Role tagging is a convenience for readers of
        # the stats file, and the allocator reads roles from the graph rather than
        # from here. No classification failure on an architecture we have never
        # seen is worth aborting a multi-hour training run over.
        except Exception as exc:  # noqa: BLE001  # pragma: no cover -- architecture-dependent
            _log.warning(
                "could not classify %s for role tagging (%s); "
                "falling back to name-based roles in the stats file",
                type(self.model).__name__,
                exc,
            )
            return {}
        return {info.name: info.role for info in graph}

    def _discover(self) -> None:
        """Find every module that owns a weight the quantizer will care about.

        Outermost-wins. Under PEFT the same weight is reachable as
        ``...q_proj`` (the ``lora.Linear`` wrapper) and ``...q_proj.base_layer``
        (the original ``nn.Linear``), and both canonicalise to ``...q_proj``.
        ``named_modules`` yields parents first, so taking the first claim on a
        canonical name selects the wrapper -- which is the one worth hooking: its
        input is the base weight's input, and because ``Y = base(X) + lora(X)``
        sums into a single output node, the gradient arriving at the wrapper's
        output *is* the base weight's output gradient. Hooking the wrapper
        therefore measures the frozen base weight exactly.
        """
        roles = self._resolve_roles()
        claimed: set[str] = set()
        unmeasurable_params = 0
        for name, module in self.model.named_modules():
            if not name:
                continue
            if is_adapter_name(name):
                continue
            weight = _quantizable_weight(module)
            if weight is None:
                refused = _unmeasurable_experts(name, module)
                if refused:
                    self._skipped.update(refused)
                    unmeasurable_params += sum(p.numel() for _, p in batched_expert_params(module))
                continue

            key = canonical_name(name)
            if key in claimed:
                continue
            kind = _module_kind(module)
            if kind == "embedding" and not self.config.track_embeddings:
                self._skipped[key] = "embeddings excluded by config"
                continue
            if not self._selected(key):
                self._skipped[key] = "filtered by include/exclude"
                continue

            claimed.add(key)
            entry = _Tracked(
                slot=len(self._tracked),
                name=key,
                module=module,
                weight=weight,
                role=roles.get(key) or role_of_name(key),
                kind=kind,
            )
            self._tracked.append(entry)
            self._by_module[id(module)] = entry

        if not self._tracked:
            raise SignalCollectionError(
                "no quantizable modules found to track. DynQuant looks for modules "
                "owning a weight with 2 or more dimensions (Linear, Embedding, "
                "Conv1d/2d, or a PEFT wrapper around one). Check the include/exclude "
                "patterns, or that a model rather than a bare module was passed."
            )
        _log.info(
            "tracking %d modules (%d skipped), estimator=%s",
            len(self._tracked),
            len(self._skipped),
            self._estimator.value,
        )
        if unmeasurable_params:
            tracked_params = sum(entry.weight.numel() for entry in self._tracked)
            share = unmeasurable_params / max(tracked_params + unmeasurable_params, 1)
            _log.warning(
                "%.1f%% of quantizable parameters (%s) sit in batched MoE expert banks "
                "and are NOT being measured -- their signals are not separable at the "
                "bank's module boundary. The allocator will see them as unmeasured and "
                "give them a neutral score, so bit widths across the experts will not be "
                "signal-driven. See dynquant.graph.experts for what per-expert tracking "
                "would require.",
                share * 100.0,
                f"{unmeasurable_params:,}",
            )

    def _selected(self, key: str) -> bool:
        if self.config.include and not any(fnmatch.fnmatch(key, p) for p in self.config.include):
            return False
        return not any(fnmatch.fnmatch(key, p) for p in self.config.exclude)

    def _infer_device(self) -> torch.device:
        """Put accumulators wherever the weights already are.

        A CPU accumulator updated from a CUDA hook would force a transfer per
        write, which is the sync this design exists to avoid.
        """
        for entry in self._tracked:
            if entry.weight.device.type != "meta":
                return entry.weight.device
        return torch.device("cpu")

    def _allocate(self) -> None:
        n = len(self._tracked)
        opts: dict[str, Any] = {"dtype": torch.float32, "device": self._device}
        # Biased EMA of activation RMS; debiased on the way out.
        self._act_ema = torch.zeros(n, **opts)
        # Sum of this optimizer step's micro-batch gradient norms, and Welford
        # state over completed steps.
        self._grad_stage = torch.zeros(n, **opts)
        self._grad_mean = torch.zeros(n, **opts)
        self._grad_m2 = torch.zeros(n, **opts)
        self._grad_n = torch.zeros(n, **opts)
        self._coh_ema = torch.zeros(n, **opts)

    # ------------------------------------------------------------------
    # Attach / detach
    # ------------------------------------------------------------------

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def tracked_names(self) -> tuple[str, ...]:
        return tuple(e.name for e in self._tracked)

    @property
    def skipped(self) -> dict[str, str]:
        """Quantizable tensors this tracker is *not* measuring, and why.

        The counterpart to :attr:`tracked_names`, and the more important of the two: a
        tensor missing from the stats file is indistinguishable from one that was
        measured and found unremarkable, and the allocator treats both as neutral. This
        is what lets a caller tell the difference before spending the training run.
        """
        return dict(self._skipped)

    def attach(self) -> SignalTracker:
        """Register hooks. Idempotent."""
        if self._attached:
            return self
        for entry in self._tracked:
            self._handles.append(entry.module.register_forward_hook(self._forward_hook))
        self._attached = True
        return self

    def detach(self) -> SignalTracker:
        """Remove every hook and drop transient state. Accumulators survive."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        # Flushed, not cleared: unlike the backward state below, a pending saliency
        # norm is a completed measurement, and dropping it would silently lose the
        # last forward pass of a run that ends by leaving the context manager.
        self._flush_saliency()
        self._flush_backward()
        self._gram_x.clear()
        self._prev_channel.clear()
        self._last_gram = None
        self._attached = False
        return self

    def __enter__(self) -> SignalTracker:
        return self.attach()

    def __exit__(self, *exc: object) -> None:
        self.detach()

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _forward_hook(self, module: nn.Module, args: tuple[Any, ...], output: Any) -> None:
        entry = self._by_module.get(id(module))
        if entry is None:
            return
        try:
            self._observe_forward(entry, args, output)
        except Exception as exc:  # noqa: BLE001 -- see "Hook failures" in the module docstring
            self._note_error(entry, "forward", exc)

    def _observe_forward(self, entry: _Tracked, args: tuple[Any, ...], output: Any) -> None:
        out = _first_tensor(output)
        inp = _first_tensor(args)

        # A checkpoint replay is not a new observation. See _in_backward and
        # TrackerConfig.observe_recompute. forward_calls is skipped along with the update
        # because it is the k in the EMA's 1 - beta**k debias, so counting a fold that did
        # not happen would debias by the wrong exponent -- and because routing_hits is
        # forward_calls for expert roles, where a replay is not a routing decision either.
        #
        # The guard stops here rather than returning. The gradient half is already correct
        # under checkpointing -- grad_norm_count was 4 over 4 steps with or without
        # checkpointing, while forward_calls went 4 -> 8 -- so there is nothing to fix
        # below, and which of the two registrations is the live one has not been
        # established. If it is the replay's, returning early would silently drop the
        # plasticity signal for every checkpointed module. The saliency read is ~76 of the
        # ~101 us of forward-side work, so stopping here gets essentially all of the cost
        # back without betting on an untested answer.
        #
        # _in_backward is called unconditionally rather than short-circuited behind
        # observe_recompute, so the count lands in the stats file in both modes: a file
        # collected with observe_recompute=True is exactly the one whose reader needs to be
        # told its EMA horizon is uneven. The default path always evaluated the call anyway,
        # which is where the "free when there is no replay" measurement (150.6 -> 152.1 us
        # per module, inside the 0.12% noise floor) comes from.
        replay = _in_backward()
        if replay:
            self._recompute_calls += 1
        if not replay or self.config.observe_recompute:
            entry.forward_calls += 1
            source = out if self.config.saliency_source == "output" else inp
            if source is not None and source.numel() and source.is_floating_point():
                self._update_saliency(entry, source)
            # Same replay rule, same reason: a checkpoint replay re-reads identical
            # activations, and counting them twice would weight those modules'
            # moments by 2 against the rest of the model.
            if self._moment_active:
                self._accumulate_input_moment(entry, inp)

        # Register the gradient hook on the output tensor rather than using
        # register_full_backward_hook. The latter rewires the module's autograd
        # graph through a synthetic Function, which misbehaves on modules that
        # return tuples or take keyword-only tensors; a hook on the output tensor
        # hands us dL/dY directly and composes with gradient checkpointing (the
        # recompute pass simply re-registers).
        #
        # One hook serves both consumers. The output moment needs dL/dY exactly as
        # outer_exact does, and registering a second hook would double the per-module
        # Python callback cost to collect a tensor we already have in hand.
        if out is None or not out.requires_grad or not torch.is_grad_enabled():
            return
        gram_x = None
        if self._estimator is GradEstimatorMode.OUTER_EXACT:
            gram_x = self._shared_input_gram(entry, inp)
            if gram_x is not None:
                self._gram_x[entry.slot] = gram_x
        if gram_x is None and not self._moment_active:
            return
        slot = entry.slot
        out.register_hook(  # type: ignore[no-untyped-call]
            lambda grad: self._grad_output_hook(slot, grad)
        )

    def _update_saliency(self, entry: _Tracked, source: Tensor) -> None:
        """Reduce the activation to per-channel norms; defer the rest to a flush.

        The signal is per-channel RMS, averaged, folded into a biased EMA --
        ``mean_c sqrt(mean_dims(x_c^2) + eps)``, mirroring the research code
        exactly, including ``eps`` *inside* the square root. That is why a module
        firing on all-zero activations reports ``sqrt(eps)`` rather than ``0``, and
        why ``forward_calls`` rather than a zero test detects one that never fired.

        What changed is where the arithmetic happens, and it is a step-time fix
        rather than a numerical one. Written directly, the formula costs about ten
        Python-level torch calls per module -- ``detach``, ``float``, ``mul``,
        ``mean``, ``add``, ``sqrt``, ``mean``, a slice, ``reshape``, ``lerp_`` --
        and on a 197-module model that is ~2000 dispatches per step at a few
        microseconds of CPU each, enough to leave the GPU waiting to be fed. It
        measured as +26.75% of step time on a dense Qwen3-0.6B against a 3% gate,
        and the cost was near-identical in *absolute* terms on a 3x larger model,
        which is the signature of a fixed per-module cost rather than real work.

        So this keeps only the one reduction that genuinely must run per module,
        fused: :func:`torch.linalg.vector_norm` with ``dtype=torch.float32`` gives
        per-channel L2 norms with fp32 accumulation and no full-size fp32 copy of
        the activation -- the old ``.float().mul()`` pair allocated two. Everything
        downstream is arithmetic on a length-``d_out`` vector and is batched across
        all modules in :meth:`_flush_saliency`.

        ``detach`` is gone rather than deferred: under ``no_grad`` the result
        already carries no graph, and the norm is fresh storage rather than a view,
        so holding it until the flush pins nothing else alive.
        """
        with torch.no_grad():
            dims = tuple(range(source.ndim - 1)) if source.ndim >= 2 else (0,)
            norm = torch.linalg.vector_norm(source, dim=dims, dtype=torch.float32)
        # A slot recorded twice in one flush would need its two EMA updates applied
        # in order, which a batched update cannot express. Flushing first keeps the
        # per-forward-call semantics exact under gradient accumulation, where every
        # module fires once per micro-batch.
        if entry.slot in self._sal_pending:
            self._flush_saliency()
        self._sal_slots.append(entry.slot)
        self._sal_norms.append(norm)
        # Elements reduced over, so the flush can turn a sum of squares into a mean.
        self._sal_counts.append(source.numel() / max(norm.numel(), 1))
        self._sal_pending.add(entry.slot)

    def _discard_queues(self) -> None:
        """Empty every deferred-observation queue without folding anything in.

        Only :meth:`reset` wants this. The flushes clear one half each, deliberately:
        a saliency flush that also dropped the backward queue would lose the
        observations a mid-step ``snapshot`` is there to preserve.
        """
        self._sal_slots, self._sal_norms, self._sal_counts = [], [], []
        self._sal_pending.clear()
        self._last_gram = None
        self._bwd_slots, self._bwd_values = [], []
        self._ctr_slots, self._ctr_gram_x, self._ctr_gram_d = [], [], []
        self._ctr_bytes = 0
        self._coh_slots, self._coh_current, self._coh_previous = [], [], []
        self._bwd_pending.clear()

    def _flush_saliency(self) -> None:
        """Fold every pending per-channel norm into the EMA, in ~12 dispatches total.

        Multi-tensor ops do the whole model at once regardless of module count, so
        the per-step dispatch bill stops scaling with depth. The chain reconstructs
        ``mean_c sqrt(v_c^2 / N + eps)`` from the norms, then applies one indexed
        ``lerp``.

        ``_foreach_norm(..., 1)`` is doing duty as a per-tensor *sum*: after
        ``sqrt_`` every entry is non-negative, so its L1 norm is its sum, and this
        is the only fused reduction that returns one scalar per tensor for a list of
        differing lengths. The pinned test on the formula catches it if that ever
        stops holding.

        The pending lists are taken and cleared up front. A failure mid-chain then
        loses one step's saliency rather than leaving state that grows without bound
        for the rest of the run.
        """
        if not self._sal_slots:
            return
        slots, norms, counts = self._sal_slots, self._sal_norms, self._sal_counts
        self._sal_slots, self._sal_norms, self._sal_counts = [], [], []
        self._sal_pending.clear()

        with torch.no_grad():
            # Channel widths before the in-place chain; a Python loop, but no dispatch.
            widths = [float(max(t.numel(), 1)) for t in norms]
            torch._foreach_mul_(norms, norms)
            torch._foreach_div_(norms, counts)
            torch._foreach_add_(norms, self.config.eps)
            torch._foreach_sqrt_(norms)
            sums = torch.stack(torch._foreach_norm(norms, 1)).to(self._device, torch.float32)
            means = sums / torch.as_tensor(widths, dtype=torch.float32, device=self._device)

            # An indexed read-modify-write rather than a whole-tensor lerp_: slots that
            # did not fire this step must not decay toward zero.
            index = torch.as_tensor(slots, dtype=torch.long, device=self._device)
            previous = self._act_ema.index_select(0, index)
            self._act_ema.index_copy_(
                0, index, previous.lerp(means, 1.0 - self.config.activation_ema_beta)
            )

    def _shared_input_gram(self, entry: _Tracked, inp: Tensor | None) -> Tensor | None:
        """``_input_gram``, reusing the previous module's result on the same tensor.

        A transformer block hands *the same tensor object* to ``q_proj``, ``k_proj`` and
        ``v_proj``, and another to ``gate_proj`` and ``up_proj``. Those modules fire
        consecutively, so a one-entry cache turns the seven input Grams of a Qwen3 layer
        into four -- and an input Gram is one of the two matmuls that are now the whole
        remaining per-module cost.

        The test is ``is``, not equality: two tensors with equal contents may still be
        different objects, and comparing contents would cost more than the matmul it
        saves. One entry rather than a dict keyed by ``id`` because ``id`` is reused after
        a tensor is freed, and a cache that outlives its key can return another module's
        Gram. Holding the tensor keeps the id valid for exactly as long as the entry.

        Kinds are keyed on too: an embedding's Gram is the token-equality matrix, so an
        embedding and a Linear reading the same ids must not share one.
        """
        if inp is None:
            return None
        cached = self._last_gram
        if cached is not None and cached[0] is inp and cached[1] == entry.kind:
            return cached[2]
        gram_x = self._input_gram(entry, inp)
        # Cached even when None, so a repeated unsupported input is rejected once.
        self._last_gram = (inp, entry.kind, gram_x)
        return gram_x

    def _input_gram(self, entry: _Tracked, inp: Tensor | None) -> Tensor | None:
        if inp is None or inp.numel() == 0:
            return None
        with torch.no_grad():
            if entry.kind == "embedding":
                if inp.is_floating_point():
                    return None
                return embedding_gram(inp, self.config.subsample_tokens)
            if entry.kind != "linear" or not inp.is_floating_point():
                # Convolutions need an im2col before the Gram identity applies;
                # the depthwise conv1d in a linear-attention block is tiny and
                # carries a high floor anyway, so it collects saliency only and
                # says so via has_gradient_signal rather than guessing.
                return None
            if inp.ndim < 2:
                return None
            return gram(inp, self.config.subsample_tokens)

    def _accumulate_input_moment(self, entry: _Tracked, inp: Tensor | None) -> None:
        """Add this micro-batch's ``sum_t x_c^2`` to the running per-channel sum.

        One fused reduction, the same primitive and the same reason as
        :meth:`_update_saliency`: ``vector_norm`` with ``dtype=float32`` accumulates
        in fp32 without materialising an fp32 copy of the activation, where a
        ``.float().pow(2).sum()`` chain allocates one the size of the input.

        Squaring the norm rather than summing squares directly is not a detour --
        ``|x_c|^2`` *is* ``sum_t x_c^2``, and this is the only route to it that
        stays inside one dispatch.

        Every token is used, unlike the Gram path's ``subsample_tokens``. The two
        have different cost classes: a Gram is ``O(T^2 d)`` and has to be capped,
        while this is ``O(T d)`` -- one read of a tensor that is already resident.
        Capping it would save a fraction of a single memory pass and pay for it in
        accuracy on a statistic that is already only sampled every
        ``channel_moment_every`` steps.
        """
        if inp is None or inp.ndim < 2 or not inp.is_floating_point() or not inp.numel():
            return
        # Linear modules only, and the shape is checked rather than assumed. An
        # embedding takes integer ids (rejected above); a conv1d's input axis is not
        # the weight's second axis, and silently pairing the two would produce a
        # number for a module the identity does not describe.
        if entry.kind != "linear" or inp.shape[-1] != entry.weight.shape[1]:
            return
        dims = tuple(range(inp.ndim - 1))
        with torch.no_grad():
            squares = torch.linalg.vector_norm(inp, dim=dims, dtype=torch.float32)
            squares.mul_(squares)
        self._fold_moment(
            self._moment_x, self._moment_x_tokens, entry.slot, squares, inp.numel() / inp.shape[-1]
        )
        self._moment_obs[entry.slot] = self._moment_obs.get(entry.slot, 0) + 1

    def _accumulate_output_moment(self, entry: _Tracked, grad: Tensor) -> None:
        """The backward twin of :meth:`_accumulate_input_moment`, over ``dL/dY``.

        This is the factor that decides whether the estimate works at all. Dropping
        it leaves ``sum_c E[x_c^2] ||dW[:, c]||^2`` -- the quantity a calibration-only
        method computes -- which measured **-0.338** mean within-role Spearman
        against true loss disturbance, against +0.521 with it. Not weaker: the wrong
        sign, in 11 of 12 roles.
        """
        if grad.ndim < 2 or not grad.is_floating_point() or not grad.numel():
            return
        if entry.kind != "linear" or grad.shape[-1] != entry.weight.shape[0]:
            return
        dims = tuple(range(grad.ndim - 1))
        squares = torch.linalg.vector_norm(grad, dim=dims, dtype=torch.float32)
        squares.mul_(squares)
        self._fold_moment(
            self._moment_d,
            self._moment_d_tokens,
            entry.slot,
            squares,
            grad.numel() / grad.shape[-1],
        )

    @staticmethod
    def _fold_moment(
        sums: dict[int, Tensor],
        tokens: dict[int, float],
        slot: int,
        squares: Tensor,
        count: float,
    ) -> None:
        previous = sums.get(slot)
        if previous is None or previous.shape != squares.shape:
            # A shape change means the module was re-entered with a different
            # in_features, which cannot happen for a fixed model; restarting the
            # accumulation is the conservative response to a state that should not
            # exist, and the token count restarts with it so the mean stays a mean.
            sums[slot] = squares
            tokens[slot] = count
            return
        previous.add_(squares)
        tokens[slot] = tokens.get(slot, 0.0) + count

    def _grad_output_hook(self, slot: int, grad: Tensor) -> None:
        entry = self._tracked[slot]
        gram_x = self._gram_x.pop(slot, None)
        if gram_x is None and not self._moment_active:
            return
        try:
            with torch.no_grad():
                if self._moment_active:
                    self._accumulate_output_moment(entry, grad)
                if gram_x is None:
                    return
                gram_d = gram(grad, self.config.subsample_tokens)
                if gram_d.shape[0] != gram_x.shape[0]:
                    # The forward that produced gram_x saw a different token count than
                    # this backward. Normal training cannot do that; an unusual
                    # checkpointing or pipeline schedule can, and a wrong-shaped product
                    # is worse than a skipped observation. Reading ``.shape`` is a Python
                    # attribute access, so the guard stays inside the no-dispatch budget.
                    return
                # Same duplicate rule as the saliency flush, and one test covers both
                # queues because a single hook call fills them together.
                if slot in self._bwd_pending:
                    self._flush_backward()
                # The contraction is deferred, not skipped: see _flush_backward for why
                # every module's is the same shape and can therefore run as one.
                self._ctr_slots.append(slot)
                self._ctr_gram_x.append(gram_x)
                self._ctr_gram_d.append(gram_d)
                self._ctr_bytes += 4 * (gram_x.numel() + gram_d.numel())
                entry.pending_observations += 1
                self._bwd_pending.add(slot)
                if self.config.coherence_ema_beta is not None:
                    self._queue_coherence(entry, grad)
                if self._ctr_bytes >= _CONTRACTION_QUEUE_BYTES:
                    self._flush_backward()
        except Exception as exc:  # noqa: BLE001 -- a hook must not kill the run
            self._note_error(entry, "backward", exc)

    def _stage(self, slot: int, norm: Tensor) -> None:
        """Queue one micro-batch observation. No dispatch: the fold is batched."""
        self._bwd_slots.append(slot)
        self._bwd_values.append(norm)

    def _queue_coherence(self, entry: _Tracked, grad: Tensor) -> None:
        """Reduce the output gradient to per-channel norms; compare at the flush.

        Two dispatches rather than the fourteen a per-module cosine costs. The
        vector is kept for the next call to compare against, which is why this
        stores into ``_prev_channel`` before deciding whether a comparison is
        possible at all.
        """
        current = channel_norm(grad)
        previous = self._prev_channel.get(entry.slot)
        self._prev_channel[entry.slot] = current
        if previous is None or previous.shape != current.shape:
            # Shape changes are exactly the condition that raised inside the
            # supplement's bare except. Here it costs one skipped observation and
            # is visible in coherence_calls.
            return
        self._coh_slots.append(entry.slot)
        self._coh_current.append(current)
        self._coh_previous.append(previous)
        entry.coherence_calls += 1

    def _flush_backward(self) -> None:
        """Fold queued gradient norms and coherence pairs, batched across modules.

        The backward twin of :meth:`_flush_saliency`, and the same trade: the hook
        keeps only reductions that must run per module, and the arithmetic joining
        them to the accumulators runs once for the whole model.

        The cosine batches on the same non-negativity argument the saliency flush
        uses, one step removed. ``channel_norm`` returns L2 norms, so both operands
        are elementwise non-negative, so their product is too -- and the L1 norm of
        that product is therefore the dot product the cosine needs.

        The ``outer_exact`` contraction batches for a different reason: a Gram matrix
        is ``[T, T]`` in the *token* count, and ``subsample_tokens`` caps ``T``, so
        every module's contraction is identically shaped. Grouping by ``T`` is not
        decoration -- an MoE expert sees only its routed tokens, so a single step
        genuinely produces more than one shape.
        """
        if not (self._bwd_slots or self._ctr_slots or self._coh_slots):
            return
        slots, values = self._bwd_slots, self._bwd_values
        ctr_slots, gram_xs, gram_ds = self._ctr_slots, self._ctr_gram_x, self._ctr_gram_d
        coh_slots, currents, previouses = self._coh_slots, self._coh_current, self._coh_previous
        self._bwd_slots, self._bwd_values = [], []
        self._ctr_slots, self._ctr_gram_x, self._ctr_gram_d = [], [], []
        self._ctr_bytes = 0
        self._coh_slots, self._coh_current, self._coh_previous = [], [], []
        self._bwd_pending.clear()

        with torch.no_grad():
            # (slots, norms) groups: the param/lowrank estimators stage scalars already,
            # outer_exact contributes one group per distinct token count.
            groups: list[tuple[list[int], Tensor]] = []
            if values:
                groups.append((slots, torch.stack(values).reshape(-1)))
            for tokens in dict.fromkeys(t.shape[-1] for t in gram_xs):
                picked = [i for i, t in enumerate(gram_xs) if t.shape[-1] == tokens]
                norms = outer_grad_sq_batched(
                    [gram_xs[i] for i in picked], [gram_ds[i] for i in picked]
                ).sqrt()
                groups.append(([ctr_slots[i] for i in picked], norms))

            for group_slots, norms in groups:
                index = torch.as_tensor(group_slots, dtype=torch.long, device=self._device)
                # index_add_, not index_copy_: micro-batch observations for one module
                # sum, and on_optimizer_step divides by the count to get the mean.
                self._grad_stage.index_add_(0, index, norms.to(self._device, torch.float32))

            if not coh_slots:
                return
            beta = self.config.coherence_ema_beta
            assert beta is not None  # only queued when coherence is enabled
            dots = torch.stack(torch._foreach_norm(torch._foreach_mul(currents, previouses), 1))
            cur_norm = torch.stack(torch._foreach_norm(currents))
            prev_norm = torch.stack(torch._foreach_norm(previouses))
            cosine = (dots / (cur_norm * prev_norm + self.config.eps)).to(
                self._device, torch.float32
            )
            coh_index = torch.as_tensor(coh_slots, dtype=torch.long, device=self._device)
            self._coh_ema.index_copy_(
                0, coh_index, self._coh_ema.index_select(0, coh_index).lerp(cosine, 1.0 - beta)
            )

    def _note_error(self, entry: _Tracked, phase: str, exc: Exception) -> None:
        entry.hook_errors += 1
        key = f"{entry.name}:{phase}"
        if key not in self._logged_errors:
            self._logged_errors.add(key)
            _log.warning(
                "signal %s hook failed for %s (%s: %s); continuing, and the count "
                "will be reported in provenance.notes['hook_errors']",
                phase,
                entry.name,
                type(exc).__name__,
                exc,
                exc_info=_log.isEnabledFor(10),
            )

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def on_optimizer_step(self) -> None:
        """Fold this step's staged observations into Welford. One step, one fold.

        This is where plasticity becomes a variance over *optimizer steps* rather
        than over micro-batches. Must be called before gradients are zeroed --
        ``param`` and ``lowrank`` read ``.grad`` here.
        """
        # This step's forward passes are done, so nothing more can arrive for the
        # slots still pending. Folding them here keeps the deferral invisible to
        # anyone reading the accumulators between steps.
        self._flush_saliency()
        if self._estimator is GradEstimatorMode.PARAM:
            self._collect_param_grads()
        elif self._estimator is GradEstimatorMode.LOWRANK:
            self._collect_lowrank_grads()
        # After the collectors, not before: `lowrank` stages from inside them, and a
        # flush that ran first would leave its observations for the next step.
        self._flush_backward()
        # Releases the one activation the input-Gram cache pins. Nothing can hit it
        # again -- the next forward pass allocates new tensors.
        self._last_gram = None

        pending = torch.as_tensor(
            [float(e.pending_observations) for e in self._tracked],
            dtype=torch.float32,
            device=self._device,
        )
        active = pending > 0
        # Mean of the micro-batch norms is the step's observation. Under gradient
        # accumulation this is not the norm of the accumulated gradient -- that
        # would need the cross-micro-batch terms, which no bounded-memory scheme
        # recovers. It is a monotone proxy, which is all a percentile rank needs,
        # and `param` mode measures the accumulated gradient exactly for anyone
        # who needs that instead.
        observation = self._grad_stage / pending.clamp_min(1.0)
        zero = torch.zeros((), dtype=torch.float32, device=self._device)

        count = self._grad_n + active.to(self._grad_n.dtype)
        delta = observation - self._grad_mean
        self._grad_mean.add_(torch.where(active, delta / count.clamp_min(1.0), zero))
        self._grad_m2.add_(torch.where(active, delta * (observation - self._grad_mean), zero))
        self._grad_n = count

        self._grad_stage.zero_()
        for entry in self._tracked:
            entry.pending_observations = 0
        self._optimizer_steps += 1
        # Decided once per step, after the increment, so every micro-batch of the
        # next step sees the same answer on both the forward and the backward side.
        # Step 0 samples: a run shorter than channel_moment_every should still
        # produce moments rather than an empty sidecar.
        self._moment_active = (
            self.config.collect_channel_moments
            and self._optimizer_steps % self.config.channel_moment_every == 0
        )

    def _collect_param_grads(self) -> None:
        """Legacy estimator: the norm of whatever gradient exists.

        ``torch._foreach_norm`` computes every norm in one fused multi-tensor
        launch -- the same primitive ``clip_grad_norm_`` uses -- so the whole model
        costs a couple of kernels instead of one per module.
        """
        grads: list[Tensor] = []
        slots: list[int] = []
        for entry in self._tracked:
            grad = _legacy_param_grad(entry)
            if grad is not None:
                grads.append(grad)
                slots.append(entry.slot)
                entry.pending_observations = 1
        if not grads:
            return
        norms = torch.stack(torch._foreach_norm(grads)).to(self._device, torch.float32)
        index = torch.as_tensor(slots, dtype=torch.long, device=self._device)
        self._grad_stage.index_copy_(0, index, norms)

    def _collect_lowrank_grads(self) -> None:
        for entry in self._tracked:
            factors = _lora_factors(entry.module)
            if factors is None:
                continue
            a, b, scaling = factors
            if a.grad is None or b.grad is None:
                continue
            try:
                value = lowrank_grad_sq(a, b, a.grad, b.grad, scaling).sqrt()
            except Exception as exc:  # noqa: BLE001 -- a hook must not kill the run
                self._note_error(entry, "lowrank", exc)
                continue
            self._stage(entry.slot, value.to(self._device, torch.float32))
            entry.pending_observations = 1

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> StatsFile:
        """Materialise the accumulators into a :class:`StatsFile`.

        The only place the host reads device state: one ``.cpu()`` on a stacked
        ``[5, n_modules]`` tensor, so one synchronisation for the entire model no
        matter how many modules it has.
        """
        self._check_hook_errors()
        # Before reading _act_ema, not after: a snapshot taken mid-step (log_every, or
        # a callback firing between micro-batches) would otherwise omit observations
        # already made.
        self._flush_saliency()
        # Coherence lands in _coh_ema, which is read below. Staged gradient norms move
        # into _grad_stage, which on_optimizer_step still folds afterwards -- the flush
        # clears its queues, so nothing is counted twice.
        self._flush_backward()
        stacked = torch.stack(
            [self._act_ema, self._grad_mean, self._grad_m2, self._grad_n, self._coh_ema]
        )
        assert stacked.shape[0] == _N_DEVICE_ACCUMULATORS
        act, mean, m2, count, coherence = (row.tolist() for row in stacked.cpu())

        beta_a = self.config.activation_ema_beta
        beta_c = self.config.coherence_ema_beta
        layers: dict[str, LayerStats] = {}
        for entry in self._tracked:
            i = entry.slot
            n = round(count[i])
            layers[entry.name] = LayerStats(
                name=entry.name,
                activation_rms_ema=_debias(act[i], beta_a, entry.forward_calls),
                grad_norm_count=n,
                grad_norm_mean=mean[i],
                grad_norm_var=(m2[i] / (n - 1)) if n >= 2 else 0.0,
                coherence_ema=(
                    _debias(coherence[i], beta_c, entry.coherence_calls)
                    if beta_c is not None and entry.coherence_calls > 0
                    else None
                ),
                param_count=entry.weight.numel(),
                role=entry.role.value,
                routing_hits=(entry.forward_calls if entry.role in _MOE_EXPERT_ROLES else None),
                forward_calls=entry.forward_calls,
                grad_estimator=self._estimator.value,
            )

        return StatsFile(
            layers=layers,
            activation_ema_beta=beta_a,
            coherence_ema_beta=beta_c,
            eps=self.config.eps,
            provenance=self._provenance(),
        )

    def channel_moments(self, *, reduce: bool = True) -> ChannelMoments:
        """Per-channel second moments, normalised to means and moved to host memory.

        A module appears only if it has a non-empty accumulation *and* a positive
        token count, and the two halves are independent: half a pair is kept rather
        than dropped, because the estimator needs both and says so, and a missing
        half is far easier to diagnose when the present half is visible.

        Reads the accumulators, never writes them, so a run with ``log_every`` set
        can call this at every flush and at the end without any observation being
        counted twice -- which an in-place all-reduce would do.
        """
        from .moments import ChannelMoments

        totals = self._reduced_moments() if reduce else self._local_moments()
        moments = ChannelMoments()
        for entry in self._tracked:
            for (sums, tokens), target in zip(
                totals[:2], (moments.input_sq, moments.output_grad_sq), strict=True
            ):
                total = sums.get(entry.slot)
                seen = tokens.get(entry.slot, 0.0)
                if total is None or seen <= 0:
                    continue
                target[entry.name] = (total / seen).to("cpu", torch.float32)
            observations = totals[2].get(entry.slot, 0)
            if observations:
                moments.observations[entry.name] = observations
        return moments

    def _local_moments(self) -> tuple[_MomentHalf, _MomentHalf, dict[int, int]]:
        return (
            (self._moment_x, self._moment_x_tokens),
            (self._moment_d, self._moment_d_tokens),
            self._moment_obs,
        )

    def _reduced_moments(self) -> tuple[_MomentHalf, _MomentHalf, dict[int, int]]:
        """Sum the accumulators across ranks into fresh tensors. Local state untouched.

        Sums and token counts are both additive, which is why the running state is a
        sum and not a mean: reducing means would need a weighted average, and would be
        wrong the moment two ranks saw different token counts -- which under a
        variable-length collator they routinely do.

        The collective walks :attr:`_tracked` in a fixed order and materialises zeros
        for slots that never fired here, so every rank issues the same sequence of
        all-reduces even when routing sent an expert no tokens on one of them. Key
        sets that differ between ranks do not raise; they hang.
        """
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return self._local_moments()

        linear = [e for e in self._tracked if e.kind == "linear"]
        halves: list[_MomentHalf] = []
        for sums, tokens, axis in (
            (self._moment_x, self._moment_x_tokens, 1),
            (self._moment_d, self._moment_d_tokens, 0),
        ):
            out_sums: dict[int, Tensor] = {}
            out_tokens: dict[int, float] = {}
            counters = torch.as_tensor(
                [tokens.get(e.slot, 0.0) for e in linear],
                dtype=torch.float32,
                device=self._device,
            )
            torch.distributed.all_reduce(counters)
            for entry, seen in zip(linear, counters.tolist(), strict=True):
                local = sums.get(entry.slot)
                total = (
                    local.clone()
                    if local is not None
                    else torch.zeros(
                        entry.weight.shape[axis], dtype=torch.float32, device=self._device
                    )
                )
                torch.distributed.all_reduce(total)
                out_sums[entry.slot] = total
                out_tokens[entry.slot] = seen
            halves.append((out_sums, out_tokens))

        observed = torch.as_tensor(
            [float(self._moment_obs.get(e.slot, 0)) for e in linear],
            dtype=torch.float32,
            device=self._device,
        )
        torch.distributed.all_reduce(observed)
        counts = {e.slot: round(c) for e, c in zip(linear, observed.tolist(), strict=True)}
        return halves[0], halves[1], counts

    def _check_hook_errors(self) -> None:
        total = sum(e.hook_errors for e in self._tracked)
        if total <= self.config.max_hook_errors:
            return
        worst = sorted(self._tracked, key=lambda e: -e.hook_errors)[:5]
        detail = ", ".join(f"{e.name} ({e.hook_errors})" for e in worst if e.hook_errors)
        raise SignalCollectionError(
            f"{total} signal hook failures exceeded max_hook_errors="
            f"{self.config.max_hook_errors}. The resulting stats would be quietly "
            f"incomplete, which is worse than no stats. Worst offenders: {detail}. "
            f"Enable DYNQUANT_LOG_LEVEL=DEBUG for tracebacks, or raise "
            f"max_hook_errors to accept the degraded collection."
        )

    def _provenance(self) -> Provenance:
        config = getattr(self.model, "config", None)
        notes: dict[str, Any] = {
            "saliency_source": self.config.saliency_source,
            "subsample_tokens": self.config.subsample_tokens,
            "tracked_modules": len(self._tracked),
            "log_every": self.config.log_every,
        }
        if self.config.collect_channel_moments:
            # The cadence belongs in the stats file, not only in the sidecar: a reader
            # comparing two runs needs to know that one averaged its moments over 30
            # steps and the other over 500 before it compares their allocations.
            notes["channel_moments"] = {
                "every": self.config.channel_moment_every,
                "sampled_steps": 1 + self._optimizer_steps // self.config.channel_moment_every,
                "modules": len(self._moment_obs),
            }
        tied = self._tied_groups()
        if tied:
            notes["tied_parameters"] = tied
        unexercised = sorted(e.name for e in self._tracked if e.forward_calls == 0)
        if unexercised:
            notes["unexercised_modules"] = unexercised
        errors = {e.name: e.hook_errors for e in self._tracked if e.hook_errors}
        if errors:
            notes["hook_errors"] = errors
        if self._skipped:
            notes["skipped_modules"] = dict(sorted(self._skipped.items()))
        if self._recompute_calls:
            # Recorded as what happened, not as the config flag, because the flag alone does
            # not say whether it mattered: observe_recompute=True on a run without
            # checkpointing produces identical statistics to the default. A non-zero count
            # with "observed": true is the one combination where the saliency EMA has two
            # horizons -- ~1/(1-beta) optimizer steps for modules outside the checkpointed
            # blocks, half that for the ones inside -- and score's percentile ranking then
            # compares two different statistics. Both shipped v1 stats files were collected
            # that way, before this counter existed; they carry no such note, which is itself
            # the reason to emit one now.
            notes["recompute_forward_calls"] = {
                "count": self._recompute_calls,
                "observed": self.config.observe_recompute,
            }

        return Provenance(
            dynquant_version=__version__,
            model_name_or_path=getattr(config, "_name_or_path", None),
            model_type=getattr(config, "model_type", None),
            num_optimizer_steps=self._optimizer_steps,
            grad_estimator=self._estimator.value,
            world_size=_world_size(),
            created_at_unix=time.time(),
            canonical_names=True,
            notes=notes,
        )

    def _tied_groups(self) -> dict[str, list[str]]:
        """Modules that share one storage, and therefore one bit width.

        Not a curiosity. Qwen3.5-2B sets ``tie_word_embeddings: true`` over a
        248,320-entry vocabulary, so ``embed_tokens.weight`` and ``lm_head.weight``
        are the same 508M-parameter tensor -- a quarter of the model. The paper's
        policy of "4-bit embedding, 8-bit LM head" is not expressible on it: one
        tensor admits one bit width. Recording the tie here is what lets the
        allocator make a single decision at the stricter of the two floors instead
        of silently letting whichever module it visits second win.
        """
        groups: dict[int, list[str]] = {}
        for entry in self._tracked:
            groups.setdefault(id(entry.weight), []).append(entry.name)
        return {names[0]: names[1:] for names in groups.values() if len(names) > 1}

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save(self, path: str | Path, *, reduce: bool = True) -> Path | None:
        """Write the stats file. Returns the path, or ``None`` on a non-zero rank.

        Reduces across ranks first, so the artifact reflects every micro-batch the
        job saw rather than only rank 0's slice.

        The channel-moment sidecar is written beside it when there is one. Beside
        rather than inside: it is two vectors per module against the stats file's
        four scalars, and every consumer that only wants the scalars would otherwise
        have to parse it. The stats file is complete and usable without it.
        """
        stats = self.snapshot()
        # Before the rank-0 early return below, because it is collective: every rank
        # has to reach the all-reduce or the ones that did are still waiting.
        moments = (
            self.channel_moments(reduce=reduce) if self.config.collect_channel_moments else None
        )
        if reduce:
            from .reduce import reduce_stats

            stats = reduce_stats(stats)
            if not _is_main_rank():
                return None
        written = stats.save(path)
        self._flushes += 1
        _log.info("wrote signal stats for %d modules to %s", len(stats), written)
        if moments is not None and len(moments):
            from .moments import save_moments

            sidecar = save_moments(moments, written.parent)
            _log.info("wrote channel moments to %s -- %s", sidecar, moments.summary())
        return written

    def maybe_flush(self, path: str | Path) -> Path | None:
        """Write intermediate stats if ``log_every`` says it is time.

        Cheap insurance on preemptible hardware: a run that dies at step 900 of
        1000 otherwise yields nothing at all.
        """
        every = self.config.log_every
        if every <= 0 or self._optimizer_steps == 0:
            return None
        if self._optimizer_steps % every != 0:
            return None
        return self.save(path)

    def reset(self) -> None:
        """Zero every accumulator. Hooks stay attached."""
        for buffer in (
            self._act_ema,
            self._grad_stage,
            self._grad_mean,
            self._grad_m2,
            self._grad_n,
            self._coh_ema,
        ):
            buffer.zero_()
        self._gram_x.clear()
        self._prev_channel.clear()
        for accumulator in (
            self._moment_x,
            self._moment_d,
            self._moment_x_tokens,
            self._moment_d_tokens,
            self._moment_obs,
        ):
            accumulator.clear()
        self._moment_active = self.config.collect_channel_moments
        # Dropped rather than flushed, the opposite of detach(): a queued observation
        # belongs to the history this call exists to forget, and folding it into a
        # freshly zeroed EMA would leak one measurement across the boundary.
        self._discard_queues()
        for entry in self._tracked:
            entry.forward_calls = 0
            entry.coherence_calls = 0
            entry.pending_observations = 0
            entry.hook_errors = 0
        self._optimizer_steps = 0
        self._recompute_calls = 0

    def __len__(self) -> int:
        return len(self._tracked)

    def __iter__(self) -> Iterator[str]:
        return iter(self.tracked_names)

    def __repr__(self) -> str:
        state = "attached" if self._attached else "detached"
        return (
            f"SignalTracker({len(self._tracked)} modules, {state}, "
            f"estimator={self._estimator.value}, steps={self._optimizer_steps})"
        )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _debias(value: float, beta: float, calls: int) -> float:
    """Undo the zero-initialisation bias of an EMA after ``calls`` updates.

    Returns ``value / (1 - beta**calls)``. At ``calls == 1`` that is exactly the
    first observation, matching a first-call assignment without needing to know on
    the host whether a call had happened yet.
    """
    if calls <= 0:
        return 0.0
    correction = 1.0 - beta**calls
    if correction <= 0.0:  # beta**calls underflowed to 1.0 only if beta == 1
        return value
    return value / correction


def _first_tensor(value: Any) -> Tensor | None:
    """First tensor in a hook payload, which may be a tensor, tuple, list or dict."""
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
        return None
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _quantizable_weight(module: nn.Module) -> nn.Parameter | None:
    """The weight this module owns, if the quantizer would want it.

    ``ndim >= 2`` is the test: it admits Linear, Embedding, Conv1d and Conv2d and
    excludes every norm, bias and scalar gate without needing to enumerate the
    dozens of norm classes across architectures. Unwraps PEFT's ``base_layer`` and
    ``original_module`` so a wrapped module reports the weight that will actually
    be written to disk.
    """
    weight = getattr(module, "weight", None)
    if isinstance(weight, nn.Parameter) and weight.ndim >= 2:
        return weight
    for attribute in ("base_layer", "original_module"):
        inner = getattr(module, attribute, None)
        if isinstance(inner, nn.Module):
            return _quantizable_weight(inner)
    return None


def _unmeasurable_experts(name: str, module: nn.Module) -> dict[str, str]:
    """Batched MoE expert tensors this tracker declines to measure, and why.

    Empty for anything that is not an expert bank.

    The graph and the quantizer both handle these tensors (see
    :mod:`dynquant.graph.experts`). The tracker does not, and the reason is not
    plumbing -- it is that **neither signal is observable at the bank's module
    boundary**:

    * The forward hook sees the bank's input, the routed hidden states. That is the
      input to ``gate_up_proj``, but ``down_proj``'s input is the post-SwiGLU
      intermediate, which never crosses a module boundary.
    * The backward hook sees the gradient of the bank's output. That is
      ``down_proj``'s output gradient, but not ``gate_up_proj``'s.

    So a boundary hook hands each of the two tensors one correct signal and one
    measured from the wrong tensor. The importance score is the *product* of the two
    ranks, so a wrong rank does not raise -- it yields a complete, plausible bit map
    driven partly by the wrong ordering. That is the same failure shape as the
    exact-zero rank artifact, which took a full re-run to catch, and it is worse here
    because there is no anomalous number to notice.

    Refusing leaves these tensors unmeasured, which the allocator treats as a neutral
    score. On a large MoE that is most of the model, so the method's central claim is
    inoperative there until per-expert tracking exists -- hence the warning in
    :meth:`SignalTracker._discover` states the parameter share out loud rather than
    letting an empty stats entry pass for a measured one.
    """
    params = batched_expert_params(module)
    if not params:
        return {}
    bank = canonical_name(name)
    reason = (
        "batched MoE expert tensor: activation and gradient signals are not separable "
        "at the bank's module boundary (see dynquant.graph.experts)"
    )
    return {f"{bank}.{param_name}": reason for param_name, _ in params}


def _module_kind(module: nn.Module) -> str:
    for attribute in ("base_layer", "original_module"):
        inner = getattr(module, attribute, None)
        if isinstance(inner, nn.Module):
            return _module_kind(inner)
    if isinstance(module, nn.Embedding):
        return "embedding"
    if isinstance(module, nn.Linear):
        return "linear"
    # Duck-typed: quantization-aware and custom linear layers (bitsandbytes
    # Linear4bit, tensor-parallel shards, Megatron ColumnParallelLinear) are not
    # nn.Linear subclasses but are shaped like one, and the Gram identity depends
    # on the shape, not the class.
    if isinstance(module, nn.modules.conv._ConvNd):
        return "other"
    weight = getattr(module, "weight", None)
    if isinstance(weight, nn.Parameter) and weight.ndim == 2:
        return "linear"
    return "other"


def _legacy_param_grad(entry: _Tracked) -> Tensor | None:
    """The gradient the ``param`` estimator can see.

    Prefers the weight itself, which is right under full fine-tuning. Under LoRA
    the weight is frozen, so this falls back to the adapter tensors -- the
    research code's behaviour, reproduced deliberately for ``--preset paper-3.15``
    and flagged as ``grad_estimator="param"`` so nobody mistakes it for a
    measurement of the base weight.
    """
    if entry.weight.grad is not None:
        return entry.weight.grad
    for _, parameter in entry.module.named_parameters(recurse=True):
        if parameter.grad is not None:
            return parameter.grad
    return None


def _lora_factors(module: nn.Module) -> tuple[Tensor, Tensor, float] | None:
    """Extract ``(A, B, scaling)`` from a PEFT LoRA layer, if this is one.

    Reaches into ``lora_A`` / ``lora_B`` ``ModuleDict``s by adapter name rather
    than assuming ``"default"``, because a multi-adapter model has none.
    """
    lora_a = getattr(module, "lora_A", None)
    lora_b = getattr(module, "lora_B", None)
    if not isinstance(lora_a, nn.ModuleDict) or not isinstance(lora_b, nn.ModuleDict):
        return None
    active = getattr(module, "active_adapters", None) or list(lora_a.keys())
    for name in active:
        if name not in lora_a or name not in lora_b:
            continue
        a = getattr(lora_a[name], "weight", None)
        b = getattr(lora_b[name], "weight", None)
        if not isinstance(a, Tensor) or not isinstance(b, Tensor):
            continue
        scaling = getattr(module, "scaling", {})
        factor = float(scaling.get(name, 1.0)) if isinstance(scaling, dict) else 1.0
        return a, b, factor
    return None


def _world_size() -> int | None:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return None
    return int(torch.distributed.get_world_size())


def _is_main_rank() -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return int(torch.distributed.get_rank()) == 0
