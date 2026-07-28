"""Combining saliency and plasticity into one number per module."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dynquant._logging import get_logger
from dynquant.graph.roles import ModuleRole

from .ranks import percentile_ranks

if TYPE_CHECKING:
    from dynquant.graph.classify import ModelGraph, ModuleInfo
    from dynquant.signals.schema import StatsFile

__all__ = ["COMBINE_MODES", "ScoreConfig", "ScoredModule", "score_modules"]

_log = get_logger(__name__)

COMBINE_MODES = ("plasticity", "rank_product")
"""How the ranks become one number. See :attr:`ScoreConfig.combine`."""

NEUTRAL_RANK = 0.5
"""Rank given to a module the training run never exercised.

Zero would be a lie of a specific kind: it says "measured, and found unimportant",
when the truth is "not measured". On a sparse MoE that distinction decides whether
a rarely-routed expert is quantized to 2 bits on the strength of evidence that was
never collected. The midpoint says "no information", which leaves the role floor in
charge -- the correct authority when the signal is silent."""


@dataclass(frozen=True, slots=True)
class ScoreConfig:
    """How the two signals are combined."""

    rank_within_role: bool = True
    """Rank each module against others in its own role rather than against the
    whole model.

    Global ranking sounds more principled and is worse in practice, because roles
    have systematically different magnitudes. Every MLP down-projection carries
    more activation RMS than every decay projection, so a global ranking sorts
    mostly by role and the within-role comparison -- which layer's gate matters
    most -- is squeezed into the noise. It gets pathological on MoE: 128 experts
    of one role can crowd attention out of the distribution entirely.

    Ranking within role asks the question the allocator actually needs answered:
    given that I must choose between these two MLP up-projections, which one?"""

    combine: str = "plasticity"
    """How the ranks become a score. One of :data:`COMBINE_MODES`.

    ``"rank_product"`` is the paper's Eq. (4), ``Rank(saliency) x Rank(plasticity)``,
    a soft AND: a module scores high only if it is both carrying magnitude and still
    moving. It is a good argument. It is also, on the only model where it has been
    checked against ground truth, worse than either factor alone.

    187 modules of a fine-tuned Qwen3.5-2B were quantized one at a time to 3 bits
    and the task loss measured. Ranked against the resulting disturbance *within
    role* -- the regime :attr:`rank_within_role` puts the allocator in, and the one
    that matters, since role floors already separate roles from each other:

    ==================================  =======  ==============
    combination                            rho    right sign in
    ==================================  =======  ==============
    plasticity alone                     +0.491           11/12
    rank product (saliency x plasticity) +0.231            9/12
    saliency alone                       -0.301            2/12
    ==================================  =======  ==============

    Activation RMS does not merely fail to help. It anti-correlates with disturbance
    in 10 of 12 roles, and multiplying it in drags a +0.491 signal down to +0.231.
    The reason is visible once stated: a module with large activations has large
    *outputs*, and the quantization error it contributes is relative to those
    outputs, so downstream normalisation divides much of it back out. Saliency
    measures scale; what the allocator needs is curvature.

    So the default drops it. ``rank_product`` stays for ``--preset paper-3.15``,
    which reproduces published numbers and therefore has to reproduce this too.

    Neither mode is the best available option -- see
    :mod:`dynquant.score.sensitivity`, which scores +0.521 by replacing the ranks
    entirely with a measured per-width error. This is the fallback for modules whose
    channel moments were not collected."""

    use_coherence: bool = False
    """Fold the gradient-coherence EMA in as a third rank.

    Off by default. The signal is real -- consistently-aligned gradients mean a
    weight is moving in a settled direction -- but it is collected per output
    channel and reduced to a scalar, and the supplement's own implementation
    computed it from mismatched LoRA factors, so no published number depends on
    it. Left available and off rather than removed."""

    min_forward_calls: int = 1
    """Below this many observed forward passes a module counts as unexercised and
    takes :data:`NEUTRAL_RANK` rather than its measured value."""

    def __post_init__(self) -> None:
        if self.combine not in COMBINE_MODES:
            raise ValueError(f"combine must be one of {COMBINE_MODES}, got {self.combine!r}")


@dataclass(frozen=True, slots=True)
class ScoredModule:
    """One module's score and the ranks it came from, kept for `dynquant inspect`."""

    name: str
    role: ModuleRole
    num_params: int
    score: float
    saliency_rank: float
    plasticity_rank: float
    coherence_rank: float | None = None
    exercised: bool = True

    @property
    def explain(self) -> str:
        parts = [f"saliency {self.saliency_rank:.3f}", f"plasticity {self.plasticity_rank:.3f}"]
        if self.coherence_rank is not None:
            parts.append(f"coherence {self.coherence_rank:.3f}")
        if not self.exercised:
            parts.append("UNEXERCISED")
        return ", ".join(parts)


@dataclass(slots=True)
class ScoreReport:
    """Scores plus what could not be scored, so silence is visible."""

    modules: dict[str, ScoredModule] = field(default_factory=dict)
    missing_stats: tuple[str, ...] = ()
    """Quantizable modules with no entry in the stats file at all -- a name
    mismatch, or a module the fine-tune never touched."""

    unexercised: tuple[str, ...] = ()
    """Present in the stats file but with no forward passes recorded."""

    def scores(self) -> dict[str, float]:
        return {name: scored.score for name, scored in self.modules.items()}

    def summary(self) -> str:
        lines = [f"scored {len(self.modules)} modules"]
        if self.missing_stats:
            lines.append(
                f"  {len(self.missing_stats)} with no stats entry "
                f"(e.g. {', '.join(self.missing_stats[:3])})"
            )
        if self.unexercised:
            lines.append(
                f"  {len(self.unexercised)} never exercised "
                f"(e.g. {', '.join(self.unexercised[:3])})"
            )
        return "\n".join(lines)


def score_modules(
    graph: ModelGraph,
    stats: StatsFile,
    config: ScoreConfig | None = None,
) -> ScoreReport:
    """Score every quantizable module in ``graph`` from the signals in ``stats``.

    The graph, not the stats file, decides which modules exist. A stats file can be
    missing entries (a module the run never reached) or carry stale ones (a module
    that was renamed), and letting it drive the iteration would silently drop
    tensors from the allocation -- which does not fail, it just quietly leaves them
    at whatever the quantizer defaults to.
    """
    config = config or ScoreConfig()
    targets = list(graph.quantizable())

    saliency: dict[str, float] = {}
    plasticity: dict[str, float] = {}
    coherence: dict[str, float] = {}
    missing: list[str] = []
    unexercised: list[str] = []

    for info in targets:
        layer = stats.layers.get(info.name)
        if layer is None:
            missing.append(info.name)
            continue
        calls = layer.forward_calls
        if (calls is not None and calls < config.min_forward_calls) or layer.grad_norm_count == 0:
            unexercised.append(info.name)
            continue

        saliency[info.name] = layer.activation_rms_ema
        # log1p, not the raw variance. Gradient-norm variance spans many orders of
        # magnitude and its upper tail is a handful of layers; ranking is
        # monotone so the transform cannot change the order, but it keeps the
        # value finite and comparable when it is reported to a human.
        plasticity[info.name] = math.log1p(max(layer.grad_norm_var, 0.0))
        if layer.coherence_ema is not None:
            coherence[info.name] = layer.coherence_ema

    groups = _rank_groups(targets, saliency, plasticity, coherence, config)
    report = ScoreReport(missing_stats=tuple(missing), unexercised=tuple(unexercised))

    for info in targets:
        measured = info.name in saliency
        s_rank = groups["saliency"].get(info.name, NEUTRAL_RANK)
        p_rank = groups["plasticity"].get(info.name, NEUTRAL_RANK)
        c_rank = groups["coherence"].get(info.name) if config.use_coherence else None

        score = p_rank if config.combine == "plasticity" else s_rank * p_rank
        if c_rank is not None:
            score *= c_rank

        report.modules[info.name] = ScoredModule(
            name=info.name,
            role=info.role,
            num_params=info.num_params,
            score=score,
            saliency_rank=s_rank,
            plasticity_rank=p_rank,
            coherence_rank=c_rank,
            exercised=measured,
        )

    if missing:
        _log.warning(
            "%d quantizable modules have no stats entry; they take a neutral score "
            "and their role floor decides. First few: %s",
            len(missing),
            ", ".join(missing[:5]),
        )
    return report


def _rank_groups(
    targets: list[ModuleInfo],
    saliency: dict[str, float],
    plasticity: dict[str, float],
    coherence: dict[str, float],
    config: ScoreConfig,
) -> dict[str, dict[str, float]]:
    """Rank each signal, either globally or once per role."""
    if not config.rank_within_role:
        return {
            "saliency": percentile_ranks(saliency),
            "plasticity": percentile_ranks(plasticity),
            "coherence": percentile_ranks(coherence),
        }

    by_role: dict[ModuleRole, list[str]] = {}
    for info in targets:
        by_role.setdefault(info.role, []).append(info.name)

    out: dict[str, dict[str, float]] = {"saliency": {}, "plasticity": {}, "coherence": {}}
    for names in by_role.values():
        members = set(names)
        for signal, values in (
            ("saliency", saliency),
            ("plasticity", plasticity),
            ("coherence", coherence),
        ):
            subset = {name: value for name, value in values.items() if name in members}
            out[signal].update(percentile_ranks(subset))
    return out
