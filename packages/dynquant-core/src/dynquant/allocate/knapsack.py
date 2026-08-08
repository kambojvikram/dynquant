"""Greedy ROI allocation of bit widths under a size budget.

The problem is a multiple-choice knapsack: each module picks one width from
``{2,3,4,8}``, widths cost params * bits, and total cost must fit the budget. Exact
solutions exist but are not worth it -- the scores driving the objective are
percentile ranks, so the objective is ordinal, and optimising it to the last bit
optimises noise. Greedy by marginal-value-per-bit is the standard answer and it is
what the supplement used.

Two things here are not what the supplement did.

**Marginal value is concave in bits.** Uniform quantization error falls roughly
4x per added bit, so the *improvement* from 2→3 bits is far larger than from 4→8.
Valuing every upgrade step equally -- the supplement's model -- makes an 8-bit
upgrade of a small important tensor compete evenly with lifting a huge 2-bit MLP,
and it picks wrong often enough that the authors had to hardcode the 8-bit upgrade
out of the loop entirely with a comment about "gibberish". Modelling the concavity
removes both the wrong choice and the need for the hack.

**Infeasible targets allocate rather than give up.** When floors cost more than the
budget the supplement returned the floor map, so at every target below ~3.5 bits
the importance scores had no effect at all. Here the allocator starts from the
floors and *downgrades* by lowest damage, so scores drive the result in both
directions, and it reports exactly which floors it had to breach.

Both directions price a move as ``score * num_params * Δerror`` against a cost
measured in bits. The ``num_params`` factor is load-bearing and easy to leave out:
quantization damage is *extensive*, because perturbing N weights by a relative error
injects error proportional to N. Without it the ratio reduces to ``score * Δerror /
num_params``, which sorts by size rather than by importance -- the biggest tensor in
the model becomes the cheapest thing to cut and the smallest becomes the best thing
to upgrade. With it, ``num_params`` cancels against the cost term and the decision
falls to ``score * Δerror / Δbits``, which is the ordinal comparison the percentile
ranks were built for.

**Prefer a measured price when there is one.** Everything above describes the
fallback. ``score * num_params * Δerror`` is two guesses multiplied: a percentile
rank standing in for importance, and a *universal* ``4**-bits`` curve standing in
for what a width change actually costs this particular tensor. The second guess is
the weaker one -- it is identical for every module in the model, so it can express
"more bits is better" and nothing else, while the entire purpose of a bit allocator
is to exploit the fact that some tensors lose far more to 3-bit than others.

When :mod:`dynquant.score.sensitivity` has run, ``allocate_bits`` takes its table
instead: a per-module, per-width estimate of the loss increase, computed from the
*actual* quantization error at the *actual* candidate width weighted by channel
statistics from the fine-tune. A move is then priced by subtraction --
``sens[3] - sens[4]`` is exactly what the fourth bit buys -- and both the percentile
rank and the universal curve drop out of the arithmetic entirely. Measured against
true loss disturbance on a fine-tuned Qwen3.5-2B it ranks at +0.521 mean within-role
Spearman where the rank product manages +0.231.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field, replace
from statistics import median
from typing import TYPE_CHECKING

from dynquant._logging import get_logger
from dynquant.errors import DynQuantError
from dynquant.graph.roles import ModuleRole

from .budget import Budget, module_stored_bits
from .policy import AllocationPolicy

if TYPE_CHECKING:
    from collections.abc import Mapping

    from dynquant.graph.classify import ModelGraph
    from dynquant.score.sensitivity import SensitivityTable

__all__ = ["BitMap", "FloorViolation", "InfeasibleTargetError", "Pricing", "allocate_bits"]

_log = get_logger(__name__)


class InfeasibleTargetError(DynQuantError):
    """The budget cannot fit even the structural floors."""


def _error_scale(bits: int) -> float:
    """Relative quantization error at a given width.

    Uniform quantization with a fixed range halves its step size per added bit, so
    squared error falls by ~4x. This is the term that makes 2→3 worth more than
    4→8, which is the whole reason the allocator does not need an artificial cap
    on 8-bit upgrades.
    """
    return 4.0**-bits


@dataclass(frozen=True, slots=True)
class FloorViolation:
    """A module the budget forced below its role's preferred width."""

    name: str
    role: ModuleRole
    floor_bits: int
    assigned_bits: int
    num_params: int


@dataclass(frozen=True, slots=True)
class Pricing:
    """Which of the two prices the allocator used, and on how much of the model.

    Recorded because the split is not visible anywhere else and it can be nearly the
    whole model. On a batched-expert MoE the Gauss-Newton estimate is unavailable *by
    construction* -- a bank's forward spans two matmuls and a non-linearity, so no
    module boundary yields the ``dY = dW x`` pairing the Kronecker form needs -- and
    on LFM2.5-8B-A1B that is 44 tensors carrying 91.5% of the quantizable parameters.
    Forty-four of a hundred and thirty-three reads like a footnote; the same fact in
    parameters is the headline. Both are here so a reader cannot get the first without
    the second.

    ``scale`` is the multiplier :func:`_apply_fallback_scale` derived. It is ``1.0``
    when no rescaling was needed -- every module measured, or none of them, both being
    cases where one formula priced the whole heap and its order means something. It is
    ``None`` only for the case where the order does not: a mix of the two prices with
    no positive overlap to calibrate on. That distinction is the reason this is a field
    rather than a log line, since the run that hits it is the run whose bit map should
    not be trusted, and it looked identical to every other run in the artifacts.
    """

    measured_modules: int = 0
    proxied_modules: int = 0
    measured_params: int = 0
    proxied_params: int = 0
    scale: float | None = None

    @property
    def proxied_share(self) -> float:
        """Fraction of quantizable parameters priced by the proxy, not by measurement."""
        total = self.measured_params + self.proxied_params
        return self.proxied_params / total if total else 0.0

    def summary(self) -> str:
        if not self.proxied_modules:
            return f"all {self.measured_modules} modules priced from measured sensitivity"
        if not self.measured_modules:
            return f"all {self.proxied_modules} modules priced from the score proxy"
        scale = "no common scale -- their order is arbitrary"
        if self.scale is not None:
            scale = f"rescaled by {self.scale:.3e}"
        return (
            f"{self.measured_modules} modules measured, {self.proxied_modules} proxied "
            f"({self.proxied_share * 100:.1f}% of parameters, {scale})"
        )


@dataclass(slots=True)
class BitMap:
    """The allocation, plus everything needed to judge whether to trust it."""

    bits: dict[str, int] = field(default_factory=dict)
    violations: tuple[FloorViolation, ...] = ()
    budget_bits: float = 0.0
    allocated_bits: float = 0.0
    denominator: int = 0
    target_label: str = ""
    pricing: Pricing | None = None
    """How the modules were priced -- see :class:`Pricing`. ``None`` on maps built
    before the field existed, and on the hard-floor paths that never price anything."""

    @property
    def average_bits(self) -> float:
        """Stored bits per parameter, metadata and unquantized tensors included."""
        return self.allocated_bits / max(self.denominator, 1)

    @property
    def nbytes(self) -> int:
        return int(self.allocated_bits // 8)

    def histogram(self) -> dict[int, int]:
        """*Modules* at each width -- the fastest sanity check on a bit map.

        Modules and not parameters, which is worth saying because the two differ by six
        orders of magnitude here and a reader who assumes the other one gets a plausible
        answer rather than an error. ``summary()`` has always labelled it correctly;
        this docstring did not, and a table downstream printed module counts through a
        billions-scale formatter and rendered every width as ``0K``.
        """
        out: dict[int, int] = {}
        for width in self.bits.values():
            out[width] = out.get(width, 0) + 1
        return dict(sorted(out.items()))

    def summary(self) -> str:
        lines = [
            f"target {self.target_label}: {self.average_bits:.4f} avg bits "
            f"({self.nbytes / 1024**3:.3f} GiB over {self.denominator / 1e9:.4f}B params)",
            f"  modules per width: {self.histogram()}",
        ]
        if self.violations:
            worst = sorted(self.violations, key=lambda v: -v.num_params)[:5]
            lines.append(f"  {len(self.violations)} floors breached to fit the budget:")
            lines.extend(
                f"    {v.name} ({v.role.value}) {v.floor_bits}b -> {v.assigned_bits}b, "
                f"{v.num_params / 1e6:.1f}M params"
                for v in worst
            )
        return "\n".join(lines)


@dataclass(slots=True)
class _Candidate:
    name: str
    role: ModuleRole
    num_params: int
    score: float
    floor: int
    structural: bool
    bits: int
    cost_at: dict[int, float]

    sens: dict[int, float] | None = None
    """Measured loss estimate per width, when the channel moments were collected."""

    fallback_scale: float = 1.0
    """Multiplier putting the proxy price on the sensitivity table's scale.

    Only used when :attr:`sens` is ``None`` while other modules have one. See
    :func:`_apply_fallback_scale`."""

    def move_value(self, lower: int, upper: int) -> float:
        """What the width difference ``lower -> upper`` is worth, in loss units.

        Non-negative by construction. The MSE clipping search optimises per group,
        so on rare tensors it finds a slightly better clip at the lower width and
        the measured difference comes out below zero. That is a real property of the
        search, not an error, but it is not a reason to price an *upgrade* as
        harmful -- a zero says "this bit buys nothing", which is the honest reading.
        """
        if self.sens is not None:
            lo, hi = self.sens.get(lower), self.sens.get(upper)
            if lo is not None and hi is not None:
                return max(lo - hi, 0.0)
        proxy = self.score * self.num_params * (_error_scale(lower) - _error_scale(upper))
        return max(self.fallback_scale * proxy, 0.0)


def allocate_bits(
    graph: ModelGraph,
    scores: Mapping[str, float],
    budget: Budget,
    policy: AllocationPolicy | None = None,
    sensitivity: SensitivityTable | None = None,
) -> BitMap:
    """Assign a bit width to every quantizable module in ``graph``.

    Args:
        scores: Importance per module name, from :func:`dynquant.score.score_modules`.
            A missing name scores 0, which makes it the first thing downgraded and
            the last thing upgraded -- the right default for a module the training
            run never said anything about.
        sensitivity: Measured loss estimates per width, from
            :func:`dynquant.score.estimate_sensitivity`. When present it replaces
            ``scores`` for every module it covers; ``scores`` still prices the rest.
            Optional because it needs channel moments, which a stats file collected
            before they existed -- or with ``collect_channel_moments=False`` -- does
            not carry.

    Raises:
        InfeasibleTargetError: The budget cannot hold the structural floors even
            with everything else at the minimum width. Raised rather than
            silently returning an oversized map, because the caller asked for a
            size and would otherwise get a checkpoint that misses it.
    """
    policy = policy or AllocationPolicy()

    candidates: list[_Candidate] = []
    for info in graph.quantizable():
        floor = policy.floor_for(info.role, info.tied_roles)
        widths = sorted(set(policy.bit_options) | {floor})
        cost_at = {
            width: module_stored_bits(
                info,
                width,
                group_size=policy.group_size,
                symmetric=policy.symmetric,
            )
            for width in widths
            if width <= max(policy.max_bits, floor)
        }
        candidates.append(
            _Candidate(
                name=info.name,
                role=info.role,
                num_params=info.num_params,
                score=float(scores.get(info.name, 0.0)),
                floor=floor,
                structural=policy.is_structural(info.role, info.tied_roles),
                bits=floor,
                cost_at=cost_at,
                sens=sensitivity.values.get(info.name) if sensitivity is not None else None,
            )
        )
    pricing = _apply_fallback_scale(candidates)

    spent = budget.fixed_bits + sum(c.cost_at[c.bits] for c in candidates)
    if spent > budget.total_bits:
        spent = _downgrade(candidates, budget, policy, spent)
    # Always finish with an upgrade pass. After a downgrade run it reclaims what
    # the last cut overshot by: widths are discrete, so the step that finally got
    # under the budget usually undershoots it, and that slack is real precision
    # sitting unspent. The pass cannot undo the cut that created the slack -- that
    # would cost exactly what it saved and put the map back over budget -- so it
    # only ever finds a strictly cheaper upgrade elsewhere.
    spent = _upgrade(candidates, budget, spent)

    violations = tuple(
        FloorViolation(
            name=c.name,
            role=c.role,
            floor_bits=c.floor,
            assigned_bits=c.bits,
            num_params=c.num_params,
        )
        for c in candidates
        if c.bits < c.floor
    )
    result = BitMap(
        bits={c.name: c.bits for c in candidates},
        violations=violations,
        budget_bits=budget.total_bits,
        allocated_bits=spent,
        denominator=budget.denominator,
        target_label=budget.label,
        pricing=pricing,
    )
    _log.info("%s", result.summary())
    return result


def _apply_fallback_scale(candidates: list[_Candidate]) -> Pricing:
    """Put proxy-priced modules on the same scale as measured ones, or say why not.

    A heap that mixes ``sens[3] - sens[4]``, which is in loss units and typically
    ``1e-4``, with ``score * num_params * Δerror``, which is a rank times a
    parameter count and typically ``1e5``, is not making a comparison -- it is
    sorting every proxy-priced module ahead of every measured one, or behind, purely
    by which formula produced the number. Whichever way it falls, the modules with
    the *worse* information decide the allocation.

    So the proxy is rescaled by the ratio of medians. Medians rather than means
    because both populations are heavy-tailed, and a common multiplier rather than a
    per-module fit because there is nothing per-module to fit against: if there were,
    the module would have a measured value already.

    Both medians are taken over a *rung-normalised* price -- each module's next-step
    value divided by the width span that step covers -- because the two populations
    generally sit at different floors, and a module stepping 3->4 is worth about four
    times one stepping 4->8 for reasons that have nothing to do with units. Both
    formulas already carry that factor themselves, so calibrating on the raw step
    values applies it twice. On a model where the unmeasured modules happen to share
    a lower floor than the rest -- MLP up/down against 4-bit attention, which is the
    common case -- that double count scaled them by 0.33 and made them the first
    thing the downgrade pass cut, on no evidence at all.

    This makes the two comparable in aggregate. It does not make an individual
    proxy-priced module accurate, so a run where the fallback is doing most of the work
    has to be visible rather than inferred -- which is why the split comes back as a
    :class:`Pricing` and is written into the saved map, not only logged. A log line is
    not an artifact: the panel captures stdout, runs at the default WARNING level, and
    would have carried a bit map whose dominant pricing decision appeared nowhere in it.
    Nothing is rescaled when every module is measured, or none is; the census is
    returned either way, because "all measured" is also a fact worth being able to read.
    """
    measured = [c for c in candidates if c.sens is not None]
    proxied = [c for c in candidates if c.sens is None]
    census = Pricing(
        measured_modules=len(measured),
        proxied_modules=len(proxied),
        measured_params=sum(c.num_params for c in measured),
        proxied_params=sum(c.num_params for c in proxied),
        # 1.0 means "no rescaling was needed", which is true both when every module
        # was measured and when none was: one formula priced the whole heap either
        # way, so the order it produced is meaningful. None is reserved for the
        # single case where it is not -- a mix that could not be calibrated.
        scale=1.0 if not proxied or not measured else None,
    )
    if not measured or not proxied:
        return census

    def next_step(c: _Candidate) -> tuple[int, int] | None:
        above = [b for b in c.cost_at if b > c.bits]
        return (c.bits, min(above)) if above else None

    def typical(group: list[_Candidate], use_sens: bool) -> float:
        values = []
        for c in group:
            step = next_step(c)
            if step is None:
                continue
            lower, upper = step
            span = _error_scale(lower) - _error_scale(upper)
            if span <= 0:
                continue
            if use_sens:
                assert c.sens is not None
                lo, hi = c.sens.get(lower), c.sens.get(upper)
                value = max(lo - hi, 0.0) / span if lo is not None and hi is not None else 0.0
            else:
                value = c.score * c.num_params
            if value > 0:
                values.append(value)
        return median(values) if values else 0.0

    reference = typical(measured, use_sens=True)
    proxy = typical(proxied, use_sens=False)
    if reference <= 0 or proxy <= 0:
        # No overlap to calibrate against. Leaving the scale at 1.0 is arbitrary, and
        # saying so beats picking a number that looks principled and is not.
        _log.warning(
            "%d of %d modules have no measured sensitivity and no common scale could "
            "be established; their prices are not comparable with the rest and the "
            "allocation for them is effectively arbitrary",
            len(proxied),
            len(candidates),
        )
        return census

    scale = reference / proxy
    for c in proxied:
        c.fallback_scale = scale
    _log.info(
        "%d of %d modules priced from measured sensitivity; the remaining %d use the "
        "score proxy rescaled by %.3e to match",
        len(measured),
        len(candidates),
        len(proxied),
        scale,
    )
    return replace(census, scale=scale)


def _upgrade(
    candidates: list[_Candidate],
    budget: Budget,
    spent: float,
) -> float:
    """Spend the remaining budget on the highest value-per-bit upgrades.

    A max-heap keyed on marginal value per bit, re-pushed after each accepted
    move: taking a module from 2→3 changes what its 3→4 step is worth, so the
    next step has to be re-priced rather than pre-computed.
    """
    heap: list[tuple[float, str, int, int]] = []

    def push(index: int) -> None:
        c = candidates[index]
        options = [b for b in c.cost_at if b > c.bits]
        if not options:
            return
        nxt = min(options)
        gain = c.move_value(c.bits, nxt)
        cost = c.cost_at[nxt] - c.cost_at[c.bits]
        if cost <= 0:
            return
        heapq.heappush(heap, (-gain / cost, c.name, index, nxt))

    for index in range(len(candidates)):
        push(index)

    while heap:
        _, _, index, width = heapq.heappop(heap)
        c = candidates[index]
        if width <= c.bits:  # stale entry from before an earlier upgrade
            continue
        cost = c.cost_at[width] - c.cost_at[c.bits]
        if spent + cost > budget.total_bits:
            # Cannot afford this one. A cheaper upgrade may still fit, so keep
            # draining the heap rather than stopping at the first refusal.
            continue
        c.bits = width
        spent += cost
        push(index)
    return spent


def _downgrade(
    candidates: list[_Candidate],
    budget: Budget,
    policy: AllocationPolicy,
    spent: float,
) -> float:
    """Cut below the floors, cheapest damage first, until the budget is met.

    Structural floors are excluded from consideration entirely. If what remains
    cannot get under the budget, the target is infeasible and saying so is more
    useful than emitting a checkpoint that misses the size the caller asked for.
    """
    minimum = min(policy.bit_options)
    if not policy.soft_floors:
        _log.warning(
            "floors cost %.3f Gbit against a %.3f Gbit budget and soft_floors is off; "
            "returning the floor map unchanged (this is the paper's behaviour, and it "
            "means the importance scores had no effect on this allocation)",
            spent / 1e9,
            budget.total_bits / 1e9,
        )
        return spent

    movable = [i for i, c in enumerate(candidates) if not c.structural and c.bits > minimum]
    floor_cost = sum(
        c.cost_at[minimum] if not c.structural else c.cost_at[c.floor] for c in candidates
    )
    if budget.fixed_bits + floor_cost > budget.total_bits:
        raise InfeasibleTargetError(
            f"target {budget.label} needs {budget.total_bits / 1e9:.3f} Gbit but the "
            f"structural floors plus {minimum}-bit everywhere else already cost "
            f"{(budget.fixed_bits + floor_cost) / 1e9:.3f} Gbit. Raise the target, or "
            f"relax the structural roles if you have reason to."
        )

    heap: list[tuple[float, str, int, int]] = []

    def push(index: int) -> None:
        c = candidates[index]
        options = [b for b in c.cost_at if b < c.bits and b >= minimum]
        if not options:
            return
        nxt = max(options)
        damage = c.move_value(nxt, c.bits)
        saving = c.cost_at[c.bits] - c.cost_at[nxt]
        if saving <= 0:
            return
        heapq.heappush(heap, (damage / saving, c.name, index, nxt))

    for index in movable:
        push(index)

    while heap and spent > budget.total_bits:
        _, _, index, width = heapq.heappop(heap)
        c = candidates[index]
        if width >= c.bits:
            continue
        spent -= c.cost_at[c.bits] - c.cost_at[width]
        c.bits = width
        push(index)

    return spent
