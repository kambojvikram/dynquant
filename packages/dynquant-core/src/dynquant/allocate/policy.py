"""What the allocator is allowed to do.

The floors in :mod:`dynquant.graph.roles` encode two different kinds of knowledge
that the supplement's allocator treated identically, and that conflation is what
made its 3-bit configuration unreachable.

**Structural floors** are correctness constraints. An MoE router below 8 bits does
not degrade the model, it changes which experts fire; an MLA ``kv_a`` projection
below 8 bits corrupts the compressed KV cache every subsequent layer reads. No
budget justifies breaking these, and if the budget cannot fit them the honest
answer is that the target is infeasible.

**Quality floors** are preferences. An MLP up-projection is *better* at 3 bits than
at 2, but 2 bits is a working model. These must yield when the budget is tight,
because the alternative -- what the supplement did -- is to return the floor map
untouched and silently ignore the importance scores entirely at every target below
about 3.5 bits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dynquant.constants import BIT_OPTIONS, DEFAULT_GROUP_SIZE
from dynquant.graph.roles import DEFAULT_FLOOR_BITS, ModuleRole

__all__ = ["STRUCTURAL_ROLES", "AllocationPolicy"]

STRUCTURAL_ROLES: frozenset[ModuleRole] = frozenset(
    {
        # A discrete top-k decision. Rounding error here does not blur an output,
        # it routes a token to a different expert.
        ModuleRole.MOE_ROUTER,
        # Everything downstream reads the compressed KV latent these produce, so
        # their error is not local -- it is multiplied through every later layer.
        ModuleRole.MLA_KV_A,
        ModuleRole.MLA_Q_A,
        # Gate and decay coefficients for a recurrence. Error is not averaged away
        # across a sequence, it compounds along it.
        ModuleRole.LIN_ATTN_A,
        ModuleRole.LIN_ATTN_B,
        ModuleRole.SSM_X,
        ModuleRole.SSM_DT,
    }
)
"""Roles whose floor is a hard constraint rather than a preference.

``LM_HEAD`` is deliberately *not* here, despite being the tensor that produces every
logit. Its 8-bit floor is a strong preference, not a correctness constraint: a 4-bit
head gives slightly worse logits, where a 4-bit router gives a different expert. The
distinction is not academic. On a tied model the head *is* the embedding -- 26% of
Qwen3.5-2B's parameters -- so pinning it at 8 bits spends 4.07 of the 5.65 Gbit
available at a 3.0-bit target, leaving 1.15 bits each for everything else. Every
target below about 3.9 bits then becomes arithmetically impossible, which is exactly
the dead end this phase exists to remove; raising :class:`InfeasibleTargetError`
instead of silently returning the floor map would be a more honest way to fail, but
it would still be failing at a target that is perfectly reachable.

It keeps the highest floor in :data:`DEFAULT_FLOOR_BITS`, so the allocator still
gives it up reluctantly and reports the breach when it does."""


@dataclass(frozen=True, slots=True)
class AllocationPolicy:
    """Bit widths, floors, and how hard the floors bind."""

    bit_options: tuple[int, ...] = BIT_OPTIONS
    group_size: int = DEFAULT_GROUP_SIZE
    symmetric: bool = False

    soft_floors: bool = True
    """Allow quality floors to be breached when the budget demands it.

    ``False`` reproduces the supplement's behaviour, where an infeasible target
    silently returns the floor map. Kept for the ``paper-3.15`` preset, which has
    to reproduce published numbers including their defects."""

    structural_roles: frozenset[ModuleRole] = STRUCTURAL_ROLES
    floor_overrides: dict[ModuleRole, int] = field(default_factory=dict)
    max_bits: int = 8

    def floor_for(self, role: ModuleRole, tied_roles: tuple[ModuleRole, ...] = ()) -> int:
        """The preferred width for a role, strictest-wins across a tie."""
        floor = self.floor_overrides.get(role, DEFAULT_FLOOR_BITS.get(role, 4))
        for other in tied_roles:
            floor = max(floor, self.floor_overrides.get(other, DEFAULT_FLOOR_BITS.get(other, 4)))
        return floor

    def is_structural(self, role: ModuleRole, tied_roles: tuple[ModuleRole, ...] = ()) -> bool:
        return role in self.structural_roles or any(r in self.structural_roles for r in tied_roles)

    def widths_at_or_above(self, bits: int) -> tuple[int, ...]:
        return tuple(b for b in sorted(self.bit_options) if b >= bits and b <= self.max_bits)

    def widths_below(self, bits: int) -> tuple[int, ...]:
        return tuple(b for b in sorted(self.bit_options, reverse=True) if b < bits)
