"""Turning importance scores into a per-module bit width under a size budget.

Three pieces: :mod:`~dynquant.allocate.budget` resolves what "3 bits" or "6.5GiB"
means in stored bits, :mod:`~dynquant.allocate.policy` says which floors are
constraints and which are preferences, and :mod:`~dynquant.allocate.knapsack` does
the greedy assignment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .budget import Budget, module_stored_bits, parse_size
    from .knapsack import BitMap, FloorViolation, InfeasibleTargetError, allocate_bits
    from .policy import STRUCTURAL_ROLES, AllocationPolicy

__all__ = [
    "STRUCTURAL_ROLES",
    "AllocationPolicy",
    "BitMap",
    "Budget",
    "FloorViolation",
    "InfeasibleTargetError",
    "allocate_bits",
    "module_stored_bits",
    "parse_size",
]

_LAZY = {
    "Budget": "budget",
    "module_stored_bits": "budget",
    "parse_size": "budget",
    "AllocationPolicy": "policy",
    "STRUCTURAL_ROLES": "policy",
    "BitMap": "knapsack",
    "FloorViolation": "knapsack",
    "InfeasibleTargetError": "knapsack",
    "allocate_bits": "knapsack",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value
