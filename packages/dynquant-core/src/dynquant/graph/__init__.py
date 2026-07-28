"""Model-graph analysis: what each module *is*, structurally.

Role classification is the difference between DynQuant working on a new
architecture and destroying it. The research supplement matched substrings
against module names, which meant a Mixtral router (``mlp.gate``) fell through
to an ``"mlp"`` catch-all and got 3 bits -- enough to scramble expert routing
for the whole model.

:mod:`dynquant.graph.roles` defines the vocabulary; :mod:`dynquant.graph.classify`
does the structural inference; :mod:`dynquant.graph.registry` holds the
per-architecture plugins for the conventions structure cannot reveal.

``roles`` and ``naming`` are pure Python and imported eagerly. ``classify`` needs
torch, so it is imported on first use -- reading a stats file should not pay for a
torch import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .naming import canonical_name, is_adapter_name
from .roles import DEFAULT_FLOOR_BITS, ModuleRole, RowPartition, role_of_name

if TYPE_CHECKING:
    from .classify import ModelGraph, ModuleInfo, classify_model
    from .registry import ModuleContext, register_arch

__all__ = [
    "DEFAULT_FLOOR_BITS",
    "ModelGraph",
    "ModuleContext",
    "ModuleInfo",
    "ModuleRole",
    "RowPartition",
    "canonical_name",
    "classify_model",
    "is_adapter_name",
    "register_arch",
    "role_of_name",
]

_LAZY = {
    "ModelGraph": "classify",
    "ModuleInfo": "classify",
    "classify_model": "classify",
    "ModuleContext": "registry",
    "register_arch": "registry",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value  # cache, so the indirection costs one lookup
    return value
