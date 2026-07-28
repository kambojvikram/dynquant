"""Architecture plugins: per-family knowledge the generic path cannot infer.

Most modules classify structurally. A few cannot, because the information needed
is a convention rather than a shape -- which half of a fused tensor is the gate,
whether ``in_proj_a`` feeds an exponential, whether a 16-row projection is
important or vestigial. That knowledge is per-architecture, so it lives in small
plugins keyed on ``config.model_type``.

A plugin answers two questions and may decline both:

``role_for(ctx)``
    The role, or ``None`` to defer to structural inference and then name matching.
    Declining is the correct answer for anything the plugin does not recognise --
    it should not re-implement the generic path.

``partitions_for(ctx, role)``
    Row ranges for a fused tensor. Only a plugin can answer this, because row
    order is a convention: whether a fused QKV is ``[q; k; v]`` contiguous or
    interleaved per head cannot be read off the shape. A wrong answer silently
    gives the gate's bits to the wrong rows, which is worse than not partitioning,
    so declining is safe and guessing is not.

Plugins are registered by decorator and looked up by exact ``model_type``, with a
fallback to the ``text_config``'s type so a VLM-derived text tower finds the plugin
registered for its own family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from .roles import ModuleRole, RowPartition

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor, nn

__all__ = [
    "ArchPlugin",
    "ModuleContext",
    "plugin_for",
    "register_arch",
    "registered_model_types",
]


@dataclass(frozen=True, slots=True)
class ModuleContext:
    """Everything a classifier may look at when deciding one module's role."""

    name: str
    """Canonical dotted name."""

    module: nn.Module
    weight: Tensor
    config: Any
    """The config governing *this* module's tower -- ``text_config`` when the model
    has one, so ``head_dim`` and friends are the text tower's, not the outer
    wrapper's ``None``."""

    ancestors: tuple[str, ...]
    """Ancestor class names, outermost first, e.g.
    ``("Qwen3_5ForCausalLM", "Qwen3_5Model", "Qwen3_5DecoderLayer",
    "Qwen3_5GatedDeltaNet")``. This is how a plugin tells a linear-attention
    projection from a self-attention one when the names alone are ambiguous."""

    leaf: str
    """Final dotted component, e.g. ``in_proj_qkv``."""

    parent: nn.Module | None = None
    """The module's parent, for sibling tests. Supplied by the caller because
    ``named_modules`` yields no parent links and re-deriving one per module by
    walking from the root is quadratic."""

    # -- predicates plugins use ------------------------------------------------

    def in_ancestor(self, *fragments: str) -> bool:
        """True if any ancestor's class name contains any fragment (case-insensitive)."""
        lowered = [a.lower() for a in self.ancestors]
        return any(f.lower() in a for f in fragments for a in lowered)

    def module_class_endswith(self, *suffixes: str) -> bool:
        return type(self.module).__name__.endswith(suffixes)

    def has_sibling(self, name: str) -> bool:
        """True if the module's parent also has a child called ``name``.

        The second half of the router test: an output width equal to the expert
        count is suggestive, an ``experts`` sibling makes it certain.
        """
        return self.parent is not None and hasattr(self.parent, name)

    @property
    def out_features(self) -> int:
        return int(self.weight.shape[0])

    @property
    def in_features(self) -> int:
        return int(self.weight.shape[-1])

    def cfg(self, *names: str) -> int | None:
        """First positive integer attribute among ``names``, or ``None``."""
        for name in names:
            value = getattr(self.config, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        return None

    def flag(self, name: str) -> bool:
        return bool(getattr(self.config, name, False))


class ArchPlugin(Protocol):
    """The interface a plugin implements. Both methods may return ``None``."""

    name: str

    def role_for(self, ctx: ModuleContext) -> ModuleRole | None: ...

    def partitions_for(
        self, ctx: ModuleContext, role: ModuleRole
    ) -> tuple[RowPartition, ...] | None: ...


_REGISTRY: dict[str, ArchPlugin] = {}


def register_arch(*model_types: str) -> Callable[[type], type]:
    """Class decorator registering a plugin for one or more ``model_type`` values.

    Registering the same type twice raises rather than overwriting: a silent
    override would make classification depend on import order, which is the kind
    of bug that only shows up as slightly wrong accuracy numbers.
    """

    def decorate(cls: type) -> type:
        instance = cls()
        for model_type in model_types:
            existing = _REGISTRY.get(model_type)
            if existing is not None and type(existing) is not cls:
                raise ValueError(
                    f"model_type {model_type!r} is already handled by "
                    f"{type(existing).__name__}; two plugins claiming one architecture "
                    f"would make classification depend on import order"
                )
            _REGISTRY[model_type] = instance
        return cls

    return decorate


def plugin_for(model_type: str, config: Any = None) -> ArchPlugin | None:
    """Look up a plugin, falling back to the ``text_config``'s model type.

    A VLM's outer ``model_type`` (``qwen3_5``) differs from its text tower's
    (``qwen3_5_text``), and loading through ``AutoModelForCausalLM`` gives a text
    tower with the *outer* type on its config. Trying both means one registration
    covers both entry points.
    """
    _load_builtin_plugins()
    plugin = _REGISTRY.get(model_type)
    if plugin is not None:
        return plugin
    inner = getattr(config, "text_config", None)
    inner_type = getattr(inner, "model_type", None)
    if isinstance(inner_type, str):
        return _REGISTRY.get(inner_type)
    return None


def registered_model_types() -> tuple[str, ...]:
    _load_builtin_plugins()
    return tuple(sorted(_REGISTRY))


_loaded = False


def _load_builtin_plugins() -> None:
    """Import the bundled plugins so their decorators run.

    Deferred rather than imported at module scope: the plugin modules import from
    this one, so doing it eagerly is a circular import.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    from . import arch

    _ = arch
