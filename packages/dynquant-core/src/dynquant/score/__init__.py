"""Turning collected training signals into a per-module importance score.

Two signals come out of a fine-tune (:mod:`dynquant.signals`):

**Saliency** -- the EMA of activation RMS. How much magnitude flows through this
module. A weight multiplying large activations produces large errors when it is
rounded.

**Plasticity** -- the variance of ``‖∇W‖`` across optimizer steps. How much the
training run kept *changing its mind* about this weight. A tensor whose gradient
norm is stable has settled; one whose gradient norm swings is still contested, and
contested weights sit near a decision boundary where a rounding error changes the
answer rather than just the magnitude.

Neither is usable raw. They live on different scales, in different units, and both
are heavy-tailed across a model -- an embedding's activation RMS and a decay
projection's are not comparable numbers. So each is converted to a **percentile
rank** within its comparison group, and the two ranks are multiplied.

The product is a soft AND: a module scores high only if it is *both* carrying
magnitude and still moving. Either alone is cheap to get wrong. Addition would let
one extreme signal carry a module on its own, which is how you end up spending
precision on a large but frozen tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .importance import ScoreConfig, ScoredModule, score_modules
    from .null import NULL_LADDER, NULL_MODES, NullReport, apply_null
    from .ranks import percentile_ranks
    from .sensitivity import (
        SensitivityTable,
        estimate_sensitivity,
        module_weights,
        weight_only_sensitivity,
    )

__all__ = [
    "NULL_LADDER",
    "NULL_MODES",
    "NullReport",
    "ScoreConfig",
    "ScoredModule",
    "SensitivityTable",
    "apply_null",
    "estimate_sensitivity",
    "module_weights",
    "percentile_ranks",
    "score_modules",
    "weight_only_sensitivity",
]

_LAZY = {
    "NULL_LADDER": "null",
    "NULL_MODES": "null",
    "NullReport": "null",
    "apply_null": "null",
    "ScoreConfig": "importance",
    "ScoredModule": "importance",
    "score_modules": "importance",
    "percentile_ranks": "ranks",
    "SensitivityTable": "sensitivity",
    "estimate_sensitivity": "sensitivity",
    "module_weights": "sensitivity",
    "weight_only_sensitivity": "sensitivity",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value
