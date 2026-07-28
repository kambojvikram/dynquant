"""Adapter-aware naming: making a LoRA run's stats line up with the merged checkpoint.

DynQuant collects signals from a model that looks like this::

    base_model.model.model.layers.0.self_attn.q_proj.base_layer
    base_model.model.model.layers.0.self_attn.q_proj.lora_A.default
    base_model.model.model.layers.0.self_attn.q_proj.lora_B.default

and quantizes a model that looks like this::

    model.layers.0.self_attn.q_proj

Every name must survive that transition, or coverage silently collapses and the
allocator works from whatever fraction happened to match. The supplement handled it
at *read* time, in ``_normalize_stats_layer_name``, by guessing which prefixes to
strip from keys it found in a JSON file. Guessing later is strictly worse than
recording correctly earlier: by then the model is gone and there is nothing left to
check the guess against. Here the canonical name is what gets written, and this
module supplies the two things that decision needs -- a way to see the adapter
topology while the model is still in memory, and a way to verify the result.

The shipped stats files carry ``"collapsed_lora_into_base": true``, produced by a
script that is not in the supplement. :func:`collapse_adapter_stats` is that missing
step, made explicit and exact: adapter observations attribute to the base weight
they will be merged into, and colliding keys merge through Chan's formula rather
than last-write-wins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dynquant._logging import get_logger
from dynquant.errors import MissingDependencyError
from dynquant.graph.naming import canonical_name, is_adapter_name
from dynquant.signals.schema import CoverageReport, StatsFile

if TYPE_CHECKING:
    from torch import nn

__all__ = [
    "adapter_target_names",
    "check_stats_cover_model",
    "collapse_adapter_stats",
    "describe_adapters",
    "is_peft_model",
    "merge_adapters",
    "quantizable_module_names",
    "unwrap_model",
]

_log = get_logger(__name__)

_WRAPPER_ATTRIBUTES = ("_orig_mod", "module", "base_model", "model")
"""Attributes that mean "the real model is one level down".

Order matters: ``torch.compile`` wraps outermost, then DDP, then PEFT. ``model`` is
last because it is also a legitimate submodule name on ``PreTrainedModel`` itself,
so it is only followed when the current object is not already the thing we want.
"""


def is_peft_model(model: nn.Module) -> bool:
    """True if a PEFT adapter is attached, without importing ``peft``.

    Duck-typed on purpose. Importing ``peft`` to answer a question about an object
    that may not involve ``peft`` turns an optional dependency into a required one,
    and the attributes tested here are PEFT's public surface.
    """
    return hasattr(model, "peft_config") or hasattr(model, "base_model")


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip ``torch.compile`` / DDP / FSDP / PEFT wrappers to the inner model.

    Stops as soon as it finds something with a ``config``, which is the signature
    of a ``PreTrainedModel`` and therefore of the level whose module names match
    the checkpoint on disk.
    """
    current = model
    for _ in range(8):  # bounded: real wrapper stacks are two or three deep
        if hasattr(current, "config") and not is_peft_model(current):
            return current
        for attribute in _WRAPPER_ATTRIBUTES:
            inner = getattr(current, attribute, None)
            if inner is not None and inner is not current and hasattr(inner, "forward"):
                current = inner
                break
        else:
            return current
    return current


def describe_adapters(model: nn.Module) -> dict[str, Any]:
    """Summarise the adapter configuration, for the stats file's provenance.

    Recorded because the plasticity signal is not comparable across ranks, targets
    or estimators, and "which modules even had an adapter" is the first thing worth
    knowing when two runs disagree.
    """
    configs = getattr(model, "peft_config", None)
    if not isinstance(configs, dict):
        return {}
    out: dict[str, Any] = {}
    for name, config in configs.items():
        target = getattr(config, "target_modules", None)
        out[name] = {
            "peft_type": str(getattr(config, "peft_type", "unknown")),
            "r": getattr(config, "r", None),
            "lora_alpha": getattr(config, "lora_alpha", None),
            "target_modules": sorted(target)
            if isinstance(target, (set, frozenset, list))
            else target,
            "modules_to_save": sorted(getattr(config, "modules_to_save", None) or []),
        }
    return out


def adapter_target_names(model: nn.Module) -> tuple[str, ...]:
    """Canonical names of the modules that actually carry an adapter.

    Read off the module tree rather than off ``target_modules``, which holds
    suffixes and regexes that were resolved against the tree at attach time. The
    tree is the ground truth about what happened.
    """
    found: set[str] = set()
    for name, module in model.named_modules():
        if not name or not _has_adapter(module):
            continue
        found.add(canonical_name(name))
    return tuple(sorted(found))


def quantizable_module_names(model: nn.Module) -> tuple[str, ...]:
    """Canonical names of every module the quantizer will want stats for.

    The same eligibility test the tracker uses -- a weight with two or more
    dimensions -- so a coverage report built from this compares like with like
    instead of measuring the difference between two definitions of "quantizable".
    """
    from dynquant.signals.tracker import _quantizable_weight  # local import: avoids a cycle

    names: set[str] = set()
    for name, module in model.named_modules():
        if not name or is_adapter_name(name):
            continue
        if _quantizable_weight(module) is not None:
            names.add(canonical_name(name))
    return tuple(sorted(names))


def collapse_adapter_stats(stats: StatsFile) -> StatsFile:
    """Attribute adapter observations to the base weights they will merge into.

    The missing "collapse" step. ``...q_proj.lora_A.default`` and
    ``...q_proj.lora_B.default`` both canonicalise to ``...q_proj``, and their
    Welford states combine exactly through Chan's parallel formula -- so two
    partial views of one weight's training dynamics become one, rather than one
    overwriting the other.

    Idempotent, and safe on a file that was already written with canonical keys.
    """
    before = len(stats)
    collapsed = stats.canonicalized()
    if len(collapsed) != before:
        _log.info(
            "collapsed %d adapter-scoped stats keys into %d base weights", before, len(collapsed)
        )
    return collapsed


def check_stats_cover_model(
    stats: StatsFile,
    model: nn.Module,
    *,
    threshold: float = 0.9,
    raise_on_shortfall: bool = True,
) -> CoverageReport:
    """Compare a stats file against the model it is about to quantize.

    This is the check that turns "the wrong stats file" from a silent 40%-coverage
    allocation into an error with the missing names in it. Also surfaces modules the
    training data never exercised, which are covered but carry no evidence.
    """
    report = stats.coverage(quantizable_module_names(model))
    _log.info("signal coverage: %s", report.summary())
    if report.unexercised:
        _log.warning(
            "%d modules were never exercised by the training data (e.g. %s). Their "
            "zero saliency is absence of evidence, not evidence of unimportance.",
            len(report.unexercised),
            ", ".join(report.unexercised[:3]),
        )
    if raise_on_shortfall:
        report.raise_if_insufficient(threshold)
    return report


def merge_adapters(model: nn.Module, *, safe_merge: bool = True) -> nn.Module:
    """Fold LoRA weights into the base model and return the bare model.

    Quantization happens on merged weights: the whole method measures the base
    weight's dynamics in order to decide the base weight's precision, and a
    checkpoint that still needed an adapter at inference time would defeat the
    point. ``safe_merge`` makes PEFT verify the merged result contains no NaNs
    before committing, which costs one extra copy per module and is worth it -- an
    adapter merged into garbage quantizes into garbage without complaint.
    """
    if not is_peft_model(model):
        return model
    merge = getattr(model, "merge_and_unload", None)
    if merge is None:
        raise MissingDependencyError(
            "peft",
            feature="merging LoRA adapters before quantization",
            extra="train",
        )
    _log.info("merging adapters into base weights (safe_merge=%s)", safe_merge)
    merged: nn.Module = merge(safe_merge=safe_merge)
    return merged


def _has_adapter(module: nn.Module) -> bool:
    from torch import nn as torch_nn

    for attribute in ("lora_A", "lora_embedding_A", "vera_A", "ia3_l"):
        container = getattr(module, attribute, None)
        if isinstance(container, torch_nn.ModuleDict | torch_nn.ParameterDict) and len(container):
            return True
    return False
