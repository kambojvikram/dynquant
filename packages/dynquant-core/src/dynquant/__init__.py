"""DynQuant -- training-dynamics-driven mixed-precision LLM quantization.

DynQuant decides *per weight matrix* how many bits it deserves, using two signals
harvested from the fine-tune you were going to run anyway:

* **Activation saliency** -- an EMA of activation RMS. How much signal flows through.
* **Gradient plasticity** -- the online variance of gradient norms. How much the
  weight was still moving when training stopped.

High on both means the weight is both load-bearing and unsettled: give it bits.
Low on either means it can be crushed. Combined multiplicatively as a soft AND,
percentile-ranked so the two incommensurable scales can be multiplied at all.

Typical use -- collect during training, allocate and pack afterwards::

    from transformers import Trainer
    from dynquant import DynQuantCallback

    trainer = Trainer(model=model, ..., callbacks=[DynQuantCallback("stats/")])
    trainer.train()          # writes stats/dynquant_stats.json

then::

    dynquant export ./merged --stats stats/dynquant_stats.json --target 3.0 -o ./q3

and load the result through plain transformers -- with the quantizer registered::

    import dynquant; dynquant.register_hf_quantizer()
    model = AutoModelForCausalLM.from_pretrained("./q3")   # stays packed in VRAM

That first line is not boilerplate. transformers answers a ``quant_method`` it
cannot resolve with a ``logger.warning`` and ``pre_quantized = False``, so the
packed tensors match no parameter the model has and ``from_pretrained`` returns a
**randomly initialised model without raising**.

``dynquant quantize`` is the other writer and a different artifact: it holds the
quantized *values* in the compute dtype, so it loads with no DynQuant installed
and measures the quantized model's accuracy at fp16 footprint. ``export`` is the
one whose VRAM is the packed size.

Attribute access is lazy: ``import dynquant`` does not import torch-heavy or
transformers-dependent submodules until you actually touch them, which is why
registration is a call rather than an import side effect.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._version import (
    CHECKPOINT_FORMAT_VERSION,
    KERNEL_ABI_VERSION,
    STATS_SCHEMA_VERSION,
    __version__,
)
from .errors import (
    AllocationError,
    BackendUnavailableError,
    BudgetInfeasibleError,
    ClassificationError,
    DynQuantError,
    FormatVersionError,
    KernelAbiMismatchError,
    MissingDependencyError,
    PackingError,
    SignalCollectionError,
    StatsCoverageError,
)

# Lazily-resolved public names -> the submodule that defines them.
# Keeping this table explicit (rather than star-importing) is what makes
# `import dynquant` cheap and keeps transformers optional.
_LAZY: dict[str, str] = {
    # graph
    "ModuleRole": "dynquant.graph.roles",
    "RowPartition": "dynquant.graph.roles",
    "role_of_name": "dynquant.graph.roles",
    # quant
    "QuantLayout": "dynquant.quant.tensor",
    "QuantTensor": "dynquant.quant.tensor",
    "pack_nbit": "dynquant.quant.pack",
    "unpack_nbit": "dynquant.quant.pack",
    # signals
    "CoverageReport": "dynquant.signals.schema",
    "LayerStats": "dynquant.signals.schema",
    "StatsFile": "dynquant.signals.schema",
    "load_stats": "dynquant.signals.schema",
    "save_stats": "dynquant.signals.schema",
    # signal collection. DynQuantCallback needs transformers, so resolving it
    # lazily is what keeps `import dynquant` working without it.
    "DynQuantCallback": "dynquant.signals.callback",
    "SignalTracker": "dynquant.signals.tracker",
    "TrackerConfig": "dynquant.signals.tracker",
    "track_signals": "dynquant.signals.context",
    # serving. Resolving this lazily is the reason `register_hf_quantizer` is a
    # call and not something `import dynquant` does for you: reaching it imports
    # transformers, which is the cost the whole table exists to defer.
    "register_hf_quantizer": "dynquant.integration.hf_quantizer",
}

if TYPE_CHECKING:  # pragma: no cover -- import-time cost avoided at runtime
    from .graph.roles import ModuleRole, RowPartition, role_of_name
    from .integration.hf_quantizer import register_hf_quantizer
    from .quant.pack import pack_nbit, unpack_nbit
    from .quant.tensor import QuantLayout, QuantTensor
    from .signals.callback import DynQuantCallback
    from .signals.context import track_signals
    from .signals.schema import CoverageReport, LayerStats, StatsFile, load_stats, save_stats
    from .signals.tracker import SignalTracker, TrackerConfig

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "KERNEL_ABI_VERSION",
    "STATS_SCHEMA_VERSION",
    "AllocationError",
    "BackendUnavailableError",
    "BudgetInfeasibleError",
    "ClassificationError",
    "CoverageReport",
    "DynQuantCallback",
    "DynQuantError",
    "FormatVersionError",
    "KernelAbiMismatchError",
    "LayerStats",
    "MissingDependencyError",
    "ModuleRole",
    "PackingError",
    "QuantLayout",
    "QuantTensor",
    "RowPartition",
    "SignalCollectionError",
    "SignalTracker",
    "StatsCoverageError",
    "StatsFile",
    "TrackerConfig",
    "__version__",
    "load_stats",
    "pack_nbit",
    "register_hf_quantizer",
    "role_of_name",
    "save_stats",
    "track_signals",
    "unpack_nbit",
]


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'dynquant' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache so subsequent access skips this path
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
