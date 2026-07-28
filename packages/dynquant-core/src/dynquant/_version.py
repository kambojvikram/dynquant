"""Single source of truth for versions and cross-distribution contracts.

Three separate version numbers, deliberately decoupled:

``__version__``
    The Python-package version of ``dynquant-core``. Ordinary semver.

``KERNEL_ABI_VERSION``
    The contract between ``dynquant-core`` and the compiled
    ``dynquant-kernels`` wheel: tensor layouts, argument order, op names.
    A compiled wheel built against a different value must refuse to load
    rather than silently return garbage -- see
    :mod:`dynquant.runtime.dispatch`. Bump on ANY change to a kernel
    signature or to the packed memory layout.

``CHECKPOINT_FORMAT_VERSION``
    The on-disk format of a quantized checkpoint (manifest + safetensors
    layout). Readers refuse a newer major and migrate an older one.

``STATS_SCHEMA_VERSION``
    The on-disk format of a collected-signals file.
"""

from __future__ import annotations

__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "KERNEL_ABI_VERSION",
    "MIN_KERNEL_ABI_VERSION",
    "STATS_SCHEMA_VERSION",
    "__version__",
]

__version__ = "0.1.0"

KERNEL_ABI_VERSION = 2
"""Current kernel ABI the Python side speaks.

2 added ``dequant`` and ``gemv`` -- the ops the packed runtime calls on every
forward. 1 shipped only the build-pipeline probes.
"""

MIN_KERNEL_ABI_VERSION = 2
"""Oldest kernel ABI this core still accepts. Raise to drop old wheels.

Raised to 2 with the ops above: an ABI-1 wheel loads fine and then fails on the
first quantized Linear, so refusing it at import is strictly better.
"""

CHECKPOINT_FORMAT_VERSION = 2
"""v1 == the research supplement's ``dynquant_packed_checkpoint_v1``."""

STATS_SCHEMA_VERSION = 2
"""v1 == the research supplement's ``unified_dynquant_stats_v1``."""
