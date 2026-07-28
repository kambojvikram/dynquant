"""Version of the compiled kernels wheel.

Read at build time by scikit-build-core's regex provider (see pyproject.toml), so
the string below is the single source of truth for the wheel version.

CI appends a PEP 440 local version identifying the binary variant before
building each matrix cell, e.g. ``0.1.0+cu124torch27``. That is the same scheme
torch uses, and it is what lets one index hold a wheel per (CUDA, torch) pair
while ``pip install dynquant-kernels`` still resolves the default build.

Not required to match ``dynquant-core``'s version -- see the pin comment in
``packages/dynquant/pyproject.toml``. Compatibility is the ABI number's job.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0.dev0"
