"""DynQuant compiled kernels.

This package is a thin, dependency-light shell around one shared object. It does
not import :mod:`dynquant`, and it must stay that way: the kernels wheel has to be
installable and diagnosable on its own, so that ``pip install dynquant-kernels``
followed by ``python -c "import dynquant_kernels; print(dynquant_kernels.report())"``
is a complete diagnostic even when the rest of the stack is broken.

Importing this module never raises. Use :func:`is_available` to decide whether the
compiled path can be used, and :func:`report` to find out why not.
"""

from __future__ import annotations

from typing import Any

from ._loader import KernelLoadResult, load
from ._version import __version__

__all__ = [
    "ABI_VERSION",
    "KernelLoadResult",
    "__version__",
    "build_info",
    "compiled_architectures",
    "is_available",
    "load_result",
    "report",
    "unavailable_reason",
]

ABI_VERSION: int = 2
"""ABI this shell expects, mirroring ``dynquant._version.KERNEL_ABI_VERSION``.

Repeated here rather than imported so this package stays independent of
``dynquant``. ``tests/test_abi.py`` asserts that this constant, the Python
constant in core, and the ``#define`` in ``csrc/include/dynquant/abi.h`` are all
the same number -- three declarations, one lint, no runtime coupling.
"""

_result: KernelLoadResult | None = None


def load_result() -> KernelLoadResult:
    """Import the extension once and cache the outcome.

    Lazy rather than at module import so that merely importing ``dynquant`` does
    not pay for loading a large shared object, and so that a machine with a broken
    driver can still run the CPU-side quantizer without touching CUDA.
    """
    global _result
    if _result is None:
        _result = load(ABI_VERSION)
        _result.raise_if_strict()
    return _result


def is_available() -> bool:
    """True when the CUDA kernels are loaded and safe to call."""
    return load_result().available


def unavailable_reason() -> str:
    """Empty when available, else ``"<reason>"`` with the remedy appended."""
    result = load_result()
    if result.available:
        return ""
    return f"{result.error}\n{result.remedy}".rstrip()


def build_info() -> dict[str, Any]:
    """What the binary was compiled against. Empty dict if there is no stamp."""
    return dict(load_result().build_info)


def compiled_architectures() -> tuple[int, ...]:
    """SM versions with a cubin in this binary, e.g. ``(75, 80, 86, 89, 90)``.

    Empty when the extension is unavailable or was built by a CUDA older than 11.5
    (which does not expose ``__CUDA_ARCH_LIST__``). An empty tuple therefore means
    "cannot tell", not "no architectures" -- callers must not use it to conclude
    that a device is unsupported.
    """
    result = load_result()
    if result.module is None:
        return ()
    import torch

    try:
        raw = str(torch.ops.dynquant.compiled_architectures())
    except Exception:  # noqa: BLE001 - diagnostics must not raise
        return ()
    if not raw or raw == "unknown":
        return ()
    out: list[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if piece.isdigit():
            # nvcc reports 750 for sm_75; report the familiar two-digit form.
            out.append(int(piece) // 10)
    return tuple(sorted(set(out)))


def report() -> dict[str, Any]:
    """Everything `dynquant doctor` needs about the compiled half, as plain data."""
    result = load_result()
    return {
        "wheel_version": __version__,
        "available": result.available,
        "abi_expected": ABI_VERSION,
        "error": result.error,
        "remedy": result.remedy,
        "build": dict(result.build_info),
        "compiled_architectures": list(compiled_architectures()),
    }
