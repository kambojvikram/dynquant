"""Backend selection: which implementation actually runs a quantized op.

Three backends, in preference order:

``cuda``
    The compiled ``dynquant-kernels`` extension. Real n-bit kernels; weights stay
    packed in VRAM.
``triton``
    Portability fallback for ROCm and for NVIDIA architectures newer than the
    wheel. Same semantics, no prebuilt binary needed.
``torch``
    Pure PyTorch reference. Dequantizes to fp16 to compute, so it saves no memory
    and is not fast -- but it is the oracle every other backend is tested against,
    it runs anywhere, and it is what makes the *quantization* half of DynQuant
    usable on a laptop.

The rules that matter
--------------------
1. **Selection is explicit and reported.** ``DYNQUANT_BACKEND`` overrides
   detection, and an override that cannot be honoured raises rather than silently
   falling back. A benchmark that quietly ran on the torch backend is worse than
   one that failed: it produces a number that looks like a result.
2. **Detection never raises.** :func:`available_backends` is safe to call on a
   broken install; that is precisely when ``dynquant doctor`` needs it.
3. **The reason is always retained.** Every unavailable backend carries a sentence
   saying why, so no user has to guess whether they are missing a wheel, a driver,
   or a GPU.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from dynquant._version import KERNEL_ABI_VERSION, MIN_KERNEL_ABI_VERSION
from dynquant.constants import ENV_BACKEND, KERNELS_DISTRIBUTION, KERNELS_IMPORT_NAME
from dynquant.errors import BackendUnavailableError, KernelAbiMismatchError

__all__ = [
    "Backend",
    "BackendStatus",
    "available_backends",
    "backend_report",
    "resolve_backend",
]


class Backend(str, Enum):
    """A compute backend. ``str`` so it round-trips through JSON and env vars."""

    CUDA = "cuda"
    TRITON = "triton"
    TORCH = "torch"


@dataclass(frozen=True, slots=True)
class BackendStatus:
    """Whether one backend can be used, and why not if it cannot."""

    backend: Backend
    available: bool
    reason: str = ""
    remedy: str = ""
    detail: dict[str, object] | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend.value,
            "available": self.available,
            "reason": self.reason,
            "remedy": self.remedy,
            "detail": self.detail or {},
        }


# --------------------------------------------------------------------------
# Individual probes
# --------------------------------------------------------------------------


def _prebuilt_wheel_exists() -> bool:
    """Whether a ``dynquant-kernels`` wheel is published for this platform.

    Mirrors ``[tool.dynquant] prebuilt-wheel-marker`` in the ``dynquant``
    meta-package. It mirrored an environment marker on that package's kernels
    *dependency* until that dependency was removed -- it was pinning a torch minor,
    and pip satisfies such a pin by downgrading the user's torch. The platform
    question outlived the requirement that used to answer it.

    Kept in sync by tests/test_packaging.py: if the two ever disagree, the doctor
    starts handing out advice that cannot work.
    """
    return platform.system() == "Linux" and platform.machine() == "x86_64"


def _probe_cuda() -> BackendStatus:
    if importlib.util.find_spec(KERNELS_IMPORT_NAME) is None:
        # Rule 3 in the module docstring applies hardest here. "Install the
        # kernels" is only actionable where a wheel exists; on Windows, macOS or
        # ARM it sends the user into a source build of CUDA code that the
        # meta-package's marker deliberately declines to attempt on their behalf.
        # Saying so is the difference between a diagnosis and a dead end.
        if _prebuilt_wheel_exists():
            remedy = (
                "pip install 'dynquant[kernels]'   (a prebuilt wheel is published for "
                f"your platform), or pip install {KERNELS_DISTRIBUTION} directly. "
                "Check what pip proposes before accepting it: the kernels are a "
                "compiled extension pinned to one torch minor, so if your torch is "
                "not that minor, pip will offer to change your torch rather than "
                "decline the wheel. That is why this is an extra and not something "
                "`pip install dynquant` did to you."
            )
        else:
            remedy = (
                f"No prebuilt {KERNELS_DISTRIBUTION} wheel exists for "
                f"{platform.system()}/{platform.machine()} -- wheels are published for "
                "Linux x86_64 only, so the torch backend is the expected path here and "
                "the quantization pipeline is fully usable on it. To build from source "
                "anyway (needs nvcc, cmake >= 3.26 and a C++ toolchain): "
                "pip install 'dynquant[kernels]'."
            )
        return BackendStatus(
            Backend.CUDA,
            False,
            reason=f"{KERNELS_DISTRIBUTION} is not installed",
            remedy=remedy,
        )

    try:
        kernels = importlib.import_module(KERNELS_IMPORT_NAME)
    except Exception as exc:  # noqa: BLE001 - detection must never raise
        return BackendStatus(
            Backend.CUDA,
            False,
            reason=f"{KERNELS_IMPORT_NAME} could not be imported: {exc}",
            remedy=f"pip install --force-reinstall {KERNELS_DISTRIBUTION}",
        )

    # The kernels package reports its own ABI without loading the binary, so an
    # ABI mismatch is caught before paying for the shared object -- and, more
    # importantly, before any op is called.
    found = int(getattr(kernels, "ABI_VERSION", -1))
    if not MIN_KERNEL_ABI_VERSION <= found <= KERNEL_ABI_VERSION:
        error = KernelAbiMismatchError(
            found=found,
            expected=KERNEL_ABI_VERSION,
            minimum=MIN_KERNEL_ABI_VERSION,
            kernels_version=str(getattr(kernels, "__version__", "unknown")),
        )
        return BackendStatus(
            Backend.CUDA,
            False,
            reason=str(error).splitlines()[0],
            remedy="pip install --upgrade dynquant",
            detail={"abi_found": found, "abi_expected": KERNEL_ABI_VERSION},
        )

    if not kernels.is_available():
        return BackendStatus(
            Backend.CUDA,
            False,
            reason=kernels.load_result().error,
            remedy=kernels.load_result().remedy,
            detail=kernels.report(),
        )
    return BackendStatus(Backend.CUDA, True, detail=kernels.report())


def _probe_triton() -> BackendStatus:
    if importlib.util.find_spec("triton") is None:
        return BackendStatus(
            Backend.TRITON,
            False,
            reason="triton is not installed",
            remedy="pip install 'dynquant[triton]'",
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a hard dependency
        return BackendStatus(Backend.TRITON, False, reason=f"torch not importable: {exc}")
    if not torch.cuda.is_available():
        return BackendStatus(
            Backend.TRITON,
            False,
            reason="triton is installed but no GPU is visible to torch",
            remedy="Check nvidia-smi / rocm-smi and CUDA_VISIBLE_DEVICES.",
        )
    return BackendStatus(Backend.TRITON, True)


def _probe_torch() -> BackendStatus:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        return BackendStatus(Backend.TORCH, False, reason=f"torch not importable: {exc}")
    return BackendStatus(Backend.TORCH, True, detail={"torch_version": torch.__version__})


_PROBES = {
    Backend.CUDA: _probe_cuda,
    Backend.TRITON: _probe_triton,
    Backend.TORCH: _probe_torch,
}

#: Preference order. torch is last and always available, so resolution terminates.
PREFERENCE: tuple[Backend, ...] = (Backend.CUDA, Backend.TRITON, Backend.TORCH)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _statuses() -> tuple[BackendStatus, ...]:
    return tuple(_PROBES[backend]() for backend in PREFERENCE)


def available_backends(*, refresh: bool = False) -> tuple[BackendStatus, ...]:
    """Status of every backend, in preference order. Never raises.

    Cached, because probing imports a shared object and queries the driver. Pass
    ``refresh=True`` after installing a wheel in the same process -- which really
    only happens in tests and in notebooks.
    """
    if refresh:
        _statuses.cache_clear()
    return _statuses()


def resolve_backend(requested: str | Backend | None = None) -> Backend:
    """Pick the backend to use.

    Args:
        requested: Explicit choice. Falls back to ``$DYNQUANT_BACKEND``, then to
            automatic selection.

    Raises:
        BackendUnavailableError: If a backend was explicitly requested -- by
            argument or environment variable -- and is not usable. Honouring the
            request or failing is the only safe pair of outcomes: silently
            substituting the torch backend for a requested ``cuda`` turns a
            benchmark into a number that means nothing and a memory-savings claim
            into a false one.
    """
    raw = requested if requested is not None else os.environ.get(ENV_BACKEND)
    statuses = {status.backend: status for status in available_backends()}

    if raw is not None and str(raw) != "":
        try:
            backend = Backend(str(raw).strip().lower())
        except ValueError:
            choices = ", ".join(b.value for b in Backend)
            raise BackendUnavailableError(
                str(raw), f"unknown backend name; choose one of {choices}"
            ) from None
        status = statuses[backend]
        if not status.available:
            source = "argument" if requested is not None else f"${ENV_BACKEND}"
            raise BackendUnavailableError(
                backend.value,
                f"{status.reason} (requested explicitly via {source})",
                remedy=status.remedy or None,
            )
        return backend

    for backend in PREFERENCE:
        if statuses[backend].available:
            return backend
    # Unreachable: the torch probe only fails if torch is missing, and torch is a
    # hard dependency, so the package could not have imported this far.
    raise BackendUnavailableError("torch", "no backend is usable, including the pure-torch one")


def backend_report() -> dict[str, object]:
    """Structured backend diagnosis for ``dynquant doctor``."""
    statuses = available_backends()
    override = os.environ.get(ENV_BACKEND) or ""
    selected: str | None
    try:
        selected = resolve_backend().value
    except BackendUnavailableError as exc:
        selected = None
        override_error: str | None = str(exc)
    else:
        override_error = None
    return {
        "selected": selected,
        "override": override,
        "override_error": override_error,
        "backends": [status.as_dict() for status in statuses],
    }
