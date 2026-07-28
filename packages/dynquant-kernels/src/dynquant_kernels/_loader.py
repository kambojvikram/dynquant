"""Load the compiled extension, or explain precisely why it could not be loaded.

The whole value of this module is in the failure paths. A user whose extension
does not match their torch currently gets::

    ImportError: /.../_C.cpython-311-x86_64-linux-gnu.so: undefined symbol:
    _ZN3c104impl23ExcludeDispatchKeyGuardC1ENS_11DispatchKeyE

which says nothing about what to install. Every branch below turns a load failure
into a sentence naming the mismatch and the command that fixes it.

Importing this module never raises. A broken kernels install must degrade to the
pure-torch backend with a loud diagnostic, not take down ``import dynquant`` --
otherwise a GPU-node misconfiguration also breaks quantizing on a CPU box.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Any

__all__ = ["KernelLoadResult", "load"]

_ENV_FORCE_ERROR = "DYNQUANT_KERNELS_STRICT"
"""When set, a load failure raises instead of being reported. For CI, where a
silent fallback to the torch backend would let a broken wheel pass the test
suite by simply not being used.

It does *not* fire for outcomes flagged :attr:`KernelLoadResult.expected` -- a
CPU-only build on a machine with no GPU is the designed configuration of the
CPU-only CI job, and a flag that failed there would have to be left unset in the
one place it was written for."""


@dataclass(slots=True)
class KernelLoadResult:
    """Outcome of trying to import the extension.

    Attributes:
        available: True only if ``_C`` imported *and* the ABI handshake passed.
            Callers must not treat "imported" as "usable".
        module: The ``_C`` module, or None.
        build_info: Contents of the CMake-generated ``_build_info``, as a plain
            dict so it can be printed and serialised. Empty if unavailable.
        error: One-line reason, suitable for a log line.
        remedy: The command or action that would fix it. Empty when there is
            nothing actionable (e.g. a CPU-only build on a CPU-only machine,
            which is working as intended).
        expected: True when unavailability is the correct outcome on this machine
            rather than a fault. Only ``$DYNQUANT_KERNELS_STRICT`` reads it, and
            only to decide whether to raise. Deliberately a separate field rather
            than inferred from an empty ``remedy``: a new failure branch that
            forgot its remedy would otherwise exempt itself from the strict check
            that exists to catch exactly that class of mistake.
    """

    available: bool
    module: Any = None
    build_info: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    remedy: str = ""
    expected: bool = False

    def raise_if_strict(self) -> None:
        if self.available or self.expected or not os.environ.get(_ENV_FORCE_ERROR):
            return
        message = self.error
        if self.remedy:
            message = f"{message}\n{self.remedy}"
        raise ImportError(f"{_ENV_FORCE_ERROR} is set and the kernels are unusable: {message}")


def _read_build_info() -> dict[str, Any]:
    """Load the CMake-generated stamp, tolerating its absence.

    Absent means the wheel was assembled by something other than this CMake
    project -- a hand-built extension, or an editable install pointing at a source
    tree that was never configured. That is a legitimate developer situation, so
    it is reported rather than treated as corruption.
    """
    try:
        info = importlib.import_module("dynquant_kernels._build_info")
    except ImportError:
        return {}
    return {
        key: getattr(info, key)
        for key in (
            "ABI_VERSION",
            "TORCH_VERSION",
            "TORCH_CUDA_VERSION",
            "CUDA_VERSION",
            "WITH_CUDA",
            "CUDA_ARCHITECTURES",
            "CXX_COMPILER",
            "BUILD_TYPE",
        )
        if hasattr(info, key)
    }


def _torch_minor(version: str) -> tuple[int, int] | None:
    """``"2.7.1+cu126"`` -> ``(2, 7)``. None if unparseable."""
    head = version.split("+", 1)[0]
    parts = head.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None


def _diagnose_import_error(exc: ImportError, build_info: dict[str, Any]) -> tuple[str, str]:
    """Turn a linker-level ImportError into something a user can act on."""
    import torch

    text = str(exc)
    built_torch = str(build_info.get("TORCH_VERSION", ""))
    built_cuda = str(build_info.get("TORCH_CUDA_VERSION", ""))
    running_torch = torch.__version__
    running_cuda = torch.version.cuda or ""

    # The overwhelmingly common cause. Symbol names in libtorch are C++-mangled and
    # change across minor releases, so a mismatch shows up as a missing symbol
    # rather than as a version check anywhere.
    if "undefined symbol" in text or "cannot open shared object" in text or "DLL load" in text:
        if built_torch and _torch_minor(built_torch) != _torch_minor(running_torch):
            remedy = (
                f"Install the kernels wheel built for torch {running_torch}:\n"
                f"    pip install --force-reinstall dynquant-kernels"
                f"=={_variant_hint(running_torch, running_cuda)}\n"
                f"or build from source against your torch:\n"
                f"    pip install --no-build-isolation dynquant-kernels"
            )
            return (
                f"the extension was compiled against torch {built_torch} but torch "
                f"{running_torch} is installed",
                remedy,
            )
        if built_cuda and running_cuda and built_cuda.split(".")[0] != running_cuda.split(".")[0]:
            return (
                f"the extension was compiled for CUDA {built_cuda} but torch reports "
                f"CUDA {running_cuda}",
                "Install the kernels wheel matching your CUDA major version.",
            )
        if platform.system() == "Windows":
            return (
                f"the extension could not be loaded: {text}",
                "On Windows the CUDA runtime DLLs must be locatable. Importing torch "
                "before dynquant normally arranges that; if this persists, check that "
                "the CUDA Toolkit bin directory is on PATH.",
            )
    return (
        f"the extension failed to import: {text}",
        "Please report this with the output of `dynquant doctor`.",
    )


def _variant_hint(torch_version: str, cuda_version: str) -> str:
    """Best-guess local version for the wheel this environment wants."""
    minor = _torch_minor(torch_version)
    torch_tag = f"torch{minor[0]}{minor[1]}" if minor else "torch"
    cuda_tag = f"cu{cuda_version.replace('.', '')}" if cuda_version else "cpu"
    return f"{cuda_tag}{torch_tag}"


def load(expected_abi: int) -> KernelLoadResult:
    """Import ``_C`` and verify it is safe to use.

    Args:
        expected_abi: ``dynquant._version.KERNEL_ABI_VERSION``. Passed in rather
            than imported so this package keeps no dependency on ``dynquant`` --
            the kernels wheel must be installable and diagnosable on its own.
    """
    build_info = _read_build_info()

    if "torch" not in sys.modules:
        # Importing torch first is not optional: the extension links libtorch, and
        # on Windows torch's __init__ is what puts the CUDA DLL directory on the
        # search path. Doing it here rather than relying on the caller means the
        # order cannot be got wrong by an import that looks innocuous.
        try:
            importlib.import_module("torch")
        except ImportError as exc:  # pragma: no cover - torch is a hard dependency
            return KernelLoadResult(
                available=False,
                build_info=build_info,
                error=f"torch is not importable: {exc}",
                remedy="Install torch >= 2.4.",
            )

    try:
        module = importlib.import_module("dynquant_kernels._C")
    except ImportError as exc:
        error, remedy = _diagnose_import_error(exc, build_info)
        return KernelLoadResult(available=False, build_info=build_info, error=error, remedy=remedy)

    import torch

    # The handshake. Every check below guards against a *silent* wrong answer, so
    # each one fails closed: an unusable extension reports unavailable and the
    # torch backend takes over.
    try:
        actual_abi = int(torch.ops.dynquant.abi_version())
    except Exception as exc:  # noqa: BLE001 - any failure here means unusable
        return KernelLoadResult(
            available=False,
            module=module,
            build_info=build_info,
            error=f"the extension imported but its operators are not registered: {exc}",
            remedy="This indicates a corrupt build. Reinstall dynquant-kernels.",
        )

    if actual_abi != expected_abi:
        newer = "newer" if actual_abi > expected_abi else "older"
        return KernelLoadResult(
            available=False,
            module=module,
            build_info=build_info,
            error=(
                f"kernel ABI {actual_abi} does not match the {expected_abi} this "
                f"dynquant expects (the extension is {newer})"
            ),
            remedy=(
                "Upgrade both together:\n"
                "    pip install --upgrade dynquant\n"
                "The ABI covers the packed weight layout, so mixing versions would "
                "decode weights at the wrong shift and produce plausible but wrong "
                "outputs rather than an error."
            ),
        )

    if not bool(torch.ops.dynquant.built_with_cuda()):
        # A working extension with no device code. Correct on a CPU-only machine
        # and pointless on a GPU one, so the message depends on what is present.
        if torch.cuda.is_available():
            return KernelLoadResult(
                available=False,
                module=module,
                build_info=build_info,
                error="the installed extension was built without CUDA, but a CUDA device is present",
                remedy=(
                    "You most likely installed the sdist on a machine with no CUDA "
                    "toolkit at build time. Install a prebuilt wheel:\n"
                    "    pip install --force-reinstall --no-cache-dir dynquant-kernels"
                ),
            )
        return KernelLoadResult(
            available=False,
            module=module,
            build_info=build_info,
            error="CPU-only extension and no CUDA device; using the torch backend",
            remedy="",
            # Nothing is wrong here: no GPU, so no device code to run. This is the
            # one outcome strict mode must not turn into an error, because it is the
            # steady state of the CPU-only CI job -- which exists to prove the
            # extension builds, registers its ops and agrees on the ABI, all of
            # which it can do without a device.
            expected=True,
        )

    if not torch.cuda.is_available():
        return KernelLoadResult(
            available=False,
            module=module,
            build_info=build_info,
            error="CUDA kernels are installed but torch reports no available device",
            remedy=(
                "Check `nvidia-smi` and that CUDA_VISIBLE_DEVICES is not empty. "
                "Quantizing works without a GPU; only accelerated inference needs one."
            ),
        )

    # Registers meta/fake implementations. Imported only now, because it calls
    # torch.library.register_fake against schemas that only exist once _C is in.
    importlib.import_module("dynquant_kernels.ops")

    return KernelLoadResult(available=True, module=module, build_info=build_info)
