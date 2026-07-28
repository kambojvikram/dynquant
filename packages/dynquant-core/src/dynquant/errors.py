"""DynQuant exception hierarchy.

Every error carries a concrete remedy in its message. A user who hits one should
not have to read the source to find out what to do next -- the research code's
habit of ``except Exception: pass`` (which silently degraded the gradient signal
to zeros for every LoRA layer) is exactly what this module exists to prevent.
"""

from __future__ import annotations

__all__ = [
    "AllocationError",
    "BackendUnavailableError",
    "BudgetInfeasibleError",
    "ClassificationError",
    "DynQuantError",
    "FormatVersionError",
    "KernelAbiMismatchError",
    "MissingDependencyError",
    "PackingError",
    "SignalCollectionError",
    "StatsCoverageError",
]


class DynQuantError(Exception):
    """Base class for every DynQuant error."""


# --------------------------------------------------------------------------
# Environment / dependency
# --------------------------------------------------------------------------


class MissingDependencyError(DynQuantError, ImportError):
    """An optional dependency is required for the requested feature."""

    def __init__(self, package: str, *, feature: str, extra: str) -> None:
        super().__init__(
            f"{feature} requires the '{package}' package, which is not installed.\n"
            f"  Install it with:  pip install 'dynquant[{extra}]'"
        )
        self.package = package
        self.feature = feature
        self.extra = extra


class BackendUnavailableError(DynQuantError, RuntimeError):
    """A specific compute backend was requested but cannot be used."""

    def __init__(self, backend: str, reason: str, *, remedy: str | None = None) -> None:
        msg = f"DynQuant backend '{backend}' is unavailable: {reason}"
        if remedy:
            msg += f"\n  {remedy}"
        msg += "\n  Run 'dynquant doctor' for a full diagnosis."
        super().__init__(msg)
        self.backend = backend
        self.reason = reason


class KernelAbiMismatchError(DynQuantError, RuntimeError):
    """The compiled kernels wheel speaks a different ABI than this core.

    This is the guard that stops a stale binary from silently producing wrong
    numbers. Layout changes are invisible at the C boundary -- both sides see a
    ``uint32*`` -- so the version handshake is the only thing standing between a
    mismatched wheel and a model that decodes to noise.
    """

    def __init__(self, *, found: int, expected: int, minimum: int, kernels_version: str) -> None:
        super().__init__(
            f"dynquant-kernels {kernels_version} was built against kernel ABI v{found}, "
            f"but this dynquant-core needs v{minimum}..v{expected}.\n"
            f"  The packed weight layout differs between ABI versions; loading anyway "
            f"would produce silently incorrect results.\n"
            f"  Fix with:  pip install --upgrade --force-reinstall dynquant"
        )
        self.found = found
        self.expected = expected


# --------------------------------------------------------------------------
# On-disk formats
# --------------------------------------------------------------------------


class FormatVersionError(DynQuantError, ValueError):
    """An on-disk file declares a schema this build cannot read."""

    def __init__(self, *, path: str, kind: str, found: object, supported: object) -> None:
        super().__init__(
            f"{kind} at {path} declares schema {found!r}, which this DynQuant "
            f"build cannot read (supported: {supported!r}).\n"
            f"  If the file is older, run:  dynquant migrate {path}\n"
            f"  If it is newer, upgrade:    pip install --upgrade dynquant"
        )
        self.path = path
        self.found = found


class PackingError(DynQuantError, ValueError):
    """A tensor cannot be packed or unpacked with the given parameters."""


# --------------------------------------------------------------------------
# Pipeline stages
# --------------------------------------------------------------------------


class SignalCollectionError(DynQuantError, RuntimeError):
    """Signal collection could not proceed. Never silently degraded."""


class StatsCoverageError(DynQuantError, ValueError):
    """The stats file does not cover enough of the model to allocate safely.

    Raised instead of quantizing uncovered layers at a blind default. The
    supplement's behaviour -- treat a missing gradient signal as ``0.0`` and a
    missing coherence signal as the neutral ``1.0`` -- makes an unhooked layer
    look maximally *unimportant*, so it gets crushed to the lowest bit-width.
    Silence in the signal must never read as "safe to destroy".
    """

    def __init__(self, *, covered: int, total: int, threshold: float, examples: list[str]) -> None:
        shown = "\n    ".join(examples[:8])
        more = f"\n    ... and {len(examples) - 8} more" if len(examples) > 8 else ""
        super().__init__(
            f"Signal stats cover only {covered}/{total} quantizable modules "
            f"({covered / max(total, 1):.1%}), below the required {threshold:.0%}.\n"
            f"  Uncovered modules would be allocated blind. Examples:\n    {shown}{more}\n"
            f"  Either collect signals for them (check the tracker's include/exclude "
            f"filters and that the stats file matches this model), or lower the bar "
            f"explicitly with --min-coverage."
        )
        self.covered = covered
        self.total = total


class ClassificationError(DynQuantError, ValueError):
    """A module's structural role could not be determined."""


class AllocationError(DynQuantError, ValueError):
    """Bit allocation failed."""


class BudgetInfeasibleError(AllocationError):
    """The requested budget cannot be met even at the lowest bit-width."""

    def __init__(self, *, target_bits: float, floor_bits: float, min_bits: int) -> None:
        super().__init__(
            f"Target of {target_bits:.3f} average bits is below the {floor_bits:.3f} bits "
            f"reachable when every quantizable module is at the {min_bits}-bit minimum.\n"
            f"  Raise --target, or allow a lower minimum width."
        )
        self.target_bits = target_bits
        self.floor_bits = floor_bits
