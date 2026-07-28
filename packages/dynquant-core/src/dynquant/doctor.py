"""``dynquant doctor`` -- diagnose an installation before it is trusted.

Two things are checked, and the distinction is the point:

**Environment** -- what is installed, what the GPU is, which backend was chosen
and why the others were not. Answers "will this run?".

**Self-check** -- actually quantize tensors, actually invoke the kernels, and
compare against an independent expectation. Answers "will this run *correctly*?".

The second is the one that matters, because the failure mode this package has to
defend against is not a crash. A kernel compiled for the wrong ABI, or a fatbin
with no cubin for the running GPU, or an extension linked against a different
libtorch, can all produce a tensor of the right shape and dtype full of wrong
numbers. Downstream that appears as "the quantized model is a bit worse", which is
indistinguishable from quantization simply being lossy -- and so it never gets
reported as a bug. Every check below is chosen to fail loudly instead.

Nothing here raises on a broken install; that is the situation it exists for. The
return value carries the verdict, and the CLI turns it into an exit code.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any

from ._version import KERNEL_ABI_VERSION, __version__
from .constants import BIT_OPTIONS, DEFAULT_GROUP_SIZE, KERNELS_IMPORT_NAME

__all__ = ["Check", "DoctorReport", "diagnose", "render"]

_OK = "ok"
_WARN = "warn"
_FAIL = "fail"
_SKIP = "skip"


@dataclass(slots=True)
class Check:
    """One diagnostic outcome.

    ``status`` is one of ``ok`` / ``warn`` / ``fail`` / ``skip``. The distinction
    between ``warn`` and ``fail`` is whether results produced by this installation
    can be trusted: a missing CUDA wheel on a CPU box is a warning (quantizing
    still works), a numerical mismatch is a failure (nothing can be trusted).
    """

    name: str
    status: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.status == _FAIL


@dataclass(slots=True)
class DoctorReport:
    environment: dict[str, Any]
    backends: dict[str, Any]
    checks: list[Check]

    @property
    def ok(self) -> bool:
        """True when nothing that would invalidate results went wrong."""
        return not any(check.failed for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "environment": self.environment,
            "backends": self.backends,
            "checks": [
                {"name": c.name, "status": c.status, "message": c.message, "detail": c.detail}
                for c in self.checks
            ],
        }


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def _environment() -> dict[str, Any]:
    info: dict[str, Any] = {
        "dynquant_core": __version__,
        "kernel_abi_expected": KERNEL_ABI_VERSION,
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
    }
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        info["torch"] = f"NOT IMPORTABLE: {exc}"
        return info

    info["torch"] = torch.__version__
    info["torch_cuda"] = torch.version.cuda or ""
    info["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        info["driver"] = getattr(torch.version, "cuda", "") or ""
        devices = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "sm": f"{props.major}.{props.minor}",
                    "memory_gib": round(props.total_memory / (1 << 30), 2),
                    "multiprocessors": props.multi_processor_count,
                }
            )
        info["devices"] = devices

    try:
        kernels = __import__(KERNELS_IMPORT_NAME)
    except Exception:  # noqa: BLE001 - absence is the normal case
        info["dynquant_kernels"] = ""
    else:
        info["dynquant_kernels"] = getattr(kernels, "__version__", "unknown")
    return info


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def _check_packing() -> Check:
    """Bit packing is a bijection at every width. Runs anywhere, needs no GPU.

    First because everything downstream is meaningless if it fails: a wrong shift
    here corrupts weights at quantization time, on the machine doing the
    quantizing, with no GPU involved at all.
    """
    try:
        import torch

        from .quant.pack import pack_nbit, unpack_nbit

        for bits in BIT_OPTIONS:
            # 130 is deliberately not a multiple of anything: it forces a partial
            # trailing word, which is where an off-by-one in the layout hides.
            codes = torch.randint(0, 2**bits, (3, 130), dtype=torch.uint8)
            if not torch.equal(unpack_nbit(pack_nbit(codes, bits), bits, 130), codes):
                return Check(
                    "packing",
                    _FAIL,
                    f"{bits}-bit pack/unpack is not a bijection on this machine",
                )
    except Exception as exc:  # noqa: BLE001
        return Check("packing", _FAIL, f"packing raised: {exc}")
    return Check("packing", _OK, f"{'/'.join(map(str, BIT_OPTIONS))}-bit round-trip exact")


def _check_quantization_error() -> Check:
    """Reconstruction error matches the uniform-quantizer prediction.

    Independent of the encoder: the predicted step comes from each group's own
    min/max computed with plain torch. An encoder that widened ranges, dropped a
    zero-point, or mixed up group strides would land outside the band even though
    every shape stayed right -- which is exactly the class of bug that otherwise
    reads as "quantization is lossy".
    """
    try:
        import math

        import torch

        from .quant.tensor import QuantTensor

        generator = torch.Generator().manual_seed(0)
        weight = torch.randn(64, 512, generator=generator, dtype=torch.float32) * 0.02
        worst: tuple[str, float, float] | None = None
        for bits in BIT_OPTIONS:
            qt = QuantTensor.from_dense(
                weight.to(torch.float16), bits=bits, group_size=DEFAULT_GROUP_SIZE
            )
            measured = qt.quantization_error(weight.to(torch.float16))["rel_fro"]

            groups = weight.reshape(64, -1, DEFAULT_GROUP_SIZE)
            step = (groups.amax(dim=-1) - groups.amin(dim=-1)) / (2**bits - 1)
            predicted = (
                math.sqrt(float((step.pow(2) / 12.0).mean()))
                / float(weight.pow(2).mean().sqrt())
                * 0.90
            )
            if not 0.6 * predicted <= measured <= 1.4 * predicted:
                worst = (f"{bits}-bit", measured, predicted)
                break
        if worst is not None:
            label, measured, predicted = worst
            return Check(
                "quantization-error",
                _FAIL,
                f"{label} reconstruction error {measured:.5f} is not the predicted "
                f"{predicted:.5f}; this installation would produce degraded checkpoints",
                detail={"measured": measured, "predicted": predicted},
            )
    except Exception as exc:  # noqa: BLE001
        return Check("quantization-error", _FAIL, f"quantization raised: {exc}")
    return Check("quantization-error", _OK, "within theory at every bit-width")


def _check_kernels() -> Check:
    """Invoke the compiled probes and compare against the CPU reference.

    Each probe covers a distinct link-time dependency (a raw launch, Thrust, and
    cuBLASLt), so a failure names which one is broken rather than reporting
    "kernels don't work".
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        return Check("kernels", _SKIP, f"torch not importable: {exc}")

    from .runtime.backend import Backend, available_backends

    cuda = next(s for s in available_backends() if s.backend is Backend.CUDA)
    if not cuda.available:
        status = _WARN if torch.cuda.is_available() else _SKIP
        note = (
            "a GPU is present but the compiled kernels are unusable, so inference "
            "will fall back to the reference path (correct, but no VRAM saving)"
            if torch.cuda.is_available()
            else "no compiled kernels and no GPU; the reference path is the right one here"
        )
        return Check("kernels", status, f"{note}: {cuda.reason}", detail={"remedy": cuda.remedy})

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    failures: list[str] = []

    x = torch.randn(4096, generator=generator, device=device, dtype=torch.float16)
    y = torch.randn(4096, generator=generator, device=device, dtype=torch.float16)
    try:
        got = torch.ops.dynquant.probe_axpy(x, y, 2.5)
        want = (x.float() * 2.5 + y.float()).to(torch.float16)
        if not torch.equal(got, want):
            worst = (got.float() - want.float()).abs().max().item()
            failures.append(f"probe_axpy (kernel launch) differs by up to {worst:.3e}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"probe_axpy (kernel launch) raised: {exc}")

    try:
        got_sum = float(torch.ops.dynquant.probe_reduce(x))
        want_sum = float(x.float().sum())
        # fp32 sequential vs tree reduction over 4096 halves: relative 1e-4 is the
        # honest bound. A wrong reduction is off by orders of magnitude, not ulps.
        if abs(got_sum - want_sum) > 1e-4 * max(1.0, abs(want_sum)):
            failures.append(f"probe_reduce (Thrust) gave {got_sum:.6g}, expected {want_sum:.6g}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"probe_reduce (Thrust) raised: {exc}")

    a = torch.randn(128, 256, generator=generator, device=device, dtype=torch.float16)
    b = torch.randn(256, 64, generator=generator, device=device, dtype=torch.float16)
    try:
        got_gemm = torch.ops.dynquant.probe_gemm(a, b).float()
        want_gemm = a.float() @ b.float()
        scale = max(1.0, float(want_gemm.abs().max()))
        deviation = float((got_gemm - want_gemm).abs().max()) / scale
        # fp16 output of an fp32-accumulated GEMM: one fp16 ulp near the top of the
        # range is ~1e-3 relative. Tighter than that would flag correct hardware.
        if deviation > 5e-3:
            failures.append(f"probe_gemm (cuBLASLt) deviates by {deviation:.3e} relative")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"probe_gemm (cuBLASLt) raised: {exc}")

    detail = dict(cuda.detail or {})
    if failures:
        return Check("kernels", _FAIL, "; ".join(failures), detail=detail)
    return Check(
        "kernels", _OK, "launch, Thrust and cuBLASLt all agree with the reference", detail=detail
    )


def _check_cubin_coverage() -> Check:
    """Does the wheel contain machine code for the GPU in this box?

    A fatbin with only PTX for the running architecture still works -- the driver
    JITs it -- but the first launch of each kernel stalls for seconds and the result
    is not tuned for the target. Users experience that as "dynquant is slow to warm
    up" and have no way to attribute it, so it gets named here.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return Check("cubin-coverage", _SKIP, "no CUDA device")
        kernels = __import__(KERNELS_IMPORT_NAME)
        if not kernels.is_available():
            return Check("cubin-coverage", _SKIP, "compiled kernels unavailable")
        compiled = kernels.compiled_architectures()
        if not compiled:
            return Check(
                "cubin-coverage",
                _SKIP,
                "the extension does not report its architecture list (built with CUDA < 11.5)",
            )
        props = torch.cuda.get_device_properties(0)
        device_sm = props.major * 10 + props.minor
        if device_sm in compiled:
            return Check(
                "cubin-coverage",
                _OK,
                f"native cubin present for sm_{device_sm}",
                detail={"compiled": list(compiled), "device": device_sm},
            )
        return Check(
            "cubin-coverage",
            _WARN,
            f"no cubin for sm_{device_sm} (wheel has "
            f"{', '.join(f'sm_{a}' for a in compiled)}); kernels will be JIT-compiled "
            f"from PTX on first launch, costing seconds once per process",
            detail={"compiled": list(compiled), "device": device_sm},
        )
    except Exception as exc:  # noqa: BLE001
        return Check("cubin-coverage", _SKIP, f"could not determine: {exc}")


def _check_abi_agreement() -> Check:
    """The Python and binary halves must claim the same ABI number."""
    try:
        kernels = __import__(KERNELS_IMPORT_NAME)
    except Exception:  # noqa: BLE001
        return Check("abi", _SKIP, "compiled kernels not installed")

    shell = int(getattr(kernels, "ABI_VERSION", -1))
    if shell != KERNEL_ABI_VERSION:
        return Check(
            "abi",
            _FAIL,
            f"dynquant-kernels claims ABI v{shell} but dynquant-core speaks "
            f"v{KERNEL_ABI_VERSION}; the two wheels are not a matched pair",
            detail={"kernels": shell, "core": KERNEL_ABI_VERSION},
        )

    stamp = kernels.build_info().get("ABI_VERSION")
    if stamp is not None and int(stamp) != shell:
        return Check(
            "abi",
            _FAIL,
            f"the compiled binary was built at ABI v{stamp} but its Python shell "
            f"claims v{shell}; this wheel is internally inconsistent",
            detail={"binary": int(stamp), "shell": shell},
        )
    return Check("abi", _OK, f"core, shell and binary all at ABI v{KERNEL_ABI_VERSION}")


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def diagnose() -> DoctorReport:
    """Run every check. Never raises."""
    from .runtime.backend import backend_report

    try:
        backends = backend_report()
    except Exception as exc:  # noqa: BLE001 - diagnosis must survive anything
        backends = {"error": str(exc)}

    checks = [
        _check_packing(),
        _check_quantization_error(),
        _check_abi_agreement(),
        _check_kernels(),
        _check_cubin_coverage(),
    ]
    return DoctorReport(environment=_environment(), backends=backends, checks=checks)


_GLYPH = {_OK: "PASS", _WARN: "WARN", _FAIL: "FAIL", _SKIP: "SKIP"}


def render(report: DoctorReport) -> str:
    """Human-readable form. Plain ASCII: this output ends up in bug reports,
    pasted through terminals and CI logs with every imaginable encoding."""
    lines: list[str] = ["DynQuant doctor", "=" * 60, "", "Environment"]
    for key, value in report.environment.items():
        if key == "devices":
            for device in value:
                lines.append(
                    f"  gpu[{device['index']}]       {device['name']} "
                    f"(sm_{device['sm'].replace('.', '')}, {device['memory_gib']} GiB, "
                    f"{device['multiprocessors']} SMs)"
                )
            continue
        lines.append(f"  {key:<22} {value if value != '' else '-'}")

    lines += ["", "Backends"]
    selected = report.backends.get("selected")
    lines.append(f"  selected               {selected or 'NONE'}")
    if report.backends.get("override"):
        lines.append(
            f"  override               ${'DYNQUANT_BACKEND'}={report.backends['override']}"
        )
    for entry in report.backends.get("backends", []):
        mark = "  active" if entry["available"] else "unusable"
        lines.append(f"  {entry['backend']:<10} {mark}  {entry['reason']}".rstrip())
        if entry.get("remedy"):
            for remedy_line in str(entry["remedy"]).splitlines():
                lines.append(f"             -> {remedy_line}")

    lines += ["", "Checks"]
    for check in report.checks:
        lines.append(f"  [{_GLYPH[check.status]}] {check.name:<20} {check.message}")
        if check.status in (_WARN, _FAIL) and check.detail.get("remedy"):
            for remedy_line in str(check.detail["remedy"]).splitlines():
                lines.append(f"             -> {remedy_line}")

    lines += [""]
    if report.ok:
        lines.append("Result: OK -- this installation produces trustworthy output.")
        if any(c.status == _WARN for c in report.checks):
            lines.append("        Warnings above affect speed or memory, not correctness.")
    else:
        lines.append("Result: FAILED -- do not trust output from this installation.")
    return "\n".join(lines)
