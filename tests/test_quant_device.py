"""Where encoding runs must not change what encoding produces.

The compute device is a performance choice -- forty minutes against one on a 7 B --
and a performance choice that altered the *quality* of the checkpoint would be a
correctness bug wearing a speedup's clothes. Two properties are pinned here:

* placement is plumbed correctly and results land where the caller expects, which
  runs everywhere; and
* CPU and CUDA encode *equally well*, which runs only where there is a GPU.

The second is a tolerance and not an equality, which was worth establishing by
measurement rather than by assumption. The first version of this file asserted
byte-identical codes on the reasoning that the encoder is affine arithmetic and a
``round``. It is, and they still differ, in two ways that were measured on an A100
against a 4096x4096 weight:

* One group scale in 131,072 differs by a single fp16 ulp. ``amin``/``amax`` agree
  exactly and the clip ratios agree exactly, so this is not reduction order: it is
  floating-point contraction, the ``centre +/- half`` clip arithmetic compiling to a
  fused multiply-add on one device and not the other.
* The clip search is an ``argmin`` over eight candidates, so groups where two
  candidates sit within float noise of each other tie-break differently -- 2 groups
  in 131,072 at 4 bits, none at 2, 3 or 8.

Neither is a defect and neither is fixable by asserting harder. What matters is that
the two encodings are equally good, and they are: relative RMSE differs by 0 at 2
and 3 bits, 6.5e-8 at 4 and 1.0e-6 at 8. So the tolerances below are tight enough
that a real regression -- a wrong scale, a misgrouped tensor, an off-by-one in the
packing -- moves them by orders of magnitude, while ulp-level disagreement passes.

The practical consequence belongs in the docs and not only here: a packed checkpoint
is reproducible on a given device, not across devices. Quantizing the same weights
on CPU and on GPU yields two encodings of equal quality that are not the same file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch
from torch import nn

if TYPE_CHECKING:
    from collections.abc import Callable

from dynquant.cli import build_parser
from dynquant.errors import DynQuantError
from dynquant.quant import device as device_module
from dynquant.quant import grid
from dynquant.quant.device import (
    COMPUTE_DEVICE_ENV,
    quantize_tensor,
    resolve_compute_device,
)
from dynquant.quant.quantizer import quantize_model
from dynquant.runtime.linear import DynQuantEmbedding, pack_model

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _weight(rows: int = 64, cols: int = 256, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    # Heavy tails on purpose: the clipping search only has anything to choose
    # between when some groups are outlier-dominated, so a gaussian would test the
    # placement while leaving the interesting half of the arithmetic unexercised.
    weight = torch.randn(rows, cols, generator=generator)
    weight[:, ::37] *= 12.0
    return weight


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------


def test_none_means_stay_put() -> None:
    assert resolve_compute_device(None) is None


@pytest.mark.parametrize("spec", ["none", "None", " same ", "keep"])
def test_opt_out_spellings(spec: str) -> None:
    assert resolve_compute_device(spec) is None


def test_auto_is_cuda_when_available() -> None:
    resolved = resolve_compute_device("auto")
    if torch.cuda.is_available():
        assert resolved is not None and resolved.type == "cuda"
    else:
        # Not an error and not a warning: a CPU-only machine encoding on the CPU is
        # doing the only thing it can, and saying so every time would be noise.
        assert resolved is None


def test_explicit_cpu_is_honoured() -> None:
    assert resolve_compute_device("cpu") == torch.device("cpu")


def test_torch_device_passes_through() -> None:
    assert resolve_compute_device(torch.device("cpu")) == torch.device("cpu")


def test_nonsense_device_is_rejected() -> None:
    with pytest.raises(DynQuantError, match="is not a device"):
        resolve_compute_device("gpu0")


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
def test_explicit_cuda_without_cuda_raises() -> None:
    # Explicit means the caller wanted that device. Silently falling back would hand
    # them the slow path they were trying to avoid, and nothing would say so.
    with pytest.raises(DynQuantError, match="no CUDA"):
        resolve_compute_device("cuda")


def test_env_var_overrides_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(COMPUTE_DEVICE_ENV, "cpu")
    assert resolve_compute_device("auto") == torch.device("cpu")


def test_env_var_does_not_override_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(COMPUTE_DEVICE_ENV, "cpu")
    assert resolve_compute_device(None) is None


def test_env_var_saying_auto_does_not_recurse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(COMPUTE_DEVICE_ENV, "auto")
    resolve_compute_device("auto")  # must terminate


# --------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------


def test_result_lands_on_the_compute_device() -> None:
    quantized, _ = quantize_tensor(_weight(), bits=4, group_size=128, device=torch.device("cpu"))
    assert quantized.device == torch.device("cpu")


def test_opting_out_matches_encoding_in_place() -> None:
    weight = _weight()
    stayed, _ = quantize_tensor(weight, bits=3, group_size=128, device=None)
    named, _ = quantize_tensor(weight, bits=3, group_size=128, device=torch.device("cpu"))
    assert torch.equal(stayed.packed, named.packed)
    assert torch.equal(stayed.scales, named.scales)


def test_pack_model_leaves_the_model_where_it_was() -> None:
    model = nn.Sequential(nn.Linear(256, 64, bias=False))
    pack_model(model, {"0": 4}, group_size=128)
    for buffer in model.buffers():
        assert buffer.device == torch.device("cpu")


# --------------------------------------------------------------------------
# CPU / CUDA agreement -- the property the speedup depends on
# --------------------------------------------------------------------------
#
# "Equal quality", quantified. Both bounds are far tighter than any real defect
# would clear and far looser than ulp noise, which is the only band where a
# tolerance is doing useful work.

RMSE_TOLERANCE = 1e-4
"""Relative. Measured divergence is <=1e-6; a wrong scale or a misgrouped tensor
moves reconstruction error by orders of magnitude, not by one part in 10^4."""

DIVERGENT_GROUP_FRACTION = 1e-3
"""Share of groups allowed to encode differently. Measured: 1.5e-5 at 4 bits."""


def _rmse(quantized: object, reference: torch.Tensor) -> float:
    recon = quantized.dequantize(dtype=torch.float32).cpu()  # type: ignore[attr-defined]
    return float(torch.sqrt(torch.mean((reference.to(torch.float32) - recon) ** 2)))


@requires_cuda
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_cuda_encodes_as_well_as_cpu(bits: int) -> None:
    """Equal reconstruction error, and disagreement confined to a few groups."""
    weight = _weight(seed=bits)
    on_cpu, _ = quantize_tensor(weight, bits=bits, group_size=128, device=torch.device("cpu"))
    on_gpu, _ = quantize_tensor(weight, bits=bits, group_size=128, device=torch.device("cuda"))

    assert _rmse(on_gpu, weight) == pytest.approx(_rmse(on_cpu, weight), rel=RMSE_TOLERANCE)

    differing = (on_cpu.scales.cpu() != on_gpu.scales.cpu()).sum().item()
    assert differing / on_cpu.scales.numel() < DIVERGENT_GROUP_FRACTION, (
        f"{bits}-bit: {differing} of {on_cpu.scales.numel()} group scales differ by device, "
        f"which is more than float contraction and argmin ties account for"
    )


@requires_cuda
def test_pack_model_of_a_cpu_model_using_the_gpu() -> None:
    """The case this exists for: model in host RAM, arithmetic on the accelerator.

    Both halves matter. The packed buffers must come back to the CPU -- a model with
    some modules on one device and some on another fails at the first forward pass,
    not here -- and the encoding must be as good as the CPU's.
    """
    reference = nn.Sequential(nn.Linear(256, 64, bias=False))
    borrowed = nn.Sequential(nn.Linear(256, 64, bias=False))
    borrowed[0].weight.data.copy_(reference[0].weight.data)
    original = reference[0].weight.detach().clone()

    pack_model(reference, {"0": 4}, group_size=128, compute_device=None)
    pack_model(borrowed, {"0": 4}, group_size=128, compute_device="cuda")

    assert borrowed[0].qweight.device == torch.device("cpu")
    assert _rmse(borrowed[0].weight_qt, original) == pytest.approx(
        _rmse(reference[0].weight_qt, original), rel=RMSE_TOLERANCE
    )


@requires_cuda
def test_embedding_packs_on_the_gpu_and_comes_home() -> None:
    original = torch.randn(512, 128)
    home = nn.Embedding(512, 128)
    away = nn.Embedding(512, 128)
    home.weight.data.copy_(original)
    away.weight.data.copy_(original)

    packed_here = DynQuantEmbedding.from_embedding(home, 4, group_size=128, compute_device=None)
    packed_there = DynQuantEmbedding.from_embedding(away, 4, group_size=128, compute_device="cuda")

    assert packed_there.qweight.device == torch.device("cpu")
    assert _rmse(packed_there.weight_qt, original) == pytest.approx(
        _rmse(packed_here.weight_qt, original), rel=RMSE_TOLERANCE
    )


# --------------------------------------------------------------------------
# the out-of-memory fallback
# --------------------------------------------------------------------------
#
# Error handling that runs only when a card is full, which is to say only when
# nobody is watching. Nothing else in the suite reaches it, and it is where the
# per-tensor design claim lives -- one oversized weight costs its own encode and
# not the whole model's speedup. Untested, that claim is a sentence in a commit
# message rather than a property of the code.
#
# None of this needs a GPU: the encoder is replaced by one that fails on command,
# so the tests exercise the recovery rather than the arithmetic.

_OOM_MESSAGE = "CUDA out of memory. Tried to allocate 2.00 GiB"


def _record_encoder(
    monkeypatch: pytest.MonkeyPatch,
    fails: Callable[[torch.Tensor, torch.device | None], bool],
    message: str = _OOM_MESSAGE,
) -> list[tuple[int, str | None]]:
    """Swap in an encoder that records each call and fails when told to.

    Calls are recorded as ``(rows, device type)`` so a test can assert *which*
    tensors reached the accelerator, rather than merely that one of them did.
    ``fails`` is asked per call rather than once, so a test can fail the device
    attempt and let the retry succeed -- otherwise a wrongly-swallowed error is
    indistinguishable from a correctly-propagated one, both arriving as the same
    exception from the second call.
    """
    calls: list[tuple[int, str | None]] = []
    real = grid.quantize_with_search

    def fake(weight: torch.Tensor, **kwargs: Any) -> Any:
        device = kwargs.pop("device", None)
        calls.append((weight.shape[0], None if device is None else device.type))
        if fails(weight, device):
            raise RuntimeError(message)
        return real(weight, **kwargs)

    monkeypatch.setattr("dynquant.quant.device.quantize_with_search", fake)
    return calls


def test_out_of_memory_retries_on_the_weights_own_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _record_encoder(monkeypatch, lambda _weight, device: device is not None)
    weight = _weight()

    quantized, _ = quantize_tensor(weight, bits=4, group_size=128, device=torch.device("cuda"))

    assert calls == [(64, "cuda"), (64, None)], "should try the card once, then the weight"
    assert quantized.device == weight.device

    # And the fallback is a real encode, not a degraded one.
    expected, _ = quantize_tensor(weight, bits=4, group_size=128, device=None)
    assert torch.equal(quantized.packed, expected.packed)
    assert torch.equal(quantized.scales, expected.scales)


def test_the_fallback_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence here would be a run that is slower than asked for and never explains why."""
    _record_encoder(monkeypatch, lambda _weight, device: device is not None)
    said: list[str] = []
    monkeypatch.setattr(
        device_module._log, "warning", lambda message, *args: said.append(message % args)
    )

    quantize_tensor(_weight(), bits=4, group_size=128, device=torch.device("cuda"))

    assert len(said) == 1
    assert "out of memory" in said[0]
    assert "cuda" in said[0]


def test_a_genuine_error_is_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Failing only the device attempt is what gives this test teeth: if the error
    # were misread as an allocation failure, the retry would succeed on the CPU and
    # the bug would vanish into a slower run instead of raising.
    calls = _record_encoder(
        monkeypatch,
        lambda _weight, device: device is not None,
        message="group_size 128 does not divide in_features 100",
    )

    with pytest.raises(RuntimeError, match="does not divide"):
        quantize_tensor(_weight(), bits=4, group_size=128, device=torch.device("cuda"))

    assert calls == [(64, "cuda")], "a real bug must not be retried anywhere"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("CUDA out of memory. Tried to allocate 2.00 GiB", True),
        ("CUDA OUT OF MEMORY", True),
        ("group_size does not divide in_features", False),
        ("expected scalar type Half but found Float", False),
    ],
)
def test_out_of_memory_is_read_from_the_message(message: str, expected: bool) -> None:
    assert device_module._is_out_of_memory(RuntimeError(message)) is expected


def test_the_typed_error_is_recognised_whatever_it_says() -> None:
    """The reason the type is checked and not only the message."""
    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom is None:
        pytest.skip("this torch predates torch.cuda.OutOfMemoryError")
    assert device_module._is_out_of_memory(oom("allocation failed")) is True


def test_the_fallback_is_per_tensor_and_not_per_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """One oversized weight must not cost every other module the accelerator."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = _record_encoder(
        monkeypatch, lambda weight, device: device is not None and weight.shape[0] >= 256
    )

    model = nn.Sequential(nn.Linear(256, 256, bias=False), nn.Linear(256, 8, bias=False))
    quantize_model(model, {"0": 4, "1": 4}, group_size=128, compute_device="cuda")

    assert (256, "cuda") in calls, "the big weight should have been tried on the card"
    assert (256, None) in calls, "and retried on its own device"
    assert (8, "cuda") in calls, "the small weight should still get the card"
    assert (8, None) not in calls, "and must not be dragged onto the CPU with it"


# --------------------------------------------------------------------------
# CLI wiring
# --------------------------------------------------------------------------


def test_the_flag_reaches_the_commands_that_encode() -> None:
    parser = build_parser()
    assert parser.parse_args(["quantize", "m", "-o", "o"]).compute_device == "auto"
    argv = ["quantize", "m", "-o", "o", "--compute-device", "cpu"]
    assert parser.parse_args(argv).compute_device == "cpu"
    assert parser.parse_args(["eval", "m", "--task", "gsm8k"]).compute_device == "auto"


def test_the_flag_is_absent_where_nothing_is_encoded() -> None:
    # ``inspect`` reads shapes and scores. A knob that silently does nothing is worse
    # than no knob: it is one a user can set, and then wonder about.
    assert not hasattr(build_parser().parse_args(["inspect", "m"]), "compute_device")


@requires_cuda
def test_quantize_model_of_a_cpu_model_using_the_gpu() -> None:
    reference = nn.Sequential(nn.Linear(256, 64, bias=False))
    borrowed = nn.Sequential(nn.Linear(256, 64, bias=False))
    borrowed[0].weight.data.copy_(reference[0].weight.data)

    on_cpu = quantize_model(reference, {"0": 3}, group_size=128, compute_device=None)
    on_gpu = quantize_model(borrowed, {"0": 3}, group_size=128, compute_device="cuda")

    # The reconstruction must come home to the CPU model, whatever encoded it.
    assert borrowed[0].weight.device == torch.device("cpu")
    # Same reported damage. A report that drifted while the weights were equally
    # good would make per-layer error incomparable across runs, which is the one
    # thing the report exists to support.
    assert on_cpu.layers["0"].rmse == pytest.approx(on_gpu.layers["0"].rmse, rel=RMSE_TOLERANCE)
