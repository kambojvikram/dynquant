"""A checkpoint that says it is packed has to come back packed, or say why not.

Both Qwen3-Omni packed arms scored **0.0%** on 500 SLURP items -- 499 and 500
unparseable generations, 478 s and 482 s of box time -- and nothing raised. The
model was randomly initialised. ``dynquant eval`` never called
``register_hf_quantizer``, and transformers' response to a quantization method it
cannot resolve is not an error: ``supports_quant_method`` logs a warning and
returns ``False``, ``pre_quantized`` goes ``False``, ``hf_quantizer`` goes
``None``, and the packed tensors then match no parameter the model has.

That failure is indistinguishable from a catastrophic accuracy result by looking
at the number, which is why the guard counts modules instead. The first test here
is a negative control against transformers itself: if upstream ever starts raising
on an unknown method, the premise of this whole file is gone and the test says so.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from dynquant.commands import _shared  # noqa: E402
from dynquant.constants import HF_QUANT_METHOD  # noqa: E402
from dynquant.errors import DynQuantError  # noqa: E402


class _Config:
    def __init__(self, quantization_config=None):
        if quantization_config is not None:
            self.quantization_config = quantization_config


class _Model(nn.Module):
    def __init__(self, config, child=None):
        super().__init__()
        self.config = config
        if child is not None:
            self.child = child


def test_transformers_downgrades_an_unknown_method_instead_of_raising():
    """The negative control. Read the dependency's behaviour, do not assume it."""
    auto = pytest.importorskip("transformers.quantizers.auto")
    supported = auto.AutoHfQuantizer.supports_quant_method(
        {"quant_method": "a-method-nobody-registered"}
    )
    assert supported is False, (
        "transformers now rejects an unknown quant_method outright; the silent-random-init "
        "failure this module guards against can no longer happen the same way"
    )


def test_load_model_registers_the_quantizer_before_from_pretrained(monkeypatch):
    """Registration has to happen *before* the load, not after it."""
    auto = pytest.importorskip("transformers.quantizers.auto")
    seen = {}

    class _Loader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            seen["registered"] = HF_QUANT_METHOD in auto.AUTO_QUANTIZER_MAPPING
            return _Model(_Config())

    monkeypatch.setattr(_shared, "_model_loader", lambda *a, **k: _Loader)
    _shared.load_model("some/path", device="cpu")

    assert seen["registered"] is True


def test_a_dynquant_checkpoint_that_came_back_dense_raises():
    """The measurement that separates a failed load from a bad model."""
    model = _Model(_Config({"quant_method": HF_QUANT_METHOD}), child=nn.Linear(4, 4))

    with pytest.raises(DynQuantError) as excinfo:
        _shared._check_packed_load("out/packed-3bit", model)

    message = str(excinfo.value)
    assert "not one module came back packed" in message
    assert "register_hf_quantizer" in message  # names the remedy


def test_banks_alone_satisfy_the_guard():
    """On this MoE 91.40% of the parameters are banks and no Linear is packed."""
    from dynquant.quant.device import quantize_tensor
    from dynquant.runtime.experts import DynQuantExpertBank

    quantized, _ = quantize_tensor(torch.randn(2, 8, 64).half(), bits=4, device=None)
    model = _Model(_Config({"quant_method": HF_QUANT_METHOD}), child=DynQuantExpertBank(quantized))

    _shared._check_packed_load("out/packed-4bit", model)  # does not raise


def test_a_checkpoint_that_never_claimed_to_be_packed_is_left_alone():
    model = _Model(_Config(), child=nn.Linear(4, 4))
    _shared._check_packed_load("out/merged", model)  # does not raise


def test_a_foreign_quantizer_is_left_alone():
    model = _Model(_Config({"quant_method": "gptq"}), child=nn.Linear(4, 4))
    _shared._check_packed_load("out/gptq-4bit", model)  # does not raise


@pytest.mark.parametrize(
    "quant_config",
    [
        {"quant_method": HF_QUANT_METHOD},
        pytest.param(type("Q", (), {"quant_method": HF_QUANT_METHOD})(), id="config-object"),
    ],
)
def test_declares_dynquant_reads_both_shapes_the_config_can_be_in(quant_config):
    """A dict when the quantizer was skipped, an object when it was not.

    The dict shape is the one that matters: it is what a *skipped* load leaves
    behind, so a check that only understood the object shape would return False on
    exactly the checkpoints the guard exists for.
    """
    assert _shared._declares_dynquant(_Config(quant_config)) is True
