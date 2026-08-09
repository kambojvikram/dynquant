"""The packed runtime holding a batched MoE expert bank.

91.5% of LFM2.5-8B-A1B is two 3-D parameters per layer. Until this existed the
encoder could write them and nothing could read them back, so both DynQuant arms of
the campaign were directories that were honest about their size and useless.

The model below is not a mock of a bank -- it is the structure of
``transformers.models.lfm2_moe.modeling_lfm2_moe.Lfm2MoeExperts``: two
``nn.Parameter``\\s of rank 3 and a loop that reaches an expert by indexing one of
them. Which is the whole reason this design works without knowing the architecture:
the parent is never edited, and the only thing standing where the parameter stood has
to answer is ``bank[expert]``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from transformers import LlamaConfig  # noqa: E402

from dynquant.constants import QUANT_TENSOR_SUFFIXES  # noqa: E402
from dynquant.errors import DynQuantError  # noqa: E402
from dynquant.integration import hf_quantizer  # noqa: E402
from dynquant.quant.checkpoint import export_packed_checkpoint  # noqa: E402
from dynquant.quant.quantizer import quantize_model  # noqa: E402
from dynquant.runtime.linear import (  # noqa: E402
    DynQuantExpertBank,
    ExpertBank,
    pack_model,
    packed_bytes,
    resolve_target,
)

PACKED = QUANT_TENSOR_SUFFIXES["packed"]

EXPERTS, HIDDEN, INTER, TOP_K = 4, 256, 384, 2
BANKS = ("block.experts.gate_up_proj", "block.experts.down_proj")


class _Experts(nn.Module):
    """``Lfm2MoeExperts``: the banks, and the loop that indexes them."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(EXPERTS, 2 * INTER, HIDDEN))
        self.down_proj = nn.Parameter(torch.empty(EXPERTS, HIDDEN, INTER))

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final = torch.zeros_like(hidden_states)
        mask = F.one_hot(top_k_index, num_classes=EXPERTS).permute(2, 1, 0)
        for expert_idx in torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero():
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            state = hidden_states[token_idx]
            # The line this whole class exists for: a 0-d tensor index into a bank.
            gate, up = F.linear(state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            out = F.linear(F.silu(gate) * up, self.down_proj[expert_idx])
            out = out * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, out.to(final.dtype))
        return final


class _Router(nn.Module):
    """``Lfm2MoeTopKRouter``: a bare 2-D parameter and ``F.linear``, not a ``Linear``."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(EXPERTS, HIDDEN))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights, index = torch.topk(F.linear(hidden_states, self.weight), k=TOP_K, dim=-1)
        return weights.sigmoid(), index


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = _Router()
        self.experts = _Experts()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weights, index = self.gate(hidden_states)
        return self.experts(hidden_states, index, weights)


class _MoEModel(nn.Module):
    """A model shaped like the MoE half of LFM2, with a transformers config attached.

    The config is real (``export_packed_checkpoint`` calls ``config.save_pretrained``)
    and untied, so nothing here depends on the tied-head path tested elsewhere.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.block = _Block()
        self.config = LlamaConfig(hidden_size=HIDDEN, tie_word_embeddings=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(self.proj(hidden_states))


def _fresh() -> _MoEModel:
    torch.manual_seed(0)
    model = _MoEModel()
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.05)
    return model.half().eval()


def _bits() -> dict[str, int]:
    # Two widths across the two banks, so a runtime that reads one file-level default
    # instead of the per-module entry decodes the second bank as noise.
    return {"proj": 4, BANKS[0]: 4, BANKS[1]: 3}


def _x() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(6, HIDDEN).half()


def _out(model: nn.Module) -> torch.Tensor:
    with torch.no_grad():
        return model(_x()).float()


# --------------------------------------------------------------------------
# The bank as its parent sees it
# --------------------------------------------------------------------------


def test_a_packed_bank_answers_its_parent_unchanged() -> None:
    """The parent's own loop, untouched, over packed banks.

    Two references, because either alone is passable. Matching the *encoder* -- the
    same widths written back over the parameters in place -- says the packed bank
    decodes to what the checkpoint means. Being far from *fp16* says the packing
    happened at all: a ``__getitem__`` that quietly returned a dense cached copy
    would match fp16 exactly and look perfect against no other yardstick.

    Turns red when: the row band ``__getitem__`` slices drifts (a bank is
    ``[E, out, in]`` flattened to ``[E * out, in]``, and an off-by-one-expert slice
    is still the right *shape*), or a width stops being read per module.
    """
    fp16 = _out(_fresh())

    reference_model = _fresh()
    quantize_model(reference_model, _bits(), in_place=True, compute_device=None)
    encoder = _out(reference_model)

    packed = _fresh()
    pack_model(packed, _bits(), compute_device=None)
    got = _out(packed)

    to_encoder = (got - encoder).abs().max().item()
    to_fp16 = (got - fp16).abs().max().item()
    assert to_encoder < 1e-3, to_encoder
    # A ratio, so it cannot pass by both numbers being large.
    assert to_fp16 > 20 * to_encoder, (to_fp16, to_encoder)


def test_one_expert_is_dequantized_at_a_time_and_addressed_without_a_copy() -> None:
    """The saving is that the other experts stay packed while one is read.

    A bank that dequantized itself and cached the result would pass every accuracy
    test in this file and defeat the entire point, so the property is checked
    structurally: the row band an expert occupies is a *view* of the packed buffer,
    sharing storage, and the dense tensor that comes back is one expert's.

    Turns red when: ``QuantTensor.rows`` starts copying, or ``__getitem__`` grows a
    cache of dequantized experts.
    """
    model = _fresh()
    pack_model(model, {BANKS[0]: 4}, compute_device=None)
    bank = model.block.experts.gate_up_proj
    assert isinstance(bank, DynQuantExpertBank)

    assert bank.shape == torch.Size([EXPERTS, 2 * INTER, HIDDEN])
    assert len(bank) == EXPERTS
    assert bank[2].shape == (2 * INTER, HIDDEN)

    band = bank.out_features
    whole = bank.weight_qt
    slice_of_it = whole.rows(2 * band, 3 * band)
    assert slice_of_it.packed.data_ptr() != 0
    assert slice_of_it.packed.untyped_storage().data_ptr() == (
        whole.packed.untyped_storage().data_ptr()
    )
    assert torch.equal(bank[2], slice_of_it.dequantize())
    assert torch.equal(bank[2], bank.dequantize()[2])
    assert torch.equal(bank[-1], bank[EXPERTS - 1])


def test_a_bank_stands_where_the_parameter_stood() -> None:
    """Same name, so the parent reaches it and the checkpoint keys line up.

    ``nn.Module.__setattr__`` refuses to put a module over a registered parameter, so
    this is the assertion that the surgery deregistered the parameter rather than
    installing the bank under some adjacent name.

    Turns red when: ``replace_module`` stops clearing ``_parameters``, or the buffer
    names drift from the keys ``dynquant export`` writes.
    """
    model = _fresh()
    pack_model(model, {BANKS[0]: 4}, compute_device=None)

    experts = model.block.experts
    assert "gate_up_proj" not in experts._parameters
    assert isinstance(experts._modules["gate_up_proj"], DynQuantExpertBank)
    assert dict(model.named_modules())[BANKS[0]] is experts.gate_up_proj

    keys = {key for key in model.state_dict() if key.startswith(BANKS[0])}
    assert keys == {f"{BANKS[0]}.{suffix}" for suffix in ("qweight", "scales", "offsets")}
    # And the untouched bank is still exactly one tensor under its own name.
    assert BANKS[1] in model.state_dict()


def test_packed_bytes_counts_the_bare_parameters_it_used_to_miss() -> None:
    """``named_modules`` cannot see a weight that is a parameter of something else.

    A bank and an MoE router are both bare parameters on modules of neither class, so
    the accounting walked straight past them: on LFM2.5-8B-A1B that is 91.5% of the
    model missing from the denominator, and every compression ratio computed from it
    was flattering by that much.

    Turns red when: the parameter pass is dropped, or starts counting rank-1 tensors
    (norms and biases are not quantization targets and would move the goalposts the
    other way).
    """
    model = _fresh()
    before = packed_bytes(model)
    bank_bytes = sum(model.get_parameter(name).numel() * 2 for name in BANKS)  # fp16, both banks
    router_bytes = EXPERTS * HIDDEN * 2
    proj_bytes = HIDDEN * HIDDEN * 2

    assert before["packed_bytes"] == 0
    assert before["dense_bytes"] == bank_bytes + router_bytes + proj_bytes
    assert set(before["dense_modules"]) == {"proj", *BANKS, "block.gate.weight"}

    pack_model(model, _bits(), compute_device=None)
    after = packed_bytes(model)

    # Only the router is left dense, and the total shrank by roughly the ratio the
    # widths imply rather than by nothing.
    assert after["dense_modules"] == ["block.gate.weight"]
    assert after["dense_bytes"] == router_bytes
    assert after["packed_bytes"] < before["dense_bytes"] / 3


# --------------------------------------------------------------------------
# What is still refused, and why
# --------------------------------------------------------------------------


def test_a_bare_2d_parameter_is_refused_by_a_reason_the_grouped_path_will_not_fix() -> None:
    """A router is not a small expert bank, and the message must not say it is.

    ``_Router`` holds ``[num_experts, hidden]`` and passes it *whole* to ``F.linear``.
    No module can stand where it stands, because nothing indexes it -- which is a
    different sentence from "the grouped path is not built yet", and that is the
    sentence this case used to get.

    Turns red when: the rank test in ``resolve_target`` widens to accept rank 2, which
    would install a bank whose parent then calls ``F.linear`` on a module.
    """
    model = _fresh()
    with pytest.raises(DynQuantError) as excinfo:
        resolve_target(model, "block.gate.weight")
    message = str(excinfo.value)
    assert "2-D parameter" in message
    assert "F.linear" in message
    assert "expert bank" in message  # says what it is *not*
    assert "--map-apply encode" in message  # and what does work

    # The module form of the same tensor keeps its own, different refusal.
    with pytest.raises(DynQuantError, match="owns a weight but is not a"):
        resolve_target(model, "block.gate")


def test_a_bank_is_indexed_one_expert_at_a_time() -> None:
    """Slicing would dequantize a range, which is what holding it packed avoids.

    Turns red when: ``__getitem__`` starts forwarding arbitrary indices to the
    dequantized tensor, at which point ``bank[:]`` silently materialises the bank.
    """
    model = _fresh()
    pack_model(model, {BANKS[0]: 4}, compute_device=None)
    bank = model.block.experts.gate_up_proj

    with pytest.raises(TypeError, match="one expert at a time"):
        _ = bank[0:2]
    with pytest.raises(IndexError, match="out of range"):
        _ = bank[EXPERTS]


def test_a_stack_deeper_than_a_bank_is_not_silently_flattened() -> None:
    """``[E, out, in]`` is the contract; rank 4 has no expert axis to index.

    Turns red when: the rank check is relaxed to ``>= 3``, which would make
    ``bank[e]`` return a slice whose shape the parent never asked for.
    """
    from dynquant.quant.device import quantize_tensor

    quantized, _ = quantize_tensor(torch.randn(2, 3, 8, 64).half(), bits=4, device=None)
    with pytest.raises(ValueError, match=r"\[experts, out, in\]"):
        DynQuantExpertBank(quantized)


# --------------------------------------------------------------------------
# The load path
# --------------------------------------------------------------------------


def _shelled(path):
    """A fresh skeleton prepared by the quantizer, exactly as ``from_pretrained`` does."""
    import json

    payload = json.loads((path / "config.json").read_text(encoding="utf-8"))
    config = hf_quantizer.build_config_class()(**payload["quantization_config"])
    quantizer = hf_quantizer.build_quantizer_class()(config)
    model = _fresh()
    quantizer._process_model_before_weight_loading(model)
    return model, quantizer


def test_an_exported_bank_loads_back_into_a_shell_of_the_right_geometry(tmp_path) -> None:
    """Export, then load through the quantizer's own preparation step.

    The shell is built from the *model's* shape, not the checkpoint's
    ``out_features`` -- which records the packer's flattened row count, ``E * out``,
    and cannot rebuild rank 3 on its own. So a shell built from the spec alone would
    be the right number of bytes and the wrong tensor, and ``load_state_dict`` would
    accept it.

    Turns red when: ``_shell`` starts trusting ``spec.out_features``, or the bank's
    buffer geometry stops matching what the encoder wrote.
    """
    from safetensors.torch import load_file

    source = _fresh()
    out = tmp_path / "ckpt"
    report = export_packed_checkpoint(source, _bits(), output_dir=out, compute_device=None)
    assert set(report.banks) == set(BANKS)

    model, _ = _shelled(out)
    for name in BANKS:
        assert isinstance(model.get_submodule(name), DynQuantExpertBank)

    state: dict[str, torch.Tensor] = {}
    for shard in sorted(out.glob("*.safetensors")):
        state.update(load_file(str(shard)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    # And the loaded thing decodes to what the encoder encoded, not merely to
    # something of the right shape.
    reference = _fresh()
    quantize_model(reference, _bits(), in_place=True, compute_device=None)
    encoder = _out(reference)
    loaded = _out(model)
    assert (loaded - encoder).abs().max().item() < 1e-3


def test_the_dense_key_a_bank_leaves_behind_is_the_one_that_is_dropped(tmp_path) -> None:
    """A bank *is* the parameter, so its missing key is ``name``, not ``name.weight``.

    Every other packed target contributed ``name.weight`` to the state dict, and a
    ``update_missing_keys`` written for those alone leaves one warning per bank -- 44
    of them on LFM2.5-8B-A1B, reading exactly like the silent-random-model failure the
    quantizer exists to prevent.

    Turns red when: the dense key is derived from the name instead of from what the
    target actually was.
    """
    source = _fresh()
    out = tmp_path / "ckpt"
    export_packed_checkpoint(source, _bits(), output_dir=out, compute_device=None)
    model, quantizer = _shelled(out)

    reported = [*BANKS, "proj.weight", f"{BANKS[0]}.weight", "block.gate.weight"]
    kept = quantizer.update_missing_keys(model, list(reported), "model")
    assert kept == [f"{BANKS[0]}.weight", "block.gate.weight"]

    prefixed = [f"model.{key}" for key in reported]
    assert quantizer.update_missing_keys(model, prefixed, "model") == [
        f"model.{BANKS[0]}.weight",
        "model.block.gate.weight",
    ]


def test_the_load_path_and_the_packing_path_classify_a_bank_the_same_way() -> None:
    """One resolver, two callers -- the property the fifth copy of it violated.

    Turns red when: the load path grows its own idea of what a bank is, which is how
    every previous divergence in this project started.
    """
    model = _fresh()
    for name in BANKS:
        for source in ("bit map", "quantization_config"):
            target = resolve_target(model, name, source=source)
            assert isinstance(target, ExpertBank)
            assert target.name == name
            assert target.weight is model.get_parameter(name)
