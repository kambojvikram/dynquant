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
from dynquant.runtime.experts import DISPATCH_NAME  # noqa: E402
from dynquant.runtime.linear import (  # noqa: E402
    DynQuantExpertBank,
    ExpertBank,
    pack_model,
    packed_bytes,
    resolve_target,
)

PACKED = QUANT_TENSOR_SUFFIXES["packed"]


def _served_dispatch() -> str:
    """Which dispatch a packed bank lands on *here*, read off transformers and not off us.

    There are two right answers and which one is right is a fact about the installed
    transformers, not about the packer: 5.14 has ``ALL_EXPERTS_FUNCTIONS`` to register
    ``dynquant`` into, and every release before it has nothing, where ``eager`` is the
    only indexing path there is. Pinning either literal makes this file pass on one line
    and fail on the other -- which is exactly what it did, green on 4.53.2 for weeks while
    the ``transformers 5.14.1`` job failed on ``assert 'dynquant' == 'eager'``.

    So the expectation is derived from the same import the fallback branches on, and what
    the tests below assert is the invariant that holds either way: the model left a
    dispatch that hands the bank to a grouped matmul, and arrived at one that indexes it.
    """
    try:
        from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS  # noqa: F401
    except ImportError:
        return "eager"
    return DISPATCH_NAME


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


# --------------------------------------------------------------------------
# Rank 3 is a shape, not a promise
# --------------------------------------------------------------------------


class _ShortConv(nn.Module):
    """``Lfm2MoeShortConv``, reduced to the part that matters: a real ``nn.Conv1d``."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(HIDDEN, HIDDEN, kernel_size=3, groups=HIDDEN, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.conv(hidden_states.transpose(1, 2)).transpose(1, 2)


class _NamedWeightExperts(nn.Module):
    """``JetMoeParallelExperts``: a genuine bank that happens to be called ``weight``."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(EXPERTS, HIDDEN, HIDDEN))

    def forward(self, hidden_states: torch.Tensor, expert: int) -> torch.Tensor:
        return F.linear(hidden_states, self.weight[expert])


def test_a_conv_kernel_is_rank_3_and_is_not_a_batched_expert_bank() -> None:
    """The defect the genuine ``Lfm2MoeExperts`` found that no fixture here could.

    Everything above was checked against banks, and rank 3 was the whole test for what
    a bank is. ``nn.Conv1d`` keeps its kernel as ``[out, in / groups, width]`` -- rank 3,
    named ``weight``, and on LFM2.5-8B-A1B eighteen of twenty-four layers are conv. So a
    map naming one under ``--map-apply pack`` got a :class:`DynQuantExpertBank` installed
    where ``F.conv1d`` expects a tensor, and died inside torch with a type error listing
    seven positional arguments rather than at the resolver with a reason.

    It is refused now, and refused for its own reason: a conv kernel is not indexed one
    expert at a time by anybody, so no grouped path will ever hold it, and the message
    says that rather than repeating the bank sentence.

    Turns red when: rank alone decides again -- and, in the other direction, when the
    discriminator is written against the *name*. That is the tempting rule and it is
    wrong: of the eight rank-3 ``weight``/``bias`` declarations in transformers 5.14.1,
    ``JetMoeParallelExperts`` reaches its through ``F.linear(x, self.weight[i])``. Both
    directions are asserted below, because a name rule passes the conv half alone.
    """

    class _Mixed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.short_conv = _ShortConv()
            self.block = _Block()
            self.jetmoe = _NamedWeightExperts()
            self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
            # A bank hung on a bare nn.Module, which is a namespace and not a layer.
            self.namespace = nn.Module()
            self.namespace.gate_up_proj = nn.Parameter(torch.empty(EXPERTS, HIDDEN, HIDDEN))

    torch.manual_seed(0)
    model = _Mixed().half().eval()

    kernel = model.short_conv.conv.weight
    assert kernel.ndim == 3, "the premise: a conv kernel and a bank share a rank"

    with pytest.raises(DynQuantError) as caught:
        resolve_target(model, "short_conv.conv.weight")
    refusal = str(caught.value)
    assert "Conv1d" in refusal
    assert "indexed by nothing" in refusal
    assert "encode" in refusal
    # Not the bank sentence, and not the wrong-map one.
    assert "indexed one expert at a time" not in refusal
    assert "not a module of this model" not in refusal

    # The owner is what decides, so a bank keeps resolving whatever it is called -- and
    # `nn.Module` is a namespace, not a layer, which the exporter's own fixture relies on.
    for name in ("block.experts.gate_up_proj", "jetmoe.weight", "namespace.gate_up_proj"):
        target = resolve_target(model, name)
        assert isinstance(target, ExpertBank), name
        assert target.weight is model.get_parameter(name)

    # A Linear is a torch.nn layer too, and its weight is 2-D. The conv sentence is
    # about a rank it does not have, so it keeps the message it always had.
    with pytest.raises(DynQuantError) as linear:
        resolve_target(model, "proj.weight")
    assert "bare 2-D parameter" in str(linear.value)
    assert "conv kernel" not in str(linear.value)


def test_packing_a_bank_beside_a_conv_leaves_the_conv_running() -> None:
    """The failure was a broken forward, so the fix is checked by running one.

    A refusal message is easy to assert and easy to get right while still installing the
    wrong object somewhere else in the same pass. This packs the bank of a model that
    also owns a conv and then calls it: if the resolver ever reaches past its map, or
    ``replace_module`` deregisters a parameter it should not, ``F.conv1d`` says so.

    Turns red when: packing touches anything the map did not name, or the conv kernel
    stops being a tensor by the time torch sees it.
    """

    class _ConvThenMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.short_conv = _ShortConv()
            self.block = _Block()
            self.config = LlamaConfig(hidden_size=HIDDEN, tie_word_embeddings=False)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            padded = F.pad(hidden_states.unsqueeze(0), (0, 0, 2, 0))
            return self.block(self.short_conv(padded).squeeze(0))

    torch.manual_seed(0)
    model = _ConvThenMoE()
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.05)
    model = model.half().eval()

    before = _out(model)
    pack_model(model, {BANKS[0]: 4, BANKS[1]: 3}, compute_device=None)

    assert isinstance(model.block.experts.gate_up_proj, DynQuantExpertBank)
    assert isinstance(model.short_conv.conv.weight, torch.nn.Parameter)

    after = _out(model)
    assert torch.isfinite(after).all()
    # Quantization moved the answer, and the conv still contributed to it: a conv whose
    # kernel had been replaced would not have produced a number at all.
    assert (after - before).abs().max() > 0.0


# --------------------------------------------------------------------------
# The dispatch that stopped calling the loop
# --------------------------------------------------------------------------


class _DispatchedExperts(_Experts):
    """``Lfm2MoeExperts`` as transformers 5.14.1 actually reaches it.

    Every ``*Experts`` class this design was measured against indexes its bank, and on
    5.14.1 that stopped being what runs. ``@use_experts_implementation`` replaces the
    class's ``forward`` with a dispatcher that reads ``config._experts_implementation``
    and picks from ``ALL_EXPERTS_FUNCTIONS``; the loop above is only the ``eager`` entry,
    and the default is ``grouped_mm``, which passes the bank whole.

    Reduced here to the one line that decides it: ``weight.transpose(-2, -1)``, which is
    where ``integrations/moe.py`` asked a :class:`DynQuantExpertBank` for a tensor method
    and got ``AttributeError`` out of ``nn.Module``.
    """

    def __init__(self, config: object) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # `dynquant` sits with `eager` and not with the rest: it is the real
        # `ALL_EXPERTS_FUNCTIONS` entry this package registers, and it reaches an expert
        # by `bank[e]` -- the same contract the loop above holds to, and the reason a
        # packed bank can stand where the parameter stood. What it does differently is
        # keep the grouped path's reduction order, which is arithmetic this fixture does
        # not model either way; see `tests/test_experts_dispatch.py` for that.
        if getattr(self.config, "_experts_implementation", "eager") in ("eager", DISPATCH_NAME):
            return super().forward(hidden_states, top_k_index, top_k_weights)
        return self._grouped(hidden_states, top_k_index, top_k_weights)

    def _grouped(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # One batched matmul over every expert at once -- the shape `torch._grouped_mm`
        # exists to serve, and the reason a module cannot stand where the bank stood.
        gate, up = torch.matmul(hidden_states, self.gate_up_proj.transpose(-2, -1)).chunk(2, dim=-1)
        dense = torch.matmul(F.silu(gate) * up, self.down_proj.transpose(-2, -1))
        mask = F.one_hot(top_k_index, num_classes=EXPERTS)
        weights = (mask * top_k_weights.unsqueeze(-1)).sum(dim=1)
        return (dense * weights.t().unsqueeze(-1)).sum(dim=0).to(hidden_states.dtype)


class _DispatchedModel(nn.Module):
    """``_MoEModel``'s parameters exactly, behind ``_experts_implementation``.

    The names match so an export written from one loads into the other, which is how
    the load path is checked below without a second checkpoint.
    """

    def __init__(self, implementation: str = "grouped_mm") -> None:
        super().__init__()
        self.config = LlamaConfig(hidden_size=HIDDEN, tie_word_embeddings=False)
        self.config._experts_implementation = implementation
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.block = nn.Module()
        self.block.gate = _Router()
        self.block.experts = _DispatchedExperts(self.config)
        self.moves: list[str] = []

    def set_experts_implementation(self, implementation: str) -> None:
        self.moves.append(implementation)
        self.config._experts_implementation = implementation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = self.proj(hidden_states)
        weights, index = self.block.gate(projected)
        return self.block.experts(projected, index, weights)


def _fresh_dispatched(implementation: str = "grouped_mm") -> _DispatchedModel:
    torch.manual_seed(0)
    model = _DispatchedModel(implementation)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.05)
    return model.half().eval()


def test_the_fixtures_two_dispatches_are_transcriptions_of_each_other() -> None:
    """Fixture fidelity, and nothing about what the two dispatches do on a real model.

    This test was called ``..._agree_so_moving_between_them_is_free`` and was read as the
    premise of the switch: that packing changes a model's answer once, by quantizing, and
    not twice. It cannot carry that claim. ``_DispatchedExperts._grouped`` is a
    hand-written dense-and-mask transcription that never calls ``torch._grouped_mm``, and
    the fixture is one MoE layer -- so the mechanism by which the two dispatches actually
    diverge is structurally absent from it. A top-k router turns a last-bit numeric
    difference into a different set of experts, and depth compounds it: on LFM2.5-8B-A1B
    the first MoE layer routes bit-identically and the last agrees on 7% of its slots,
    ending at 1.24% teacher-forced token disagreement, 0.29x the effect of quantizing that
    model. One layer has no second layer to compound into.

    So what is checked here is that the fixture's two paths are the same arithmetic, which
    is what makes every *other* assertion in this file about the switch non-vacuous. The
    real question is measured on the real model and lives in
    ``docs/reports/phase4-packed-moe-runtime.md`` section 8; the pin that keeps a panel
    from straddling it is ``_pin_experts_dispatch`` in ``commands/evaluate.py``.

    Turns red when: the fixture's grouped path stops transcribing its eager one.
    """
    grouped = _out(_fresh_dispatched("grouped_mm"))
    eager = _out(_fresh_dispatched("eager"))
    assert (grouped - eager).abs().max().item() < 1e-2, (grouped - eager).abs().max().item()


def test_packing_a_bank_moves_the_model_off_the_dispatch_that_cannot_hold_one() -> None:
    """A packed bank answers ``bank[e]`` and nothing else, so eager is not a preference.

    This is the second defect the genuine ``Lfm2MoeExperts`` found: the resolver was
    fixed, the bank installed cleanly, and the forward still died -- inside transformers,
    at ``weight.transpose(-2, -1)``, because 5.14.1 routes experts through a dispatcher
    whose default hands the whole bank to a grouped matmul.

    Turns red when: the switch stops being applied, or is applied by writing the config
    attribute directly while the model has a setter that does more than that.
    """
    model = _fresh_dispatched("grouped_mm")
    assert model.config._experts_implementation == "grouped_mm"

    report = pack_model(model, {BANKS[0]: 4, BANKS[1]: 3}, compute_device=None)

    assert isinstance(model.block.experts.gate_up_proj, DynQuantExpertBank)
    assert model.config._experts_implementation == _served_dispatch()
    assert model.moves == [_served_dispatch()], "the model's own setter is what should have run"
    assert report.experts_implementation == "grouped_mm"
    assert "grouped_mm" in report.summary() and _served_dispatch() in report.summary()

    # And the model runs, which is the whole claim -- previously an AttributeError.
    packed = _out(model)
    assert torch.isfinite(packed).all()

    reference = _fresh_dispatched("eager")
    quantize_model(reference, {BANKS[0]: 4, BANKS[1]: 3}, in_place=True, compute_device=None)
    assert (packed - _out(reference)).abs().max().item() < 1e-3


def test_nothing_moves_for_a_model_that_has_nowhere_to_move() -> None:
    """Three ways to have no dispatch, one answer, and no lie in the report.

    ``use_eager_experts`` is called on every pack that installs a bank, and most models
    it will ever see have no such config -- every transformers before 5.14, and every
    dense model after. Reporting a move that did not happen would put a sentence in the
    summary that no operator could act on.

    Turns red when: the helper starts inventing the attribute it is looking for, or the
    report records a move for a pack that installed no bank at all.
    """
    from dynquant.runtime.linear import use_eager_experts

    already = _fresh_dispatched("eager")
    assert use_eager_experts(already) is None
    assert already.moves == []

    # `_MoEModel`'s config is a plain LlamaConfig: no such attribute, nothing to do.
    plain = _fresh()
    assert use_eager_experts(plain) is None
    report = pack_model(plain, _bits(), compute_device=None)
    assert report.experts_implementation is None
    assert "dispatch" not in report.summary()

    # A pack that installs no bank must not touch a dispatch it never depended on.
    dense_only = _fresh_dispatched("grouped_mm")
    dense_report = pack_model(dense_only, {"proj": 4}, compute_device=None)
    assert dense_report.experts_implementation is None
    assert dense_only.config._experts_implementation == "grouped_mm"
    assert dense_only.moves == []


def test_the_load_path_moves_the_dispatch_too(tmp_path) -> None:
    """``from_pretrained`` installs banks without going anywhere near ``pack_model``.

    The two paths share ``resolve_target`` and ``_shell`` and nothing else, so a fix
    written into the packer alone leaves the published checkpoint -- the artifact anyone
    but us will ever touch -- raising ``AttributeError`` on its first forward.

    Turns red when: the load path grows its own idea of which dispatch a bank needs.
    """
    from safetensors.torch import load_file

    out = tmp_path / "ckpt"
    export_packed_checkpoint(_fresh(), _bits(), output_dir=out, compute_device=None)

    import json

    payload = json.loads((out / "config.json").read_text(encoding="utf-8"))
    config = hf_quantizer.build_config_class()(**payload["quantization_config"])
    quantizer = hf_quantizer.build_quantizer_class()(config)
    model = _fresh_dispatched("grouped_mm")
    quantizer._process_model_before_weight_loading(model)

    assert model.config._experts_implementation == _served_dispatch()
    assert model.moves == [_served_dispatch()]

    state: dict[str, torch.Tensor] = {}
    for shard in sorted(out.glob("*.safetensors")):
        state.update(load_file(str(shard)))
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    reference = _fresh()
    quantize_model(reference, _bits(), in_place=True, compute_device=None)
    assert (_out(model) - _out(reference)).abs().max().item() < 1e-3


# --------------------------------------------------------------------------
# The dtype the scales are stored in, and the dtype the bank hands back
# --------------------------------------------------------------------------


def _fresh32() -> _MoEModel:
    """``_fresh()`` without the ``.half()``. Everything below turns on that one call.

    Three copies of the scale-dtype rule agreed on fp16 and bf16, which is every model
    anyone ships, so the disagreement could only ever surface here.
    """
    torch.manual_seed(0)
    model = _MoEModel()
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.05)
    return model.eval()


def test_the_packer_and_the_encoder_encode_an_fp32_bank_identically() -> None:
    """Same weights, same width, two code paths -- and they used to differ by 0.0082.

    ``pack_model`` and ``quantize_model(in_place=True)`` are the artifact and its
    yardstick: every accuracy this project reports for a DynQuant arm is measured on the
    encoder, and every byte is measured on the packer. If the two encode differently then
    the panel's numbers describe a model nobody can download. They did, on fp32, by about
    one 4-bit step -- 16% of what quantizing the model moved at all -- because the packer
    stored fp32 scales and the encoder stored fp16 ones.

    Exact equality, not a tolerance: these are the same arithmetic on the same inputs, so
    any gap at all is two rules again.

    Turns red when: a fourth caller grows its own idea of the metadata dtype.
    """
    packed = _fresh32()
    pack_model(packed, _bits(), compute_device=None)

    encoded = _fresh32()
    quantize_model(encoded, _bits(), in_place=True, compute_device=None)

    for name in BANKS:
        bank = packed.get_submodule(name)
        dense = encoded.get_parameter(name)
        for expert in range(EXPERTS):
            gap = (bank[expert] - dense[expert]).abs().max().item()
            assert gap == 0.0, f"{name}[{expert}] differs by {gap}"


def test_an_fp32_bank_stores_16_bit_metadata_and_still_hands_back_fp32() -> None:
    """The two halves of the resolution, which pull in opposite directions.

    The budget prices metadata at 16 bits -- ``metadata_bits: int = 16``, "an fp16 scale
    and an fp16 offset per 128" -- and every average-bits figure in this project is
    computed against that constant, so fp32 scales would put a model 0.25 bits/weight
    above its own manifest at group 128. That settles the storage question.

    It does not settle what ``bank[e]`` returns. A packed Linear never has to decide:
    ``quantized_matmul`` dequantizes to ``x.dtype``, following the activation. A bank is
    asked for a weight before any activation is in sight, so it is told at construction
    what its parent computes in -- otherwise an fp32 model gets fp16 weights and
    ``F.linear`` raises on the mismatch.

    Turns red when: the metadata rule follows the weight again (the byte claim breaks), or
    the bank hands back its metadata dtype again (the fp32 forward breaks).
    """
    model = _fresh32()
    pack_model(model, _bits(), compute_device=None)

    bank = model.get_submodule(BANKS[0])
    assert bank.weight_qt.scales.dtype is torch.float16
    assert bank.weight_qt.offsets is not None and bank.weight_qt.offsets.dtype is torch.float16
    assert bank.out_dtype is torch.float32
    assert bank[0].dtype is torch.float32
    assert bank.dequantize().dtype is torch.float32

    # And the parent's own loop runs, which is what the mismatch would have stopped.
    with torch.no_grad():
        out = model(torch.randn(6, HIDDEN))
    assert out.dtype is torch.float32 and torch.isfinite(out).all()


def test_the_output_dtype_moves_with_the_model() -> None:
    """``.half()`` on the parent has to reach the bank, or the next forward raises.

    A caller who packs in fp32 and then casts the model down is doing something ordinary,
    and the bank's answer is not a constant chosen at construction -- it is a property of
    the live model. Holding it in a zero-element buffer is what makes ``nn.Module._apply``
    carry it, the same mechanism that carries the scales.

    Turns red when: the dtype goes back to being a plain Python attribute, which survives
    ``.to()`` unchanged and starts lying at the first cast.
    """
    model = _fresh32()
    pack_model(model, _bits(), compute_device=None)
    bank = model.get_submodule(BANKS[0])
    assert bank.out_dtype is torch.float32

    model.half()
    assert bank.out_dtype is torch.float16
    assert bank[0].dtype is torch.float16
    with torch.no_grad():
        assert torch.isfinite(model(_x())).all()

    model.float()
    assert bank.out_dtype is torch.float32
    assert bank[0].dtype is torch.float32

    # It is not in the checkpoint: a bank loaded into an fp32 model should serve fp32,
    # whatever the model it was written from computed in.
    assert not [key for key in model.state_dict() if key.endswith("_out_dtype")]


def test_an_fp32_bank_survives_the_round_trip_through_a_checkpoint(tmp_path) -> None:
    """Export and load are a third and fourth copy of the same decision.

    ``export_packed_checkpoint`` writes the scales and ``_shell`` builds the skeleton that
    receives them; the low-memory loader assigns rather than copies, so a skeleton in the
    wrong dtype is not corrected on the way in. Both ends call ``storage_dtype`` now, and
    this is the test that keeps them calling the same one.

    Turns red when: either end starts deriving the metadata dtype from the model instead.
    """
    from safetensors.torch import load_file

    out = tmp_path / "ckpt"
    export_packed_checkpoint(_fresh32(), _bits(), output_dir=out, compute_device=None)

    state: dict[str, torch.Tensor] = {}
    for shard in sorted(out.glob("*.safetensors")):
        state.update(load_file(str(shard)))
    scale_key = BANKS[0] + "." + QUANT_TENSOR_SUFFIXES["scale"]
    assert state[scale_key].dtype is torch.float16, "the exporter owes the budget 16 bits"

    import json

    payload = json.loads((out / "config.json").read_text(encoding="utf-8"))
    config = hf_quantizer.build_config_class()(**payload["quantization_config"])
    quantizer = hf_quantizer.build_quantizer_class()(config)
    loaded = _fresh32()
    quantizer._process_model_before_weight_loading(loaded)

    bank = loaded.get_submodule(BANKS[0])
    assert bank.weight_qt.scales.dtype is torch.float16
    assert bank.out_dtype is torch.float32

    missing, unexpected = loaded.load_state_dict(state, strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    reference = _fresh32()
    quantize_model(reference, _bits(), in_place=True, compute_device=None)
    with torch.no_grad():
        probe = torch.randn(6, HIDDEN)
        assert (loaded(probe) - reference(probe)).abs().max().item() == 0.0
