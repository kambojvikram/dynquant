"""The packed modules and the surgery that installs them.

These run on the torch backend with no extension built, which is the point: the
dispatch layer is what makes ``DYNQUANT_BACKEND=torch`` a real comparison rather
than a different code path, so the module semantics have to be established
somewhere that does not need a GPU. ``tests/test_kernels_parity.py`` then
establishes that the compiled path computes the same thing.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from dynquant.errors import DynQuantError
from dynquant.quant.grid import quantize_with_search
from dynquant.runtime import ops
from dynquant.runtime.backend import Backend
from dynquant.runtime.linear import (
    DynQuantEmbedding,
    DynQuantLinear,
    pack_model,
    packed_bytes,
)


class TiedTiny(nn.Module):
    """The shape that matters: an embedding whose table is also the output head.

    On Qwen3.5-2B this pair is 508.6M of 1881.3M parameters -- 27% -- so whether
    the two share one packed table or hold two is the difference between the
    memory claim being true and being off by a quarter.
    """

    def __init__(self, vocab: int = 256, dim: int = 128) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.mid = nn.Linear(dim, dim, bias=True)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.mid(self.emb(ids)))


@pytest.fixture
def model() -> TiedTiny:
    torch.manual_seed(0)
    return TiedTiny()


# --------------------------------------------------------------------------
# The modules
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_packed_linear_computes_what_the_dense_reconstruction_would(bits):
    """The module must not introduce error of its own.

    Quantization error is expected and measured elsewhere. What is asserted here is
    that running the packed weight gives the *same* answer as running the
    reconstruction of it -- so any accuracy number measured with simulated
    quantization transfers to the packed model unchanged, which is the assumption
    every earlier eval in this project rests on.
    """
    torch.manual_seed(bits)
    dense = nn.Linear(256, 64, bias=True)
    packed = DynQuantLinear.from_linear(dense, bits)

    x = torch.randn(4, 256)
    reference = torch.nn.functional.linear(
        x, packed.weight_qt.dequantize(dtype=x.dtype), dense.bias
    )
    torch.testing.assert_close(packed(x), reference, rtol=1e-5, atol=1e-5)


def test_packed_linear_accepts_arbitrary_leading_dimensions():
    """``[batch, seq, hidden]`` is what a transformer actually passes."""
    dense = nn.Linear(128, 32, bias=False)
    packed = DynQuantLinear.from_linear(dense, 4)
    x = torch.randn(2, 5, 128)
    assert packed(x).shape == (2, 5, 32)
    # And the flattening is not doing anything to the values.
    flat = packed(x.reshape(-1, 128)).reshape(2, 5, 32)
    torch.testing.assert_close(packed(x), flat, rtol=0, atol=0)


def test_packed_embedding_gathers_the_same_rows():
    torch.manual_seed(1)
    dense = nn.Embedding(512, 64)
    packed = DynQuantEmbedding.from_embedding(dense, 4)
    ids = torch.randint(0, 512, (3, 7))
    reference = packed.weight_qt.dequantize()[ids.reshape(-1)].reshape(3, 7, 64)
    torch.testing.assert_close(packed(ids), reference, rtol=0, atol=0)


def test_packed_modules_have_no_dense_weight_attribute():
    """A ``weight`` on a packed module would be an attractive nuisance: code that
    reaches for it -- and transformers has plenty -- would either crash or, worse,
    get a freshly materialised fp16 copy and quietly undo the compression."""
    packed = DynQuantLinear.from_linear(nn.Linear(64, 32), 4)
    assert not hasattr(packed, "weight")


def test_moving_a_packed_module_does_not_dequantize():
    packed = DynQuantLinear.from_linear(nn.Linear(64, 32), 3)
    before = packed.get_buffer("qweight").dtype
    packed = packed.to(torch.device("cpu"))
    assert packed.get_buffer("qweight").dtype is before is torch.int32


def test_state_dict_keys_match_the_checkpoint_format():
    """A live model's ``state_dict`` and a DynQuant checkpoint use the same names,
    so a checkpoint can be loaded into a packed model with ``load_state_dict`` and
    no key translation."""
    packed = DynQuantLinear.from_linear(nn.Linear(64, 32, bias=True), 4)
    keys = set(packed.state_dict())
    assert keys == {"qweight", "scales", "offsets", "bias"}


# --------------------------------------------------------------------------
# Surgery
# --------------------------------------------------------------------------


def test_pack_model_replaces_named_modules(model):
    report = pack_model(model, {"mid": 4})
    assert isinstance(model.mid, DynQuantLinear)
    assert isinstance(model.emb, nn.Embedding)  # untouched
    assert set(report.modules) == {"mid"}
    assert report.packed_bytes < report.fp16_bytes


def test_pack_model_follows_tied_weights_without_being_told(model):
    """The bit map lists one representative per tied group. If the surgery packed
    only that one, the ``lm_head`` of every tied model would keep a dense parameter
    alive and the memory measurement would be wrong by the size of the vocabulary
    projection -- with nothing raising."""
    report = pack_model(model, {"emb": 4})

    assert isinstance(model.emb, DynQuantEmbedding)
    assert isinstance(model.head, DynQuantLinear)
    assert report.tied == {"head": "emb"}
    assert model.emb.weight_qt.packed.data_ptr() == model.head.weight_qt.packed.data_ptr(), (
        "the tied pair must share one packed table, not hold two copies of it"
    )


def test_only_one_of_a_tied_pair_registers_the_table(model):
    """Sharing has to be structural, not two registrations of one tensor.

    ``nn.Module._apply`` -- which is all ``model.to(device)`` is -- calls its
    conversion function once per registered buffer and memoizes nothing. Two buffers
    that happen to be the same tensor therefore come out as two tensors, and the
    saving disappears exactly when the model reaches the GPU, where it was supposed
    to be realised. So the tied module registers no table at all: there is one
    tensor because there is one buffer.

    Asserted through ``state_dict`` because that is also the checkpoint claim -- a
    tied table is written once, the way ``transformers`` writes tied weights.
    """
    pack_model(model, {"emb": 4})

    assert set(model.emb.state_dict()) == {"qweight", "scales", "offsets"}
    # Nothing: the table is the embedding's and this head has no bias of its own.
    assert set(model.head.state_dict()) == set()
    assert model.head.holder is model.emb


def test_moving_a_packed_model_keeps_the_tied_table_shared(model):
    """The regression this design exists to prevent, on the operation that used to
    break it. ``.to()`` is a device change here only in name -- CPU to CPU with a
    dtype conversion exercises the same ``_apply`` path that a real ``.cuda()``
    does, and it is the path that used to double 27% of a Qwen3.5-2B."""
    pack_model(model, {"emb": 4, "mid": 3})
    moved = model.to(torch.float64)

    assert moved.emb.weight_qt.packed.data_ptr() == moved.head.weight_qt.packed.data_ptr()
    assert moved.emb.weight_qt.scales.dtype is torch.float64
    # The packed words are integers and must not have been swept along into the
    # float conversion; `_apply` skips non-floating buffers, and if it ever stopped
    # doing so the table would silently become 8 bytes per word.
    assert moved.emb.weight_qt.packed.dtype is torch.int32


def test_packed_bytes_counts_a_tied_table_once(model):
    pack_model(model, {"emb": 4, "mid": 3})
    accounting = packed_bytes(model)
    assert accounting["dense_bytes"] == 0
    assert accounting["dense_modules"] == []
    expected = model.emb.nbytes + model.mid.nbytes
    assert accounting["packed_bytes"] == expected


def test_a_packed_model_still_runs(model):
    ids = torch.randint(0, 256, (2, 6))
    before = model(ids)
    pack_model(model, {"emb": 8, "mid": 8})
    after = model(ids)
    assert after.shape == before.shape
    assert torch.isfinite(after).all()
    # 8-bit end to end should track the fp32 model closely; this is a smoke test of
    # the wiring, not of quantization quality.
    assert (after - before).abs().max() < 0.5 * before.abs().max()


def test_pack_model_raises_on_a_name_the_model_does_not_have(model):
    with pytest.raises(DynQuantError, match="not a module"):
        pack_model(model, {"decoder.layers.0.attn.q_proj": 4})


def test_pack_model_raises_rather_than_skipping_an_unsupported_module(model):
    # The empty name resolves to the root module, which owns no weight of its own --
    # the container case, distinct from a router (a weight with no Linear forward) and
    # from a batched bank (a weight with no module at all). All three are refused; only
    # this one has no encoder route to offer instead.
    with pytest.raises(DynQuantError, match="owns no weight") as caught:
        pack_model(model, {"": 4})
    assert "encode" not in str(caught.value)


# --------------------------------------------------------------------------
# Dispatch, and when it is allowed to happen
# --------------------------------------------------------------------------


@pytest.fixture
def cold_dispatch(monkeypatch):
    """A process that has not yet resolved its backend, with the probe counted.

    ``monkeypatch.setattr`` restores both globals afterwards, so a test that
    leaves the dispatch layer cold cannot slow down or mislead the next one.
    """
    calls = []

    def counted(*args, **kwargs):
        calls.append(args)
        return Backend.TORCH

    monkeypatch.setattr(ops, "_ACTIVE_BACKEND", None)
    monkeypatch.setattr(ops, "_GEMV_MAX_ROWS", None)
    monkeypatch.setattr(ops, "resolve_backend", counted)
    return calls


def test_the_backend_is_probed_once_and_then_never_again(cold_dispatch):
    """The property `torch.compile` needs: after warming, these read a constant.

    Dynamo traces through a cache decorator rather than honouring it, so a probe
    that is still *reachable* on the first call is a probe dynamo will try to
    trace -- and `importlib.util.find_spec`, which resolving a backend does, is
    not traceable. vLLM compiles with `fullgraph`, so that is a hard failure at
    the first quantized linear rather than a slow path.

    Counting the calls rather than asserting the return value is deliberate: the
    return value would still be right if the probe ran on every forward.
    """
    ops.warm_dispatch()
    assert len(cold_dispatch) == 1

    for _ in range(5):
        ops.active_backend()
        ops.gemv_max_rows()
        ops.uses_compiled_kernels(torch.device("cpu"))
    assert len(cold_dispatch) == 1


def test_building_a_packed_module_warms_the_dispatch_layer(cold_dispatch):
    """Construction is the last point at which the probe is an ordinary call.

    vLLM builds every layer before it traces anything, so warming here is what
    makes the constant above already be a constant by the time compilation
    starts. Asserting on construction rather than on the plugin keeps the
    guarantee testable without vLLM installed -- `vllm_plugin.linear` calls the
    same function from its `create_weights`.
    """
    quantized, _ = quantize_with_search(
        torch.randn(32, 128), bits=4, group_size=128, compute_dtype=torch.float16
    )
    DynQuantLinear(quantized, bias=None)
    assert len(cold_dispatch) == 1


# --------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("bits", "ceiling"), [(2, 2.3), (3, 3.3), (4, 4.3), (8, 8.3)])
def test_effective_bits_include_the_metadata(bits, ceiling):
    """The stored cost is the packed payload *plus* scales and offsets. At group
    128 with fp16 metadata that is an extra 0.25 bits/weight, which is why this
    project reports 3.25 rather than 3.00 for a 3-bit map."""
    torch.manual_seed(bits)
    quantized, _ = quantize_with_search(
        torch.randn(64, 512), bits=bits, group_size=128, compute_dtype=torch.float16
    )
    assert bits < quantized.bits_per_weight <= ceiling


def test_an_embedding_lookup_never_materialises_the_whole_table():
    """The gather is the point, not an optimisation detail worth leaving unpinned.

    ``embedding_lookup`` used to dequantize the table and index the result, which
    is correct and allocates an fp32 intermediate the size of the table on every
    forward -- 1 GiB on Mistral's 32k x 4096 table, 5 GiB on a 248k x 5120 one. A
    model whose packed weights fit in VRAM could still fail to decode one token,
    and the failure looked like the weights being too big rather than like a
    lookup being written the expensive way. Counting rows is the observation that
    distinguishes the two implementations; ``test_packed_embedding_gathers_the_
    same_rows`` above already establishes they agree on the values.
    """
    from dynquant.quant.tensor import QuantTensor

    dense = nn.Embedding(512, 64)
    packed = DynQuantEmbedding.from_embedding(dense, 4)
    ids = torch.randint(0, 512, (3, 7))

    seen: list[int] = []
    original = QuantTensor.dequantize

    def counting(self, **kwargs):
        seen.append(self.num_rows)
        return original(self, **kwargs)

    QuantTensor.dequantize = counting  # type: ignore[method-assign]
    try:
        out = packed(ids)
    finally:
        QuantTensor.dequantize = original  # type: ignore[method-assign]

    assert out.shape == (3, 7, 64)
    assert seen == [21], f"dequantized {seen} rows for 21 ids out of a 512-row table"


def test_select_rows_is_a_gather_of_the_packed_rows():
    """``select_rows`` against the table-wide route, including duplicate ids.

    A gather that dropped duplicates or reordered them would still pass a test
    written on a permutation, so the index here repeats and is out of order.
    """
    torch.manual_seed(3)
    table = DynQuantEmbedding.from_embedding(nn.Embedding(64, 32), 3).weight_qt
    index = torch.tensor([7, 7, 0, 63, 31, 7])

    gathered = table.select_rows(index)
    assert gathered.logical_shape == (6, 32)
    assert gathered.row_offset == 0
    torch.testing.assert_close(gathered.dequantize(), table.dequantize()[index], rtol=0, atol=0)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_a_band_reconstructs_exactly_the_rows_it_covers(bits):
    """The invariant banding actually has: the weights are bit-identical.

    Asserted on the reconstruction rather than on the matmul because the matmul is
    not bit-identical and should not be claimed to be -- ``F.linear`` picks its
    kernel by shape, so a 40-column GEMM and forty 1-column GEMVs accumulate in
    different orders and land a few fp32 ULPs apart. What must not move is which
    weight a band reconstructs, and that is exact.
    """
    torch.manual_seed(bits)
    qt = DynQuantLinear.from_linear(nn.Linear(96, 40), bits).weight_qt
    whole = qt.dequantize()

    rebuilt = torch.cat([qt.rows(s, min(s + 7, 40)).dequantize() for s in range(0, 40, 7)])
    torch.testing.assert_close(rebuilt, whole, rtol=0, atol=0)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_banded_and_whole_tensor_matmuls_agree(bits, monkeypatch):
    """And the product agrees to fp32, which is the claim the runtime makes."""
    from dynquant.runtime import ops as _ops

    torch.manual_seed(bits)
    packed = DynQuantLinear.from_linear(nn.Linear(96, 40), bits)
    x = torch.randn(5, 96)

    whole = packed(x)
    # 96 elements a band -> one output row at a time, the most adversarial split.
    monkeypatch.setattr(_ops, "TORCH_DEQUANT_BAND_ELEMENTS", 96)
    torch.testing.assert_close(packed(x), whole, rtol=1e-5, atol=1e-6)


def test_banding_bounds_the_reconstruction_peak():
    """The point of the band is the peak, so the peak is what is asserted.

    ``down_proj`` on a 7B is 4096 x 14336; reconstructing it whole to multiply by
    it wants 448 MiB of fp32 intermediates for a single token, which is how a
    model whose packed weights fit in VRAM fails to decode. Counting the rows each
    reconstruction covers is that quantity, measured without needing the GPU.
    """
    from dynquant.quant.tensor import QuantTensor
    from dynquant.runtime import ops as _ops

    packed = DynQuantLinear.from_linear(nn.Linear(128, 64), 4)
    seen: list[int] = []
    original = QuantTensor.dequantize

    def counting(self, **kwargs):
        seen.append(self.num_rows)
        return original(self, **kwargs)

    band = _ops.TORCH_DEQUANT_BAND_ELEMENTS
    QuantTensor.dequantize = counting  # type: ignore[method-assign]
    _ops.TORCH_DEQUANT_BAND_ELEMENTS = 128 * 8
    try:
        packed(torch.randn(2, 128))
    finally:
        QuantTensor.dequantize = original  # type: ignore[method-assign]
        _ops.TORCH_DEQUANT_BAND_ELEMENTS = band

    assert seen == [8] * 8, f"expected eight 8-row bands over 64 rows, got {seen}"
