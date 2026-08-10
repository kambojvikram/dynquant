"""Loading a packed checkpoint through ``from_pretrained``.

The property under test is not really "the model loads". It is that the only two
outcomes are a correct load and a loud refusal, because the third outcome -- what
happens with no quantizer registered -- is a randomly initialised model returned
successfully. See :mod:`dynquant.integration.hf_quantizer` for that measurement.
"""

from __future__ import annotations

import json
import warnings

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from torch import nn  # noqa: E402
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM  # noqa: E402

from dynquant.commands._shared import _dtype_kwarg  # noqa: E402
from dynquant.constants import QUANT_TENSOR_SUFFIXES  # noqa: E402
from dynquant.errors import DynQuantError  # noqa: E402
from dynquant.integration import hf_quantizer  # noqa: E402
from dynquant.quant.checkpoint import export_packed_checkpoint  # noqa: E402
from dynquant.quant.quantizer import quantize_model  # noqa: E402
from dynquant.runtime.linear import (  # noqa: E402
    DynQuantEmbedding,
    DynQuantLinear,
    ExpertBank,
    RestoredWeight,
    packed_bytes,
    resolve_target,
)

IDS = [[1, 5, 9, 33, 7, 2, 11, 4]]
PACKED = QUANT_TENSOR_SUFFIXES["packed"]


def _config(*, tied: bool) -> LlamaConfig:
    return LlamaConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=512,
        tie_word_embeddings=tied,
    )


def _bit_map() -> dict[str, int]:
    # Four widths, not one. A loader that reads the file-level default instead of the
    # per-module entry -- the shape of bug 9 in the legacy audit -- passes a
    # single-width map and fails here.
    widths = {
        "self_attn.q_proj": 4,
        "self_attn.k_proj": 4,
        "self_attn.v_proj": 8,
        "self_attn.o_proj": 4,
        "mlp.gate_proj": 4,
        "mlp.up_proj": 3,
        "mlp.down_proj": 2,
    }
    bits = {f"model.layers.{i}.{k}": v for i in range(2) for k, v in widths.items()}
    bits["model.embed_tokens"] = 4
    return bits


def _fresh(cfg: LlamaConfig) -> LlamaForCausalLM:
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).half().eval()


def _exported(tmp_path, *, tied: bool):
    """A packed directory, the fp16 model it came from, the map, and the config."""
    cfg = _config(tied=tied)
    model = _fresh(cfg)
    bits = _bit_map()
    out = tmp_path / "ckpt"
    export_packed_checkpoint(model, bits, output_dir=out, compute_device=None)
    return out, model, bits, cfg


def _load(path):
    hf_quantizer.register_hf_quantizer()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = AutoModelForCausalLM.from_pretrained(
            str(path), **{_dtype_kwarg(transformers): torch.float16}
        )
    return loaded.eval()


def _logits(model) -> torch.Tensor:
    with torch.no_grad():
        return model(torch.tensor(IDS)).logits.float()


def _encoder_reference(cfg, bits) -> torch.Tensor:
    """The same widths applied in place -- what the checkpoint is supposed to mean."""
    model = _fresh(cfg)
    quantize_model(model, bits, in_place=True, compute_device=None)
    return _logits(model)


# --------------------------------------------------------------------------
# The load itself
# --------------------------------------------------------------------------


def test_a_loaded_checkpoint_decodes_what_the_encoder_encoded(tmp_path) -> None:
    """Round trip, checked against two references because either alone is passable.

    Matching the *encoder* says the packed buffers were filled with the right values
    and read back with the right geometry. Being far from *fp16* says quantization
    happened at all -- a loader that quietly kept dense weights would match fp16
    exactly and look perfect against no other yardstick, and a loader that swapped
    nothing at all is precisely the silent failure this module exists to prevent.

    The second assertion is a ratio rather than an absolute bound so it cannot pass by
    both numbers being large.

    Turns red when: a width is read from the file-level default instead of the module
    entry, offsets are dropped, or the shells are filled in the wrong order.
    """
    out, base, bits, cfg = _exported(tmp_path, tied=False)
    loaded = _load(out)

    got = _logits(loaded)
    to_encoder = (got - _encoder_reference(cfg, bits)).abs().max().item()
    to_dense = (got - _logits(base)).abs().max().item()

    assert to_encoder < 1e-2, to_encoder
    assert to_dense > 20 * to_encoder, (to_dense, to_encoder)


def test_every_module_the_config_names_comes_back_packed(tmp_path) -> None:
    """The count, and the absence of a dense remainder.

    ``packed_bytes`` walks the live tree rather than the config, so a module the swap
    missed appears in ``dense_modules`` instead of being excluded from the total.

    Turns red when: the swap silently skips a module -- which otherwise surfaces only
    as a model slightly larger than the manifest promised.
    """
    out, _base, bits, _cfg = _exported(tmp_path, tied=False)
    loaded = _load(out)

    assert sorted(hf_quantizer.packed_module_names(loaded)) == sorted(bits)
    # `lm_head` is untied here, so it is not in the map and the checkpoint carries it
    # as an ordinary dense tensor. It is the only thing that may be left.
    assert packed_bytes(loaded)["dense_modules"] == ["lm_head"]
    assert hf_quantizer.dense_weight_bytes(loaded) == 512 * 128 * 2


def test_the_dense_weight_of_a_packed_module_is_not_reported_missing(tmp_path) -> None:
    """A quantized module's ``.weight`` is not missing; it is replaced.

    Left unfiltered, the loader emits one "newly initialized" line per quantized
    module -- textually identical to the silent-random-model failure. Someone reading
    that warning could not tell a working load from a broken one, so the warning has
    to be true.

    Driven through the quantizer directly rather than through ``from_pretrained``,
    because a version of transformers that logs nothing would make the same assertion
    on captured output pass while measuring nothing.

    Turns red when: the filter stops handling either the bare or the prefixed form of
    the key, which are both shapes transformers has passed here.
    """
    out, _base, bits, cfg = _exported(tmp_path, tied=False)
    block = json.loads((out / "config.json").read_bytes().decode("utf-8"))["quantization_config"]

    quantizer = hf_quantizer.build_quantizer_class()(hf_quantizer.build_config_class()(**block))
    skeleton = _fresh(cfg)
    quantizer._process_model_before_weight_loading(skeleton)

    dense = [f"{name}.weight" for name in bits]
    survivors = quantizer.update_missing_keys(
        skeleton, [*dense, "model.norm.weight", "lm_head.weight"], "model"
    )
    assert survivors == ["model.norm.weight", "lm_head.weight"]
    # The prefixed form too: transformers strips the base prefix on some paths and not
    # others, and a filter that handles one of them is half a filter.
    prefixed = [f"model.{key}" for key in dense]
    assert quantizer.update_missing_keys(skeleton, prefixed, "model") == []


def test_a_checkpoint_naming_a_module_this_model_lacks_raises_at_load(tmp_path) -> None:
    """The end-to-end guarantee, stated as the thing that used to not happen.

    Without a registered quantizer, transformers warns "Unknown quantization type ...
    we will skip the quantization" and returns a randomly initialised model. This
    asserts that an unloadable DynQuant checkpoint now stops instead of handing back
    noise -- which is what makes a batched-expert-bank checkpoint safe to publish,
    since the same path refuses that too.

    Turns red when: registration stops taking effect, or a refusal raised inside
    ``_process_model_before_weight_loading`` is caught and downgraded by the loader.
    """
    out, _base, _bits, _cfg = _exported(tmp_path, tied=False)
    path = out / "config.json"
    config = json.loads(path.read_bytes().decode("utf-8"))
    config["quantization_config"]["modules"]["model.layers.0.mlp.not_a_proj"] = {"bits": 4}
    path.write_bytes(json.dumps(config).encode("utf-8"))

    with pytest.raises(DynQuantError, match="not a module of this model"):
        _load(out)


# --------------------------------------------------------------------------
# Tied embeddings
# --------------------------------------------------------------------------


def test_a_tied_head_reads_the_input_table_instead_of_a_second_copy(tmp_path) -> None:
    """One table, two readers -- and the load used to die rather than produce it.

    ``from_pretrained`` calls ``model.tie_weights()`` after loading, which does
    ``output.weight = input.weight``; a packed embedding has no ``.weight``, so a tied
    model raised ``AttributeError: 'DynQuantEmbedding' object has no attribute
    'weight'`` from inside transformers. LFM2.5-8B-A1B and Qwen3.5-2B are both tied,
    so that was every tied-head model in the campaign.

    Checked at the buffer and not at the type: a head that is a ``DynQuantLinear``
    holding its *own* copy of the table would satisfy a type assertion and still
    double the 27% of a Qwen3.5-2B its embedding accounts for.

    Turns red when: the head is given its own registration, or transformers stops
    calling ``tie_weights`` and the neutralisation is quietly doing nothing.
    """
    out, _base, _bits, _cfg = _exported(tmp_path, tied=True)
    loaded = _load(out)

    head, table = loaded.lm_head, loaded.model.embed_tokens
    assert isinstance(head, DynQuantLinear)
    assert isinstance(table, DynQuantEmbedding)
    assert head.holder is table
    assert head.weight_qt.packed.data_ptr() == table.weight_qt.packed.data_ptr()
    assert PACKED not in dict(head.named_buffers(recurse=False))
    # Written once, so read once: a second key here is a second table on disk.
    state = loaded.state_dict()
    assert f"model.embed_tokens.{PACKED}" in state
    assert f"lm_head.{PACKED}" not in state


def test_a_tied_model_leaves_nothing_dense(tmp_path) -> None:
    """With the head tied and the map covering the rest, nothing is left unpacked.

    Zero is the only passing value: any dense Linear or Embedding still standing is a
    tensor the manifest already counted as compressed.
    """
    out, _base, _bits, _cfg = _exported(tmp_path, tied=True)
    loaded = _load(out)

    accounting = packed_bytes(loaded)
    assert accounting["dense_bytes"] == 0, accounting["dense_modules"]
    assert hf_quantizer.dense_weight_bytes(loaded) == 0
    # And the shared table is counted once, not once per reader.
    unique = {b.data_ptr(): b.numel() * b.element_size() for b in loaded.buffers()}
    assert accounting["packed_bytes"] <= sum(unique.values())


def test_a_tied_model_decodes_correctly_and_not_merely_without_error(tmp_path) -> None:
    """Structure is not correctness: the shared table must also be read correctly.

    A head reading the right buffer with the wrong geometry satisfies every identity
    assertion above and produces nonsense.
    """
    out, base, bits, cfg = _exported(tmp_path, tied=True)
    loaded = _load(out)

    got = _logits(loaded)
    to_encoder = (got - _encoder_reference(cfg, bits)).abs().max().item()
    to_dense = (got - _logits(base)).abs().max().item()

    assert to_encoder < 1e-2, to_encoder
    assert to_dense > 20 * to_encoder, (to_dense, to_encoder)


def test_no_tie_bookkeeping_survives_naming_a_parameter_the_packed_model_lacks(
    tmp_path,
) -> None:
    """The tie is structural after the swap, so the parameter-level record must go.

    ``tie_weights`` is not the only consumer of the tie, and treating it as though it
    were is what broke: transformers v5 also keeps ``all_tied_weights_keys`` and hands
    every name in it to ``get_parameter`` from ``mark_tied_weights_as_initialized`` --
    at the *end* of a load that had otherwise fully succeeded. The three tests above
    catch that on a v5 install and cannot catch it anywhere else, because
    ``from_pretrained`` only calls what the installed version has. This one states the
    invariant those three depend on, so it is checked wherever the suite runs: the
    mapping may name only tensors the model actually owns.

    The mapping is seeded rather than assumed, since transformers below v5 does not
    build one -- the seeded value is exactly what v5 computes for a tied Llama, and on
    v5 it overwrites that with itself.

    Turns red when: the pruning is dropped, or widened to empty the mapping outright
    (which would untie a model that ties something besides its head), or a future
    packed head starts registering a ``weight`` again.
    """
    out, _base, _bits, _cfg = _exported(tmp_path, tied=True)

    block = json.loads((out / "config.json").read_bytes().decode("utf-8"))
    quantizer = hf_quantizer.build_quantizer_class()(
        hf_quantizer.build_config_class()(**block["quantization_config"])
    )
    model = _fresh(_config(tied=True))
    # Parameters, not buffers: ``mark_tied_weights_as_initialized`` reaches the kept
    # entry through ``get_parameter``, which refuses a buffer. So a pair this prunes
    # *to* has to be the thing transformers expects to find there.
    for extra in ("pair_target", "pair_source", "orphan_source", "orphan_target"):
        model.register_parameter(extra, nn.Parameter(torch.zeros(2)))
    model.all_tied_weights_keys = {
        # The real entry: after the swap neither name is a parameter of anything.
        "lm_head.weight": "model.embed_tokens.weight",
        # Both sides resolve, so "prune what is absent" is distinguishable from
        # "empty the mapping" -- a model tying something besides its head keeps it.
        "pair_target": "pair_source",
        # One broken side each, because the crash is on the target and the re-tie is
        # on the source, so an entry is unusable if *either* name has gone.
        "gone_target": "orphan_source",
        "orphan_target": "gone_source",
    }

    quantizer._process_model_before_weight_loading(model)

    assert model.all_tied_weights_keys == {"pair_target": "pair_source"}
    owned = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    owned |= {name for name, _ in model.named_buffers(remove_duplicate=False)}
    assert not set(model.all_tied_weights_keys) - owned

    # And the v5 call site itself, where it exists: the assertion above is the reason
    # it no longer raises, not a proxy for it.
    mark = getattr(model, "mark_tied_weights_as_initialized", None)
    if mark is not None:
        mark(_LoadingInfo())


class _LoadingInfo:
    """The fields ``mark_tied_weights_as_initialized`` may touch, and nothing else."""

    def __init__(self) -> None:
        self.missing_keys: set[str] = set()
        self.unexpected_keys: set[str] = set()


# --------------------------------------------------------------------------
# What must refuse, and by whose rule
# --------------------------------------------------------------------------


class _Bank(nn.Module):
    """A batched expert bank: experts as one 3-D parameter, indexed by the parent."""

    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(4, 256, 128))


class _Router(nn.Module):
    """Shaped like ``Lfm2MoeTopKRouter``: a bare weight, ``F.linear``, no ``Linear``."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 128))


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = _Bank()
        self.gate = _Router()
        self.o_proj = nn.Linear(128, 128, bias=False)


def _refuse(name: str) -> str:
    with pytest.raises(DynQuantError) as caught:
        resolve_target(_Tiny(), name, source="quantization_config")
    return str(caught.value)


def test_the_load_path_refuses_by_the_packing_rule_and_not_a_copy_of_it() -> None:
    """All four outcomes, reached through the load path's own ``source``.

    This test exists because of a near miss: the load path started as a private
    three-branch reimplementation of ``resolve_target`` -- narrower by the container
    case, which is how each previous copy of this resolver in the project began, and
    each of those shipped a refusal naming the wrong reason. The duplicate is gone;
    this pins that it stays gone by asserting the *fourth* branch, the one a fresh
    copy omits.

    ``source`` is the only thing the two callers may differ in, so it is asserted too:
    a load-time refusal that blames a "bit map" sends someone looking for a file they
    do not have.

    Turns red when: a second resolver grows here, or any two branches collapse.
    """
    # The tensor branch resolves rather than refusing, and resolves to the same thing
    # the packing caller gets. A load path that re-derived this would be free to
    # disagree, which is the whole failure mode above.
    held = resolve_target(_Tiny(), "experts.gate_up_proj", source="quantization_config")
    assert isinstance(held, ExpertBank)

    bare = _refuse("gate.weight")
    assert "2-D parameter" in bare
    assert "encode" in bare
    assert "not a module of this model" not in bare

    router = _refuse("gate")
    assert "encode" in router
    assert "not a batched expert bank" in router

    absent = _refuse("experts.absent_proj")
    assert "not a module of this model" in absent
    assert "quantization_config names" in absent
    assert "bit map" not in absent

    # The branch a re-implementation forgets: a container owns nothing to pack, so
    # there is no encoder route to offer either.
    container = _refuse("experts")
    assert "owns no weight" in container
    assert "encode" not in container


# --------------------------------------------------------------------------
# The one target the two sides of the format disagreed about
# --------------------------------------------------------------------------


def test_a_router_is_refused_for_packing_and_restored_for_loading() -> None:
    """One resolver, two contracts, and the flag that names which one is asking.

    The disk contract is "a tensor exists to pack" -- ``dynquant export`` writes
    anything ``target_tensor`` resolves. The memory contract is "a forward exists to
    replace". An ``Lfm2MoeTopKRouter`` satisfies the first and not the second, and
    while both sides asked the same *question* they got the packer's answer, so the
    8B export wrote 22 routers into a checkpoint ``from_pretrained`` then refused to
    read: an artifact rejected by its own reader.

    ``restore`` is that difference made explicit rather than resolved by a second
    copy of the resolver -- the failure this whole section exists to prevent, and the
    fifth time in this project that two copies of one registry have agreed until
    they did not.

    Turns red when: ``restore`` starts rescuing a case that has no tensor at all
    (which would hand the loader a target it cannot fill), or the default stops
    refusing (which would let ``pack_model`` install a module where the parent
    indexes a parameter).
    """
    restored = resolve_target(_Tiny(), "gate", source="quantization_config", restore=True)
    assert isinstance(restored, RestoredWeight)
    assert restored.name == "gate"
    assert restored.weight.shape == (8, 128)

    # The default is still the packer's answer, so `pack_model` is unchanged.
    assert "not a batched expert bank" in _refuse("gate")

    # And `restore` is not a general amnesty. A container has no tensor to fill, so
    # rescuing it would produce a target the loader cannot load rather than one it
    # can, and a name this model lacks is wrong under either contract.
    for name, reason in (("experts", "owns no weight"), ("absent", "not a module of this model")):
        with pytest.raises(DynQuantError, match=reason):
            resolve_target(_Tiny(), name, source="quantization_config", restore=True)

    # The two cases that resolve resolve identically under both flags: ``restore``
    # changes one branch, and a flag that quietly changed the others would make the
    # loaded model a different object from the scored one.
    for name in ("experts.gate_up_proj", "o_proj"):
        kept = resolve_target(_Tiny(), name, source="quantization_config", restore=True)
        assert type(kept) is type(resolve_target(_Tiny(), name, source="quantization_config"))


class _Routed(nn.Module):
    """A model whose targets include the shape that has no forward to replace.

    Deliberately not a ``transformers`` architecture: no released model class has a
    router this test could reach through ``from_pretrained`` without a trust-remote-code
    download, and the property under test belongs to the loader rather than to any
    architecture. Driven through the quantizer's own hooks the way
    :mod:`tests.test_expert_bank` drives the bank path.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate = _Router()
        self.o_proj = nn.Linear(128, 128, bias=False)
        # Real, because `export_packed_checkpoint` writes config.json through
        # `config.save_pretrained` and reads the block back on the way in.
        self.config = _config(tied=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(x) @ self.gate.weight.t()


def _routed_bits() -> dict[str, int]:
    # Two widths, and the router is not given the Linear's: a restore path reading a
    # file-level default rather than the per-module entry would still produce a
    # plausible tensor of the right shape.
    return {"gate": 8, "o_proj": 4}


def _fresh_routed() -> _Routed:
    torch.manual_seed(0)
    return _Routed().eval()


def _routed_state(path) -> dict:
    from safetensors.torch import load_file

    state: dict[str, torch.Tensor] = {}
    for shard in sorted(path.glob("*.safetensors")):
        state.update(load_file(str(shard)))
    return state


def _routed_loaded(tmp_path):
    """Export a router-bearing model and bring it back through the quantizer's hooks."""
    out = tmp_path / "ckpt"
    export_packed_checkpoint(_fresh_routed(), _routed_bits(), output_dir=out, compute_device=None)

    block = json.loads((out / "config.json").read_bytes().decode("utf-8"))["quantization_config"]
    quantizer = hf_quantizer.build_quantizer_class()(hf_quantizer.build_config_class()(**block))
    model = _fresh_routed()
    quantizer._process_model_before_weight_loading(model)
    # `assign=True` because that is what the low-memory loader does, and it is the
    # harder case: it *replaces* every buffer object rather than filling it, so a
    # restore that closed over the tensors it registered before the load would
    # dequantize the empty ones and this would still be green under the default.
    missing, unexpected = model.load_state_dict(_routed_state(out), strict=False, assign=True)
    quantizer._process_model_after_weight_loading(model)
    return model, quantizer, out, (missing, unexpected)


def test_everything_the_exporter_writes_for_a_router_is_something_the_loader_reads(
    tmp_path,
) -> None:
    """The contract that was missing, stated as a set equality on tensor names.

    Every other test here exports and loads the same architecture, so the two sides
    of the format could only disagree about a target no test used -- which is exactly
    what happened, on the one target the exporter was deliberately widened to accept
    and the loader was not. This asserts the general form instead: the keys
    ``export_packed_checkpoint`` wrote are the keys the prepared skeleton offers, with
    nothing missing and nothing left over.

    Turns red when: either side of the format grows a tensor the other does not know
    about.
    """
    _model, _quantizer, out, (missing, unexpected) = _routed_loaded(tmp_path)
    assert not missing and not unexpected, (missing, unexpected)

    expected = {
        f"{module}.{suffix}"
        for module in _routed_bits()
        for suffix in (QUANT_TENSOR_SUFFIXES[kind] for kind in ("packed", "scale", "offset"))
    }
    assert set(_routed_state(out)) == expected


def test_a_restored_router_holds_what_the_encoder_would_have_written(tmp_path) -> None:
    """Dequantized once at load, to the encoder's numbers and not merely to numbers.

    Exact rather than within a tolerance: the packer and the encoder run the same
    arithmetic on the same weight, so any gap at all is the scale-dtype split of
    :mod:`tests.test_expert_bank` wearing a third face.

    Turns red when: the restore reads the pre-load buffers instead of the loaded ones
    -- which ``assign=True`` replaces rather than fills -- or the output dtype stops
    following the model it is loading into.
    """
    model, _quantizer, _out, _keys = _routed_loaded(tmp_path)

    reference = _fresh_routed()
    quantize_model(reference, _routed_bits(), in_place=True, compute_device=None)

    assert model.gate.weight.dtype == reference.gate.weight.dtype
    assert (model.gate.weight - reference.gate.weight).abs().max().item() == 0.0

    with torch.no_grad():
        x = torch.randn(2, 128)
        assert (model(x) - reference(x)).abs().max().item() < 1e-4
        # And moved from the unquantized weight, or the line above is satisfied by a
        # loader that quietly kept the dense tensor the skeleton started with.
        assert (model.gate.weight - _fresh_routed().gate.weight).abs().max().item() > 0


def test_a_restored_router_saves_back_packed_and_not_dense(tmp_path) -> None:
    """The dense weight is materialised in memory and must not reach the disk again.

    ``is_serializable`` claims a load-then-save round trip is the identity on the
    packed tensors. A router carrying its dequantized weight as an ordinary buffer
    writes a fourth tensor per router -- bytes the manifest never priced, in a
    checkpoint that then disagrees with its own ``quantization_config``. It is
    registered non-persistent for exactly that reason, which is a one-word property
    with no other symptom.

    Turns red when: either ``persistent=False`` is dropped, or the parameter is
    shadowed instead of deregistered -- a live ``_parameters["weight"]`` is reported
    missing at load, one warning per router, which reads like the silent-random-model
    failure.
    """
    model, _quantizer, out, _keys = _routed_loaded(tmp_path)

    state = model.state_dict()
    assert "gate.weight" not in state
    assert set(state) == set(_routed_state(out))
    assert "weight" not in model.gate._parameters


def test_a_restored_router_is_reported_in_both_totals_and_named_as_restored(tmp_path) -> None:
    """The one place in the format where both encodings are resident, so say so.

    ``packed_bytes`` walks the module tree, and a restored router is not a packed
    module, not a dense ``Linear`` and not a bare ``Parameter`` -- it is a module
    holding buffers, the one shape all three passes were blind to. Reporting nothing
    for it is the tied-embedding error running backwards: a denominator that omits
    what the loaded model actually costs.

    Turns red when: these are counted twice, or the module pass stops recognising
    them and the 2.9 MB on LFM2.5-8B-A1B goes back to being invisible.
    """
    model, _quantizer, _out, _keys = _routed_loaded(tmp_path)
    accounting = packed_bytes(model)

    dense = model.gate.weight
    assert accounting["dense_bytes"] == dense.numel() * dense.element_size()
    assert accounting["dense_modules"] == ["gate.weight (restored)"]

    buffers = dict(model.gate.named_buffers(recurse=False))
    router_packed = sum(
        buffers[name].numel() * buffers[name].element_size()
        for name in QUANT_TENSOR_SUFFIXES.values()
        if name in buffers
    )
    assert accounting["packed_bytes"] == router_packed + model.o_proj.nbytes
    assert accounting["total_bytes"] == accounting["packed_bytes"] + accounting["dense_bytes"]


# --------------------------------------------------------------------------
# Registration and the config object
# --------------------------------------------------------------------------


def test_registering_twice_is_a_no_op_rather_than_an_error() -> None:
    """A library may register defensively without knowing whether the app already did.

    transformers raises on a duplicate name, so the second call has to be absorbed.

    Turns red when: ``register_hf_quantizer`` starts writing unconditionally, which
    breaks any process where two callers both want to be sure.
    """
    hf_quantizer.register_hf_quantizer()
    assert hf_quantizer.register_hf_quantizer() is False

    from transformers.quantizers.auto import (
        AUTO_QUANTIZATION_CONFIG_MAPPING,
        AUTO_QUANTIZER_MAPPING,
    )

    assert "dynquant" in AUTO_QUANTIZER_MAPPING
    assert "dynquant" in AUTO_QUANTIZATION_CONFIG_MAPPING


def test_the_two_lines_in_the_model_card_are_the_two_lines_that_work() -> None:
    """``dynquant.register_hf_quantizer`` is the documented entry point, so it must exist.

    Loading a DynQuant checkpoint takes an explicit call -- transformers has no
    entry-point discovery for quantizers and ``import dynquant`` deliberately stays
    lazy -- which makes this name the difference between a correct model and a
    randomly initialised one. A rename that only updated the module would leave every
    published model card pointing at an ``AttributeError``.

    Turns red when: the top-level export is dropped, or drifts from the function the
    module actually defines.
    """
    import dynquant

    assert dynquant.register_hf_quantizer is hf_quantizer.register_hf_quantizer
    assert "register_hf_quantizer" in dynquant.__all__


def test_the_line_the_exporter_prints_is_the_line_that_works() -> None:
    """The model card is the only defence, and this is where its author gets the text.

    ``dynquant export`` is the last thing run before somebody writes a card, and the
    directory it just wrote loads into transformers only if the reader registers the
    quantizer first -- otherwise the quantization is skipped and a randomly
    initialised model comes back without an exception. The test above pins that the
    exported name is the real one; this pins that the exporter is telling people
    about it at all.

    Turns red when: the registration function is renamed without the message
    following, or the transformers half of the message is dropped and only
    ``vllm serve`` survives -- which is everything it said until banks became
    loadable and made the directory worth publishing.
    """
    from pathlib import Path

    from dynquant.commands.export import _how_to_load

    text = _how_to_load(Path("qwen3-dynquant-3bit"))

    assert f"dynquant.{hf_quantizer.register_hf_quantizer.__name__}()" in text
    assert "from_pretrained" in text
    assert "vllm serve qwen3-dynquant-3bit" in text
    assert "randomly initialised" in text


def test_the_config_object_defers_to_the_serving_schema(tmp_path) -> None:
    """One format, one parser -- the vLLM and SGLang plugins read the same object.

    Turns red when: a second interpretation of ``quantization_config`` grows here,
    which is how a checkpoint comes to load in one runtime and not another.
    """
    out, _base, bits, _cfg = _exported(tmp_path, tied=False)
    block = json.loads((out / "config.json").read_bytes().decode("utf-8"))["quantization_config"]
    config = hf_quantizer.build_config_class()(**block)

    assert set(config.schema.modules) == set(bits)
    assert config.schema.modules["model.layers.0.mlp.down_proj"].bits == 2
    assert config.to_dict()["quant_method"] == "dynquant"


def test_a_malformed_block_fails_while_the_traceback_points_at_the_config() -> None:
    """Parsed in ``__init__``, not lazily during the module swap.

    A ``modules`` map that is missing or empty means there is no allocation, and
    DynQuant has no single width to fall back on. Discovering that halfway through
    surgery on the module tree reports it as a swap failure instead.
    """
    with pytest.raises(DynQuantError, match="modules is missing or empty"):
        hf_quantizer.build_config_class()(quant_method="dynquant", modules={})
