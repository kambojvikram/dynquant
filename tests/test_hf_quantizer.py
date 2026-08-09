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
    bank = _refuse("experts.gate_up_proj")
    assert "batched expert bank" in bank
    assert "encode" in bank

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
