"""The parameter and linear port: where SGLang's loaders differ from vLLM's.

``test_vllm_tp_placement.py`` already asks the question this file's middle section
asks -- reassemble what the ranks hold, require it to equal the checkpoint -- and
the answer comes from the same framework-free
:mod:`dynquant.integration.serving_common.geometry`. Repeating it here is not
duplication: the *routes* into that arithmetic are different, and each of the four
differences below is a way the port could look right and place weights wrong.

1. ``tp_rank`` is an argument, not ``self.tp_rank``. vLLM's parameter base class
   caches the rank from the process group; SGLang's passes it into each hook,
   because its linear layers take an explicit ``tp_rank``/``tp_size`` pair.
2. ``use_presharded_weights`` -- a tensor that has already been sliced for this
   rank. Slicing it again loads a quarter of a rank's weights, silently.
3. ``skip_block_quant_check`` is passed to every column-parallel
   ``create_weights``. Swallowed by ``**extra_weight_attrs`` it would be stapled
   onto all three tensors.
4. The base classes decide the *call convention*. ``weight_loader_v2`` gates on
   ``isinstance``, and its fallback branch re-calls without the rank.

Runs against :mod:`_sglang_stub`, whose parameter classes are copied from 0.5.16
rather than sketched, precisely because ``DynQuantPackedParameter`` inherits from
them for real. :mod:`test_sglang_stub_conformance` is what keeps that copy honest.
"""

from __future__ import annotations

import pytest
import torch

from dynquant.errors import DynQuantError
from dynquant.integration.serving_common.schema import ModuleQuantSpec
from dynquant.quant.pack import row_geometry

from _sglang_stub import fake_sglang

TP = 2
IN_FEATURES = 2048
GROUP_SIZE = 128


@pytest.fixture
def sglang():
    """The stub tree, yielding the plugin's linear module built against it."""
    with fake_sglang():
        import dynquant.integration.sglang_plugin.linear as linear

        yield linear


def spec(bits: int, out_features: int, *, group_size: int = GROUP_SIZE) -> ModuleQuantSpec:
    return ModuleQuantSpec(
        bits=bits, group_size=group_size, symmetric=False, out_features=out_features
    )


def checkpoint_tensors(bits: int, out_features: int, in_features: int = IN_FEATURES):
    """One module's packed words and scales, spanning every rank.

    Values are an ascending ramp rather than a real quantization: this asks where
    bytes land, and a ramp makes a misplacement a mismatch at a nameable index
    instead of two plausible-looking tensors.
    """
    geom = row_geometry(bits, GROUP_SIZE, in_features)
    words = torch.arange(out_features * geom.words_per_row, dtype=torch.int32)
    scales = torch.arange(out_features * geom.num_groups, dtype=torch.bfloat16)
    return words.view(out_features, geom.words_per_row), scales.view(out_features, geom.num_groups)


def build(module, shards, output_partition_sizes, *, in_per_partition=IN_FEATURES, **extra):
    layer = torch.nn.Module()
    method = module.DynQuantLinearMethod(quant_config=None, shards=shards)
    method.create_weights(
        layer,
        input_size_per_partition=in_per_partition,
        output_partition_sizes=list(output_partition_sizes),
        input_size=IN_FEATURES,
        output_size=sum(output_partition_sizes),
        params_dtype=torch.bfloat16,
        weight_loader=None,
        **extra,
    )
    return layer


# --------------------------------------------------------------------------
# 1. The base classes, which decide how the hooks get called
# --------------------------------------------------------------------------


def test_the_parameter_passes_both_isinstance_gates(sglang):
    """``ColumnParallelLinear`` tests one, ``RowParallelLinear`` the other.

    ``weight_loader_v2`` branches on ``isinstance(param, _ColumnvLLMParameter)``
    (``linear.py:449``) and ``isinstance(param, RowvLLMParameter)`` (``:1541``).
    Failing either lands in an ``except TypeError`` fallback that re-calls the
    hook *without* ``tp_rank`` -- which is how a parameter ends up holding rank
    0's rows on every rank. Inheriting from both, in that order, is what SGLang's
    own ``ModelWeightParameter`` does.
    """
    from sglang.srt.layers.parameter import RowvLLMParameter, _ColumnvLLMParameter

    layer = build(sglang, [("l.q_proj", spec(4, 1024))], [1024])
    for name in ("qweight", "scales", "offsets"):
        param = getattr(layer, name)
        assert isinstance(param, _ColumnvLLMParameter), name
        assert isinstance(param, RowvLLMParameter), name


def test_the_hooks_refuse_a_call_that_omits_the_rank(sglang):
    """The other half of the gate above, and the reason ``tp_rank`` has no default.

    If the ``isinstance`` test above ever stopped holding, SGLang would try the
    hook with a rank and, on ``TypeError``, retry without one. A signature with a
    defaulted ``tp_rank`` would accept that retry and place rank 0's rows
    everywhere. A required one turns the fallback back into a crash, which is the
    outcome worth having.
    """
    words, _ = checkpoint_tensors(4, 1024 * TP)
    layer = build(sglang, [("l.q_proj", spec(4, 1024))], [1024])

    with pytest.raises(TypeError):
        layer.qweight.load_column_parallel_weight(words)
    with pytest.raises(TypeError):
        layer.qweight.load_row_parallel_weight(words)


def test_the_dims_describe_the_checkpoint_tensor_and_not_the_buffer(sglang):
    """``output_dim``/``input_dim`` are inherited, and mean the loaded tensor.

    Nothing indexes *our* parameter by them -- it is flat. They exist because the
    base classes' properties are what ``_load_fused_module_from_checkpoint``
    (``linear.py:739``, ``:1035``) narrows a fused-on-disk checkpoint with, and
    because SGLang's loaders read them off the parameter rather than the layer.

    ``packed_dim`` stays absent: setting it would make the loaders divide row
    offsets by a packing factor, which is right when packing runs along the output
    dimension and wrong here, where it runs along the input dimension.
    """
    layer = build(sglang, [("l.q_proj", spec(4, 1024))], [1024])
    for name in ("qweight", "scales", "offsets"):
        param = getattr(layer, name)
        assert param.output_dim == 0
        assert param.input_dim == 1
        assert param.data.dim() == 1, "the buffer is flat; the dims describe the checkpoint"
        assert not hasattr(param, "packed_dim"), name


# --------------------------------------------------------------------------
# 2. Placement, with the rank arriving as an argument
# --------------------------------------------------------------------------


def test_column_parallel_ranks_tile_the_checkpoint_rows(sglang):
    rows = 1024
    words, scales = checkpoint_tensors(4, rows * TP)
    shards = [("l.q_proj", spec(4, rows))]

    for rank in range(TP):
        layer = build(sglang, shards, [rows])
        layer.qweight.load_column_parallel_weight(words, tp_rank=rank)
        layer.scales.load_column_parallel_weight(scales, tp_rank=rank)
        geometry = layer.dynquant_geometry

        assert torch.equal(
            geometry.view_qweight(layer.qweight.data, 0), words[rank * rows : (rank + 1) * rows]
        )
        assert torch.equal(
            geometry.view_scale(layer.scales.data, 0), scales[rank * rows : (rank + 1) * rows]
        )


def test_qkv_ranks_tile_the_checkpoint_rows(sglang):
    """Three modules of three different widths, split across two ranks.

    Different bits per shard is the case no other quantization method has: with
    one width the flat buffer is a rectangle and SGLang's own ``narrow`` would do.
    Here q, k and v have 256-, 192- and 512-word rows over the same input, so
    every offset in the placement is shard-dependent.
    """
    widths = {"q": (4, 2048), "k": (3, 512), "v": (8, 512)}
    full = {name: checkpoint_tensors(bits, rows) for name, (bits, rows) in widths.items()}
    shards = [(f"l.{name}_proj", spec(bits, rows)) for name, (bits, rows) in widths.items()]
    partitions = [rows // TP for _, rows in widths.values()]

    for rank in range(TP):
        layer = build(sglang, shards, partitions)
        geometry = layer.dynquant_geometry

        offset = 0
        for shard_id, (name, (_, rows)) in enumerate(widths.items()):
            size = rows // TP
            for attr, source in (("qweight", full[name][0]), ("scales", full[name][1])):
                getattr(layer, attr).load_qkv_weight(
                    source,
                    tp_rank=rank,
                    shard_offset=offset,
                    shard_size=size,
                    shard_id=name,
                    num_heads=1,
                )
            offset += size

            want = full[name][0][rank * size : (rank + 1) * size]
            assert torch.equal(geometry.view_qweight(layer.qweight.data, shard_id), want), (
                f"{name} rows on rank {rank}"
            )
            want_scales = full[name][1][rank * size : (rank + 1) * size]
            assert torch.equal(geometry.view_scale(layer.scales.data, shard_id), want_scales), (
                f"{name} scales on rank {rank}"
            )


def test_kv_heads_replicated_across_ranks_read_the_same_rows(sglang):
    """Fewer KV heads than ranks: k and v are copied, not split.

    SGLang signals this with ``num_heads=num_kv_head_replicas``, and getting it
    wrong is silent -- rank 1 would read rows past the end of a tensor that has
    exactly one head's worth, or wrap onto rank 0's. Both ranks must end up with
    byte-identical k.
    """
    k_rows = 128
    k_words, _ = checkpoint_tensors(4, k_rows)
    shards = [("l.q_proj", spec(4, 2048)), ("l.k_proj", spec(4, k_rows))]

    held = []
    for rank in range(TP):
        layer = build(sglang, shards, [1024, k_rows])
        layer.qweight.load_qkv_weight(
            k_words,
            tp_rank=rank,
            shard_offset=1024,
            shard_size=k_rows,
            shard_id="k",
            num_heads=TP,  # replicated: the source index is tp_rank // num_heads == 0
        )
        held.append(layer.dynquant_geometry.view_qweight(layer.qweight.data, 1).clone())

    assert torch.equal(held[0], k_words)
    assert torch.equal(held[0], held[1])


def test_merged_column_ranks_tile_the_checkpoint_rows(sglang):
    """``gate_up_proj``: the same split, reached through SGLang's other hook."""
    gate = checkpoint_tensors(4, 4096)
    up = checkpoint_tensors(3, 4096)
    shards = [("l.gate_proj", spec(4, 4096)), ("l.up_proj", spec(3, 4096))]

    for rank in range(TP):
        layer = build(sglang, shards, [2048, 2048])
        for shard_id, (source, offset) in enumerate(((gate, 0), (up, 2048))):
            layer.qweight.load_merged_column_weight(
                source[0], shard_offset=offset, shard_size=2048, tp_rank=rank, tp_size=TP
            )
            got = layer.dynquant_geometry.view_qweight(layer.qweight.data, shard_id)
            assert torch.equal(got, source[0][rank * 2048 : (rank + 1) * 2048])


def test_the_merged_hook_says_which_key_is_missing(sglang):
    """Subscripted, unlike SGLang's own implementation, which ``.get``s and then
    multiplies the ``None``. Every call site that can reach our class passes all
    four; the two that omit them are guarded by an exact-type test we do not
    satisfy. So an absent key means the convention moved, and the traceback should
    say which key rather than ``unsupported operand type(s) for *: 'NoneType'``."""
    gate, _ = checkpoint_tensors(4, 4096)
    layer = build(sglang, [("l.gate_proj", spec(4, 2048))], [2048])

    with pytest.raises(KeyError, match="tp_rank"):
        layer.qweight.load_merged_column_weight(gate, shard_offset=0, shard_size=2048)


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_row_parallel_ranks_tile_the_word_axis(sglang, bits):
    """``o_proj``/``down_proj``: every row, this rank's columns.

    Parametrised over all four widths because this is the split that is only legal
    on a group boundary, and the words-per-group differs per width -- 3-bit in
    particular packs 32 values into 3 words, so a split that is clean for 4-bit
    tells you nothing about it.
    """
    rows = 1024
    words, scales = checkpoint_tensors(bits, rows)
    shards = [("l.o_proj", spec(bits, rows))]

    halves, scale_halves = [], []
    for rank in range(TP):
        layer = build(sglang, shards, [rows], in_per_partition=IN_FEATURES // TP)
        assert layer.dynquant_row_split is not None
        layer.qweight.load_row_parallel_weight(words, tp_rank=rank)
        layer.scales.load_row_parallel_weight(scales, tp_rank=rank)
        geometry = layer.dynquant_geometry
        halves.append(geometry.view_qweight(layer.qweight.data, 0).clone())
        scale_halves.append(geometry.view_scale(layer.scales.data, 0).clone())

    assert torch.equal(torch.cat(halves, dim=1), words)
    assert torch.equal(torch.cat(scale_halves, dim=1), scales)


def test_row_parallel_split_off_a_group_boundary_is_refused_at_build_time(sglang):
    """Three groups over two ranks. Named module, before any weight is read."""
    in_features = 3 * GROUP_SIZE
    method = sglang.DynQuantLinearMethod(quant_config=None, shards=[("l.down_proj", spec(4, 512))])
    with pytest.raises(DynQuantError, match="down_proj"):
        method.create_weights(
            torch.nn.Module(),
            input_size_per_partition=in_features // TP,
            output_partition_sizes=[512],
            input_size=in_features,
            output_size=512,
            params_dtype=torch.bfloat16,
            weight_loader=None,
        )


def test_row_parallel_fused_layer_is_refused(sglang):
    """More than one width on the reduction axis has no single word split."""
    shards = [("l.a", spec(4, 512)), ("l.b", spec(3, 512))]
    with pytest.raises(DynQuantError, match="more than one"):
        build(sglang, shards, [256, 256], in_per_partition=IN_FEATURES // TP)


# --------------------------------------------------------------------------
# 3. Presharded weights -- the source is already this rank's
# --------------------------------------------------------------------------


def test_presharded_column_parallel_takes_the_tensor_as_given(sglang):
    """Every rank is handed its own rows, so no rank may take an offset.

    A port that ignored the flag would have rank 1 narrow past the end of a
    tensor that holds exactly one rank's rows -- which raises, loudly, and only
    at ``--tp-size 2``. Rank 0 would load fine, which is what makes it a
    configuration-dependent bug rather than an obvious one.
    """
    rows = 1024
    shards = [("l.q_proj", spec(4, rows))]

    for rank in range(TP):
        mine, mine_scales = checkpoint_tensors(4, rows)
        layer = build(sglang, shards, [rows])
        layer.qweight.load_column_parallel_weight(mine, tp_rank=rank, use_presharded_weights=True)
        layer.scales.load_column_parallel_weight(
            mine_scales, tp_rank=rank, use_presharded_weights=True
        )
        assert torch.equal(layer.dynquant_geometry.view_qweight(layer.qweight.data, 0), mine)
        assert torch.equal(layer.dynquant_geometry.view_scale(layer.scales.data, 0), mine_scales)


def test_presharded_row_parallel_skips_the_word_slice(sglang):
    """The row-parallel case has *two* things to skip, not one.

    Row-parallel placement takes no source-row offset in either mode -- every rank
    holds all the rows -- so the flag's whole effect here is the column slice. A
    port that zeroed the row offset and forgot the word slice would pass the
    column-parallel test above and load a quarter of a rank's words here.
    """
    rows = 512
    half_in = IN_FEATURES // TP
    shards = [("l.down_proj", spec(4, rows))]

    for rank in range(TP):
        mine, mine_scales = checkpoint_tensors(4, rows, in_features=half_in)
        layer = build(sglang, shards, [rows], in_per_partition=half_in)
        layer.qweight.load_row_parallel_weight(mine, tp_rank=rank, use_presharded_weights=True)
        layer.scales.load_row_parallel_weight(
            mine_scales, tp_rank=rank, use_presharded_weights=True
        )
        assert torch.equal(layer.dynquant_geometry.view_qweight(layer.qweight.data, 0), mine)
        assert torch.equal(layer.dynquant_geometry.view_scale(layer.scales.data, 0), mine_scales)


def test_presharded_qkv_takes_the_tensor_as_given(sglang):
    """Including for a replicated K, where the source index would have been 0
    anyway -- so this is the one presharded case a broken port passes by luck at
    ``tp_rank=0``, and the loop over both ranks is what catches it."""
    k_rows = 128
    shards = [("l.q_proj", spec(4, 2048)), ("l.k_proj", spec(4, k_rows))]

    for rank in range(TP):
        mine, _ = checkpoint_tensors(4, k_rows)
        layer = build(sglang, shards, [1024, k_rows])
        layer.qweight.load_qkv_weight(
            mine,
            tp_rank=rank,
            use_presharded_weights=True,
            shard_offset=1024,
            shard_size=k_rows,
            shard_id="k",
            num_heads=1,
        )
        assert torch.equal(layer.dynquant_geometry.view_qweight(layer.qweight.data, 1), mine)


# --------------------------------------------------------------------------
# 4. The kwarg SGLang passes that vLLM does not
# --------------------------------------------------------------------------


def test_the_block_quant_flag_is_accepted(sglang):
    """``ColumnParallelLinear`` passes it unconditionally (``linear.py:366``).

    A ``create_weights`` that did not accept it would raise ``TypeError:
    create_weights() got an unexpected keyword argument`` for *every* attention
    and MLP layer in the model -- so this is not a corner case, it is whether the
    plugin can build a layer at all under SGLang.
    """
    layer = build(sglang, [("l.q_proj", spec(4, 1024))], [1024], skip_block_quant_check=True)
    assert layer.qweight.numel() > 0


def test_the_block_quant_flag_does_not_become_a_weight_attribute(sglang):
    """Left in ``**extra_weight_attrs`` it would reach ``set_weight_attrs``.

    Harmless-looking -- the assert there only objects to *overwriting* -- and so
    it would ship: three tensors carrying an FP8 flag that means nothing to
    DynQuant, and a later reader with no way to tell it from something load-bearing.
    """
    layer = build(sglang, [("l.q_proj", spec(4, 1024))], [1024], skip_block_quant_check=True)
    for name in ("qweight", "scales", "offsets"):
        assert not hasattr(getattr(layer, name), "skip_block_quant_check"), name


def test_the_weight_loader_is_installed_once(sglang):
    """``BasevLLMParameter.__init__`` takes one; ``set_weight_attrs`` asserts
    ``not hasattr``. Forwarding the whole attrs dict would raise ``Overwriting
    existing tensor attribute: weight_loader`` while the layer is being built --
    so this test is red the moment the strip in ``create_weights`` is dropped,
    and it also pins that the loader does arrive on all three tensors."""

    def loader(param, loaded_weight):  # pragma: no cover - never called
        raise AssertionError

    layer = torch.nn.Module()
    method = sglang.DynQuantLinearMethod(quant_config=None, shards=[("l.q_proj", spec(4, 1024))])
    method.create_weights(
        layer,
        input_size_per_partition=IN_FEATURES,
        output_partition_sizes=[1024],
        input_size=IN_FEATURES,
        output_size=1024,
        params_dtype=torch.bfloat16,
        weight_loader=loader,
    )
    for name in ("qweight", "scales", "offsets"):
        assert getattr(layer, name).weight_loader is loader, name


# --------------------------------------------------------------------------
# 5. dtype
# --------------------------------------------------------------------------


def test_scales_wider_than_the_layer_are_refused_rather_than_rounded(sglang):
    """The reason ``_place`` ends in SGLang's ``copy_with_check``.

    An fp32-scales checkpoint served at ``--dtype float16`` is a real mistake --
    the exporter's dtype and the server's are set in different places. A plain
    ``copy_`` would take it and quietly lose the exponent range that made a 2-bit
    group's scale meaningful; every other SGLang parameter refuses, and so should
    this one.
    """
    rows = 512
    _, wide = checkpoint_tensors(4, rows)
    layer = build(sglang, [("l.q_proj", spec(4, rows))], [rows])

    with pytest.raises(ValueError, match="Downcasting not allowed"):
        layer.scales.load_column_parallel_weight(
            wide.to(torch.float32), tp_rank=0, use_presharded_weights=True
        )


def test_a_shape_mismatch_names_the_module_and_the_config(sglang):
    """Our check runs before ``copy_with_check``'s, which is the point of having
    both: SGLang's says ``target.shape=..., loaded_weight.shape=...`` and ours
    says which module, at what width, and that the config and the weights came
    from different runs."""
    words, _ = checkpoint_tensors(3, 512)
    layer = build(sglang, [("l.q_proj", spec(4, 512))], [512])

    with pytest.raises(DynQuantError) as excinfo:
        layer.qweight.load_column_parallel_weight(words, tp_rank=0, use_presharded_weights=True)

    message = str(excinfo.value)
    assert "l.q_proj" in message
    assert "4-bit" in message


# --------------------------------------------------------------------------
# 6. Why the registry entry is not decoration
# --------------------------------------------------------------------------


def test_the_registered_name_is_this_class(sglang):
    """``register()`` appends a string literal; SGLang matches ``__name__``.

    Two independent spellings of one fact, and nothing else would notice them
    drifting apart -- a rename would leave the literal pointing at a class that no
    longer exists, and the only symptom is the v1 loader being selected.
    """
    from dynquant.integration.sglang_plugin import _LINEAR_METHOD_NAME

    assert sglang.DynQuantLinearMethod.__name__ == _LINEAR_METHOD_NAME


def test_the_embedding_method_is_deliberately_not_registered(sglang):
    """It is a subclass, so its ``__name__`` differs and it is *not* in the list.

    Correct rather than an oversight: ``VocabParallelEmbedding`` and
    ``ParallelLMHead`` never consult ``WEIGHT_LOADER_V2_SUPPORTED``. They install
    their own ``weight_loader``, which is why the embedding buffers are 2-D
    ``[rows, words]`` while the linear ones are flat.
    """
    import sglang.srt.layers.linear as sgl_linear

    from dynquant.integration.sglang_plugin import register

    register()
    assert sglang.DynQuantLinearMethod.__name__ in sgl_linear.WEIGHT_LOADER_V2_SUPPORTED
    assert sglang.DynQuantEmbeddingMethod.__name__ not in sgl_linear.WEIGHT_LOADER_V2_SUPPORTED


def test_the_v1_loader_cannot_place_these_buffers(sglang):
    """Copies of SGLang's three v1 placement bodies, run against our parameter.

    What this proves, and what it does not. It does *not* prove the v1 loader
    corrupts silently -- it does not; every v1 path ends in ``assert
    param_data.shape == loaded_weight.shape`` before the copy (``linear.py:437``,
    ``:736``), and one of those asserts carries no message at all. What it proves
    is the real justification for the registry entry: the flat layout has no v1
    loader that can *express* it, because v1 places shards by narrowing the
    parameter along ``output_dim`` on the assumption that every row is the same
    width. Here they are not, and the arithmetic has nowhere to land.

    So the failure without the registry entry is loud, and useless: an
    ``AssertionError`` or an ``IndexError`` raised inside a spawned scheduler
    subprocess, on a line that names neither the module nor quantization.

    A copied-body test goes stale if SGLang rewrites those loaders, which is why
    the line numbers are here and why
    :func:`test_the_weight_loader_v2_list_is_still_class_names_as_strings` in the
    conformance file checks the selection mechanism itself against a real SGLang.
    """
    rows = 1024
    words, _ = checkpoint_tensors(4, rows * TP)
    layer = build(sglang, [("l.q_proj", spec(4, rows))], [rows])
    param_data = layer.qweight.data

    # ColumnParallelLinear.weight_loader, linear.py:408-440.
    with pytest.raises(RuntimeError):
        shard_size = param_data.shape[0]
        param_data.copy_(words.narrow(0, 0 * shard_size, shard_size))

    # MergedColumnParallelLinear.weight_loader, linear.py:~700-736 -- the narrow
    # succeeds, on the wrong elements, and the bare assert is what stops it.
    with pytest.raises(AssertionError):
        narrowed = param_data.narrow(0, 0, rows)
        assert narrowed.shape == words.shape

    # RowParallelLinear.weight_loader, linear.py:~1470-1530: input_dim is 1 and
    # the buffer has one dimension.
    with pytest.raises(IndexError):
        param_data.shape[layer.qweight.input_dim]


# --------------------------------------------------------------------------
# 7. The embedding method
# --------------------------------------------------------------------------


def embedding_layer(sglang, bits: int = 4, rows: int = 4096):
    layer = torch.nn.Module()
    method = sglang.DynQuantEmbeddingMethod(quant_config=None, shards=[("embed", spec(bits, rows))])
    method.create_weights(
        layer,
        IN_FEATURES,
        [rows],
        IN_FEATURES,
        rows,
        params_dtype=torch.bfloat16,
        weight_loader=None,
    )
    return layer, method


def test_the_embedding_buffers_are_two_dimensional(sglang):
    """``VocabParallelEmbedding.weight_loader`` does ``param[:n].data.copy_()`` and
    ``param[n:].data.fill_(0)`` (``vocab_parallel_embedding.py:501-502``), which
    only mean the right thing if dim 0 is the vocabulary. An embedding is never
    fused, so the flat layout buys nothing here and would cost correctness."""
    layer, _ = embedding_layer(sglang)
    geom = row_geometry(4, GROUP_SIZE, IN_FEATURES)

    assert layer.qweight.shape == (4096, geom.words_per_row)
    assert layer.scales.shape == (4096, geom.num_groups)
    assert layer.offsets.shape == (4096, geom.num_groups)
    assert layer.qweight.output_dim == 0


def test_the_embedding_buffers_start_zeroed(sglang):
    """Why ``offsets`` exists even for a symmetric table.

    The loader zero-fills padded rows with zero *codes*, and only
    ``q * scale + offset`` with a zero offset reconstructs to the zero vector. A
    ``None`` offsets buffer would leave those rows holding whatever the symmetric
    decode maps code 0 to -- which for an asymmetric group is not zero.
    """
    layer, _ = embedding_layer(sglang)
    assert torch.count_nonzero(layer.offsets) == 0
    assert torch.count_nonzero(layer.scales) == 0
    assert torch.count_nonzero(layer.qweight) == 0


def test_embedding_is_defined_on_the_class_itself(sglang):
    """``method_has_implemented_embedding`` (``base_config.py:260``) compares
    ``inspect.getattr_static`` of the method class against the base, and
    ``VocabParallelEmbedding.__init__`` raises ``NotImplementedError`` when they
    match. Inheriting it from somewhere convenient would not be enough."""
    assert "embedding" in vars(sglang.DynQuantEmbeddingMethod)


def test_there_is_no_tie_weights_override(sglang):
    """Deliberate, and the one place the vLLM twin has a method this port must not.

    SGLang's ``ParallelLMHead.tie_weights`` (``vocab_parallel_embedding.py:654``)
    is ``self.weight = embed_tokens.weight; return self`` -- it never consults
    ``quant_method``, where vLLM's delegates to it. An override here would be dead
    code that looked live, and the next reader would conclude tying was handled.

    It is survivable because the dense models we serve tie by assigning the whole
    module (``models/qwen3.py:488``), which carries all three buffers and the
    stash with them. If that ever changes, this test is where the note lives.
    """
    assert not hasattr(sglang.DynQuantEmbeddingMethod, "tie_weights")


def test_a_fused_embedding_is_refused(sglang):
    """Two shards for a vocabulary means the map does not describe this model."""
    method = sglang.DynQuantEmbeddingMethod(
        quant_config=None, shards=[("a", spec(4, 2048)), ("b", spec(4, 2048))]
    )
    with pytest.raises(DynQuantError, match="cannot be fused"):
        method.create_weights(
            torch.nn.Module(),
            IN_FEATURES,
            [4096],
            IN_FEATURES,
            4096,
            params_dtype=torch.bfloat16,
            weight_loader=None,
        )
