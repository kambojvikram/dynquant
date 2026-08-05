"""The architecture matrix: classification against real ``transformers`` module trees.

P3's exit gate asked for this and it was never written, so until now the only
architectures any test had seen were ``llama``, ``mixtral`` and ``qwen3_5`` --
while ``graph/arch/`` shipped exactly one plugin. Everything else went through
generic inference that nothing checked.

**Why these fixtures come from ``transformers`` and not from hand-built stubs.**
[test_graph_classify.py][1] deliberately hand-builds its fixtures, and for what it
tests that is right: it is checking our reasoning about shapes and ancestry, and a
stub cannot drift under it. This file is checking the opposite thing -- whether our
assumptions still match what ``transformers`` actually constructs. A stub of
``Phi3MLP`` would keep asserting that the gate comes first long after upstream
stopped doing that, which is precisely the failure it needs to catch. So the
fixtures are the real classes at 1/100th scale, built via ``from_config`` with no
downloads and no weights loaded from anywhere.

The four architectures are the phase-3 evaluation set
([docs/phase3-generalization-plan.md][2]): Llama-3.1 (the control), Gemma-3
(interleaved local/global attention plus a vision tower), Phi-4-mini (tied
embeddings and *both* fused projections), Ministral (sliding-window attention).
Between them they cover every structural feature that set exercises.

Four things were wrong when this file was first run, and each test below that names
one is the regression guard for it:

1. Phi's ``qkv_proj`` and ``gate_up_proj`` -- 60% of its quantizable parameters --
   got no row partitions at all, because row order is plugin-only knowledge and
   there was no ``phi3`` plugin.
2. Gemma-3's ``vision_model.embeddings.position_embedding`` fell through to
   ``OTHER``.
3. ``nn.MultiheadAttention.in_proj_weight`` matched the substring ``in_proj`` and
   read as Mamba's SSM input projection.
4. Three tensors per Gemma-3 -- including the multimodal projector -- were owned by
   no module's ``.weight`` and so were invisible to the graph and to the parameter
   count the average-bits figure divides by.

[1]: test_graph_classify.py
[2]: ../docs/phase3-generalization-plan.md
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from dynquant.graph import ModuleRole, classify_model
from dynquant.graph.registry import ModuleContext, plugin_for

if TYPE_CHECKING:
    from torch import nn

transformers = pytest.importorskip("transformers", reason="the arch matrix classifies real models")

pytestmark = pytest.mark.needs_hf

# Small enough that all four instantiate in well under a second, large enough that
# the ratios classification depends on stay distinguishable. In particular the GQA
# ratio is 4:2 rather than 1:1, so a QKV partition that splits evenly instead of by
# head count produces the wrong boundaries rather than accidentally right ones.
HIDDEN = 64
INTERMEDIATE = 128
HEADS, KV_HEADS = 4, 2
LAYERS = 2
VOCAB = 128

# `head_dim` is pinned to something other than `HIDDEN // HEADS` for Gemma-3
# because that is Gemma-3's actual departure from convention: it sets head_dim
# independently (256 at full scale, against hidden/heads = 640). A fixture that let
# them coincide would not notice code that assumed the quotient.
GEMMA_HEAD_DIM = 16

_TOKENS = {"pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2}


def _llama() -> tuple[nn.Module, Any]:
    cfg = transformers.LlamaConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        num_hidden_layers=LAYERS,
        vocab_size=VOCAB,
        **_TOKENS,
    )
    return transformers.AutoModelForCausalLM.from_config(cfg), cfg


def _phi3(*, tie: bool = True) -> tuple[nn.Module, Any]:
    cfg = transformers.Phi3Config(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        num_hidden_layers=LAYERS,
        vocab_size=VOCAB,
        tie_word_embeddings=tie,
        partial_rotary_factor=0.75,
        **_TOKENS,
    )
    return transformers.AutoModelForCausalLM.from_config(cfg), cfg


def _phi_4_mini_config() -> Any:
    """``microsoft/Phi-4-mini-instruct``'s real geometry, snapshot ``cfbefac``.

    The fixture above runs at hidden 64 with a 4:2 head ratio. Two properties only
    exist at the checkpoint's own scale: a 24:8 GQA ratio the fixture never produces,
    and a payload large enough to dominate the group metadata -- at hidden 64 a group
    of 128 pads every row, so the allocator's arithmetic is metadata, not weights.
    Note ``head_dim`` is deliberately unset, as it is in the real config.
    """
    return transformers.Phi3Config(
        hidden_size=3072,
        intermediate_size=8192,
        num_attention_heads=24,
        num_key_value_heads=8,
        num_hidden_layers=32,
        vocab_size=200064,
        tie_word_embeddings=True,
        partial_rotary_factor=0.75,
        **_TOKENS,
    )


def _mistral(*, sliding_window: int | None = 16) -> tuple[nn.Module, Any]:
    cfg = transformers.MistralConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        num_hidden_layers=LAYERS,
        vocab_size=VOCAB,
        sliding_window=sliding_window,
        **_TOKENS,
    )
    return transformers.AutoModelForCausalLM.from_config(cfg), cfg


def _gemma3() -> tuple[nn.Module, Any]:
    cfg = transformers.Gemma3Config(
        text_config={
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "num_attention_heads": HEADS,
            "num_key_value_heads": KV_HEADS,
            "head_dim": GEMMA_HEAD_DIM,
            "num_hidden_layers": LAYERS,
            "vocab_size": VOCAB,
            **_TOKENS,
        },
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": LAYERS,
            "num_attention_heads": HEADS,
            "image_size": 16,
            "patch_size": 8,
        },
    )
    return transformers.Gemma3ForConditionalGeneration(cfg), cfg


BUILDERS = {
    "llama": _llama,
    "phi3": _phi3,
    "mistral": _mistral,
    "gemma3": _gemma3,
}


def _graph(model_type: str):
    model, cfg = BUILDERS[model_type]()
    return classify_model(model, config=cfg), model


# --------------------------------------------------------------------------
# The gate: nothing unclassified, nothing unaccounted for
# --------------------------------------------------------------------------


@pytest.mark.parametrize("model_type", list(BUILDERS))
def test_no_module_lands_in_other(model_type: str) -> None:
    """Every quantizable weight gets a real role on all four architectures.

    ``OTHER`` is not a crash -- it is a conservative 4-bit floor and a warning --
    which is exactly why it needs a test. A module that silently lands there keeps
    the pipeline working while quietly opting one tensor out of role-aware
    allocation, and on a vision tower that was one module per model.

    Turns red when: a new ``transformers`` release renames a projection, or a fifth
    architecture is added to the matrix before its names are covered.
    """
    graph, _ = _graph(model_type)
    assert graph.unclassified() == (), (
        f"{model_type}: unclassified modules {list(graph.unclassified())}"
    )


@pytest.mark.parametrize("model_type", list(BUILDERS))
def test_every_matrix_parameter_is_accounted_for(model_type: str) -> None:
    """No ``ndim>=2`` parameter is invisible to the graph.

    The graph is what the budget divides by, so a tensor it never sees is a tensor
    the average-bits figure does not divide by -- while the tensor is still on
    disk. That is the tied-embedding error running in the opposite direction, and
    it costs the manifest its claim to be exact.

    Gemma-3 had three such tensors, because ``named_modules`` only finds a weight
    whose owner spells it ``self.weight`` and both ``nn.MultiheadAttention`` and
    ``Gemma3MultiModalProjector`` keep theirs in bare ``nn.Parameter`` attributes.

    Turns red when: an architecture introduces another raw-parameter weight, or the
    sweep that picks them up is removed.
    """
    from dynquant.graph.naming import canonical_name

    graph, model = _graph(model_type)
    known = set(graph.modules) | set(graph.skipped)

    for raw_name, param in model.named_parameters():
        if sum(1 for dim in param.shape if dim > 1) < 2:
            continue  # a vector, however many singleton axes it wears
        name = canonical_name(raw_name)
        assert name in known, (
            f"{model_type}: {name} {tuple(param.shape)} is in neither the graph "
            f"nor the skip list -- it would be absent from the parameter count"
        )


@pytest.mark.parametrize("model_type", list(BUILDERS))
def test_floor_cost_counts_every_parameter(model_type: str) -> None:
    """``total_params()`` equals the model's own distinct-parameter count.

    The arithmetic form of the test above: it is not enough for a tensor to appear
    somewhere in the graph, it has to reach the denominator. Ties are counted once
    on both sides, which is what makes this an equality rather than a bound.

    Turns red when: a tensor is classified but excluded from the count, or a tie is
    double-counted.
    """
    graph, model = _graph(model_type)

    seen: dict[int, int] = {}
    for _, param in model.named_parameters():
        if sum(1 for dim in param.shape if dim > 1) >= 2:
            seen[id(param)] = param.numel()

    assert graph.total_params() == sum(seen.values())


# --------------------------------------------------------------------------
# Phi: the fused projections, which are 60% of the model
# --------------------------------------------------------------------------


def test_phi_fused_projections_are_row_partitioned() -> None:
    """Phi's two fused tensors carry row partitions rather than one flat width.

    Without partitions the whole of ``gate_up_proj`` sits at the SwiGLU gate's
    4-bit floor even though the ``up`` half would take 3, and ``qkv_proj`` cannot
    price Q separately from K and V. On Phi-4-mini those two tensors are 60% of the
    quantizable parameters, so this is not a rounding error -- it is most of the
    model declining to participate in the thing DynQuant does.

    Turns red when: the ``phi3`` plugin is unregistered, or its guards start
    declining on a shape they should accept.
    """
    graph, _ = _graph("phi3")
    fused = [m for m in graph if m.role.is_fused]

    assert fused, "phi3 should have fused projections at all"
    for info in fused:
        assert info.partitions, f"{info.name} ({info.role.value}) has no row partitions"
        assert info.partitions[0].start == 0
        assert info.partitions[-1].stop == info.shape[0]
        # Contiguous and non-overlapping: a gap would leave rows with no assigned
        # width, an overlap would give some rows two.
        for earlier, later in zip(info.partitions, info.partitions[1:], strict=False):
            assert earlier.stop == later.start


def test_phi_gate_up_splits_at_intermediate_size_with_gate_first() -> None:
    """``gate_up_proj`` is ``[gate; up]``, split at ``intermediate_size``.

    ``Phi3MLP.forward`` does ``gate, up = gate_up_proj(x).chunk(2, dim=-1)``, so the
    gate is the *first* half. Reversing it would hand the gate's higher floor to the
    up-projection and starve the gate -- a checkpoint that loads, runs, and is
    quietly worse, with no symptom that points here.

    Turns red when: the partition order is flipped, or the split stops being read
    from ``intermediate_size``.
    """
    graph, _ = _graph("phi3")
    gate_ups = [m for m in graph if m.role is ModuleRole.MLP_GATE_UP]
    assert gate_ups

    for info in gate_ups:
        gate, up = info.partitions
        assert gate.role is ModuleRole.MLP_GATE
        assert up.role is ModuleRole.MLP_UP
        assert (gate.start, gate.stop) == (0, INTERMEDIATE)
        assert (up.start, up.stop) == (INTERMEDIATE, 2 * INTERMEDIATE)


def test_phi_qkv_partition_follows_the_gqa_ratio() -> None:
    """``qkv_proj`` splits by head count, not into three equal blocks.

    Under GQA the query block is ``num_attention_heads`` wide and the key and value
    blocks are ``num_key_value_heads`` wide -- 8x apart on the real Phi-4-mini. An
    even three-way split sums to the right total while putting every boundary in
    the wrong place, which is the failure mode a shape-only check would miss.

    Turns red when: the split reverts to ``out_features // 3``, or stops reading
    ``num_key_value_heads``.
    """
    graph, _ = _graph("phi3")
    qkvs = [m for m in graph if m.role is ModuleRole.ATTN_QKV]
    assert qkvs

    head_dim = HIDDEN // HEADS
    q_width, kv_width = HEADS * head_dim, KV_HEADS * head_dim
    for info in qkvs:
        q, k, v = info.partitions
        assert (q.role, q.num_rows) == (ModuleRole.ATTN_Q, q_width)
        assert (k.role, k.num_rows) == (ModuleRole.ATTN_K, kv_width)
        assert (v.role, v.num_rows) == (ModuleRole.ATTN_V, kv_width)
        assert q.num_rows != k.num_rows, "fixture must not let an even split pass"


def test_phi_tied_embedding_takes_the_lm_head_floor() -> None:
    """Phi-4-mini ties ``embed_tokens`` to ``lm_head``; the tie has to win.

    One tensor, one bit-width, and the strictest of the two roles' floors -- 8 from
    the head, not 4 from the embedding. Taking the representative's own floor would
    make the floor of a large fraction of the model depend on which name
    ``named_modules`` happened to yield first.

    Phi-4-mini is the only tied model in the phase-3 set, so it is the only place
    this is exercised against a real architecture.

    Turns red when: tie detection breaks, or the floor stops being maxed over
    ``tied_roles``.
    """
    graph, _ = _graph("phi3")

    assert graph.tied_groups, "phi3 fixture is configured with tie_word_embeddings=True"
    representative = graph.tied_groups[0][0]
    info = graph[representative]

    assert info.role is ModuleRole.EMBEDDING
    assert ModuleRole.LM_HEAD in info.tied_roles
    assert info.floor_bits == 8

    followers = [m for m in graph if m.is_tied_follower]
    assert followers and all(m.name not in {i.name for i in graph.quantizable()} for m in followers)


def test_phi_partitions_hold_at_phi_4_mini_s_real_geometry() -> None:
    """The same two splits, at the numbers the campaign actually quantizes.

    The fixture above runs at hidden 64 with a 4:2 head ratio. The checkpoint phase 3
    spends its GPU-hours on runs at hidden 3072 with 24:8, and it does not set
    ``head_dim`` at all -- so the real model exercises the ``hidden // heads``
    fallback and a 3:1 block ratio that the fixture never produces. Both boundaries
    are asserted as literals rather than recomputed from the config, because
    recomputing them here would just restate the code under test.

    Values read from ``microsoft/Phi-4-mini-instruct``'s ``config.json``
    (snapshot ``cfbefac``). Weights are on the meta device: only ``shape`` is read,
    and materialising 15.7 M rows to check a boundary would be its own bug.

    Turns red when: the fallback for an absent ``head_dim`` is dropped, a guard
    starts declining on this shape, or Phi-4-mini's geometry moves under us.
    """
    import torch

    plugin = plugin_for("phi3")
    assert plugin is not None

    cfg = _phi_4_mini_config()
    assert getattr(cfg, "head_dim", None) in (None, 128), "fallback path must stay reachable"

    def _ctx(leaf: str, out_features: int) -> ModuleContext:
        return ModuleContext(
            name=f"model.layers.0.{'self_attn' if 'qkv' in leaf else 'mlp'}.{leaf}",
            module=torch.nn.Identity(),
            weight=torch.empty(out_features, 3072, device="meta"),
            config=cfg,
            ancestors=(),
            leaf=leaf,
        )

    qkv = plugin.partitions_for(_ctx("qkv_proj", 5120), ModuleRole.ATTN_QKV)
    assert qkv is not None, "5120 = 24x128 + 2x(8x128); the guard must accept it"
    assert [(p.role, p.start, p.stop) for p in qkv] == [
        (ModuleRole.ATTN_Q, 0, 3072),
        (ModuleRole.ATTN_K, 3072, 4096),
        (ModuleRole.ATTN_V, 4096, 5120),
    ]

    gate_up = plugin.partitions_for(_ctx("gate_up_proj", 16384), ModuleRole.MLP_GATE_UP)
    assert gate_up is not None
    assert [(p.role, p.start, p.stop) for p in gate_up] == [
        (ModuleRole.MLP_GATE, 0, 8192),
        (ModuleRole.MLP_UP, 8192, 16384),
    ]


def test_soft_floors_reach_phi_s_fused_tensors() -> None:
    """A fused tensor can be pushed below its maxed floor like any other.

    ``floor_bits`` on a fused tensor is the *strictest* of its partitions' floors --
    ``gate_up_proj`` inherits the SwiGLU gate's 4 even though half its rows are an
    up-projection that would take 3. On Phi-4-mini that rule applies to 55% of the
    parameters, and it lifts the whole model's floor cost to 4.43 average bits, so
    every target the campaign runs sits below it. If soft floors did not reach these
    tensors the allocator would have nothing left to trade and would miss the budget
    on the one model in the panel that is fused.

    Every allocator test in ``test_allocate.py`` runs on an unfused synthetic model,
    which is why this lives here: it is the fused half of the same guard.

    Turns red when: ``_downgrade`` starts treating a fused role as structural, or
    fused tensors stop appearing in the violation report that makes the trade visible.
    """
    import torch

    from dynquant.allocate.budget import Budget
    from dynquant.allocate.knapsack import allocate_bits

    # Real geometry, not the fixture: at hidden 64 a group of 128 pads every row and
    # the scale/offset metadata outweighs the payload, so 2 bits everywhere costs more
    # than a 3.25-bit budget and the allocator correctly refuses before it can descend.
    # The property under test only exists at a scale where the payload dominates.
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(_phi_4_mini_config())
    graph = classify_model(model, config=_phi_4_mini_config())

    fused = {m.name for m in graph.quantizable() if m.partitions}
    assert fused, "phi3 should have fused tensors to allocate over"

    unaffordable = graph.floor_cost_bits() / graph.total_params()
    assert unaffordable > 4.0, f"Phi-4-mini's floors should be unaffordable, got {unaffordable:.2f}"

    budget = Budget.from_target(graph, target_bits=3.25)
    scores = {info.name: 0.5 for info in graph.quantizable()}
    result = allocate_bits(graph, scores, budget)
    assert abs(result.average_bits - 3.25) < 0.01, result.summary()

    breached = {v.name for v in result.violations}
    assert fused & breached, (
        f"no fused tensor was traded down at 3.25b against {unaffordable:.2f}b of "
        f"floors; breached instead: {sorted(breached)[:5]}"
    )
    for name in fused:
        assert result.bits[name] >= 2


def test_untied_phi_keeps_the_embedding_floor() -> None:
    """The 8-bit floor above comes from the tie, not from being an embedding.

    Without this, ``test_phi_tied_embedding_takes_the_lm_head_floor`` would still
    pass if the embedding floor were simply raised to 8 everywhere, and the tie
    logic could rot undetected.

    Turns red when: the embedding floor is changed globally instead of by tie.
    """
    model, cfg = _phi3(tie=False)
    graph = classify_model(model, config=cfg)

    assert graph.tied_groups == ()
    embedding = next(m for m in graph if m.role is ModuleRole.EMBEDDING)
    assert embedding.floor_bits == 4


# --------------------------------------------------------------------------
# Gemma-3: the vision tower
# --------------------------------------------------------------------------


def test_gemma3_multimodal_projector_is_in_the_graph() -> None:
    """The vision-to-text bridge is classified, not skipped into nothing.

    ``Gemma3MultiModalProjector`` holds the whole bridge in a bare
    ``nn.Parameter``, so it has no ``.weight`` and ``named_modules`` walks straight
    past it. ``DEFAULT_FLOOR_BITS`` rates the projector as one of the least
    compressible tensors in a VLM -- small, no redundancy, every image token passes
    through it -- which makes silently omitting it the worst available outcome.

    Turns red when: the raw-parameter sweep is removed, or ``MM_PROJECTOR`` stops
    matching the name.
    """
    graph, _ = _graph("gemma3")
    projectors = [m for m in graph if m.role is ModuleRole.MM_PROJECTOR]

    assert len(projectors) == 1
    assert projectors[0].source == "parameter"
    assert projectors[0].floor_bits == 8


def test_gemma3_vision_position_embedding_is_not_other() -> None:
    """SigLIP's positional table classifies, and as a vision tensor.

    It shares no substring with ``embed_tokens``, so before the ``position_embedding``
    leaf rule every Gemma-3 shipped one module in ``OTHER``. It is added directly to
    the patch embeddings, so it inherits their floor rather than the token
    embedding's.

    Turns red when: the leaf rule is dropped, or the vision remap stops applying to
    embeddings.
    """
    graph, _ = _graph("gemma3")
    names = {m.name: m for m in graph}
    positional = [m for n, m in names.items() if n.endswith("embeddings.position_embedding")]

    assert positional, "fixture should build a SigLIP tower with a positional table"
    assert all(m.role is ModuleRole.VISION_PATCH_EMBED for m in positional)


def test_gemma3_multihead_attention_qkv_is_not_read_as_ssm() -> None:
    """``in_proj_weight`` is a fused QKV, not Mamba's ``in_proj``.

    ``nn.MultiheadAttention`` -- which SigLIP's pooling head uses -- keeps Q, K and V
    in one raw parameter called ``in_proj_weight``. The substring pass contains
    ``in_proj`` for Mamba, so without an exact-leaf rule this lands on the SSM input
    floor: a confident wrong answer, in a tensor the name sweep only reaches at all
    because of the raw-parameter fix.

    Turns red when: the ``in_proj_weight`` leaf rule is removed and the substring
    pass reclaims it.
    """
    graph, _ = _graph("gemma3")
    heads = [m for m in graph if m.name.endswith("in_proj_weight")]

    assert heads, "fixture should build the SigLIP pooling head"
    for info in heads:
        assert info.role is ModuleRole.VISION_ATTN
        assert not info.role.is_ssm


def test_gemma3_vision_tower_is_separable_from_the_text_tower() -> None:
    """Vision weights are identifiable as such, so the tower can be left dense.

    The phase-3 plan quantizes Gemma-3's language model only and reports the
    compression ratio both ways. That is only possible if every vision tensor
    carries a vision role -- an exclusion list built from name globs would be one
    ``transformers`` rename away from silently quantizing the tower.

    Turns red when: a vision tensor starts classifying as a text role, which would
    put it on the wrong side of the exclusion and out of the language-model-only
    byte accounting.
    """
    graph, _ = _graph("gemma3")

    vision = [m for m in graph if m.role.is_vision]
    assert vision, "gemma3 is multimodal; the tower must be visible"
    assert all("vision" in m.name or "multi_modal" in m.name for m in vision)

    text = [m for m in graph if not m.role.is_vision]
    assert all("vision_tower" not in m.name for m in text)


# --------------------------------------------------------------------------
# Ministral: sliding-window attention must not reach allocation
# --------------------------------------------------------------------------


def test_sliding_window_does_not_change_any_role() -> None:
    """Ministral's interleaved sliding window is a KV-cache property, not a weight one.

    Sliding-window attention changes which keys a query attends to. It does not
    change what any weight tensor *is*, so it must not reach role assignment. This
    asserts that rather than assuming it: the phase-3 set includes Ministral
    precisely because its distinguishing feature is one we expect to be a no-op
    here, and an expectation worth stating is worth testing.

    Turns red when: a config-reading rule starts keying on ``sliding_window`` or
    ``layer_types`` and makes allocation depend on attention span.
    """
    windowed, cfg_w = _mistral(sliding_window=16)
    plain, cfg_p = _mistral(sliding_window=None)

    roles_w = {m.name: m.role for m in classify_model(windowed, config=cfg_w)}
    roles_p = {m.name: m.role for m in classify_model(plain, config=cfg_p)}

    assert roles_w == roles_p


# --------------------------------------------------------------------------
# The guard that keeps a wrong partition from being worse than none
# --------------------------------------------------------------------------


def test_phi_plugin_declines_when_widths_do_not_add_up() -> None:
    """A shape the config cannot explain gets no partition rather than a guessed one.

    Partitioning is only safe because the boundary is derived from the config and
    checked against the tensor. If a checkpoint disagrees with its own config, the
    correct output is ``None`` -- one width for the whole tensor, merely
    suboptimal. Slicing anyway would assign the gate's bits to arbitrary rows,
    which is silently wrong.

    Turns red when: either guard is relaxed into an unchecked ``out_features // 2``
    or ``// 3``.
    """
    import torch

    plugin = plugin_for("phi3")
    assert plugin is not None

    _, cfg = _phi3()
    bad = ModuleContext(
        name="model.layers.0.mlp.gate_up_proj",
        module=torch.nn.Identity(),
        weight=torch.zeros(2 * INTERMEDIATE + 8, HIDDEN),  # not 2 x intermediate
        config=cfg,
        ancestors=(),
        leaf="gate_up_proj",
    )
    assert plugin.partitions_for(bad, ModuleRole.MLP_GATE_UP) is None

    bad_qkv = ModuleContext(
        name="model.layers.0.self_attn.qkv_proj",
        module=torch.nn.Identity(),
        weight=torch.zeros(HIDDEN + 7, HIDDEN),
        config=cfg,
        ancestors=(),
        leaf="qkv_proj",
    )
    assert plugin.partitions_for(bad_qkv, ModuleRole.ATTN_QKV) is None
