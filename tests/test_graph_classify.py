"""Phase-3 role classification.

The tests that matter here are the ones where the *name* and the *function*
disagree, because that gap is what the supplement's substring matcher fell into:

- ``mlp.gate`` is a SwiGLU gate in Llama and an expert router in Mixtral.
- ``out_proj`` is attention output in self-attention and the DeltaNet output
  projection inside a gated linear-attention block.
- ``q_proj`` is a query projection, unless ``attn_output_gate`` is set, in which
  case half its rows are a sigmoid gate.
- ``embed_tokens`` and ``lm_head`` can be one tensor wearing two names.

Model fixtures are hand-built rather than instantiated from ``transformers``: the
shapes and class names are what classification reads, no downloads are involved,
and a fixture cannot drift when a library version changes.
"""

# ruff: noqa: N801 -- fixture classes copy the real transformers class names
# verbatim (`Qwen3_5GatedDeltaNet`, not `Qwen35GatedDeltaNet`) because ancestry
# matching reads `type(module).__name__`. Renaming them to satisfy CapWords would
# stop the tests exercising the strings the plugin actually sees.

from __future__ import annotations

import pytest
import torch
from torch import nn

from dynquant.graph import ModuleRole, classify_model
from dynquant.graph.registry import plugin_for, registered_model_types
from dynquant.graph.roles import UNQUANTIZED_FLOOR

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


class _Cfg:
    """Stand-in for a transformers config: attribute access is all that matters."""

    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)


# Qwen3.5-2B at 1/16 scale in every width, with the real layer count and the real
# 3-linear-to-1-full interleave. Scaling widths rather than depth is what keeps the
# size *proportions* -- embed 27%, MLP 48%, linear-attn 21%, full-attn 4.6% -- and
# those proportions are what the floor-cost and budget assertions below actually
# test. Full width would be 1.88B params instantiated once per test; this is 7.4M.
_SCALE = 16
HIDDEN = 2048 // _SCALE  # 128
INTERMEDIATE = 6144 // _SCALE  # 384
VOCAB = 248320 // _SCALE  # 15520
LIN_HEAD_DIM = 128 // _SCALE  # 8
# Deliberately *not* 256 // _SCALE. At full width the two head configs coincide --
# full attention is 8 x 256 and linear attention 16 x 128, both 2048 -- so a plugin
# reading the wrong one still slices in the right place and the bug hides. 24 makes
# 8 x 24 = 192 differ from 16 x 8 = 128, turning that coincidence into a failure.
HEAD_DIM = 24
N_HEADS, N_KV_HEADS, N_LIN_HEADS = 8, 2, 16
N_LAYERS = 24

Q_GATE_OUT = 2 * N_HEADS * HEAD_DIM  # 256 -- the doubled width, [Q; gate]
KV_OUT = N_KV_HEADS * HEAD_DIM  # 32
LIN_BLOCK = N_LIN_HEADS * LIN_HEAD_DIM  # 128, one of q/k/v
DECAY_ROWS = 16  # in_proj_a / in_proj_b, unscaled: it is a head count, not a width

QWEN35_TEXT_CFG = _Cfg(
    model_type="qwen3_5",
    vocab_size=VOCAB,
    hidden_size=HIDDEN,
    num_attention_heads=N_HEADS,
    num_key_value_heads=N_KV_HEADS,
    head_dim=HEAD_DIM,
    attn_output_gate=True,
    linear_num_key_heads=N_LIN_HEADS,
    linear_num_value_heads=N_LIN_HEADS,
    linear_key_head_dim=LIN_HEAD_DIM,
    linear_value_head_dim=LIN_HEAD_DIM,
    linear_conv_kernel_dim=4,
    tie_word_embeddings=True,
)


class Qwen3_5GatedDeltaNet(nn.Module):
    """Real class name -- ancestry is what the plugin matches on."""

    def __init__(self) -> None:
        super().__init__()
        self.in_proj_qkv = nn.Linear(HIDDEN, 3 * LIN_BLOCK, bias=False)
        self.in_proj_z = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.in_proj_b = nn.Linear(HIDDEN, DECAY_ROWS, bias=False)
        self.in_proj_a = nn.Linear(HIDDEN, DECAY_ROWS, bias=False)
        self.conv1d = nn.Conv1d(
            3 * LIN_BLOCK, 3 * LIN_BLOCK, kernel_size=4, groups=3 * LIN_BLOCK, bias=False
        )
        self.out_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.norm = nn.RMSNorm(LIN_HEAD_DIM)


class Qwen3_5Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, Q_GATE_OUT, bias=False)  # [Q; output_gate]
        self.k_proj = nn.Linear(HIDDEN, KV_OUT, bias=False)
        self.v_proj = nn.Linear(HIDDEN, KV_OUT, bias=False)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.q_norm = nn.RMSNorm(HEAD_DIM)
        self.k_norm = nn.RMSNorm(HEAD_DIM)


class Qwen3_5MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, *, linear: bool) -> None:
        super().__init__()
        if linear:
            self.linear_attn = Qwen3_5GatedDeltaNet()
        else:
            self.self_attn = Qwen3_5Attention()
        self.mlp = Qwen3_5MLP()
        self.input_layernorm = nn.RMSNorm(HIDDEN)
        self.post_attention_layernorm = nn.RMSNorm(HIDDEN)


class Qwen3_5Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        # 3 linear : 1 full, the real Qwen3.5 interleaving -- so layer 3 is the
        # full-attention layer every test below reaches for.
        self.layers = nn.ModuleList(
            Qwen3_5DecoderLayer(linear=(i % 4) != 3) for i in range(N_LAYERS)
        )
        self.norm = nn.RMSNorm(HIDDEN)


class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, *, tie: bool = True) -> None:
        super().__init__()
        self.config = QWEN35_TEXT_CFG
        self.model = Qwen3_5Model()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        if tie:
            self.lm_head.weight = self.model.embed_tokens.weight


# ---- a Mixtral-shaped MoE, for the router test ----------------------------

MIXTRAL_CFG = _Cfg(model_type="mixtral", vocab_size=32000, hidden_size=512, num_local_experts=8)


class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(512, 1024, bias=False)
        self.w2 = nn.Linear(1024, 512, bias=False)
        self.w3 = nn.Linear(512, 1024, bias=False)


class MixtralSparseMoeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # The trap: named `gate`, output width 8. A substring matcher reads "gate"
        # as a SwiGLU gate and gives a router 3 or 4 bits.
        self.gate = nn.Linear(512, 8, bias=False)
        self.experts = nn.ModuleList(MixtralBlockSparseTop2MLP() for _ in range(8))


class MixtralDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block_sparse_moe = MixtralSparseMoeBlock()


class MixtralForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = MIXTRAL_CFG
        self.embed_tokens = nn.Embedding(32000, 512)
        self.layers = nn.ModuleList([MixtralDecoderLayer()])


# --------------------------------------------------------------------------
# Qwen3.5 hybrid linear attention
# --------------------------------------------------------------------------


@pytest.fixture
def qwen_graph() -> object:
    return classify_model(Qwen3_5ForCausalLM())


def test_linear_attention_projections_get_linear_attention_roles(qwen_graph: object) -> None:
    """The four split projections, none of which the generic path would know.

    Without the plugin ``in_proj_*`` matches Mamba's ``in_proj`` -> ``ssm.in``,
    which is wrong for all four and specifically dangerous for ``in_proj_a``.
    """
    g = qwen_graph
    assert g["model.layers.0.linear_attn.in_proj_qkv"].role is ModuleRole.LIN_ATTN_QKV
    assert g["model.layers.0.linear_attn.in_proj_z"].role is ModuleRole.LIN_ATTN_Z
    assert g["model.layers.0.linear_attn.in_proj_a"].role is ModuleRole.LIN_ATTN_A
    assert g["model.layers.0.linear_attn.in_proj_b"].role is ModuleRole.LIN_ATTN_B
    assert g["model.layers.0.linear_attn.conv1d"].role is ModuleRole.LIN_ATTN_CONV
    assert all(
        m.source == "plugin:qwen_linear_attn" for m in g.quantizable() if "linear_attn" in m.name
    )


def test_out_proj_means_different_things_in_different_blocks(qwen_graph: object) -> None:
    """Same leaf name, two roles, decided by ancestry alone."""
    g = qwen_graph
    assert g["model.layers.0.linear_attn.out_proj"].role is ModuleRole.LIN_ATTN_OUT
    assert g["model.layers.3.self_attn.o_proj"].role is ModuleRole.ATTN_O


def test_decay_projections_are_floored_high(qwen_graph: object) -> None:
    """``in_proj_a``/``_b`` feed an exponential, and cost ~nothing to protect."""
    g = qwen_graph
    a = g["model.layers.0.linear_attn.in_proj_a"]
    assert a.floor_bits == 8
    assert a.num_params == DECAY_ROWS * HIDDEN
    # The whole reason 8 bits is affordable: these are a rounding error in size.
    # The bound is loose because these rows are a *head count* and so do not shrink
    # with the fixture's widths -- at full width they are 0.06% of the model, here
    # 1%. Either way, protecting them costs ~1.2 Mbit on the real 1.88B model.
    total = g.unique_params()
    tiny = sum(
        m.num_params
        for m in g.quantizable()
        if m.role in {ModuleRole.LIN_ATTN_A, ModuleRole.LIN_ATTN_B}
    )
    assert tiny / total < 0.02


def test_depthwise_conv1d_is_left_unquantized_but_still_budgeted(qwen_graph: object) -> None:
    """4 taps per channel: nothing to compress, but still bits on disk.

    It must be out of :meth:`quantizable` (no decision to make) and in
    :meth:`unquantized` (16 bits per parameter that the manifest's average has to
    account for). Dropping it from both is how a reported "3.0 average bits" ends
    up describing a file that is larger than that.
    """
    conv = qwen_graph["model.layers.0.linear_attn.conv1d"]
    assert conv.floor_bits == UNQUANTIZED_FLOOR
    assert conv.is_quantizable is False
    assert conv not in qwen_graph.quantizable()
    assert conv in qwen_graph.unquantized()
    assert qwen_graph.total_params() == (
        qwen_graph.unique_params() + qwen_graph.unquantized_params()
    )


def test_fused_q_and_output_gate_is_row_partitioned(qwen_graph: object) -> None:
    """``attn_output_gate`` doubles q_proj; the halves need different precision."""
    half = N_HEADS * HEAD_DIM
    q = qwen_graph["model.layers.3.self_attn.q_proj"]
    assert q.role is ModuleRole.ATTN_Q_GATE
    assert q.shape == (2 * half, HIDDEN)
    assert q.partitions == (
        (ModuleRole.ATTN_Q, 0, half),
        (ModuleRole.MLP_GATE, half, 2 * half),
    )
    assert [p.num_rows for p in q.partitions] == [half, half]


def test_fused_deltanet_qkv_partitions_use_the_linear_head_config(qwen_graph: object) -> None:
    """Linear heads are 16 x LIN_HEAD_DIM, distinct from full attention's 8 x HEAD_DIM.

    Reading the full-attention head config here would slice at ``8 * HEAD_DIM``
    instead of ``16 * LIN_HEAD_DIM``, which happens to be the same number at full
    scale -- the fixture keeps them distinct so the wrong config is a visible
    failure rather than a coincidence.
    """
    qkv = qwen_graph["model.layers.0.linear_attn.in_proj_qkv"]
    assert qkv.shape == (3 * LIN_BLOCK, HIDDEN)
    assert qkv.partitions == (
        (ModuleRole.ATTN_Q, 0, LIN_BLOCK),
        (ModuleRole.ATTN_K, LIN_BLOCK, 2 * LIN_BLOCK),
        (ModuleRole.ATTN_V, 2 * LIN_BLOCK, 3 * LIN_BLOCK),
    )
    assert N_HEADS * HEAD_DIM != LIN_BLOCK, "fixture must keep the two head configs apart"


def test_q_gate_partitioning_declines_when_the_shape_disagrees() -> None:
    """A config flag alone is not enough -- a wrong boundary beats no boundary.

    Here the config claims a fused gate but the tensor is plain-Q width, so the
    plugin must decline both the role and the partition rather than slice 2048 rows
    into two 1024-row halves that mean nothing.
    """

    class PlainQAttention(Qwen3_5Attention):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(HIDDEN, N_HEADS * HEAD_DIM, bias=False)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = QWEN35_TEXT_CFG  # still says attn_output_gate=True
            self.self_attn = PlainQAttention()

    graph = classify_model(Model())
    q = graph["self_attn.q_proj"]
    assert q.role is ModuleRole.ATTN_Q
    assert q.partitions == ()


# --------------------------------------------------------------------------
# Ties and budget accounting
# --------------------------------------------------------------------------


def test_tied_embedding_and_head_are_one_decision_and_counted_once() -> None:
    """The 27%-of-the-model accounting bug, guarded.

    Summing ``weight.numel()`` over modules counts the shared tensor twice. On the
    real Qwen3.5-2B that is 508.6M of 1.88B, so a budget built that way is 27% too
    large and the resulting map does not fit on disk.
    """
    graph = classify_model(Qwen3_5ForCausalLM(tie=True))
    assert graph.tied_groups == (("model.embed_tokens", "lm_head"),)

    # Naive over *quantizable-role* modules, so the excluded conv1d does not muddy
    # the comparison: the whole difference must be the one shared tensor.
    naive = sum(m.num_params for m in graph if m.is_quantizable)
    assert naive - graph.unique_params() == VOCAB * HIDDEN

    # Exactly one of the pair carries the decision; the other defers to it.
    assert graph["model.embed_tokens"].is_tied_follower is False
    assert graph["lm_head"].tied_to == "model.embed_tokens"
    assert graph["lm_head"] not in graph.quantizable()


def test_a_tie_takes_the_strictest_floor_of_the_roles_it_serves() -> None:
    """One tensor, two jobs, so it must satisfy the harder one.

    ``embed_tokens`` floors at 4 bits and ``lm_head`` at 8. Tied, there is a single
    tensor and it produces every logit, so 8 is the answer. Taking the
    representative's own floor would resolve this by ``named_modules`` order --
    embedding first, therefore 4 bits -- and quietly halve the precision of the
    output projection of 27% of the model.
    """
    tied = classify_model(Qwen3_5ForCausalLM(tie=True))
    shared = tied["model.embed_tokens"]
    assert shared.role is ModuleRole.EMBEDDING
    assert shared.tied_roles == (ModuleRole.LM_HEAD,)
    assert shared.floor_bits == 8

    untied = classify_model(Qwen3_5ForCausalLM(tie=False))
    assert untied["model.embed_tokens"].tied_roles == ()
    assert untied["model.embed_tokens"].floor_bits == 4  # free to go lower alone


def test_untied_model_reports_no_tie_and_counts_both() -> None:
    graph = classify_model(Qwen3_5ForCausalLM(tie=False))
    assert graph.tied_groups == ()
    assert graph.unique_params() == sum(m.num_params for m in graph.quantizable())
    assert graph["lm_head"].tied_to is None


def test_floor_cost_exposes_an_unsatisfiable_budget() -> None:
    """The bug-4 diagnostic, on a model where it genuinely bites.

    With the tie resolved to the stricter LM-head floor of 8 bits, 27% of the
    parameters would demand 8 bits and the floors alone exceed a 3-bit target. The
    graph has to be able to say so -- an allocator that discovers this by silently
    returning the floor map is how the supplement's headline config stopped
    depending on its own scores.
    """
    graph = classify_model(Qwen3_5ForCausalLM())
    total = graph.total_params()

    # The conclusion only transfers to the real model if the fixture's size mix
    # does. Guard the mix, not just the verdict.
    share = {
        "embed": graph["model.embed_tokens"].num_params / total,
        "mlp": sum(
            m.num_params
            for m in graph.quantizable()
            if m.role in {ModuleRole.MLP_GATE, ModuleRole.MLP_UP, ModuleRole.MLP_DOWN}
        )
        / total,
    }
    assert 0.25 < share["embed"] < 0.29, share  # real model: 27.0%
    assert 0.45 < share["mlp"] < 0.51, share  # real model: 48.4%

    avg_floor = graph.floor_cost_bits() / total
    assert avg_floor > 3.0, "expected floors to be unsatisfiable at a 3-bit target"
    assert avg_floor < 8.0


# --------------------------------------------------------------------------
# Structural inference (no plugin involved)
# --------------------------------------------------------------------------


def test_moe_router_is_found_structurally_not_by_name() -> None:
    """``mlp.gate``-style routers, for every MoE family at once.

    No Mixtral plugin is registered. The router is identified by output width
    equal to ``num_local_experts`` plus an ``experts`` sibling, which is why this
    generalises to families that do not exist yet.
    """
    graph = classify_model(MixtralForCausalLM())
    router = graph["layers.0.block_sparse_moe.gate"]
    assert router.role is ModuleRole.MOE_ROUTER
    assert router.source == "structural"
    assert router.floor_bits == 8

    # And the expert weights are still recognised, by name, as expert FFN parts.
    expert = graph["layers.0.block_sparse_moe.experts.0.w1"]
    assert expert.role is ModuleRole.MOE_EXPERT_GATE


def test_a_wide_mlp_is_not_mistaken_for_a_router() -> None:
    """The sibling test is what stops the width coincidence from firing."""

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(512, 8, bias=False)  # width == num_experts

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = MIXTRAL_CFG
            self.mlp = Block()

    graph = classify_model(Model())
    assert graph["mlp.gate_proj"].role is ModuleRole.MLP_GATE


def test_embeddings_are_classified_by_class_and_vocab_width() -> None:
    graph = classify_model(Qwen3_5ForCausalLM())
    embed = graph["model.embed_tokens"]
    assert embed.role is ModuleRole.EMBEDDING
    assert embed.source == "structural"


# --------------------------------------------------------------------------
# Exclusions, overrides, reporting
# --------------------------------------------------------------------------


def test_norms_are_skipped_with_a_recorded_reason() -> None:
    """Excluded by rank, not by a class-name list that goes stale."""
    graph = classify_model(Qwen3_5ForCausalLM())
    assert "model.norm" in graph.skipped
    assert "model.layers.0.input_layernorm" in graph.skipped
    assert "1-D" in graph.skipped["model.norm"]
    assert not [n for n in graph.names if "layernorm" in n]


def test_overrides_win_over_every_other_step() -> None:
    graph = classify_model(
        Qwen3_5ForCausalLM(),
        overrides={"model.layers.*.mlp.down_proj": ModuleRole.MOE_ROUTER},
    )
    assert graph["model.layers.0.mlp.down_proj"].role is ModuleRole.MOE_ROUTER
    assert graph["model.layers.0.mlp.down_proj"].source == "override"
    assert graph["model.layers.0.mlp.up_proj"].role is ModuleRole.MLP_UP


def test_unclassified_modules_get_a_cautious_floor_not_the_minimum() -> None:
    """An unseen architecture must degrade to cautious, never to destroyed."""

    class Weird(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = _Cfg(model_type="something_new", vocab_size=100)
            self.thingamajig = nn.Linear(64, 64, bias=False)

    graph = classify_model(Weird())
    info = graph["thingamajig"]
    assert info.role is ModuleRole.OTHER
    assert info.floor_bits == 4
    assert graph.unclassified() == ("thingamajig",)


def test_report_names_the_tie_and_the_floor_cost() -> None:
    text = classify_model(Qwen3_5ForCausalLM()).report()
    assert "model.embed_tokens == lm_head" in text
    assert "floor cost" in text
    assert "lin_attn.qkv" in text


def test_classification_is_stable_under_a_peft_style_wrapper() -> None:
    """Canonical keys, so the graph and the stats file agree on names."""

    class FakeLora(nn.Module):
        def __init__(self, base: nn.Linear) -> None:
            super().__init__()
            self.base_layer = base
            self.lora_A = nn.ModuleDict({"default": nn.Linear(base.in_features, 4, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(4, base.out_features, bias=False)})

    model = Qwen3_5ForCausalLM()
    layer = model.model.layers[3].self_attn
    layer.q_proj = FakeLora(layer.q_proj)  # type: ignore[assignment]

    graph = classify_model(model)
    assert "model.layers.3.self_attn.q_proj" in graph
    assert graph["model.layers.3.self_attn.q_proj"].role is ModuleRole.ATTN_Q_GATE
    assert not [n for n in graph.names if "base_layer" in n or "lora_" in n]


def test_plugin_registry_lookup_falls_back_to_the_text_config() -> None:
    assert "qwen3_5" in registered_model_types()
    assert plugin_for("qwen3_5") is not None
    assert plugin_for("llama") is None
    # A VLM's outer type with a registered inner type resolves through text_config.
    outer = _Cfg(model_type="some_vlm", text_config=_Cfg(model_type="qwen3_5_text"))
    assert plugin_for("some_vlm", outer) is not None


def test_registering_two_plugins_for_one_architecture_is_refused() -> None:
    """Silent override would make classification depend on import order."""
    from dynquant.graph.registry import register_arch

    with pytest.raises(ValueError, match="already handled by"):

        @register_arch("qwen3_5")
        class Rival:
            name = "rival"

            def role_for(self, ctx: object) -> None:
                return None

            def partitions_for(self, ctx: object, role: object) -> None:
                return None


def test_conv2d_and_conv1d_weights_are_eligible_by_rank() -> None:
    """``ndim >= 2`` admits convolutions without special-casing them."""

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = _Cfg(model_type="x", vocab_size=10)
            self.a = nn.Conv1d(8, 8, 3)
            self.b = nn.Conv2d(3, 8, 2)

    graph = classify_model(Model())
    assert graph["a"].shape == (8, 8, 3)
    assert graph["b"].shape == (8, 3, 2, 2)


def test_graph_keys_are_sorted_for_stable_reports() -> None:
    graph = classify_model(Qwen3_5ForCausalLM())
    assert list(graph.names) == sorted(graph.names)


def test_dtype_and_device_do_not_affect_classification() -> None:
    model = Qwen3_5ForCausalLM().to(torch.float16)
    graph = classify_model(model)
    assert graph["model.layers.3.self_attn.q_proj"].role is ModuleRole.ATTN_Q_GATE
