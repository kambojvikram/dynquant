"""The vocabulary of structural roles a module can play, and their safety floors.

Why roles instead of names
--------------------------
Quantization sensitivity is a property of a tensor's *function*, not its label.
A router that picks 8 experts from 128 is a tiny matrix whose every output feeds
an ``argmax``: perturb it and tokens go to the wrong expert, which is
unrecoverable downstream. An MLP ``down_proj`` is enormous and averages thousands
of contributions, so it absorbs noise gracefully. Bits should follow that, and
the only robust way to know which is which is to look at where the module sits
in the graph.

The supplement's ``_min_bits_floor`` matched substrings and therefore mis-handled
every architecture whose naming it had not seen:

===============================  ==================  ==========================
module                           supplement's floor  what it actually needs
===============================  ==================  ==========================
``mlp.gate`` (MoE router)        3-bit (mlp match)   8-bit -- routing is discrete
``mlp.gate_up_proj`` (Phi-4)     3-bit (mlp match)   4-bit on the gate half
``kv_a_proj_with_mqa`` (MLA)     2-bit (no match)    8-bit -- rank-512 bottleneck
``in_proj`` (Mamba)              2-bit (no match)    4-bit -- carries the SSM state
===============================  ==================  ==========================

Every one of those is a silent quality collapse, not a crash.

This module is pure data -- no torch import -- so it stays cheap and testable.
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

__all__ = [
    "DEFAULT_FLOOR_BITS",
    "NEVER_QUANTIZE",
    "UNQUANTIZED_FLOOR",
    "ModuleRole",
    "RowPartition",
    "role_of_name",
]


class ModuleRole(str, Enum):
    """What a module does in the network.

    ``str``-valued so it serialises straight into JSON manifests and compares
    against plain strings in user-supplied override maps.
    """

    # ---- token in / logits out -------------------------------------------
    EMBEDDING = "embedding"
    LM_HEAD = "lm_head"

    # ---- standard multi-head / grouped-query attention -------------------
    ATTN_Q = "attn.q"
    ATTN_K = "attn.k"
    ATTN_V = "attn.v"
    ATTN_O = "attn.o"
    ATTN_QKV = "attn.qkv"  # fused Q|K|V (Phi-3/4, GPT-NeoX, Falcon)
    ATTN_KV = "attn.kv"  # fused K|V (some GQA implementations)
    ATTN_Q_GATE = "attn.q_gate"
    """Fused Q|output-gate, as produced by ``attn_output_gate: true`` (Qwen3.5).

    Easy to miss and expensive to miss: the tensor is ``[2 * n_heads * head_dim,
    hidden]``, so a classifier that reads the name alone calls it ``attn.q`` and
    quantizes a sigmoid gate as though it were a query projection. The gate
    multiplies the whole attention output, so its error is amplified rather than
    averaged -- the same argument that gives the SwiGLU gate its 4-bit floor."""

    # ---- gated linear attention / DeltaNet (Qwen3-Next, Qwen3.5) ----------
    # A different mechanism from Mamba's SSM despite the family resemblance, and
    # named differently again in every release, so these get their own roles
    # rather than being folded into `ssm.*`.
    LIN_ATTN_QKV = "lin_attn.qkv"  # fused q|k|v into the delta rule
    LIN_ATTN_Z = "lin_attn.z"  # output gate
    LIN_ATTN_A = "lin_attn.a"
    """Decay/alpha projection. Tiny (``[16, hidden]``) and its output is
    exponentiated to form the state decay, so a small absolute error becomes a
    multiplicative error compounding over the whole sequence."""
    LIN_ATTN_B = "lin_attn.b"  # beta / write strength; tiny, same reasoning
    LIN_ATTN_OUT = "lin_attn.out"
    LIN_ATTN_CONV = "lin_attn.conv"  # depthwise conv1d, a handful of taps per channel

    # ---- multi-head latent attention (DeepSeek-V2/V3) --------------------
    # The `_a` projections compress to a low-rank latent (e.g. 5120 -> 512).
    # They are small and every downstream head reads through them, so an error
    # here is amplified across the entire attention block.
    MLA_Q_A = "mla.q_a"
    MLA_Q_B = "mla.q_b"
    MLA_KV_A = "mla.kv_a"
    MLA_KV_B = "mla.kv_b"

    # ---- dense feed-forward ----------------------------------------------
    MLP_GATE = "mlp.gate"  # SwiGLU gate: feeds a nonlinearity, needs headroom
    MLP_UP = "mlp.up"
    MLP_DOWN = "mlp.down"
    MLP_GATE_UP = "mlp.gate_up"  # fused gate|up (Phi-3/4, Gemma)
    MLP_FC = "mlp.fc"  # non-gated FFN (GPT-2/OPT/BERT style)

    # ---- mixture of experts ----------------------------------------------
    MOE_ROUTER = "moe.router"  # discrete top-k decision -- never compress hard
    MOE_EXPERT_GATE = "moe.expert.gate"
    MOE_EXPERT_UP = "moe.expert.up"
    MOE_EXPERT_DOWN = "moe.expert.down"
    MOE_EXPERT_GATE_UP = "moe.expert.gate_up"
    MOE_EXPERT_FC = "moe.expert.fc"
    MOE_SHARED_GATE = "moe.shared.gate"  # always-on expert (DeepSeek, Qwen)
    MOE_SHARED_UP = "moe.shared.up"
    MOE_SHARED_DOWN = "moe.shared.down"

    # ---- state-space / hybrid (Mamba, Jamba, RWKV-ish) -------------------
    SSM_IN = "ssm.in"  # in_proj: produces both the state input and the gate
    SSM_X = "ssm.x"  # x_proj: emits B, C, dt -- tiny and highly sensitive
    SSM_DT = "ssm.dt"  # dt_proj: timestep, exponentiated -> keep precise
    SSM_OUT = "ssm.out"
    SSM_CONV = "ssm.conv"  # depthwise conv1d, tiny

    # ---- vision / multimodal ---------------------------------------------
    VISION_PATCH_EMBED = "vision.patch_embed"
    VISION_ATTN = "vision.attn"
    VISION_MLP = "vision.mlp"
    MM_PROJECTOR = "mm.projector"  # vision->text bridge; small, unforgiving

    # ---- not quantized ---------------------------------------------------
    NORM = "norm"
    BIAS = "bias"
    ROTARY = "rotary"

    # ---- unknown ---------------------------------------------------------
    OTHER = "other"
    """Unclassified. Gets a conservative floor and a loud warning -- never the
    2-bit minimum. An architecture we have not seen must degrade to *cautious*,
    not to *destroyed*."""

    # -- convenience predicates -------------------------------------------

    @property
    def is_quantizable(self) -> bool:
        return self not in NEVER_QUANTIZE

    @property
    def is_attention(self) -> bool:
        return self.value.startswith(("attn.", "mla."))

    @property
    def is_mlp(self) -> bool:
        return self.value.startswith("mlp.")

    @property
    def is_moe(self) -> bool:
        return self.value.startswith("moe.")

    @property
    def is_expert(self) -> bool:
        """True for per-expert weights, which dominate MoE parameter counts."""
        return self.value.startswith("moe.expert.")

    @property
    def is_ssm(self) -> bool:
        return self.value.startswith("ssm.")

    @property
    def is_linear_attention(self) -> bool:
        return self.value.startswith("lin_attn.")

    @property
    def is_vision(self) -> bool:
        return self.value.startswith(("vision.", "mm."))

    @property
    def is_fused(self) -> bool:
        """True if the module concatenates several logical projections by row."""
        return self in _FUSED_PARTS

    @property
    def fused_parts(self) -> tuple[ModuleRole, ...]:
        """The sub-roles a fused module splits into, in row order.

        Row order matters: a fused ``gate_up_proj`` is ``[gate; up]``, so the
        first half of its rows is the SwiGLU gate and gets the gate's floor
        while the second half gets ``up``'s. Assigning one bit-width to the
        whole tensor would either waste bits on ``up`` or starve the gate.
        """
        return _FUSED_PARTS.get(self, ())


NEVER_QUANTIZE: frozenset[ModuleRole] = frozenset(
    {ModuleRole.NORM, ModuleRole.BIAS, ModuleRole.ROTARY}
)
"""Kept in the compute dtype. Together these are well under 0.1% of parameters
and quantizing them buys nothing while risking numerical blowups (a LayerNorm
scale near zero, an inverse-frequency table that must stay monotone)."""


_FUSED_PARTS: dict[ModuleRole, tuple[ModuleRole, ...]] = {
    ModuleRole.ATTN_QKV: (ModuleRole.ATTN_Q, ModuleRole.ATTN_K, ModuleRole.ATTN_V),
    ModuleRole.ATTN_KV: (ModuleRole.ATTN_K, ModuleRole.ATTN_V),
    # The gate half is scored like an MLP gate, for the same reason: it is a
    # multiplicative mask on the block's output, so error there is amplified.
    ModuleRole.ATTN_Q_GATE: (ModuleRole.ATTN_Q, ModuleRole.MLP_GATE),
    ModuleRole.MLP_GATE_UP: (ModuleRole.MLP_GATE, ModuleRole.MLP_UP),
    ModuleRole.MOE_EXPERT_GATE_UP: (ModuleRole.MOE_EXPERT_GATE, ModuleRole.MOE_EXPERT_UP),
    ModuleRole.LIN_ATTN_QKV: (ModuleRole.ATTN_Q, ModuleRole.ATTN_K, ModuleRole.ATTN_V),
}


class RowPartition(NamedTuple):
    """A contiguous row range of a fused weight and the role it plays.

    Row-partitioning is what lets DynQuant give a fused ``qkv_proj`` different
    precision per projection without unfusing the model. It works because
    group-wise quantization already stores one scale per (row, group): rows are
    independent, so a bit-width that varies by row costs no extra machinery in
    the format -- only a ``row_bits`` side table.

    ``stop`` is exclusive, matching Python slicing.
    """

    role: ModuleRole
    start: int
    stop: int

    @property
    def num_rows(self) -> int:
        return self.stop - self.start

    def __repr__(self) -> str:  # pragma: no cover -- debugging aid
        return f"RowPartition({self.role.value}, {self.start}:{self.stop})"


# --------------------------------------------------------------------------
# Default safety floors
# --------------------------------------------------------------------------

UNQUANTIZED_FLOOR = 16
"""Floor value meaning "leave this tensor in the compute dtype".

Not a packable width -- it is deliberately outside :data:`BIT_OPTIONS`, so a floor
of 16 is unsatisfiable by any packing and the allocator has to route the tensor
around quantization instead of spending 8 bits on it.

Used for weights where quantization cannot pay: a depthwise conv1d with a handful
of taps per channel stores one fp16 scale per 4 values, so the scale dominates the
payload, and there is no redundancy across 4 numbers to exploit. Such tensors stay
in the *graph* and in the size accounting -- they occupy 16 bits per parameter on
disk and the manifest's average must say so -- they simply are not candidates for a
bit-width decision."""

DEFAULT_FLOOR_BITS: dict[ModuleRole, int] = {
    # Shared vocabulary tensors. Errors here hit every token.
    ModuleRole.EMBEDDING: 4,
    ModuleRole.LM_HEAD: 8,
    # Attention. Q/K errors compound through the softmax.
    ModuleRole.ATTN_Q: 4,
    ModuleRole.ATTN_K: 4,
    ModuleRole.ATTN_V: 4,
    ModuleRole.ATTN_O: 4,
    ModuleRole.ATTN_QKV: 4,
    ModuleRole.ATTN_KV: 4,
    ModuleRole.ATTN_Q_GATE: 4,  # refined per-partition; see fused_parts
    # Gated linear attention. The bulk (`qkv`, `z`, `out`) behaves like ordinary
    # attention. `a` and `b` are ~0.03% of the block and feed an exponential, so
    # 8 bits there is free insurance against a decay term drifting.
    ModuleRole.LIN_ATTN_QKV: 4,
    ModuleRole.LIN_ATTN_Z: 4,
    ModuleRole.LIN_ATTN_A: 8,
    ModuleRole.LIN_ATTN_B: 8,
    ModuleRole.LIN_ATTN_OUT: 4,
    ModuleRole.LIN_ATTN_CONV: UNQUANTIZED_FLOOR,
    # MLA. The `_a` bottlenecks are ~1% of the block's parameters and carry all
    # of its information, so paying 8 bits for them costs almost nothing.
    ModuleRole.MLA_Q_A: 8,
    ModuleRole.MLA_KV_A: 8,
    ModuleRole.MLA_Q_B: 4,
    ModuleRole.MLA_KV_B: 4,
    # Dense FFN. The paper's ablation (Sec. 4.4) shows the SwiGLU gate is the
    # one FFN tensor that will not tolerate 3 bits: it multiplies the up-branch,
    # so its error is amplified rather than averaged.
    ModuleRole.MLP_GATE: 4,
    ModuleRole.MLP_UP: 3,
    ModuleRole.MLP_DOWN: 3,
    ModuleRole.MLP_GATE_UP: 4,  # refined per-partition; see fused_parts
    ModuleRole.MLP_FC: 3,
    # MoE. The router is the whole game: it is O(hidden x num_experts) -- often
    # under 0.01% of the model -- and a wrong argmax sends a token through the
    # wrong 500M parameters. 8 bits here is free.
    ModuleRole.MOE_ROUTER: 8,
    ModuleRole.MOE_EXPERT_GATE: 4,
    ModuleRole.MOE_EXPERT_UP: 2,
    ModuleRole.MOE_EXPERT_DOWN: 2,
    ModuleRole.MOE_EXPERT_GATE_UP: 4,
    ModuleRole.MOE_EXPERT_FC: 2,
    # Shared experts run for every token, so they behave like a dense FFN.
    ModuleRole.MOE_SHARED_GATE: 4,
    ModuleRole.MOE_SHARED_UP: 3,
    ModuleRole.MOE_SHARED_DOWN: 3,
    # SSM. x_proj/dt_proj feed an exponential; conv1d and dt are negligible in
    # size. in_proj/out_proj are the bulk.
    ModuleRole.SSM_IN: 4,
    ModuleRole.SSM_X: 8,
    ModuleRole.SSM_DT: 8,
    ModuleRole.SSM_OUT: 4,
    # Mamba's conv1d is the same tensor shape as a DeltaNet's -- `[channels, 1, 4]`
    # -- so it gets the same answer. Anything else would make the floor depend on
    # which block the identical tensor happens to sit in.
    ModuleRole.SSM_CONV: UNQUANTIZED_FLOOR,
    # Vision towers are a few percent of a VLM's parameters but carry all the
    # image information; the projector is tiny and has no redundancy at all.
    ModuleRole.VISION_PATCH_EMBED: 8,
    ModuleRole.VISION_ATTN: 4,
    ModuleRole.VISION_MLP: 4,
    ModuleRole.MM_PROJECTOR: 8,
    # Unknown: cautious, and the CLI names every module that lands here.
    ModuleRole.OTHER: 4,
}
"""Minimum bits per role under the default policy.

These are *defaults*, and under the default preset they are soft: if the budget
cannot accommodate them the allocator lowers the least valuable ones and reports
exactly which, rather than silently abandoning the score-driven allocation the
way the supplement's ``allocate_bits_pareto`` did. ``--preset paper-3.15`` makes
them hard, to reproduce published numbers.
"""

# MoE experts start lower than dense FFN on purpose: with 128 experts and top-8
# routing, any given expert sees ~6% of tokens, so per-expert precision matters
# far less than the router's correctness. This is where the parameter budget of
# a large MoE actually goes, and where compression has the most room.


# --------------------------------------------------------------------------
# Legacy name-based classification (last-resort fallback only)
# --------------------------------------------------------------------------

# Ordered longest/most-specific first: `gate_up_proj` must win over `gate_proj`,
# which must win over a bare `gate`. Dict order is the match order.
_NAME_PATTERNS: tuple[tuple[str, ModuleRole], ...] = (
    # fused first
    ("gate_up_proj", ModuleRole.MLP_GATE_UP),
    ("gate_proj", ModuleRole.MLP_GATE),
    ("up_proj", ModuleRole.MLP_UP),
    ("down_proj", ModuleRole.MLP_DOWN),
    ("qkv_proj", ModuleRole.ATTN_QKV),
    ("query_key_value", ModuleRole.ATTN_QKV),
    ("kv_proj", ModuleRole.ATTN_KV),
    # MLA before plain q/k/v so `q_a_proj` is not read as `q_proj`
    ("kv_a_proj_with_mqa", ModuleRole.MLA_KV_A),
    ("kv_a_proj", ModuleRole.MLA_KV_A),
    ("kv_b_proj", ModuleRole.MLA_KV_B),
    ("q_a_proj", ModuleRole.MLA_Q_A),
    ("q_b_proj", ModuleRole.MLA_Q_B),
    ("q_proj", ModuleRole.ATTN_Q),
    ("k_proj", ModuleRole.ATTN_K),
    ("v_proj", ModuleRole.ATTN_V),
    ("o_proj", ModuleRole.ATTN_O),
    ("out_proj", ModuleRole.ATTN_O),
    # Falcon / GPT-NeoX spell the FFN with `dense_*`, and those must be read as
    # FFN rather than as attention output -- so they precede the bare `dense`.
    ("dense_h_to_4h", ModuleRole.MLP_UP),
    ("dense_4h_to_h", ModuleRole.MLP_DOWN),
    ("dense", ModuleRole.ATTN_O),
    # SSM
    ("in_proj", ModuleRole.SSM_IN),
    ("x_proj", ModuleRole.SSM_X),
    ("dt_proj", ModuleRole.SSM_DT),
    ("conv1d", ModuleRole.SSM_CONV),
    # heads / embeddings
    ("lm_head", ModuleRole.LM_HEAD),
    ("embed_tokens", ModuleRole.EMBEDDING),
    ("embed_out", ModuleRole.LM_HEAD),
    ("wte", ModuleRole.EMBEDDING),
    # non-gated FFN
    ("fc1", ModuleRole.MLP_UP),
    ("fc2", ModuleRole.MLP_DOWN),
    ("c_fc", ModuleRole.MLP_UP),
    ("c_proj", ModuleRole.MLP_DOWN),
    # norms
    ("layernorm", ModuleRole.NORM),
    ("layer_norm", ModuleRole.NORM),
    ("rmsnorm", ModuleRole.NORM),
    ("_norm", ModuleRole.NORM),
    ("rotary_emb", ModuleRole.ROTARY),
)

# A bare final component of exactly one of these is a MoE router, never a
# SwiGLU gate. Checked on the *last* dotted component so `mlp.gate_proj` is
# unaffected.
_ROUTER_LEAF_NAMES: frozenset[str] = frozenset(
    {
        "gate",  # Mixtral, Qwen3-MoE, OLMoE, DeepSeek
        "router",  # GPT-OSS, Switch
        "wg",  # some Megablocks-derived implementations
        "w_gate",
        "gating",
        "switch",
        "shared_expert_gate",  # Qwen2-MoE: sigmoid weight on the always-on expert
    }
)

# Names that are only meaningful as a *whole* leaf component. Substring matching
# on one- and two-character names would false-positive constantly ("w1" appears
# inside plenty of unrelated identifiers), so these are matched exactly.
_LEAF_EXACT: dict[str, ModuleRole] = {
    # Mixtral / Megablocks experts: w1 gates, w3 lifts, w2 projects back down.
    # Without these, every expert weight in a Mixtral-family model -- which is
    # ~95% of its parameters -- falls through to OTHER.
    "w1": ModuleRole.MLP_GATE,
    "w3": ModuleRole.MLP_UP,
    "w2": ModuleRole.MLP_DOWN,
    # T5 / UL2-style FFN.
    "wi_0": ModuleRole.MLP_GATE,
    "wi_1": ModuleRole.MLP_UP,
    "wi": ModuleRole.MLP_FC,
    "wo": ModuleRole.MLP_DOWN,
    # Vision patch embeddings appear under several spellings.
    "patch_embed": ModuleRole.VISION_PATCH_EMBED,
    "patch_embedding": ModuleRole.VISION_PATCH_EMBED,
    "proj_in": ModuleRole.VISION_PATCH_EMBED,
    # Norms whose leaf carries no "norm" substring the pattern list would catch.
    # `model.norm` -- the final norm in every Llama-family model -- is the one
    # that matters: unclassified it would land in OTHER and be quantized, and a
    # quantized final norm is a numerically unstable divide applied to every token.
    "norm": ModuleRole.NORM,
    "ln_f": ModuleRole.NORM,  # GPT-2
    "ln_1": ModuleRole.NORM,
    "ln_2": ModuleRole.NORM,
    "final_norm": ModuleRole.NORM,
}


def role_of_name(name: str) -> ModuleRole:
    """Best-effort role from a module's dotted name alone.

    This is the **last resort** in the resolution chain
    (user override -> architecture plugin -> structural inference -> here).
    Names lie: ``mlp.gate`` means "SwiGLU gate" in Llama and "expert router" in
    Mixtral, and no amount of string matching can tell them apart. Prefer
    :func:`dynquant.graph.classify.classify_module`, which inspects the module
    tree and the model config.

    Kept because it is the only classifier available when all you have is a
    stats file -- e.g. ``dynquant inspect`` on a JSON with no model present.
    """
    lowered = name.lower()
    parts = lowered.split(".")
    leaf = parts[-1] if parts else lowered

    # A MoE router is identified by its leaf name, before any substring pass,
    # because `gate` is a substring of nothing useful but a prefix of `gate_proj`.
    if leaf in _ROUTER_LEAF_NAMES:
        return ModuleRole.MOE_ROUTER

    in_experts = "experts." in lowered or leaf.startswith("expert")
    is_shared = "shared_expert" in lowered or "shared_mlp" in lowered
    in_vision = any(
        tag in lowered for tag in ("vision_model", "visual.", "vision_tower", "image_encoder")
    )

    leaf_role = _LEAF_EXACT.get(leaf)
    if leaf_role is not None:
        return _contextualise(
            leaf_role, in_experts=in_experts, is_shared=is_shared, in_vision=in_vision
        )

    for pattern, role in _NAME_PATTERNS:
        if pattern not in lowered:
            continue
        return _contextualise(role, in_experts=in_experts, is_shared=is_shared, in_vision=in_vision)

    if "multi_modal_projector" in lowered or "mm_projector" in lowered:
        return ModuleRole.MM_PROJECTOR
    return ModuleRole.OTHER


def _contextualise(
    role: ModuleRole, *, in_experts: bool, is_shared: bool, in_vision: bool
) -> ModuleRole:
    """Re-map an FFN role based on where in the graph the module sits."""
    if in_vision:
        if role.is_attention:
            return ModuleRole.VISION_ATTN
        if role.is_mlp:
            return ModuleRole.VISION_MLP
        return role
    if is_shared:
        return _SHARED_REMAP.get(role, role)
    if in_experts:
        return _EXPERT_REMAP.get(role, role)
    return role


_EXPERT_REMAP: dict[ModuleRole, ModuleRole] = {
    ModuleRole.MLP_GATE: ModuleRole.MOE_EXPERT_GATE,
    ModuleRole.MLP_UP: ModuleRole.MOE_EXPERT_UP,
    ModuleRole.MLP_DOWN: ModuleRole.MOE_EXPERT_DOWN,
    ModuleRole.MLP_GATE_UP: ModuleRole.MOE_EXPERT_GATE_UP,
    ModuleRole.MLP_FC: ModuleRole.MOE_EXPERT_FC,
}

_SHARED_REMAP: dict[ModuleRole, ModuleRole] = {
    ModuleRole.MLP_GATE: ModuleRole.MOE_SHARED_GATE,
    ModuleRole.MLP_UP: ModuleRole.MOE_SHARED_UP,
    ModuleRole.MLP_DOWN: ModuleRole.MOE_SHARED_DOWN,
}
