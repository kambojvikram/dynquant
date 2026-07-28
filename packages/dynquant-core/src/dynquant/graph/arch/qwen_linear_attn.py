"""Qwen hybrid linear-attention models: Qwen3.5 and Qwen3-Next.

These interleave gated-DeltaNet layers with ordinary full-attention layers
(Qwen3.5-2B: ``layer_types`` = three ``linear_attention`` then one
``full_attention``, 18 and 6 of 24). Two things here defeat generic
classification, and both were verified against a loaded checkpoint rather than
inferred from the config:

**The projection names differ between releases.** Qwen3-Next fuses into
``in_proj_qkvz`` and ``in_proj_ba``; Qwen3.5 splits the same computation into four
tensors, ``in_proj_qkv``, ``in_proj_z``, ``in_proj_b``, ``in_proj_a``. Assuming
either naming breaks the other, and the fallback name matcher reads ``in_proj`` as
Mamba's ``ssm.in`` -- close enough to look right and wrong enough to matter, since
it puts the exponentiated decay projection on a 4-bit floor.

**``q_proj`` is not a Q projection.** With ``attn_output_gate: true`` the tensor is
``[2 * n_heads * head_dim, hidden]`` holding ``[Q; output_gate]``. On Qwen3.5-2B
that is ``(4096, 2048)`` where a plain Q projection would be ``(2048, 2048)``. The
gate multiplies the entire attention output, so it needs gate-grade precision
while the Q half does not, which is exactly what a row partition expresses.

The ``a``/``b`` projections are ``[16, 2048]`` -- 0.03% of a layer -- and their
outputs are exponentiated into the state decay, so an absolute error there becomes
a multiplicative error compounding along the sequence. Keeping them at 8 bits costs
about 1.2 Mbit across the whole model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dynquant.graph.registry import ModuleContext, register_arch
from dynquant.graph.roles import ModuleRole, RowPartition

if TYPE_CHECKING:
    pass

__all__ = ["QwenLinearAttentionPlugin"]

# Leaf name -> role, for modules inside a gated-DeltaNet block. Both spellings are
# listed because both ship: `_qkvz`/`_ba` is Qwen3-Next, the split four is Qwen3.5.
_LINEAR_ATTN_LEAVES: dict[str, ModuleRole] = {
    "in_proj_qkv": ModuleRole.LIN_ATTN_QKV,
    "in_proj_qkvz": ModuleRole.LIN_ATTN_QKV,
    "in_proj_z": ModuleRole.LIN_ATTN_Z,
    "in_proj_a": ModuleRole.LIN_ATTN_A,
    "in_proj_b": ModuleRole.LIN_ATTN_B,
    "in_proj_ba": ModuleRole.LIN_ATTN_A,
    "out_proj": ModuleRole.LIN_ATTN_OUT,
    "conv1d": ModuleRole.LIN_ATTN_CONV,
}

_ATTENTION_ANCESTORS = ("GatedDeltaNet", "LinearAttention", "DeltaNet")


@register_arch("qwen3_5", "qwen3_5_text", "qwen3_next")
class QwenLinearAttentionPlugin:
    """Roles and row partitions for Qwen hybrid linear-attention text towers."""

    name = "qwen_linear_attn"

    def role_for(self, ctx: ModuleContext) -> ModuleRole | None:
        # Ancestry, not the name, decides which block we are in: `out_proj` means
        # `lin_attn.out` inside a DeltaNet and `attn.o` inside self-attention, and
        # only the ancestor class distinguishes them.
        if ctx.in_ancestor(*_ATTENTION_ANCESTORS):
            return _LINEAR_ATTN_LEAVES.get(ctx.leaf)

        if ctx.leaf == "q_proj" and self._q_is_gated(ctx):
            return ModuleRole.ATTN_Q_GATE

        # Everything else in these models -- q/k/v/o, the dense SwiGLU MLP,
        # embeddings -- is conventional, so defer rather than restate it.
        return None

    def partitions_for(
        self, ctx: ModuleContext, role: ModuleRole
    ) -> tuple[RowPartition, ...] | None:
        if role is ModuleRole.ATTN_Q_GATE:
            heads, head_dim = ctx.cfg("num_attention_heads"), ctx.cfg("head_dim")
            if not heads or not head_dim:
                return None
            split = heads * head_dim
            if ctx.out_features != 2 * split:
                return None
            # Row order is [Q; gate]: the model computes q_proj(x), splits on the
            # last dim, and applies sigmoid to the second half.
            return (
                RowPartition(ModuleRole.ATTN_Q, 0, split),
                RowPartition(ModuleRole.MLP_GATE, split, ctx.out_features),
            )

        if role is ModuleRole.LIN_ATTN_QKV:
            return self._deltanet_qkv_partitions(ctx)

        return None

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _q_is_gated(ctx: ModuleContext) -> bool:
        """True when ``q_proj`` carries a fused output gate.

        Checks the shape as well as the flag. The flag alone would mis-partition a
        checkpoint whose config claims the gate but whose tensor does not have the
        doubled width -- and a wrong partition boundary is worse than none, because
        it hands the gate's bit-width to the wrong rows.
        """
        if not ctx.flag("attn_output_gate"):
            return False
        heads, head_dim = ctx.cfg("num_attention_heads"), ctx.cfg("head_dim")
        return bool(heads and head_dim and ctx.out_features == 2 * heads * head_dim)

    @staticmethod
    def _deltanet_qkv_partitions(ctx: ModuleContext) -> tuple[RowPartition, ...] | None:
        """Split ``in_proj_qkv`` into its q, k and v row blocks.

        Widths come from the *linear* head config (``linear_num_key_heads`` etc.),
        which is separate from the full-attention head config -- on Qwen3.5-2B the
        linear heads are 16x128 while the full-attention heads are 8x256. Returns
        ``None`` unless the three blocks account for exactly the tensor's rows, so a
        naming variant that fuses ``z`` in as well (Qwen3-Next's ``in_proj_qkvz``)
        declines instead of mis-slicing.
        """
        k_heads = ctx.cfg("linear_num_key_heads")
        v_heads = ctx.cfg("linear_num_value_heads")
        k_dim = ctx.cfg("linear_key_head_dim")
        v_dim = ctx.cfg("linear_value_head_dim")
        if not (k_heads and v_heads and k_dim and v_dim):
            return None

        q_width = k_width = k_heads * k_dim
        v_width = v_heads * v_dim
        if q_width + k_width + v_width != ctx.out_features:
            return None
        return (
            RowPartition(ModuleRole.ATTN_Q, 0, q_width),
            RowPartition(ModuleRole.ATTN_K, q_width, q_width + k_width),
            RowPartition(ModuleRole.ATTN_V, q_width + k_width, ctx.out_features),
        )
