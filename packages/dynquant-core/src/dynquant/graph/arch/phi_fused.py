"""Phi-3 / Phi-4 family: row partitions for the two fused projections.

Phi is the architecture where fusion costs the most to ignore. It fuses both of
the tensors a decoder layer has to fuse -- ``self_attn.qkv_proj`` and
``mlp.gate_up_proj`` -- and on Phi-4-mini those two are **60% of the quantizable
parameters** (40% gate_up, 20% qkv). Without partitions every one of those rows
takes a single width, which means the whole 60% sits at the strictest floor its
sub-roles need: 4 bits for ``gate_up`` because the SwiGLU gate demands 4, even
though the ``up`` half would have been perfectly happy at 3.

Name-based classification already gets the *roles* right here -- ``qkv_proj`` and
``gate_up_proj`` are unambiguous strings -- so this plugin declines
:meth:`role_for` entirely and exists only to answer the question the generic path
is forbidden to guess at: where the row boundaries are.

Both orders are read off the modelling code rather than assumed, because the cost
of being wrong is silent. ``Phi3MLP.forward`` does::

    up_states = self.gate_up_proj(hidden_states)
    gate, up_states = up_states.chunk(2, dim=-1)
    up_states = up_states * self.activation_fn(gate)

so the gate is the *first* ``intermediate_size`` rows. ``Phi3Attention.forward``
slices ``[0, query_pos)``, ``[query_pos, +kv)``, ``[.., end)``, so QKV is
contiguous ``[q; k; v]`` and not interleaved per head.

Getting that backwards would hand the gate's bit-width to the up-projection and
vice versa -- a checkpoint that loads, runs, and is quietly worse. Hence every
partition here is guarded by an arithmetic check against ``out_features``: if the
widths do not account for exactly the tensor's rows, the plugin declines and the
tensor keeps one width, which is merely suboptimal rather than wrong.
"""

from __future__ import annotations

from dynquant.graph.registry import ModuleContext, register_arch
from dynquant.graph.roles import ModuleRole, RowPartition

__all__ = ["PhiFusedPlugin"]


@register_arch("phi3")
class PhiFusedPlugin:
    """Row partitions for Phi-3/Phi-4's fused QKV and gate-up projections."""

    name = "phi_fused"

    def role_for(self, ctx: ModuleContext) -> ModuleRole | None:
        """Always defer.

        Phi spells every projection unambiguously, so there is nothing here the
        generic path gets wrong. Restating it would only create a second place
        for the answer to drift.
        """
        return None

    def partitions_for(
        self, ctx: ModuleContext, role: ModuleRole
    ) -> tuple[RowPartition, ...] | None:
        if role is ModuleRole.ATTN_QKV:
            return self._qkv_partitions(ctx)
        if role is ModuleRole.MLP_GATE_UP:
            return self._gate_up_partitions(ctx)
        return None

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _qkv_partitions(ctx: ModuleContext) -> tuple[RowPartition, ...] | None:
        """Split ``qkv_proj`` into ``[q; k; v]`` row blocks.

        Under GQA the three blocks are different sizes -- on Phi-4-mini the query
        block is 8x the key and value blocks -- so an even three-way split would
        put the boundaries in the wrong place while still summing correctly.
        """
        heads = ctx.cfg("num_attention_heads")
        kv_heads = ctx.cfg("num_key_value_heads") or heads
        if not heads:
            return None

        hidden = ctx.cfg("hidden_size")
        head_dim = ctx.cfg("head_dim") or (hidden // heads if hidden else None)
        if not head_dim:
            return None

        q_width = heads * head_dim
        kv_width = kv_heads * head_dim
        if q_width + 2 * kv_width != ctx.out_features:
            return None
        return (
            RowPartition(ModuleRole.ATTN_Q, 0, q_width),
            RowPartition(ModuleRole.ATTN_K, q_width, q_width + kv_width),
            RowPartition(ModuleRole.ATTN_V, q_width + kv_width, ctx.out_features),
        )

    @staticmethod
    def _gate_up_partitions(ctx: ModuleContext) -> tuple[RowPartition, ...] | None:
        """Split ``gate_up_proj`` into ``[gate; up]`` halves.

        The split is checked against ``intermediate_size`` rather than taken as
        ``out_features // 2``. Halving would produce a plausible-looking partition
        for a tensor whose row count is even but whose real layout is something
        else, which is precisely the failure this guard exists to prevent.
        """
        intermediate = ctx.cfg("intermediate_size")
        if not intermediate or 2 * intermediate != ctx.out_features:
            return None
        return (
            RowPartition(ModuleRole.MLP_GATE, 0, intermediate),
            RowPartition(ModuleRole.MLP_UP, intermediate, ctx.out_features),
        )
