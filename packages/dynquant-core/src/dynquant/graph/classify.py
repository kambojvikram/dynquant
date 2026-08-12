"""Assign a :class:`ModuleRole` to every quantizable weight in a live model.

Resolution order, first match wins::

    1. user override map        explicit, fnmatch over canonical names
    2. architecture plugin      registered for config.model_type
    3. structural inference     reads the module tree and the config
    4. name substrings          `roles.role_of_name`, last resort

Steps 1-3 look at what a module *is*. Step 4 looks at what it is *called*, which
is why it is last: the supplement had only step 4, and names lie in ways that are
specifically dangerous. ``mlp.gate`` is a SwiGLU gate in Llama and an expert
router in Mixtral -- 4-bit-ish versus never-below-8-bit, decided by identical
strings. Structural inference asks instead whether the module's output width
equals ``num_experts`` and whether it has an ``experts`` sibling, which is true of
every MoE family at once, including ones not yet released.

Why a graph and not a dict
--------------------------
Two facts about a model are invisible to per-module classification and both change
the answer:

**Ties.** ``embed_tokens`` and ``lm_head`` can be the same tensor. Then there is
one weight, one bit-width, and one contribution to the parameter budget. Summing
``weight.numel()`` per module double-counts it -- on Qwen3.5-2B by 508.6M, which is
27% of the model, so a budget computed that way overshoots by 27% and the
allocator hands back a map that does not fit. :class:`ModelGraph` resolves ties by
``id(weight)`` and reports one representative plus its followers.

**Fusion.** A fused ``[Q; output_gate]`` or ``[gate; up]`` tensor holds two
logical projections with different sensitivities. Group-wise quantization already
stores one scale per (row, group), so rows are independent and a bit-width that
varies by row costs nothing in the format -- but only if something knows where the
boundary is. That "something" needs the config, not the module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any

from dynquant._logging import get_logger
from dynquant.constants import BIT_OPTIONS, DEFAULT_GROUP_SIZE
from dynquant.quant.pack import stored_bits

from .experts import IN_OUT, OUT_IN, bank_orientation, batched_expert_params
from .naming import canonical_name
from .registry import ModuleContext, plugin_for
from .roles import (
    DEFAULT_FLOOR_BITS,
    UNQUANTIZED_FLOOR,
    ModuleRole,
    RowPartition,
    role_of_name,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from torch import nn

__all__ = ["ModelGraph", "ModuleInfo", "SkippedTensor", "classify_model"]

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SkippedTensor:
    """A weight the graph refused, and how much of the file it still occupies.

    Refusing to quantize a tensor is a decision about *bits*, not about existence. Every
    one of these is written to the checkpoint, so a budget that does not know their size
    targets a number smaller than the folder it produces, and an average-bits figure
    computed without their parameters is an average over a subset of the file. That is a
    rounding error at one end of the range and not at the other: LFM2.5-8B-A1B's norms and
    router biases come to 205,056 bytes against 4.4 GB, while a batched expert bank refused
    for orientation is 91.5% of the same model.

    Priced at :data:`UNQUANTIZED_FLOOR` rather than at the width each is actually stored
    at, which is exact for the norms and half the truth for the 22 fp32 buffers beside
    them -- 1,408 bytes on that model. See :func:`_persistent_buffers`.

    Carries a count rather than a shape, because the count is what a budget needs and the
    shape is usually already in :attr:`reason`.
    """

    reason: str
    num_params: int
    tied_to: str | None = None
    """Set when an earlier name owns the same tensor, so :meth:`ModelGraph.skipped_params`
    counts it once. Two modules sharing one norm is unusual; counting a shared tensor twice
    is the error that made a tied embedding 27% of a model twice over, and the guard is
    four lines."""


@dataclass(frozen=True, slots=True)
class ModuleInfo:
    """One quantizable weight, and everything needed to decide its bit-width."""

    name: str
    """Canonical name -- the spelling the weight has in the bare, merged model,
    which is also the key the stats file uses. One name, whatever wrapped it."""

    role: ModuleRole
    module_type: str
    shape: tuple[int, ...]
    num_params: int
    source: str
    """Which resolution step decided the role: ``override``, ``plugin:<name>``,
    ``structural``, or ``name``. Carried into the manifest so a surprising bit
    assignment can be traced back to the decision that produced it."""

    partitions: tuple[RowPartition, ...] = ()
    """Row ranges with distinct sub-roles, empty for an unfused tensor."""

    tied_to: str | None = None
    """Set on the *followers* of a tie, naming the representative. A follower has
    no independent bit-width and contributes no parameters to the budget."""

    tied_roles: tuple[ModuleRole, ...] = ()
    """Roles of the other names this same tensor answers to, set on a tie
    representative.

    A tied tensor has to satisfy the strictest of its roles at once, because there
    is only one of it. ``embed_tokens`` floors at 4 bits and ``lm_head`` at 8; when
    they are one tensor it needs 8, since the same numbers that look up an
    embedding also produce every logit. Taking the representative's own floor would
    make the floor of 27% of Qwen3.5-2B depend on which name ``named_modules``
    happened to yield first."""

    @property
    def is_tied_follower(self) -> bool:
        return self.tied_to is not None

    @property
    def floor_bits(self) -> int:
        """Minimum bits under the default policy.

        For a fused tensor this is the strictest of its partitions' floors: the
        tensor as a whole cannot go below what its most sensitive rows need, even
        though individual row blocks may end up higher. For a tied tensor it is
        likewise the strictest over every role it serves.
        """
        if self.partitions:
            floor = max(DEFAULT_FLOOR_BITS.get(p.role, 4) for p in self.partitions)
        else:
            floor = DEFAULT_FLOOR_BITS.get(self.role, 4)
        for role in self.tied_roles:
            floor = max(floor, DEFAULT_FLOOR_BITS.get(role, 4))
        return floor

    @property
    def pays_for_itself(self) -> bool:
        """Whether quantizing this tensor is cheaper than storing it dense.

        Usually so obvious it need not be asked, and for one shape in the phase-3
        set it is false. Gemma-3's ``vision_tower...patch_embedding`` has shape
        ``[1152, 3, 14, 14]``, which folds to 48384 rows of **14** columns; a group
        of 128 pads that row by 9.1x and then charges a scale and an offset on top.
        Measured through :class:`~dynquant.quant.tensor.QuantTensor`: 20.6 bits per
        weight at 2-bit, 38.9 at 4-bit, 75.4 at 8-bit -- against 16 for leaving it
        alone. Every width is worse than fp16, and the wide ones are worse by more,
        so no amount of budget pressure makes quantizing it the right move.

        :data:`~dynquant.graph.roles.UNQUANTIZED_FLOOR` already exists for exactly
        this failure ("a depthwise conv1d with a handful of taps per channel stores
        one fp16 scale per 4 values"), but it is keyed on *role*, and a role cannot
        see a shape. A short trailing dimension is a property of the tensor, and
        every architecture grows its own: this one is a patch embedding, the next
        will be something else.

        Priced at :data:`~dynquant.constants.DEFAULT_GROUP_SIZE` and at the
        narrowest width on offer, which is the same "under the default policy"
        scope :attr:`floor_bits` carries. A policy with a smaller group would pad
        less and could make some of these worth quantizing again; leaving them
        dense then costs a few unnecessary bits on a tiny tensor, which is the
        direction to be wrong in.
        """
        if len(self.shape) < 2:  # pragma: no cover -- rank-1 tensors have no role
            return True
        columns = self.shape[-1]
        if columns <= 0:  # pragma: no cover -- an empty weight
            return True
        cheapest = stored_bits(1, columns, min(BIT_OPTIONS), group_size=DEFAULT_GROUP_SIZE)
        return cheapest < columns * UNQUANTIZED_FLOOR

    @property
    def is_quantizable(self) -> bool:
        return (
            self.role.is_quantizable
            and self.floor_bits < UNQUANTIZED_FLOOR
            and self.pays_for_itself
        )


@dataclass(frozen=True, slots=True)
class ModelGraph:
    """Every quantizable module in a model, classified, with ties resolved."""

    modules: dict[str, ModuleInfo]
    model_type: str
    architectures: tuple[str, ...] = ()
    tied_groups: tuple[tuple[str, ...], ...] = ()
    skipped: dict[str, SkippedTensor] = field(default_factory=dict)
    """Weights deliberately not quantized, mapped to the reason and their size. Norms,
    biases and refused expert banks land here. Recorded rather than dropped so
    ``dynquant inspect`` can show that a tensor was considered and excluded, not
    overlooked -- and priced, because being left out of the map does not leave it out of
    the file."""

    def __len__(self) -> int:
        return len(self.modules)

    def __iter__(self) -> Iterator[ModuleInfo]:
        return iter(self.modules.values())

    def __getitem__(self, name: str) -> ModuleInfo:
        return self.modules[name]

    def __contains__(self, name: str) -> bool:
        return name in self.modules

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.modules)

    def quantizable(self) -> tuple[ModuleInfo, ...]:
        """Modules that get a bit-width: quantizable role, not a tie follower."""
        return tuple(
            m for m in self.modules.values() if m.is_quantizable and not m.is_tied_follower
        )

    def unique_params(self) -> int:
        """Parameter count with ties counted once.

        The number a budget must be computed against. See the class docstring for
        what going without it costs.
        """
        return sum(m.num_params for m in self.quantizable())

    def total_params(self) -> int:
        """Every distinct weight the graph holds, quantizable or not.

        The denominator for an average-bits figure, so that it matches
        :meth:`floor_cost_bits` and any allocation cost built the same way. Using
        :meth:`unique_params` instead would divide a numerator that includes the
        compute-dtype tensors by a denominator that excludes their parameters.

        :meth:`skipped_params` is in here for the same reason, one step further out. A
        norm is not a decision the allocator makes, and it is still a tensor a user
        downloads; "the model at 3 bits" is a claim about the folder, not about the part
        of the folder this graph happened to classify.
        """
        return self.unique_params() + self.unquantized_params() + self.skipped_params()

    def unquantized(self) -> tuple[ModuleInfo, ...]:
        """Weights that stay in the compute dtype: floor at :data:`UNQUANTIZED_FLOOR`.

        Distinct from :attr:`skipped`, which never entered the graph. These are real
        weight tensors that occupy space on disk at 16 bits per parameter, so the
        budget has to know about them even though there is no decision to make.
        """
        return tuple(
            m for m in self.modules.values() if not m.is_quantizable and not m.is_tied_follower
        )

    def unquantized_params(self) -> int:
        return sum(m.num_params for m in self.unquantized())

    def skipped_params(self) -> int:
        """Parameters in tensors the graph refused, ties counted once.

        Distinct from :meth:`unquantized_params`, which counts weights that entered the
        graph and were floored at compute dtype. These never entered it: there is no
        :class:`ModuleInfo`, no role and no decision. They are the same thing to a
        filesystem, which is the only reader a budget answers to.
        """
        return sum(s.num_params for s in self.skipped.values() if s.tied_to is None)

    def unclassified(self) -> tuple[str, ...]:
        return tuple(n for n, m in self.modules.items() if m.role is ModuleRole.OTHER)

    def by_role(self) -> dict[ModuleRole, tuple[str, ...]]:
        grouped: dict[ModuleRole, list[str]] = {}
        for name, info in self.modules.items():
            grouped.setdefault(info.role, []).append(name)
        return {role: tuple(names) for role, names in grouped.items()}

    def floor_cost_bits(self) -> int:
        """Total bits if every module sat exactly on its floor.

        Compared against the budget this is the single most useful diagnostic
        there is: when it exceeds the budget, hard floors are unsatisfiable and an
        allocator that treats them as hard has no score-driven decisions left to
        make. That is precisely the state the supplement's allocator was in at its
        own headline 3-bit setting, where it silently returned the floor map.

        Counts the compute-dtype tensors at :data:`UNQUANTIZED_FLOOR` too, and the
        refused ones beside them. Neither is a decision, both are bits on disk, and a
        budget that omits them reports an average the file does not have. The two are
        summed together because nothing downstream can tell them apart: a tensor that was
        floored and a tensor that was never classified cost the same 16 bits.
        """
        payload = sum(m.num_params * m.floor_bits for m in self.quantizable())
        dense = self.unquantized_params() + self.skipped_params()
        return payload + UNQUANTIZED_FLOOR * dense

    def report(self) -> str:
        """Human-readable summary, grouped by role and sorted by parameter count."""
        by_role: dict[ModuleRole, tuple[int, int, int]] = {}
        for info in self.quantizable():
            count, params, floor = by_role.get(info.role, (0, 0, 0))
            # The *effective* floor, so a tie or a fused partition shows the number
            # that will actually constrain the allocator. The role's table entry
            # would say 4b for a tied `embedding` that has to deliver 8.
            by_role[info.role] = (count + 1, params + info.num_params, max(floor, info.floor_bits))

        total = self.total_params() or 1
        lines = [
            f"{self.model_type}: {len(self.quantizable())} quantizable modules, "
            f"{total / 1e9:.3f}B params"
        ]
        for role, (count, params, floor) in sorted(by_role.items(), key=lambda kv: -kv[1][1]):
            base = DEFAULT_FLOOR_BITS.get(role, 4)
            note = f"floor {floor}b" + (f" (role {base}b)" if floor != base else "")
            lines.append(
                f"  {role.value:<22} {count:>4}x  {params / 1e6:>9.2f}M  "
                f"({100 * params / total:>5.1f}%)  {note}"
            )
        if self.tied_groups:
            for group in self.tied_groups:
                lines.append(f"  tied: {' == '.join(group)}")
        if unclassified := self.unclassified():
            lines.append(f"  UNCLASSIFIED ({len(unclassified)}): {', '.join(unclassified[:8])}")
        if kept := self.unquantized():
            lines.append(
                f"  compute dtype ({len(kept)}): {self.unquantized_params() / 1e6:.2f}M params "
                f"at {UNQUANTIZED_FLOOR}b"
            )
        if refused := self.skipped_params():
            # Beside the floored tensors rather than folded into them, because the sizes
            # say different things. A few hundred KB of norms is the expected shape; a
            # refused expert bank is most of the model sitting at fp16, and is the loudest
            # available signal that the target will be missed for a reason nobody chose.
            lines.append(
                f"  refused ({len(self.skipped)}): {refused / 1e6:.2f}M params "
                f"at {UNQUANTIZED_FLOOR}b ({100 * refused / total:.1f}%)"
            )
        lines.append(f"  floor cost: {self.floor_cost_bits() / total:.4f} avg bits")
        return "\n".join(lines)


def classify_model(
    model: nn.Module,
    *,
    config: Any = None,
    overrides: Mapping[str, str | ModuleRole] | None = None,
) -> ModelGraph:
    """Walk ``model`` and classify every quantizable weight.

    Args:
        model: A live module. Wrappers (PEFT, DDP, ``torch.compile``) are fine;
            names are canonicalised, so the result keys match a bare model's and
            therefore match the stats file's.
        config: The HF config. Taken from ``model.config`` when omitted. Supplies
            the dimensions structural inference needs -- ``num_experts``,
            ``head_dim``, ``attn_output_gate`` -- so passing a config for a
            *different* model produces confident nonsense.
        overrides: ``{glob: role}``, highest priority. ``fnmatch`` over canonical
            names, e.g. ``{"model.layers.*.mlp.gate": "moe.router"}``.

    Returns:
        A :class:`ModelGraph`. Unrecognised modules get :attr:`ModuleRole.OTHER`,
        a conservative 4-bit floor, and a warning naming every one of them --
        never a silent drop to the 2-bit minimum.
    """
    config = config if config is not None else getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "") or "unknown")
    architectures = tuple(getattr(config, "architectures", None) or ())
    plugin = plugin_for(model_type, config)

    # Sub-configs carry the dimensions for their own tower. On a VLM-derived text
    # model like Qwen3.5 the interesting numbers -- head_dim, attn_output_gate,
    # layer_types -- live in `text_config`, and reading the outer config instead
    # yields None for all of them.
    inner = getattr(config, "text_config", None) or config

    parents = _parent_classes(model)
    by_raw_name = dict(model.named_modules())
    overrides = overrides or {}

    modules: dict[str, ModuleInfo] = {}
    skipped: dict[str, SkippedTensor] = {}
    seen_weights: dict[int, str] = {}
    seen_skipped: dict[int, str] = {}
    """Ids of refused tensors, so two modules sharing one norm are priced once. Kept apart
    from `seen_weights` because a tie among refused tensors has no representative to push a
    role onto -- there is no `ModuleInfo` at either end."""
    tied: dict[str, list[str]] = {}
    claimed: set[int] = set()
    """Ids of every parameter some module accounted for, so the sweep below can
    tell a genuinely unowned tensor from one already in the graph."""

    for raw_name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None:
            # A batched MoE expert bank owns its experts as 3-D parameters and has no
            # `.weight`, so it lands here beside the genuine containers. Skipping it
            # is how ~91% of a 128-expert model used to become invisible.
            found, refused = _expert_bank(raw_name, module, inner, overrides)
            modules.update(found)
            _record_skipped(refused, into=skipped, seen=seen_skipped)
            claimed.update(id(param) for _, param in batched_expert_params(module))
            continue
        claimed.add(id(weight))
        name = canonical_name(raw_name)
        if not name or name in modules or name in skipped:
            continue
        if weight.ndim < 2:
            # One dimension means a norm scale or a bias. Excluding by rank rather
            # than by class name covers every architecture's norm without an
            # enumeration that goes stale.
            _record_skipped(
                [
                    (
                        name,
                        SkippedTensor(
                            reason=f"{type(module).__name__} weight is {weight.ndim}-D",
                            num_params=weight.numel(),
                        ),
                        weight,
                    )
                ],
                into=skipped,
                seen=seen_skipped,
            )
            continue

        ctx = ModuleContext(
            name=name,
            module=module,
            weight=weight,
            config=inner,
            ancestors=parents.get(raw_name, ()),
            leaf=name.rsplit(".", 1)[-1],
            parent=by_raw_name.get(raw_name.rpartition(".")[0]),
        )
        role, source = _resolve(ctx, plugin, overrides)

        tied_to: str | None = None
        first = seen_weights.get(id(weight))
        if first is None:
            seen_weights[id(weight)] = name
        else:
            tied_to = first
            tied.setdefault(first, [first]).append(name)

        modules[name] = ModuleInfo(
            name=name,
            role=role,
            module_type=type(module).__name__,
            shape=tuple(weight.shape),
            num_params=weight.numel(),
            source=source,
            partitions=_partitions(ctx, role, plugin),
            tied_to=tied_to,
        )

    unowned, refused = _unowned_parameters(model, claimed=claimed, overrides=overrides)
    refused += _persistent_buffers(model, claimed=claimed)
    _record_skipped(
        [entry for entry in refused if entry[0] not in modules], into=skipped, seen=seen_skipped
    )
    for name, info, weight in unowned:
        if name in modules or name in skipped:
            continue
        first = seen_weights.get(id(weight))
        if first is None:
            seen_weights[id(weight)] = name
        else:
            info = replace(info, tied_to=first)
            tied.setdefault(first, [first]).append(name)
        modules[name] = info

    # Push each follower's role onto its representative, now that every name is
    # known. Done as a second pass because a tie is only discoverable once both
    # ends have been walked, and the representative is written first.
    for representative, group in tied.items():
        followers = tuple(modules[n].role for n in group if n != representative)
        modules[representative] = replace(modules[representative], tied_roles=followers)

    graph = ModelGraph(
        modules=dict(sorted(modules.items())),
        model_type=model_type,
        architectures=architectures,
        tied_groups=tuple(tuple(g) for g in tied.values()),
        skipped=dict(sorted(skipped.items())),
    )
    _warn_about_gaps(graph)
    return graph


# --------------------------------------------------------------------------
# Resolution steps
# --------------------------------------------------------------------------


def _resolve(
    ctx: ModuleContext,
    plugin: Any,
    overrides: Mapping[str, str | ModuleRole],
) -> tuple[ModuleRole, str]:
    for pattern, override_role in overrides.items():
        if fnmatchcase(ctx.name, pattern):
            return ModuleRole(override_role), "override"

    if plugin is not None:
        claimed: ModuleRole | None = plugin.role_for(ctx)
        if claimed is not None:
            return claimed, f"plugin:{plugin.name}"

    structural = _structural_role(ctx)
    if structural is not None:
        return structural, "structural"

    return role_of_name(ctx.name), "name"


def _structural_role(ctx: ModuleContext) -> ModuleRole | None:
    """Infer from the module tree and config, for anything a plugin did not claim.

    Only returns a role when the structure is genuinely decisive. Returning
    ``None`` defers to name matching, which is the right outcome for a module this
    cannot reason about -- a confident wrong answer is worse than falling through.
    """
    out_features = ctx.weight.shape[0]

    # A router is the one classification that must never be wrong, so it is tested
    # structurally: output width equal to the expert count, with an `experts`
    # sibling to rule out the coincidence of an MLP that happens to be that wide.
    num_experts = _first_attr(ctx.config, "num_experts", "num_local_experts", "n_routed_experts")
    if num_experts and out_features == num_experts and ctx.has_sibling("experts"):
        return ModuleRole.MOE_ROUTER

    # An Embedding whose rows are the vocabulary. Distinguishing embedding from LM
    # head by class rather than by name matters because they are often the same
    # tensor under two names.
    if ctx.module_class_endswith("Embedding") and out_features == _first_attr(
        ctx.config, "vocab_size"
    ):
        return ModuleRole.EMBEDDING

    return None


def _expert_bank(
    raw_name: str,
    module: nn.Module,
    config: Any,
    overrides: Mapping[str, str | ModuleRole],
) -> tuple[dict[str, ModuleInfo], list[tuple[str, SkippedTensor, Any]]]:
    """Classify a batched MoE expert bank's 3-D parameters, or refuse them.

    Returns ``({}, [])`` for anything that is not a bank, so the caller can treat
    every ``.weight``-less module the same way. A refusal here is the one that most
    needs pricing: these tensors are the bulk of a MoE, and refusing them leaves that
    bulk at compute dtype.

    Each tensor is named ``<bank>.<param>`` -- e.g.
    ``model.layers.0.mlp.experts.gate_up_proj`` -- which is the name the whole
    pipeline already agrees on: it is what the state dict calls the tensor, so the
    stats file, the bit map and the manifest all key on it without translation.

    Two deliberate departures from the ``.weight`` path:

    *Name and overrides only, no structural inference.* ``_structural_role`` reads
    ``weight.shape[0]`` as the output width. On a 3-D expert tensor that axis is the
    *expert count*, so the router test -- output width equals ``num_experts``, with an
    ``experts`` sibling -- matches every bank in the model, and the bank is its own
    ``experts`` sibling. Every expert tensor would be classified ``MOE_ROUTER`` and
    floored at 8 bits: a confident, structural, catastrophically wrong answer. The
    name is unambiguous here anyway (``experts.`` in the path already remaps the MLP
    roles to their MoE counterparts), so the weaker signal is the correct one.

    *No row partitions.* A fused ``gate_up_proj`` bank does hold gate rows and up rows
    that deserve different widths, but :class:`RowPartition` indexes the rows of a
    matrix and this is a stack of matrices. Splitting it needs a partition scheme
    defined over the expert axis, which is a format change. So the whole bank gets one
    width, and bug 5's fix does not yet reach MoE -- stated here rather than papered
    over by a partition that would index the wrong axis.
    """
    params = batched_expert_params(module)
    if not params:
        return {}, []

    bank = canonical_name(raw_name)
    orientation = bank_orientation(module, config)
    if orientation != OUT_IN:
        # Grouping along the wrong axis produces a checkpoint that round-trips and
        # reports a plausible reconstruction error, with scales that average over the
        # output channels instead of the input ones. There is no symptom to notice
        # later, so this refuses now.
        detail = (
            "input axis is not last"
            if orientation == IN_OUT
            else "orientation could not be determined from the config"
        )
        return {}, [
            (
                f"{bank}.{param_name}",
                SkippedTensor(
                    reason=(
                        f"batched expert tensor {tuple(param.shape)}: {detail}. The encoder "
                        f"groups along the last axis; quantizing this would share scales "
                        f"across the wrong channels. Pass an explicit layout when the format "
                        f"supports one."
                    ),
                    num_params=param.numel(),
                ),
                param,
            )
            for param_name, param in params
        ]

    found: dict[str, ModuleInfo] = {}
    for param_name, param in params:
        name = f"{bank}.{param_name}"
        override = next(
            (role for pattern, role in overrides.items() if fnmatchcase(name, pattern)), None
        )
        found[name] = ModuleInfo(
            name=name,
            role=ModuleRole(override) if override is not None else role_of_name(name),
            module_type=type(module).__name__,
            shape=tuple(param.shape),
            num_params=param.numel(),
            source="override" if override is not None else "batched-expert",
        )
    return found, []


def _record_skipped(
    entries: list[tuple[str, SkippedTensor, Any]],
    *,
    into: dict[str, SkippedTensor],
    seen: dict[int, str],
) -> None:
    """File refused tensors under their names, pricing each distinct tensor once.

    One function for all three refusal sites rather than the same four lines three times,
    because the thing being got right is a total and a total is only as right as its least
    careful contributor. First name wins the parameters; the rest record who has them.
    """
    for name, entry, tensor in entries:
        if name in into:
            continue
        first = seen.setdefault(id(tensor), name)
        into[name] = entry if first == name else replace(entry, tied_to=first)


def _unowned_parameters(
    model: nn.Module,
    *,
    claimed: set[int],
    overrides: Mapping[str, str | ModuleRole],
) -> tuple[list[tuple[str, ModuleInfo, Any]], list[tuple[str, SkippedTensor, Any]]]:
    """Pick up weight tensors that no module exposes as ``.weight``.

    ``named_modules`` finds a tensor only if its owner spells it ``self.weight``.
    Real architectures routinely do not:

    * ``nn.MultiheadAttention`` keeps its fused Q|K|V in ``in_proj_weight``;
    * Gemma-3's ``Gemma3MultiModalProjector`` holds the entire vision-to-text
      bridge in a bare ``nn.Parameter`` called ``mm_input_projection_weight``.

    Those tensors are on disk and in the parameter count whether or not the graph
    knows about them, and a graph that omits them reports an average-bits figure
    computed over a *subset* of the file -- the same class of error as
    double-counting a tied embedding, in the opposite direction. On Gemma-3 the
    omission is three tensors including the multimodal projector, which
    :data:`DEFAULT_FLOOR_BITS` rates as one of the least compressible things in
    the model.

    Classification is by name only. There is no module to inspect, so structural
    inference has nothing to read and a plugin has no ``ModuleContext`` to be
    handed; the fallback is the honest option rather than a degraded one.

    Returns:
        ``(found, refused)`` -- both as ``(name, info, tensor)`` triples so the caller can
        resolve ties by tensor identity. ``named_parameters`` already deduplicates shared
        tensors, so this sweep cannot produce a tie of its own; it can still name a tensor
        the module walk refused under another name.
    """
    found: list[tuple[str, ModuleInfo, Any]] = []
    refused: list[tuple[str, SkippedTensor, Any]] = []

    for raw_name, param in model.named_parameters():
        if id(param) in claimed:
            continue
        name = canonical_name(raw_name)
        if not name:
            continue
        # Rank alone is the wrong test here. SigLIP's pooling `probe` is
        # `[1, 1, hidden]`: three dimensions, but only one of them is real, so it
        # is a vector wearing a matrix's shape. Grouped quantization would give it
        # a single group and one scale per 128 values, which is all cost and no
        # compression -- the same reasoning that keeps norms and biases out by
        # rank, applied to a shape that would otherwise slip past.
        if sum(1 for dim in param.shape if dim > 1) < 2:
            refused.append(
                (
                    name,
                    SkippedTensor(
                        reason=(f"{tuple(param.shape)} has fewer than two non-trivial dimensions"),
                        num_params=param.numel(),
                    ),
                    param,
                )
            )
            continue

        override = next(
            (role for pattern, role in overrides.items() if fnmatchcase(name, pattern)), None
        )
        found.append(
            (
                name,
                ModuleInfo(
                    name=name,
                    role=ModuleRole(override) if override is not None else role_of_name(name),
                    module_type="Parameter",
                    shape=tuple(param.shape),
                    num_params=param.numel(),
                    source="override" if override is not None else "parameter",
                ),
                param,
            )
        )
    return found, refused


def _persistent_buffers(
    model: nn.Module, *, claimed: set[int]
) -> list[tuple[str, SkippedTensor, Any]]:
    """Tensors that reach the checkpoint without ever having been parameters.

    :func:`_unowned_parameters` walks ``named_parameters``, which by construction never
    yields a buffer, and every total this module builds descends from that walk. A
    persistent buffer is therefore in the download and in no denominator.

    Not hypothetical. LFM2.5-8B-A1B keeps its router's load-balancing bias in
    ``feed_forward.expert_bias``: 22 tensors of 32 values, ``requires_grad=False``,
    registered as buffers and written to the shard in fp32. The graph counted
    8,467,856,128 parameters against the file's 8,467,856,832, and the missing 704 were
    exactly these. This is the tied-embedding error running backwards for the second
    time -- there a shared tensor was counted twice, here a real one is counted zero
    times -- and the first backwards case, bare :class:`torch.nn.Parameter` weights, is
    what :func:`_unowned_parameters` above exists to catch.

    Persistence is read from ``state_dict()``, not from a module's private
    non-persistent set, because ``state_dict`` *is* the definition of what gets written.
    A rotary ``inv_freq`` is a buffer on this same model and is not in there; charging
    for it would be the opposite error, and by exactly the number of bytes a downloader
    never pays.

    Every one of these is refused rather than classified. A buffer is state -- a routing
    bias, a cached mask, a running statistic -- and there is no version of quantizing one
    that trades size for accuracy instead of simply damaging it.

    One term stays unpriced and is named here rather than absorbed: a refused tensor is
    charged at :data:`UNQUANTIZED_FLOOR`, and a buffer stored in fp32 costs twice that on
    disk. On LFM2.5-8B-A1B that is 704 parameters and 1,408 bytes against 4.4 GB. Pricing
    it properly means carrying a per-tensor width through the budget, which is a wider
    change than the error justifies; leaving it undocumented is what would not be.
    """
    persistent = set(model.state_dict())
    refused: list[tuple[str, SkippedTensor, Any]] = []
    for raw_name, buffer in model.named_buffers():
        if raw_name not in persistent or id(buffer) in claimed:
            continue
        name = canonical_name(raw_name)
        if not name:
            continue
        refused.append(
            (
                name,
                SkippedTensor(
                    reason=f"persistent buffer {tuple(buffer.shape)}, not a weight",
                    num_params=buffer.numel(),
                ),
                buffer,
            )
        )
    return refused


def _partitions(ctx: ModuleContext, role: ModuleRole, plugin: Any) -> tuple[RowPartition, ...]:
    """Row ranges for a fused tensor, or ``()``.

    A plugin gets first refusal because only it knows its architecture's row order
    -- whether a fused QKV is ``[q; k; v]`` or interleaved per head, whether the
    gate comes before or after the up-projection. Guessing wrong here silently
    assigns the gate's bits to the up-projection's rows, which is worse than not
    partitioning at all, so there is no generic fallback: without a plugin the
    tensor keeps one bit-width for all its rows.
    """
    if not role.is_fused:
        return ()
    if plugin is not None:
        parts: tuple[RowPartition, ...] | None = plugin.partitions_for(ctx, role)
        if parts:
            return parts
    return ()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _parent_classes(model: nn.Module) -> dict[str, tuple[str, ...]]:
    """Map every module's raw name to its ancestors' class names, outermost first.

    Built in one pass over ``named_modules``. Ancestry is what tells a plugin that
    a ``Linear`` sits inside a ``Qwen3_5GatedDeltaNet`` rather than a
    ``Qwen3_5Attention`` when both spell their projections ``in_proj``-ish.
    """
    classes = {name: type(mod).__name__ for name, mod in model.named_modules()}
    out: dict[str, tuple[str, ...]] = {}
    for name in classes:
        chain: list[str] = []
        parts = name.split(".")
        for depth in range(len(parts)):
            ancestor = ".".join(parts[:depth])
            if ancestor in classes:
                chain.append(classes[ancestor])
        out[name] = tuple(chain)
    return out


def _first_attr(config: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _warn_about_gaps(graph: ModelGraph) -> None:
    """Say out loud what was not understood.

    An unclassified module is not an error -- it gets a cautious 4-bit floor and
    the model still quantizes. But it is the signal that this architecture needs a
    plugin, and staying quiet about it is how a model ships with its router at
    3 bits and nobody notices until the accuracy numbers come back wrong.
    """
    unclassified = graph.unclassified()
    if unclassified:
        _log.warning(
            "%d module(s) could not be classified for model_type=%r and default to a "
            "conservative %d-bit floor: %s%s. Pass overrides= to name them, or add an "
            "architecture plugin.",
            len(unclassified),
            graph.model_type,
            DEFAULT_FLOOR_BITS[ModuleRole.OTHER],
            ", ".join(unclassified[:10]),
            " ..." if len(unclassified) > 10 else "",
        )
    for group in graph.tied_groups:
        _log.info(
            "tied weights share one bit-width: %s (one tensor, one decision)",
            " == ".join(group),
        )
