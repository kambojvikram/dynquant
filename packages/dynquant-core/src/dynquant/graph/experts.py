"""Batched MoE expert tensors: finding them, and knowing which way round they are.

Modern ``transformers`` does not give each expert its own ``nn.Linear``. Every MoE
family on 5.x -- qwen3_moe, qwen2_moe, mixtral, olmoe, gpt_oss -- collapses the whole
expert bank into a single module holding 3-D ``nn.Parameter`` tensors::

    Qwen3MoeExperts
      gate_up_proj   [num_experts, 2 * moe_intermediate, hidden]
      down_proj      [num_experts, hidden, moe_intermediate]

and applies them with ``F.linear(x, self.gate_up_proj[expert_idx])`` inside a loop
over the experts that actually received tokens.

Why this module exists
----------------------
Every consumer in this package used to find weights the same way: walk
``named_modules()`` and read ``module.weight``. A batched expert module has no
``.weight``, so it was skipped -- silently, by the same ``if weight is None:
continue`` that correctly skips containers. On a 128-expert model that is roughly
91% of the parameters invisible to the graph, so the allocator would quantize
attention and the router and leave every expert at full precision: a checkpoint far
larger than its target, with nothing in any report saying why. The architecture
matrix did not catch it because those tests hand-build stubs modelling the *old*
per-expert ``nn.Linear`` layout, which no released ``transformers`` has produced for
some time -- the tests and the library had drifted apart while both stayed green.

So discovery lives here, once, and the graph, the tracker and the quantizer all call
it. Three copies of this rule would be three chances to fix the bug in two places.

Orientation is not guessable
----------------------------
Group-wise quantization shares one scale across a run of *input* channels. That makes
the input axis load-bearing, and the families disagree about where it is:

===========  ==========================  =========================
family       ``down_proj`` shape         input axis
===========  ==========================  =========================
qwen3_moe    ``[E, hidden, inter]``      last   (``[E, out, in]``)
qwen2_moe    ``[E, hidden, inter]``      last
mixtral      ``[E, hidden, inter]``      last
olmoe        ``[E, hidden, inter]``      last
gpt_oss      ``[E, inter, hidden]``      middle (``[E, in, out]``)
===========  ==========================  =========================

The encoder groups along the last axis. For the first four that is right. For
``gpt_oss`` it would group along the *output* axis instead -- which still yields a
well-formed, round-trippable checkpoint with a plausible reconstruction error, just
one whose scales average over the wrong direction and whose group boundaries mean
nothing to a kernel. A silently-wrong axis is precisely the class of bug this package
exists to not have, so :func:`bank_orientation` decides structurally from the config
and returns :data:`UNKNOWN` rather than assuming, and callers refuse the tensor
instead of encoding it the wrong way round.

Deciding it needs the parameter *name*, not just the shape. ``[E, 32, 16]`` is
``[E, out, in]`` for a ``down_proj`` and would be ``[E, in, out]`` for an ``up_proj``;
the shape cannot say which, but the name says which side ``hidden`` belongs on.

Not yet: per-expert bit-widths
------------------------------
One batched tensor gets one role, one score and one width for all its experts. A
rarely-routed expert cannot currently be given fewer bits than a hot one. That needs
:class:`~dynquant.graph.roles.RowPartition` extended over the expert axis and
per-expert routing counters in the tracker; it is a feature, it is absent, and this
note is here so its absence is not mistaken for its presence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = [
    "IN_OUT",
    "OUT_IN",
    "UNKNOWN",
    "bank_orientation",
    "batched_expert_params",
    "is_expert_container",
    "owning_configs",
    "reads_hidden",
]

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import nn

OUT_IN = "out_in"
"""``[E, out, in]`` -- input axis last, which is what the encoder assumes."""

IN_OUT = "in_out"
"""``[E, in, out]`` -- input axis in the middle. gpt_oss."""

UNKNOWN = "unknown"
"""The config did not settle it. Refuse rather than pick."""

_EXPERT_CLASS_SUFFIX = "Experts"
_BIAS_SUFFIX = "_bias"

# Which side of the matrix ``hidden`` sits on, by parameter name. A down-projection
# reads the expert width and writes the residual stream; everything else does the
# reverse. Anything not listed is not orientable by this rule and gets UNKNOWN.
_READS_HIDDEN: tuple[str, ...] = ("gate_up_proj", "gate_proj", "up_proj", "fc1", "w1", "w3")
_WRITES_HIDDEN: tuple[str, ...] = ("down_proj", "fc2", "w2")


def is_expert_container(module: nn.Module) -> bool:
    """Whether ``module`` looks like a batched expert bank.

    Tested by class-name suffix *and* by the presence of 3-D parameters, because
    either alone is wrong. The class name alone would admit a future ``...Experts``
    that went back to submodules; the 3-D test alone would admit any module that
    happens to own a rank-3 parameter of its own.
    """
    if not type(module).__name__.endswith(_EXPERT_CLASS_SUFFIX):
        return False
    return any(p.ndim == 3 for _, p in module.named_parameters(recurse=False))


def batched_expert_params(module: nn.Module) -> tuple[tuple[str, Any], ...]:
    """The quantizable 3-D parameters ``module`` owns directly, as ``(name, param)``.

    Empty for anything that is not a batched expert bank, so a caller can treat this
    as "extra tensors this module contributes" without a type test first.

    Rank 3 exactly, and never a bias. ``gpt_oss`` carries ``gate_up_proj_bias`` and
    ``down_proj_bias`` alongside the weights: rank 2, so the rank test already
    excludes them, and the name test is belt-and-braces against a family that stores a
    rank-3 bias. Quantizing a bias is not a size/accuracy trade -- biases are a
    rounding error of the parameter count and they land directly on the residual
    stream.
    """
    if not is_expert_container(module):
        return ()
    return tuple(
        (name, param)
        for name, param in module.named_parameters(recurse=False)
        if param.ndim == 3 and not name.endswith(_BIAS_SUFFIX)
    )


def reads_hidden(param_name: str) -> bool | None:
    """Whether this expert tensor's *input* is the residual stream.

    ``True`` for a gate/up projection, ``False`` for a down projection, and ``None``
    when the name is not one this module recognises. Three-way for the same reason
    :func:`bank_orientation` is: which side of a matrix the residual stream sits on
    is not guessable, and a caller that guesses gets a plausible number computed
    against the wrong activation.

    Callers need this because a bank is one module holding two weights. Its forward
    reads the residual stream, projects up, applies a non-linearity, projects down
    and returns to the residual stream -- so of the four activations involved, only
    the first and last cross the module boundary where a hook can see them. This says
    which of the two a given tensor owns.
    """
    if param_name in _READS_HIDDEN:
        return True
    if param_name in _WRITES_HIDDEN:
        return False
    return None


def owning_configs(modules: Mapping[str, nn.Module], raw_name: str) -> tuple[Any, ...]:
    """Every config that could describe ``raw_name``'s widths, most specific first.

    A composite model does not have *a* config. ``qwen3_omni_moe`` carries a Thinker
    and a Talker, each a full MoE with its own hidden size and its own expert width,
    and the top-level ``Qwen3OmniMoeConfig`` holds neither number -- they live two
    levels down, in ``thinker_config.text_config`` and ``talker_config.text_config``.
    Hand the outer config to :func:`bank_orientation` and ``_widths`` returns empty
    sets, so every bank in the model is refused for want of a dimension that is
    present in the file the whole time. On this family that is 90.8% of the
    parameters declined.

    Resolution is *structural*: the answer for a bank is the config of the nearest
    module that encloses it and owns one. ``thinker.model.layers.0.mlp.experts`` is
    enclosed by ``thinker.model``, whose config is the Thinker's text config, and
    that is the authority on the Thinker's widths. Walking outward from the bank
    rather than inward from the root is what keeps the Talker's numbers away from a
    Thinker bank: the Talker is not an ancestor of it, so it is never a candidate.

    Two properties make this safe to use as a fallback rather than a guess. Every
    candidate is an ancestor, so it describes a subtree that *contains* this bank --
    none of them is some other model's config. And orientation is decided by matching
    the config's widths against the tensor's actual shape, so a config from the wrong
    tower does not quietly win: measured on Qwen3-Omni, the Thinker's text config
    against a Talker bank returns :data:`UNKNOWN`, and the reverse does too. A wrong
    candidate refuses; it does not mislead.

    Args:
        modules: ``dict(model.named_modules())``. Taken as a mapping rather than a
            model because the caller already built it -- rebuilding it per bank is
            an extra full walk of the module tree for each of 68 banks.
        raw_name: The bank's un-canonicalised name, as ``named_modules`` gave it.

    Returns:
        Candidate configs, nearest enclosing module first, then outward to the root.
        Each config is followed by its ``text_config`` when it has one, since a
        wrapper config's dimensions usually sit one level in. Deduplicated by
        identity, so a sub-model that shares its parent's config is offered once.
        Empty when no ancestor owns a config.
    """
    parts = raw_name.split(".") if raw_name else []
    found: list[Any] = []
    seen: set[int] = set()
    for depth in range(len(parts) - 1, -1, -1):
        ancestor = modules.get(".".join(parts[:depth]))
        if ancestor is None:
            continue
        config = getattr(ancestor, "config", None)
        if config is None:
            continue
        for candidate in (config, getattr(config, "text_config", None)):
            if candidate is not None and id(candidate) not in seen:
                seen.add(id(candidate))
                found.append(candidate)
    return tuple(found)


def bank_orientation(module: nn.Module, config: Any) -> str:
    """Which axis of this bank's expert tensors is the input dimension.

    Orientation is a property of the bank, not of one tensor: a family stores all its
    expert matrices the same way round. So this takes the first tensor that can be
    decided and applies the answer to the bank. That matters because a fused
    ``gate_up_proj`` is often undecidable on its own -- when ``hidden == 2 * inter``
    both readings fit -- while the ``down_proj`` beside it is unambiguous.
    """
    for name, param in batched_expert_params(module):
        decided = _orientation_of(name, param, config)
        if decided != UNKNOWN:
            return decided
    return UNKNOWN


def _orientation_of(name: str, param: Any, config: Any) -> str:
    """Orientation from one named tensor, or :data:`UNKNOWN` if it cannot be settled.

    Returns :data:`UNKNOWN` for four real cases: an unrecognised parameter name, a
    config missing the width fields, a shape matching neither reading, and a
    degenerate config where both readings fit. Every one of them has to refuse rather
    than pick, because picking is a coin flip on whether the scales average over the
    right channels.
    """
    if param.ndim != 3:
        return UNKNOWN
    hidden = _widths(config, "hidden_size", "d_model", "n_embd")
    inter = _widths(config, "moe_intermediate_size", "intermediate_size", "ffn_dim")
    if not hidden or not inter:
        return UNKNOWN

    # A fused gate_up writes 2x the expert width; nothing reads a doubled hidden.
    expert_side = inter | {2 * w for w in inter}
    if name in _READS_HIDDEN:
        expects_in, expects_out = hidden, expert_side
    elif name in _WRITES_HIDDEN:
        expects_in, expects_out = expert_side, hidden
    else:
        return UNKNOWN

    _, middle, last = (int(d) for d in param.shape)
    out_in = last in expects_in and middle in expects_out
    in_out = middle in expects_in and last in expects_out
    if out_in and not in_out:
        return OUT_IN
    if in_out and not out_in:
        return IN_OUT
    return UNKNOWN


def _widths(config: Any, *names: str) -> set[int]:
    """Every plausible value of one dimension, across the names families use for it.

    A set rather than a first-match because families disagree about which field is
    authoritative -- ``gpt_oss`` sizes its experts by ``intermediate_size`` while
    carrying an unused ``moe_intermediate_size``, so first-match would test the wrong
    number and return :data:`UNKNOWN` for a bank that is perfectly determinable.
    """
    return {
        value
        for name in names
        if isinstance(value := getattr(config, name, None), int) and value > 0
    }
