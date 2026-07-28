"""Canonical module names -- one spelling for a weight, whatever wrapped it.

The same ``q_proj`` answers to many names depending on what is wrapped around it
at the moment you ask::

    model.layers.0.self_attn.q_proj                                  bare
    base_model.model.model.layers.0.self_attn.q_proj.base_layer      + PEFT LoRA
    module.model.layers.0.self_attn.q_proj                           + DDP
    _orig_mod.model.layers.0.self_attn.q_proj                        + torch.compile
    _fsdp_wrapped_module.model.layers...._fsdp_wrapped_module.q_proj  + FSDP

Signals are collected under whichever spelling was live during training, and
quantization runs later against the *merged* model, which uses the bare spelling.
Something has to reconcile them.

The supplement reconciled them at **read** time
(``dynquant_paper/run_quantization.py:17-36``): the quantizer took each stats key
and guessed its bare form by stripping ``base_model.``, collapsing
``model.model.`` to ``model.``, and special-casing ``model.lm_head``. Guessing at
read time is the wrong end of the pipe -- it has to anticipate every wrapper
combination the trainer might have used, and when it guesses wrong the layer
simply is not found, its signal reads as absent, and the allocator treats it as
unimportant.

Canonicalising at **write** time instead means the stats file is
wrapper-independent the moment it is created, and this module is the single place
that knows about wrappers. It also supplies the "collapse LoRA into base" step
that the shipped stats files claim to have had applied
(``"collapsed_lora_into_base": true``) with a script that was not in the
supplement.

Pure string manipulation -- no torch, no peft.
"""

from __future__ import annotations

__all__ = [
    "ADAPTER_COMPONENTS",
    "canonical_name",
    "is_adapter_name",
    "strip_wrappers",
]

_LEADING_WRAPPERS: frozenset[str] = frozenset(
    {
        "module",  # DataParallel / DistributedDataParallel
        "_orig_mod",  # torch.compile
        "_fsdp_wrapped_module",  # FSDP
    }
)
"""Only stripped at the *front* of a path. ``module`` in particular is too common
a word to remove from the middle of a name."""

_INNER_WRAPPERS: frozenset[str] = frozenset(
    {
        "base_layer",  # peft.tuners.lora.Linear -> the original nn.Linear
        "_orig_mod",
        "_fsdp_wrapped_module",
        "_checkpoint_wrapped_module",  # activation checkpointing
        "_flat_param",
    }
)
"""Stripped anywhere in the path. All are private//framework-reserved names that
no model architecture uses for a real submodule."""

ADAPTER_COMPONENTS: frozenset[str] = frozenset(
    {
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_magnitude_vector",  # DoRA
        "vera_A",
        "vera_B",
        "loha_w1_a",
        "loha_w1_b",
        "lokr_w1",
        "lokr_w2",
        "ia3_l",
        "adapter",
        "adapters",
        "modules_to_save",
    }
)
"""Adapter-internal components. A name containing one of these refers to a
*trainable adapter tensor*, not to the base weight that will be quantized.

This distinction is the heart of bug 7. The supplement's tracker, unable to hook
a frozen base weight, hooked ``lora_A.weight`` and ``lora_B.weight`` and recorded
their statistics **under the parent module's name**. Those two tensors have
transposed shapes -- ``[r, in]`` and ``[out, r]`` -- so the per-channel coherence
buffer alternated length between ``r`` and ``out`` on successive steps, and the
``torch.dot`` comparing them raised every time, inside a bare ``except`` that
discarded the error. The coherence signal was silently dead for every LoRA layer
in the paper's own experiments.
"""


def is_adapter_name(name: str) -> bool:
    """True if the name refers to adapter-internal weights rather than a base weight."""
    return any(part in ADAPTER_COMPONENTS for part in name.split("."))


def strip_wrappers(name: str) -> str:
    """Remove framework wrapper components without touching PEFT prefixes."""
    parts = name.split(".")
    while parts and parts[0] in _LEADING_WRAPPERS:
        parts.pop(0)
    return ".".join(p for p in parts if p not in _INNER_WRAPPERS)


def canonical_name(name: str, *, strip_param_suffix: bool = True) -> str:
    """Return the name this module would have in the bare, merged model.

    Args:
        name: A dotted module path, possibly with wrappers, possibly a parameter
            path ending in ``.weight``.
        strip_param_suffix: Drop a trailing ``.weight``. Parameter and module
            paths are constantly confused between the collection and
            quantization sides, and normalising both to the module path removes
            the whole class of mismatch.

    Returns:
        The canonical module path. Adapter tensors canonicalise to the base
        module they adapt, so ``...q_proj.lora_A.default`` becomes
        ``...q_proj`` -- which is exactly what you want when attributing a LoRA
        gradient to the base weight it will be merged into.

    >>> canonical_name("base_model.model.model.layers.0.self_attn.q_proj.base_layer")
    'model.layers.0.self_attn.q_proj'
    >>> canonical_name("base_model.model.lm_head")
    'lm_head'
    >>> canonical_name("module._orig_mod.model.layers.3.mlp.gate_proj.weight")
    'model.layers.3.mlp.gate_proj'
    >>> canonical_name("base_model.model.model.layers.0.mlp.up_proj.lora_A.default.weight")
    'model.layers.0.mlp.up_proj'
    """
    parts = name.split(".")

    while parts and parts[0] in _LEADING_WRAPPERS:
        parts.pop(0)

    # PEFT: PeftModel.base_model is the tuner, tuner.model is the real model.
    # Removing both leaves the path the unwrapped model would use, which handles
    # `model.layers...` and `lm_head` with one rule instead of two special cases.
    if parts[:1] == ["base_model"]:
        parts.pop(0)
        if parts[:1] == ["model"]:
            parts.pop(0)

    parts = [p for p in parts if p not in _INNER_WRAPPERS]

    # Truncate at the first adapter component: everything after it is
    # adapter-internal structure (`lora_A.default.weight`) and the base weight is
    # what we are naming.
    for i, part in enumerate(parts):
        if part in ADAPTER_COMPONENTS:
            parts = parts[:i]
            break

    if strip_param_suffix and parts[-1:] == ["weight"]:
        parts.pop()

    return ".".join(parts)
