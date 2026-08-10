"""Loading a packed DynQuant checkpoint through ``transformers``.

Why this exists, measured rather than assumed
---------------------------------------------
``AutoModelForCausalLM.from_pretrained`` on an exported DynQuant directory does not
fail. With no quantizer registered, ``transformers`` logs

    Unknown quantization type, got dynquant - supported types are: [...].
    Hence, we will skip the quantization.

and then loads the directory as though it were dense: every ``qweight``/``scales``/
``offsets`` entry is reported as an unused checkpoint key, every real weight as
"newly initialized", and ``from_pretrained`` **returns a randomly initialised model**.
No exception, no non-zero exit -- a model that generates noise, from a package whose
entire claim is that its checkpoints are size-honest. Publishing a directory with that
failure mode invites the conclusion that the quantizer is bad rather than absent.

So the point of this module is not only that a checkpoint loads. It is that one which
cannot load says so.

How the load works
------------------
The packed modules already name their buffers ``qweight``, ``scales`` and ``offsets``
-- the checkpoint's own keys (see :mod:`dynquant.runtime.linear`). So there is no
weight-copying code here and no second reader: this hook swaps each named module for a
correctly shaped but *uninitialised* packed shell before loading, and ``transformers``
fills those buffers by name exactly as it fills any other tensor. What gets swapped is
whatever ``quantization_config.modules`` lists, which is what the exporter wrote, so
the load cannot disagree with the export about which modules are packed.

Shapes come from :meth:`~dynquant.quant.tensor.QuantTensor.empty`, which derives them
through ``row_geometry`` -- the same resolver the encoder used. The shells hold
``torch.empty`` and not zeros, so a tensor the loader fails to fill decodes to garbage
instead of to a plausible-looking model.

What a bank and a router each become
------------------------------------
Neither is a ``Linear``, they are unlike each other, and the loader does a different
thing with each. Which one is ``resolve_target``'s decision -- literally the same
boundary :func:`dynquant.runtime.linear.pack_model` draws, so it is drawn by the same
function, called here with ``source="quantization_config"`` and there with a bit map.
A second copy of that resolver has already been the cause of two wrong refusals in
this project, and this module briefly held a third.

A batched expert bank stays packed. It is a 3-D ``nn.Parameter`` its parent indexes
directly, so it is swapped for a :class:`~dynquant.runtime.linear.DynQuantExpertBank`
registered under the parameter's own name, and the parent's untouched
``bank[expert]`` reaches ``__getitem__``. No parent is edited and no architecture is
enumerated. The grouped GEMM is still P8, so k routed experts cost k slice
dequantizations -- a speed limit, not a correctness one.

A router is not that shape one rank down. An ``Lfm2MoeTopKRouter`` holds
``[num_experts, hidden]`` as its own parameter and hands it *whole* to ``F.linear``,
so there is no index at which a module could stand. Those are restored instead: the
packed buffers are filled by name like any other, dequantized once after the state
dict lands, and the module's forward reads a dense weight it never learns was
quantized. Disk is saved, VRAM is not, and on LFM2.5-8B-A1B that is 22 tensors of
``[32, 2048]``. Refusing them here instead wrote 22 routers into a checkpoint that
``from_pretrained`` then declined to read.

Both of these were refused by name when this module was written, and both sentences
saying so outlived the code by two commits.

The registration gap that remains
---------------------------------
``transformers`` resolves ``quant_method`` through a process-global mapping and has no
entry-point discovery, so nothing here takes effect until someone calls
:func:`register_hf_quantizer`. Two lines, and both are needed::

    import dynquant; dynquant.register_hf_quantizer()
    model = AutoModelForCausalLM.from_pretrained("...-dynquant-4bit")

It is not done for you by ``import dynquant``, and that is a deliberate cost decision
rather than an oversight: the top-level module resolves everything lazily so that
``import dynquant`` stays free of torch and transformers, and registering eagerly would
put a transformers import on every ``dynquant`` CLI invocation that never touches a
model. Measured, that is 9.8 s against 0.06 s for ``import dynquant`` itself, with
neither torch nor transformers in ``sys.modules`` afterwards. Calling it twice is a no-op, so a library may call it
defensively.

Nothing here can help a reader who has not installed ``dynquant`` at all, or who
forgets the call: they get the silent path described above, from ``transformers``
itself. The only defences are the model card and the two lines above appearing in it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import torch
from torch import nn

from dynquant._logging import get_logger
from dynquant.constants import HF_QUANT_METHOD, QUANT_TENSOR_SUFFIXES
from dynquant.errors import MissingDependencyError
from dynquant.integration.serving_common.schema import (
    ModuleQuantSpec,
    QuantizationConfigSchema,
)
from dynquant.quant.tensor import QuantTensor, storage_dtype
from dynquant.runtime.experts import use_dynquant_experts
from dynquant.runtime.linear import (
    DynQuantEmbedding,
    DynQuantExpertBank,
    DynQuantLinear,
    ExpertBank,
    RestoredWeight,
    _PackedModule,
    replace_module,
    resolve_target,
)

__all__ = [
    "build_config_class",
    "build_quantizer_class",
    "dense_weight_bytes",
    "packed_module_names",
    "register_hf_quantizer",
]

_log = get_logger(__name__)


def _require_transformers() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise MissingDependencyError(
            "transformers",
            feature="loading a packed checkpoint with `from_pretrained`",
            extra="hf",
        ) from exc


# --------------------------------------------------------------------------
# The two transformers-derived classes
# --------------------------------------------------------------------------
#
# Both are built inside functions rather than declared at module scope, because their
# base classes live in `transformers`. A module-scope `class DynQuantConfig(
# QuantizationConfigMixin)` would make importing this module -- which
# `dynquant.__init__` does on a best-effort basis -- import transformers, several
# seconds paid by every CLI invocation that never touches a model.


def build_config_class() -> type:
    """``config.json -> quantization_config``, as ``transformers`` wants it."""
    from transformers.utils.quantization_config import QuantizationConfigMixin

    class DynQuantConfig(QuantizationConfigMixin):
        """A thin adapter, and deliberately not a second schema.

        Everything it is asked is answered by :class:`QuantizationConfigSchema`, the
        object the vLLM and SGLang plugins already read. A checkpoint those two accept
        and this one rejects would be a disagreement between two copies of the format,
        which is a bill this project has paid more than once.
        """

        def __init__(self, **kwargs: Any) -> None:
            self.quant_method = HF_QUANT_METHOD  # type: ignore[assignment]
            self._raw = dict(kwargs)
            # Parsed eagerly, so a malformed block fails while the traceback still
            # points at the config rather than later, inside a module swap.
            self.schema = QuantizationConfigSchema.from_dict(self._raw)
            self.modules_to_not_convert = list(self.schema.modules_to_not_convert)

        def to_dict(self) -> dict[str, Any]:
            return self.schema.to_dict()

        def __repr__(self) -> str:
            return (
                f"DynQuantConfig(modules={len(self.schema.modules)}, "
                f"group_size={self.schema.group_size}, "
                f"symmetric={self.schema.symmetric})"
            )

    return DynQuantConfig


def build_quantizer_class() -> type:
    """The ``HfQuantizer`` that swaps packed modules in before the weights are read."""
    from transformers.quantizers.base import HfQuantizer

    class DynQuantHfQuantizer(HfQuantizer):
        """Prepare the module tree so the checkpoint's own keys land in it."""

        requires_calibration = False
        """The widths come from the checkpoint; nothing is measured at load time."""

        requires_parameters_quantization = False
        """Nothing is quantized here. The checkpoint arrives packed -- this is a
        loader, and the encode happened in ``dynquant export``."""

        required_packages = ("dynquant",)  # type: ignore[assignment]

        def __init__(self, quantization_config: Any, **kwargs: Any) -> None:
            super().__init__(quantization_config, **kwargs)
            self._packed_names: tuple[str, ...] = ()
            self._dense_keys: frozenset[str] = frozenset()
            self._restored: dict[str, tuple[QuantTensor, torch.dtype]] = {}

        def validate_environment(self, *args: Any, **kwargs: Any) -> None:
            # Nothing beyond torch. The reference decode path is pure torch and the
            # CUDA kernels are an acceleration, not a requirement; demanding a GPU
            # here would make the CPU correctness check impossible to run, and that
            # is the check most likely to catch a format regression.
            return None

        def _process_model_before_weight_loading(
            self,
            model: Any,
            keep_in_fp32_modules: list[str] | None = None,
            **kwargs: Any,
        ) -> None:
            # The config object is ours (`build_config_class`), but the attribute is
            # read through transformers' base-class annotation, which knows only
            # `QuantizationConfigMixin`.
            schema: QuantizationConfigSchema = cast(Any, self.quantization_config).schema
            shells: dict[str, nn.Module] = {}
            dense_keys: set[str] = set()
            restored: dict[str, tuple[QuantTensor, torch.dtype]] = {}
            for name, spec in sorted(schema.modules.items()):
                target = resolve_target(model, name, source="quantization_config", restore=True)
                # The key that used to hold this target's dense weight, which is the
                # key the loader will now miss. A module contributed `name.weight`;
                # a bank *is* the parameter, so it contributed `name`.
                dense_keys.add(name if isinstance(target, ExpertBank) else f"{name}.weight")
                if isinstance(target, RestoredWeight):
                    # No forward to swap, so nothing is replaced: the module keeps its
                    # own and is given packed buffers to receive, which the after-hook
                    # turns back into the weight it already knows how to use.
                    restored[name] = _prepare_restore(target, spec)
                    continue
                shells[name] = _shell(target, spec)
                replace_module(model, name, shells[name])
            packed = set(schema.modules)
            tied = _tie_output_embedding(model, packed)
            packed |= set(tied)
            dense_keys |= {f"{name}.weight" for name in tied}
            self._packed_names = tuple(sorted(packed))
            self._dense_keys = frozenset(dense_keys)
            self._restored = restored
            # A bank installed here reaches a forward the same way a packed one does,
            # so it needs the same dispatch. `pack_model` does this for the quantize
            # path; nothing shared runs on this one. `use_dynquant_experts` keeps the
            # default path's reduction order and falls back to `use_eager_experts` on a
            # transformers with nothing to register into.
            if any(isinstance(shell, DynQuantExpertBank) for shell in shells.values()):
                use_dynquant_experts(model)
            _log.info(
                "prepared %d packed modules for loading%s%s",
                len(self._packed_names),
                f", {tied[0]} reading the input table" if tied else "",
                f", {len(restored)} restored dense after load" if restored else "",
            )

        def _process_model_after_weight_loading(self, model: Any, **kwargs: Any) -> Any:
            # Everything else is already a packed module and stays packed. These are the
            # targets with no forward to replace -- routers -- and this is the one point
            # in the format where a dense tensor is materialised on purpose. It happens
            # here rather than at export because the bytes on disk are the bytes the
            # manifest priced, and it happens at all because the alternative was an
            # artifact our own `from_pretrained` refused.
            dense = 0
            for name, (empty, out_dtype) in self._restored.items():
                module = model.get_submodule(name)
                weight = _restore_weight(module, empty, out_dtype)
                dense += weight.numel() * weight.element_size()
            if self._restored:
                _log.info(
                    "restored %d weight(s) dense after load, %s B: %s",
                    len(self._restored),
                    f"{dense:,}",
                    "no module could stand where they stand",
                )
            return model

        def update_missing_keys(
            self, model: Any, missing_keys: list[str], prefix: str
        ) -> list[str]:
            """Drop the dense key of every target that is now packed.

            Without this the loader reports one missing key per quantized module, and
            that warning reads exactly like the silent-random-model failure this
            quantizer exists to prevent -- indistinguishable, to a reader, from
            nothing having worked at all.
            """
            if not self._dense_keys:
                return missing_keys
            dense = self._dense_keys
            return [
                key
                for key in missing_keys
                if key not in dense and key.partition(f"{prefix}.")[2] not in dense
            ]

        def is_serializable(self, safe_serialization: bool | None = None) -> bool:
            # `save_pretrained` writes the buffers back under the names they were
            # loaded from, so a round trip is the identity on the packed tensors.
            return True

        @property
        def is_trainable(self) -> bool:
            # Packed weights have no gradient, and `runtime.linear` refuses a backward
            # for the same reason: a fine-tune whose updates the format cannot
            # represent is worse than one that never starts.
            return False

    return DynQuantHfQuantizer


# --------------------------------------------------------------------------
# Resolution and module construction
# --------------------------------------------------------------------------


def _tie_output_embedding(model: Any, packed: set[str]) -> tuple[str, ...]:
    """Point a tied ``lm_head`` at the packed input table, and stop it being re-tied.

    A tied model stores one table and the checkpoint contains it once, under the input
    embedding's name. So after the swap the output embedding is a dense ``Linear``
    whose weight is in no shard -- and ``from_pretrained`` calls ``model.tie_weights()``
    after loading, which does ``output.weight = input.weight`` and dies on
    ``AttributeError: 'DynQuantEmbedding' object has no attribute 'weight'``.

    Both ways out of that are worse than this one. Giving the packed embedding a
    ``weight`` property would hand ``tie_weights`` a *dequantized* table to assign,
    quietly restoring the dense tensor the packing exists to avoid -- 27% of a
    Qwen3.5-2B, and the manifest would still claim it was saved. Leaving the head dense
    and letting the loader fill it needs a copy of the table in the checkpoint, which
    is the same 27% on disk.

    So the head is replaced by a ``DynQuantLinear`` that registers no tensors of its
    own and reads the embedding's (see ``_PackedModule.__init__``), and ``tie_weights``
    is made a no-op because the tie it would establish already exists structurally --
    one buffer, two readers, which survives ``.to(device)`` where a shared registration
    would not.
    """
    if not getattr(model.config, "tie_word_embeddings", False):
        return ()
    output = model.get_output_embeddings()
    if output is None or isinstance(output, _PackedModule):
        return ()
    table = model.get_input_embeddings()
    if not isinstance(table, DynQuantEmbedding):
        return ()

    name = next((n for n, m in model.named_modules() if m is output), None)
    if name is None:  # pragma: no cover - an output embedding outside the tree
        return ()
    replace_module(model, name, DynQuantLinear(table.weight_qt, None, tied_to=table))
    _disable_retying(model)
    return (name,)


def _disable_retying(model: Any) -> None:
    """Make ``tie_weights`` a no-op on this instance, for the reason above.

    Bound on the instance rather than handled by setting
    ``config.tie_word_embeddings = False``, which would be a lie that ``save_pretrained``
    writes to disk: the model *is* tied, and a later reader told otherwise would
    allocate two tables.
    """

    def _already_tied(*_args: Any, **_kwargs: Any) -> None:
        return None

    model.tie_weights = _already_tied
    _forget_tied_parameter_names(model)


def _forget_tied_parameter_names(model: Any) -> None:
    """Drop tie bookkeeping that names tensors this model no longer owns.

    ``tie_weights`` was the only thing that acted on the tie when the line above it was
    written. It is not any more. transformers v5 records the same fact a second way, in
    an ``all_tied_weights_keys`` mapping built at ``post_init``, and
    ``mark_tied_weights_as_initialized`` walks that mapping calling ``get_parameter`` on
    every name -- ``AttributeError: DynQuantLinear has no attribute 'weight'``, raised
    from ``_finalize_model_loading`` after the entire checkpoint has been read
    correctly. Silencing one reader of a fact does not change the fact, and this is the
    sixth time in this project that a second copy of one registry has agreed with the
    first right up until it did not.

    So the fact is corrected rather than another reader silenced. After the swap
    ``lm_head.weight`` is not a parameter of anything: the head registers no tensors and
    reads the embedding's buffers (see ``_PackedModule.__init__``). That is a tie the
    module tree carries structurally and the parameter namespace cannot express, so the
    honest entry for it is no entry. Pairs whose two sides both still resolve are kept,
    so a model tying something other than its head keeps that bookkeeping.

    ``remove_duplicate=False`` because a tie transformers has *already* established
    hides one of its two names from the default iteration; dropping an entry for being
    invisible rather than for being absent would silently untie a live pair.
    """
    tied = getattr(model, "all_tied_weights_keys", None)
    if not tied:
        return
    present = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    present |= {name for name, _ in model.named_buffers(remove_duplicate=False)}
    model.all_tied_weights_keys = {
        target: source for target, source in tied.items() if target in present and source in present
    }


def _empty_like(weight: torch.Tensor, spec: ModuleQuantSpec) -> QuantTensor:
    """Correctly shaped, unfilled packed buffers for one target.

    The geometry comes from the *model* -- a checkpoint records ``out_features`` as
    the packer's row count, which is enough to check a shard against and not enough
    to rebuild rank 3 from -- and the scale dtype from ``storage_dtype``, the same
    function the exporter wrote them with.
    """
    logical_shape = tuple(int(extent) for extent in weight.shape)
    return QuantTensor.empty(
        bits=spec.bits,
        group_size=spec.group_size,
        in_features=logical_shape[-1],
        logical_shape=logical_shape,
        # A `meta` skeleton still carries the dtype the model was built in, and
        # `storage_dtype` turns that into the dtype the exporter wrote the scales in --
        # 16-bit, unless the weight was already half. Calling the same function both
        # ends keeps the skeleton shaped like the shards without a second rule here.
        # It does not have to be exactly right: `load_state_dict` casts on copy. It has
        # to be right for `assign=True`, which the low-memory loader uses, where a
        # mismatch would silently leave the buffer in whichever dtype won the race.
        dtype=storage_dtype(weight),
        device=weight.device,
        symmetric=spec.symmetric,
        has_offsets=not spec.symmetric,
    )


def _prepare_restore(
    target: RestoredWeight, spec: ModuleQuantSpec
) -> tuple[QuantTensor, torch.dtype]:
    """Give a router packed buffers to be loaded into, and take its parameter away.

    The parameter has to go: ``transformers`` reports every key the skeleton has and
    the checkpoint does not, and a module still advertising ``weight`` would produce
    one missing-key warning per router -- indistinguishable, to a reader, from the
    silent-random-model failure this quantizer exists to prevent.

    ``weight`` comes back immediately as a **non-persistent** buffer holding nothing,
    so the module is never without the attribute its own forward reads, and so that
    ``state_dict`` -- and therefore ``save_pretrained`` -- carries the three packed
    tensors and not a dense fourth. A checkpoint saved from a loaded model is then
    the one that was loaded, which is what ``is_serializable`` claims.
    """
    empty = _empty_like(target.weight, spec)
    module = target.module
    del module._parameters["weight"]
    for suffix, tensor in (
        (QUANT_TENSOR_SUFFIXES["packed"], empty.packed),
        (QUANT_TENSOR_SUFFIXES["scale"], empty.scales),
        (QUANT_TENSOR_SUFFIXES["offset"], empty.offsets),
    ):
        if tensor is not None:
            module.register_buffer(suffix, tensor, persistent=True)
    module.register_buffer("weight", target.weight.detach(), persistent=False)
    return empty, target.weight.dtype


def _restore_weight(module: nn.Module, empty: QuantTensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Dequantize a router's packed buffers back into the weight its forward reads.

    Reads the buffers off the module rather than closing over the tensors handed to
    :func:`_prepare_restore`: the low-memory loader copies with ``assign=True``, which
    *replaces* the buffer object, so the tensors held from before the load are the
    empty ones and comparing against them would compare against garbage.
    """
    loaded = replace(
        empty,
        packed=module.get_buffer(QUANT_TENSOR_SUFFIXES["packed"]),
        scales=module.get_buffer(QUANT_TENSOR_SUFFIXES["scale"]),
        offsets=(
            None if empty.offsets is None else module.get_buffer(QUANT_TENSOR_SUFFIXES["offset"])
        ),
    )
    loaded.validate()
    weight = loaded.dequantize(dtype=out_dtype)
    module.register_buffer("weight", weight, persistent=False)
    return weight


def _shell(target: nn.Linear | nn.Embedding | ExpertBank, spec: ModuleQuantSpec) -> nn.Module:
    """An empty packed module of the right shape, on the original's own device.

    ``device=target.weight.device`` rather than a concrete device: under
    ``low_cpu_mem_usage`` the skeleton sits on ``meta``, and allocating real storage
    here would materialise the model in RAM at precisely the moment the loader is
    trying not to.

    The shape comes from the *model*, not from ``spec``. A checkpoint records
    ``out_features`` as the packer's row count -- ``E * out`` for a bank -- which is
    enough to check a shard against but not enough to rebuild rank 3 from, and the
    skeleton transformers has already built from ``config.json`` knows the real one.
    """
    weight = target.weight
    empty = _empty_like(weight, spec)
    if isinstance(target, ExpertBank):
        return DynQuantExpertBank(empty, out_dtype=weight.dtype)
    if isinstance(target, nn.Embedding):
        return DynQuantEmbedding(empty, padding_idx=target.padding_idx)
    return DynQuantLinear(empty, target.bias)


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

_registered = False


def register_hf_quantizer() -> bool:
    """Make ``quant_method: dynquant`` resolvable. Idempotent; returns whether it ran.

    Exported at the top level as ``dynquant.register_hf_quantizer`` -- that is the
    name in the model card, so it is the name that has to keep working.

    ``transformers`` keeps one process-global mapping and raises when a name is
    registered twice, so a second call has to be a no-op rather than an error:
    a library that wants to be certain registration happened must be able to say
    so without knowing whether the application already did.
    """
    global _registered
    if _registered:
        return False
    _require_transformers()
    from transformers.quantizers.auto import (
        AUTO_QUANTIZATION_CONFIG_MAPPING,
        AUTO_QUANTIZER_MAPPING,
        register_quantization_config,
        register_quantizer,
    )

    if (
        HF_QUANT_METHOD in AUTO_QUANTIZER_MAPPING
        or HF_QUANT_METHOD in AUTO_QUANTIZATION_CONFIG_MAPPING
    ):
        # Already there -- a reload, or a second dynquant on the path. Not an error,
        # and not something to overwrite: whichever object is registered is the one
        # the rest of the process has already been handed.
        _registered = True
        return False

    register_quantization_config(HF_QUANT_METHOD)(build_config_class())
    register_quantizer(HF_QUANT_METHOD)(build_quantizer_class())
    _registered = True
    return True


def packed_module_names(model: nn.Module) -> list[str]:
    """Every module in a loaded model whose weight is held packed.

    The measurement behind "did the load actually work": a model that loaded through
    this quantizer and reports an empty list is one whose modules were never swapped,
    which is the failure mode being guarded against and is not otherwise visible.
    """
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, DynQuantLinear | DynQuantEmbedding)
    ]


def dense_weight_bytes(model: nn.Module) -> int:
    """Bytes still held as dense Linear/Embedding weights after a load.

    Pairs with :func:`packed_module_names`: the claim is not that some modules are
    packed but that the ones the config named are, and a non-trivial number here on a
    fully-quantized checkpoint means the swap missed something.
    """
    seen: set[int] = set()
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear | nn.Embedding):
            weight = module.weight
            if weight.data_ptr() in seen or weight.is_meta:
                continue
            seen.add(weight.data_ptr())
            total += weight.numel() * weight.element_size()
    return total
