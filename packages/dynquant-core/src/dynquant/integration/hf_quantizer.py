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

What it does not do yet
-----------------------
Batched MoE expert banks and routers that own a bare weight are refused by name, with
the reason and the alternative. A bank is a 3-D ``nn.Parameter`` its parent indexes
directly; a router runs top-k rather than a plain matmul. Neither has a forward this
can replace -- literally the same boundary :func:`dynquant.runtime.linear.pack_model`
draws, so it is drawn by the same function: ``resolve_target``, called here with
``source="quantization_config"`` and there with a bit map. A second copy of that
resolver has already been the cause of two wrong refusals in this project, and this
module briefly held a third. On an MoE checkpoint the refusal is now a hard error at
load time, which is the whole improvement over the silent random model above.

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

from typing import Any, cast

from torch import nn

from dynquant._logging import get_logger
from dynquant.constants import HF_QUANT_METHOD
from dynquant.errors import MissingDependencyError
from dynquant.integration.serving_common.schema import (
    ModuleQuantSpec,
    QuantizationConfigSchema,
)
from dynquant.quant.tensor import QuantTensor, storage_dtype
from dynquant.runtime.linear import (
    DynQuantEmbedding,
    DynQuantExpertBank,
    DynQuantLinear,
    ExpertBank,
    _PackedModule,
    replace_module,
    resolve_target,
    use_eager_experts,
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
            for name, spec in sorted(schema.modules.items()):
                target = resolve_target(model, name, source="quantization_config")
                shells[name] = _shell(target, spec)
                # The key that used to hold this target's dense weight, which is the
                # key the loader will now miss. A module contributed `name.weight`;
                # a bank *is* the parameter, so it contributed `name`.
                dense_keys.add(name if isinstance(target, ExpertBank) else f"{name}.weight")
                replace_module(model, name, shells[name])
            packed = set(schema.modules)
            tied = _tie_output_embedding(model, packed)
            packed |= set(tied)
            dense_keys |= {f"{name}.weight" for name in tied}
            self._packed_names = tuple(sorted(packed))
            self._dense_keys = frozenset(dense_keys)
            # A bank installed here reaches a forward the same way a packed one does,
            # so it needs the same dispatch. `pack_model` does this for the quantize
            # path; nothing shared runs on this one.
            if any(isinstance(shell, DynQuantExpertBank) for shell in shells.values()):
                use_eager_experts(model)
            _log.info(
                "prepared %d packed modules for loading%s",
                len(self._packed_names),
                f", {tied[0]} reading the input table" if tied else "",
            )

        def _process_model_after_weight_loading(self, model: Any, **kwargs: Any) -> Any:
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
    logical_shape = tuple(int(extent) for extent in weight.shape)
    in_features = logical_shape[-1]
    empty = QuantTensor.empty(
        bits=spec.bits,
        group_size=spec.group_size,
        in_features=in_features,
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
