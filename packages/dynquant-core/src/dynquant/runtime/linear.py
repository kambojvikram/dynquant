"""Modules that hold their weights packed, and the surgery that installs them.

This is where the compression stops being a property of a file and becomes a
property of a running model. :func:`dynquant.quant.quantizer.quantize_model` with
``in_place=True`` writes reconstructed values back over fp16 parameters: correct
numbers, unchanged memory. The modules here store the ``int32`` words and nothing
else, so a 3.25-bit model occupies 3.25 bits per weight in VRAM and the dense
weight is never allocated on the decode path.

Why the buffers are held flat rather than as a QuantTensor field
----------------------------------------------------------------
``nn.Module`` only moves and saves things it knows are tensors. A
:class:`~dynquant.quant.tensor.QuantTensor` stored as a plain attribute would be
invisible to ``.to(device)`` and absent from ``state_dict()``. So the three tensors
are registered buffers and the ``QuantTensor`` is rebuilt per call from them plus
the immutable integer metadata. Rebuilding is a dataclass construction with no
tensor work; it is free next to the matmul that follows.

The buffer names (``qweight``, ``scales``, ``offsets``) are the ones the checkpoint
format uses, so a ``state_dict`` from a live model and a DynQuant checkpoint have
the same keys.

How a tied table stays one table
--------------------------------
Registering the *same* tensor as a buffer on two modules does not survive a device
move -- ``nn.Module._apply`` converts each registered buffer independently and
memoizes nothing, so one CPU tensor comes back as two on the GPU. Since a tied
embedding and LM head are 27% of a Qwen3.5-2B, that would have made the memory
claim false precisely when the weights reached the device it is claimed about. So
the tied module registers no table at all and reads the owner's: see
``_PackedModule.__init__``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from dynquant._logging import get_logger
from dynquant.constants import DEFAULT_GROUP_SIZE, QUANT_TENSOR_SUFFIXES
from dynquant.errors import DynQuantError
from dynquant.quant.grid import CLIP_CANDIDATES, quantize_with_search
from dynquant.quant.tensor import QuantLayout, QuantTensor

from . import ops

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "DynQuantEmbedding",
    "DynQuantLinear",
    "PackReport",
    "pack_model",
    "packed_bytes",
]

_log = get_logger(__name__)

_PACKED = QUANT_TENSOR_SUFFIXES["packed"]
_SCALES = QUANT_TENSOR_SUFFIXES["scale"]
_OFFSETS = QUANT_TENSOR_SUFFIXES["offset"]


class _PackedModule(nn.Module):
    """Shared storage and plumbing for the two packed module types."""

    def __init__(self, weight: QuantTensor, *, tied_to: _PackedModule | None = None) -> None:
        super().__init__()
        if weight.layout is not QuantLayout.LINEAR:
            raise NotImplementedError(f"layout {weight.layout.value} is not runnable yet")
        if len(weight.logical_shape) != 2:
            raise ValueError(f"expected a 2-D weight, got {weight.logical_shape}")

        if tied_to is None:
            self.register_buffer(_PACKED, weight.packed)
            self.register_buffer(_SCALES, weight.scales)
            # Passed through as None when the tensor has none, rather than filled
            # with zeros: numerically identical, but a zero buffer costs one metadata
            # entry per group and would push a checkpoint that saved them back up to
            # the asymmetric footprint. `register_buffer` accepts None and keeps the
            # name in `state_dict` handling, so the two cases stay one code path.
            self.register_buffer(_OFFSETS, weight.offsets)
        else:
            # A tied module registers no tensors of its own and reads the owner's.
            #
            # Sharing one tensor between two `register_buffer` calls does *not*
            # survive `model.to("cuda")`: `nn.Module._apply` calls its conversion
            # function once per registered buffer with no memo, so one CPU tensor
            # referenced twice becomes two CUDA tensors. On Qwen3.5-2B the tied
            # embedding/LM-head table is 508.6M of 1881.3M parameters, so the model
            # would arrive on the GPU 27% larger than the manifest says -- the packed
            # runtime's entire reason for existing, falsified silently at the moment
            # the weights move.
            #
            # Holding the owner instead makes the sharing structural rather than
            # incidental: there is one tensor because there is one buffer. It is
            # stored through `object.__setattr__` so `nn.Module.__setattr__` does not
            # file it under `_modules`, where it would show up a second time in
            # `named_modules()` and in every `state_dict` key prefix.
            #
            # The consequence is that a tied module's `state_dict` carries only its
            # bias. That is the same convention `transformers` uses for tied weights
            # (`_tied_weights_keys`), and it is what a checkpoint should contain: one
            # table, written once.
            object.__setattr__(self, "_tied_to", tied_to)

        self.bits = weight.bits
        self.group_size = weight.group_size
        self.in_features = weight.in_features
        self.out_features = weight.num_rows
        self.symmetric = weight.symmetric
        self.row_offset = weight.row_offset

    @property
    def holder(self) -> _PackedModule:
        """The module the buffers actually live on: this one, or the one it is tied to."""
        owner: _PackedModule = self.__dict__.get("_tied_to", self)
        return owner

    @property
    def weight_qt(self) -> QuantTensor:
        """A view of this module's buffers as a QuantTensor. Copies nothing."""
        holder = self.holder
        return QuantTensor(
            packed=holder.get_buffer(_PACKED),
            scales=holder.get_buffer(_SCALES),
            offsets=getattr(holder, _OFFSETS),
            bits=self.bits,
            group_size=self.group_size,
            in_features=self.in_features,
            logical_shape=(self.out_features, self.in_features),
            row_offset=self.row_offset,
            layout=QuantLayout.LINEAR,
            symmetric=self.symmetric,
        )

    @property
    def nbytes(self) -> int:
        """Bytes this module's weight occupies, metadata included."""
        return self.weight_qt.nbytes

    def extra_repr(self) -> str:
        group = "per-row" if self.group_size < 0 else f"g{self.group_size}"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, {group}, {self.weight_qt.bits_per_weight:.2f} bits/weight"
        )


class DynQuantLinear(_PackedModule):
    """``nn.Linear`` whose weight lives packed.

    Drop-in for the forward direction only. There is no backward: the packed
    representation has no gradient with respect to anything, and pretending
    otherwise (a straight-through estimator, say) would let someone fine-tune a
    model whose weights cannot represent what they learn. Quantize after training.
    """

    # A bare annotation, so no class attribute is created and `register_buffer`
    # behaves exactly as it would without it. It is here because `nn.Module`'s
    # `__getattr__` is typed as returning `Tensor | Module`, which makes every read
    # of a registered buffer ambiguous to a type checker; narrowing at the call site
    # instead would put an `isinstance` check on the decode path.
    bias: torch.Tensor | None

    def __init__(
        self,
        weight: QuantTensor,
        bias: torch.Tensor | None = None,
        *,
        tied_to: _PackedModule | None = None,
    ) -> None:
        super().__init__(weight, tied_to=tied_to)
        # Registered on this module even when the weight is tied. Two modules sharing
        # a weight table do not thereby share a bias -- a tied LM head typically has
        # none while the embedding it shares with has none either, but nothing in the
        # format says they must agree, and borrowing the owner's would be wrong the
        # first time they differ.
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int,
        *,
        group_size: int = DEFAULT_GROUP_SIZE,
        symmetric: bool = False,
        candidates: Sequence[float] = CLIP_CANDIDATES,
    ) -> DynQuantLinear:
        quantized, _ = quantize_with_search(
            linear.weight.detach(),
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            candidates=candidates,
            compute_dtype=_storage_dtype(linear.weight),
        )
        return cls(quantized, linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ops.quantized_matmul(x, self.weight_qt, self.bias)


class DynQuantEmbedding(_PackedModule):
    """``nn.Embedding`` whose table lives packed.

    The table is gathered *packed* and only the gathered rows are dequantized, so
    the 151k-row table of a Qwen3.5-2B never exists densely -- which is the whole
    27% of that model's parameters that its tied embedding accounts for.
    """

    def __init__(
        self,
        weight: QuantTensor,
        *,
        padding_idx: int | None = None,
        tied_to: _PackedModule | None = None,
    ) -> None:
        super().__init__(weight, tied_to=tied_to)
        self.padding_idx = padding_idx
        # Names matching nn.Embedding's, so code that introspects an embedding
        # (resize_token_embeddings, tokenizer sanity checks) still finds them.
        self.num_embeddings = self.out_features
        self.embedding_dim = self.in_features

    @classmethod
    def from_embedding(
        cls,
        embedding: nn.Embedding,
        bits: int,
        *,
        group_size: int = DEFAULT_GROUP_SIZE,
        symmetric: bool = False,
        candidates: Sequence[float] = CLIP_CANDIDATES,
    ) -> DynQuantEmbedding:
        quantized, _ = quantize_with_search(
            embedding.weight.detach(),
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            candidates=candidates,
            compute_dtype=_storage_dtype(embedding.weight),
        )
        return cls(quantized, padding_idx=embedding.padding_idx)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return ops.embedding_lookup(self.weight_qt, ids)


# --------------------------------------------------------------------------
# Model surgery
# --------------------------------------------------------------------------


class PackReport:
    """What ``pack_model`` did, in bytes.

    The two totals are the claim: ``fp16_bytes`` is what the same weights occupied
    before, ``packed_bytes`` is what they occupy now, and both are measured from
    the tensors rather than predicted from the bit map.
    """

    def __init__(self) -> None:
        self.modules: dict[str, tuple[int, int, int]] = {}
        """name -> (bits, dense bytes before, packed bytes after)."""
        self.tied: dict[str, str] = {}
        """name -> the name whose packed tensor it shares."""
        self.skipped: list[str] = []

    @property
    def fp16_bytes(self) -> int:
        return sum(before for _, before, _ in self.modules.values())

    @property
    def packed_bytes(self) -> int:
        return sum(after for _, _, after in self.modules.values())

    def summary(self) -> str:
        if not self.modules:
            return "packed nothing"
        ratio = self.fp16_bytes / max(self.packed_bytes, 1)
        lines = [
            f"packed {len(self.modules)} modules: "
            f"{self.fp16_bytes / 2**30:.3f} GiB -> {self.packed_bytes / 2**30:.3f} GiB "
            f"({ratio:.2f}x)"
        ]
        if self.tied:
            shared = sum(
                self.modules[target][2]
                for target in set(self.tied.values())
                if target in self.modules
            )
            lines.append(
                f"  {len(self.tied)} tied module(s) share {shared / 2**30:.3f} GiB "
                f"of that, counted once"
            )
        if self.skipped:
            lines.append(f"  {len(self.skipped)} left dense: {', '.join(self.skipped[:5])}")
        return "\n".join(lines)


def pack_model(
    model: nn.Module,
    bits: Mapping[str, int],
    *,
    group_size: int = DEFAULT_GROUP_SIZE,
    symmetric: bool = False,
    candidates: Sequence[float] = CLIP_CANDIDATES,
    progress: Callable[[int, int], None] | None = None,
) -> PackReport:
    """Replace every module named in ``bits`` with its packed equivalent, in place.

    Modules are replaced one at a time and the dense weight is dropped as soon as
    its packed form exists, so peak memory is the model plus one weight -- not the
    model plus a second copy of it.

    Tied weights
    ------------
    A bit map lists one representative per tied group (see
    :mod:`dynquant.quant.quantizer`), so packing only the names it contains would
    leave the ``lm_head`` of a tied model pointing at a dense parameter that is now
    referenced by nothing else -- 27% of a Qwen3.5-2B still resident in fp16, and
    the memory measurement silently wrong. So after the named modules are packed,
    every remaining Linear/Embedding whose weight shared storage with one of them is
    replaced too, sharing the *same* packed buffers. One table, two views, counted
    once.
    """
    report = PackReport()
    targets = sorted(bits.items())

    # Snapshot the storage identity of every candidate weight before anything is
    # replaced. Once a module is swapped out its Parameter may be freed, and a
    # data_ptr collected afterwards could belong to a different tensor entirely.
    identity: dict[int, list[str]] = {}
    for name, module in _quantizable_modules(model):
        identity.setdefault(module.weight.data_ptr(), []).append(name)

    owner_by_ptr: dict[int, tuple[str, QuantTensor, _PackedModule]] = {}

    for index, (name, width) in enumerate(targets):
        module = _resolve_module(model, name)
        weight = module.weight
        ptr = weight.data_ptr()
        dense_bytes = weight.numel() * weight.element_size()

        with torch.no_grad():
            quantized, _ = quantize_with_search(
                weight.detach(),
                bits=width,
                group_size=group_size,
                symmetric=symmetric,
                candidates=candidates,
                compute_dtype=_storage_dtype(weight),
            )
        replacement = _wrap(module, quantized)
        owner_by_ptr[ptr] = (name, quantized, replacement)
        _replace(model, name, replacement)
        report.modules[name] = (width, dense_bytes, quantized.nbytes)

        if progress is not None:
            progress(index + 1, len(targets))

    # Second pass: the aliases. Each is given the *owning module*, not a second
    # registration of the owner's tensors -- see `_PackedModule.__init__` for why
    # the difference decides whether the saving survives `model.to("cuda")`.
    for ptr, names in identity.items():
        owner = owner_by_ptr.get(ptr)
        if owner is None:
            continue
        owner_name, quantized, holder = owner
        for name in names:
            if name in report.modules:
                continue
            module = _resolve_module(model, name)
            _replace(model, name, _wrap(module, quantized, tied_to=holder))
            report.tied[name] = owner_name

    for name, _module in _quantizable_modules(model):
        report.skipped.append(name)

    _log.info("%s", report.summary())
    return report


def _quantizable_modules(model: nn.Module) -> list[tuple[str, nn.Linear | nn.Embedding]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear | nn.Embedding)
    ]


def _resolve_module(model: nn.Module, name: str) -> nn.Linear | nn.Embedding:
    try:
        module = model.get_submodule(name)
    except AttributeError as exc:
        raise DynQuantError(f"bit map names {name!r}, which is not a module of this model") from exc
    if not isinstance(module, nn.Linear | nn.Embedding):
        raise DynQuantError(
            f"{name!r} is a {type(module).__name__}; the packed runtime replaces Linear and "
            f"Embedding only. Batched MoE expert banks need the grouped path."
        )
    return module


def _wrap(
    module: nn.Linear | nn.Embedding,
    quantized: QuantTensor,
    *,
    tied_to: _PackedModule | None = None,
) -> _PackedModule:
    if isinstance(module, nn.Embedding):
        return DynQuantEmbedding(quantized, padding_idx=module.padding_idx, tied_to=tied_to)
    return DynQuantLinear(quantized, module.bias, tied_to=tied_to)


def _replace(model: nn.Module, name: str, replacement: nn.Module) -> None:
    parent_name, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, leaf, replacement)


def _storage_dtype(weight: torch.Tensor) -> torch.dtype:
    """The dtype scales and offsets are stored in: the weight's own.

    Not "whatever is smallest". The scale dtype is the module's compute dtype --
    it is what ``dequantize`` returns and what the kernels require the activation
    to match -- so choosing fp16 metadata for an fp32 model makes every packed
    module emit fp16 into an fp32 graph, and the first Linear downstream fails on
    a dtype mismatch. The metadata saving would be real (0.25 bits/weight at group
    128) and irrelevant: nobody serves an fp32 model.
    """
    return weight.dtype


def packed_bytes(model: nn.Module) -> dict[str, Any]:
    """Measured VRAM accounting for a packed model: what is packed, what is not.

    Walks the live module tree rather than the bit map, so anything the surgery
    missed shows up in ``dense_bytes`` instead of being quietly excluded from the
    total. A model that reports a small ``packed_bytes`` and a large
    ``dense_bytes`` is a model whose compression did not happen.
    """
    packed = 0
    dense = 0
    dense_names: list[str] = []
    seen: set[int] = set()

    for name, module in model.named_modules():
        if isinstance(module, _PackedModule):
            # Through `holder`, so a tied alias reports the table it reads rather
            # than raising for a buffer it deliberately does not register.
            ptr = module.holder.get_buffer(_PACKED).data_ptr()
            if ptr in seen:
                continue  # a tied alias; its bytes are already counted
            seen.add(ptr)
            packed += module.nbytes
        elif isinstance(module, nn.Linear | nn.Embedding):
            weight = module.weight
            if weight.data_ptr() in seen:
                continue
            seen.add(weight.data_ptr())
            dense += weight.numel() * weight.element_size()
            dense_names.append(name)

    return {
        "packed_bytes": packed,
        "dense_bytes": dense,
        "dense_modules": dense_names,
        "total_bytes": packed + dense,
    }
