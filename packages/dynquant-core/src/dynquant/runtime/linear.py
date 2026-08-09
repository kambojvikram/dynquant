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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from dynquant._logging import get_logger
from dynquant.constants import DEFAULT_GROUP_SIZE, QUANT_TENSOR_SUFFIXES
from dynquant.errors import DynQuantError
from dynquant.quant.device import quantize_tensor, resolve_compute_device
from dynquant.quant.grid import CLIP_CANDIDATES
from dynquant.quant.tensor import QuantLayout, QuantTensor

from . import ops

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "DynQuantEmbedding",
    "DynQuantExpertBank",
    "DynQuantLinear",
    "ExpertBank",
    "PackReport",
    "pack_model",
    "packed_bytes",
    "replace_module",
    "resolve_target",
]

_log = get_logger(__name__)

_PACKED = QUANT_TENSOR_SUFFIXES["packed"]
_SCALES = QUANT_TENSOR_SUFFIXES["scale"]
_OFFSETS = QUANT_TENSOR_SUFFIXES["offset"]


class _PackedModule(nn.Module):
    """Shared storage and plumbing for the two packed module types."""

    def __init__(self, weight: QuantTensor, *, tied_to: _PackedModule | None = None) -> None:
        super().__init__()
        # Decide the backend here, where it is an ordinary function call, rather
        # than in the first forward, where it would be one `torch.compile` is
        # tracing. See the comment on the caches in `runtime.ops`.
        ops.warm_dispatch()
        if weight.layout is not QuantLayout.LINEAR:
            raise NotImplementedError(f"layout {weight.layout.value} is not runnable yet")
        if len(weight.logical_shape) < 2:
            raise ValueError(f"expected a matrix, or a stack of them, got {weight.logical_shape}")

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
        self.logical_shape = tuple(weight.logical_shape)
        # Rows of *one* matrix. Identical to `num_rows` for a 2-D weight, which is
        # every case this attribute had before a stack of them existed; on a bank
        # `[E, out, in]` it is `out`, which is what the parent means by out_features
        # and what one expert's dequantized slice has to be.
        self.out_features = int(self.logical_shape[-2])
        self.num_rows = weight.num_rows
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
            logical_shape=self.logical_shape,
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
        compute_device: str | torch.device | None = "auto",
    ) -> DynQuantLinear:
        quantized, _ = quantize_tensor(
            linear.weight.detach(),
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            candidates=candidates,
            compute_dtype=_storage_dtype(linear.weight),
            device=resolve_compute_device(compute_device),
        )
        return cls(quantized.to(linear.weight.device), linear.bias)

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
        compute_device: str | torch.device | None = "auto",
    ) -> DynQuantEmbedding:
        quantized, _ = quantize_tensor(
            embedding.weight.detach(),
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            candidates=candidates,
            compute_dtype=_storage_dtype(embedding.weight),
            device=resolve_compute_device(compute_device),
        )
        return cls(quantized.to(embedding.weight.device), padding_idx=embedding.padding_idx)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return ops.embedding_lookup(self.weight_qt, ids)


class DynQuantExpertBank(_PackedModule):
    """A batched MoE expert bank, packed, standing where the 3-D Parameter stood.

    Why this is a module and not a forward
    --------------------------------------
    Every other packed type replaces a module's forward. A bank has none to replace:
    ``Lfm2MoeExperts`` (and Qwen3-Next's, and GPT-OSS's) keeps its experts as one bare
    ``nn.Parameter`` of shape ``[E, out, in]`` and calls ``F.linear`` on a *slice*
    of one, inside its own loop over the experts a token actually hit. So the packed
    thing has to be reachable exactly where the parameter was and answer exactly the
    question the parent asks of it, which is ``bank[expert]``.

    That is what this is. It is registered under the parameter's own name, so
    ``self.gate_up_proj[expert_idx]`` in an untouched parent reaches
    :meth:`__getitem__` and gets back a dense ``[out, in]`` matrix for that one
    expert. No parent is edited, no architecture is enumerated, and a family whose
    MoE block has not been written yet works if it indexes its bank -- which is the
    only access pattern a per-expert loop has.

    What it costs and what it saves
    -------------------------------
    One expert's worth of fp16 exists at a time, transiently, against the whole bank
    resident for the whole run. On LFM2.5-8B-A1B a layer's ``gate_up_proj`` is
    ``[32, 2688, 2048]`` -- 352 MiB in bf16 -- and one expert of it is 11 MiB. The
    saving is the 44 banks that are 91.5% of that model; the cost is that a token
    routed to 4 experts dequantizes 4 slices rather than reading 4 slices of packed
    memory in a fused kernel. That fused kernel is the other half of P8 and lands
    against this interface, not instead of it: ``weight_qt.rows()`` is already the
    segment addressing a grouped GEMM needs.

    What it deliberately does not support
    -------------------------------------
    A parent that treats the bank as one tensor -- ``torch.bmm(x, bank)``, or an
    ``einsum`` over all experts -- will fail, because a module is not a tensor. It
    fails loudly at the first forward rather than quietly, which is the right side
    to be on, and :meth:`dequantize` is the escape hatch for anyone who genuinely
    wants the whole bank dense and has the memory for it.
    """

    def __init__(self, weight: QuantTensor) -> None:
        if len(weight.logical_shape) != 3:
            raise ValueError(f"an expert bank is [experts, out, in]; got {weight.logical_shape}")
        super().__init__(weight)
        self.num_experts = int(self.logical_shape[0])

    @classmethod
    def from_parameter(
        cls,
        bank: torch.Tensor,
        bits: int,
        *,
        group_size: int = DEFAULT_GROUP_SIZE,
        symmetric: bool = False,
        candidates: Sequence[float] = CLIP_CANDIDATES,
        compute_device: str | torch.device | None = "auto",
    ) -> DynQuantExpertBank:
        quantized, _ = quantize_tensor(
            bank.detach(),
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            candidates=candidates,
            compute_dtype=_storage_dtype(bank),
            device=resolve_compute_device(compute_device),
        )
        return cls(quantized.to(bank.device))

    def __getitem__(self, index: Any) -> torch.Tensor:
        """Expert ``index``, dequantized. The only access pattern a bank needs.

        ``index`` arrives as a 0-d tensor from a ``torch.where``/``nonzero`` loop as
        often as it arrives as an ``int``, so it is taken through ``__index__``
        rather than type-checked.
        """
        if isinstance(index, slice):
            raise TypeError(
                "a packed expert bank is indexed one expert at a time; slicing it "
                "would dequantize a range, which is what holding it packed avoids. "
                "Use `dequantize()` if the whole bank really is wanted."
            )
        expert = int(index)
        if expert < 0:
            expert += self.num_experts
        if not 0 <= expert < self.num_experts:
            raise IndexError(f"expert {int(index)} out of range for a bank of {self.num_experts}")
        band = self.out_features
        return self.weight_qt.rows(expert * band, (expert + 1) * band).dequantize()

    def __len__(self) -> int:
        return self.num_experts

    def dequantize(self) -> torch.Tensor:
        """The whole bank, dense. Defeats the point; exists for tests and parity checks."""
        return self.weight_qt.dequantize()

    # `nn.Module` has none of these, and a parent that asks a parameter for its shape
    # or dtype before using it -- a dtype check, an `empty_like` -- is asking something
    # this can answer truthfully without materialising anything.

    @property
    def shape(self) -> torch.Size:
        return torch.Size(self.logical_shape)

    @property
    def ndim(self) -> int:
        return len(self.logical_shape)

    @property
    def dtype(self) -> torch.dtype:
        return self.holder.get_buffer(_SCALES).dtype

    @property
    def device(self) -> torch.device:
        return self.holder.get_buffer(_PACKED).device

    def extra_repr(self) -> str:
        group = "per-row" if self.group_size < 0 else f"g{self.group_size}"
        return (
            f"experts={self.num_experts}, out_features={self.out_features}, "
            f"in_features={self.in_features}, bits={self.bits}, {group}, "
            f"{self.weight_qt.bits_per_weight:.2f} bits/weight"
        )


@dataclass(frozen=True)
class ExpertBank:
    """A bare 3-D parameter :func:`resolve_target` accepts, and where it lives.

    Carries ``weight`` under that name so the three things a caller can be handed --
    a ``Linear``, an ``Embedding`` and this -- all answer ``target.weight`` with the
    dense tensor to encode. The packing loop then needs no branch at all until the
    moment it builds the replacement.
    """

    name: str
    weight: torch.Tensor


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
    compute_device: str | torch.device | None = "auto",
    progress: Callable[[int, int], None] | None = None,
) -> PackReport:
    """Replace every module named in ``bits`` with its packed equivalent, in place.

    Modules are replaced one at a time and the dense weight is dropped as soon as
    its packed form exists, so peak memory is the model plus one weight -- not the
    model plus a second copy of it.

    Where the encoding runs
    -----------------------
    ``compute_device`` defaults to the accelerator and is independent of where the
    model sits, which matters most for the case this function exists to serve:
    packing a CPU-resident model precisely so that no dense weight ever reaches the
    GPU. Doing the arithmetic there anyway costs one tensor's working set,
    transiently, and each packed result is returned to the model's own device before
    the module is replaced -- so the model does not move and a subsequent
    ``model.to("cuda")`` still carries packed buffers and nothing else. The saving
    is large: a 7 B that took forty minutes to pack takes about one.

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
    device = resolve_compute_device(compute_device)

    # Snapshot the storage identity of every candidate weight before anything is
    # replaced. Once a module is swapped out its Parameter may be freed, and a
    # data_ptr collected afterwards could belong to a different tensor entirely.
    identity: dict[int, list[str]] = {}
    for name, module in _quantizable_modules(model):
        identity.setdefault(module.weight.data_ptr(), []).append(name)

    owner_by_ptr: dict[int, tuple[str, QuantTensor, _PackedModule]] = {}

    for index, (name, width) in enumerate(targets):
        target = resolve_target(model, name)
        weight = target.weight
        ptr = weight.data_ptr()
        dense_bytes = weight.numel() * weight.element_size()

        with torch.no_grad():
            quantized, _ = quantize_tensor(
                weight.detach(),
                bits=width,
                group_size=group_size,
                symmetric=symmetric,
                candidates=candidates,
                compute_dtype=_storage_dtype(weight),
                device=device,
            )
            # Back to the model's device before the swap. Leaving it where the
            # arithmetic happened would build a model whose modules sit on
            # different devices depending on which ones were quantized, and the
            # first forward pass would be where that is discovered.
            quantized = quantized.to(weight.device)
        replacement = _wrap(target, quantized)
        owner_by_ptr[ptr] = (name, quantized, replacement)
        replace_module(model, name, replacement)
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
            alias = resolve_target(model, name)
            if isinstance(alias, ExpertBank):  # pragma: no cover - see `identity`
                raise DynQuantError(
                    f"{name!r} resolved to an expert bank in the alias pass, which walks "
                    f"Linear and Embedding modules only. Two banks sharing storage is not "
                    f"something the packer knows how to tie."
                )
            replace_module(model, name, _wrap(alias, quantized, tied_to=holder))
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


_ENCODER_REMEDY = (
    "The encoder reaches it, so score this map with `dynquant eval --map-apply encode` or "
    "write it with `dynquant quantize --map`; both give the same accuracy, and "
    "`dynquant export` packs it at the bytes its manifest claims."
)
"""What to do instead, for every target the packed runtime declines but the encoder takes.

One sentence and one copy of it. Each refusal below adds only what is specific to its own
case, because these three messages differ in *why* and not in *what next*, and the version
that spelled the remedy out once and left the other branches as dead ends is what this
replaces.
"""


def resolve_target(
    model: nn.Module, name: str, *, source: str = "bit map"
) -> nn.Linear | nn.Embedding | ExpertBank:
    """What the packed runtime will replace, or a refusal that says which kind.

    Public, and shared with the ``from_pretrained`` load path, because that path asks
    exactly this question: which named targets can be swapped for a packed module. It
    had its own three-branch copy of this for about an hour -- narrower than this one by
    the container case, which is how every previous copy in this project started. The
    only thing that genuinely differs between the two callers is what to call the thing
    that named the target, so that is the only thing parameterised: a ``bit map`` when
    packing a live model, a ``quantization_config`` when loading a checkpoint.

    Three failures, three messages, and each of the first two once wore the wrong one.

    A name this model does not have and a name addressing a *tensor* both raise
    ``AttributeError`` out of ``get_submodule``, and both were reported as "not a module of
    this model" -- which reads as a map built for another checkpoint. It is not: a batched
    MoE expert bank keeps its experts as bare 3-D parameters, so the allocator priced it
    and the encoder encodes it. The runtime now holds one too, as
    :class:`DynQuantExpertBank`, so that case returns rather than refusing -- but only at
    rank 3. A bare *2-D* parameter is still refused, and for a reason that survives the
    grouped path: its owner passes it whole to ``F.linear``, so there is no index at which
    a module could stand in for it.

    The third is a module that owns a weight and is neither class. An
    ``Lfm2MoeTopKRouter`` -- and Qwen3-Next's router, and GPT-OSS's -- holds
    ``[num_experts, hidden]`` as its own parameter and calls ``F.linear`` on it, so there is
    a weight to quantize but no ``Linear`` forward to replace. That case was being told it
    was a batched expert bank and pointed at a grouped path it does not need. The refusal
    itself is right in all three: packing swaps a module's forward, and these are the shapes
    that have no forward to swap.
    """
    try:
        module = model.get_submodule(name)
    except AttributeError as exc:
        from dynquant.quant.quantizer import resolves_to_weight, target_tensor

        if resolves_to_weight(model, name):
            tensor = target_tensor(model, name)
            if tensor.ndim == 3:
                return ExpertBank(name=name, weight=tensor)
            raise DynQuantError(
                f"{name!r} is a bare {tensor.ndim}-D parameter, not a module. The packed "
                f"runtime holds a batched expert bank -- rank 3, indexed one expert at a "
                f"time by its parent -- and this is not one: whatever owns it passes it "
                f"whole to `F.linear`, so no module can stand where it stands. "
                f"{_ENCODER_REMEDY}"
            ) from exc
        raise DynQuantError(
            f"{source} names {name!r}, which is not a module of this model"
        ) from exc
    if not isinstance(module, nn.Linear | nn.Embedding):
        from dynquant.quant.quantizer import resolves_to_weight

        if resolves_to_weight(model, name):
            raise DynQuantError(
                f"{name!r} is a {type(module).__name__}, which owns a weight but is not a "
                f"Linear or an Embedding, so the packed runtime has no forward to replace "
                f"-- an MoE router calling `F.linear` on its own parameter is the usual "
                f"case, and it is not a batched expert bank. {_ENCODER_REMEDY}"
            )
        raise DynQuantError(
            f"{name!r} is a {type(module).__name__} and owns no weight to quantize. A "
            f"{source} names the leaves that hold weights, not the containers that hold "
            f"them."
        )
    return module


def _wrap(
    target: nn.Linear | nn.Embedding | ExpertBank,
    quantized: QuantTensor,
    *,
    tied_to: _PackedModule | None = None,
) -> _PackedModule:
    if isinstance(target, ExpertBank):
        # No `tied_to`: aliasing is discovered by walking Linear and Embedding
        # modules, and a bank is in no alias group -- two layers' experts are two
        # tensors. If that ever stops being true the alias pass will find them.
        return DynQuantExpertBank(quantized)
    if isinstance(target, nn.Embedding):
        return DynQuantEmbedding(quantized, padding_idx=target.padding_idx, tied_to=tied_to)
    return DynQuantLinear(quantized, target.bias, tied_to=tied_to)


def replace_module(model: nn.Module, name: str, replacement: nn.Module) -> None:
    """Install ``replacement`` where ``name`` points, module or bare parameter.

    Public because the ``from_pretrained`` path does the same surgery and had its own
    copy, which is how the resolver above came to exist in five versions.

    The ``_parameters`` line is what makes a bank work. ``nn.Module.__setattr__``
    checks the parameter registry first and raises ``TypeError: cannot assign ... as
    parameter`` when a name registered there is handed anything but a ``Parameter`` --
    so a bank cannot be replaced by a module while the parameter is still registered
    under that name. For every other target the pop finds nothing, because a Linear
    lives in ``_modules``.
    """
    parent_name, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    parent._parameters.pop(leaf, None)
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

    Three passes, because two of them cannot see everything. The module pass finds
    packed modules and dense Linears and Embeddings. The third finds what
    ``named_modules`` structurally cannot: a weight that is a bare
    :class:`~torch.nn.Parameter` on a module of some other class -- an unpacked
    expert bank, an MoE router holding ``[num_experts, hidden]`` and calling
    ``F.linear`` on it. Those were previously in neither total, so on
    LFM2.5-8B-A1B the denominator omitted 91.5% of the model and every ratio
    computed from it was flattering. Rank < 2 is skipped: norms and biases are not
    quantization targets and counting them would move the goalposts the other way.
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

    for name, param in model.named_parameters():
        if param.ndim < 2 or param.data_ptr() in seen:
            continue
        seen.add(param.data_ptr())
        dense += param.numel() * param.element_size()
        dense_names.append(name)

    return {
        "packed_bytes": packed,
        "dense_bytes": dense,
        "dense_modules": dense_names,
        "total_bytes": packed + dense,
    }
