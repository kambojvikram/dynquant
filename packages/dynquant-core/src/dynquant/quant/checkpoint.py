"""Write a packed checkpoint that vLLM and transformers can both load.

``dynquant quantize`` writes *values-quantized* weights: every element has been
through quantize-then-decode, so the accuracy is the quantized model's accuracy,
but the container is still fp16 and the directory is the same size as the
original. That output exists to measure accuracy, and its own manifest says so.

This module writes the other thing -- the one that is actually smaller. Each
quantized module becomes three tensors (``qweight``, ``scales``, ``offsets``)
instead of one ``weight``, and the widths go into ``config.json`` so a loader can
reconstruct the geometry without guessing it from tensor shapes. That guessing is
what the research code did (``scale.numel() // out_features``, with a hardcoded
128 fallback) and it is why its checkpoints could not be read back.

The encoder is :func:`dynquant.quant.device.quantize_tensor` -- the same call
:func:`dynquant.runtime.linear.pack_model` makes, with the same clipping grid.
That is deliberate and load-bearing: a checkpoint served by vLLM and the same
model packed in-process must produce the same logits, and they can only do that
if one encoder produced both.

Layout
------
Standard HF names -- ``model-00001-of-0000N.safetensors`` plus
``model.safetensors.index.json`` -- so vLLM's default loader, transformers, and
``huggingface-cli upload`` all work with no special case. Unquantized tensors
(norms, biases, rotary caches) are written unchanged under their original names,
which is what lets each model's own ``load_weights`` route them without knowing
anything about DynQuant.
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from dynquant.constants import (
    DEFAULT_GROUP_SIZE,
    HF_CONFIG_FILENAME,
    HF_SHARD_PATTERN,
    HF_WEIGHTS_FILENAME,
    HF_WEIGHTS_INDEX_FILENAME,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
)
from dynquant.errors import DynQuantError
from dynquant.integration.serving_common.schema import (
    ModuleQuantSpec,
    QuantizationConfigSchema,
)
from dynquant.quant.device import quantize_tensor, resolve_compute_device
from dynquant.quant.grid import CLIP_CANDIDATES

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = ["DEFAULT_SHARD_BYTES", "ExportReport", "export_packed_checkpoint"]

DEFAULT_SHARD_BYTES: int = 4 * 1024**3
"""Target bytes per safetensors shard.

Matches the transformers default, which is what every downstream tool's memory
expectations were built around -- including vLLM's loader, which mmaps one shard
at a time.
"""


@dataclass(slots=True)
class ExportReport:
    """What was written, in the terms someone would check it against."""

    output_dir: Path
    quantized_modules: int
    tied: dict[str, str]
    skipped_modules: tuple[str, ...]
    packed_bytes: int
    dense_bytes: int
    quantized_elements: int
    files: tuple[str, ...]
    layers: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return self.packed_bytes + self.dense_bytes

    @property
    def average_bits(self) -> float:
        """Bits per *quantized* weight, scales and offsets amortised in."""
        if self.quantized_elements == 0:
            return 0.0
        return self.packed_bytes * 8.0 / self.quantized_elements

    def summary(self) -> str:
        tied = f", {len(self.tied)} tied module(s) sharing a table" if self.tied else ""
        return (
            f"{self.quantized_modules} modules packed at {self.average_bits:.4f} "
            f"average bits{tied}: {self.total_bytes / 2**30:.3f} GiB on disk "
            f"({self.packed_bytes / 2**30:.3f} quantized + "
            f"{self.dense_bytes / 2**30:.3f} left dense) across "
            f"{len(self.files)} file(s)"
        )


def export_packed_checkpoint(
    model: nn.Module,
    bits: Mapping[str, int],
    *,
    output_dir: str | Path,
    group_size: int = DEFAULT_GROUP_SIZE,
    symmetric: bool = False,
    candidates: Sequence[float] = CLIP_CANDIDATES,
    compute_device: str | torch.device | None = "auto",
    max_shard_bytes: int = DEFAULT_SHARD_BYTES,
    provenance: Mapping[str, Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> ExportReport:
    """Quantize, pack, and write ``model`` as a loadable checkpoint directory.

    Args:
        model: A loaded transformers model. Not modified -- weights are read, and
            the packed results go straight to the write buffer.
        bits: Module name to bit width, as produced by the allocator or read from
            a saved bit map. Names address modules, not parameters:
            ``model.layers.0.mlp.up_proj``, not ``...up_proj.weight``.
        compute_device: Where the quantization arithmetic runs, independent of
            where the model sits. ``"auto"`` takes the accelerator, which is
            roughly an order of magnitude faster and numerically identical --
            the clipping search is a reduction, not an iteration.
        provenance: Free-form dict recorded in the manifest (stats file,
            allocator, target). Nothing reads it back; it is there so a number can
            be traced to the run that produced it.

    Returns:
        An :class:`ExportReport`. Its byte counts are measured from the tensors
        that were written, not predicted from the bit map.

    Raises:
        DynQuantError: If a name in ``bits`` is not a Linear or Embedding module
            of ``model``, or if the model has no ``config`` to write.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = resolve_compute_device(compute_device)
    aliases = _tied_aliases(model)

    tensors: dict[str, torch.Tensor] = {}
    specs: dict[str, ModuleQuantSpec] = {}
    layers: dict[str, dict[str, Any]] = {}
    tied: dict[str, str] = {}
    packed_bytes = 0
    quantized_elements = 0
    consumed: set[str] = set()

    targets = sorted(bits.items())
    for index, (name, width) in enumerate(targets):
        module = _resolve(model, name)
        weight = module.weight

        with torch.no_grad():
            quantized, search = quantize_tensor(
                weight.detach(),
                bits=width,
                group_size=group_size,
                symmetric=symmetric,
                candidates=candidates,
                compute_dtype=_storage_dtype(weight),
                device=device,
            )
        # Off the accelerator before it lands in the write buffer. The point of
        # encoding on the GPU is the arithmetic; holding every packed tensor in
        # VRAM until the end would make exporting a 7B need as much VRAM as
        # loading it.
        tensors.update({key: value.cpu() for key, value in quantized.state_dict(name).items()})

        # num_rows, not `weight.shape[0]`: for a module the packer treats as more
        # than a matrix the row count is the packer's, and the loader compares
        # against packed rows.
        specs[name] = ModuleQuantSpec(
            bits=width,
            group_size=group_size,
            symmetric=symmetric,
            out_features=quantized.num_rows,
        )
        layers[name] = {
            **quantized.metadata(),
            "clipped_fraction": round(search.clipped_fraction, 6),
            "clip_improvement": round(search.improvement, 6),
        }
        packed_bytes += quantized.nbytes
        quantized_elements += quantized.num_dense_elements

        # Every module sharing this weight's storage is now represented on disk,
        # and writing the table again would cost 27% of a tied 2B model for a
        # tensor the loader discards -- vLLM re-ties after loading, and
        # safetensors refuses two keys backed by one storage anyway.
        for alias in aliases.get(weight.data_ptr(), ()):
            consumed.add(f"{alias}.weight")
            if alias != name:
                tied[alias] = name

        del quantized, search
        if progress is not None:
            progress(index + 1, len(targets))

    # Everything the map did not name is written through untouched. Norms, biases
    # and rotary caches are the intended cases; anything else surviving here is a
    # module the allocator declined, and it stays correct rather than silently
    # disappearing.
    dense_bytes = 0
    for key, value in _unique_state_dict(model).items():
        if key in consumed:
            continue
        tensor = value.detach().cpu()
        tensors[key] = tensor
        dense_bytes += tensor.numel() * tensor.element_size()

    files = _write_shards(out, tensors, max_shard_bytes=max_shard_bytes)
    del tensors

    schema = QuantizationConfigSchema(
        modules=specs,
        group_size=group_size,
        symmetric=symmetric,
        lm_head_quantized=any(name.split(".")[-1] == "lm_head" for name in specs),
        version=_package_version(),
    )
    _write_config(out, model, schema)

    report = ExportReport(
        output_dir=out,
        quantized_modules=len(specs),
        tied=tied,
        skipped_modules=tuple(
            name
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear | nn.Embedding)
            and name not in bits
            and name not in tied
        ),
        packed_bytes=packed_bytes,
        dense_bytes=dense_bytes,
        quantized_elements=quantized_elements,
        files=files,
        layers=layers,
    )
    _write_manifest(out, schema=schema, report=report, provenance=provenance)
    return report


# --------------------------------------------------------------------------
# Pieces
# --------------------------------------------------------------------------


def _resolve(model: nn.Module, name: str) -> nn.Linear | nn.Embedding:
    try:
        module = model.get_submodule(name)
    except AttributeError as exc:
        from dynquant.quant.quantizer import resolves_to_weight

        if resolves_to_weight(model, name):
            raise DynQuantError(
                f"{name!r} is a tensor, not a module -- a batched expert bank. The packed "
                f"format stores one entry per module and there is no module here, so this "
                f"map cannot be exported yet. `dynquant quantize --map` writes the same "
                f"widths as encoded values in a loadable checkpoint."
            ) from exc
        raise DynQuantError(
            f"the bit map names {name!r}, which is not a module of this model. A map built "
            f"for a different checkpoint does not transfer."
        ) from exc
    if not isinstance(module, nn.Linear | nn.Embedding):
        raise DynQuantError(
            f"{name!r} is a {type(module).__name__}; the packed format covers Linear and "
            f"Embedding only. Batched MoE expert banks need the grouped path."
        )
    return module


def _tied_aliases(model: nn.Module) -> dict[int, list[str]]:
    """Storage address -> every quantizable module whose weight lives there.

    Collected before anything is encoded. A bit map lists one representative per
    tied group, so this is how the ``lm_head`` of a tied model is recognised as
    already written rather than emitted a second time in fp16.
    """
    groups: dict[int, list[str]] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear | nn.Embedding):
            groups.setdefault(module.weight.data_ptr(), []).append(name)
    return groups


def _storage_dtype(weight: torch.Tensor) -> torch.dtype:
    """The dtype scales are stored in: the weight's own, unless it is fp32."""
    return weight.dtype if weight.dtype in (torch.float16, torch.bfloat16) else torch.float16


def _unique_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """``state_dict`` with shared-storage duplicates removed.

    safetensors refuses to serialise two keys backed by one storage, and any model
    with ``tie_word_embeddings`` reports exactly that. The first key wins, which
    for every architecture in the matrix is ``model.embed_tokens.weight`` -- the
    one both transformers and vLLM re-tie from.
    """
    state = model.state_dict()
    first_at: dict[tuple[int, int], str] = {}
    out: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        # `storage_offset()` is `int | SymInt`, and a SymInt is not hashable into
        # the same bucket as the int it will become. Nothing reaches here on a
        # traced model, so int() is the fact rather than a coercion.
        identity = (value.untyped_storage().data_ptr(), int(value.storage_offset()))
        earlier = first_at.get(identity)
        if earlier is not None and value.shape == state[earlier].shape:
            continue
        first_at[identity] = key
        out[key] = value
    return out


def _write_shards(
    out: Path, tensors: Mapping[str, torch.Tensor], *, max_shard_bytes: int
) -> tuple[str, ...]:
    """Write safetensors shards plus an index, HF-style.

    A checkpoint that fits in one file gets ``model.safetensors`` and **no**
    index, which is what transformers does and what vLLM's loader expects: it
    globs ``*.safetensors`` and only consults the index when the glob returns
    more than one file.
    """
    from safetensors.torch import save_file

    groups: list[dict[str, torch.Tensor]] = [{}]
    sizes = [0]
    for key in sorted(tensors):
        value = tensors[key].contiguous()
        nbytes = value.numel() * value.element_size()
        if sizes[-1] and sizes[-1] + nbytes > max_shard_bytes:
            groups.append({})
            sizes.append(0)
        groups[-1][key] = value
        sizes[-1] += nbytes

    if len(groups) == 1:
        save_file(groups[0], out / HF_WEIGHTS_FILENAME, metadata={"format": "pt"})
        return (HF_WEIGHTS_FILENAME,)

    names = [HF_SHARD_PATTERN.format(index=i + 1, total=len(groups)) for i in range(len(groups))]
    weight_map: dict[str, str] = {}
    for name, group in zip(names, groups, strict=True):
        save_file(group, out / name, metadata={"format": "pt"})
        weight_map.update(dict.fromkeys(group, name))

    index = {
        "metadata": {"total_size": sum(sizes)},
        "weight_map": {key: weight_map[key] for key in sorted(weight_map)},
    }
    (out / HF_WEIGHTS_INDEX_FILENAME).write_text(json.dumps(index, indent=2), encoding="utf-8")
    return (*names, HF_WEIGHTS_INDEX_FILENAME)


def _write_config(out: Path, model: nn.Module, schema: QuantizationConfigSchema) -> None:
    """Write the model's own config with ``quantization_config`` added.

    transformers writes it first and this function edits the result, rather than
    serialising ``config.to_dict()`` directly: nested sub-configs, dtype fields
    and ``transformers_version`` all have version-specific serialisation rules,
    and a config.json that differs from the one ``save_pretrained`` would have
    written is a difference the user has to debug later.
    """
    config = getattr(model, "config", None)
    if config is None:
        raise DynQuantError(
            "the model has no `config`, so there is nothing to write config.json from. "
            "Export expects a transformers model."
        )
    config.save_pretrained(str(out))

    path = out / HF_CONFIG_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["quantization_config"] = schema.to_dict()
    # `architectures` is how every loader picks a model class, and
    # `config.save_pretrained` does not write it -- transformers stamps it from
    # `PreTrainedModel.save_pretrained` instead, which this does not call. A
    # config built in memory rather than read from a checkpoint therefore has
    # none, and the directory would be unloadable by anything.
    payload.setdefault("architectures", [type(model).__name__])
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        # A generation config that fails its own validation is not worth failing an
        # export over -- the model still serves, on defaults.
        with contextlib.suppress(ValueError, OSError):
            generation_config.save_pretrained(str(out))


def _write_manifest(
    out: Path,
    *,
    schema: QuantizationConfigSchema,
    report: ExportReport,
    provenance: Mapping[str, Any] | None,
) -> None:
    """The human-readable record beside the checkpoint.

    Nothing loads from this. ``config.json`` is the contract, because that is the
    only file a loader hands to ``from_config``. The manifest carries the
    per-layer reconstruction error, the tie map and the provenance -- what
    somebody reproducing a number needs and no loader does.
    """
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "dynquant_core": _package_version(),
        "quantization_config": schema.to_dict(),
        "files": list(report.files),
        "accounting": {
            "packed_nbytes": report.packed_bytes,
            "dense_nbytes": report.dense_bytes,
            "total_nbytes": report.total_bytes,
            "quantized_elements": report.quantized_elements,
            "average_bits": round(report.average_bits, 6),
        },
        "tied": dict(sorted(report.tied.items())),
        "provenance": dict(provenance or {}),
        "layers": {name: dict(meta) for name, meta in sorted(report.layers.items())},
    }
    (out / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _package_version() -> str:
    from dynquant._version import __version__

    return str(__version__)
