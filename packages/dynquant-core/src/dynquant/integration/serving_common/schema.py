"""The ``quantization_config`` block, parsed without importing a serving framework.

A DynQuant checkpoint's widths are per module, so ``config.json`` has to carry the
whole map -- there is no ``"bits": 4`` that describes the model. That is the one
real difference from every other entry in a server's quantization registry and it
forces a design choice this module makes explicit.

Why the map is embedded rather than read from a sidecar file
------------------------------------------------------------
``QuantizationConfig.from_config`` receives the parsed ``quantization_config``
dict and nothing else -- no model path, no revision, no filesystem handle. This is
true of both servers: vLLM's signature and SGLang's are the same one, SGLang's
having been forked from it. A sidecar (``dynquant_manifest.json``, which the
exporter also writes) is therefore unreachable from inside the server without
re-deriving the download location, which would mean reimplementing the Hub cache
lookup and getting it wrong for local paths, remote revisions and offline mode. So
the map goes in ``config.json``. For a 2B model that is roughly 15 KB of JSON and
for a 70B model roughly 40 KB; ``config.json`` is read once at startup, and 40 KB is
not a cost worth a fragile second file for. The manifest stays on disk as the
human-readable record and as what ``dynquant inspect`` reads -- it is simply not on
the server's path.

One caller-side difference worth knowing here, though this module is unaffected by
it: SGLang injects ``packed_modules_mapping`` *into* that same dict before calling
``from_config`` (``model_loader/weight_utils.py:278``), whereas vLLM sets it on the
instance. :meth:`QuantizationConfigSchema.from_dict` ignores unknown keys, so the
injected entry passes through harmlessly; lifting it onto the config object is the
SGLang plugin's job, not this module's.

Nothing here imports a serving framework or torch, so the round-trip is testable
anywhere.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from dynquant.constants import (
    BIT_OPTIONS,
    DEFAULT_GROUP_SIZE,
    HF_CONFIG_FILENAME,
    HF_QUANT_METHOD,
)
from dynquant.errors import DynQuantError, FormatVersionError, PackingError

__all__ = [
    "CHECKPOINT_FORMAT",
    "SCHEMA_VERSION",
    "ModuleQuantSpec",
    "QuantizationConfigSchema",
    "expand_fused_prefix",
]

CHECKPOINT_FORMAT = "dynquant-packed"
"""``quantization_config.checkpoint_format``. Distinguishes a packed checkpoint
a server can load from the values-quantized directory ``dynquant quantize``
writes, which has the same ``quant_method`` but stores dense compute-dtype
weights and must not be loaded through this path."""

SCHEMA_VERSION = 1
"""Version of the ``quantization_config`` block itself, bumped only when a reader
of an older block would misread a newer one. Independent of the package version,
which is recorded separately as provenance."""


@dataclass(frozen=True, slots=True)
class ModuleQuantSpec:
    """The width one checkpoint module was quantized at.

    Attributes:
        out_features: Rows of this module's weight as stored, i.e. before any
            tensor-parallel division. Optional because it is not needed to
            *decode* anything -- the packed tensor carries its own shape -- but a
            serving framework needs it to line checkpoint modules up against a
            fused layer's output partitions when the two do not correspond one to
            one. vLLM's gated delta net is the case that forces it: one
            ``in_proj_qkv`` tensor backs three of ``in_proj_qkvz``'s four
            partitions and ``in_proj_z`` backs the fourth, and nothing else in
            what a server hands a quantization config distinguishes that from a
            4096+4096 split. ``None`` for checkpoints written before this field
            existed, which is why every consumer treats it as optional rather
            than assuming it.
    """

    bits: int
    group_size: int
    symmetric: bool
    out_features: int | None = None

    def __post_init__(self) -> None:
        if self.out_features is not None and self.out_features <= 0:
            raise DynQuantError(f"out_features must be positive, got {self.out_features}")
        if self.bits not in BIT_OPTIONS:
            # PackingError rather than FormatVersionError: the block may declare a
            # schema this build understands perfectly and still name a width no
            # kernel here can decode, and the two are worth telling apart. The
            # schema check upstream is what catches a genuinely newer format.
            raise PackingError(
                f"bits={self.bits} is not one of {list(BIT_OPTIONS)}; this checkpoint "
                f"was written by a different version of DynQuant"
            )

    def to_dict(self, *, default_group_size: int, default_symmetric: bool) -> dict[str, Any]:
        """Emit only what differs from the file-level defaults.

        Per-module keys are the common case for ``bits`` and the rare case for
        everything else, and a map of 200 modules that repeats
        ``"group_size": 128, "symmetric": false`` on every line is three times the
        size for no information.
        """
        out: dict[str, Any] = {"bits": self.bits}
        if self.group_size != default_group_size:
            out["group_size"] = self.group_size
        if self.symmetric != default_symmetric:
            out["symmetric"] = self.symmetric
        if self.out_features is not None:
            out["out_features"] = self.out_features
        return out


@dataclass(frozen=True, slots=True)
class QuantizationConfigSchema:
    """``config.json -> quantization_config`` for a packed DynQuant checkpoint."""

    modules: Mapping[str, ModuleQuantSpec]
    group_size: int = DEFAULT_GROUP_SIZE
    symmetric: bool = False
    lm_head_quantized: bool = False
    modules_to_not_convert: tuple[str, ...] = ()
    version: str = ""
    """The ``dynquant-core`` version that wrote the checkpoint. Provenance only:
    never branched on, because a version string that gates behaviour turns every
    future release into a compatibility matrix. :data:`SCHEMA_VERSION` is the
    thing readers are allowed to test."""

    # -- serialisation -----------------------------------------------------

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> QuantizationConfigSchema:
        method = config.get("quant_method")
        if method is not None and method != HF_QUANT_METHOD:
            raise DynQuantError(
                f"quantization_config.quant_method is {method!r}, not {HF_QUANT_METHOD!r}"
            )

        schema_version = int(config.get("schema_version", SCHEMA_VERSION))
        if schema_version > SCHEMA_VERSION:
            raise FormatVersionError(
                path=f"{HF_CONFIG_FILENAME} -> quantization_config",
                kind="quantization_config block",
                found=schema_version,
                supported=f"up to v{SCHEMA_VERSION}",
            )

        fmt = config.get("checkpoint_format", CHECKPOINT_FORMAT)
        if fmt != CHECKPOINT_FORMAT:
            raise DynQuantError(
                f"checkpoint_format is {fmt!r}, not {CHECKPOINT_FORMAT!r}. Only a packed "
                f"DynQuant checkpoint can be served; a directory written by "
                f"`dynquant quantize` stores dense compute-dtype weights and has to be "
                f"re-exported with `dynquant export`."
            )

        group_size = int(config.get("group_size", DEFAULT_GROUP_SIZE))
        symmetric = bool(config.get("symmetric", False))

        raw_modules = config.get("modules")
        if not isinstance(raw_modules, Mapping) or not raw_modules:
            raise DynQuantError(
                "quantization_config.modules is missing or empty. DynQuant allocates a "
                "width per module, so the map is the configuration -- there is no "
                "single bit width to fall back on."
            )

        modules = {
            name: _spec_from_entry(name, entry, group_size, symmetric)
            for name, entry in raw_modules.items()
        }
        return cls(
            modules=modules,
            group_size=group_size,
            symmetric=symmetric,
            lm_head_quantized=bool(config.get("lm_head_quantized", False)),
            modules_to_not_convert=tuple(config.get("modules_to_not_convert", ())),
            version=str(config.get("version", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "quant_method": HF_QUANT_METHOD,
            "checkpoint_format": CHECKPOINT_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "version": self.version,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            "lm_head_quantized": self.lm_head_quantized,
            "modules_to_not_convert": list(self.modules_to_not_convert),
            "modules": {
                name: spec.to_dict(
                    default_group_size=self.group_size, default_symmetric=self.symmetric
                )
                for name, spec in sorted(self.modules.items())
            },
        }

    # -- queries -----------------------------------------------------------

    def get(self, name: str) -> ModuleQuantSpec | None:
        """The spec for a checkpoint module, or ``None`` if it stayed dense."""
        if name in self.modules_to_not_convert:
            return None
        return self.modules.get(name)

    def remap(self, apply_list: Callable[[list[str]], list[str]]) -> QuantizationConfigSchema:
        """Rewrite module names through vLLM's HF-to-vLLM name mapper.

        Some vLLM model implementations rename modules relative to the HF
        checkpoint (``transformer.h`` to ``model.layers``, and so on) and hand the
        quantization config a mapper so per-module rules keep matching. Returns a
        new schema rather than mutating, because ``apply_vllm_mapper`` may be
        called on a config that other layers already hold a reference to and a
        half-remapped map is worse than either end state.
        """
        names = list(self.modules)
        mapped = apply_list(names)
        if len(mapped) != len(names):
            raise DynQuantError(
                f"vLLM's name mapper returned {len(mapped)} names for {len(names)} modules"
            )
        return QuantizationConfigSchema(
            modules={new: self.modules[old] for old, new in zip(names, mapped, strict=True)},
            group_size=self.group_size,
            symmetric=self.symmetric,
            lm_head_quantized=self.lm_head_quantized,
            modules_to_not_convert=tuple(apply_list(list(self.modules_to_not_convert))),
            version=self.version,
        )

    def resolve_shards(
        self, prefix: str, packed_modules_mapping: Mapping[str, Sequence[str]]
    ) -> list[tuple[str, ModuleQuantSpec]] | None:
        """Which checkpoint modules back one server-side layer, and at what widths.

        Both servers fuse ``q``/``k``/``v`` into one ``qkv_proj`` layer whose prefix
        names no tensor in the checkpoint, so the prefix is expanded through
        ``packed_modules_mapping``. How that mapping reaches the config differs --
        vLLM sets it on the instance, SGLang injects it into the dict handed to
        ``from_config`` and leaves the lifting to the plugin -- which is why it is a
        parameter here rather than something this module reads for itself.

        Returns ``None`` when the layer is not quantized at all, so the caller can
        hand back an unquantized method. Raises when *some* of a fused layer's
        shards are quantized and others are not: the fused parameter is one
        buffer, so a mixed layer is not representable, and silently promoting the
        dense shard to a guessed width would change the model without saying so.
        This mirrors what GPTQ does in the same situation.
        """
        names = expand_fused_prefix(prefix, packed_modules_mapping)
        found = [(name, self.get(name)) for name in names]
        present = [(name, spec) for name, spec in found if spec is not None]
        if not present:
            return None
        if len(present) != len(found):
            missing = [name for name, spec in found if spec is None]
            raise DynQuantError(
                f"{prefix} fuses {names}, but {missing} are not in this checkpoint's "
                f"quantization map while the others are. All shards of a fused layer "
                f"share one packed buffer, so they must all be quantized or none. "
                f"Re-export with these modules quantized, or add every shard to "
                f"modules_to_not_convert."
            )
        return present


def expand_fused_prefix(
    prefix: str, packed_modules_mapping: Mapping[str, Sequence[str]]
) -> list[str]:
    """A serving layer's prefix to the checkpoint module names it was fused from.

    ``model.layers.0.self_attn.qkv_proj`` with ``{"qkv_proj": ["q_proj", "k_proj",
    "v_proj"]}`` becomes the three ``model.layers.0.self_attn.*_proj`` names. A
    prefix with no entry, or a self-referential entry like
    ``{"qkv_proj": ["qkv_proj"]}`` (used for models whose checkpoint is already
    fused on disk), yields the prefix itself.
    """
    if not prefix:
        return [prefix]
    leaf = prefix.rsplit(".", 1)[-1]
    subs = packed_modules_mapping.get(leaf)
    if not subs:
        return [prefix]
    stem = prefix[: len(prefix) - len(leaf)]
    return [prefix if sub == leaf else stem + sub for sub in subs]


def _spec_from_entry(
    name: str, entry: Any, default_group_size: int, default_symmetric: bool
) -> ModuleQuantSpec:
    """One ``modules`` value, accepting the bare-int shorthand.

    ``{"model.layers.0.mlp.up_proj": 3}`` is accepted alongside the dict form
    because a hand-written or externally generated map is far more likely to be
    written that way, and rejecting it buys nothing.
    """
    if isinstance(entry, int) and not isinstance(entry, bool):
        return ModuleQuantSpec(entry, default_group_size, default_symmetric)
    if not isinstance(entry, Mapping):
        raise DynQuantError(
            f"quantization_config.modules[{name!r}] should be an int or an object with "
            f"a 'bits' key, got {type(entry).__name__}"
        )
    if "bits" not in entry:
        raise DynQuantError(f"quantization_config.modules[{name!r}] has no 'bits'")
    out_features = entry.get("out_features")
    return ModuleQuantSpec(
        bits=int(entry["bits"]),
        group_size=int(entry.get("group_size", default_group_size)),
        symmetric=bool(entry.get("symmetric", default_symmetric)),
        out_features=None if out_features is None else int(out_features),
    )
