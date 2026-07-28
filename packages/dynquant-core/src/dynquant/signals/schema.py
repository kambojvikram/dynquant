"""The on-disk contract for collected signals, and lossless migration from v1.

A stats file is the only thing that travels between the training run and the
quantization run, often across machines and weeks. It therefore has to be
self-describing, wrapper-independent, and honest about what it does *not* know.

What v2 changes
---------------
=========================  ====================================================
v1 (the supplement)         v2
=========================  ====================================================
keys as-collected, so       keys canonicalised at write time, so the file is
``base_model.model.model``  independent of PEFT/DDP/FSDP/compile wrapping
``...base_layer``           (:func:`dynquant.graph.naming.canonical_name`)
no provenance               model id, step count, estimator mode, world size,
                            dynquant version -- so a mismatched pairing of stats
                            and model is detectable instead of silent
``count == 0`` and          ``count < 2`` recognised as *no variance observed*
``count == 1`` both look    and reported as missing signal rather than as
like "variance 0"           "perfectly stable"
no parameter counts         optional ``param_count`` so allocation can run
                            without loading the model
no MoE awareness            ``routing_hits``, so a rarely-routed expert is not
                            mistaken for a stable one
=========================  ====================================================

The third row is the subtle one. ``grad_norm_var`` is a *sample* variance,
``m2 / (count - 1)``, and the supplement's ``_WelfordState.variance()`` returns
``0.0`` when ``count < 2``. So a layer that was hooked once and a layer that never
moved both emit exactly ``0.0`` -- and since the scorer maps low variance to low
importance, an unobserved layer is indistinguishable from a maximally
compressible one. ``qwen3_14b``'s shipped stats contain precisely this case:
``lm_head`` has ``grad_norm_count: 0``. It survives only because the LM head has
an 8-bit floor; anything without a floor would have been crushed on the strength
of a measurement that never happened.

Because ``grad_norm_var`` is a sample variance with a known count, the second
moment ``m2 = var * (count - 1)`` is recoverable exactly -- which is what makes
:meth:`LayerStats.merged_with` able to combine two accumulators with no loss
(Chan's parallel variance). That single fact supplies both the DDP reduction and
the "collapse LoRA into base" step that the shipped stats files claim
(``"collapsed_lora_into_base": true``) but whose script was never included.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from dynquant._version import STATS_SCHEMA_VERSION, __version__
from dynquant.constants import LEGACY_STATS_SCHEMAS, STATS_SCHEMA, STATS_SCHEMA_V1
from dynquant.errors import FormatVersionError, StatsCoverageError
from dynquant.graph.naming import canonical_name

__all__ = [
    "CoverageReport",
    "LayerStats",
    "Provenance",
    "StatsFile",
    "load_stats",
    "save_stats",
]

MIN_OBSERVATIONS_FOR_VARIANCE = 2
"""Below this, ``grad_norm_var`` carries no information -- it is 0 by definition."""


# --------------------------------------------------------------------------
# Per-layer record
# --------------------------------------------------------------------------


@dataclass(slots=True)
class LayerStats:
    """Signals collected for one module.

    Field names for the four v1 quantities are kept verbatim so that a v1 file
    round-trips through v2 and back without renaming, and so the ``paper-3.15``
    compat path can feed the vendored scorer untouched.
    """

    name: str
    activation_rms_ema: float = 0.0
    """EMA of mean per-channel activation RMS. The saliency signal."""

    grad_norm_count: int = 0
    """Welford observation count. ``< 2`` means no variance was observed."""

    grad_norm_mean: float = 0.0
    grad_norm_var: float = 0.0
    """**Sample** variance of the gradient L2 norm, ``m2 / (count - 1)``."""

    coherence_ema: float | None = None
    """EMA of cosine similarity between consecutive per-channel gradient RMS
    vectors. Present in the research code and in its score product, but absent
    from the paper's Eq. (4) -- the published formula is a two-way product while
    the shipped code computes a three-way one. Retained because reproducing the
    paper's numbers requires it; the default v2 scorer treats it as optional."""

    param_count: int | None = None
    """Elements in the weight. Lets allocation run without loading the model."""

    role: str | None = None
    """:class:`~dynquant.graph.roles.ModuleRole` value, if known at collection."""

    routing_hits: int | None = None
    """MoE only: times this expert was selected. A rarely-routed expert
    accumulates few gradient observations, which looks like stability but is
    really absence of evidence -- so the scorer needs the denominator."""

    forward_calls: int | None = None
    """Times the module's forward ran during collection.

    ``0`` means the module was **never exercised** by the training data, which is
    categorically different from "exercised and found unimportant" and must not be
    scored as though it were. A text-only fine-tune of a VLM never runs the vision
    tower -- on Qwen3.5-2B that is a 24-layer ViT, about 15% of the parameters,
    every one of which reports zero saliency and zero plasticity for the entirely
    innocent reason that no image was ever passed. An allocator reading those
    zeros as evidence sends the whole tower to the 2-bit floor and destroys any
    later multimodal use of the checkpoint.

    ``activation_rms_ema == 0`` cannot substitute for this: the collection path
    computes ``sqrt(mean(x^2) + eps)``, so a module that fires on genuinely
    all-zero activations still reports ``sqrt(eps)``, and one that never fires
    reports a structural zero. Only an explicit call count separates the two.

    ``None`` means the producing collector did not record it (every v1 file), in
    which case absence of signal is genuinely ambiguous and consumers should say
    so rather than guess."""

    grad_estimator: str | None = None
    """Which estimator produced the gradient stats: ``outer_exact``,
    ``lowrank`` or ``param``. Different modes are not comparable, so mixing them
    within one file is a bug worth being able to detect."""

    # -- derived -----------------------------------------------------------

    @property
    def has_gradient_signal(self) -> bool:
        """True only if a variance was actually observable.

        Deliberately stricter than ``count > 0``. See the module docstring.
        """
        return self.grad_norm_count >= MIN_OBSERVATIONS_FOR_VARIANCE

    @property
    def has_activation_signal(self) -> bool:
        return self.activation_rms_ema > 0.0

    @property
    def was_exercised(self) -> bool | None:
        """Did the training data ever route through this module?

        ``None`` when unknowable -- the file predates :attr:`forward_calls`. Three
        states, not two, because "we did not measure" and "we measured nothing"
        warrant different handling and collapsing them to ``False`` is how a
        vision tower ends up at 2 bits.
        """
        if self.forward_calls is None:
            return None
        return self.forward_calls > 0

    @property
    def m2(self) -> float:
        """Welford's second moment, recovered from the sample variance."""
        if self.grad_norm_count < MIN_OBSERVATIONS_FOR_VARIANCE:
            return 0.0
        return self.grad_norm_var * (self.grad_norm_count - 1)

    @property
    def plasticity_raw(self) -> float:
        """``log(1 + Var_t||grad||)`` -- the paper's Eq. (2) inner term.

        The log compresses a distribution that spans several orders of magnitude
        before percentile-ranking, which keeps a handful of extreme layers from
        occupying the entire top of the ranking.
        """
        return math.log1p(max(0.0, self.grad_norm_var))

    @property
    def saliency_raw(self) -> float:
        return max(0.0, self.activation_rms_ema)

    # -- combination -------------------------------------------------------

    def merged_with(self, other: LayerStats) -> LayerStats:
        """Exactly combine two accumulators over disjoint observation sets.

        Uses Chan's parallel variance formula, which is numerically stable and
        gives bit-comparable results to a single-stream Welford pass. Used for:

        * **DDP/FSDP reduction** -- each rank saw different micro-batches.
        * **Collapsing LoRA into base** -- ``lora_A`` and ``lora_B`` gradient
          observations both attribute to the base weight they will be merged
          into, and canonicalisation maps them to the same key.

        EMAs (activation, coherence) cannot be combined exactly -- an EMA is
        path-dependent, so there is no algebra that recovers the joint history
        from two summaries. They are combined by observation-count-weighted mean,
        which is the right first-order estimate, and the approximation is
        documented rather than hidden.
        """
        n_a, n_b = self.grad_norm_count, other.grad_norm_count
        n = n_a + n_b
        if n == 0:
            mean = 0.0
            var = 0.0
        else:
            delta = other.grad_norm_mean - self.grad_norm_mean
            mean = self.grad_norm_mean + delta * (n_b / n)
            m2 = self.m2 + other.m2 + delta * delta * n_a * n_b / n
            var = m2 / (n - 1) if n >= MIN_OBSERVATIONS_FOR_VARIANCE else 0.0

        return LayerStats(
            name=self.name,
            activation_rms_ema=_blend(
                self.activation_rms_ema,
                other.activation_rms_ema,
                self.has_activation_signal,
                other.has_activation_signal,
                n_a,
                n_b,
            ),
            grad_norm_count=n,
            grad_norm_mean=mean,
            grad_norm_var=var,
            coherence_ema=_blend_optional(self.coherence_ema, other.coherence_ema, n_a, n_b),
            param_count=self.param_count if self.param_count is not None else other.param_count,
            role=self.role or other.role,
            routing_hits=_add_optional(self.routing_hits, other.routing_hits),
            forward_calls=_add_optional(self.forward_calls, other.forward_calls),
            grad_estimator=self.grad_estimator or other.grad_estimator,
        )

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "activation_rms_ema": self.activation_rms_ema,
            "grad_norm_count": self.grad_norm_count,
            "grad_norm_mean": self.grad_norm_mean,
            "grad_norm_var": self.grad_norm_var,
        }
        for key, value in (
            ("coherence_ema", self.coherence_ema),
            ("param_count", self.param_count),
            ("role", self.role),
            ("routing_hits", self.routing_hits),
            ("forward_calls", self.forward_calls),
            ("grad_estimator", self.grad_estimator),
        ):
            if value is not None:
                out[key] = value
        return out

    @classmethod
    def from_dict(cls, name: str, raw: Mapping[str, Any]) -> LayerStats:
        return cls(
            name=name,
            activation_rms_ema=float(raw.get("activation_rms_ema", 0.0) or 0.0),
            grad_norm_count=int(raw.get("grad_norm_count", 0) or 0),
            grad_norm_mean=float(raw.get("grad_norm_mean", 0.0) or 0.0),
            grad_norm_var=float(raw.get("grad_norm_var", 0.0) or 0.0),
            coherence_ema=_opt_float(raw.get("coherence_ema")),
            param_count=_opt_int(raw.get("param_count")),
            role=raw.get("role"),
            routing_hits=_opt_int(raw.get("routing_hits")),
            forward_calls=_opt_int(raw.get("forward_calls")),
            grad_estimator=raw.get("grad_estimator"),
        )


def _blend(a: float, b: float, a_valid: bool, b_valid: bool, n_a: int, n_b: int) -> float:
    if not b_valid:
        return a
    if not a_valid:
        return b
    total = n_a + n_b
    if total == 0:
        return 0.5 * (a + b)
    return (a * n_a + b * n_b) / total


def _blend_optional(a: float | None, b: float | None, n_a: int, n_b: int) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    total = n_a + n_b
    if total == 0:
        return 0.5 * (a + b)
    return (a * n_a + b * n_b) / total


def _add_optional(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _opt_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _opt_int(value: Any) -> int | None:
    return None if value is None else int(value)


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Provenance:
    """Where a stats file came from.

    Exists so that pairing a stats file with the wrong model is a detectable
    error rather than a quiet 40%-coverage allocation.
    """

    dynquant_version: str = __version__
    model_name_or_path: str | None = None
    model_type: str | None = None
    num_optimizer_steps: int | None = None
    grad_estimator: str | None = None
    world_size: int | None = None
    created_at_unix: float | None = None
    canonical_names: bool = True
    migrated_from: str | None = None
    """Schema tag of the file this was migrated from, if any."""

    notes: dict[str, Any] = field(default_factory=dict)
    """Free-form extras. v1 top-level keys that have no v2 home land here so
    migration is genuinely lossless."""

    def to_dict(self) -> dict[str, Any]:
        out = {
            "dynquant_version": self.dynquant_version,
            "canonical_names": self.canonical_names,
        }
        for key, value in (
            ("model_name_or_path", self.model_name_or_path),
            ("model_type", self.model_type),
            ("num_optimizer_steps", self.num_optimizer_steps),
            ("grad_estimator", self.grad_estimator),
            ("world_size", self.world_size),
            ("created_at_unix", self.created_at_unix),
            ("migrated_from", self.migrated_from),
        ):
            if value is not None:
                out[key] = value
        if self.notes:
            out["notes"] = self.notes
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Provenance:
        return cls(
            dynquant_version=str(raw.get("dynquant_version", "unknown")),
            model_name_or_path=raw.get("model_name_or_path"),
            model_type=raw.get("model_type"),
            num_optimizer_steps=_opt_int(raw.get("num_optimizer_steps")),
            grad_estimator=raw.get("grad_estimator"),
            world_size=_opt_int(raw.get("world_size")),
            created_at_unix=_opt_float(raw.get("created_at_unix")),
            canonical_names=bool(raw.get("canonical_names", True)),
            migrated_from=raw.get("migrated_from"),
            notes=dict(raw.get("notes", {})),
        )


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


@dataclass(slots=True)
class CoverageReport:
    """How well a stats file covers a model's quantizable modules."""

    matched: tuple[str, ...]
    missing_signal: tuple[str, ...]
    """In the model, absent from the stats -- would be allocated blind."""

    partial_signal: tuple[str, ...]
    """Present but with no usable gradient variance (``count < 2``)."""

    unmatched_stats: tuple[str, ...]
    """In the stats, absent from the model -- usually a wrong-model pairing."""

    total_model_modules: int

    unexercised: tuple[str, ...] = ()
    """Matched, but the training data never routed through them
    (:attr:`LayerStats.forward_calls` is 0).

    Separated from :attr:`partial_signal` because the remedy differs. Partial
    signal means "train longer". Unexercised means "this data cannot exercise this
    module at all" -- a text-only run against a VLM, or an expert no prompt in the
    corpus routes to. Allocating from those zeros as though they were measurements
    is the failure this field exists to make visible."""

    @property
    def fraction(self) -> float:
        if self.total_model_modules == 0:
            return 1.0
        return len(self.matched) / self.total_model_modules

    @property
    def usable_fraction(self) -> float:
        """Matched *and* carrying a real gradient variance."""
        if self.total_model_modules == 0:
            return 1.0
        usable = len(self.matched) - len(self.partial_signal)
        return max(0, usable) / self.total_model_modules

    def raise_if_insufficient(self, threshold: float) -> None:
        if self.fraction >= threshold:
            return
        raise StatsCoverageError(
            covered=len(self.matched),
            total=self.total_model_modules,
            threshold=threshold,
            examples=list(self.missing_signal),
        )

    def summary(self) -> str:
        parts = [
            f"{len(self.matched)}/{self.total_model_modules} modules matched ({self.fraction:.1%})",
            f"{len(self.partial_signal)} with no gradient variance",
            f"{len(self.missing_signal)} missing",
            f"{len(self.unmatched_stats)} stats entries unmatched",
        ]
        if self.unexercised:
            parts.append(f"{len(self.unexercised)} never exercised by the training data")
        return "; ".join(parts)


# --------------------------------------------------------------------------
# The file
# --------------------------------------------------------------------------


@dataclass(slots=True)
class StatsFile:
    """A collection of :class:`LayerStats` plus the hyperparameters that made them."""

    layers: dict[str, LayerStats] = field(default_factory=dict)
    activation_ema_beta: float = 0.99
    coherence_ema_beta: float | None = 0.95
    eps: float = 1e-12
    provenance: Provenance = field(default_factory=Provenance)

    def __post_init__(self) -> None:
        """Sorted key order is canonical -- in memory as well as on disk.

        :meth:`to_dict` sorts so that two runs of the same job produce files that
        diff cleanly. Sorting *only* there would make ``names`` mean "module tree
        order" before a save and "alphabetical" after one, so anything that zips
        against it, slices it, or reports "the first N layers" would quietly change
        behaviour across a round-trip. Canonicalising at construction also makes a
        merge order-independent: two ranks whose modules arrive in different orders
        produce identical files.
        """
        names = list(self.layers)
        if names != sorted(names):
            self.layers = {name: self.layers[name] for name in sorted(names)}

    # -- mapping-ish behaviour --------------------------------------------

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self) -> Iterator[str]:
        return iter(self.layers)

    def __contains__(self, name: str) -> bool:
        return name in self.layers

    def __getitem__(self, name: str) -> LayerStats:
        return self.layers[name]

    def get(self, name: str) -> LayerStats | None:
        return self.layers.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.layers)

    # -- integrity ---------------------------------------------------------

    def canonicalized(self) -> StatsFile:
        """Return a copy with every key canonical, merging any collisions exactly.

        Idempotent. Two keys collide when they name the same base weight through
        different wrappers -- ``...q_proj`` and ``...q_proj.base_layer``, or the
        ``lora_A``/``lora_B`` pair of one module. Merging rather than
        last-write-wins is what makes this the "collapse LoRA into base" step.
        """
        merged: dict[str, LayerStats] = {}
        for name, stats in self.layers.items():
            key = canonical_name(name)
            record = replace(stats, name=key)
            existing = merged.get(key)
            merged[key] = existing.merged_with(record) if existing else record
        return replace(
            self,
            layers=merged,
            provenance=replace(self.provenance, canonical_names=True),
        )

    def merged_with(self, other: StatsFile) -> StatsFile:
        """Combine two stats files over disjoint observations (DDP reduction)."""
        layers = dict(self.layers)
        for name, stats in other.layers.items():
            existing = layers.get(name)
            layers[name] = existing.merged_with(stats) if existing else stats
        return replace(self, layers=layers)

    def coverage(self, model_module_names: Iterable[str]) -> CoverageReport:
        """Compare against the modules a model actually wants quantized."""
        wanted = [canonical_name(n) for n in model_module_names]
        wanted_set = set(wanted)
        have = set(self.layers)

        matched = tuple(sorted(wanted_set & have))
        return CoverageReport(
            matched=matched,
            missing_signal=tuple(sorted(wanted_set - have)),
            partial_signal=tuple(n for n in matched if not self.layers[n].has_gradient_signal),
            unmatched_stats=tuple(sorted(have - wanted_set)),
            total_model_modules=len(wanted_set),
            unexercised=tuple(n for n in matched if self.layers[n].was_exercised is False),
        )

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STATS_SCHEMA,
            "schema_version": STATS_SCHEMA_VERSION,
            "provenance": self.provenance.to_dict(),
            "hyperparameters": {
                "activation_ema_beta": self.activation_ema_beta,
                "coherence_ema_beta": self.coherence_ema_beta,
                "eps": self.eps,
            },
            # Sorted again rather than trusting __post_init__: `layers` is a plain
            # mutable dict and callers may have inserted into it since construction.
            "layers": {name: self.layers[name].to_dict() for name in sorted(self.layers)},
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], *, path: str = "<dict>") -> StatsFile:
        """Parse either schema version, migrating v1 transparently."""
        schema = raw.get("schema")
        if schema == STATS_SCHEMA:
            return cls._from_v2(raw)
        if schema in LEGACY_STATS_SCHEMAS or (schema is None and "layers" in raw):
            # `schema is None` covers hand-edited files, which carry the same
            # field layout without the tag.
            return cls._from_v1(raw)
        raise FormatVersionError(
            path=path,
            kind="Signal stats file",
            found=schema,
            supported=(STATS_SCHEMA, *LEGACY_STATS_SCHEMAS),
        )

    @classmethod
    def _from_v2(cls, raw: Mapping[str, Any]) -> StatsFile:
        hyper = raw.get("hyperparameters", {})
        return cls(
            layers={
                name: LayerStats.from_dict(name, body)
                for name, body in raw.get("layers", {}).items()
            },
            activation_ema_beta=float(hyper.get("activation_ema_beta", 0.99)),
            coherence_ema_beta=_opt_float(hyper.get("coherence_ema_beta")),
            eps=float(hyper.get("eps", 1e-12)),
            provenance=Provenance.from_dict(raw.get("provenance", {})),
        )

    @classmethod
    def _from_v1(cls, raw: Mapping[str, Any]) -> StatsFile:
        """Migrate the supplement's format.

        Lossless in both directions that matter: every numeric quantity is
        carried over unchanged, unrecognised top-level keys are preserved in
        ``provenance.notes``, and key collisions created by canonicalisation are
        merged with Chan's formula rather than overwritten.
        """
        known_top_level = {
            "schema",
            "layers",
            "activation_ema_beta",
            "coherence_ema_beta",
            "eps",
            "created_at_unix",
        }
        notes = {k: v for k, v in raw.items() if k not in known_top_level}

        parsed = cls(
            layers={
                name: LayerStats.from_dict(name, body)
                for name, body in raw.get("layers", {}).items()
            },
            activation_ema_beta=float(raw.get("activation_ema_beta", 0.99)),
            coherence_ema_beta=_opt_float(raw.get("coherence_ema_beta")),
            eps=float(raw.get("eps", 1e-12)),
            provenance=Provenance(
                created_at_unix=_opt_float(raw.get("created_at_unix")),
                canonical_names=False,
                migrated_from=str(raw.get("schema", STATS_SCHEMA_V1)),
                # v1 files were written by the research code, whose only gradient
                # estimator was the one that hooks whatever parameter happens to
                # require grad -- LoRA factors under QLoRA, not the base weight.
                grad_estimator="param",
                notes=notes,
            ),
        )
        return parsed.canonicalized()

    # -- io ----------------------------------------------------------------

    def save(self, path: str | Path, *, stamp_time: bool = True) -> Path:
        target = Path(path)
        if target.is_dir():
            from dynquant.constants import STATS_FILENAME

            target = target / STATS_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        if stamp_time and self.provenance.created_at_unix is None:
            self.provenance.created_at_unix = time.time()
        payload = json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False)
        # Write-then-rename: a crash mid-write must not leave a truncated stats
        # file that a later run would silently under-cover from.
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> StatsFile:
        source = Path(path)
        if source.is_dir():
            source = _find_stats_in_dir(source)
        raw = json.loads(source.read_text(encoding="utf-8"))
        return cls.from_dict(raw, path=str(source))


def _find_stats_in_dir(directory: Path) -> Path:
    """Locate a stats file in a directory, accepting every historical name."""
    from dynquant.constants import LEGACY_STATS_FILENAMES, STATS_FILENAME

    for name in (STATS_FILENAME, *LEGACY_STATS_FILENAMES):
        candidate = directory / name
        if candidate.is_file():
            return candidate
    tried = ", ".join((STATS_FILENAME, *LEGACY_STATS_FILENAMES))
    raise FileNotFoundError(f"no DynQuant stats file found in {directory}. Looked for: {tried}")


def load_stats(path: str | Path) -> StatsFile:
    """Load a stats file or directory, migrating older schemas transparently."""
    return StatsFile.load(path)


def save_stats(stats: StatsFile, path: str | Path) -> Path:
    """Write a stats file atomically. Returns the path written."""
    return stats.save(path)
