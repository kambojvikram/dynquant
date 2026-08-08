"""Loading, allocation and bit-map IO shared by the subcommands.

Every function here exists because two commands would otherwise each have their
own copy, and the copies would drift in a way that changes results rather than
crashing. ``inspect`` and ``quantize`` must build the *same* bit map from the same
inputs -- otherwise inspecting an allocation tells you nothing about the one you
are about to write -- and the only durable way to get that is for them to call one
function.

The bit-map file is the seam between them: ``inspect --save-map`` writes it,
``quantize --map`` reads it, and a map that has been reviewed is then the thing
that gets quantized, with no second allocation in between that could differ.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dynquant.constants import (
    ALLOCATION_FILENAME,
    ALLOCATION_SCHEMA,
    COMPUTE_DTYPES,
    DEFAULT_GROUP_SIZE,
)
from dynquant.errors import DynQuantError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import torch
    from torch import nn

    from dynquant.allocate.budget import Budget
    from dynquant.allocate.knapsack import BitMap
    from dynquant.allocate.policy import AllocationPolicy
    from dynquant.graph.classify import ModelGraph
    from dynquant.score.sensitivity import SensitivityTable

__all__ = [
    "DTYPES",
    "AllocationInputs",
    "allocate",
    "build_inputs",
    "check_map_covers",
    "load_model",
    "load_tokenizer",
    "parse_overrides",
    "progress_printer",
    "read_bit_map",
    "resolve_dtype",
    "uniform_map",
    "write_bit_maps",
]

DTYPES = COMPUTE_DTYPES
"""Compute dtypes the CLI will load a model in.

Re-exported from :mod:`dynquant.constants` because :mod:`dynquant.cli` needs the
same tuple to build ``--dtype``'s ``choices`` and must not import this module to
get it -- parser construction imports nothing beyond the standard library, so
``dynquant doctor`` still runs on an install where ``torch`` is broken.
"""


def resolve_dtype(name: str) -> torch.dtype:
    import torch

    try:
        dtype = getattr(torch, name)
    except AttributeError as exc:  # pragma: no cover -- argparse constrains this
        raise DynQuantError(f"unknown dtype {name!r}; expected one of {DTYPES}") from exc
    if not isinstance(dtype, torch.dtype):  # pragma: no cover
        raise DynQuantError(f"{name!r} is not a torch dtype")
    return dtype


def load_model(
    path: str,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    trust_remote_code: bool = False,
) -> Any:
    """Load a causal LM, in one dtype, on one device.

    ``device="cpu"`` is meaningful for :mod:`~dynquant.commands.inspect`, which
    reads only names and shapes: a second copy of the weights on the GPU while an
    evaluation is running turns an analysis into an OOM in the run that matters.

    ``use_cache`` is left as the config has it. Both values are correct for
    something -- ``False`` for training and for the calibration pass, ``True`` for
    generation -- so the command that knows which it is sets it, rather than this
    function guessing on everyone's behalf.
    """
    transformers = _require_transformers()
    return transformers.AutoModelForCausalLM.from_pretrained(
        path,
        device_map=device,
        trust_remote_code=trust_remote_code,
        **{_dtype_kwarg(transformers): resolve_dtype(dtype)},
    )


def _dtype_kwarg(transformers: Any) -> str:
    """``"dtype"`` or ``"torch_dtype"``, whichever this transformers understands.

    4.56 renamed the argument. The old name still works there but warns; the new
    name on an older release is not an error at the call site -- it falls through
    ``**kwargs`` into the model constructor and surfaces as
    ``LlamaForCausalLM.__init__() got an unexpected keyword argument 'dtype'``,
    which names neither transformers nor the argument that is actually wrong.
    Since ``dynquant-core`` declares ``transformers>=4.45``, both halves of that
    range have to work.
    """
    version = getattr(transformers, "__version__", "")
    try:
        major, minor = (int(part) for part in version.split(".")[:2])
    except ValueError:  # a dev build like "5.0.0.dev0" still parses; "" does not
        return "dtype"
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def load_tokenizer(path: str, *, trust_remote_code: bool = False) -> Any:
    """Load a tokenizer and guarantee a pad token.

    A missing pad token is not a crash: ``transformers`` pads with 0, which is a
    real token in most vocabularies, so the batch is silently poisoned and the
    accuracy comes out low but plausible.
    """
    transformers = _require_transformers()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        path, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _require_transformers() -> Any:
    from dynquant.errors import MissingDependencyError

    try:
        import transformers
    except ImportError as exc:
        raise MissingDependencyError(
            "transformers", feature="loading a model from a checkpoint", extra="train"
        ) from exc
    return transformers


def progress_printer(label: str, *, every: int = 25) -> Callable[[int, int], None]:
    """A ``(done, total)`` callback that prints every ``every`` items and at the end.

    Every item would flood a log with 700 lines for one quantization; a spinner
    would print nothing a redirected log can use. This prints something a `tail -f`
    can read and a CI log can hold.
    """

    def report(done: int, total: int) -> None:
        if done == total or done % every == 0:
            print(f"  [{label}] {done}/{total}", flush=True)

    return report


# --------------------------------------------------------------------------
# Allocation
# --------------------------------------------------------------------------


class AllocationInputs:
    """Everything needed to allocate, resolved once and reused per target.

    Held as an object rather than passed as eight arguments because ``inspect``
    allocates several targets from one set of inputs and the expensive parts --
    classification, scoring, and above all the sensitivity pricing, which
    quantizes every tensor at every candidate width -- must not be repeated per
    target.
    """

    def __init__(
        self,
        graph: ModelGraph,
        scores: Mapping[str, float],
        *,
        policy: AllocationPolicy,
        sensitivity: SensitivityTable | None = None,
        score_report: Any = None,
        group_size: int = DEFAULT_GROUP_SIZE,
    ) -> None:
        self.graph = graph
        self.scores = scores
        self.policy = policy
        self.sensitivity = sensitivity
        self.score_report = score_report
        self.group_size = group_size

    @property
    def allocator(self) -> str:
        """Which quantity ordered the widths -- recorded in every artifact.

        Two bit maps at the same average width can come from different estimators,
        and the paper's own history is the reason this is written down: a map from
        the rank-product score and a map from measured sensitivity differ on
        roughly two-thirds of modules at 3.25 bits, and nothing in the map itself
        says which one produced it.
        """
        return "sensitivity" if self.sensitivity is not None else "rank_product"


def build_inputs(
    model: nn.Module,
    *,
    stats: str | None,
    moments: str | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    symmetric: bool = False,
    overrides: Mapping[str, str] | None = None,
    soft_floors: bool = True,
    verbose: bool = True,
) -> AllocationInputs:
    """Classify, score, and price sensitivity if the moments are there."""
    from dynquant.allocate.policy import AllocationPolicy
    from dynquant.graph.classify import classify_model
    from dynquant.score.importance import score_modules
    from dynquant.signals.schema import load_stats

    graph = classify_model(model, overrides=overrides)

    # No stats is not an error here, because not every map is allocated from a score.
    # `--uniform` needs the graph and nothing else, and it is precisely the arm a user
    # without fine-tuning signals has: refusing it -- while recommending it in the
    # refusal -- made the control arm reachable only by first collecting the signals it
    # exists to be compared against. `allocate` still refuses, which is where the
    # requirement actually is.
    report = None
    scores: Mapping[str, float] = {}
    if stats is not None:
        report = score_modules(graph, load_stats(stats))
        scores = report.scores()
        if verbose:
            print(report.summary(), flush=True)

    table: SensitivityTable | None = None
    if moments is not None:
        table = _price_sensitivity(
            model, graph, moments, group_size=group_size, symmetric=symmetric, verbose=verbose
        )

    policy = AllocationPolicy(group_size=group_size, symmetric=symmetric, soft_floors=soft_floors)
    return AllocationInputs(
        graph,
        scores,
        policy=policy,
        sensitivity=table,
        score_report=report,
        group_size=group_size,
    )


def _price_sensitivity(
    model: nn.Module,
    graph: ModelGraph,
    moments: str,
    *,
    group_size: int,
    symmetric: bool,
    verbose: bool,
) -> SensitivityTable:
    from dynquant.score.sensitivity import estimate_sensitivity, module_weights
    from dynquant.signals.moments import load_moments

    loaded = load_moments(moments)
    table = estimate_sensitivity(
        graph,
        loaded,
        module_weights(model),
        group_size=group_size,
        symmetric=symmetric,
    )
    if verbose:
        print(table.summary(), flush=True)
    return table


def allocate(
    inputs: AllocationInputs,
    *,
    target_bits: float | None = None,
    target_size: str | None = None,
    target_ratio: float | None = None,
) -> BitMap:
    """One bit map, from one budget, through the shipped knapsack."""
    from dynquant.allocate.budget import Budget
    from dynquant.allocate.knapsack import allocate_bits

    if not inputs.scores:
        raise DynQuantError(
            "allocation needs the signals collected during fine-tuning: pass --stats "
            "with the directory DynQuantCallback wrote to. Without it there is no "
            "importance ordering, and a uniform map is the honest alternative "
            "(`--uniform BITS`)."
        )

    budget: Budget = Budget.from_target(
        inputs.graph,
        target_bits=target_bits,
        target_size=target_size,
        target_ratio=target_ratio,
        group_size=inputs.group_size,
    )
    return allocate_bits(
        inputs.graph,
        inputs.scores,
        budget,
        inputs.policy,
        sensitivity=inputs.sensitivity,
    )


def uniform_map(
    graph: ModelGraph,
    width: int,
    *,
    policy: AllocationPolicy | None = None,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> BitMap:
    """Every module at ``width``, except structural roles, which keep their floor.

    The control arm. It is here rather than in a script because a uniform
    comparison is only worth reporting if it went through the same accounting as
    the allocated one -- the number to beat is *stored* bits, and a uniform map
    that ignored scale and offset overhead would be compared at the wrong budget.

    Structural roles are exempt because they are not a quality trade-off: a router
    at 2 bits does not degrade an MoE model's answers, it sends every token to the
    wrong expert. A "uniform 2-bit" arm that included them would measure a broken
    model rather than a low-precision one.

    Every *other* floor it breaches is reported, by the same rule the knapsack uses.
    The exemption covers structural roles only, so a uniform 3-bit map puts the
    embedding and the LM head at 3 bits as well -- under their 4- and 8-bit floors --
    and a control that reported no violations while the allocated arm at the same
    byte count reported seventy-nine would read as the allocated arm being the
    reckless one. Both arms breach; only one was counting.
    """
    from dynquant.allocate.budget import module_stored_bits
    from dynquant.allocate.knapsack import BitMap, FloorViolation
    from dynquant.allocate.policy import AllocationPolicy
    from dynquant.graph.roles import UNQUANTIZED_FLOOR

    policy = policy or AllocationPolicy(group_size=group_size)
    bits: dict[str, int] = {}
    violations: list[FloorViolation] = []
    total = float(graph.unquantized_params() * UNQUANTIZED_FLOOR)
    for info in graph.quantizable():
        assigned = width
        floor = policy.floor_for(info.role, info.tied_roles)
        if policy.is_structural(info.role, info.tied_roles):
            assigned = max(width, floor)
        if assigned < floor:
            violations.append(
                FloorViolation(
                    name=info.name,
                    role=info.role,
                    floor_bits=floor,
                    assigned_bits=assigned,
                    num_params=info.num_params,
                )
            )
        bits[info.name] = assigned
        total += module_stored_bits(info, assigned, group_size=group_size)

    return BitMap(
        bits=bits,
        violations=tuple(violations),
        budget_bits=total,
        allocated_bits=total,
        denominator=graph.total_params(),
        target_label=f"uniform {width} bit",
    )


# --------------------------------------------------------------------------
# Bit-map files
# --------------------------------------------------------------------------


def write_bit_maps(
    path: str | Path,
    maps: Mapping[str, BitMap],
    *,
    model: str,
    stats: str | None,
    allocator: str,
    group_size: int,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write one or more bit maps plus the provenance needed to reproduce them.

    Provenance is not decoration. A bare ``{name: bits}`` map cannot be checked
    against the model it was built for, and applying one to the wrong checkpoint
    fails only if a name happens to be missing -- otherwise it quantizes a
    similarly-named tensor at a width chosen for a different one.
    """
    from dynquant._version import __version__

    destination = Path(path)
    if destination.is_dir():
        destination = destination / ALLOCATION_FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "schema": ALLOCATION_SCHEMA,
        "dynquant_core": __version__,
        "model": model,
        "stats": stats,
        "allocator": allocator,
        "group_size": group_size,
        "maps": {key: _map_payload(value, group_size) for key, value in maps.items()},
    }
    if extra:
        payload.update(extra)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


def _map_payload(bit_map: BitMap, group_size: int) -> dict[str, Any]:
    return {
        "target_label": bit_map.target_label,
        "average_bits": bit_map.average_bits,
        "nbytes": bit_map.nbytes,
        "group_size": group_size,
        "histogram": {str(width): params for width, params in sorted(bit_map.histogram().items())},
        "violations": [
            {
                "name": v.name,
                "role": v.role.value,
                "floor_bits": v.floor_bits,
                "assigned_bits": v.assigned_bits,
                "num_params": v.num_params,
            }
            for v in bit_map.violations
        ],
        "bits": bit_map.bits,
    }


def read_bit_map(path: str, *, key: str | None = None) -> tuple[dict[str, int], dict[str, Any]]:
    """Read a bit map written by ``--save-map``, or by the experiment scripts.

    Returns ``(bits, metadata)``. Three shapes are accepted, because the
    experiment scripts predate this file format and their outputs are the only
    real bit maps in existence:

    * ``{"maps": {"3.00": {"bits": {...}}}}`` -- this writer, and the stage scripts
    * ``{"bits": {...}}`` -- a single map
    * ``{"model.layers.0...": 3, ...}`` -- a bare map, hand-written

    ``key`` selects among several maps; with one map it is optional, and with
    several it is required rather than defaulted, because picking "the first one"
    would quantize at a budget nobody asked for.
    """
    source = Path(path)
    if source.is_dir():
        source = source / ALLOCATION_FILENAME
    if not source.is_file():
        raise DynQuantError(f"no bit map at {source}")

    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DynQuantError(f"{source} is not valid JSON: {exc}") from exc

    if not isinstance(raw, dict):
        raise DynQuantError(f"{source} does not hold a bit map")

    metadata = {k: v for k, v in raw.items() if k not in {"maps", "bits"}}

    if "maps" in raw:
        maps = raw["maps"]
        if not isinstance(maps, dict) or not maps:
            raise DynQuantError(f"{source} has an empty 'maps' section")
        if key is None:
            if len(maps) > 1:
                raise DynQuantError(
                    f"{source} holds {len(maps)} bit maps ({', '.join(sorted(maps))}); "
                    f"pick one with --map-key"
                )
            key = next(iter(maps))
        elif key not in maps:
            raise DynQuantError(f"{source} has no map {key!r}; it holds {', '.join(sorted(maps))}")
        entry = maps[key]
        metadata["map_key"] = key
        metadata.update({k: v for k, v in entry.items() if k != "bits"})
        return _as_bits(entry.get("bits", {}), source), metadata

    if "bits" in raw:
        return _as_bits(raw["bits"], source), metadata

    return _as_bits(raw, source), {}


def _as_bits(raw: Any, source: Path) -> dict[str, int]:
    from dynquant.constants import BIT_OPTIONS

    if not isinstance(raw, dict) or not raw:
        raise DynQuantError(f"{source} holds no widths")
    bits: dict[str, int] = {}
    for name, width in raw.items():
        if not isinstance(width, int) or isinstance(width, bool):
            raise DynQuantError(f"{source}: width for {name!r} is {width!r}, not an integer")
        if width not in BIT_OPTIONS:
            raise DynQuantError(
                f"{source}: {name!r} is assigned {width} bits, which is not one of "
                f"{BIT_OPTIONS}. Every kernel is templated over exactly those widths."
            )
        bits[str(name)] = width
    return bits


def check_map_covers(model: nn.Module, bits: Mapping[str, int]) -> None:
    """Fail before touching a weight if the map names tensors this model lacks.

    Applying a mismatched map is the failure worth catching early: the names that
    do resolve get quantized at widths chosen for a different model, and the
    result is a working model with damage nobody allocated.

    Asked through the quantizer's own resolver rather than ``get_submodule``. A bit map
    entry does not have to be a module -- a batched MoE expert bank keeps its experts as
    bare parameters, which is what the state dict, the graph and the stats file all call
    them -- and a guard that only knows about modules refuses those maps while the
    quantizer behind it would have applied them. That is what it did on LFM2.5-8B-A1B.
    """
    from dynquant.quant.quantizer import resolves_to_weight

    missing = [name for name in bits if not resolves_to_weight(model, name)]
    if missing:
        shown = ", ".join(sorted(missing)[:5])
        raise DynQuantError(
            f"the bit map names {len(missing)} tensor(s) this model does not have "
            f"({shown}{', ...' if len(missing) > 5 else ''}). Was it built for a "
            f"different checkpoint?"
        )


def parse_overrides(pairs: Sequence[str] | None) -> dict[str, str] | None:
    """``--role NAME=ROLE`` pairs into the mapping ``classify_model`` takes."""
    if not pairs:
        return None
    out: dict[str, str] = {}
    for pair in pairs:
        name, _, role = pair.partition("=")
        if not name or not role:
            raise DynQuantError(f"--role expects NAME=ROLE, got {pair!r}")
        out[name] = role
    return out
