"""Serve a DynQuant checkpoint with SGLang, without patching SGLang.

    pip install dynquant
    python -m sglang.launch_server --model-path my-org/qwen3-2b-dynquant-3bit

:func:`register` is wired to SGLang's ``sglang.srt.plugins`` entry-point group in
``pyproject.toml``, so SGLang calls it once per process during startup and
``quant_method: "dynquant"`` in the checkpoint's ``config.json`` then resolves.

Why registration is three writes and not one
--------------------------------------------
vLLM has ``register_quantization_config`` and
``register_weight_loader_v2_supported_method``. SGLang has neither: its
quantization layer was forked from vLLM ``v0.5.5``, before those decorators
existed, and the containers are still plain module-level mutables. So this module
inserts into them directly:

===============================  ==================================  ============
Container                        Location (0.5.16)                   Public setter
===============================  ==================================  ============
``QUANTIZATION_METHODS``         ``layers/quantization/__init__.py``  none
``QUANTIZATION_CHOICES``         ``server_args.py``                   yes
``WEIGHT_LOADER_V2_SUPPORTED``   ``layers/linear.py``                 none
===============================  ==================================  ============

Two of the three are private by any reasonable reading, which is why every write
here is preceded by a check that the target exists and has the shape we expect,
and why failure raises :class:`SGLangIncompatibleError` naming the installed
version. The alternative is an ``AttributeError`` or a ``TypeError`` three frames
deep inside a spawned scheduler subprocess, whose traceback reaches the user as a
worker that died during startup.

``WEIGHT_LOADER_V2_SUPPORTED`` is the load-bearing one. It is a list of class-name
*strings*, tested at ``linear.py:369`` and ``:1454`` to decide which of two weight
loaders a layer gets. The v1 loader places shards with
``param.data.narrow(output_dim, ...)``, which assumes every row of the parameter is
the same width. DynQuant's buffers are flat and its shards have different word
counts, so there is no v1 loader that can *express* this layout at all -- missing
this write does not mean weights land in the wrong place, it means
``DynQuantPackedParameter``'s four placement hooks are never called.

The resulting failure is loud but useless. Every v1 path ends in ``assert
param_data.shape == loaded_weight.shape`` before the copy (``linear.py:437`` with a
message, ``:736`` bare), so the load stops -- as a bare ``AssertionError``, a
``RuntimeError`` from an out-of-range ``narrow``, or an ``IndexError`` from
indexing a 1-D buffer by ``input_dim``, raised inside a spawned scheduler
subprocess on a line that names neither the module nor quantization.
``test_sglang_linear.py`` proves the write is not decoration.

Ordering
--------
``load_plugins()`` runs before anything reads the containers, in every process that
matters -- notably as the first statement of ``run_scheduler_process()``, which is
where ``ModelConfig`` is actually built. The scheduler is *spawned*, so it does not
inherit the parent's registry mutations; a plugin system that only ran in the parent
would be useless here. It is idempotent behind a ``_plugins_loaded`` flag but may
still be reached more than once per process by other routes, so :func:`register`
is written to be re-entrant.

The package is named ``sglang_plugin`` and not ``sglang`` for the same reason
``vllm_plugin`` is not ``vllm``: a submodule named after the framework shadows the
real one for any relative-looking import inside the package, and the resulting
ImportError names the wrong project entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dynquant.errors import DynQuantError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable
    from types import ModuleType

__all__ = ["SGLangIncompatibleError", "register"]


class SGLangIncompatibleError(DynQuantError):
    """The installed SGLang does not expose a surface the plugin has to write to.

    Raised at registration rather than at model load, because registration is the
    last moment at which the message can still name what is missing. Past that
    point the symptom is a ``KeyError: 'dynquant'`` or a silently wrong weight
    loader, neither of which points at the cause.
    """


def _sglang_version() -> str:
    """Best-effort version string for error messages only.

    Deliberately total: this runs on the failure path, and a plugin that raised
    while composing the message for another failure would replace a diagnosable
    problem with an undiagnosable one.
    """
    try:
        from importlib.metadata import version

        return version("sglang")
    except Exception:  # noqa: BLE001 - see above; any failure here means "unknown"
        return "unknown"


def _incompatible(detail: str) -> SGLangIncompatibleError:
    return SGLangIncompatibleError(
        f"sglang {_sglang_version()} {detail}. "
        f"DynQuant registers itself by writing to SGLang's quantization containers "
        f"directly, because SGLang -- unlike vLLM -- publishes no decorator for it, "
        f"so a rename or a type change upstream lands here. Please open an issue at "
        f"https://github.com/kambojvikram/dynquant/issues with this message and the "
        f'output of `python -c "import sglang; print(sglang.__version__)"`.'
    )


_MISSING = object()


def _import(name: str) -> ModuleType:
    """Import one SGLang module, or say which module moved.

    A renamed module is the coarsest way this can break and the one whose native
    error is least useful -- ``ModuleNotFoundError: No module named
    'sglang.srt.server_args'`` inside a scheduler subprocess reads as a broken
    SGLang install rather than as a plugin that needs updating.
    """
    from importlib import import_module

    try:
        return import_module(name)
    except ImportError as exc:
        raise _incompatible(f"has no module {name}, which DynQuant registers into") from exc


def _lookup(module: ModuleType, attribute: str, expectation: str, ok: Callable[[Any], bool]) -> Any:
    """Read one container off a module, or explain what changed about it.

    Spelled as ``getattr`` rather than ``from sglang... import X``, which reads
    better, for a reason worth stating: a missing name in a ``from`` import raises
    :class:`ImportError` at the top of :func:`register`, *before* any check can run.
    That traceback says only that a name could not be imported from a module the
    user has never heard of. Reading the attribute here puts the absent case and the
    wrong-type case through the same message, which is the only reason the guard is
    worth having at all.
    """
    symbol = f"{module.__name__}.{attribute}"
    value = getattr(module, attribute, _MISSING)
    if value is _MISSING:
        raise _incompatible(f"does not expose {symbol} at all; DynQuant needs it {expectation}")
    if not ok(value):
        raise _incompatible(
            f"exposes {symbol} as {type(value).__name__}, but DynQuant needs it {expectation}"
        )
    return value


_LINEAR_METHOD_NAME = "DynQuantLinearMethod"
"""Spelled as a literal, and not as ``DynQuantLinearMethod.__name__``.

Reaching for the attribute would import the linear module -- and torch, and the
kernels -- inside :func:`register`, on every process start, only to read a string
that is already known. The cost of the literal is that a rename could desynchronise
it, which the equality check in ``test_sglang_linear.py`` guards once that module
exists.
"""


def register() -> None:
    """Add ``dynquant`` to SGLang's quantization registry. Idempotent.

    Every import is inside the function body. This module is reachable from
    ``dynquant.integration``, and importing SGLang at module scope would make
    ``import dynquant`` cost several seconds and a CUDA context on any machine that
    happens to have SGLang installed -- including machines only ever running
    ``dynquant quantize``, which needs neither.
    """
    from dynquant.constants import HF_QUANT_METHOD

    linear = _import("sglang.srt.layers.linear")
    quantization = _import("sglang.srt.layers.quantization")
    server_args = _import("sglang.srt.server_args")

    # Insert into QUANTIZATION_METHODS, not BASE_QUANTIZATION_METHODS. 0.5.16 split
    # the registry in two so platform backends can override entries, and the former
    # is built as `{**BASE_QUANTIZATION_METHODS}` at import time -- by the time any
    # plugin runs, the copy has been taken, and a write to BASE_ would be a no-op
    # that silently leaves `--quantization dynquant` unresolvable.
    methods = _lookup(quantization, "QUANTIZATION_METHODS", "as a dict", _is_dict)
    choices = _lookup(server_args, "QUANTIZATION_CHOICES", "as a list", _is_list)
    add_choices = _lookup(server_args, "add_quantization_method_choices", "as a callable", callable)
    loader_v2 = _lookup(linear, "WEIGHT_LOADER_V2_SUPPORTED", "as a list", _is_list)

    # Checked against the live containers rather than a module-level flag, for the
    # same reason as the vLLM plugin: the registry is the thing whose state matters,
    # and it outlives any bookkeeping we could keep beside it. `add_choices` is a
    # bare `list.extend` upstream, so deduplication is entirely our problem.
    if HF_QUANT_METHOD not in methods:
        # Imported here and not with the others so that every container check fails
        # before the config module -- and therefore torch, and therefore a CUDA
        # context -- is touched at all. On an incompatible SGLang the error is meant
        # to be cheap as well as readable.
        from dynquant.integration.sglang_plugin.config import DynQuantConfig

        methods[HF_QUANT_METHOD] = DynQuantConfig

    if HF_QUANT_METHOD not in choices:
        add_choices([HF_QUANT_METHOD])

    # A string, matched by class name. Both halves of that -- the exact spelling and
    # the fact that it is the *method* class rather than the config -- are checked
    # against the real class by the linear tests, so a rename cannot leave this
    # literal pointing at nothing.
    if _LINEAR_METHOD_NAME not in loader_v2:
        loader_v2.append(_LINEAR_METHOD_NAME)


def _is_dict(value: Any) -> bool:
    return isinstance(value, dict)


def _is_list(value: Any) -> bool:
    # Not `Sequence`: the write is an in-place mutation, and a tuple or a str would
    # both pass a Sequence check and then fail on `append` or on `extend`.
    return isinstance(value, list)
