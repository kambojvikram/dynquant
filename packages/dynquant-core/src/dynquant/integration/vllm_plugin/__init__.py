"""Serve a DynQuant checkpoint with vLLM, without patching vLLM.

    pip install dynquant
    vllm serve my-org/qwen3-2b-dynquant-3bit

That is the whole user-facing surface. :func:`register` is wired to vLLM's
``vllm.general_plugins`` entry-point group in ``pyproject.toml``, so vLLM calls it
once per process during startup -- engine, workers and API server alike -- and
``quant_method: "dynquant"`` in the checkpoint's ``config.json`` then resolves.
No fork, no ``--quantization`` flag, no code in the user's project.

Why a plugin rather than a patch
--------------------------------
The alternative -- shipping a diff against vLLM, or monkey-patching its registry
on import -- ties a DynQuant release to a vLLM release and breaks on every
upgrade. The plugin group and
``vllm.model_executor.layers.quantization.register_quantization_config`` are both
supported extension points, and everything this package touches inside vLLM is
one of those or a documented base class. The one thing that looked like it would
force a patch, ``WEIGHT_LOADER_V2_SUPPORTED`` being a hardcoded list, has a
public decorator for exactly this case; see
:mod:`dynquant.integration.vllm_plugin.linear`.

Layout of this package
----------------------
* :mod:`~dynquant.integration.vllm_plugin.config` -- the ``QuantizationConfig``.
* :mod:`~dynquant.integration.vllm_plugin.parameter` -- flat packed parameter.
* :mod:`~dynquant.integration.vllm_plugin.linear` -- the linear and embedding
  methods.

Everything above imports vLLM. The half that does not -- the flat-buffer and
tensor-parallel arithmetic, the ``quantization_config`` block, and the anti-inductor
custom op -- lives in :mod:`dynquant.integration.serving_common`, because SGLang's
quantization layer is a fork of vLLM's and needs the identical arithmetic. That
package is importable with nothing but torch, which is what makes it testable on a
laptop; this one is not, and its tests skip where vLLM is absent.

The package is named ``vllm_plugin`` and not ``vllm``: a submodule called
``dynquant.integration.vllm`` shadows the real ``vllm`` for any relative-looking
import inside this package and produces an ImportError whose message points at
the wrong project entirely.
"""

from __future__ import annotations

__all__ = ["register"]


def register() -> None:
    """Add ``dynquant`` to vLLM's quantization registry. Idempotent.

    vLLM loads general plugins once per process but a process may import this
    module by other routes too, and re-registering a name raises inside vLLM. The
    guard is a membership test rather than a module-level flag because the
    registry, not this module, is the thing whose state matters.

    Every import is inside the function body. This module is reachable from
    ``dynquant.integration``, and importing vLLM at module scope would make
    ``import dynquant`` cost several seconds and a CUDA context on any machine
    that happens to have vLLM installed.
    """
    from vllm.model_executor.layers.quantization import (
        QUANTIZATION_METHODS,
        register_quantization_config,
    )

    from dynquant.constants import HF_QUANT_METHOD

    if HF_QUANT_METHOD in QUANTIZATION_METHODS:
        return

    from dynquant.integration.vllm_plugin.config import DynQuantConfig

    register_quantization_config(HF_QUANT_METHOD)(DynQuantConfig)
