"""Integration with the training and serving ecosystem.

Everything in here depends on a package that :mod:`dynquant` treats as optional --
``transformers``, ``peft``, ``trl`` -- so nothing is imported eagerly and each
submodule raises :class:`~dynquant.errors.MissingDependencyError` with the right
``pip install`` line if the dependency is absent.

* :mod:`dynquant.integration.peft_utils` -- adapter-aware naming and merging, the
  bridge between "what the tracker saw during a LoRA run" and "what the quantizer
  will find in the merged checkpoint".
* ``dynquant.integration.hf_quantizer`` -- ``HfQuantizer`` registration, so
  ``AutoModelForCausalLM.from_pretrained`` loads a DynQuant checkpoint (phase 9).
"""

from __future__ import annotations

__all__: list[str] = []
