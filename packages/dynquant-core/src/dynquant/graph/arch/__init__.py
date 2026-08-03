"""Bundled architecture plugins.

Importing this package registers every plugin in it. The registry imports it
lazily on first lookup, so a user who never classifies a model never pays for it.

Adding support for an architecture is additive: drop a module in here, decorate a
class with ``@register_arch("model_type")``, and import it below. Nothing else in
the package needs to change, and architectures without a plugin still classify
through structural inference and name matching -- a plugin makes the answer
better, it is not a prerequisite for the model working.
"""

from __future__ import annotations

from . import phi_fused, qwen_linear_attn

__all__ = ["phi_fused", "qwen_linear_attn"]
