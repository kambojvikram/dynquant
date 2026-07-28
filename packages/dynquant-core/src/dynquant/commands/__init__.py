"""Bodies of the ``dynquant`` subcommands, one module each.

:mod:`dynquant.cli` owns the parser and nothing else. The work lives here, for two
reasons that both showed up while wiring these up.

**Import cost.** ``dynquant doctor`` has to run on a machine whose CUDA install is
broken -- that is what it is *for* -- so the CLI must not import a quantizer, a
model loader or ``transformers`` to build its parser. Keeping each command in its
own module makes that structural instead of a rule someone has to remember:
:mod:`dynquant.cli` imports one of these modules inside one handler, after
argument parsing has already succeeded.

**Testability.** Every command is a thin shell around a function that takes an
``nn.Module`` and returns a dataclass. ``run(args)`` resolves paths, loads a model
and prints; the function underneath does the work and is tested on the synthetic
graphs in ``tests/`` without a checkpoint, a GPU or a download. A command whose
logic is only reachable through ``argv`` is a command that gets tested by running
it once by hand.
"""

from __future__ import annotations

__all__ = ["bench", "evaluate", "inspect", "quantize"]
