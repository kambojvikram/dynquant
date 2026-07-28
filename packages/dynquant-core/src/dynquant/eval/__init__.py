"""Task evaluation, for measuring what quantization actually cost.

A quantization method's only interesting number is the accuracy it keeps, so the
evaluator has to be trustworthy in a specific way: **the same prompt, the same
decode, and the same scorer at every measurement point.** A comparison across
before-fine-tune / after-fine-tune / 4-bit / 3-bit is only a comparison if the
sole thing that changed between runs is the weights.

That rules out a few conveniences. Greedy decoding, not sampling -- a temperature
above zero turns a 1-point gap into noise you would need many seeds to see through.
A fixed few-shot prefix, including for the fine-tuned model, so the fine-tune is
measured as "did it get better at this task" rather than "did it learn a different
prompt format". And a scorer that reads the answer the same way whatever produced
it.

It also rules out reporting a difference as if the two runs were independent. They
are not: the same problems, in the same order, scored twice. :mod:`dynquant.eval.compare`
does the paired arithmetic, and every task result carries the per-problem correctness
vector it needs, because that vector cannot be recovered after the GPU-hours are spent.

One more thing the tasks have to be chosen for, which is not a property of the
harness: a task can only show what a fine-tune bought if the base model is not
already at the ceiling. :mod:`dynquant.eval.gsm8k` turned out not to be -- Qwen3.5-2B
scores 66% on it untouched -- and a flat result there says nothing about the
allocator. :mod:`dynquant.eval.casehold` and :mod:`dynquant.eval.banking77` were each
picked by measuring base-model headroom on several candidates first, against the model
that would actually be fine-tuned; both docstrings record the screen tables, including
the candidates that were rejected and why.

:mod:`dynquant.eval.harness` is the batched greedy decode loop shared by every task,
which is what makes "the same decode at every measurement point" structural rather
than a thing each task has to remember.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .banking77 import Banking77Result, evaluate_banking77, load_banking77
    from .casehold import CaseholdResult, evaluate_casehold, load_casehold
    from .compare import PairedComparison, compare_paired, mcnemar_exact
    from .gsm8k import Gsm8kResult, evaluate_gsm8k, load_gsm8k
    from .harness import EvalConfig, generate_batched

__all__ = [
    "Banking77Result",
    "CaseholdResult",
    "EvalConfig",
    "Gsm8kResult",
    "PairedComparison",
    "compare_paired",
    "evaluate_banking77",
    "evaluate_casehold",
    "evaluate_gsm8k",
    "generate_batched",
    "load_banking77",
    "load_casehold",
    "load_gsm8k",
    "mcnemar_exact",
]

_LAZY = {
    "Banking77Result": "banking77",
    "evaluate_banking77": "banking77",
    "load_banking77": "banking77",
    "CaseholdResult": "casehold",
    "evaluate_casehold": "casehold",
    "load_casehold": "casehold",
    "Gsm8kResult": "gsm8k",
    "evaluate_gsm8k": "gsm8k",
    "load_gsm8k": "gsm8k",
    "EvalConfig": "harness",
    "generate_batched": "harness",
    "PairedComparison": "compare",
    "compare_paired": "compare",
    "mcnemar_exact": "compare",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value
