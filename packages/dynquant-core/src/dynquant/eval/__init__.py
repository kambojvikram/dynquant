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
than a thing each task has to remember. It also owns the tokenize/detokenize
boundary, so :mod:`dynquant.eval.backends` can swap the engine underneath -- a task
is handed either a ``transformers`` model or a vLLM one through the same argument and
cannot tell which it got. Serving every arm through vLLM is what makes a campaign of
this size affordable, and evaluating through the runtime the checkpoints are actually
served with is worth having on its own; neither is worth a prompt that differs between
the two arms, which is why the ids, not the strings, are what crosses the boundary.

:mod:`dynquant.eval.ifeval` breaks the few-shot convention, deliberately. Its prompts
are zero-shot instructions addressed to an assistant, so it renders them through the
model's chat template instead; a fixed prefix would be measuring the wrong thing, since
the constraint under test *is* the prompt. The rest of the discipline is unchanged --
greedy decode, a parameterless scorer, per-item hits -- and the two things a template
introduces that a few-shot prefix does not (a second BOS token, and a base model with
no template at all) are handled explicitly rather than left to chance.

:mod:`dynquant.eval.humaneval` and :mod:`dynquant.eval.mbpp` are the only tasks whose
scorer cannot be argued with: the model's output is executed, and it either passes the
assertions or it does not. That costs a sandbox, so **execution is opt-in** -- both
refuse to run until the caller passes ``allow_execution=True``, because importing an
evaluation module should never be enough to run code a language model wrote. What the
subprocess isolation covers, and what it does not, is set out in
:mod:`dynquant.eval._code_exec`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._code_exec import CodeEvalResult
    from .backends import VllmBackend
    from .banking77 import Banking77Result, evaluate_banking77, load_banking77
    from .casehold import CaseholdResult, evaluate_casehold, load_casehold
    from .compare import PairedComparison, compare_paired, mcnemar_exact
    from .gsm8k import Gsm8kResult, evaluate_gsm8k, load_gsm8k
    from .harness import EvalBackend, EvalConfig, generate_batched
    from .humaneval import evaluate_humaneval, load_humaneval
    from .ifeval import IfevalResult, evaluate_ifeval, load_ifeval
    from .mbpp import evaluate_mbpp, load_mbpp

__all__ = [
    "Banking77Result",
    "CaseholdResult",
    "CodeEvalResult",
    "EvalBackend",
    "EvalConfig",
    "Gsm8kResult",
    "IfevalResult",
    "PairedComparison",
    "VllmBackend",
    "compare_paired",
    "evaluate_banking77",
    "evaluate_casehold",
    "evaluate_gsm8k",
    "evaluate_humaneval",
    "evaluate_ifeval",
    "evaluate_mbpp",
    "generate_batched",
    "load_banking77",
    "load_casehold",
    "load_gsm8k",
    "load_humaneval",
    "load_ifeval",
    "load_mbpp",
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
    "IfevalResult": "ifeval",
    "evaluate_ifeval": "ifeval",
    "load_ifeval": "ifeval",
    "CodeEvalResult": "_code_exec",
    "evaluate_humaneval": "humaneval",
    "load_humaneval": "humaneval",
    "evaluate_mbpp": "mbpp",
    "load_mbpp": "mbpp",
    "EvalBackend": "harness",
    "EvalConfig": "harness",
    "generate_batched": "harness",
    "VllmBackend": "backends",
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
