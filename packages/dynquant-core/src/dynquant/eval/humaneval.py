"""HumanEval: 164 hand-written Python problems, scored by running the tests.

The reason to pay the cost of an execution sandbox is that this is the only task in the
phase-3 set where the scorer cannot be fooled. IFEval asks a Python function whether a
response followed an instruction; GSM8K compares a number. Here the model's output is
*run*, and a program that passes the tests is correct in a sense no regex has to be
trusted for. It is also the most brittle-in-the-right-way task in the set: an off-by-one
from a corrupted weight fails the assertion outright rather than degrading a score by
tenths.

Prompt framing
--------------
Two framings, and picking the wrong one costs more than the quantization does.

``completion`` is the original: the prompt is a signature and a docstring ending in a
newline, and the model continues the body. Correct for a base checkpoint.

``chat`` renders an instruction through the model's chat template and reads the fenced
block back out. Correct for an *instruct* checkpoint -- which all four phase-3 models
are. An instruct model scored under the completion framing appends "Sure! Here's the
function:" to a function signature, and every problem fails to parse. That arm reads as
destroyed, and nothing in the output says the harness did it.

``style="auto"`` follows the tokenizer, which gets this right without the caller having
to remember. The framing actually used is recorded on the result, because two arms
prompted differently are not comparable and the pass rates alone do not say so.

Scoring
-------
Greedy pass@1: one sample per problem, the same problems in the same order at every
measurement point, and a per-problem hit vector so every A/B is an exact McNemar test.
Not the unbiased pass@k estimator -- that needs sampling, and sampling turns a
two-point effect into noise. See :class:`~dynquant.eval._code_exec.CodeEvalResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from dynquant._logging import get_logger

from ._code_exec import (
    DEFAULT_MEMORY_MB,
    DEFAULT_TIMEOUT,
    CodeEvalResult,
    prepare_decode,
    resolve_style,
    score_generations,
)
from .harness import EvalConfig, generate_batched

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ._code_exec import PromptStyle

__all__ = [
    "COMPLETION_STOPS",
    "DEFAULT_CHAT_CONFIG",
    "DEFAULT_COMPLETION_CONFIG",
    "HumanEvalExample",
    "build_prompt",
    "build_test_program",
    "evaluate_humaneval",
    "load_humaneval",
]

_log = get_logger(__name__)

COMPLETION_STOPS = ("\nclass ", "\ndef ", "\n#", "\nif ", "\nprint(", "\n@", "\nassert ")
"""Where a completion ends: the first token that could only start a *new* top-level
statement. All anchored at column zero, so the function's own indented ``if`` and
``print`` survive -- a stop list without the leading newline truncates most solutions at
their first conditional and scores a correct model at near zero."""

DEFAULT_COMPLETION_CONFIG = EvalConfig(
    max_new_tokens=512,
    batch_size=16,
    max_prompt_tokens=1024,
    stop_sequences=COMPLETION_STOPS,
)

DEFAULT_CHAT_CONFIG = EvalConfig(
    max_new_tokens=1024,
    batch_size=16,
    max_prompt_tokens=2048,
    add_special_tokens=False,
    # No stop sequences: an instruct model's answer is a fenced block inside prose, and
    # the extractor finds it. Stopping at "\ndef " here would cut a solution that
    # defines a helper function above the one being asked for.
    stop_sequences=(),
)

_INSTRUCTION = (
    "Complete the following Python function. Write the entire function, including the "
    "signature, inside a single ```python code block. Do not write tests, examples, or "
    "an explanation.\n\n```python\n{prompt}```"
)


@dataclass(frozen=True, slots=True)
class HumanEvalExample:
    task_id: str
    prompt: str
    """Imports, signature and docstring, ending in a newline. Fed to the model verbatim
    under the completion framing and quoted inside the instruction under the chat one."""

    entry_point: str
    test: str
    """The dataset's ``check(candidate)`` definition. Not runnable on its own -- see
    :func:`build_test_program`."""

    canonical_solution: str = ""


def load_humaneval(*, cache_dir: str | None = None) -> list[HumanEvalExample]:
    """Load all 164 problems. Requires ``datasets``."""
    from datasets import load_dataset

    raw = load_dataset("openai/openai_humaneval", split="test", cache_dir=cache_dir)
    return [
        HumanEvalExample(
            task_id=row["task_id"],
            prompt=row["prompt"],
            entry_point=row["entry_point"],
            test=row["test"],
            canonical_solution=row.get("canonical_solution", ""),
        )
        for row in raw
    ]


def build_test_program(example: HumanEvalExample) -> str:
    """The suite to append after the candidate.

    The dataset's ``test`` field only *defines* ``check``; nothing calls it. A harness
    that appends the field alone runs a program that defines two functions, exits zero,
    and scores every arm at 100 %.
    """
    return f"{example.test.rstrip()}\n\ncheck({example.entry_point})"


def build_prompt(example: HumanEvalExample, tokenizer: Any, *, style: PromptStyle) -> str:
    """Render one problem in the requested framing."""
    if style == "completion":
        return example.prompt
    message = [{"role": "user", "content": _INSTRUCTION.format(prompt=example.prompt)}]
    return str(tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True))


def evaluate_humaneval(
    model: Any,
    tokenizer: Any,
    examples: Sequence[HumanEvalExample],
    *,
    label: str,
    allow_execution: bool = False,
    style: PromptStyle | Literal["auto"] = "auto",
    config: EvalConfig | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    memory_mb: int = DEFAULT_MEMORY_MB,
    max_workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    keep_predictions: int = 0,
) -> CodeEvalResult:
    """Score a model on HumanEval by executing its solutions.

    Args:
        allow_execution: Required. This runs code the model wrote; see
            :mod:`dynquant.eval._code_exec` for what the sandbox does and does not
            protect against.
        keep_predictions: Retain this many (generation, extracted candidate, failure
            detail) records. A pass rate alone cannot distinguish a model that writes
            wrong Python from one that has stopped writing Python, and those mean very
            different things about the bit map.
    """
    resolved = resolve_style(tokenizer, style)
    if config is None:
        config = DEFAULT_CHAT_CONFIG if resolved == "chat" else DEFAULT_COMPLETION_CONFIG
    elif resolved == "completion" and not config.stop_sequences:
        # `replace`, not a rebuild: a rebuild silently reverts every field it forgets.
        config = replace(config, stop_sequences=COMPLETION_STOPS)
    config = prepare_decode(tokenizer, config, style=resolved, label=label)

    subset = list(examples[: config.limit] if config.limit else examples)
    prompts = [build_prompt(example, tokenizer, style=resolved) for example in subset]
    generations = generate_batched(model, tokenizer, prompts, config, progress=progress)

    return score_generations(
        label=label,
        task="humaneval",
        prompt_style=resolved,
        keys=[example.task_id for example in subset],
        prompts=[example.prompt for example in subset],
        generations=generations,
        entry_points=[example.entry_point for example in subset],
        tests=[build_test_program(example) for example in subset],
        allow_execution=allow_execution,
        timeout=timeout,
        memory_mb=memory_mb,
        max_workers=max_workers,
        keep_predictions=keep_predictions,
    )
