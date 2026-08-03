"""MBPP: short Python tasks described in one sentence, scored by running the tests.

Paired with :mod:`dynquant.eval.humaneval` rather than replacing it. HumanEval's 164
problems are too few to resolve a two-point difference on their own -- the McNemar
window at that n is wide enough to swallow most of what quantization does -- and MBPP's
500-problem test split roughly triples the sample on the same skill with independently
written problems. Two code tasks that agree is also the cheapest evidence that a result
is about code rather than about one dataset's quirks.

**The task statement is not enough to write the function.** "Write a function to find
the similar elements from the given two tuple lists" does not say whether to call it
``similar_elements`` or ``find_common``, and the tests call one specific name. So the
asserts are part of the prompt -- every serious MBPP harness does this, and one that
does not is measuring the model's luck at guessing identifiers. It also means the entry
point is recoverable from the tests rather than from the gold solution; see
:func:`entry_point_of`.

Prompt framing and scoring follow :mod:`dynquant.eval.humaneval` exactly: ``auto``
picks chat for an instruct checkpoint and completion for a base one, decoding is greedy,
pass@1 is over one sample, and the per-problem hit vector is always stored.
"""

from __future__ import annotations

import ast
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
    "DEFAULT_CHAT_CONFIG",
    "DEFAULT_COMPLETION_CONFIG",
    "FEWSHOT_STOP",
    "MbppExample",
    "build_prompt",
    "build_test_program",
    "entry_point_of",
    "evaluate_mbpp",
    "load_mbpp",
]

_log = get_logger(__name__)

FEWSHOT_STOP = "[DONE]"
"""The original format's end-of-answer marker. A few-shot prompt teaches the model to
emit it, so it is the model's own turn boundary rather than an arbitrary cutoff."""

DEFAULT_COMPLETION_CONFIG = EvalConfig(
    max_new_tokens=512,
    batch_size=16,
    max_prompt_tokens=2048,
    stop_sequences=(FEWSHOT_STOP,),
)

DEFAULT_CHAT_CONFIG = EvalConfig(
    max_new_tokens=1024,
    batch_size=16,
    max_prompt_tokens=2048,
    add_special_tokens=False,
    stop_sequences=(),
)

_COMPLETION_BLOCK = (
    "You are an expert Python programmer, and here is your task: {text} "
    "Your code should pass these tests:\n\n{tests}\n[BEGIN]\n"
)

_CHAT_INSTRUCTION = (
    "You are an expert Python programmer. Write a Python function for this task:\n\n"
    "{text}\n\nYour code must pass these tests:\n\n```python\n{tests}\n```\n\n"
    "Return only the function, in a single ```python code block, with no explanation."
)

# Names that appear as the outermost call in an assert without being the thing under
# test. `assert set(similar_elements(a, b)) == set(c)` is the common shape.
_WRAPPERS = frozenset(
    """abs all any bool dict float frozenset int len list max min round set sorted str
    sum tuple type""".split()  # noqa: SIM905
)


@dataclass(frozen=True, slots=True)
class MbppExample:
    task_id: str
    text: str
    tests: tuple[str, ...]
    """The asserts, shown to the model *and* used to score it. Showing them is not
    leakage in any sense that matters here: they are the only statement of the required
    interface, and both arms of every comparison see the same ones."""

    setup: str = ""
    """``test_setup_code`` / ``test_imports``, which a handful of problems need and
    which is not part of the prompt."""

    code: str = ""
    """The reference solution. Never used for scoring -- kept for the entry-point
    fallback and for eyeballing failures."""


def load_mbpp(
    split: str = "test",
    *,
    config_name: Literal["full", "sanitized"] = "full",
    cache_dir: str | None = None,
) -> list[MbppExample]:
    """Load an MBPP split. Requires ``datasets``.

    ``full``/``test`` is the standard 500-problem evaluation set (task ids 11-510);
    ``sanitized`` is the 427-problem hand-checked subset, whose problem statements are
    clearer but whose sample is smaller.
    """
    from datasets import load_dataset

    raw = load_dataset(
        "google-research-datasets/mbpp", config_name, split=split, cache_dir=cache_dir
    )
    examples = []
    for row in raw:
        # The sanitized config renames `text` to `prompt` and `test_setup_code` to
        # `test_imports` (a list). Reading whichever exists keeps one loader.
        text = row.get("text") or row.get("prompt") or ""
        setup = row.get("test_setup_code") or ""
        if not setup and row.get("test_imports"):
            setup = "\n".join(row["test_imports"])
        examples.append(
            MbppExample(
                task_id=str(row["task_id"]),
                text=text.strip(),
                tests=tuple(row["test_list"]),
                setup=setup,
                code=row.get("code", ""),
            )
        )
    return examples


def entry_point_of(example: MbppExample) -> str:
    """The function name the tests call.

    Read out of the asserts by walking their AST, because the task statement never says
    it. Attribute calls are skipped so ``assert math.isclose(area(3), 9)`` yields
    ``area`` rather than nothing, and the obvious wrappers are skipped so
    ``assert set(f(x)) == set(y)`` yields ``f`` rather than ``set``.

    Only ever used to choose which fenced block is the answer, so a miss degrades to
    "take the first block" rather than to a wrong verdict.
    """
    for test in example.tests:
        try:
            tree = ast.parse(test)
        except SyntaxError:  # pragma: no cover -- the published asserts all parse
            continue
        for node in ast.walk(tree):
            # `math.isclose(...)` is an Attribute call, not a Name call, so it is skipped
            # here rather than returning the module name.
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in _WRAPPERS:
                return node.func.id
    for line in example.code.splitlines():
        stripped = line.strip()
        if stripped.startswith("def "):
            return stripped[4:].split("(")[0].strip()
    return ""


def build_test_program(example: MbppExample) -> str:
    """The suite to append after the candidate: setup first, then every assert."""
    parts = [example.setup.strip(), *(test.strip() for test in example.tests)]
    return "\n".join(part for part in parts if part)


def build_prompt(
    example: MbppExample,
    tokenizer: Any,
    *,
    style: PromptStyle,
    shots: Sequence[MbppExample] = (),
) -> str:
    """Render one problem, with the tests included in both framings."""
    tests = "\n".join(example.tests)
    if style == "completion":
        blocks = [
            _COMPLETION_BLOCK.format(text=shot.text, tests="\n".join(shot.tests))
            + f"{shot.code.strip()}\n{FEWSHOT_STOP}\n"
            for shot in shots
        ]
        blocks.append(_COMPLETION_BLOCK.format(text=example.text, tests=tests))
        return "\n".join(blocks)
    message = [
        {"role": "user", "content": _CHAT_INSTRUCTION.format(text=example.text, tests=tests)}
    ]
    return str(tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True))


def evaluate_mbpp(
    model: Any,
    tokenizer: Any,
    examples: Sequence[MbppExample],
    *,
    label: str,
    allow_execution: bool = False,
    style: PromptStyle | Literal["auto"] = "auto",
    shots: Sequence[MbppExample] = (),
    config: EvalConfig | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    memory_mb: int = DEFAULT_MEMORY_MB,
    max_workers: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    keep_predictions: int = 0,
) -> CodeEvalResult:
    """Score a model on MBPP by executing its solutions.

    Args:
        allow_execution: Required. See :mod:`dynquant.eval._code_exec`.
        shots: Few-shot exemplars for the completion framing, conventionally MBPP's own
            ``prompt`` split (task ids 1-10). Ignored under the chat framing, where an
            instruct model needs the instruction rather than the pattern. Must be the
            same list at every measurement point.
    """
    resolved = resolve_style(tokenizer, style)
    if config is None:
        config = DEFAULT_CHAT_CONFIG if resolved == "chat" else DEFAULT_COMPLETION_CONFIG
    elif resolved == "completion" and FEWSHOT_STOP not in config.stop_sequences:
        config = replace(config, stop_sequences=(*config.stop_sequences, FEWSHOT_STOP))
    config = prepare_decode(tokenizer, config, style=resolved, label=label)
    if shots and resolved == "chat":
        _log.info("chat framing ignores the %d few-shot exemplar(s) for %s", len(shots), label)

    subset = list(examples[: config.limit] if config.limit else examples)
    prompts = [build_prompt(example, tokenizer, style=resolved, shots=shots) for example in subset]
    generations = generate_batched(model, tokenizer, prompts, config, progress=progress)

    return score_generations(
        label=label,
        task="mbpp",
        prompt_style=resolved,
        keys=[example.task_id for example in subset],
        # Empty on purpose, in both framings. MBPP asks for a whole function rather
        # than the body of one already begun, so there is no signature to prepend and
        # no imports to recover -- prepending the natural-language task statement would
        # put English at the top of the program and fail every problem.
        prompts=["" for _ in subset],
        generations=generations,
        entry_points=[entry_point_of(example) for example in subset],
        tests=[build_test_program(example) for example in subset],
        allow_execution=allow_execution,
        timeout=timeout,
        memory_mb=memory_mb,
        max_workers=max_workers,
        keep_predictions=keep_predictions,
    )
