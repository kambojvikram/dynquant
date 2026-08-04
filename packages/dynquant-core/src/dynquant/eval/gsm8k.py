"""GSM8K: grade-school math word problems, scored by exact match.

Chosen for this experiment because it is *brittle in the right way*. A word problem
needs a multi-step chain where every step is right; a single corrupted weight
shows up as a wrong final number rather than as slightly worse prose. Perplexity
would move by tenths and leave you guessing whether the model still works. This
either gets 72 or it does not.

The dataset ships each answer as a worked solution ending in ``#### <number>``, so
the gold answer needs no parsing heuristics -- only the model's output does.

Prompt format
-------------
A fixed few-shot prefix drawn from the *train* split, identical at every
measurement point and identical to what the fine-tune trains on::

    Question: {question}
    Answer: {reasoning}
    #### {number}

Few-shot even for the fine-tuned model. Dropping the prefix after fine-tuning
would confound "the fine-tune taught it math" with "the fine-tune taught it this
output format", and those have very different implications for what the signal map
is measuring.

Scoring
-------
Exact match on the final number. The comparison is numeric, not textual, after
stripping the separators models emit inconsistently (``1,000`` / ``$1000`` /
``1000.00``) -- otherwise the score measures formatting compliance and the
quantized models get punished for a cosmetic drift that a human grader would not
count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from dynquant._logging import get_logger

from .harness import EvalConfig, generate_batched

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "FEWSHOT_STOP",
    "Gsm8kExample",
    "Gsm8kResult",
    "build_prompt",
    "evaluate_gsm8k",
    "extract_answer",
    "format_training_text",
    "load_gsm8k",
]

_log = get_logger(__name__)

FEWSHOT_STOP = "Question:"
"""Where a continuation ends: the model's own turn boundary rather than an
arbitrary cutoff.

Deliberately *not* ``"\\n\\nQuestion:"``, which is how the few-shot prefix separates
its exemplars and therefore the obvious thing to cut on. Models do not reliably
reproduce the separator. Qwen2.5-1.5B-Instruct writes ``"the answer is 366.
Question: There are 12 more green apples..."`` -- a new problem, on the same line,
after a single space -- and a stop that requires the blank line matches nothing.
Generation then runs to ``max_new_tokens`` through two or three invented problems,
and :func:`extract_answer`'s "last number in the text" fallback returns an answer to
one of *those*. That is worse than an unparseable generation: it is a wrong answer
that looks like bad arithmetic, and it cost 24 points of GSM8K on the runtime-parity
gate before it was found. Cutting on the bare word costs nothing, because a
continuation that says "Question:" has stopped answering either way."""

_GOLD = re.compile(r"####\s*(.+?)\s*$", re.DOTALL)
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
_CALCULATOR = re.compile(r"<<[^>]*>>")


@dataclass(frozen=True, slots=True)
class Gsm8kExample:
    question: str
    reasoning: str
    """The worked solution with the ``#### n`` line and the dataset's inline
    ``<<3*4=12>>`` calculator annotations removed. The annotations are an artefact
    of how the dataset was collected; training on them teaches the model to emit
    markup that the scorer then has to ignore."""

    answer: str


@dataclass(slots=True)
class Gsm8kResult:
    """One measurement point."""

    label: str
    correct: int
    total: int
    unparseable: int
    """Generations with no number in them at all. Distinguished from wrong answers
    because they mean something different: a model that produces no number has
    stopped following the format, which is what catastrophic quantization damage
    looks like before it looks like bad arithmetic."""

    hits: list[bool] = field(default_factory=list)
    """Per-problem correctness, in dataset order. Always recorded.

    Two models scored on the same fixed problem set is a *paired* design, and the
    pairing is where most of the statistical power lives: comparing two accuracies
    as independent proportions throws away the fact that both models got the same
    easy problems right and the same hard ones wrong. On GSM8K the two-proportion
    interval is roughly twice as wide as McNemar's on the same data, which is the
    difference between resolving a two-point effect and reporting "inconclusive".

    Keeping only a summary count makes that test impossible after the fact, and the
    run costs GPU-hours -- so the vector is not optional and not sampled. It is 1319
    booleans."""

    predictions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def summary(self) -> str:
        return (
            f"{self.label:<28} {self.accuracy:6.2%}  "
            f"({self.correct}/{self.total} exact match, {self.unparseable} unparseable)"
        )


def load_gsm8k(split: str = "test", *, cache_dir: str | None = None) -> list[Gsm8kExample]:
    """Load a GSM8K split. Requires ``datasets``."""
    from datasets import load_dataset

    raw = load_dataset("openai/gsm8k", "main", split=split, cache_dir=cache_dir)
    return [
        Gsm8kExample(
            question=row["question"].strip(),
            reasoning=_CALCULATOR.sub("", row["answer"].split("####")[0]).strip(),
            answer=_normalise(_GOLD.search(row["answer"]).group(1)),  # type: ignore[union-attr]
        )
        for row in raw
    ]


def format_training_text(example: Gsm8kExample) -> tuple[str, str]:
    """Return ``(prompt, completion)`` for supervised fine-tuning.

    Split rather than concatenated so the trainer can mask the loss to the
    completion. Training on the question tokens too would spend capacity modelling
    the prompt distribution -- and, worse for this experiment, would make the
    gradient signal partly about question text rather than about solving.
    """
    return (
        f"Question: {example.question}\nAnswer:",
        f" {example.reasoning}\n#### {example.answer}",
    )


def build_prompt(example: Gsm8kExample, shots: Sequence[Gsm8kExample]) -> str:
    """The few-shot prompt, in exactly the format the fine-tune trains on."""
    blocks = [
        f"Question: {shot.question}\nAnswer: {shot.reasoning}\n#### {shot.answer}" for shot in shots
    ]
    blocks.append(f"Question: {example.question}\nAnswer:")
    return "\n\n".join(blocks)


def extract_answer(text: str) -> str | None:
    """Pull the predicted number out of a generation.

    Prefers the ``#### n`` marker the format asks for. Falls back to the last
    number in the text, because a model that reasons correctly and then forgets the
    marker has still solved the problem, and scoring that as a failure would
    overstate the damage from quantization -- format compliance degrades before
    arithmetic does.
    """
    marked = _GOLD.search(text)
    if marked:
        candidates = _NUMBER.findall(marked.group(1))
        if candidates:
            return _normalise(candidates[0])
    candidates = _NUMBER.findall(text)
    return _normalise(candidates[-1]) if candidates else None


def evaluate_gsm8k(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Gsm8kExample],
    *,
    label: str,
    shots: Sequence[Gsm8kExample] = (),
    config: EvalConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
    keep_predictions: int = 0,
) -> Gsm8kResult:
    """Score a model on GSM8K by exact match.

    Args:
        shots: Few-shot exemplars, from the *train* split. Must be the same list at
            every measurement point for the numbers to be comparable.
        keep_predictions: Retain this many (question, generation, gold) triples for
            inspection. A score alone cannot tell you whether a 3-bit model is
            doing worse arithmetic or has stopped speaking English.
    """
    config = config or EvalConfig(stop_sequences=(FEWSHOT_STOP,))
    if FEWSHOT_STOP not in config.stop_sequences:
        # `replace`, not a field-by-field rebuild: a rebuild silently reverts any
        # field it forgets to list, and the fields it is most likely to forget are
        # the ones added after it was written.
        config = replace(config, stop_sequences=(*config.stop_sequences, FEWSHOT_STOP))

    subset = list(examples[: config.limit] if config.limit else examples)
    prompts = [build_prompt(example, shots) for example in subset]
    generations = generate_batched(model, tokenizer, prompts, config, progress=progress)

    result = Gsm8kResult(label=label, correct=0, total=len(subset), unparseable=0)
    for index, (example, text) in enumerate(zip(subset, generations, strict=True)):
        predicted = extract_answer(text)
        if predicted is None:
            result.unparseable += 1
        hit = predicted is not None and _numeric_match(predicted, example.answer)
        result.correct += int(hit)
        result.hits.append(hit)
        if index < keep_predictions:
            result.predictions.append(
                {
                    "question": example.question,
                    "generation": text.strip(),
                    "predicted": predicted,
                    "gold": example.answer,
                    "correct": hit,
                }
            )

    _log.info("%s", result.summary())
    return result


# --------------------------------------------------------------------------
# Answer normalisation
# --------------------------------------------------------------------------


def _normalise(raw: str) -> str:
    """Strip the separators models use inconsistently, keeping the digits."""
    return raw.strip().replace(",", "").replace("$", "").replace("%", "").rstrip(".")


def _numeric_match(predicted: str, gold: str) -> bool:
    """Compare as numbers where possible, as strings otherwise.

    ``1000`` and ``1000.00`` are the same answer; a string comparison would call
    them different and quietly deflate every score by a percent or two. Falls back
    to string equality when either side will not parse, so a non-numeric gold
    answer still has a defined result rather than raising mid-run.
    """
    if predicted == gold:
        return True
    try:
        return abs(float(predicted) - float(gold)) < 1e-6
    except ValueError:
        return False
