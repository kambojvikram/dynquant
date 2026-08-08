"""CaseHOLD: pick which of five legal holdings a case citation supports.

Why this task rather than GSM8K
-------------------------------
The four-point experiment ran first on GSM8K and the fine-tune moved it -0.99
points (140 problems flipped one way, 153 the other, p=0.48) -- a tie, not a
regression, but a useless one: with nothing separating the base and tuned models
there was no fine-tuning gain for quantization to be measured against.

The cause was measured, not guessed. Qwen3.5-2B-Base already scores 66.11% on
GSM8K, because grade-school arithmetic is saturated in modern pre-training. There
was no headroom. Three candidate replacements were probed against the *base* model
before committing to a second run:

===================  ==============  =======  =====================  ==========
candidate            base few-shot   chance   supervised reference   headroom
===================  ==============  =======  =====================  ==========
CaseHOLD             34.3%           20%      ~69%                   ~35 pts
sql-create-context   54.0%            0%      ~80%                   ~26 pts
MedMCQA              47.7%           25%      ~45-50%                ~0 pts
===================  ==============  =======  =====================  ==========

MedMCQA is GSM8K again -- the base model is already at what supervised training
reaches -- and it would have cost a second wasted fine-tune to discover that after
the fact. CaseHOLD has the most room and the cleanest attribution: the answer is one
of five holdings and the few-shot prefix already establishes that format, so a gain
cannot be the model "learning to answer in the right shape". It has to be
discrimination the model did not previously have.

What this module does not claim
-------------------------------
The chance floor is 20%, not 0%. A model destroyed by quantization returns to
roughly 20% here, where on GSM8K it went to 13.6% and looked dramatic. The headline
percentages are therefore compressed relative to GSM8K while carrying the same
information, which is why :attr:`CaseholdResult.hits` matters even more than it did
there -- the paired test reads which problems flipped, and is unaffected by where
the floor sits.

Prompt length is the hazard to watch. A CaseHOLD prompt runs several times longer
than a GSM8K one, so it can reach ``max_prompt_tokens``. The shared harness cuts
from the front and says so; see :mod:`dynquant.eval.harness`. Cutting from the back
would remove the holdings and the ``Answer:`` cue, and score the model on a question
it never saw.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from dynquant._logging import get_logger

from .harness import EvalConfig, generate_batched, strip_reasoning

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "CaseholdExample",
    "CaseholdResult",
    "build_prompt",
    "evaluate_casehold",
    "extract_answer",
    "format_training_text",
    "load_casehold",
]

_log = get_logger(__name__)

N_HOLDINGS = 5
CHANCE = 1.0 / N_HOLDINGS
"""What guessing scores. Quoted beside every result: "34%" means something quite
different against a 20% floor than against a 0% one, and a collapsed model returns
here rather than to zero."""

FEWSHOT_STOP = "\n\n"
"""Exemplars are separated by a blank line, so this is the model's own turn
boundary -- the point at which it would start inventing the next question."""

_PARQUET = (
    "https://huggingface.co/datasets/casehold/casehold/resolve/"
    "refs%2Fconvert%2Fparquet/all/{split}/0000.parquet"
)
"""The canonical repo still ships a loading script, which ``datasets`` 4.x refuses
to execute. The Hub's auto-converted parquet mirror is the same rows without one."""

_LEADING_CHOICE = re.compile(r"^\W*([0-4])")
_ANY_CHOICE = re.compile(r"[0-4]")


@dataclass(frozen=True, slots=True)
class CaseholdExample:
    citing_prompt: str
    """The citing text, with the cited holding blanked out as ``(<HOLDING>)``."""

    holdings: tuple[str, ...]
    answer: str
    """Index of the correct holding, as a string ``"0"``-``"4"``."""


@dataclass(slots=True)
class CaseholdResult:
    """One measurement point."""

    label: str
    correct: int
    total: int
    unparseable: int
    """Generations containing no digit 0-4 anywhere. Tracked separately from wrong
    answers because it means something different: the model has stopped answering
    the question rather than answering it badly. On a five-way choice a damaged
    model usually keeps emitting *a* digit, so a rising unparseable count is the
    earliest visible sign that quantization has broken format compliance."""

    hits: list[bool] = field(default_factory=list)
    """Per-problem correctness, in dataset order. Always recorded, never sampled.

    Two models scored on one fixed problem set is a paired design and the pairing is
    where the power lives -- most problems are answered the same way by both models
    and carry no information about which is better. Keeping only a count makes
    McNemar's test impossible after the GPU time has been spent. See
    :mod:`dynquant.eval.compare`."""

    predictions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def above_chance(self) -> float:
        """Accuracy minus the 20% floor. The part of the score that is skill."""
        return self.accuracy - CHANCE

    def summary(self) -> str:
        return (
            f"{self.label:<28} {self.accuracy:6.2%}  "
            f"({self.correct}/{self.total} exact match, "
            f"{self.above_chance:+.2%} vs chance, {self.unparseable} unparseable)"
        )


def load_casehold(split: str = "test", *, cache_dir: str | None = None) -> list[CaseholdExample]:
    """Load a CaseHOLD split. Requires ``datasets``."""
    from datasets import load_dataset

    raw = load_dataset(
        "parquet", data_files=_PARQUET.format(split=split), split="train", cache_dir=cache_dir
    )
    return [
        CaseholdExample(
            citing_prompt=row["citing_prompt"].strip(),
            holdings=tuple(row[f"holding_{i}"].strip() for i in range(N_HOLDINGS)),
            answer=str(row["label"]).strip(),
        )
        for row in raw
    ]


def _question(example: CaseholdExample) -> str:
    options = "\n".join(f"{i}. {holding}" for i, holding in enumerate(example.holdings))
    return f"{example.citing_prompt}\n\nWhich holding is cited?\n{options}\nAnswer:"


def format_training_text(example: CaseholdExample) -> tuple[str, str]:
    """Return ``(prompt, completion)`` for supervised fine-tuning.

    Split rather than concatenated so the trainer masks the loss to the completion,
    and here that is not an optimisation but the difference between a fine-tune that
    works and one that does nothing. The prompt is ~400 tokens of appellate prose and
    the completion is a single digit. Train on the whole sequence and the answer
    carries under 1% of the gradient: the run is spent learning to write legal prose,
    which the model already does well, and the evaluation comes back flat -- with a
    healthy-looking training loss the whole way down.
    """
    return _question(example), f" {example.answer}"


def build_prompt(example: CaseholdExample, shots: Sequence[CaseholdExample]) -> str:
    """The few-shot prompt, in exactly the format the fine-tune trains on."""
    blocks = [_question(shot) + f" {shot.answer}" for shot in shots]
    blocks.append(_question(example))
    return "\n\n".join(blocks)


def extract_answer(text: str) -> str | None:
    """Pull the chosen holding index out of a generation.

    Prefers a digit at the very start, which is what the format asks for. Falls back
    to the first 0-4 anywhere, because a model that answers ``"holding 3"`` has
    chosen; scoring that as a failure would attribute a formatting wobble to
    quantization damage. Returns ``None`` only when there is no candidate digit at
    all, which is the genuinely different outcome the format has broken down.

    A reasoning trace is cut first (:func:`~dynquant.eval.harness.strip_reasoning`), a
    no-op for a model that does not emit one. It matters here because this extractor
    falls back to scanning the whole text: inside a trace the candidate it would find is
    one the model may still have been arguing with.
    """
    text = strip_reasoning(text)
    leading = _LEADING_CHOICE.match(text.strip())
    if leading:
        return leading.group(1)
    anywhere = _ANY_CHOICE.search(text)
    return anywhere.group(0) if anywhere else None


def evaluate_casehold(
    model: Any,
    tokenizer: Any,
    examples: Sequence[CaseholdExample],
    *,
    label: str,
    shots: Sequence[CaseholdExample] = (),
    config: EvalConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
    keep_predictions: int = 0,
) -> CaseholdResult:
    """Score a model on CaseHOLD by exact match on the holding index.

    Args:
        shots: Few-shot exemplars, from the *train* split, and the same list at
            every measurement point. They must also be excluded from the
            fine-tuning set -- an exemplar shown in every prompt and trained on is
            an answer handed to the model for free.
        keep_predictions: Retain this many (generation, gold) pairs. A score alone
            cannot distinguish a model choosing the wrong holding from one that has
            stopped emitting digits.
    """
    config = config or EvalConfig(
        max_new_tokens=8, stop_sequences=(FEWSHOT_STOP,), max_prompt_tokens=4096
    )
    if FEWSHOT_STOP not in config.stop_sequences:
        # `replace`, not a field-by-field rebuild: a rebuild silently reverts any
        # field it forgets to list, and the fields it is most likely to forget are
        # the ones added after it was written.
        config = replace(config, stop_sequences=(*config.stop_sequences, FEWSHOT_STOP))

    subset = list(examples[: config.limit] if config.limit else examples)
    prompts = [build_prompt(example, shots) for example in subset]
    generations = generate_batched(model, tokenizer, prompts, config, progress=progress)

    result = CaseholdResult(label=label, correct=0, total=len(subset), unparseable=0)
    for index, (example, text) in enumerate(zip(subset, generations, strict=True)):
        predicted = extract_answer(text)
        if predicted is None:
            result.unparseable += 1
        hit = predicted is not None and predicted == example.answer
        result.correct += int(hit)
        result.hits.append(hit)
        if index < keep_predictions:
            result.predictions.append(
                {
                    "generation": text.strip(),
                    "predicted": predicted,
                    "gold": example.answer,
                    "correct": hit,
                }
            )

    _log.info("%s", result.summary())
    return result
