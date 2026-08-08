"""Banking77: route a customer message to one of 77 fine-grained banking intents.

Why this task
-------------
Chosen by measurement, not by reputation. The rule comes from GSM8K, where a
six-arm run was spent before anyone checked that the base model already scored
66% and there was therefore no fine-tuning gain for quantization to be measured
against. So four unused open datasets were screened against *base*
Mistral-7B-Instruct-v0.3 first, a few hundred rows each, in the same harness that
scores the real runs:

===========  ==============  ========  =====================  ==========
candidate    base few-shot   chance    supervised reference   headroom
===========  ==============  ========  =====================  ==========
Banking77    36.3%  [1]        1.3%    ~93% (BERT-base)       ~57 pts
MNLI         71.0%  (300)     33.3%    ~90% (BERT-large)      ~19 pts
PubMedQA     57.5%  (200)     33.3%    ~73%                   ~15 pts
LogiQA       41.3%  (300)     25.0%    ~40% (RoBERTa-large)   none
===========  ==============  ========  =====================  ==========

[1] Full 3,080-row test set, not the screen's 300. The screen reported 41.0%, and
that figure was wrong for the reason :func:`load_banking77` now exists to prevent:
this split alone of the four ships in label order, so its first 300 rows cover 8
intents of 77. The other three interleave (540, 116 and 6,515 label changes
respectively) and their numbers stand as screened. The correction moves Banking77
in the direction that widens its lead, so the choice below is unaffected -- but it
was luck that it did, and the ranking was made on a number that had not been
earned.

LogiQA is the interesting rejection: at 41.3% the *base* model is already above
the supervised reference, so a fine-tune there could only move the number down.
Banking77 wins on every axis that matters here. Fifty-seven points of headroom is
the widest gap the screen found; the distinctions the task turns on
(``card_payment_fee_charged`` against ``extra_charge_on_statement``, six separate
top-up failure modes) are label semantics no pre-training corpus teaches, so a
gain is discrimination the model did not previously have rather than format
compliance; and the inputs are one sentence each, which makes 10,003 training
rows cheap.

What the 1.3% chance floor buys
-------------------------------
CaseHOLD bottoms out at 20% and GSM8K at 0%. Here it is 1/77, which makes this the
most sensitive of the three to quantization damage: a model whose weights have
been destroyed cannot land on the right intent by luck, so the gap between "still
works" and "broken" is nearly the full range of the scale. The cost is the mirror
image -- there is no cushion, so a small real regression is visible rather than
buried, which is the direction an evaluation should err in.

Why the intent list is in the prompt
------------------------------------
The model answers with an index, and an index is meaningless without the list that
numbers it. :data:`HEADER` is therefore part of the prompt at every measurement
point *and* in :func:`format_training_text`, so the fine-tune trains against the
same 77 lines it is scored against. Dropping it from training would mean the model
learns a mapping from query to a numbering it never saw, and the evaluation would
measure the mismatch rather than the model.

The header costs ~600 of the ~710 prompt tokens and the completion is a single
digit-pair, so more than 98% of each training sequence is masked out of the loss.
That is deliberate and it is not waste: it is the same asymmetry CaseHOLD has, and
the alternative -- training on the whole sequence -- spends the run teaching the
model to recite the intent list.
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
    "INTENTS",
    "SHUFFLE_SEED",
    "Banking77Example",
    "Banking77Result",
    "build_prompt",
    "deterministic_order",
    "evaluate_banking77",
    "extract_answer",
    "format_training_text",
    "load_banking77",
]

_log = get_logger(__name__)

INTENTS: tuple[str, ...] = (
    "activate my card",
    "age limit",
    "apple pay or google pay",
    "atm support",
    "automatic top up",
    "balance not updated after bank transfer",
    "balance not updated after cheque or cash deposit",
    "beneficiary not allowed",
    "cancel transfer",
    "card about to expire",
    "card acceptance",
    "card arrival",
    "card delivery estimate",
    "card linking",
    "card not working",
    "card payment fee charged",
    "card payment not recognised",
    "card payment wrong exchange rate",
    "card swallowed",
    "cash withdrawal charge",
    "cash withdrawal not recognised",
    "change pin",
    "compromised card",
    "contactless not working",
    "country support",
    "declined card payment",
    "declined cash withdrawal",
    "declined transfer",
    "direct debit payment not recognised",
    "disposable card limits",
    "edit personal details",
    "exchange charge",
    "exchange rate",
    "exchange via app",
    "extra charge on statement",
    "failed transfer",
    "fiat currency support",
    "get disposable virtual card",
    "get physical card",
    "getting spare card",
    "getting virtual card",
    "lost or stolen card",
    "lost or stolen phone",
    "order physical card",
    "passcode forgotten",
    "pending card payment",
    "pending cash withdrawal",
    "pending top up",
    "pending transfer",
    "pin blocked",
    "receiving money",
    "refund not showing up",
    "request refund",
    "reverted card payment",
    "supported cards and currencies",
    "terminate account",
    "top up by bank transfer charge",
    "top up by card charge",
    "top up by cash or cheque",
    "top up failed",
    "top up limits",
    "top up reverted",
    "topping up by card",
    "transaction charged twice",
    "transfer fee charged",
    "transfer into account",
    "transfer not received by recipient",
    "transfer timing",
    "unable to verify identity",
    "verify my identity",
    "verify source of funds",
    "verify top up",
    "virtual card not working",
    "visa or mastercard",
    "why verify identity",
    "wrong amount of cash received",
    "wrong exchange rate for cash withdrawal",
)
"""The taxonomy, in the dataset's own label order -- index *is* the gold answer, so
this tuple's order is a correctness invariant, not a presentation choice.

Rendered for a reader rather than for a parser: upstream ships
``card_payment_fee_charged``, ``Refund_not_showing_up`` (capitalised, alone among
the 77) and ``reverted_card_payment?``. Underscores become spaces, the stray
capital and question mark go, because the model has to tell these apart as English
and snake_case tokenizes into fragments that blur an already fine-grained
distinction. Order is untouched, which is the part that matters.
"""

N_INTENTS = len(INTENTS)
CHANCE = 1.0 / N_INTENTS
"""1.3%. Quoted beside every result: a collapsed model returns here, and against a
floor this low there is nowhere for damage to hide."""

FEWSHOT_STOP = "\n\n"
"""Exemplars are separated by a blank line, so this is the model's own turn
boundary -- the point at which it would start inventing the next query."""

HEADER = (
    "Classify the customer query into one of these banking intents.\nAnswer with the number only.\n"
) + "\n".join(f"{index}. {intent}" for index, intent in enumerate(INTENTS))
"""Prepended once per prompt, ahead of the exemplars, and carried into training.

Once, not per exemplar: repeating 77 lines four times would treble the prompt for
no information the model does not already have in front of it.
"""

_SOURCE = "legacy-datasets/banking77"
"""Not ``PolyAI/banking77``, the canonical repo. That one is script-only,
``datasets`` 5.x refuses to execute a loading script, and the Hub never produced a
``refs/convert/parquet`` revision for it -- so unlike CaseHOLD there is no mirror
to fall back to. This repo is the same 10,003 train / 3,080 test rows as parquet
with the 77-way ``ClassLabel`` intact."""

SHUFFLE_SEED = 0
"""Fixes the load order. Both splits arrive sorted by label -- see :func:`load_banking77`
for why that has to be undone, and why it has to be undone the same way every time."""

_LEADING_INDEX = re.compile(r"^\W*(\d+)")
_ANY_INDEX = re.compile(r"\d+")


@dataclass(frozen=True, slots=True)
class Banking77Example:
    text: str
    """The customer's message, one or two sentences."""

    answer: str
    """Index into :data:`INTENTS`, as a string ``"0"``-``"76"``."""


@dataclass(slots=True)
class Banking77Result:
    """One measurement point."""

    label: str
    correct: int
    total: int
    unparseable: int
    """Generations containing no in-range index anywhere. Distinct from a wrong
    answer: it means the model has stopped answering the question rather than
    answering it badly. Note that ``"81"`` counts here -- it is a digit string, but
    it names no intent, so treating it as a wrong answer would hide a model that
    has lost track of the list it was given."""

    hits: list[bool] = field(default_factory=list)
    """Per-problem correctness, in dataset order. Always recorded, never sampled --
    the comparison that matters is paired, and keeping only a count makes McNemar's
    test impossible after the GPU time has been spent. See
    :mod:`dynquant.eval.compare`."""

    predictions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def above_chance(self) -> float:
        return self.accuracy - CHANCE

    def summary(self) -> str:
        return (
            f"{self.label:<28} {self.accuracy:6.2%}  "
            f"({self.correct}/{self.total} exact match, "
            f"{self.above_chance:+.2%} vs chance, {self.unparseable} unparseable)"
        )


def load_banking77(
    split: str = "test", *, cache_dir: str | None = None, seed: int = SHUFFLE_SEED
) -> list[Banking77Example]:
    """Load a Banking77 split, deterministically shuffled. Requires ``datasets``.

    The shuffle is not a nicety. Upstream ships both splits in *label order* -- the test
    split is 77 contiguous blocks of 40 -- so any prefix of it is a single-intent sample.
    Evaluating with ``--limit 32`` on the raw order scores the model on intent 11 and
    nothing else, which reads as 3% accuracy against a real figure ten times that, and
    reads as a destroyed model when the arm is a quantized one. Measured, not supposed:
    that is exactly what the first smoke run reported.

    A fixed seed rather than no shuffle, because the paired design needs the same
    problems in the same order at every measurement point -- the per-problem ``hits``
    vectors are compared position by position, and an order that varied between arms
    would silently pair each problem with a different one.
    """
    from datasets import load_dataset

    raw = load_dataset(_SOURCE, split=split, cache_dir=cache_dir)
    names = getattr(raw.features["label"], "names", None)
    if names is not None and len(names) != N_INTENTS:
        raise ValueError(
            f"{_SOURCE} split {split!r} has {len(names)} classes, not {N_INTENTS}. "
            f"INTENTS is indexed by the gold label, so a taxonomy that has moved "
            f"would score every answer against the wrong intent."
        )
    return deterministic_order(
        [Banking77Example(text=row["text"].strip(), answer=str(row["label"])) for row in raw],
        seed=seed,
    )


def deterministic_order(
    examples: list[Banking77Example], *, seed: int = SHUFFLE_SEED
) -> list[Banking77Example]:
    """Return ``examples`` shuffled by a fixed seed. Separate from the loader so the
    property that matters -- a prefix is class-diverse, and two calls agree -- can be
    tested without a network round trip."""
    import random

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def _question(example: Banking77Example) -> str:
    return f"Customer query: {example.text}\nIntent:"


def format_training_text(example: Banking77Example) -> tuple[str, str]:
    """Return ``(prompt, completion)`` for supervised fine-tuning.

    Split rather than concatenated so the trainer masks the loss to the completion.
    Here that is not a refinement but the whole run: the prompt is ~615 tokens of
    intent taxonomy and the completion is one index. Train on the full sequence and
    the answer carries well under 1% of the gradient -- the run is spent learning
    to recite a list the model is shown anyway, and the evaluation comes back flat
    under a training loss that fell the whole way.
    """
    return f"{HEADER}\n\n{_question(example)}", f" {example.answer}"


def build_prompt(example: Banking77Example, shots: Sequence[Banking77Example]) -> str:
    """The few-shot prompt, in exactly the format the fine-tune trains on."""
    blocks = [_question(shot) + f" {shot.answer}" for shot in shots]
    blocks.append(_question(example))
    return f"{HEADER}\n\n" + "\n\n".join(blocks)


def extract_answer(text: str) -> str | None:
    """Pull the chosen intent index out of a generation.

    Prefers an integer at the very start, which is what the format asks for, and
    falls back to the first in-range integer anywhere, because a model that answers
    ``"intent 41"`` has chosen and scoring that as a failure would charge a
    formatting wobble to quantization. Out-of-range integers are skipped rather
    than accepted, so ``"81"`` is unparseable and not merely wrong.

    A reasoning trace is cut first (:func:`~dynquant.eval.harness.strip_reasoning`), a
    no-op for a model that does not emit one. It matters here because this extractor
    falls back to scanning the whole text: inside a trace the candidate it would find is
    one the model may still have been arguing with.
    """
    text = strip_reasoning(text)
    leading = _LEADING_INDEX.match(text.strip())
    if leading and int(leading.group(1)) < N_INTENTS:
        return leading.group(1)
    for match in _ANY_INDEX.finditer(text):
        if int(match.group(0)) < N_INTENTS:
            return match.group(0)
    return None


def evaluate_banking77(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Banking77Example],
    *,
    label: str,
    shots: Sequence[Banking77Example] = (),
    config: EvalConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
    keep_predictions: int = 0,
) -> Banking77Result:
    """Score a model on Banking77 by exact match on the intent index.

    Args:
        shots: Few-shot exemplars, from the *train* split, and the same list at
            every measurement point. They must also be excluded from the
            fine-tuning set -- an exemplar shown in every prompt and trained on is
            an answer handed to the model for free.
        keep_predictions: Retain this many (generation, gold) pairs. A score alone
            cannot distinguish a model picking the wrong intent from one that has
            stopped emitting indices.
    """
    config = config or EvalConfig(
        max_new_tokens=6, stop_sequences=(FEWSHOT_STOP,), max_prompt_tokens=1536
    )
    if FEWSHOT_STOP not in config.stop_sequences:
        # `replace`, not a field-by-field rebuild: a rebuild silently reverts any
        # field it forgets to list, and the fields it is most likely to forget are
        # the ones added after it was written.
        config = replace(config, stop_sequences=(*config.stop_sequences, FEWSHOT_STOP))

    subset = list(examples[: config.limit] if config.limit else examples)
    prompts = [build_prompt(example, shots) for example in subset]
    generations = generate_batched(model, tokenizer, prompts, config, progress=progress)

    result = Banking77Result(label=label, correct=0, total=len(subset), unparseable=0)
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
                    "predicted_intent": INTENTS[int(predicted)] if predicted else None,
                    "gold": example.answer,
                    "gold_intent": INTENTS[int(example.answer)],
                    "correct": hit,
                }
            )

    _log.info("%s", result.summary())
    return result
