"""IFEval: instruction following, scored by program rather than by judge.

Why this task leads the phase-3 set. Every other candidate for measuring what an SFT
run bought either needs a judge model (which introduces a second set of weights whose
own behaviour drifts between runs) or reduces to multiple choice (which is the
classification framing this campaign is deliberately not repeating, and which is
insensitive to the thing SFT changes). IFEval is neither: each prompt carries a list of
machine-checkable constraints -- "at least 3 bullet points", "no commas", "wrap your
whole answer in double quotes" -- and a Python function decides whether each was met.
The scorer has no parameters, so it cannot drift, and it costs nothing to run.

It is also the right *shape* of task for this experiment. Instruction following is
learned during SFT and is fragile under quantization in a way arithmetic is not: a
model that has lost a little precision keeps answering the question but stops counting
its own bullet points. That failure is invisible to perplexity and nearly invisible to
GSM8K, and it is exactly what a serving user notices first.

The four numbers
----------------
IFEval defines four metrics and this module reports all four, because reporting one is
how the benchmark gets inflated -- loose instruction-level accuracy runs some ten points
above strict prompt-level accuracy on the same generations, and papers quoting a single
"IFEval" number rarely say which.

* **prompt-level strict** -- every constraint on a prompt satisfied by the raw output.
  The headline, and the vector stored in :attr:`IfevalResult.hits`.
* **prompt-level loose** -- the same, allowing the eight cosmetic rewrites below.
* **instruction-level strict / loose** -- the same two, counted per constraint rather
  than per prompt. More sensitive, since a prompt with four constraints is scored on
  all four rather than collapsing to one bit.

Strict and loose differ only in whether the response is allowed to be tidied first: the
loose pass retries each check against the response with markdown asterisks stripped,
with its first line dropped, with its last line dropped, and the four combinations of
those. This forgives the two things chat models do unbidden -- opening with "Sure, here
is..." and bolding things -- neither of which is the constraint being tested.

Pairing
-------
:attr:`IfevalResult.hits` is prompt-level strict, one boolean per prompt in dataset
order, so any two arms are an exact McNemar test through :mod:`dynquant.eval.compare`.
:meth:`IfevalResult.flat_instruction_hits` gives the same thing at instruction
granularity -- roughly 1.5x the items, and the narrower test of the two when a change
moves a few constraints rather than a few prompts.

What can make two runs incomparable
-----------------------------------
Three things, all of them recorded rather than assumed:

* the **scorer fingerprint** (:attr:`IfevalResult.scorer`), since sentence splitting
  falls back to a regex when NLTK is absent -- see :mod:`._ifeval_instructions`;
* the **prompt style** (:attr:`IfevalResult.prompt_style`), since a model with a chat
  template is prompted through it and one without is not, and IFEval scores fall by a
  wide margin without a template;
* **dropped prompts** (:attr:`IfevalResult.dropped`), if ``langdetect`` is missing and
  the caller asked to proceed anyway.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from dynquant._logging import get_logger
from dynquant.errors import DynQuantError

from ._ifeval_instructions import (
    Checker,
    build_checker,
    missing_capabilities,
    requirements_for,
    scorer_fingerprint,
)
from .harness import EvalConfig, chat_prompt_style, generate_batched

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "IfevalExample",
    "IfevalResult",
    "build_prompt",
    "evaluate_ifeval",
    "load_ifeval",
    "loose_variants",
]

_log = get_logger(__name__)

DEFAULT_CONFIG = EvalConfig(
    max_new_tokens=1024,
    batch_size=16,
    max_prompt_tokens=2048,
    add_special_tokens=False,
)
"""Decoding defaults for this task.

``max_new_tokens=1024`` because the constraints are frequently *length* constraints --
"at least 400 words", "write 5 paragraphs" -- and a cap that truncates the answer scores
the cap rather than the model. ``add_special_tokens=False`` because
:func:`build_prompt` emits a chat template that already carries its own BOS.

No stop sequences: a chat-templated prompt ends at the model's own EOS, and a stop
string would risk cutting a response short at a sequence the model was *asked* to emit.
"""


@dataclass(frozen=True, slots=True)
class IfevalExample:
    """One prompt and the constraints it must satisfy."""

    key: int
    prompt: str
    instruction_ids: tuple[str, ...]
    kwargs: tuple[dict[str, Any], ...]
    """Per-instruction arguments, positionally aligned with :attr:`instruction_ids`.
    The published dataset stores every possible argument name on every row with
    ``None`` for the ones that do not apply; :func:`load_ifeval` strips those, so a
    missing argument here is a real missing argument."""


@dataclass(slots=True)
class IfevalResult:
    """One measurement point, carrying all four official metrics."""

    label: str
    total: int
    prompt_strict: int
    prompt_loose: int
    instruction_total: int
    instruction_strict: int
    instruction_loose: int
    empty: int
    """Generations with no non-whitespace content. Reported separately because they
    mean something different from a violated constraint: a model that emits nothing has
    stopped working, and at 2 bits that is the expected failure, not bad formatting."""

    scorer: str
    prompt_style: Literal["chat-template", "raw"]
    hits: list[bool] = field(default_factory=list)
    """Prompt-level strict correctness, one per prompt in dataset order. The paired
    vector; see :mod:`dynquant.eval.compare` for why it is not optional."""

    loose_hits: list[bool] = field(default_factory=list)
    instruction_hits: list[list[bool]] = field(default_factory=list)
    instruction_loose_hits: list[list[bool]] = field(default_factory=list)
    instruction_ids: list[tuple[str, ...]] = field(default_factory=list)
    keys: list[int] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)
    """Keys of prompts excluded because their constraints could not be scored. Empty
    in any run that should be reported; non-empty runs are comparable only to other
    runs with the identical list."""

    predictions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Prompt-level strict, the headline. Matches :attr:`hits`."""
        return self.prompt_strict / self.total if self.total else 0.0

    @property
    def prompt_loose_accuracy(self) -> float:
        return self.prompt_loose / self.total if self.total else 0.0

    @property
    def instruction_strict_accuracy(self) -> float:
        return self.instruction_strict / self.instruction_total if self.instruction_total else 0.0

    @property
    def instruction_loose_accuracy(self) -> float:
        return self.instruction_loose / self.instruction_total if self.instruction_total else 0.0

    def flat_instruction_hits(self, *, strict: bool = True) -> list[bool]:
        """Every constraint's verdict, flattened in prompt order.

        A paired vector at instruction granularity. Two arms produce aligned vectors
        because the constraint list is a property of the dataset, not of the model --
        but only if both were scored on the same prompts, which is what
        :attr:`dropped` exists to let you check.
        """
        source = self.instruction_hits if strict else self.instruction_loose_hits
        return [hit for row in source for hit in row]

    def summary(self) -> str:
        return (
            f"{self.label:<28} "
            f"prompt {self.accuracy:6.2%}/{self.prompt_loose_accuracy:6.2%}  "
            f"instr {self.instruction_strict_accuracy:6.2%}/"
            f"{self.instruction_loose_accuracy:6.2%}  "
            f"(strict/loose, {self.total} prompts, {self.instruction_total} instructions, "
            f"{self.empty} empty, {self.scorer}, {self.prompt_style})"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "total": self.total,
            "prompt_strict_accuracy": self.accuracy,
            "prompt_loose_accuracy": self.prompt_loose_accuracy,
            "instruction_strict_accuracy": self.instruction_strict_accuracy,
            "instruction_loose_accuracy": self.instruction_loose_accuracy,
            "instruction_total": self.instruction_total,
            "empty": self.empty,
            "scorer": self.scorer,
            "prompt_style": self.prompt_style,
            "dropped": list(self.dropped),
        }


def load_ifeval(split: str = "train", *, cache_dir: str | None = None) -> list[IfevalExample]:
    """Load IFEval. Requires ``datasets``.

    The published dataset has a single split named ``train`` despite being an
    evaluation set -- there is nothing to train on and nothing held out. The default
    reflects that rather than hiding it behind a rename.
    """
    from datasets import load_dataset

    raw = load_dataset("google/IFEval", split=split, cache_dir=cache_dir)
    return [
        IfevalExample(
            key=int(row["key"]),
            prompt=row["prompt"],
            instruction_ids=tuple(row["instruction_id_list"]),
            # Arrow pads every row to a union of all argument names, so an unused
            # argument arrives as an explicit None rather than as an absent key.
            kwargs=tuple(
                {name: value for name, value in entry.items() if value is not None}
                for entry in row["kwargs"]
            ),
        )
        for row in raw
    ]


def build_prompt(example: IfevalExample, tokenizer: Any) -> str:
    """Render one prompt, through the model's chat template where it has one.

    IFEval is zero-shot and its constraints are addressed to an assistant, so a model
    with a template must be prompted through it -- an instruct checkpoint given bare
    text continues the text rather than answering it, and scores near the floor for
    reasons that have nothing to do with quantization.

    A base checkpoint has no template and gets the raw prompt. That is a real
    difference in measurement, not a fallback to paper over, which is why
    :attr:`IfevalResult.prompt_style` records which happened.

    "Has one" is :func:`~dynquant.eval.harness.chat_prompt_style`, which asks rather
    than inspects: some instruct tokenizers apply a template without exposing one.
    """
    if chat_prompt_style(tokenizer) == "chat-template":
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": example.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return str(rendered)
    return example.prompt


def loose_variants(response: str) -> list[str]:
    """The eight rewrites the loose metric is allowed to try.

    Verbatim from the reference implementation, order included: a constraint counts as
    followed if *any* variant satisfies it, so the set matters and the order does not,
    but keeping the order makes a diff against upstream readable.

    All of them forgive presentation rather than content -- a stripped preamble, a
    dropped sign-off, markdown emphasis removed. None of them can turn a response that
    ignored a constraint into one that met it.
    """
    lines = response.split("\n")
    without_first = "\n".join(lines[1:]).strip()
    without_last = "\n".join(lines[:-1]).strip()
    without_both = "\n".join(lines[1:-1]).strip()
    return [
        response,
        response.replace("*", ""),
        without_first,
        without_last,
        without_both,
        without_first.replace("*", ""),
        without_last.replace("*", ""),
        without_both.replace("*", ""),
    ]


def evaluate_ifeval(
    model: Any,
    tokenizer: Any,
    examples: Sequence[IfevalExample],
    *,
    label: str,
    config: EvalConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
    keep_predictions: int = 0,
    on_unverifiable: Literal["raise", "drop"] = "raise",
) -> IfevalResult:
    """Score a model on IFEval, reporting all four official metrics.

    Args:
        on_unverifiable: What to do about prompts carrying a constraint this process
            cannot check -- in practice, the three language constraints when
            ``langdetect`` is not installed. ``"raise"`` refuses to produce a number.
            ``"drop"`` excludes those prompts and records their keys on the result;
            it exists so a smoke run is not blocked by an optional dependency, and a
            run that used it is comparable only against another run that dropped the
            same keys.
        keep_predictions: Retain this many generations with their per-constraint
            verdicts. A score says instruction following fell; only the generations
            say whether the model stopped counting bullets or stopped writing English.
    """
    config = config or DEFAULT_CONFIG
    # Probed once for the run rather than per example, and shared with `build_prompt`
    # below, so the style the result reports is by construction the style the prompts
    # were built in.
    prompt_style: Literal["chat-template", "raw"] = chat_prompt_style(tokenizer)
    if config.add_special_tokens and prompt_style == "chat-template":
        # Silently correctable, and correcting it silently is the wrong call: the
        # double-BOS it prevents costs a few points and leaves no trace, so a caller
        # who overrode the default deserves to know their override was overridden.
        _log.warning(
            "add_special_tokens=True with a chat template will prepend a second BOS; "
            "forcing it off for %s",
            label,
        )
        config = replace(config, add_special_tokens=False)

    subset = list(examples[: config.limit] if config.limit else examples)
    subset, dropped = _resolve_unverifiable(subset, on_unverifiable=on_unverifiable)

    # Every checker is built before a single token is generated. A malformed kwarg or
    # an uncompilable section spliter then costs two seconds; discovered while scoring,
    # it would cost the whole generation pass.
    checkers = [
        [build_checker(instruction_id, kwargs) for instruction_id, kwargs in _bind(example)]
        for example in subset
    ]

    prompts = [build_prompt(example, tokenizer) for example in subset]
    generations = generate_batched(model, tokenizer, prompts, config, progress=progress)

    result = IfevalResult(
        label=label,
        total=len(subset),
        prompt_strict=0,
        prompt_loose=0,
        instruction_total=0,
        instruction_strict=0,
        instruction_loose=0,
        empty=0,
        scorer=scorer_fingerprint(),
        prompt_style=prompt_style,
        dropped=dropped,
    )

    for index, (example, bound, text) in enumerate(zip(subset, checkers, generations, strict=True)):
        if not text.strip():
            result.empty += 1
        strict = [_follows_strict(checker, text) for checker in bound]
        loose = [_follows_loose(checker, text) for checker in bound]

        result.prompt_strict += int(all(strict))
        result.prompt_loose += int(all(loose))
        result.instruction_total += len(bound)
        result.instruction_strict += sum(strict)
        result.instruction_loose += sum(loose)

        result.hits.append(all(strict))
        result.loose_hits.append(all(loose))
        result.instruction_hits.append(strict)
        result.instruction_loose_hits.append(loose)
        result.instruction_ids.append(example.instruction_ids)
        result.keys.append(example.key)

        if index < keep_predictions:
            result.predictions.append(
                {
                    "key": example.key,
                    "prompt": example.prompt,
                    "generation": text.strip(),
                    "instructions": [
                        {"id": checker.instruction_id, "strict": is_strict, "loose": is_loose}
                        for checker, is_strict, is_loose in zip(bound, strict, loose, strict=True)
                    ],
                }
            )

    _log.info("%s", result.summary())
    return result


# --------------------------------------------------------------------------
# Scoring internals
# --------------------------------------------------------------------------


def _bind(example: IfevalExample) -> list[tuple[str, dict[str, Any]]]:
    """Pair each instruction id with its arguments, checking they line up.

    A length mismatch would silently shift every argument onto the wrong constraint --
    "at least 3 paragraphs" checked with the word-count arguments -- and produce a
    plausible score from nonsense, which is the failure mode this file is most exposed
    to.
    """
    if len(example.instruction_ids) != len(example.kwargs):
        raise DynQuantError(
            f"IFEval prompt {example.key} has {len(example.instruction_ids)} instructions "
            f"but {len(example.kwargs)} argument sets"
        )
    return list(zip(example.instruction_ids, example.kwargs, strict=True))


def _follows_strict(checker: Checker, response: str) -> bool:
    """An empty response follows nothing, matching the reference implementation."""
    return bool(response.strip()) and checker(response)


def _follows_loose(checker: Checker, response: str) -> bool:
    return any(variant.strip() and checker(variant) for variant in loose_variants(response))


def _resolve_unverifiable(
    subset: Sequence[IfevalExample],
    *,
    on_unverifiable: Literal["raise", "drop"],
) -> tuple[list[IfevalExample], list[int]]:
    """Split off prompts whose constraints this process cannot check."""
    missing = missing_capabilities(
        instruction_id for example in subset for instruction_id in example.instruction_ids
    )
    if not missing:
        return list(subset), []

    affected = [
        example
        for example in subset
        if any(requirements_for(i) & missing for i in example.instruction_ids)
    ]
    packages = ", ".join(sorted(missing))
    if on_unverifiable == "raise":
        raise DynQuantError(
            f"{len(affected)}/{len(subset)} IFEval prompts carry constraints that need "
            f"{packages}, which is not installed. Install it (`pip install {packages}`) "
            "or pass on_unverifiable='drop' to score the rest -- but a dropped run is "
            "only comparable against another run that dropped the same prompts."
        )

    dropped = {example.key for example in affected}
    _log.warning(
        "dropping %d/%d IFEval prompts that need %s; this result is comparable only "
        "against runs with the identical dropped set",
        len(dropped),
        len(subset),
        packages,
    )
    return [example for example in subset if example.key not in dropped], sorted(dropped)
