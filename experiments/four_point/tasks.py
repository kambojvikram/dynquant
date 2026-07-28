"""The tasks this experiment has been run on, and everything that differs.

Why there is more than one
--------------------------
The experiment ran first on GSM8K, the paper's own task, and the fine-tune moved it
-0.99 points (p=0.48). Not a regression -- a tie. But a tie is useless here: the whole
design reads quantization damage against a fine-tuning gain, and there was no gain to
read it against. Qwen3.5-2B-Base already scores 66.11% on GSM8K, because grade-school
arithmetic is saturated in modern pre-training.

So every task since has been chosen by measuring base-model headroom on candidates
*before* committing a fine-tune to any of them, against the model that would actually
be tuned. :mod:`screen_headroom` holds the Qwen screen and the reasoning behind it:

===================  ==============  =======  ====================  ==========
candidate            base few-shot   chance   supervised reference  headroom
===================  ==============  =======  ====================  ==========
CaseHOLD             34.3%           20%      ~69%                  ~35 pts
sql-create-context   54.0%            0%      ~80%                  ~26 pts
MedMCQA              47.7%           25%      ~45-50%               ~0 pts
GSM8K (control)      66.1%            0%      65.1% measured        ~0 pts
===================  ==============  =======  ====================  ==========

MedMCQA is GSM8K again. CaseHOLD has the most room, and its format is a single digit
already demonstrated by the few-shot prefix, so a gain there cannot be the model
learning to answer in the right shape.

Banking77 came out of the same screen re-run against Mistral-7B-Instruct-v0.3, since
headroom is a property of the *pair* and not of the dataset; the table is in
:mod:`dynquant.eval.banking77`. It won on ~52 points of headroom, and it is the most
sensitive of the three to quantization damage: a 77-way choice bottoms out at 1.3%, so
there is no cushion for a small regression to hide under.

GSM8K stays in this file rather than being deleted. Its records are on disk, the
GSM8K arm of ``RESULTS.md`` is a real finding about the task, and a reader who wants
to check that the CaseHOLD numbers came out of the same pipeline needs to be able to
run both from the same scripts.

Why a registry rather than one directory per run
------------------------------------------------
Of the eight stage scripts, six are entirely task-agnostic -- allocation, quantization,
the results table and the two inspectors never mention a dataset, and after ``DQ_MODEL``
none of them mentions a model either. Copying the directory would have duplicated ~700
lines and let the copies drift, and drift between the runs' *shared* machinery is
exactly what would make their tables incomparable. Everything that genuinely differs is
in this file: how to load, how to build a training row, how to score.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

TASK_ENV = "DQ_TASK"
DEFAULT_TASK = "casehold"


@dataclass(frozen=True)
class Task:
    """Everything the stage scripts need to know about a dataset."""

    key: str
    n_shots: int
    """Held fixed across all six measurement points, including the fine-tuned one.
    Dropping the prefix for the tuned model would confound "got better at the task"
    with "learned this output format", and the signal map is meant to be measuring
    the former."""

    chance: float
    """Accuracy from guessing, as a fraction. Reported beside every number: a 5-way
    choice bottoms out at 0.20, so a destroyed model lands near 20% rather than near
    zero, and 34% means something quite different against 0.20 than against 0.00."""

    max_new_tokens: int
    max_prompt_tokens: int
    train_max_len: int
    """Training sequence cap. Sequences longer than this are cut from the *left*,
    keeping the trailing cue -- see :meth:`training_row`."""

    eval_batch_size: int
    load: Callable[[str], list[Any]]
    evaluate: Callable[..., Any]
    format_training_text: Callable[[Any], tuple[str, str]]
    fewshot_stop: str
    """What separates one exemplar from the next, and therefore where generation stops.
    Taken from the task's own eval module rather than written out here, so the string the
    decoder stops on cannot drift from the string the prompt builder joins with."""

    train_split: str = "train"
    test_split: str = "test"

    def stop_sequence(self) -> str:
        return self.fewshot_stop

    def training_row(self, example: Any, tokenizer: Any) -> dict[str, list[int]] | None:
        """Tokenize one example into ``input_ids``/``labels``, or ``None`` to drop it.

        The prompt is masked out of the loss in every task, but how much that matters
        varies enormously. A GSM8K completion is a whole worked solution, so the
        answer tokens are a healthy fraction of the sequence either way. A CaseHOLD
        completion is *one digit* after ~400 tokens of case-law prose: train on the
        concatenation and the answer carries under 1% of the gradient, the run teaches
        the model to write appellate prose, and the evaluation comes back flat with a
        perfectly healthy-looking training loss the whole way down.
        """
        prompt, completion = self.format_training_text(example)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        completion_ids = [*completion_ids, tokenizer.eos_token_id]

        room = self.train_max_len - len(completion_ids)
        if room <= 0:
            return None
        if len(prompt_ids) > room:
            prompt_ids = self._truncate_prompt(prompt_ids, room)
            if prompt_ids is None:
                return None

        return {
            "input_ids": prompt_ids + completion_ids,
            "labels": [-100] * len(prompt_ids) + completion_ids,
        }

    def _truncate_prompt(self, prompt_ids: list[int], room: int) -> list[int] | None:
        raise NotImplementedError


@dataclass(frozen=True)
class DropTooLong(Task):
    """Drop overlong examples instead of cutting them.

    Right when the cap is generous relative to the data, so dropping costs a handful of
    examples, and when there is no expendable end to cut. GSM8K: 512 tokens against a
    measured max of ~330, and cutting a worked solution in half would train the model to
    stop mid-reasoning. Banking77: every row is one sentence appended to a fixed header,
    so nothing should exceed the cap at all -- and if the header alone did, cutting
    either end would remove either the numbering the answer indexes into or the query
    being classified. The printed drop count is then the alarm.
    """

    def _truncate_prompt(self, prompt_ids: list[int], room: int) -> list[int] | None:
        return None


@dataclass(frozen=True)
class TruncateLeft(Task):
    """Cut overlong prompts at the front.

    Right for CaseHOLD, where a large fraction of examples exceed any reasonable cap
    and dropping them would silently train on an easy subset. The five holdings and
    the ``Answer:`` cue are at the *end* of the prompt and are the part the model has
    to act on; the citing text at the front is the expendable part. Cutting from the
    right would remove the options and the cue, and train the model to answer a
    question it was not shown.
    """

    def _truncate_prompt(self, prompt_ids: list[int], room: int) -> list[int] | None:
        return prompt_ids[-room:]


def _gsm8k() -> Task:
    from dynquant.eval.gsm8k import FEWSHOT_STOP, evaluate_gsm8k, format_training_text, load_gsm8k

    return DropTooLong(
        key="gsm8k",
        n_shots=5,
        chance=0.0,
        max_new_tokens=320,
        max_prompt_tokens=2048,
        train_max_len=512,
        eval_batch_size=48,
        load=load_gsm8k,
        evaluate=evaluate_gsm8k,
        format_training_text=format_training_text,
        fewshot_stop=FEWSHOT_STOP,
    )


def _casehold() -> Task:
    from dynquant.eval.casehold import (
        FEWSHOT_STOP,
        evaluate_casehold,
        format_training_text,
        load_casehold,
    )

    return TruncateLeft(
        key="casehold",
        # Two, not five. A CaseHOLD prompt is ~400 tokens against GSM8K's ~90, so
        # five shots would put the prefix past 2k tokens for no measured benefit --
        # the probe used two and cleared chance by 14 points.
        n_shots=2,
        chance=0.2,
        # The answer is one digit. Eight tokens is room for " 3" plus whatever
        # punctuation the model adds before the stop sequence, and a tight budget
        # keeps the six-arm run to minutes.
        max_new_tokens=8,
        # Larger than GSM8K's because the prompt is the long part here, not the
        # answer: two exemplars plus the question run ~1.2k tokens and the tail must
        # survive. The harness counts and warns about anything it still has to cut.
        max_prompt_tokens=3072,
        train_max_len=640,
        eval_batch_size=32,
        load=load_casehold,
        evaluate=evaluate_casehold,
        format_training_text=format_training_text,
        fewshot_stop=FEWSHOT_STOP,
    )


def _banking77() -> Task:
    from dynquant.eval.banking77 import (
        FEWSHOT_STOP,
        evaluate_banking77,
        format_training_text,
        load_banking77,
    )

    return DropTooLong(
        key="banking77",
        # Four. The prompt is dominated by the 77-line intent header rather than by the
        # exemplars, which are one sentence each, so shots are nearly free here in a way
        # they are not on CaseHOLD.
        n_shots=4,
        # 1.3%. The lowest floor of the three tasks and the reason this one is the
        # sharpest test of quantization damage: there is nothing for a small regression
        # to hide under, and a destroyed model cannot guess its way back to a
        # plausible-looking score.
        chance=1.0 / 77,
        # Room for a two-digit index plus whatever punctuation precedes the stop.
        max_new_tokens=6,
        # The header is ~615 tokens under a Qwen tokenizer and more under Mistral's;
        # 1536 leaves headroom over header + four one-sentence shots + the query so the
        # harness never has to cut. Front-truncation would remove the numbering the
        # answer indexes into and score the model on a list it never saw.
        max_prompt_tokens=1536,
        # Training rows are zero-shot: header + one query + one index. The cap is well
        # clear of that, so a nonzero drop count means the header itself did not fit --
        # which is a configuration error, not a data property, and should be loud.
        train_max_len=1024,
        eval_batch_size=32,
        load=load_banking77,
        evaluate=evaluate_banking77,
        format_training_text=format_training_text,
        fewshot_stop=FEWSHOT_STOP,
    )


_BUILDERS = {"gsm8k": _gsm8k, "casehold": _casehold, "banking77": _banking77}


def get_task(key: str | None = None) -> Task:
    """Resolve the task from an explicit key, then ``DQ_TASK``, then the default."""
    key = key or os.environ.get(TASK_ENV, DEFAULT_TASK)
    builder = _BUILDERS.get(key)
    if builder is None:
        raise SystemExit(f"unknown task {key!r}; choose from {', '.join(sorted(_BUILDERS))}")
    return builder()


def split_task(task: Task, seed: int) -> tuple[list[Any], list[Any], list[Any]]:
    """Return ``(train, test, shots)``.

    The exemplars come from the *train* split and are chosen by a fixed seed, so the
    prompt is byte-identical across stages and across restarts. They are also removed
    from the training set: leaving them in lets the tuned model memorise the very
    examples sitting in its own prompt, which flatters the after-fine-tune number for
    a reason that has nothing to do with quantization.
    """
    train = task.load(task.train_split)
    test = task.load(task.test_split)

    rng = random.Random(seed)
    shot_idx = sorted(rng.sample(range(len(train)), task.n_shots))
    shots = [train[i] for i in shot_idx]
    held_out = set(shot_idx)
    return [ex for i, ex in enumerate(train) if i not in held_out], test, shots
