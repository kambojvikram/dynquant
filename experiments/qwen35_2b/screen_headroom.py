"""Stage 0: how much room does a fine-tune have, before spending one to find out.

This script exists because the first pass of the four-point experiment did not run it.
GSM8K was chosen because it is the paper's own task; it turned out that
``Qwen/Qwen3.5-2B-Base`` already scores 66.11% at 5-shot, and two epochs of SFT on
GSM8K's own train split moved that -0.99 points (p=0.48). A base model that has
effectively seen the task in pre-training cannot be improved by seeing it again -- and
finding that out cost a full six-arm run, because the flat line only appears at the end.

The screen is **two-sided**, and that is the whole point:

1. the base model must score far *above* chance -- otherwise the task is simply beyond a
   2B model and the fine-tune has nothing to build on; and
2. the base model must score far *below* what supervised training is known to reach --
   otherwise there is nothing left for the fine-tune to teach.

Failing either side produces an identical flat line in the final table, so the diagnosis
is ambiguous afterwards and cheap beforehand: a few hundred examples of base-model eval.
GSM8K fails (2). It is kept in the candidate list as the control -- a screen that does
not reject the task already known to fail is not a screen.

One further requirement, not mechanised because it is a judgement about the dataset: the
answer format must be conveyable by the few-shot prefix (a digit, a letter, one SQL
line). Few-shot then supplies the format, so any gain the fine-tune produces is task
skill rather than format compliance. "The tuned model finally learned to answer in the
right shape" is the cheap way to manufacture a gain and it would not be a real one.

What was measured, and what came of it
--------------------------------------
====================  =============  =======  =============  ==============  =======
candidate             base few-shot  chance   supervised     to reference    verdict
====================  =============  =======  =============  ==============  =======
CaseHOLD              34.3%          20%      ~69%           ~35 pts         pass
sql-create-context    54.0%           0%      ~80%           ~26 pts         pass
MedMCQA               47.7%          25%      ~45-50%        ~0 pts          reject
GSM8K (control)       66.1%           0%      65.1% measured  -1 pt          reject
====================  =============  =======  =============  ==============  =======

CaseHOLD was taken, and the screen held up: the full-split base measurement came in at
35.13% on n=5314, and a 500-step SFT probe reached 75.67% -- a +41 point move on the task
where GSM8K moved -1.

Two honesty notes about this table. It was produced by the scratch version of this
script, which drew its own few-shot exemplars; the committed version routes registered
candidates through ``tasks.split_task`` so the screen's exemplars are the *run's*
exemplars, which is strictly better and means a re-run lands within sampling error of the
numbers above rather than on them. And a screen number must never be reported in
``RESULTS.md``: it is a few hundred examples on a subsample, and the only figures in that
table are the ones the shared ``common.run_eval`` produced on the full split.

Why the candidates are not all ``tasks.py`` entries
---------------------------------------------------
Two of the four here will never become tasks. Requiring a full :class:`tasks.Task` --
loader, training-row builder with its own truncation rule, stop sequence, scorer -- just
to *consider* a dataset would make the cheap step-zero screen the expensive one, which is
how it ends up skipped again. So candidates come in two kinds:

``Registered``
    Already in ``tasks.py``. Measured by that task's own ``evaluate``, with the decode
    settings from ``common.eval_config`` -- the same three things the results table uses,
    so the number is directly comparable to it.
``AdHoc``
    A prompt builder and a scorer, inline. Not comparable to the table, only to the other
    candidates in this run.

Both kinds decode through ``dynquant.eval.harness.generate_batched``, so left padding,
left truncation, stop sequences and batching are shared. That matters more than it
sounds: the scratch version of this script set ``padding_side`` but not
``truncation_side``, so its longest CaseHOLD prompts were cut from the *right* -- which
removes the five holdings and the ``Answer:`` cue, and scores the model on a question it
was never shown.

Usage::

    export PYTHONPATH=/path/to/packages/dynquant-core/src
    python screen_headroom.py                              # all four, n=300 each
    python screen_headroom.py --datasets casehold medmcqa --n 500
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from common import MODEL_ID, SEED, eval_config, load_model, load_tokenizer, progress

from dynquant.eval.harness import EvalConfig, generate_batched

if TYPE_CHECKING:
    from collections.abc import Callable

ABOVE_CHANCE_MIN = 10.0
"""Points above the guessing floor the base model must clear. Below this the task is
plausibly outside a 2B model's reach, and a fine-tune that moves nothing is telling you
about model capacity rather than about headroom."""

TO_REFERENCE_MIN = 15.0
"""Points the base model must sit *below* the supervised reference. This is the side
GSM8K failed, and the side that is invisible if you only check that the task is hard."""


@dataclass(frozen=True)
class Candidate:
    """A dataset being considered, and what is already known about its ceiling."""

    key: str
    chance: float
    """Accuracy from guessing, in percent. The floor a collapsed model returns to, and
    the baseline every "gain" here is measured against."""

    supervised_percent: float
    """What supervised training on this task is reported to reach. A number rather than
    a prose citation so the verdict below is mechanical: the moment this is a sentence,
    the second half of the screen becomes a judgement call and gets skipped."""

    supervised_source: str

    def measure(self, model: Any, tokenizer: Any, n: int) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class Registered(Candidate):
    """A candidate already in ``tasks.py``, measured through the shared harness."""

    def measure(self, model: Any, tokenizer: Any, n: int) -> dict[str, Any]:
        from tasks import get_task, split_task

        task = get_task(self.key)
        # A chance floor recorded in two places is a chance floor that will disagree,
        # and disagreement here silently rescales "above chance" for one candidate only.
        if abs(task.chance * 100.0 - self.chance) > 1e-9:
            raise SystemExit(
                f"{self.key}: chance floor is {self.chance}% here and "
                f"{task.chance * 100.0}% in tasks.py"
            )
        _, test, shots = split_task(task, SEED)
        subset = random.Random(SEED).sample(test, min(n, len(test)))
        result = task.evaluate(
            model,
            tokenizer,
            subset,
            label=f"screen:{self.key}",
            shots=shots,
            # limit=None: the subset was already drawn, and letting the config cut it
            # again would take the first N of a sample that is no longer in dataset
            # order.
            config=eval_config(None, task=task),
            progress=progress(self.key),
        )
        return {
            "harness": "shared",
            "shots": task.n_shots,
            "n": result.total,
            "correct": result.correct,
            "unparseable": result.unparseable,
            "accuracy": result.accuracy,
        }


@dataclass(frozen=True)
class AdHoc(Candidate):
    """A candidate with an inline prompt builder and scorer.

    For datasets being considered but not adopted. The numbers are comparable to the
    other candidates in this run and to nothing else.
    """

    shots: int
    max_new_tokens: int
    build: Callable[[], tuple[list[dict[str, str]], list[dict[str, str]]]]
    score: Callable[[str, str], bool]

    def measure(self, model: Any, tokenizer: Any, n: int) -> dict[str, Any]:
        rng = random.Random(SEED)
        train, test = self.build()
        shot_idx = rng.sample(range(len(train)), self.shots)
        prefix = "\n\n".join(train[i]["prompt"] + train[i]["answer"] for i in shot_idx) + "\n\n"
        subset = rng.sample(test, min(n, len(test)))

        config = EvalConfig(
            max_new_tokens=self.max_new_tokens,
            batch_size=8,
            stop_sequences=("\n\n",),
            max_prompt_tokens=3072,
        )
        generations = generate_batched(
            model,
            tokenizer,
            [prefix + row["prompt"] for row in subset],
            config,
            progress=progress(self.key),
        )
        hits = [
            self.score(text, row["answer"]) for text, row in zip(generations, subset, strict=True)
        ]
        return {
            "harness": "adhoc",
            "shots": self.shots,
            "n": len(hits),
            "correct": sum(hits),
            "unparseable": None,
            "accuracy": 100.0 * sum(hits) / len(hits),
            "prefix_tokens": len(tokenizer(prefix)["input_ids"]),
        }


def _fmt_medmcqa(row: dict[str, Any]) -> dict[str, str]:
    letters = "ABCD"
    options = "\n".join(
        f"{letters[i]}. {row[key]}" for i, key in enumerate(("opa", "opb", "opc", "opd"))
    )
    return {
        "prompt": f"Question: {row['question'].strip()}\n{options}\nAnswer:",
        "answer": f" {letters[int(row['cop'])]}",
    }


def _medmcqa() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    from datasets import load_dataset

    train = load_dataset("openlifescienceai/medmcqa", split="train")
    # `test` ships without labels; `validation` is the split with public answers.
    test = load_dataset("openlifescienceai/medmcqa", split="validation")
    return [_fmt_medmcqa(row) for row in train], [_fmt_medmcqa(row) for row in test]


def _fmt_sql(row: dict[str, Any]) -> dict[str, str]:
    return {
        "prompt": f"Schema: {row['context'].strip()}\nQuestion: {row['question'].strip()}\nSQL:",
        "answer": f" {row['answer'].strip()}",
    }


def _sql() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    from datasets import load_dataset

    # Ships one split only, so the held-out set is carved here by a fixed index rather
    # than a random sample -- the same rows are then held out on every future run.
    rows = [_fmt_sql(row) for row in load_dataset("b-mc2/sql-create-context", split="train")]
    return rows[:-4000], rows[-4000:]


def _norm_sql(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().rstrip(";")).lower()


def _first_token_match(generated: str, gold: str) -> bool:
    """Exact match on the first non-space token -- the whole answer for an MCQ."""
    got = generated.strip().split()
    return bool(got) and got[0].rstrip(".") == gold.strip()


CANDIDATES: dict[str, Candidate] = {
    "casehold": Registered(
        key="casehold",
        chance=20.0,
        supervised_percent=69.0,
        supervised_source="fine-tuned BERT-base, Zheng et al. 2021",
    ),
    "sql": AdHoc(
        key="sql",
        chance=0.0,
        supervised_percent=80.0,
        supervised_source="reported exact match for SFT on b-mc2/sql-create-context",
        shots=3,
        max_new_tokens=64,
        build=_sql,
        score=lambda got, gold: _norm_sql(got.split("\n")[0]) == _norm_sql(gold),
    ),
    "medmcqa": AdHoc(
        key="medmcqa",
        chance=25.0,
        supervised_percent=47.5,
        supervised_source="fine-tuned small LMs, ~45-50%",
        shots=5,
        max_new_tokens=4,
        build=_medmcqa,
        score=_first_token_match,
    ),
    # The control. Its reference is the only measured one in this table, and measuring
    # it is exactly what this script exists to avoid having to do again.
    "gsm8k": Registered(
        key="gsm8k",
        chance=0.0,
        supervised_percent=65.1,
        supervised_source="measured: 2 epochs SFT on GSM8K train, this experiment",
    ),
}


def verdict(accuracy: float, cand: Candidate) -> dict[str, Any]:
    """Apply both sides of the screen and say which one failed."""
    above_chance = accuracy - cand.chance
    to_reference = cand.supervised_percent - accuracy
    reasons = []
    if above_chance < ABOVE_CHANCE_MIN:
        reasons.append(f"only {above_chance:+.1f} pts above chance (need {ABOVE_CHANCE_MIN:.0f})")
    if to_reference < TO_REFERENCE_MIN:
        reasons.append(
            f"only {to_reference:+.1f} pts below the supervised reference "
            f"(need {TO_REFERENCE_MIN:.0f})"
        )
    return {
        "above_chance": above_chance,
        "to_reference": to_reference,
        "verdict": "reject" if reasons else "pass",
        "why": "; ".join(reasons) or "room on both sides",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(CANDIDATES))
    parser.add_argument(
        "--n", type=int, default=300, help="examples per candidate; a screen, not a result"
    )
    parser.add_argument("--out", default="screen_headroom.json")
    args = parser.parse_args()

    unknown = [key for key in args.datasets if key not in CANDIDATES]
    if unknown:
        raise SystemExit(f"unknown candidate(s) {unknown}; choose from {sorted(CANDIDATES)}")

    tokenizer = load_tokenizer()
    model = load_model()

    results: list[dict[str, Any]] = []
    for key in args.datasets:
        cand = CANDIDATES[key]
        print(f"\n=== {key} ===", flush=True)
        started = time.time()
        # One candidate failing to load must not cost the others their measurement: the
        # model is already resident and loading it is the expensive part. The exception
        # is reported in the record, not swallowed.
        try:
            measured = cand.measure(model, tokenizer, args.n)
        except Exception as exc:  # noqa: BLE001 - recorded per candidate, then reported
            results.append({"dataset": key, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        record = {
            "dataset": key,
            **measured,
            "chance": cand.chance,
            "supervised_percent": cand.supervised_percent,
            "supervised_source": cand.supervised_source,
            **verdict(measured["accuracy"], cand),
            "seconds": round(time.time() - started, 1),
        }
        results.append(record)
        print(json.dumps(record), flush=True)

    print()
    header = (
        f"{'candidate':14s} {'harness':8s} {'n':>5s} {'base':>7s} "
        f"{'chance':>7s} {'ref':>7s} {'verdict':>8s}  why"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        if "error" in row:
            print(
                f"{row['dataset']:14s} {'--':8s} {'':>5s} {'':>7s} {'':>7s} {'':>7s} "
                f"{'error':>8s}  {row['error']}"
            )
            continue
        print(
            f"{row['dataset']:14s} {row['harness']:8s} {row['n']:5d} "
            f"{row['accuracy']:6.2f}% {row['chance']:6.1f}% {row['supervised_percent']:6.1f}% "
            f"{row['verdict']:>8s}  {row['why']}"
        )

    passing = [row["dataset"] for row in results if row.get("verdict") == "pass"]
    print(f"\npasses both sides: {', '.join(passing) if passing else 'none'}")

    Path(args.out).write_text(
        json.dumps({"model": MODEL_ID, "n_per_candidate": args.n, "candidates": results}, indent=2),
        encoding="utf-8",
    )
    print(f"-> wrote {args.out}")
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
