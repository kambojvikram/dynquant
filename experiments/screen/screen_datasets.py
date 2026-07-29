"""Headroom screen: which open dataset has room for a Mistral-7B fine-tune to move?

Step zero, before any GPU time is spent on training. The rule this enforces was
learned the expensive way: GSM8K cost a full six-arm run to discover that
Qwen3.5-2B-Base already scored 66% on it, so there was no fine-tuning gain for
quantization to be measured against and every arm was measuring noise.

What matters is not "is this benchmark well known" but the gap between where the
*base* model already sits and where supervised training on the task is known to
reach. Four candidates, none of them used in any previous DynQuant experiment
(CaseHOLD, GSM8K and MedMCQA are all spent), chosen to differ in kind so the
screen is not four samples of one thing:

  banking77   77-way fine-grained intent classification, 10k train rows. The
              distinctions are between things like "card_payment_fee_charged" and
              "extra_charge_on_statement" -- label semantics no pre-training
              corpus teaches. Chance is 1.3%.
  pubmedqa    3-way yes/no/maybe over a biomedical abstract. Medical is the domain
              where MedMCQA turned out saturated, so this is the control on that
              finding: a different task shape in the same domain.
  logiqa      4-way logical reading comprehension from civil-service exams. Hard
              for its size; the risk here is the opposite one, that supervised
              training cannot move it either.
  mnli        3-way natural language inference. The oldest and most saturated of
              the four; included precisely because if it screens as flat that is
              information about the screen working.

Prompts are plain few-shot completion, not the chat template, even though this is
an Instruct model. The fine-tune will train on exactly this format, so using one
format here and another there would mean the measured gain includes "learned the
new format" -- which is not the thing being measured. Decoding runs through the
same `generate_batched` the real evaluation uses, so a number from this screen and
a number from `dynquant eval` are comparable.

Running it
----------

    DQ_MODEL=mistralai/Mistral-7B-Instruct-v0.3 python screen_datasets.py 300
    DQ_MODEL=... python screen_datasets.py 300 logiqa,mnli    # re-screen a subset

The candidates above are the four that were screened for Mistral-7B-Instruct-v0.3;
`SPECS` and `CANDIDATES` are where a different shortlist goes. Nothing here is
specific to those four beyond those two tables.

A note on the numbers this produced
-----------------------------------
The table recorded in `dynquant.eval.banking77` was produced by the version of this
script that sliced `test[:LIMIT]` without shuffling, and Banking77's row of it was
wrong by 4.7 points as a result -- see the shuffle in `main` and the footnote on
that table. Re-running this script now will not reproduce that row.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dynquant.eval.harness import EvalConfig, generate_batched

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 300
ONLY = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
"""Re-run a subset without re-paying for the candidates that already screened.
Results merge into the existing file rather than replacing it."""

SEED = 0
"""Fixes the shot draw *and* the row sample. See the shuffle in :func:`main`."""

# Read from the environment for the same reason the six-stage harness does: headroom
# is a property of the model/dataset *pair*, so a screen table is only evidence about
# the model it was run against, and a screen script with a model baked into it is a
# screen script that gets copied and edited instead of re-run.
MODEL = os.environ.get("DQ_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
OUT = os.environ.get(
    "DQ_SCREEN_OUT",
    f"/workspace/runs/{re.sub(r'[^a-z0-9]+', '_', MODEL.rsplit('/', 1)[-1].lower())}_headroom.json",
)


# --------------------------------------------------------------------------- data


def _load(repo: str, config: str | None, split: str, mirrors: list[str]) -> Any:
    """Load a split, falling back to the Hub's auto-converted parquet mirror.

    `datasets` 5.x refuses to execute loading scripts, and several of these repos
    still ship one. The mirror is the same rows without a script; its path layout
    changed across conversion eras, hence a list rather than one URL.
    """
    from datasets import load_dataset

    try:
        return load_dataset(repo, config, split=split)
    except Exception as exc:  # noqa: BLE001 -- any failure means try the mirror
        print(f"  ({repo}: direct load failed, {type(exc).__name__}; trying parquet)")
    for url in mirrors:
        try:
            return load_dataset("parquet", data_files=url, split="train")
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f"could not load {repo}/{config}/{split}")


def _mirrors(repo: str, config: str, split: str) -> list[str]:
    base = f"https://huggingface.co/datasets/{repo}/resolve/refs%2Fconvert%2Fparquet"
    name = repo.split("/")[-1]
    return [
        f"{base}/{config}/{split}/0000.parquet",
        f"{base}/{config}/{name}-{split}.parquet",
        f"{base}/{config}/partial-{split}/0000.parquet",
    ]


@dataclass
class Candidate:
    key: str
    n_choices: int
    shots: int
    supervised: str
    """Published/expected accuracy after supervised training on this task, for a
    comparably sized model. The screen is only meaningful against this number --
    a low base score with an equally low supervised ceiling is LogiQA, not
    headroom."""
    build: Callable[[Any], tuple[str, str]]
    """example -> (question block, gold answer string)."""
    header: str = ""
    """Text prepended once, before the shots. Only banking77 needs one; its label
    list is 77 lines and repeating it per exemplar would triple the prompt."""


def _numbered(options: list[str]) -> str:
    return "\n".join(f"{i}. {o}" for i, o in enumerate(options))


# banking77 --------------------------------------------------------------------
_B77_LABELS: list[str] = []


def _b77(row: Any) -> tuple[str, str]:
    return (
        f"Customer query: {row['text'].strip()}\nIntent:",
        str(row["label"]),
    )


# pubmedqa ---------------------------------------------------------------------
_PQA_CHOICES = ["yes", "no", "maybe"]


def _pqa(row: Any) -> tuple[str, str]:
    contexts = row["context"]
    if isinstance(contexts, dict):
        contexts = contexts.get("contexts", [])
    abstract = " ".join(contexts)[:2400]
    return (
        f"Abstract: {abstract}\n\nQuestion: {row['question'].strip()}\n"
        f"{_numbered(_PQA_CHOICES)}\nAnswer:",
        str(_PQA_CHOICES.index(row["final_decision"].strip().lower())),
    )


# logiqa -----------------------------------------------------------------------
def _logiqa(row: Any) -> tuple[str, str]:
    return (
        f"{row['context'].strip()}\n\nQuestion: {row['query'].strip()}\n"
        f"{_numbered(list(row['options']))}\nAnswer:",
        str(row["correct_option"]),
    )


# mnli -------------------------------------------------------------------------
_MNLI_CHOICES = ["entailment", "neutral", "contradiction"]


def _mnli(row: Any) -> tuple[str, str]:
    return (
        f"Premise: {row['premise'].strip()}\nHypothesis: {row['hypothesis'].strip()}\n"
        f"{_numbered(_MNLI_CHOICES)}\nAnswer:",
        str(row["label"]),
    )


SPECS = {
    # Not PolyAI/banking77, the canonical repo: it is script-only, `datasets` 5.x
    # will not execute a script, and the Hub never auto-converted it -- there is no
    # refs/convert/parquet revision to fall back to. legacy-datasets/banking77 is
    # the same 10,003/3,080 rows as parquet, with the 77-way ClassLabel and its
    # names intact, which the label list in the prompt header depends on.
    "banking77": ("legacy-datasets/banking77", None, "train", "test"),
    "pubmedqa": ("qiaojin/PubMedQA", "pqa_labeled", "train", "train"),
    "logiqa": ("lucasmccabe/logiqa", None, "train", "test"),
    "mnli": ("nyu-mll/glue", "mnli", "train", "validation_matched"),
}

CANDIDATES = [
    Candidate("banking77", 77, 4, "~93% (BERT-base reference is 93.7)", _b77),
    Candidate("pubmedqa", 3, 2, "~73% (BioGPT/Meditron range)", _pqa),
    Candidate("logiqa", 4, 2, "~40% (RoBERTa-large is 35.3)", _logiqa),
    Candidate("mnli", 3, 4, "~90% (BERT-large is 86.6)", _mnli),
]


def fetch(candidate: Candidate) -> tuple[list[Any], list[Any], Any]:
    """Returns ``(train rows, eval rows, train dataset)``.

    The dataset object comes back alongside the rows because banking77's label
    *names* live in ``features``, not in any row, and the header needs them.
    """
    repo, config, train_split, eval_split = SPECS[candidate.key]
    cfg = config or "default"
    train = _load(repo, config, train_split, _mirrors(repo, cfg, train_split))
    if train_split == eval_split:
        # PubMedQA's labelled subset ships one split of 1000. Hold out the tail so
        # the shots and the scored rows are disjoint -- an exemplar that is also
        # graded is the answer handed over for free.
        rows = list(train)
        return rows[:800], rows[800:], train
    test = _load(repo, config, eval_split, _mirrors(repo, cfg, eval_split))
    return list(train), list(test), train


def prompt_for(candidate: Candidate, example: Any, shots: list[Any]) -> str:
    blocks = []
    for shot in shots:
        question, answer = candidate.build(shot)
        blocks.append(f"{question} {answer}")
    blocks.append(candidate.build(example)[0])
    body = "\n\n".join(blocks)
    return f"{candidate.header}\n\n{body}" if candidate.header else body


def extract(text: str, n_choices: int) -> str | None:
    """First integer in range. Leading position preferred, as the format asks."""
    leading = re.match(r"^\W*(\d+)", text.strip())
    if leading and int(leading.group(1)) < n_choices:
        return leading.group(1)
    for match in re.finditer(r"\d+", text):
        if int(match.group(0)) < n_choices:
            return match.group(0)
    return None


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    global _B77_LABELS

    print(f"loading {MODEL} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    model.config.use_cache = True
    print(f"loaded, {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    results = []
    for candidate in CANDIDATES:
        if ONLY and candidate.key not in ONLY:
            continue
        started = time.time()
        print(f"\n=== {candidate.key} ===", flush=True)
        try:
            train, test, train_ds = fetch(candidate)
        except Exception as exc:  # noqa: BLE001 -- one dead dataset must not end the screen
            print(f"  SKIPPED: {type(exc).__name__}: {exc}", flush=True)
            results.append({"task": candidate.key, "error": f"{type(exc).__name__}: {exc}"})
            continue

        if candidate.key == "banking77":
            _B77_LABELS = _b77_label_names(train_ds)
            candidate.header = (
                "Classify the customer query into one of these banking intents.\n"
                "Answer with the number only.\n" + _numbered(_B77_LABELS)
            )

        # Fixed shots, fixed order, drawn from train -- the same discipline the real
        # harness uses, so the two are comparable.
        rng = random.Random(SEED)
        shots = [train[i] for i in sorted(rng.sample(range(len(train)), candidate.shots))]
        # Shuffled before slicing, because a screen scores a few hundred rows of a
        # split it did not choose the order of. Banking77 ships sorted by label -- 77
        # contiguous blocks of 40 -- so `test[:300]` asked about 8 intents of 77 and
        # returned 41.0% where the full split gives 36.3%. That number then ranked
        # four candidates against each other. The seed is fixed so re-screening one
        # candidate later scores the same rows it scored the first time.
        subset = list(test)
        random.Random(SEED).shuffle(subset)
        subset = subset[:LIMIT]
        prompts = [prompt_for(candidate, e, shots) for e in subset]
        golds = [candidate.build(e)[1] for e in subset]

        tokens = len(tokenizer(prompts[0])["input_ids"])
        print(f"  {len(subset)} rows, {candidate.shots} shots, prompt ~{tokens} tokens", flush=True)

        config = EvalConfig(
            max_new_tokens=6,
            batch_size=16 if tokens > 900 else 32,
            stop_sequences=("\n\n",),
            max_prompt_tokens=4096,
        )
        generations = generate_batched(model, tokenizer, prompts, config)

        correct = unparseable = 0
        for text, gold in zip(generations, golds, strict=True):
            predicted = extract(text, candidate.n_choices)
            unparseable += predicted is None
            correct += predicted == gold
        accuracy = correct / len(subset)
        chance = 1.0 / candidate.n_choices
        print(
            f"  base {accuracy:.1%}  (chance {chance:.1%}, {unparseable} unparseable, "
            f"supervised {candidate.supervised}, {time.time() - started:.0f}s)",
            flush=True,
        )
        print(f"  sample: {generations[0]!r} gold={golds[0]!r}", flush=True)
        results.append(
            {
                "task": candidate.key,
                "n": len(subset),
                "base_accuracy": accuracy,
                "chance": chance,
                "unparseable": unparseable,
                "supervised_reference": candidate.supervised,
                "shots": candidate.shots,
                "prompt_tokens": tokens,
                "seconds": round(time.time() - started, 1),
            }
        )

    merged = _merge(results)

    print("\n" + "=" * 78)
    print(f"{'task':<14}{'base':>8}{'chance':>9}{'unparse':>9}  supervised reference")
    for row in merged:
        if "error" in row:
            print(f"{row['task']:<14}{'--':>8}{'--':>9}{'--':>9}  {row['error']}")
            continue
        print(
            f"{row['task']:<14}{row['base_accuracy']:>7.1%}{row['chance']:>9.1%}"
            f"{row['unparseable']:>9}  {row['supervised_reference']}"
        )
    with Path(OUT).open("w", encoding="utf-8") as handle:
        json.dump({"model": MODEL, "limit": LIMIT, "results": merged}, handle, indent=2)
    print(f"-> {OUT}")
    return 0


def _merge(fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold this run's rows over whatever a previous run left behind.

    A re-run of one candidate should not silently drop the other three from the
    record; the table is only a decision if every arm is on it.
    """
    by_task: dict[str, dict[str, Any]] = {}
    try:
        with Path(OUT).open(encoding="utf-8") as handle:
            for row in json.load(handle)["results"]:
                by_task[row["task"]] = row
    except (OSError, KeyError, json.JSONDecodeError):
        pass
    for row in fresh:
        by_task[row["task"]] = row
    order = [c.key for c in CANDIDATES]
    return sorted(by_task.values(), key=lambda r: order.index(r["task"]))


def _b77_label_names(dataset: Any) -> list[str]:
    """The 77 intent names, in label-index order.

    Underscores become spaces: the model has to read these as English, and
    ``card_payment_fee_charged`` tokenizes into fragments that make an already
    fine-grained distinction harder to see than it needs to be.
    """
    feature = dataset.features["label"]
    names = getattr(feature, "names", None)
    if names is None:
        raise RuntimeError(
            "banking77 loaded without ClassLabel names -- the parquet mirror carries "
            "bare integers, so the label list cannot be reconstructed from it."
        )
    return [name.replace("_", " ") for name in names]


if __name__ == "__main__":
    raise SystemExit(main())
