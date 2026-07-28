"""Shared setup for the four-point experiment.

Every measurement point in this experiment loads a model, evaluates it, and writes
a JSON record. What makes the six numbers comparable is that *only* the weights
differ between them -- so everything else lives here, once, and no stage is
allowed to pick its own prompt, its own decode settings, or its own subset.

The stages are separate scripts rather than one driver because the run is long and
GPU-bound: a failure in quantization should not cost the fine-tune, and each stage
is restartable from the artifacts on disk.

Which task
----------
Selected by ``DQ_TASK`` (``casehold`` by default) and resolved in :mod:`tasks`, which
holds everything that differs between them. ``RUN_DIR`` includes the task key, so two
runs' records cannot land on top of each other -- a shared directory would let a stale
GSM8K record be read into a CaseHOLD table as a row that looks measured and is not,
which is the kind of mistake that survives review.

Which model
-----------
Selected by ``DQ_MODEL``. It was a constant for the first two runs, and hoisting it to
an environment variable is what lets a second model reuse this machinery instead of
forking it -- the same argument :mod:`tasks` makes for keeping both tasks in one file.
``RUN_DIR`` therefore includes a model slug as well as the task key, for the same
reason: a Mistral record read into a Qwen table is the same failure as a GSM8K record
read into a CaseHOLD one, and less obvious.

The committed Qwen records were written before the slug existed, under
``/workspace/runs/qwen35_2b_<task>``. The derived slug is ``qwen3_5_2b_base``, so
reproducing against those directories means passing ``DQ_RUN_DIR`` explicitly rather
than relying on the default.
"""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tasks import get_task, split_task

MODEL_ID = os.environ.get("DQ_MODEL", "Qwen/Qwen3.5-2B-Base")
SEED = 0
DTYPE = torch.bfloat16

TASK = get_task()


def model_slug(model_id: str = MODEL_ID) -> str:
    """A filesystem-safe name for the model, for run directories and record fields."""
    return re.sub(r"[^a-z0-9]+", "_", model_id.rsplit("/", 1)[-1].lower()).strip("_")


RUN_DIR = Path(os.environ.get("DQ_RUN_DIR", f"/workspace/runs/{model_slug()}_{TASK.key}"))

N_SHOTS = TASK.n_shots
"""Re-exported so the stage scripts need not each reach into the task object for the
one field they print. Held fixed across all six measurement points -- including the
fine-tuned one."""


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval_config(limit: int | None = None, *, task: Any = None) -> Any:
    """The decode settings, identical at every measurement point.

    ``task`` overrides the selected one. It exists for :mod:`screen_headroom`, which
    measures several candidates in a single process and so cannot rely on ``DQ_TASK``.
    The override is a parameter rather than a second copy of this function because the
    screen's numbers are only worth anything if they were produced by the same decode
    settings as the table's -- and two functions that build an ``EvalConfig`` would
    agree at first and diverge later.
    """
    from dynquant.eval.harness import EvalConfig

    task = task or TASK
    return EvalConfig(
        max_new_tokens=task.max_new_tokens,
        batch_size=task.eval_batch_size,
        stop_sequences=(task.stop_sequence(),),
        max_prompt_tokens=task.max_prompt_tokens,
        limit=limit,
    )


def load_task() -> tuple[list[Any], list[Any], list[Any]]:
    """Return ``(train, test, shots)``. See :func:`tasks.split_task` for the split rule."""
    return split_task(TASK, SEED)


def load_tokenizer() -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(path: str | Path = MODEL_ID, *, device: str = "cuda") -> Any:
    """Load in the compute dtype, on ``device``.

    Pass ``device="cpu"`` for the inspection scripts. They only read module shapes,
    names and the tree structure, so the weights never need to reach the GPU -- and
    holding a second copy of the model there while an evaluation is running is a way to
    turn an analysis into an OOM in the run that matters.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        dtype=DTYPE,
        device_map=device,
    )
    model.config.use_cache = True
    return model


def progress(label: str) -> Any:
    """A callback that prints without holding a line buffer open."""

    def report(done: int, total: int) -> None:
        print(f"  [{label}] {done}/{total} prompts", flush=True)

    return report


def run_eval(
    model: Any,
    *,
    label: str,
    name: str,
    limit: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Any:
    """Score an already-loaded model and write the record.

    Every measurement point goes through here. That is the whole point of the
    function: two call sites that each build their own prompt and decode settings
    will agree today and drift tomorrow, and the drift would show up as an accuracy
    difference indistinguishable from one the quantizer caused.
    """
    import time

    _, test, shots = load_task()
    tokenizer = load_tokenizer()

    started = time.time()
    result = TASK.evaluate(
        model,
        tokenizer,
        test,
        label=label,
        shots=shots,
        config=eval_config(limit),
        progress=progress(name),
        keep_predictions=12,
    )
    elapsed = time.time() - started

    print("\n" + result.summary(), flush=True)
    record(
        name,
        {
            "task": TASK.key,
            "label": result.label,
            "accuracy": result.accuracy,
            "correct": result.correct,
            "total": result.total,
            "unparseable": result.unparseable,
            # Stored per record rather than assumed by the reader: a 5-way task
            # bottoms out at 0.20, and a table that omits the floor makes a collapsed
            # arm look like a mildly damaged one.
            "chance": TASK.chance,
            "seconds": round(elapsed, 1),
            **(extra or {}),
            # Per-problem correctness, so the six measurement points can be compared
            # as the paired design they are. The first version of this harness stored
            # only the counts, and the headline comparison -- allocated against
            # uniform at the same budget -- then had to be reported as "not
            # separated" by an unpaired test that is roughly twice as wide as the
            # paired one on this data. Recovering the vector meant re-running every
            # arm on the GPU. It is one boolean per problem.
            "hits": result.hits,
            "predictions": result.predictions,
        },
    )
    return result


def record(name: str, payload: Any) -> Path:
    """Write one stage's result, and echo it so the log alone is readable."""
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    path = RUN_DIR / f"{name}.json"
    data = asdict(payload) if is_dataclass(payload) and not isinstance(payload, type) else payload
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"\n-> wrote {path}", flush=True)
    return path


def read_record(name: str) -> Any | None:
    path = RUN_DIR / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
