#!/usr/bin/env python3
"""S1: what each phase-3 model already scores, before anything is spent on it.

The screen exists because a fine-tune arm that cannot move is indistinguishable from
one that moved and was undone by quantization. A task the base model already saturates
has no headroom to show a gain, and one it scores at chance has no signal to lose --
either way the arm costs its full GPU-hours and answers nothing. GSM8K taught this the
expensive way: a flat arm was diagnosed only after a full six-arm run.

So this measures the four base models on the four scored tasks first, and the decision
it feeds is which of the sixteen (model, task) cells are worth carrying into S2.

Every cell is one ``dynquant eval`` invocation rather than an in-process loop. That is
deliberate: the campaign's results come from that command, and a driver that called the
harness directly would be a second evaluator whose prompts could drift from the one
being reported. It costs an engine build per cell, which is minutes against hours.

**Resume is checked against the code, not just the file.** A record on disk is only
reusable if it was produced by the commit that is checked out now; skip-if-output-exists
cannot see that an output predates its input, and a stale record staples last week's
prompt-assembly to this week's conclusion. Each record therefore gets a ``.commit``
sidecar and is re-run when it disagrees with ``HEAD``.

Usage::

    python scripts/run_s1_headroom.py --out reports/s1
    python scripts/run_s1_headroom.py --out reports/s1 --models phi4-mini --limit 200
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from dynquant.commands.evaluate import TASKS as TASK_REGISTRY

#: The four models the campaign screens, and whether the Hub requires an accepted
#: licence to download them. Gating is recorded here rather than discovered at failure
#: time so the driver can say up front which cells it cannot run, instead of dying
#: partway through a sixteen-cell sweep with a 401.
MODELS: dict[str, dict[str, object]] = {
    "llama31-8b": {"repo": "meta-llama/Llama-3.1-8B-Instruct", "gated": True},
    "gemma3-4b": {"repo": "google/gemma-3-4b-it", "gated": True},
    "phi4-mini": {"repo": "microsoft/Phi-4-mini-instruct", "gated": False},
    "ministral-8b": {"repo": "mistralai/Ministral-8B-Instruct-2410", "gated": False},
}

#: Ordered cheapest-first so a broken cell surfaces before the expensive ones run.
TASKS = ("gsm8k", "ifeval", "humaneval", "mbpp")

#: Which per-task flags each cell needs, taken from the registry rather than written
#: out here. `dynquant eval` refuses to score a code task without --allow-execution,
#: and that refusal arrives *after* the model is loaded -- so a task that grew the
#: requirement and a driver that did not know would cost a load per cell to discover.
TASKS_EXECUTING_CODE = frozenset(name for name, spec in TASK_REGISTRY.items() if spec.executes_code)
TASKS_UNVERIFIABLE = frozenset(name for name, spec in TASK_REGISTRY.items() if spec.unverifiable)

#: What :func:`summarize` reads out of each record.
RECORD_FIELDS = ("total", "accuracy", "chance", "unparseable")


def _git_head(repo: Path) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _cell_command(repo_id: str, task: str, out: Path, args: argparse.Namespace) -> list[str]:
    """One ``dynquant eval`` invocation. The model is positional, not ``--model``."""
    cmd = [
        sys.executable,
        "-m",
        "dynquant",
        "eval",
        repo_id,
        "--task",
        task,
        "--backend",
        args.backend,
        "--out",
        str(out),
        "--dtype",
        args.dtype,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    if TASKS_EXECUTING_CODE & {task}:
        # Not defaulted on: pass@1 means running model-written code, and the opt-in is
        # the whole point of the flag. Read off the registry rather than listed here,
        # so a task added to the campaign cannot arrive without it.
        cmd.append("--allow-execution")
    if TASKS_UNVERIFIABLE & {task}:
        # IFEval carries instructions no checker in this process can verify. Dropping
        # them scores the rest and records which; the alternative is a task that
        # refuses to produce a number at all when `langdetect` is missing.
        cmd += ["--on-unverifiable", "drop"]
    return cmd


def _needs_run(record: Path, commit: str, *, force: bool) -> bool:
    if force or not record.exists():
        return True
    stamp = record.with_suffix(".commit")
    if not stamp.exists():
        return True
    return stamp.read_text(encoding="utf-8").strip() != commit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="directory the records are written to")
    parser.add_argument("--repo", default=".", help="the dynquant checkout, for the commit stamp")
    parser.add_argument("--models", nargs="*", choices=sorted(MODELS), default=sorted(MODELS))
    parser.add_argument("--tasks", nargs="*", choices=TASKS, default=list(TASKS))
    parser.add_argument("--backend", default="vllm", choices=("transformers", "vllm"))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=None, help="smoke runs only")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="some Hub checkpoints ship their own modelling code and will not load without it",
    )
    parser.add_argument("--force", action="store_true", help="re-run cells already recorded")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    commit = _git_head(Path(args.repo))
    have_token = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))

    planned, blocked = [], []
    for name in args.models:
        if MODELS[name]["gated"] and not have_token:
            blocked.append(name)
            continue
        planned.extend((name, task) for task in args.tasks)

    if blocked:
        print(
            f"SKIPPING {len(blocked)} gated model(s) with no HF_TOKEN in the environment: "
            f"{', '.join(blocked)}. Their licences are accepted per-account on the Hub, so "
            f"this cannot be worked around from here -- the remaining cells still run.",
            flush=True,
        )
    print(f"s1: {len(planned)} cell(s) at {commit[:8]}", flush=True)

    failures = []
    for index, (name, task) in enumerate(planned, start=1):
        record = out_dir / f"{name}.{task}.json"
        if not _needs_run(record, commit, force=args.force):
            print(f"[{index}/{len(planned)}] {name} {task}: already at {commit[:8]}", flush=True)
            continue

        repo_id = str(MODELS[name]["repo"])
        print(f"[{index}/{len(planned)}] {name} {task}: running", flush=True)
        proc = subprocess.run(_cell_command(repo_id, task, record, args), check=False)
        if proc.returncode != 0:
            # Recorded and carried past rather than raised: one model failing to load
            # should not cost the other fifteen cells their run.
            failures.append(f"{name}.{task} (exit {proc.returncode})")
            print(f"  FAILED {name} {task}: exit {proc.returncode}", flush=True)
            continue
        record.with_suffix(".commit").write_text(commit, encoding="utf-8")

    print("\n" + summarize(out_dir), flush=True)
    if failures:
        print(f"\n{len(failures)} cell(s) failed: {', '.join(failures)}", flush=True)
    if blocked:
        print(f"{len(blocked)} model(s) not screened for want of a token: {', '.join(blocked)}")
    return 1 if failures else 0


def summarize(out_dir: Path) -> str:
    """The screen's actual output: accuracy against the task's chance floor.

    Accuracy alone cannot say whether a cell has headroom. A score at chance and a score
    at ceiling are both unusable and they sit at opposite ends of the column, so the
    floor is printed beside every number.
    """
    rows = ["model            task        n      acc%    chance%  unscored"]
    for path in sorted(out_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        missing = [field for field in RECORD_FIELDS if field not in record]
        if missing:
            # Named rather than raised as a KeyError, because the records are the
            # expensive part and they are already on disk: the fix is to this function,
            # and it should not require reading a traceback to find which field moved.
            rows.append(f"{path.name}: no {', '.join(missing)} in the record")
            continue
        name, task = path.stem.split(".", 1)
        rows.append(
            f"{name:<16} {task:<11} {record['total']:<6} "
            f"{100 * record['accuracy']:>6.2f}  {100 * record['chance']:>7.2f}  "
            f"{record['unparseable']}"
        )
    return "\n".join(rows) if len(rows) > 1 else "no records yet"


if __name__ == "__main__":
    raise SystemExit(main())
