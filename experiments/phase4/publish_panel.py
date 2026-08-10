#!/usr/bin/env python3
"""Publish every scored arm of a panel as a loadable directory, in the order that survives.

The panel produces records, not checkpoints. `arms_lfm2.py run` loads a model, applies a
recipe or a bit map, scores 12,000 problems in process, and writes JSON. Nothing anyone can
download comes out of it. Turning the seven rows of a finished table into six directories is
therefore a second pass over every arm, and this is the script that runs it.

It reads `arms.json` and derives everything else. The panel already wrote down which arms
exist, what each one's method and anchor is, which record it produced and -- for a DynQuant
arm -- which map priced it. Restating any of that here would be a second copy of the panel's
own registry, and this campaign has now found seven of those, each one agreeing with the
original until the first case that separated them. So the only thing this file contributes
is the *order*, the guards, and the two command shapes.

**Two command shapes, because there are two kinds of arm.**

A baseline arm re-runs its recipe. GPTQ and AWQ calibrate against data, and the panel kept
no checkpoint, so the weights in the published directory come from a pass that has not
happened yet. `--scored` is what connects it back: the arm's own `<label>.quant.json`,
compared field by field against what the second pass produced. Without it the label on the
directory is the only claim being made.

A DynQuant arm does not re-calibrate. Its allocation is a saved map, priced by the allocator
and checked against the baselines' anchor before a single problem was scored, and `export`
applies the identical encoder at the identical widths. So the map *is* the artifact the
score came from, and there is no second pass to reconcile -- the check that belongs on these
is the byte count, which the map already states exactly.

**The order is cost-to-redo, descending.** `/workspace` on the campaign box is not a volume:
a recycle keeps nothing. A GPTQ pass over 8 B parameters is 32 minutes and an AWQ pass 47,
so the four baselines are between two and three hours that cannot be recovered by re-running
a cheap command. A DynQuant export from an existing map is minutes. Baselines first,
therefore, and 4 bits before 3 within each method only because the 4-bit pair is what the
table's separated comparison is against. Nothing here pushes to the Hub; that is a decision
about publication, not a step in producing the artifacts.

Nothing in this file has been run against the 8B. The publish path it drives has been proven
end to end on a four-layer `lfm2_moe` -- all six arms carrying within 0.068 code steps -- and
what remains between that and this is scale, not model class. That distinction is the whole
content of the claim, so it is stated here rather than discovered later.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE_SRC = HERE.parents[1] / "packages" / "dynquant-core" / "src"

# Every child this launches needs `dynquant` importable, and nothing installs it in the
# venv on the campaign box. The recipe children get there on their own -- `baselines_lfm2`
# puts the source on `sys.path` once it is running -- but `python -m dynquant export`
# cannot, because the module has to resolve before any code of ours runs. Set here rather
# than asked of whoever types the command: an env var the operator has to remember is the
# same arrangement that let panel_table.py ship unable to run outside my own shell.
if CORE_SRC.is_dir():  # pragma: no cover - exercised through a subprocess
    _inherited = os.environ.get("PYTHONPATH", "")
    if str(CORE_SRC) not in _inherited.split(os.pathsep):
        os.environ["PYTHONPATH"] = (
            f"{CORE_SRC}{os.pathsep}{_inherited}" if _inherited else str(CORE_SRC)
        )

#: Arms this script knows how to publish, and what it calls the two shapes. A `bf16` ceiling
#: is not published: it is the checkpoint the panel started from, already on disk, and
#: writing a copy of it under a new name would be the one directory in the set whose
#: contents are not a result.
RECIPE_KINDS = ("gptq", "awq", "rtn")
MAP_KINDS = ("dq",)

#: Cost to redo, descending, and the reason each position is what it is. A recipe arm cannot
#: be recovered without another calibration pass over 8 B parameters; a map arm can be
#: re-exported in minutes from a file that already exists. Within the recipe arms the 4-bit
#: pair goes first because the panel's separated comparison is against it.
ORDER = ("gptq_4b", "awq_4b", "gptq_3b", "awq_3b", "dq_4b", "dq_3b")


@dataclass(frozen=True)
class Step:
    """One arm's publish, with what it will write and what that has to weigh."""

    label: str
    kind: str
    out: Path
    cmd: list[str]
    #: What the panel's own accounting says this arm is. For a recipe arm it is the
    #: compressed-tensors anchor, which the DynQuant container does *not* match -- 4.25 bits
    #: against 4.15625 -- so it is carried to be reported against, not asserted.
    scored_bytes: int | None
    #: The record this pass has to reproduce, or None for an arm whose widths are a file.
    scored: Path | None


def _python() -> str:
    """The interpreter running this, not whichever one is first on PATH.

    The box has two: a system python and the venv that carries llm-compressor. A publish
    that resolved `python` from PATH would calibrate under one and be scored under the other.
    """
    return sys.executable


def recipe_step(spec: dict, panel: Path, out_root: Path, arms: dict) -> Step:
    """A baseline arm: re-run its recipe, and compare the result to what it scored."""
    label = str(spec["label"])
    record = panel / f"{label}.quant.json"
    out = out_root / label
    cmd = [
        _python(),
        str(HERE / "baselines_lfm2.py"),
        "publish",
        "--model",
        str(arms["model"]),
        "--save-to",
        str(out),
        "--method",
        str(spec["kind"]),
        "--bits",
        str(spec["anchor"]),
        "--group-size",
        str(arms["group_size"]),
        "--scored",
        str(record),
    ]
    return Step(label, spec["kind"], out, cmd, spec.get("nbytes"), record)


def map_step(spec: dict, out_root: Path, arms: dict) -> Step:
    """A DynQuant arm: export the map that was scored, at the key it was priced under.

    `--map-key` is the arm's byte target and not its achieved `nbytes`. A map file holds one
    allocation per target it was asked for, keyed by the target, and the achieved count is
    what that allocation came out at -- so keying by the achieved count would miss on every
    arm that did not land exactly on its anchor, which is every DynQuant arm in this panel.
    """
    label = str(spec["label"])
    out = out_root / label
    cmd = [
        _python(),
        "-m",
        "dynquant",
        "export",
        str(arms["model"]),
        "--output",
        str(out),
        "--map",
        str(spec["map"]),
        "--map-key",
        str(spec["target_bytes"]),
        "--group-size",
        str(arms["group_size"]),
    ]
    return Step(label, spec["kind"], out, cmd, spec.get("nbytes"), None)


def plan(arms_path: Path, out_root: Path, only: list[str] | None = None) -> list[Step]:
    """Derive the publish plan from the panel's own registry.

    Refuses an arm that was never scored rather than publishing it. An unscored arm is not
    a missing convenience: the directory it would write is a model with no row, and the
    entire point of the second pass is that the directory and the row are the same model.
    """
    arms = json.loads(arms_path.read_text(encoding="utf-8"))
    panel = arms_path.parent
    by_label = {str(a["label"]): a for a in arms["arms"]}

    wanted = list(only) if only else [a for a in ORDER if a in by_label]
    unknown = [label for label in wanted if label not in by_label]
    if unknown:
        raise SystemExit(f"{arms_path} has no arm named {unknown}; it has {sorted(by_label)}")

    # Every publishable arm in the registry has to be in ORDER, or a seventh arm added to a
    # future panel would be dropped here in silence -- which is the failure this whole file
    # is arranged against.
    publishable = {
        label for label, spec in by_label.items() if spec.get("kind") in RECIPE_KINDS + MAP_KINDS
    }
    missed = sorted(publishable - set(ORDER))
    if missed:
        raise SystemExit(
            f"{arms_path} carries publishable arm(s) {missed} that this script has no place "
            f"for. Add them to ORDER with the reason for the position."
        )

    steps: list[Step] = []
    for label in wanted:
        spec = by_label[label]
        kind = str(spec.get("kind"))
        if kind not in RECIPE_KINDS + MAP_KINDS:
            raise SystemExit(f"{label} is a {kind!r} arm, which is not a thing to publish")
        if spec.get("record") is None or spec.get("nbytes") is None:
            raise SystemExit(
                f"{label} was never scored -- {arms_path} has no record for it. Publishing "
                f"it would write a directory with no row in the table. If the panel is "
                f"deliberately short an arm, name the ones to publish with --only."
            )
        if kind in MAP_KINDS:
            if not spec.get("map"):
                raise SystemExit(f"{label} is a map arm with no map recorded in {arms_path}")
            steps.append(map_step(spec, out_root, arms))
        else:
            steps.append(recipe_step(spec, panel, out_root, arms))
    return steps


def guard(steps: list[Step], *, force: bool) -> None:
    """Everything refusable before the first hour of GPU is spent.

    Ordered cheapest first and all of it before any command runs, because a plan that dies
    on arm three has already spent the two hours arms one and two took.
    """
    running = subprocess.run(
        ["pgrep", "-af", r"arms_lfm2\.py run"], capture_output=True, text=True, check=False
    )
    if running.stdout.strip():
        raise SystemExit(
            "the panel driver is still running and this wants the same card:\n"
            f"{running.stdout.strip()}"
        )

    for step in steps:
        if step.scored is not None and not step.scored.is_file():
            raise SystemExit(f"{step.label}: {step.scored} is not there")
        if step.out.exists() and not force:
            raise SystemExit(
                f"{step.label}: {step.out} already exists. It is either a finished publish "
                f"or half of one, and this script cannot tell which. Move it aside, or pass "
                f"--force to overwrite."
            )

    # Last, because it is the only check here that costs a process. It is a check at all
    # because the map arms are published last: `-m dynquant` failing to resolve, or a clone
    # too old to carry `export`, is an error that would otherwise arrive two and a half
    # hours in with every recipe arm already paid for. Probed with the subcommand rather
    # than bare `--help` so a resolvable module without this command is caught too.
    if any(step.kind in MAP_KINDS for step in steps):
        probe = subprocess.run(
            [_python(), "-m", "dynquant", "export", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            raise SystemExit(
                f"{_python()} cannot run `-m dynquant export`, which every map arm is "
                f"published with, so those arms would fail after the recipe arms had "
                f"run:\n{(probe.stderr or probe.stdout).strip()}"
            )


def report(step: Step, out: Path) -> str:
    """What the directory weighs, against what the arm was scored at.

    The two kinds of arm answer this differently, and the line is misread if they are not
    told apart.

    A **map arm** was priced by the allocator in DynQuant's own container and exports into
    that same container, so the written bytes should land on the arm's own figure. `dq_4b`
    is 4,397,666,304 B at 4.1547 average bits by the map's accounting and should write that
    plus a tokenizer; a gap larger than that is a disagreement between the pricing model and
    the writer, which `export` also reports on its own.

    A **recipe arm** was scored under compressed-tensors -- 4 + 20/128 bits at 4, the zero
    point itself packed to four -- and republishes by carrying the identical codes into
    DynQuant's container, which spends a full bf16 zero per group: 4 + 32/128. Same codes,
    same numbers on dequantization, about 2.3% more disk. So the gap on those four is the
    container and not a discrepancy, and the line says so rather than leaving it to be
    discovered in a directory listing.
    """
    written = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    if not step.scored_bytes:
        return f"{step.label}: wrote {written:,} B"
    delta = written - step.scored_bytes
    pct = 100.0 * delta / step.scored_bytes
    return (
        f"{step.label}: wrote {written:,} B against the arm's {step.scored_bytes:,} B "
        f"({delta:+,} B, {pct:+.2f}% -- the container, not the model)"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--arms", default="/workspace/runs/s4/panel/arms.json")
    p.add_argument("--out", default="/workspace/runs/s4/published")
    p.add_argument(
        "--only",
        default=None,
        help="comma-separated labels, in the order given, instead of the full plan",
    )
    p.add_argument("--dry-run", action="store_true", help="print the plan and stop")
    p.add_argument("--force", action="store_true", help="overwrite an existing directory")
    args = p.parse_args(argv)

    only = [s for s in args.only.split(",") if s] if args.only else None
    out_root = Path(args.out)
    steps = plan(Path(args.arms), out_root, only)

    for step in steps:
        print(f"{step.label:<10} {step.kind:<5} -> {step.out}")
        print(f"           {' '.join(step.cmd)}")
    if args.dry_run:
        return 0

    guard(steps, force=args.force)
    out_root.mkdir(parents=True, exist_ok=True)

    for step in steps:
        if step.out.exists() and args.force:
            shutil.rmtree(step.out)
        print(f"\n=== {step.label} ===", flush=True)
        result = subprocess.run(step.cmd, check=False)
        if result.returncode != 0:
            # Not fatal to the rest. Each arm is independent, the expensive ones are first,
            # and a box that is not a volume makes "finish what you can" the right default.
            print(f"{step.label}: FAILED with {result.returncode}", flush=True)
            continue
        print(report(step, step.out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
