#!/usr/bin/env python3
"""Push the Qwen3.8-27B DynQuant arms to the Hub, each under the card that describes it.

[phase4/push_to_hub.py][1] does this for a six-arm baseline panel, and its guards are the
right guards -- they are imported below rather than restated, because a second copy of a
rule agrees with the original until the day it quietly does not. What that file cannot do
is this campaign's shape. It resolves every directory through a `table.json` written by
`publish_panel.py`, and there is no such table here: this campaign has no baseline arms, by
decision, so there is nothing for a panel table to be a table *of*. Its cards also have no
field for the architectural floor budget, which on this model is the fact a reader most
needs -- see [model_card.py][2].

**What is pushed, and what is not.** The two quantized arms. Not the bf16 merge: it is the
reference the arms are scored against, not an artifact anyone asked to publish, and at 51 GiB
it would be a long upload of a checkpoint that is a `merge_and_unload` away from a base model
already on the Hub. It stays on the box as the yardstick and is named in both cards.

**Each card is generated here, against the repo id this file is about to push to.** The card
bakes the repo id into its load snippet, so one generated under a different `--repo-prefix`
tells every reader to `from_pretrained` a path that 404s -- and they reach that 404 only
after deciding to trust the numbers above it. Reading a card off disk and trusting that the
prefixes matched would make that a hope. Generating it here, from the same string used to
create the repo, makes it structural.

**Guards, all of them before the first upload**, so one invocation names every problem
rather than finding the second one after a public repo already exists:

* every arm has a directory, a `config.json` and weights;
* every arm agrees with the merge on `architectures` and `vocab_size` -- what a
  cross-contaminated export pass looks like from outside;
* every arm carries a DynQuant `quantization_config`, since both were written by DynQuant's
  exporter and one that says otherwise was written by something else;
* the two arms do not carry the *same* allocation, which is this campaign's version of the
  swap phase4 could not see. There it was invisible because six arms shared one container;
  here there are only two and they are supposed to differ, so "3-bit and 4-bit came out
  byte-identical" is checkable rather than merely printable;
* no shard exceeds the Hub's per-file ceiling;
* a repo that already holds files is refused without `--force`.

The token comes from `HF_TOKEN` and nowhere else -- never a flag, which would land it in
shell history and in `ps` for every other process on a shared box to read.

Usage, after both exports and the scoring panel have finished::

    export HF_TOKEN=...
    python experiments/phase5/push_to_hub.py \\
        --arm dq4=/workspace/exports/qwen38-27b-text2sql-dq4 \\
        --arm dq3=/workspace/exports/qwen38-27b-text2sql-dq3 \\
        --target dq4=4.02 --target dq3=3.00 \\
        --merged /workspace/runs-s2/qwen38-27b.text2sql/merged \\
        --finetune /workspace/runs-s2/qwen38-27b.text2sql/s2_finetune.json \\
        --export-record /workspace/records/export-{arm}.json \\
        --eval /workspace/evals/{arm}.json --inspect /workspace/floors.json \\
        --base-model Qwen/Qwen3.8-27B \\
        --repo-prefix VikramPal/Qwen3.8-27B-text2sql-DynQuant --dry-run

Drop `--dry-run` to upload.

[1]: ../phase4/push_to_hub.py
[2]: model_card.py
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "phase4"))
# The string this file refuses on has to be the string the loader accepts, so it is imported
# from the package rather than spelled here.
sys.path.insert(0, str(HERE.parents[1] / "packages" / "dynquant-core" / "src"))

import model_card  # noqa: E402

from dynquant.constants import HF_QUANT_METHOD  # noqa: E402


def _phase4() -> Any:
    """phase4's pusher, loaded by path under a name that cannot collide with this file.

    Both files are called ``push_to_hub.py``. A plain ``import push_to_hub`` would resolve by
    whichever directory sits earlier on ``sys.path`` -- and when this file runs as a script
    its own module name is ``__main__``, so the import is perfectly capable of loading a
    *second copy of this file* and calling its ``check`` instead of phase4's. That failure
    would be silent and would look like the guards passing.
    """
    import importlib.util

    path = HERE.parent / "phase4" / "push_to_hub.py"
    spec = importlib.util.spec_from_file_location("phase4_push_to_hub", path)
    if spec is None or spec.loader is None:  # pragma: no cover -- the file is in the tree
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase4_push_to_hub"] = module
    spec.loader.exec_module(module)
    return module


phase4 = _phase4()

#: The Hub rejects a file above this. The bf16 merge of this model comes out as a single
#: 49.8 GiB shard and would land just under it, which is the kind of margin that becomes a
#: failed upload on the next model rather than a caught one -- so it is checked, in bytes,
#: against every file actually about to be sent.
MAX_FILE_NBYTES = 50 * 1000**3


@dataclass(frozen=True)
class Arm:
    """One quantized arm: where its weights are, what budget it was exported at, its repo."""

    label: str
    source: Path
    target: str
    repo: str


def parse_pairs(values: list[str], flag: str) -> dict[str, str]:
    """``label=value`` pairs, refused rather than guessed when they are not that shape."""
    pairs: dict[str, str] = {}
    for value in values or []:
        label, sep, rest = value.partition("=")
        if not sep or not label or not rest:
            raise SystemExit(f"{flag} wants label=value, got {value!r}")
        if label in pairs:
            raise SystemExit(f"{flag} named {label!r} twice")
        pairs[label] = rest
    return pairs


def allocation(source: Path) -> dict[str, int]:
    """The per-module widths this directory's exporter wrote, read back off its config."""
    modules = (phase4._config(source).get("quantization_config") or {}).get("modules") or {}
    return {name: int(spec["bits"]) for name, spec in modules.items() if "bits" in spec}


def check(arms: list[Arm], merged: Path) -> list[str]:
    """Everything knowable without the network, reported at once rather than one per run."""
    problems: list[str] = []
    if not (merged / "config.json").is_file():
        return [
            f"{merged / 'config.json'} does not exist, and it is the checkpoint every arm "
            f"here is checked against. Point --merged at the fine-tune's merge."
        ]

    reference = phase4._config(merged)
    maps: dict[str, dict[str, int]] = {}
    for arm in arms:
        if not (arm.source / "config.json").is_file():
            problems.append(f"{arm.label}: {arm.source} has no config.json -- export it first")
            continue
        if not list(arm.source.glob("*.safetensors")):
            problems.append(f"{arm.label}: {arm.source} holds no weights")
            continue

        config = phase4._config(arm.source)
        for field in ("architectures", "vocab_size"):
            if config.get(field) != reference.get(field):
                problems.append(
                    f"{arm.label}: its {field} is {config.get(field)!r} where the merge says "
                    f"{reference.get(field)!r}. These are supposed to be one checkpoint "
                    f"quantized two ways; a directory that disagrees is a different model."
                )

        method = (config.get("quantization_config") or {}).get("quant_method")
        if method != HF_QUANT_METHOD:
            problems.append(
                f"{arm.label}: quant_method is {method!r}, not {HF_QUANT_METHOD!r}. Both arms "
                f"are written by DynQuant's exporter, so a directory that says anything else "
                f"was not written by the pass that was supposed to write it."
            )

        maps[arm.label] = allocation(arm.source)

        for shard in sorted(arm.source.glob("*.safetensors")):
            nbytes = shard.stat().st_size
            if nbytes > MAX_FILE_NBYTES:
                problems.append(
                    f"{arm.label}: {shard.name} is {nbytes / 1000**3:.1f} GB, above the Hub's "
                    f"{MAX_FILE_NBYTES / 1000**3:.0f} GB per-file limit. Re-save with a "
                    f"smaller max_shard_size; the upload would fail partway."
                )

    # The swap phase4 documents as unseeable is seeable here, because there are two arms and
    # they are supposed to differ. Identical maps mean one export overwrote the other's
    # directory, or both ran at the same target -- either way two repos would claim two
    # budgets over one set of weights.
    labels = sorted(maps)
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            if maps[left] and maps[left] == maps[right]:
                problems.append(
                    f"{left} and {right} carry byte-identical allocations over "
                    f"{len(maps[left])} modules. They are supposed to be different budgets; "
                    f"one export wrote into the other's directory, or both ran at one target."
                )
    return problems


def render(arm: Arm, args: argparse.Namespace, out: Path) -> str:
    """The card, through ``model_card.main`` and the repo id this file will push to."""
    argv = [
        "--arm",
        arm.label,
        "--repo",
        arm.repo,
        "--base-model",
        args.base_model,
        "--finetune",
        args.finetune,
        "--export",
        args.export_record.format(arm=arm.label),
        "--eval",
        args.eval.format(arm=arm.label),
        "--inspect",
        args.inspect,
        "--inspect-target",
        arm.target,
        "--out",
        str(out),
    ]
    if args.panel:
        argv += ["--panel", args.panel]
    if args.reference_eval:
        argv += ["--reference-eval", args.reference_eval]
    model_card.main(argv)
    return out.read_text(encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="repeatable; the exported directory for each arm",
    )
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        metavar="LABEL=BITS",
        help="repeatable; which inspect row each arm was exported at",
    )
    parser.add_argument("--merged", required=True, help="the bf16 merge, as the reference")
    parser.add_argument("--finetune", required=True)
    parser.add_argument(
        "--export-record",
        required=True,
        help="path to each arm's export json; {arm} is substituted",
    )
    parser.add_argument(
        "--eval", required=True, help="path to each arm's eval json; {arm} is substituted"
    )
    parser.add_argument("--inspect", required=True)
    parser.add_argument("--reference-eval")
    parser.add_argument("--panel")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--repo-prefix", required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="upload into repos that already hold files"
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    directories = parse_pairs(args.arm, "--arm")
    targets = parse_pairs(args.target, "--target")
    if set(directories) != set(targets):
        raise SystemExit(
            f"--arm names {sorted(directories)} and --target names {sorted(targets)}. Every "
            f"arm needs the budget its card will describe."
        )

    arms = [
        Arm(
            label=label,
            source=Path(directories[label]),
            target=targets[label],
            repo=f"{args.repo_prefix}-{label}",
        )
        for label in sorted(directories)
    ]

    problems = check(arms, Path(args.merged))
    if problems:
        for problem in problems:
            print(f"  refused: {problem}")
        return 1

    print(f"{'arm':<8} {'widths':<16} {'repo'}")
    for arm in arms:
        print(f"{arm.label:<8} {phase4.widths(arm.source):<16} {arm.repo}")

    # Rendered before the token is even looked for: a card that cannot be generated is a
    # problem to find now, not after the first 13 GB has gone up.
    cards = {arm.label: render(arm, args, arm.source / "README.md") for arm in arms}
    for arm in arms:
        print(f"\n--- {arm.repo} card, first lines ---")
        print("\n".join(cards[arm.label].splitlines()[:12]))

    if args.dry_run:
        print("\ndry run: nothing uploaded. Cards are written beside the weights.")
        return 0

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise SystemExit("HF_TOKEN is not set. It is read from the environment and nowhere else.")

    taken = phase4.occupied(
        [phase4.Push(label=a.label, source=a.source, repo=a.repo, kind="quantized") for a in arms],
        token,
    )
    if taken and not args.force:
        for line in taken:
            print(f"  refused: {line}. Pass --force to replace it.")
        return 1

    for arm in arms:
        push = phase4.Push(label=arm.label, source=arm.source, repo=arm.repo, kind="quantized")
        url = phase4.upload(push, cards[arm.label], token, private=args.private)
        print(f"pushed {arm.label} -> {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
