#!/usr/bin/env python3
"""Push a finished panel to the Hub, generating each card against the id it is pushed to.

`publish_panel.py` turns a table of scored arms into directories. `model_cards.py` turns the
same table into the README above each one. Neither uploads, and until this file there was no
step that did -- six directories and their cards sat on a box whose `/workspace` is not a
volume, one recycle away from being a campaign nobody outside it can read.

**The card is written here rather than read from the directory.** `model_cards.py` bakes the
repo id into the usage snippet, so a card generated with one `--repo-prefix` and pushed under
another tells every reader to load a repo that does not exist -- a 404 reached only after
they have decided to trust the numbers above it. Reading the card off disk and hoping the
prefixes matched would make that a check; generating it here, from the same `card()` and the
same `slug()` this file pushes with, makes it impossible. The rendered card is written back
beside the weights before the upload, so the local copy is the pushed copy.

**Which directory an arm is.** Quantized arms come from `--published`, which is
`publish_panel.py`'s `--out`. The ceiling does not: on this campaign it is the merged
fine-tune itself, and the directory holding it is named by the fine-tune's own manifest. That
is read from `output` rather than rebuilt from a run root, for the reason this campaign has
now paid for twice -- an env-derived path once sent four Mistral arms into a Qwen directory,
and a reconstructed path answers "where did this go" differently without ever saying so.

**What is checked before anything uploads.** A repo is public and permanent enough that the
guards are worth more than the minute they cost, and all of them run before the first upload
so one invocation names every problem:

* every arm the table declares publishable has a directory, a `config.json` and weights;
* every directory agrees with the merge on `architectures` and `vocab_size`, which is what a
  cross-contaminated publish pass looks like from the outside;
* every quantized directory carries a DynQuant `quantization_config`, because all of them
  are written by DynQuant's exporter and one that says otherwise was written by something
  that was not the publish pass;
* the ceiling carries no `quantization_config` at all -- the same claim the card generator
  makes in prose, checked against the weights it is written above;
* a repo that already holds files is refused without `--force`, so a re-run cannot quietly
  replace a checkpoint somebody has already downloaded.

One thing these cannot check, stated because it is the gap a reader of this file would
otherwise assume closed: a swap *between two quantized arms* is invisible here. The recipe
arms and the DynQuant arms are written by the same exporter into the same container, which
records the widths but not which pass chose them, so `gptq_4b` holding the DynQuant export
has a config indistinguishable from the right one. What the plan prints instead is how many
distinct widths each map carries -- a recipe arm is uniform at its anchor and a DynQuant arm
usually is not -- which is a fact a human can read at the moment of the decision, not a rule
this file is willing to refuse on. A DynQuant allocation *can* come out uniform when the
budget stops binding, and a guard that fires on a legitimate run gets disabled.

**The token comes from `HF_TOKEN` and from nowhere else.** Not a flag: an argument lands in
shell history and in `ps`, where every other process on a shared box can read it. Not a file
either. A token passed on the command line is refused rather than used.

Usage, after the panel and the publish pass have both finished::

    export HF_TOKEN=...
    python experiments/phase4/push_to_hub.py --table PANEL/table.json
        --finetune RUN/s2_finetune.json --published PUBLISHED
        --repo-prefix myorg/mistral-7b-instruct-v0.3-text2sql --include-ceiling --dry-run

Drop `--dry-run` to upload. `--only` takes the same comma-separated labels as its neighbours.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# The string this file refuses on has to be the string the loader accepts, so it is imported
# from the package rather than spelled here.
sys.path.insert(0, str(HERE.parents[1] / "packages" / "dynquant-core" / "src"))

import model_cards  # noqa: E402

from dynquant.constants import HF_QUANT_METHOD  # noqa: E402

#: Uploaded from every directory. `publish_panel.py` writes the weights and the tokenizer;
#: everything else in the source directory -- a training checkpoint, an optimizer state, a
#: log -- is the run's business and not the reader's. An explicit list beats an ignore list:
#: a file nobody thought about stays local instead of being published by default.
KEEP = (
    "*.safetensors",
    "*.json",
    "*.model",
    "*.txt",
    "*.md",
    "*.jinja",
)


@dataclass(frozen=True)
class Push:
    """One arm, resolved: where it is now and what it will be called."""

    label: str
    source: Path
    repo: str
    kind: str


def sources(
    table: dict[str, Any], finetune: dict[str, Any], published: Path, labels: list[str], prefix: str
) -> list[Push]:
    """Where each label's weights are, and the repo id its own card will name.

    The ceiling is the exception and the reason this is a function: it is the merged
    fine-tune, which lives where the trainer put it, while every other arm lives where the
    publish pass put it.
    """
    merged = Path(str(finetune.get("output", "")))
    out = []
    for label in labels:
        row = model_cards.find_arm(table, label)
        kind = str(row.get("kind"))
        source = merged if kind == model_cards.CEILING else published / label
        repo = f"{prefix}-{model_cards.slug(label, kind, row['anchor'])}"
        out.append(Push(label=label, source=source, repo=repo, kind=kind))
    return out


def _config(path: Path) -> dict[str, Any]:
    return json.loads((path / "config.json").read_text(encoding="utf-8"))


def widths(source: Path) -> str:
    """How many distinct bit widths this directory's map carries, for the plan to print.

    Not a guard -- see the note in the module docstring. It is the one line that separates a
    uniform recipe export from an allocation, read off the map the exporter wrote rather
    than recomputed from the allocator, and it belongs in front of a human before six public
    repos exist rather than in a rule that would refuse a legitimate uniform allocation.
    """
    modules = (_config(source).get("quantization_config") or {}).get("modules") or {}
    found = sorted({int(spec["bits"]) for spec in modules.values() if "bits" in spec})
    if not found:
        return "unquantized"
    return f"{len(found)}x{{{','.join(str(b) for b in found)}}}b"


def check(pushes: list[Push], merged: Path) -> list[str]:
    """Everything that can be known without the network, reported all at once.

    Returns the problems rather than raising on the first, because each of these is a
    separate mistake and finding them one upload at a time means finding some of them after
    a repo already exists.
    """
    problems: list[str] = []
    if not (merged / "config.json").is_file():
        problems.append(
            f"{merged / 'config.json'} does not exist, and it is the checkpoint every other "
            f"directory here is checked against. The fine-tune's manifest names {merged} as "
            f"its output; point --finetune at a manifest whose merge is present."
        )
        return problems

    reference = _config(merged)
    for push in pushes:
        if not (push.source / "config.json").is_file():
            problems.append(f"{push.label}: {push.source} has no config.json -- publish it first")
            continue
        if not list(push.source.glob("*.safetensors")):
            problems.append(f"{push.label}: {push.source} holds no weights")

        config = _config(push.source)
        for field in ("architectures", "vocab_size"):
            if config.get(field) != reference.get(field):
                problems.append(
                    f"{push.label}: its {field} is {config.get(field)!r} where the merge says "
                    f"{reference.get(field)!r}. These are supposed to be the same checkpoint "
                    f"quantized; a directory that disagrees is a different model."
                )

        method = (config.get("quantization_config") or {}).get("quant_method")
        if push.kind == model_cards.CEILING:
            if config.get("quantization_config") is not None:
                problems.append(
                    f"{push.label}: the ceiling's config carries a quantization_config "
                    f"({method!r}). Its card claims an unquantized checkpoint."
                )
        elif method != HF_QUANT_METHOD:
            problems.append(
                f"{push.label}: quant_method is {method!r}, not {HF_QUANT_METHOD!r}. Every "
                f"arm in this panel is written by DynQuant's exporter -- the baselines carry "
                f"their recipe's codes into its container -- so a directory that says "
                f"anything else was not written by the pass that was supposed to write it."
            )
    return problems


def occupied(pushes: list[Push], token: str) -> list[str]:
    """Which target repos already hold files. Costs one API call each and is worth it."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    taken = []
    for push in pushes:
        try:
            files = [
                f
                for f in api.list_repo_files(push.repo)
                if f not in ("README.md", ".gitattributes")
            ]
        except Exception:  # noqa: BLE001 -- absent, private to someone else, or unreachable
            continue
        if files:
            taken.append(f"{push.repo} already holds {len(files)} files")
    return taken


def upload(push: Push, card: str, token: str, *, private: bool) -> str:
    """Write the card beside the weights, then push the directory. Returns the repo url."""
    from huggingface_hub import HfApi

    (push.source / "README.md").write_text(card, encoding="utf-8", newline=model_cards.NL)
    api = HfApi(token=token)
    api.create_repo(push.repo, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=push.repo,
        folder_path=str(push.source),
        allow_patterns=list(KEEP),
        commit_message=f"{push.label}: weights and card from the text-to-SQL panel",
    )
    return f"https://huggingface.co/{push.repo}"


def main(argv: list[str] | None = None) -> int:
    # Ahead of the parser, not merely ahead of the work. `parse_args` rejects an unknown
    # flag by printing the offending argument *and its value* to stderr, so a `--token`
    # nobody added reaches the log through the error that refuses it -- which is the same
    # disclosure this check exists to prevent, arriving by the path nobody watches.
    if any(part.startswith("hf_") for part in (argv if argv is not None else sys.argv[1:])):
        raise SystemExit(
            "a token was passed as an argument. It is now in this shell's history and was "
            "visible in `ps` to every process on this box for the length of this run. Treat "
            "it as compromised, revoke it, and pass the replacement in HF_TOKEN."
        )

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--table", required=True, help="what panel_table.py --json-out wrote")
    p.add_argument("--finetune", required=True, help="s2_finetune.json from the training run")
    p.add_argument("--published", required=True, help="publish_panel.py's --out")
    p.add_argument("--repo-prefix", required=True, help="owner/name; each arm appends its slug")
    p.add_argument("--only", default=None, help="comma-separated labels instead of every arm")
    p.add_argument(
        "--adapter-repo",
        default=None,
        help="Hub id of the published LoRA adapter, carried into every card",
    )
    p.add_argument("--include-ceiling", action="store_true", help="also push the merged fine-tune")
    p.add_argument("--private", action="store_true", help="create the repos private")
    p.add_argument(
        "--dry-run", action="store_true", help="check and print the plan, upload nothing"
    )
    p.add_argument("--force", action="store_true", help="push into repos that already hold files")
    args = p.parse_args(argv)

    token = os.environ.get("HF_TOKEN", "")
    if not token and not args.dry_run:
        raise SystemExit("HF_TOKEN is not set, and it is the only place this reads a token from.")

    table = json.loads(Path(args.table).read_text(encoding="utf-8"))
    finetune = json.loads(Path(args.finetune).read_text(encoding="utf-8"))
    labels = (
        [s for s in args.only.split(",") if s]
        if args.only
        else model_cards.publishable(table, include_ceiling=args.include_ceiling)
    )
    if not labels:
        raise SystemExit(f"{args.table} carries no scored arm to publish.")

    pushes = sources(table, finetune, Path(args.published), labels, args.repo_prefix)
    problems = check(pushes, Path(str(finetune.get("output", ""))))
    if not args.dry_run and not args.force:
        problems += occupied(pushes, token)
    if problems:
        print("REFUSING to push:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    for push in pushes:
        card = model_cards.card(
            table,
            push.label,
            finetune,
            repo_prefix=args.repo_prefix,
            adapter_repo=args.adapter_repo,
        )
        size = sum(f.stat().st_size for f in push.source.glob("*.safetensors"))
        if args.dry_run:
            print(
                f"{push.label:>10}  {model_cards.gib(size):>10}  {widths(push.source):>14}  "
                f"{push.source} -> {push.repo}"
            )
            continue
        print(f"{push.label}: pushing {model_cards.gib(size)} to {push.repo}", flush=True)
        print(f"{push.label}: {upload(push, card, token, private=args.private)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
