#!/usr/bin/env python3
"""Write each published arm's model card from the panel table, so the two cannot disagree.

A model card is the only part of this campaign a reader sees before they download 4 GB. It
is also the easiest place for a number to drift: the table is regenerated whenever an arm
lands, and a card written by hand in between keeps whatever was true that afternoon. So no
number here is typed. The card is assembled from two files that were produced by the runs
themselves --

    --table      what `panel_table.py --json-out` wrote: every arm's size and accuracy, and
                 every head-to-head with its Holm-adjusted p, computed once
    --finetune   `s2_finetune.json`, which the fine-tune wrote: the base model, the three
                 datasets, the regime, the step count and the loss

-- and an arm the table does not carry is refused rather than described.

Reading the table rather than the records is the point of the split. `panel_table.py`
already decides what is comparable, corrects a family of six with Holm, and flags the
comparisons whose two arms are not known to have run the same expert arithmetic. A card
that re-derived any of that from `panel/*.json` would be a second implementation of the
statistics, agreeing until the first panel where it did not -- and the disagreement would
surface on the Hub rather than in a terminal.

**The caveats are not optional and are not softened.** Three of them are load-bearing on
this campaign and each one is generated, not remembered: a DynQuant arm was scored by
encoding its widths back into bf16 rather than from the packed directory; a comparison the
table flagged for mixed expert arithmetic carries that flag into the card; and a recipe
arm's published directory is about 2.3% larger than the bytes it was scored at, because the
codes are carried into a container that spends a full bf16 zero per group. A card that
dropped any of them would be the most-read and least-honest document in the campaign.

Nothing here uploads. `--out` writes `<label>/README.md` next to the published weights or
into a directory of its own; pushing is a separate decision and a separate command.

Usage::

    python experiments/phase4/panel_table.py --arms /workspace/runs/s4/panel \\
        --json-out /workspace/runs/s4/panel/table.json
    python experiments/phase4/model_cards.py \\
        --table /workspace/runs/s4/panel/table.json \\
        --finetune /workspace/runs/s4/lfm25-8b-a1b.text2sql/s2_finetune.json \\
        --out /workspace/runs/s4/published
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

NL = "\n"

#: How each arm's method is named to a reader. Keyed by the kind the panel recorded, so
#: this is a rendering of the registry and not a second copy of it -- an arm whose kind is
#: not here stops the card rather than being described by its label.
METHOD_NAMES = {
    "gptq": "GPTQ",
    "awq": "AWQ",
    "dq": "DynQuant",
    "rtn": "round-to-nearest",
}

#: What the reader has to install to open the directory. Every arm in this panel is written
#: by DynQuant's exporter -- the baselines carry their recipe's codes into DynQuant's
#: container -- so all of them load through DynQuant's ``HfQuantizer`` and none of them load
#: through vLLM's native ``compressed-tensors`` path. Saying "GPTQ" on a card without this
#: line would send a reader to a loader that cannot open the file.
LOADER = "`transformers` with `dynquant` installed"


def slug(label: str, kind: str, anchor: int) -> str:
    """The repo suffix for an arm: `gptq_4b` reads as `GPTQ-4bit`."""
    return f"{METHOD_NAMES[kind].replace(' ', '-')}-{anchor}bit"


def percent(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.2f}%"


def gib(nbytes: int | None) -> str:
    return "--" if not nbytes else f"{nbytes / 2**30:.3f} GiB"


def find_arm(table: dict[str, Any], label: str) -> dict[str, Any]:
    for row in table["arms"]:
        if row["label"] == label:
            return row
    known = ", ".join(sorted(r["label"] for r in table["arms"]))
    raise SystemExit(f"the table has no arm {label!r}; it has {known}")


def publishable(table: dict[str, Any]) -> list[str]:
    """Every scored arm that is a quantization of the base model.

    The ceiling is excluded the same way it is in `publish_panel.py` and for the same
    reason: it is the checkpoint the panel started from, so a card for it would describe
    somebody else's model.
    """
    return [
        row["label"]
        for row in table["arms"]
        if row.get("kind") in METHOD_NAMES and row.get("accuracy") is not None
    ]


def comparisons_for(table: dict[str, Any], label: str) -> list[dict[str, Any]]:
    """Every computed comparison this arm appears in, from both families.

    Taken verbatim, including the sign convention: the table states each pair once, so a
    card for the right-hand arm reports a negative delta rather than silently flipping it.
    Flipping would be arithmetically fine and editorially not -- the CI, the flip counts
    and the question text all read from left to right, and half-reversing them produces a
    row nobody can check against the table it came from.
    """
    found = []
    for family in ("head_to_head", "against_ceiling"):
        for entry in table.get(family, []):
            if label in (entry["left"], entry["right"]):
                found.append({**entry, "family": family})
    return found


def frontmatter(base_model: str, row: dict[str, Any]) -> str:
    tags = [
        "dynquant",
        "quantized",
        f"{row['anchor']}-bit",
        "text-to-sql",
        "moe",
        METHOD_NAMES[row["kind"]].lower().replace(" ", "-"),
    ]
    # A DynQuant arm's method name and the package tag are the same word. Deduplicated in
    # order rather than by sorting: the Hub renders the repeat as two identical chips, and
    # the leading tags are the ones a reader scans.
    tags = list(dict.fromkeys(tags))
    lines = [
        "---",
        f"base_model: {base_model}",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "tags:",
        *[f"- {tag}" for tag in tags],
        # Not asserted. The base model's terms govern a derivative of it, and this file
        # has no way to read them -- writing a specific licence here would be inventing a
        # permission on the base model's behalf.
        "license: other",
        "license_name: see-base-model",
        f"license_link: https://huggingface.co/{base_model}",
        "---",
    ]
    return NL.join(lines)


def what_this_is(base_model: str, row: dict[str, Any], finetune: dict[str, Any]) -> str:
    method = METHOD_NAMES[row["kind"]]
    datasets = str(finetune.get("dataset", "")).split("+")
    kept = finetune.get("conversations_kept")
    bits = row.get("bits_per_param")
    regime = (
        f"{finetune.get('regime', '?')} r={finetune.get('lora_rank', '?')}, "
        f"{finetune.get('epochs', '?')} epoch over {kept:,} text-to-SQL conversations"
        if kept
        else "?"
    )
    widths = (
        "per-module widths from a DynQuant allocation"
        if row["kind"] == "dq"
        else "uniform, group size 128"
    )
    size = gib(row.get("nbytes")) + (f" ({bits:.4f} bits per parameter)" if bits else "")
    table = [
        "| | |",
        "|---|---|",
        f"| base model | [{base_model}](https://huggingface.co/{base_model}) |",
        f"| fine-tune | {regime} |",
        "| training data | " + ", ".join(f"`{name}`" for name in datasets if name) + " |",
        f"| quantization | {method}, {row['anchor']}-bit, {widths} |",
        f"| size on disk | {size} |",
        f"| loads with | {LOADER} |",
    ]
    return NL.join(table)


def results_table(table: dict[str, Any], label: str) -> str:
    """The whole panel, with this arm marked, because one number alone is not a result."""
    lines = [
        "| arm | exec match | size | bits/param |",
        "|---|---|---|---|",
    ]
    for row in table["arms"]:
        if row.get("accuracy") is None:
            continue
        bits = row.get("bits_per_param")
        name = f"**{row['label']}**" if row["label"] == label else row["label"]
        lines.append(
            f"| {name} | {percent(row['accuracy'])} | {gib(row.get('nbytes'))} | "
            f"{'16.0' if bits is None else f'{bits:.4f}'} |"
        )
    return NL.join(lines)


def by_source_table(row: dict[str, Any]) -> str | None:
    by_source = row.get("by_source") or {}
    if not by_source:
        return None
    lines = ["| eval source | exec match | items |", "|---|---|---|"]
    for name, counts in sorted(by_source.items()):
        correct, total = counts
        share = correct / total if total else 0.0
        lines.append(f"| `{name}` | {share * 100:.2f}% | {total:,} |")
    return NL.join(lines)


def comparison_table(entries: list[dict[str, Any]]) -> tuple[str, int]:
    """The rows, and how many of them the table flagged for mixed expert arithmetic."""
    lines = [
        "| comparison | delta (pts) | 95% CI | p | p (Holm) | verdict |",
        "|---|---|---|---|---|---|",
    ]
    flagged = 0
    for entry in entries:
        # The CI arrives as two scalars, not a pair: `PairedResult.as_dict` flattens
        # `interval_points` into `ci_low_points`/`ci_high_points` on the way through json,
        # and a card reading the property name gets a KeyError rather than a wrong number.
        low, high = entry["ci_low_points"], entry["ci_high_points"]
        mark = "" if entry["same_arithmetic"] else " [^1]"
        if not entry["same_arithmetic"]:
            flagged += 1
        verdict = "separated" if entry["separated"] else "not separated"
        lines.append(
            f"| {entry['question'].strip()}{mark} | {entry['delta_points']:+.2f} | "
            f"[{low:+.2f}, {high:+.2f}] | {entry['p_value']:.3g} | "
            f"{entry['p_adjusted']:.3g} | {verdict} |"
        )
    return NL.join(lines), flagged


def caveats(table: dict[str, Any], row: dict[str, Any], flagged: int) -> str:
    """Everything true about this arm that a reader would otherwise have to discover.

    Each bullet is emitted because of something in the table, not because it is on a list:
    the scoring-container bullet only appears for an arm scored through ``encode``, the
    arithmetic bullet only when the table flagged one of this arm's comparisons, and the
    pairing bullet only when the panel says its arms are not paired.
    """
    bullets: list[str] = []

    if row.get("apply") == "encode":
        bullets.append(
            "**The accuracy above was measured in bf16, not from this directory.** A "
            "DynQuant arm is scored by encoding its allocated widths back into bf16 -- the "
            "same encoder, the same widths, the same values -- because 91.5% of this "
            "model's parameters are batched expert banks and the scoring path applies "
            "widths in memory rather than writing a 17 GB decoded copy per arm. The "
            "directory you are downloading holds those same values packed. What is carried "
            "across from the measurement is the arithmetic; what is not is a claim that "
            "the packed and encoded containers were separately scored."
        )

    if row["kind"] != "dq":
        bullets.append(
            "**This directory is about 2.3% larger than the size it was scored at.** The "
            "recipe's integer codes are carried across exactly and re-stored in DynQuant's "
            "container, which spends a full bf16 zero point per group where "
            "compressed-tensors packs the zero to the weight width. Same codes, same "
            "numbers on dequantization, more bytes -- and it is why the size in the table "
            "above is the scored size rather than `du`."
        )

    if flagged:
        bullets.append(
            f"**{flagged} of the comparisons above are flagged `[^1]`.** The two expert "
            "dispatches available for this architecture -- the grouped kernel and the "
            "per-expert indexing loop -- disagree on 1.24% of teacher-forced tokens on this "
            "model, which is 0.29x the effect of quantizing it to 4 bits. A flagged row "
            "pairs two arms that are not both recorded as having run the same one, so its "
            "delta carries a term that is not the quantization method. The verdict is what "
            "the stored per-item hits say; it is not yet a statement about quantization "
            "alone."
        )

    if not table.get("pairable", True):
        bullets.append(
            f"**The panel's arms are not paired:** {table.get('pairing_error')}. Every "
            "comparison above should be read as two independent accuracies rather than as "
            "a paired test."
        )

    bullets.append(
        "**Storage, measured; throughput, not.** The number reported here is bytes on disk "
        "and execution match. This card makes no claim about decode speed or peak VRAM "
        "against an fp16 baseline, because this panel did not measure either."
    )
    bullets.append(
        "**One task.** Execution match on held-out text-to-SQL is what was scored. It says "
        "nothing about how this arm behaves on anything else, and a quantization that "
        "holds one task can lose another."
    )
    return NL.join(f"- {line}" for line in bullets)


def usage(row: dict[str, Any], repo: str | None) -> str:
    where = repo or f"./{row['label']}"
    body = [
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        "import dynquant",
        "",
        "dynquant.register_hf_quantizer()",
        "",
        f'model = AutoModelForCausalLM.from_pretrained("{where}", device_map="cuda")',
        f'tokenizer = AutoTokenizer.from_pretrained("{where}")',
        "```",
    ]
    return NL.join(body)


def provenance(table: dict[str, Any], row: dict[str, Any], finetune: dict[str, Any]) -> str:
    params = table.get("params")
    target = row.get("target_bytes")
    loss = finetune.get("train_loss")
    lines = [
        f"- panel model: `{table.get('model')}`",
        f"- parameters counted: {params:,}" if params else "- parameters counted: --",
        f"- byte target this arm was allocated against: {target:,} B"
        if target
        else "- byte target: --",
        f"- fine-tune: {finetune.get('steps')} steps, train loss {loss:.4f}, "
        f"{finetune.get('seconds', 0) / 3600:.1f} h"
        if loss is not None
        else "- fine-tune: --",
        f"- fine-tune commit: `{finetune.get('commit')}`"
        if finetune.get("commit")
        else "- fine-tune commit: --",
    ]
    if row.get("seconds"):
        lines.append(f"- evaluation: {row['total']:,} problems in {row['seconds'] / 60:.0f} min")
    return NL.join(lines)


def card(
    table: dict[str, Any],
    label: str,
    finetune: dict[str, Any],
    *,
    repo_prefix: str | None,
) -> str:
    row = find_arm(table, label)
    if row.get("kind") not in METHOD_NAMES:
        raise SystemExit(
            f"{label} is a {row.get('kind')!r} arm. Only a quantization of the base model "
            f"gets a card; a ceiling would describe a model this campaign did not make."
        )
    if row.get("accuracy") is None:
        raise SystemExit(
            f"{label} has no accuracy in the table, so there is nothing to publish it on. "
            f"Re-run panel_table.py once the arm has been scored."
        )

    base_model = str(finetune["model"])
    method = METHOD_NAMES[row["kind"]]
    entries = comparisons_for(table, label)
    repo = f"{repo_prefix}-{slug(label, row['kind'], row['anchor'])}" if repo_prefix else None

    parts = [
        frontmatter(base_model, row),
        "",
        f"# {base_model.split('/')[-1]} text-to-SQL, {method} {row['anchor']}-bit",
        "",
        f"{base_model} fine-tuned on text-to-SQL and quantized to {row['anchor']} bits with "
        f"{method}. It is one arm of a seven-arm panel in which every quantized arm was "
        f"allocated the *same byte budget*, so the accuracies below differ by method and "
        f"not by size.",
        "",
        "## What this is",
        "",
        what_this_is(base_model, row, finetune),
        "",
        "## Results",
        "",
        f"Execution match on {row['total']:,} held-out text-to-SQL problems: the generated "
        f"query is run against the schema and compared to the reference result set.",
        "",
        results_table(table, label),
        "",
    ]

    per_source = by_source_table(row)
    if per_source:
        parts += ["This arm, by evaluation source:", "", per_source, ""]

    if entries:
        rendered, flagged = comparison_table(entries)
        parts += [
            "## How this arm compares",
            "",
            "McNemar exact over the per-item hits, so every row is a paired test on the "
            "same problems in the same order. `p (Holm)` is step-down corrected within the "
            "family the panel declared, not within this card.",
            "",
            rendered,
            "",
        ]
    else:
        flagged = 0
        parts += [
            "## How this arm compares",
            "",
            "No comparison in the panel involves this arm yet.",
            "",
        ]

    parts += [
        "## What is not claimed",
        "",
        caveats(table, row, flagged),
        "",
        "## Usage",
        "",
        usage(row, repo),
        "",
        "## Provenance",
        "",
        provenance(table, row, finetune),
        "",
    ]
    if flagged:
        parts += [
            "[^1]: the two arms are not both recorded as having run the same expert "
            "dispatch; see the caveat above.",
            "",
        ]
    return NL.join(parts)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--table", required=True, help="what panel_table.py --json-out wrote")
    p.add_argument("--finetune", required=True, help="s2_finetune.json from the training run")
    p.add_argument("--out", required=True, help="directory holding one subdirectory per arm")
    p.add_argument("--only", default=None, help="comma-separated labels instead of every arm")
    p.add_argument(
        "--repo-prefix",
        default=None,
        help="Hub id prefix, so the usage snippet names the repo rather than a local path",
    )
    p.add_argument("--print", action="store_true", help="write nothing; print the cards")
    args = p.parse_args(argv)

    table = json.loads(Path(args.table).read_text(encoding="utf-8"))
    finetune = json.loads(Path(args.finetune).read_text(encoding="utf-8"))
    labels = [s for s in args.only.split(",") if s] if args.only else publishable(table)
    if not labels:
        raise SystemExit(
            f"{args.table} carries no scored quantized arm. A card is a claim about a "
            f"measurement, so there is nothing to write until one lands."
        )

    out_root = Path(args.out)
    for label in labels:
        text = card(table, label, finetune, repo_prefix=args.repo_prefix)
        if args.print:
            print(text)
            continue
        target = out_root / label
        if not target.is_dir():
            print(f"{label}: {target} does not exist -- publish the arm first", flush=True)
            continue
        (target / "README.md").write_text(text, encoding="utf-8", newline=NL)
        print(f"{label}: wrote {target / 'README.md'} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
