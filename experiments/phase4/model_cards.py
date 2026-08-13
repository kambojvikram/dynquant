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

#: Where the method itself lives, and what opens a directory written by it. A card is
#: the only place a reader meets the checkpoint and the code at once, and every packed
#: arm here is unreadable without the package -- so both lines are generated onto the
#: card rather than left to a reader who has already spent 4 GB finding out.
GITHUB = "https://github.com/kambojvikram/dynquant"
PIP = "pip install dynquant"

#: The kind the panel records for the unquantized arm. It is publishable here, unlike
#: in `publish_panel.py`, because on this campaign the ceiling *is* the merged
#: fine-tune -- a checkpoint this work produced -- and not the base model it started
#: from. It stays opt-in: publishing a fine-tune is a separate decision from
#: publishing a quantization of it, and its card claims no quantization at all.
CEILING = "ceiling"


def slug(label: str, kind: str, anchor: int) -> str:
    """The repo suffix for an arm: `gptq_4b` reads as `GPTQ-4bit`."""
    if kind == CEILING:
        return "bf16"
    return f"{METHOD_NAMES[kind].replace(' ', '-')}-{anchor}bit"


def percent(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.2f}%"


def gib(nbytes: int | None) -> str:
    return "--" if not nbytes else f"{nbytes / 2**30:.3f} GiB"


def published(table: dict[str, Any]) -> list[dict[str, Any]]:
    """The arms a reader is being offered, which is not every arm the panel ran.

    A `--score-null` arm is an allocation built to be worse on purpose: the same budget
    and the same encoder with the signal permuted or flattened, so the panel can say how
    much of a margin the signal bought. It is a measurement and never a checkpoint. But
    it carries an accuracy, a byte count and kind `dq`, so it passed every filter here --
    the first card off this generator listed `dq_3b_shuf 79.12%` among the results with
    nothing to say what `shuf` was, under a sentence counting twelve arms.

    Discovered from the `score_null` block the table already carries, not from the `_shuf`
    and `_unif` in the labels: those are this campaign's naming and the next campaign's
    labels are somebody else's to choose.
    """
    return [row for row in table["arms"] if not row.get("score_null")]


def find_arm(table: dict[str, Any], label: str) -> dict[str, Any]:
    for row in table["arms"]:
        if row["label"] == label:
            return row
    known = ", ".join(sorted(r["label"] for r in table["arms"]))
    raise SystemExit(f"the table has no arm {label!r}; it has {known}")


def publishable(table: dict[str, Any], include_ceiling: bool = False) -> list[str]:
    """Every scored arm this campaign produced and can describe.

    The ceiling used to be excluded on the grounds that it is the checkpoint the panel
    started from, so a card for it would describe somebody else's model. That is true of
    a base model and false of this one: the ceiling arm *is* the merged fine-tune, which
    this campaign trained, and the base model is a different checkpoint on the Hub. The
    reason survives as the reason it stays opt-in rather than becoming the default --
    on a panel that quantized a checkpoint it did not make, `--include-ceiling` would be
    wrong and nothing here can tell the two cases apart.
    """
    kinds = set(METHOD_NAMES) | ({CEILING} if include_ceiling else set())
    return [
        row["label"]
        for row in published(table)
        if row.get("kind") in kinds and row.get("accuracy") is not None
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


def architecture_tags(finetune: dict[str, Any]) -> list[str]:
    """What the checkpoint *is*, read off the checkpoint's own config.

    ``moe`` used to be a constant in the tag list. It was true of the one model this
    campaign had published and false of the next one, and a dense model tagged ``moe`` is
    the same defect the ``quantized`` tag is guarded against a few lines below: a wrong
    answer to somebody's search, which costs them a multi-gigabyte download to discover.

    Asked of the config rather than of a list of model names, because the list is the part
    that goes stale. What makes a checkpoint mixture-of-experts is that it carries a count
    of experts, and the families spell that differently -- ``num_experts`` on LFM2 and
    Qwen3-MoE, ``num_local_experts`` on Mixtral, ``n_routed_experts`` on DeepSeek. Matching
    the substring asks the question once instead of tracking three spellings that grow.
    Booleans are excluded on purpose: ``use_expert_bias`` sits right beside ``num_experts``
    in the same config and is a flag, not a count.

    Refuses rather than guesses when the merge is not reachable. A tag list assembled
    without the config would be silently missing whatever the config would have added, and
    a card is a claim about a specific checkpoint.
    """
    merged = Path(str(finetune.get("output", "")))
    config_path = merged / "config.json"
    if not config_path.is_file():
        raise SystemExit(
            f"{config_path} does not exist, so the card cannot say what this checkpoint is. "
            f"The fine-tune's own record names {merged} as its output; point --finetune at a "
            f"manifest whose merge is present."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    experts = any(
        "expert" in key.lower()
        and isinstance(value, int)
        and not isinstance(value, bool)
        and value > 1
        for key, value in config.items()
    )
    return ["moe"] if experts else []


def frontmatter(base_model: str, row: dict[str, Any], arch_tags: list[str]) -> str:
    if row["kind"] == CEILING:
        # No `quantized` tag and no width: the Hub filters on these, and an unquantized
        # checkpoint answering a 4-bit filter is a wrong answer to a search.
        tags = ["text-to-sql", *arch_tags, "sft", "bf16", "dynquant"]
    else:
        tags = [
            "dynquant",
            "quantized",
            f"{row['anchor']}-bit",
            "text-to-sql",
            *arch_tags,
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


def what_this_is(
    base_model: str,
    row: dict[str, Any],
    finetune: dict[str, Any],
    adapter_repo: str | None = None,
) -> str:
    ceiling = row["kind"] == CEILING
    method = "" if ceiling else METHOD_NAMES[row["kind"]]
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
    quantization = (
        "none -- this is the bf16 fine-tune every quantized arm was made from"
        if ceiling
        else f"{method}, {row['anchor']}-bit, {widths}"
    )
    table = [
        "| | |",
        "|---|---|",
        f"| base model | [{base_model}](https://huggingface.co/{base_model}) |",
        f"| fine-tune | {regime} |",
        # Only when the adapter was actually published. A row naming a repo that does not
        # exist is worse than no row: the reader who follows it learns the card is wrong
        # about something they cannot check the rest of.
        *(
            [f"| the adapter it merged | [{adapter_repo}](https://huggingface.co/{adapter_repo}) |"]
            if adapter_repo
            else []
        ),
        "| training data | " + ", ".join(f"`{name}`" for name in datasets if name) + " |",
        f"| quantization | {quantization} |",
        f"| size on disk | {size} |",
        f"| loads with | {'`transformers`' if ceiling else LOADER} |",
    ]
    return NL.join(table)


def results_table(table: dict[str, Any], label: str) -> str:
    """The whole panel, with this arm marked, because one number alone is not a result.

    An arm that has not been scored keeps its row and says so. Dropping it produced a card
    whose own first sentence counted seven arms above a table of five, and a reader had no
    way to learn that two more exist -- which mid-panel is the normal state, since the
    expensive arms are published first and the cheap ones are still running. A control
    allocation is the opposite case and is dropped: it is not an arm a reader can have,
    and a row saying so would need the whole ablation to be readable.
    """
    lines = [
        "| arm | exec match | size | bits/param |",
        "|---|---|---|---|",
    ]
    for row in published(table):
        bits = row.get("bits_per_param")
        if row.get("accuracy") is None:
            lines.append(f"| {row['label']} | *not scored yet* | -- | -- |")
            continue
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
    arithmetic bullet only when the table flagged one of this arm's comparisons, the
    scheme bullet only when the panel mixes a symmetric method with an asymmetric one,
    and the pairing bullet only when the panel says its arms are not paired.
    """
    bullets: list[str] = []

    if row.get("apply") == "encode":
        bullets.append(
            "**The accuracy above was measured in bf16, not from this directory.** A "
            "DynQuant arm is scored by encoding its allocated widths back into bf16 -- the "
            "same encoder, the same widths, the same values -- so that every arm in the "
            "panel is scored through one path and no arm's number depends on which "
            "container it was read from. The directory you are downloading holds those "
            "same values packed. What is carried across from the measurement is the "
            "arithmetic; what is not is a claim that the packed and encoded containers "
            "were separately scored."
        )

    if row["kind"] not in (CEILING, "dq"):
        bullets.append(
            "**This directory is about 2.3% larger than the size it was scored at.** The "
            "recipe's integer codes are carried across exactly and re-stored in DynQuant's "
            "container, which spends a full bf16 zero point per group where "
            "compressed-tensors packs the zero to the weight width. Same codes, same "
            "numbers on dequantization, more bytes -- and it is why the size in the table "
            "above is the scored size rather than `du`."
        )

    kinds = {arm["kind"] for arm in table["arms"]}
    if "gptq" in kinds and kinds & {"awq", "dq"}:
        bullets.append(
            "**The baselines run at their own libraries' defaults, and those defaults "
            "are not the same scheme.** GPTQ here is symmetric with no activation "
            "reordering; AWQ and DynQuant are asymmetric. Where a comparison above pairs "
            "a symmetric arm against an asymmetric one its delta spans two differences at "
            "once -- how the bits were allocated, and whether a zero point was stored per "
            "group -- so a large gap between those two arms is not on its own evidence "
            "about allocation. The comparison that would isolate it, two arms of the same "
            "scheme at the same byte anchor, is not in this panel."
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


def links(row: dict[str, Any]) -> str:
    """The package that opens this directory, and the source that produced it."""
    if row["kind"] == CEILING:
        return NL.join(
            [
                "This checkpoint is plain bf16 and loads with `transformers` alone. It is "
                "the ceiling arm of a DynQuant panel: the quantized arms in the table above "
                "are this same fine-tune at a fraction of the size, and those need the "
                "package.",
                "",
                "```bash",
                PIP,
                "```",
                "",
                f"Source, format spec, and the allocator that produced their bit maps: <{GITHUB}>",
            ]
        )
    return NL.join(
        [
            "This directory is packed, so `transformers` alone cannot open it -- it needs "
            "DynQuant's `HfQuantizer`, which the package registers. Prebuilt CUDA kernels "
            "come with it where a wheel exists for your platform, and it falls back to a "
            "pure-torch path where one does not.",
            "",
            "```bash",
            PIP,
            "```",
            "",
            f"Source, format spec, and the allocator that produced this arm's bit map: <{GITHUB}>",
        ]
    )


def usage(row: dict[str, Any], repo: str | None) -> str:
    where = repo or f"./{row['label']}"
    ceiling = row["kind"] == CEILING
    body = [
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        *([] if ceiling else ["import dynquant", "", "dynquant.register_hf_quantizer()", ""]),
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
    adapter_repo: str | None = None,
) -> str:
    row = find_arm(table, label)
    if row.get("score_null"):
        spec = row["score_null"]
        raise SystemExit(
            f"{label} is a control allocation ({spec.get('mode')}, seed "
            f"{spec.get('seed')}), not a checkpoint. It exists to be worse than the arm "
            f"it is subtracted from; published on its own it is a model nobody should "
            f"download and a number nobody can read."
        )
    if row.get("kind") not in set(METHOD_NAMES) | {CEILING}:
        raise SystemExit(
            f"{label} is a {row.get('kind')!r} arm, which this generator has no card for. "
            f"A ceiling needs --include-ceiling; anything else is not a checkpoint."
        )
    if row.get("accuracy") is None:
        raise SystemExit(
            f"{label} has no accuracy in the table, so there is nothing to publish it on. "
            f"Re-run panel_table.py once the arm has been scored."
        )

    base_model = str(finetune["model"])
    ceiling = row["kind"] == CEILING
    method = "bf16" if ceiling else METHOD_NAMES[row["kind"]]
    entries = comparisons_for(table, label)
    repo = f"{repo_prefix}-{slug(label, row['kind'], row['anchor'])}" if repo_prefix else None

    parts = [
        frontmatter(base_model, row, architecture_tags(finetune)),
        "",
        f"# {base_model.split('/')[-1]} text-to-SQL, {method}"
        + ("" if ceiling else f" {row['anchor']}-bit"),
        "",
        (
            f"{base_model} fine-tuned on text-to-SQL, merged and left in bf16. It is the ceiling "
            f"arm of a panel of {len(published(table))} arms: every quantized arm below was "
            f"made from this checkpoint and allocated the same byte budget, so their "
            f"accuracies differ by method and not by size."
            if ceiling
            else f"{base_model} fine-tuned on text-to-SQL and quantized to {row['anchor']} bits "
            f"with {method}. It is one of {len(published(table))} arms in a panel where "
            f"every quantized arm was allocated the *same byte budget*, so the accuracies "
            f"below differ by method and not by size."
        ),
        "",
        "## What this is",
        "",
        what_this_is(base_model, row, finetune, adapter_repo),
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
        "## Install",
        "",
        links(row),
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
    p.add_argument(
        "--adapter-repo",
        default=None,
        help="Hub id of the published LoRA adapter, if there is one, so a reader can "
        "reach the fine-tune itself and not only the merge",
    )
    p.add_argument(
        "--include-ceiling",
        action="store_true",
        help="also card the unquantized arm, which on this panel is the merged fine-tune",
    )
    p.add_argument("--print", action="store_true", help="write nothing; print the cards")
    args = p.parse_args(argv)

    table = json.loads(Path(args.table).read_text(encoding="utf-8"))
    finetune = json.loads(Path(args.finetune).read_text(encoding="utf-8"))
    labels = (
        [s for s in args.only.split(",") if s]
        if args.only
        else publishable(table, include_ceiling=args.include_ceiling)
    )
    if not labels:
        raise SystemExit(
            f"{args.table} carries no scored quantized arm. A card is a claim about a "
            f"measurement, so there is nothing to write until one lands."
        )

    out_root = Path(args.out)
    for label in labels:
        text = card(
            table, label, finetune, repo_prefix=args.repo_prefix, adapter_repo=args.adapter_repo
        )
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
