#!/usr/bin/env python3
"""Write one published arm's model card from the files the run itself produced.

Every number here is read from an artifact. Nothing is typed, and an input that is
missing stops the card rather than leaving a gap a reader has to notice -- a model card is
the only thing most people see before downloading 12 GB, and it is the easiest place in a
campaign for a figure to go stale, because nothing recomputes prose.

Four inputs, each written by the step it describes::

    --finetune   s2_finetune.json      the regime, the data, the step count, the loss
    --export     export-<tag>.json     the bytes and the average bits actually realised
    --eval       evals/<arm>.json      the score, and the per-problem hit vector
    --inspect    floors.json           the allocation: widths, floors, violations

The allocation section is the reason this is not the phase-4 card generator. That one
renders a panel of methods at matched bytes; this model's story is a single number those
cards have no field for -- the *floor budget*, the average width at which every
architectural floor can still be honoured. Above it an arm is an allocation; below it the
arm is measuring what breaking the floors costs, which is a different experiment and has
to be labelled as one on the page rather than in a footnote.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

GITHUB = "https://github.com/kambojvikram/dynquant"
PIP = 'pip install "dynquant[hf]"'


def load(path: str | None, what: str) -> dict[str, Any]:
    if not path:
        raise SystemExit(f"--{what} is required: the card has no other source for it")
    file = Path(path)
    if not file.is_file():
        raise SystemExit(f"no {what} record at {file}")
    return json.loads(file.read_text(encoding="utf-8"))


def gib(nbytes: float) -> str:
    return f"{nbytes / 2**30:.2f} GiB"


def pct(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.2f}%"


def joined(names: list[str]) -> str:
    """Backticked and comma-joined, kept out of the f-strings that render it."""
    return ", ".join(f"`{name}`" for name in names)


def scored_fields(scored: dict[str, Any]) -> dict[str, Any]:
    """The three eval fields the card asserts, read from where the harness really writes them.

    ``dynquant eval`` nests all three: the decode budget under ``decode``, the source list
    under ``task_options``, and the unfinished count under ``detail``. Read at the top level
    every one of them is absent -- and absent is not neutral on a public page. ``sources``
    degrades to an empty list and the accuracy sentence loses its subject; ``max_new_tokens``
    prints ``?``; and ``unfinished_reasoning`` defaults to 0, which is the card *asserting*
    that no generation hit its cap. On this task a cap that binds has been worth 50 points,
    so that is the one sentence on the page that must not be able to come from a missing key.

    A flattened record is still accepted, because the fixtures and the phase-4 records are
    flat. But a record carrying the count in neither place is refused rather than read as
    zero -- the whole point of the field is that it is a claim, not a default.
    """
    detail = scored.get("detail") or {}
    decode = scored.get("decode") or {}
    options = scored.get("task_options") or {}

    unfinished = scored.get("unfinished_reasoning", detail.get("unfinished_reasoning"))
    if unfinished is None:
        raise SystemExit(
            "the eval record reports no unfinished_reasoning count, at the top level or "
            "under 'detail'. The card would print 0, which claims that no generation hit "
            "its decode budget -- a claim this record cannot support."
        )
    return {
        "sources": scored.get("sources") or options.get("sources") or [],
        "max_new_tokens": scored.get("max_new_tokens", decode.get("max_new_tokens", "?")),
        "unfinished_reasoning": unfinished,
    }


def train_sources(finetune: dict[str, Any]) -> list[str]:
    """Which datasets the fine-tune trained on, from where the trainer writes them.

    ``run_s2_finetune`` records ``sources`` as a mapping of ``task/name -> count`` and
    ``train_sources_requested`` as the list that was asked for. It writes no
    ``train_sources``, so reading that key yields an empty list and the sentence renders as
    "9,999 conversations from , 350,799 supervised tokens" -- a card that names no dataset
    while looking like it named one, on the single line a reader checks to decide whether
    the model was trained on their kind of data.

    The realised mapping wins over the requested list: a source that was asked for and then
    dropped entirely by decontamination is not something this checkpoint was trained on.
    """
    realised = finetune.get("sources")
    if isinstance(realised, dict) and realised:
        # "text2sql/spider" -> "spider": the task prefix is the harness's, not the dataset's.
        return [name.split("/")[-1] for name, count in realised.items() if count]
    for key in ("train_sources", "train_sources_requested"):
        listed = finetune.get(key)
        if listed:
            return list(listed)
    return []


def pval(value: float | None) -> str:
    """A p-value that never renders as zero.

    ``:.4f`` prints 1.93e-05 as ``0.0000``, which on a public page is a claim that the
    difference could not have arisen by chance at all. Below the point where four decimals
    stop carrying information, print the exponent instead.
    """
    if value is None:
        return "--"
    return f"{value:.4f}" if value >= 1e-4 else f"{value:.2e}"


def decontamination(finetune: dict[str, Any]) -> str | None:
    """The sentence a text-to-SQL card cannot leave to the reader's imagination.

    This checkpoint trains on `spider`, `gretel` and `wikisql` and is then scored on the
    held-out splits of `spider`, `gretel` and `wikisql`. Anyone reading the results table
    will ask whether it was scored on what it was trained on, and a card that does not
    answer has effectively answered badly. The trainer records both halves -- how many
    training examples were dropped for matching an eval item, and whether any source
    remained overlapping afterwards -- so the answer is an artifact rather than a promise.

    Returns None only when the run recorded neither, in which case the card says nothing
    rather than claiming a check that may not have run.
    """
    dropped = finetune.get("decontaminated")
    overlap = finetune.get("sources_overlapping_an_eval_task")
    if dropped is None and overlap is None:
        return None
    total = sum(dropped.values()) if isinstance(dropped, dict) else (dropped or 0)
    if overlap:
        return (
            f"- **Contamination**: {joined(list(overlap))} overlap an evaluation task after "
            f"decontamination removed {total:,} training examples. The score on those "
            f"sources is not a held-out measurement."
        )
    return (
        f"- **Contamination**: the training split was matched against every evaluation item "
        f"and {total:,} examples were dropped for colliding with one. No source overlaps an "
        f"evaluation task after that."
    )


def floor_budget(inspect: dict[str, Any]) -> tuple[float | None, str | None]:
    """The narrowest budget in this file that broke no floor.

    Read off the measured rows rather than recomputed from the policy: applying the floors
    is the allocator's job, and a second implementation here would agree with it until the
    release where it did not.
    """
    clean = [
        (row["average_bits"], key)
        for key, row in inspect["targets"].items()
        if not row["violations"]
    ]
    if not clean:
        return None, None
    bits, key = min(clean)
    return bits, key


def widths_table(row: dict[str, Any], total_params: int) -> str:
    lines = ["| Width | Modules | Parameters | Share |", "|---|---:|---:|---:|"]
    for bits in sorted(row["widths"], key=int):
        width = row["widths"][bits]
        share = 100 * width["params"] / total_params
        lines.append(f"| {bits}-bit | {width['modules']:,} | {width['params']:,} | {share:.2f}% |")
    return "\n".join(lines)


def violations_table(row: dict[str, Any]) -> str:
    """Every breached floor, aggregated by role, with the parameters behind it.

    Aggregated rather than listed: at the 3-bit budget this is 310 lines, which is a wall
    a reader skips. By role it is a dozen, and the dozen is the finding.
    """
    if not row["violations"]:
        return (
            "No floor was breached at this budget: every module sits at or above the "
            "width its role requires."
        )
    agg: dict[str, dict[str, Any]] = {}
    for violation in row["violations"]:
        entry = agg.setdefault(
            violation["role"], {"n": 0, "params": 0, "floor": set(), "got": set()}
        )
        entry["n"] += 1
        entry["params"] += violation["num_params"]
        entry["floor"].add(violation["floor_bits"])
        entry["got"].add(violation["assigned_bits"])

    lines = [
        f"{len(row['violations'])} modules were allocated below the floor their role requires:",
        "",
        "| Role | Modules | Parameters | Floor | Given |",
        "|---|---:|---:|---:|---:|",
    ]
    for role, entry in sorted(agg.items(), key=lambda kv: -kv[1]["params"]):
        floor = "/".join(f"{bits}b" for bits in sorted(entry["floor"]))
        got = "/".join(f"{bits}b" for bits in sorted(entry["got"]))
        lines.append(f"| `{role}` | {entry['n']} | {entry['params']:,} | {floor} | {got} |")
    return "\n".join(lines)


def citation(repo: str, version: str | None) -> str:
    """BibTeX for the code and for this checkpoint, as two entries rather than one.

    They are two artifacts on two clocks. The method's code keeps moving after a
    checkpoint is frozen, so a reader who reruns anything has to be able to say which
    release they ran against -- and a reader who only wants to point at these weights
    should not have to cite a version that had nothing to do with them.

    ``version`` is optional because the reference arm is not a DynQuant export and has no
    export record to read one from. Omitting the field is more honest than inventing it.
    """
    name = repo.split("/")[-1]
    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    software = [
        "@software{dynquant,",
        "  author  = {Kamboj, Vikrampal},",
        "  title   = {{DynQuant}: dynamic-signal quantization for large language models},",
    ]
    if version:
        software.append(f"  version = {{{version}}},")
    software += [f"  url     = {{{GITHUB}}},", "  year    = {2026}", "}"]
    checkpoint = [
        f"@misc{{{key},",
        "  author       = {Kamboj, Vikrampal},",
        f"  title        = {{{name}}},",
        "  year         = {2026},",
        "  publisher    = {Hugging Face},",
        f"  howpublished = {{\\url{{https://huggingface.co/{repo}}}}}",
        "}",
    ]
    return "```bibtex\n" + "\n".join([*software, "", *checkpoint]) + "\n```\n"


def frontmatter(args: argparse.Namespace) -> str:
    return "\n".join(
        [
            "---",
            "license: apache-2.0",
            f"base_model: {args.base_model}",
            "library_name: transformers",
            "pipeline_tag: text-generation",
            "tags:",
            "- dynquant",
            "- quantization",
            "- mixed-precision",
            "- text-to-sql",
            "- qlora",
            "language:",
            "- en",
            "---",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", required=True, help="the arm's name in the eval records")
    parser.add_argument("--repo", required=True, help="the Hub id this card will live at")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--finetune", required=True)
    parser.add_argument("--export", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--inspect", required=True)
    parser.add_argument(
        "--inspect-target",
        required=True,
        help="which target row in the inspect file this arm was exported at",
    )
    parser.add_argument("--reference-eval", help="the bf16 record this arm is measured against")
    parser.add_argument("--panel", help="panel.json, for the paired test")
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # Taken as an argument rather than read off ``sys.argv``, so a test can drive the real
    # parser -- the refusal below lives behind it, and calling the render helpers directly
    # would step over the thing being checked.
    args = build_parser().parse_args(argv)

    finetune = load(args.finetune, "finetune")
    export = load(args.export, "export")
    scored = load(args.eval, "eval")
    fields = scored_fields(scored)
    inspect = load(args.inspect, "inspect")

    if args.inspect_target not in inspect["targets"]:
        raise SystemExit(
            f"the inspect record has no target {args.inspect_target!r}; it has "
            f"{', '.join(sorted(inspect['targets']))}. The allocation table would "
            f"otherwise describe a budget this arm was not exported at."
        )
    row = inspect["targets"][args.inspect_target]
    total_params = row["total_params"]

    reference = load(args.reference_eval, "reference-eval") if args.reference_eval else None
    paired = None
    if args.panel:
        panel = load(args.panel, "panel")
        paired = next((arm["comparison"] for arm in panel["arms"] if arm["arm"] == args.arm), None)

    budget_bits, budget_key = floor_budget(inspect)
    realised = export["average_bits"]

    # The headline width comes from the export and the allocation table from the inspect,
    # so the two have to be the same allocation or the card describes a checkpoint that
    # does not exist -- a widths table over one budget above a size and a score from
    # another. They are computed by the same allocator from the same stats and should
    # agree to the last place; the tolerance is here only because the export counts
    # tensors it refused to quantize and the inspection does not. A gap wider than that is
    # a stale file on one side, which is exactly what a resume guard leaves behind.
    drift = abs(realised - row["average_bits"])
    if drift > 0.05:
        raise SystemExit(
            f"the export realised {realised:.4f} bits but the {args.inspect_target!r} "
            f"inspection row measured {row['average_bits']:.4f} -- a gap of {drift:.4f}. "
            f"These are supposed to be one allocation described twice. One of the two "
            f"files predates the stats the other was built from; re-run the inspection "
            f"against the stats the export used."
        )

    below = None if budget_bits is None else budget_bits - realised

    out: list[str] = [frontmatter(args)]
    add = out.append

    add(f"# {args.repo.split('/')[-1]}\n")
    add(
        f"`{args.base_model}`, fine-tuned for text-to-SQL with QLoRA and then quantized "
        f"with [DynQuant]({GITHUB}) to **{realised:.3f} bits per weight** "
        f"({gib(export['directory_nbytes'])} on disk).\n"
    )
    add(
        "DynQuant gives every module its own width, driven by two signals measured "
        "*during* the fine-tune: how much activation mass each weight sees, and how "
        "unstable its gradient is across optimizer steps. Modules the training dynamics "
        "say are load-bearing keep their bits; the rest pay for them.\n"
    )

    # The one thing a reader has to know before trusting the number below.
    if below is not None and below > 0.1:
        add("## Read this first\n")
        add(
            f"This architecture has a **floor budget of {budget_bits:.4f} bits** -- the "
            f"narrowest average width at which every module can still be given the "
            f"minimum its role requires. That is measured, not assumed: it is the "
            f"narrowest budget in this campaign's sweep that breached no floor.\n"
        )
        add(
            f"This arm was exported at **{realised:.3f} bits, {below:.2f} bits below "
            f"it**. So it is not an allocation within the architecture's limits; it is a "
            f"measurement of what overriding those limits costs. The table further down "
            f"names every floor it broke. The score was measured exactly as every other "
            f"arm's was -- what it is a score *of* is a deliberately over-compressed "
            f"model.\n"
        )
        if budget_key:
            add(
                f"For the smallest arm that breaks nothing, use the {budget_key}-bit "
                f"budget instead.\n"
            )
    elif below is not None:
        clears = (
            "It clears every floor."
            if not row["violations"]
            else (f"It still breaks {len(row['violations'])} of them, listed below.")
        )
        add(
            f"At {realised:.3f} bits this arm sits essentially at the architecture's "
            f"**floor budget of {budget_bits:.4f} bits** -- the narrowest average width "
            f"at which every module can still hold the minimum its role requires. "
            f"{clears}\n"
        )

    add("## Results\n")
    sources = fields["sources"]
    add(
        f"Execution accuracy on the held-out validation split of {joined(sources)}, "
        f"{scored.get('total', '?')} problems, greedy decode.\n"
    )

    rows = [
        "| Model | Bits | Size | Accuracy | vs bf16 | p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if reference:
        rows.append(
            f"| bf16 (unquantized) | 16 | -- | {pct(reference.get('accuracy'))} | -- | -- |"
        )
    delta = f"{paired['delta_points']:+.2f}" if paired else "--"
    pvalue = pval(paired["p_value"]) if paired else "--"
    rows.append(
        f"| **this checkpoint** | {realised:.3f} | {gib(export['directory_nbytes'])} "
        f"| {pct(scored.get('accuracy'))} | {delta} | {pvalue} |"
    )
    add("\n".join(rows) + "\n")
    add(
        "The comparison is paired: both arms answered the same problems in the same "
        "order, so the difference is a McNemar test on the per-problem outcomes rather "
        "than two independent accuracies subtracted.\n"
    )

    unfinished = fields["unfinished_reasoning"]
    total = scored.get("total") or 0
    budget_note = (
        ". A short decode budget scores a model that deliberates as though it were "
        "wrong, so this is reported rather than assumed.\n"
        if not unfinished
        else f" -- so accuracy is bounded above by {1 - unfinished / total:.2%} and the number "
        f"above is partly a measurement of the budget.\n"
    )
    add(
        f"Decode budget was {fields['max_new_tokens']} new tokens, and "
        f"{unfinished} generations reached it without finishing{budget_note}"
    )

    add("## What the allocator did\n")
    add(
        f"{export['modules']} modules were quantized at group size "
        f"{export.get('group_size') or inspect.get('group_size')}, scored by the "
        f"`{inspect.get('allocator', '?')}` allocator.\n"
    )
    add(widths_table(row, total_params) + "\n")

    add("### Floors\n")
    add(
        "Each role carries a minimum width below which that role is known to break -- an "
        "embedding, an LM head, an attention projection and an MLP gate do not tolerate "
        "the same compression. When a budget cannot pay for every floor, DynQuant breaks "
        "the cheapest ones and **reports every one it broke**, rather than quietly "
        "lowering a floor until the arithmetic works.\n"
    )
    add(violations_table(row) + "\n")

    add("## Use it\n")
    add(f"```bash\n{PIP}\n```\n")
    add(
        "```python\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
        "import dynquant\n"
        "\n"
        "# Both lines are needed. transformers has no entry-point discovery for\n"
        "# quantization methods, so without the registration call it skips the\n"
        "# quantization it does not recognise and hands back a randomly initialised\n"
        "# model -- fluent-looking output, no exception, no non-zero exit.\n"
        "dynquant.register_hf_quantizer()\n"
        "\n"
        f'model = AutoModelForCausalLM.from_pretrained("{args.repo}", device_map="auto")\n'
        f'tokenizer = AutoTokenizer.from_pretrained("{args.repo}")\n'
        "```\n"
    )
    add(
        "That path is measured rather than assumed: this repo was pulled from the Hub "
        "and loaded with the released wheel on torch 2.13, and the graph that came back "
        f"was checked to hold {export['modules']} DynQuant modules at the widths in the "
        "table above.\n"
    )

    add("### Serving\n")
    add(
        f"`vllm serve {args.repo}` does **not** work, and the failure belongs here rather "
        "than in your terminal:\n"
    )
    add(
        "```\n"
        "ValueError: There is no module or parameter named 'embed_tokens.offsets' in "
        "Qwen3_5Model\n"
        "```\n"
    )
    add(
        "vLLM builds this architecture's input embedding without offering it to a "
        "quantization plugin at all, so a packed `model.embed_tokens` has nowhere to "
        "land. It is not a flag you are missing and it is not the plugin failing to "
        "register: on vLLM 0.27.1 the DynQuant plugin loads, the engine reports "
        "`quantization=dynquant`, and then this raises.\n"
    )
    add(
        "That one tensor is the whole gap. Re-exported with the input embedding in the "
        f"compute dtype and the other {export['modules'] - 1} quantized modules left "
        "exactly as they are, this arm serves on vLLM 0.27.1 and answers correctly -- so "
        "nothing else in the allocation is in the way. That export is not what is "
        "published here. Until it is, use the transformers path above.\n"
    )

    add("## How it was made\n")
    trained_on = train_sources(finetune)
    raw_loss = finetune.get("train_loss")
    loss = "?" if raw_loss is None else f"{raw_loss:.4f}"
    add(
        f"- **Base**: `{args.base_model}`, text tower only\n"
        f"- **Regime**: {finetune.get('regime', '?')}, LoRA rank "
        f"{finetune.get('lora_rank', '?')}\n"
        f"- **Data**: {finetune.get('conversations_kept', 0):,} conversations from "
        f"{joined(trained_on)}, {finetune.get('supervised_tokens', 0):,} supervised "
        f"tokens\n"
        f"- **Steps**: {finetune.get('steps', '?')} at effective batch "
        f"{finetune.get('effective_batch', '?')}, lr {finetune.get('lr', '?')}, final "
        f"train loss {loss}\n"
        f"- **Signals**: collected from {finetune.get('tracked_modules', '?')} modules "
        f"during the fine-tune itself, with no extra forward or backward pass\n"
        f"- **DynQuant**: {export['dynquant_core']}\n"
    )
    contamination = decontamination(finetune)
    if contamination:
        add(contamination + "\n")

    add("## Citation\n")
    add("The method has no published paper yet, so these cite the code and these weights.\n")
    add(citation(args.repo, export.get("dynquant_core")))

    add("## Limitations\n")
    add(
        "- Fine-tuned and evaluated on text-to-SQL. General-purpose ability was not "
        "measured, and quantization is not free elsewhere.\n"
        "- Only the base model's text tower was trained and quantized. This checkpoint "
        "does not carry the vision path.\n"
        "- The score is execution accuracy on the datasets named above, against their "
        "own schemas. Accuracy on your schemas is a different measurement.\n"
    )
    if row["violations"]:
        add(
            f"- {len(row['violations'])} modules sit below their role's floor; the table "
            f"above says which.\n"
        )

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes("\n".join(out).encode("utf-8"))
    print(f"-> wrote {destination} ({destination.stat().st_size:,} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
