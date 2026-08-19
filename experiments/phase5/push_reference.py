"""Publish the unquantized bf16 arm -- the yardstick the two DynQuant repos are measured against.

This is a separate script from [the arm pusher][1] rather than a flag on it, because every
guard that file runs is about quantization: it reads ``quantization_config.modules`` to compare
allocations, and it refuses two arms whose maps agree. A bf16 merge has no map. Bolting a
"reference" mode onto that file would mean threading ``None`` through each of those checks and
quietly turning all of them off for this arm -- which is the arm most likely to be pointed at
the wrong directory, since it is the only one whose source is not a DynQuant export.

So the guards here are the inverse ones, and they are the mistakes that would survive
everything upstream and become permanent the moment the upload starts:

* a source that **is** quantized, published under a name that promises bf16 (pointing
  ``--source`` at ``q-4p0`` produces a working repo whose every number is wrong by an arm),
* a comparison table built from a panel that scored a different reference, and
* the 49.83 GB shard ``save_pretrained`` leaves behind, which lands *under* the Hub's 50 GB
  ceiling and so is never caught by the limit -- it has to be caught by the recommendation.

[1]: ./push_to_hub.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import model_card  # noqa: E402


def _phase4() -> Any:
    """phase4's pusher, loaded by path -- the same collision this directory's other pusher hits.

    Several files in this tree are called ``push_to_hub.py``. Under a plain import the winner
    is whichever directory sits earlier on ``sys.path``, and nothing raises when the wrong one
    wins; the upload simply runs through another phase's helpers.
    """
    path = HERE.parent / "phase4" / "push_to_hub.py"
    spec = importlib.util.spec_from_file_location("phase4_push_to_hub", path)
    if spec is None or spec.loader is None:  # pragma: no cover -- the file is in the tree
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase4_push_to_hub"] = module
    spec.loader.exec_module(module)
    return module


phase4 = _phase4()

#: The Hub's per-file ceiling.
MAX_FILE_NBYTES = 50 * 1000**3

#: What a shard should be, as opposed to what it is allowed to be. ``save_pretrained`` on a
#: 27B model in bf16 emits one 49.83 GB shard, which clears MAX_FILE_NBYTES with 0.17 GB to
#: spare and then makes every download a single un-resumable file. The gap between these two
#: numbers is the whole reason this check is separate from the limit.
RECOMMENDED_SHARD_NBYTES = 5 * 1000**3


@dataclass(frozen=True)
class Peer:
    """A quantized arm of the same panel, as it will be linked from this card."""

    label: str
    repo: str
    export: str | None = None

    def nbytes(self) -> int | None:
        """That arm's size on disk, from the record its own export wrote.

        Taken from the export record rather than measured here, because the arm's directory
        is not on this machine at card time and a size the card invents is the one number a
        reader cannot check against anything.
        """
        if not self.export:
            return None
        record = json.loads(Path(self.export).read_text(encoding="utf-8"))
        return record.get("directory_nbytes")


def peers(values: list[str]) -> list[Peer]:
    out = []
    for value in values:
        label, _, rest = value.partition("=")
        repo, _, export = rest.partition("=")
        if not label or not repo:
            raise SystemExit(f"--peer wants label=repo[=export.json], got {value!r}")
        out.append(Peer(label=label, repo=repo, export=export or None))
    return out


def quantized(source: Path) -> bool:
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    return bool(config.get("quantization_config"))


def tensor_names(directory: Path) -> set[str] | None:
    """Every tensor the directory claims, from its index. ``None`` if there is no index."""
    index = directory / "model.safetensors.index.json"
    if not index.is_file():
        return None
    return set(json.loads(index.read_text(encoding="utf-8"))["weight_map"])


def same_weights(source: Path, scored: dict[str, Any]) -> str | None:
    """That the directory being published holds the weights the eval record scored.

    The obvious version of this check compares paths, and it is wrong here: the merge was
    resharded before upload, so the published directory legitimately is not the scored one.
    What has to match is the *content* -- the same tensors, the same total bytes. Comparing
    names would either refuse a correct publish or, once someone relaxed it, pass a directory
    that has nothing to do with the number printed above it.

    Skipped, with a note, when the scored directory is no longer on this machine: an absent
    original is not evidence of a mismatch, and refusing on it would make the check something
    people route around rather than keep.
    """
    original = Path(scored["model"]) if scored.get("model") else None
    if original is None or not original.is_dir():
        return None
    here, there = tensor_names(source), tensor_names(original)
    if here is None or there is None:
        return None
    if here != there:
        missing, extra = sorted(there - here)[:3], sorted(here - there)[:3]
        return (
            f"{source} does not hold the tensors {original} was scored with: "
            f"{len(there - here)} absent {missing}, {len(here - there)} unexpected {extra}."
        )
    a = sum(p.stat().st_size for p in source.glob("*.safetensors"))
    b = sum(p.stat().st_size for p in original.glob("*.safetensors"))
    # Resharding moves tensors between files, so per-file sizes differ by construction and
    # only the total is comparable. The slack is the safetensors header, which grows with the
    # shard count -- 13 headers instead of 2, not a difference in weights.
    if abs(a - b) > 4 * 1024**2:
        return (
            f"{source} holds {a:,} bytes of weights but the scored {original} holds {b:,}. "
            "A reshard moves tensors between files; it does not change how many there are."
        )
    return None


def check(source: Path, scored: dict[str, Any], panel: dict[str, Any] | None) -> list[str]:
    """Everything that has to hold before two public repos become three."""
    problems = []

    if not (source / "config.json").is_file():
        return [f"{source} holds no config.json -- it is not a model directory"]

    if quantized(source):
        problems.append(
            f"{source} carries a quantization_config. This script publishes the bf16 "
            "reference; pointing it at an exported arm would put quantized weights behind a "
            "name that promises unquantized ones, and every comparison on the card would be "
            "that arm against itself."
        )

    shards = sorted(source.glob("*.safetensors"))
    if not shards:
        problems.append(f"{source} holds no weights")
    for shard in shards:
        nbytes = shard.stat().st_size
        if nbytes > MAX_FILE_NBYTES:
            problems.append(
                f"{shard.name} is {nbytes / 1000**3:.2f} GB, above the Hub's "
                f"{MAX_FILE_NBYTES / 1000**3:.0f} GB per-file limit."
            )
        elif nbytes > RECOMMENDED_SHARD_NBYTES:
            problems.append(
                f"{shard.name} is {nbytes / 1000**3:.2f} GB. It is under the Hub's limit and "
                "would upload, but a shard that large cannot be fetched in parallel and "
                "resumes badly. Re-save with max_shard_size='4GB'."
            )

    mismatch = same_weights(source, scored)
    if mismatch:
        problems.append(mismatch)

    if panel is not None and panel.get("reference") != scored.get("label"):
        problems.append(
            f"the panel's reference is {panel.get('reference')!r} but this arm is "
            f"{scored.get('label')!r}. Every delta in the comparison table is measured against "
            "the panel's reference, so the table would describe a model this card is not for."
        )

    return problems


def comparison(panel: dict[str, Any] | None, linked: list[Peer], nbytes: int) -> list[str]:
    """The table that tells a reader they probably want one of the other two repos."""
    if not panel:
        return []
    by_label = {peer.label: peer for peer in linked}
    rows = [
        "| model | on disk | accuracy | vs bf16 | 95% CI | separated? |",
        "|---|---|---|---|---|---|",
        f"| **this repo** (bf16) | {nbytes / 2**30:.2f} GiB | "
        f"{model_card.pct(panel.get('reference_accuracy'))} | -- | -- | -- |",
    ]
    for arm in panel.get("arms", []):
        comp = arm.get("comparison") or {}
        peer = by_label.get(arm["arm"])
        name = f"[`{peer.repo}`](https://huggingface.co/{peer.repo})" if peer else arm["arm"]
        size = peer.nbytes() if peer else None
        if size is None:
            size = arm.get("directory_nbytes")
        rows.append(
            f"| {name} | {model_card.gib(size) if size else '--'} | "
            f"{model_card.pct(arm.get('accuracy'))} | "
            f"{comp.get('delta_points', 0):+.2f} pts | "
            f"[{comp.get('ci_low_points', 0):+.2f}, {comp.get('ci_high_points', 0):+.2f}] | "
            f"{'**yes**' if comp.get('separated') else 'no'} (p = "
            f"{model_card.pval(comp.get('p_value'))}) |"
        )
    return rows


def render(
    args: argparse.Namespace,
    source: Path,
    finetune: dict[str, Any],
    scored: dict[str, Any],
    panel: dict[str, Any] | None,
    linked: list[Peer],
) -> str:
    fields = model_card.scored_fields(scored)
    nbytes = sum(p.stat().st_size for p in source.glob("*.safetensors"))
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    add("---")
    add("license: apache-2.0")
    add(f"base_model: {args.base_model}")
    add("library_name: transformers")
    add("pipeline_tag: text-generation")
    add("tags:")
    for tag in ("text-to-sql", "qlora", "bf16", "sft"):
        add(f"- {tag}")
    add("language:")
    add("- en")
    add("---")
    add()
    add(f"# {args.repo.split('/')[-1]}")
    add()
    add(
        f"The **unquantized** arm of a three-arm panel: {args.base_model}'s text tower, QLoRA "
        "fine-tuned on text-to-SQL and merged back to bf16. It is published because the two "
        "DynQuant arms' headline numbers are differences *against this model*, measured in the "
        "same run on the same items -- and a difference whose reference nobody can download is "
        "not a measurement anybody can check."
    )
    add()

    table = comparison(panel, linked, nbytes)
    if table:
        add("## You may want a smaller arm")
        add()
        add(
            f"This repo is {nbytes / 2**30:.2f} GiB. All three arms were scored in the same "
            f"run, on the same {scored.get('total', '?')} held-out items, at the same decode "
            "settings, and paired item by item (McNemar):"
        )
        add()
        lines.extend(table)
        add()
        add(
            '"Separated: no" means the paired test could not tell that arm apart from this one '
            "at this sample size. That is **not** a claim that the two are identical -- read "
            "the confidence interval, which is the range of differences the data is consistent "
            "with."
        )
        add()

    add("## What this is")
    add()
    add(
        f"- **Base**: [`{args.base_model}`](https://huggingface.co/{args.base_model}), text "
        "tower only. The vision tower is not fine-tuned and not evaluated here."
    )
    trained_on = model_card.train_sources(finetune)
    loss = finetune.get("train_loss")
    add(
        f"- **Fine-tune**: {finetune.get('regime', '?')}, LoRA rank "
        f"{finetune.get('lora_rank', '?')}, {finetune.get('epochs', '?')} epoch at lr "
        f"{finetune.get('lr', '?')}, effective batch {finetune.get('effective_batch', '?')} "
        f"over {finetune.get('steps', '?')} steps, train loss "
        f"{'?' if loss is None else format(loss, '.4f')}."
    )
    add(
        f"- **Data**: {finetune.get('conversations_kept', 0):,} conversations from "
        f"{model_card.joined(trained_on)}, {finetune.get('supervised_tokens', 0):,} supervised "
        "tokens (loss is taken on the answer only)."
    )
    add(
        "- **Merged**: the adapter is folded into the base weights, so this is a plain bf16 "
        "checkpoint with no PEFT dependency at load time."
    )
    # No blank line between this and the bullets above it: a blank line inside a markdown list
    # ends the list, and the contamination sentence would render as a second one-item list
    # sitting under the first -- which reads as a footnote rather than as part of the recipe.
    contamination = model_card.decontamination(finetune)
    if contamination:
        add(contamination)
    add()

    add("## How it was scored")
    add()
    add(
        f"{model_card.pct(scored.get('accuracy'))} "
        f"({scored.get('correct', '?')}/{scored.get('total', '?')}) on held-out text-to-SQL "
        f"drawn from {model_card.joined(fields['sources'])}, {scored.get('shots', '?')}-shot, "
        f"execution-free logic match. Decode was greedy with a budget of "
        f"{fields['max_new_tokens']} new tokens, and {fields['unfinished_reasoning']} "
        f"generations reached it without finishing. {scored.get('unparseable', '?')} "
        "predictions were unparseable."
    )
    add()
    add(
        "`generation_config.json` in this repo pins greedy decode, because that is how the "
        "number above was measured. If you sample, you are not running the configuration that "
        "produced it."
    )
    add()

    add("## Use")
    add()
    add("```python")
    add("import torch")
    add("from transformers import AutoModelForCausalLM, AutoTokenizer")
    add()
    add(f'tok = AutoTokenizer.from_pretrained("{args.repo}")')
    add("model = AutoModelForCausalLM.from_pretrained(")
    add(f'    "{args.repo}", dtype=torch.bfloat16, device_map="auto"')
    add(")")
    add()
    add('schema = "CREATE TABLE singer (singer_id INT, name TEXT, age INT)"')
    add('prompt = f"Given the schema: {schema}\\nQuestion: How many singers are there?\\nSQL:"')
    add("enc = tok.apply_chat_template(")
    add('    [{"role": "user", "content": prompt}],')
    add("    tokenize=True, add_generation_prompt=True,")
    add('    return_tensors="pt", return_dict=True,')
    add(").to(model.device)")
    add("out = model.generate(**enc, max_new_tokens=256, do_sample=False)")
    add('print(tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True))')
    add("```")
    add()
    add(
        "The base is a reasoning model and its answers open with a `</think>` marker before the "
        "SQL. The scorer cuts at the first `SELECT`; if you consume the output programmatically, "
        "do the same rather than assuming the first line is the query."
    )
    add()

    add("## Limits")
    add()
    add(
        f"- Scored on {scored.get('total', '?')} items. Differences of well under a point are "
        "not resolvable at that size, which is why the table above reports intervals and a "
        "paired test rather than two accuracies side by side."
    )
    add(
        "- Logic match is not execution accuracy: it compares query structure, not results "
        "against a live database."
    )
    add("- English prompts only; the fine-tune added no other language.")
    add(
        "- Text only. The base model's vision tower is untouched by this fine-tune and its "
        "behaviour here is unmeasured."
    )
    add()
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--source", required=True, help="the bf16 merge, resharded for the Hub")
    p.add_argument("--finetune", required=True)
    p.add_argument("--eval", required=True, dest="eval_record")
    p.add_argument("--panel")
    p.add_argument(
        "--peer",
        action="append",
        default=[],
        metavar="LABEL=REPO",
        help="a quantized arm of the same panel, linked from the comparison table",
    )
    p.add_argument("--base-model", required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--private", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    if any(part.startswith("hf_") for part in (argv if argv is not None else sys.argv[1:])):
        raise SystemExit(
            "a token was passed as an argument. It is now in this shell's history and was "
            "visible in `ps` to every process on this box for the length of this run. Treat it "
            "as compromised, revoke it, and pass the replacement in HF_TOKEN."
        )

    args = build_parser().parse_args(argv)
    source = Path(args.source)
    finetune = model_card.load(args.finetune, "finetune")
    scored = model_card.load(args.eval_record, "eval")
    panel = model_card.load(args.panel, "panel") if args.panel else None
    linked = peers(args.peer)

    if panel is not None:
        panel = dict(panel)
        panel.setdefault("reference_accuracy", scored.get("accuracy"))

    problems = check(source, scored, panel)
    if problems:
        for problem in problems:
            print(f"REFUSED: {problem}")
        return 2

    card = render(args, source, finetune, scored, panel, linked)
    out = source / "README.md"
    out.write_text(card, encoding="utf-8", newline=phase4.model_cards.NL)
    print(f"card -> {out} ({len(card):,} chars)")

    if args.dry_run:
        print("dry run: nothing uploaded")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set")

    push = phase4.Push(label="bf16", source=source, repo=args.repo)
    for line in phase4.occupied([push], token):
        print(f"note: {line}")
    url = phase4.upload(push, card, token, private=args.private)
    print(f"pushed bf16 -> {url}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
