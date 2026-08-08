"""The seven-arm panel as one table, assembled from what the run left on disk.

Nothing here re-scores anything or recomputes a size. Every number is a number some
process already wrote down, and an arm that has not run shows up as a missing row rather
than as a gap the reader has to notice.

Where the size column comes from, and why not the record
--------------------------------------------------------

Not from the eval record, which does not carry one and should not. Four of the six
quantized arms are scored on a model ``compressed-tensors`` wrote, and the other two are
scored by encoding the allocator's widths back into bf16 weights -- so the *resident* size
of a DynQuant arm is the fp16 size, and a column filled from the loaded model would report
16 bits for the arm whose compression is the claim.

The honest source is ``arms.json``: for a baseline the size is its format's own accounting
at that width, and for a DynQuant arm it is the byte count the allocator realised and
``check_matched`` already held against the baselines' anchor. This script re-states that
drift rather than trusting that the check ran, because a manifest can be assembled by hand
and a table that only prints matched sizes is not the same as one that checks them.

The fp16 row is the single derived number. It is ``params * 2`` with ``params`` read from a
baseline's ``.quant.json`` side file -- the count that arm was itself sized against, so the
ceiling and the arms beneath it count the same tensors and the ratio between them is real.
No literal parameter count appears anywhere here: one written for this model would be
silently wrong for the next one.

Two families, corrected separately
----------------------------------

Twelve comparisons at alpha=0.05 expect half a false positive, and the headline of this
panel is one of the twelve. So each block carries a Holm-adjusted p and the verdict follows
the adjusted one.

The split into two blocks is not a way of shrinking the multiplier on the claim. The blocks
answer different kinds of question. "Does DynQuant beat GPTQ at these bytes" is a hypothesis
test and belongs in a corrected family. "What did quantizing to 4 bits cost" is a
measurement whose answer is the interval; nobody doubts the sign. Both are corrected anyway,
each within its own block, and the block sizes are printed so a reader who disagrees with
the split can multiply by twelve instead.

Run::

    python experiments/phase4/panel_table.py --arms runs/s4/arms
    python experiments/phase4/panel_table.py --arms runs/s4/arms --json > table.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

#: Head-to-head at matched bytes. This is the claim, and the family it is corrected in.
HEAD_TO_HEAD: tuple[tuple[str, str, str], ...] = (
    ("dq_4b", "gptq_4b", "4b  DynQuant vs GPTQ"),
    ("dq_4b", "awq_4b", "4b  DynQuant vs AWQ"),
    ("gptq_4b", "awq_4b", "4b  GPTQ vs AWQ"),
    ("dq_3b", "gptq_3b", "3b  DynQuant vs GPTQ"),
    ("dq_3b", "awq_3b", "3b  DynQuant vs AWQ"),
    ("gptq_3b", "awq_3b", "3b  GPTQ vs AWQ"),
)

#: What each method cost against the unquantized model it was built from.
AGAINST_CEILING: tuple[tuple[str, str, str], ...] = (
    ("gptq_4b", "bf16", "4b  GPTQ vs bf16"),
    ("awq_4b", "bf16", "4b  AWQ vs bf16"),
    ("dq_4b", "bf16", "4b  DynQuant vs bf16"),
    ("gptq_3b", "bf16", "3b  GPTQ vs bf16"),
    ("awq_3b", "bf16", "3b  AWQ vs bf16"),
    ("dq_3b", "bf16", "3b  DynQuant vs bf16"),
)

FP16_BYTES_PER_PARAM = 2


def compact(count: int) -> str:
    """A parameter count at a width a reader can compare across rows.

    Used for the breached-floor mass, which is the one place here that counts parameters
    and where they span four orders of magnitude between roles -- fixed G units printed a
    real million-parameter role as ``0.00G``, a mass the budget did take, rendered as
    though it had taken nothing. The width histogram counts *modules* and is printed as an
    integer; the two are not interchangeable and reading one as the other is what this
    formatter was originally, wrongly, applied to.
    """
    if count >= 1_000_000_000:
        return f"{count / 1e9:.2f}G"
    if count >= 1_000_000:
        return f"{count / 1e6:.0f}M"
    return f"{count / 1e3:.0f}K"


def standard_error(accuracy: float, total: int) -> float:
    """One binomial SE, in percentage points."""
    return math.sqrt(accuracy * (1.0 - accuracy) / total) * 100.0 if total else 0.0


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the order given.

    Step-down rather than plain Bonferroni: it controls the same family-wise error rate
    and is uniformly more powerful, so using Bonferroni here would be discarding real
    findings for no gain in rigour. Monotonicity is enforced on the way up, which is what
    makes the adjusted values interpretable as p-values rather than as a sorted list of
    multiplied numbers.
    """
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def load_panel(out: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """The manifest plus every record it names that exists."""
    manifest_path = out / "arms.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"{manifest_path} does not exist. The table is assembled from the manifest the "
            f"run writes, not from whatever json files are in the directory -- a stray "
            f"record from another run would otherwise enter the panel as an arm."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: dict[str, dict[str, Any]] = {}
    for arm in manifest["arms"]:
        found = resolve_record(out, arm.get("record"))
        if found is not None:
            records[arm["label"]] = json.loads(found.read_text(encoding="utf-8"))
    return manifest, records


def resolve_record(out: Path, stored: str | None) -> Path | None:
    """The record a manifest entry names, found from beside the manifest if need be.

    The driver writes ``str(out / f"{label}.json")``, so a run launched with a relative
    ``--out`` stores relative paths -- which resolve against whatever directory the *table*
    is later run from, not the one the panel ran in. Read literally, a manifest moved off
    the box, or simply read from a different cwd, is a panel in which no arm was scored:
    seven arms, seven silent misses, and a table that prints ``0/7`` for a run that finished.

    So a stored path that is not a file is retried beside the manifest. That cannot pick up
    a foreign record, because the writer's own invariant is that the file is named for the
    arm and sits in ``out`` -- the same two facts the retry uses.
    """
    if not stored:
        return None
    direct = Path(stored)
    if direct.is_file():
        return direct
    beside = out / direct.name
    return beside if beside.is_file() else None


def check_pairable(records: dict[str, dict[str, Any]]) -> str | None:
    """Refuse to pair records that were not scored under the same settings.

    ``arms_lfm2 run`` already checks this as each arm lands. Checked again because this
    script can be pointed at a directory the run did not assemble -- a resumed panel, a
    hand-merged one -- and pairing two hit vectors that describe different problem sets
    produces a number rather than an error.
    """
    from dynquant.commands.evaluate import _comparability

    if not records:
        return None
    reference = "bf16" if "bf16" in records else next(iter(records))
    expected = _comparability(records[reference])
    for label, record in records.items():
        found = _comparability(record)
        if found != expected:
            differed = {k: (expected[k], found[k]) for k in expected if expected[k] != found[k]}
            return (
                f"{label} was not scored under the same settings as {reference}: {differed} "
                f"as ({reference}, {label}). Their hit vectors are not paired."
            )
    return None


def infer_params(manifest: dict[str, Any]) -> int | None:
    """The parameter count the baselines sized themselves against.

    Read from a ``.quant.json`` side file rather than counted here, so the fp16 row and
    the quantized rows are denominated in the same tensors. Returns ``None`` when no
    baseline arm has run, and the fp16 size column then says so instead of guessing.
    """
    for arm in manifest["arms"]:
        if arm.get("kind") not in ("gptq", "awq"):
            continue
        record = arm.get("record")
        if not record:
            continue
        side = Path(str(record)).with_suffix(".quant.json")
        if side.is_file():
            payload = json.loads(side.read_text(encoding="utf-8"))
            if payload.get("params"):
                return int(payload["params"])
    return None


def allocation_of(arm: dict[str, Any]) -> dict[str, Any] | None:
    """The allocator's own account of a DynQuant arm: widths and breached floors.

    The floor violations are the part that cannot be recovered later. Two arms can land on
    the same average bits with and without a breached expert bank, and only one of them is
    a knapsack result -- so an allocation that reports zero breaches at 4 bits and a
    breached ``gate_up`` at 3 is the pre-registered prediction being confirmed or not.
    """
    path, key = arm.get("map"), str(arm.get("target_bytes"))
    if not path or not Path(path).is_file():
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entry = payload.get("maps", {}).get(key)
    if entry is None:
        return None
    return {
        "average_bits": entry.get("average_bits"),
        "nbytes": entry.get("nbytes"),
        "histogram": entry.get("histogram", {}),
        "violations": entry.get("violations", []),
    }


def rows(
    manifest: dict[str, Any], records: dict[str, dict[str, Any]], params: int | None
) -> list[dict[str, Any]]:
    """One row per planned arm, in the order the panel planned them."""
    built = []
    for arm in manifest["arms"]:
        label = arm["label"]
        record = records.get(label)
        nbytes = arm.get("nbytes")
        if nbytes is None and arm.get("kind") == "ceiling" and params:
            nbytes = params * FP16_BYTES_PER_PARAM
        detail = (record or {}).get("detail") or {}
        target = arm.get("target_bytes")
        built.append(
            {
                "label": label,
                "kind": arm.get("kind"),
                "anchor": arm.get("anchor"),
                "nbytes": nbytes,
                "target_bytes": target,
                "drift": (nbytes - target) / target if (nbytes and target) else None,
                "bits_per_param": (nbytes * 8 / params) if (nbytes and params) else None,
                "accuracy": (record or {}).get("accuracy"),
                "correct": (record or {}).get("correct"),
                "total": (record or {}).get("total"),
                "unparseable": (record or {}).get("unparseable"),
                "errored": detail.get("errored"),
                "exact": detail.get("exact"),
                "unfinished": detail.get("unfinished_reasoning"),
                "by_source": detail.get("by_source") or {},
                "apply": ((record or {}).get("packed") or {}).get("apply"),
                "allocation": allocation_of(arm),
                "seconds": (record or {}).get("seconds"),
            }
        )
    return built


def print_sizes(built: list[dict[str, Any]], params: int | None, tolerance: float) -> None:
    header = (
        f"{'arm':10s} {'method':8s} {'GiB':>8s} {'b/param':>8s} {'off anchor':>12s} "
        f"{'vs bf16':>8s} {'apply':>7s}"
    )
    print(header)
    print("-" * len(header))
    ceiling = next((row["nbytes"] for row in built if row["kind"] == "ceiling"), None)
    for row in built:
        nbytes = row["nbytes"]
        gib = f"{nbytes / 2**30:.3f}" if nbytes else "--"
        bpp = f"{row['bits_per_param']:.4f}" if row["bits_per_param"] else "--"
        if row["drift"] is None:
            drift = "--"
        else:
            drift = f"{row['drift'] * 100:+.4f}%" + ("!" if abs(row["drift"]) > tolerance else "")
        ratio = f"{ceiling / nbytes:.2f}x" if (ceiling and nbytes) else "--"
        kind = row["kind"] or "--"
        apply_mode = row["apply"] or "--"
        print(
            f"{row['label']:10s} {kind:8s} {gib:>8s} {bpp:>8s} {drift:>12s} "
            f"{ratio:>8s} {apply_mode:>7s}"
        )
    if params:
        print(f"  denominated in {params:,} parameters, from a baseline's own accounting")
    else:
        print("  no baseline .quant.json found: the bf16 size and b/param columns are unavailable")
    if any(row["drift"] is not None and abs(row["drift"]) > tolerance for row in built):
        raise SystemExit(
            f"an arm marked ! is further than {tolerance:.3%} from its anchor, so the panel "
            f"is not byte-matched and every accuracy difference in it is confounded with "
            f"size. The rest of the table is not printed for a panel that cannot support one."
        )


def print_accuracy(built: list[dict[str, Any]], chance: float | None) -> None:
    header = (
        f"{'arm':10s} {'exec match':>11s} {'+-1SE':>6s} {'correct':>11s} "
        f"{'exact':>7s} {'no query':>9s} {'sql error':>10s} {'unfinished':>11s} {'min':>6s}"
    )
    print(header)
    print("-" * len(header))
    for row in built:
        if row["accuracy"] is None:
            print(f"{row['label']:10s} {'not run':>11s}")
            continue
        se = standard_error(row["accuracy"], row["total"] or 0)
        minutes = f"{row['seconds'] / 60:.0f}" if row["seconds"] else "--"
        unfinished = row["unfinished"]
        counted = f"{row['correct']}/{row['total']}"
        # A ! rather than a footnote: a non-zero count here caps the headline above it,
        # and that is a fact about the row it sits on, not about the table.
        flagged = f"{unfinished if unfinished is not None else '--'}{'!' if unfinished else ''}"
        exact, no_query, errored = (
            f"{row[field] if row[field] is not None else '--'}"
            for field in ("exact", "unparseable", "errored")
        )
        print(
            f"{row['label']:10s} {row['accuracy'] * 100:10.2f}% {se:6.2f} {counted:>11s} "
            f"{exact:>7s} {no_query:>9s} {errored:>10s} {flagged:>11s} {minutes:>6s}"
        )
    if chance:
        print(f"{'(guessing)':10s} {chance * 100:10.2f}%")


def print_by_source(built: list[dict[str, Any]]) -> None:
    """Per-source accuracy, because one number over three datasets can hide a collapse.

    A method that damages one source's distribution and leaves the others alone moves the
    headline by a couple of points and moves one column here by twenty. The mixture exists
    so that is visible; printing only the mixture would waste it.
    """
    sources = sorted({name for row in built for name in row["by_source"]})
    if not sources:
        return
    header = f"{'arm':10s}" + "".join(f"{name:>18s}" for name in sources)
    print(header)
    print("-" * len(header))
    for row in built:
        if not row["by_source"]:
            continue
        cells = ""
        for name in sources:
            pair = row["by_source"].get(name)
            if not pair or not pair[1]:
                cells += f"{'--':>18s}"
            else:
                correct, total = pair
                cells += f"{f'{correct / total * 100:.1f}% ({total})':>18s}"
        print(f"{row['label']:10s}{cells}")


def print_allocation(built: list[dict[str, Any]]) -> None:
    """What the allocator did, for the arms that had one.

    A DynQuant arm's average bits and its breached floors are the only evidence of whether
    the budget was still binding. Zero breaches means the floors fitted and the allocation
    was a knapsack over the slack; a breach names the role the budget could not afford.
    Both were predicted before either arm ran, and neither can be recovered from the
    accuracy afterwards.
    """
    for row in built:
        report = row["allocation"]
        if not report:
            continue
        # Modules, not parameters. The saved map's histogram counts tensors -- checked
        # against a real one rather than assumed -- and this line said "params" and ran the
        # counts through a billions-scale formatter, so a 187-module width printed as
        # `0K`: the allocator's whole answer, rendered as though it had assigned nothing.
        # The parameter mass is not in the map, but it is in `violations`, which is where
        # the question that needs it is actually asked.
        widths = "  ".join(
            f"{width}b {int(modules)}"
            for width, modules in sorted(report["histogram"].items(), key=lambda kv: int(kv[0]))
        )
        total = sum(int(modules) for modules in report["histogram"].values())
        print(f"{row['label']}: {report['average_bits']:.4f} avg bits over the quantized set")
        print(f"  widths, modules at each: {widths}   ({total} quantized)")
        breaches = report["violations"]
        if not breaches:
            print("  floors: none breached -- the budget was not binding on any role")
            continue
        by_role: dict[str, list[int]] = {}
        for breach in breaches:
            by_role.setdefault(str(breach["role"]), []).append(int(breach["num_params"]))
        print(f"  floors: {len(breaches)} breached")
        for role, counts in sorted(by_role.items()):
            print(f"    {role:16s} {len(counts):3d} tensors  {compact(sum(counts))} params")


def print_comparisons(
    title: str,
    family: tuple[tuple[str, str, str], ...],
    records: dict[str, dict[str, Any]],
    pairable: str | None,
) -> list[dict[str, Any]]:
    from dynquant.eval.compare import compare_paired

    print(title)
    header = (
        f"{'comparison':28s} {'delta':>7s} {'95% CI':>18s} {'flips':>11s} "
        f"{'p':>10s} {'p (Holm)':>10s}  verdict"
    )
    print(header)
    print("-" * len(header))

    computed: list[dict[str, Any]] = []
    for left, right, question in family:
        if left not in records or right not in records:
            print(f"{question:28s} (needs both arms)")
            continue
        if pairable is not None:
            print(f"{question:28s} (records are not comparable)")
            continue
        a, b = records[left], records[right]
        if not a.get("hits") or not b.get("hits"):
            print(f"{question:28s} (no per-item hits recorded)")
            continue
        paired = compare_paired(a["hits"], b["hits"], label_a=left, label_b=right)
        computed.append({"left": left, "right": right, "question": question, "paired": paired})

    adjusted = holm([entry["paired"].p_value for entry in computed])
    for entry, p_adj in zip(computed, adjusted, strict=True):
        paired = entry["paired"]
        low, high = paired.interval_points
        entry["p_adjusted"] = p_adj
        entry["separated"] = p_adj < 0.05
        print(
            f"{entry['question']:28s} {paired.delta_points:+6.2f} "
            f"{f'[{low:+.2f}, {high:+.2f}]':>18s} "
            f"{f'{paired.a_only}/{paired.b_only}':>11s} "
            f"{paired.p_value:10.3g} {p_adj:10.3g}  "
            f"{'separated' if entry['separated'] else 'NOT separated'}"
        )
    if computed:
        print(f"  Holm-adjusted over the {len(computed)} comparisons in this block")
    return computed


def as_json(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "left": entry["left"],
            "right": entry["right"],
            **entry["paired"].as_dict(),
            "p_adjusted": entry["p_adjusted"],
            "separated": entry["separated"],
        }
        for entry in entries
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="assemble the phase-4 panel table")
    parser.add_argument("--arms", required=True, help="the --out directory arms_lfm2 run wrote")
    parser.add_argument("--json", action="store_true", help="emit the assembled table as json")
    args = parser.parse_args(argv)

    manifest, records = load_panel(Path(args.arms))
    params = infer_params(manifest)
    built = rows(manifest, records, params)
    pairable = check_pairable(records)
    tolerance = float(manifest.get("tolerance", 0.001))

    chance = next((r.get("chance") for r in records.values() if r.get("chance")), None)
    scored = sum(1 for row in built if row["accuracy"] is not None)
    print(f"panel: {scored}/{len(built)} arms scored   model: {manifest.get('model')}")
    anchors = manifest.get("anchors") or {}
    ordered = sorted(anchors.items(), key=lambda kv: int(kv[0]), reverse=True)
    print("anchors: " + ", ".join(f"{w}b -> {int(b):,} B" for w, b in ordered))
    print()
    print_sizes(built, params, tolerance)
    print()
    print_accuracy(built, chance)
    print()
    print_by_source(built)
    print()
    print_allocation(built)
    print()

    if pairable is not None:
        print(f"NOT PAIRED: {pairable}")
        print()
    head = print_comparisons("head to head, at matched bytes", HEAD_TO_HEAD, records, pairable)
    print()
    ceiling = print_comparisons("what each method cost", AGAINST_CEILING, records, pairable)
    print()
    for line in (
        "delta = left minus right, percentage points, on the same problems in the same order.",
        "CI and p are McNemar exact over the discordant pairs; flips = only-left-right /",
        "only-right-right. The verdict follows the Holm-adjusted p within its own block.",
        "Sizes are the manifest's, not the loaded model's: a DynQuant arm is scored by",
        "encoding its widths back into bf16, so it holds fp16 and claims the allocator's",
        "bytes -- which is the size the same map writes when packed to disk.",
    ):
        print(line)

    if args.json:
        print(
            json.dumps(
                {
                    "model": manifest.get("model"),
                    "anchors": anchors,
                    "params": params,
                    "pairable": pairable is None,
                    "pairing_error": pairable,
                    "arms": built,
                    "head_to_head": as_json(head),
                    "against_ceiling": as_json(ceiling),
                },
                indent=2,
                default=str,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
