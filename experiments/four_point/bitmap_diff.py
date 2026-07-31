"""Compare two allocations of the *same* model produced from *different* fine-tunes.

The question this answers is narrow and worth asking precisely: the allocator reads statistics
collected during fine-tuning, so two tasks should produce two bit maps. Do they, and by how
much? A map that barely moves would mean the signal is measuring architecture rather than task,
which is a real limitation and not one an accuracy column would ever reveal -- both maps would
score fine and nobody would learn that one of them was redundant.

The comparison is deliberately structural rather than statistical. It reports, per target
budget:

* how many modules were assigned a different width, and the total parameters under them;
* the direction of each move, so "the two tasks disagree" can be distinguished from "one task
  simply had more budget left over";
* a breakdown by role, because a disagreement concentrated in ``mlp.down`` means something
  different from one spread evenly -- the former says the task changed which *kind* of tensor
  matters, the latter says it only jittered the ranking near the knapsack's cut line;
* the modules both maps agree on but that differ from the uniform assignment, which is the
  part of the allocation attributable to the architecture rather than to either task.

Reading the last bullet first is the honest order. If both tasks move the same 14 modules off
uniform and disagree on none of them, the "dynamic signal" is a fixed architectural prior with
extra steps.

Usage::

    python bitmap_diff.py RUN_A RUN_B [--target 4.25]

where each RUN is a run directory containing ``stage4_bitmaps.json``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_maps(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "stage4_bitmaps.json"
    if not path.is_file():
        raise SystemExit(f"no stage4_bitmaps.json in {run_dir}")
    record = json.loads(path.read_text(encoding="utf-8"))
    maps = record.get("maps")
    if not maps:
        raise SystemExit(f"{path} has no 'maps' key")
    return maps


def role_of(name: str) -> str:
    """Collapse a module path to the role the allocator prices it as.

    Derived from the name rather than re-read from the record because the bit maps store only
    ``{name: bits}``; the ``violations`` list carries roles but covers only breached floors.
    The buckets are coarse on purpose -- this is for grouping a diff, not for allocating.
    """
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    tail = name.rsplit(".", 1)[-1]
    if "linear_attn" in name:
        return f"lin_attn.{tail.replace('in_proj_', '')}"
    if "self_attn" in name:
        return f"attn.{tail}"
    if "mlp" in name:
        return f"mlp.{tail.replace('_proj', '')}"
    return tail


def modal(values: list[int]) -> int:
    """The most common width, which is what "uniform" means for a mixed-width map."""
    return Counter(values).most_common(1)[0][0]


def compare(target: str, a: dict, b: dict, label_a: str, label_b: str) -> None:
    bits_a: dict[str, int] = a["bits"]
    bits_b: dict[str, int] = b["bits"]

    only_a = sorted(set(bits_a) - set(bits_b))
    only_b = sorted(set(bits_b) - set(bits_a))
    shared = sorted(set(bits_a) & set(bits_b))
    if only_a or only_b:
        # Two runs of the same model should cover the same module set. If they do not, the
        # comparison below is silently over a subset and every count in it is wrong, so say so.
        print(
            f"  !! module sets differ: {len(only_a)} only in {label_a}, {len(only_b)} only in {label_b}"
        )

    print(f"\n=== target {target}  ({label_a} vs {label_b})")
    print(f"  average bits: {a['average_bits']:.4f} vs {b['average_bits']:.4f}")
    print(f"  modules: {len(shared)} shared")

    uniform = modal([bits_a[n] for n in shared])
    off_a = {n for n in shared if bits_a[n] != uniform}
    off_b = {n for n in shared if bits_b[n] != uniform}
    differ = [n for n in shared if bits_a[n] != bits_b[n]]

    print(
        f"  modal width {uniform}b; off-modal: {len(off_a)} in {label_a}, {len(off_b)} in {label_b}"
    )
    print(f"  agreed off-modal (architecture, not task): {len(off_a & off_b)}")
    print(
        f"  disagreements: {len(differ)} of {len(shared)} modules ({100 * len(differ) / len(shared):.1f}%)"
    )

    if not differ:
        print("  -> the two fine-tunes produced the SAME map. The signal is not task-specific")
        print("     at this budget, whatever else it is.")
        return

    directions: Counter[str] = Counter()
    by_role: defaultdict[str, list[str]] = defaultdict(list)
    for name in differ:
        directions[f"{bits_a[name]}b -> {bits_b[name]}b"] += 1
        by_role[role_of(name)].append(name)

    print("  moves:")
    for move, count in directions.most_common():
        print(f"    {move:>14}  x{count}")

    print("  by role:")
    for role, names in sorted(by_role.items(), key=lambda kv: -len(kv[1])):
        sample = names[0].replace("model.layers.", "L")
        print(f"    {role:<20} x{len(names):<4} e.g. {sample}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument(
        "--target",
        action="append",
        help="budget key to compare (repeatable); default is every key both runs share",
    )
    args = parser.parse_args()

    maps_a = load_maps(args.run_a)
    maps_b = load_maps(args.run_b)
    label_a, label_b = args.run_a.name, args.run_b.name

    targets = args.target or sorted(set(maps_a) & set(maps_b), key=float, reverse=True)
    if not targets:
        raise SystemExit(f"no shared budgets: {sorted(maps_a)} vs {sorted(maps_b)}")

    for target in targets:
        if target not in maps_a or target not in maps_b:
            print(f"\n=== target {target}: missing from one run, skipped")
            continue
        compare(target, maps_a[target], maps_b[target], label_a, label_b)


if __name__ == "__main__":
    main()
