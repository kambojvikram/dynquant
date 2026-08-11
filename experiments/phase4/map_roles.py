"""What the allocator did, by role, from an exported bit map.

A panel row says DynQuant beat a uniform baseline by nineteen points. It does not say what
DynQuant *did*, and on a mixture-of-experts model the answer is not diffuse: 133 quantized
modules, four widths, and a handful of decisions carrying almost all of the margin. The map is on
disk -- `arms_lfm2.py` exports one per DynQuant arm -- and reading it by hand into a report is the
transcription step `panel_table.py` and `report_tables.py` were written to remove, arriving one
file to the right. Eight roles by four widths is thirty-two cells and nothing recomputes them.

Two maps side by side is the useful form, because a width is only interesting against the width
the same role got at a different budget. So the columns are maps and the rows are roles.

**The roles here are derived from module names, and the allocator's were not.** It classified with
the model in hand -- `classify_module` reads the module tree and the config -- while all this has
is a JSON of dotted names, so it falls back to `role_of_name`, which the package documents as the
last resort precisely because names lie. That would be a quiet second opinion presented as the
first one, except that the map records the true role for every module whose floor was breached.
Those are checked, and a disagreement refuses rather than printing a plausible table. It is a
partial check -- it covers the breached modules and not the rest -- and it is the only one the
format admits, so the refusal is worth more than the coverage suggests: the roles it can check are
the ones the allocator singled out.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dynquant.graph import DEFAULT_FLOOR_BITS, role_of_name


def entries(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """(name, entry) for every target in one exported map file.

    A map file is keyed by byte budget and usually holds one, but nothing in the format says so,
    and a file holding two would otherwise have its second target silently dropped -- the failure
    this whole tool exists one level up from.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    maps = doc.get("maps")
    if not maps:
        raise SystemExit(f"{path} has no `maps` block -- is it an allocation export?")
    stem = path.stem
    if len(maps) == 1:
        return [(stem, next(iter(maps.values())))]
    return [(f"{stem}@{target}", entry) for target, entry in sorted(maps.items())]


def resolve(entry: dict[str, Any], where: str) -> dict[str, str]:
    """Module name -> role name, checked against every role the map actually recorded."""
    resolved = {name: role_of_name(name).value for name in entry["bits"]}
    for breach in entry.get("violations", ()):
        name, recorded = breach["name"], breach["role"]
        got = resolved.get(name)
        if got != recorded:
            raise SystemExit(
                f"{where}: the allocator classified `{name}` as `{recorded}` with the model in\n"
                f"hand; from the name alone it reads as `{got}`. Every role in the table below\n"
                "comes from the name, so one that is demonstrably wrong makes the whole column\n"
                "unreadable rather than slightly off."
            )
    return resolved


def crosstab(entry: dict[str, Any], resolved: dict[str, str]) -> dict[str, Counter[int]]:
    table: dict[str, Counter[int]] = {}
    for name, bits in entry["bits"].items():
        table.setdefault(resolved[name], Counter())[int(bits)] += 1
    counted = sum(sum(row.values()) for row in table.values())
    declared = sum(int(n) for n in entry["histogram"].values())
    if counted != declared:
        raise SystemExit(
            f"the crosstab covers {counted} modules and the map's own histogram counts "
            f"{declared}. One of them is not reading the `bits` block."
        )
    return table


def default_floor_of(role: str) -> int:
    """The role's floor in the package's default table, or 0 for a role it does not name."""
    for known, bits in DEFAULT_FLOOR_BITS.items():
        if known.value == role:
            return int(bits)
    return 0


def recorded_floors(named: list[tuple[str, dict[str, Any], dict[str, str]]]) -> dict[str, int]:
    """The floor the allocator actually enforced, per role, as the map recorded it.

    `DEFAULT_FLOOR_BITS` is a default, and this campaign's own model breaks it: `embedding`
    defaults to 4 and `model.embed_tokens` was held to 8, because it is tied to the LM head and
    `Policy.floor_for` takes the strictest floor across a tie. Printing 4 in a column headed
    "floor" beside a breach row that says 8 is the failure this file already refuses in the other
    direction -- a derived value standing where a recorded one exists. So where the map records a
    floor, it wins, and `main` names every role where the two disagree rather than quietly
    preferring one.

    Two maps recording different floors for one role would mean they were allocated under
    different policies, which makes a side-by-side column comparison meaningless rather than
    slightly off.
    """
    floors: dict[str, int] = {}
    for name, entry, _ in named:
        for breach in entry.get("violations", ()):
            role, floor = breach["role"], int(breach["floor_bits"])
            if floors.setdefault(role, floor) != floor:
                raise SystemExit(
                    f"`{role}` was held to {floors[role]} bits in one map and {floor} in "
                    f"`{name}`. These maps were allocated under different policies, so a row "
                    "putting their widths side by side would compare two different rules."
                )
    return floors


def floor_of(role: str, recorded: dict[str, int]) -> int:
    """The floor that applied, recorded first, default second, 0 for a role neither names.

    Sorting on it puts the roles the method protects at the top, which is the order the mechanism
    reads in -- and it is the allocator's own ordering rather than one chosen to make a point.
    """
    if role in recorded:
        return recorded[role]
    return default_floor_of(role)


def widths_table(
    named: list[tuple[str, dict[str, Any], dict[str, str]]], recorded: dict[str, int]
) -> str:
    tables = [(name, crosstab(entry, resolved)) for name, entry, resolved in named]
    roles = sorted(
        {role for _, table in tables for role in table},
        key=lambda role: (-floor_of(role, recorded), role),
    )
    widths = sorted({bits for _, table in tables for row in table.values() for bits in row})
    spread = "/".join(f"{bits}b" for bits in widths)

    head = ["role", "floor", "n"] + [f"{name} @ {spread}" for name, _ in tables]
    lines = ["| " + " | ".join(head) + " |", "|---|---:|---:|" + "---|" * len(tables)]
    for role in roles:
        counts = [table.get(role, Counter()) for _, table in tables]
        total = max(sum(row.values()) for row in counts)
        cells = [" / ".join(str(row.get(bits, 0)) for bits in widths) for row in counts]
        # A star marks a floor the map recorded rather than one read out of the default table.
        floor = floor_of(role, recorded)
        shown = f"{floor}*" if role in recorded else (str(floor) if floor else "--")
        lines.append(f"| `{role}` | {shown} | {total} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def mixed_parents(named: list[tuple[str, dict[str, Any], dict[str, str]]]) -> list[str]:
    """Roles whose members sit under more than one parent module, which names cannot separate.

    The one check the map's `violations` block cannot provide. A role is only breached if the
    budget bound on it, so an unbreached role gets no second opinion at all -- and this model has
    one that needs it: `role_of_name` files 24 modules as `attn.o`, of which 6 are
    `self_attn.out_proj` and 18 are `conv.out_proj`, the output of the short-convolution block.
    The allocator, reading the module tree, did not confuse them; both happen to carry the same
    floor, so the widths are right and only the row label is wrong.

    Which is the point. There is no way to notice that from inside a role's own count, and a
    reader comparing `attn.o` n=24 against `attn.q` n=6 has no reason to suspect the row covers
    two different blocks. The parent segment is the cheapest evidence available and it is generic:
    any role assembled from two structurally different places shows up here without this file
    knowing anything about LFM2.
    """
    # One map, not all of them: `main` has already refused unless every map covers the same
    # modules, so walking each one would count every module once per column.
    _, entry, resolved = named[0]
    grouped: dict[str, Counter[str]] = {}
    for name in entry["bits"]:
        parts = name.split(".")
        parent = parts[-2] if len(parts) > 1 else "(root)"
        grouped.setdefault(resolved[name], Counter())[parent] += 1
    lines = []
    for role, parents in sorted(grouped.items()):
        if len(parents) > 1:
            spread = ", ".join(f"{count} under `{p}`" for p, count in sorted(parents.items()))
            lines.append(f"  `{role}` is not one block: {spread}.")
    return lines


def floors_table(name: str, entry: dict[str, Any]) -> str:
    breaches = entry.get("violations") or []
    if not breaches:
        return f"`{name}`: no floor breached -- the budget was not binding on any role."

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for breach in breaches:
        key = (breach["role"], int(breach["floor_bits"]), int(breach["assigned_bits"]))
        grouped.setdefault(key, []).append(breach)

    lines = [
        f"`{name}`: {len(breaches)} floor(s) breached.",
        "",
        "| role | floor | assigned | modules | params |",
        "|---|---:|---:|---:|---:|",
    ]
    for (role, floor, assigned), members in sorted(grouped.items(), key=lambda kv: -kv[0][1]):
        params = sum(int(m["num_params"]) for m in members)
        lines.append(
            f"| `{role}` | {floor} | **{assigned}** | {len(members)} | {params / 1e9:.2f}G |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--map",
        dest="maps",
        required=True,
        action="append",
        type=Path,
        help="an allocation export from `arms_lfm2 run`; repeatable, one column each",
    )
    args = parser.parse_args(argv)

    named = []
    for path in args.maps:
        for name, entry in entries(path):
            named.append((name, entry, resolve(entry, f"{path}::{name}")))

    covered = [set(entry["bits"]) for _, entry, _ in named]
    if len({frozenset(names) for names in covered}) > 1:
        missing = set.union(*covered) - set.intersection(*covered)
        raise SystemExit(
            "these maps do not cover the same modules, so a row would compare a width to a "
            f"blank: {len(missing)} module(s) appear in some and not others, e.g. "
            f"{sorted(missing)[:3]}"
        )

    for name, entry, _ in named:
        print(
            f"`{name}`: {entry['average_bits']:.4f} average bits over "
            f"{len(entry['bits'])} quantized modules, {int(entry['nbytes']):,} B"
        )
    recorded = recorded_floors(named)
    print()
    print(widths_table(named, recorded))
    for role, floor in sorted(recorded.items()):
        default = default_floor_of(role)
        if default != floor:
            print()
            print(f"  `{role}`* was held to {floor} bits, not the {default} of the default table.")
            print("  A floor is raised across a tie -- the strictest role in the tie wins -- so a")
            print("  role that looks cheap in isolation is not, and the star is where that")
            print("  happened.")
    mixed = mixed_parents(named)
    if mixed:
        print()
        print("  These roles come from the module name, and one name covers two blocks:")
        for line in mixed:
            print(line)
        print("  The allocator classified from the module tree and did not merge them; the")
        print("  widths in that row are right and the label is the tool's, not the method's.")
    print()
    for name, entry, _ in named:
        print(floors_table(name, entry))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
