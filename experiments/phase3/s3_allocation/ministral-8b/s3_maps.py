"""Read the S3 bit maps directly, rather than inferring their shape from the evals.

``s3_table.py`` answers what the arms scored. This answers what they actually assigned, and
it exists because the first draft of the report got that wrong: it explained the inert 4.25-bit
anchor by saying the maps converged, which was inferred from zero floor violations plus low
eval discordance and is false. At 4.25 bits ``dq`` differs from uniform on 99 of 254 modules --
more than it does at 3.25. The allocation is not degenerate there; the model is just insensitive
to it.

Three questions, none of which an eval table can answer:

* **How far is each arm from uniform?** The number of modules whose width differs. An arm that
  scores like uniform because it *allocated* like uniform is a different finding from one that
  reallocated heavily and gained nothing.
* **What is the signal's footprint?** ``dq`` and ``shuf`` share an allocator, a budget and a
  byte total, so the modules where they disagree are exactly the modules the measured signal
  moved. Everything the signal is worth is bought there.
* **Which floors were breached?** The allocator downgrades by lowest ROI when floors exceed
  budget rather than silently returning the floor map, so the violation list is the honest
  record of what the budget could not afford.

    python experiments/phase3/s3_allocation/ministral-8b/s3_maps.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

MAPS = Path(__file__).parent / "maps"
ARMS = ("rtn", "rank", "shuf", "dq")
ANCHORS = {"3": "3.25 bits", "4": "4.25 bits"}


def entry(arm: str, anchor: str) -> dict:
    """The map body for one arm at one anchor.

    ``rtn`` carries both anchors in a single file under ``uniform-3``/``uniform-4`` -- which is
    why seven map files cover eight arms -- while every other arm has one file holding one
    budget keyed by its byte total.
    """
    if arm == "rtn":
        both = json.loads((MAPS / "map.rtn.json").read_text(encoding="utf-8"))
        return both["maps"][f"uniform-{anchor}"]
    one = json.loads((MAPS / f"map.{arm}{anchor}.json").read_text(encoding="utf-8"))
    (body,) = one["maps"].values()
    return body


def role(name: str) -> str:
    """``model.layers.7.self_attn.o_proj`` -> ``self_attn.o_proj``; leaf tensors keep their name."""
    inside = re.match(r"model\.layers\.\d+\.(.*)", name)
    return inside.group(1) if inside else name


def layer(name: str) -> int:
    """Layer index, or -1 for the embedding and the head."""
    found = re.match(r"model\.layers\.(\d+)\.", name)
    return int(found.group(1)) if found else -1


def main() -> None:
    for anchor, label in ANCHORS.items():
        bodies = {arm: entry(arm, anchor) for arm in ARMS}
        maps = {arm: body["bits"] for arm, body in bodies.items()}
        names = sorted(maps["rtn"])
        n = len(names)

        print("=" * 96)
        print(
            f"anchor {label} -- {n} quantized modules, all four arms at {bodies['rtn']['nbytes']} B"
        )
        print("=" * 96)
        for arm in ARMS:
            body, assigned = bodies[arm], maps[arm]
            moved = sum(1 for m in names if assigned[m] != maps["rtn"][m])
            breached = Counter(v["role"] for v in body["violations"])
            hist = str(dict(sorted(body["histogram"].items())))
            print(
                f"  {arm + anchor:<6} hist={hist:<40} "
                f"differs from uniform on {moved:>3}/{n}   "
                f"floors breached: {len(body['violations']):>3} "
                f"{dict(breached.most_common(3)) if breached else ''}"
            )

        print("\n  modules assigned identical widths:")
        for a, b in combinations(ARMS, 2):
            same = sum(1 for m in names if maps[a][m] == maps[b][m])
            print(f"    {a}{anchor} vs {b}{anchor}: {same:>3}/{n}  ({same / n * 100:.1f}%)")

        # dq and shuf share allocator, budget and byte total, so their disagreements are the
        # measured signal's entire footprint -- the only modules where it changed an outcome.
        dq, shuf = maps["dq"], maps["shuf"]
        footprint = [m for m in names if dq[m] != shuf[m]]
        touched = sorted({layer(m) for m in footprint if layer(m) >= 0})
        depth = max(layer(m) for m in names) + 1
        print(f"\n  the signal's footprint -- dq{anchor} != shuf{anchor} on {len(footprint)}/{n}:")
        print(f"    by role:  {dict(Counter(role(m) for m in footprint).most_common())}")
        print(f"    layers:   {len(touched)} of {depth}")
        moves = Counter((shuf[m], dq[m]) for m in footprint)
        print(f"    moves:    {dict(sorted(moves.items()))}  (shuf width -> dq width)")
        print()


if __name__ == "__main__":
    main()
