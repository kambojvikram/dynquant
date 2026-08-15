"""The table is the only part of the panel a reader ever sees, so it is where a wrong
number does the most damage and costs the least to produce.

Seven hours of GPU time land as seven json files. Everything after that is arithmetic and
formatting, which means every failure worth covering here is a failure that would print a
plausible number rather than crash:

* a size column filled from the loaded model, which would report 16 bits for the DynQuant
  arms -- they are scored by encoding their widths back into bf16, so what they *hold* and
  what they *cost* are different numbers and only one of them is the claim;
* a byte-matched panel that is not byte-matched, printed anyway;
* a verdict that follows the raw p across a family of six, where two of these six
  comparisons are significant raw and neither survives correction;
* hit vectors from records scored under different settings, paired into a p-value;
* an allocation whose breached floors are not reported, which is the difference between a
  knapsack result and a budget that could not afford a role.

Nothing here loads a model or scores anything. The panel is synthetic and its joint hit
patterns are laid out by hand, so every count in the output is a count this file chose.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "phase4" / "panel_table.py"
DRIVER = REPO_ROOT / "experiments" / "phase4" / "arms_lfm2.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


#: The two anchors the baselines' accounting produces on this checkpoint, and a parameter
#: count in the model's range. Real numbers rather than round ones: the size column's job
#: is to make a 2.3% difference visible, and round numbers hide exactly that.
ANCHORS = {4: 4_399_629_312, 3: 3_332_904_576}
PARAMS = 8_343_543_808

#: Joint correctness patterns for the three 4-bit arms, in ``(dq, gptq, awq)`` order.
#:
#: Laid out as a joint distribution rather than as three independent vectors because the
#: three pairwise tables are not independent: the accuracy differences have to sum around
#: the triangle, and a fixture that ignores that produces a panel no run could produce.
#: These counts give dq-gptq 14/4 (p=0.031) and dq-awq 24/10 (p=0.024) -- both significant
#: raw, neither surviving correction over six. That is the case the verdict column exists
#: for and it cannot be built out of a clean win.
FOUR_BIT = {"TTT": 280, "FFF": 84, "TFF": 13, "TFT": 1, "FTF": 1, "FTT": 3, "TTF": 11, "FFT": 7}

#: The same for the 3-bit arms: dq-gptq 30/4 and dq-awq 28/4, which do survive correction.
#: A fixture where nothing separates cannot tell a broken correction from a strict one.
THREE_BIT = {"TTT": 240, "FFF": 116, "TFF": 20, "TFT": 10, "FTF": 2, "FTT": 2, "TTF": 8, "FFT": 2}

TOTAL = 400


@pytest.fixture(scope="module")
def table() -> Any:
    return _load("_dq_panel_table", SCRIPT)


def _trio(counts: dict[str, int]) -> dict[int, list[bool]]:
    """Three hit vectors realising an exact joint pattern count."""
    vectors: dict[int, list[bool]] = {0: [], 1: [], 2: []}
    for pattern, count in counts.items():
        for _ in range(count):
            for index, flag in enumerate(pattern):
                vectors[index].append(flag == "T")
    for vector in vectors.values():
        assert len(vector) == TOTAL
    return vectors


def _record(
    label: str, hits: list[bool], *, by_source: dict[str, list[int]] | None = None
) -> dict[str, Any]:
    correct = sum(1 for hit in hits if hit)
    packed = (
        {"map": f"maps/{label}.json", "apply": "encode", "group_size": 128, "modules": 300}
        if label.startswith("dq_")
        else None
    )
    return {
        "packed": packed,
        "dynquant_core": "0.3.0",
        "label": label,
        "model": "/runs/s4/merged",
        "task": "text2sql",
        "backend": "transformers",
        "split": "test",
        "shots": 2,
        "shot_seed": 7,
        "limit": TOTAL,
        "accuracy": correct / len(hits),
        "correct": correct,
        "total": len(hits),
        "unparseable": 3,
        "chance": 0.0,
        "seconds": 1800.0,
        "detail": {
            "errored": 5,
            "exact": correct - 20,
            "unfinished_reasoning": 0,
            "prompt_style": "chat",
            "by_source": by_source
            or {
                "gretel": [correct // 2, TOTAL // 2],
                "wikisql": [correct - correct // 2, TOTAL // 2],
            },
        },
        "decode": {
            "max_new_tokens": 1024,
            "batch_size": 8,
            "max_prompt_tokens": 2048,
            "greedy": True,
        },
        "hits": hits,
    }


def _write_map(path: Path, width: int, nbytes: int, *, breach: bool) -> None:
    """One allocation file, written by the writer the allocator uses.

    Not a hand-built dict. The field this fixture originally guessed wrong was
    ``histogram``, which counts *modules* -- a dict typed here with parameter-scale values
    agreed with the reader's wrong reading of it and both stayed green. Going through
    ``write_bit_maps`` means the fixture cannot hold a shape the allocator does not produce,
    and the numbers below are chosen so ``BitMap`` derives the ones the table prints.
    """
    from dynquant.allocate.knapsack import BitMap, FloorViolation, Pricing
    from dynquant.commands._shared import write_bit_maps
    from dynquant.graph.roles import ModuleRole

    bits = dict.fromkeys(("m.head", "m.embed", "m.norm", "m.conv", "m.router"), 8)
    bits["m.tiny"] = 2
    bits |= {f"m.layers.{index}.proj": 4 for index in range(181)}
    average = 4.1563 if width == 4 else 3.1488
    denominator = round(nbytes * 8 / average)
    violations = tuple(
        FloorViolation(
            name=f"model.layers.{index}.feed_forward.experts.gate_up_proj",
            role=ModuleRole.MOE_EXPERT_GATE_UP,
            floor_bits=4,
            assigned_bits=3,
            num_params=150_000_000,
        )
        for index in range(22 if breach else 0)
    )
    bit_map = BitMap(
        bits=bits,
        violations=violations,
        budget_bits=float(nbytes * 8),
        allocated_bits=float(nbytes * 8),
        denominator=denominator,
        target_label=f"{nbytes}B",
        # The real panel's shape, not a tidy one: a minority of modules holding the
        # large majority of the parameters is priced by the proxy, because on a
        # batched-expert MoE the measured estimate does not exist for a bank.
        pricing=Pricing(
            measured_modules=143,
            proxied_modules=44,
            measured_params=round(denominator * 0.085),
            proxied_params=round(denominator * 0.915),
            scale=2.5e-13,
        ),
    )
    write_bit_maps(
        path,
        {str(ANCHORS[width]): bit_map},
        model="/runs/s4/merged",
        stats="/runs/s4/stats.json",
        allocator="greedy",
        group_size=128,
    )


def _write_panel(
    out: Path,
    *,
    drift: int = 0,
    breach_at_3b: bool = True,
    omit: tuple[str, ...] = (),
    ceiling_correct: int = 320,
    nulls: tuple[str, ...] = (),
    draws: tuple[tuple[str, int], ...] = (),
    schemes: dict[str, dict[str, Any]] | None = None,
    extra_arms: tuple[dict[str, Any], ...] = (),
) -> Path:
    """A full seven-arm panel on disk, in the shape ``arms_lfm2 run`` writes.

    The manifest goes out through the driver's own ``write_manifest`` rather than through a
    copy of its keys here. A reader tested against a hand-written copy of the writer's shape
    is tested against this file's belief about the writer, which is exactly the belief a
    rename in the driver would leave untouched and green.
    """
    out.mkdir(parents=True, exist_ok=True)
    maps = out / "maps"
    maps.mkdir(exist_ok=True)

    four, three = _trio(FOUR_BIT), _trio(THREE_BIT)
    hits = {
        "bf16": [index < ceiling_correct for index in range(TOTAL)],
        "dq_4b": four[0],
        "gptq_4b": four[1],
        "awq_4b": four[2],
        "dq_3b": three[0],
        "gptq_3b": three[1],
        "awq_3b": three[2],
    }

    # Label, mode and seed for every control arm: the first draw of each mode, then any
    # further draws of one. Built once and read by both the hits below and the manifest
    # further down, so a redrawn arm cannot end up scored under one name and planned under
    # another -- which is a failure the table would report as a missing row.
    controls = [(f"dq_3b_{mode[:4]}", mode, 0) for mode in nulls]
    controls += [(f"dq_3b_{mode[:4]}s{seed}", mode, seed) for mode, seed in draws]

    # A control's hits sit between the arm it controls and the baseline it is measured
    # against, which is the only shape that makes both of its rows non-degenerate: a
    # control identical to `dq_3b` gives a zero signal row and a control identical to
    # `gptq_3b` gives a zero shape row, and either one would let a printer that dropped a
    # row still pass. Each control hands back a little more than the one before it, so the
    # ladder's rungs are all positive and a reversed row shows up as a sign. Deterministic
    # and index-based rather than sampled, because a fixture that reseeds is a fixture
    # whose failures do not reproduce.
    for index, (label, _, _) in enumerate(controls):
        share = 3 + index
        hits[label] = [
            base if (mine and not base and position % share) else mine
            for position, (mine, base) in enumerate(
                zip(hits["dq_3b"], hits["gptq_3b"], strict=True)
            )
        ]

    arms: list[dict[str, Any]] = [
        {"label": "bf16", "kind": "ceiling", "anchor": None, "target_bytes": None, "nbytes": None}
    ]
    for width in (4, 3):
        for kind in ("gptq", "awq", "dq"):
            label = f"{kind}_{width}b"
            entry: dict[str, Any] = {
                "label": label,
                "kind": kind,
                "anchor": width,
                "target_bytes": ANCHORS[width],
                "nbytes": ANCHORS[width],
            }
            if kind == "dq":
                path = maps / f"{label}.json"
                entry["nbytes"] = ANCHORS[width] + (drift if width == 4 else 0)
                entry["map"] = str(path)
                _write_map(path, width, entry["nbytes"], breach=width == 3 and breach_at_3b)
            arms.append(entry)
    for label, mode, seed in controls:
        path = maps / f"{label}.json"
        _write_map(path, 3, ANCHORS[3], breach=breach_at_3b)
        arms.append(
            {
                "label": label,
                "kind": "dq",
                "anchor": 3,
                "target_bytes": ANCHORS[3],
                "nbytes": ANCHORS[3],
                "map": str(path),
                "score_null": {"mode": mode, "seed": seed},
            }
        )

    # An extra arm needs hits before the loop below asks for them. Built on the same
    # ladder as a control's -- a little better than `gptq_3b` and short of `dq_3b` --
    # so a scheme arm added here is a live row in every comparison it appears in and
    # not a degenerate one that would let a dropped row still pass.
    for index, entry in enumerate(extra_arms):
        share = 3 + len(controls) + index
        hits.setdefault(
            entry["label"],
            [
                base if (mine and not base and position % share) else mine
                for position, (mine, base) in enumerate(
                    zip(hits["dq_3b"], hits["gptq_3b"], strict=True)
                )
            ],
        )
    arms.extend(dict(entry) for entry in extra_arms)
    for arm in arms:
        label = arm["label"]
        if label in omit:
            arm["record"] = None
            continue
        record = out / f"{label}.json"
        record.write_text(json.dumps(_record(label, hits[label]), indent=2), encoding="utf-8")
        arm["record"] = str(record)
        if arm["kind"] in ("gptq", "awq"):
            # No `symmetric` key unless a test asks for one, because the arms this
            # panel was built from have none: both scheme flags postdate them. That
            # absence is the case `scheme_of` recovers, so it is the fixture default.
            payload: dict[str, Any] = {
                "method": arm["kind"],
                "bits": arm["anchor"],
                "params": PARAMS,
            }
            payload.update((schemes or {}).get(label, {}))
            record.with_suffix(".quant.json").write_text(json.dumps(payload), encoding="utf-8")

    driver = _load("_dq_arms_lfm2", DRIVER)
    driver.write_manifest(
        out,
        argparse.Namespace(
            model="/runs/s4/merged",
            stats="/runs/s4/stats.json",
            moments=None,
            group_size=128,
        ),
        ANCHORS,
        [
            driver.Arm(
                label=arm["label"],
                kind=arm["kind"],
                anchor=arm["anchor"],
                target_bytes=arm["target_bytes"],
                nbytes=arm["nbytes"],
                record=arm.get("record"),
                extra={key: arm[key] for key in ("map", "score_null") if key in arm},
            )
            for arm in arms
        ],
    )
    return out


def _run(table: Any, out: Path, *extra: str) -> str:
    buffer = StringIO()
    with redirect_stdout(buffer):
        table.main(["--arms", str(out), *extra])
    return buffer.getvalue()


def _run_refusing(table: Any, out: Path) -> str:
    """What the table printed before it refused, which is where the offending row is."""
    buffer = StringIO()
    with pytest.raises(SystemExit, match="byte-matched"), redirect_stdout(buffer):
        table.main(["--arms", str(out)])
    return buffer.getvalue()


def test_holm_is_step_down_and_monotone(table: Any) -> None:
    """Plain Bonferroni would multiply every p by the family size; Holm does not.

    Pinned because the difference is the whole reason to prefer it, and because a
    correction that silently degrades to Bonferroni costs real findings without changing
    a single line of output format. The monotone clause is the other half: without it the
    adjusted values are not p-values, and a smaller raw p can print a smaller adjusted p
    than a larger one that outranks it.
    """
    assert table.holm([0.04]) == [0.04]

    adjusted = table.holm([0.01, 0.02, 0.03, 0.04])
    assert adjusted[0] == pytest.approx(0.04), "smallest p takes the full family multiplier"
    assert adjusted[1] == pytest.approx(0.06), "second takes m-1, which Bonferroni would not"
    assert adjusted == sorted(adjusted), "adjusted order follows raw order"

    # The step-down value for the second entry would be 3 * 0.001 = 0.003, below the first
    # entry's 0.004. Monotonicity pulls it up rather than letting it print smaller.
    assert table.holm([0.001, 0.001, 0.5]) == pytest.approx([0.003, 0.003, 0.5])
    assert all(0.0 <= value <= 1.0 for value in table.holm([0.4, 0.5, 0.9]))


def _verdict(line: str) -> str:
    """The verdict, with the expert-arithmetic mark stripped off the end.

    The mark is a separate claim -- "these two arms are not known to have run the same
    arithmetic" -- and a test about which p the verdict followed should not have to know
    whether it is there.
    """
    return line.rstrip().removesuffix("!").rstrip()


def _question(line: str) -> str:
    """The comparison's name, read off the fixed-width field the printer writes it into."""
    return line[:28].strip()


def _aggregate_block(printed: str) -> str:
    """The head-to-head block alone, bounded at the first partition block's title.

    Every block below it prints rows beginning with the same six comparison names, so a
    parser that stops at a landmark further down the table silently returns some of theirs.
    Four of the copies this replaced were bounded at ``"head to head, on"``, which is a
    title the table only prints when it was given sources -- so on the panels that had none
    they bounded nothing at all, and passed until a block that does not need sources was
    added between them and their landmark.

    Every partition block titles itself ``... on <subset> alone (n of m items)`` and the
    aggregate does not, which makes `" alone ("` the one bound that stays correct when a
    fourth partition arrives.
    """
    assert "head to head, at matched bytes" in printed, printed
    return printed.split("head to head, at matched bytes")[1].split(" alone (")[0]


def test_the_verdict_follows_the_corrected_p_and_not_the_raw_one(
    table: Any, tmp_path: Path
) -> None:
    """Two of the six 4-bit comparisons are significant raw and neither survives Holm.

    This is the case the correction exists for, and it is also the case where getting it
    wrong is most tempting: the raw p prints 0.031 next to a +2.5 point delta, and a table
    that called that separated would be publishing the panel's headline off a family-wise
    error rate of about a quarter. The 3-bit arms are built to survive, so a correction
    that simply refused everything would fail here too.
    """
    out = _write_panel(tmp_path / "arms")
    printed = _run(table, out)

    four_bit = next(
        line for line in printed.splitlines() if line.startswith("4b  DynQuant vs GPTQ")
    )
    assert "0.0309" in four_bit, "the raw p is still printed"
    assert "0.0972" in four_bit, "and so is the adjusted one"
    assert _verdict(four_bit).endswith("NOT separated"), "the verdict is the corrected one"

    three_bit = next(
        line for line in printed.splitlines() if line.startswith("3b  DynQuant vs GPTQ")
    )
    assert _verdict(three_bit).endswith("separated")
    assert not _verdict(three_bit).endswith("NOT separated")
    assert "Holm-adjusted over 6 of 6 comparisons in this block" in printed


def test_the_size_column_is_the_manifests_and_the_ceiling_is_derived_from_a_count_on_disk(
    table: Any, tmp_path: Path
) -> None:
    """A DynQuant arm holds fp16 and costs the allocator's bytes; the table prints the cost.

    Both numbers are true and only one is the claim, so the column has to come from the
    manifest -- and the record has to keep saying ``encode``, which is how a reader knows
    the resident size is not the printed one.

    The ceiling's size is the one derived number in the table. It comes from a baseline's
    own parameter count, so the compression ratio is between two figures denominated in the
    same tensors; a literal count written for this model would be silently wrong on the
    next one, which is the failure that produced this rule.
    """
    out = _write_panel(tmp_path / "arms")
    printed = _run(table, out)

    expected_gib = f"{ANCHORS[4] / 2**30:.3f}"
    dq_line = next(line for line in printed.splitlines() if line.startswith("dq_4b "))
    assert expected_gib in dq_line, "the dq arm claims the allocator's bytes, not fp16"
    assert f"{ANCHORS[4] * 8 / PARAMS:.4f}" in dq_line

    bf16_line = next(line for line in printed.splitlines() if line.startswith("bf16 "))
    assert f"{PARAMS * 2 / 2**30:.3f}" in bf16_line
    assert "16.0000" in bf16_line, "the ceiling denominates in the same parameters"
    assert f"{PARAMS:,} parameters" in printed
    assert "encode" in dq_line, "and the record still says how the map reached the weights"

    # And the ceiling column is a real ratio between them rather than a nominal 16/4.
    assert f"{PARAMS * 2 / ANCHORS[4]:.2f}x" in dq_line


def test_a_panel_that_is_not_byte_matched_prints_no_comparisons(table: Any, tmp_path: Path) -> None:
    """0.5% off the anchor is a size advantage that will be read as accuracy.

    ``arms_lfm2`` refuses at run time, and this refuses again at read time, because the
    directory this script is pointed at need not be the one that run assembled. Loud
    rather than a footnote: a table that prints a drift column and then the comparisons
    underneath it has already told the reader the numbers are comparable.
    """
    # Both directions. Under-budget is the one that actually happens: --target-size is a
    # ceiling and an allocator that cannot spend the last few bits lands beneath it, so a
    # signed check would wave through the only case there is.
    for name, drift, shown in (
        ("over", 22_000_000, "+0.5000%"),
        ("under", -22_000_000, "-0.5000%"),
    ):
        printed = _run_refusing(table, _write_panel(tmp_path / name, drift=drift))
        row = next(line for line in printed.splitlines() if line.startswith("dq_4b "))
        assert shown + "!" in row, "the offending arm is marked, not just the panel"
        assert "head to head" not in printed, "and nothing comparable is printed under it"

    # Within tolerance the same panel prints through to the comparisons.
    printed = _run(table, _write_panel(tmp_path / "ok", drift=1_000_000))
    assert "head to head, at matched bytes" in printed
    assert "+0.0227%" in printed, "the drift is stated even when it passes"
    assert "!" not in printed.split("head to head")[0], "and carries no marker when it passes"


def test_records_scored_under_different_settings_are_not_paired(table: Any, tmp_path: Path) -> None:
    """A leftover record's only claim to provenance is its filename.

    The pairing contract lives in the eval command and is read from there, so a field
    added to it reaches this guard without a second copy being updated. What is pinned
    here is that a mismatch turns into no p-value at all rather than into a p-value over
    two different problem sets -- and that it does so for the comparisons the stale arm is
    actually in, not for the whole table.

    That second half is the part that used to be wrong. `pairable` was one string for the
    panel, so a single arm scored at the wrong decode budget blanked `GPTQ vs AWQ` too, a
    row it appears nowhere in. Comparability is a property of a pair.

    Turns red when: the check goes back to a panel-wide flag, or the row stops naming the
    field and an operator has to diff two 120 KB records to find it.
    """
    out = _write_panel(tmp_path / "arms")
    record = out / "gptq_4b.json"
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["decode"]["max_new_tokens"] = 320
    record.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    printed = _run(table, out)
    block = _aggregate_block(printed)
    rows = {_question(line): line for line in block.splitlines() if line.startswith("4b ")}
    assert "NOT PAIRED" in printed
    assert "not comparable: decode.max_new_tokens" in rows["4b  DynQuant vs GPTQ"]
    assert "not comparable" not in rows["4b  DynQuant vs AWQ"], (
        "neither arm of this row is the stale one, so it is still a comparison"
    )
    assert _verdict(rows["4b  DynQuant vs AWQ"]).endswith("separated")
    assert "p (Holm)" in printed, "the block still prints, so the absence is visible"
    assert not _verdict(rows["4b  DynQuant vs GPTQ"]).endswith("separated"), (
        "no verdict is issued off unpaired vectors"
    )


def test_the_allocation_reports_which_floors_the_budget_could_not_afford(
    table: Any, tmp_path: Path
) -> None:
    """Zero breaches and twenty-two breaches are different results at the same average bits.

    Pre-registered before either arm ran: at 4 bits the floors leave slack and the
    allocation is a knapsack over it; at 3 bits they cannot fit and the expert gate_up
    banks have to be breached whatever the signal says. Neither fact can be recovered from
    the accuracy afterwards, and an arm reported without them reads as though the allocator
    was free to choose when it was not.
    """
    out = _write_panel(tmp_path / "arms")
    printed = _run(table, out)

    assert "dq_4b: 4.1563 avg bits" in printed
    assert "floors: none breached -- the budget was not binding on any role" in printed
    assert "dq_3b: 3.1488 avg bits" in printed
    assert "floors: 22 breached" in printed
    # `moe.expert.gate_up`, the role's own value -- the hand-written fixture this replaced
    # asserted an invented `moe_expert_gate`, which no allocation has ever emitted.
    assert "moe.expert.gate_up" in printed, "the breached role is named, not just counted"
    assert "3.30G params" in printed, "the breached mass is a parameter count, at its scale"
    # The histogram counts modules. Printed through the parameter formatter, all three
    # widths rendered as `0K` -- the allocator's whole answer, shown as nothing assigned.
    assert "widths, modules at each: 2b 1  4b 181  8b 5   (187 quantized)" in printed

    clean = _write_panel(tmp_path / "clean", breach_at_3b=False)
    assert "floors: 22 breached" not in _run(table, clean)


def test_an_arm_that_did_not_run_is_a_missing_row_and_not_a_missing_comparison(
    table: Any, tmp_path: Path
) -> None:
    """Half a panel is a normal state to read a table in, and it must not be silent.

    The comparisons that needed the absent arm say so by name. A block that simply omitted
    them would make a five-comparison Holm family look like a six-comparison one that
    happened to agree, and would hide which arm still has to run.
    """
    out = _write_panel(tmp_path / "arms", omit=("awq_3b",))
    printed = _run(table, out)

    assert any(
        line.startswith("awq_3b") and line.endswith("not run") for line in printed.splitlines()
    )
    # Four blocks ask the head-to-head family on this panel -- the aggregate, the two
    # difficulty strata, and the fidelity family -- and two of its six comparisons name
    # awq_3b, so eight rows do. The ceiling family names it once more. The per-source and
    # source-by-difficulty blocks do not run here: `_write_panel` writes no sources.json.
    assert printed.count("(needs both arms)") == 9
    assert "3/7 arms scored" not in printed and "6/7 arms scored" in printed
    assert "Holm-adjusted over 4 of 6 comparisons" in printed, "the head family shrinks with it"
    assert "Holm-adjusted over 5 of 6 comparisons" in printed, "and so does the ceiling family"
    # A short family is corrected less than the finished panel will be, so an adjusted
    # p read mid-run can only move the unfavourable way -- and the note says which way,
    # because "weaker" reads as either the evidence or the number and they point opposite
    # ways here. Once per block that ran short: the four above plus the ceiling at five of six.
    assert printed.count("a short family, so these adjusted p can only rise") == 5


def test_a_named_comparison_joins_the_family_and_a_misnamed_one_is_refused(
    table: Any, tmp_path: Path
) -> None:
    """A control run after the panel has to be tested against it, not beside it.

    The six standing rows are a constant because every campaign makes them. A control is
    the other case -- one more arm at an anchor already scored, whose whole point is a
    paired test against an arm already in the table. Left out, its number would be computed
    in a script beside the panel and cited from there, which is the second copy of an
    arithmetic nothing in the table could contradict.

    Joining the family is the claim being tested, not merely appearing: it has to be
    corrected with the others, or its p is read against a threshold nothing else in the
    block was held to.

    Turns red when: a named comparison stops reaching the block, stops entering the Holm
    family, stops reaching the per-source and per-difficulty partitions, or an unknown
    label starts printing the standing family's "(needs both arms)" placeholder -- which
    for a comparison the caller asked for by name would read as a test that was run.
    """
    out = _write_panel(tmp_path / "arms", nulls=("shuffle",))

    plain = _run(table, out)
    assert "DynQuant vs its own shuffle" not in plain
    assert "Holm-adjusted over 6 of 6 comparisons" in plain

    named = _run(table, out, "--compare", "dq_3b:dq_3b_shuf:3b  DynQuant vs its own shuffle")
    assert "3b  DynQuant vs its own shuffle" in named
    assert "Holm-adjusted over 7 of 7 comparisons" in named
    # The partitions run the same family, so the row is in the per-difficulty blocks too --
    # one appearance would mean it reached the headline and none of the stratifications.
    assert named.count("3b  DynQuant vs its own shuffle") > 1

    for spec, expected in (
        ("dq_3b:dq_3b_typo:whatever", "which the manifest does not"),
        ("dq_3b:dq_3b_shuf", "is not LEFT:RIGHT:QUESTION"),
    ):
        with pytest.raises(SystemExit, match=expected):
            table.main(["--arms", str(out), "--compare", spec])


def test_a_block_reads_in_the_families_declared_order_however_much_of_it_ran(
    table: Any, tmp_path: Path
) -> None:
    """The rows a partial panel can compute belong where the family put them.

    Classification and printing were one loop, so a skipped comparison printed immediately
    and a computed one printed after every skip. At two arms of seven that puts the single
    comparison the panel *can* make underneath five placeholders -- the ordering only reads
    correctly once nothing is missing, which is when it has stopped mattering. This is the
    state a running panel is read in, repeatedly, for a day and a half.
    """
    # Dropping a 3-bit arm leaves the four earlier rows computable and the last two not,
    # so declared order and availability order disagree and the old code is discriminated
    # against: it printed the two skips first.
    out = _write_panel(tmp_path / "arms", omit=("awq_3b",))
    printed = _run(table, out)

    block = _aggregate_block(printed).splitlines()
    rows = [line for line in block if line[:2] in {"4b", "3b"}]
    assert [line[:28].rstrip() for line in rows] == [q for _, _, q in table.HEAD_TO_HEAD], (
        "every row is present and in the family's order, computed or not"
    )
    skipped = [i for i, line in enumerate(rows) if "(needs both arms)" in line]
    assert skipped == [4, 5], "the placeholders sit where the family put them, not on top"


def test_a_manifest_read_from_another_directory_still_finds_its_records(
    table: Any, tmp_path: Path
) -> None:
    """The record paths a run stores are only meaningful from the directory it ran in.

    ``do_run`` stores ``str(out / f"{label}.json")``, so ``--out runs/s4/arms`` writes seven
    relative paths. Read from any other cwd -- the repo root, a laptop the directory was
    copied to -- every one of them misses. Nothing raises: each arm is simply not scored, and
    the table prints ``0/7`` for a panel that finished, with the anchors and the allocation
    still correct above it, which is the shape of an answer rather than of a failure.

    The manifest names three kinds of path and they must all survive the move, which is why
    this rewrites the map as well as the record. Records resolve, and the panel looks whole --
    while the fp16 ceiling loses the parameter count it reads from the baseline's
    ``.quant.json`` side file, and every DynQuant arm loses the allocation the §12 prediction
    is about. Both print as absence rather than as an error.

    The map is the case that cannot be fixed by filename alone: it lives in ``out/maps/`` and
    the record of the same arm lives in ``out/``, so a bare-name retry reads the record, finds
    no ``maps`` key, and reports no allocation -- the wrong file, read successfully.

    Turns red when: the fallback beside the manifest is dropped, applied to only one of the
    three path kinds, or made to prefer the shortest tail over the longest.
    """
    out = _write_panel(tmp_path / "arms")
    manifest = out / "arms.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for arm in payload["arms"]:
        arm["record"] = f"runs/s4/arms/{arm['label']}.json"
        if arm.get("map"):
            arm["map"] = f"runs/s4/arms/maps/{arm['label']}.json"
    manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    printed = _run(table, out)
    assert "panel: 7/7 arms scored" in printed
    assert "4b  DynQuant vs GPTQ" in printed
    assert f"denominated in {PARAMS:,} parameters" in printed, "the fp16 row lost its count"
    assert "dq_3b: 3.1488 avg bits" in printed, "the allocation was read from the wrong file"
    assert "moe.expert.gate_up" in printed, "the floor breach vanished with the map"


def test_the_json_carries_the_verdict_the_table_printed(table: Any, tmp_path: Path) -> None:
    """The report quotes the json, so the two must not be able to disagree.

    Specifically the adjusted p and the boolean beside it: a json that carried only the
    raw p would let a writeup call separated exactly the comparison the table refused to.
    """
    out = _write_panel(tmp_path / "arms")
    printed = _run(table, out, "--json")
    payload = json.loads(printed[printed.index("{") :])

    assert payload["params"] == PARAMS
    assert payload["pairable"] is True
    assert {arm["label"] for arm in payload["arms"]} == {
        "bf16",
        "gptq_4b",
        "awq_4b",
        "dq_4b",
        "gptq_3b",
        "awq_3b",
        "dq_3b",
    }
    assert next(a for a in payload["arms"] if a["label"] == "dq_4b")["apply"] == "encode"
    assert next(a for a in payload["arms"] if a["label"] == "gptq_4b")["apply"] is None

    head = {(entry["left"], entry["right"]): entry for entry in payload["head_to_head"]}
    borderline = head[("dq_4b", "gptq_4b")]
    assert borderline["p_value"] < 0.05 < borderline["p_adjusted"]
    assert borderline["separated"] is False
    assert head[("dq_3b", "gptq_3b")]["separated"] is True
    assert all(
        entry["separated"] == (entry["p_adjusted"] < 0.05)
        for entry in payload["head_to_head"] + payload["against_ceiling"]
    )


def test_the_json_carries_the_mark_and_the_file_is_what_the_flag_printed(
    table: Any, tmp_path: Path
) -> None:
    """A downstream reader gets the caveat or it gets a delta with the caveat removed.

    The printed table says two things about every comparison: what it is called, and --
    with a trailing ``!`` -- whether its two arms are known to have run the same expert
    arithmetic. Both were dropped on the way into json, which was survivable while the only
    consumer was a person reading the terminal. It is not survivable now: `model_cards.py`
    builds six Hub READMEs from this payload, and a serialisation that carries
    ``separated`` without ``same_arithmetic`` publishes a verdict with its confound
    stripped -- on this model a dispatch difference worth 0.29x the effect being reported.

    ``--json-out`` is checked against ``--json`` rather than parsed on its own, because the
    file exists so a downstream tool does not have to find where the human output stopped.
    Two constructions of the payload would be two tables, which is the thing this whole
    split exists to prevent.

    Turns red when: ``as_json`` drops either field, or the file and the stream are built
    separately and diverge.
    """
    out = _write_panel(tmp_path / "arms")
    for label in ("gptq_4b", "awq_4b", "gptq_3b", "awq_3b"):
        _set_linearization(out, label, {"banks_before": 22, "banks_after": 0})

    printed = _run(table, out, "--json")
    payload = json.loads(printed[printed.index("{") :])
    block = _aggregate_block(printed)
    marked = {
        _question(line)
        for line in block.splitlines()
        if line[:2] in {"4b", "3b"} and line.rstrip().endswith("!")
    }
    assert marked == {
        "4b  DynQuant vs GPTQ",
        "4b  DynQuant vs AWQ",
        "3b  DynQuant vs GPTQ",
        "3b  DynQuant vs AWQ",
    }, "the four rows pairing an unrecorded DynQuant arm against a recovered baseline"

    entries = {entry["question"].strip(): entry for entry in payload["head_to_head"]}
    assert set(entries) == {q for _, _, q in table.HEAD_TO_HEAD}
    assert {q for q, e in entries.items() if not e["same_arithmetic"]} == marked

    dest = tmp_path / "table.json"
    _run(table, out, "--json-out", str(dest))
    assert json.loads(dest.read_text(encoding="utf-8")) == payload


def test_the_per_source_columns_show_a_collapse_the_headline_hides(
    table: Any, tmp_path: Path
) -> None:
    """The mixture exists so a one-source collapse is visible; the table has to print it.

    A method that destroys one dataset's distribution and leaves the others alone moves
    the headline a couple of points. That is indistinguishable from mild damage everywhere,
    and it is the difference between a usable arm and one that cannot be shipped.
    """
    out = _write_panel(tmp_path / "arms")
    record = out / "awq_3b.json"
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["detail"]["by_source"] = {"gretel": [190, 200], "wikisql": [12, 200]}
    record.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    printed = _run(table, out)
    lines = printed.splitlines()
    start = next(
        index for index, line in enumerate(lines) if "gretel" in line and "wikisql" in line
    )
    row = next(line for line in lines[start:] if line.startswith("awq_3b "))
    assert "95.0% (200)" in row and "6.0% (200)" in row


def test_the_allocation_block_says_which_price_chose_the_widths(tmp_path: Path, table) -> None:
    """A DynQuant arm's widths are only evidence about the signal if the signal set them.

    On this architecture most of the model is priced by the rank-product proxy rather
    than by measured sensitivity, and nothing in the table said so: the arm printed its
    average bits, its width histogram and its floor breaches, all of which are true and
    none of which distinguish "the measurement decided this" from "the measurement was
    unavailable for 91.5% of it". The share is asserted rather than the module count
    because the count is the reassuring one.
    """
    out = _write_panel(tmp_path / "arms")
    printed = _run(table, out)

    assert "priced: 143 modules measured, 44 from the score proxy" in printed
    assert "91.5% of parameters" in printed, "the count without the mass is the wrong story"
    assert "rescaled by 2.500e-13" in printed


def test_an_uncalibrated_price_is_called_out_not_left_blank(table) -> None:
    """``scale: null`` is the run whose bit map should not be trusted.

    It means a mix of two prices with no overlap to calibrate on, so the proxy-priced
    modules sit at an arbitrary offset against the measured ones -- and it renders
    identically to a healthy run unless the table says the word.
    """
    line = table.describe_pricing(
        {
            "measured_modules": 89,
            "proxied_modules": 44,
            "proxied_share": 0.915,
            "scale": None,
        }
    )
    assert "NO COMMON SCALE" in line
    assert "arbitrary" in line


def test_a_run_that_measured_nothing_does_not_read_as_a_measured_run(table) -> None:
    """All-proxy is a legitimate allocation and an illegitimate claim about the signal."""
    line = table.describe_pricing(
        {"measured_modules": 0, "proxied_modules": 133, "proxied_share": 1.0, "scale": 1.0}
    )
    assert "nothing was measured" in line


# --- which arithmetic each arm ran ----------------------------------------------------


def _set_experts(out: Path, label: str, value: Any) -> None:
    """Put an ``experts`` block into a stored record, or a bare ``null``."""
    record = out / f"{label}.json"
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["experts"] = value
    record.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _set_linearization(out: Path, label: str, value: Any) -> None:
    """Put a ``linearization`` block into the arm's ``.quant.json`` side file."""
    side = out / f"{label}.quant.json"
    payload = json.loads(side.read_text(encoding="utf-8"))
    payload["linearization"] = value
    side.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_a_null_dispatch_and_a_missing_one_are_the_same_absence_to_the_guard(
    table: Any, tmp_path: Path
) -> None:
    """The straddle a panel can make without being told, and where it becomes visible.

    ``_pin_experts_dispatch`` writes ``null`` for a model whose config carries no
    ``_experts_implementation`` -- a dense model. A record written before the field existed
    has no key. ``_comparability`` reads both as ``_ABSENT`` and pairs them, which is
    correct for the ``null`` and is carried along for free by the missing key.

    So a panel can hold "this model had nothing to dispatch" beside "we do not know what
    this ran" and the guard will not say so. On LFM2.5-8B-A1B the two dispatches disagree
    on 1.24% of teacher-forced tokens, 0.29x what quantizing to 4 bits moves, so the
    unknown is on the same axis as the result. The last assertion is the reason the block
    exists: nothing else in the table reports it.

    Turns red when: the two absences stop rendering differently, the warning stops naming
    the arms it applies to, or the block is dropped because "the pairing guard covers it".
    """
    out = _write_panel(tmp_path / "arms")
    _set_experts(out, "gptq_4b", None)

    printed = _run(table, out)
    block = printed.split("experts dispatch")[1]
    assert "none (dense)" in block, "a null is a fact about the model, not a gap"
    assert "not recorded" in block, "a missing key is a gap, and has to read as one"
    named = block.split("no linearization report: ")[1].split(".")[0]
    assert "bf16" in named and "gptq_4b" not in named
    assert "NOT PAIRED" not in printed, (
        "the guard reads both absences as _ABSENT and pairs them, which is why this "
        "block is the only place the difference appears"
    )


def test_a_recorded_dispatch_is_priced_on_the_row_and_does_not_refuse_it(
    table: Any, tmp_path: Path
) -> None:
    """A dispatch difference is a caveat on a comparison, not a reason to withhold it.

    This reverses an earlier decision in this file, so the reason is worth writing down.
    ``experts.ran`` went into the pairing contract because two arms on different
    dispatches produce a delta contaminated by dispatch, which is true. The mistake was
    the response: "paired" has a technical meaning -- the two hit vectors index the same
    items in the same order -- and a dispatch difference does not break it. The concern is
    about what the delta *means*, and the panel already has a place to say that, which is
    the ``!`` mark and the priced footnote under the block.

    What forced the issue is that the guard, operating exactly as designed, destroyed the
    result it guards. Re-scoring three arms on ``--experts-impl eager`` gives them
    ``experts.ran`` while the four ``llm-compressor`` baselines predate the field, the
    panel-wide flag fires, and every row in every block -- including the two that pair
    baselines with each other -- goes to "not comparable". The re-score would clear the
    caveat and blank the numbers it annotates.

    So the delta prints, the mark says the arms are not known to have matched, and the
    magnitude is on screen. Strictly more information than a blank row.

    Turns red when: the experts block re-enters ``problem_set_difference``, or the row
    prints a number with the caveat dropped.
    """
    out = _write_panel(tmp_path / "arms")
    _set_experts(out, "dq_4b", {"found": "grouped_mm", "ran": "eager"})

    printed = _run(table, out)
    block = _aggregate_block(printed)
    rows = {_question(line): line for line in block.splitlines() if line.startswith("4b ")}
    assert "eager (from grouped_mm)" in printed.split("experts dispatch")[1]
    assert "NOT PAIRED" not in printed, "the dispatch is not a problem-set difference"
    assert "not comparable" not in block
    assert rows["4b  DynQuant vs GPTQ"].rstrip().endswith("!"), "priced, not withheld"
    assert _verdict(rows["4b  DynQuant vs GPTQ"]).endswith("separated")
    assert "1.24%" in block and "0.29x" in block


def test_a_bank_count_from_the_scoring_process_recovers_a_missing_dispatch(
    table: Any, tmp_path: Path
) -> None:
    """A missing field is not always a missing fact, and the difference is auditable.

    ``baselines_lfm2.do_run`` linearises, calibrates and scores one object: ``quantize``
    returns the model in memory and ``score`` hands that same object to ``evaluate.run``
    with no save and no reload. So ``banks_after: 0`` in the ``.quant.json`` beside the
    record is a count of the weights that were then scored, and a model with no batched
    bank has no grouped kernel to take -- the arithmetic was the loop.

    That matters because the alternative reading, "four arms ran we-know-not-what", makes
    the panel's main comparison unfalsifiable and argues for ~22 GPU-hours of re-runs. The
    real unknown is the complement: the arms with no such report. This pins that the two
    are told apart, and that the recovered arms drop out of the re-score list.

    Turns red when: the recovery is dropped and every field-less record reads as unknown,
    or it widens to treat any ``linearization`` block as proof regardless of the count.
    """
    out = _write_panel(tmp_path / "arms")
    for label in ("gptq_4b", "awq_4b", "gptq_3b", "awq_3b"):
        _set_linearization(out, label, {"banks_before": 22, "banks_after": 0})

    printed = _run(table, out)
    block = printed.split("experts dispatch")[1]
    assert "loop (22 banks -> 0)" in block, "the count is the evidence and has to be shown"

    recovered = block.split("linearised to zero banks: ")[1].split(".")[0]
    assert all(label in recovered for label in ("gptq_4b", "awq_4b", "gptq_3b", "awq_3b"))

    unknown = block.split("no linearization report: ")[1].split(".")[0]
    assert "bf16" in unknown and "dq_4b" in unknown and "dq_3b" in unknown, (
        "the arms with no report are the re-score set, and the block has to name them"
    )
    assert "gptq_4b" not in unknown, "recovered is not unknown"


def test_a_linearization_that_left_banks_behind_recovers_nothing(
    table: Any, tmp_path: Path
) -> None:
    """The recovery is the zero, not the presence of a report.

    ``load_linearized`` refuses a run whose ``banks_after`` is non-zero, so this state
    should not reach a stored panel -- which is exactly why the reader must not treat the
    block's existence as the fact. A partially linearised model still holds a batched bank
    for the grouped kernel to take, and what it dispatched is then unknown in the full
    sense.

    Turns red when: the check softens to ``"linearization" in payload`` or to a truthiness
    test on the report, either of which would certify an arm that still had a bank.
    """
    out = _write_panel(tmp_path / "arms")
    _set_linearization(out, "gptq_4b", {"banks_before": 22, "banks_after": 3})

    printed = _run(table, out)
    block = printed.split("experts dispatch")[1]
    assert "linearised to zero banks" not in block, "three banks is not zero banks"
    unknown = block.split("no linearization report: ")[1].split(".")[0]
    assert "gptq_4b" in unknown


def _all_indexed(out: Path) -> None:
    """Put every arm on the indexed path: recovered for the baselines, recorded for the rest."""
    for label in ("gptq_4b", "awq_4b", "gptq_3b", "awq_3b"):
        _set_linearization(out, label, {"banks_before": 22, "banks_after": 0})
    for label in ("bf16", "dq_4b", "dq_3b"):
        _set_experts(out, label, {"found": "grouped_mm", "ran": "eager"})


def test_a_verdict_on_two_dispatches_is_marked_and_the_reason_is_priced(
    table: Any, tmp_path: Path
) -> None:
    """The panel's headline line, with the confound that sits inside it.

    `4b DynQuant vs GPTQ` is the sentence this campaign exists to write. On the landed
    panel the DynQuant arm kept its batched bank and the GPTQ arm was linearised to none,
    so the delta contains a dispatch difference worth 1.24% of teacher-forced tokens --
    0.29x the quantization effect -- on top of the method. A reader who takes `separated`
    from that row without the dispatch census two blocks up has been misled by this
    script, so the row carries the mark and the block prices it.

    The unflagged row is as much of the test as the flagged ones: `GPTQ vs AWQ` pairs two
    recovered arms and must stay clean, or the mark means "this table has a caveat" rather
    than "this comparison has one".

    Turns red when: the flag stops reaching the row, an unknown starts counting as a
    match, or the footnote drops the magnitude and leaves the mark unexplained.
    """
    out = _write_panel(tmp_path / "arms")
    for label in ("gptq_4b", "awq_4b", "gptq_3b", "awq_3b"):
        _set_linearization(out, label, {"banks_before": 22, "banks_after": 0})

    block = _aggregate_block(_run(table, out))
    rows = {
        _question(line): line
        for line in block.splitlines()
        if line.startswith("4b ") or line.startswith("3b ")
    }
    assert rows["4b  DynQuant vs GPTQ"].rstrip().endswith("!")
    assert rows["4b  DynQuant vs AWQ"].rstrip().endswith("!")
    assert not rows["4b  GPTQ vs AWQ"].rstrip().endswith("!"), (
        "two recovered arms ran the same arithmetic and a mark on them makes the mark noise"
    )
    assert "dq_4b = unrecorded, gptq_4b = indexed" in block
    assert "1.24%" in block and "0.29x" in block, (
        "a mark whose size the reader cannot see is a mark they will learn to skip"
    )


def test_the_mark_clears_when_every_arm_is_on_the_indexed_path(table: Any, tmp_path: Path) -> None:
    """What the re-score is for, stated as the condition that clears the flag.

    `eager` and the linearised loop are one class: both take a single expert at a time,
    and on a four-layer model their outputs are bitwise identical. So once the three arms
    that kept their banks are re-scored on `eager`, every comparison pairs `indexed` with
    `indexed` and the caveat is gone -- while `_comparability` still refuses the four
    field-less records, which is the distinction the census block draws.

    A clean table is the right output here and also the most dangerous one, because the
    collapse holding it up -- `eager` and the linearised loop are one class -- rests on a
    four-layer CPU fp32 model and section 8 is an agreement at small scale that did not
    survive to 8B. So the census prints that claim exactly when both buckets are occupied,
    which is the state the re-score creates. A panel showing no caveats has to say what it
    is resting on.

    Turns red when: `eager` and a recovered loop stop collapsing to one class, which would
    make the re-score unable to clear the flag it was run to clear; or the clean table
    stops naming the unmeasured collapse and reads as though nothing is owed.
    """
    out = _write_panel(tmp_path / "arms")
    _all_indexed(out)

    printed = _run(table, out)
    census = printed.split("experts dispatch")[1].split("by source")[0]
    assert "expert arithmetic" not in printed, "every arm is indexed; nothing to flag"
    assert "NOT PAIRED" not in printed, (
        "the dispatch was the only difference and it is not a problem-set difference"
    )
    assert "3 arm(s) ran `eager` and 4 ran the linearised loop" in census
    assert "did not survive to 8B" in census, "the clean table names what it rests on"
    assert "separated" in printed, "the re-score has to leave the numbers standing"


def test_two_unrecorded_arms_do_not_count_as_agreeing(table: Any, tmp_path: Path) -> None:
    """Absence twice over is not evidence of a match.

    The pairing guard deliberately pairs absence with absence, because a dense model has
    no dispatch and never will. That exemption is about bookkeeping. Applying the same
    leniency to the arithmetic check would let the panel certify a comparison between two
    arms that nothing says anything about, which is the exact reading this block exists to
    prevent.

    Turns red when: the check becomes an inequality test and two `None`s compare equal.
    """
    out = _write_panel(tmp_path / "arms")
    block = _aggregate_block(_run(table, out))
    assert "dq_4b = unrecorded, gptq_4b = unrecorded" in block


# ---------------------------------------------------------------------------
# Per-source heterogeneity.
#
# The per-source blocks predate any test that reaches them, because `_write_panel`
# writes no `sources.json` and `load_sources` returns `(None, None)` without one. So
# these fixtures are the first thing to exercise that path at all, and the labels are
# not written blind: `load_sources` refuses a vector that disagrees with any arm's own
# `by_source` tally, so the helper recomputes every record's tally from the labels it
# just chose. A helper that wrote labels and left the tallies alone would be testing
# the refusal, on every one of these tests, for the rest of the file's life.
# ---------------------------------------------------------------------------


def _write_sources(out: Path, chooser: Any) -> list[str]:
    """Label every item, and rewrite each record's ``by_source`` to agree."""
    labels = [chooser(index) for index in range(TOTAL)]
    (out / "sources.json").write_text(json.dumps(labels), encoding="utf-8")
    for record in sorted(out.glob("*.json")):
        if record.name in {"arms.json", "sources.json"} or record.name.endswith(".quant.json"):
            continue
        payload = json.loads(record.read_text(encoding="utf-8"))
        hits = payload.get("hits")
        if not hits:
            continue
        payload["detail"]["by_source"] = {
            name: [
                sum(1 for hit, src in zip(hits, labels, strict=True) if src == name and hit),
                sum(1 for src in labels if src == name),
            ]
            for name in sorted(set(labels))
        }
        record.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return labels


def _rewrite_hits(out: Path, label: str, hits: list[bool]) -> None:
    """Replace one arm's hit vector, keeping the counts it wrote about itself in step.

    A record whose ``hits`` disagree with its own ``correct`` is a record no run produces,
    and `load_sources` checks every arm against its stored per-source totals -- so a fixture
    that edited only the vector would be rejected, for a reason that has nothing to do with
    what it was written to test.
    """
    path = out / f"{label}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    correct = sum(1 for hit in hits if hit)
    payload["hits"] = hits
    payload["correct"] = correct
    payload["accuracy"] = correct / len(hits)
    payload["detail"]["exact"] = correct - 20
    payload["detail"]["by_source"] = {
        "gretel": [correct // 2, TOTAL // 2],
        "wikisql": [correct - correct // 2, TOTAL // 2],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _split_the_flips(out: Path) -> list[bool]:
    """Rewrite the ceiling so the two difficulty strata hold flips of opposite sign.

    `_write_panel` lays its joint patterns out in contiguous blocks and makes the ceiling a
    prefix of the item order, so every discordant 4-bit pair falls in one stratum and the
    other has none at all. The block under test would still print, with real numbers in it,
    and would pin nothing.

    This is the shape the real panel has: DynQuant leads by +1.18 where the ceiling is right
    and trails by -2.22 where it is wrong. Every dq-over-gptq flip goes to ceiling-right and
    every gptq-over-dq flip to ceiling-wrong, which is that sign change at its extreme; the
    concordant items alternate, so neither stratum is a handful of items.

    Must run before `_write_sources`, which recomputes every record's ``by_source`` from its
    own hits -- including the one this rewrites.
    """
    dq = json.loads((out / "dq_4b.json").read_text(encoding="utf-8"))["hits"]
    gptq = json.loads((out / "gptq_4b.json").read_text(encoding="utf-8"))["hits"]
    hits = []
    for index, (left, right) in enumerate(zip(dq, gptq, strict=True)):
        if left and not right:
            hits.append(True)
        elif right and not left:
            hits.append(False)
        else:
            hits.append(index % 2 == 0)
    _rewrite_hits(out, "bf16", hits)
    return hits


def _difficulty_rows(printed: str, stratum: str) -> dict[str, str]:
    """The rows of one difficulty block, keyed by the comparison they name.

    The near bound is the title's whole ``on <stratum> alone (`` and not the bare stratum
    name, because the crossed blocks below title themselves ``on wikisql/ceiling-wrong
    alone (`` and a parser looking for the name alone would find those too.
    """
    head = f"on {stratum} alone ("
    assert head in printed, printed
    block = printed.split(head)[1].split("is the margin the same at both difficulties?")[0]
    block = block.split(" alone (")[0]
    rows = {}
    for line in block.splitlines():
        for _, _, question in _load("_dq_ds", SCRIPT).HEAD_TO_HEAD:
            if line.startswith(question):
                rows[question.strip()] = line
    return rows


def _stack_the_flips(out: Path) -> list[str]:
    """Every dq-over-gptq flip into gretel, every gptq-over-dq flip into wikisql.

    The extreme the test needs and one no real mixture produces: the same 14/4 the
    aggregate sees, arranged so one source holds all of the wins and the other all of the
    losses. The concordant items alternate, so the two subsets stay near the same size and
    the spread is the effect rather than a difference in how much each source weighs.
    """
    dq = json.loads((out / "dq_4b.json").read_text(encoding="utf-8"))["hits"]
    gptq = json.loads((out / "gptq_4b.json").read_text(encoding="utf-8"))["hits"]

    def chooser(index: int) -> str:
        if dq[index] and not gptq[index]:
            return "gretel"
        if gptq[index] and not dq[index]:
            return "wikisql"
        return "gretel" if index % 2 else "wikisql"

    return _write_sources(out, chooser)


def _heterogeneity_rows(printed: str) -> dict[str, str]:
    """The rows of the heterogeneity block, keyed by the comparison they name.

    Bounded at both ends. Seven blocks in this table print rows that begin with the same
    comparison names -- the head-to-head, the three partitions and their spreads, and the
    fidelity family -- so a parser that only located the header would return whichever of
    them came last, and would go on returning rows after the block under test had stopped
    printing any.

    The far bound is the next block's title rather than a landmark further down the table,
    because the two blocks between this one and `what each method cost` were added after
    this parser was written and it went on returning real rows from the wrong one.
    """
    assert "is the margin the same on every source?" in printed, printed
    block = printed.split("is the margin the same on every source?")[1]
    block = block.split("head to head, by difficulty")[0]
    rows = {}
    for line in block.splitlines():
        for _, _, question in _load("_dq_ht", SCRIPT).HEAD_TO_HEAD:
            if line.startswith(question):
                rows[question.strip()] = line
    return rows


def _fidelity_heterogeneity_rows(printed: str) -> dict[str, str]:
    """The rows of the *fidelity* heterogeneity block, keyed by the comparison they name.

    The fourth block in this table to print rows beginning with these names, and the last,
    so the far bound is the table's closing legend.
    """
    header = "is the fidelity margin the same on every source?"
    assert header in printed, printed
    block = printed.split(header)[1].split("delta = left minus right")[0]
    rows = {}
    for line in block.splitlines():
        for _, _, question in _load("_dq_fq", SCRIPT).HEAD_TO_HEAD:
            if line.startswith(question):
                rows[question.strip()] = line
    return rows


def test_the_chi_square_tail_matches_its_textbook_values(table: Any) -> None:
    """Both branches of the incomplete gamma, against numbers that are not this code's.

    The series and the continued fraction are separate implementations of the same
    function and the input decides which one runs, so a table of critical values that
    only spanned one branch would leave the other free to return anything. df=1 at
    Q=3.841 takes the continued fraction; df=3 at Q=0.35 takes the series.

    Turns red when: either branch is transcribed wrong, or the transition test flips and
    sends inputs to the branch that does not converge there.
    """
    for statistic, df, want in [
        (3.841, 1, 0.05),
        (6.635, 1, 0.01),
        (5.991, 2, 0.05),
        (9.210, 2, 0.01),
        (7.815, 3, 0.05),
        (11.345, 3, 0.01),
        (0.35, 3, 0.9505),
        (0.10, 1, 0.7518),
        # The small-Q corner, which is the only region where the two branches disagree
        # -- the continued fraction returns 0.863 at Q=0.001 where the answer is 0.975 --
        # and it is exactly what two sources in close agreement produce. Independently
        # checkable without a table: at df=1 the tail is 2 * (1 - Phi(sqrt(Q))).
        (0.001, 1, 0.974773),
        (0.010, 1, 0.920344),
    ]:
        got = table.chi_square_sf(statistic, df)
        assert abs(got - want) < 1e-3, f"sf({statistic}, df={df}) = {got}, want {want}"
    assert table.chi_square_sf(0.0, 1) == 1.0
    assert table.chi_square_sf(-1.0, 1) == 1.0
    assert table.chi_square_sf(100.0, 1) < 1e-20
    with pytest.raises(ValueError, match="df must be at least 1"):
        table.chi_square_sf(1.0, 0)


def test_cochran_q_is_zero_on_agreement_and_pools_by_precision(table: Any) -> None:
    """Q measures spread around the pooled centre, and the centre is not the mean.

    Both halves matter and they fail differently. Equal-weighted pooling still gives Q=0
    when the estimates agree, so the first assertion alone would not notice it; the second
    pins the weighting by using estimates whose precisions differ by 4x, where the mean
    (+2.00) and the inverse-variance pool (+1.40) are far apart.
    """
    pooled, q, p_value = table.cochran_q([1.0, 1.0, 1.0], [0.5, 0.25, 1.0])
    assert pooled == pytest.approx(1.0)
    assert q == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)

    pooled, q, p_value = table.cochran_q([1.0, 3.0], [0.5, 1.0])
    assert pooled == pytest.approx(1.4)  # not 2.0
    assert q == pytest.approx((1.0 - 1.4) ** 2 / 0.25 + (3.0 - 1.4) ** 2 / 1.0)
    assert 0.05 < p_value < 0.20


def test_cochran_q_refuses_the_inputs_that_would_decide_the_answer_alone(table: Any) -> None:
    """A zero standard error carries infinite weight, and would set the pooled estimate.

    Reachable, not hypothetical: a source on which two arms never disagree has a delta of
    exactly zero with no width. The block skips that row and says so; this pins that the
    function underneath refuses rather than returning a nan a caller might print.
    """
    with pytest.raises(ValueError, match="infinite weight"):
        table.cochran_q([1.0, 2.0], [0.5, 0.0])
    with pytest.raises(ValueError, match="at least two"):
        table.cochran_q([1.0], [0.5])
    with pytest.raises(ValueError, match="same length"):
        table.cochran_q([1.0, 2.0], [0.5])


def test_a_margin_that_lives_in_one_source_is_called_heterogeneous(
    table: Any, tmp_path: Path
) -> None:
    """The finding the aggregate cannot express, and the reason this block exists.

    On the real panel the 4-bit DynQuant-over-GPTQ margin is +1.03 on one dataset and
    -0.49 on the other, and the aggregate reports +0.64 with no indication that the number
    is an average of two different things. Here the same 14/4 is arranged to the extreme.
    Reading the two intervals side by side is not a substitute -- that is what the block
    above already offers and what this one exists because it cannot do.

    Turns red when: the deltas stop being tested against each other, or the block loses
    the Holm correction and starts promoting rows on the raw p.
    """
    out = _write_panel(tmp_path / "arms")
    _stack_the_flips(out)
    printed = _run(table, out)

    row = _heterogeneity_rows(printed)["4b  DynQuant vs GPTQ"]
    assert "HETEROGENEOUS" in row, row
    # The spread is printed in the header's source order, and it has to carry the signs:
    # a row that prints two magnitudes hides that the sources disagree about direction.
    assert "+6.83" in row and "-2.05" in row, row
    assert "consistent" not in row


def test_a_margin_that_holds_on_both_sources_is_called_consistent(
    table: Any, tmp_path: Path
) -> None:
    """The control, and it is the load-bearing half.

    Any test whose only case is a positive is satisfied by a block that always separates.
    Same arms, same 14/4, same subset sizes -- the flips split evenly instead of stacking,
    and the verdict has to flip with them.
    """
    out = _write_panel(tmp_path / "arms")
    _write_sources(out, lambda index: "gretel" if index % 2 else "wikisql")
    printed = _run(table, out)

    row = _heterogeneity_rows(printed)["4b  DynQuant vs GPTQ"]
    assert "consistent" in row, row
    assert "HETEROGENEOUS" not in row
    assert "No row separates" in printed
    # Consistent is a failure to reject, and the block has to say so rather than let a
    # reader take it for a demonstration that the margins are equal.
    assert "Consistent is not the same as equal" in printed


def test_the_heterogeneity_row_inherits_the_mark_it_cannot_re_derive(
    table: Any, tmp_path: Path
) -> None:
    """A flagged comparison stays flagged when it is re-partitioned and re-tested.

    The dispatch confound is a property of which two arms are being compared, so it
    survives every slicing of the items -- both halves of a flagged comparison carry it,
    and so does a statistic computed from the two halves. The mark is inherited from the
    entry rather than re-derived from the arithmetic map, and this pins the two to agree:
    a row marked in the block above and unmarked here would read as a spread that is
    cleaner than the deltas it came from.
    """
    out = _write_panel(tmp_path / "arms")
    _stack_the_flips(out)
    printed = _run(table, out)

    source_block = printed.split("head to head, on gretel alone")[1].split("head to head, on w")[0]
    for _, _, question in table.HEAD_TO_HEAD:
        per_source = next(
            (line for line in source_block.splitlines() if line.startswith(question)), None
        )
        if per_source is None or "needs both arms" in per_source:
            continue
        row = _heterogeneity_rows(printed).get(question.strip())
        assert row is not None, question
        assert row.rstrip().endswith("!") == per_source.rstrip().endswith("!"), (
            f"{question}: per-source {per_source!r} vs heterogeneity {row!r}"
        )


def test_the_printed_fidelity_spread_is_cochran_on_agreement_not_on_accuracy(
    table: Any, tmp_path: Path
) -> None:
    """The claim this block exists to support is a *disagreement* between the two spreads.

    On the real panel DynQuant's accuracy margin over GPTQ varies by source and its
    fidelity margin over AWQ does not, and reading those two rows together is what says
    the accuracy spread is the ceiling's arithmetic rather than the method's behaviour.
    Hand this block the accuracy per-source blocks and it prints the accuracy spread under
    a fidelity heading: three rows, every one of them a real Cochran statistic computed
    over real deltas, and the conclusion drawn from them is the opposite one.

    So this reads what the table printed and re-derives it from the derived records, and
    then checks the fixture can actually tell the two apart -- otherwise the loop above
    would pass against either set of blocks and this test would be pinning nothing.

    Turns red when: the fidelity source blocks are built from ``records``; the spread is
    computed from the accuracy blocks; or either block stops naming its own indicator, at
    which point the two headers collide and the parser cannot find the right one.
    """
    out = _write_panel(tmp_path / "arms")
    labels = _stack_the_flips(out)
    printed = _run(table, out)
    _, records = table.load_panel(out)
    arithmetic = dict.fromkeys(records, "grouped")

    def spread_over(source_records: dict[str, Any]) -> list[dict[str, Any]]:
        blocks = {
            name: table.print_comparisons(
                name, table.HEAD_TO_HEAD, table.restrict(source_records, labels, name), arithmetic
            )
            for name in sorted(set(labels))
        }
        return table.print_heterogeneity(blocks)

    with redirect_stdout(StringIO()):
        expected = spread_over(table.agreement_records(records))
        accuracy = spread_over(records)

    assert expected, "the fixture produced no fidelity spread rows to check"
    rows = _fidelity_heterogeneity_rows(printed)
    for entry in expected:
        row = rows[entry["question"].strip()]
        assert f"{entry['pooled_points']:+.2f}" in row, (entry["question"], row)
        assert f"{entry['q']:.2f}" in row, (entry["question"], row)

    by_question = {entry["question"]: entry["q"] for entry in accuracy}
    assert any(entry["q"] != by_question[entry["question"]] for entry in expected), (
        "every fidelity Q equalled its accuracy Q, so this fixture cannot discriminate"
    )


def test_the_json_carries_the_fidelity_spread_beside_the_accuracy_one(
    table: Any, tmp_path: Path
) -> None:
    """The card reads the payload, and one spread without the other inverts the reading.

    A consumer that gets ``source_heterogeneity`` alone sees a margin that varies by
    dataset and has no way to ask whether the method varied or the ceiling did. Both keys
    or neither.

    Turns red when: either key is dropped, or both are filled from the same spread.
    """
    out = _write_panel(tmp_path / "arms")
    _stack_the_flips(out)
    destination = tmp_path / "table.json"
    _run(table, out, "--json-out", str(destination))
    payload = json.loads(destination.read_text(encoding="utf-8"))

    fidelity = payload["fidelity_source_heterogeneity"]
    assert fidelity, "the table printed a fidelity spread and the payload carries none"
    for entry in fidelity:
        assert set(entry["sources"]) == {"gretel", "wikisql"}
        assert entry["df"] == 1

    # The blocks the spread was computed from, tied to the spread rather than merely
    # present. Filled from the accuracy blocks they would still carry both sources and six
    # plausible deltas -- the spread's own per-source values are the only thing in the
    # payload that says which records they came from.
    by_source = payload["fidelity_head_to_head_by_source"]
    assert set(by_source) == {"gretel", "wikisql"}
    for entry in fidelity:
        for source, delta in entry["sources"].items():
            row = next(r for r in by_source[source] if r["question"] == entry["question"])
            assert row["delta_points"] == delta, (entry["question"], source)

    accuracy = {entry["question"]: entry["q"] for entry in payload["source_heterogeneity"]}
    assert any(entry["q"] != accuracy[entry["question"]] for entry in fidelity), (
        "both spread keys carry the same statistics, so one of them was filled from the other"
    )


def _deltas(
    table: Any, records: Any, labels: list[str], name: str, arithmetic: Any
) -> dict[str, float]:
    """``{question: delta}`` for one subset, derived the way the block under test derives it."""
    with redirect_stdout(StringIO()):
        entries = table.print_comparisons(
            name, table.HEAD_TO_HEAD, table.restrict(records, labels, name), arithmetic
        )
    return {entry["question"].strip(): entry["paired"].delta_points for entry in entries}


def test_the_difficulty_blocks_split_on_the_ceiling_and_not_on_anything_else(
    table: Any, tmp_path: Path
) -> None:
    """The rows section 13 opens with, which until now came from a script in a scratch dir.

    Two ways for this block to be wrong while printing six believable rows in each half:
    split on something other than the ceiling's own hits, or restrict nothing and print the
    aggregate twice under two headings. Both are caught by deriving the labels from the
    ceiling's vector, deriving the deltas from those labels, and reading what the table
    printed -- and then by requiring the fixture to discriminate, because a panel whose two
    strata agree with each other and with the aggregate would pass against all three.

    Turns red when: the split stops being the ceiling's answer, the restriction is dropped,
    or the strata stop being printed under their own names.
    """
    out = _write_panel(tmp_path / "arms")
    ceiling = _split_the_flips(out)
    printed = _run(table, out)
    _, records = table.load_panel(out)
    arithmetic = dict.fromkeys(records, "grouped")

    difficulty = table.difficulty_labels(records)
    off_the_ceiling = [table.CEILING_RIGHT if hit else table.CEILING_WRONG for hit in ceiling]
    assert difficulty == off_the_ceiling, "the labels are not the ceiling's own answers"

    strata = {
        name: _deltas(table, records, difficulty, name, arithmetic)
        for name in (table.CEILING_RIGHT, table.CEILING_WRONG)
    }
    for name, expected in strata.items():
        rows = _difficulty_rows(printed, name)
        assert set(rows) == set(expected), (name, sorted(rows), sorted(expected))
        for question, delta in expected.items():
            assert f"{delta:+6.2f}" in rows[question], (name, question, rows[question])

    aggregate = _deltas(table, records, [""] * len(ceiling), "", arithmetic)
    right, wrong = strata[table.CEILING_RIGHT], strata[table.CEILING_WRONG]
    assert any(right[q] > 0 > wrong[q] or wrong[q] > 0 > right[q] for q in right), (
        "no comparison changes sign between the strata, so this fixture cannot tell a "
        "difficulty split from two copies of the aggregate"
    )
    assert any(right[q] != aggregate[q] for q in right), "a stratum reproduced the aggregate"


def test_the_ceiling_wrong_stratum_is_the_fidelity_margin_with_its_sign_flipped(
    table: Any, tmp_path: Path
) -> None:
    """The two rows are one finding, and the block's closing note says so. This is why.

    Inside ceiling-right a hit *is* agreement with the ceiling; inside ceiling-wrong a hit
    is disagreement with it. So the second stratum's accuracy margin is the same arms'
    fidelity margin negated -- exactly, down to the two discordant counts swapping and the
    exact McNemar p being the same number. A method that tracks the ceiling more closely has
    to win the first row and lose the second, which is what makes the sign change an
    identity rather than a second result the panel went out and found.

    Asserted because the note tells a reader to read the block that way and a note is not a
    mechanism: if the strata ever stopped being cut on the same arm the fidelity indicator
    is measured against, the note would go false while every number under it stayed real.

    Turns red when: the split stops being the ceiling's, or the fidelity indicator stops
    being agreement with the arm the strata are cut on.
    """
    out = _write_panel(tmp_path / "arms")
    _split_the_flips(out)
    _, records = table.load_panel(out)
    arithmetic = dict.fromkeys(records, "grouped")
    difficulty = table.difficulty_labels(records)

    with redirect_stdout(StringIO()):
        accuracy = table.print_comparisons(
            "acc",
            table.HEAD_TO_HEAD,
            table.restrict(records, difficulty, table.CEILING_WRONG),
            arithmetic,
        )
        fidelity = table.print_comparisons(
            "fid",
            table.HEAD_TO_HEAD,
            table.restrict(table.agreement_records(records), difficulty, table.CEILING_WRONG),
            arithmetic,
        )

    assert accuracy, "the fixture produced no comparisons in the ceiling-wrong stratum"
    mirrored = {entry["question"]: entry["paired"] for entry in fidelity}
    moved = 0
    for entry in accuracy:
        theirs, mine = mirrored[entry["question"]], entry["paired"]
        assert mine.delta_points == pytest.approx(-theirs.delta_points), entry["question"]
        assert mine.p_value == pytest.approx(theirs.p_value), entry["question"]
        assert (mine.a_only, mine.b_only) == (theirs.b_only, theirs.a_only), entry["question"]
        moved += mine.delta_points != 0
    assert moved, "every delta was zero, so the negation asserted nothing"


def test_stratifying_a_family_that_compares_the_ceiling_is_refused(table: Any) -> None:
    """`AGAINST_CEILING` is a family in the same file, one keyword argument away.

    Split the ceiling-vs-arm comparisons on the ceiling's own hits and the halves separate
    by construction: ceiling-right is the set of items the ceiling got right, so it scores
    100% there and no arm can match it. The rows would be large, significant, and entirely
    an artifact of the split, and nothing about them would look wrong.

    Turns red when: the guard is dropped, or narrowed to the default family it was written
    against rather than the family it is handed.
    """
    records = {"bf16": {"hits": [True, False]}, "dq_4b": {"hits": [True, True]}}
    with pytest.raises(ValueError, match="conditions on the outcome"):
        table.difficulty_labels(records, family=table.AGAINST_CEILING)
    assert table.difficulty_labels(records) == [table.CEILING_RIGHT, table.CEILING_WRONG]


def test_the_cross_is_printed_only_when_it_is_two_axes(table: Any, tmp_path: Path) -> None:
    """Crossing an axis with a constant reprints the other axis under a heading naming two.

    On a single-source panel the four cells collapse to the two difficulty strata already
    printed above, and a reader has no way to see that the second block is the first one
    again -- every number in it agrees, which reads as corroboration.

    Turns red when: the cross stops being suppressed on one axis, stops being printed on
    two, or is printed over the wrong labels.
    """
    assert table.crossed(["a", "a"], ["x", "y"]) is None
    assert table.crossed(["a", "b"], ["x", "x"]) is None
    assert table.crossed(["a", "b"], ["x", "y"]) == ["a/x", "b/y"]
    assert table.crossed(None, ["x", "y"]) is None
    assert table.crossed(["a", "b"], ["x", "y", "z"]) is None

    out = _write_panel(tmp_path / "arms")
    _split_the_flips(out)
    labels = _write_sources(out, lambda index: "gretel" if index % 3 else "wikisql")
    printed = _run(table, out)
    _, records = table.load_panel(out)
    arithmetic = dict.fromkeys(records, "grouped")

    both = table.crossed(labels, table.difficulty_labels(records))
    assert both is not None
    for cell in sorted(set(both)):
        title = f"by source and difficulty, on {cell} alone"
        assert title in printed, cell
        block = printed.split(title)[1].split(" alone (")[0]
        expected = _deltas(table, records, both, cell, arithmetic)
        assert expected, cell
        for question, delta in expected.items():
            row = next(line for line in block.splitlines() if line.startswith(question))
            assert f"{delta:+6.2f}" in row, (cell, question, row)

    lonely = _write_panel(tmp_path / "lonely")
    _split_the_flips(lonely)
    _write_sources(lonely, lambda index: "wikisql")
    alone = _run(table, lonely)
    assert "by difficulty, on ceiling-right alone" in alone, "the axis that needs no sources"
    assert "by source and difficulty" not in alone


def test_a_partition_wider_than_two_still_lines_its_columns_up(table: Any, tmp_path: Path) -> None:
    """Four cells print a spread string of 26 characters into a field sized for two.

    Nothing about the statistics changes -- pooled, Q and both p stay exactly right -- and
    the row simply pushes every column to its right out of position. This is the one block
    in the table a reader is meant to scan down rather than across, comparing four cells of
    the same comparison against each other, so a misaligned Q column is the failure that
    matters most and is the one no assertion on a value would catch.

    Turns red when: the spread field goes back to a constant width, or a wider partition is
    added without the field following it.
    """
    out = _write_panel(tmp_path / "arms")
    _split_the_flips(out)
    _write_sources(out, lambda index: "gretel" if index % 3 else "wikisql")
    destination = tmp_path / "table.json"
    printed = _run(table, out, "--json-out", str(destination))
    payload = json.loads(destination.read_text(encoding="utf-8"))

    # Two ways for this block to come out ragged and only one of them is the new code. The
    # rows can disagree with each other, which is what a fixed-width spread field does to a
    # four-cell partition; or they can agree with each other and not with their own heading,
    # which is what this table did in both of its blocks until the field widths were made to
    # match. Both are checked, because fixing either one alone still leaves the column a
    # reader is scanning under the wrong name.
    widths = {
        len(", ".join(f"{delta:+.2f}" for _, delta in sorted(entry["sources"].items())))
        for entry in payload["source_and_difficulty_heterogeneity"]
    }
    assert len(widths) > 1, (
        f"every cell rendered {widths} characters of spread, so this block would line up "
        f"with or without a partition-sized column"
    )

    # Bounded at the next block, as everything in this file that reads a block has to be.
    # The fidelity spread further down prints HETEROGENEOUS rows too, over a two-subset
    # spread of its own width, and an unbounded read collects those as disagreement.
    block = printed.split("is the margin the same in every source-difficulty cell?")[1]
    block = block.split("what each method cost")[0]
    questions = [question for _, _, question in _load("_dq_wq", SCRIPT).HEAD_TO_HEAD]
    columns = set()
    for line in block.splitlines():
        if not any(line.startswith(question) for question in questions):
            continue
        verdict = "HETEROGENEOUS" if "HETEROGENEOUS" in line else "consistent"
        if verdict in line:
            columns.add(line.index(verdict))
    header = next(line for line in block.splitlines() if line.startswith("comparison"))
    assert columns == {header.index("verdict")}, (columns, header, block)


def test_a_spread_row_carries_the_family_its_p_was_corrected_in(table: Any, tmp_path: Path) -> None:
    """The block warns once about a short family; a serialised row travels alone.

    Holm's multiplier is the number of comparisons actually corrected, so the same row over
    three comparisons and over six is two different claims -- on this panel's five-arm run
    that moved the one heterogeneous row from 0.0359 to 0.0717, which is the verdict rather
    than a decimal place. A consumer holding the row has no other way to see it.

    Turns red when: either field is dropped, or the corrected count stops counting the rows
    Holm actually ran over -- which is what it is, not the family that was declared, because
    a comparison whose subset produced no flips at all is named and skipped.
    """
    out = _write_panel(tmp_path / "arms", omit=("dq_3b", "awq_3b"))
    _stack_the_flips(out)
    _write_sources(out, lambda index: "gretel" if index % 3 else "wikisql")
    destination = tmp_path / "table.json"
    _run(table, out, "--json-out", str(destination))
    payload = json.loads(destination.read_text(encoding="utf-8"))

    spread = payload["source_heterogeneity"]
    assert spread, "the table printed a spread and the payload carries none"
    declared = len(table.HEAD_TO_HEAD)
    for entry in spread:
        assert entry["holm_corrected"] == len(spread), entry["question"]
        assert entry["holm_family"] == declared, entry["question"]
    assert len(spread) < declared, (
        f"this fixture corrects all {declared} comparisons, so it cannot tell a count of the "
        f"rows Holm ran over from a count of the family that was declared"
    )

    whole = _write_panel(tmp_path / "full")
    _stack_the_flips(whole)
    _write_sources(whole, lambda index: "gretel" if index % 3 else "wikisql")
    finished = tmp_path / "full.json"
    _run(table, whole, "--json-out", str(finished))
    rows = json.loads(finished.read_text(encoding="utf-8"))["source_heterogeneity"]
    assert {entry["holm_corrected"] for entry in rows} == {declared}


def test_the_json_carries_the_difficulty_blocks_and_their_spread(
    table: Any, tmp_path: Path
) -> None:
    """The model cards read this payload, and section 13 opens with these six rows.

    A consumer holding ``source_heterogeneity`` alone has the question -- the margin is not
    one number -- and nothing to answer it with, because on this panel the source axis and
    the difficulty axis are confounded and only one of them is carried. Both, or the
    heterogeneous row means whichever of the two the reader already believed.

    Turns red when: any of the four keys is dropped, filled from the source blocks, or
    filled with deltas other than the ones the table printed.
    """
    out = _write_panel(tmp_path / "arms")
    _split_the_flips(out)
    _write_sources(out, lambda index: "gretel" if index % 3 else "wikisql")
    destination = tmp_path / "table.json"
    printed = _run(table, out, "--json-out", str(destination))
    payload = json.loads(destination.read_text(encoding="utf-8"))

    by_difficulty = payload["head_to_head_by_difficulty"]
    assert set(by_difficulty) == {table.CEILING_RIGHT, table.CEILING_WRONG}
    for name, entries in by_difficulty.items():
        rows = _difficulty_rows(printed, name)
        assert entries, name
        for entry in entries:
            assert f"{entry['delta_points']:+6.2f}" in rows[entry["question"].strip()]

    spread = payload["difficulty_heterogeneity"]
    assert spread, "the table printed a difficulty spread and the payload carries none"
    for entry in spread:
        assert set(entry["sources"]) == {table.CEILING_RIGHT, table.CEILING_WRONG}
        assert entry["df"] == 1
        for name, delta in entry["sources"].items():
            row = next(r for r in by_difficulty[name] if r["question"] == entry["question"])
            assert row["delta_points"] == delta, (entry["question"], name)

    cells = payload["head_to_head_by_source_and_difficulty"]
    assert set(cells) == {
        f"{source}/{stratum}"
        for source in ("gretel", "wikisql")
        for stratum in (table.CEILING_RIGHT, table.CEILING_WRONG)
    }
    assert all(cells.values()), "a crossed cell carried no comparisons"
    assert all(entry["df"] == 3 for entry in payload["source_and_difficulty_heterogeneity"])

    source_side = {entry["question"]: entry["q"] for entry in payload["source_heterogeneity"]}
    shared = [entry for entry in spread if entry["question"] in source_side]
    assert shared, "no comparison reached both spreads, so this fixture cannot discriminate"
    assert any(entry["q"] != source_side[entry["question"]] for entry in shared), (
        "the difficulty spread equals the source spread, so one was filled from the other"
    )


def test_one_source_is_not_a_mixture_and_prints_no_block(table: Any, tmp_path: Path) -> None:
    """Nothing to compare against, so no header with an empty table under it.

    Cochran's Q needs two estimates. A panel evaluated on a single dataset is the ordinary
    case for every other model in this campaign, and it must not print a heading promising
    an answer it cannot give.
    """
    out = _write_panel(tmp_path / "arms")
    _write_sources(out, lambda index: "wikisql")
    printed = _run(table, out)
    assert "head to head, on wikisql alone" in printed
    assert "agreement with bf16, on wikisql alone" in printed
    assert "is the margin the same on every source?" not in printed
    assert "is the fidelity margin the same on every source?" not in printed
    assert "is the margin the same in every source-difficulty cell?" not in printed


def test_the_json_carries_the_spread_the_table_printed(table: Any, tmp_path: Path) -> None:
    """A consumer reading only the aggregate cannot tell a broad win from a narrow one.

    The model card is generated from this payload, so a heterogeneous margin that appears
    in the terminal and not in the file is a margin the card will state as one number.
    Every field the row prints is asserted present, because a serialisation that carries
    the verdict and drops the spread is the one that reads as complete.
    """
    out = _write_panel(tmp_path / "arms")
    _stack_the_flips(out)
    destination = tmp_path / "table.json"
    printed = _run(table, out, "--json-out", str(destination))

    payload = json.loads(destination.read_text(encoding="utf-8"))
    spread = {entry["question"].strip(): entry for entry in payload["source_heterogeneity"]}
    entry = spread["4b  DynQuant vs GPTQ"]
    assert entry["heterogeneous"] is True
    assert entry["df"] == 1
    assert set(entry["sources"]) == {"gretel", "wikisql"}
    assert entry["sources"]["gretel"] > 0 > entry["sources"]["wikisql"]
    assert entry["p_adjusted"] >= entry["p_value"]
    assert entry["same_arithmetic"] == (
        not _heterogeneity_rows(printed)["4b  DynQuant vs GPTQ"].rstrip().endswith("!")
    )


def _stack_every_flip_into_one_source(out: Path) -> list[str]:
    """All 18 dq-vs-gptq discordant pairs into gretel, leaving wikisql with none.

    The degenerate input the block has to survive rather than divide by: a source on
    which two arms never disagree has a delta of exactly zero and a standard error of
    exactly zero, so its inverse-variance weight is infinite and it would decide the
    pooled estimate on its own.
    """
    dq = json.loads((out / "dq_4b.json").read_text(encoding="utf-8"))["hits"]
    gptq = json.loads((out / "gptq_4b.json").read_text(encoding="utf-8"))["hits"]

    def chooser(index: int) -> str:
        if dq[index] != gptq[index]:
            return "gretel"
        return "gretel" if index % 2 else "wikisql"

    return _write_sources(out, chooser)


class _StandInPaired:
    """Just the two fields the heterogeneity block reads off a comparison.

    A real :class:`PairedComparison` is built from hit vectors, and hit vectors that
    produce three chosen p-values to two significant figures are a puzzle rather than a
    fixture. The block reads a delta and a standard error; this supplies them directly.
    """

    def __init__(self, delta: float, error: float) -> None:
        self.delta_points = delta
        self.standard_error_points = error


def _blocks(cases: dict[str, list[tuple[float, float]]]) -> dict[str, list[dict[str, Any]]]:
    """``{question: [(delta, se) per source]}`` in the shape print_source_blocks returns."""
    sources = ["gretel", "wikisql"]
    return {
        source: [
            {
                "left": f"left_{index}",
                "right": f"right_{index}",
                "question": question,
                "paired": _StandInPaired(*pairs[position]),
                "same_arithmetic": True,
            }
            for index, (question, pairs) in enumerate(cases.items())
        ]
        for position, source in enumerate(sources)
    }


def test_a_row_that_separates_only_before_correction_does_not_separate(
    table: Any, capsys: Any
) -> None:
    """The verdict follows the Holm-adjusted p, and one row exists to prove it.

    Asking this question of every comparison in a panel makes a family, and the block says
    so. The claim is untestable on the real fixture -- its spreads are far from the
    boundary either way -- so the deltas are supplied directly and chosen to straddle it:
    three comparisons, the middle one significant raw and not after being multiplied by
    two.

    Turns red when: the block drops the correction, or applies it and then reads the raw p
    when deciding the verdict.
    """
    computed = table.print_heterogeneity(
        _blocks(
            {
                "wide apart": [(0.0, 0.5), (2.4, 0.5)],
                "just inside": [(0.0, 0.5), (1.55, 0.5)],
                "together": [(0.0, 0.5), (0.6, 0.5)],
            }
        )
    )
    printed = capsys.readouterr().out

    straddling = [entry for entry in computed if entry["p_value"] < 0.05 <= entry["p_adjusted"]]
    assert straddling, [(e["question"], e["p_value"], e["p_adjusted"]) for e in computed]
    for entry in straddling:
        assert entry["heterogeneous"] is False
        row = next(line for line in printed.splitlines() if line.startswith(entry["question"]))
        assert "consistent" in row and "HETEROGENEOUS" not in row, row
    # And the block still finds the one that clears correction, or the test above would be
    # satisfied by a version that never separates anything.
    assert any(entry["heterogeneous"] for entry in computed)


def test_a_source_with_no_flips_is_skipped_by_name_and_not_weighted(
    table: Any, tmp_path: Path
) -> None:
    """Zero flips is zero width, and an infinite weight would decide the pooled estimate.

    Reachable on a real panel, not a constructed edge: two arms that differ on a handful
    of items can easily agree completely on the smaller dataset. The row is dropped and
    said out loud rather than printed as a spread of one number, and it must not reach the
    json either -- a consumer cannot tell a dropped comparison from one that was never run
    if the only difference is a line of terminal output.
    """
    out = _write_panel(tmp_path / "arms")
    _stack_every_flip_into_one_source(out)
    destination = tmp_path / "table.json"
    printed = _run(table, out, "--json-out", str(destination))

    block = printed.split("is the margin the same on every source?")[1]
    skipped = [line for line in block.splitlines() if "no flips at all" in line]
    assert any(line.startswith("4b  DynQuant vs GPTQ") for line in skipped), block

    payload = json.loads(destination.read_text(encoding="utf-8"))
    questions = {entry["question"].strip() for entry in payload["source_heterogeneity"]}
    assert "4b  DynQuant vs GPTQ" not in questions


def test_a_heterogeneity_verdict_on_a_half_run_panel_says_it_is_provisional(
    table: Any, tmp_path: Path
) -> None:
    """Holm's multiplier is the family that ran, so a mid-run verdict can only get worse.

    Not a general caution: on the panel this was written against, the single heterogeneous
    row sits at 0.0359 over three comparisons and 0.0717 over six. Finishing the run flips
    the word. The block above carries this warning and this one did not, which left the
    reader most likely to be misled -- someone glancing at a running panel -- with a
    verdict and no way to see that it was about how much of the panel had run.

    Turns red when: the warning goes away, or starts appearing on a complete family, where
    it would train a reader to ignore it.
    """
    out = _write_panel(tmp_path / "arms", omit=("dq_3b", "awq_3b"))
    _stack_the_flips(out)
    printed = _run(table, out)
    short = printed.split("is the margin the same on every source?")[1]
    assert "Holm-adjusted over 3 of 6 comparisons -- a short family" in short, short
    assert "may read consistent then" in short

    whole = _write_panel(tmp_path / "full")
    _stack_the_flips(whole)
    complete = _run(table, whole).split("is the margin the same on every source?")[1]
    assert "short family" not in complete, complete


def test_the_table_runs_with_nothing_already_on_the_path(tmp_path: Path) -> None:
    """A subprocess, because in-process this test cannot fail.

    Every other test here calls `main` after conftest has put the core package source on
    `sys.path`, and the shell I run the suite from exports PYTHONPATH as well. The script
    imports core inside three functions, so a missing bootstrap does not break collection
    or `--help` -- it waits for a reader who runs the table on real records, on the box,
    where nothing installs `dynquant` and no conftest is involved.

    Run from `tmp_path` rather than the repo root so that a bootstrap resolved against the
    working directory instead of the file would fail here too.

    Turns red when: the `CORE_SRC` insert is dropped, or rewritten relative to cwd.
    """
    out = _write_panel(tmp_path / "arms")
    stripped = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--arms", str(out)],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=stripped,
    )
    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr, result.stderr
    # And it did the work, rather than exiting cleanly with an empty table.
    for label in ("bf16", "gptq_4b", "dq_4b"):
        assert label in result.stdout, result.stdout
    assert "head to head" in result.stdout, result.stdout


def test_the_fidelity_columns_reconstruct_the_accuracy_column(table: Any, tmp_path: Path) -> None:
    """A hit either matches the ceiling or flips it, so this is an identity, not a model.

    accuracy = c * (where right) + (1 - c) * (1 - (where wrong))

    Asserted exactly rather than approximately because there is no estimation anywhere in
    it: the three quantities are counts over one partition of one problem set. Any slip in
    which partition a column is counted over, or in which direction it is counted, breaks
    it -- and every one of those slips still prints four plausible percentages.

    Turns red when: the strata are swapped; either column is computed as P(arm right) in
    its stratum rather than P(arm agrees); the partition is taken from an arm's hits rather
    than the ceiling's; or ``ceiling_accuracy`` is read off the wrong arm.
    """
    out = _write_panel(tmp_path / "arms")
    manifest, records = table.load_panel(out)
    built = table.rows(out, manifest, records, PARAMS)
    with redirect_stdout(StringIO()):
        entries = table.print_fidelity(built, records)

    assert entries, "the panel has six quantized arms and none were measured"
    for entry in entries:
        c = entry["ceiling_accuracy"]
        reconstructed = c * entry["agreement_where_ceiling_right"] + (1.0 - c) * (
            1.0 - entry["agreement_where_ceiling_wrong"]
        )
        assert reconstructed == pytest.approx(entry["accuracy"], abs=1e-12), entry["label"]


def test_an_arm_that_never_flips_the_ceiling_scores_the_ceiling(table: Any, tmp_path: Path) -> None:
    """The identity's fixed point, which pins the sense of both columns at once.

    An arm that reproduces bf16 item for item must land on bf16's accuracy -- not above it,
    which is what reading "where wrong" as P(arm right | ceiling wrong) would produce, and
    not at 100%, which is what dropping the ``1 -`` would produce.
    """
    out = _write_panel(tmp_path / "arms")
    manifest, records = table.load_panel(out)
    # Every scored field together, not just the hits. `print_fidelity` reads accuracy
    # off the record and the two fidelity columns off the hits, so a record carrying one
    # arm's vector under another's accuracy is a state no run produces and the identity
    # is not claimed to hold for it.
    for field in ("hits", "accuracy", "correct"):
        records["dq_4b"][field] = records["bf16"][field]
    built = table.rows(out, manifest, records, PARAMS)
    with redirect_stdout(StringIO()):
        entries = table.print_fidelity(built, records)

    clone = next(entry for entry in entries if entry["label"] == "dq_4b")
    assert clone["agreement"] == 1.0
    assert clone["agreement_where_ceiling_right"] == 1.0
    assert clone["agreement_where_ceiling_wrong"] == 1.0
    assert clone["accuracy"] == pytest.approx(clone["ceiling_accuracy"])


def test_the_ceiling_is_not_given_a_row_agreeing_with_itself(table: Any, tmp_path: Path) -> None:
    """100.00% is the only number bf16 could print here and it is not a measurement.

    Turns red when: the ceiling stops being skipped, in the table or in the derived records
    -- which would also put a self-comparison into any family that included it.
    """
    out = _write_panel(tmp_path / "arms")
    _, records = table.load_panel(out)
    assert table.CEILING not in table.agreement_records(records)


def test_agreement_records_change_the_indicator_and_nothing_else(
    table: Any, tmp_path: Path
) -> None:
    """The derived records have to pair, or the fidelity family silently prints nothing.

    Every field except ``hits`` is what decides comparability -- the task, the split, the
    limit, the decode budget. Carrying them through is not tidiness: `print_comparisons`
    refuses any pair whose comparability fields differ, so a derivation that rebuilt the
    record from scratch would produce a block of "(not comparable)" rows rather than a
    wrong number, and only on a panel with real records.

    Turns red when: the derived record is built fresh instead of copied, or the indicator
    stops being elementwise equality with the ceiling.
    """
    out = _write_panel(tmp_path / "arms")
    _, records = table.load_panel(out)
    derived = table.agreement_records(records)

    truth = records["bf16"]["hits"]
    for label, record in derived.items():
        assert record["hits"] == [
            bool(hit) == bool(fact) for hit, fact in zip(records[label]["hits"], truth, strict=True)
        ], label
        for field, value in records[label].items():
            if field != "hits":
                assert record[field] == value, (label, field)


def _fidelity_rows(printed: str) -> dict[str, str]:
    """Rows of the fidelity family, keyed by the comparison they name.

    Bounded at both ends, like `_heterogeneity_rows` and for the same reason: this is the
    third block in the table to print rows beginning with these names.
    """
    assert "on agreement with" in printed, printed
    block = printed.split("on agreement with")[1].split("delta = left minus right")[0]
    # And stop at the first per-source fidelity block, which prints the same six
    # comparison names one partition down. Without this the rows returned are whichever
    # source printed last, on a panel that has sources -- and every one of them is a real
    # number, so nothing downstream would notice.
    block = block.split(" alone (")[0]
    rows = {}
    for line in block.splitlines():
        for _, _, question in _load("_dq_ht", SCRIPT).HEAD_TO_HEAD:
            if line.startswith(question):
                rows[question.strip()] = line
    return rows


def test_the_printed_fidelity_family_is_computed_on_agreement_not_on_accuracy(
    table: Any, tmp_path: Path
) -> None:
    """The wiring, not the function. Both halves had a test and the line joining them did not.

    Handing that call ``records`` rather than the derived ones prints the *accuracy* family
    under a fidelity heading: six rows, every one of them a real number, and nothing that
    reads only the functions can tell. So this reads what the table printed.

    The reason the two families can be told apart at all is that they disagree, and on the
    real 4-bit arms they disagree in the direction that matters -- DynQuant leads GPTQ by
    +0.64 on accuracy and +1.34 on agreement with bf16, and it is the second number that
    explains why the first changes sign with difficulty.

    Turns red when: the fidelity family is handed ``records``, or the block stops being
    printed, or its rows stop carrying the deltas the derived records produce.
    """
    out = _write_panel(tmp_path / "arms")
    printed = _run(table, out)
    _, records = table.load_panel(out)
    arithmetic = dict.fromkeys(records, "grouped")
    with redirect_stdout(StringIO()):
        fidelity = table.print_comparisons(
            "fid", table.HEAD_TO_HEAD, table.agreement_records(records), arithmetic
        )
        accuracy = table.print_comparisons("acc", table.HEAD_TO_HEAD, records, arithmetic)

    rows = _fidelity_rows(printed)
    assert len(rows) == len(table.HEAD_TO_HEAD), rows
    for entry in fidelity:
        assert f"{entry['paired'].delta_points:+.2f}" in rows[entry["question"].strip()]

    # And the fixture can actually tell them apart -- otherwise the loop above would pass
    # against either set of records and this test would be pinning nothing.
    by_question = {entry["question"]: entry["paired"].delta_points for entry in accuracy}
    assert any(
        entry["paired"].delta_points != by_question[entry["question"]] for entry in fidelity
    ), "every fidelity delta equalled its accuracy delta, so the fixture cannot discriminate"


def test_a_panel_with_no_ceiling_says_so_rather_than_dividing_by_it(
    table: Any, tmp_path: Path
) -> None:
    """Half the panel's blocks survive a missing bf16 arm; this one cannot, and the run it
    would abort is a seven-hour one. Named and skipped, not raised.
    """
    out = _write_panel(tmp_path / "arms", omit=("bf16",))
    manifest, records = table.load_panel(out)
    built = table.rows(out, manifest, records, PARAMS)
    buffer = StringIO()
    with redirect_stdout(buffer):
        assert table.print_fidelity(built, records) == []
    assert "nothing to agree with" in buffer.getvalue()
    assert table.agreement_records(records) == {}


def _decomposition_block(table: Any, printed: str) -> str:
    """The block, from its title through the last line of its own closing note.

    Bounded by the note's own text rather than by the next blank line: the arithmetic
    marker prints a blank line and four more of its own between the rows and the Holm
    count, so a slice that stopped at the first gap would cut the block in half and a
    test asserting on the family size would pass by not reaching it.
    """
    start = printed.index("the decomposition:")
    closing = table.DECOMPOSITION_NOTE[-1]
    return printed[start : printed.index(closing, start) + len(closing)]


def test_a_panel_with_no_control_prints_no_decomposition_block(table: Any, tmp_path: Path) -> None:
    """The seven-arm panel does not ask this question and must not appear to answer it.

    Six placeholder rows reading "(needs both arms)" would be a block about a control that
    was never planned, in a table whose every other missing row means an arm that was
    planned and failed. And the banked tables are committed: a block that appears
    unconditionally makes every one of them differ from a re-run for a reason that is not
    a measurement.

    Turns red when: the family stops being built from what the panel actually holds.
    """
    printed = _run(table, _write_panel(tmp_path / "panel"))

    assert "the decomposition" not in printed
    assert "dq-null" not in printed


def test_the_decomposition_partitions_the_margin_it_decomposes(table: Any, tmp_path: Path) -> None:
    """The two rows have to add up, or the block is two comparisons rather than a split.

    An accuracy difference is a difference of two means over the same items, so
    (dq - control) + (control - gptq) is (dq - gptq) identically -- this asserts the
    printer put the right two arms on the right two rows in the right two orders, which is
    the only way the identity can be broken. Asserted to a hundredth because that is the
    width the table prints, and a reader adding the printed rows should get the printed
    total.

    Turns red when: a row is reversed, or points at the wrong reference arm.
    """
    out = _write_panel(tmp_path / "panel", nulls=("shuffle", "uniform"))
    printed = _run(table, out, "--json-out", str(tmp_path / "panel.json"))
    payload = json.loads((tmp_path / "panel.json").read_text(encoding="utf-8"))

    margin = next(
        row["delta_points"]
        for row in payload["head_to_head"]
        if (row["left"], row["right"]) == ("dq_3b", "gptq_3b")
    )
    pairs = [(row["left"], row["right"]) for row in payload["decomposition"]]
    # One chain, not one fan per control: the right-hand arm of each row is the left-hand
    # arm of the next, so the rows visit each arm once and the column adds to the margin
    # once. Two independent splits would also each sum to the margin -- and the block would
    # sum to twice it, which is the arithmetic this asserts against.
    assert pairs == [
        ("dq_3b", "dq_3b_shuf"),
        ("dq_3b_shuf", "dq_3b_unif"),
        ("dq_3b_unif", "gptq_3b"),
    ]
    total = sum(row["delta_points"] for row in payload["decomposition"])
    assert round(total, 2) == round(margin, 2)
    assert all(row["delta_points"] > 0 for row in payload["decomposition"]), (
        "the fixture degrades each control further than the one before it"
    )

    block = _decomposition_block(table, printed)
    assert "signal: dq vs shuffle" in block
    assert "signal: shuffle vs uniform" in block
    assert "shape: uniform vs GPTQ" in block


def test_the_decomposition_is_corrected_in_its_own_family(table: Any, tmp_path: Path) -> None:
    """It answers a different question from the head-to-head, so it is a different family.

    Holm's multiplier is the size of the family a comparison is corrected in. Folding the
    decomposition rows into the six head-to-head rows would inflate every published margin's
    adjusted p by a factor of 9/6 for having asked a further question about one of them --
    and the further question is not about whether the method won, which is what that block
    is corrected for.

    Turns red when: the block is merged into another family, in either direction.
    """
    out = _write_panel(tmp_path / "panel", nulls=("shuffle", "uniform"))
    printed = _run(table, out)

    assert "Holm-adjusted over 3 of 3 comparisons" in _decomposition_block(table, printed)
    assert "Holm-adjusted over 6 of 6 comparisons" in _aggregate_block(printed)


def test_a_control_is_read_off_its_provenance_and_not_off_its_name(
    table: Any, tmp_path: Path
) -> None:
    """A label is a filename someone typed; the manifest is what the driver wrote.

    This file already refuses to trust labels for the source vectors, for the same reason.
    An arm named like a control that carries no `score_null` block did not have its signal
    nulled -- it is a DynQuant arm with an unusual name -- and putting it in the
    decomposition would report a control that was never run, at whatever accuracy that arm
    happens to have.

    Turns red when: the family is discovered by parsing labels.
    """
    out = _write_panel(tmp_path / "panel", nulls=("shuffle",))
    manifest = json.loads((out / "arms.json").read_text(encoding="utf-8"))
    for arm in manifest["arms"]:
        arm.pop("score_null", None)
    (out / "arms.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    printed = _run(table, out)

    assert "dq_3b_shuf" in printed, "the arm is still a row"
    assert "the decomposition" not in printed
    assert "dq-null" not in printed


def test_a_control_does_not_print_in_the_same_method_column_as_a_real_arm(
    table: Any, tmp_path: Path
) -> None:
    """The size table is where someone scanning the panel decides what each row is.

    A control runs the DynQuant path end to end, so its manifest `kind` is `dq` and has to
    stay `dq` for the driver to dispatch it -- which in the table would put a signal-free
    arm in the same column as the arm it is the control for, at the same bytes, one row
    apart. The one confusion the whole block exists to prevent.

    Turns red when: the method column goes back to printing `kind` unconditionally.
    """
    printed = _run(table, _write_panel(tmp_path / "panel", nulls=("uniform",)))

    # The size table alone. Every later block is also keyed by arm label in column one,
    # and a parse over the whole output reads whichever one came last -- which is how the
    # first version of this test compared an accuracy against the string "dq".
    body = printed.split("arm        method", 1)[1].split("\n\n", 1)[0]
    rows = {line.split()[0]: line.split()[1] for line in body.splitlines() if line[:1].isalnum()}
    assert rows["dq_3b"] == "dq"
    assert rows["dq_3b_unif"] == "dq-null"
    assert rows["gptq_3b"] == "gptq" and rows["bf16"] == "ceiling"


def test_one_control_ladders_to_the_two_rows_it_always_had(table: Any, tmp_path: Path) -> None:
    """The chain generalises the two-row block; it must not have replaced it.

    A ladder over one control is `dq -> control -> gptq`, which is the same two rows and the
    same two labels the block printed before it was a chain -- and every banked panel with a
    single control was read that way. If this ever printed something else, the change would
    have been a rewrite of a published block dressed as a generalisation.

    Turns red when: the single-control case stops being the block it was.
    """
    out = _write_panel(tmp_path / "panel", nulls=("shuffle",))
    printed = _run(table, out, "--json-out", str(tmp_path / "panel.json"))
    payload = json.loads((tmp_path / "panel.json").read_text(encoding="utf-8"))

    assert [(row["left"], row["right"]) for row in payload["decomposition"]] == [
        ("dq_3b", "dq_3b_shuf"),
        ("dq_3b_shuf", "gptq_3b"),
    ]
    block = _decomposition_block(table, printed)
    assert "signal: dq vs shuffle" in block
    assert "shape: shuffle vs GPTQ" in block
    assert "Holm-adjusted over 2 of 2 comparisons" in block


def test_the_ladder_is_ordered_by_the_package_and_not_by_the_manifest(
    table: Any, tmp_path: Path
) -> None:
    """Which control removes more is a property of the mode, and one file owns it.

    `NULL_MODES` is declared in increasing order of how much each mode removes, so the rung
    between two controls is only interpretable if the chain follows it. Manifest order is
    the order someone typed on a command line -- here, deliberately the reverse -- and a
    ladder that followed it would report the rung as what `shuffle` bought over `uniform`
    while printing it as the other way round. The sum would still be the margin, which is
    exactly why this needs asserting rather than eyeballing.

    Turns red when: this file starts keeping its own opinion of which null is stronger.
    """
    from dynquant.score.null import NULL_MODES

    assert NULL_MODES.index("shuffle") < NULL_MODES.index("uniform")

    out = _write_panel(tmp_path / "panel", nulls=("uniform", "shuffle"))
    _run(table, out, "--json-out", str(tmp_path / "panel.json"))
    payload = json.loads((tmp_path / "panel.json").read_text(encoding="utf-8"))

    assert [(row["left"], row["right"]) for row in payload["decomposition"]] == [
        ("dq_3b", "dq_3b_shuf"),
        ("dq_3b_shuf", "dq_3b_unif"),
        ("dq_3b_unif", "gptq_3b"),
    ]


def _redrawn_block(table: Any, printed: str) -> str:
    """The redrawn block, sliced the way :func:`_decomposition_block` slices its own."""
    start = printed.index("the same rung, redrawn:")
    closing = table.REPLICATE_NOTE[-1]
    return printed[start : printed.index(closing, start) + len(closing)]


def test_a_further_draw_of_a_rung_does_not_become_a_further_rung(
    table: Any, tmp_path: Path
) -> None:
    """Redrawing a control measures the rung again; it does not add a step to the ladder.

    The ladder partitions the margin because each rung runs from the arm the one before it
    ran to. Chaining four shuffle draws would still sum to the margin -- any chain of the
    same endpoints does -- while printing three rows that read "signal: shuffle vs shuffle",
    each worth the difference between two permutations of the same nulled score, and
    splitting the rung shuffle actually buys across the four of them. The block would then
    say the signal was worth a quarter of what it is worth, in rows that all look valid.

    Turns red when: the chain is built by pairing every control arm in rank order.
    """
    out = _write_panel(
        tmp_path / "panel", nulls=("shuffle", "uniform"), draws=(("shuffle", 1), ("shuffle", 2))
    )
    printed = _run(table, out, "--json-out", str(tmp_path / "panel.json"))
    payload = json.loads((tmp_path / "panel.json").read_text(encoding="utf-8"))

    margin = next(
        row["delta_points"]
        for row in payload["head_to_head"]
        if (row["left"], row["right"]) == ("dq_3b", "gptq_3b")
    )
    assert [(row["left"], row["right"]) for row in payload["decomposition"]] == [
        ("dq_3b", "dq_3b_shuf"),
        ("dq_3b_shuf", "dq_3b_unif"),
        ("dq_3b_unif", "gptq_3b"),
    ]
    assert round(sum(row["delta_points"] for row in payload["decomposition"]), 2) == round(
        margin, 2
    )

    # Every further draw runs from the arm its own rung runs from -- `dq_3b` here, since the
    # shuffle rung is the first -- so the spread is the spread on that rung and not on some
    # other quantity wearing the same row.
    assert [(row["left"], row["right"]) for row in payload["replicates"]] == [
        ("dq_3b", "dq_3b_shufs1"),
        ("dq_3b", "dq_3b_shufs2"),
    ]
    block = _redrawn_block(table, printed)
    assert "redrawn: shuffle @1" in block and "redrawn: shuffle @2" in block
    assert "shuffle vs shuffle" not in printed


def test_the_ladder_prints_the_draw_a_rule_picked_and_not_the_one_that_flatters_it(
    table: Any, tmp_path: Path
) -> None:
    """Which draw is the rung is decided by lowest seed, fixed before any of them ran.

    A block that printed, say, the median or the best of four draws would be choosing the
    published rung with the numbers in hand. The fixture degrades each successive control
    further, so seed 0 is here the *smallest* of the three shuffle rungs -- if the printer
    were picking by size in either direction this fails, and if it were picking by seed it
    passes for the reason the docstring gives.

    Turns red when: the representative draw is chosen from the measurements.
    """
    out = _write_panel(
        tmp_path / "panel", nulls=("shuffle",), draws=(("shuffle", 1), ("shuffle", 2))
    )
    _run(table, out, "--json-out", str(tmp_path / "panel.json"))
    payload = json.loads((tmp_path / "panel.json").read_text(encoding="utf-8"))

    rung = next(row for row in payload["decomposition"] if row["right"] == "dq_3b_shuf")
    others = [row["delta_points"] for row in payload["replicates"]]
    assert len(others) == 2
    assert all(other > rung["delta_points"] for other in others), (
        "the fixture makes each later draw a wider rung, so a size rule would not pick seed 0"
    )


def test_a_rung_drawn_once_prints_no_redrawn_block(table: Any, tmp_path: Path) -> None:
    """A panel whose controls each ran once has no spread, and must not imply it has one.

    An empty block titled "the same rung, redrawn" is a claim that the question was asked,
    in a table where every other block appears only when its arms did. The banked panels
    were drawn once and their committed tables must keep re-printing identically.

    Turns red when: the block stops being guarded by whether anything was redrawn.
    """
    printed = _run(table, _write_panel(tmp_path / "panel", nulls=("shuffle", "uniform")))

    assert "the decomposition" in printed
    assert "redrawn" not in printed


def test_a_redrawn_rung_is_corrected_in_its_own_family(table: Any, tmp_path: Path) -> None:
    """The spread on a rung is not a rung, and must not enlarge the ladder's multiplier.

    Holm's multiplier is the size of the family a comparison is corrected in. Folding two
    further draws into the three-row ladder would raise every rung's adjusted p by a factor
    of 5/3 for having asked how much the choice of permutation was worth -- a question about
    the control's variance, not about what the allocator put back.

    Turns red when: the two blocks are corrected together, in either direction.
    """
    out = _write_panel(
        tmp_path / "panel", nulls=("shuffle", "uniform"), draws=(("shuffle", 1), ("shuffle", 2))
    )
    printed = _run(table, out)

    assert "Holm-adjusted over 3 of 3 comparisons" in _decomposition_block(table, printed)
    assert "Holm-adjusted over 2 of 2 comparisons" in _redrawn_block(table, printed)


def test_a_third_control_lengthens_the_ladder_and_still_partitions_the_margin(
    table: Any, tmp_path: Path
) -> None:
    """The chain is k+1 rungs over k controls, for whatever k the panel holds.

    The two-control ladder could be produced by code that knew there were two, and the
    third control is the first that would catch it. What it must not do is displace a rung:
    the nested modes each remove more than the last, so the rungs stay in declaration order,
    the new one lands between the two it was declared between, and the column still adds to
    the same head-to-head margin -- a chain of any length over the same endpoints does, and
    that is exactly why the ordering has to be asserted alongside the sum.

    Turns red when: the chain hard-codes its length, or orders rungs by anything but rank.
    """
    out = _write_panel(tmp_path / "panel", nulls=("shuffle", "flat", "uniform"))
    printed = _run(table, out, "--json-out", str(tmp_path / "panel.json"))
    payload = json.loads((tmp_path / "panel.json").read_text(encoding="utf-8"))

    margin = next(
        row["delta_points"]
        for row in payload["head_to_head"]
        if (row["left"], row["right"]) == ("dq_3b", "gptq_3b")
    )
    assert [(row["left"], row["right"]) for row in payload["decomposition"]] == [
        ("dq_3b", "dq_3b_shuf"),
        ("dq_3b_shuf", "dq_3b_flat"),
        ("dq_3b_flat", "dq_3b_unif"),
        ("dq_3b_unif", "gptq_3b"),
    ]
    assert round(sum(row["delta_points"] for row in payload["decomposition"]), 2) == round(
        margin, 2
    )

    block = _decomposition_block(table, printed)
    # Only the last rung is the shape rung, however many rungs precede it: everything above
    # it is a control against a control and is signal, and a block that named two of them
    # "shape" would be claiming the allocator's structure was put back twice.
    assert block.count("shape:") == 1
    assert "shape: uniform vs GPTQ" in block
    assert "signal: shuffle vs flat" in block and "signal: flat vs uniform" in block


def test_a_scheme_is_read_when_recorded_and_recovered_from_the_method_when_not(
    table: Any, tmp_path: Path
) -> None:
    """A scheme is a recipe input, so nothing downstream can measure it back off the weights.

    A symmetric arm and an asymmetric one at the same anchor are the same size, the same
    width and the same shape -- the only place the difference survives is the side file the
    driver wrote. So the row carries it, and a comparison between two arms can say whether
    it spans the scheme as well as the allocation.

    Both flags postdate the panel that needed them, so the arms that motivated the control
    have no `symmetric` key at all. Reading that as unknown would leave the table unable to
    name a pair it does contain, and what those arms ran is knowable: the method's default.
    The recovery is labelled `method-default` rather than passed off as `recorded`, and it
    goes through `_llmc.default_symmetric` -- the recipe builder's own rule, not a copy --
    so a change to what a method defaults to cannot leave this reading the old answer.

    Turns red when: the field stops being carried, the recovery starts claiming it was
    recorded, or the default is re-spelled here instead of imported.
    """
    out = _write_panel(
        tmp_path / "arms", schemes={"gptq_3b": {"symmetric": False, "actorder": "group"}}
    )
    dest = tmp_path / "table.json"
    _run(table, out, "--json-out", str(dest))
    built = json.loads(dest.read_text(encoding="utf-8"))
    schemes = {arm["label"]: arm["scheme"] for arm in built["arms"]}

    assert schemes["gptq_3b"] == {
        "symmetric": False,
        "actorder": "group",
        "source": "recorded",
    }
    assert schemes["gptq_4b"] == {"symmetric": True, "actorder": None, "source": "method-default"}
    assert schemes["awq_4b"] == {"symmetric": False, "actorder": None, "source": "method-default"}
    # Nothing to read and nothing to recover: the ceiling and the arms this repository
    # quantizes itself have no side file, and DynQuant's asymmetry is a property of the
    # encoder rather than something any run here wrote down.
    assert schemes["bf16"] is None
    assert schemes["dq_3b"] is None

    driver = _load("_dq_llmc_default", REPO_ROOT / "experiments" / "_llmc.py")
    assert schemes["gptq_4b"]["symmetric"] == driver.default_symmetric("gptq")
    assert schemes["awq_4b"]["symmetric"] == driver.default_symmetric("awq")
