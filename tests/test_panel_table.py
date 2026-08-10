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

    for arm in arms:
        label = arm["label"]
        if label in omit:
            arm["record"] = None
            continue
        record = out / f"{label}.json"
        record.write_text(json.dumps(_record(label, hits[label]), indent=2), encoding="utf-8")
        arm["record"] = str(record)
        if arm["kind"] in ("gptq", "awq"):
            record.with_suffix(".quant.json").write_text(
                json.dumps({"method": arm["kind"], "bits": arm["anchor"], "params": PARAMS}),
                encoding="utf-8",
            )

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
                extra={"map": arm["map"]} if "map" in arm else {},
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
    block = printed.split("head to head, at matched bytes")[1].split("head to head, on")[0]
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
    # Two head-to-head rows name awq_3b, and one ceiling row does.
    assert printed.count("(needs both arms)") == 3
    assert "3/7 arms scored" not in printed and "6/7 arms scored" in printed
    assert "Holm-adjusted over 4 of 6 comparisons" in printed, "the head family shrinks with it"
    assert "Holm-adjusted over 5 of 6 comparisons" in printed, "and so does the ceiling family"
    # A short family is corrected less than the finished panel will be, so an adjusted
    # p read mid-run can only move the unfavourable way. The count does not say that.
    assert printed.count("a short family, so these adjusted p are weaker") == 2


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

    block = (
        printed.split("head to head, at matched bytes")[1]
        .split("what each method cost")[0]
        .splitlines()
    )
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
    block = printed.split("head to head, at matched bytes")[1].split("what each method cost")[0]
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
    block = printed.split("head to head, at matched bytes")[1].split("head to head, on")[0]
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

    block = _run(table, out).split("head to head, at matched bytes")[1].split("head to head, on")[0]
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
    block = _run(table, out).split("head to head, at matched bytes")[1].split("head to head, on")[0]
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
    """The rows of the heterogeneity block, keyed by the comparison they name."""
    assert "is the margin the same on every source?" in printed, printed
    block = printed.split("is the margin the same on every source?")[1]
    rows = {}
    for line in block.splitlines():
        for _, _, question in _load("_dq_ht", SCRIPT).HEAD_TO_HEAD:
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
    assert "is the margin the same on every source?" not in printed


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
