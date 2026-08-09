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
    two different problem sets.
    """
    out = _write_panel(tmp_path / "arms")
    record = out / "gptq_4b.json"
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["decode"]["max_new_tokens"] = 320
    record.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    printed = _run(table, out)
    assert "NOT PAIRED" in printed
    assert "max_new_tokens" in printed, "the differing field is named"
    assert "records are not comparable" in printed
    assert "p (Holm)" in printed, "the block still prints, so the absence is visible"
    for line in printed.splitlines():
        assert not line.endswith(" separated"), "no verdict is issued off unpaired vectors"


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


def test_a_recorded_dispatch_refuses_to_pair_with_a_record_that_predates_the_field(
    table: Any, tmp_path: Path
) -> None:
    """The other half, and the panel's own next step.

    Re-scoring the three arms that kept their expert banks on ``--experts-impl eager``
    puts ``{found, ran}`` into their records and leaves the ``llm-compressor`` baselines
    as they are. ``_ABSENT != "eager"``, so the guard fires -- correctly, because a
    baseline written before the field cannot say what it ran. This pins that the fix to
    the arithmetic is not silent about the bookkeeping it breaks, so the re-score cannot
    be read as a clean seven-arm panel when it is a three-arm one beside four unknowns.

    Turns red when: ``experts.ran`` leaves ``EXPERTS_PAIRING_FIELDS``, or the exemption
    for an absent field widens from "pairs with absence" to "pairs with anything".
    """
    out = _write_panel(tmp_path / "arms")
    _set_experts(out, "dq_4b", {"found": "grouped_mm", "ran": "eager"})

    printed = _run(table, out)
    assert "eager (from grouped_mm)" in printed.split("experts dispatch")[1]
    assert "NOT PAIRED" in printed
    assert "experts.ran" in printed


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

    Turns red when: `eager` and a recovered loop stop collapsing to one class, which would
    make the re-score unable to clear the flag it was run to clear.
    """
    out = _write_panel(tmp_path / "arms")
    _all_indexed(out)

    printed = _run(table, out)
    assert "expert arithmetic" not in printed, "every arm is indexed; nothing to flag"
    assert "NOT PAIRED" in printed, (
        "the records still cannot pair -- recovery reads a sibling file and pairing reads "
        "the record, and this is where those two come apart"
    )


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
