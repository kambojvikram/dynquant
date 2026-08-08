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
    spec = importlib.util.spec_from_file_location("_dq_panel_table", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dq_panel_table"] = module
    spec.loader.exec_module(module)
    return module


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


def _write_panel(
    out: Path,
    *,
    drift: int = 0,
    breach_at_3b: bool = True,
    omit: tuple[str, ...] = (),
    ceiling_correct: int = 320,
) -> Path:
    """A full seven-arm panel on disk, in the shape ``arms_lfm2 run`` writes."""
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
                violations = (
                    [
                        {
                            "name": f"model.layers.{n}.feed_forward.experts.gate_up_proj",
                            "role": "moe_expert_gate",
                            "floor_bits": 4,
                            "assigned_bits": 3,
                            "num_params": 150_000_000,
                        }
                        for n in range(22)
                    ]
                    if width == 3 and breach_at_3b
                    else []
                )
                path.write_text(
                    json.dumps(
                        {
                            "schema": "dynquant/allocation/1",
                            "model": "/runs/s4/merged",
                            "group_size": 128,
                            "maps": {
                                str(ANCHORS[width]): {
                                    "target_label": f"{ANCHORS[width]}B",
                                    "average_bits": 4.1563 if width == 4 else 3.1488,
                                    "nbytes": entry["nbytes"],
                                    "group_size": 128,
                                    "histogram": {
                                        "2": 1_000_000,
                                        "4": 7_000_000_000,
                                        "8": 300_000_000,
                                    },
                                    "violations": violations,
                                    "bits": {},
                                }
                            },
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
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

    (out / "arms.json").write_text(
        json.dumps(
            {
                "model": "/runs/s4/merged",
                "group_size": 128,
                "tolerance": 0.001,
                "anchors": {str(width): value for width, value in ANCHORS.items()},
                "arms": arms,
            },
            indent=2,
        ),
        encoding="utf-8",
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
    assert four_bit.endswith("NOT separated"), "the verdict is the corrected one"

    three_bit = next(
        line for line in printed.splitlines() if line.startswith("3b  DynQuant vs GPTQ")
    )
    assert three_bit.endswith("separated") and not three_bit.endswith("NOT separated")
    assert "Holm-adjusted over the 6 comparisons" in printed


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
    assert "moe_expert_gate" in printed, "the breached role is named, not just counted"
    assert "3.30G params" in printed
    # The 2-bit row holds a million parameters, not zero. Fixed G units printed 0.00G.
    assert "2b 1M" in printed

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
    assert "Holm-adjusted over the 4 comparisons" in printed, "the head family shrinks with it"
    assert "Holm-adjusted over the 5 comparisons" in printed, "and so does the ceiling family"


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
