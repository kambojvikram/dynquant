"""The four ways a progress-line profile can turn a length signal into a fixed cost.

The profile exists to answer one question -- is an arm uniformly slower, or slower in
places -- and every failure mode below answers it wrongly rather than failing loudly:

* splitting arms on a time gap, which cuts a slow block in half and invents a boundary;
* letting the first stamp of an arm open an interval, which divides a model load and a
  32-to-47-minute quantization by 800 items and calls it decode;
* pairing blocks by position when the two arms start at different items, so block 4000 of
  one arm is divided by block 1600 of the other;
* printing a fixed-cost ceiling for an arm that is *faster* than the other somewhere,
  which is the one number in the report a reader would quote.

No log file and no box. Every stamp here is written by hand so the arithmetic in the
assertions is arithmetic this file can do.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "phase4" / "rate_profile.py"
EPOCH = datetime(2026, 8, 9, 0, 0, 0)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def profile() -> Any:
    return _load("rate_profile", SCRIPT)


def _log(entries: list[tuple[int, int]]) -> str:
    """(seconds since epoch, items done) -> the sampler's own line format."""
    return "\n".join(
        f"{(EPOCH + timedelta(seconds=offset)):%Y-%m-%dT%H:%M:%SZ}   [text2sql] {done}/12000"
        for offset, done in entries
    )


def _blocks(profile: Any, entries: list[tuple[int, int]]) -> list[Any]:
    return profile.blocks(profile.parse(_log(entries)))


def test_an_arm_boundary_is_a_counter_reset_and_never_a_time_gap(profile: Any) -> None:
    """A slow block is longer than the gap that separates two arms.

    In the real log ``awq_4b``'s block 9600 took 3,781 s while the quantization between
    ``dq_4b`` and ``gptq_3b`` took about 2,200 s, so any gap threshold big enough to catch
    the boundary also splits that arm in two and reports its two halves as separate models.
    Only the counter going backwards is a boundary.

    Turns red when: the splitter grows a duration heuristic.
    """
    stamps = profile.parse(_log([(0, 800), (3800, 1600), (5000, 2400), (7200, 800), (8000, 1600)]))
    arms = profile.split_arms(stamps)
    assert [len(arm) for arm in arms] == [3, 2]
    assert [arm[0].done for arm in arms] == [800, 800]
    assert [b.seconds for b in profile.blocks(arms[0])] == [3800.0, 1200.0]


def test_the_first_stamp_of_an_arm_opens_no_interval(profile: Any) -> None:
    """The gap before it holds a load and a quantization, which is not eval time.

    ``gptq_3b``'s first stamp sits about 86 minutes after ``dq_4b``'s last one, and 2,208 s
    of that is its own quantization. Charging it to 800 items would put the arm's opening
    block near 6.5 s/item and make every ratio computed from it wrong in the direction the
    report is trying to test.

    Turns red when: blocks are derived across an arm boundary, or an arm's opening stamp
    starts contributing.
    """
    stamps = profile.parse(_log([(0, 11200), (600, 12000), (5760, 800), (6400, 1600)]))
    first, second = profile.split_arms(stamps)
    assert [b.seconds for b in profile.blocks(first)] == [600.0]
    assert [b.seconds for b in profile.blocks(second)] == [640.0]
    assert all(b.seconds < 5000 for arm in (first, second) for b in profile.blocks(arm))


def test_blocks_pair_by_item_index_and_not_by_position(profile: Any) -> None:
    """The sampler started part-way through the first arm, so positions do not line up.

    ``awq_4b`` is stamped from item 3200 and ``dq_4b`` from item 800. Zipping them pairs
    ``awq_4b``'s block 4000 with ``dq_4b``'s block 1600 -- different questions on the two
    sides of every ratio, which is the one property that makes the profile readable at all.

    Turns red when: ``align`` starts zipping, or joins on anything but the end index.
    """
    left = _blocks(profile, [(0, 3200), (100, 4000), (200, 4800)])
    right = _blocks(profile, [(0, 800), (400, 1600), (800, 2400), (1200, 4000), (1600, 4800)])
    pairs = profile.align(left, right)
    assert [end for end, _, _ in pairs] == [4000, 4800]
    assert [one.items for _, one, _ in pairs] == [800, 800]
    assert [other.items for _, _, other in pairs] == [1600, 800]


def test_a_ratio_below_one_withdraws_the_fixed_cost_ceiling_it_would_licence(
    profile: Any,
) -> None:
    """The load-bearing claim, and the condition under which it is not available.

    A fixed per-forward cost -- unpacking 3-bit weights, 704 module calls instead of 22
    grouped matmuls -- is the same work on every block, so the arm carrying it can never be
    the faster one. One block where it *is* faster says the excess elsewhere is decode
    steps, not fixed work, and the ceiling has to say so rather than print its minimum and
    let the minimum be quoted.

    Turns red when: the ceiling is emitted for one direction only, or ``ceiling_usable``
    stops tracking the 1.0 boundary.
    """
    slow = _blocks(profile, [(0, 800), (2000, 1600), (2800, 2400)])
    fast = _blocks(profile, [(0, 800), (1000, 1600), (3000, 2400)])
    row = profile.compare("slow", slow, "fast", fast)
    assert row["blocks"][0]["ratio"] == pytest.approx(2.0)
    assert row["blocks"][1]["ratio"] == pytest.approx(0.4)
    assert row["fixed_cost_ceiling"] == {"slow": pytest.approx(0.4), "fast": pytest.approx(0.5)}
    assert row["ceiling_usable"] == {"slow": False, "fast": False}

    always_slower = _blocks(profile, [(0, 800), (1800, 1600), (5400, 2400)])
    row = profile.compare("always_slower", always_slower, "fast", fast)
    assert row["fixed_cost_ceiling"]["always_slower"] == pytest.approx(1.8)
    assert row["ceiling_usable"] == {"always_slower": True, "fast": False}


def test_arms_with_no_shared_block_are_refused_rather_than_compared(profile: Any) -> None:
    """An empty join is a refusal, not an empty table with a ceiling attached.

    Two arms whose stamps never land on the same item index share no question, and a pair
    row carrying ``ratio_min`` over zero blocks would read as a measured result.

    Turns red when: the refusal path starts emitting ceilings.
    """
    left = _blocks(profile, [(0, 800), (100, 1600)])
    right = _blocks(profile, [(0, 900), (100, 1700)])
    row = profile.compare("left", left, "right", right)
    assert "refused" in row and "same 800 items" in row["refused"]
    assert "fixed_cost_ceiling" not in row and "blocks" not in row


def test_a_wrong_number_of_arm_names_refuses_instead_of_labelling_by_position(
    profile: Any, tmp_path: Path
) -> None:
    """Naming the arms is positional, so a miscount silently renames every row.

    ``rescore_eager.sh`` calls this with a hard-coded arm list and treats a failure as
    non-fatal, which is only safe while the failure is a refusal. A resumed panel, a
    restarted arm or a sampler launched late all change the arm count, and the version of
    this that zipped the shorter of the two would report ``dq_4b``'s blocks under
    ``gptq_3b``'s name and hand the report a ratio between two models that were never
    compared.

    Turns red when: the name list is zipped against the arms instead of checked.
    """
    log = tmp_path / "rate.log"
    log.write_text(_log([(0, 800), (100, 1600), (200, 800), (300, 1600)]), encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        profile.main([str(log), "--arms", "a,b,c"])
    assert "2 arm(s)" in str(caught.value) and "3 name(s)" in str(caught.value)

    assert profile.main([str(log), "--arms", "a,b"]) == 0


def test_a_block_the_clock_could_not_separate_is_dropped_and_counted(profile: Any) -> None:
    """Zero seconds is a resolution failure, and dividing by it is a crash, not a refusal.

    ``rescore_eager.sh`` stamps the driver's lines as they arrive rather than polling, so
    two progress lines emitted inside one second carry the same stamp -- which a short
    ``--limit`` smoke run produces on the first try. The caller treats a non-zero exit as
    "refused, re-run by hand", so a traceback there is a wrong message as well as an ugly
    one, and a surviving block that silently loses a same-second neighbour is worse: the
    aggregate would be over fewer blocks than the row count claims.

    Turns red when: the ratio divides before filtering, or the dropped count stops being
    reported alongside the blocks that survived.
    """
    left = _blocks(profile, [(0, 800), (0, 1600), (1000, 2400)])
    right = _blocks(profile, [(0, 800), (100, 1600), (600, 2400)])
    row = profile.compare("left", left, "right", right)
    assert [block["block_end"] for block in row["blocks"]] == [2400]
    assert row["blocks_dropped_below_clock_resolution"] == 1
    assert row["ratio_min"] == pytest.approx(2.0)

    every_block = profile.compare("left", left[:1], "right", right[:1])
    assert "blocks" not in every_block
    assert "under one second" in every_block["refused"]
    assert "share no block boundary" not in every_block["refused"]


def test_a_short_final_block_divides_by_its_own_item_count(profile: Any) -> None:
    """12,000 is not a multiple of 800 in every campaign, and the tail must not be scaled.

    Turns red when: the per-item cost divides by a constant block size instead of the
    items the block actually covered.
    """
    tail = _blocks(profile, [(0, 11600), (400, 12000)])
    assert tail[0].items == 400
    assert tail[0].per_item == pytest.approx(1.0)
