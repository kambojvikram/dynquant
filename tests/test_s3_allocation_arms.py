"""S3's four arms are only a controlled comparison if two things hold on the maps.

The campaign quantizes Ministral four ways at each anchor -- ``rtn`` (uniform, no signal),
``rank`` (the rank-product baseline), ``shuf`` (the signal, permuted within role) and ``dq``
(the headline) -- and reads the accuracy differences as statements about the allocator and
the signal. That reading needs the maps themselves to satisfy two preconditions, and each
has already failed once:

* **Matched bytes.** An arm that lands on a different on-disk size is being compared at a
  different compression ratio, and any accuracy gap is confounded by the size gap. The
  allocator hits a target *size*, not a nominal bit width, so this is checkable exactly
  rather than to a tolerance.
* **A control that ablates.** ``shuf`` is supposed to destroy the correspondence between a
  module and its measurements while keeping the signal's distribution intact. Before
  ``d74059a`` the permutation was applied to the stats but not the moments, and the
  sensitivity table it produced was byte-identical to ``dq``'s: the control moved 0 of 129
  modules and the arm measured nothing at all, while looking like a clean null result.

Both run against the committed maps rather than a fixture, because a fixture proves the
assertions parse. Anchors are discovered from the files present, so the 4-bit arms are
covered by the same checks the moment they are committed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MAPS = REPO_ROOT / "experiments" / "phase3" / "s3_allocation" / "ministral-8b" / "maps"

#: Arms that carry one anchor each, keyed by the digit in their filename.
SIGNAL_ARMS = ("rank", "shuf", "dq")


def _record(arm: str) -> dict:
    return json.loads((MAPS / f"map.{arm}.json").read_text())


def _sole_map(arm: str) -> dict:
    (entry,) = _record(arm)["maps"].values()
    return entry


def _anchors() -> list[str]:
    """The anchor digits with at least one signal arm on disk -- ``["3"]``, ``["3", "4"]``.

    Discovery is what lets the 4-bit arms join without an edit, and it is also the way every
    check in this file could go vacuously green: an empty maps directory has no anchors and
    no assertions to fail. The 3.25 anchor is phase 2's headline and is committed, so its
    presence is asserted rather than discovered.
    """
    anchors = sorted({p.name[len("map.") + len("rank")] for p in MAPS.glob("map.rank?.json")})
    assert "3" in anchors, f"the 3.25-bit anchor is missing from {MAPS}"
    return anchors


@pytest.fixture(scope="module")
def rtn() -> dict:
    return _record("rtn")


def test_the_arms_are_compared_at_identical_bytes(rtn) -> None:
    """Every arm at an anchor must land on the same ``nbytes``, exactly.

    ``run_s3_allocate.py`` passes ``--target-size`` in bytes and allows a 0.001-bit slack,
    so equality here is a fact about the run rather than a rounding coincidence: the 3.25
    anchor puts all four arms on 3 257 925 632 bytes with a pairwise delta of zero. If an
    arm ever misses, the S3 table stops being a comparison of allocators and becomes a
    comparison of sizes, and the report would have no way to tell from the accuracy numbers.

    Turns red when: an arm is re-run against a different target, or the size accounting
    changes for one allocator path and not the others.
    """
    for anchor in _anchors():
        sizes = {
            arm: _sole_map(f"{arm}{anchor}")["nbytes"]
            for arm in SIGNAL_ARMS
            if _present(arm, anchor)
        }
        sizes["rtn"] = rtn["maps"][f"uniform-{anchor}"]["nbytes"]
        assert len(set(sizes.values())) == 1, f"anchor {anchor} is not byte-matched: {sizes}"


def test_the_shuffled_control_actually_ablates() -> None:
    """``shuf`` and ``dq`` must disagree somewhere, or the control arm measures nothing.

    This is the direct guard on the failure that produced a false null: a permutation that
    reaches the stats but not the moments leaves the sensitivity table -- which is what
    ``allocate_bits`` actually prices from -- untouched, and the two arms allocate
    identically. At 3.25 the repaired control moves 39 of 254 modules.

    Turns red when: the permutation stops covering every artifact the allocator reads, or a
    seed lands on an identity permutation that ``run_s3_allocate.py``'s ``moved`` guard did
    not catch because the moments carried the signal anyway.
    """
    for anchor in _anchors():
        if not (_present("shuf", anchor) and _present("dq", anchor)):
            continue
        shuffled = _sole_map(f"shuf{anchor}")["bits"]
        headline = _sole_map(f"dq{anchor}")["bits"]
        assert shuffled.keys() == headline.keys()
        moved = [name for name, bits in shuffled.items() if bits != headline[name]]
        assert moved, f"anchor {anchor}: the control allocates identically to the headline arm"


def test_each_arm_records_which_allocator_priced_it() -> None:
    """The arm names are a claim about code paths; the maps have to carry the evidence.

    ``rank`` is meant to price every module from the percentile score and ``dq`` from
    measured sensitivity wherever the channel moments reach -- two different branches of
    ``_Candidate.move_value``, selected by whether ``--moments`` was passed. Nothing about a
    finished map's bit widths reveals which branch ran, so an arm launched without its
    moments would produce a plausible file and a silently duplicated baseline. The
    ``allocator`` field is the only place that distinction survives into the artifact.

    Turns red when: an arm is regenerated without the flags its name implies.
    """
    for anchor in _anchors():
        for arm in ("shuf", "dq"):
            if _present(arm, anchor):
                assert _record(f"{arm}{anchor}")["allocator"] == "sensitivity"
        assert _record(f"rank{anchor}")["allocator"] == "rank_product"

    # The no-signal baseline has to have had no signal available to it.
    assert _record("rtn")["stats"] is None


def test_the_control_cannot_reach_the_two_tensors_the_report_is_about() -> None:
    """``lm_head`` and ``model.embed_tokens`` get the same width in ``shuf`` and ``dq``.

    Ministral is untied, so each of these is alone in its role group, and
    ``permutation_within_role`` is a fixed point on a group of one -- there is no other
    module of that role to swap with. They are nonetheless not *guaranteed* to match,
    because the budget is shared and a reallocation elsewhere can still move them; that they
    do match is an observed fact about this run, and it is the fact that bounds what the
    control arm can be read as saying. Together they are 13.4% of the model, so an S3
    ``shuf``-vs-``dq`` gap is a statement about the other 252 modules and silent on the two
    largest tensors.

    Turns red when: the control gains reach over the singletons -- which would be an
    improvement, and would make ``docs/reports/phase3-s2-ministral-signal-map.md`` wrong
    where it says the ablation cannot touch them.
    """
    for anchor in _anchors():
        if not (_present("shuf", anchor) and _present("dq", anchor)):
            continue
        shuffled = _sole_map(f"shuf{anchor}")["bits"]
        headline = _sole_map(f"dq{anchor}")["bits"]
        for singleton in ("lm_head", "model.embed_tokens"):
            assert shuffled[singleton] == headline[singleton], (
                f"anchor {anchor}: the control moved {singleton}, which the reports say it cannot"
            )


def _present(arm: str, anchor: str) -> bool:
    return (MAPS / f"map.{arm}{anchor}.json").exists()
