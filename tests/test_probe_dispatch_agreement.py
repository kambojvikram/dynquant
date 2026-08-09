"""Three dispatches, three pairs, and the two ways this probe could lie about which.

The probe itself needs a GPU and an 8B checkpoint. What is covered here is everything
between the forwards, which is where the mistakes this campaign has actually made live:

* labelling the linearised loop by ``config._experts_implementation``, which
  ``linearize_moe`` does not touch -- the config keeps saying whatever was set before the
  banks were rewritten, so a probe that trusts it reports two ``eager`` passes and one
  pair that never existed;
* printing a zero for a pair whose two sides ran the same dispatch, which reads as
  agreement and means the probe failed to move the model.

No torch, no model. The passes are built by hand so every token in the output is a token
this file chose.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "phase4" / "probe_dispatch_agreement.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def probe() -> Any:
    return _load("probe_dispatch_agreement", SCRIPT)


class _Config:
    def __init__(self, implementation: str | None) -> None:
        self._experts_implementation = implementation


class _Model:
    """Just enough of a model to be asked what it will run."""

    def __init__(self, implementation: str | None) -> None:
        self.config = _Config(implementation)


def _pass(probe: Any, name: str, argmax: list[int], seconds: float = 1.0) -> Any:
    return probe.Pass(name=name, argmax=argmax, seconds=seconds, banks=0, linears=0)


def test_a_model_with_no_banks_is_the_loop_whatever_its_config_still_says(probe: Any) -> None:
    """The config is not a witness once the modules have been rewritten.

    ``linearize_moe`` replaces every bank with 32 ``Linear`` modules and leaves
    ``config._experts_implementation`` exactly as it found it. The probe sets ``eager``
    before it linearises, so the stale attribute says ``eager`` at the moment the loop is
    running -- and a probe that reads it labels two of its three passes identically, which
    then collapses the pair it was written to measure into a refusal.

    Turns red when: ``dispatch_name`` goes back to reading the config first.
    """
    assert probe.dispatch_name(_Model("eager"), banks=0) == "the loop"
    assert probe.dispatch_name(_Model("grouped_mm"), banks=0) == "the loop"
    assert probe.dispatch_name(_Model("eager"), banks=22) == "eager"
    assert probe.dispatch_name(_Model("grouped_mm"), banks=22) == "grouped_mm"
    assert probe.dispatch_name(_Model(None), banks=22) == "unset"


def test_every_pair_of_three_dispatches_is_reported_and_named(probe: Any) -> None:
    """Three passes make three pairs, and the report has to print all three.

    This campaign has carried a number across a pair boundary once already: 1.24% is
    ``eager`` against ``grouped_mm``, the panel's 1.9-2.3x clock is the loop against
    ``grouped_mm``, and the bitwise-identical result is ``eager`` against the loop. A
    report that emits one rate and calls it "the dispatch difference" is how that happens
    again.

    Turns red when: the pairing collapses to consecutive passes, or a pair loses the two
    dispatch names from its label.
    """
    passes = [
        _pass(probe, "grouped_mm", [1, 2, 3, 4]),
        _pass(probe, "eager", [1, 9, 3, 4]),
        _pass(probe, "the loop", [1, 9, 3, 8]),
    ]
    pairs = {row["pair"]: row for row in probe.report(passes)["pairs"]}
    assert set(pairs) == {
        "grouped_mm vs eager",
        "grouped_mm vs the loop",
        "eager vs the loop",
    }
    assert pairs["grouped_mm vs eager"]["differing"] == 1
    assert pairs["grouped_mm vs the loop"]["differing"] == 2
    assert pairs["eager vs the loop"]["differing"] == 1
    assert pairs["eager vs the loop"]["rate"] == pytest.approx(0.25)


def test_two_passes_on_one_dispatch_are_refused_rather_than_scored_as_perfect_agreement(
    probe: Any,
) -> None:
    """The zero that means the probe did nothing.

    If ``use_eager_experts`` returns without moving anything -- an older transformers with
    no such dispatch, a config that never had the attribute -- the second pass re-runs the
    first. Every token matches, the rate is 0.0000, and the honest reading of that number
    is "these two dispatches are identical", which is exactly the conclusion four places in
    this package asserted and had to withdraw.

    Turns red when: an identical-dispatch pair starts reporting a rate.
    """
    passes = [_pass(probe, "eager", [1, 2, 3]), _pass(probe, "eager", [1, 2, 3])]
    row = probe.report(passes)["pairs"][0]
    assert "refused" in row
    assert "did not move the model" in row["refused"]
    assert "rate" not in row and "differing" not in row


def test_the_clock_travels_with_the_pair_it_belongs_to(probe: Any) -> None:
    """Both seconds and their ratio, per pair, in the pair's own order.

    The seconds are the cheap half of a measurement the alternative to which is an 8.5 to
    17 hour re-score, so they are worth as much care as the rates. The ratio is
    right-over-left so it reads the same direction as the pair label.

    Turns red when: the ratio inverts, or the pair keeps a single scalar instead of both
    sides' cost.
    """
    passes = [
        _pass(probe, "grouped_mm", [1, 2], seconds=10.0),
        _pass(probe, "the loop", [1, 3], seconds=25.0),
    ]
    row = probe.report(passes)["pairs"][0]
    assert row["pair"] == "grouped_mm vs the loop"
    assert row["seconds"] == [10.0, 25.0]
    assert row["ratio"] == pytest.approx(2.5)


def test_unequal_pass_lengths_compare_the_overlap_and_say_how_much_it_was(probe: Any) -> None:
    """A short pass truncates the comparison instead of raising or padding.

    Passes cannot legitimately differ in length -- the same items produce the same gold
    positions -- so this is a defensive path, and the thing that matters about it is that
    the token count in the row is the count actually compared. A rate over 4 tokens
    labelled as being over 6 is worse than a crash.

    Turns red when: the denominator stops being the compared length.
    """
    passes = [_pass(probe, "eager", [1, 2, 3, 4, 5, 6]), _pass(probe, "the loop", [1, 9, 3, 4])]
    row = probe.report(passes)["pairs"][0]
    assert row["tokens"] == 4
    assert row["differing"] == 1
    assert row["rate"] == pytest.approx(0.25)
