"""The signal-map verifier has to name every tensor the ranking left uninformed.

The verifier's job is to say whether a finished S2 map can be handed to S3, and it
answers that by finding tensors the allocator will place on no information. Two ways
of missing such a tensor have already happened once each, and both look like a clean
report:

* **Counting "scored" as "informed."** A module alone in its role group scores 0.5
  from the shipped per-role ranker no matter what was measured about it -- a
  percentile rank against a set of one. It is neither missing nor unexercised, so a
  coverage check built from those two lists never names it. On an untied checkpoint
  that module is the LM head, measured 1 492 times and carrying the highest saliency
  in the model.
* **Reading a counterfactual off one width.** The budget is shared, so a score that
  moves a large tensor's place in the ROI order changes how much budget reaches
  everything below it -- whether or not that tensor's own width changes. Phi's alias
  substitution was recorded as costing the checkpoint nothing on the strength of its
  own width holding at four targets, while between five and twelve other modules
  moved at each.

Both are checked against the two real phase-3 maps rather than a fixture, because
what is under test is the verifier's ability to find these tensors in the files it
will actually be run on -- and a fixture written to contain them proves only that the
author knew to put them there. Phi is tied and Ministral is not, which is the
distinction that produces the two cases.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "phase3" / "s3_allocation" / "verify_signal_map.py"
RUNS = REPO_ROOT / "experiments" / "phase3" / "s2_runs"

# The verifier imports `floor_headroom`, which builds the two real configs and therefore
# imports transformers at module scope. Without it the module cannot load at all, which
# in the core-only `test` matrix job is a collection error rather than a skip; the two
# pinned-transformers jobs are where these run.
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def verifier():
    spec = importlib.util.spec_from_file_location("_dq_verify", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dq_verify"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def phi(verifier) -> dict:
    return verifier.verify(RUNS / "phi4-mini.tulu3" / "stats" / "dynquant_stats.json", "phi4-mini")


@pytest.fixture(scope="module")
def ministral(verifier) -> dict:
    return verifier.verify(
        RUNS / "ministral-8b.tulu3" / "stats" / "dynquant_stats.json", "ministral-8b"
    )


def test_a_measured_module_alone_in_its_role_group_is_still_probed(ministral) -> None:
    """Ministral is untied, so ``lm_head`` is its own tensor in a role group of one.

    It has a complete measurement -- 1 492 gradient observations and the highest
    activation RMS in the model -- and the shipped ranker hands it 0.5 regardless. If
    the probe set is ever narrowed back to ``missing_stats | unexercised``, the
    allocator's placement of 6.7% of an 8B model goes unexamined and this report says
    so.

    Turns red when: the probe set stops including singleton role groups, or
    ``coverage.informed`` is computed without subtracting them.
    """
    coverage = ministral["coverage"]
    assert "lm_head" in coverage["singleton_role_groups"]
    assert "lm_head" not in coverage["missing_stats"]
    assert "lm_head" not in coverage["unexercised"]
    assert coverage["informed"] == coverage["scored"] - len(coverage["singleton_role_groups"])

    entry = ministral["neutral_module_sensitivity"]["lm_head"]
    assert entry["reasons"] == ["role group of one"]
    assert entry["shipped_score"] == pytest.approx(0.5)
    # Its own measurements, ranked against the model instead of against itself.
    assert entry["own_score"] > 0.9


def test_the_counterfactual_reports_the_map_and_not_one_width(phi) -> None:
    """Phi's alias substitution leaves ``embed_tokens`` put and rewrites the tail.

    This is the case that makes ``changes_width`` insufficient on its own: at every
    probed target the substituted score lands the tied embedding on the width it
    already had, and at every probed target other modules move. A verifier that
    reported only the first number would call this free.

    Turns red when: ``_counterfactual`` stops diffing the whole allocation, or the
    per-target record drops ``other_modules_moved``.
    """
    entry = phi["neutral_module_sensitivity"]["model.embed_tokens"]
    assert entry["alias"] == "lm_head"

    arms = [target["alias"] for target in entry["by_target"].values()]
    assert not any(arm["changes_width"] for arm in arms), "the width this was read off"
    assert all(arm["other_modules_moved"] > 0 for arm in arms), (
        "and the reallocation it was read without"
    )


def test_an_unexercised_module_has_no_own_signal_to_fall_back_on(phi, ministral) -> None:
    """``own_score`` is not a workaround for a missing measurement.

    ``model.embed_tokens`` carries no gradient observations under ``outer_exact`` on
    either checkpoint, so ranking it globally returns the same 0.5 the per-role ranker
    gave it and moves nothing. That equality is the evidence that the global-ranking
    counterfactual reflects measurements rather than manufacturing a number -- on
    Ministral's ``lm_head``, where a measurement exists, the same code path returns
    0.978 and moves 24 modules.

    Turns red when: the global-ranking path stops going through ``score_modules`` and
    starts synthesising a score for modules that have none.
    """
    for result in (phi, ministral):
        entry = result["neutral_module_sensitivity"]["model.embed_tokens"]
        assert "unexercised or no gradient observations" in entry["reasons"]
        assert entry["own_score"] == pytest.approx(entry["shipped_score"])
        for target in entry["by_target"].values():
            assert target["own"]["modules_moved"] == 0


def test_the_reports_quote_this_artifact_and_not_another_run() -> None:
    """The write-ups quote hard numbers off Ministral's stats file; pin them to it.

    ``phase3-s2-ministral-signal-map.md`` and ``docs/reports/README.md`` both say the
    discarded tensor is rank 1 of 254 on saliency at 1.78x the runner-up and rank 6 of
    254 on plasticity -- the pair of facts that makes the singleton 0.5 worth a report
    rather than a footnote, since a head that topped saliency alone could be doing it
    for scale reasons. Those numbers were read off this file. Swapping in another run's
    artifact, or regenerating this one under a different estimator, would leave the
    prose quoting measurements the repository no longer contains, and nothing else here
    would notice: every other check in this file is about relative structure and would
    stay green on any well-formed map.

    Turns red when: the Ministral stats artifact is replaced or regenerated without the
    two reports being re-read.
    """
    artifact = RUNS / "ministral-8b.tulu3" / "stats" / "dynquant_stats.json"
    stats = json.loads(artifact.read_text())["layers"]
    assert len(stats) == 254

    by_saliency = sorted(stats, key=lambda n: -(stats[n].get("activation_rms_ema") or 0.0))
    by_plasticity = sorted(stats, key=lambda n: -(stats[n].get("grad_norm_var") or 0.0))

    assert by_saliency[0] == "lm_head"
    head, runner_up = (stats[n]["activation_rms_ema"] for n in by_saliency[:2])
    assert head / runner_up == pytest.approx(1.78, abs=0.01)
    assert by_plasticity.index("lm_head") + 1 == 6


def test_the_record_names_the_allocator_it_describes(phi, ministral) -> None:
    """A bracket computed without a sensitivity table is a fact about one arm only.

    ``allocate_bits`` prices a module from measured loss wherever the channel moments
    cover it and from the percentile score only where they do not, so everything this
    script measures describes the ``rank`` baseline. Read as the headline ``dq`` arm it
    inverts: Ministral's ``lm_head`` scores 0.5 here and takes 3 bits, while ``dq3``
    prices it from its moments and gives it 4. The record therefore has to say which
    map it is about, out loud, rather than leaving it to be inferred from a keyword
    argument that is absent three files away.

    Turns red when: the verifier starts passing ``sensitivity=`` without relabelling
    its output, or the label is dropped.
    """
    for result in (phi, ministral):
        assert result["allocator"] == "rank_product"


def test_the_bracket_still_brackets(ministral) -> None:
    """Score 0.0 and 1.0 must span the realistic counterfactual, or one of them is wrong.

    The bracket is the outer bound and the counterfactual is a point inside it. If a
    real score ever lands outside the forced-0/forced-1 range, the two are being
    computed against different graphs or budgets and neither number means what the
    report says.

    Turns red when: the bracket and the counterfactual stop sharing a budget, a graph,
    or a score map.
    """
    for entry in ministral["neutral_module_sensitivity"].values():
        for target in entry["by_target"].values():
            lo = min(target["bits_at_score_0"], target["bits_at_score_1"])
            hi = max(target["bits_at_score_0"], target["bits_at_score_1"])
            assert lo <= target["own"]["bits"] <= hi
