"""An executable record of the research allocator's behaviour.

This is not a regression test. Nothing here will ever be "fixed" -- it pins what
the supplement's ``allocate_bits_pareto`` actually does, so that

* ``docs/legacy-audit.md`` item 4 stays a checkable claim rather than a story, and
* ``--preset paper-3.15`` has an executable definition of the behaviour it must
  reproduce bit-for-bit.

The finding it pins: at the paper's headline 3-bit target the greedy ROI loop
never runs, because the stability floors alone already cost more than the budget.
``allocator.py:137`` returns the floor map, and the importance score -- the
method's central contribution -- is never read. The decisive check is
:func:`test_inverting_every_score_changes_nothing`: negate every score and not one
of 282 modules changes width.

Runs against the **vendored** copy in :mod:`dynquant._legacy`, not against the
research tree, so it executes everywhere the package installs -- including CI,
where the supplement is not checked out. ``tests/test_legacy_provenance.py`` is
what ties the vendored copy back to the original.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STATS = REPO_ROOT / "stats" / "qwen3_14b" / "unified_gasq_stats_collapsed.json"

# Qwen3-14B geometry. The stats file records no parameter counts -- that omission
# is itself audit item 10 -- so they are derived from the published config.
HIDDEN, INTERMEDIATE, VOCAB = 5120, 17408, 151936
Q_DIM, KV_DIM = 40 * 128, 8 * 128  # 40 attention heads, 8 KV heads, head_dim 128

pytestmark = [
    pytest.mark.golden,
    pytest.mark.skipif(
        not STATS.is_file(),
        reason="shipped stats fixture not present",
    ),
]


def _param_count(name: str) -> int:
    if "embed_tokens" in name or "lm_head" in name:
        return VOCAB * HIDDEN
    if "q_proj" in name:
        return HIDDEN * Q_DIM
    if "k_proj" in name or "v_proj" in name:
        return HIDDEN * KV_DIM
    if "o_proj" in name:
        return Q_DIM * HIDDEN
    if any(x in name for x in ("gate_proj", "up_proj", "down_proj")):
        return HIDDEN * INTERMEDIATE
    raise AssertionError(f"unclassified module in the shipped stats: {name}")


@pytest.fixture(scope="module")
def legacy():
    """The supplement's allocator, vendored verbatim.

    Imported from ``dynquant._legacy`` rather than from ``dynquant_paper`` at the
    repo root: the research tree is not committed (it carries the credential
    literal and the author's VM paths), so an import from there would make this
    file skip in CI -- exactly where the audit's claims most need checking.
    """
    from dynquant._legacy.allocator import _min_bits_floor, allocate_bits_pareto

    return allocate_bits_pareto, _min_bits_floor


@pytest.fixture(scope="module")
def shipped():
    layers = json.loads(STATS.read_text(encoding="utf-8"))["layers"]
    params = {n: _param_count(n) for n in layers}
    scores = {n: float(layers[n].get("grad_norm_var", 0.0)) for n in layers}
    return layers, params, scores


def test_the_shipped_stats_are_the_shape_the_audit_describes(shipped):
    layers, params, _ = shipped
    assert len(layers) == 282, "2 (embed, lm_head) + 40 layers x 7 projections"
    assert sum(params.values()) == pytest.approx(14.768e9, rel=1e-3)
    # Audit item 10: the LM head really did record zero gradient observations, and
    # survives only because a floor rescues it.
    assert layers["base_model.model.lm_head"]["grad_norm_count"] == 0


@pytest.mark.parametrize(
    ("target", "loop_runs"),
    [(3.0, False), (3.15, False), (3.5, False), (4.0, True)],
)
def test_greedy_loop_only_runs_above_the_floor_cost(shipped, legacy, target, loop_runs):
    """``remaining < 0`` for every target at or below the allocator's own default.

    ``allocate_bits_pareto``'s default is ``target_avg_bits=3.5``, and the floors
    cost 3.5477 -- so the shipped default configuration also early-returns.
    """
    allocate, min_floor = legacy
    layers, params, scores = shipped

    variable = [n for n in layers if "lm_head" not in n]  # safety floor, off-budget
    variable_params = sum(params[n] for n in variable)
    floor_cost = sum(params[n] * (_floor_of(n, layers, min_floor, target) - 2) for n in variable)
    remaining = variable_params * target - variable_params * 2.0 - floor_cost

    assert (remaining >= 0) is loop_runs, f"remaining={remaining / 1e9:+.3f} Gbit"

    allocation = allocate(scores, params, layers, target_avg_bits=target)
    achieved = sum(allocation[n] * params[n] for n in variable) / variable_params
    if loop_runs:
        assert achieved == pytest.approx(target, abs=1e-3)
    else:
        # Silently misses its own target by half a bit, and reports nothing.
        assert achieved == pytest.approx(3.5477, abs=1e-3)


def _floor_of(name, layers, min_floor, target):
    if "embed_tokens" in name:
        return 4
    rms = float(layers[name].get("activation_rms_ema", 0.0))
    return int(min_floor(name, rms, target, False))


@pytest.mark.parametrize("target", [3.0, 3.15, 3.5])
def test_inverting_every_score_changes_nothing(shipped, legacy, target):
    """The decisive check for audit item 4.

    If the importance score influenced allocation at all, negating every score
    would move at least one module. It moves none, at every target the paper
    reports.
    """
    allocate, _ = legacy
    layers, params, scores = shipped

    baseline = allocate(scores, params, layers, target_avg_bits=target)
    inverted = allocate({n: -v for n, v in scores.items()}, params, layers, target_avg_bits=target)

    changed = [n for n in baseline if baseline[n] != inverted[n]]
    assert changed == [], f"{len(changed)}/{len(baseline)} modules changed"


def test_scores_do_bite_once_the_budget_clears_the_floors(shipped, legacy):
    """The same check at 3.8, to prove the test above is not vacuous.

    Without this, a harness that silently compared a map to itself would pass
    ``test_inverting_every_score_changes_nothing`` and the audit's central claim
    would rest on nothing.

    3.8 rather than 4.0, and the difference is the point. At 4.0 the budget is
    large enough to lift *every* remaining module to 4-bit, so there is no choice
    left for a score to influence and inverting the scores changes nothing there
    either -- for a completely different reason. Only in the band between the floor
    cost and 4.0 does the greedy ROI ordering actually select.
    """
    allocate, _ = legacy
    layers, params, scores = shipped

    baseline = allocate(scores, params, layers, target_avg_bits=3.8)
    inverted = allocate({n: -v for n, v in scores.items()}, params, layers, target_avg_bits=3.8)

    changed = [n for n in baseline if baseline[n] != inverted[n]]
    assert len(changed) > 50, "at 3.8 the greedy loop must be selecting on score"


@pytest.mark.parametrize(
    ("target", "expected_changed"),
    [(3.5, 0), (3.6, 16), (3.7, 46), (3.8, 64), (3.9, 32), (3.95, 16), (4.0, 0)],
)
def test_the_window_in_which_the_score_matters_at_all(shipped, legacy, target, expected_changed):
    """Map the full window, both edges included.

    Zero at 3.5 because the floors early-return; zero at 4.0 because every
    variable module reaches 4-bit whatever the order. The method's contribution
    influences the outcome only in between -- and the paper's headline setting is
    3.0, outside it.
    """
    allocate, _ = legacy
    layers, params, scores = shipped

    baseline = allocate(scores, params, layers, target_avg_bits=target)
    inverted = allocate({n: -v for n, v in scores.items()}, params, layers, target_avg_bits=target)

    changed = sum(1 for n in baseline if baseline[n] != inverted[n])
    assert changed == expected_changed


def test_the_returned_map_is_the_hand_written_floor_map(shipped, legacy):
    """What the paper reports as its 3-bit configuration is ``_min_bits_floor``.

    Embeddings 4, attention 4, MLP gate 4, MLP up/down 3 (except where the
    activation-spike rule lifts them), LM head 8.
    """
    allocate, _ = legacy
    layers, params, scores = shipped
    allocation = allocate(scores, params, layers, target_avg_bits=3.0)

    def widths(fragment):
        return {allocation[n] for n in allocation if fragment in n}

    assert widths("embed_tokens") == {4}
    assert widths("lm_head") == {8}
    assert widths("gate_proj") == {4}
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert widths(projection) == {4}, projection
    # up/down are 3-bit except where activation_rms > 2.5 forces 4.
    assert widths("up_proj") | widths("down_proj") == {3, 4}
    spiking = [n for n in layers if float(layers[n].get("activation_rms_ema", 0.0)) > 2.5]
    assert len(spiking) == 42
    assert sum(1 for n in spiking if "up_proj" in n or "down_proj" in n) == 9
