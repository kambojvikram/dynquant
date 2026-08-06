"""The S3 driver's two claims have to hold before a GPU-week is spent on them.

S3 exists to separate three explanations for an accuracy win -- a bigger file, the
allocator's structure, the signal -- and it can only do that if two properties are
true of the run rather than of the report. Both are cheap to check here and expensive
to discover afterwards, because discovering either one late invalidates every arm:

* **The arms are the same size.** ``--target-size`` is a ceiling, so an allocator that
  cannot spend the last bits lands under it. Arms that drift apart turn a comparison of
  assignments into a comparison of sizes, and the table would not show it.
* **The shuffled control is a real control.** It has to carry exactly the same
  measurements as the treatment, attached to different modules, and hold every
  structural fact fixed. A permutation that changed the score *distribution* would
  measure the distribution; one that was near-identity would measure nothing.

Nothing here launches a subprocess or loads a model. The permutation is pure data and
the byte check is arithmetic, so the whole surface that decides whether S3's numbers
mean anything is reachable from CPU CI.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from dynquant.signals.schema import LayerStats, StatsFile, load_stats

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "scripts" / "run_s3_allocate.py"

#: The signal map from the first phase-3 fine-tune. Used rather than a synthetic file
#: because the properties under test are properties of *this* map -- 130 modules, two
#: singleton roles, one tensor with no gradient signal -- and a fixture built to be
#: convenient would not have had the tied embedding that makes a fixed point.
PHI_STATS = REPO_ROOT / "experiments" / "phase3" / "s2_runs" / "phi4-mini.tulu3" / "stats"


@pytest.fixture(scope="module")
def s3():
    spec = importlib.util.spec_from_file_location("_dq_s3", DRIVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dq_s3"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def phi() -> StatsFile:
    return load_stats(PHI_STATS)


def _synthetic(sizes: dict[str, int]) -> StatsFile:
    """A stats file with ``sizes[role]`` modules per role, each distinguishable."""
    layers = {}
    n = 0
    for role, count in sizes.items():
        for index in range(count):
            n += 1
            layers[f"m{n}.{role}.{index}"] = LayerStats(
                name=f"m{n}.{role}.{index}",
                role=role,
                activation_rms_ema=float(n),
                grad_norm_mean=float(n) * 10,
                grad_norm_var=float(n) * 100,
                grad_norm_count=n,
                param_count=1000 + n,
            )
    return StatsFile(layers=layers)


# --------------------------------------------------------------------------
# The shuffled control
# --------------------------------------------------------------------------


def test_the_control_carries_exactly_the_same_measurements(s3, phi) -> None:
    """The permuted arm must differ in correspondence only, never in distribution.

    This is the property that makes ``shuf`` an ablation of the *signal* rather than
    of the score distribution. Turns red the moment the permutation starts synthesising
    values -- zeroing them, resampling them, or permuting a subset of the fields so a
    module ends up with one donor's activation and another's gradient, which is a third
    distribution belonging to no module at all.
    """
    shuffled = s3.shuffle_within_role(phi, seed=0)

    def multiset(stats: StatsFile) -> list[tuple[float, ...]]:
        return sorted(
            tuple(getattr(layer, field) or 0.0 for field in s3.PERMUTED_FIELDS)
            for layer in stats.layers.values()
        )

    assert multiset(shuffled) == multiset(phi)


def test_the_control_actually_moves_measurements(s3, phi) -> None:
    """A control that leaves the map alone measures nothing and would read as a null."""
    shuffled = s3.shuffle_within_role(phi, seed=0)
    moved = [
        name
        for name, layer in shuffled.layers.items()
        if any(getattr(layer, f) != getattr(phi.layers[name], f) for f in s3.PERMUTED_FIELDS)
    ]
    assert len(moved) > len(phi.layers) // 2, f"only {len(moved)}/{len(phi.layers)} moved"


def test_identity_fields_stay_with_their_module(s3, phi) -> None:
    """Name, role and parameter count describe the module, not its behaviour.

    If ``param_count`` or ``role`` travelled with the measurements, the control would
    differ from the treatment in what the allocator is *pricing* -- role floors and
    tensor sizes -- and its accuracy gap would no longer isolate the signal.
    """
    shuffled = s3.shuffle_within_role(phi, seed=0)
    for name, layer in phi.layers.items():
        after = shuffled.layers[name]
        assert after.name == layer.name
        assert after.role == layer.role
        assert after.param_count == layer.param_count
        assert after.grad_estimator == layer.grad_estimator


def test_measurements_never_cross_a_role_boundary(s3) -> None:
    """Within-role, so the ranking the scorer actually performs is what gets destroyed.

    A global permutation would hand an embedding's activation scale to an attention
    projection, changing the between-role structure that per-role ranking deliberately
    holds fixed -- and the arm would then ablate the role policy as well as the signal.
    """
    stats = _synthetic({"attn.qkv": 6, "mlp.down": 6})
    shuffled = s3.shuffle_within_role(stats, seed=3)
    for name, layer in shuffled.layers.items():
        donor = next(
            other
            for other in stats.layers.values()
            if other.activation_rms_ema == layer.activation_rms_ema
        )
        assert donor.role == stats.layers[name].role


def test_a_singleton_role_is_a_fixed_point(s3, phi) -> None:
    """Phi's tied embedding and its head are alone in their roles, so they cannot move.

    Recorded as a test rather than a caveat because it is the one place the control is
    structurally weaker than it looks: those two tensors are 16% of the model and the
    permuted arm scores them identically to the treatment. A future model with more
    members in those roles will simply stop being a fixed point; a future *bug* that
    silently drops singletons from the permutation looks the same from the outside,
    and this pins which one is happening.
    """
    shuffled = s3.shuffle_within_role(phi, seed=0)
    singletons = [
        name
        for name, layer in phi.layers.items()
        if sum(1 for other in phi.layers.values() if other.role == layer.role) == 1
    ]
    assert set(singletons) == {"lm_head", "model.embed_tokens"}
    for name in singletons:
        assert all(
            getattr(shuffled.layers[name], f) == getattr(phi.layers[name], f)
            for f in s3.PERMUTED_FIELDS
        )


def test_the_permutation_is_reproducible_from_its_seed(s3, phi) -> None:
    """Two seeds give two controls; one seed twice gives one control.

    Without this the arm cannot be rebuilt from ``arms.json``, and a re-run that
    disagreed with the recorded number would be indistinguishable from a real effect.
    """
    assert s3.shuffle_within_role(phi, seed=1) == s3.shuffle_within_role(phi, seed=1)
    assert s3.shuffle_within_role(phi, seed=1) != s3.shuffle_within_role(phi, seed=2)


def test_an_identity_permutation_is_refused_rather_than_run(s3, tmp_path, monkeypatch) -> None:
    """A model whose roles are all singletons cannot have this control at all.

    Better to fail here than to spend the GPU hours and report a null result that is
    an artifact of the permutation having nothing to permute.
    """
    from dynquant.signals.schema import save_stats

    source = tmp_path / "dynquant_stats.json"
    save_stats(_synthetic({"embedding": 1, "lm_head": 1}), source)
    with pytest.raises(SystemExit, match="identity permutation"):
        s3.write_variants(source, tmp_path / "out", seed=0)


def test_the_untouched_variant_round_trips_byte_identical(s3, tmp_path) -> None:
    """Treatment and control must be read through the same loader.

    ``write_variants`` re-saves the real signal instead of pointing at the original, so
    that ``dq`` and ``shuf`` differ in the permutation and in nothing else -- not in
    which writer produced the file they were parsed from.
    """
    written = s3.write_variants(PHI_STATS, tmp_path / "out", seed=0)
    before, after = load_stats(PHI_STATS), load_stats(written["signal"]["path"])
    assert after.layers == before.layers


# --------------------------------------------------------------------------
# Matched bytes
# --------------------------------------------------------------------------


def _arm(s3, name: str, nbytes: int):
    return s3.Arm(
        name=name,
        anchor=3,
        kind=name,
        map_path=Path("map.json"),
        map_key="k",
        nbytes=nbytes,
        average_bits=3.0,
        violations=0,
    )


def test_arms_within_tolerance_are_accepted(s3) -> None:
    anchor = _arm(s3, "rtn", 1_000_000_000)
    s3.check_matched([anchor, _arm(s3, "dq", 999_500_000)], anchor)


def test_an_arm_that_bought_itself_bytes_is_refused(s3) -> None:
    """The failure this whole file exists for.

    A one-percent size advantage is larger than the accuracy differences S3 reports, so
    an arm that drifted would win on size and be written up as winning on method. Red
    if the tolerance is loosened, if the check stops looking at the widest arm, or if a
    future allocator starts overshooting a ``--target-size`` it is meant to treat as a
    ceiling.
    """
    anchor = _arm(s3, "rtn", 1_000_000_000)
    with pytest.raises(SystemExit, match="not byte-matched"):
        s3.check_matched([anchor, _arm(s3, "dq", 1_010_000_000)], anchor)


def test_the_check_reports_the_worst_arm_not_the_last(s3) -> None:
    """One bad arm among several must not be averaged away by the ones that matched."""
    anchor = _arm(s3, "rtn", 1_000_000_000)
    arms = [anchor, _arm(s3, "rank", 999_999_000), _arm(s3, "shuf", 900_000_000)]
    with pytest.raises(SystemExit, match="arm shuf is"):
        s3.check_matched(arms, anchor)


# --------------------------------------------------------------------------
# Arm identity
# --------------------------------------------------------------------------


def test_arm_labels_are_names_the_evaluator_will_accept(s3) -> None:
    """S1 files records as ``{name}.{task}.json`` and splits on the first dot.

    ``resolve_model`` rejects a dot for that reason, and it rejects it *after* this
    driver has quantized every arm. Checking the labels here costs nothing and catches
    a naming change that would otherwise strand a completed sweep.
    """
    for kind in s3.ARMS:
        for anchor in s3.ANCHORS:
            label = _arm(s3, kind, 1).label
            assert "." not in label and label
            assert replace(_arm(s3, kind, 1), anchor=anchor).label == f"{kind}{anchor}"


def test_every_arm_names_a_signal_variant_that_gets_written(s3) -> None:
    """An arm whose variant is never produced dies at allocation, mid-sweep."""
    produced = {"signal", "shuffled", None}
    assert all(spec["variant"] in produced for spec in s3.ARMS.values())
    assert s3.ARMS["rtn"]["variant"] is None, "the control arm allocates from nothing"
