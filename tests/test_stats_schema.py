"""Stats schema: exact accumulator merging and lossless v1 migration.

The merge tests are the load-bearing ones. DDP reduction and "collapse LoRA into
base" are both the same operation -- combining Welford accumulators over disjoint
observation sets -- and both must be *exact*, not approximate, or the plasticity
signal drifts with world size and the allocation becomes irreproducible.
"""

from __future__ import annotations

import math
import random
import re

import pytest

from dynquant.constants import STATS_FILENAME, STATS_SCHEMA
from dynquant.errors import FormatVersionError, StatsCoverageError
from dynquant.graph.naming import canonical_name
from dynquant.signals.schema import LayerStats, StatsFile, load_stats

from _oracle import shipped_stats_params  # type: ignore[import-not-found]

# --------------------------------------------------------------------------
# Welford / Chan
# --------------------------------------------------------------------------


def _single_stream_welford(xs):
    """Reference: the algorithm the supplement's tracker runs, sample variance."""
    n, mean, m2 = 0, 0.0, 0.0
    for x in xs:
        n += 1
        delta = x - mean
        mean += delta / n
        m2 += delta * (x - mean)
    return n, mean, (m2 / (n - 1) if n >= 2 else 0.0)


def _stats_from(xs, name="w"):
    n, mean, var = _single_stream_welford(xs)
    return LayerStats(name=name, grad_norm_count=n, grad_norm_mean=mean, grad_norm_var=var)


@pytest.fixture(scope="module")
def sample():
    rng = random.Random(7)
    # log-normal, because gradient norms span orders of magnitude and that is
    # where a naive sum-of-squares variance would lose precision
    return [rng.lognormvariate(-3.0, 1.5) for _ in range(500)]


@pytest.mark.parametrize("split", [1, 2, 3, 137, 250, 497, 499])
def test_chan_merge_equals_single_stream(sample, split):
    ref = _stats_from(sample)
    merged = _stats_from(sample[:split]).merged_with(_stats_from(sample[split:]))
    assert merged.grad_norm_count == ref.grad_norm_count
    assert merged.grad_norm_mean == pytest.approx(ref.grad_norm_mean, rel=1e-12)
    assert merged.grad_norm_var == pytest.approx(ref.grad_norm_var, rel=1e-9)


def test_chan_merge_is_associative(sample):
    """Reduction order must not matter -- DDP does not promise one."""
    a, b, c = (_stats_from(chunk) for chunk in (sample[:100], sample[100:300], sample[300:]))
    left = a.merged_with(b).merged_with(c)
    right = a.merged_with(b.merged_with(c))
    assert left.grad_norm_var == pytest.approx(right.grad_norm_var, rel=1e-9)
    assert left.grad_norm_var == pytest.approx(_stats_from(sample).grad_norm_var, rel=1e-9)


def test_m2_recovers_from_stored_sample_variance(sample):
    """The whole merge rests on ``m2 = var * (n - 1)`` being recoverable."""
    n, _, var = _single_stream_welford(sample)
    stats = _stats_from(sample)
    assert stats.m2 == pytest.approx(var * (n - 1), rel=1e-12)


def test_count_below_two_means_no_signal():
    """The trap: ``var == 0.0`` is emitted both for "never moved" and "never
    measured", and the scorer reads low variance as unimportant."""
    assert not LayerStats(name="w", grad_norm_count=0).has_gradient_signal
    one = LayerStats(name="w", grad_norm_count=1, grad_norm_mean=5.0, grad_norm_var=0.0)
    assert not one.has_gradient_signal
    assert one.m2 == 0.0
    two = one.merged_with(LayerStats(name="w", grad_norm_count=1, grad_norm_mean=7.0))
    assert two.has_gradient_signal
    assert two.grad_norm_var == pytest.approx(2.0)  # sample var of {5, 7}


def test_merging_empty_accumulators_is_safe():
    empty = LayerStats(name="w")
    assert empty.merged_with(empty).grad_norm_count == 0
    assert empty.merged_with(empty).grad_norm_var == 0.0


def test_merge_preserves_optional_metadata():
    a = LayerStats(name="w", grad_norm_count=3, param_count=100, routing_hits=7)
    b = LayerStats(name="w", grad_norm_count=3, role="moe_expert_up", routing_hits=5)
    m = a.merged_with(b)
    assert m.param_count == 100
    assert m.role == "moe_expert_up"
    assert m.routing_hits == 12, "routing hits are counts and must add"


def test_plasticity_is_log1p_of_variance():
    stats = LayerStats(name="w", grad_norm_count=10, grad_norm_var=3.0)
    assert stats.plasticity_raw == pytest.approx(math.log1p(3.0))
    # negative variance can only come from a corrupt file; must not produce NaN
    assert LayerStats(name="w", grad_norm_var=-1.0).plasticity_raw == 0.0


# --------------------------------------------------------------------------
# Migration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", shipped_stats_params())
def test_shipped_v1_migrates_losslessly(path, v1_stats_raw):
    raw = v1_stats_raw(path)
    v1_layers = raw["layers"]
    stats = load_stats(path)

    groups: dict[str, list[str]] = {}
    for key in v1_layers:
        groups.setdefault(canonical_name(key), []).append(key)

    assert set(groups) == set(stats), "canonical key set must match exactly"

    # total observations conserved -- the single strongest losslessness signal
    assert sum(s.grad_norm_count for s in stats.layers.values()) == sum(
        int(v.get("grad_norm_count", 0) or 0) for v in v1_layers.values()
    )

    for dest, sources in groups.items():
        acc = None
        for src in sources:
            record = LayerStats.from_dict(dest, v1_layers[src])
            acc = record if acc is None else acc.merged_with(record)
        assert acc is not None
        got = stats[dest]
        assert got.grad_norm_count == acc.grad_norm_count
        assert got.grad_norm_var == pytest.approx(acc.grad_norm_var, rel=1e-9, abs=1e-30)
        assert got.grad_norm_mean == pytest.approx(acc.grad_norm_mean, rel=1e-9, abs=1e-30)


@pytest.mark.parametrize("path", shipped_stats_params())
def test_shipped_v1_keeps_hyperparameters_and_notes(path, v1_stats_raw):
    raw = v1_stats_raw(path)
    stats = load_stats(path)
    assert stats.activation_ema_beta == raw["activation_ema_beta"]
    assert stats.coherence_ema_beta == raw["coherence_ema_beta"]
    assert stats.provenance.migrated_from == raw["schema"]
    # top-level keys with no v2 home survive rather than being dropped
    assert "collapsed_lora_into_base" in stats.provenance.notes


@pytest.mark.parametrize("path", shipped_stats_params())
def test_canonicalisation_is_idempotent(path):
    once = load_stats(path)
    twice = once.canonicalized()
    assert set(once) == set(twice)
    for key in once:
        assert once[key].to_dict() == twice[key].to_dict()


@pytest.mark.parametrize("path", shipped_stats_params())
def test_roundtrip_through_disk(path, tmp_path):
    stats = load_stats(path)
    written = stats.save(tmp_path / "dynquant_stats.json")
    back = load_stats(written)
    assert set(back) == set(stats)
    for key in stats:
        assert back[key].to_dict() == stats[key].to_dict()
    assert back.provenance.notes == stats.provenance.notes
    assert back.provenance.migrated_from == stats.provenance.migrated_from


@pytest.mark.parametrize("path", shipped_stats_params())
def test_migrated_keys_carry_no_wrapper_debris(path):
    """The point of write-time canonicalisation."""
    stats = load_stats(path)
    for key in stats:
        assert not key.startswith("base_model."), key
        assert ".base_layer" not in key, key
        assert "lora_" not in key, key
        assert not key.endswith(".weight"), key


def test_v2_declares_the_current_schema(tmp_path):
    stats = StatsFile(layers={"model.x": LayerStats(name="model.x", grad_norm_count=4)})
    payload = stats.to_dict()
    assert payload["schema"] == STATS_SCHEMA
    assert "provenance" in payload and "hyperparameters" in payload


def test_load_finds_legacy_filenames_in_a_directory(tmp_path):
    import json

    (tmp_path / "unified_gasq_stats_collapsed.json").write_text(
        json.dumps(
            {
                "schema": "unified_gasq_stats_v1",
                "activation_ema_beta": 0.99,
                "coherence_ema_beta": 0.95,
                "layers": {
                    "base_model.model.model.layers.0.mlp.up_proj.base_layer": {
                        "activation_rms_ema": 1.5,
                        "grad_norm_count": 10,
                        "grad_norm_mean": 0.1,
                        "grad_norm_var": 0.01,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    stats = load_stats(tmp_path)
    assert list(stats) == ["model.layers.0.mlp.up_proj"]


def test_untagged_but_layer_shaped_file_is_read_as_v1():
    stats = StatsFile.from_dict({"layers": {"model.x": {"grad_norm_count": 3}}})
    assert len(stats) == 1


def test_future_schema_is_rejected_with_an_actionable_message():
    with pytest.raises(FormatVersionError, match="upgrade"):
        StatsFile.from_dict({"schema": "dynquant_stats_v99", "layers": {}}, path="s.json")


def test_missing_stats_file_names_what_it_looked_for(tmp_path):
    with pytest.raises(FileNotFoundError, match=re.escape(STATS_FILENAME)):
        load_stats(tmp_path)


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def test_coverage_separates_missing_from_unusable():
    stats = StatsFile(
        layers={
            "a": LayerStats(name="a", grad_norm_count=10, grad_norm_var=0.5),
            "b": LayerStats(name="b", grad_norm_count=0),  # hooked but never observed
        }
    )
    report = stats.coverage(["a", "b", "c"])
    assert report.matched == ("a", "b")
    assert report.missing_signal == ("c",)
    assert report.partial_signal == ("b",)
    assert report.fraction == pytest.approx(2 / 3)
    assert report.usable_fraction == pytest.approx(1 / 3)


def test_coverage_normalises_model_side_names_too():
    """A caller passing wrapped module names must still match."""
    stats = StatsFile(layers={"model.layers.0.mlp.up_proj": LayerStats(name="x")})
    report = stats.coverage(["base_model.model.model.layers.0.mlp.up_proj.base_layer"])
    assert len(report.matched) == 1


def test_coverage_reports_wrong_model_pairings():
    stats = StatsFile(layers={"decoder.block.0.wi": LayerStats(name="decoder.block.0.wi")})
    report = stats.coverage(["model.layers.0.mlp.up_proj"])
    assert report.matched == ()
    assert report.unmatched_stats == ("decoder.block.0.wi",)


def test_insufficient_coverage_raises_with_examples():
    stats = StatsFile(layers={"a": LayerStats(name="a", grad_norm_count=5)})
    report = stats.coverage([f"m{i}" for i in range(20)] + ["a"])
    with pytest.raises(StatsCoverageError) as excinfo:
        report.raise_if_insufficient(0.95)
    message = str(excinfo.value)
    assert "min-coverage" in message, "must tell the user how to override"
    assert "m0" in message, "must name uncovered modules"


def test_empty_model_is_trivially_covered():
    assert StatsFile().coverage([]).fraction == 1.0


# --------------------------------------------------------------------------
# Reduction
# --------------------------------------------------------------------------


def test_ddp_reduction_matches_single_process(sample):
    rank0 = StatsFile(layers={"a": _stats_from(sample[:250], "a")})
    rank1 = StatsFile(
        layers={"a": _stats_from(sample[250:], "a"), "b": _stats_from(sample[:10], "b")}
    )
    reduced = rank0.merged_with(rank1)
    reference = _stats_from(sample)
    assert reduced["a"].grad_norm_var == pytest.approx(reference.grad_norm_var, rel=1e-9)
    assert "b" in reduced, "keys present on only one rank must survive"
