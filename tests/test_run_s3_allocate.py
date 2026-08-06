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
  measure the distribution; one that was near-identity would measure nothing; and one
  that permuted only *some* of what the allocator reads would measure nothing while
  looking like it had measured everything -- the failure that actually happened, and
  that the moments tests below exist to keep from happening twice.

Nothing here launches a subprocess or loads a model. The permutation is pure data and
the byte check is arithmetic, so the whole surface that decides whether S3's numbers
mean anything is reachable from CPU CI.
"""

from __future__ import annotations

import argparse
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


@pytest.fixture(scope="module")
def phi_moments():
    from dynquant.signals.moments import load_moments

    return load_moments(PHI_STATS)


def _synthetic_moments(stats: StatsFile, widths: dict[str, tuple[int, int]]):
    """Channel vectors whose *values* identify their module, so a donor is traceable."""
    import torch

    from dynquant.signals.moments import ChannelMoments

    moments = ChannelMoments()
    for index, (name, layer) in enumerate(sorted(stats.layers.items()), start=1):
        rows, cols = widths[layer.role or ""]
        moments.input_sq[name] = torch.full((cols,), float(index))
        moments.output_grad_sq[name] = torch.full((rows,), float(index) * 1000.0)
        moments.observations[name] = index
    return moments


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
    shuffled = s3.shuffle_stats(phi, s3.permutation_within_role(phi, seed=0))

    def multiset(stats: StatsFile) -> list[tuple[float, ...]]:
        return sorted(
            tuple(getattr(layer, field) or 0.0 for field in s3.PERMUTED_FIELDS)
            for layer in stats.layers.values()
        )

    assert multiset(shuffled) == multiset(phi)


def test_the_control_actually_moves_measurements(s3, phi) -> None:
    """A control that leaves the map alone measures nothing and would read as a null."""
    shuffled = s3.shuffle_stats(phi, s3.permutation_within_role(phi, seed=0))
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
    shuffled = s3.shuffle_stats(phi, s3.permutation_within_role(phi, seed=0))
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
    shuffled = s3.shuffle_stats(stats, s3.permutation_within_role(stats, seed=3))
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
    shuffled = s3.shuffle_stats(phi, s3.permutation_within_role(phi, seed=0))
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
    assert s3.permutation_within_role(phi, seed=1) == s3.permutation_within_role(phi, seed=1)
    assert s3.permutation_within_role(phi, seed=1) != s3.permutation_within_role(phi, seed=2)


def test_an_identity_permutation_is_refused_rather_than_run(s3, tmp_path) -> None:
    """A model whose roles are all singletons cannot have this control at all.

    Better to fail here than to spend the GPU hours and report a null result that is
    an artifact of the permutation having nothing to permute.
    """
    from dynquant.signals.moments import save_moments
    from dynquant.signals.schema import save_stats

    stats = _synthetic({"embedding": 1, "lm_head": 1})
    source = tmp_path / "dynquant_stats.json"
    save_stats(stats, source)
    moments_path = tmp_path / "dynquant_moments.safetensors"
    save_moments(_synthetic_moments(stats, {"embedding": (8, 4), "lm_head": (4, 8)}), moments_path)

    with pytest.raises(SystemExit, match="identity permutation"):
        s3.write_variants(source, moments_path, tmp_path / "out", seed=0)


def test_the_untouched_variant_round_trips_byte_identical(s3, tmp_path) -> None:
    """Treatment and control must be read through the same loader.

    ``write_variants`` re-saves the real signal instead of pointing at the original, so
    that ``dq`` and ``shuf`` differ in the permutation and in nothing else -- not in
    which writer produced the file they were parsed from.
    """
    import torch

    from dynquant.signals.moments import load_moments

    written = s3.write_variants(PHI_STATS, PHI_STATS, tmp_path / "out", seed=0)
    before, after = load_stats(PHI_STATS), load_stats(written["signal"]["stats"])
    assert after.layers == before.layers

    m_before, m_after = load_moments(PHI_STATS), load_moments(written["signal"]["moments"])
    assert m_after.names == m_before.names
    assert m_after.observations == m_before.observations
    for name in m_before.names:
        assert torch.equal(m_after.input_sq[name], m_before.input_sq[name])
        assert torch.equal(m_after.output_grad_sq[name], m_before.output_grad_sq[name])


# --------------------------------------------------------------------------
# The control has to permute what the allocator actually reads
# --------------------------------------------------------------------------


def test_the_channel_moments_are_permuted_too(s3, phi, phi_moments) -> None:
    """The bug this section exists for: a control that ablated nothing.

    ``move_value`` prices a width change from the measured sensitivity table whenever
    the module has one, and only falls back to the stats-derived score when it does
    not. Phi's moments cover all 129 quantizable modules, so an arm that permuted the
    stats and passed the real moments allocated identically to the treatment -- and
    would have been written up as "the signal does not matter" on the strength of an
    ablation that never took place.

    Red if the moments stop being permuted, or are permuted for fewer modules than the
    stats are.
    """
    permutation = s3.permutation_within_role(phi, seed=0, moments=phi_moments)
    shuffled = s3.shuffle_moments(phi_moments, permutation)

    moved_stats = {t for t, d in permutation.items() if t != d}
    assert moved_stats, "the permutation is the identity; nothing below proves anything"

    with_moments = moved_stats & set(phi_moments.input_sq)
    assert len(with_moments) > 100, f"only {len(with_moments)} moved modules carry moments"
    for name in sorted(with_moments):
        donor = permutation[name]
        assert shuffled.input_sq[name] is phi_moments.input_sq[donor]
        assert shuffled.output_grad_sq[name] is phi_moments.output_grad_sq[donor]


def test_the_scalars_and_the_vectors_name_the_same_donor(s3, phi, phi_moments) -> None:
    """One permutation, both artifacts.

    Permuting the two independently would give a module one donor's activation scale
    and another's channel geometry -- the same "third distribution belonging to no
    module at all" that the multiset test rules out for the scalars, reintroduced
    across files where no single file's invariants can see it.
    """
    import torch

    permutation = s3.permutation_within_role(phi, seed=0, moments=phi_moments)
    stats = s3.shuffle_stats(phi, permutation)
    moments = s3.shuffle_moments(phi_moments, permutation)

    for name in sorted(set(phi.layers) & set(phi_moments.input_sq)):
        donor_by_scalar = next(
            other
            for other, layer in phi.layers.items()
            if layer.activation_rms_ema == stats.layers[name].activation_rms_ema
            and layer.grad_norm_var == stats.layers[name].grad_norm_var
        )
        assert torch.equal(moments.input_sq[name], phi_moments.input_sq[donor_by_scalar])


def test_the_moments_are_relabelled_and_not_rewritten(s3, phi, phi_moments) -> None:
    """Same vectors, different modules: the multiset property, for the channel data.

    A permutation that dropped a module's moments would change *which* modules the
    allocator can price at all, which is a structural difference and not a relabelling
    -- the control would then be ablating coverage as well as correspondence.
    """
    permutation = s3.permutation_within_role(phi, seed=0, moments=phi_moments)
    shuffled = s3.shuffle_moments(phi_moments, permutation)

    assert shuffled.names == phi_moments.names
    assert sorted(shuffled.complete_names()) == sorted(phi_moments.complete_names())
    for field_name in ("input_sq", "output_grad_sq"):
        before = getattr(phi_moments, field_name)
        after = getattr(shuffled, field_name)
        assert sorted(id(t) for t in after.values()) == sorted(id(t) for t in before.values())


def test_a_channel_vector_never_lands_on_a_module_of_another_shape(s3) -> None:
    """Length, not just role, decides who can donate to whom.

    A vector of the wrong length does not raise where it is consumed; it broadcasts,
    or it silently prices a subset of the channels. Grouping by ``(role, in, out)``
    makes that unrepresentable. On a dense model the refinement is a no-op, which is
    why the fixture deliberately gives one role two geometries.
    """
    stats = _synthetic({"attn.qkv": 8})
    names = sorted(stats.layers)
    moments = _synthetic_moments(stats, {"attn.qkv": (16, 32)})

    import torch

    for name in names[:4]:  # half the role gets a different geometry
        moments.input_sq[name] = torch.ones(64)

    permutation = s3.permutation_within_role(stats, seed=5, moments=moments)
    shuffled = s3.shuffle_moments(moments, permutation)
    for name in names:
        assert shuffled.input_sq[name].numel() == moments.input_sq[name].numel()
        assert shuffled.output_grad_sq[name].numel() == moments.output_grad_sq[name].numel()
        assert permutation[name] in (names[:4] if name in names[:4] else names[4:])


def test_both_variants_are_written_with_their_own_moments(s3, tmp_path) -> None:
    """``arms.json`` has to name four files, not three.

    The shuffled arm needs its own sidecar on disk; if ``write_variants`` emits only
    the permuted stats, the driver has nothing to hand ``--moments`` but the real file.
    """
    import torch

    from dynquant.signals.moments import load_moments

    written = s3.write_variants(PHI_STATS, PHI_STATS, tmp_path / "out", seed=0)
    assert set(written) == {"signal", "shuffled"}
    for variant in written.values():
        assert Path(variant["stats"]).is_file()
        assert Path(variant["moments"]).is_file()
    assert written["signal"]["moments"] != written["shuffled"]["moments"]

    real = load_moments(written["signal"]["moments"])
    control = load_moments(written["shuffled"]["moments"])
    differing = [
        name for name in real.names if not torch.equal(real.input_sq[name], control.input_sq[name])
    ]
    assert len(differing) > 100, f"only {len(differing)}/{len(real.names)} moments differ"


def test_each_arm_is_allocated_from_its_own_moments(s3, tmp_path, monkeypatch) -> None:
    """The end of the wire, checked at the argv the subprocess would receive.

    Everything above can be correct and the arm still be a null if ``allocate_arm``
    reaches past the variant for ``args.moments``. This is the line that was wrong, so
    this is the line the test pins.
    """
    import json

    variants = {
        "signal": {"stats": "s.json", "moments": "m.safetensors"},
        "shuffled": {"stats": "s.shuf.json", "moments": "m.shuf.safetensors"},
    }
    anchor = _arm(s3, "rtn", 1_000_000)
    args = argparse.Namespace(
        model="/some/merge",
        group_size=128,
        trust_remote_code=False,
        moments="/the/real/moments.safetensors",
    )

    seen: list[list[str]] = []

    def fake_run(cmd: list[str], *, what: str) -> None:
        seen.append(cmd)
        Path(cmd[cmd.index("--save-map") + 1]).write_text(
            json.dumps(
                {"maps": {"1000000": {"nbytes": 1_000_000, "average_bits": 3.0, "violations": []}}}
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(s3, "_run", fake_run)

    for kind, spec in s3.ARMS.items():
        if spec["variant"] is None:
            continue
        seen.clear()
        s3.allocate_arm(args, tmp_path, kind=kind, anchor=anchor, variants=variants)
        (cmd,) = seen
        assert variants[spec["variant"]]["stats"] in cmd
        if spec["moments"]:
            assert cmd[cmd.index("--moments") + 1] == variants[spec["variant"]]["moments"]
            assert args.moments not in cmd, f"{kind} was allocated from the unpermuted moments"
        else:
            assert "--moments" not in cmd


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


# --------------------------------------------------------------------------
# The commands the driver builds
# --------------------------------------------------------------------------


def test_the_anchor_command_asks_for_every_anchor_width(s3, tmp_path) -> None:
    """Parsed by the CLI's own parser, because that is what decides what gets written.

    ``--uniform`` is ``nargs="+"``: one flag per width silently keeps only the last, so
    the driver asked for two anchors and got one, and the run continued until an arm
    reached for the anchor that was never allocated. Checking the argv against the real
    parser rather than against a hardcoded list means a future change to the flag's
    arity turns this red instead of turning a sweep red.
    """
    from dynquant.cli import build_parser

    args = argparse.Namespace(model="/some/merge", group_size=128, trust_remote_code=False)
    cmd = s3.anchor_cmd(args, tmp_path / "map.rtn.json")

    assert cmd[1:4] == ["-m", "dynquant", "inspect"], cmd
    parsed = build_parser().parse_args(cmd[3:])
    assert parsed.uniform == list(s3.ANCHORS)
    assert parsed.stats is None, "the RTN anchor must not be allocated from a signal"
    assert parsed.json and parsed.save_map


def test_the_anchor_widths_are_widths_the_cli_will_accept(s3) -> None:
    """``--uniform`` has ``choices``, so an anchor off the packing grid is an argparse
    error two minutes into a model load rather than a wrong number."""
    from dynquant.constants import BIT_OPTIONS

    assert set(s3.ANCHORS) <= set(BIT_OPTIONS)
