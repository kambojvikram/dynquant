"""Channel moments: the second-moment sidecar the cardinal estimator runs on.

Two things are being guarded here, and only one of them is serialisation.

The first is that the moments *mean what they say*. ``input_sq[name]`` must be
``E[x_c^2]`` over the module's input channels and ``output_grad_sq[name]`` must be
``E[delta_r^2]`` over its output channels -- indexed that way round, in that
orientation. A transposed axis broadcasts silently on a square weight and the
resulting sensitivity table is wrong everywhere without raising anywhere, which is
how the first attempt at this measurement produced numbers for two thirds of a model
before anything complained. So the accumulation is checked against an explicit
per-channel mean computed the slow way.

The second is that the sidecar is only ever written from a *read* of the running
accumulators. The tracker flushes on ``log_every``, and an implementation that
drained its accumulators while writing would silently halve the denominator of every
subsequent moment.
"""

from __future__ import annotations

import pytest
import torch
from test_signals_tracker import TinyVlm, _train_steps

from dynquant.constants import MOMENTS_FILENAME
from dynquant.signals import SignalTracker, TrackerConfig
from dynquant.signals.moments import ChannelMoments, load_moments, save_moments


def _tracker(model: torch.nn.Module, every: int = 1) -> SignalTracker:
    config = TrackerConfig(collect_channel_moments=True, channel_moment_every=every)
    return SignalTracker(model, config=config).attach()


class _Solo(torch.nn.Module):
    """One Linear with a parent, because the tracker discovers by walking children."""

    def __init__(self, fan_in: int = 6, fan_out: int = 4) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(fan_in, fan_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# --------------------------------------------------------------------------
# What the numbers are
# --------------------------------------------------------------------------


def test_input_moment_is_the_per_input_channel_mean_square() -> None:
    """``E[x_c^2]``, indexed by input channel, averaged over every token seen."""
    torch.manual_seed(0)
    model = _Solo()
    tracker = _tracker(model)

    batches = [torch.randn(3, 6), torch.randn(5, 6)]
    for x in batches:
        model(x).square().sum().backward()
    tracker.on_optimizer_step()

    moments = tracker.channel_moments(reduce=False)
    stacked = torch.cat(batches)
    torch.testing.assert_close(
        moments.input_sq["proj"], stacked.pow(2).mean(0), rtol=1e-5, atol=1e-6
    )
    assert moments.input_sq["proj"].shape == (model.proj.in_features,)


def test_output_moment_is_the_per_output_channel_mean_square_of_the_grad() -> None:
    """``E[delta_r^2]``, indexed by *output* channel.

    The half that carries the signal. Dropping it -- keeping only the input Gram,
    which is the shape a calibration-free method can compute -- turns a +0.52 mean
    within-role rank correlation against measured loss disturbance into -0.34, so an
    axis error here is not a rounding issue, it is the difference between the
    estimator working and pointing backwards.
    """
    torch.manual_seed(0)
    model = _Solo()
    tracker = _tracker(model)

    x = torch.randn(7, 6)
    y = model(x)
    captured: dict[str, torch.Tensor] = {}

    def capture(grad: torch.Tensor) -> None:
        captured["delta"] = grad

    y.register_hook(capture)
    (y * torch.randn_like(y)).sum().backward()
    tracker.on_optimizer_step()

    moments = tracker.channel_moments(reduce=False)
    torch.testing.assert_close(
        moments.output_grad_sq["proj"], captured["delta"].pow(2).mean(0), rtol=1e-5, atol=1e-6
    )
    assert moments.output_grad_sq["proj"].shape == (model.proj.out_features,)


def test_embeddings_are_skipped_rather_than_given_a_transposed_moment() -> None:
    """An embedding's input is a token id, so the pairing its forward produces is
    transposed relative to how ``[num_embeddings, dim]`` is stored.

    Storing it under the same field names would be a number of the right shape and
    the wrong meaning -- worse than an absence, because an absence is reported.
    """
    model = TinyVlm(tie=False)
    tracker = _tracker(model)
    _train_steps(tracker, model, steps=2)

    moments = tracker.channel_moments(reduce=False)
    assert "model.embed_tokens" not in moments
    assert "lm_head" in moments.complete_names()


def test_every_moment_matches_its_weight_orientation() -> None:
    """The invariant :mod:`dynquant.score.sensitivity` refuses to proceed without."""
    model = TinyVlm(tie=False)
    tracker = _tracker(model)
    _train_steps(tracker, model, steps=2)

    moments = tracker.channel_moments(reduce=False)
    assert moments.complete_names()
    for name in moments.complete_names():
        weight = model.get_submodule(name).weight
        assert moments.input_sq[name].shape[0] == weight.shape[1], name
        assert moments.output_grad_sq[name].shape[0] == weight.shape[0], name


# --------------------------------------------------------------------------
# Collection policy
# --------------------------------------------------------------------------


def test_sampling_every_n_steps_collects_on_a_subset_of_steps() -> None:
    """Moments are per-channel means, so they converge in far fewer steps than the
    signal they support -- hence sampling, which is what keeps the tracker's measured
    +2.3% overhead from drifting up against a 3% gate."""
    model = TinyVlm(tie=False)
    dense = _tracker(model)
    _train_steps(dense, model, steps=4)

    sparse = _tracker(model, every=4)
    _train_steps(sparse, model, steps=4)

    name = sparse.channel_moments(reduce=False).complete_names()[0]
    assert (
        dense.channel_moments(reduce=False).observations[name]
        > (sparse.channel_moments(reduce=False).observations[name])
    )


def test_collection_can_be_turned_off_entirely() -> None:
    model = TinyVlm(tie=False)
    tracker = SignalTracker(model, config=TrackerConfig(collect_channel_moments=False)).attach()
    _train_steps(tracker, model, steps=2)
    assert len(tracker.channel_moments(reduce=False)) == 0


def test_reading_the_moments_does_not_drain_them() -> None:
    """``save()`` reads; it must not consume.

    With ``log_every`` set, the tracker writes several times in one run. An
    implementation that reduced in place would leave the second write covering half
    the observations of the first, with nothing to indicate it.
    """
    model = TinyVlm(tie=False)
    tracker = _tracker(model)
    _train_steps(tracker, model, steps=3)

    first = tracker.channel_moments(reduce=False)
    second = tracker.channel_moments(reduce=False)
    name = first.complete_names()[0]
    assert first.observations[name] == second.observations[name]
    torch.testing.assert_close(first.input_sq[name], second.input_sq[name])


def test_a_rejected_step_count_is_rejected_loudly() -> None:
    with pytest.raises(ValueError, match="channel_moment_every"):
        TrackerConfig(channel_moment_every=0).validate()


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def test_round_trip_preserves_tensors_and_observation_counts(tmp_path) -> None:
    moments = ChannelMoments(
        input_sq={"model.layers.0.mlp.up_proj": torch.rand(8)},
        output_grad_sq={"model.layers.0.mlp.up_proj": torch.rand(4)},
        observations={"model.layers.0.mlp.up_proj": 37},
    )
    written = save_moments(moments, tmp_path)
    assert written.name == MOMENTS_FILENAME

    back = load_moments(tmp_path)
    assert back.names == moments.names
    assert back.observations == moments.observations
    for name in moments.names:
        torch.testing.assert_close(back.input_sq[name], moments.input_sq[name])
        torch.testing.assert_close(back.output_grad_sq[name], moments.output_grad_sq[name])


def test_load_accepts_the_file_as_well_as_its_directory(tmp_path) -> None:
    written = save_moments(ChannelMoments(input_sq={"a": torch.ones(3)}), tmp_path)
    assert load_moments(written).names == ("a",)


def test_a_missing_sidecar_says_why_it_might_be_missing(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="collect_channel_moments"):
        load_moments(tmp_path)


def test_a_half_collected_module_is_reported_as_incomplete() -> None:
    """Input without output is unusable by the estimator and must not look usable."""
    moments = ChannelMoments(input_sq={"a": torch.ones(3)}, output_grad_sq={"b": torch.ones(2)})
    assert set(moments.names) == {"a", "b"}
    assert moments.complete_names() == ()
    assert "input only: 1" in moments.summary()


def test_the_tracker_writes_a_sidecar_next_to_the_stats_file(tmp_path) -> None:
    model = TinyVlm(tie=False)
    tracker = _tracker(model)
    _train_steps(tracker, model, steps=2)

    stats_path = tracker.save(tmp_path, reduce=False)
    assert stats_path is not None
    sidecar = stats_path.parent / MOMENTS_FILENAME
    assert sidecar.is_file()
    assert load_moments(sidecar).complete_names()


def test_no_sidecar_is_written_when_collection_is_off(tmp_path) -> None:
    model = TinyVlm(tie=False)
    tracker = SignalTracker(model, config=TrackerConfig(collect_channel_moments=False)).attach()
    _train_steps(tracker, model, steps=2)

    stats_path = tracker.save(tmp_path, reduce=False)
    assert stats_path is not None
    assert not (stats_path.parent / MOMENTS_FILENAME).exists()
