"""Phase-2 signal collection: exactness of the estimators, and the bug guards.

Three groups of tests, matching the three claims made in
:mod:`dynquant.signals.estimators` and :mod:`dynquant.signals.tracker`:

1. **Exactness.** The Gram identity, the embedding one-hot variant and the
   low-rank expansion are each asserted against a gradient that ``autograd``
   computed the direct way. Any of them being merely "approximately right" would
   make the plasticity ranking approximately meaningless, so they are checked to
   floating-point tolerance rather than to a correlation.

2. **Bug guards**, one test per numbered finding in ``docs/legacy-audit.md``.
   These are the tests that would have failed against the research code.

3. **Plumbing** -- naming, coverage, ties, serialisation, error accounting.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import pytest
import torch
import torch.utils.checkpoint  # not re-exported by `import torch`; see _CheckpointedVlm
from torch import Tensor, nn

from dynquant.errors import SignalCollectionError
from dynquant.signals import SignalTracker, TrackerConfig, load_stats
from dynquant.signals.estimators import (
    GradEstimatorMode,
    channel_norm,
    embedding_gram,
    gram,
    lowrank_grad_sq,
    outer_grad_sq_batched,
    outer_grad_sq_from_gram,
    stride_subsample,
)

LIMIT = 4096  # far above any token count here, so nothing is subsampled away


# --------------------------------------------------------------------------
# Fixtures: a tiny model with the structural features that break allocators
# --------------------------------------------------------------------------


class _Attn(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.o_proj(torch.tanh(self.q_proj(x)))


class _Mlp(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d, 2 * d, bias=False)
        self.down_proj = nn.Linear(2 * d, d, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)))


class _Layer(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.self_attn = _Attn(d)
        self.mlp = _Mlp(d)
        self.input_layernorm = nn.LayerNorm(d)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.mlp(self.self_attn(self.input_layernorm(x)))


class _Inner(nn.Module):
    def __init__(self, vocab: int, d: int, layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(_Layer(d) for _ in range(layers))

    def forward(self, ids: Tensor) -> Tensor:
        h = self.embed_tokens(ids)
        for layer in self.layers:
            h = layer(h)
        return h


class _VisionTower(nn.Module):
    """Never called by a text-only forward. That is the entire point of it.

    Stands in for the 24-layer ViT in Qwen3.5-2B -- about 15% of the parameters,
    structurally unreachable from a GSM8K batch.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, d, kernel_size=2)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, pixels: Tensor) -> Tensor:
        return self.proj(self.patch_embed(pixels).flatten(2).transpose(1, 2))


class TinyVlm(nn.Module):
    """Text tower + unused vision tower + tied LM head."""

    def __init__(self, vocab: int = 16, d: int = 8, layers: int = 2, *, tie: bool = True) -> None:
        super().__init__()
        self.model = _Inner(vocab, d, layers)
        self.vision_tower = _VisionTower(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        if tie:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, ids: Tensor) -> Tensor:
        return self.lm_head(self.model(ids))


def _capture_output_grad(tensor: Tensor) -> dict[str, Tensor]:
    """Grab ``dL/dY`` for ``tensor``, the way the tracker's own backward hook does.

    Returns a dict filled in during ``backward()``. The hook returns ``None``
    explicitly -- a ``register_hook`` callback that returns a value *replaces* the
    gradient flowing onward, which would corrupt the very autograd result the test
    compares against.
    """
    captured: dict[str, Tensor] = {}

    def hook(grad: Tensor) -> None:
        captured["delta"] = grad

    tensor.register_hook(hook)
    return captured


def _train_steps(
    tracker: SignalTracker,
    model: TinyVlm,
    *,
    steps: int,
    micro_batches: int = 1,
    vocab: int = 16,
) -> None:
    """Run a loop with explicit micro-batch and optimizer-step boundaries."""
    generator = torch.Generator().manual_seed(0)
    for _ in range(steps):
        for _ in range(micro_batches):
            ids = torch.randint(0, vocab, (2, 5), generator=generator)
            logits = model(ids)
            logits.square().mean().backward()
        tracker.on_optimizer_step()
        model.zero_grad(set_to_none=True)


# --------------------------------------------------------------------------
# 1. Exactness of the estimators
# --------------------------------------------------------------------------


def test_gram_identity_is_exact_for_a_linear() -> None:
    """``<dd^T, xx^T> == ||grad W||_F^2``, checked against autograd.

    The claim that lets the tracker skip forming a ``[out, in]`` matrix per module
    per step. If it were only approximate, the whole cost argument would collapse
    back to the naive implementation.
    """
    torch.manual_seed(0)
    layer = nn.Linear(6, 4, bias=False)
    x = torch.randn(5, 6)
    y = layer(x)

    captured = _capture_output_grad(y)
    (y * torch.randn_like(y)).sum().backward()

    assert layer.weight.grad is not None
    truth = layer.weight.grad.pow(2).sum() / x.shape[0]
    estimate = outer_grad_sq_from_gram(gram(x, LIMIT), captured["delta"], LIMIT)
    torch.testing.assert_close(estimate, truth, rtol=1e-5, atol=1e-6)


def test_gram_identity_holds_through_extra_token_dimensions() -> None:
    """Real activations arrive as ``[batch, seq, d]``, not ``[T, d]``."""
    torch.manual_seed(1)
    layer = nn.Linear(7, 3, bias=False)
    x = torch.randn(2, 5, 7)
    y = layer(x)

    captured = _capture_output_grad(y)
    (y * torch.randn_like(y)).sum().backward()

    assert layer.weight.grad is not None
    truth = layer.weight.grad.pow(2).sum() / (2 * 5)
    estimate = outer_grad_sq_from_gram(gram(x, LIMIT), captured["delta"], LIMIT)
    torch.testing.assert_close(estimate, truth, rtol=1e-5, atol=1e-6)


def test_embedding_gram_recovers_the_true_embedding_gradient() -> None:
    """Repeated tokens contribute cross terms; the one-hot Gram keeps them.

    ``indices = [1, 3, 1, 7]`` sends two different output gradients to row 1, and
    the true squared norm of that row is ``||d_0 + d_2||^2``, not
    ``||d_0||^2 + ||d_2||^2``. A per-token approximation drops the cross term; the
    token-equality Gram matrix reproduces it exactly.
    """
    torch.manual_seed(2)
    emb = nn.Embedding(10, 4)
    indices = torch.tensor([1, 3, 1, 7])
    y = emb(indices)

    captured = _capture_output_grad(y)
    (y * torch.randn_like(y)).sum().backward()

    assert emb.weight.grad is not None
    truth = emb.weight.grad.pow(2).sum() / indices.numel()
    estimate = outer_grad_sq_from_gram(embedding_gram(indices, LIMIT), captured["delta"], LIMIT)
    torch.testing.assert_close(estimate, truth, rtol=1e-5, atol=1e-6)


def test_batched_contraction_equals_the_scalar_one_module_by_module() -> None:
    """The batched form is an optimisation, so it is pinned to the form it replaced.

    ``outer_grad_sq_from_gram`` is the version the three exactness tests above check
    against autograd. Pinning the batched path to it here is what transfers that
    guarantee: the tracker calls only the batched one, and this is the link in the
    chain that says they agree.
    """
    torch.manual_seed(11)
    widths = [3, 8, 1, 5]  # including 1, where a stacked sum could silently broadcast
    tokens = 6
    grams_x = [gram(torch.randn(tokens, w), LIMIT) for w in widths]
    deltas = [torch.randn(tokens, w) for w in widths]

    batched = outer_grad_sq_batched(grams_x, [gram(d, LIMIT) for d in deltas])
    scalar = torch.stack(
        [outer_grad_sq_from_gram(gx, d, LIMIT) for gx, d in zip(grams_x, deltas, strict=True)]
    )
    torch.testing.assert_close(batched, scalar, rtol=1e-6, atol=1e-7)


def test_a_module_with_its_own_token_count_still_gets_its_own_norm() -> None:
    """One step, two Gram shapes -- the MoE case the flush groups for.

    An expert sees only the tokens routed to it, so ``T`` differs between modules
    within a single backward pass. If the flush stacked regardless it would raise;
    if it grouped by the wrong key it would attribute one expert's norm to another.
    Two modules with deliberately different token counts, driven through the real
    hook, catch both.
    """
    torch.manual_seed(12)
    model = nn.ModuleDict(
        {"wide": nn.Linear(4, 3, bias=False), "narrow": nn.Linear(4, 3, bias=False)}
    )
    tracker = SignalTracker(model, TrackerConfig(grad_estimator="outer_exact")).attach()
    try:
        wide, narrow = torch.randn(9, 4), torch.randn(2, 4)
        (model["wide"](wide).square().sum() + model["narrow"](narrow).square().sum()).backward()
        tracker._flush_backward()

        slots = {e.name: e.slot for e in tracker._tracked}
        for name, x in (("wide", wide), ("narrow", narrow)):
            weight = model[name].weight
            assert weight.grad is not None
            truth = (weight.grad.pow(2).sum() / x.shape[0]).sqrt()
            got = tracker._grad_stage[slots[name]]
            torch.testing.assert_close(got, truth, rtol=1e-5, atol=1e-6)
    finally:
        tracker.detach()


def test_modules_sharing_an_input_each_get_their_own_gradient_norm() -> None:
    """The shared input Gram is an optimisation, so each module's answer is checked.

    ``q_proj`` and ``o_proj`` here read the *same* tensor object, which is what lets
    the one-entry cache skip the second Gram. If sharing leaked -- one module's Gram
    used for the other's contraction, or one norm overwriting the other -- the values
    would still be finite and plausible, so both are compared against autograd rather
    than against each other.
    """
    torch.manual_seed(14)

    class _Twins(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(4, 3, bias=False)
            self.o_proj = nn.Linear(4, 6, bias=False)

        def forward(self, x: Tensor) -> Tensor:
            # One tensor, two modules: the sharing the cache is built for.
            return self.q_proj(x).sum() + self.o_proj(x).sum()

    model = _Twins()
    tracker = SignalTracker(model, TrackerConfig(grad_estimator="outer_exact")).attach()
    try:
        x = torch.randn(7, 4)
        model(x).backward()
        tracker._flush_backward()

        slots = {e.name: e.slot for e in tracker._tracked}
        assert tracker._last_gram is not None  # the cache was populated, so it was exercised
        for name in ("q_proj", "o_proj"):
            weight = getattr(model, name).weight
            assert weight.grad is not None
            truth = (weight.grad.pow(2).sum() / x.shape[0]).sqrt()
            torch.testing.assert_close(
                tracker._grad_stage[slots[name]], truth, rtol=1e-5, atol=1e-6
            )
    finally:
        tracker.detach()


def test_capping_the_gram_queue_does_not_change_the_answer(monkeypatch) -> None:
    """Splitting one step into several flushes must be invisible in the result.

    The cap exists so that deferring the contraction cannot hold gigabytes of Gram
    matrices on a model with thousands of experts. It is only safe because the fold is
    an ``index_add_``: a sum does not care how the terms were grouped. Forcing a cap
    small enough to flush after every single module is the strongest available check
    that the grouping really is free.
    """
    torch.manual_seed(13)
    ids = torch.randint(0, 16, (2, 5))

    def staged(cap: int) -> Tensor:
        monkeypatch.setattr("dynquant.signals.tracker._CONTRACTION_QUEUE_BYTES", cap)
        torch.manual_seed(13)
        model = TinyVlm()
        tracker = SignalTracker(model, TrackerConfig(grad_estimator="outer_exact")).attach()
        try:
            model(ids).square().mean().backward()
            tracker._flush_backward()
            return tracker._grad_stage.clone()
        finally:
            tracker.detach()

    one_flush = staged(1 << 30)
    many_flushes = staged(1)
    assert one_flush.count_nonzero() > 0  # a silent no-op would pass the comparison
    torch.testing.assert_close(many_flushes, one_flush, rtol=1e-6, atol=1e-7)


def test_lowrank_expansion_matches_the_explicit_product() -> None:
    """``||grad_B A + B grad_A||^2`` from r-by-r traces only."""
    torch.manual_seed(3)
    out_features, in_features, rank, scaling = 6, 5, 2, 0.5
    a, b = torch.randn(rank, in_features), torch.randn(out_features, rank)
    grad_a, grad_b = torch.randn(rank, in_features), torch.randn(out_features, rank)

    explicit = (scaling * (grad_b @ a + b @ grad_a)).pow(2).sum()
    torch.testing.assert_close(
        lowrank_grad_sq(a, b, grad_a, grad_b, scaling), explicit, rtol=1e-5, atol=1e-6
    )


def test_channel_norm_is_always_the_output_width() -> None:
    """Bug 7's shape mismatch, made impossible by construction.

    The research code's coherence buffer alternated between length ``r`` and length
    ``out`` depending on which adapter factor fired, and every comparison between
    consecutive entries raised inside a bare ``except``. Keying the vector to
    ``d_out`` -- a property of the module, not of the adapter -- means consecutive
    observations are always comparable, whatever the rank.
    """
    for shape in [(5, 3), (2, 5, 3), (1, 1, 7, 3)]:
        assert channel_norm(torch.randn(*shape)).shape == (3,)


def test_stride_subsample_is_evenly_spaced_and_deterministic() -> None:
    t = torch.arange(100).unsqueeze(1).float()
    picked = stride_subsample(t, 10)
    assert picked.shape[0] == 10
    assert picked.squeeze(1).tolist() == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    torch.testing.assert_close(picked, stride_subsample(t, 10))
    # Below the budget, nothing is dropped.
    assert stride_subsample(t, 500).shape[0] == 100


# --------------------------------------------------------------------------
# 2. Bug guards
# --------------------------------------------------------------------------


def test_welford_counts_optimizer_steps_not_micro_batches() -> None:
    """Bug 10. Appendix H defines the variance over optimizer steps.

    The research code updated Welford from inside the gradient hook, which fires
    once per micro-batch, so at 4-step accumulation it recorded four observations
    per step -- inflating the count fourfold and measuring within-batch gradient
    noise instead of the step-to-step movement plasticity is supposed to capture.
    """
    torch.manual_seed(0)
    model = TinyVlm()
    tracker = SignalTracker(model, TrackerConfig()).attach()
    _train_steps(tracker, model, steps=3, micro_batches=4)
    stats = tracker.snapshot()
    tracker.detach()

    observed = {s.grad_norm_count for s in stats.layers.values() if s.grad_norm_count}
    assert observed == {3}, f"expected 3 optimizer-step observations, got {observed}"


def test_collection_never_synchronises_until_snapshot() -> None:
    """Bug 8. ``float(t.cpu().item())`` in the hooks meant a stall per module per step.

    Enforced rather than benchmarked, because a timing assertion would be flaky on
    shared CI while this one fails deterministically the moment a sync reappears:
    ``.item()``, ``.tolist()`` and ``.cpu()`` are the three ways a device tensor's
    value reaches the host, so they are counted for the duration of training and the
    count must be zero.

    Counted rather than raised. Raising from inside a hook would be caught by the
    tracker's own ``except Exception`` error accounting -- the guard would be
    swallowed by the very error tolerance it is meant to check, and the test would
    pass with a sync sitting right there. The counter cannot be swallowed.
    """
    torch.manual_seed(0)
    model = TinyVlm()
    tracker = SignalTracker(model, TrackerConfig(coherence_ema_beta=0.95)).attach()

    syncs: list[str] = []

    def counting(name: str, original: Any) -> Any:
        def wrapper(self: Tensor, *args: Any, **kwargs: Any) -> Any:
            syncs.append(name)
            return original(self, *args, **kwargs)

        return wrapper

    originals = {name: getattr(Tensor, name) for name in ("item", "tolist", "cpu")}
    for name, original in originals.items():
        setattr(Tensor, name, counting(name, original))
    try:
        _train_steps(tracker, model, steps=3, micro_batches=2)
    finally:
        for name, original in originals.items():
            setattr(Tensor, name, original)

    assert not syncs, f"host syncs during collection: {sorted(set(syncs))} ({len(syncs)} calls)"

    # The hooks did run -- otherwise a tracker that collects nothing at all would
    # pass the assertion above trivially.
    stats = tracker.snapshot()
    tracker.detach()
    assert any(s.grad_norm_var > 0 for s in stats.layers.values())
    assert any(s.coherence_ema is not None for s in stats.layers.values())


def test_keys_are_canonical_through_a_peft_style_wrapper() -> None:
    """Bugs 3 and 7. Canonicalise at write time, so nothing has to be guessed later.

    The wrapper here is shaped like ``peft.tuners.lora.Linear``: the original module
    demoted to ``base_layer``, adapter factors alongside it. The tracker must hook
    the *wrapper* -- whose output gradient is the base weight's output gradient,
    since ``Y = base(X) + lora(X)`` -- and file the result under the name the weight
    will have once merged.
    """

    class FakeLoraLinear(nn.Module):
        def __init__(self, base: nn.Linear, rank: int = 2) -> None:
            super().__init__()
            self.base_layer = base
            self.lora_A = nn.ModuleDict({"default": nn.Linear(base.in_features, rank, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, base.out_features, bias=False)})
            self.scaling = {"default": 1.0}
            self.active_adapters = ["default"]

        def forward(self, x: Tensor) -> Tensor:
            return self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x))

    torch.manual_seed(0)
    model = TinyVlm()
    for layer in model.model.layers:
        layer.self_attn.q_proj = FakeLoraLinear(layer.self_attn.q_proj)  # type: ignore[assignment]

    tracker = SignalTracker(model, TrackerConfig()).attach()
    _train_steps(tracker, model, steps=2)
    stats = tracker.snapshot()
    tracker.detach()

    assert "model.layers.0.self_attn.q_proj" in stats.layers
    # Neither the wrapper's internals nor the adapter factors may appear as keys.
    leaked = [n for n in stats.layers if "base_layer" in n or "lora_" in n]
    assert not leaked, f"non-canonical keys leaked into the stats file: {leaked}"
    # And the wrapped module still carries a real gradient signal, measured on the
    # frozen base weight rather than on an adapter factor.
    assert stats.layers["model.layers.0.self_attn.q_proj"].grad_norm_count == 2


def test_unexercised_modules_are_recorded_as_such_not_as_unimportant() -> None:
    """The vision-tower failure. Absence of evidence is not evidence of absence.

    A text-only fine-tune never routes through the vision tower, so its saliency
    and plasticity are structurally zero. Scoring from those zeros floors the whole
    tower; ``forward_calls == 0`` is what lets a consumer tell the two situations
    apart.
    """
    torch.manual_seed(0)
    model = TinyVlm()
    tracker = SignalTracker(model, TrackerConfig()).attach()
    _train_steps(tracker, model, steps=2)
    stats = tracker.snapshot()
    tracker.detach()

    vision = stats.layers["vision_tower.proj"]
    text = stats.layers["model.layers.0.self_attn.q_proj"]

    assert vision.forward_calls == 0
    assert vision.was_exercised is False
    assert text.forward_calls > 0
    assert text.was_exercised is True

    notes = stats.provenance.notes
    assert "vision_tower.proj" in notes["unexercised_modules"]
    assert "vision_tower.patch_embed" in notes["unexercised_modules"]

    report = stats.coverage(stats.names)
    assert set(report.unexercised) == set(notes["unexercised_modules"])
    assert "never exercised" in report.summary()


def test_tied_embedding_and_lm_head_are_reported_as_one_decision() -> None:
    """One tensor admits one bit width, whatever the per-role policy says.

    Qwen3.5-2B ties a 248,320-entry embedding to its LM head, so the paper's
    "4-bit embedding, 8-bit LM head" is not expressible on it -- and quietly
    letting whichever module the allocator visits second win would be a 25%-of-
    parameters coin flip.
    """
    torch.manual_seed(0)
    model = TinyVlm(tie=True)
    tracker = SignalTracker(model, TrackerConfig()).attach()
    _train_steps(tracker, model, steps=1)
    tied = tracker.snapshot().provenance.notes["tied_parameters"]
    tracker.detach()

    groups = [{key, *values} for key, values in tied.items()]
    assert {"model.embed_tokens", "lm_head"} in groups

    untied = TinyVlm(tie=False)
    untied_tracker = SignalTracker(untied, TrackerConfig()).attach()
    _train_steps(untied_tracker, untied, steps=1)
    assert "tied_parameters" not in untied_tracker.snapshot().provenance.notes
    untied_tracker.detach()


class _CheckpointedVlm(TinyVlm):
    """:class:`TinyVlm` with every decoder layer wrapped in a checkpoint.

    Deliberately checkpoints only the layers, leaving ``embed_tokens`` and ``lm_head``
    outside -- which is what ``gradient_checkpointing_enable`` does on a real
    ``transformers`` model, and the asymmetry is the whole point: the bug is not that the
    EMA horizon changes, it is that it changes for *some* modules.

    Both ``use_reentrant`` modes are exercised because they are not the same experiment.
    The reentrant implementation runs the *first* forward under ``no_grad`` and rebuilds the
    graph on the replay, so the tracker's ``out.requires_grad`` guard skips gradient-hook
    registration on the real forward and the replay's registration is the live one. The
    non-reentrant implementation is the other way round. A guard that returned early on
    ``_in_backward`` instead of only skipping the saliency read would therefore lose the
    plasticity signal outright under the reentrant mode -- which is what the research code
    got by calling ``gradient_checkpointing_enable()`` with no arguments.
    """

    def __init__(self, *args: Any, use_reentrant: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.use_reentrant = use_reentrant

    def forward(self, ids: Tensor) -> Tensor:
        h = self.model.embed_tokens(ids)
        for layer in self.model.layers:
            h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=self.use_reentrant)
        return self.lm_head(h)


def test_in_backward_actually_discriminates() -> None:
    """The guard below is worthless if the predicate it rests on silently stops working.

    ``_in_backward`` reads a private torch symbol and reports ``False`` when it is absent,
    so a rename upstream would turn every recompute back into a counted observation with no
    error anywhere. This test fails instead.
    """
    from dynquant.signals.tracker import _GRAPH_TASK_ID, _in_backward

    assert _GRAPH_TASK_ID is not None, "torch no longer exposes _current_graph_task_id"
    assert _in_backward() is False

    seen: list[bool] = []
    x = torch.randn(4, requires_grad=True)
    x.register_hook(lambda grad: seen.append(_in_backward()) or None)
    (x * 2).sum().backward()
    assert seen == [True]


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_a_checkpoint_replay_is_not_a_second_observation(use_reentrant: bool) -> None:
    """Gradient checkpointing replays the forward, and forward hooks fire again.

    Measured on a checkpointed Qwen3-0.6B before the guard existed:
    ``layers.0.self_attn.q_proj`` reported ``forward_calls=8`` over 4 steps while
    ``lm_head`` reported 4. Two EMA updates per micro-batch on identical data square the
    decay, so the checkpointed modules end up averaging over half the horizon the config
    asks for -- and the unchecked modules do not, which is what makes the cross-module
    percentile ranking in :mod:`dynquant.score` compare two different statistics.
    """
    steps = 4
    torch.manual_seed(0)
    plain = TinyVlm()
    plain_tracker = SignalTracker(plain, TrackerConfig()).attach()
    _train_steps(plain_tracker, plain, steps=steps)
    plain_stats = plain_tracker.snapshot()
    plain_tracker.detach()

    torch.manual_seed(0)
    checkpointed = _CheckpointedVlm(use_reentrant=use_reentrant)
    checkpointed.load_state_dict(plain.state_dict())
    ckpt_tracker = SignalTracker(checkpointed, TrackerConfig()).attach()
    _train_steps(ckpt_tracker, checkpointed, steps=steps)
    ckpt_stats = ckpt_tracker.snapshot()
    ckpt_tracker.detach()

    inside = "model.layers.0.self_attn.q_proj"
    assert plain_stats.layers[inside].forward_calls == steps
    assert ckpt_stats.layers[inside].forward_calls == steps, (
        "the recompute pass was counted as a second observation"
    )
    # Outside the checkpointed block, so it was never at risk -- included because a guard
    # that suppressed every second forward everywhere would also satisfy the line above.
    assert ckpt_stats.layers["lm_head"].forward_calls == steps
    # And every other module, because *which* ones replay is not ours to predict: torch's
    # non-reentrant path stops recomputing once the saved tensors it needs exist, so on a
    # real Qwen3-0.6B 168 of 198 modules doubled and every mlp.down_proj escaped. One count
    # per step for everything that ran is the invariant that survives that.
    assert {layer.forward_calls for layer in ckpt_stats.layers.values()} == {0, steps}

    # The signal itself, not just the counter. Same weights, same seed, same data, so a
    # module inside a checkpointed block must report the same saliency either way.
    for name in (inside, "model.layers.1.mlp.down_proj", "lm_head"):
        assert ckpt_stats.layers[name].activation_rms_ema == pytest.approx(
            plain_stats.layers[name].activation_rms_ema, rel=1e-5
        ), f"{name}: checkpointing changed the saliency signal"

    # The plasticity half must survive the fix. Under use_reentrant=True the only gradient
    # hook that ever gets registered is the *replay's*, because the real forward runs under
    # no_grad -- so a guard that returned early instead of skipping just the saliency read
    # would zero this out, and no assertion about forward_calls would notice.
    assert ckpt_stats.layers[inside].grad_norm_count == steps
    assert ckpt_stats.layers[inside].grad_norm_var > 0

    # The artifact says what happened to it. Absent on the plain run -- the note exists to
    # flag an uneven EMA horizon, and an unconditional key would make every stats file look
    # like it had one -- and present with observed=false here, which is how a consumer tells
    # "no replays" apart from "replays, skipped".
    assert "recompute_forward_calls" not in plain_stats.provenance.notes
    note = ckpt_stats.provenance.notes["recompute_forward_calls"]
    assert note["count"] > 0, "the replay was not detected at all"
    assert note["observed"] is False


@pytest.mark.parametrize("use_reentrant", [True, False])
def test_observe_recompute_restores_the_legacy_double_count(use_reentrant: bool) -> None:
    """The compat escape hatch, because the research code had no such guard.

    ``supervised_finetuning.py`` calls ``gradient_checkpointing_enable()`` unconditionally
    and passes ``gradient_checkpointing=True`` to ``TrainingArguments``, so both shipped
    stats files were collected *with* the double count. Reproducing one needs the old
    behaviour available, and pinning it here is what stops the flag becoming a no-op.
    """
    steps = 3
    torch.manual_seed(0)
    model = _CheckpointedVlm(use_reentrant=use_reentrant)
    tracker = SignalTracker(model, TrackerConfig(observe_recompute=True)).attach()
    _train_steps(tracker, model, steps=steps)
    stats = tracker.snapshot()
    tracker.detach()

    assert stats.layers["model.layers.0.self_attn.q_proj"].forward_calls == 2 * steps
    assert stats.layers["lm_head"].forward_calls == steps

    # A file collected this way is the one that most needs the note, since nothing else in it
    # distinguishes a 50-step average from a 100-step one. Pinned against the counters it
    # explains: the count is exactly the number of observations beyond one per step.
    note = stats.provenance.notes["recompute_forward_calls"]
    assert note["observed"] is True
    exercised = [layer for layer in stats.layers.values() if layer.forward_calls]
    assert note["count"] == sum(layer.forward_calls for layer in exercised) - steps * len(exercised)


# --------------------------------------------------------------------------
# 3. Plumbing
# --------------------------------------------------------------------------


def test_debiased_ema_equals_the_first_observation_after_one_update() -> None:
    """The reason no branch on "is this the first call?" is needed -- and no sync.

    A zero-seeded EMA holds ``(1 - beta) * x`` after one update, 1% of the truth at
    beta=0.99. Dividing by ``1 - beta**k`` recovers ``x`` exactly, which is what the
    research code's first-call assignment did at the cost of reading a device
    counter on the host.
    """
    torch.manual_seed(0)
    model = TinyVlm()
    tracker = SignalTracker(model, TrackerConfig(activation_ema_beta=0.99)).attach()

    ids = torch.zeros(1, 3, dtype=torch.long)
    target = model.model.layers[0].self_attn.q_proj
    seen: list[Tensor] = []
    probe = target.register_forward_hook(lambda m, a, o: seen.append(o.detach().clone()))
    model(ids)
    probe.remove()
    tracker.detach()

    out = seen[0].float()
    expected = out.mul(out).mean(dim=(0, 1)).add(1e-12).sqrt().mean().item()
    recorded = tracker.snapshot().layers["model.layers.0.self_attn.q_proj"]
    assert recorded.activation_rms_ema == pytest.approx(expected, rel=1e-5)
    assert recorded.forward_calls == 1


def test_saliency_ema_matches_per_call_updates_under_gradient_accumulation() -> None:
    """The batched saliency fold must equal applying each forward call in turn.

    The EMA is folded once per *forward call*, and the flush batches across modules
    to keep per-module dispatch off the step. Batching across calls to the same
    module would not be equivalent -- ``lerp`` is not associative in the way that
    would require -- so the flush is supposed to break whenever a slot repeats.
    Gradient accumulation is exactly that case: every module fires once per
    micro-batch, so this asserts the sequential result over three of them rather
    than trusting the duplicate test to fire.
    """
    torch.manual_seed(0)
    beta = 0.99
    model = TinyVlm()
    tracker = SignalTracker(model, TrackerConfig(activation_ema_beta=beta)).attach()

    target = model.model.layers[0].self_attn.q_proj
    seen: list[Tensor] = []
    probe = target.register_forward_hook(lambda m, a, o: seen.append(o.detach().clone()))
    generator = torch.Generator().manual_seed(0)
    for _ in range(3):
        model(torch.randint(0, 16, (2, 5), generator=generator))
    probe.remove()
    tracker.detach()

    assert len(seen) == 3
    ema = 0.0
    for out in seen:
        value = out.float().square().mean(dim=(0, 1)).add(1e-12).sqrt().mean().item()
        ema += (1.0 - beta) * (value - ema)
    recorded = tracker.snapshot().layers["model.layers.0.self_attn.q_proj"]
    assert recorded.forward_calls == 3
    # Debiasing divides by 1 - beta**3, so compare against the biased EMA undone the
    # same way rather than reimplementing the debias here.
    assert recorded.activation_rms_ema == pytest.approx(ema / (1.0 - beta**3), rel=1e-5)


def test_snapshot_sees_saliency_from_a_forward_pass_with_no_optimizer_step() -> None:
    """Deferring the fold must stay invisible to a reader of the accumulators.

    ``log_every`` snapshots land between micro-batches, so a flush that only ran on
    the optimizer step would report ``0.0`` for every module -- indistinguishable
    from a module that never fired, which is the one thing ``forward_calls`` exists
    to disambiguate.
    """
    torch.manual_seed(0)
    model = TinyVlm()
    tracker = SignalTracker(model, TrackerConfig()).attach()
    model(torch.zeros(1, 3, dtype=torch.long))

    stats = tracker.snapshot()  # no on_optimizer_step, still attached
    recorded = stats.layers["model.layers.0.self_attn.q_proj"]
    assert recorded.forward_calls == 1
    assert recorded.activation_rms_ema > 0.0
    tracker.detach()


def test_saliency_of_a_module_that_did_not_fire_does_not_decay() -> None:
    """A batched fold must be indexed, not whole-tensor.

    The flush applies one ``lerp`` across every slot observed this step. Applying it
    to the whole accumulator instead would pull absent modules toward zero once per
    step -- silently, and worst for exactly the rarely-routed MoE experts whose
    scores the tracker is meant to keep honest.
    """
    torch.manual_seed(0)
    model = TinyVlm()
    tracker = SignalTracker(model, TrackerConfig()).attach()

    # Fire everything once, then read one module's EMA before and after a second
    # forward pass in which that module is the only one held out.
    model(torch.zeros(1, 3, dtype=torch.long))
    name = "model.layers.0.self_attn.q_proj"
    before = tracker.snapshot().layers[name].activation_rms_ema
    assert before > 0.0

    held_out = model.model.layers[0].self_attn.q_proj
    tracker._by_module.pop(id(held_out))  # the module still runs; the tracker ignores it
    model(torch.zeros(1, 3, dtype=torch.long))
    after = tracker.snapshot().layers[name]
    tracker.detach()

    assert after.forward_calls == 1
    assert after.activation_rms_ema == pytest.approx(before, rel=1e-12)


def test_coherence_equals_a_hand_computed_cosine_ema() -> None:
    """The batched cosine must equal the per-module one, to floating point.

    Coherence had no exact pin, only shape and presence checks, and the batched
    flush computes its dot products as the L1 norm of an elementwise product --
    valid only because both operands are per-channel L2 norms and therefore
    non-negative. That is a real assumption about ``channel_norm``, so it gets a
    test that fails if either side of it changes.
    """
    torch.manual_seed(0)
    beta = 0.95
    model = TinyVlm()
    config = TrackerConfig(coherence_ema_beta=beta, subsample_tokens=LIMIT)
    tracker = SignalTracker(model, config).attach()

    target = model.model.layers[0].self_attn.q_proj
    grads: list[Tensor] = []

    # Returns None deliberately: a forward hook's return value *replaces* the module
    # output, so handing back register_hook's RemovableHandle would rewire the model.
    def capture(module: nn.Module, args: Any, output: Tensor) -> None:
        output.register_hook(lambda g: grads.append(g.detach().clone()))

    probe = target.register_forward_hook(capture)
    _train_steps(tracker, model, steps=3)
    probe.remove()
    stats = tracker.snapshot()
    tracker.detach()

    assert len(grads) == 3
    channel = [g.reshape(-1, g.shape[-1]).float().pow(2).sum(dim=0).sqrt() for g in grads]
    # The first call has nothing to compare against, so three forwards give two
    # observations -- which is also what the debias divisor has to use.
    ema, calls = 0.0, 0
    for previous, current in itertools.pairwise(channel):
        cosine = (current @ previous) / (current.norm() * previous.norm() + config.eps)
        ema += (1.0 - beta) * (cosine.item() - ema)
        calls += 1

    assert calls == 2
    recorded = stats.layers["model.layers.0.self_attn.q_proj"].coherence_ema
    assert recorded is not None
    assert recorded == pytest.approx(ema / (1.0 - beta**calls), rel=1e-5)


def test_staged_gradient_norms_sum_across_micro_batches() -> None:
    """Staging is an ``index_add_``, and the distinction is not cosmetic.

    ``on_optimizer_step`` divides the staged total by the observation count to get
    the step's mean, so a batched fold that *overwrote* instead of accumulating
    would silently report the last micro-batch rather than the mean -- correct at
    ``micro_batches=1`` and wrong for every gradient-accumulation run, which is the
    configuration real fine-tunes use.
    """
    name = "model.layers.0.self_attn.q_proj"

    def run(model: TinyVlm, tracker: SignalTracker, *, isolate: bool) -> list[float]:
        """Staged value after each of three micro-batches, read via an explicit flush.

        With ``isolate`` the accumulator is cleared first, so each entry is that one
        micro-batch's observation; without it, each entry is the running total. No
        optimizer step runs in either case -- that is what consumes the staging.
        """
        slot = next(e for e in tracker._tracked if e.name == name).slot
        generator = torch.Generator().manual_seed(0)
        values: list[float] = []
        for _ in range(3):
            model(torch.randint(0, 16, (2, 5), generator=generator)).square().mean().backward()
            if isolate:
                tracker._grad_stage.zero_()
            tracker._flush_backward()
            values.append(tracker._grad_stage[slot].item())
        return values

    # Two passes over identical inputs. Nothing updates the weights, so the same
    # micro-batch produces the same observation in both, and the totals are
    # comparable -- which is what makes this an independent check rather than the
    # accumulator agreeing with itself.
    torch.manual_seed(0)
    accumulating = TinyVlm()
    tracker_a = SignalTracker(accumulating, TrackerConfig()).attach()
    running = run(accumulating, tracker_a, isolate=False)

    torch.manual_seed(0)
    isolated = TinyVlm()
    tracker_b = SignalTracker(isolated, TrackerConfig()).attach()
    singles = run(isolated, tracker_b, isolate=True)

    assert all(v > 0.0 for v in singles)
    assert running[-1] == pytest.approx(sum(singles), rel=1e-5)
    # The load-bearing inequality: an overwriting fold would leave only the last
    # micro-batch, and this is what separates the two.
    assert running[-1] > singles[-1] * 1.5

    tracker_a.on_optimizer_step()
    recorded = tracker_a.snapshot().layers[name]
    tracker_a.detach()
    tracker_b.detach()
    assert recorded.grad_norm_count == 1
    assert recorded.grad_norm_mean == pytest.approx(sum(singles) / 3.0, rel=1e-5)


def test_variance_matches_a_reference_welford_pass() -> None:
    """The vectorized fold must equal a plain single-stream Welford.

    Driven through ``param`` mode with hand-set gradients, so the observations are
    known exactly and the test measures the accumulator rather than the estimator.
    """
    torch.manual_seed(0)
    model = TinyVlm()
    config = TrackerConfig(grad_estimator=GradEstimatorMode.PARAM.value)
    tracker = SignalTracker(model, config).attach()

    target = model.model.layers[0].mlp.down_proj
    norms = [3.0, 5.0, 11.0, 2.0, 7.0]
    for norm in norms:
        grad = torch.zeros_like(target.weight)
        grad.view(-1)[0] = norm  # a one-hot gradient has exactly this L2 norm
        target.weight.grad = grad
        tracker.on_optimizer_step()
    stats = tracker.snapshot()
    tracker.detach()

    recorded = stats.layers["model.layers.0.mlp.down_proj"]
    mean = sum(norms) / len(norms)
    variance = sum((n - mean) ** 2 for n in norms) / (len(norms) - 1)
    assert recorded.grad_norm_count == len(norms)
    assert recorded.grad_norm_mean == pytest.approx(mean, rel=1e-5)
    assert recorded.grad_norm_var == pytest.approx(variance, rel=1e-5)
    assert recorded.grad_estimator == "param"


def test_stats_round_trip_through_disk(tmp_path: Any) -> None:
    torch.manual_seed(0)
    model = TinyVlm()
    tracker = SignalTracker(model, TrackerConfig(coherence_ema_beta=0.95)).attach()
    _train_steps(tracker, model, steps=3)
    written = tracker.save(tmp_path)
    original = tracker.snapshot()
    tracker.detach()

    assert written is not None
    reloaded = load_stats(written)
    assert reloaded.names == original.names
    for name, layer in original.layers.items():
        other = reloaded.layers[name]
        assert other.grad_norm_count == layer.grad_norm_count
        assert other.forward_calls == layer.forward_calls
        assert other.grad_norm_var == pytest.approx(layer.grad_norm_var)
    # forward_calls must survive as a first-class field, not land in notes.
    raw = json.loads(written.read_text(encoding="utf-8"))
    assert "forward_calls" in raw["layers"]["model.layers.0.mlp.gate_proj"]


def test_coherence_is_off_by_default_and_shape_stable_when_on() -> None:
    torch.manual_seed(0)
    model = TinyVlm()
    plain = SignalTracker(model, TrackerConfig()).attach()
    _train_steps(plain, model, steps=3)
    default_stats = plain.snapshot()
    plain.detach()
    assert default_stats.coherence_ema_beta is None
    assert all(s.coherence_ema is None for s in default_stats.layers.values())

    torch.manual_seed(0)
    model = TinyVlm()
    tracked = SignalTracker(model, TrackerConfig(coherence_ema_beta=0.95)).attach()
    _train_steps(tracked, model, steps=3)
    with_coherence = tracked.snapshot()
    tracked.detach()
    observed = [
        s.coherence_ema for s in with_coherence.layers.values() if s.coherence_ema is not None
    ]
    assert observed, "coherence was requested but never recorded"
    assert all(-1.0001 <= value <= 1.0001 for value in observed)


def test_include_and_exclude_select_modules_by_canonical_name() -> None:
    torch.manual_seed(0)
    model = TinyVlm()
    tracker = SignalTracker(model, TrackerConfig(include=("model.layers.*.mlp.*",)))
    assert set(tracker.tracked_names) == {
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.1.mlp.gate_proj",
        "model.layers.1.mlp.down_proj",
    }

    excluded = SignalTracker(model, TrackerConfig(exclude=("vision_tower.*", "lm_head")))
    assert not [n for n in excluded.tracked_names if n.startswith("vision_tower")]
    assert "lm_head" not in excluded.tracked_names


def test_norms_and_biases_are_never_tracked() -> None:
    """A 1-D weight is a norm or a bias; ``ndim >= 2`` excludes every one of them
    without needing to enumerate the norm classes of every architecture."""
    model = TinyVlm()
    tracker = SignalTracker(model, TrackerConfig())
    assert not [n for n in tracker.tracked_names if "layernorm" in n]
    assert "vision_tower.patch_embed" in tracker.tracked_names  # Conv2d: ndim 4


def test_hook_failures_are_counted_and_then_refused() -> None:
    """Never crash training, never hide the damage either.

    A bare ``except: return`` is exactly what kept the supplement's coherence signal
    silently dead through every experiment in the paper. Here failures are counted,
    reported in provenance, and past a threshold refused outright -- because stats
    that are quietly 60% complete are worse than no stats.
    """
    torch.manual_seed(0)
    model = TinyVlm()
    tracker = SignalTracker(model, TrackerConfig(max_hook_errors=2)).attach()

    def explode(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("synthetic hook failure")

    tracker._update_saliency = explode  # type: ignore[method-assign]
    _train_steps(tracker, model, steps=1)

    with pytest.raises(SignalCollectionError, match="max_hook_errors"):
        tracker.snapshot()
    tracker.detach()

    tracker.config = TrackerConfig(max_hook_errors=10_000)
    notes = tracker.snapshot().provenance.notes
    assert notes["hook_errors"], "failures must be reported, not swallowed"


def test_a_model_with_nothing_to_track_is_an_error_not_an_empty_file() -> None:
    with pytest.raises(SignalCollectionError, match="no quantizable modules"):
        SignalTracker(nn.Sequential(nn.LayerNorm(4), nn.ReLU()), TrackerConfig())


def test_config_validation_rejects_impossible_settings() -> None:
    for kwargs in (
        {"activation_ema_beta": 1.0},
        {"activation_ema_beta": 0.0},
        {"coherence_ema_beta": 1.5},
        {"saliency_source": "middle"},
        {"subsample_tokens": 0},
        {"log_every": -1},
        {"grad_estimator": "vibes"},
    ):
        with pytest.raises(ValueError):
            TrackerConfig(**kwargs).validate()  # type: ignore[arg-type]


def test_provenance_records_what_would_make_two_runs_incomparable() -> None:
    torch.manual_seed(0)
    model = TinyVlm()
    tracker = SignalTracker(model, TrackerConfig(subsample_tokens=64)).attach()
    _train_steps(tracker, model, steps=2)
    provenance = tracker.snapshot().provenance
    tracker.detach()

    assert provenance.num_optimizer_steps == 2
    assert provenance.grad_estimator == "outer_exact"
    assert provenance.canonical_names is True
    assert provenance.notes["saliency_source"] == "output"
    assert provenance.notes["subsample_tokens"] == 64


def test_reset_clears_accumulators_but_keeps_hooks() -> None:
    torch.manual_seed(0)
    model = TinyVlm()
    tracker = SignalTracker(model, TrackerConfig()).attach()
    _train_steps(tracker, model, steps=2)
    tracker.reset()

    cleared = tracker.snapshot()
    assert all(s.grad_norm_count == 0 for s in cleared.layers.values())
    assert all(s.forward_calls == 0 for s in cleared.layers.values())
    assert tracker.attached

    _train_steps(tracker, model, steps=1)
    assert any(s.forward_calls > 0 for s in tracker.snapshot().layers.values())
    tracker.detach()


def test_detach_removes_every_hook() -> None:
    torch.manual_seed(0)
    model = TinyVlm()
    with SignalTracker(model, TrackerConfig()) as tracker:
        _train_steps(tracker, model, steps=1)
    assert not tracker.attached

    before = tracker.snapshot()
    model(torch.zeros(1, 3, dtype=torch.long))
    after = tracker.snapshot()
    assert before.layers["lm_head"].forward_calls == after.layers["lm_head"].forward_calls
