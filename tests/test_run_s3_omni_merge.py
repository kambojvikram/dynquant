"""The merge verification has to detect movement, and a norm difference does not.

This file exists because the first version of `run_s3_omni_merge.py` aborted a merge that
had worked. It fingerprinted each parameter by its L2 norm and called a weight "moved"
when the relative norm changed by more than 1e-6. But `||W+D|| - ||W||` is `<W,D>/||W||`
to first order, so a delta near-orthogonal to the base weight registers only at second
order -- and a LoRA delta, fitted against an NF4 base and folded into a bf16 one, is very
nearly orthogonal to it.

Measured on the Qwen3-Omni-30B SLURP adapter, in float64, over all 384 targets:

* every target had a real delta, `||D||/||W||` from 2.7e-3 to 1.2e-2;
* 48% to 91% of the *stored bf16 elements* changed value;
* the relative norm change ran 7.7e-7 .. 1.2e-4 -- one continuous band, no gap, and the
  1e-6 cut fell inside it: passing minimum 1.073e-6 against failing maximum 9.888e-7.

Nine weights were declared unmoved and the script exited 5 on a correct merge. The tests
below pin the geometry that caused it, so a future return to any norm-difference test
fails here rather than after a 63 GiB load.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "scripts" / "run_s3_omni_merge.py"

# The driver imports the S2 trainer for LORA_TARGETS/load_thinker, which reaches
# transformers. Declared at module scope so collection reports a skip rather than
# erroring on every test in the core-only matrix job. peft is deliberately *not*
# required: everything under test here is fingerprint arithmetic on plain tensors, and
# these are the checks that must run in CI rather than only on a box with a merge on it.
pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def s3():
    spec = importlib.util.spec_from_file_location("_dq_s3", DRIVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dq_s3"] = module
    spec.loader.exec_module(module)
    return module


def _orthogonal_delta(weight: torch.Tensor, rho: float) -> torch.Tensor:
    """A perturbation of relative size ``rho`` carrying no component along ``weight``.

    This is the worst case for a norm test and the ordinary case for a merge: with the
    ``<W,D>`` term gone, ``||W+D||/||W||`` is exactly ``sqrt(1 + rho**2)``, so the norm
    moves by ``rho**2 / 2`` while every element moves by order ``rho``.
    """
    raw = torch.randn_like(weight)
    raw -= weight * (torch.dot(raw.flatten(), weight.flatten()) / weight.pow(2).sum())
    return raw * (rho * torch.linalg.vector_norm(weight) / torch.linalg.vector_norm(raw))


def _linear(weight: torch.Tensor) -> nn.Linear:
    module = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        module.weight.copy_(weight)
    return module


def test_a_near_orthogonal_delta_moves_almost_every_element_and_barely_the_norm(s3):
    """The defect's premise, stated as arithmetic before it is stated as a fingerprint."""
    torch.manual_seed(0)
    weight = torch.randn(256, 128)
    delta = _orthogonal_delta(weight, rho=1e-3)

    reference = torch.linalg.vector_norm(weight.double())
    relative_norm_change = (
        torch.linalg.vector_norm((weight + delta).double()) - reference
    ) / reference

    # Under the old test this weight is "unchanged" -- by a factor of two, in float64,
    # with no reduction noise to blame it on.
    assert abs(relative_norm_change.item()) < 1e-6
    # Yet it is a different tensor in every element that matters.
    assert (weight + delta != weight).double().mean().item() > 0.99


def test_the_fingerprint_calls_that_delta_moved(s3):
    torch.manual_seed(0)
    weight = torch.randn(256, 128)
    delta = _orthogonal_delta(weight, rho=1e-3)

    module = _linear(weight)
    before = s3.fingerprint(module)
    with torch.no_grad():
        module.weight.add_(delta)
    after = s3.fingerprint(module)

    assert s3.changed_share(before, after, "weight") > 0.99


def test_an_untouched_parameter_is_exactly_zero_not_merely_small(s3):
    """The other direction, and the one that matters most.

    Exit 4 asserts no expert bank moved. A tolerance there is a hole: a bank perturbed
    just under it reads as clean. Sampled equality has no tolerance to sit under.
    """
    torch.manual_seed(1)
    module = _linear(torch.randn(64, 32))
    before = s3.fingerprint(module)
    after = s3.fingerprint(module)

    assert s3.changed_share(before, after, "weight") == 0.0


def test_a_delta_confined_to_a_few_elements_is_still_movement(s3):
    """A bank corrupted in one slice changes a small share of elements, not half of them.

    The sample is 4096 elements, so this is probabilistic in general -- but a change to
    one full row of a 64x32 weight is 1/64 of it, which the sample sees essentially
    always. Pinned so a future shrink of SAMPLE_ELEMENTS shows up as a failure here.
    """
    torch.manual_seed(2)
    module = _linear(torch.randn(64, 32))
    before = s3.fingerprint(module)
    with torch.no_grad():
        module.weight[7].add_(1.0)
    after = s3.fingerprint(module)

    share = s3.changed_share(before, after, "weight")
    assert share > 0.0
    assert share == pytest.approx(1 / 64, abs=0.02)


def test_the_sample_is_stable_across_calls_and_specific_to_the_tensor(s3):
    """Before and after must sample the same slots, or every weight reads as moved."""
    first = s3.sample_index(4096, "model.layers.0.self_attn.q_proj.weight")
    again = s3.sample_index(4096, "model.layers.0.self_attn.q_proj.weight")
    other = s3.sample_index(4096, "model.layers.1.self_attn.q_proj.weight")
    reshaped = s3.sample_index(2048, "model.layers.0.self_attn.q_proj.weight")

    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    assert not torch.equal(first[:2048], reshaped)


def test_the_sample_is_keyed_by_the_unwrapped_name(s3):
    """`fingerprint` seeds on `base_name(...)`, so a peft-wrapped weight and its merged
    self sample the same slots. Seeding on the raw name would make every LoRA target
    read as moved and every assertion in the script vacuous."""
    wrapped = "model.layers.0.self_attn.q_proj.base_layer.weight"
    plain = "model.layers.0.self_attn.q_proj.weight"

    assert s3.base_name(wrapped) == plain
    assert torch.equal(s3.sample_index(4096, s3.base_name(wrapped)), s3.sample_index(4096, plain))


def test_a_tensor_smaller_than_the_sample_is_fingerprinted_whole(s3):
    module = _linear(torch.randn(8, 8))
    before = s3.fingerprint(module)

    assert before["weight"][0] == 64
    assert before["weight"][1].numel() == 64
