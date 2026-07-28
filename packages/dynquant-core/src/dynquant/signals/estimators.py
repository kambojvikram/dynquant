"""Gradient-norm estimators: three ways to ask "is training still pushing this weight?"

DynQuant's plasticity signal is ``Var_t ||grad W_i||`` -- the variance, over
optimizer steps, of the gradient norm of the weight that will be quantized. The
awkward part is that under (Q)LoRA **that weight is frozen**, so it has no
``.grad`` to read. The research supplement resolved this by hooking whatever
tensor did require grad -- ``lora_A.weight`` and ``lora_B.weight`` -- and
recording the result under the parent module's name. Two problems followed:

1. ``lora_A`` is ``[r, in]`` and ``lora_B`` is ``[out, r]``, so the
   per-output-channel coherence buffer alternated length between ``r`` and
   ``out`` on consecutive hook firings and every ``torch.dot`` between them
   raised, inside a bare ``except`` that discarded the error.
2. Even setting the crash aside, the number measured is the gradient of a rank-``r``
   factor, not of the ``[out, in]`` weight being quantized. Its magnitude scales
   with the LoRA rank and initialisation, not with the base weight's importance.

The fix is not a better fallback -- it is to measure the right tensor. The paper's
own Eq. (1) gives the identity needed::

    grad W = delta^T X          delta = dL/dY : [T, out]      X : [T, in]

Both factors are available to hooks even when ``W.requires_grad`` is ``False``:
``X`` from a forward hook, ``delta`` from a full backward hook. And this holds
whether or not a LoRA adapter is attached, because ``Y = base(X) + lora(X)``
routes the *same* output gradient to both branches -- so ``delta`` observed at the
wrapper is exactly the base weight's output gradient.

**The cost problem, and the trick that removes it.** Forming ``delta^T X`` costs
``T * out * in`` FLOPs and ``out * in`` bytes of workspace -- 67 MB per module at
4096x4096, hundreds of GFLOP per step across a model. But the norm never needs
the matrix::

    ||delta^T X||_F^2 = tr((delta^T X)^T (delta^T X))
                      = tr(X^T delta delta^T X)
                      = tr((delta delta^T)(X X^T))
                      = <delta delta^T, X X^T>_F        both factors symmetric

So two ``[T, T]`` Gram matrices suffice. At the default ``T = 256`` subsample that
is 256 KB of workspace and ``T^2 * (out + in)`` FLOPs -- about 16x less arithmetic
than the naive form at 4096x4096, and 250x less memory. The Gram of ``X`` is built
in the forward hook and is the *only* thing stashed for the backward pass, which
is what keeps activation stashing from costing gigabytes.

The identity is exact, not an approximation: subsampling tokens changes *which*
gradient is measured (one over a token subset), not the fidelity with which it is
measured.

Three modes, and they are **not comparable to each other** -- a stats file records
which one produced it so that mixing them is a detectable error rather than a
silently meaningless ranking:

``outer_exact`` (default)
    The identity above. Measures the base weight. Shape-stable regardless of LoRA
    rank, or of whether LoRA is attached at all.

``lowrank``
    The Frobenius norm of ``d(BA)/dt = grad_B @ A + B @ grad_A``, computed exactly
    from ``r x r`` traces without forming the ``[out, in]`` product. Measures how
    fast the *adapter's* contribution is moving. Cheaper than ``outer_exact`` and
    useful when you care about the update rather than the weight.

``param``
    ``weight.grad.norm()``. What the research code did. Correct and exact under
    full fine-tuning; under LoRA it falls back to adapter tensors and inherits the
    original semantics (minus the crash). Retained for ``--preset paper-3.15``.
"""

from __future__ import annotations

from enum import Enum

import torch
from torch import Tensor

__all__ = [
    "GradEstimatorMode",
    "channel_norm",
    "embedding_gram",
    "gram",
    "lowrank_grad_sq",
    "outer_grad_sq_batched",
    "outer_grad_sq_from_gram",
    "param_grad_sq",
    "stride_subsample",
    "tokens_2d",
]


class GradEstimatorMode(str, Enum):
    """Which tensor the plasticity signal is measured on."""

    OUTER_EXACT = "outer_exact"
    LOWRANK = "lowrank"
    PARAM = "param"

    @classmethod
    def parse(cls, value: str | GradEstimatorMode) -> GradEstimatorMode:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError:
            valid = ", ".join(m.value for m in cls)
            raise ValueError(
                f"unknown grad_estimator {value!r}. Expected one of: {valid}"
            ) from None


# --------------------------------------------------------------------------
# Token handling
# --------------------------------------------------------------------------


def tokens_2d(t: Tensor) -> Tensor:
    """Collapse every leading dimension into a token axis: ``[..., d] -> [T, d]``.

    ``reshape`` rather than ``view`` because activations arriving from attention
    kernels are frequently non-contiguous.
    """
    return t.reshape(-1, t.shape[-1])


def stride_subsample(t: Tensor, limit: int) -> Tensor:
    """Keep at most ``limit`` rows, evenly spaced across the token axis.

    Evenly spaced rather than random for three reasons: it is deterministic, so a
    rerun reproduces the same numbers; it costs no RNG draw and therefore no
    host-device sync; and it spreads the sample across sequence position instead
    of concentrating it in the prompt the way a prefix slice would.

    The same stride applied to the input and to the output gradient of one module
    selects the same tokens, which is what makes the Gram identity hold.
    """
    if limit <= 0:
        return t
    total = t.shape[0]
    if total <= limit:
        return t
    step = total // limit
    return t[: step * limit : step]


# --------------------------------------------------------------------------
# outer_exact
# --------------------------------------------------------------------------


def gram(t: Tensor, limit: int) -> Tensor:
    """Return ``S @ S.T`` for ``S`` the subsampled, fp32 view of ``t``.

    fp32 regardless of the training dtype: this is a sum over the feature axis of
    up to tens of thousands of products, and in bf16 the accumulation error is
    comparable to the signal.
    """
    s = stride_subsample(tokens_2d(t), limit).float()
    return s @ s.T


def embedding_gram(indices: Tensor, limit: int) -> Tensor:
    """The Gram matrix of the one-hot rows an ``nn.Embedding`` lookup selects.

    An embedding has no feature axis on its input, so there is no ``X`` to take a
    Gram of -- but the identity still applies, because the lookup *is* a matmul
    against a one-hot ``X``. The Gram of one-hot rows is the token-equality
    matrix::

        (X X^T)[t, s] = 1 if indices[t] == indices[s] else 0

    which gives exactly ``||grad W||_F^2 = sum_v ||sum_{t: idx_t = v} delta_t||^2``
    -- the true embedding gradient norm, including the cross terms from repeated
    tokens that a per-token approximation would drop.
    """
    idx = stride_subsample(indices.reshape(-1, 1), limit).squeeze(1)
    return (idx.unsqueeze(0) == idx.unsqueeze(1)).float()


def outer_grad_sq_from_gram(gram_x: Tensor, grad_output: Tensor, limit: int) -> Tensor:
    """``||grad W||_F^2 / T`` from a stashed input Gram and the output gradient.

    Normalised by the sampled token count so that the estimate is a *per-token*
    gradient norm. Without that division a step with more padding, or a shorter
    final batch, produces a smaller norm for purely mechanical reasons -- spurious
    variance injected into the one quantity the plasticity signal is measuring.

    Returns a 0-dim fp32 tensor and never synchronises.
    """
    gram_d = gram(grad_output, limit)
    n = gram_d.shape[0]
    if n != gram_x.shape[0]:
        # The forward that produced gram_x saw a different token count than this
        # backward. Under normal training this cannot happen; under an unusual
        # checkpointing or pipeline schedule it can, and a wrong-shaped product is
        # worse than a skipped observation.
        return torch.zeros((), dtype=torch.float32, device=gram_d.device)
    return (gram_d * gram_x).sum().clamp_min(0.0) / float(n)


def outer_grad_sq_batched(gram_x: list[Tensor], gram_d: list[Tensor]) -> Tensor:
    """:func:`outer_grad_sq_from_gram` for many modules at once. Returns ``[M]``.

    The batching is possible for a reason specific to this estimator: a Gram matrix is
    ``[T, T]`` in the *token* count, not in any feature dimension, and
    ``subsample_tokens`` caps ``T`` at one value for every module in a step. So the
    per-module contractions are identically shaped and stack, which the feature-space
    quantities elsewhere in the tracker never do.

    Callers must group by ``T`` before calling -- MoE experts see only their routed
    tokens, so a single step can produce more than one shape.

    Equivalent to the scalar version elementwise, and pinned to it by test. The
    normalisation is by ``T`` for the same per-token reason given there.
    """
    stacked_x = torch.stack(gram_x)
    stacked_d = torch.stack(gram_d)
    tokens = stacked_x.shape[-1]
    return (stacked_d * stacked_x).sum(dim=(-2, -1)).clamp_min(0.0) / float(tokens)


def channel_norm(grad_output: Tensor) -> Tensor:
    """Per-output-channel L2 norm of the output gradient: always ``[d_out]``.

    The coherence signal compares this vector against the previous step's. Keying
    it to ``d_out`` -- a property of the module, not of the adapter -- is what
    makes the comparison well-defined on every step, which is the concrete repair
    of the alternating-length ``torch.dot`` failure described in the module
    docstring.

    It is a proxy for the per-row norm of ``grad W`` rather than the thing itself;
    the exact version needs the full ``[out, in]`` matrix. Since coherence is only
    ever used as a direction, and this vector is the dominant factor in each row's
    magnitude, the proxy carries the signal at a fraction of the cost.
    """
    return tokens_2d(grad_output).float().pow(2).sum(dim=0).sqrt()


# --------------------------------------------------------------------------
# lowrank
# --------------------------------------------------------------------------


def lowrank_grad_sq(
    a: Tensor,
    b: Tensor,
    grad_a: Tensor,
    grad_b: Tensor,
    scaling: float = 1.0,
) -> Tensor:
    """``||scaling * (grad_B @ A + B @ grad_A)||_F^2`` via ``r x r`` traces only.

    ``A`` is ``[r, in]`` and ``B`` is ``[out, r]``, so the product is the ``[out,
    in]`` update actually applied to the base weight. Expanding the square::

        ||M||^2 = <grad_B^T grad_B, A A^T> + <B^T B, grad_A grad_A^T>
                  + 2 tr((grad_B^T B)(grad_A A^T)^T)

    Every factor is ``r x r``. At ``r = 16`` that is four 256-element matrices
    instead of an ``out x in`` one, so the cost is dominated by the thin matmuls
    that build them and the ``[out, in]`` product is never materialised.
    """
    a32, b32 = a.float(), b.float()
    ga32, gb32 = grad_a.float(), grad_b.float()

    term_b = (gb32.T @ gb32 * (a32 @ a32.T)).sum()
    term_a = (b32.T @ b32 * (ga32 @ ga32.T)).sum()
    cross = ((gb32.T @ b32) * (ga32 @ a32.T).T).sum()
    total = term_a + term_b + 2.0 * cross
    return total.clamp_min(0.0) * (scaling * scaling)


# --------------------------------------------------------------------------
# param
# --------------------------------------------------------------------------


def param_grad_sq(grad: Tensor) -> Tensor:
    """``||grad||_F^2`` in fp32. The legacy estimator, minus the host sync."""
    return grad.float().pow(2).sum()
