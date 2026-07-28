from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Tuple


def load_unified_stats(stats_path: str) -> Dict[str, Any]:
    with open(stats_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "layers" not in payload:
        raise ValueError("Invalid stats file: missing 'layers'")
    return payload


def _percentile_ranks(values: Dict[str, float]) -> Dict[str, float]:
    """Compute percentile ranks in [0,1] with average ranks for ties."""

    items: List[Tuple[str, float]] = [(k, float(v)) for k, v in values.items()]
    # Sort ascending; lower values => lower percentiles.
    items.sort(key=lambda kv: kv[1])

    n = len(items)
    if n == 0:
        return {}
    if n == 1:
        return {items[0][0]: 1.0}

    ranks: Dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        v = items[i][1]
        while j < n and items[j][1] == v:
            j += 1

        # items i..j-1 are equal. Assign average rank.
        # Use 1..n ranks then map to percentiles (rank-1)/(n-1).
        avg_rank = (i + 1 + j) / 2.0
        pct = (avg_rank - 1.0) / (n - 1.0)
        for k in range(i, j):
            ranks[items[k][0]] = float(pct)
        i = j

    return ranks


def compute_unified_scores(stats_path: str) -> Dict[str, float]:
    """Compute Unified DynQuant importance scores from saved stats.

    Logic:
    - Gradient variance is log-transformed (log1p) for stability.
    - log(grad_var), activation magnitude, and coherence are converted to percentile ranks.
    - Final score is the product of the three percentile ranks.
    """

    payload = load_unified_stats(stats_path)
    layers = payload["layers"]

    grad_metric: Dict[str, float] = {}
    act_metric: Dict[str, float] = {}
    coh_metric: Dict[str, float] = {}

    for name, st in layers.items():
        if not isinstance(st, dict):
            continue
        grad_var = float(st.get("grad_norm_var", 0.0))
        act = float(st.get("activation_rms_ema", 0.0))
        coh = float(st.get("coherence_ema", 0.0))

        # Log-transform to compress heavy tails.
        grad_log = math.log1p(max(0.0, grad_var))

        grad_metric[name] = grad_log
        act_metric[name] = act
        coh_metric[name] = coh

    pr_grad = _percentile_ranks(grad_metric)
    pr_act = _percentile_ranks(act_metric)
    pr_coh = _percentile_ranks(coh_metric)

    scores: Dict[str, float] = {}
    for name in set(pr_grad.keys()) | set(pr_act.keys()) | set(pr_coh.keys()):
        g = float(pr_grad.get(name, 0.0))
        a = float(pr_act.get(name, 0.0))
        # Treat missing coherence as neutral (1.0) rather than zeroing the score.
        c = float(pr_coh.get(name, 1.0))
        scores[name] = g * a * c

    return scores
