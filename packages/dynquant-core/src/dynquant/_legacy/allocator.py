from __future__ import annotations

from typing import Dict, List, Tuple


def _is_safety_floor(name: str) -> bool:
    # Safety zones: LM Head forced to 8-bit.
    # We removed "o_proj" (Attn Output) from safety floor to allow it to drop to 4-bit.
    # This frees up massive budget for the sensitive Gate projections.
    return "lm_head" in name


def _is_embedding(name: str) -> bool:
    n = name.lower()
    return any(x in n for x in ["embed", "wte", "embed_tokens"])


def _is_norm(name: str) -> bool:
    # Phi-4 norms / layer norms
    return ("norm" in name) or ("ln_" in name)


def _min_bits_floor(name: str, activation_rms: float, target_avg_bits: float, conservative_attention: bool) -> int:
    """Return a minimum bit-width floor for stability-critical weights.
    
    Updated Rules (Optimized for 3-bit performance):
    - High Activation Outliers (RMS > 2.5): 4-bit.
    - Embeddings: 4 bits.
    - Attn (Q, K, V, O): 4 bits. (Reduced O from 8 to 4).
    - MLP Gate: 4 bits (Crucial for SwiGLU stability).
    - MLP Up/Down: 3 bits (Can tolerate noise better than Gate).
    """

    # 1. Activation Spike Protection
    if activation_rms > 2.5:
        return 4

    # Embeddings: 4-bit (Qwen/Phi both tend to be stable here; avoids the 8-bit "tax")
    if _is_embedding(name):
        return 4
    
    # Attention: Keep purely at 4-bit to avoid 3-bit attention artifacts
    # (Q/K/V/O all 4-bit)
    if any(x in name for x in ["q_proj", "k_proj", "v_proj", "qkv_proj", "o_proj", "output"]):
        return 4

    # --- MLP STRATEGY ---
    # GATE: 4-bit. Controls the activation flow. 3-bit here destroys the signal.
    if "gate_proj" in name:
        return 4
        
    # UP/DOWN: 3-bit. 
    # Down_proj sums noise (averaging effect). Up_proj is raw value.
    # These are fine at 3-bit with Group-Wise quantization.
    if any(x in name for x in ["down_proj", "up_proj", "mlp"]):
        return 3

    return 2



def allocate_bits_pareto(
    scores: Dict[str, float],
    param_counts: Dict[str, int],
    layer_stats: Dict[str, Dict[str, float]] | None = None, # Added access to raw stats
    target_avg_bits: float = 3.5,
    conservative_attention: bool = False,
) -> Dict[str, int]:
    """Allocate per-layer bit-widths to match a target average.

    Strategy:
    - Safety floor: embeddings + lm_head/output forced to 8-bit (stable + compact).
    - Norms: kept in fp16 (16-bit) to avoid numerical drift.
    - Variable layers: start at 2-bit, apply per-layer minimum floors for stability,
      then greedily upgrade with best ROI to meet `target_avg_bits` over VARIABLE layers only.

    ROI uses: score / added_cost, where added_cost = params * delta_bits.
    """
    print(f"DEBUG: Running allocate_bits_pareto (UPDATED). Target bits: {target_avg_bits}")

    # NOTE: 3-bit packing is slower (Python loop), but supported.
    BIT_OPTIONS = [2, 3, 4, 8]

    # Only consider layers we have params for.
    layer_names = [n for n in scores.keys() if n in param_counts]

    allocation: Dict[str, int] = {}
    spent = 0.0
    variable_layers: List[str] = []
    variable_params = 0

    # 1) Apply safety floors / stability rules.
    # IMPORTANT: `target_avg_bits` describes the average bits over VARIABLE layers only.
    # Layers forced to 8-bit (safety) or 16-bit (norms) are EXCLUDED from the budget.
    for name in layer_names:
        
        # Extract Act RMS if available
        act_rms = 0.0
        if layer_stats and name in layer_stats:
            act_rms = float(layer_stats[name].get("activation_rms_ema", 0.0))
        # Handle PEFT naming difference (allocator receives "pure" name, stats might have "base_model...")
        # (This is handled by caller in run_quantization usually, but let's be safe)
        
        # Embeddings: explicitly fixed at 4-bit.
        # This prevents older configs from accidentally treating embeddings as an 8-bit safety floor.
        if _is_embedding(name):
            allocation[name] = 4
            variable_layers.append(name)
            variable_params += int(param_counts[name])
        elif _is_safety_floor(name):
            allocation[name] = 8
        elif _is_norm(name):
            allocation[name] = 16
        else:
            floor = int(_min_bits_floor(name, act_rms, target_avg_bits, conservative_attention))
            if floor not in BIT_OPTIONS:
                floor = 2
            allocation[name] = floor
            variable_layers.append(name)
            variable_params += int(param_counts[name])

    if variable_params <= 0:
        return allocation

    # 2) Build budget over VARIABLE layers only.
    variable_params_total = sum(param_counts[n] for n in variable_layers)
    base_cost = float(variable_params_total) * 2.0  # start all variable layers at 2-bit
    total_budget = float(variable_params_total) * float(target_avg_bits)

    # Initial spend: account for per-layer floors above 2-bit.
    spent = 0.0
    for name in variable_layers:
        floor_bits = int(allocation[name])
        if floor_bits > 2:
            spent += float(param_counts[name]) * float(floor_bits - 2)

    remaining = total_budget - base_cost - spent
    if remaining < 0:
        # Floors exceed the requested budget; return the forced allocation.
        return allocation

    # Pre-sort for stable iteration (determinism)
    variable_layers.sort()

    # Helper: next upgrade option
    # REMOVED 8-bit upgrade option:
    # Upgrading small attention layers to 8-bit while leaving massive MLPs at 2-bit
    # causes "gibberish" output. Capping at 4-bit forces the budget to be spent
    # on lifting the 2-bit floors to 3/4-bit first (structural integrity).
    next_bits: Dict[int, int] = {2: 3, 3: 4}

    # 3) Greedy loop: repeatedly upgrade the best ROI layer.
    while True:
        best: Tuple[float, str, int, int, float] | None = None
        # (roi, name, old_bits, new_bits, cost)

        for name in variable_layers:
            old = allocation[name]
            if old not in next_bits:
                continue
            new = next_bits[old]
            cost = float(param_counts[name] * (new - old))
            if cost <= 0 or cost > remaining:
                continue

            roi = float(scores.get(name, 0.0)) / cost
            if (best is None) or (roi > best[0]):
                best = (roi, name, old, new, cost)

        if best is None:
            break

        _, name, _old, new, cost = best
        allocation[name] = new
        remaining -= cost

        if remaining <= 0:
            break

    return allocation
