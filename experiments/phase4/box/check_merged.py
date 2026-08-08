"""Did the merge actually merge, or is it the base model under a new name?

The probe answers this statistically and cannot: at 128 items the binomial standard error
is 4.3 points, and the failure mode -- LoRA saved but never folded into the base weights --
lands the accuracy exactly on the base model's 57.75%, which is the one place a threshold
test has no power. Meanwhile a fine-tuned model that genuinely improved reads below the
threshold often enough to stop a good launch.

The question has a direct answer. Load one adapted matrix from each checkpoint and compare
the values. Not the file bytes -- safetensors metadata ordering is not stable, so a digest
of the file is not a test of its contents.

    check_merged.py <base_dir> <merged_dir>
"""

import json
import sys
from pathlib import Path

from safetensors import safe_open

base, merged = Path(sys.argv[1]), Path(sys.argv[2])
NEEDLE = "self_attn.q_proj.weight"


def load_one(root: Path) -> tuple[str, object]:
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        names = sorted(n for n in weight_map if NEEDLE in n)
        if not names:
            raise SystemExit(f"no tensor matching {NEEDLE!r} in {index}")
        name = names[0]
        shard = root / weight_map[name]
    else:
        shards = sorted(root.glob("*.safetensors"))
        for shard in shards:
            with safe_open(str(shard), framework="pt") as f:
                # safe_open exposes keys() but is not iterable, so the idiomatic
                # `n in f` rewrite would raise TypeError at the first shard.
                names = sorted(n for n in f.keys() if NEEDLE in n)  # noqa: SIM118
            if names:
                name = names[0]
                break
        else:
            raise SystemExit(f"no tensor matching {NEEDLE!r} under {root}")
    with safe_open(str(shard), framework="pt") as f:
        return name, f.get_tensor(name)


base_name, base_w = load_one(base)
merged_name, merged_w = load_one(merged)
if base_name != merged_name:
    raise SystemExit(f"compared different tensors: {base_name} vs {merged_name}")

delta = (merged_w.float() - base_w.float()).abs()
peak, mean = delta.max().item(), delta.mean().item()
scale = base_w.float().abs().mean().item()
print(f"tensor {base_name}  shape {tuple(base_w.shape)}")
print(f"max |merged - base| = {peak:.3e}   mean = {mean:.3e}   (base mean |w| = {scale:.3e})")
if peak == 0.0:
    raise SystemExit(
        "the merged checkpoint's weights are IDENTICAL to the base model's. The adapter was "
        "not folded in, and every arm would quantize the base model while the table called it "
        "fine-tuned."
    )
print(f"merged: weights moved, mean shift is {mean / scale:.2%} of the base magnitude")
