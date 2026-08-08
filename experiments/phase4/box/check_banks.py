"""Run the S2 expert-bank gate by hand against a finished stats file.

The live fine-tune launched from a commit that predates `banked_entries_missing`, so it
will not check itself when it lands. The gate needs a model only for its *names*, so this
builds one on the meta device from the checkpoint's config: no weights, no VRAM, seconds.

Usage: python check_banks.py <config-dir> <stats.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, "/workspace/dynquant/scripts")
sys.path.insert(0, "/workspace/dynquant/packages/dynquant-core/src")

from run_s2_finetune import banked_entries_missing

config_dir, stats_path = sys.argv[1], Path(sys.argv[2])
config = AutoConfig.from_pretrained(config_dir)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)

missing = banked_entries_missing(model, stats_path)

from dynquant.graph.experts import batched_expert_params  # noqa: E402
from dynquant.graph.naming import canonical_name  # noqa: E402

expected = [
    f"{canonical_name(name)}.{param}"
    for name, module in model.named_modules()
    for param, _ in batched_expert_params(module)
]
layers = json.loads(stats_path.read_text(encoding="utf-8")).get("layers", {})
print(f"stats file holds {len(layers)} modules")
print(f"expert-bank tensors this architecture has: {len(expected)}")
print(f"missing from the stats file: {len(missing)}")
for name in missing[:10]:
    print(f"  {name}")
raise SystemExit(1 if missing else 0)
