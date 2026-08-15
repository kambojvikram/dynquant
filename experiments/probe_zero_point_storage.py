"""Does a symmetric checkpoint store a zero point? Measured, not read.

Quantizes one tiny model twice -- symmetric and asymmetric, everything else held -- saves
both with `save_compressed=True`, and lists what actually landed on disk. This is the
negative control for a correction that restates published byte counts: the arithmetic being
corrected is a *dependency's*, so reading its source establishes intent and only the file
establishes fact.
"""

import json
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _llmc import build_recipe
from llmcompressor import oneshot

HID, GS, BITS = 256, 128, 4


def tiny():
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-2B-Base")
    cfg.num_hidden_layers = 1
    cfg.hidden_size = HID
    cfg.intermediate_size = 2 * HID
    cfg.vocab_size = 512
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16)


def run(symmetric: bool, out: Path) -> dict:
    model = tiny()
    # RTN, so no calibration set and no processor: the question is what
    # `save_pretrained(save_compressed=True)` writes for each scheme, and the algorithm that
    # picked the values cannot change which tensors the format stores.
    oneshot(
        model=model,
        recipe=build_recipe("rtn", BITS, GS, ignore=["lm_head"], symmetric=symmetric),
    )
    model.save_pretrained(out, save_compressed=True)
    found = {}
    for f in sorted(out.glob("*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():  # noqa: SIM118 -- safe_open is not a Mapping
                t = h.get_slice(k)
                found[k] = (t.get_dtype(), list(t.get_shape()))
    return found


with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    sym = run(True, tmp / "sym")
    asym = run(False, tmp / "asym")

for label, found in (("SYMMETRIC", sym), ("ASYMMETRIC", asym)):
    zp = {k: v for k, v in found.items() if "zero_point" in k}
    sc = {k: v for k, v in found.items() if k.endswith("weight_scale")}
    print(f"--- {label}: {len(zp)} zero-point tensors, {len(sc)} scale tensors")
    for k, v in list(zp.items())[:2]:
        print(f"      {k}  dtype={v[0]} shape={v[1]}")
    for k, v in list(sc.items())[:2]:
        print(f"      {k}  dtype={v[0]} shape={v[1]}")

print()
print("keys only in asymmetric:", sorted(set(asym) - set(sym))[:6])
print("keys only in symmetric :", sorted(set(sym) - set(asym))[:6])

# And the arithmetic this correction turns on, checked against one real tensor.
name = next(k for k in asym if k.endswith("weight_packed"))
base = name[: -len("weight_packed")]
shape = asym[base + "weight_shape"]
print()
print(json.dumps({"probe_tensor": base, "packed_shape": asym[name][1]}, indent=1))
for label, found in (("sym", sym), ("asym", asym)):
    zpk = base + "weight_zero_point"
    print(
        f"  {label}: zero_point present={zpk in found} "
        f"{found.get(zpk, '')}  scale={found[base + 'weight_scale']}"
    )
