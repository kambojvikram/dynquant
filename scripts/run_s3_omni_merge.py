#!/usr/bin/env python
"""S3 for Qwen3-Omni: fold the SLURP adapter into a bf16 Thinker and write it out.

Separate from training for the reason every QLoRA merge is: the adapter was fitted
against an NF4 base, and it is folded into the bf16 one. Those are not the same weights,
so the merge has to happen in its own process against a fresh load -- merging in-place at
the end of training would fold into the dequantized NF4 values and quietly ship a
different checkpoint than the one this writes.

**What comes out is the Thinker alone, not the whole Omni model.** The Talker and code2wav
are 3.02 B parameters that SLURP never runs, never gets a gradient for, and therefore has
no signal for. Writing them into the checkpoint would hand the allocator 3 B of parameters
priced by role floors alone and put them in the denominator of every average-bits number
the campaign reports. `dynquant quantize`, `eval`, `inspect` and `export` all already take
`--model-class`, so a Thinker-only directory is read back with

    --model-class Qwen3OmniMoeThinkerForConditionalGeneration

and nothing downstream needs a change.

The merge is verified rather than assumed. Every parameter is fingerprinted before and
after, and the run fails unless exactly the 384 LoRA-targeted projections moved and
nothing else did -- in particular not one of the 96 expert banks, which carry 91.4% of the
parameters and no adapter. A merge that silently touched them, or one that silently
touched nothing, both write a checkpoint that loads.

**The fingerprint samples values; it does not compare norms.** The first version of this
script fingerprinted each tensor by its L2 norm, and that is not a distance -- to first
order ``||W+D|| - ||W||`` is ``<W,D>/||W||``, so a delta that is real but near-orthogonal
to the base weight moves the norm only at second order. Measured on this very checkpoint:
all 384 targets had a genuine delta, ``||D||/||W||`` between 2.7e-3 and 1.2e-2 with 48% to
91% of the stored bf16 elements changing value, and yet in float64 the relative norm
change over the whole population ran from 7.7e-7 to 1.2e-4 -- a continuous band with no
gap, straddling any threshold one might pick. Nine of the 384 landed under 1e-6 and the
run aborted on a merge that had worked. Worse, the norms were reduced in fp32, whose
error on a 6.5 M-element tensor is ~1e-4 relative: a hundred times the signal. Comparing
sampled elements for inequality has no threshold, no reduction, and a categorical gap --
here 48%+ of the sample changed on every merged weight against exactly 0 everywhere else.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path
from typing import Any

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_s2_omni_finetune import (  # noqa: E402
    LORA_TARGETS,
    bank_parameters,
    load_thinker,
)

THINKER_CLASS = "Qwen3OmniMoeThinkerForConditionalGeneration"


SAMPLE_ELEMENTS = 4096


def sample_index(numel: int, name: str) -> Any:
    """The elements of ``name`` to fingerprint -- deterministic, so before and after agree.

    Seeded from the name so two different tensors sample different slots (a stride would
    correlate with row length and could land in one column of every tensor), and from the
    element count so a shape change cannot be mistaken for a value change.
    """
    import torch

    generator = torch.Generator().manual_seed(zlib.crc32(name.encode("utf-8")) ^ numel)
    return torch.randint(0, numel, (min(SAMPLE_ELEMENTS, numel),), generator=generator)


def fingerprint(model: Any) -> dict[str, tuple[int, Any]]:
    """``{name: (numel, sampled values)}`` for every parameter, one tensor at a time.

    A gather of 4096 elements, so the peak extra allocation is kilobytes rather than a
    copy of the largest tensor -- the previous norm fingerprint cast a 2.4 GiB expert
    bank to fp32 to produce one float. Values are kept exactly, promoted to fp32 on the
    host: bf16 widens losslessly, so ``!=`` between two fingerprints stays a bit-level
    comparison of what is actually stored, with no tolerance to choose.
    """
    import torch

    out: dict[str, tuple[int, Any]] = {}
    for name, param in model.named_parameters():
        with torch.no_grad():
            flat = param.detach().reshape(-1)
            index = sample_index(flat.numel(), base_name(name)).to(flat.device)
            values = flat[index].to(device="cpu", dtype=torch.float32).clone()
        out[name] = (param.numel(), values)
    return out


def changed_share(
    before: dict[str, tuple[int, Any]], after: dict[str, tuple[int, Any]], name: str
) -> float:
    """Share of ``name``'s sampled elements whose stored value differs. 0.0 means unmoved.

    This is the movement test. It is exact on the sample rather than approximate on the
    whole tensor, which is the right trade here: every failure mode worth catching --
    peft folding into an expert bank, or folding into nothing -- changes a large fraction
    of a tensor's elements, and a 4096-element sample sees a 0.1%-of-elements change with
    98% probability and a 50% change with certainty.
    """
    return float((after[name][1] != before[name][1]).double().mean().item())


def base_name(name: str) -> str:
    """The name a parameter has on the bare model, whatever peft wrapped it in.

    ``merge_and_unload`` returns the base model, so the after-fingerprint is keyed by the
    plain names and the before-fingerprint by the plain names too -- both are taken off
    unwrapped models. This exists for the one case that is not symmetric: a merged
    ``lora.Linear`` keeps its base weight at ``...base_layer.weight`` if anything went
    wrong and the adapter was not actually unloaded, and that has to be visible as a
    *missing* name rather than silently compared against nothing.
    """
    return name.replace(".base_layer.", ".")


def targeted(name: str) -> bool:
    """Whether a parameter belongs to a module LoRA was attached to.

    Matches how peft matched in the first place -- the module name ends in ``.<target>``
    -- and then takes the module's own ``weight``. A bias would also be a parameter of a
    targeted module, but ``bias="none"`` means no adapter touches one, so a moved bias is
    a real failure and must not be swallowed here.
    """
    if not name.endswith(".weight"):
        return False
    module = name[: -len(".weight")]
    return any(module.endswith(f".{target}") for target in LORA_TARGETS)


def merge(args: argparse.Namespace) -> int:
    import torch

    from dynquant.integration.peft_utils import merge_adapters

    adapter = Path(args.adapter)
    if not (adapter / "adapter_config.json").is_file():
        print(f"no adapter_config.json under {adapter}", flush=True)
        return 2

    started = time.time()
    model, census = load_thinker(
        args.model, four_bit=False, device={"": args.device}, dtype=torch.bfloat16
    )
    banks = bank_parameters(model)
    bank_names = set(banks.values())
    before = fingerprint(model)
    print(f"loaded bf16 thinker: {census['thinker_params'] / 1e9:.3f}B params", flush=True)

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model = merge_adapters(model, safe_merge=True)
    after = {base_name(name): value for name, value in fingerprint(model).items()}

    # --- the assertions the whole script exists for ------------------------------------
    missing = sorted(set(before) - set(after))
    extra = sorted(set(after) - set(before))
    if missing or extra:
        print(
            f"the merged model does not have the base model's parameters: {len(missing)} "
            f"missing (first {missing[:3]}), {len(extra)} unexpected (first {extra[:3]}). "
            f"An unexpected `...base_layer.weight` here means the adapter was not "
            f"unloaded and the checkpoint would still need peft to run.",
            flush=True,
        )
        return 3

    # `> 0.0` rather than a tolerance, because there is nothing to tolerate: a parameter
    # no adapter touches is the same tensor object with the same stored bits, so one
    # sampled element differing is already a defect. The printed separation below
    # confirms rather than assumes it -- if the smallest moved target ever approaches the
    # largest unmoved one, the sample has stopped discriminating and the number says so.
    def rel(name: str) -> float:
        return changed_share(before, after, name)

    moved = {name for name in before if rel(name) > 0.0}
    expected = {name for name in before if targeted(name)}
    unexpected_moves = sorted(moved - expected)
    unmoved_targets = sorted(expected - moved)

    if unexpected_moves:
        offenders = sorted(unexpected_moves)[:5]
        banked = [name for name in unexpected_moves if name in bank_names]
        print(
            f"MERGE MOVED {len(unexpected_moves)} PARAMETERS NO ADAPTER TOUCHED, "
            f"{len(banked)} of them expert banks. First: {offenders}",
            flush=True,
        )
        return 4
    if unmoved_targets:
        print(
            f"MERGE LEFT {len(unmoved_targets)} OF {len(expected)} TARGETED WEIGHTS "
            f"UNCHANGED, first {unmoved_targets[:5]}. A merge that folds nothing writes a "
            f"checkpoint identical to the base, which loads and evaluates as the base.",
            flush=True,
        )
        return 5

    delta = max(rel(name) for name in moved)
    floor = min(rel(name) for name in moved)
    ceiling = max((rel(name) for name in before if name not in moved), default=0.0)
    print(
        f"merge verified: {len(moved)} weights moved, all of them targeted; "
        f"{len(bank_names)} expert banks bit-identical; over the merged weights between "
        f"{floor:.1%} and {delta:.1%} of a {SAMPLE_ELEMENTS}-element sample changed value, "
        f"and over every other parameter {ceiling:.1%} did",
        flush=True,
    )

    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(destination), safe_serialization=True)

    from transformers import AutoProcessor

    AutoProcessor.from_pretrained(args.model).save_pretrained(str(destination))

    # `save_pretrained` writes whatever `model.config.architectures` says, and the eval
    # side resolves `--model-class` by that name off the `transformers` namespace. Reading
    # it back rather than trusting it, because a config that names the whole Omni class
    # over Thinker-only weights fails at load with a missing-key table, not an exception.
    written = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    architectures = written.get("architectures") or []
    if architectures != [THINKER_CLASS]:
        print(
            f"config.json says architectures={architectures}, expected [{THINKER_CLASS!r}]. "
            f"Downstream loads with --model-class {THINKER_CLASS}, so this has to agree.",
            flush=True,
        )
        return 6

    size = sum(f.stat().st_size for f in destination.glob("*.safetensors"))
    record = {
        **census,
        "base": args.model,
        "adapter": str(adapter),
        "output": str(destination),
        "model_class": THINKER_CLASS,
        "merged_weights": len(moved),
        "expert_banks_unchanged": len(bank_names),
        "fingerprint_sample_elements": SAMPLE_ELEMENTS,
        "max_sampled_change_merged": delta,
        "min_sampled_change_merged": floor,
        "max_sampled_change_unmerged": ceiling,
        "safetensors_bytes": size,
        "seconds": round(time.time() - started, 1),
    }
    (destination / "s3_omni_merge.json").write_text(
        json.dumps(record, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"-> wrote {destination} ({size / 1024**3:.1f} GiB) in {record['seconds']:.0f}s",
        flush=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", default="/workspace/models/qwen3-omni-30b")
    parser.add_argument("--adapter", default="/workspace/runs/omni-slurp/adapter")
    parser.add_argument("--out", default="/workspace/runs/omni-slurp/merged")
    parser.add_argument("--device", default=0, help="0 for cuda:0, or 'cpu'")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.device != "cpu":
        args.device = int(args.device)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
