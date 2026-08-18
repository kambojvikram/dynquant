#!/usr/bin/env python3
"""Rewrite a Qwen3.5 vision-language checkpoint as the text tower alone.

Qwen ships the 27B as ``Qwen3_5ForConditionalGeneration``: one checkpoint holding a
text tower under ``model.language_model.*``, a 27-layer vision tower under
``model.visual.*``, an ``lm_head`` and a multi-token-prediction head under ``mtp.*``.
``Qwen3_5ForCausalLM`` -- the class the fine-tune, the signal tracker, the allocator
and the quantizer all expect -- reads ``model.*``. The two never meet, and the way
they fail to meet is the problem: ``from_pretrained`` answers an all-missing state
dict with a logged table and a randomly initialised model, not an exception. A run
started that way trains, evaluates and quantizes a model that never held Qwen's
weights, and every number it produces is internally consistent.

So the conversion is done once, explicitly, and checked by name rather than by
count: every tensor the target class declares must be present in what was written,
and nothing else may be. ``--verify-only`` re-runs that check against an existing
output directory without rewriting it.

What is dropped is dropped deliberately:

``model.visual.*``
    The vision tower. The point of the exercise -- the campaign fine-tunes and
    quantizes the text-to-text side.

``mtp.*``
    The multi-token-prediction head. ``Qwen3_5ForCausalLM`` does not declare it
    (a census of the class built from ``text_config`` shows 64 decoder layers and
    no ``mtp``), so carrying the tensors would produce an UNEXPECTED table on every
    load -- the same printed-not-raised signal this script exists to eliminate.
    Speculative decoding is the thing given up, not any generation quality.

Usage::

    python scripts/extract_text_tower.py --repo Qwen/Qwen3.8-27B --out /workspace/models/qwen38-27b-text
    python scripts/extract_text_tower.py --out /workspace/models/qwen38-27b-text --verify-only
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

#: Checkpoint prefix rewrites, longest first. A key matching none of these and none
#: of `DROP_PREFIXES` is an error rather than a passthrough: this script's whole job
#: is to be exhaustive about a name mapping, and a silent passthrough is how a tensor
#: ends up in the output under a name nothing reads.
RENAME_PREFIXES: tuple[tuple[str, str], ...] = (
    ("model.language_model.", "model."),
    ("lm_head.", "lm_head."),
)

DROP_PREFIXES: tuple[str, ...] = ("model.visual.", "mtp.")

#: Copied verbatim when present. The vision and video preprocessor configs are
#: deliberately absent: a text-only directory that carries them makes `AutoProcessor`
#: build an image processor for a model with no vision tower.
SIDECAR_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "generation_config.json",
    "LICENSE",
)

SHARD_TEMPLATE = "model-{index:05d}-of-{total:05d}.safetensors"


def text_config(source_config: dict[str, Any]) -> dict[str, Any]:
    """The ``text_config`` block, promoted to a top-level causal-LM config.

    ``architectures`` is rewritten rather than dropped: it is what a bare
    ``AutoModelForCausalLM.from_pretrained`` on the output directory dispatches on,
    and leaving the conditional-generation class there would send the loader back to
    the model this directory exists to escape.
    """
    inner = source_config.get("text_config")
    if not isinstance(inner, dict):
        raise SystemExit(
            "the source config has no `text_config` block, so it is not a Qwen3.5 VLM "
            "checkpoint -- nothing to extract"
        )
    out = dict(inner)
    out["architectures"] = ["Qwen3_5ForCausalLM"]
    # `tie_word_embeddings` lives in both blocks on this repo and they agree. Read it
    # from the outer one anyway: it is the one that described the checkpoint being
    # read, and a tie asserted by a config the loader no longer sees is exactly the
    # class of error where storage and config part company.
    out["tie_word_embeddings"] = bool(source_config.get("tie_word_embeddings", False))
    for key in ("transformers_version", "dtype", "torch_dtype"):
        if key in source_config and key not in out:
            out[key] = source_config[key]
    return out


def target_keys(config: dict[str, Any]) -> set[str]:
    """Every parameter and persistent buffer ``Qwen3_5ForCausalLM`` declares.

    Built on the meta device, so this costs no memory and no download -- the
    question is which *names* the class wants, and the answer does not need values.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.for_model(**config) if "model_type" in config else None
    if cfg is None:  # pragma: no cover -- every Qwen3.5 text config carries one
        raise SystemExit("the extracted config has no `model_type`")
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    keys = set(model.state_dict())
    if getattr(cfg, "tie_word_embeddings", False):
        # A tied head is one storage under two names; the checkpoint stores it once
        # and the loader re-ties on load, so demanding both names would fail a
        # correct file. Derived from the config here because there is no storage to
        # ask -- the model is on meta.
        keys.discard("lm_head.weight")
    return keys


def rename(key: str) -> str | None:
    """The output name for a checkpoint key, or ``None`` if it is dropped."""
    for prefix in DROP_PREFIXES:
        if key.startswith(prefix):
            return None
    for old, new in RENAME_PREFIXES:
        if key.startswith(old):
            return new + key[len(old) :]
    return ""  # unmapped: the caller turns this into an error naming the key


def convert(source: Path, out: Path) -> dict[str, str]:
    """Stream the shards, rewriting names. Returns the written weight map."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    index_path = source / "model.safetensors.index.json"
    if index_path.is_file():
        shards = sorted(
            {
                source / name
                for name in json.loads(index_path.read_text("utf-8"))["weight_map"].values()
            }
        )
    else:
        shards = sorted(source.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no safetensors shards under {source}")

    out.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    total_bytes = 0
    written = 0
    unmapped: list[str] = []

    for shard in shards:
        tensors: dict[str, Any] = {}
        with safe_open(str(shard), framework="pt") as handle:
            for key in handle.keys():  # noqa: SIM118 -- safe_open has no __iter__
                new = rename(key)
                if new is None:
                    continue
                if new == "":
                    unmapped.append(key)
                    continue
                tensors[new] = handle.get_tensor(key)
        if not tensors:
            print(f"  {shard.name}: nothing to keep", flush=True)
            continue
        written += 1
        name = SHARD_TEMPLATE.format(index=written, total=len(shards))
        save_file(tensors, str(out / name), metadata={"format": "pt"})
        for key, tensor in tensors.items():
            weight_map[key] = name
            total_bytes += tensor.numel() * tensor.element_size()
        print(f"  {shard.name} -> {name}: {len(tensors)} tensors", flush=True)
        del tensors

    if unmapped:
        raise SystemExit(
            f"{len(unmapped)} checkpoint key(s) matched neither a rename nor a drop, "
            f"e.g. {unmapped[:5]}. Add them to RENAME_PREFIXES or DROP_PREFIXES rather "
            f"than letting them through unnamed."
        )

    # Renumber: shards that lost every tensor leave gaps, and `x-of-y` has to count
    # the files that exist or `from_pretrained` looks for ones that do not.
    final = {}
    names = sorted(set(weight_map.values()))
    for position, old_name in enumerate(names, start=1):
        new_name = SHARD_TEMPLATE.format(index=position, total=len(names))
        if new_name != old_name:
            (out / old_name).rename(out / new_name)
        final[old_name] = new_name
    weight_map = {k: final[v] for k, v in weight_map.items()}

    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_bytes}, "weight_map": weight_map}, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    return weight_map


def verify(out: Path) -> int:
    """Assert the written names are exactly the names the target class declares."""
    config = json.loads((out / "config.json").read_text("utf-8"))
    index = json.loads((out / "model.safetensors.index.json").read_text("utf-8"))
    present = set(index["weight_map"])
    wanted = target_keys(config)

    missing = sorted(wanted - present)
    unexpected = sorted(present - wanted)
    print(f"declared {len(wanted)} | written {len(present)}", flush=True)
    for label, names in (("MISSING", missing), ("UNEXPECTED", unexpected)):
        if names:
            print(f"{label}: {len(names)}", flush=True)
            for name in names[:20]:
                print(f"   {name}", flush=True)
    if missing or unexpected:
        print("VERIFY FAILED", flush=True)
        return 1
    print("VERIFY OK -- every declared tensor is present and nothing else is", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo", default="Qwen/Qwen3.8-27B", help="Hub repo to fetch when --source is absent"
    )
    parser.add_argument("--source", type=Path, help="a checkpoint already on disk")
    parser.add_argument(
        "--out", type=Path, required=True, help="directory to write the text tower into"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="check an existing --out and exit"
    )
    args = parser.parse_args(argv)

    if args.verify_only:
        return verify(args.out)

    source = args.source
    if source is None:
        from huggingface_hub import snapshot_download

        print(f"fetching {args.repo}", flush=True)
        source = Path(snapshot_download(args.repo))
    print(f"source: {source}", flush=True)

    config = json.loads((source / "config.json").read_text("utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(
        json.dumps(text_config(config), indent=2), encoding="utf-8", newline="\n"
    )

    convert(source, args.out)

    for name in SIDECAR_FILES:
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, args.out / name)
            print(f"  copied {name}", flush=True)

    return verify(args.out)


if __name__ == "__main__":
    sys.exit(main())
