"""The VLM-to-text-tower conversion has to be exhaustive about names, not approximate.

The failure it exists to prevent is silent: ``from_pretrained`` answers a state dict
whose every key is wrong with a logged MISSING table and a randomly initialised model.
A conversion that drops a tensor, or carries one through under a name the target class
does not read, therefore produces a model that trains, evaluates and quantizes without
raising anything -- and none of its numbers describe Qwen's weights.

So the tests here are about names. The load-bearing one is the negative control: a key
matching neither a rename nor a drop must stop the conversion rather than pass through,
because a passthrough is precisely how an unnamed tensor reaches the output. Nothing
here downloads a checkpoint; ``convert`` is exercised against a synthetic one holding
one tensor from each of the four key families the real repo has.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "extract_text_tower.py"


@pytest.fixture(scope="module")
def extract():
    spec = importlib.util.spec_from_file_location("_dq_extract", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dq_extract"] = module
    spec.loader.exec_module(module)
    return module


# The four families Qwen/Qwen3.8-27B's index actually contains, one key each.
TEXT_KEY = "model.language_model.layers.0.self_attn.q_proj.weight"
HEAD_KEY = "lm_head.weight"
VISION_KEY = "model.visual.blocks.0.attn.qkv.weight"
MTP_KEY = "mtp.layers.0.input_layernorm.weight"


def test_the_text_tower_loses_its_language_model_prefix(extract):
    assert extract.rename(TEXT_KEY) == "model.layers.0.self_attn.q_proj.weight"


def test_the_head_is_carried_through_unchanged(extract):
    assert extract.rename(HEAD_KEY) == HEAD_KEY


@pytest.mark.parametrize("key", [VISION_KEY, MTP_KEY])
def test_the_vision_tower_and_the_mtp_head_are_dropped(extract, key):
    assert extract.rename(key) is None


def test_an_unmapped_key_is_not_a_passthrough(extract):
    """The negative control.

    ``model.audio_tower.…`` is not a Qwen3.5 key; it stands for the next family Qwen
    adds. The point is that ``rename`` refuses to guess -- were the fallback ``return
    key``, a tensor under a name ``Qwen3_5ForCausalLM`` never reads would land in the
    output and the only symptom would be an UNEXPECTED line in a log nobody reads.
    """
    assert extract.rename("model.audio_tower.layers.0.weight") == ""


def test_the_promoted_config_names_the_causal_lm(extract):
    promoted = extract.text_config(
        {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "tie_word_embeddings": False,
            "text_config": {"model_type": "qwen3_5_text", "num_hidden_layers": 64},
            "vision_config": {"depth": 27},
        }
    )
    assert promoted["architectures"] == ["Qwen3_5ForCausalLM"]
    assert promoted["model_type"] == "qwen3_5_text"
    assert promoted["num_hidden_layers"] == 64
    assert "vision_config" not in promoted


def test_a_config_without_a_text_block_is_refused(extract):
    with pytest.raises(SystemExit):
        extract.text_config({"architectures": ["LlamaForCausalLM"], "model_type": "llama"})


def _write_source(tmp_path: Path, keys: list[list[str]]) -> Path:
    """A synthetic multi-shard checkpoint, one 2x2 tensor per key."""
    source = tmp_path / "src"
    source.mkdir()
    weight_map = {}
    for index, shard_keys in enumerate(keys, start=1):
        name = f"model-{index:05d}-of-{len(keys):05d}.safetensors"
        save_file(
            {key: torch.zeros(2, 2, dtype=torch.bfloat16) for key in shard_keys},
            str(source / name),
            metadata={"format": "pt"},
        )
        for key in shard_keys:
            weight_map[key] = name
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}), encoding="utf-8"
    )
    return source


def test_convert_writes_the_renamed_keys_and_only_those(extract, tmp_path):
    source = _write_source(tmp_path, [[TEXT_KEY, VISION_KEY], [HEAD_KEY, MTP_KEY]])
    out = tmp_path / "out"
    written = extract.convert(source, out)

    assert set(written) == {"model.layers.0.self_attn.q_proj.weight", HEAD_KEY}
    index = json.loads((out / "model.safetensors.index.json").read_text("utf-8"))
    assert set(index["weight_map"]) == set(written)
    # 2 tensors x 2x2 x bf16
    assert index["metadata"]["total_size"] == 16


def test_a_shard_that_loses_every_tensor_leaves_no_gap(extract, tmp_path):
    """``x-of-y`` has to count the files that exist.

    Shard 2 here is all vision, so it writes nothing. If the surviving shards kept
    their original ordinals the index would name ``model-00003-of-00003`` while only
    two files sit on disk, and ``from_pretrained`` opens files by the name in the
    index -- so the whole load fails on a file that was never meant to exist.
    """
    source = _write_source(tmp_path, [[TEXT_KEY], [VISION_KEY], [HEAD_KEY]])
    out = tmp_path / "out"
    extract.convert(source, out)

    index = json.loads((out / "model.safetensors.index.json").read_text("utf-8"))
    named = set(index["weight_map"].values())
    on_disk = {path.name for path in out.glob("*.safetensors")}
    assert (
        named == on_disk == {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
    )


def test_convert_refuses_a_checkpoint_holding_an_unmapped_family(extract, tmp_path):
    source = _write_source(tmp_path, [[TEXT_KEY, "model.audio_tower.layers.0.weight"]])
    with pytest.raises(SystemExit, match="neither a rename nor a drop"):
        extract.convert(source, tmp_path / "out")


def test_verify_fails_when_a_declared_tensor_is_absent(extract, tmp_path, monkeypatch):
    """``verify`` is a set comparison, so give it sets and check both directions.

    ``target_keys`` builds a real model to answer "which names does the class want";
    stubbing it here keeps the test off the meta device and, more to the point, makes
    the assertion about ``verify``'s own logic rather than about transformers'.
    """
    out = tmp_path / "out"
    out.mkdir()
    (out / "config.json").write_text(json.dumps({"model_type": "qwen3_5_text"}), encoding="utf-8")
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.a": "s.safetensors", "model.extra": "s.safetensors"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(extract, "target_keys", lambda config: {"model.a", "model.b"})
    assert extract.verify(out) == 1  # model.b missing, model.extra unexpected

    monkeypatch.setattr(extract, "target_keys", lambda config: {"model.a", "model.extra"})
    assert extract.verify(out) == 0
