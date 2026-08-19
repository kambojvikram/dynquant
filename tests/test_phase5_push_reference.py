"""The refusals that stand between a correct third repo and a permanent wrong one.

The bf16 arm is the one whose source directory is not a DynQuant export, which makes it the
one most likely to be pointed somewhere else -- and the only arm whose mistakes the [arm
pusher's][1] guards do not cover, because every check in that file reads an allocation this
arm does not have.

Two of the tests here are about the same class of error from opposite sides: a check that is
too strict refuses a correct publish (the reshard changes the path, legitimately), and one
that is too loose passes a directory that has nothing to do with the number printed above it.

[1]: ../experiments/phase5/push_to_hub.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "phase5" / "push_reference.py"

FINETUNE = {
    "regime": "qlora",
    "lora_rank": 32,
    "epochs": 1.0,
    "lr": 1e-4,
    "effective_batch": 16,
    "steps": 625,
    "train_loss": 0.09632793507575989,
    "conversations_kept": 9999,
    "supervised_tokens": 350_799,
    "sources": {
        "text2sql/spider": 2500,
        "text2sql/gretel": 2500,
        "text2sql/wikisql": 2500,
        "text2sql/create-context": 2500,
    },
    "train_sources_requested": ["spider", "gretel", "wikisql", "create-context"],
    "sources_overlapping_an_eval_task": [],
    "decontaminated": {"spider": 2, "gretel": 2, "wikisql": 4, "create-context": 593},
}

EVAL = {
    "label": "merged-bf16",
    "accuracy": 0.855,
    "correct": 342,
    "total": 400,
    "unparseable": 0,
    "shots": 2,
    "decode": {"max_new_tokens": 1024, "greedy": True},
    "detail": {"unfinished_reasoning": 0},
    "task_options": {"sources": ["spider", "gretel", "wikisql"]},
}

PANEL = {
    "reference": "merged-bf16",
    "arms": [
        {
            "arm": "4bit",
            "accuracy": 0.8425,
            "comparison": {
                "delta_points": -1.2499999999999956,
                "ci_low_points": -2.870522678027057,
                "ci_high_points": 0.37052267802706584,
                "p_value": 0.2265625,
                "separated": False,
            },
        },
        {
            "arm": "3bit",
            "accuracy": 0.795,
            "comparison": {
                "delta_points": -5.999999999999995,
                "ci_low_points": -8.708773892372704,
                "ci_high_points": -3.2912261076272853,
                "p_value": 1.9301194697618484e-05,
                "separated": True,
            },
        },
    ],
}


@pytest.fixture(scope="module")
def push() -> Any:
    spec = importlib.util.spec_from_file_location("phase5_push_reference", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase5_push_reference"] = module
    spec.loader.exec_module(module)
    return module


def _dir(path: Path, *, shards: dict[str, int], quantization: dict | None = None) -> Path:
    """A model directory: a config, an index, and shards of the requested sizes."""
    path.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {"architectures": ["Qwen3_5ForCausalLM"], "vocab_size": 248_064}
    if quantization:
        config["quantization_config"] = quantization
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    weight_map = {}
    for i, (name, nbytes) in enumerate(shards.items()):
        (path / name).write_bytes(b"\0" * nbytes)
        weight_map[f"model.layers.{i}.mlp.up_proj.weight"] = name
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )
    return path


def _inputs(tmp_path: Path, **over: Any) -> dict[str, str]:
    paths = {}
    for name, payload in (
        ("finetune", over.get("finetune") or FINETUNE),
        ("eval", over.get("eval") or EVAL),
        ("panel", over.get("panel") or PANEL),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = str(path)
    return paths


def test_an_exported_arm_is_refused_behind_a_bf16_name(push: Any, tmp_path: Path) -> None:
    """``--source /workspace/q-4p0`` uploads perfectly well and is wrong by a whole arm.

    Nothing downstream notices: the weights load, the tokenizer matches, generation works.
    The only thing wrong is that a card claiming 85.50% unquantized sits above 4-bit weights,
    and every delta in its table is that arm measured against itself.

    Turns red when: the quantization_config check stops running for the reference arm.
    """
    source = _dir(
        tmp_path / "q-4p0",
        shards={"model-00001-of-00001.safetensors": 32},
        quantization={"quant_method": "dynquant", "modules": {"a": {"bits": 4}}},
    )
    problems = push.check(source, EVAL, PANEL)
    assert any("quantization_config" in p for p in problems), problems


def test_the_shard_that_fits_under_the_limit_is_still_refused(push: Any, tmp_path: Path) -> None:
    """49.83 GB clears the Hub's 50 GB ceiling by 0.17 GB, and that is the whole problem.

    A limit check passes it. Every future downloader then pulls one un-resumable file. The
    recommendation has to be a separate check from the ceiling or this arm ships as
    ``save_pretrained`` left it.

    Turns red when: the recommendation collapses into the limit, or is written in GiB against
    a value in GB.
    """
    source = _dir(tmp_path / "merged", shards={"model-00001-of-00002.safetensors": 64})
    push.RECOMMENDED_SHARD_NBYTES = 32  # a 64-byte shard stands in for 49.83 GB
    try:
        problems = push.check(source, EVAL, PANEL)
    finally:
        push.RECOMMENDED_SHARD_NBYTES = 5 * 1000**3
    assert any("max_shard_size" in p for p in problems), problems
    assert not any("per-file limit" in p for p in problems), problems


def test_a_reshard_is_not_mistaken_for_a_different_model(push: Any, tmp_path: Path) -> None:
    """The correct publish must pass, or the check gets deleted the first time it fires.

    The directory being uploaded is deliberately not the directory that was scored -- it was
    rewritten into 13 shards from 2. A path comparison would refuse this, so what is compared
    is the tensor set and the total bytes.

    Turns red when: the check compares paths, shard counts, or per-file sizes.
    """
    original = _dir(
        tmp_path / "merged",
        shards={"model-00001-of-00002.safetensors": 900, "model-00002-of-00002.safetensors": 100},
    )
    resharded = tmp_path / "merged-resharded"
    resharded.mkdir()
    (resharded / "config.json").write_text(
        (original / "config.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    names = list(json.loads((original / "model.safetensors.index.json").read_text())["weight_map"])
    for i in range(1, len(names) + 1):
        (resharded / f"model-{i:05d}-of-{len(names):05d}.safetensors").write_bytes(b"\0" * 500)
    (resharded / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    name: f"model-{i:05d}-of-{len(names):05d}.safetensors"
                    for i, name in enumerate(names, start=1)
                }
            }
        ),
        encoding="utf-8",
    )

    scored = {**EVAL, "model": str(original)}
    assert push.same_weights(resharded, scored) is None
    assert push.check(resharded, scored, PANEL) == []


def test_a_directory_that_is_not_what_was_scored_is_refused(push: Any, tmp_path: Path) -> None:
    """The other side of the same check: content that does not match, at a matching-ish size.

    This is the failure the loose version of the check lets through -- a directory assembled
    from a different run, published under an accuracy it never earned.

    Turns red when: the tensor-set comparison is dropped in favour of counting files or bytes.
    """
    original = _dir(tmp_path / "merged", shards={"model-00001-of-00001.safetensors": 500})
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(
        (original / "config.json").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (other / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 500)
    (other / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.99.self_attn.q_proj.weight": "model-00001-of-00001.safetensors"
                }
            }
        ),
        encoding="utf-8",
    )
    problems = push.check(other, {**EVAL, "model": str(original)}, PANEL)
    assert any("was scored with" in p for p in problems), problems


def test_a_missing_original_does_not_refuse_the_publish(push: Any, tmp_path: Path) -> None:
    """`/workspace` is not persistent, so the scored directory may simply be gone.

    An absent original is not evidence of a mismatch. Refusing on it would make this the
    check people pass ``--force`` to rather than the one they keep.

    Turns red when: a missing scored directory starts being treated as a failed comparison.
    """
    source = _dir(tmp_path / "merged", shards={"model-00001-of-00001.safetensors": 500})
    assert push.same_weights(source, {**EVAL, "model": "/workspace/gone"}) is None


def test_a_panel_that_scored_another_reference_is_refused(push: Any, tmp_path: Path) -> None:
    """Every delta in the table is against the panel's reference, not against this card's arm.

    Turns red when: the panel is rendered without checking whose reference it is.
    """
    source = _dir(tmp_path / "merged", shards={"model-00001-of-00001.safetensors": 500})
    problems = push.check(source, EVAL, {**PANEL, "reference": "some-other-run"})
    assert any("panel's reference" in p for p in problems), problems


def test_the_table_carries_the_sizes_that_make_it_worth_reading(push: Any, tmp_path: Path) -> None:
    """A comparison table without the sizes is a table saying "bf16 wins".

    The whole argument for the quantized arms is accuracy *per byte*, so the size column is
    not decoration -- and it comes from each arm's own export record, because those
    directories are not on the machine that renders this card.

    Turns red when: the peer's export record stops being read and the column falls back to a
    dash, which renders as a perfectly tidy table that has lost the point.
    """
    source = _dir(tmp_path / "merged", shards={"model-00001-of-00001.safetensors": 500})
    for label, nbytes in (("4bit", 13_463_150_592), ("3bit", 10_105_663_488)):
        (tmp_path / f"export-{label}.json").write_text(
            json.dumps({"directory_nbytes": nbytes}), encoding="utf-8"
        )
    inputs = _inputs(tmp_path)

    code = push.main(
        [
            "--source",
            str(source),
            "--finetune",
            inputs["finetune"],
            "--eval",
            inputs["eval"],
            "--panel",
            inputs["panel"],
            "--peer",
            f"4bit=Org/model-4bit={tmp_path / 'export-4bit.json'}",
            "--peer",
            f"3bit=Org/model-3bit={tmp_path / 'export-3bit.json'}",
            "--base-model",
            "Qwen/Qwen3.8-27B",
            "--repo",
            "Org/model-bf16",
            "--dry-run",
        ]
    )
    assert code == 0
    card = (source / "README.md").read_text(encoding="utf-8")
    assert "12.54 GiB" in card, card
    assert "9.41 GiB" in card, card
    assert "Org/model-4bit" in card and "Org/model-3bit" in card


def test_the_card_reports_the_interval_and_not_only_the_verdict(push: Any, tmp_path: Path) -> None:
    """The verdict is the sentence a reader will over-read, so the CI has to be next to it.

    p = 0.2266 at 400 items means the test could not tell the arms apart. It does not mean
    they are the same, and the interval [-2.87, +0.37] is what says so. A card that prints the
    verdict alone is making a stronger claim than the run supports.

    Turns red when: the interval is dropped, or a tiny p-value renders as 0.0000 and reads as
    exactly zero.
    """
    source = _dir(tmp_path / "merged", shards={"model-00001-of-00001.safetensors": 500})
    inputs = _inputs(tmp_path)
    assert (
        push.main(
            [
                "--source",
                str(source),
                "--finetune",
                inputs["finetune"],
                "--eval",
                inputs["eval"],
                "--panel",
                inputs["panel"],
                "--base-model",
                "Qwen/Qwen3.8-27B",
                "--repo",
                "Org/model-bf16",
                "--dry-run",
            ]
        )
        == 0
    )
    card = (source / "README.md").read_text(encoding="utf-8")
    assert "[-2.87, +0.37]" in card, card
    assert "1.93e-05" in card, card
    assert "0.0000" not in card, card
    assert "not** a claim" in card, card


def test_a_token_in_the_arguments_stops_the_run(push: Any) -> None:
    """Before the parser, because argparse echoes an unknown flag *and its value* to stderr.

    Turns red when: the scan moves below parse_args, where the error that refuses the flag is
    itself the disclosure.
    """
    with pytest.raises(SystemExit, match="compromised"):
        push.main(["--token", "hf_averyrealtoken", "--source", "/nowhere"])


def test_the_upload_path_is_constructed_the_way_phase4_expects(
    push: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every other test here stops at --dry-run, which is where this bug lived.

    ``phase4.Push`` takes a ``kind``. Omitting it raises TypeError on the line that runs
    *only* on the irreversible path -- after the card is written, after the guards pass, with
    the token already in the environment. A suite that exercises only the dry run reports
    green on a script that cannot upload at all.

    ``kind`` also has to be CEILING specifically, not any string: phase4's arm check branches
    on it to refuse a quantization_config on this arm, and any other value routes the bf16
    merge into the branch that demands one.

    Turns red when: the upload path is constructed with the wrong arity or the wrong kind.
    """
    source = _dir(tmp_path / "merged", shards={"model-00001-of-00001.safetensors": 500})
    inputs = _inputs(tmp_path)
    seen: dict[str, Any] = {}

    monkeypatch.setenv("HF_TOKEN", "not-a-real-token")
    monkeypatch.setattr(push.phase4, "occupied", lambda pushes, token: [])
    monkeypatch.setattr(
        push.phase4,
        "upload",
        lambda p, card, token, *, private: (
            seen.update(push=p, private=private, card=card)
            or "https://huggingface.co/Org/model-bf16"
        ),
    )

    code = push.main(
        [
            "--source",
            str(source),
            "--finetune",
            inputs["finetune"],
            "--eval",
            inputs["eval"],
            "--panel",
            inputs["panel"],
            "--base-model",
            "Qwen/Qwen3.8-27B",
            "--repo",
            "Org/model-bf16",
        ]
    )
    assert code == 0
    assert seen["push"].kind == push.phase4.model_cards.CEILING
    assert seen["push"].repo == "Org/model-bf16"
    assert seen["private"] is False
    assert "85.50%" in seen["card"]
