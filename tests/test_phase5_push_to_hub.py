"""The guards that run before two public repos exist, and the one that cannot be undone.

An upload is the only step in this campaign with no rollback that matters: a checkpoint
somebody has already pulled stays pulled, and a repo id that has been posted stays posted.
So the tests here are about refusals, not about rendering -- specifically about the three
mistakes that would survive every earlier check and become permanent:

* a directory that is not the model it claims to be (a cross-contaminated export),
* two arms holding **one** allocation, which is this campaign's shape of the arm-swap
  [phase4's pusher][1] documents as invisible to it, and
* a card generated against a repo id other than the one being pushed to, which sends every
  reader to a 404 they reach only after deciding to trust the numbers above it.

The module-loading test is not ceremony either. Both pushers are named ``push_to_hub.py``,
and a plain import would resolve by ``sys.path`` order into a second copy of the wrong file
-- a failure that looks exactly like the guards passing.

[1]: ../experiments/phase4/push_to_hub.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "phase5" / "push_to_hub.py"


@pytest.fixture(scope="module")
def push() -> Any:
    spec = importlib.util.spec_from_file_location("phase5_push_to_hub", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["phase5_push_to_hub"] = module
    spec.loader.exec_module(module)
    return module


def _write_dir(path: Path, *, bits: int, modules: int = 4, **config: Any) -> Path:
    """An exported arm on disk: a config with an allocation, and something shaped like weights."""
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "architectures": ["Qwen3_5ForCausalLM"],
        "vocab_size": 248_064,
        "quantization_config": {
            "quant_method": "dynquant",
            "modules": {f"model.layers.{i}.mlp.up_proj": {"bits": bits} for i in range(modules)},
        },
    }
    payload.update(config)
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    (path / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 32)
    return path


def _merge(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps({"architectures": ["Qwen3_5ForCausalLM"], "vocab_size": 248_064}),
        encoding="utf-8",
    )
    return path


def _inputs(tmp_path: Path) -> dict[str, str]:
    """The four records the card generator reads, with ``{arm}`` paths that resolve.

    One export record and one eval record serve both arms: the ids under test are the repo
    ids, and giving each arm its own copy of an identical file would only make the fixture
    longer.
    """
    payloads = {
        "finetune": {
            "regime": "qlora",
            "lora_rank": 32,
            "epochs": 1,
            "lr": 1e-4,
            "effective_batch": 16,
            "steps": 625,
            "train_loss": 0.086,
            "tracked_modules": 654,
            "conversations_kept": 9999,
            "supervised_tokens": 350_799,
            "train_sources": ["spider", "gretel", "wikisql", "create-context"],
        },
        "export": {
            "dynquant_core": "0.5.2",
            "average_bits": 3.0,
            "modules": 498,
            "directory_nbytes": 10_085_628_928,
            "group_size": 128,
        },
        "eval": {
            "label": "arm",
            "accuracy": 0.41,
            "total": 400,
            "correct": 164,
            "max_new_tokens": 1024,
            "unfinished_reasoning": 0,
            "sources": ["spider", "gretel", "wikisql", "create-context"],
        },
        "inspect": {
            "allocator": "rank_product",
            "group_size": 128,
            "targets": {
                "3.00": {
                    "average_bits": 2.9998,
                    "total_params": 26_893_352_960,
                    "widths": {"3": {"modules": 290, "params": 19_991_101_440}},
                    "violations": [
                        {
                            "name": "lm_head",
                            "role": "lm_head",
                            "floor_bits": 8,
                            "assigned_bits": 3,
                            "num_params": 1_271_398_400,
                        }
                    ],
                },
                "4.02": {
                    "average_bits": 4.019577,
                    "total_params": 26_893_352_960,
                    "widths": {"4": {"modules": 265, "params": 13_749_452_800}},
                    "violations": [],
                },
            },
        },
    }
    paths = {}
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = str(path)
    return paths


def _arms(push: Any, tmp_path: Path, dq4: Path, dq3: Path) -> list[Any]:
    return [
        push.Arm(label="dq4", source=dq4, target="4.02", repo="Org/model-dq4"),
        push.Arm(label="dq3", source=dq3, target="3.00", repo="Org/model-dq3"),
    ]


def test_phase4s_guards_are_the_ones_that_run(push: Any) -> None:
    """Loaded from phase4's file, not from a same-named copy of this one.

    Both scripts are ``push_to_hub.py``. Under a plain import the winner is whichever
    directory sits earlier on ``sys.path``, and this file's own module is ``__main__`` when
    it runs as a script -- so the import can quietly load a second copy of the phase5 file
    and call *its* helpers believing they are phase4's. Nothing would raise.

    Turns red when: the explicit path load is replaced by an import statement.
    """
    assert push.phase4.__file__ is not None
    assert Path(push.phase4.__file__).parent.name == "phase4"
    assert hasattr(push.phase4, "occupied") and hasattr(push.phase4, "upload")


def test_a_directory_that_is_a_different_model_is_refused(push: Any, tmp_path: Path) -> None:
    """Same exporter, same container, different checkpoint -- invisible once uploaded.

    A `vocab_size` that disagrees with the merge is what an export pass pointed at the wrong
    run root leaves behind, and this campaign has already paid once for a path that resolved
    somewhere other than where it was read: four Mistral arms landed in a Qwen directory with
    a Qwen tokenizer, and nothing said so.

    Turns red when: the fields checked against the merge stop being checked.
    """
    merged = _merge(tmp_path / "merged")
    dq4 = _write_dir(tmp_path / "dq4", bits=4)
    dq3 = _write_dir(tmp_path / "dq3", bits=3, vocab_size=151_936)

    problems = push.check(_arms(push, tmp_path, dq4, dq3), merged)
    assert any("vocab_size" in p for p in problems), problems


def test_two_arms_holding_one_allocation_are_refused(push: Any, tmp_path: Path) -> None:
    """Two repos, two claimed budgets, one set of weights.

    phase4 states plainly that it cannot see an arm swap: six arms written by one exporter
    into one container are indistinguishable from each other. That reasoning does not carry
    here, and the difference is worth a guard -- there are two arms, they are supposed to
    hold different budgets, and identical maps mean one export wrote into the other's
    directory or both ran at the same target. Either way one of the two public cards would
    describe a model that does not exist.

    Turns red when: the comparison starts looking at the average bits rather than the map,
    which two runs at one target would agree on just as happily.
    """
    merged = _merge(tmp_path / "merged")
    dq4 = _write_dir(tmp_path / "dq4", bits=4)
    dq3 = _write_dir(tmp_path / "dq3", bits=4)

    problems = push.check(_arms(push, tmp_path, dq4, dq3), merged)
    assert any("identical allocations" in p for p in problems), problems

    # And the honest case stays quiet, or the guard is one nobody keeps.
    _write_dir(tmp_path / "dq3", bits=3)
    assert push.check(_arms(push, tmp_path, dq4, tmp_path / "dq3"), merged) == []


def test_a_shard_above_the_hubs_limit_is_refused_before_the_upload(
    push: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Because the alternative is finding out partway through 13 GB.

    Turns red when: the ceiling is checked in GiB against a limit written in GB, or not at all.
    """
    merged = _merge(tmp_path / "merged")
    dq4 = _write_dir(tmp_path / "dq4", bits=4)
    dq3 = _write_dir(tmp_path / "dq3", bits=3)
    # The limit is shrunk rather than a 50 GB file written, which is the only way to
    # exercise this on a machine anyone runs tests on.
    monkeypatch.setattr(push, "MAX_FILE_NBYTES", 8)
    problems = push.check(_arms(push, tmp_path, dq4, dq3), merged)
    assert any("per-file limit" in p for p in problems), problems


def test_an_arm_without_a_budget_is_refused(push: Any, tmp_path: Path) -> None:
    """``--arm`` and ``--target`` name the same set, or the card describes another budget.

    Turns red when: a missing target starts defaulting to some row instead of raising.
    """
    with pytest.raises(SystemExit, match="needs the budget"):
        push.main(
            [
                "--arm",
                "dq4=/nowhere",
                "--arm",
                "dq3=/nowhere",
                "--target",
                "dq4=4.02",
                "--merged",
                "/nowhere",
                "--finetune",
                "/nowhere",
                "--export-record",
                "/nowhere",
                "--eval",
                "/nowhere",
                "--inspect",
                "/nowhere",
                "--base-model",
                "Qwen/Qwen3.8-27B",
                "--repo-prefix",
                "Org/model",
                "--dry-run",
            ]
        )


def test_the_card_is_generated_against_the_repo_it_is_pushed_to(push: Any, tmp_path: Path) -> None:
    """One string builds the repo id and reaches the card, so the two cannot disagree.

    A card carries its own load snippet. Generated under one prefix and pushed under
    another, it tells every reader to open a repo that 404s -- reached only after they have
    decided to trust the numbers above it. Reading a card off disk instead would make the
    match a hope; building both from one string makes it structural.

    Driven end to end through ``main --dry-run`` rather than by reading ``render``'s source,
    because the property is that the id in the file equals the id in the repo, and a source
    assertion would go green on a reformat while saying nothing about either.

    Turns red when: the card's repo id stops coming from the ``Arm`` that is pushed.
    """
    merged = _merge(tmp_path / "merged")
    dq4 = _write_dir(tmp_path / "dq4", bits=4)
    dq3 = _write_dir(tmp_path / "dq3", bits=3)
    inputs = _inputs(tmp_path)

    code = push.main(
        [
            "--arm",
            f"dq4={dq4}",
            "--arm",
            f"dq3={dq3}",
            "--target",
            "dq4=4.02",
            "--target",
            "dq3=3.00",
            "--merged",
            str(merged),
            "--finetune",
            inputs["finetune"],
            "--export-record",
            inputs["export"],
            "--eval",
            inputs["eval"],
            "--inspect",
            inputs["inspect"],
            "--base-model",
            "Qwen/Qwen3.8-27B",
            "--repo-prefix",
            "Org/qwen38-27b-text2sql-DynQuant",
            "--dry-run",
        ]
    )
    assert code == 0

    for label in ("dq3", "dq4"):
        card = (tmp_path / label / "README.md").read_text(encoding="utf-8")
        assert f"Org/qwen38-27b-text2sql-DynQuant-{label}" in card
        # And not the *other* arm's id, which is what a shared render would leak.
        other = "dq4" if label == "dq3" else "dq3"
        assert f"DynQuant-{other}" not in card
