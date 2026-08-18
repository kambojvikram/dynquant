"""The card that ships with the Qwen3.8-27B arms, and the two claims it must not soften.

A model card is the most-read and least-verified document a campaign produces: it is prose,
nothing recomputes it, and the reader has no way to check it before spending a download. So
the tests here are not about formatting. They are about the two sentences that decide
whether a reader is misled --

* whether the arm sits **below the architecture's floor budget**, which makes its score a
  measurement of override damage rather than of an allocation, and
* whether the **registration call** appears, without which ``from_pretrained`` returns a
  randomly initialised model that generates fluent text and raises nothing.

Both have a failure mode of quietly disappearing: the first when a branch stops firing, the
second when a code block is reflowed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "phase5" / "model_card.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def card() -> Any:
    return _load("phase5_model_card", SCRIPT)


#: Two budgets from the same sweep, shaped like the ones this model really produced: one
#: that breaks nothing and one that breaks a great deal. The floor budget is whatever the
#: cleanest row measured, so the fixture never states it -- which is the property under
#: test, since a hardcoded floor budget is exactly the number that goes stale.
INSPECT: dict[str, Any] = {
    "allocator": "rank_product",
    "group_size": 128,
    "targets": {
        "3.00": {
            "average_bits": 2.9998,
            "total_params": 26_893_352_960,
            "widths": {
                "2": {"modules": 112, "params": 6_878_658_560},
                "3": {"modules": 290, "params": 19_991_101_440},
                "8": {"modules": 96, "params": 23_592_960},
            },
            "violations": [
                {
                    "name": "lm_head",
                    "role": "lm_head",
                    "floor_bits": 8,
                    "assigned_bits": 3,
                    "num_params": 1_271_398_400,
                },
                {
                    "name": "model.layers.0.mlp.gate_proj",
                    "role": "mlp.gate",
                    "floor_bits": 4,
                    "assigned_bits": 2,
                    "num_params": 89_128_960,
                },
            ],
        },
        "4.02": {
            "average_bits": 4.019577,
            "total_params": 26_893_352_960,
            "widths": {
                "3": {"modules": 135, "params": 11_759_779_840},
                "4": {"modules": 265, "params": 13_749_452_800},
                "8": {"modules": 97, "params": 1_294_991_360},
            },
            "violations": [],
        },
    },
}

FINETUNE = {
    "regime": "qlora",
    "lora_rank": 32,
    "epochs": 1,
    "lr": 1e-4,
    "effective_batch": 16,
    "steps": 625,
    "train_loss": 0.0864,
    "tracked_modules": 654,
    "conversations_kept": 9999,
    "supervised_tokens": 350_799,
    "train_sources": ["spider", "gretel", "wikisql", "create-context"],
}

EVAL = {
    "label": "dq3",
    "accuracy": 0.4125,
    "total": 400,
    "correct": 165,
    "max_new_tokens": 1024,
    "unfinished_reasoning": 0,
    "sources": ["spider", "gretel", "wikisql", "create-context"],
}


def _write(tmp_path: Path, **overrides: Any) -> dict[str, str]:
    """Lay the four inputs on disk the way the run leaves them."""
    export = {
        "dynquant_core": "0.5.2",
        "average_bits": 2.9998,
        "modules": 498,
        "directory_nbytes": 10_085_628_928,
        "group_size": 128,
    }
    export.update(overrides.get("export", {}))
    files = {
        "finetune": FINETUNE,
        "export": export,
        "eval": {**EVAL, **overrides.get("eval", {})},
        "inspect": INSPECT,
    }
    paths = {}
    for name, payload in files.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = str(path)
    return paths


def _render(card: Any, tmp_path: Path, target: str, **overrides: Any) -> str:
    """Through ``main`` and a real argument list, because both are what runs.

    Calling the render helpers directly would step over the argument parsing and the
    refusal that lives behind it, and the refusal is one of the behaviours under test.
    """
    paths = _write(tmp_path, **overrides)
    out = tmp_path / f"README-{target}-{len(overrides)}.md"
    card.main(
        [
            "--arm",
            "dq3",
            "--repo",
            "Org/model",
            "--base-model",
            "Qwen/Qwen3.8-27B",
            "--finetune",
            paths["finetune"],
            "--export",
            paths["export"],
            "--eval",
            paths["eval"],
            "--inspect",
            paths["inspect"],
            "--inspect-target",
            target,
            "--out",
            str(out),
        ]
    )
    return out.read_text(encoding="utf-8")


def test_an_arm_below_the_floor_budget_says_so_before_it_says_its_score(
    card: Any, tmp_path: Path
) -> None:
    """The sentence that stops a number being read as something it is not.

    At 3.0 bits on this architecture a quarter of the parameters are at 2 bits and the LM
    head has fallen from its 8-bit floor to 3. The accuracy that comes out is a real
    measurement of a deliberately over-compressed model, and a card that prints it next to
    the bf16 number without that framing invites the reader to conclude the method costs
    what this arm costs.

    Turns red when: the branch stops firing, or the warning drifts below the results.
    """
    text = _render(card, tmp_path, "3.00")
    assert "floor budget" in text
    # Measured off the clean row, never stated by the fixture.
    assert "4.0196 bits" in text, text[:800]
    assert "1.02 bits below" in text
    assert text.index("Read this first") < text.index("## Results")


def test_the_arm_at_the_floor_budget_is_not_given_the_warning(card: Any, tmp_path: Path) -> None:
    """Because a warning that appears on every card is one nobody reads.

    Turns red when: the threshold widens far enough to catch a clean arm.
    """
    text = _render(
        card, tmp_path, "4.02", export={"average_bits": 4.0196, "directory_nbytes": 13_447_625_728}
    )
    assert "Read this first" not in text
    assert "clears every floor" in text


def test_every_breached_floor_reaches_the_card(card: Any, tmp_path: Path) -> None:
    """Aggregated by role, but no role dropped -- the LM head above all.

    Turns red when: the table starts truncating, which is how a 310-violation arm comes to
    look like a 5-violation one.
    """
    text = _render(card, tmp_path, "3.00")
    assert "`lm_head`" in text and "8b" in text
    assert "`mlp.gate`" in text
    assert "2 modules were allocated below the floor" in text


def test_the_registration_call_is_in_the_load_snippet(card: Any, tmp_path: Path) -> None:
    """Without it ``from_pretrained`` returns a randomly initialised model and raises nothing.

    transformers has no entry-point discovery for quantization methods, so a reader who
    copies a snippet missing this line gets fluent output from untrained weights. The card
    is the only place that defence can live.

    Turns red when: the snippet is reflowed and the call is lost with it.
    """
    text = _render(card, tmp_path, "3.00")
    assert "dynquant.register_hf_quantizer()" in text
    assert text.index("import dynquant") < text.index("from_pretrained")
    assert "randomly initialised" in text


def test_a_decode_budget_that_bound_is_reported_as_a_bound(card: Any, tmp_path: Path) -> None:
    """A generation that never finished is scored wrong, so the accuracy has a ceiling.

    On this task a 256-token cap once scored 5.50% where 1024 scored 57.75% on the same
    problems. If a run hits its cap the card has to say the number is partly a measurement
    of the cap.

    Turns red when: `unfinished_reasoning` stops reaching the page.
    """
    text = _render(card, tmp_path, "3.00", eval={"unfinished_reasoning": 40})
    assert "bounded above by 90.00%" in text

    clean = _render(card, tmp_path, "3.00")
    assert "bounded above" not in clean


def test_a_target_the_inspection_never_measured_is_refused(card: Any, tmp_path: Path) -> None:
    """Otherwise the allocation table describes a budget this arm was not exported at.

    Turns red when: the lookup starts defaulting to some row rather than raising.
    """
    with pytest.raises(SystemExit, match="no target"):
        _render(card, tmp_path, "3.50")
