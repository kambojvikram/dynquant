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
        # The budget the 4-bit arm is really exported at: 0.02 bits under the floor, which
        # is a handful of mild breaches rather than the wholesale override the 3.00 row is.
        "4.00": {
            "average_bits": 4.00496,
            "total_params": 26_893_352_960,
            "widths": {
                "3": {"modules": 138, "params": 11_884_249_088},
                "4": {"modules": 263, "params": 13_735_567_360},
                "8": {"modules": 96, "params": 1_273_536_512},
            },
            "violations": [
                {
                    "name": "model.layers.3.mlp.down_proj",
                    "role": "mlp.down",
                    "floor_bits": 3,
                    "assigned_bits": 2,
                    "num_params": 89_128_960,
                }
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
        "finetune": overrides.get("finetune") or FINETUNE,
        "export": export,
        "eval": overrides.get("eval_record") or {**EVAL, **overrides.get("eval", {})},
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
    extra = ["--arm", "dq3", "--panel", overrides["panel"]] if overrides.get("panel") else []
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
            *extra,
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


def test_the_arm_just_under_the_floor_budget_still_admits_what_it_broke(
    card: Any, tmp_path: Path
) -> None:
    """The case actually shipped, which neither of the two above it is.

    The 4-bit arm is exported at a 4.00 target against a 4.0196-bit floor budget. That is
    0.02 bits under, far too little for the override warning the 3-bit arm earns and far
    too much to call clean: a few modules do drop below their role's floor. A card that
    rounded this to either neighbour would be wrong in one of the two directions that
    matter -- alarming a reader about a sound arm, or quietly nodding through breaches.

    Turns red when: the near-floor branch stops counting violations and starts asserting
    the arm clears every floor because it is close enough to the budget.
    """
    text = _render(card, tmp_path, "4.00", export={"average_bits": 4.00496})
    assert "Read this first" not in text
    assert "sits essentially at" in text
    assert "still breaks 1 of them" in text
    assert "clears every floor" not in text


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


def test_a_size_from_one_run_and_a_table_from_another_are_refused(
    card: Any, tmp_path: Path
) -> None:
    """The card reads its width from the export and its widths table from the inspection.

    Those are one allocation described twice, and nothing but this guard makes them so. The
    inspection is cheap and gets re-run; the export takes forty minutes and does not. So the
    reachable mistake is an inspection against a newer stats file than the export was built
    from -- and this campaign has already seen one stats snapshot give 3.99989 bits with 9
    violations where another gave 4.00496 with 7. Either table looks entirely plausible
    under either headline. A reader cannot tell, and the numbers are not far enough apart
    for anyone to notice by eye.

    Turns red when: the two records are read without being compared, which is how the file
    shipped before this test existed.
    """
    with pytest.raises(SystemExit, match="one allocation described twice"):
        _render(card, tmp_path, "3.00", export={"average_bits": 4.019577})


#: The record ``dynquant eval`` actually writes. Every field the card asserts about the
#: scoring run is nested one level down, and none of the three is where the flat fixture
#: above puts it -- which is the whole reason these two tests exist alongside it.
NESTED_EVAL = {
    "label": "3bit",
    "accuracy": 0.4125,
    "total": 400,
    "correct": 165,
    "split": "test",
    "limit": 400,
    "unparseable": 0,
    "decode": {
        "max_new_tokens": 1024,
        "batch_size": 32,
        "max_prompt_tokens": 3072,
        "greedy": True,
    },
    "task_options": {"sources": ["spider", "gretel", "wikisql"]},
    "detail": {
        "unfinished_reasoning": 0,
        "by_source": {"gretel": [105, 133], "spider": [108, 134], "wikisql": [124, 133]},
    },
}


def test_the_record_the_harness_writes_reaches_the_card(card: Any, tmp_path: Path) -> None:
    """Read flat, all three of these are missing, and two of them fail silently.

    ``dynquant eval`` puts the decode budget under ``decode``, the sources under
    ``task_options`` and the unfinished count under ``detail``. A card generator reading
    them at the top level renders a page that is wrong in three places and looks finished in
    all three: the accuracy sentence names no dataset, the decode budget prints ``?``, and
    the unfinished count falls back to 0 -- which reads as the run having been measured and
    found clean rather than never having been asked.

    Turns red when: the reads go back to the top level, which is where they were when the
    generator was written against a hand-made fixture instead of a real record.
    """
    text = _render(card, tmp_path, "3.00", eval_record=NESTED_EVAL)
    assert "`spider`, `gretel`, `wikisql`" in text
    assert "Decode budget was 1024 new tokens" in text
    assert "?" not in text.split("## Results")[1].split("## What the allocator")[0]

    # And the count itself, from a run where it is not zero -- which is the only version of
    # this assertion a top-level read cannot also satisfy. Both records here carry 0 under
    # `detail`, so a card defaulting to 0 would agree with a correct one on every one of
    # them and be wrong on the only run where the answer changes anything.
    bound = {**NESTED_EVAL, "detail": {**NESTED_EVAL["detail"], "unfinished_reasoning": 40}}
    text = _render(card, tmp_path, "3.00", eval_record=bound)
    assert "40 generations reached it without finishing" in text
    assert "bounded above by 90.00%" in text


def test_a_record_that_never_counted_unfinished_generations_is_refused(
    card: Any, tmp_path: Path
) -> None:
    """Zero is a measurement here, so it may not come from a key that is not there.

    A 256-token cap once scored 5.50% on this task where 1024 scored 57.75% on the same
    problems. "0 generations reached it without finishing" is therefore load-bearing prose,
    and ``.get(field, 0)`` would print it for a record that never looked.

    Turns red when: the default comes back.
    """
    stripped = {k: v for k, v in NESTED_EVAL.items() if k != "detail"}
    with pytest.raises(SystemExit, match="no unfinished_reasoning count"):
        _render(card, tmp_path, "3.00", eval_record=stripped)


#: The manifest ``run_s2_finetune`` really writes. It has no ``train_sources`` key at all --
#: it records the realised counts under ``sources`` and the request under
#: ``train_sources_requested`` -- which is why the flat FINETUNE fixture above cannot catch
#: the read this exercises.
REAL_FINETUNE = {
    **FINETUNE,
    "sources": {
        "text2sql/spider": 2500,
        "text2sql/gretel": 2500,
        "text2sql/wikisql": 2500,
        "text2sql/create-context": 2500,
    },
    "train_sources_requested": ["spider", "gretel", "wikisql", "create-context"],
    "decontaminated": {"spider": 2, "gretel": 2, "wikisql": 4, "create-context": 593},
    "sources_overlapping_an_eval_task": [],
    "train_loss": 0.09632793507575989,
}
del REAL_FINETUNE["train_sources"]


def test_the_datasets_reach_the_card_from_the_key_the_trainer_writes(
    card: Any, tmp_path: Path
) -> None:
    """What the other read gives is: 9,999 conversations from , 350,799 supervised tokens.

    The card asked for ``train_sources``; the trainer writes ``sources`` as a
    ``task/name -> count`` mapping. Missing, the list renders empty and the sentence keeps
    its commas -- a line that names no dataset while looking like it named one, on the one
    bullet a reader checks to decide whether the model saw their kind of data.

    Turns red when: the read goes back to a key the manifest does not have.
    """
    text = _render(card, tmp_path, "3.00", finetune=REAL_FINETUNE)
    assert "`spider`, `gretel`, `wikisql`, `create-context`" in text
    assert "conversations from ," not in text


def test_a_source_dropped_to_nothing_is_not_claimed_as_training_data(
    card: Any, tmp_path: Path
) -> None:
    """The realised counts win over the requested list, because a request is not a fact.

    Decontamination can empty a source entirely. It was still asked for, and
    ``train_sources_requested`` still names it, but the checkpoint never saw it.

    Turns red when: the requested list is preferred over the realised mapping.
    """
    finetune = {**REAL_FINETUNE, "sources": {**REAL_FINETUNE["sources"], "text2sql/wikisql": 0}}
    text = _render(card, tmp_path, "3.00", finetune=finetune)
    assert "`spider`, `gretel`, `create-context`" in text
    assert "`wikisql`" not in text.split("## How it was made")[1]


def test_a_tiny_p_value_does_not_render_as_zero(card: Any, tmp_path: Path) -> None:
    """``:.4f`` prints 1.93e-05 as 0.0000, which claims a difference that cannot be chance.

    Turns red when: the fixed-decimal format comes back.
    """
    panel = tmp_path / "panel.json"
    panel.write_text(
        json.dumps(
            {"arms": [{"arm": "dq3", "comparison": {"delta_points": -6.0, "p_value": 1.93e-05}}]}
        ),
        encoding="utf-8",
    )
    text = _render(card, tmp_path, "3.00", panel=str(panel))
    assert "1.93e-05" in text
    assert "0.0000" not in text


def test_a_full_float_repr_of_the_loss_does_not_reach_the_page(card: Any, tmp_path: Path) -> None:
    """0.09632793507575989 is a float repr; 17 significant figures is not a measurement.

    Turns red when: the raw value is interpolated again.
    """
    text = _render(card, tmp_path, "3.00", finetune=REAL_FINETUNE)
    assert "train loss 0.0963" in text
    assert "0.09632793507575989" not in text


def test_the_contamination_answer_comes_from_the_run(card: Any, tmp_path: Path) -> None:
    """Trained on spider/gretel/wikisql, scored on spider/gretel/wikisql -- so the card says.

    A reader of the results table will ask whether the model was scored on what it was
    trained on. The trainer records both halves of the answer, so the card states it rather
    than leaving the reader to assume either way.

    Turns red when: a run that overlaps an eval task is described as decontaminated, which
    is the direction that misleads.
    """
    clean = _render(card, tmp_path, "3.00", finetune=REAL_FINETUNE)
    assert "601 examples were dropped" in clean
    assert "No source overlaps an evaluation task" in clean

    dirty = _render(
        card,
        tmp_path,
        "3.00",
        finetune={**REAL_FINETUNE, "sources_overlapping_an_eval_task": ["spider"]},
    )
    assert "`spider` overlap an evaluation task" in dirty
    assert "not a held-out measurement" in dirty

    # A run that recorded neither says nothing, rather than claiming a check it never ran.
    silent = _render(
        card,
        tmp_path,
        "3.00",
        finetune={
            k: v
            for k, v in REAL_FINETUNE.items()
            if k not in ("decontaminated", "sources_overlapping_an_eval_task")
        },
    )
    assert "Contamination" not in silent
