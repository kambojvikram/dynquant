"""The publish plan is derived from the panel's registry, and that is the thing to test.

Nothing here runs a recipe. What can go wrong before one runs is the plan: an arm dropped
because this file has never heard of it, a record pointed at the wrong label, a map keyed by
a number that looks right and is not. Each of those produces a directory rather than an
error, and a directory that loads is indistinguishable from a directory that is the arm.

The fixture transcribes the real panel's numbers instead of inventing round ones, because
two of these tests turn on a coincidence in the real values -- a DynQuant arm whose achieved
byte count is *near* its target and not equal to it, which is what separates keying by one
from keying by the other.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "experiments" / "phase4" / "publish_panel.py"

MODEL = "/workspace/runs/s4/lfm25-8b-a1b.text2sql/merged"
ANCHOR_4B = 4399629312
ANCHOR_3B = 3332904576


@pytest.fixture(scope="module")
def driver() -> Any:
    spec = importlib.util.spec_from_file_location("_dq_publish_panel", DRIVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dq_publish_panel"] = module
    spec.loader.exec_module(module)
    return module


def _arms(tmp_path: Any, *extra: dict[str, Any], drop: tuple[str, ...] = ()) -> Path:
    """The seven-arm panel as `arms_lfm2.py` writes it, finished.

    `dq_4b`'s `nbytes` is 4,397,666,304 against a `target_bytes` of 4,399,629,312: the
    allocator lands *under* the anchor because the last width it could afford would have
    gone over. That 1,963,008-byte gap is the whole content of the map-key test.
    """
    arms = [
        {
            "label": "bf16",
            "kind": "ceiling",
            "anchor": None,
            "target_bytes": None,
            "nbytes": None,
            "record": "/panel/bf16.json",
        },
        {
            "label": "gptq_4b",
            "kind": "gptq",
            "anchor": 4,
            "target_bytes": ANCHOR_4B,
            "nbytes": ANCHOR_4B,
            "record": "/panel/gptq_4b.json",
        },
        {
            "label": "awq_4b",
            "kind": "awq",
            "anchor": 4,
            "target_bytes": ANCHOR_4B,
            "nbytes": ANCHOR_4B,
            "record": "/panel/awq_4b.json",
        },
        {
            "label": "dq_4b",
            "kind": "dq",
            "anchor": 4,
            "target_bytes": ANCHOR_4B,
            "nbytes": 4397666304,
            "record": "/panel/dq_4b.json",
            "map": "/panel/maps/dq_4b.json",
        },
        {
            "label": "gptq_3b",
            "kind": "gptq",
            "anchor": 3,
            "target_bytes": ANCHOR_3B,
            "nbytes": ANCHOR_3B,
            "record": "/panel/gptq_3b.json",
        },
        {
            "label": "awq_3b",
            "kind": "awq",
            "anchor": 3,
            "target_bytes": ANCHOR_3B,
            "nbytes": ANCHOR_3B,
            "record": "/panel/awq_3b.json",
        },
        {
            "label": "dq_3b",
            "kind": "dq",
            "anchor": 3,
            "target_bytes": ANCHOR_3B,
            "nbytes": 3330000000,
            "record": "/panel/dq_3b.json",
            "map": "/panel/maps/dq_3b.json",
        },
    ]
    arms = [a for a in arms if a["label"] not in drop]
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "arms.json"
    path.write_text(
        json.dumps({"model": MODEL, "group_size": 128, "arms": [*arms, *extra]}),
        encoding="utf-8",
    )
    return path


def _flag(cmd: list[str], name: str) -> str:
    return cmd[cmd.index(name) + 1]


def test_the_plan_is_the_panels_arm_list_and_not_a_second_copy_of_it(
    driver: Any, tmp_path: Any
) -> None:
    """An arm this file has never heard of has to stop the plan, not fall out of it.

    ``ORDER`` exists to say *when* each arm runs, and a list that also decides *which* arms
    run is the seventh duplicated registry this campaign has found -- it would agree with
    ``arms.json`` on every panel until the one that added an arm, and then publish six of
    seven in silence. So the labels are checked both ways: the plan may not name an arm the
    registry does not have, and the registry may not hold a publishable arm the plan skips.
    """
    eighth = {
        "label": "rtn_4b",
        "kind": "rtn",
        "anchor": 4,
        "target_bytes": ANCHOR_4B,
        "nbytes": ANCHOR_4B,
        "record": "/panel/rtn_4b.json",
    }
    with pytest.raises(SystemExit, match="rtn_4b"):
        driver.plan(_arms(tmp_path, eighth), Path("/out"))

    with pytest.raises(SystemExit, match="no arm named"):
        driver.plan(_arms(tmp_path / "b", drop=("dq_3b",)), Path("/out"), ["dq_3b"])


def test_a_ceiling_is_not_a_variant_and_is_not_published(driver: Any, tmp_path: Any) -> None:
    """``bf16`` is the checkpoint the panel started from, already on disk under its own name.

    It is in ``arms.json`` because it is a row in the table. A copy of it in the published
    set would be the one directory whose contents are not a result, and -- since it is the
    only unquantized arm -- also the largest thing in the set by four times.
    """
    steps = driver.plan(_arms(tmp_path), Path("/out"))
    assert [s.label for s in steps] == list(driver.ORDER)
    assert "bf16" not in {s.label for s in steps}


def test_each_recipe_arm_is_told_to_reproduce_its_own_record(driver: Any, tmp_path: Any) -> None:
    """The whole point of the second pass, and the one place it can be wired to the wrong file.

    Every published baseline directory comes from a calibration pass that has not happened
    yet, so ``--scored`` is what makes it that arm's directory rather than a directory with
    that arm's name. Pointed one label over it would compare AWQ's pass against GPTQ's
    record and refuse, which is survivable; pointed at nothing it publishes whatever came
    out, which is not.
    """
    steps = {s.label: s for s in driver.plan(_arms(tmp_path), Path("/out"))}
    for label, bits in (("gptq_4b", "4"), ("awq_4b", "4"), ("gptq_3b", "3"), ("awq_3b", "3")):
        cmd = steps[label].cmd
        assert _flag(cmd, "--scored").endswith(f"{label}.quant.json")
        assert _flag(cmd, "--bits") == bits
        assert _flag(cmd, "--method") == label.split("_")[0]
        assert _flag(cmd, "--model") == MODEL
        assert _flag(cmd, "--group-size") == "128"


def test_a_map_arm_is_exported_at_the_target_it_was_priced_under_not_the_bytes_it_hit(
    driver: Any, tmp_path: Any
) -> None:
    """The two numbers are three decimal digits apart and only one of them is a key.

    A map file holds one allocation per target it was asked for, keyed by the target.
    ``dq_4b`` was asked for 4,399,629,312 and the allocation came out at 4,397,666,304,
    because the next width it could have bought would have gone over. Keying by what it
    achieved looks right in a diff, reads right in a review, and misses -- on every
    DynQuant arm in this panel, since none of them land exactly on an anchor.
    """
    steps = {s.label: s for s in driver.plan(_arms(tmp_path), Path("/out"))}
    for label, target, achieved in (
        ("dq_4b", ANCHOR_4B, 4397666304),
        ("dq_3b", ANCHOR_3B, 3330000000),
    ):
        cmd = steps[label].cmd
        assert _flag(cmd, "--map-key") == str(target)
        assert _flag(cmd, "--map-key") != str(achieved)
        assert _flag(cmd, "--map").endswith(f"{label}.json")
        assert "--scored" not in cmd
        assert steps[label].scored is None


def test_an_arm_the_panel_never_scored_is_refused_rather_than_published(
    driver: Any, tmp_path: Any
) -> None:
    """A directory with no row is the failure this script exists to prevent, inverted.

    Mid-panel every unscored arm is a `null` record in ``arms.json``. Publishing one writes
    a real, loadable, correct-looking model that no table describes -- and the six variants
    this campaign promised are six rows, not six directories.
    """
    unscored = _arms(tmp_path, drop=("awq_3b",))
    payload = json.loads(unscored.read_text(encoding="utf-8"))
    payload["arms"].append(
        {
            "label": "awq_3b",
            "kind": "awq",
            "anchor": 3,
            "target_bytes": ANCHOR_3B,
            "nbytes": None,
            "record": None,
        }
    )
    unscored.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="awq_3b was never scored"):
        driver.plan(unscored, Path("/out"))

    # And it says how to publish the rest, because mid-panel that is the actual next move.
    remaining = driver.plan(unscored, Path("/out"), ["gptq_4b", "dq_4b"])
    assert [s.label for s in remaining] == ["gptq_4b", "dq_4b"]


def test_the_expensive_arms_go_first_because_the_box_is_not_a_volume(
    driver: Any, tmp_path: Any
) -> None:
    """Order is the only thing this file decides on its own, so it is the only thing to pin.

    A recycle keeps nothing on that box. A recipe arm is 32 to 47 minutes of calibration
    that cannot be recovered any other way; a map arm re-exports in minutes from a file that
    already exists. So every recipe arm precedes every map arm, and a reordering that puts
    the cheap ones first -- which is what alphabetical does, and what "DynQuant first, it is
    ours" does -- goes red here.
    """
    steps = driver.plan(_arms(tmp_path), Path("/out"))
    kinds = [s.kind for s in steps]
    last_recipe = max(i for i, k in enumerate(kinds) if k in driver.RECIPE_KINDS)
    first_map = min(i for i, k in enumerate(kinds) if k in driver.MAP_KINDS)
    assert last_recipe < first_map
    assert sorted(s.label for s in steps) != [s.label for s in steps]


def _exec_driver() -> Any:
    """A fresh module object, because the thing under test is an import side effect."""
    spec = importlib.util.spec_from_file_location("_dq_publish_fresh", DRIVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registered before exec, as the fixture above does: a `@dataclass` in the module body
    # resolves its own annotations through `sys.modules[cls.__module__]`.
    sys.modules["_dq_publish_fresh"] = module
    spec.loader.exec_module(module)
    return module


def _scored_records_exist(steps: list[Any]) -> None:
    """Satisfy the guard's own file check, which is not what these two tests are about."""
    for step in steps:
        if step.scored is not None:
            step.scored.write_text("{}", encoding="utf-8")


class _Fake:
    """Stand-in for `subprocess.run`, answering the two calls `guard` makes."""

    def __init__(self, *, probe_returncode: int) -> None:
        self.probe_returncode = probe_returncode
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> Any:
        self.calls.append(list(cmd))
        if "pgrep" in cmd[0]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(
            cmd, self.probe_returncode, "", "No module named dynquant"
        )

    @property
    def probes(self) -> list[list[str]]:
        return [cmd for cmd in self.calls if "dynquant" in cmd]


def test_a_map_arm_that_cannot_be_exported_is_refused_before_the_recipes_run(
    driver: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    """The failure this guard exists to move to the front, and the one it did not check.

    `python -m dynquant export` is how both map arms are published, and on the campaign
    box nothing installs `dynquant` in the venv -- so the command resolves only if the
    environment carries the package source. The map arms run *last*, after 2.7 h of GPTQ
    and AWQ passes that a recycle cannot give back, so an unresolvable CLI is discovered
    at the one point where discovering it is worthless.

    Turns red when: the probe is dropped, or moved after the first recipe runs, or reduced
    to a bare `-m dynquant --help` that a clone without the `export` command would pass.
    """
    fake = _Fake(probe_returncode=1)
    monkeypatch.setattr(driver.subprocess, "run", fake)
    steps = driver.plan(_arms(tmp_path), tmp_path / "out")
    _scored_records_exist(steps)

    with pytest.raises(SystemExit, match="after the recipe arms had run"):
        driver.guard(steps, force=False)

    assert fake.probes, fake.calls
    assert fake.probes[0][-3:] == ["dynquant", "export", "--help"], fake.probes[0]


def test_a_plan_with_no_map_arm_is_not_asked_about_a_cli_it_never_calls(
    driver: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    """A guard that refuses for a reason the plan cannot reach is a guard people learn to
    pass `--force` around. Only the map arms use `-m dynquant`; a recipe-only publish must
    not be blocked by whether it resolves.
    """
    fake = _Fake(probe_returncode=1)
    monkeypatch.setattr(driver.subprocess, "run", fake)
    steps = driver.plan(_arms(tmp_path), tmp_path / "out", only=["gptq_4b"])
    _scored_records_exist(steps)

    driver.guard(steps, force=False)
    assert fake.probes == [], fake.calls


def test_the_children_can_import_core_with_nothing_exported(monkeypatch: Any) -> None:
    """The fix itself, checked by spawning a child rather than reading the variable back.

    Asserting that PYTHONPATH contains a string would pass for a path that points nowhere.
    What the map arms need is that a fresh interpreter, started from this environment,
    can import `dynquant` -- so that is what is asserted, with PYTHONPATH removed first so
    the shell I happen to run the suite from cannot supply the answer.

    Turns red when: the environment plumbing is dropped, or CORE_SRC stops resolving to
    the package source.
    """
    monkeypatch.delenv("PYTHONPATH", raising=False)
    fresh = _exec_driver()
    assert fresh.CORE_SRC.is_dir(), fresh.CORE_SRC

    result = subprocess.run(
        [sys.executable, "-c", "import dynquant; print(dynquant.__file__)"],
        capture_output=True,
        text=True,
        env=os.environ,
    )
    assert result.returncode == 0, result.stderr
    assert "dynquant-core" in result.stdout, result.stdout
