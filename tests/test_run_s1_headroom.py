"""The S1 driver has to be right before it is run, because running it is the cost.

Sixteen cells, each a model download and an engine build, is most of a GPU-day. Every
failure this file covers is one that costs a cell or the whole sweep and produces no
number: a flag spelled the way the driver's own parser spells it rather than the way
``dynquant eval`` does, a code task that reaches the execution refusal *after* the
model is loaded, a resume guard that hands back a record produced by different code.

Nothing here runs a cell. ``_cell_command`` returns a list and ``_needs_run`` reads a
file, so the whole surface that decides what gets run is reachable from CPU CI.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from dynquant.cli import build_parser
from dynquant.commands.evaluate import TASKS

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "scripts" / "run_s1_headroom.py"


@pytest.fixture(scope="module")
def s1():
    spec = importlib.util.spec_from_file_location("_dq_s1", DRIVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dq_s1"] = module
    spec.loader.exec_module(module)
    return module


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "backend": "vllm",
        "dtype": "bfloat16",
        "gpu_memory_utilization": 0.85,
        "limit": None,
        "trust_remote_code": False,
    }
    return argparse.Namespace(**{**defaults, **overrides})


# --------------------------------------------------------------------------
# The command each cell runs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task", ["gsm8k", "ifeval", "humaneval", "mbpp"])
def test_every_cell_is_a_command_the_eval_parser_accepts(s1, task: str, tmp_path) -> None:
    """The driver and ``dynquant eval`` are two argument lists over one parser.

    Turns red when: a flag is renamed on either side, or the model goes back to being
    passed as ``--model`` -- it is positional, and argparse rejects the flag form with
    "unrecognized arguments" *before* anything is loaded, which is the cheap version
    of this failure only because this test exists. Sixteen cells each failing that way
    is a sweep that produces nothing.
    """
    cmd = s1._cell_command("some/repo", task, tmp_path / f"x.{task}.json", _args())
    assert cmd[:4] == [sys.executable, "-m", "dynquant", "eval"]

    parsed = build_parser().parse_args(cmd[3:])
    assert parsed.model == "some/repo"
    assert parsed.task == task
    assert parsed.out == str(tmp_path / f"x.{task}.json")


def test_a_code_task_opts_into_execution_and_nothing_else_does(s1, tmp_path) -> None:
    """``--allow-execution`` is refused at ``_task_kwargs``, after the model is loaded.

    Reading the requirement off the registry rather than off a literal list here is
    the point: a task added to the campaign with ``executes_code`` set cannot arrive
    in this driver without the flag.

    Turns red when: the flag is dropped, or handed to a task that does not run code --
    which would be an evaluation quietly permitted to execute model-written Python.
    """
    for task in TASKS:
        cmd = s1._cell_command("r", task, tmp_path / "x.json", _args())
        assert ("--allow-execution" in cmd) == TASKS[task].executes_code, task


def test_ifeval_drops_what_it_cannot_check_rather_than_refusing_to_score(s1, tmp_path) -> None:
    """The default is ``raise``, and on this box that means no number at all.

    Turns red when: the flag goes, or spreads to a task with nothing unverifiable in
    it -- where "drop" would silently shrink a denominator for no stated reason.
    """
    for task in TASKS:
        cmd = s1._cell_command("r", task, tmp_path / "x.json", _args())
        dropped = "--on-unverifiable" in cmd and cmd[cmd.index("--on-unverifiable") + 1] == "drop"
        assert dropped == TASKS[task].unverifiable, task


def test_the_smoke_limit_is_absent_unless_asked_for(s1, tmp_path) -> None:
    """A ``--limit`` left in from a smoke run is a screen of the first N examples.

    The splits arrive label-sorted, so a prefix limit on a classification task samples
    one class -- and this driver's whole output is "does this cell have headroom".

    Turns red when: the limit acquires a default other than None.
    """
    assert "--limit" not in s1._cell_command("r", "gsm8k", tmp_path / "x.json", _args())
    assert "--limit" in s1._cell_command("r", "gsm8k", tmp_path / "x.json", _args(limit=200))


def test_the_tasks_screened_are_tasks_the_command_can_score(s1) -> None:
    """A typo in ``TASKS`` is sixteen cells that die on argparse choices.

    Turns red when: a task is renamed in the registry, or added to this driver that
    ``dynquant eval`` does not offer.
    """
    assert set(s1.TASKS) <= set(TASKS)


# --------------------------------------------------------------------------
# Resume
# --------------------------------------------------------------------------


def test_a_record_written_by_older_code_is_re_run(s1, tmp_path) -> None:
    """Skip-if-output-exists cannot see that an output predates its input.

    This is the failure that has already happened once in this project: a resume guard
    stapled a stale artifact to a fresh conclusion, and nothing in the output said the
    two came from different code.

    Turns red when: the commit stamp stops being compared, which makes the guard a
    plain existence check again.
    """
    record = tmp_path / "phi4-mini.gsm8k.json"
    record.write_text("{}", encoding="utf-8")
    record.with_suffix(".commit").write_text("0" * 40, encoding="utf-8")

    assert s1._needs_run(record, "0" * 40, force=False) is False
    assert s1._needs_run(record, "1" * 40, force=False) is True


def test_a_record_with_no_stamp_at_all_is_re_run(s1, tmp_path) -> None:
    """An unstamped record was written by something this driver cannot vouch for.

    Turns red when: a missing sidecar starts being treated as "current", which is what
    a ``stamp.exists() and ...`` would do.
    """
    record = tmp_path / "phi4-mini.gsm8k.json"
    record.write_text("{}", encoding="utf-8")
    assert s1._needs_run(record, "0" * 40, force=False) is True


def test_force_re_runs_a_record_that_is_current(s1, tmp_path) -> None:
    record = tmp_path / "phi4-mini.gsm8k.json"
    record.write_text("{}", encoding="utf-8")
    record.with_suffix(".commit").write_text("0" * 40, encoding="utf-8")
    assert s1._needs_run(record, "0" * 40, force=True) is True


# --------------------------------------------------------------------------
# The screen's output
# --------------------------------------------------------------------------


def test_the_summary_prints_accuracy_against_the_tasks_chance_floor(s1, tmp_path) -> None:
    """Accuracy alone cannot say whether a cell has headroom.

    A score at chance and a score at ceiling are both unusable and sit at opposite ends
    of the column, so the floor belongs beside the number.

    Turns red when: the floor column goes, and the screen starts reporting a number
    that cannot be acted on.
    """
    (tmp_path / "phi4-mini.gsm8k.json").write_text(
        json.dumps({"total": 1319, "accuracy": 0.6042, "chance": 0.0, "unparseable": 3}),
        encoding="utf-8",
    )
    summary = s1.summarize(tmp_path)

    assert "phi4-mini" in summary
    assert "60.42" in summary
    assert "0.00" in summary


def test_a_record_missing_a_field_names_it_instead_of_raising(s1, tmp_path) -> None:
    """The records are the expensive part and they are already written.

    A ``KeyError`` at the end of a sixteen-cell sweep is recoverable -- the records
    survive -- but only after reading a traceback to find which field moved. Naming it
    costs one line.

    Turns red when: the fields are indexed directly again.
    """
    (tmp_path / "phi4-mini.gsm8k.json").write_text(
        json.dumps({"total": 10, "accuracy": 0.5}), encoding="utf-8"
    )
    summary = s1.summarize(tmp_path)

    assert "no chance, unparseable in the record" in summary


def test_an_empty_output_directory_is_not_an_error(s1, tmp_path) -> None:
    assert s1.summarize(tmp_path) == "no records yet"


# --------------------------------------------------------------------------
# Gating
# --------------------------------------------------------------------------


def test_the_gated_models_are_the_ones_the_hub_actually_gates(s1) -> None:
    """Measured against the Hub API, not assumed: only Llama-3.1 and Gemma-3 are
    ``gated=manual``. Recording it here is what lets the driver say up front which
    cells it cannot run, rather than dying partway through with a 401.

    Turns red when: a model is added without its gating stated, which would put the
    401 back where it costs the sweep.
    """
    gated = {name for name, entry in s1.MODELS.items() if entry["gated"]}
    assert gated == {"llama31-8b", "gemma3-4b"}
    assert set(s1.MODELS) == {"llama31-8b", "gemma3-4b", "phi4-mini", "ministral-8b"}


def test_gated_models_are_skipped_loudly_when_there_is_no_token(
    s1, tmp_path, capsys, monkeypatch
) -> None:
    """Licence acceptance is per-account on the Hub and cannot be worked around here.

    The remaining cells still run -- refusing the whole sweep for two models would
    waste the two that are open.

    Turns red when: a missing token stops being reported, or aborts the sweep.
    """
    for variable in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(variable, raising=False)

    calls: list[list[str]] = []

    class _Proc:
        returncode = 0

    def _fake_run(cmd, check=False, **kwargs):
        calls.append(cmd)
        return _Proc()

    monkeypatch.setattr(s1.subprocess, "run", _fake_run)
    monkeypatch.setattr(s1, "_git_head", lambda repo: "a" * 40)

    code = s1.main(["--out", str(tmp_path), "--tasks", "gsm8k"])

    out = capsys.readouterr().out
    assert "SKIPPING 2 gated model(s)" in out
    assert "llama31-8b" in out and "gemma3-4b" in out
    assert code == 0

    ran = {cmd[4] for cmd in calls}
    assert ran == {"microsoft/Phi-4-mini-instruct", "mistralai/Ministral-8B-Instruct-2410"}


def test_a_failed_cell_does_not_cost_the_others_their_run(
    s1, tmp_path, capsys, monkeypatch
) -> None:
    """One model failing to load should not end a sweep the rest of which would work.

    It is still reported and still exits non-zero -- a sweep with a hole in it that
    returned 0 would be read as a complete screen.

    Turns red when: the driver raises on the first non-zero exit, or stops counting
    failures into its own exit code.
    """
    monkeypatch.setenv("HF_TOKEN", "not-a-real-token")
    seen: list[str] = []

    class _Proc:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode

    def _fake_run(cmd, check=False, **kwargs):
        seen.append(cmd[4])
        return _Proc(1 if cmd[4].startswith("google/") else 0)

    monkeypatch.setattr(s1.subprocess, "run", _fake_run)
    monkeypatch.setattr(s1, "_git_head", lambda repo: "a" * 40)

    code = s1.main(["--out", str(tmp_path), "--tasks", "gsm8k"])

    assert len(seen) == 4, "every cell was attempted"
    assert code == 1
    assert "1 cell(s) failed: gemma3-4b.gsm8k (exit 1)" in capsys.readouterr().out
    assert not (tmp_path / "gemma3-4b.commit").exists(), "a failed cell is not stamped"


# --------------------------------------------------------------------------
# Scoring a checkpoint on disk -- S3's ~32 arms go through this same driver
# --------------------------------------------------------------------------


def _checkpoint(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text('{"model_type": "phi3"}', encoding="utf-8")
    return path


def test_a_registry_key_and_a_path_both_resolve(s1, tmp_path) -> None:
    """S3's arms are directories; S1's models are Hub ids. One driver takes both.

    Turns red when: ``--models`` goes back to being a closed set of registry keys, and
    S3 has to grow a second evaluator whose prompts can drift from this one.
    """
    assert s1.resolve_model("phi4-mini") == ("phi4-mini", "microsoft/Phi-4-mini-instruct", False)
    assert s1.resolve_model("llama31-8b")[2] is True, "gating is read off the registry"

    arm = _checkpoint(tmp_path, "dq_3p25")
    assert s1.resolve_model(f"dq3p25={arm}") == ("dq3p25", str(arm), False)


def test_a_local_checkpoint_is_never_gated(s1, tmp_path) -> None:
    """Gating is a Hub licence. A directory on disk has already settled it.

    Turns red when: a path inherits a registry entry's gating, so an S3 arm quantized
    *from* a gated model is skipped for want of a token it does not need.
    """
    arm = _checkpoint(tmp_path, "from-a-gated-base")
    assert s1.resolve_model(f"arm={arm}")[2] is False


def test_a_record_name_with_a_dot_is_refused(s1, tmp_path) -> None:
    """``dq_3.25`` would file under model ``dq_3``, task ``25.gsm8k``.

    ``summarize`` splits the stem on the first dot, so the row would name the wrong arm
    and still read as measured. Turns red when the name stops being validated, or when
    the record filename stops being dot-separated and this check is no longer the right
    one to make.
    """
    arm = _checkpoint(tmp_path, "arm")
    with pytest.raises(SystemExit, match="cannot be empty or contain a dot"):
        s1.resolve_model(f"dq_3.25={arm}")
    with pytest.raises(SystemExit, match="cannot be empty or contain a dot"):
        s1.resolve_model(f"={arm}")

    record = tmp_path / "dq_3.25.gsm8k.json"
    record.write_text(
        json.dumps({"total": 1, "accuracy": 1.0, "chance": 0.0, "unparseable": 0}),
        encoding="utf-8",
    )
    assert "dq_3 " in s1.summarize(tmp_path), "the mis-split this refusal prevents"


def test_a_path_that_is_not_a_checkpoint_costs_a_message_not_an_engine_build(s1, tmp_path) -> None:
    """An arm is produced by a separate step; a missing one should fail before the GPU.

    Turns red when: the check is dropped, and a mistyped path costs a model load per
    task before it says anything.
    """
    with pytest.raises(SystemExit, match="not a checkpoint"):
        s1.resolve_model(f"arm={tmp_path / 'never-written'}")

    empty = tmp_path / "half-written"
    empty.mkdir()
    with pytest.raises(SystemExit, match="not a checkpoint"):
        s1.resolve_model(f"arm={empty}")


def test_an_unknown_bare_name_lists_the_registry(s1) -> None:
    """A typo should say what the options are, including that a path is one.

    Turns red when: argparse choices come back (the message would be argparse's, which
    cannot mention the name=path form), or the error stops naming the alternatives.
    """
    with pytest.raises(SystemExit) as excinfo:
        s1.resolve_model("phi4mini")
    message = str(excinfo.value)
    assert "phi4-mini" in message and "name=path" in message


def test_two_arms_cannot_file_under_one_record_name(s1, tmp_path, monkeypatch) -> None:
    """Thirty-two arms per model is where a name collision stops being hypothetical.

    The loser's records are overwritten by the winner's and the survivor looks correct.
    Turns red when the collision check is dropped.
    """
    monkeypatch.setattr(s1, "_git_head", lambda repo: "a" * 40)
    first, second = _checkpoint(tmp_path, "a"), _checkpoint(tmp_path, "b")
    with pytest.raises(SystemExit, match="both record as 'dq'"):
        s1.main(["--out", str(tmp_path / "out"), "--models", f"dq={first}", f"dq={second}"])


def test_a_checkpoint_arm_runs_the_same_command_a_registry_model_does(
    s1, tmp_path, monkeypatch
) -> None:
    """The arm's path reaches ``dynquant eval`` as its positional model.

    Turns red when: the driver keeps resolving the model through ``MODELS`` at call time,
    so a path resolves to a KeyError or -- worse -- to whatever registry entry shares its
    name, and the record says one model while another was scored.
    """
    monkeypatch.setattr(s1, "_git_head", lambda repo: "a" * 40)
    arm = _checkpoint(tmp_path, "dq_3p25")
    seen: list[list[str]] = []

    class _Proc:
        returncode = 0

    def _fake_run(cmd, check=False, **kwargs):
        seen.append(cmd)
        return _Proc()

    monkeypatch.setattr(s1.subprocess, "run", _fake_run)
    out = tmp_path / "out"
    code = s1.main(["--out", str(out), "--models", f"dq3p25={arm}", "--tasks", "gsm8k"])

    assert code == 0
    assert len(seen) == 1
    assert seen[0][4] == str(arm), "the checkpoint path is the positional model"
    assert (out / "dq3p25.gsm8k.commit").exists()
