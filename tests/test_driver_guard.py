"""Three campaign scripts refuse to run while the panel driver is scoring. Only one kind of
process is the panel driver.

`pgrep -f` matches a whole command line, and this campaign generates a steady supply of command
lines carrying the driver's name without being it: a monitor polling every 240 seconds, a `grep`
over these scripts, and the `ssh host bash -c '...'` wrapper that launches one of the guarded
scripts with a diagnostic appended to the same line. That last one refused a relaunch on
2026-08-09 with the GPU at 1 MiB and nothing scoring, and printed its own PID as the evidence.

The refusal is why it matters. A guard that stops work cannot afford a false positive, because a
false positive is indistinguishable from the true positive it exists to produce -- both print a
PID and a plausible command line -- and the only move it leaves an operator is to decide the guard
is wrong. Teaching that is worse than not guarding.

So the filter is tested here against command lines of both kinds, and it is tested by *extracting
it from the scripts* rather than by restating it: a copy of the rule in this file would agree with
the scripts until the day someone edited one of them. Three things turn this red -- relaxing the
filter back to matching a mention, tightening it to something that misses a real driver, and the
three copies drifting apart.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PHASE4 = REPO_ROOT / "experiments" / "phase4"
GUARDED = ("rescore_eager.sh", "sync_clone.sh", "build_kernels.sh")

# `pgrep -af` prints "PID cmdline", so field 2 is argv[0] and field 3 is argv[1].
PIPELINE = re.compile(r"pgrep -af '(?P<pattern>[^']*)' \| (?P<awk>awk '[^']*')")

# Command lines of both kinds, in the shape `pgrep -af` prints them. The first two are processes
# running the driver -- under an interpreter, which is how every script here launches it, and under
# its own shebang, which is how someone would launch it by hand. The rest only mention it.
RUNNING = [
    "181717 /workspace/venv-llmc/bin/python experiments/phase4/arms_lfm2.py run "
    "--model /workspace/runs/s4/lfm25-8b-a1b.text2sql/merged --out /workspace/runs/s4/panel "
    "--device cuda --limit 12000 --resume --rescore dq_4b,dq_3b --experts-impl eager",
    "182003 ./experiments/phase4/arms_lfm2.py run --resume --rescore bf16",
]
MENTIONING = [
    # The ssh wrapper that launched the guarded script and a diagnostic on one command line.
    "24110 bash -c cd /workspace/dq-next && nohup bash experiments/phase4/rescore_eager.sh "
    "& sleep 20; pgrep -af arms_lfm2.py run",
    # The monitor, polling for exactly this process on a fixed interval.
    "24250 bash -lc pgrep -af arms_lfm2.py run >/dev/null && echo DRIVER_UP || echo DRIVER_GONE",
    # Someone reading the scripts while the box is idle.
    "24310 /usr/bin/grep -n arms_lfm2.py run experiments/phase4/rescore_eager.sh",
    # An inbound session whose own command line carries the pattern.
    "24400 sshd: root@notty bash -c pgrep -af arms_lfm2.py run",
]


def _filters() -> dict[str, re.Match[str]]:
    out = {}
    for name in GUARDED:
        source = (PHASE4 / name).read_text(encoding="utf-8")
        found = PIPELINE.search(source)
        assert found, f"{name} has no `pgrep -af '...' | awk '...'` driver filter"
        out[name] = found
    return out


def _run_awk(program: str, lines: list[str]) -> list[str]:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash to run the extracted awk program under")
    if subprocess.run([bash, "-c", "command -v awk"], capture_output=True).returncode != 0:
        pytest.skip("no awk on this shell")
    done = subprocess.run(
        [bash, "-c", program],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in done.stdout.splitlines() if line.strip()]


def test_the_three_scripts_filter_identically() -> None:
    """A guard that reads differently in three places is three guards.

    They are three copies of one rule, and the only thing keeping them one rule is that a change
    to any of them has to be a change to all of them. Nothing else in the repo compares them.
    """
    filters = _filters()
    patterns = {name: found.group("pattern") for name, found in filters.items()}
    awks = {name: found.group("awk") for name, found in filters.items()}
    assert len(set(patterns.values())) == 1, patterns
    assert len(set(awks.values())) == 1, awks


def test_a_process_running_the_driver_is_reported() -> None:
    """Both launch forms, because the false negative is the expensive direction.

    A missed driver does not stop anything: `sync_clone.sh` rewrites the clone under a live panel
    and the arms scored after it ran different code than the arms before, with nothing recording
    which was which. Requiring `python` in field 2 would pass this file's first case and silently
    drop the second.
    """
    program = next(iter(_filters().values())).group("awk")
    assert _run_awk(program, RUNNING) == RUNNING


def test_a_process_that_only_mentions_the_driver_is_not() -> None:
    """The monitor, the grep, the ssh wrapper, and the launcher's own command line."""
    program = next(iter(_filters().values())).group("awk")
    assert _run_awk(program, MENTIONING) == []


def test_the_driver_is_found_among_the_mentions() -> None:
    """The two lists interleaved, which is what the box actually looks like.

    Separately they only prove the filter has a preference; together they prove it discriminates,
    and this is the arrangement that produced the 2026-08-09 refusal -- one real driver absent, one
    monitor and one ssh wrapper present, and a guard that could not tell the difference.
    """
    program = next(iter(_filters().values())).group("awk")
    interleaved = [MENTIONING[0], RUNNING[0], MENTIONING[1], MENTIONING[2], RUNNING[1]]
    assert _run_awk(program, interleaved) == [RUNNING[0], RUNNING[1]]


@pytest.mark.parametrize("name", GUARDED)
def test_every_pgrep_goes_through_the_filter(name: str) -> None:
    """One `pgrep` per script, inside `driver_procs`.

    The old code called it twice -- once for the test and once to print the evidence -- so the
    lines shown as the reason for a refusal were a second sample and not necessarily the lines
    that caused it. A second raw call is also how the filter gets bypassed by accident.
    """
    source = (PHASE4 / name).read_text(encoding="utf-8")
    # Comment lines are prose about the rule; every script quotes `pgrep -af` while explaining
    # which field is argv[0]. It is the calls that have to be one.
    calls = [
        line
        for line in source.splitlines()
        if "pgrep -af" in line and not line.lstrip().startswith("#")
    ]
    assert len(calls) == 1, f"{name} calls pgrep -af {len(calls)} times, not once: {calls}"
    assert "driver_procs()" in source
    assert source.count("running=$(driver_procs)") == 1
