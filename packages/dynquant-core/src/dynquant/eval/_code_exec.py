"""Running model-written code, and deciding whether it passed.

Everything here sits between a generation and a verdict without being part of either
task, because HumanEval and MBPP differ only in their prompts and their test programs
-- the hard parts, extraction and execution, are the same problem twice.

Execution is opt-in
-------------------
Nothing in this module runs until a caller passes ``allow_execution=True``. That is
not ceremony. A pass@1 evaluation executes text a language model wrote, and importing
an evaluation module should never be sufficient to do that. The upstream HumanEval
release ships its ``exec`` call commented out for the same reason; a required argument
is the same guarantee with a better error message.

What the isolation is, and is not
---------------------------------
Each candidate runs in its own subprocess, in its own temporary directory, with stdin
at ``/dev/null``, a wall-clock timeout, an address-space and file-size rlimit where the
platform has them, and a minimal environment. That is isolation from *accidents* --
infinite loops, memory bombs, a solution that calls ``input()``, one that writes files
next to your checkpoints. It is **not** a security boundary against code written to
break out of one, and it is not claimed to be. The threat model is a model you
fine-tuned yourself producing a wrong program, not an adversary producing a hostile
one.

Three failure modes are worth naming, because each produces a plausible number rather
than an error:

**``exit(0)`` reads as a pass.** A candidate that calls :func:`sys.exit` or
:func:`os._exit` before the assertions run leaves a process that exited zero, and an
exit-code-only harness scores that as correct. So the runner requires *two* things: a
zero exit code **and** a sentinel file that only ``_GUARD_SOURCE`` writes, after the
program has returned normally. ``SystemExit`` propagating out of the program skips the
write; ``os._exit`` never reaches it.

**A timeout is a property of the machine, not only of the model.** A solution that
takes 5.9 s alone times out under eight-way contention, and the arm that happened to
run while the box was busy loses points that have nothing to do with its weights. So
every timeout is re-run once, serially, before it is counted -- and the timeout and
memory bounds are recorded in :func:`sandbox_fingerprint` next to the score, because
two runs under different bounds are not comparable.

**Unbounded child output.** A candidate printing in a loop can emit hundreds of
megabytes before its timeout expires. Reading that through a pipe would put it in the
parent's memory. Child stdout goes to ``/dev/null`` -- the verdict is the exit status,
never the output -- and stderr goes to a file that is read from the end, so what the
parent holds is bounded by :data:`_STDERR_TAIL` regardless of what the child wrote.
"""

from __future__ import annotations

import os
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dynquant._logging import get_logger
from dynquant.errors import DynQuantError
from dynquant.eval.harness import chat_prompt_style

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "CodeEvalResult",
    "ExecOutcome",
    "ExecStatus",
    "PromptStyle",
    "extract_code",
    "prepare_decode",
    "resolve_style",
    "run_program",
    "run_programs",
    "sandbox_fingerprint",
    "score_generations",
]

_log = get_logger(__name__)

ExecStatus = Literal["passed", "failed", "timeout", "empty"]
PromptStyle = Literal["completion", "chat"]

DEFAULT_TIMEOUT = 8.0
"""Seconds per candidate. Upstream HumanEval uses 3.0, which is comfortable for a
correct solution and tight for a merely slow one; the extra margin costs nothing on
programs that pass and stops a loaded box from turning slowness into failure."""

DEFAULT_MEMORY_MB = 4096

_SENTINEL_NAME = "__dynquant_passed__"
_STDERR_TAIL = 2000

# Run by the child instead of the candidate, for two reasons. A traceback then points
# at the candidate's own line numbers, because the candidate file is executed as-is
# rather than with a prelude stapled to the front of it. And the sentinel write lives
# in a namespace the candidate never sees, so no amount of shadowing `open`, `Path` or
# `__builtins__` inside the candidate can reach it.
#
# The rlimits are set here rather than through `preexec_fn` deliberately: `preexec_fn`
# runs between fork and exec and is documented as unsafe in the presence of threads,
# and `run_programs` is a thread pool.
_GUARD_SOURCE = """\
import runpy
import sys
from pathlib import Path

memory_mb = int(sys.argv[1])
program = sys.argv[2]
sentinel = sys.argv[3]

try:
    import resource
except ImportError:
    # Windows has no rlimits. The wall-clock timeout is the only bound there, which
    # `sandbox_fingerprint` reports so the difference is visible in the results.
    #
    # macOS is the case in between and the one to watch: `resource` imports, RLIMIT_AS
    # sets without error, and Darwin then does not enforce it -- a 3 GiB allocation
    # under a 256 MB ceiling succeeds. So on macOS the memory bound is nominal and the
    # timeout is again the only real one. `sandbox_fingerprint` carries
    # `platform.system()`, which is what keeps that from being invisible in a results
    # table; the campaign's code evaluations run on Linux, where it is enforced.
    resource = None

if resource is not None:
    wanted = [("RLIMIT_AS", memory_mb * 1024 * 1024), ("RLIMIT_FSIZE", 64 * 1024 * 1024)]
    for name, value in wanted:
        which = getattr(resource, name, None)
        if which is None or value <= 0:
            continue
        try:
            _, hard = resource.getrlimit(which)
            ceiling = value if hard in (resource.RLIM_INFINITY, -1) else min(value, hard)
            resource.setrlimit(which, (ceiling, hard))
        except (ValueError, OSError):
            # A limit we cannot lower is not a reason to refuse to run the candidate;
            # the timeout still applies.
            pass

runpy.run_path(program, run_name="__main__")

# Reached only if the program returned normally. `sys.exit(0)` raises SystemExit
# through `run_path` and lands here as a non-zero exit; `os._exit(0)` never arrives.
Path(sentinel).write_text("ok", encoding="utf-8")
"""


@dataclass(frozen=True, slots=True)
class ExecOutcome:
    """What happened to one candidate program."""

    status: ExecStatus
    detail: str = ""
    """The tail of stderr, or the reason there is none. For inspection only -- nothing
    scores off it."""

    duration: float = 0.0

    @property
    def passed(self) -> bool:
        return self.status == "passed"


@dataclass(slots=True)
class CodeEvalResult:
    """One measurement point on an execution-scored task.

    Shared by HumanEval and MBPP because the arithmetic is identical and the paired
    comparison downstream does not care which set of problems it is looking at, only
    that both arms saw the same ones in the same order.
    """

    label: str
    task: str
    total: int
    passed: int
    timeouts: int
    """Counted after the serial re-run, so these are candidates that were too slow on
    an idle machine rather than candidates that lost a race for a core."""

    empty: int
    """Generations with no extractable code at all. Held apart from failures because
    they mean something different: a model that stops emitting code has stopped
    following the format, which is what catastrophic quantization damage looks like
    before it looks like wrong logic."""

    prompt_style: PromptStyle
    sandbox: str
    hits: list[bool] = field(default_factory=list)
    """Per-problem correctness, in dataset order. Always recorded -- the paired test
    downstream is the whole point, and the vector cannot be recovered once the
    GPU-hours are spent."""

    statuses: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)
    predictions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pass_at_1(self) -> float:
        """Greedy pass@1.

        Not the unbiased estimator over ``n`` samples, and deliberately so: every
        measurement point in this package decodes greedily, because a temperature above
        zero turns the effect being measured into noise you would need many seeds to
        see through. With one sample per problem the estimator collapses to the mean,
        and the result is a paired vector rather than a distribution.
        """
        return self.passed / self.total if self.total else 0.0

    @property
    def accuracy(self) -> float:
        """Alias, so a caller can treat every task result the same way."""
        return self.pass_at_1

    def summary(self) -> str:
        return (
            f"{self.label:<28} {self.pass_at_1:6.2%}  "
            f"({self.passed}/{self.total} pass@1, {self.timeouts} timeout, "
            f"{self.empty} no code)  [{self.task}/{self.prompt_style}]"
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "task": self.task,
            "pass_at_1": self.pass_at_1,
            "passed": self.passed,
            "total": self.total,
            "timeouts": self.timeouts,
            "empty": self.empty,
            "prompt_style": self.prompt_style,
            "sandbox": self.sandbox,
            "hits": self.hits,
            "statuses": self.statuses,
            "keys": self.keys,
        }


def sandbox_fingerprint(*, timeout: float, memory_mb: int) -> str:
    """A short string identifying the execution rules in force.

    The analogue of :func:`~dynquant.eval._ifeval_instructions.scorer_fingerprint`, and
    it exists for the same reason: two arms judged under different bounds are not
    comparable, and nothing in the accuracies themselves says so. Records the
    interpreter too, because a solution using ``match`` or ``itertools.batched`` passes
    on one minor version and fails on another.
    """
    rlimits = "rlimits" if _has_rlimits() else "no-rlimits"
    version = f"py{sys.version_info.major}.{sys.version_info.minor}"
    return f"exec/{platform.system().lower()}/{version}/{rlimits}/t={timeout:g}s/m={memory_mb}MB"


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

# `\Z` as an alternative to the closing fence on purpose. A long solution can hit
# `max_new_tokens` mid-block, and a regex that insisted on the closing fence would find
# nothing and score a nearly-complete answer as "produced no code" -- a harness failure
# that reads exactly like a model that has stopped writing Python.
_FENCE = re.compile(
    r"```[ \t]*(?:python3?|py)?[ \t]*\r?\n(.*?)(?:```|\Z)", re.DOTALL | re.IGNORECASE
)
_TOP_LEVEL_IMPORT = re.compile(r"^(?:import[ \t]+\S.*|from[ \t]+\S+[ \t]+import[ \t]+\S.*)$", re.M)


def extract_code(
    generation: str,
    *,
    prompt: str,
    entry_point: str,
    style: PromptStyle,
) -> str:
    """Turn one generation into a program that defines ``entry_point``.

    ``completion`` style is the historical framing: the prompt ends mid-function and the
    model continues it, so the program is the concatenation. The harness has already cut
    the continuation at the stop sequences.

    ``chat`` style is what an *instruct* checkpoint actually produces -- prose, then a
    fenced block, then more prose -- and getting this wrong is the largest harness
    artefact on this benchmark. An instruct model scored by concatenation has its
    `````python`` line appended to a function signature; every problem then fails
    for a syntax error, and the arm reads as catastrophically damaged. All four phase-3
    models are instruct checkpoints, so this is the default path, not the exotic one.
    """
    if style == "completion":
        return prompt + generation

    body = _preferred_block(generation, entry_point=entry_point)
    if not body.strip():
        return ""
    if re.search(rf"^[ \t]*def[ \t]+{re.escape(entry_point)}\b", body, re.MULTILINE):
        # The block redefines the function, so the prompt's signature must not be
        # prepended -- but its imports must, or every problem whose signature is
        # annotated `List[int]` dies with a NameError the model did not cause.
        return _prompt_imports(prompt) + body
    # No signature: treat it as the body the completion framing would have produced.
    return prompt + body


def _preferred_block(generation: str, *, entry_point: str) -> str:
    """The fenced block most likely to be the answer, or the whole text if unfenced.

    Prefers a block that defines the entry point over the first block, because models
    routinely open with a fenced restatement of the *tests* or of the signature before
    writing the solution.
    """
    blocks: list[str] = [block for block in _FENCE.findall(generation) if block.strip()]
    if not blocks:
        return generation
    needle = re.compile(rf"^[ \t]*def[ \t]+{re.escape(entry_point)}\b", re.MULTILINE)
    for block in blocks:
        if needle.search(block):
            return block
    return blocks[0]


def _prompt_imports(prompt: str) -> str:
    lines = _TOP_LEVEL_IMPORT.findall(prompt)
    return "".join(f"{line}\n" for line in lines) + "\n" if lines else ""


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def run_program(
    source: str,
    *,
    allow_execution: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> ExecOutcome:
    """Run one self-contained program and report whether it exited cleanly.

    "Cleanly" means a zero exit code *and* the sentinel, so a candidate that exits
    before its assertions run is a failure rather than a pass.
    """
    _require_opt_in(allow_execution)
    if not source.strip():
        return ExecOutcome(status="empty", detail="no code was extracted from the generation")

    with tempfile.TemporaryDirectory(prefix="dynquant-exec-") as workdir:
        root = Path(workdir)
        program = root / "candidate.py"
        program.write_text(source, encoding="utf-8")
        guard = root / "_dq_guard.py"
        guard.write_text(_GUARD_SOURCE, encoding="utf-8")
        sentinel = root / _SENTINEL_NAME
        stderr_path = root / "_dq_stderr.txt"

        argv = [
            sys.executable,
            "-s",  # no user site-packages
            "-B",  # no __pycache__ next to a file we are about to delete
            str(guard),
            str(memory_mb),
            str(program),
            str(sentinel),
        ]
        started = time.perf_counter()
        with stderr_path.open("wb") as stderr_file:
            process = subprocess.Popen(
                argv,
                cwd=workdir,
                # DEVNULL, not a pipe: a candidate calling `input()` would otherwise
                # block until its timeout and be scored as an infinite loop.
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
                env=_child_env(),
                **_isolation_kwargs(),
            )
            try:
                returncode: int | None = process.wait(timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                _terminate(process)
                returncode, timed_out = None, True
        duration = time.perf_counter() - started

        if timed_out:
            return ExecOutcome(
                status="timeout",
                detail=f"no result within {timeout:g}s",
                duration=duration,
            )
        if returncode == 0 and sentinel.exists():
            return ExecOutcome(status="passed", duration=duration)
        return ExecOutcome(
            status="failed",
            detail=_stderr_tail(stderr_path) or f"exited with status {returncode}",
            duration=duration,
        )


def run_programs(
    sources: Sequence[str],
    *,
    allow_execution: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    memory_mb: int = DEFAULT_MEMORY_MB,
    max_workers: int | None = None,
    retry_timeouts: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> list[ExecOutcome]:
    """Run every program, in parallel, and return outcomes in input order.

    Args:
        retry_timeouts: Re-run each timeout once with nothing else running. A timeout is
            the only outcome that can be a property of the machine rather than of the
            model, and leaving it unconfirmed makes the score depend on how busy the box
            was -- which is indistinguishable, in the results table, from quantization
            damage.
    """
    _require_opt_in(allow_execution)
    if not sources:
        return []

    workers = max_workers if max_workers is not None else min(8, (os.cpu_count() or 2))
    outcomes: list[ExecOutcome | None] = [None] * len(sources)
    done = 0

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                run_program,
                source,
                allow_execution=True,
                timeout=timeout,
                memory_mb=memory_mb,
            ): index
            for index, source in enumerate(sources)
        }
        for future in futures:
            outcomes[futures[future]] = future.result()
            done += 1
            if progress is not None:
                progress(done, len(sources))

    if retry_timeouts:
        contended = [
            i for i, outcome in enumerate(outcomes) if outcome and outcome.status == "timeout"
        ]
        if contended:
            _log.info(
                "re-running %d timed-out candidate(s) serially to separate a slow "
                "solution from a busy machine",
                len(contended),
            )
            for index in contended:
                outcomes[index] = run_program(
                    sources[index],
                    allow_execution=True,
                    timeout=timeout,
                    memory_mb=memory_mb,
                )

    # The futures partition the input, so a `None` here would be a dropped candidate --
    # which would shift every later verdict onto the wrong problem.
    if any(outcome is None for outcome in outcomes):
        missing = sum(outcome is None for outcome in outcomes)
        raise DynQuantError(f"execution covered {len(sources) - missing}/{len(sources)} candidates")
    return [outcome for outcome in outcomes if outcome is not None]


def resolve_style(tokenizer: Any, requested: PromptStyle | Literal["auto"]) -> PromptStyle:
    """Pick the prompting framing, defaulting to what the checkpoint actually is.

    ``auto`` follows the tokenizer: a chat template means an instruct checkpoint, and an
    instruct checkpoint answers a coding prompt with prose around a fenced block whether
    or not the harness was expecting one. Guessing wrong in this direction is the single
    largest artefact on this benchmark -- see :func:`extract_code`.

    "Has a template" is :func:`~dynquant.eval.harness.chat_prompt_style`, which asks the
    tokenizer instead of reading an attribute off it. Reading the attribute misclassified
    every Mistral checkpoint with a ``tekken.json`` as a base model.
    """
    if requested != "auto":
        return requested
    return "chat" if chat_prompt_style(tokenizer) == "chat-template" else "completion"


def prepare_decode(tokenizer: Any, config: Any, *, style: PromptStyle, label: str) -> Any:
    """Force ``add_special_tokens`` off when the prompt came from a chat template.

    The template emits BOS itself, so leaving the tokenizer's own on gives Llama-3 and
    Gemma-3 two of them -- no error, a few points of damage, and the same magnitude as
    the effect being measured.
    """
    from dataclasses import replace

    if style == "chat" and config.add_special_tokens:
        _log.warning(
            "add_special_tokens=True with a chat-templated prompt will prepend a second "
            "BOS; forcing it off for %s",
            label,
        )
        return replace(config, add_special_tokens=False)
    return config


def score_generations(
    *,
    label: str,
    task: str,
    prompt_style: PromptStyle,
    keys: Sequence[str],
    prompts: Sequence[str],
    generations: Sequence[str],
    entry_points: Sequence[str],
    tests: Sequence[str],
    allow_execution: bool = False,
    timeout: float = DEFAULT_TIMEOUT,
    memory_mb: int = DEFAULT_MEMORY_MB,
    max_workers: int | None = None,
    retry_timeouts: bool = True,
    keep_predictions: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> CodeEvalResult:
    """Extract, execute and tally -- the half of an execution-scored task that is the
    same for HumanEval and MBPP.

    ``tests[i]`` is appended after the candidate, so the caller decides what "passing"
    means for its dataset while the failure accounting stays in one place.
    """
    _require_opt_in(allow_execution)
    lengths = {
        "keys": len(keys),
        "prompts": len(prompts),
        "generations": len(generations),
        "entry_points": len(entry_points),
        "tests": len(tests),
    }
    if len(set(lengths.values())) != 1:
        # Zipping mismatched columns would score each generation against some other
        # problem's tests and report a plausible number for it.
        raise DynQuantError(f"misaligned columns: {lengths}")

    candidates = [
        extract_code(generation, prompt=prompt, entry_point=entry_point, style=prompt_style)
        for generation, prompt, entry_point in zip(generations, prompts, entry_points, strict=True)
    ]
    programs = [
        f"{candidate.rstrip()}\n\n{test}\n" if candidate.strip() else ""
        for candidate, test in zip(candidates, tests, strict=True)
    ]
    outcomes = run_programs(
        programs,
        allow_execution=True,
        timeout=timeout,
        memory_mb=memory_mb,
        max_workers=max_workers,
        retry_timeouts=retry_timeouts,
        progress=progress,
    )

    result = CodeEvalResult(
        label=label,
        task=task,
        total=len(programs),
        passed=sum(outcome.passed for outcome in outcomes),
        timeouts=sum(outcome.status == "timeout" for outcome in outcomes),
        empty=sum(outcome.status == "empty" for outcome in outcomes),
        prompt_style=prompt_style,
        sandbox=sandbox_fingerprint(timeout=timeout, memory_mb=memory_mb),
        hits=[outcome.passed for outcome in outcomes],
        statuses=[outcome.status for outcome in outcomes],
        keys=list(keys),
    )
    for index, (key, generation, candidate, outcome) in enumerate(
        zip(keys, generations, candidates, outcomes, strict=True)
    ):
        if index < keep_predictions:
            result.predictions.append(
                {
                    "key": key,
                    "generation": generation,
                    "candidate": candidate,
                    "status": outcome.status,
                    "detail": outcome.detail,
                }
            )
    _log.info("%s", result.summary())
    return result


def _require_opt_in(allow_execution: bool) -> None:
    if not allow_execution:
        raise DynQuantError(
            "this task scores by running model-written code, which is off by default. "
            "Pass allow_execution=True once you have read what the sandbox does and "
            "does not protect against (dynquant.eval._code_exec docstring)."
        )


def _has_rlimits() -> bool:
    try:
        import resource  # noqa: F401
    except ImportError:
        return False
    return True


def _child_env() -> dict[str, str]:
    """A minimal environment for the candidate.

    Allow-list rather than a copy: the parent's environment is where a fine-tuning run
    keeps its HF token, its W&B key and its CUDA configuration, and none of that belongs
    in a process running text a model wrote.

    ``PYTHONHASHSEED`` is pinned for the same reason langdetect is seeded -- a solution
    whose output depends on set iteration order would otherwise flip between runs and
    read as quantization noise. The thread caps matter because a candidate importing
    numpy inside eight concurrent workers otherwise starts eight full thread pools and
    the box spends the evaluation context-switching.
    """
    keep = (
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
    )
    env = {name: os.environ[name] for name in keep if name in os.environ}
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONWARNINGS": "ignore",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return env


def _isolation_kwargs() -> dict[str, Any]:
    """Put the child somewhere it can be killed as a unit.

    On POSIX a new session means :func:`_terminate` can signal the whole process group,
    so a candidate that spawned a helper does not leave it running after the timeout. On
    Windows the equivalent needs a Job object; the new process group gets us a clean
    ``kill`` of the child itself, and a grandchild would survive. Model-written HumanEval
    solutions do not spawn processes, so this is a documented gap rather than a live one.

    Branched on ``sys.platform`` rather than ``os.name`` because only the former narrows
    for a type checker: ``subprocess.CREATE_NEW_PROCESS_GROUP`` does not exist in the
    POSIX stubs and ``os.killpg`` does not exist in the Windows ones, so under ``os.name``
    each platform's mypy run reports the *other* platform's branch as an error. The two
    tests are equivalent at runtime; this one is also checkable.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        # An `else` rather than a fall-through return, which reads better without it.
        # Under `warn_unreachable` mypy elides the branch its platform did not take but
        # still walks the statement *after* an always-returning `if`, so the bare
        # trailing return is an "unreachable" error on Windows and the `else` is not.
        return {"start_new_session": True}


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if sys.platform != "win32":
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            process.kill()
    else:
        process.kill()
    # Reap it. Without this the child stays a zombie holding the stderr fd, and the
    # temporary directory cannot be removed on Windows.
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover -- an unkillable child
        _log.warning("candidate process %d survived SIGKILL", process.pid)


def _stderr_tail(path: Path, *, limit: int = _STDERR_TAIL) -> str:
    """The last ``limit`` bytes of stderr.

    Read from the end rather than whole, because the file is as large as the candidate
    chose to make it and this runs once per problem.
    """
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            raw = handle.read()
    except OSError:  # pragma: no cover -- the file is created before the child starts
        return ""
    text = raw.decode("utf-8", errors="replace").strip()
    return f"...{text}" if size > limit else text
