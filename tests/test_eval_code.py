"""HumanEval and MBPP: the sandbox, the extractor, and the two tasks on top of them.

An execution-scored benchmark has two halves that can each be wrong without saying so.
The **sandbox** decides what "passed" means, and its interesting failures all point the
same way -- a candidate that exits before its assertions run, a timeout that was really
a busy machine, a child that inherited the parent's environment -- so each of those has
a test rather than a comment. The **extractor** decides what code the model actually
wrote, and getting it wrong on an instruct checkpoint (which all four phase-3 models
are) fails every problem for a syntax error and reads as catastrophic quantization
damage.

Subprocesses are real here. Stubbing them would test the tally and nothing else, and
the tally is the part that was never in doubt. No model is loaded; generation is stubbed.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from dynquant.errors import DynQuantError
from dynquant.eval import _code_exec, humaneval, mbpp
from dynquant.eval._code_exec import (
    ExecOutcome,
    extract_code,
    run_program,
    run_programs,
    sandbox_fingerprint,
    score_generations,
)
from dynquant.eval.harness import EvalConfig
from dynquant.eval.humaneval import (
    COMPLETION_STOPS,
    HumanEvalExample,
    build_test_program,
    evaluate_humaneval,
)
from dynquant.eval.mbpp import MbppExample, entry_point_of, evaluate_mbpp

# A short timeout everywhere it cannot change a verdict: the passing programs here run
# in milliseconds, and the suite pays this per subprocess.
FAST = 15.0
SLOW = 1.5


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


class _Tokenizer:
    """Just enough tokenizer to choose a framing and render a template."""

    def __init__(self, chat_template: str | None = "template") -> None:
        self.chat_template = chat_template

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        return f"<|user|>{messages[0]['content']}<|assistant|>"


ADD = HumanEvalExample(
    task_id="HumanEval/test",
    prompt=(
        "from typing import List\n"
        "\n"
        "\n"
        "def add_all(numbers: List[int]) -> int:\n"
        '    """Return the sum of the list."""\n'
    ),
    entry_point="add_all",
    test=(
        "METADATA = {}\n"
        "\n"
        "\n"
        "def check(candidate):\n"
        "    assert candidate([1, 2, 3]) == 6\n"
        "    assert candidate([]) == 0\n"
    ),
)

SHARED = MbppExample(
    task_id="7",
    text="Write a function to find the shared items in two lists.",
    tests=("assert set(shared_items([1, 2], [2, 3])) == {2}",),
    code="def shared_items(a, b):\n    return set(a) & set(b)\n",
)


def _run_humaneval(
    generations: list[str],
    examples: list[HumanEvalExample],
    *,
    tokenizer: Any | None = None,
    generate: Any | None = None,
    **kwargs: Any,
) -> Any:
    tokenizer = tokenizer if tokenizer is not None else _Tokenizer()
    stub = generate or (lambda *args, **_kw: list(generations))
    with mock.patch.object(humaneval, "generate_batched", stub):
        return evaluate_humaneval(
            object(),
            tokenizer,
            examples,
            label="test",
            allow_execution=True,
            timeout=FAST,
            max_workers=2,
            **kwargs,
        )


def _run_mbpp(
    generations: list[str],
    examples: list[MbppExample],
    *,
    tokenizer: Any | None = None,
    generate: Any | None = None,
    **kwargs: Any,
) -> Any:
    tokenizer = tokenizer if tokenizer is not None else _Tokenizer()
    stub = generate or (lambda *args, **_kw: list(generations))
    with mock.patch.object(mbpp, "generate_batched", stub):
        return evaluate_mbpp(
            object(),
            tokenizer,
            examples,
            label="test",
            allow_execution=True,
            timeout=FAST,
            max_workers=2,
            **kwargs,
        )


def _fenced(code: str, *, preamble: str = "Sure, here you go:\n\n") -> str:
    return f"{preamble}```python\n{code}\n```\n\nHope that helps!"


# --------------------------------------------------------------------------
# The opt-in gate
# --------------------------------------------------------------------------


def test_execution_is_off_until_the_caller_opts_in() -> None:
    """Importing an evaluation module must not be enough to run code a model wrote.

    Turns red when: the default flips to True. Nothing else in the package would
    notice, because every caller in this repo passes it explicitly.
    """
    for call in (
        lambda: run_program("pass"),
        lambda: run_programs(["pass"]),
        lambda: _code_exec.score_generations(
            label="x",
            task="t",
            prompt_style="chat",
            keys=[],
            prompts=[],
            generations=[],
            entry_points=[],
            tests=[],
        ),
    ):
        with pytest.raises(DynQuantError, match="allow_execution"):
            call()


def test_the_task_entry_points_refuse_too() -> None:
    """Turns red when: the gate is checked in the runner but not on the way in.

    An evaluation that only fails after generating 164 completions has already spent
    the GPU time it was supposed to protect.
    """
    with (
        pytest.raises(DynQuantError, match="allow_execution"),
        mock.patch.object(humaneval, "generate_batched", lambda *a, **k: [""]),
    ):
        evaluate_humaneval(object(), _Tokenizer(), [ADD], label="x")


# --------------------------------------------------------------------------
# The sandbox
# --------------------------------------------------------------------------


def test_a_passing_program_passes_and_a_failing_one_fails() -> None:
    assert run_program("assert 1 + 1 == 2", allow_execution=True, timeout=FAST).status == "passed"
    outcome = run_program("assert 1 + 1 == 3", allow_execution=True, timeout=FAST)
    assert outcome.status == "failed"
    assert "AssertionError" in outcome.detail


@pytest.mark.parametrize("exit_call", ["import sys; sys.exit(0)", "import os; os._exit(0)"])
def test_exiting_zero_before_the_tests_run_is_not_a_pass(exit_call: str) -> None:
    """A zero exit code is necessary but not sufficient.

    Turns red when: the sentinel check is dropped and the verdict becomes the exit code
    alone. Every candidate that calls ``exit()`` -- which models emit, especially when
    they think they are writing a script -- then scores as correct, and the arm that
    does it most looks the best.
    """
    outcome = run_program(f"{exit_call}\nassert False", allow_execution=True, timeout=FAST)
    assert outcome.status == "failed"


def test_an_infinite_loop_times_out_instead_of_hanging_the_run() -> None:
    outcome = run_program("while True:\n    pass", allow_execution=True, timeout=SLOW)
    assert outcome.status == "timeout"
    assert outcome.duration < SLOW * 10


def test_reading_stdin_fails_immediately_instead_of_burning_the_timeout() -> None:
    """Turns red when: stdin becomes a pipe or is inherited.

    A candidate calling ``input()`` would then block until the timeout and be counted
    as an infinite loop -- and with a few dozen of them, the evaluation's wall clock
    stops being about the model at all.
    """
    outcome = run_program("value = input()", allow_execution=True, timeout=SLOW * 8)
    assert outcome.status == "failed"
    assert outcome.duration < SLOW * 4


def test_no_code_is_its_own_outcome() -> None:
    """Turns red when: an empty generation is folded into ordinary failures.

    A model that stops emitting code has stopped following the format, which is a
    different diagnosis from one that emits wrong code -- and it is the one that says
    the bit map went too far.
    """
    assert run_program("   \n  ", allow_execution=True, timeout=FAST).status == "empty"


def test_the_child_cannot_see_the_parents_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turns red when: the allow-list is replaced by a copy of ``os.environ``.

    The process running this is the one holding an HF token, a W&B key and whatever
    else a fine-tuning box accumulates. None of it belongs in a process executing text
    a language model produced.
    """
    monkeypatch.setenv("DYNQUANT_TEST_SECRET", "hunter2")
    program = "import os\nassert 'DYNQUANT_TEST_SECRET' not in os.environ"
    assert run_program(program, allow_execution=True, timeout=FAST).status == "passed"


def test_a_candidate_runs_in_its_own_directory() -> None:
    """Turns red when: ``cwd`` stops being the temporary directory.

    Model-written code writes files -- caches, plots, "output.txt". Inheriting the
    caller's working directory scatters those through a checkpoint tree, and the
    failure surfaces days later as a mysteriously dirty repository.
    """
    program = (
        "from pathlib import Path\n"
        "Path('scratch.txt').write_text('x', encoding='utf-8')\n"
        "assert 'dynquant-exec-' in str(Path.cwd()), Path.cwd()\n"
    )
    assert run_program(program, allow_execution=True, timeout=FAST).status == "passed"
    assert not Path("scratch.txt").exists()


def test_unbounded_child_output_does_not_reach_the_parent() -> None:
    """Turns red when: stderr is read whole, or stdout becomes a pipe.

    A candidate printing in a loop emits as much as its timeout allows. Through a pipe
    that lands in the parent's memory, once per problem, in parallel.
    """
    program = "import sys\nfor _ in range(20000):\n    sys.stderr.write('x' * 100 + '\\n')\nassert False\n"
    outcome = run_program(program, allow_execution=True, timeout=FAST)
    assert outcome.status == "failed"
    assert len(outcome.detail) < 4000


def test_outcomes_come_back_in_input_order() -> None:
    """Turns red when: results are collected in completion order.

    Every verdict would then be attributed to whichever problem happened to finish in
    that slot, and the paired vector -- the entire point of storing hits -- would be
    scrambled against the other arm's.
    """
    sources = ["assert True", "assert False", "assert True", "assert False"]
    outcomes = run_programs(sources, allow_execution=True, timeout=FAST, max_workers=4)
    assert [outcome.status for outcome in outcomes] == ["passed", "failed", "passed", "failed"]


def test_a_timeout_is_confirmed_serially_before_it_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Turns red when: the re-run is removed.

    A timeout is the only outcome that can be a property of the machine rather than of
    the model: a 5.9 s solution passes alone and fails under eight-way contention. Left
    unconfirmed, the score depends on how busy the box was -- which is indistinguishable,
    in the results table, from quantization damage.
    """
    calls: list[str] = []

    def flaky(source: str, **_kwargs: Any) -> ExecOutcome:
        calls.append(source)
        first = calls.count(source) == 1
        return ExecOutcome(status="timeout" if first else "passed")

    monkeypatch.setattr(_code_exec, "run_program", flaky)
    outcomes = run_programs(["slow"], allow_execution=True)
    assert calls == ["slow", "slow"]
    assert outcomes[0].status == "passed"

    calls.clear()
    outcomes = run_programs(["slow"], allow_execution=True, retry_timeouts=False)
    assert calls == ["slow"]
    assert outcomes[0].status == "timeout"


def test_the_sandbox_fingerprint_names_the_bounds_that_can_change_a_verdict() -> None:
    """Turns red when: a bound stops being recorded.

    Two arms run under different timeouts are not comparable, and the pass rates alone
    do not say so. The interpreter is in there too: a solution using ``match`` passes on
    one minor version and is a syntax error on another.
    """
    fingerprint = sandbox_fingerprint(timeout=8.0, memory_mb=2048)
    assert "t=8s" in fingerprint
    assert "2048MB" in fingerprint
    assert f"py{sys.version_info.major}.{sys.version_info.minor}" in fingerprint
    assert sandbox_fingerprint(timeout=3.0, memory_mb=2048) != fingerprint


@pytest.mark.skipif(os.name == "nt", reason="Windows has no rlimits")
def test_a_memory_bomb_is_bounded_rather_than_taking_the_box_with_it() -> None:
    """Turns red when: the rlimit is dropped on the platforms that have one.

    Eight workers each allocating without a ceiling is how an evaluation gets killed by
    the OOM reaper -- and the process the reaper picks is usually the largest one, which
    is the parent holding the model.
    """
    program = "big = bytearray(3 * 1024 * 1024 * 1024)\n"
    outcome = run_program(program, allow_execution=True, timeout=FAST, memory_mb=256)
    assert outcome.status == "failed"


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def test_an_instruct_answer_is_read_out_of_its_fenced_block() -> None:
    """Turns red when: the chat framing concatenates instead of extracting.

    The prose around the block is appended to a function signature, every problem is a
    syntax error, and the arm reads as destroyed by quantization.
    """
    code = extract_code(
        _fenced("def f(x):\n    return x + 1"),
        prompt="def f(x):\n",
        entry_point="f",
        style="chat",
    )
    assert "Sure" not in code
    assert code.strip().startswith("def f(x):")


def test_a_fence_the_generation_never_closed_is_still_read() -> None:
    """Turns red when: the closing fence becomes mandatory.

    A long solution can hit ``max_new_tokens`` mid-block. Requiring the close scores a
    nearly complete answer as "produced no code", and it does so most often on the
    longest problems -- exactly where a quantized model is already weakest, so the
    artefact points the same way as the effect and is invisible in the totals.
    """
    truncated = "Here:\n```python\ndef f(x):\n    return x + 1"
    code = extract_code(truncated, prompt="def f(x):\n", entry_point="f", style="chat")
    assert code.strip().endswith("return x + 1")
    # The body alone is not enough: with the close made mandatory the regex matches
    # nothing, `_preferred_block` hands back the whole generation, and the code still
    # *ends* with the right line -- while beginning with "Here:" and a stray fence.
    ast.parse(code)


def test_the_block_that_defines_the_function_wins_over_the_first_block() -> None:
    """Models restate the tests, or the signature, in a fence before answering.

    Turns red when: the first block is taken unconditionally. The program is then the
    example the model was quoting, which usually runs and usually fails.
    """
    generation = (
        "First, the tests:\n```python\nassert f(1) == 2\n```\n"
        "Now the solution:\n```python\ndef f(x):\n    return x + 1\n```"
    )
    code = extract_code(generation, prompt="def f(x):\n", entry_point="f", style="chat")
    assert "assert" not in code
    assert "return x + 1" in code


def test_a_standalone_answer_keeps_the_prompts_imports() -> None:
    """Turns red when: imports stop being carried over.

    HumanEval signatures are annotated ``List[int]``, and the import lives in the prompt
    the model was shown but did not repeat. Without it every annotated problem dies with
    a NameError the model did not cause -- roughly a fifth of the benchmark, silently.
    """
    code = extract_code(
        _fenced("def f(xs: List[int]) -> int:\n    return sum(xs)"),
        prompt="from typing import List\n\n\ndef f(xs: List[int]) -> int:\n",
        entry_point="f",
        style="chat",
    )
    assert code.startswith("from typing import List")
    assert code.count("def f") == 1


def test_a_bare_body_is_appended_to_the_prompt() -> None:
    """Some instruct models answer with the body only, having been given the signature."""
    code = extract_code(
        _fenced("    return sum(xs)"),
        prompt="def f(xs):\n",
        entry_point="f",
        style="chat",
    )
    assert code.rstrip() == "def f(xs):\n    return sum(xs)"


def test_an_unfenced_answer_is_taken_whole() -> None:
    code = extract_code(
        "def f(x):\n    return x + 1\n", prompt="def f(x):\n", entry_point="f", style="chat"
    )
    assert code.strip().startswith("def f(x):")


def test_completion_style_concatenates_prompt_and_generation() -> None:
    code = extract_code(
        "    return x + 1\n", prompt="def f(x):\n", entry_point="f", style="completion"
    )
    assert code == "def f(x):\n    return x + 1\n"


# --------------------------------------------------------------------------
# HumanEval
# --------------------------------------------------------------------------


def test_the_test_field_defines_check_but_never_calls_it() -> None:
    """Turns red when: the ``check(entry_point)`` call is dropped.

    The dataset's ``test`` field only defines the function. Appended alone it produces a
    program that defines two functions, exits zero, and scores every arm at 100 % --
    including a model that emitted nothing but a stub.
    """
    assert "check(add_all)" not in ADD.test
    assert build_test_program(ADD).rstrip().endswith("check(add_all)")


def test_completion_stops_are_anchored_at_column_zero() -> None:
    """Turns red when: a stop loses its leading newline.

    ``"if"`` or ``"print"`` unanchored truncates most solutions at their first
    conditional. The model looks like it cannot finish a function; the harness could
    not let it.
    """
    assert all(stop.startswith("\n") for stop in COMPLETION_STOPS)


def test_an_instruct_checkpoint_is_prompted_as_one() -> None:
    """Turns red when: ``auto`` stops following the tokenizer.

    All four phase-3 models are instruct checkpoints. Scored under the completion
    framing they append "Sure! Here's the function:" to a signature and fail every
    problem for a syntax error.
    """
    assert _run_humaneval([_fenced("def add_all(n):\n    return sum(n)")], [ADD]).prompt_style == (
        "chat"
    )
    result = _run_humaneval(
        ["    return sum(numbers)\n"], [ADD], tokenizer=_Tokenizer(chat_template=None)
    )
    assert result.prompt_style == "completion"
    assert result.hits == [True]


def test_a_chat_framing_forces_add_special_tokens_off() -> None:
    """The template already emits BOS; the tokenizer would emit a second.

    Turns red when: the override is removed. Costs a few points, reports no error, and
    is the same size as the effect being measured.
    """
    seen: dict[str, Any] = {}

    def spy(
        _model: Any, _tok: Any, prompts: list[str], config: EvalConfig, **_kw: Any
    ) -> list[str]:
        seen["add_special_tokens"] = config.add_special_tokens
        return [""] * len(prompts)

    _run_humaneval([], [ADD], config=EvalConfig(add_special_tokens=True), generate=spy)
    assert seen["add_special_tokens"] is False


def test_a_correct_solution_passes_and_a_wrong_one_fails() -> None:
    result = _run_humaneval(
        [
            _fenced("def add_all(numbers: List[int]) -> int:\n    return sum(numbers)"),
            _fenced("def add_all(numbers: List[int]) -> int:\n    return sum(numbers) + 1"),
        ],
        [ADD, ADD],
    )
    assert result.hits == [True, False]
    assert result.passed == 1
    assert result.pass_at_1 == pytest.approx(0.5)


def test_hits_and_keys_are_one_per_problem_in_dataset_order() -> None:
    """Turns red when: either vector stops lining up with the problems.

    Two arms are compared position by position. A vector that is one short, or in
    another order, still produces a McNemar table and still produces a p-value.
    """
    result = _run_humaneval([_fenced("def add_all(n):\n    return sum(n)")] * 2, [ADD, ADD])
    assert len(result.hits) == len(result.keys) == len(result.statuses) == 2
    assert result.keys == [ADD.task_id, ADD.task_id]


def test_misaligned_columns_are_refused() -> None:
    """Turns red when: the length check goes.

    ``zip`` would then score each generation against some other problem's tests and
    report a plausible number for it.
    """
    with pytest.raises(DynQuantError, match="misaligned"):
        score_generations(
            label="x",
            task="humaneval",
            prompt_style="chat",
            keys=["a", "b"],
            prompts=["p"],
            generations=["g"],
            entry_points=["f"],
            tests=["assert True"],
            allow_execution=True,
        )


def test_the_result_survives_a_round_trip_to_a_dict() -> None:
    result = _run_humaneval([_fenced("def add_all(n):\n    return sum(n)")], [ADD])
    payload = result.as_dict()
    assert payload["hits"] == result.hits
    assert payload["sandbox"] == result.sandbox
    assert payload["prompt_style"] == "chat"


# --------------------------------------------------------------------------
# MBPP
# --------------------------------------------------------------------------


def test_the_tests_are_shown_because_the_statement_does_not_name_the_function() -> None:
    """Turns red when: the asserts are dropped from either framing.

    "Write a function to find the shared items in two lists" does not say whether to
    call it ``shared_items`` or ``common``. The tests call one. Without them the score
    measures the model's luck at guessing identifiers, and it measures it lower for
    every arm equally -- so the comparison flattens rather than breaking.
    """
    tokenizer = _Tokenizer()
    for style in ("completion", "chat"):
        prompt = mbpp.build_prompt(SHARED, tokenizer, style=style)  # type: ignore[arg-type]
        assert "shared_items" in prompt
        assert SHARED.text in prompt


@pytest.mark.parametrize(
    ("assertion", "expected"),
    [
        ("assert shared_items([1], [1]) == [1]", "shared_items"),
        ("assert set(shared_items([1], [1])) == {1}", "shared_items"),
        ("assert math.isclose(area(2), 4.0, rel_tol=0.001)", "area"),
        ("assert len(sorted(pack(x))) == 3", "pack"),
    ],
)
def test_the_entry_point_is_read_out_of_the_asserts(assertion: str, expected: str) -> None:
    """The task statement never names the function, so the tests have to.

    Turns red when: the wrapper skip list or the attribute-call skip goes. The entry
    point then comes back as ``set`` or as nothing, the wrong fenced block is chosen,
    and the failure looks like a model that answered a different question.
    """
    example = MbppExample(task_id="1", text="t", tests=(assertion,))
    assert entry_point_of(example) == expected


def test_the_entry_point_falls_back_to_the_reference_definition() -> None:
    """Turns red when: the fallback goes and an unparseable assert yields nothing."""
    example = MbppExample(
        task_id="1", text="t", tests=("assert ((( broken",), code="def solve(a):\n    return a\n"
    )
    assert entry_point_of(example) == "solve"


def test_setup_code_runs_before_the_asserts() -> None:
    """A handful of MBPP problems need an import the candidate was never asked for.

    Turns red when: ``test_setup_code`` is dropped. Those problems fail with a NameError
    for every arm, which looks like a hard subset rather than a missing line.
    """
    example = MbppExample(
        task_id="1",
        text="t",
        tests=("assert math.isclose(area(2), 4.0)",),
        setup="import math",
    )
    program = mbpp.build_test_program(example)
    assert program.splitlines()[0] == "import math"


def test_mbpp_does_not_prepend_its_task_statement_to_the_program() -> None:
    """Turns red when: the prompt is passed through to the extractor.

    MBPP's prompt is English. Prepended to the candidate it is a syntax error on every
    problem -- an arm that scores zero for a reason that has nothing to do with weights.
    """
    result = _run_mbpp(
        [_fenced("def shared_items(a, b):\n    return list(set(a) & set(b))")], [SHARED]
    )
    assert result.hits == [True]
    assert result.task == "mbpp"

    # A block that defines the entry point discards the prompt either way, so the pass
    # above cannot see the difference. The prompt is only *used* when the block does not
    # define what the tests call -- and then it decides whether the run reports the
    # model's mistake or the harness's. English at the top is a SyntaxError on every
    # problem; an empty prompt leaves the model's own NameError.
    misnamed = _run_mbpp(
        [_fenced("def common(a, b):\n    return list(set(a) & set(b))")],
        [SHARED],
        keep_predictions=1,
    )
    assert misnamed.hits == [False]
    assert "NameError" in misnamed.predictions[0]["detail"]
    assert "SyntaxError" not in misnamed.predictions[0]["detail"]


def test_a_wrong_mbpp_solution_fails() -> None:
    result = _run_mbpp([_fenced("def shared_items(a, b):\n    return []")], [SHARED])
    assert result.hits == [False]
    assert result.statuses == ["failed"]


def test_a_chat_run_ignores_few_shot_exemplars() -> None:
    """Turns red when: exemplars leak into a chat prompt.

    An instruct model given both an instruction and a worked pattern answers in the
    pattern's format, which the extractor was not built for -- and the exemplars'
    reference solutions are gold code sitting in the context window.
    """
    prompt = mbpp.build_prompt(SHARED, _Tokenizer(), style="chat", shots=[SHARED])
    assert prompt.count("shared_items") == SHARED.tests[0].count("shared_items")
    assert SHARED.code not in prompt


def test_a_generation_with_no_code_is_counted_apart_from_a_failure() -> None:
    result = _run_mbpp(["I'm sorry, I can't help with that."], [SHARED])
    assert result.empty == 0  # prose is still *something*; it just does not run
    assert result.hits == [False]

    blank = _run_mbpp([""], [SHARED])
    assert blank.empty == 1
    assert blank.statuses == ["empty"]
