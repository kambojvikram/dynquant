"""The guard must fire on the real files it was written for.

``scripts/check_no_confidential.py`` shipped with a pattern that looked right and
caught nothing: ``hf_[A-Za-z0-9]{8,}`` cannot match ``hf_token_here``, because the
underscore ends the character class after five characters. The guard passed the
exact file whose literal is quoted in its own docstring.

A secret scanner that never fires is worse than no scanner, because the green tick
is taken as evidence. So the fixtures here are the actual research files, not
synthetic strings -- a synthetic ``token="hf_abcd1234"`` would have passed the
broken pattern too, and the bug would have survived its own test.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD = REPO_ROOT / "scripts" / "check_no_confidential.py"

#: The two research files carrying the literals, and what must be found in each.
REAL_OFFENDERS = {
    "fine-tuning_and_stats_hook/supervised_finetuning.py": "Hugging Face token",
    "inference/inference_3bits.py": "home directory",
}


@pytest.fixture(scope="module")
def guard():
    spec = importlib.util.spec_from_file_location("_dq_guard", GUARD)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dq_guard"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("relpath", "expected"), sorted(REAL_OFFENDERS.items()))
def test_the_guard_fires_on_the_real_research_files(guard, relpath, expected):
    target = REPO_ROOT / relpath
    if not target.is_file():
        pytest.skip(f"{relpath} not present (research tree is not part of the wheel)")

    problems = guard.check([str(target.relative_to(REPO_ROOT))])
    assert problems, f"guard passed {relpath}, which carries a literal it must catch"
    assert any(expected in p for p in problems), problems


def test_the_placeholder_form_is_what_gets_caught(guard, tmp_path):
    """``hf_token_here`` is a placeholder, and that is exactly why it must fail.

    Matching only a well-formed token means the placeholder ships, and whoever
    fills it in ships a live credential with no warning.
    """
    sample = tmp_path / "sft.py"
    sample.write_text('    token="hf_token_here"  # replace me\n', encoding="utf-8")
    assert guard.check([str(sample)])


def test_environment_reads_are_not_flagged(guard, tmp_path):
    """The fix has to be accepted, or the guard just teaches people to bypass it."""
    sample = tmp_path / "ok.py"
    sample.write_text(
        "token = os.environ.get('HF_TOKEN')\n"
        'token = os.getenv("HF_TOKEN")\n'
        "token = None\n"
        'parser.add_argument("--model-path", default="/home/user/models")\n',
        encoding="utf-8",
    )
    assert guard.check([str(sample)]) == []


def test_prose_may_contain_absolute_paths(guard, tmp_path):
    """A path in a README is an example; a path in source is a defect."""
    doc = tmp_path / "README.md"
    doc.write_text('Run it with `--out "/home/alice/models"`.\n', encoding="utf-8")
    assert guard.check([str(doc)]) == []

    code = tmp_path / "run.py"
    code.write_text('OUT = "/home/alice/models"\n', encoding="utf-8")
    assert guard.check([str(code)])


def test_the_reviewer_pdf_can_never_be_committed(guard):
    """The confidentiality half. ``.gitignore`` is bypassable with ``git add -f``."""
    assert guard.check(["20710_DynQuant_Dynamic_Signal_.pdf"])
    assert guard.check(["docs/whatever.PDF"])
    assert guard.check(["model.safetensors"])


def test_the_guard_exempts_the_files_that_must_quote_the_literals(guard):
    """The audit and this test quote the offending lines verbatim, as evidence."""
    assert guard.check(["docs/legacy-audit.md"]) == []
    assert guard.check(["tests/test_confidential_guard.py"]) == []
    assert guard.check(["scripts/check_no_confidential.py"]) == []


def test_the_moment_sidecars_the_ignore_rules_invite_are_not_then_refused(guard):
    """``.gitignore`` un-ignores these two paths; the guard has to agree with it.

    It did not, and CI was red for two pushes over it: ``*.safetensors`` in the guard
    against ``!**/stats/dynquant_moments.safetensors`` in ``.gitignore``, one file
    inviting exactly what the other refuses. Nothing caught it locally, because the
    repository-wide test below filtered to ``TEXT_SUFFIXES`` and so never looked at a
    single file of the type the rule is about.
    """
    for arm in ("ministral-8b.tulu3", "phi4-mini.tulu3"):
        assert (
            guard.check([f"experiments/phase3/s2_runs/{arm}/stats/dynquant_moments.safetensors"])
            == []
        )
    assert guard.check(["tests/fixtures/tiny.safetensors"]) == []


def test_the_carve_out_is_the_filename_and_not_the_directory(guard):
    """A checkpoint dropped beside the sidecars is still refused.

    The exception is granted to one measurement file, not to ``stats/``. Widening it to
    the directory would make the obvious accident -- ``cp model.safetensors stats/`` --
    the one thing that passes.
    """
    assert guard.check(["experiments/phase3/s2_runs/x/stats/model.safetensors"])
    assert guard.check(["experiments/phase3/s2_runs/x/dynquant_moments.safetensors"])
    assert guard.check(["experiments/phase3/s2_runs/x/stats/dynquant_moments.bin"])


def test_the_whole_repository_is_clean_except_the_research_tree(guard):
    """Everything DynQuant itself ships must pass, or the hook is unusable.

    The file list is ``git ls-files``, which is the same list the CI job pipes into the
    guard -- so this test and that job cannot reach different verdicts. It used to be an
    ``rglob`` narrowed to ``TEXT_SUFFIXES``, and that filter is what let two committed
    ``.safetensors`` files turn CI red while the suite stayed green: the test excluded
    the one file type the rule it was checking is written about. Tracked files also
    means an untracked local ``out/`` full of checkpoints cannot fail somebody's run.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode != 0:
        pytest.skip("not a git checkout")
    paths = [p for p in tracked.stdout.split("\0") if p]
    assert len(paths) > 100, "git ls-files returned too little to be the repository"

    # The research tree is the thing being audited; it is expected to fail.
    audited = ("dynquant_paper/", "inference/", "fine-tuning_and_stats_hook/", "pipeline/")
    problems = guard.check([p for p in paths if not p.startswith(audited)])
    assert problems == [], problems
