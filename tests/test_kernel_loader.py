"""Strict-mode semantics of the kernels loader.

``$DYNQUANT_KERNELS_STRICT`` is what stops a broken kernels wheel from passing CI
by simply never being used: without it, a wheel that fails to load degrades to the
torch backend and every test still passes. So the flag's behaviour is itself load
bearing, and it has exactly one subtlety -- it must *not* fire when the extension
is unavailable for a legitimate reason, namely a CPU-only build on a machine with
no GPU, which is the steady state of the CPU-only CI job.

The module is loaded by path rather than imported, for the same reason
``test_abi.py`` reads source text: ``dynquant_kernels`` is a compiled wheel that is
not installed on a CPU development box, and a guard that only runs where the wheel
exists is a guard that never runs. ``_loader.py`` imports nothing but the standard
library at module scope, which is what makes this possible -- and is worth keeping
that way.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LOADER_PATH = (
    REPO_ROOT / "packages" / "dynquant-kernels" / "src" / "dynquant_kernels" / "_loader.py"
)
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

ENV_NAME = "DYNQUANT_KERNELS_STRICT"


@pytest.fixture(scope="module")
def loader():
    """``_loader`` executed under a private module name.

    Private so it cannot shadow or be shadowed by an installed
    ``dynquant_kernels._loader`` -- on a machine where the wheel *is* installed,
    both must be able to coexist in one interpreter.
    """
    assert LOADER_PATH.is_file(), f"{LOADER_PATH} is missing"
    name = "_dynquant_loader_under_test"
    spec = importlib.util.spec_from_file_location(name, LOADER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def strict(monkeypatch):
    monkeypatch.setenv(ENV_NAME, "1")


def test_the_env_var_name_is_the_one_ci_sets(loader):
    """Renaming the variable in one place and not the other silently removes the
    only thing making the kernels CI job meaningful."""
    assert loader._ENV_FORCE_ERROR == ENV_NAME
    assert CI_WORKFLOW.is_file(), f"{CI_WORKFLOW} is missing"
    assert ENV_NAME in CI_WORKFLOW.read_text(encoding="utf-8")


def test_a_working_extension_never_raises(loader, strict):
    loader.KernelLoadResult(available=True).raise_if_strict()


def test_nothing_raises_without_the_env_var(loader, monkeypatch):
    monkeypatch.delenv(ENV_NAME, raising=False)
    loader.KernelLoadResult(available=False, error="broken").raise_if_strict()


def test_a_broken_extension_raises_under_strict(loader, strict):
    result = loader.KernelLoadResult(
        available=False,
        error="the extension imported but its operators are not registered",
        remedy="Reinstall dynquant-kernels.",
    )
    with pytest.raises(ImportError) as excinfo:
        result.raise_if_strict()
    message = str(excinfo.value)
    # Both halves must survive into the exception: the error says what happened,
    # the remedy is the only part a reader can act on, and a CI log is often all
    # anyone ever sees of either.
    assert "operators are not registered" in message
    assert "Reinstall dynquant-kernels." in message
    assert ENV_NAME in message


def test_an_expected_outcome_does_not_raise_under_strict(loader, strict):
    """The CPU-only-build-on-a-CPU-box case, which the CPU kernels job depends on."""
    result = loader.KernelLoadResult(
        available=False,
        error="CPU-only extension and no CUDA device; using the torch backend",
        expected=True,
    )
    result.raise_if_strict()


def test_expected_defaults_to_false(loader):
    """Fail closed. A failure branch added later without thinking about strict mode
    must raise, not quietly opt itself out of the check."""
    assert loader.KernelLoadResult(available=False).expected is False


def test_only_the_cpu_only_branch_is_exempt(loader):
    """``expected=True`` appears exactly once in the loader.

    A second exemption is not necessarily wrong, but it is a decision about what CI
    stops catching, so it has to be a deliberate edit to this test rather than a
    line added to a long function.
    """
    source = LOADER_PATH.read_text(encoding="utf-8")
    assert source.count("expected=True") == 1


def test_the_loader_imports_only_the_standard_library(loader):
    """What makes this file testable at all -- and what makes the kernels wheel
    diagnosable on its own, with no dependency on ``dynquant``."""
    source = LOADER_PATH.read_text(encoding="utf-8")
    module_level = [
        line
        for line in source.splitlines()
        if (line.startswith("import ") or line.startswith("from ")) and "__future__" not in line
    ]
    allowed = {"importlib", "os", "platform", "sys", "dataclasses", "typing"}
    for line in module_level:
        root = line.split()[1].split(".")[0]
        assert root in allowed, f"module-level import of {root!r} in _loader.py: {line}"
