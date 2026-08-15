"""The phase-2 baseline driver must not grow a second copy of the recipe it runs.

`experiments/_llmc.py` owns what scheme each method is fitted under. `stage8_baselines.py`
used to own a second copy of that answer, and the two agreed right up until one of them was
fixed: `_llmc` learned `--symmetric` and `--actorder` so the scheme control could be run,
and stage 8 did not, so phase 2's GPTQ arm could not be re-fitted to match its opponent at
all. These tests fail the diff that reintroduces the split rather than the run that suffers
from it, because the run costs a GPU-hour and reads as a result rather than as a defect.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE8 = REPO_ROOT / "experiments" / "four_point" / "stage8_baselines.py"
LLMC = REPO_ROOT / "experiments" / "_llmc.py"


@pytest.fixture(scope="module")
def llmc() -> Any:
    """Loaded by path, not imported: `experiments/` is not a package."""
    spec = importlib.util.spec_from_file_location("_dq_llmc", LLMC)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dq_llmc"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stage8_tree() -> ast.Module:
    """Parsed rather than imported.

    Importing it would execute `common`, which builds the CaseHOLD task at module scope and
    reaches the network to do it. The question here is about what the file declares, which
    the syntax tree answers exactly and without a download.
    """
    return ast.parse(STAGE8.read_text(encoding="utf-8"))


def test_stage8_declares_no_recipe_builder_of_its_own(stage8_tree: ast.Module) -> None:
    defined = {n.name for n in stage8_tree.body if isinstance(n, ast.FunctionDef)}
    assert not defined & {"build_recipe", "quant_args"}, (
        "stage 8 has grown its own recipe builder again; the copy that is not the one "
        "under test is the one that will be missing the next control's flag"
    )


def test_stage8_takes_the_recipe_builder_from_the_module_that_owns_it(
    stage8_tree: ast.Module,
) -> None:
    imported = {
        alias.name
        for node in stage8_tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "_llmc"
        for alias in node.names
    }
    assert {"METHODS", "build_recipe", "default_symmetric"} <= imported


def test_the_published_default_of_every_method_is_pinned(llmc: Any) -> None:
    """The line stage 8 used to carry, kept as an assertion instead of as a second copy.

    Every phase-2 baseline in the campaign was fitted under these defaults. If
    `default_symmetric` ever moves, those arms stop reproducing and nothing else in the
    repository would say so -- the records carry the resolved value, so old and new arms
    would simply disagree without either being wrong on its face.
    """
    for method in llmc.METHODS:
        assert llmc.default_symmetric(method) == (method != "awq"), method


def test_the_ignore_list_cannot_be_defaulted_by_the_shared_builder(llmc: Any) -> None:
    """`ignore` must stay required and keyword-only.

    A default here would be a model-shaped decision living in the one module that is
    supposed to know nothing about the model. It also has teeth: phase 2 passes
    `["lm_head"]` and phase 4 passes `[]`, and on a model that ties `lm_head` to
    `embed_tokens` the difference is 27% of the weights and moves the measured width of a
    "4-bit" arm to 7.36 bits.
    """
    parameter = inspect.signature(llmc.build_recipe).parameters["ignore"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize("method", ["awq", "rtn"])
def test_activation_ordering_is_refused_where_no_hessian_is_estimated(
    llmc: Any, method: str
) -> None:
    """Refused, not dropped. A flag that is silently ignored is a control a caller
    believes it ran, and the arm it produces is labelled as though it did."""
    with pytest.raises(SystemExit, match="no Hessian"):
        llmc.build_recipe(method, 3, 128, ignore=[], actorder="group")
