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
import re
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


def test_a_symmetric_grid_is_not_charged_for_a_zero_point_it_never_stores(llmc: Any) -> None:
    """The defect this function was extracted to fix.

    `compressed-tensors` adds `weight_zero_point` to the state dict under
    `if not weights.symmetric` and `_remove_symmetric_zp` strips it otherwise, so a
    symmetric arm stores one fp16 scale per group and nothing else. Both baseline stages
    used to charge `16 + bits` to every arm, which over-charged every GPTQ and RTN arm this
    project published -- in the direction that makes the baseline look more expensive than
    it is, which is the direction that flatters DynQuant.
    """
    numel, in_features, group_size = 4096 * 2048, 2048, 128
    groups = numel // group_size

    symmetric = llmc.stored_meta_bits(
        numel, in_features, bits=3, group_size=group_size, symmetric=True
    )
    asymmetric = llmc.stored_meta_bits(
        numel, in_features, bits=3, group_size=group_size, symmetric=False
    )

    assert symmetric == groups * 16
    assert asymmetric == groups * (16 + 3)
    assert asymmetric > symmetric


def test_the_two_schemes_cannot_account_to_the_same_width(llmc: Any) -> None:
    """The property the phase-2 control depends on, asserted as a property.

    That control varies one flag and reads the accuracy against the size. If the size
    column cannot see the flag, the control measures nothing and still prints a table --
    which is what happened: two arms whose accuracies differed by 22 points both accounted
    to 3.1522 bits. A parametrised equality would have passed on the broken version too,
    because it was equal on purpose; the claim worth pinning is that they *differ*.
    """
    for bits in (2, 3, 4, 8):
        widths = {
            symmetric: bits
            + llmc.stored_meta_bits(
                2048 * 2048, 2048, bits=bits, group_size=128, symmetric=symmetric
            )
            / (2048 * 2048)
            for symmetric in (True, False)
        }
        assert widths[True] < widths[False], bits
        assert widths[False] - widths[True] == pytest.approx(bits / 128)


def test_activation_ordering_is_charged_for_its_index(llmc: Any) -> None:
    """`weight_g_idx` is an `int32` per input column, and nothing was counting it.

    Per-tensor rather than per-group, which is why this returns a total instead of a rate.
    No arm has shipped under `actorder=group` yet, so this pins a cost before an arm is
    priced by it rather than correcting one afterwards.
    """
    numel, in_features = 4096 * 2048, 2048
    plain = llmc.stored_meta_bits(numel, in_features, bits=4, group_size=128, symmetric=True)
    ordered = llmc.stored_meta_bits(
        numel, in_features, bits=4, group_size=128, symmetric=True, actorder="group"
    )
    assert ordered - plain == in_features * 32


def test_a_group_size_that_does_not_divide_is_refused(llmc: Any) -> None:
    """Refused rather than floored.

    `compressed_tensors` ceil-divides (`strategy_cdiv`) and pads the final group, so
    `numel // group_size` undercounts a checkpoint that is genuinely larger. A number that
    is quietly low is worse here than no number, because the whole point of this function
    is to be the denominator an accuracy is read against.
    """
    with pytest.raises(ValueError, match="not a multiple of the group size"):
        llmc.stored_meta_bits(100 * 96, 96, bits=4, group_size=128, symmetric=True)


def test_neither_baseline_stage_keeps_a_local_copy_of_the_group_metadata_rate(
    stage8_tree: ast.Module,
) -> None:
    """The copy is the failure mode, not the arithmetic.

    Both stages independently wrote `meta_bits = 16 + bits`, and the phase-4 copy carried a
    comment defending it. Fixing one and leaving the other is how the campaign ends up with
    two byte columns that disagree about the same checkpoint, so the assertion is that the
    literal is gone from the source rather than that the value is right.
    """
    sources = [ast.unparse(stage8_tree)]
    sources.append((REPO_ROOT / "experiments" / "phase4" / "baselines_lfm2.py").read_text("utf-8"))
    for source in sources:
        # Anchored so `stored_meta_bits` does not satisfy its own prohibition -- the first
        # version of this test passed the phase-4 file for exactly that reason.
        assert re.search(r"(?<!stored_)meta_bits", source) is None
        assert "stored_meta_bits" in source


def test_the_scheme_is_resolved_once_and_priced_by_the_same_answer() -> None:
    """The record and the size column must not resolve `--symmetric auto` separately.

    Phase 2's control is a pair of arms that differ in this flag and nothing else, so the
    flag has to reach both the thing that says what ran and the thing that says what it
    cost. It reached only the first: the arms recorded `symmetric: true` and
    `symmetric: false` correctly and both accounted to 3.1522 bits, and the table would
    have printed a 22-point accuracy gap at an identical width as though the scheme were
    free.
    """
    source = STAGE8.read_text("utf-8")
    assert source.count("resolved_symmetric = ") == 1
    assert '"symmetric": resolved_symmetric,' in source
    assert "symmetric=resolved_symmetric," in source
