"""The lint that holds the ABI number's three declarations together.

``KERNEL_ABI_VERSION`` is necessarily stated three times -- in ``dynquant._version``
(Python, core wheel), in ``dynquant_kernels.ABI_VERSION`` (Python, kernels wheel,
which must not import core), and as a ``#define`` in ``csrc/include/dynquant/abi.h``
(C++, baked into the shared object). They cannot be collapsed into one: the C++
side needs a compile-time constant, and the kernels wheel needs to be diagnosable
without core installed.

So the number is duplicated and this file is what makes that safe. Everything here
reads *source text* rather than importing the compiled extension, so the lint runs
on a CPU-only machine with no kernels wheel built -- which is where it needs to run
if it is to catch a mismatch before a wheel is published.

An ABI mismatch is the one failure mode that produces plausible wrong numbers
instead of an exception: a kernel compiled against an older packed layout still
returns a tensor of the right shape, dtype and device, full of garbage. Nothing
downstream can detect that, which is why it is caught here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from dynquant._version import KERNEL_ABI_VERSION, MIN_KERNEL_ABI_VERSION
from dynquant.constants import (
    BIT_OPTIONS,
    GEMV_MAX_ROWS,
    GROUP_SIZE_ALIGNMENT,
    VALUES_PER_WORD,
    WORDS_PER_BLOCK,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNELS_ROOT = REPO_ROOT / "packages" / "dynquant-kernels"
ABI_HEADER = KERNELS_ROOT / "csrc" / "include" / "dynquant" / "abi.h"
KERNELS_SHELL = KERNELS_ROOT / "src" / "dynquant_kernels" / "__init__.py"
BINDINGS = KERNELS_ROOT / "csrc" / "bindings.cpp"
CMAKELISTS = KERNELS_ROOT / "CMakeLists.txt"


def _header() -> str:
    if not ABI_HEADER.is_file():
        pytest.skip(f"{ABI_HEADER} not present (installed-package test run)")
    return ABI_HEADER.read_text(encoding="utf-8")


def _define(name: str) -> str:
    """The replacement text of an object-like ``#define``, as a raw string."""
    match = re.search(rf"^#define\s+{re.escape(name)}\s+(.+?)\s*$", _header(), re.MULTILINE)
    assert match is not None, f"{name} is not #defined in {ABI_HEADER.name}"
    return match.group(1)


# --------------------------------------------------------------------------
# The number itself
# --------------------------------------------------------------------------


def test_header_and_core_agree_on_the_abi_version():
    assert int(_define("DYNQUANT_ABI_VERSION")) == KERNEL_ABI_VERSION, (
        "csrc/include/dynquant/abi.h and dynquant._version disagree on the ABI "
        "version. Whichever was bumped, bump the other: a wheel built from this "
        "tree would load and then return wrong numbers."
    )


def test_kernels_shell_and_core_agree_on_the_abi_version():
    """Read the shell's source, not the imported module.

    Importing ``dynquant_kernels`` would skip on every machine without the wheel
    built -- i.e. exactly the CPU CI runner where this lint is cheap enough to run
    on every commit.
    """
    if not KERNELS_SHELL.is_file():
        pytest.skip(f"{KERNELS_SHELL} not present")
    source = KERNELS_SHELL.read_text(encoding="utf-8")
    match = re.search(r"^ABI_VERSION\s*:\s*int\s*=\s*(\d+)", source, re.MULTILINE)
    assert match is not None, "dynquant_kernels.__init__ no longer declares ABI_VERSION: int = N"
    assert int(match.group(1)) == KERNEL_ABI_VERSION


def test_minimum_abi_is_not_newer_than_current():
    """``MIN <= CURRENT``, or every kernels wheel including the matching one is
    rejected by :func:`dynquant.runtime.backend._probe_cuda`'s range check."""
    assert MIN_KERNEL_ABI_VERSION <= KERNEL_ABI_VERSION


def test_cmake_can_still_parse_the_header():
    """The build stamps the ABI into ``_build_info.py`` by regex-matching the
    header. A reformat that broke the match would be a FATAL_ERROR at build time on
    a GPU runner, minutes into a CUDA compile -- catch it here in milliseconds."""
    if not CMAKELISTS.is_file():
        pytest.skip("CMakeLists.txt not present")
    cmake = CMAKELISTS.read_text(encoding="utf-8")
    stated = re.search(r'string\(REGEX MATCH "([^"]+)"\s*\n?\s*_dq_abi_match', cmake)
    assert stated is not None, "the ABI regex in CMakeLists.txt has moved; update this test"
    # CMake regex syntax is a subset of Python's for this pattern; \t is written as
    # a literal backslash-t in the CMake string, which Python reads the same way.
    pattern = stated.group(1)
    match = re.search(pattern, _header())
    assert match is not None, (
        f"CMake's pattern {pattern!r} no longer matches abi.h; the build would fail "
        f"with 'Could not parse DYNQUANT_ABI_VERSION'"
    )
    assert int(match.group(1)) == KERNEL_ABI_VERSION


# --------------------------------------------------------------------------
# The layout constants that travel with it
# --------------------------------------------------------------------------


def test_bit_widths_agree():
    """``DYNQUANT_BITS_LIST`` is what the kernels are templated over. If Python
    offers a width the templates were not instantiated for, the dispatch falls off
    the end of a switch at runtime."""
    header_bits = tuple(int(piece) for piece in _define("DYNQUANT_BITS_LIST").split(","))
    assert header_bits == tuple(BIT_OPTIONS)


def test_group_size_alignment_agrees():
    assert int(_define("DYNQUANT_GROUP_SIZE_ALIGNMENT")) == GROUP_SIZE_ALIGNMENT


def test_gemv_max_rows_agrees():
    """The runtime's dispatch threshold. If Python believed a larger bound than the
    kernel was built with, every batch between the two would raise from inside a
    forward pass instead of taking the dequant + GEMM path."""
    assert int(_define("DYNQUANT_GEMV_MAX_ROWS")) == GEMV_MAX_ROWS


def test_group_size_alignment_is_actually_sufficient():
    """The alignment is not a free parameter -- it is the LCM of the per-width
    block sizes, and the whole word-aligned layout rests on it. Deriving it here
    means a new bit-width cannot be added without either satisfying the invariant
    or failing this test."""
    for bits in BIT_OPTIONS:
        values = VALUES_PER_WORD[bits]
        assert GROUP_SIZE_ALIGNMENT % values == 0, (
            f"{bits}-bit packs {values} values per block, which does not divide the "
            f"{GROUP_SIZE_ALIGNMENT}-value alignment: groups would start mid-word"
        )
        # A block must be a whole number of words, or the "constant shift" property
        # the kernels rely on does not hold.
        assert values * bits == WORDS_PER_BLOCK[bits] * 32


# --------------------------------------------------------------------------
# Op registration
# --------------------------------------------------------------------------


def test_every_tensor_returning_op_has_a_meta_kernel():
    """A new op registered in C++ but not in ``ops.py`` costs a graph break per
    call site under ``torch.compile``, and nothing fails -- it just gets slower,
    which is the kind of regression that survives for releases."""
    for path in (BINDINGS, KERNELS_ROOT / "src" / "dynquant_kernels" / "ops.py"):
        if not path.is_file():
            pytest.skip(f"{path} not present")

    schemas = re.findall(
        r'm\.def\(\s*"(\w+)\(([^)]*)\)\s*->\s*([^"]+)"', BINDINGS.read_text(encoding="utf-8")
    )
    assert schemas, "no TORCH_LIBRARY schemas found; the parse in this test has rotted"

    fakes = set(
        re.findall(
            r'register_fake\(\s*"dynquant::(\w+)"',
            (KERNELS_ROOT / "src" / "dynquant_kernels" / "ops.py").read_text(encoding="utf-8"),
        )
    )

    missing = [
        name
        for name, _args, returns in schemas
        # Only tensor-returning ops need a shape rule. `-> int`/`-> bool`/`-> str`
        # ops are metadata queries; they are never in a traced graph.
        if "Tensor" in returns and name not in fakes
    ]
    assert not missing, (
        f"ops with no meta kernel: {missing}. Add "
        f"@torch.library.register_fake('dynquant::<name>') in dynquant_kernels/ops.py."
    )


# --------------------------------------------------------------------------
# The installed artifact, when there is one
# --------------------------------------------------------------------------


def test_installed_kernels_wheel_matches_core():
    """Same assertion as above, but against what is actually importable. Skips
    everywhere the wheel is not installed, which is why the source-text tests exist
    rather than only this one."""
    kernels = pytest.importorskip("dynquant_kernels")
    assert kernels.ABI_VERSION == KERNEL_ABI_VERSION


def test_compiled_binary_matches_its_own_shell():
    """The last link: the number the *binary* was compiled with. Everything above
    compares text in a checkout; this compares text to machine code, and it is the
    only check that can catch a stale ``build/`` directory being packaged."""
    kernels = pytest.importorskip("dynquant_kernels")
    if not kernels.is_available():
        pytest.skip(f"extension unavailable: {kernels.unavailable_reason()}")

    import torch

    assert int(torch.ops.dynquant.abi_version()) == kernels.ABI_VERSION
    stamped = kernels.build_info().get("ABI_VERSION")
    if stamped is not None:
        assert int(stamped) == kernels.ABI_VERSION
