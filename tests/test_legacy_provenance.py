"""The vendored research code must stay the research code.

``dynquant/_legacy/`` is a reference oracle: ``--preset paper-3.15`` and
``tests/test_legacy_allocator.py`` are only meaningful if what they run is what
the paper ran. A vendored file drifts silently -- a formatter sweep, an
obvious-looking fix -- and the golden tests keep passing, because they now compare
the new code against itself.

Two checks, at different distances from the source:

* :func:`test_the_vendored_files_match_their_recorded_hashes` runs everywhere and
  catches an edit made inside this repository.
* :func:`test_the_vendored_files_still_match_the_supplement` runs only where the
  research tree exists and catches divergence at the source.

Plus :func:`test_nothing_else_gets_vendored`, which is the security half: the
directory is one convenient ``cp`` away from containing ``supervised_finetuning.py``
and its ``token="hf_..."`` literal, at which point a credential shape ships to
PyPI inside an installed package.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from dynquant._legacy import _provenance

REPO_ROOT = Path(__file__).resolve().parents[1]
LEGACY_DIR = Path(_provenance.__file__).parent
RESEARCH_TREE = REPO_ROOT / _provenance.SOURCE_TREE


def _normalized_sha256(path: Path) -> str:
    """Hash LF-normalized bytes.

    The supplement's files are CRLF and this repository is LF. A raw byte hash
    would flip with a ``core.autocrlf`` setting, failing for a reason that has
    nothing to do with the code -- and a test that fails for the wrong reason gets
    marked xfail, which is how the real check gets lost.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@pytest.mark.parametrize("filename", sorted(_provenance.EXPECTED_SHA256))
def test_the_vendored_files_match_their_recorded_hashes(filename):
    path = LEGACY_DIR / filename
    assert path.is_file(), f"{filename} is recorded in _provenance but missing from _legacy/"
    assert _normalized_sha256(path) == _provenance.EXPECTED_SHA256[filename], (
        f"{filename} has been modified. It is a verbatim copy kept as an oracle; "
        f"if the change is genuinely intended, update EXPECTED_SHA256 deliberately "
        f"and re-check every golden number derived from it."
    )


@pytest.mark.parametrize("module", _provenance.VENDORED_MODULES)
def test_the_vendored_files_still_match_the_supplement(module):
    original = RESEARCH_TREE / f"{module}.py"
    if not original.is_file():
        pytest.skip("research tree not present (it is not committed; see docs/legacy-audit.md)")
    assert _normalized_sha256(original) == _normalized_sha256(LEGACY_DIR / f"{module}.py")


def test_nothing_else_gets_vendored():
    """The directory holds the three oracles and its own two support modules.

    Anything else is either a driver with no oracle value or -- the case this
    guards -- a file carrying a credential literal or an absolute path from the
    author's VM. ``scripts/check_no_confidential.py`` would catch those at commit
    time, but only for a file that is *staged*; this catches the copy itself.
    """
    allowed = {f"{m}.py" for m in _provenance.VENDORED_MODULES} | {
        "__init__.py",
        "_provenance.py",
    }
    present = {p.name for p in LEGACY_DIR.iterdir() if p.is_file() and p.suffix == ".py"}
    assert present == allowed, f"unexpected in _legacy/: {sorted(present - allowed)}"

    for excluded in _provenance.EXCLUDED_FROM_VENDORING:
        assert not (LEGACY_DIR / excluded).exists(), (
            f"{excluded} must not be vendored: {_provenance.EXCLUDED_FROM_VENDORING[excluded]}"
        )


def test_the_oracles_are_self_contained():
    """They import only the standard library and torch.

    This is what makes vendoring three files possible instead of the whole
    package, and it is worth pinning: an added ``from .run_quantization import ...``
    would drag in a module that writes a filename no reader can open (audit item 2)
    and would make ``dynquant._legacy`` unimportable in the wheel.
    """
    permitted = {"__future__", "json", "math", "dataclasses", "typing", "torch"}
    for module in _provenance.VENDORED_MODULES:
        text = (LEGACY_DIR / f"{module}.py").read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not (stripped.startswith("import ") or stripped.startswith("from ")):
                continue
            root = stripped.split()[1].split(".")[0]
            assert root in permitted, f"{module}.py:{line_number}: unexpected import {root!r}"


def test_the_legacy_package_is_not_imported_on_the_main_path():
    """Importing ``dynquant`` must not drag in the research code.

    It carries the defects the audit catalogues; nothing on the supported path may
    depend on it, and the cheapest way for that to stop being true is an
    absent-minded re-export in ``dynquant/__init__.py``.
    """
    import os
    import subprocess
    import sys

    import dynquant

    # A subprocess rather than checking ``sys.modules`` here: by the time this test
    # runs, the imports at the top of this file have already loaded ``_legacy``, so
    # an in-process check would pass unconditionally. The environment has to carry
    # the source path, because the suite runs against a checkout with no install
    # step -- ``conftest.py`` puts ``src/`` on ``sys.path``, and a bare subprocess
    # inherits none of that.
    src_root = str(Path(dynquant.__file__).resolve().parents[1])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src_root, env.get("PYTHONPATH", "")]))

    result = subprocess.run(
        [sys.executable, "-c", "import sys, dynquant; print('dynquant._legacy' in sys.modules)"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert result.stdout.strip() == "False", "dynquant/__init__.py imports _legacy"
