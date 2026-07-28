"""Path setup and shared fixtures.

Two sys.path insertions, both deliberate:

* the core package's ``src/``, so the suite runs against a source checkout with no
  install step (harmless once installed -- an installed ``dynquant`` wins);
* this directory, because ``--import-mode=importlib`` does not put test modules on
  the path, and the error oracle in :mod:`_oracle` needs to be importable by name.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent

for extra in (REPO_ROOT / "packages" / "dynquant-core" / "src", TESTS_DIR):
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


@pytest.fixture(scope="session")
def torch_seeded():
    """torch with a fixed seed, or a skip if torch is absent."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    return torch


@pytest.fixture
def weight(torch_seeded):
    """A weight shaped and scaled like a real transformer projection."""
    return torch_seeded.randn(256, 1024, dtype=torch_seeded.float16) * 0.02


@pytest.fixture
def v1_stats_raw():
    """Factory: read a v1 stats file as raw JSON."""

    def _read(path: Path) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    return _read
