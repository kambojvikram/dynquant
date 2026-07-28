"""Where the vendored research modules came from, and what they must still be.

A vendored file drifts silently. Someone runs a formatter across the tree, or
fixes an obvious-looking bug, and the reference oracle quietly stops being the
thing it was copied from -- while the golden tests keep passing, because they now
compare the new code against itself.

The hashes below close that. They are taken over **LF-normalized bytes**: the
originals in the supplement are CRLF, this repository is LF, and a raw byte hash
would flip with a ``core.autocrlf`` setting for reasons having nothing to do with
the code.

This module is a plain ``.py`` rather than a JSON sidecar so that it is packaged
by any build backend without a ``package-data`` rule, and so a typo in a filename
is an import error rather than a silently-skipped check.
"""

from __future__ import annotations

from typing import Final

#: Where in the supplementary material each file was taken from, relative to the
#: research tree root. Recorded because the directory names in the supplement do
#: not survive into the package: ``dynquant_paper`` is not an importable location
#: here, and ``fine-tuning_and_stats_hook`` is not a legal Python identifier at
#: all (audit item 3).
SOURCE_TREE: Final[str] = "dynquant_paper"

#: SHA-256 over LF-normalized file bytes. Regenerate deliberately, never casually:
#: a changed hash here means the oracle changed, which means every golden number
#: derived from it needs re-checking.
EXPECTED_SHA256: Final[dict[str, str]] = {
    "scorer.py": "860a97325a58ce793134de1a0b4907026d98737c0ea5e1534d2e500f7fbfab49",
    "allocator.py": "69963b6bf2fa003d4fb7bcfb46c2e30f2b25ba1de871b19e5312fe42e342e071",
    "quantizer.py": "d42feb6d7e7cfa41a434d0b81579374510cbea16c2a07b54b950a4239c8b5004",
}

#: The vendored module names, in dependency order (there are no dependencies
#: between them -- each imports only the standard library and torch, which is why
#: vendoring three files rather than the whole package is possible at all).
VENDORED_MODULES: Final[tuple[str, ...]] = ("scorer", "allocator", "quantizer")

#: Modules deliberately **not** vendored, and why. Kept as data rather than prose
#: so ``tests/test_legacy_provenance.py`` can assert the directory contains
#: nothing outside the allowed set -- the failure mode being someone adding
#: ``supervised_finetuning.py`` here for convenience and shipping its credential
#: literal to PyPI.
EXCLUDED_FROM_VENDORING: Final[dict[str, str]] = {
    "run_quantization.py": (
        "a driver, not a computation; also writes the capitalized filename that no "
        "reader can open (audit item 2)"
    ),
    "unified_tracker.py": (
        "superseded wholesale by dynquant.signals; its per-step device sync and "
        "LoRA-shape defects (audit items 7, 8) have no oracle value"
    ),
    "supervised_finetuning.py": "carries a hardcoded Hugging Face token literal (audit item 10)",
    "inference_3bits.py": "carries absolute paths from the author's VM (audit item 10)",
    "inference_4bits.py": "loads a model at import time (audit item 10)",
}
