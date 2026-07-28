"""The research implementation, kept verbatim as a reference oracle.

Three modules from the paper's supplementary material are vendored here byte for
byte: :mod:`scorer`, :mod:`allocator` and :mod:`quantizer`. They exist for two
reasons and no others.

1. **``--preset paper-3.15`` has to reproduce published numbers.** A preset that
   merely *claims* compatibility is worth nothing; the only way to know the v2
   pipeline still lands on the paper's allocation is to run the original beside it
   and compare. That requires the original, unmodified.
2. **Golden tests need a fixed target.** :mod:`tests.test_legacy_allocator` runs
   this allocator against the shipped stats files and pins the behaviour that
   `docs/legacy-audit.md` documents -- including the defects. If the vendored copy
   were "cleaned up", the audit would stop being checkable.

**Nothing here is on the main path.** ``dynquant/__init__.py`` does not import this
package; it is reached only by an explicit ``from dynquant._legacy import ...``,
which happens in the compat preset and in tests. The code carries the defects
catalogued in the audit -- most consequentially, an allocator whose importance
score has no effect at the paper's headline 3-bit target (audit item 4). Do not
copy patterns from it.

Consistent with that, this package is excluded from ruff, from mypy and from
coverage (see the root ``pyproject.toml``). Those exclusions are not an oversight
to be tidied away later: satisfying a linter means editing the file, and editing
the file destroys the only property that makes it useful.

Why only three modules
----------------------
The rest of the supplement -- ``supervised_finetuning.py``, ``run_quantization.py``,
the inference scripts, the pipeline drivers -- is **not** vendored. Those files
carry a hardcoded credential placeholder and the author's absolute VM paths
(audit §10). Vendoring them would move those literals into an installed package,
which is the one place they must never reach. They are also the parts with no
oracle value: they are drivers, not computations, and every behaviour worth
reproducing lives in the three modules that are here.

The three vendored modules import only ``json``, ``math``, ``dataclasses``,
``typing`` and ``torch``. They read no files, take no paths and hold no
credentials, which is what makes them safe to ship.

Provenance
----------
:mod:`dynquant._legacy._provenance` records the SHA-256 of each vendored file, and
``tests/test_legacy_provenance.py`` checks two things: that the files still match
those hashes, and -- when the research tree is present next to the repo -- that
they still match the originals. The first check runs everywhere and catches a
well-meaning edit. The second runs only on a machine that has the supplement and
catches divergence at the source.

Hashes are taken over LF-normalized bytes. The originals are CRLF; the repository
is LF throughout, and a byte-level comparison that flipped with a checkout setting
would fail for reasons having nothing to do with the code.
"""

from __future__ import annotations

__all__ = ["EXPECTED_SHA256", "VENDORED_MODULES"]

from dynquant._legacy._provenance import EXPECTED_SHA256, VENDORED_MODULES
