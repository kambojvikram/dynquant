#!/usr/bin/env python3
"""Stamp the binary variant into ``dynquant-kernels``' version, PEP 440 style.

One kernels wheel is not enough: a compiled extension is valid only next to the
torch minor and CUDA major it was linked against, so a release is a *matrix* of
wheels for the same source. PEP 440 has exactly one place to record that -- the
local version segment -- and torch itself uses it (``2.7.1+cu126``). So::

    0.1.0.dev0  ->  0.1.0.dev0+cu126torch27

pip treats the local segment as a build variant of the same release, which is what
makes ``dynquant-kernels>=0.1.0.dev0,<0.2`` in the meta package resolve to
whichever variant an index offers, while a ``==`` pin on the full string selects
one exactly.

Why a script instead of a build option: scikit-build-core reads the version from
the source file by regex (see ``packages/dynquant-kernels/pyproject.toml``), and
there is no build-time hook to append to it -- the string in the file *is* the
version. So CI rewrites the file per matrix cell, before the build.

The default combination for PyPI is built with ``--plain``, which stamps nothing:
PyPI rejects local versions outright, so exactly one cell of the matrix can be
published there and the rest are served from a ``--find-links`` index.

Usage::

    python scripts/stamp_kernel_version.py --cuda 12.6 --torch 2.7.1
    python scripts/stamp_kernel_version.py --plain          # PyPI cell
    python scripts/stamp_kernel_version.py --cuda 12.6 --torch 2.7.1 --print-only
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

VERSION_FILE = (
    Path(__file__).resolve().parents[1]
    / "packages"
    / "dynquant-kernels"
    / "src"
    / "dynquant_kernels"
    / "_version.py"
)

#: Must stay in step with scikit-build-core's own regex provider, which looks for
#: this exact assignment. A change here that the build cannot parse fails at build
#: time rather than producing a mis-versioned wheel, which is the safe direction.
_ASSIGNMENT = re.compile(r'^(__version__\s*=\s*)"([^"]+)"', re.MULTILINE)


def local_label(cuda: str, torch: str) -> str:
    """``12.6``, ``2.7.1+cu126`` -> ``cu126torch27``.

    Only major.minor of each is kept. A cu12.6 build works against any 12.x torch
    (the CUDA minor compatibility guarantee), and a torch *patch* release does not
    change the C++ ABI -- so encoding either would multiply the matrix without
    describing a real incompatibility.
    """
    cuda_parts = re.findall(r"\d+", cuda)
    torch_parts = re.findall(r"\d+", torch)
    if len(cuda_parts) < 2 or len(torch_parts) < 2:
        raise SystemExit(
            f"could not read major.minor from --cuda {cuda!r} / --torch {torch!r}; "
            f"expected forms like '12.6' and '2.7.1'"
        )
    return f"cu{cuda_parts[0]}{cuda_parts[1]}torch{torch_parts[0]}{torch_parts[1]}"


def stamped_version(base: str, label: str | None) -> str:
    # Strip any existing local segment first: CI reuses a checkout across matrix
    # cells, and appending twice would produce `...+cu126torch27+cu121torch24`,
    # which is not a valid PEP 440 version at all.
    base = base.split("+", 1)[0]
    return base if label is None else f"{base}+{label}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cuda", help="CUDA toolkit version, e.g. 12.6")
    parser.add_argument("--torch", help="torch version, e.g. 2.7.1 or 2.7.1+cu126")
    parser.add_argument(
        "--plain",
        action="store_true",
        help="strip any local segment (the one cell that may be uploaded to PyPI)",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="print the version that would be written and change nothing",
    )
    args = parser.parse_args()

    if args.plain == bool(args.cuda or args.torch):
        raise SystemExit("pass either --plain or both --cuda and --torch")
    if not args.plain and not (args.cuda and args.torch):
        raise SystemExit("--cuda and --torch must be given together")

    text = VERSION_FILE.read_text(encoding="utf-8")
    match = _ASSIGNMENT.search(text)
    if match is None:
        raise SystemExit(f'no `__version__ = "..."` assignment found in {VERSION_FILE}')

    label = None if args.plain else local_label(args.cuda, args.torch)
    new_version = stamped_version(match.group(2), label)

    if args.print_only:
        print(new_version)
        return 0

    VERSION_FILE.write_text(
        _ASSIGNMENT.sub(rf'\g<1>"{new_version}"', text, count=1), encoding="utf-8"
    )
    print(f"{VERSION_FILE.name}: __version__ = {new_version!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
