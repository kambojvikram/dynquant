#!/usr/bin/env python3
"""Assert that named ``dynquant doctor`` checks reached a required status.

``dynquant doctor`` exits 0 unless a check *fails*, which is the right default for
a user: a missing kernels wheel on a CPU box is not an error, and quantization
still works. CI needs the opposite question answered -- *did the thing this job
exists to build actually get exercised?*

Without that, the CPU-only kernels job would report::

    [SKIP] abi   compiled kernels not installed

and pass, having proved nothing at all. That is the precise failure mode this
script exists to prevent: a green build whose green means "the check did not run".

Usage::

    python scripts/assert_doctor.py abi=ok
    python scripts/assert_doctor.py abi=ok kernels=ok cubin-coverage=ok

Runs the diagnosis in-process against whatever ``dynquant`` is importable, so it
tests the *installed* package when invoked from a directory that is not a source
checkout. Prints the full report on failure, because the interesting information
in CI is always the checks you did not name.
"""

from __future__ import annotations

import sys
from pathlib import Path

_VALID = ("ok", "warn", "fail", "skip")


def _parse(args: list[str]) -> dict[str, str]:
    wanted: dict[str, str] = {}
    for arg in args:
        name, sep, status = arg.partition("=")
        if not sep or not name:
            raise SystemExit(f"expected NAME=STATUS, got {arg!r}")
        if status not in _VALID:
            raise SystemExit(f"unknown status {status!r} in {arg!r}; expected one of {_VALID}")
        wanted[name] = status
    if not wanted:
        raise SystemExit(__doc__)
    return wanted


def main(argv: list[str]) -> int:
    wanted = _parse(argv[1:])

    from dynquant.doctor import diagnose, render

    report = diagnose()
    actual = {check.name: check.status for check in report.checks}

    problems: list[str] = []
    for name, status in wanted.items():
        if name not in actual:
            problems.append(
                f"{name}: no such check (this installation runs {sorted(actual)}). "
                f"A renamed check must be renamed here too, or the assertion "
                f"silently stops asserting."
            )
        elif actual[name] != status:
            problems.append(f"{name}: required {status!r}, got {actual[name]!r}")

    if not problems:
        print(f"doctor: {', '.join(f'{k}={wanted[k]}' for k in wanted)} as required")
        return 0

    print(render(report))
    print()
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    # Make a source checkout work without an install, matching tests/conftest.py, so
    # this script is runnable locally before it is ever wired into a workflow.
    _src = Path(__file__).resolve().parents[1] / "packages" / "dynquant-core" / "src"
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.append(str(_src))  # append, not insert: an installed dynquant wins
    raise SystemExit(main(sys.argv))
