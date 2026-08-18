#!/usr/bin/env python3
"""Refuse a shared object that needs a newer libstdc++ than the wheel promises.

This exists because 0.5.0 shipped a wheel that installed, imported, and did not
work. ``dynquant-kernels`` 0.5.0 was built by GCC 14, which emits
``__cxa_call_terminate`` on exception-cleanup paths. That symbol was added to
libstdc++ in GCC 14 (CXXABI_1.3.15). Ubuntu 22.04 -- and every image older than
GCC 14 -- provides CXXABI_1.3.13, so ``import dynquant_kernels`` raised
``undefined symbol: __cxa_call_terminate``, ``is_available()`` returned False,
and DynQuant silently fell back to the torch path. Measured cost of that silence
on an L40: 32 decode tokens unfinished after 32 minutes at 0% GPU utilisation,
against 20.97 tok/s with the kernels loaded. Nothing raised. Nothing warned.

**Why the existing tooling did not catch it.** ``auditwheel`` and the
``manylinux_2_34`` tag are computed from the *versioned* symbol requirements in
the ELF -- the ``.gnu.version_r`` table -- and that table topped out at
GLIBCXX_3.4.21 / CXXABI_1.3.9, comfortably inside the promise. It topped out
there because ``__cxa_call_terminate`` is an **unversioned** undefined symbol.
The one symbol that broke the wheel is the one class of symbol the platform tag
cannot see. So a version-table check is not enough, and this script does not
write one: it resolves every C++ runtime symbol by *name* against a baseline
libstdc++ and fails on anything that baseline does not define.

Two independent mechanisms, because either alone has a blind spot:

1. **Resolve against a baseline libstdc++.** Correct and general, but only as old
   as whatever library the build host happens to have -- on a GCC 14 host the
   baseline *does* define the offending symbol and the check passes wrongly.
2. **An explicit deny-list.** Narrow, but it holds no matter what the build host
   has installed, which is exactly the case mechanism 1 cannot cover.

Usage::

    python check_cxx_runtime_abi.py path/to/_C.so [--baseline libstdc++.so.6]

Set ``DYNQUANT_ABI_BASELINE_LIBSTDCXX`` to point at the oldest libstdc++ the
release intends to support. With neither flag nor variable the script falls back
to the build host's own, which still runs mechanism 2.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

#: Symbols that exist only in a libstdc++ newer than the oldest we support, keyed
#: by the toolchain that introduced them. Checked by name regardless of what the
#: build host provides -- this is the half of the check that a modern build image
#: cannot defeat.
INTRODUCED_AFTER_BASELINE: dict[str, str] = {
    "__cxa_call_terminate": "GCC 14 (CXXABI_1.3.15)",
}

#: The C++ runtime surface. Everything else undefined in the module -- ATen,
#: libtorch, libc, libcudart -- is resolved at load time from libraries that are
#: guaranteed present next to it, and is not this script's business.
CXX_RUNTIME_PREFIXES = ("__cxa_", "__cxx_", "__gxx_", "_ZSt", "_ZNSt", "_ZNKSt", "_ZN9__gnu_cxx", "_ZTVSt", "_ZTISt")


def _tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"{name} is required for the ABI check but is not on PATH")
    return found


def undefined_symbols(so: Path) -> list[str]:
    """Undefined dynamic symbols, with any ``@VERSION`` suffix stripped.

    ``nm -D --undefined-only`` prints one symbol per line prefixed by ``U``. The
    version suffix is dropped because the whole point here is to compare by name:
    the symbol that caused this script to exist carries no version at all.
    """
    out = subprocess.run(
        [_tool("nm"), "-D", "--undefined-only", str(so)],
        check=True, capture_output=True, text=True,
    ).stdout
    syms = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[-2] in {"U", "w", "W"}:
            syms.append(parts[-1].split("@", 1)[0])
        elif len(parts) == 2 and parts[0] in {"U", "w", "W"}:
            syms.append(parts[1].split("@", 1)[0])
    return sorted(set(syms))


def defined_symbols(lib: Path) -> set[str]:
    out = subprocess.run(
        [_tool("nm"), "-D", "--defined-only", str(lib)],
        check=True, capture_output=True, text=True,
    ).stdout
    return {line.split()[-1].split("@", 1)[0] for line in out.splitlines() if len(line.split()) >= 2}


#: The libraries the loader will actually resolve these symbols from. libstdc++ is
#: the one under suspicion, but it is not the only provider of a `__cxa_*` name:
#: `__cxa_atexit` and `__cxa_finalize` are glibc's, and `__cxa_call_terminate` --
#: the symbol this whole check exists for -- is libstdc++'s. Checking against
#: libstdc++ alone reports the first two as missing on every correct build, which
#: is how the first run of this script failed a wheel that was fine.
BASELINE_LIBRARIES = ("libstdc++.so.6", "libc.so.6", "libgcc_s.so.1")

#: Where to look when the compiler will not say. `-print-file-name` answers for
#: libraries the compiler links itself and echoes the bare name for the rest.
_FALLBACK_LIBDIRS = (
    "/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu", "/lib64", "/usr/lib64", "/usr/lib",
)


def _locate(soname: str) -> Path | None:
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")
    if cxx:
        got = subprocess.run(
            [cxx, f"-print-file-name={soname}"], capture_output=True, text=True
        ).stdout.strip()
        if got and got != soname and Path(got).exists():
            return Path(got).resolve()
    for d in _FALLBACK_LIBDIRS:
        cand = Path(d) / soname
        if cand.exists():
            return cand.resolve()
    return None


def find_baseline(explicit: str | None) -> list[Path]:
    """The baseline runtime: an explicit libstdc++ if given, plus libc and libgcc.

    Only libstdc++ is overridable, because it is the only one whose *age* is the
    question. A release build should point this at the oldest libstdc++ it intends
    to support; without that it falls back to the build host's, which still leaves
    the deny-list doing real work.
    """
    libs: list[Path] = []
    override = explicit or os.environ.get("DYNQUANT_ABI_BASELINE_LIBSTDCXX")
    if override:
        p = Path(override)
        if not p.is_file():
            raise SystemExit(f"baseline libstdc++ not found: {p}")
        libs.append(p)
    for soname in BASELINE_LIBRARIES:
        if override and soname.startswith("libstdc++"):
            continue
        found = _locate(soname)
        if found is not None:
            libs.append(found)
    return libs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shared_object", type=Path)
    ap.add_argument("--baseline", help="oldest libstdc++.so.6 this release supports")
    args = ap.parse_args()

    so = args.shared_object
    if not so.is_file():
        raise SystemExit(f"not a file: {so}")

    undef = undefined_symbols(so)
    failures: list[str] = []

    # Mechanism 2 first: it needs nothing but the symbol list, so it reports even
    # when no baseline library could be located.
    for sym in undef:
        if sym in INTRODUCED_AFTER_BASELINE:
            failures.append(f"  {sym}  -- introduced in {INTRODUCED_AFTER_BASELINE[sym]}")

    baseline = find_baseline(args.baseline)
    if not baseline:
        print("ABI check: no baseline libraries located; running the deny-list only.", file=sys.stderr)
    else:
        have: set[str] = set()
        for lib in baseline:
            have |= defined_symbols(lib)
        names = ", ".join(lib.name for lib in baseline)
        for sym in undef:
            if sym in INTRODUCED_AFTER_BASELINE:
                continue  # already reported, and with a better explanation
            if sym.startswith(CXX_RUNTIME_PREFIXES) and sym not in have:
                failures.append(f"  {sym}  -- not defined in any of {names}")

    if failures:
        print(
            f"\nABI CHECK FAILED for {so.name}\n\n"
            "It needs C++ runtime symbols that the oldest supported libstdc++ does not\n"
            "define. This wheel would install, import, and then fail to load at runtime --\n"
            "silently, because the loader falls back to the torch backend. Build with an\n"
            "older GCC (13 or earlier) or against an older libstdc++.\n\n"
            + "\n".join(failures)
            + "\n",
            file=sys.stderr,
        )
        return 1

    where = ", ".join(str(lib) for lib in baseline) if baseline else "deny-list only"
    print(f"ABI check passed for {so.name} ({len(undef)} undefined symbols, baseline: {where})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
