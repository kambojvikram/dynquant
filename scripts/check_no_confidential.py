#!/usr/bin/env python3
"""Refuse to let confidential material or credentials enter git history.

Run as a pre-commit hook on staged files. Two things it blocks, both learned from
this repository's own starting state:

1. **The reviewer PDF.** The supplementary material is a confidential NeurIPS
   reviewer copy, stamped on every page with "Unauthorized sharing,
   redistribution, or disclosure is strictly prohibited". ``.gitignore`` covers it,
   but ``git add -f`` bypasses ``.gitignore``, and once a blob is committed it stays
   reachable in history even after the file is deleted. CI cannot help: it runs
   after the commit exists.

2. **Credentials.** The research code carried ``token="hf_token_here"`` inline in
   three places. That exact string is a placeholder, but the shape of it -- a
   literal secret assigned in source -- is what has to stay out, because the
   version someone fills in is the version that leaks.

Exit code 1 with the offending paths named. Deliberately not overridable by a
flag: the escape hatch is ``git commit --no-verify``, which is at least a
deliberate act that shows up in a review.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

#: Paths that must never be committed, matched against the POSIX-style repo path.
FORBIDDEN_PATHS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\.pdf$", re.IGNORECASE),
    re.compile(r"\.(safetensors|gguf|ckpt|pt|bin)$", re.IGNORECASE),
    re.compile(r"(^|/)\.env(\.|$)"),
    re.compile(r"\.(pem|key|p12|pfx)$", re.IGNORECASE),
)

#: Files that legitimately contain the *word* token, key, secret and so on: this
#: script, the ignore rules that name them, and the docs explaining the policy.
#: ``docs/legacy-audit.md`` quotes the offending lines verbatim as evidence, which
#: is the point of an audit -- it has to be able to show what it found.
EXEMPT_PATHS: frozenset[str] = frozenset(
    {
        "scripts/check_no_confidential.py",
        "tests/test_confidential_guard.py",
        "docs/legacy-audit.md",
        ".gitignore",
        ".pre-commit-config.yaml",
    }
)

#: Secret-shaped assignments. Narrow on purpose -- a broad "looks like base64"
#: pattern fires on legitimate test vectors and gets disabled within a week.
SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        # The value pattern is deliberately *not* ``hf_[A-Za-z0-9]{8,}``. That was
        # the first version, and it did not match the literal this script exists to
        # stop: in ``hf_token_here`` the underscore ends the character class after
        # only five characters, so the guard passed the exact file it was written
        # for. A real token is `hf_` + 34 alphanumerics, but matching only the real
        # shape means the placeholder ships and whoever fills it in ships a live
        # credential -- the placeholder is the thing to catch.
        re.compile(
            r"""\b(hf_token|hf_hub_token|use_auth_token|token)\s*=\s*["']hf_[A-Za-z0-9_-]{4,}["']"""
        ),
        "a Hugging Face token literal (or its placeholder)",
    ),
    (
        re.compile(
            r"""\b(api_key|apikey|secret|password|passwd)\s*=\s*["'][^"'{}$][^"']{7,}["']""",
            re.IGNORECASE,
        ),
        "a credential assigned as a string literal",
    ),
    (
        re.compile(r"\b(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16})\b"),
        "a provider API key",
    ),
)

#: Absolute paths from somebody's machine. Two separate problems in one string: the
#: code does not run anywhere else, and the path leaks a real account name --
#: ``/home/azureuser/cloudfiles/code/gasq_research_qwen14/...`` appears twice in the
#: research inference script, naming the VM the experiments ran on.
#:
#: Applied to code only, not to prose. An absolute path in a README is an example;
#: an absolute path in a source file is a defect either way.
#: ``(?!user/)`` exempts the generic ``/home/user/...`` placeholder used in help
#: text and defaults. It is not a real account, and flagging it would push people
#: to invent a different placeholder rather than to stop hardcoding paths.
PERSONAL_PATH_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"""["'](/home/(?!user/)[A-Za-z0-9._-]+/[^"']*)["']"""),
        "an absolute path under someone's home directory",
    ),
    (
        re.compile(r"""["'](/Users/(?!user/)[A-Za-z0-9._-]+/[^"']*)["']"""),
        "an absolute path under someone's home directory",
    ),
    (
        re.compile(
            r"""["']([A-Za-z]:[\\/]+Users[\\/]+(?!user[\\/])[A-Za-z0-9._-]+[\\/][^"']*)["']"""
        ),
        "an absolute Windows user path",
    ),
)

#: Prose. Scanned for secrets, not for paths.
PROSE_SUFFIXES: frozenset[str] = frozenset({".md", ".txt"})

#: Extensions worth scanning for secrets. Binary and generated files are skipped.
TEXT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".py",
        ".pyi",
        ".sh",
        ".bash",
        ".ps1",
        ".yaml",
        ".yml",
        ".toml",
        ".json",
        ".cfg",
        ".ini",
        ".md",
        ".txt",
        ".cpp",
        ".cu",
        ".cuh",
        ".h",
        ".hpp",
        ".cmake",
        ".in",
    }
)

MAX_SCAN_BYTES = 2 * 1024 * 1024


def check(paths: list[str]) -> list[str]:
    problems: list[str] = []

    for raw in paths:
        path = Path(raw)
        posix = path.as_posix()

        for pattern in FORBIDDEN_PATHS:
            if pattern.search(posix):
                problems.append(
                    f"{posix}: this file type must never be committed "
                    f"(matched {pattern.pattern!r}). If it is genuinely needed, it "
                    f"belongs outside the repository or in release artifacts."
                )
                break

        if posix in EXEMPT_PATHS or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if not path.is_file() or path.stat().st_size > MAX_SCAN_BYTES:
            continue

        # errors="replace": a file that is not valid UTF-8 should still be scanned
        # rather than silently skipped, since skipping is the failure mode an
        # attacker (or a careless commit) would exploit.
        text = path.read_text(encoding="utf-8", errors="replace")
        is_prose = path.suffix.lower() in PROSE_SUFFIXES
        for line_number, line in enumerate(text.splitlines(), start=1):
            if "check_no_confidential" in line:  # this script's own doc examples
                continue
            for pattern, description in SECRET_PATTERNS:
                if pattern.search(line):
                    problems.append(
                        f"{posix}:{line_number}: {description}. Read it from the "
                        f"environment or huggingface_hub's cache instead."
                    )
                    break
            if is_prose:
                continue
            for pattern, description in PERSONAL_PATH_PATTERNS:
                match = pattern.search(line)
                if match:
                    problems.append(
                        f"{posix}:{line_number}: {description} ({match.group(1)!r}). "
                        f"Take it as a CLI argument or an environment variable."
                    )
                    break

    return problems


def main(argv: list[str]) -> int:
    problems = check(argv[1:])
    if not problems:
        return 0
    print("Refusing the commit:\n", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    print(
        "\nA commit cannot be un-made: the blob stays reachable in history even "
        "after the file is deleted.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
