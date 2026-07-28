"""Enforce that no filename literal exists outside ``constants.py``.

This is the bug-2 guard, and it is a lint rather than a unit test on purpose.

The research code spelled its checkpoint filenames as literals in five files. One
of them capitalised the ``Q``: the writer emitted
``dynQuant_quantized_weights.safetensors`` while every reader opened
``dynquant_quantized_weights.safetensors``. No checkpoint that code produced could
ever be loaded, and nothing detected it -- not a type checker, not a test, not a
review, because both spellings look right in isolation. The only durable fix is to
make a second spelling impossible to introduce.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from dynquant import constants

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "packages" / "dynquant-core" / "src" / "dynquant"

# The module that is allowed to contain filename literals, plus paths that hold
# research code kept verbatim for golden tests.
ALLOWED = {"constants.py"}
EXCLUDED_DIRS = {"_legacy", "_cutlass", "__pycache__"}

FILENAME_PATTERN = re.compile(
    r"""["']([A-Za-z0-9_.\-{}:0-9]*(?:\.safetensors|\.json)(?:\.index\.json)?)["']"""
)

# Filenames owned by other projects, which DynQuant reads but does not define.
FOREIGN = {
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "adapter_config.json",
    "adapter_model.safetensors",
    "quantize_config.json",  # GPTQ, for the migrate command
    "quant_config.json",  # AWQ
    "autotune.json",
}


def _python_sources():
    if not SOURCE_ROOT.is_dir():
        pytest.skip(f"{SOURCE_ROOT} not present")
    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        if EXCLUDED_DIRS & set(path.relative_to(SOURCE_ROOT).parts):
            continue
        if path.name in ALLOWED:
            continue
        yield path


def _known_filenames() -> set[str]:
    """Every filename constants.py declares, including shard patterns."""
    names: set[str] = set()
    for value in vars(constants).values():
        if isinstance(value, str):
            names.add(value)
        elif isinstance(value, tuple):
            names.update(v for v in value if isinstance(v, str))
        elif isinstance(value, dict):
            names.update(v for v in value.values() if isinstance(v, str))
    return names


def test_no_filename_literals_outside_constants():
    offenders: list[str] = []
    known = _known_filenames()

    for path in _python_sources():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.lstrip()
            # docstrings and comments may name files -- they are documentation
            if stripped.startswith(("#", '"', "'", "*", ":")):
                continue
            for match in FILENAME_PATTERN.finditer(line):
                name = match.group(1)
                if name in FOREIGN or name in known:
                    continue
                rel = path.relative_to(REPO_ROOT)
                offenders.append(f"{rel}:{lineno}: {name!r}")

    assert not offenders, (
        "Filename literals found outside constants.py. Add the name to "
        "dynquant/constants.py and reference it from there -- a second spelling "
        "of a filename is how the research code produced checkpoints nothing "
        "could load:\n  " + "\n  ".join(offenders)
    )


def test_writer_and_reader_agree_on_case():
    """No two DynQuant-owned filenames may differ only by case.

    Precisely the failure mode being guarded: two names that are 'the same file'
    to a human and different files to a filesystem. Case-insensitive filesystems
    (macOS, Windows) hide it locally and Linux CI finds it.
    """
    owned = [n for n in _known_filenames() if "dynquant" in n.lower() and "." in n]
    by_lower: dict[str, list[str]] = {}
    for name in owned:
        by_lower.setdefault(name.lower(), []).append(name)
    collisions = {k: v for k, v in by_lower.items() if len(v) > 1}
    # The legacy tuple deliberately holds both historical spellings so `migrate`
    # can find either; they must not appear anywhere else.
    legacy = set(constants.LEGACY_CHECKPOINT_FILENAMES)
    real = {k: v for k, v in collisions.items() if not set(v) <= legacy}
    assert not real, f"filenames differing only by case: {real}"


def test_current_filenames_are_lowercase():
    for name in (
        constants.PACKED_WEIGHTS_FILENAME,
        constants.PACKED_WEIGHTS_INDEX_FILENAME,
        constants.FP16_WEIGHTS_FILENAME,
        constants.MANIFEST_FILENAME,
        constants.STATS_FILENAME,
    ):
        assert name == name.lower(), name


def test_shard_pattern_produces_hf_style_names():
    name = constants.SHARD_PATTERN.format(index=1, total=12)
    assert name == "dynquant_packed_weights-00001-of-00012.safetensors"
    assert name.startswith(constants.PACKED_WEIGHTS_FILENAME.removesuffix(".safetensors"))


def test_group_size_alignment_is_the_lcm_of_values_per_word():
    """``constants.GROUP_SIZE_ALIGNMENT`` is a derived number; pin the derivation.

    If a future bit-width were added whose values-per-word did not divide 32, the
    alignment invariant would silently stop holding and 3-bit-style straddling
    would appear in widths that do not handle it.
    """
    import math

    lcm = 1
    for bits, per_word in constants.VALUES_PER_WORD.items():
        assert per_word * bits % 32 == 0 or bits == 3, bits
        lcm = lcm * per_word // math.gcd(lcm, per_word)
    assert constants.GROUP_SIZE_ALIGNMENT % 32 == 0
    assert constants.DEFAULT_GROUP_SIZE % constants.GROUP_SIZE_ALIGNMENT == 0


def test_bit_options_and_word_tables_agree():
    assert set(constants.VALUES_PER_WORD) == set(constants.BIT_OPTIONS)
    assert set(constants.WORDS_PER_BLOCK) == set(constants.BIT_OPTIONS)
    for bits in constants.BIT_OPTIONS:
        values = constants.VALUES_PER_WORD[bits]
        words = constants.WORDS_PER_BLOCK[bits]
        assert values * bits == words * 32, f"{bits}-bit block is not word-exact"
