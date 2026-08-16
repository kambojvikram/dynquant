"""The cache the harness reports, read where it is measured rather than at the end.

``decoded_cache_len`` exists to answer one question -- did the timed generation decode
with a KV cache at all -- and it answered it wrongly for every static-cache arm. Under
``cache_implementation="static"`` ``generate`` keeps one cache on the model and hands the
same object to every call, so a probe that returned the object and let the caller measure
it later reported whatever the *last* timed generation left behind: 54 tokens on
LFM2.5-8B-A1B and 58 on OLMoE-1B-7B, both exactly ``prompt + 32 - 1``, the capacity of the
cache the warmup sized. Under a dynamic cache each call allocates its own, so the same
code read the probe's own 30 and looked correct.

The failure is worth a test because it is invisible in the output: 54 is a plausible
number for a generation of 32 tokens, and the field was quoted in a report before the
arithmetic was checked. The regression here mutates the cache after the probe returns,
which is what the timed reps do, and fails if the value moves.

No model is loaded. The stubs return exactly the shapes ``_probe_cache`` touches, so every
number below is one this file chose.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "phase4" / "moe_end_to_end.py"


@pytest.fixture(scope="module")
def harness() -> Any:
    spec = importlib.util.spec_from_file_location("moe_end_to_end", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["moe_end_to_end"] = module
    spec.loader.exec_module(module)
    return module


class _LiveCache:
    """A cache whose length keeps moving, which is what a reused static cache does."""

    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length


class _Enc(dict):  # type: ignore[type-arg]
    def to(self, device: str) -> _Enc:
        return self


class _Tok:
    pad_token_id = 0
    eos_token_id = 2
    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        return str(messages)

    def __call__(self, text: str, **kwargs: Any) -> _Enc:
        return _Enc(input_ids=[[1, 2, 3]])


class _Out:
    def __init__(self, cache: Any) -> None:
        self.past_key_values = cache


class _Model:
    def __init__(self, cache: Any) -> None:
        self.cache = cache

    def generate(self, **kwargs: Any) -> _Out:
        return _Out(self.cache)


def test_cache_len_reports_absence_as_minus_one(harness: Any) -> None:
    """-1 and 0 are different findings: no cache at all, versus a cache holding nothing."""
    assert harness._cache_len(None) == -1
    assert harness._cache_len(_LiveCache(30)) == 30
    assert harness._cache_len(object()) == 0


def test_cache_len_falls_through_to_len(harness: Any) -> None:
    class _Legacy:
        def get_seq_length(self) -> int:
            raise IndexError("empty layer list")

        def __len__(self) -> int:
            return 7

    assert harness._cache_len(_Legacy()) == 7


def test_probe_reads_the_length_before_the_timed_reps_move_it(harness: Any) -> None:
    """The regression: returning the cache object made this read 99, not 30.

    A reused static cache is still being written to after the probe returns. Holding the
    reference and measuring at the end of ``_time_arm`` therefore reports the last timed
    generation's fill, which is how ``prompt + 32 - 1`` reached a record labelled as a
    four-token probe.
    """
    pytest.importorskip("transformers")  # `_probe_cache` builds a GenerationConfig
    cache = _LiveCache(30)
    model = _Model(cache)
    length = harness._probe_cache(model, _Tok(), "hello", "cpu", "static", True, "default")
    cache.length = 99
    assert length == 30


def test_probe_returns_minus_one_when_generate_decoded_without_a_cache(harness: Any) -> None:
    """``use_cache: false`` in a checkpoint is the case this field was added to catch."""
    pytest.importorskip("transformers")  # `_probe_cache` builds a GenerationConfig
    assert harness._probe_cache(_Model(None), _Tok(), "hello", "cpu", None, True, "default") == -1
