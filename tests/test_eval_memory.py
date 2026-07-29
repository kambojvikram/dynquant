"""The two VRAM figures `dynquant eval --pack` reports, and why they are two.

Packing borrows the accelerator for arithmetic even when the model lives in host
RAM (see ``dynquant.quant.device``), so the CUDA peak during a pack spans a working
set that is gone before anything runs. It answers "what did packing cost", which is
a question about the tool. A reader of a results table is asking a different one --
"how much does the packed model hold" -- and that is the headline claim.

Reporting the peak under either name would overstate resident VRAM by the size of a
transient, in the direction that flatters the method. So the two are recorded
separately, and the resident figure is read *after* the caching allocator has
released what packing borrowed. This file pins that ordering, because reversing it
produces no error and no warning -- only a larger number.

None of it needs a GPU: the CUDA accounting surface is replaced by one that records
the order it was called in.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from dynquant.commands.evaluate import _pack  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

IN_FEATURES = 128
OUT_FEATURES = 64


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)


def _args(tmp_path: Path) -> argparse.Namespace:
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"fc": 4}), encoding="utf-8")
    return argparse.Namespace(
        map=str(path),
        map_key=None,
        group_size=IN_FEATURES,
        # The point here is the accounting, not the arithmetic, so encoding stays
        # on the weight's own device and the stub below never has to be a real card.
        compute_device="none",
        quiet=True,
    )


def test_both_keys_are_present_without_a_card(tmp_path: Path) -> None:
    """Absent, not zero, and never missing.

    A results table that reads ``record["cuda_resident_bytes"]`` must get ``None``
    on a CPU box rather than a ``KeyError``, and must not get ``0`` -- which would
    be indistinguishable from a packed model that holds nothing.
    """
    record = _pack(_Tiny(), _args(tmp_path))

    assert "cuda_pack_peak_bytes" in record
    assert "cuda_resident_bytes" in record
    if not torch.cuda.is_available():
        assert record["cuda_pack_peak_bytes"] is None
        assert record["cuda_resident_bytes"] is None


def test_resident_is_read_after_the_transients_are_released(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Peak before the cache is emptied, resident after. Reversing it is silent."""
    events: list[str] = []

    def _record(name: str, value: int = 0) -> Any:
        def hook(*_a: object, **_k: object) -> int:
            events.append(name)
            return value

        return hook

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", _record("reset"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", _record("peak", 4_000_000_000))
    monkeypatch.setattr(torch.cuda, "empty_cache", _record("empty_cache"))
    monkeypatch.setattr(torch.cuda, "memory_allocated", _record("resident", 1_000_000_000))

    record = _pack(_Tiny(), _args(tmp_path))

    assert events == ["reset", "peak", "empty_cache", "resident"], (
        "the peak must be read before the cache is emptied and the resident figure "
        "after, or the headline number carries packing's transients"
    )
    assert record["cuda_pack_peak_bytes"] == 4_000_000_000
    assert record["cuda_resident_bytes"] == 1_000_000_000


def test_the_peak_counter_is_reset_before_packing_and_not_after(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Whatever the caller did before us is not part of what packing cost.

    Without the reset the peak is a high-water mark for the whole process -- it
    would include loading the dense model, which is precisely the allocation the
    packed path exists to avoid reporting.
    """
    events: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda *a, **k: events.append("reset")
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 1)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 1)

    def packed(*_a: object, **_k: object) -> Any:
        events.append("pack")
        raise AssertionError("unreachable")

    monkeypatch.setattr("dynquant.runtime.linear.pack_model", packed)

    with pytest.raises(AssertionError, match="unreachable"):
        _pack(_Tiny(), _args(tmp_path))

    assert events == ["reset", "pack"], "the counter must be cleared before packing starts"
