"""QuantTensor survives a real safetensors file, not just a Python dict.

Distinct from the state_dict test: safetensors imposes constraints a dict does
not -- contiguity, no shared storage, a flat string->tensor namespace, and
metadata that must be ``dict[str, str]``. Every one of those has bitten a
quantization project. The research code never wrote a checkpoint that could be
read back at all (writer emitted a capital-Q filename, readers opened lowercase),
so "it round-trips through the actual file format" is the claim worth pinning.
"""

from __future__ import annotations

import json

import pytest

from dynquant.constants import (
    MANIFEST_SCHEMA,
    PACKED_WEIGHTS_FILENAME,
    QUANT_TENSOR_SUFFIXES,
)
from dynquant.quant.tensor import QuantTensor

safetensors_torch = pytest.importorskip("safetensors.torch")


LAYERS = [
    ("model.layers.0.self_attn.q_proj", (256, 512), 4, 128),
    ("model.layers.0.self_attn.k_proj", (64, 512), 8, 128),
    ("model.layers.0.mlp.gate_proj", (512, 256), 4, 64),
    ("model.layers.0.mlp.up_proj", (512, 256), 3, 128),
    ("model.layers.0.mlp.down_proj", (256, 512), 2, 128),
    ("model.layers.0.mlp.experts_stacked", (4, 32, 256), 3, 128),
    ("model.embed_tokens", (1000, 128), 4, -1),
    ("odd.shaped.proj", (17, 300), 3, 128),
]


@pytest.fixture
def quantized_layers(torch_seeded):
    torch = torch_seeded
    out = {}
    for name, shape, bits, group_size in LAYERS:
        dense = torch.randn(*shape, dtype=torch.float16) * 0.02
        out[name] = (
            dense,
            QuantTensor.from_dense(dense, bits=bits, group_size=group_size),
        )
    return out


def test_full_checkpoint_roundtrip(quantized_layers, tmp_path):
    """Write every layer into one file with a manifest, then read it all back."""
    import torch

    tensors: dict[str, torch.Tensor] = {}
    manifest: dict[str, object] = {"schema": MANIFEST_SCHEMA, "layers": {}}
    for name, (_dense, qt) in quantized_layers.items():
        tensors.update(qt.state_dict(name))
        manifest["layers"][name] = qt.metadata()  # type: ignore[index]

    # The manifest must survive JSON, since that is how it reaches disk.
    manifest = json.loads(json.dumps(manifest))

    path = tmp_path / PACKED_WEIGHTS_FILENAME
    safetensors_torch.save_file(tensors, str(path))
    loaded = safetensors_torch.load_file(str(path))

    assert set(loaded) == set(tensors)

    for name, (dense, qt) in quantized_layers.items():
        restored = QuantTensor.from_state_dict(name, loaded, manifest["layers"][name])  # type: ignore[index]
        restored.validate()
        assert restored.bits == qt.bits
        assert restored.group_size == qt.group_size
        assert restored.in_features == qt.in_features
        assert restored.logical_shape == qt.logical_shape
        assert restored.symmetric == qt.symmetric
        # the numbers, not just the shapes
        assert torch.equal(restored.dequantize(), qt.dequantize())
        # and the error is still what it was before the file existed
        assert restored.quantization_error(dense) == pytest.approx(
            qt.quantization_error(dense), rel=1e-6
        )


def test_saved_tensors_are_contiguous_and_unshared(quantized_layers):
    """safetensors refuses shared storage, and a non-contiguous tensor would be
    silently copied -- changing the on-disk size from what the manifest claims."""
    seen: set[int] = set()
    for name, (_dense, qt) in quantized_layers.items():
        for key, tensor in qt.state_dict(name).items():
            assert tensor.is_contiguous(), key
            ptr = tensor.data_ptr()
            assert ptr not in seen, f"{key} shares storage with an earlier tensor"
            seen.add(ptr)


def test_on_disk_size_matches_reported_nbytes(quantized_layers, tmp_path):
    """The budget the allocator promises must be the size on the filesystem.

    safetensors adds a small JSON header, so the check is that the payload
    accounts for the file minus that header -- not that the numbers are equal.
    """
    tensors = {}
    expected = 0
    for name, (_dense, qt) in quantized_layers.items():
        tensors.update(qt.state_dict(name))
        expected += qt.nbytes

    path = tmp_path / PACKED_WEIGHTS_FILENAME
    safetensors_torch.save_file(tensors, str(path))
    actual = path.stat().st_size

    header = actual - expected
    assert 0 < header < 8192, (
        f"file is {actual} bytes but tensors account for {expected}; "
        f"unexplained difference of {header}"
    )


def test_key_suffixes_come_from_constants(quantized_layers):
    """Keys must be built from the shared suffix table, not from literals.

    Storage keys are the contract between the writer, the HF loader, and the CUDA
    kernels. A drifted suffix produces a KeyError at load time in the best case
    and a wrong-tensor read in the worst.
    """
    name = "model.layers.0.mlp.down_proj"
    qt = quantized_layers[name][1]
    keys = set(qt.state_dict(name))
    assert keys <= {f"{name}.{s}" for s in QUANT_TENSOR_SUFFIXES.values()}
    assert f"{name}.{QUANT_TENSOR_SUFFIXES['packed']}" in keys
    assert f"{name}.{QUANT_TENSOR_SUFFIXES['scale']}" in keys


def test_metadata_is_all_json_scalars(quantized_layers):
    """The manifest is JSON, so no enums, tuples-of-tensors, or numpy scalars may
    leak into it -- those serialise as ``str(obj)`` and never parse back."""
    for _name, (_dense, qt) in quantized_layers.items():
        meta = qt.metadata()
        reparsed = json.loads(json.dumps(meta))
        assert reparsed == json.loads(json.dumps(reparsed)), "metadata is not JSON-stable"
        for key, value in reparsed.items():
            assert isinstance(value, (str, int, float, bool, list, type(None))), (key, value)


def test_partial_load_reads_only_requested_layers(quantized_layers, tmp_path):
    """The HF load path streams one layer at a time; peak RSS depends on it."""
    tensors = {}
    for name, (_dense, qt) in quantized_layers.items():
        tensors.update(qt.state_dict(name))
    path = tmp_path / PACKED_WEIGHTS_FILENAME
    safetensors_torch.save_file(tensors, str(path))

    target = "model.layers.0.mlp.up_proj"
    with safetensors_torch.safe_open(str(path), framework="pt") as handle:
        # `.keys()` is not redundant here: safetensors' `safe_open` handle exposes
        # keys() but is not iterable, so `for k in handle` raises TypeError.
        subset = {
            k: handle.get_tensor(k)
            for k in handle.keys()  # noqa: SIM118
            if k.startswith(target + ".")
        }
    assert subset
    restored = QuantTensor.from_state_dict(target, subset, quantized_layers[target][1].metadata())
    restored.validate()
