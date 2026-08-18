"""export -> load -> forward, against module trees ``transformers`` really builds.

[test_export_checkpoint.py][1] covers the writer thoroughly, but every fixture there
is a hand-built module: it checks that what the packer wrote is what the runtime
reads, on shapes we chose. Nothing checked the same path against a real architecture,
and the two failures that motivated this file were both architectural rather than
arithmetic -- a role that exists only on one model family, and a head that is untied
on some checkpoints and tied on others.

``qwen3_5_text`` is the case that prompted it. Its linear-attention blocks contribute
``lin_attn.a`` and ``lin_attn.b``, which are bare parameters rather than Linears, and
``lin_attn.conv``, which is a ``Conv1d`` and so has a weight of rank three; and its
head is untied against a 248k-row table. None of those shapes appears anywhere in the
synthetic fixtures, so before this file the first time the packer met them was on a
27B checkpoint at the end of a four-hour fine-tune.

Architectures the installed ``transformers`` does not know are **skipped, loudly**.
That is not a gap being papered over: the alternative is a test that silently passes
on the machine least able to run it, and a skip that names the model type is the
thing that tells a reader why a guard they expected to see did not run.

[1]: test_export_checkpoint.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers", reason="round-trips real model trees")

pytestmark = pytest.mark.needs_hf

#: Small enough to build and pack in seconds, large enough that a group of 128 still
#: has something to say: ``hidden_size`` stays a multiple of the group size so the
#: packer's aligned path is what runs, and the ragged tail has its own tests elsewhere.
TINY = {
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "vocab_size": 2048,
}

ARCHITECTURES = ["llama", "qwen3", "qwen3_5_text"]


def _dtype_kwarg(func, dtype) -> dict:
    """Spell the compute dtype the way the installed ``transformers`` spells it.

    v5 renamed ``torch_dtype`` to ``dtype`` on both ``from_config`` and
    ``from_pretrained``. The old name still works there but warns; the new name is a
    ``TypeError`` on 4.x. Chosen from the signature rather than from
    ``transformers.__version__``, for the same reason the driver picks its warmup field
    that way: a version string is a claim about a distribution and this is a question
    about a function.
    """
    import inspect

    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):  # C-implemented or wrapped beyond inspection
        return {"dtype": dtype}
    if "dtype" in parameters:
        return {"dtype": dtype}
    if "torch_dtype" in parameters:
        return {"torch_dtype": dtype}
    # Neither: both spellings would raise, and the default is fp32 anyway, which is
    # what every caller here asks for.
    return {}


def _tiny_config(model_type: str):
    from transformers import AutoConfig

    try:
        config = AutoConfig.for_model(model_type)
    except ValueError:
        pytest.skip(f"the installed transformers {transformers.__version__} has no {model_type}")

    for field, value in TINY.items():
        if hasattr(config, field):
            setattr(config, field, value)

    # ``layer_types`` is per-layer and transformers validates that it still matches the
    # depth. It is also the field that decides which blocks are linear-attention, which
    # is the whole reason this architecture is here -- so the prefix is grown until it
    # contains every type the full-size default used, rather than cut to a round number
    # that might happen to hold only one kind.
    types = list(getattr(config, "layer_types", []) or [])
    if types:
        want = set(types)
        depth = next((i for i in range(1, len(types) + 1) if set(types[:i]) == want), len(types))
        config.layer_types = types[:depth]
        config.num_hidden_layers = depth

    # The special ids came from the real vocabulary and now point past the end of this
    # one. transformers warns rather than refuses, and a warning here would be noise
    # standing in front of the failure the test exists to show.
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(config, field, None)
        if isinstance(value, int) and value >= config.vocab_size:
            setattr(config, field, config.vocab_size - 1)
    return config


def _tiny_model(config):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(
        config, **_dtype_kwarg(AutoModelForCausalLM.from_config, torch.float32)
    )
    model = model.to(torch.float32)
    model.eval()
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if isinstance(getattr(model.generation_config, field, None), int):
            setattr(model.generation_config, field, config.vocab_size - 1)
    return model


def _export(src, out, bits: int) -> None:
    """Through the CLI, because that is what a user runs.

    Argument parsing and the command's own resolved output path are part of the thing
    under test, and an in-process call to the handler skips both.
    """
    # The child gets this process's import path, not the interpreter's default. Under an
    # editable or src-layout install dynquant reaches the test through sys.path rather
    # than through site-packages, and a bare subprocess would report `No module named
    # dynquant` -- a failure about the harness wearing the costume of a failure about
    # the packer.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    proc = subprocess.run(
        [
            sys.executable, "-m", "dynquant", "export", str(src),
            "--uniform", str(bits),
            "--device", "cpu", "--compute-device", "cpu",
            "-o", str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, f"export failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"


@pytest.mark.parametrize("model_type", ARCHITECTURES)
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_a_real_tree_packs_writes_loads_and_answers(model_type: str, bits: int, tmp_path) -> None:
    """The whole path, end to end, at every width an arm can be allocated.

    Every width rather than one: the packers are separate code per width, 3-bit most of
    all -- it is the only one whose values do not divide a 32-bit word evenly -- and a
    round-trip that only ever ran at 4 bits would never have met it.

    Turns red when: a new role's weight shape reaches the packer and it writes something
    no loader can read back, which stays silent until a forward pass runs.
    """
    from transformers import AutoModelForCausalLM

    config = _tiny_config(model_type)
    src, out = tmp_path / "src", tmp_path / f"q{bits}"
    _tiny_model(config).save_pretrained(src)
    _export(src, out, bits)

    from dynquant.integration.hf_quantizer import register_hf_quantizer

    # Installing dynquant does not register the quantizer with transformers -- there is
    # no entry point for it, only for vLLM and SGLang. Without this call transformers
    # cannot resolve `quant_method`, and it answers that with a *warning* and a randomly
    # initialised model rather than an exception. So the call comes first, and its
    # absence is the failure the assertion below would otherwise be unable to see.
    register_hf_quantizer()

    written = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert (written.get("quantization_config") or {}).get("quant_method") == "dynquant"

    back = AutoModelForCausalLM.from_pretrained(
        out,
        device_map="cpu",
        **_dtype_kwarg(AutoModelForCausalLM.from_pretrained, torch.float32),
    )
    packed = {
        type(module).__name__
        for module in back.modules()
        if type(module).__name__.startswith("DynQuant")
    }
    assert packed, f"{model_type} at {bits}b loaded with no DynQuant module: quantizer never ran"

    ids = torch.arange(16).unsqueeze(0) % config.vocab_size
    with torch.no_grad():
        logits = back(ids).logits
    assert logits.shape == (1, 16, config.vocab_size)
    assert torch.isfinite(logits).all(), f"{model_type} at {bits}b produced non-finite logits"


@pytest.mark.parametrize("model_type", ARCHITECTURES)
def test_narrower_is_smaller_on_disk(model_type: str, tmp_path) -> None:
    """Bytes fall with width, which is the claim the whole method rests on.

    Cheap, and it catches what a forward pass cannot: a width that packs and loads and
    answers, but stores at the width above it. That reads as a working arm, and as a
    compression ratio that quietly is not the one printed on the card.

    Turns red when: a packer falls back to a wider container for a shape it cannot
    handle, instead of refusing.
    """
    config = _tiny_config(model_type)
    src = tmp_path / "src"
    _tiny_model(config).save_pretrained(src)

    sizes = {}
    for bits in (2, 4, 8):
        out = tmp_path / f"q{bits}"
        _export(src, out, bits)
        sizes[bits] = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())

    assert sizes[2] < sizes[4] < sizes[8], sizes
