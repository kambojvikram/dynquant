"""The command line, and the shared machinery under it.

Everything here runs on CPU with no checkpoint, no download and no GPU, which is
the property that makes it run on every push rather than nightly. That is possible
because each command is a thin shell over a function taking an ``nn.Module`` and
returning data, so the synthetic Qwen3.5-shaped fixture from
:mod:`test_graph_classify` can drive the parts that matter.

The two invariants worth stating outright, because both are easy to break by
adding an innocuous import:

* **Parser construction imports nothing heavy.** ``dynquant doctor`` exists to
  diagnose an install where ``torch`` or the kernels are broken, and it cannot do
  that if building the parser imports them first. Every handler therefore imports
  inside its own function.
* **``inspect`` and ``quantize`` allocate through the same code.** A reviewed bit
  map is only worth reviewing if it is the map that gets applied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import pytest
from test_graph_classify import Qwen3_5ForCausalLM

from dynquant.allocate.knapsack import BitMap, FloorViolation
from dynquant.cli import EVAL_TASKS, EXIT_FAILED, build_parser, main
from dynquant.commands import _shared, bench, evaluate, inspect, quantize
from dynquant.constants import ALLOCATION_FILENAME, ALLOCATION_SCHEMA, BIT_OPTIONS
from dynquant.errors import DynQuantError
from dynquant.graph import ModuleRole, classify_model

COMMANDS = ("inspect", "quantize", "export", "eval", "bench", "doctor", "version")


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def test_the_cli_is_reachable_as_python_dash_m_dynquant() -> None:
    """``sys.executable -m dynquant`` has to work, because the drivers only have that.

    A console script lands in an install-chosen ``bin/`` that a subprocess cannot
    assume is on ``PATH``; ``-m`` on the running interpreter is the form that is
    guaranteed to reach *this* environment's package, so every phase-3 driver shells
    out that way. ``scripts/run_s1_headroom.py`` did it before ``__main__.py``
    existed and would have failed on every cell -- unnoticed, because the screen that
    produced the S1 records used a different command.

    Turns red if ``__main__.py`` is dropped, if it stops calling ``cli.main``, or if
    it starts importing something that is not installed in a bare environment.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run(
        [sys.executable, "-m", "dynquant", "version"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "dynquant" in proc.stdout.lower()


def test_dash_m_exits_with_the_status_main_returns() -> None:
    """A usage error through ``-m`` must be a non-zero exit, not a swallowed one.

    ``runpy`` does not propagate a return value, so ``main()``'s status is only an
    exit status if ``__main__`` passes it to ``sys.exit``. Without that the drivers'
    ``check``/``returncode`` handling sees success on every failed cell.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run(
        [sys.executable, "-m", "dynquant"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode != 0, "a bare invocation reported success"


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------


def test_every_command_is_registered_with_a_handler() -> None:
    parser = build_parser()
    (subparsers,) = [
        action
        for action in parser._actions
        if hasattr(action, "choices") and action.dest == "command"
    ]
    assert set(subparsers.choices) == set(COMMANDS)
    for name, sub in subparsers.choices.items():
        assert sub.get_default("handler") is not None, f"{name} has no handler"


def test_every_registered_eval_task_is_reachable_from_the_command_line() -> None:
    """A task in the registry that argparse refuses is implemented and unrunnable.

    This is not hypothetical. ``text2sql`` shipped with a loader, a scorer, a spec in
    ``TASKS`` and its own test file, and could not be run: ``--task`` carried a
    hand-written tuple of the other six. The usage error named those six, so it read as
    a typo rather than as the omission it was, and nothing in the suite disagreed.

    Turns red when a task is added to ``TASKS`` and the parser is not derived from it.
    """
    parser = build_parser()
    for task in EVAL_TASKS:
        assert parser.parse_args(["eval", "m", "--task", task]).task == task


def test_building_the_parser_imports_nothing_heavy() -> None:
    """``dynquant doctor`` must run on an install where torch is the broken thing.

    A subprocess rather than a check on this process's ``sys.modules``: the rest of
    the suite has already imported torch, so an in-process assertion would pass
    whatever the CLI does.
    """
    source = (
        "import sys;"
        "from dynquant.cli import build_parser;"
        "build_parser();"
        "heavy={m.split('.')[0] for m in sys.modules} &"
        " {'torch','transformers','datasets','dynquant_kernels','safetensors'};"
        "print(sorted(heavy))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False, env=env
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", f"parser construction imported {proc.stdout.strip()}"


def test_a_bare_invocation_is_a_usage_error_not_a_success() -> None:
    """Exit 0 on no arguments would read as "it worked"."""
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_a_dynquant_error_exits_one_with_the_message_and_no_traceback(capsys) -> None:
    """Every DynQuantError is written to be read by whoever hit it, so the message
    is the whole report and a traceback would only bury it."""
    assert main(["version"]) == 0  # main() itself works, so the failure below is real
    capsys.readouterr()
    code = main(["quantize", "model"])  # -o omitted: raises before anything loads
    captured = capsys.readouterr()
    assert code == EXIT_FAILED
    assert captured.err.startswith("error: ")
    assert "Traceback" not in captured.err
    assert "--dry-run" in captured.err  # the message names the way out


def test_export_defaults_to_cpu_because_it_streams_the_weights() -> None:
    """Export encodes one module at a time, so the model need not be resident."""
    assert build_parser().parse_args(["export", "m", "-o", "o"]).device == "cpu"


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("4.45.0", "torch_dtype"),
        ("4.55.4", "torch_dtype"),
        ("4.56.0", "dtype"),
        ("4.57.1", "dtype"),
        ("5.0.0.dev0", "dtype"),
        ("", "dtype"),
    ],
)
def test_the_dtype_argument_is_named_the_way_this_transformers_names_it(
    version: str, expected: str
) -> None:
    """4.56 renamed `torch_dtype` to `dtype`, and core declares `>=4.45`.

    The failure mode this guards is not a clean TypeError at the call site: the
    new name on an older release falls through `**kwargs` into the model
    constructor and comes out as `LlamaForCausalLM.__init__() got an unexpected
    keyword argument 'dtype'`, which names neither transformers nor the version.
    """
    assert _shared._dtype_kwarg(argparse.Namespace(__version__=version)) == expected


# --------------------------------------------------------------------------
# Which class loads the checkpoint
# --------------------------------------------------------------------------
#
# `AutoModelForCausalLM` is right for every model this project has published and
# wrong for the next one. A composite checkpoint -- Qwen3-Omni is two full MoE
# towers plus an audio tower, a visual tower and a vocoder in one directory --
# registers as `Qwen3OmniMoeForConditionalGeneration`, which is not in the
# causal-LM mapping at all. `from_pretrained` raises before it reads a weight, so
# every command that loads a model is unavailable on the model, not degraded on it.
#
# The shape of the fix is the same one `_bank_config` uses in the classifier:
# the historical answer is tried first and returned by identity when it applies,
# so the new path is reachable only where the old one raised.


class _FakeMapping:
    """A stand-in for `MODEL_FOR_CAUSAL_LM_MAPPING`, which is lazy in transformers.

    Membership is the only operation `_claims_causal_lm` is allowed to use:
    `__getitem__` on the real mapping imports the modelling module, and asking
    which class to load should not cost fifty megabytes of imports.
    """

    def __init__(self, *claimed: type) -> None:
        self._claimed = set(claimed)
        self.subscripted = False

    def __contains__(self, key: object) -> bool:
        return key in self._claimed

    def __getitem__(self, key: object) -> None:  # pragma: no cover -- must not be called
        self.subscripted = True
        raise AssertionError("membership must not subscript the lazy mapping")


class _CausalConfig:
    architectures: ClassVar[list[str]] = ["StubForCausalLM"]


class _CompositeConfig:
    """No entry in the causal-LM mapping; names its own class, as save_pretrained does."""

    architectures: ClassVar[list[str]] = ["StubOmniForConditionalGeneration"]


class _RemoteCodeConfig:
    """A checkpoint whose own code nominates the auto class."""

    architectures: ClassVar[list[str]] = ["SomethingNotInThisTransformers"]
    auto_map: ClassVar[dict[str, str]] = {
        "AutoModelForCausalLM": "modelling_stub.SomethingNotInThisTransformers"
    }


class _NamelessConfig:
    architectures: ClassVar[list[str]] = []


def _fake_transformers(config: object, *, claimed: tuple[type, ...] = ()) -> argparse.Namespace:
    mapping = _FakeMapping(*claimed)
    return argparse.Namespace(
        __version__="5.14.1",
        AutoConfig=argparse.Namespace(from_pretrained=lambda _p, **_k: config),
        AutoModelForCausalLM="<AutoModelForCausalLM>",
        StubForCausalLM="<StubForCausalLM>",
        StubOmniForConditionalGeneration="<StubOmniForConditionalGeneration>",
        MODEL_FOR_CAUSAL_LM_MAPPING=mapping,
    )


def test_a_causal_lm_checkpoint_still_loads_through_the_class_it_always_did() -> None:
    """The no-regression guard, by identity rather than by equality.

    Every published arm loaded through `AutoModelForCausalLM`. The fallback below
    must be unreachable for those checkpoints, not merely equivalent for them.
    """
    fake = _fake_transformers(_CausalConfig(), claimed=(_CausalConfig,))
    assert _shared._model_loader(fake, "m") is fake.AutoModelForCausalLM
    assert not fake.MODEL_FOR_CAUSAL_LM_MAPPING.subscripted


def test_a_composite_checkpoint_loads_through_the_class_that_saved_it() -> None:
    """`config.architectures` is the checkpoint's own record, not an inference of ours.

    Preferring it means the fallback cannot pick a *different* head than the
    weights came from: it finds the right class or finds none.
    """
    fake = _fake_transformers(_CompositeConfig())
    assert _shared._model_loader(fake, "m") is fake.StubOmniForConditionalGeneration


def test_remote_code_that_nominates_the_auto_class_is_believed() -> None:
    """A `trust_remote_code` checkpoint is absent from the mapping and still causal.

    Its `auto_map` is the answer; reading `architectures` first would send it to a
    class this transformers does not export.
    """
    fake = _fake_transformers(_RemoteCodeConfig())
    assert _shared._model_loader(fake, "m") is fake.AutoModelForCausalLM


def test_naming_a_class_pins_it_without_reading_the_config() -> None:
    read = []

    def _record(_path: str, **_kwargs: object) -> _CausalConfig:
        read.append(_path)
        return _CausalConfig()

    fake = _fake_transformers(_CausalConfig(), claimed=(_CausalConfig,))
    fake.AutoConfig = argparse.Namespace(from_pretrained=_record)
    loader = _shared._model_loader(fake, "m", model_class="StubOmniForConditionalGeneration")
    assert loader is fake.StubOmniForConditionalGeneration
    assert not read, "an explicit class should not need the config"


def test_a_class_this_transformers_does_not_export_is_named_in_the_error() -> None:
    fake = _fake_transformers(_CausalConfig(), claimed=(_CausalConfig,))
    with pytest.raises(DynQuantError, match="Qwen4Whatever"):
        _shared._model_loader(fake, "m", model_class="Qwen4Whatever")


def test_a_checkpoint_no_class_can_build_says_what_it_looked_for() -> None:
    """The failure this replaces said `Unrecognized configuration class` and named
    neither the class that would have worked nor where to find it."""
    fake = _fake_transformers(_CompositeConfig())
    del fake.StubOmniForConditionalGeneration
    with pytest.raises(DynQuantError, match="StubOmniForConditionalGeneration"):
        _shared._model_loader(fake, "m")

    bare = _fake_transformers(_NamelessConfig())
    with pytest.raises(DynQuantError, match="none listed"):
        _shared._model_loader(bare, "m")


@pytest.mark.parametrize("command", ["inspect", "quantize", "export", "eval"])
def test_every_command_that_loads_a_model_can_be_told_which_class(command: str) -> None:
    """One flag on the shared loading group, so the four cannot drift apart."""
    parser = build_parser()
    argv = {
        "inspect": [command, "m"],
        "quantize": [command, "m", "-o", "o"],
        "export": [command, "m", "-o", "o"],
        "eval": [command, "m", "--task", "gsm8k"],
    }[command]
    assert parser.parse_args(argv).model_class is None
    pinned = parser.parse_args([*argv, "--model-class", "Qwen3OmniMoeForConditionalGeneration"])
    assert pinned.model_class == "Qwen3OmniMoeForConditionalGeneration"


def test_inspect_defaults_to_cpu_and_quantize_to_cuda() -> None:
    """Not cosmetic: inspect reads names and shapes, and a second copy of the
    weights on the GPU during an evaluation is how an analysis becomes an OOM."""
    parser = build_parser()
    assert parser.parse_args(["inspect", "m"]).device == "cpu"
    assert parser.parse_args(["quantize", "m", "-o", "o"]).device == "cuda"


def test_a_repeated_target_flag_accumulates_rather_than_replacing() -> None:
    """The help has always called it repeatable; ``nargs="+"`` alone is not.

    With the default ``store`` action a second ``--target`` overwrites the first, so
    ``--target 3.0 --target 4.0`` measured one budget and printed one table -- and the
    table it printed was for a budget the reader had asked to *compare against*, not the
    one they were reading. Nothing warns, and the output is a well-formed report.

    Turns red when: the action goes back to the default, or the space-separated spelling
    stops working.
    """
    parser = build_parser()
    assert parser.parse_args(["inspect", "m", "--target", "3.0", "4.0"]).target == [3.0, 4.0]
    assert parser.parse_args(["inspect", "m", "--target", "3.0", "--target", "4.0"]).target == [
        3.0,
        4.0,
    ]
    # Absent entirely still means absent, not an empty run with a silent default.
    assert parser.parse_args(["inspect", "m"]).target is None


def test_every_target_label_parses_back_to_the_target_it_names() -> None:
    """Distinctness is not the property; identifying the budget is.

    At two decimals ``4.005`` and ``4.015`` become ``"4.00"`` and ``"4.01"`` -- distinct,
    so a same-run collision check waves them through, and both name a different number
    than the one that was measured. ``"4.00"`` then collides across runs with a genuine
    ``4.00``, which is the version of this bug that survives into a published table.

    Round-tripping is the stronger requirement and it gives distinctness for free.

    Turns red when: the label format goes back to a fixed width, or the check weakens to
    ``len(set(labels)) == len(labels)``.
    """
    for targets in ([3.0, 4.0], [4.005, 4.015], [4.0, 4.005, 4.01, 4.015], [3.15]):
        labels = inspect._target_labels(targets)
        assert [float(label) for label in labels] == list(targets), (targets, labels)
        assert len(set(labels)) == len(labels)

    # The ordinary case keeps the spelling every record already written uses.
    assert inspect._target_labels([3.0, 4.0]) == ["3.00", "4.00"]


def test_the_same_budget_asked_for_twice_is_refused() -> None:
    """One allocation computed twice and reported as a comparison is not a comparison.

    Turns red when: duplicates start being merged into one row, which is the silent
    behaviour this whole change exists to remove.
    """
    with pytest.raises(DynQuantError, match="asked for twice"):
        inspect._target_labels([4.0, 4.0])


def test_uniform_is_a_list_for_inspect_and_a_scalar_for_quantize() -> None:
    """inspect compares several control arms in one pass; quantize writes one."""
    parser = build_parser()
    assert parser.parse_args(["inspect", "m", "--uniform", "3", "4"]).uniform == [3, 4]
    assert parser.parse_args(["quantize", "m", "-o", "o", "--uniform", "3"]).uniform == 3


@pytest.mark.parametrize("width", [1, 5, 16])
def test_an_unsupported_width_is_rejected_at_the_parser(width: int) -> None:
    """Every kernel is templated over exactly BIT_OPTIONS; 5 bits has no kernel."""
    assert width not in BIT_OPTIONS
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["quantize", "m", "-o", "o", "--uniform", str(width)])
    assert excinfo.value.code == 2


# --------------------------------------------------------------------------
# Bit-map files
# --------------------------------------------------------------------------


@pytest.fixture
def graph():
    return classify_model(Qwen3_5ForCausalLM())


@pytest.fixture
def bit_map(graph) -> BitMap:
    """A map with real variety in it, so concordance has something to measure."""
    bits = {}
    for index, info in enumerate(graph.quantizable()):
        bits[info.name] = (2, 3, 4, 8)[index % 4]
    return BitMap(
        bits=bits,
        violations=(
            FloorViolation(
                name=next(iter(bits)),
                role=ModuleRole.MLP_DOWN,
                floor_bits=3,
                assigned_bits=2,
                num_params=1024,
            ),
        ),
        budget_bits=1e6,
        allocated_bits=1e6,
        denominator=int(4e5),
        target_label="3.00",
    )


def test_a_written_map_reads_back_identically(tmp_path: Path, bit_map: BitMap) -> None:
    written = _shared.write_bit_maps(
        tmp_path,
        {"3.00": bit_map},
        model="fixture",
        stats="stats/",
        allocator="sensitivity",
        group_size=128,
    )
    assert written.name == ALLOCATION_FILENAME
    bits, metadata = _shared.read_bit_map(str(written))
    assert bits == bit_map.bits
    assert metadata["schema"] == ALLOCATION_SCHEMA
    assert metadata["allocator"] == "sensitivity"
    assert metadata["group_size"] == 128
    assert metadata["map_key"] == "3.00"


def test_the_saved_map_carries_which_price_chose_its_widths(
    tmp_path: Path, graph, bit_map: BitMap
) -> None:
    """The pricing split has to be on the file, because the file is what survives.

    The panel runs `dynquant inspect` as a subprocess at the default log level and
    keeps `maps/<arm>.json`. Everything the allocator decided about *how* it priced
    91.5% of this model existed only in an INFO record that nothing captured, so the
    artifact and a map where every module was measured were byte-identical in every
    field a reader could check.
    """
    from dynquant.allocate.knapsack import Pricing

    priced = replace(
        bit_map,
        pricing=Pricing(
            measured_modules=89,
            proxied_modules=44,
            measured_params=720_000_000,
            proxied_params=8_000_000_000,
            scale=2.5e-13,
        ),
    )
    written = _shared.write_bit_maps(
        tmp_path,
        {"3.00": priced},
        model="m",
        stats=None,
        allocator="sensitivity",
        group_size=128,
    )
    payload = json.loads(written.read_text(encoding="utf-8"))["maps"]["3.00"]["pricing"]
    assert payload["proxied_modules"] == 44
    assert payload["proxied_params"] == 8_000_000_000
    assert payload["scale"] == 2.5e-13
    assert payload["proxied_share"] > 0.9, "the share is the number the count hides"


def test_a_map_that_priced_nothing_writes_no_pricing_key(tmp_path: Path, graph) -> None:
    """Absent, not a row of nulls -- a uniform map made no pricing decision at all,
    and nulls would read as an allocator that tried to price and measured nothing."""
    written = _shared.write_bit_maps(
        tmp_path,
        {"8": _shared.uniform_map(graph, 8)},
        model="m",
        stats=None,
        allocator="uniform",
        group_size=128,
    )
    assert "pricing" not in json.loads(written.read_text(encoding="utf-8"))["maps"]["8"]


def test_a_directory_is_accepted_on_both_sides(tmp_path: Path, bit_map: BitMap) -> None:
    """`quantize --map out/` is what anyone types after `inspect --save-map out/`."""
    _shared.write_bit_maps(
        tmp_path, {"3.00": bit_map}, model="m", stats=None, allocator="uniform", group_size=128
    )
    bits, _ = _shared.read_bit_map(str(tmp_path))
    assert bits == bit_map.bits


def test_several_maps_without_a_key_is_an_error_not_a_guess(
    tmp_path: Path, bit_map: BitMap
) -> None:
    """Picking "the first one" would quantize at a budget nobody asked for."""
    _shared.write_bit_maps(
        tmp_path,
        {"3.00": bit_map, "4.00": bit_map},
        model="m",
        stats=None,
        allocator="uniform",
        group_size=128,
    )
    with pytest.raises(DynQuantError, match="--map-key"):
        _shared.read_bit_map(str(tmp_path))
    bits, metadata = _shared.read_bit_map(str(tmp_path), key="4.00")
    assert metadata["map_key"] == "4.00"
    assert bits == bit_map.bits


def test_an_unknown_key_names_the_keys_that_do_exist(tmp_path: Path, bit_map: BitMap) -> None:
    _shared.write_bit_maps(
        tmp_path, {"3.00": bit_map}, model="m", stats=None, allocator="uniform", group_size=128
    )
    with pytest.raises(DynQuantError, match=r"3\.00"):
        _shared.read_bit_map(str(tmp_path), key="2.50")


@pytest.mark.parametrize(
    "payload",
    [
        {"bits": {"a": 4, "b": 3}},
        {"a": 4, "b": 3},
        {"maps": {"only": {"bits": {"a": 4, "b": 3}}}},
    ],
    ids=["single-map", "bare-map", "keyed-map"],
)
def test_the_three_accepted_file_shapes(tmp_path: Path, payload: dict) -> None:
    """The stage scripts predate this format and their outputs are real bit maps."""
    path = tmp_path / "m.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    bits, _ = _shared.read_bit_map(str(path))
    assert bits == {"a": 4, "b": 3}


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"a": 5}, "not one of"),
        ({"a": "4"}, "not an integer"),
        ({"a": True}, "not an integer"),
        ({}, "holds no widths"),
        ({"maps": {}}, "empty"),
    ],
)
def test_a_malformed_map_is_diagnosed(tmp_path: Path, payload: dict, match: str) -> None:
    path = tmp_path / "m.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DynQuantError, match=match):
        _shared.read_bit_map(str(path))


def test_a_missing_map_says_where_it_looked(tmp_path: Path) -> None:
    with pytest.raises(DynQuantError, match="no bit map at"):
        _shared.read_bit_map(str(tmp_path / "absent.json"))


def test_a_map_from_another_checkpoint_fails_before_a_weight_is_touched(graph) -> None:
    """The damage worth catching early: the names that *do* resolve get quantized
    at widths chosen for a different model, and the result runs."""
    model = Qwen3_5ForCausalLM()
    _shared.check_map_covers(model, {info.name: 4 for info in graph.quantizable()})
    with pytest.raises(DynQuantError, match="different checkpoint"):
        _shared.check_map_covers(model, {"model.layers.99.mlp.up_proj": 4})


def test_the_guard_accepts_every_name_the_quantizer_behind_it_would_encode() -> None:
    """A batched expert bank is a parameter, and the pre-flight check used to deny it exists.

    ``get_submodule("...experts.gate_up_proj")`` raises, because the bank keeps its experts
    as bare 3-D parameters rather than as child modules. Every other part of the pipeline
    already knows that -- the tracker writes those names, the graph classifies them, the
    allocator prices them, and ``quantize_model`` resolves them -- but the guard in front
    of all three asked ``get_submodule`` and refused. On LFM2.5-8B-A1B that is 91.5% of the
    parameters, and the refusal arrives phrased as "modules this model does not have",
    which reads as a stale map rather than as a guard that cannot see parameters.

    Asserted as an agreement rather than as two separate behaviours: the guard's whole job
    is to predict what the quantizer will do, so a test that pinned them independently
    would let them drift apart again in the same direction.

    Turns red when: the guard resolves names by itself instead of asking the quantizer.
    """
    import torch
    from torch import nn

    from dynquant.quant.quantizer import quantize_model

    class _Bank(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(4, 256, 128))
            self.down_proj = nn.Parameter(torch.randn(4, 128, 128))

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = _Bank()
            self.o_proj = nn.Linear(128, 128, bias=False)

    model = _Tiny()
    bits = {"experts.gate_up_proj": 4, "experts.down_proj": 3, "o_proj": 4}

    _shared.check_map_covers(model, bits)
    report = quantize_model(model, bits, group_size=128, compute_device=None)
    assert set(report.layers) == set(bits)

    with pytest.raises(DynQuantError, match="different checkpoint"):
        _shared.check_map_covers(model, {"experts.absent_proj": 4})


def test_what_the_packed_runtime_cannot_hold_is_told_apart_from_a_wrong_map() -> None:
    """Four outcomes, four messages, and two of them once wore another's.

    ``get_submodule`` raises ``AttributeError`` both for a name this model does not have
    and for a name that addresses a *tensor*, and both were reported as "not a module of
    this model" -- which sends someone to re-check a map that is correct.

    A batched expert bank was the case that exposed it, and the runtime now holds one
    (:class:`~dynquant.runtime.linear.DynQuantExpertBank`), so that name resolves rather
    than refusing. It is still asserted here, because "resolves" is the distinction the
    other three are being told apart from: a tensor branch that fell back to the
    wrong-map message would now be silently *packing nothing*.

    What stays refused is the same shape one rank down. A bare 2-D parameter -- an MoE
    router's -- has no index at which a module could stand, because its owner passes it
    whole to ``F.linear``, so no grouped path fixes it and the message must not imply one
    will. That is a different sentence from the router reached as a *module*, which is the
    third branch, and from a container, which is the fourth.

    Asserted as a *set of distinctions*: all in one test, because a single message that
    happens to match one of them is exactly the state this replaced.

    Turns red when: any two branches collapse into one, a case the encoder can still reach
    stops naming the mode that does work, or the bank stops resolving.
    """
    import torch
    from torch import nn

    from dynquant.runtime.linear import pack_model

    class _Bank(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(4, 256, 128))

    class _Router(nn.Module):
        """Shaped like `Lfm2MoeTopKRouter`: a bare weight, `F.linear`, no `Linear`."""

        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.randn(8, 128))

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = _Bank()
            self.gate = _Router()
            self.o_proj = nn.Linear(128, 128, bias=False)

    def refuse(name: str) -> str:
        with pytest.raises(DynQuantError) as caught:
            pack_model(_Tiny(), {name: 4}, group_size=128, compute_device=None)
        return str(caught.value)

    from dynquant.runtime.linear import DynQuantExpertBank

    held = _Tiny()
    pack_model(held, {"experts.gate_up_proj": 4}, group_size=128, compute_device=None)
    assert isinstance(held.experts.gate_up_proj, DynQuantExpertBank)

    bare = refuse("gate.weight")
    assert "2-D parameter" in bare
    assert "F.linear" in bare
    assert "encode" in bare
    assert "not a module of this model" not in bare

    router = refuse("gate")
    assert "encode" in router
    # The two it used to be confused with, and the one thing it must say about itself.
    assert "not a module of this model" not in router
    assert "not a batched expert bank" in router

    assert "not a module of this model" in refuse("experts.absent_proj")

    # A container owns nothing to pack, so there is no encoder route to offer either.
    container = refuse("experts")
    assert "owns no weight" in container
    assert "encode" not in container


def test_encoding_a_map_reaches_the_weights_packing_cannot() -> None:
    """The mode the MoE panel runs in, asserted against the mode it replaced.

    ``--map-apply encode`` exists because ``pack`` cannot represent a batched expert bank,
    and the claim that justifies substituting it is that the two run the same encoder at
    the same widths. So this pins both halves: encode changes the weights of a bank that
    packing refuses, and on a plain ``Linear`` -- where both modes work -- the values they
    produce are identical rather than merely similar.

    In bf16, which is the dtype the panel serves and the one the two modes agree in. They
    part company on an fp32 model, where packing keeps fp32 scales (so the packed module
    emits fp32 into an fp32 graph) and encoding falls back to the search's own default --
    a last-bit difference, no served model, and not what this pins.

    Turns red when: encode stops covering the banks, or the two modes drift onto different
    encoders and the panel's substitution stops being neutral.
    """
    import torch
    from torch import nn

    from dynquant.quant.quantizer import quantize_model
    from dynquant.runtime.linear import pack_model

    class _Bank(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(4, 256, 128, dtype=torch.bfloat16))

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = _Bank()
            self.o_proj = nn.Linear(128, 128, bias=False, dtype=torch.bfloat16)

    torch.manual_seed(0)
    model = _Tiny()
    before = model.experts.gate_up_proj.detach().clone()
    linear_before = model.o_proj.weight.detach().clone()

    quantize_model(
        model,
        {"experts.gate_up_proj": 4, "o_proj": 4},
        group_size=128,
        compute_device=None,
    )
    assert not torch.equal(model.experts.gate_up_proj, before), "the bank was left alone"

    packed = _Tiny()
    packed.o_proj.weight.data.copy_(linear_before)
    pack_model(packed, {"o_proj": 4}, group_size=128, compute_device=None)
    assert torch.equal(
        packed.o_proj.weight_qt.dequantize().to(model.o_proj.weight.dtype), model.o_proj.weight
    ), "pack and encode disagree on a module both can do"


def test_the_chosen_apply_mode_is_the_one_that_runs_and_the_one_recorded(
    tmp_path: Path,
) -> None:
    """A silent fallback here is a panel that dies four hours in, or one that lies.

    ``--map-apply`` has two failure shapes and neither is loud. If the dispatch ignored
    the flag, the MoE arms would take the packed path they were moved off and refuse
    mid-run. If the record omitted which mode ran, two arms scored under different
    applications would pair cleanly and their difference would be read as quantization.
    So the flag has to reach ``quantize_model`` *and* reach the record.

    The parser's ``choices`` is pinned in the same test because without it a typo is not
    an error -- ``--map-apply pak`` parses, misses the equality, and quietly packs.

    Since the packed runtime holds a bank, the two modes are told apart by *what they
    leave behind* rather than by one of them failing: encode writes reconstructed values
    back over the parameter, pack replaces it with a module. Same widths, different
    artifacts, and only one of them is what an in-memory accuracy panel scores.

    Turns red when: the dispatch stops reading the flag, the record stops carrying it, or
    an unknown mode is accepted instead of refused.
    """
    import argparse
    import json

    import torch
    from torch import nn

    from dynquant.commands.evaluate import _apply_map

    class _Bank(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.randn(4, 256, 128, dtype=torch.bfloat16))

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = _Bank()

    path = tmp_path / "maps.json"
    path.write_text(
        json.dumps({"maps": {"k": {"bits": {"experts.gate_up_proj": 4}, "group_size": 128}}}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        map=str(path), map_key="k", group_size=128, quiet=True, compute_device=None
    )

    model = _Tiny()
    before = model.experts.gate_up_proj.detach().clone()
    recorded = _apply_map(model, argparse.Namespace(**vars(args), map_apply="encode"))
    assert recorded["apply"] == "encode"
    assert not torch.equal(model.experts.gate_up_proj, before), "the flag did not reach the weights"

    from dynquant.runtime.linear import DynQuantExpertBank

    packed = _Tiny()
    assert _apply_map(packed, argparse.Namespace(**vars(args), map_apply="pack"))["apply"] == "pack"
    assert isinstance(packed.experts.gate_up_proj, DynQuantExpertBank)
    assert "gate_up_proj" not in packed.experts._parameters

    with pytest.raises(SystemExit):
        build_parser().parse_args(["eval", "m", "--task", "gsm8k", "--map-apply", "pak"])


# --------------------------------------------------------------------------
# Allocation seam
# --------------------------------------------------------------------------


def test_allocating_without_stats_points_at_the_honest_alternative() -> None:
    """The requirement is real, but it belongs to ``allocate``, not to ``build_inputs``.

    Scores are what the knapsack orders modules by, so a budgeted map without them is
    not a worse map, it is an arbitrary one -- hence the raise. It lives one level down
    from where it used to because the same inputs also feed ``uniform_map``, which needs
    no ordering at all; see the test below for the failure that cost.
    """
    inputs = _shared.build_inputs(Qwen3_5ForCausalLM(), stats=None, verbose=False)
    with pytest.raises(DynQuantError, match=r"--uniform"):
        _shared.allocate(inputs, target_bits=3.0)


def test_a_uniform_arm_is_reachable_without_ever_fine_tuning(graph) -> None:
    """``inspect --uniform`` has to run on a checkpoint that has no signals at all.

    The guard this pins used to sit in ``build_inputs``, which every ``inspect`` run
    goes through -- so asking for the RTN control arm died with an error recommending
    the exact flag that had been passed, and the baseline was reachable only by first
    collecting the fine-tuning signals it exists to be compared against. Caught on a
    real checkpoint mid-campaign; red again if the requirement drifts back up.
    """
    inputs = _shared.build_inputs(Qwen3_5ForCausalLM(), stats=None, verbose=False)
    assert inputs.scores == {}
    assert inputs.score_report is None
    assert {info.name for info in inputs.graph.quantizable()}  # classification still ran

    uniform = _shared.uniform_map(inputs.graph, 4, policy=inputs.policy)
    assert set(uniform.bits) == {info.name for info in graph.quantizable()}


def test_a_uniform_map_does_not_put_structural_roles_at_the_uniform_width(graph) -> None:
    """A router at 2 bits is not a low-precision model, it is a broken one, and a
    control arm containing one measures the wrong thing."""
    policy_floor_roles = [
        info for info in graph.quantizable() if info.role in ModuleRole.__members__.values()
    ]
    assert policy_floor_roles  # the fixture has quantizable modules at all

    two_bit = _shared.uniform_map(graph, 2)
    assert set(two_bit.bits) == {info.name for info in graph.quantizable()}
    for info in graph.quantizable():
        assigned = two_bit.bits[info.name]
        if assigned != 2:
            assert assigned > 2, f"{info.name} landed below the uniform width"
    assert two_bit.target_label == "uniform 2 bit"
    # Accounting is the same accounting the allocated arm uses, or the two are
    # being compared at different budgets.
    assert two_bit.average_bits > 2.0
    assert two_bit.nbytes > 0


def test_a_uniform_map_at_eight_bits_is_ordered_above_one_at_two(graph) -> None:
    assert _shared.uniform_map(graph, 8).nbytes > _shared.uniform_map(graph, 2).nbytes


def test_the_control_arm_counts_the_floors_it_breaches(graph) -> None:
    """A control that reports no violations is not a control that breaches none.

    The structural exemption covers routers and the like; it does not cover the
    embedding or the LM head, so a uniform 3-bit map puts both under their floors. This
    only surfaced next to a real allocated arm: at the same byte count the allocated map
    reported 79 breaches and the uniform one reported zero, which reads as the method
    being reckless when in fact both breach and only one was counting.
    """
    from dynquant.allocate.policy import AllocationPolicy

    policy = AllocationPolicy(group_size=128)
    narrow = _shared.uniform_map(graph, 2, policy=policy)

    breached = {v.name for v in narrow.violations}
    expected = {
        info.name
        for info in graph.quantizable()
        if not policy.is_structural(info.role, info.tied_roles)
        and policy.floor_for(info.role, info.tied_roles) > 2
    }
    assert breached == expected
    assert breached, "the fixture has no floor above 2 bits, so this proves nothing"
    for violation in narrow.violations:
        assert violation.assigned_bits < violation.floor_bits
        assert narrow.bits[violation.name] == violation.assigned_bits

    # A width above every floor breaches nothing -- the count tracks the map, not the
    # fact that a uniform map was asked for.
    assert _shared.uniform_map(graph, 8, policy=policy).violations == ()


@pytest.mark.parametrize(
    ("pairs", "expected"),
    [
        (None, None),
        ([], None),
        (["model.layers.0.mlp.gate=moe_router"], {"model.layers.0.mlp.gate": "moe_router"}),
    ],
)
def test_role_overrides_parse(pairs, expected) -> None:
    assert _shared.parse_overrides(pairs) == expected


@pytest.mark.parametrize("bad", ["nameonly", "=role", ""])
def test_a_malformed_role_override_is_rejected(bad: str) -> None:
    with pytest.raises(DynQuantError, match="NAME=ROLE"):
        _shared.parse_overrides([bad])


# --------------------------------------------------------------------------
# inspect
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pairs", "expected"),
    [
        ([(1.0, 2), (2.0, 4)], (1, 0)),
        ([(2.0, 2), (1.0, 4)], (0, 1)),
        ([(1.0, 4), (2.0, 4)], (0, 0)),  # equal widths carry no ordering
        ([(1.0, 2), (1.0, 4)], (0, 0)),  # equal scores carry no ordering
        ([], (0, 0)),
    ],
)
def test_concordance_counts_only_informative_pairs(pairs, expected) -> None:
    """Ties would dilute the ratio toward 0.5, which is the value meaning "broken"."""
    assert inspect.concordance(pairs) == expected


def test_a_perfectly_ordered_allocation_scores_one(graph, bit_map: BitMap) -> None:
    scores = {name: float(width) for name, width in bit_map.bits.items()}
    report = inspect.inspect_allocation(graph, scores, bit_map)
    assert report["concordance"]["value"] == pytest.approx(1.0)

    inverted = {name: -float(width) for name, width in bit_map.bits.items()}
    assert inspect.inspect_allocation(graph, inverted, bit_map)["concordance"]["value"] == 0.0


def test_the_inspection_reports_violations_and_the_narrowest_modules(
    graph, bit_map: BitMap
) -> None:
    scores = dict.fromkeys(bit_map.bits, 0.5)
    report = inspect.inspect_allocation(graph, scores, bit_map, narrowest=3)
    assert len(report["violations"]) == 1
    assert report["violations"][0]["assigned_bits"] == 2
    assert len(report["narrowest"]) == 3
    assert all(entry["bits"] == 2 for entry in report["narrowest"])
    assert set(report["widths"]) == {"2", "3", "4", "8"}
    assert inspect.render(report)  # renders without raising on every field


def _sensitivity(graph, per_param: dict[str, float]):
    """A table whose 2b->8b gain per parameter is exactly ``per_param``."""
    from dynquant.score.sensitivity import SensitivityTable

    values = {}
    for info in graph.quantizable():
        total = per_param.get(info.name, 0.0) * info.num_params
        values[info.name] = {2: total, 3: total * 0.5, 4: total * 0.25, 8: 0.0}
    return SensitivityTable(values=values, group_size=128)


def test_concordance_follows_sensitivity_not_the_score_when_moments_drove_it(
    graph, bit_map: BitMap
) -> None:
    """The defect this guards: a bit map allocated from measured dL, checked
    against the rank score, reports the agreement of a signal the allocator never
    read -- under a heading that says otherwise."""
    agreeing = _sensitivity(graph, {name: float(w) for name, w in bit_map.bits.items()})
    inverted = dict.fromkeys(bit_map.bits, 1.0)  # a score that orders nothing

    report = inspect.inspect_allocation(graph, inverted, bit_map, sensitivity=agreeing)
    assert report["concordance"]["value"] == pytest.approx(1.0)
    assert "dL" in report["quantity"]
    assert "dL" in inspect.render(report)

    # ...and the mirror image: a score that agrees perfectly must not rescue an
    # allocation the driving quantity contradicts.
    scores = {name: float(w) for name, w in bit_map.bits.items()}
    opposed = _sensitivity(graph, {name: -float(w) for name, w in bit_map.bits.items()})
    assert inspect.inspect_allocation(graph, scores, bit_map, sensitivity=opposed)["concordance"][
        "value"
    ] == pytest.approx(0.0)


def test_the_sensitivity_ordering_divides_out_tensor_size(graph, bit_map: BitMap) -> None:
    """An undivided dL grows with the tensor, so concordance against it would read
    as healthy in exactly the case the diagnostic exists to catch: size, not the
    signal, choosing the widths."""
    from dynquant.score.sensitivity import SensitivityTable

    size = {info.name: float(info.num_params) for info in graph.quantizable()}
    values = {name: {2: n, 3: n * 0.5, 4: n * 0.25, 8: 0.0} for name, n in size.items()}

    ordering, quantity, fallback = inspect.ordering_values(
        graph, {}, SensitivityTable(values=values, group_size=128)
    )
    assert not fallback
    assert "per parameter" in quantity
    # Every module has dL proportional to its own size, so per-parameter they are
    # all equal and none of them is ranked above another by being large.
    assert len({round(v, 9) for v in ordering.values()}) == 1


def test_a_module_with_no_measured_sensitivity_is_left_out_not_scored_zero(
    graph, bit_map: BitMap
) -> None:
    """Zero is an ordering claim. The reason the value is missing is that nothing
    was measured, and entering it as the smallest value would rank it last."""
    table = _sensitivity(graph, {name: float(w) for name, w in bit_map.bits.items()})
    dropped = next(iter(table.values))
    del table.values[dropped]

    ordering, _, fallback = inspect.ordering_values(graph, {}, table)
    assert fallback == [dropped]
    assert dropped not in ordering

    report = inspect.inspect_allocation(graph, {}, bit_map, sensitivity=table)
    assert report["unordered"] == [dropped]
    assert dropped in inspect.render(report)

    # And nowhere in the rendered text does it appear as a number. `narrowest`
    # printed `ordering.get(name, 0.0)`, so an unmeasured module was shown with a
    # score of 0 next to peers whose scores were real -- which is the same claim
    # the returned ordering is careful not to make, made two lines later in prose.
    assert report["unordered_params"] == graph[dropped].num_params
    entries = {e["name"]: e["score"] for e in report["narrowest"]}
    assert entries.get(dropped, "absent") in (None, "absent")
    if dropped in entries:
        assert "unmeasured" in inspect.render(report)


def test_a_width_group_nothing_measured_is_not_reported_as_scoring_zero(
    graph, bit_map: BitMap
) -> None:
    """min = median = max = 0 read as "measured and worthless", not as "not measured".

    The two are opposite conclusions about the same width group and the panel's
    3-bit arm hits the second one: on a batched-expert MoE every module at the
    narrow widths can be an expert bank, and every expert bank is unmeasurable by
    construction. Padding the group with zeros made the signal look like it had
    ranked them last.
    """
    table = _sensitivity(graph, {name: float(w) for name, w in bit_map.bits.items()})
    two_bit = [name for name, width in bit_map.bits.items() if width == 2]
    assert two_bit
    for name in two_bit:
        del table.values[name]

    report = inspect.inspect_allocation(graph, {}, bit_map, sensitivity=table)
    assert report["widths"]["2"]["score_median"] is None
    assert report["widths"]["2"]["measured"] == 0
    assert report["widths"]["2"]["modules"] == len(two_bit)
    assert "none of them measured" in inspect.render(report)


def test_without_moments_the_ordering_is_the_score_unchanged(graph, bit_map: BitMap) -> None:
    """The two definitions have to coincide where sensitivity is absent, or the
    rank-product path silently changed meaning when this was added."""
    scores = {name: float(width) for name, width in bit_map.bits.items()}
    ordering, quantity, fallback = inspect.ordering_values(graph, scores, None)
    assert ordering == scores
    assert quantity == "score"
    assert not fallback


def test_an_allocation_where_every_module_got_the_same_width_says_so(graph) -> None:
    uniform = _shared.uniform_map(graph, 8)
    report = inspect.inspect_allocation(graph, dict.fromkeys(uniform.bits, 1.0), uniform)
    assert report["concordance"]["value"] is None
    assert "nothing to correlate" in inspect.render(report)


# --------------------------------------------------------------------------
# quantize
# --------------------------------------------------------------------------


def test_the_map_key_records_the_budget_that_was_asked_for(bit_map: BitMap) -> None:
    parser = build_parser()
    args = parser.parse_args(["quantize", "m", "-o", "o", "--target", "3.25"])
    assert quantize._map_key(args, bit_map) == "3.25"
    args = parser.parse_args(["quantize", "m", "-o", "o", "--uniform", "4"])
    assert quantize._map_key(args, bit_map) == "uniform-4"
    args = parser.parse_args(["quantize", "m", "-o", "o", "--target-size", "6.5GiB"])
    assert quantize._map_key(args, bit_map) == bit_map.target_label


def test_applying_a_map_at_the_wrong_group_size_is_refused(tmp_path: Path, bit_map: BitMap) -> None:
    """The widths were priced against that grouping. Applying them at another
    changes the stored size and the error of every tensor."""
    _shared.write_bit_maps(
        tmp_path, {"3.00": bit_map}, model="m", stats=None, allocator="uniform", group_size=64
    )
    args = build_parser().parse_args(
        ["quantize", "m", "-o", "o", "--map", str(tmp_path), "--group-size", "128"]
    )
    with pytest.raises(DynQuantError, match="group_size=64"):
        quantize._resolve_widths(Qwen3_5ForCausalLM(), args)


def test_no_budget_at_all_is_an_error(tmp_path: Path) -> None:
    args = build_parser().parse_args(["quantize", "m", "-o", str(tmp_path)])
    with pytest.raises(DynQuantError, match="exactly one of"):
        quantize._resolve_widths(Qwen3_5ForCausalLM(), args)


def test_two_budgets_at_once_is_an_error() -> None:
    args = build_parser().parse_args(
        ["quantize", "m", "-o", "o", "--target", "3.0", "--target-ratio", "0.2"]
    )
    with pytest.raises(DynQuantError, match="exactly one of"):
        quantize._resolve_widths(Qwen3_5ForCausalLM(), args)


def test_a_map_read_from_a_file_is_applied_verbatim(tmp_path: Path, bit_map: BitMap) -> None:
    """The point of the seam: no second allocation between review and application."""
    _shared.write_bit_maps(
        tmp_path, {"3.00": bit_map}, model="m", stats=None, allocator="sensitivity", group_size=128
    )
    args = build_parser().parse_args(["quantize", "m", "-o", "o", "--map", str(tmp_path)])
    resolved = quantize._resolve_widths(Qwen3_5ForCausalLM(), args)
    assert resolved.bits == bit_map.bits
    assert resolved.bit_map is None  # the file's accounting is the file's, not recomputed
    assert resolved.allocator == "sensitivity"
    assert "3.00" in resolved.source


def test_the_packed_size_of_a_map_read_from_a_file_is_still_reported(
    tmp_path: Path, bit_map: BitMap
) -> None:
    """Not recomputing the file's accounting is not the same as not printing it.

    The directory ``quantize`` writes is compute-dtype sized, and the one number
    the research supplement got wrong was reading that as the quantized size. So
    the packed figure has to appear on the ``--map`` path too -- as the file's,
    which is what ``provenance`` says."""
    _shared.write_bit_maps(
        tmp_path, {"3.00": bit_map}, model="m", stats=None, allocator="sensitivity", group_size=128
    )
    args = build_parser().parse_args(["quantize", "m", "-o", "o", "--map", str(tmp_path)])
    stored = quantize._resolve_widths(Qwen3_5ForCausalLM(), args).stored
    assert stored is not None
    assert stored.nbytes == bit_map.nbytes
    assert stored.average_bits == pytest.approx(bit_map.average_bits)
    assert "as recorded" in stored.provenance


def test_a_hand_written_map_carries_no_packed_size_and_none_is_invented(tmp_path: Path) -> None:
    """A bare ``{name: bits}`` map has no accounting, and computing one would need
    the graph its writer used. Reporting ``None`` is the honest answer; reporting
    the directory size in its place is the defect."""
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"model.layers.0.self_attn.q_proj": 3}), encoding="utf-8")
    args = build_parser().parse_args(["quantize", "m", "-o", "o", "--map", str(path)])
    resolved = quantize._resolve_widths(Qwen3_5ForCausalLM(), args)
    assert resolved.stored is None
    assert resolved.bit_map is None


def test_an_allocated_map_reports_its_own_accounting_not_a_files(bit_map: BitMap) -> None:
    stored = quantize._Stored.from_bit_map(bit_map)
    assert stored.nbytes == bit_map.nbytes
    assert stored.provenance == "this allocation"
    assert stored.violations == bit_map.violations


def test_floors_breached_by_a_map_file_are_reported_when_it_is_applied(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The allocating run that wrote the map may be one nobody still has the output
    of, and a breached floor is the one thing in a bit map that is a risk to the
    model rather than an accounting detail."""
    stored = quantize._Stored(
        nbytes=1,
        average_bits=3.0,
        provenance="maps/ [3.25], as recorded",
        violations=tuple(
            {
                "name": f"model.layers.{i}.mlp.gate_proj",
                "role": "mlp.gate",
                "floor_bits": 4,
                "assigned_bits": 2,
                "num_params": i * 1_000_000,
            }
            for i in range(8)
        ),
    )
    quantize._print_violations(stored)
    out = capsys.readouterr().out
    assert "8 floors breached" in out  # the count is all of them ...
    assert out.count("mlp.gate_proj") == 5  # ... the listing is the worst five
    assert "model.layers.7.mlp.gate_proj (mlp.gate) 4b -> 2b, 7.0M params" in out
    assert "model.layers.0" not in out  # smallest, so not among the worst


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------


def test_every_task_has_defaults_and_a_stated_chance_floor() -> None:
    """A destroyed 5-way multiple choice returns to 20%, not to zero, and a table
    without the floor makes a collapsed arm look mildly damaged."""
    for spec in evaluate.TASKS.values():
        assert 0.0 <= spec.chance < 1.0
        assert spec.shots >= 0
        assert spec.max_new_tokens > 0
        assert spec.batch_size > 0
    assert evaluate.TASKS["casehold"].chance == pytest.approx(0.2)
    assert evaluate.TASKS["banking77"].chance == pytest.approx(1 / 77)


def test_every_registered_task_is_reachable_from_the_command_line() -> None:
    """``--task`` lists its choices literally, so adding a task to ``TASKS`` and
    forgetting the parser produces a registry entry no user can select -- and the
    reverse produces a choice that raises ``KeyError`` on the way in."""
    subparsers = next(
        a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction)
    )
    action = next(a for a in subparsers.choices["eval"]._actions if a.dest == "task")
    assert set(action.choices or ()) == set(evaluate.TASKS)


def test_task_dispatch_resolves_by_naming_convention() -> None:
    """``_TaskSpec`` imports ``dynquant.eval.<key>`` and calls ``load_<key>`` /
    ``evaluate_<key>``. That convention is implicit in a getattr, so a module named
    off-pattern would fail at the end of a GPU run rather than at import."""
    from importlib import import_module

    for key in evaluate.TASKS:
        module = import_module(f"dynquant.eval.{key}")
        assert callable(getattr(module, f"load_{key}"))
        assert callable(getattr(module, f"evaluate_{key}"))


def _result_class(key: str) -> type:
    """The dataclass ``evaluate_<key>`` returns, resolved without importing datasets.

    ``from __future__ import annotations`` leaves the return annotation a string, and
    ``get_type_hints`` would try to resolve every *other* annotation too -- several of
    which live under ``TYPE_CHECKING`` and do not exist at runtime. So only the return
    name is looked up, in the module that declares the function.
    """
    from importlib import import_module

    module = import_module(f"dynquant.eval.{key}")
    return getattr(module, getattr(module, f"evaluate_{key}").__annotations__["return"])  # type: ignore[no-any-return]


def _eval_args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "task": "gsm8k",
        "limit": None,
        "split": None,
        "shots": None,
        "shot_split": None,
        "shot_seed": 0,
        "on_unverifiable": "raise",
        "allow_execution": False,
        "prompt_style": "auto",
        "exec_timeout": None,
        "exec_memory_mb": None,
        "model": "m",
        "map": None,
        "device": "cuda",
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "backend": "transformers",
        "quantization": None,
        "gpu_memory_utilization": 0.85,
        "max_model_len": None,
        "tensor_parallel_size": 1,
        "enforce_eager": False,
    }
    return argparse.Namespace(**{**defaults, **overrides})


def test_every_task_is_sent_exactly_the_arguments_its_scorer_takes() -> None:
    """The registry decides what to pass by declared capability, and the scorer
    decides what it accepts. Both directions matter and they fail differently.

    Sending an argument the function does not take raises at the end of a GPU run.
    *Not* sending one it does takes the default instead -- MBPP flagged as taking no
    shots would score zero-shot and report the number as three-shot, which is a
    silently wrong row in the results table rather than a crash.

    Both the declaration and the call are checked. Agreeing on the flags is not the
    same as acting on them: a capability the registry declares but never turns into a
    keyword argument leaves the scorer on its own default, which is exactly the
    silent-wrong-row failure, and the signature check alone cannot see it.
    """
    from importlib import import_module
    from inspect import signature

    for key, spec in evaluate.TASKS.items():
        scorer = getattr(import_module(f"dynquant.eval.{key}"), f"evaluate_{key}")
        accepted = set(signature(scorer).parameters)
        assert {"label", "config", "progress", "keep_predictions"} <= accepted
        assert ("shots" in accepted) == spec.takes_shots, key
        assert ("on_unverifiable" in accepted) == spec.unverifiable, key
        assert ("style" in accepted) == spec.takes_style, key
        for name in ("allow_execution", "timeout", "memory_mb"):
            assert (name in accepted) == spec.executes_code, f"{key}.{name}"

        sent = evaluate._task_kwargs(spec, _eval_args(allow_execution=True), ["exemplar"])
        assert set(sent) <= accepted, f"{key} is sent {set(sent) - accepted}"
        assert ("shots" in sent) == spec.takes_shots, key
        assert ("on_unverifiable" in sent) == spec.unverifiable, key
        # `timeout`/`memory_mb` are deliberately absent unless asked for, so only the
        # two the task cannot run without are required here.
        # `style` is checked against `takes_style`, not `executes_code`. The two agreed
        # on every task in the registry until text2sql, which takes a framing argument
        # and runs nothing -- so the version of this assertion that read `style` off the
        # sandbox flag was passing on a coincidence, and the first task to break the
        # coincidence is the one it would have mis-scored.
        assert ("style" in sent) == spec.takes_style, key
        assert ("allow_execution" in sent) == spec.executes_code, key


def test_the_field_counting_unscorable_generations_exists_on_every_result() -> None:
    """``unscored`` names a field per task because the tasks spell it differently.
    A name that is merely plausible would read as ``0`` through ``getattr`` with a
    default -- reporting a model that emitted nothing as a model that answered wrong,
    which is the one distinction this column exists to make."""
    import dataclasses

    for key, spec in evaluate.TASKS.items():
        names = {field.name for field in dataclasses.fields(_result_class(key))}
        assert spec.unscored in names, f"{key} has no field named {spec.unscored!r}"


def test_a_task_promising_extra_metrics_can_actually_produce_them() -> None:
    """``detail=True`` puts the task's own metrics in the record. IFEval reports four
    and the code tasks separate a timeout from a wrong answer; a task that claims the
    block without an ``as_dict`` would fail after the GPU time, not before it."""
    for key, spec in evaluate.TASKS.items():
        if spec.detail:
            assert callable(getattr(_result_class(key), "as_dict", None)), key


def test_a_task_with_one_split_is_never_asked_which_split() -> None:
    """HumanEval is 164 problems in a single set and ``load_humaneval`` takes no split
    argument. The parser used to default ``--split`` to ``"test"``, which would have
    reached the loader as a positional it has no parameter for."""
    assert evaluate.TASKS["humaneval"].split is None
    assert evaluate.TASKS["ifeval"].split == "train"

    with pytest.raises(DynQuantError, match="single set"):
        evaluate._resolve_splits(evaluate.TASKS["humaneval"], _eval_args(split="test"))


@pytest.mark.parametrize(
    "flag,kwargs", [("--shots", {"shots": 5}), ("--shot-split", {"shot_split": "train"})]
)
def test_a_few_shot_option_on_a_zero_shot_task_is_refused(flag: str, kwargs: dict) -> None:
    """Not ignored. A run that quietly dropped ``--shots 5`` reports a zero-shot score
    under a five-shot label, and nothing downstream can tell."""
    with pytest.raises(DynQuantError, match=flag):
        evaluate._resolve_splits(evaluate.TASKS["ifeval"], _eval_args(**kwargs))


def test_a_task_whose_only_split_is_named_train_is_read_from_it() -> None:
    """IFEval's 541 prompts ship under `train` and there is no other split. The old
    parser default of `"test"` would have asked for one that does not exist -- and the
    three tasks whose split *is* called `test` cannot detect that, because for them the
    parser's default and the task's own answer coincide."""
    assert evaluate._resolve_splits(evaluate.TASKS["ifeval"], _eval_args()) == ("train", None, 0)
    assert evaluate._resolve_splits(evaluate.TASKS["gsm8k"], _eval_args()) == ("test", "train", 5)


def test_mbpp_draws_its_exemplars_from_a_split_it_does_not_score() -> None:
    """The `prompt` split is MBPP's own exemplar set. Drawing them from `test` would
    put the answer to a graded problem in its own prompt."""
    split, shot_split, shots = evaluate._resolve_splits(evaluate.TASKS["mbpp"], _eval_args())
    assert (split, shot_split, shots) == ("test", "prompt", 3)


def test_a_code_task_refuses_to_start_without_execution_opted_in() -> None:
    """Scoring HumanEval means running code the model wrote. The refusal happens
    before the model loads, not after the generations exist."""
    with pytest.raises(DynQuantError, match="--allow-execution"):
        evaluate._task_kwargs(evaluate.TASKS["humaneval"], _eval_args(), [])

    kwargs = evaluate._task_kwargs(
        evaluate.TASKS["humaneval"], _eval_args(allow_execution=True), []
    )
    assert kwargs["allow_execution"] is True
    # Absent, not defaulted: the sandbox budget lives in the task module, and a second
    # copy in the CLI is a second number to forget to change.
    assert "timeout" not in kwargs and "memory_mb" not in kwargs


def test_a_chat_templated_task_does_not_prepend_a_second_bos() -> None:
    """IFEval's prompt is a chat template, which carries its own BOS. Leaving the
    tokenizer's on gives Llama-3 and Gemma-3 two -- no error, and damage the same size
    as the effect being measured."""
    assert evaluate.TASKS["ifeval"].add_special_tokens is False
    assert evaluate.TASKS["gsm8k"].add_special_tokens is True


def _engine_kwargs(monkeypatch, **arg_overrides) -> dict:
    """Build the vLLM runtime with the engine stubbed, and return what it was sent."""
    from dynquant.eval.backends import VllmBackend
    from dynquant.eval.harness import EvalConfig

    captured: dict = {}

    def _fake(cls, model_path, **kwargs):
        captured.update(kwargs, model_path=model_path)
        return "engine"

    monkeypatch.setattr(VllmBackend, "from_pretrained", classmethod(_fake))
    config = EvalConfig(max_new_tokens=1024, batch_size=16, max_prompt_tokens=2048)
    model, packed = evaluate._load_runtime(_eval_args(backend="vllm", **arg_overrides), config)
    assert model == "engine"
    assert packed is None, "an engine loads a checkpoint; nothing was packed in memory"
    return captured


def test_the_engine_context_covers_the_prompt_and_the_generation(monkeypatch) -> None:
    """vLLM refuses or truncates a prompt longer than `max_model_len`, and the number
    that comes back from a truncated prompt is a real number for a different question.
    The engine is therefore sized from the harness's own budgets rather than left to
    the checkpoint's advertised context, which on a long-context model is 128k of KV
    cache reserved to run a 2k prompt."""
    captured = _engine_kwargs(monkeypatch)
    assert captured["max_model_len"] == 2048 + 1024

    override = _engine_kwargs(monkeypatch, max_model_len=8192)
    assert override["max_model_len"] == 8192


def test_engine_options_left_unset_are_not_sent_at_all(monkeypatch) -> None:
    """`quantization=None` is not the same request as omitting it: vLLM reads the
    method out of the checkpoint's own config.json, which is how the GPTQ and AWQ
    baselines are meant to load."""
    captured = _engine_kwargs(monkeypatch)
    assert "quantization" not in captured
    assert "tensor_parallel_size" not in captured

    asked = _engine_kwargs(monkeypatch, quantization="dynquant", tensor_parallel_size=2)
    assert asked["quantization"] == "dynquant"
    assert asked["tensor_parallel_size"] == 2


def test_a_bit_map_is_not_applied_to_an_engine_that_never_built_a_model() -> None:
    """`--map` swaps modules on a live `nn.Module`. The vLLM path has none, so the map
    would be accepted and silently do nothing -- an arm reported as quantized that ran
    at full precision, which is the most flattering way this could fail."""
    with pytest.raises(DynQuantError, match="dynquant quantize"):
        evaluate._load_runtime(_eval_args(backend="vllm", map="maps/3.25.json"), None)


def test_the_record_pairs_on_exactly_the_settings_that_change_the_score() -> None:
    """Pinned literally, not derived, because every other test here walks
    ``PAIRING_FIELDS`` -- so dropping an entry deletes its own test cases and the suite
    stays green while the guard quietly stops checking that setting.

    ``backend`` earns its place even though G4 measures the two runtimes as equivalent
    in *score*: equal totals are not identical per-item outcomes, and McNemar reads
    exactly the items the arms disagree on.
    """
    assert evaluate.PAIRING_FIELDS == ("task", "backend", "split", "shots", "shot_seed", "limit")


def test_the_record_carries_every_setting_the_pairing_guard_will_read() -> None:
    """The other direction: a field named in the contract but never written into the
    record makes every comparison against a record this same code produced raise."""
    args = _eval_args(task="mbpp", backend="vllm", shot_seed=7, limit=200)
    values = evaluate._pairing(args, split="test", n_shots=3)

    assert set(values) == set(evaluate.PAIRING_FIELDS)
    assert values == {
        "task": "mbpp",
        "backend": "vllm",
        "split": "test",
        "shots": 3,
        "shot_seed": 7,
        "limit": 200,
    }


def test_a_contract_the_record_cannot_satisfy_is_a_loud_bug(monkeypatch) -> None:
    """Adding a field to ``PAIRING_FIELDS`` without recording it would otherwise be
    found by a user, at the end of a run, holding a record that cannot be compared."""
    monkeypatch.setattr(evaluate, "PAIRING_FIELDS", (*evaluate.PAIRING_FIELDS, "dtype"))
    with pytest.raises(DynQuantError, match="PAIRING_FIELDS names"):
        evaluate._pairing(_eval_args(), split="test", n_shots=5)


@pytest.mark.parametrize("field", evaluate.PAIRING_FIELDS)
def test_comparing_across_settings_is_refused(tmp_path: Path, field: str) -> None:
    """A harness difference reported as a quantization effect is the failure this
    exists to prevent.

    ``backend`` belongs in this list even though G4 measures the two runtimes as
    equivalent in *score*. Equal totals are not identical per-item outcomes, and
    McNemar reads exactly the items the two arms disagree on -- so a cross-backend
    pair puts engine disagreement into the one cell being counted.
    """
    record = {
        "task": "gsm8k",
        "backend": "transformers",
        "split": "test",
        "shots": 5,
        "shot_seed": 0,
        "limit": None,
        "label": "b",
        "hits": [True, False],
    }
    other = dict(record, label="a")
    other[field] = "changed" if isinstance(other[field], str) else 999
    path = tmp_path / "other.json"
    path.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(DynQuantError, match=field):
        evaluate._compare(record, str(path))


@pytest.mark.parametrize("field", evaluate.PAIRING_FIELDS)
def test_a_record_missing_a_field_the_guard_reads_fails_loudly(tmp_path: Path, field: str) -> None:
    """The guard compares ``other.get(f)`` against ``record.get(f)``. Drop the field
    from the record `run()` writes and both sides are ``None``, so every genuine
    mismatch is waved through and the McNemar table is built from two different
    evaluations. The guard has to notice it went blind."""
    record = {
        "task": "gsm8k",
        "backend": "transformers",
        "split": "test",
        "shots": 5,
        "shot_seed": 0,
        "limit": None,
        "label": "b",
        "hits": [True, False],
    }
    # The stored record is complete and matches, so the only thing that can fire is the
    # missing-field check -- otherwise a field earlier in the tuple raises first and the
    # test passes for the wrong reason.
    other = dict(record, label="a")
    del record[field]
    path = tmp_path / "other.json"
    path.write_text(json.dumps(other), encoding="utf-8")
    with pytest.raises(DynQuantError, match=f"no {field!r} field"):
        evaluate._compare(record, str(path))


def test_comparing_different_problem_counts_is_refused(tmp_path: Path) -> None:
    record = {
        "task": "gsm8k",
        "backend": "transformers",
        "split": "test",
        "shots": 5,
        "shot_seed": 0,
        "limit": None,
        "label": "b",
        # A real record always carries this, and the budget is part of the
        # comparability contract -- a fixture without it is a record `dynquant eval`
        # could not have written.
        "decode": {"max_new_tokens": 256, "batch_size": 8, "greedy": True},
        "hits": [True, False, True],
    }
    path = tmp_path / "other.json"
    path.write_text(json.dumps(dict(record, hits=[True, False])), encoding="utf-8")
    with pytest.raises(DynQuantError, match="Not the same problem set"):
        evaluate._compare(record, str(path))


def test_a_matched_pair_compares(tmp_path: Path, capsys) -> None:
    record = {
        "task": "gsm8k",
        "backend": "transformers",
        "split": "test",
        "shots": 5,
        "shot_seed": 0,
        "limit": None,
        "label": "quantized",
        # A real record always carries this, and the budget is part of the
        # comparability contract -- a fixture without it is a record `dynquant eval`
        # could not have written.
        "decode": {"max_new_tokens": 256, "batch_size": 8, "greedy": True},
        "hits": [True, True, False, False],
    }
    path = tmp_path / "other.json"
    path.write_text(
        json.dumps(dict(record, label="fp16", hits=[True, True, True, False])), encoding="utf-8"
    )
    comparison = evaluate._compare(record, str(path))
    assert comparison
    capsys.readouterr()


def _styled(style: str | None) -> dict:
    """A text2sql record, with or without the style its task resolved."""
    record = {
        "task": "text2sql",
        "backend": "transformers",
        "split": "test",
        "shots": 2,
        "shot_seed": 0,
        "limit": 400,
        "label": "arm",
        "decode": {"max_new_tokens": 384, "batch_size": 8, "greedy": True},
        "detail": {"accuracy": 0.5, "by_source": {}},
        "hits": [True, False],
    }
    if style is not None:
        record["detail"]["prompt_style"] = style  # type: ignore[index]
    return record


def test_two_arms_that_resolved_different_prompt_styles_are_refused(tmp_path: Path) -> None:
    """The one pairing field two byte-identical commands can still disagree about.

    ``--prompt-style auto`` is answered by the tokenizer, so a quantized checkpoint whose
    saved tokenizer lost its chat template is asked bare-text questions while the ceiling
    is asked chat questions. On IFEval that gap was worth 44 points on Ministral-8B -- low
    enough to look like a broken model, stable enough to look real. Nothing on either
    command line differs, so the record is the only place it can be caught.

    Turns red when: ``prompt_style`` leaves ``DETAIL_PAIRING_FIELDS``, or the detail block
    stops being read.
    """
    path = tmp_path / "other.json"
    path.write_text(json.dumps(_styled("completion")), encoding="utf-8")

    with pytest.raises(DynQuantError, match=re.escape("detail.prompt_style")):
        evaluate._compare(_styled("chat"), str(path))


def test_a_task_that_reports_no_style_pairs_exactly_as_it_did_before(tmp_path: Path) -> None:
    """Absence is legitimate here, unlike every other field in the contract.

    Three tasks carry no detail block at all, and the ones that do carry different keys.
    So a missing style has to mean "this task does not report one" and pair cleanly
    against another record that does not report one -- otherwise adding this field would
    have broken every GSM8K and MMLU comparison the package can make.

    ``experts.ran`` earns the same exemption for a different reason, and the distinction
    is what this guard is really protecting. ``dynquant eval`` writes an ``experts`` block
    on every run now -- but it writes ``None`` into it for a dense model, which has no
    experts dispatch and never will, and ``None`` reads as absence. So absence still means
    "there was nothing to record" rather than "the command forgot", which is the only
    thing that makes an exemption safe.

    Turns red when: the missing-field guard stops exempting the detail-sourced keys, or
    the exemption widens to a field whose absence would mean ``dynquant eval`` had
    dropped it.
    """
    both_absent = _styled(None)
    path = tmp_path / "other.json"
    path.write_text(json.dumps(dict(both_absent, label="fp16")), encoding="utf-8")

    assert evaluate._compare(both_absent, str(path))

    assert set(evaluate._OPTIONAL_COMPARABILITY) == {
        "detail.prompt_style",
        "experts.ran",
        # A third exemption, earned the same way as the other two: `task_options` is
        # written on every run and is empty for every task that has no loader options, so
        # absence still means "there was nothing to record". What it buys is that a
        # text2sql run from here on cannot fail to say which sources it scored -- and
        # cannot pair against one written before the mixture widened.
        "task_options.sources",
    }
    assert not any(
        key.split(".", 1)[0] not in {"decode", "detail", "experts", "task_options"}
        for key in evaluate._OPTIONAL_COMPARABILITY
    )


def _sourced(names: list[str] | None) -> dict:
    """A text2sql record, with or without the source list its run resolved."""
    record = _styled("chat")
    record["task_options"] = {} if names is None else {"sources": names}
    return record


def test_an_arm_scored_on_a_wider_mixture_will_not_pair_with_one_scored_on_less(
    tmp_path: Path,
) -> None:
    """Two records that agree on every other field can still have asked different questions.

    text2sql is a mixture, so ``--split test`` does not finish naming the evaluation set.
    Both records here say ``test``, both say 400 items, both say two shots at seed 0 --
    and one of them scored a third dataset. A paired test lines the two hit vectors up
    element-wise, so pairing them would be comparing arm A's Gretel items against arm B's
    Spider ones and reporting the difference as a quantization effect.

    Turns red when: ``sources`` leaves ``TASK_PAIRING_FIELDS``, or the block stops being
    read by ``_comparability``.
    """
    path = tmp_path / "other.json"
    path.write_text(json.dumps(_sourced(["gretel", "wikisql"])), encoding="utf-8")

    with pytest.raises(DynQuantError, match=re.escape("task_options.sources")):
        evaluate._compare(_sourced(["gretel", "wikisql", "spider"]), str(path))


def test_a_text2sql_record_written_before_the_mixture_was_named_does_not_pair_silently(
    tmp_path: Path,
) -> None:
    """The banked arms are the ones this guard exists for, and they cannot be repaired.

    Every text2sql record this campaign wrote before Spider joined the registry scored
    two sources and said so nowhere. The new default scores three. Absence has to refuse
    against a stated list rather than compare equal to it -- otherwise the one comparison
    the widening actually breaks is the one that goes through unremarked.

    Turns red when: absence is filled in with a guess, or the missing-field exemption is
    read as "these two agree" rather than as "this task has no such option".
    """
    path = tmp_path / "other.json"
    path.write_text(json.dumps(_styled("chat")), encoding="utf-8")

    with pytest.raises(DynQuantError, match=re.escape("task_options.sources")):
        evaluate._compare(_sourced(["gretel", "wikisql"]), str(path))


def test_the_run_records_the_mixture_it_resolved_even_when_the_flag_was_silent() -> None:
    """An unset ``--sources`` still writes the list, because the default is derived.

    ``DEFAULT_TEST`` is every registry source that carries rows, so it grows on its own.
    A record that copied the flag verbatim would write ``None`` for the run that mattered
    most -- the unpinned one -- and two unpinned runs a registry edit apart would pair.

    Turns red when: ``_load_options`` records the flag instead of the resolution, or stops
    resolving through ``resolve_sources``.
    """
    from dynquant.eval.text2sql_sources import DEFAULT_TEST

    args = argparse.Namespace(task="text2sql", split=None, sources=None)
    assert evaluate._load_options(evaluate.TASKS["text2sql"], args) == {
        "sources": list(DEFAULT_TEST)
    }

    args.sources = ["gretel"]
    assert evaluate._load_options(evaluate.TASKS["text2sql"], args) == {"sources": ["gretel"]}


def test_a_task_with_no_loader_options_records_an_empty_block_and_not_a_null() -> None:
    """The five other tasks must keep pairing against every record they have ever written.

    ``sources`` is a text2sql setting. Writing ``sources: None`` onto a casehold record
    would refuse it against the whole banked casehold history, over a field casehold does
    not have -- which is the failure this block's shape exists to avoid, and is why it is
    nested and optional rather than a seventh ``PAIRING_FIELDS`` entry.

    Turns red when: ``_load_options`` returns a key for a task that declares no options.
    """
    args = argparse.Namespace(task="casehold", split=None, sources=["gretel"])
    assert evaluate._load_options(evaluate.TASKS["casehold"], args) == {}


def test_a_record_that_knows_its_style_will_not_pair_against_one_that_does_not(
    tmp_path: Path,
) -> None:
    """ "Unknown" is not "the same". It is refused in both directions.

    Records written before the task recorded a style are the realistic source, and the
    honest reading of one is that its framing cannot be established -- not that it matches
    whatever this run happened to resolve. The refusal costs a re-run; the alternative
    costs a wrong number in a table.

    Turns red when: absence starts comparing equal to a value, in either direction.
    """
    path = tmp_path / "other.json"
    path.write_text(json.dumps(_styled(None)), encoding="utf-8")
    with pytest.raises(DynQuantError, match=re.escape("detail.prompt_style=absent")):
        evaluate._compare(_styled("chat"), str(path))

    path.write_text(json.dumps(_styled("chat")), encoding="utf-8")
    with pytest.raises(DynQuantError, match="this run used nothing"):
        evaluate._compare(_styled(None), str(path))


# --------------------------------------------------------------------------
# bench
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4096x1024", ("4096x1024", 4096, 1024)),
        ("down=4096x1024", ("down", 4096, 1024)),
        ("a=b=16x8", ("a=b", 16, 8)),
    ],
)
def test_shapes_parse(text: str, expected: tuple) -> None:
    assert bench.parse_shape(text) == expected


@pytest.mark.parametrize("bad", ["4096", "4096y1024", "x", "0x8", "-4x8"])
def test_a_malformed_shape_is_diagnosed(bad: str) -> None:
    with pytest.raises(DynQuantError, match="--shape"):
        bench.parse_shape(bad)


def test_layer_indices_are_stripped_from_shape_labels() -> None:
    """Every layer has the same shape; one is measured on behalf of all of them,
    so the index in the label would be a lie about coverage."""
    assert bench._short("model.layers.11.mlp.down_proj") == "mlp.down_proj"
    assert bench._short("lm_head") == "lm_head"


def test_the_table_flags_a_baseline_that_beat_its_own_bandwidth() -> None:
    """A 25 MB weight fits in an A100's 40 MB L2, so the dense arm reads it from
    cache and posts over 100%. Real, and it flatters the baseline."""
    result = {
        "device": "fixture",
        "backend": "cuda",
        "dtype": "bfloat16",
        "rows_per_call": 1,
        "bits": [3],
        "achievable_read_gbs": 1000.0,
        "achievable_copy_gbs": 1400.0,
        "shapes": [["mlp.down_proj", 4096, 1024]],
        "rows": [
            {
                "shape": "mlp.down_proj",
                "bits": None,
                "micros": 5.0,
                "gbytes_per_s": 1200.0,
                "max_rel_err": 0.0,
            },
            {
                "shape": "mlp.down_proj",
                "bits": 3,
                "micros": 2.0,
                "gbytes_per_s": 900.0,
                "max_rel_err": 1e-3,
            },
        ],
    }
    table = bench.render(result)
    assert "120 %" in table
    assert "2.50x" in table  # 5.0 / 2.0
    assert "L2 residency" in table
    assert "1.00e-03" in table


def test_the_table_omits_the_cache_footnote_when_nothing_was_cached() -> None:
    result = {
        "device": "fixture",
        "backend": "cuda",
        "dtype": "bfloat16",
        "rows_per_call": 1,
        "bits": [4],
        "achievable_read_gbs": 1000.0,
        "achievable_copy_gbs": 1400.0,
        "shapes": [["big", 100000, 4096]],
        "rows": [
            {
                "shape": "big",
                "bits": None,
                "micros": 10.0,
                "gbytes_per_s": 800.0,
                "max_rel_err": 0.0,
            },
            {"shape": "big", "bits": 4, "micros": 3.0, "gbytes_per_s": 700.0, "max_rel_err": 4e-4},
        ],
    }
    assert "L2 residency" not in bench.render(result)


def test_eval_pins_the_experts_dispatch_unless_asked_not_to() -> None:
    """The default is the whole point: a panel is only one experiment if nobody opts in.

    Every arm of the LFM2.5 panel was launched without anyone thinking about the experts
    dispatch, and that is exactly the condition under which three of them ran different
    arithmetic. A flag defaulting to ``auto`` would reproduce it for the next campaign.

    Turns red when: the default flips, or the flag is dropped and the driver's namespace
    falls back to the ``getattr`` default in ``run`` without the CLI agreeing with it.
    """
    parser = build_parser()
    assert parser.parse_args(["eval", "m", "--task", "text2sql"]).experts_impl == "eager"
    assert (
        parser.parse_args(
            ["eval", "m", "--task", "text2sql", "--experts-impl", "auto"]
        ).experts_impl
        == "auto"
    )
