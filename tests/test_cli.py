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
import subprocess
import sys
from pathlib import Path

import pytest
from test_graph_classify import Qwen3_5ForCausalLM

from dynquant.allocate.knapsack import BitMap, FloorViolation
from dynquant.cli import EXIT_FAILED, build_parser, main
from dynquant.commands import _shared, bench, evaluate, inspect, quantize
from dynquant.constants import ALLOCATION_FILENAME, ALLOCATION_SCHEMA, BIT_OPTIONS
from dynquant.errors import DynQuantError
from dynquant.graph import ModuleRole, classify_model

COMMANDS = ("inspect", "quantize", "export", "eval", "bench", "doctor", "version")


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


def test_inspect_defaults_to_cpu_and_quantize_to_cuda() -> None:
    """Not cosmetic: inspect reads names and shapes, and a second copy of the
    weights on the GPU during an evaluation is how an analysis becomes an OOM."""
    parser = build_parser()
    assert parser.parse_args(["inspect", "m"]).device == "cpu"
    assert parser.parse_args(["quantize", "m", "-o", "o"]).device == "cuda"


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


# --------------------------------------------------------------------------
# Allocation seam
# --------------------------------------------------------------------------


def test_allocating_without_stats_points_at_the_honest_alternative() -> None:
    with pytest.raises(DynQuantError, match=r"--uniform"):
        _shared.build_inputs(Qwen3_5ForCausalLM(), stats=None, verbose=False)


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


@pytest.mark.parametrize("field", ["task", "split", "shots", "shot_seed", "limit"])
def test_comparing_across_settings_is_refused(tmp_path: Path, field: str) -> None:
    """A harness difference reported as a quantization effect is the failure this
    exists to prevent."""
    record = {
        "task": "gsm8k",
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


def test_comparing_different_problem_counts_is_refused(tmp_path: Path) -> None:
    record = {
        "task": "gsm8k",
        "split": "test",
        "shots": 5,
        "shot_seed": 0,
        "limit": None,
        "label": "b",
        "hits": [True, False, True],
    }
    path = tmp_path / "other.json"
    path.write_text(json.dumps(dict(record, hits=[True, False])), encoding="utf-8")
    with pytest.raises(DynQuantError, match="Not the same problem set"):
        evaluate._compare(record, str(path))


def test_a_matched_pair_compares(tmp_path: Path, capsys) -> None:
    record = {
        "task": "gsm8k",
        "split": "test",
        "shots": 5,
        "shot_seed": 0,
        "limit": None,
        "label": "quantized",
        "hits": [True, True, False, False],
    }
    path = tmp_path / "other.json"
    path.write_text(
        json.dumps(dict(record, label="fp16", hits=[True, True, True, False])), encoding="utf-8"
    )
    comparison = evaluate._compare(record, str(path))
    assert comparison
    capsys.readouterr()


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
