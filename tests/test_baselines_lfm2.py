"""The six-arm baseline driver has to be right before it runs, because running it is the cost.

Each arm is a calibration pass over an 8 B MoE plus a 400-item generation sweep. The failures
worth covering are the ones that produce a *number* rather than an error, because a number
enters the table:

* an eval namespace that differs from the ceiling's in a setting the record still describes as
  shared -- which makes the paired test compare two problem sets,
* a 3-bit arm saved as dequantized bf16, whose directory size a later reader takes for the
  arm's footprint,
* and a group size that does not divide a weight's contracted dimension, where the accounting
  silently omits the padding it does not model.

Nothing here loads a model or touches a GPU: the namespace builder, the width accounting's
guard and the save refusal are all reachable from CPU CI.
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from dynquant.commands.evaluate import PAIRING_FIELDS, TASKS

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "experiments" / "phase4" / "baselines_lfm2.py"


@pytest.fixture(scope="module")
def driver() -> Any:
    spec = importlib.util.spec_from_file_location("_dq_baselines_lfm2", DRIVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dq_baselines_lfm2"] = module
    spec.loader.exec_module(module)
    return module


def _args(driver: Any, argv: list[str]) -> argparse.Namespace:
    """Parse through the driver's own parser, so a renamed flag fails here."""
    return driver.build_parser().parse_args(argv)  # type: ignore[no-any-return]


RUN = [
    "run",
    "--model",
    "/runs/lfm25/finetuned",
    "--method",
    "gptq",
    "--bits",
    "4",
    "--label",
    "gptq-4b",
]


# --- the eval namespace ----------------------------------------------------------------


def test_the_namespace_is_a_real_dynquant_eval_namespace(driver: Any) -> None:
    """Parsed by the CLI's own parser, not assembled.

    Every default these arms are scored under has to be the default the bf16 ceiling was
    scored under. A hand-built namespace is a second copy of the eval contract, and the first
    field to drift in it -- a prompt style, a shot seed, a max prompt length -- is a
    difference between arms that the record still describes as shared.

    Turns red when: the namespace stops going through ``build_parser``, or a flag name the
    driver passes stops existing on the eval subcommand.
    """
    namespace = driver.eval_namespace(_args(driver, [*RUN, "--max-new-tokens", "512"]))

    assert namespace.task == "text2sql"
    assert namespace.backend == "transformers"
    assert namespace.max_new_tokens == 512
    # Parsed namespaces carry every eval flag, including ones this driver never sets. That is
    # the property being bought: a task default the CLI changes reaches these arms too.
    for field in PAIRING_FIELDS:
        assert hasattr(namespace, field) or field == "backend"


def test_every_pairing_field_the_arms_share_is_set_explicitly(driver: Any) -> None:
    """The fields a McNemar test refuses to pair across.

    ``PAIRING_FIELDS`` is what ``dynquant eval --compare`` checks before it agrees two hit
    vectors describe the same problems. Left to the parser's ``None`` default, ``split`` and
    ``shots`` would be filled in from the task spec at run time -- which is the same value
    today and is not the same *record*, because the record writes what was resolved. An arm
    whose recorded split is ``null`` cannot be paired with a ceiling whose recorded split is
    ``test``.

    The seed is asserted at a value the eval parser would *not* have produced on its own.
    Both sides default it to 0, so ``--shot-seed 0`` is indistinguishable from never passing
    it -- an assertion on the default here passed against the mutant that deleted the flag,
    and was rewritten rather than kept.

    Turns red when: one of these stops being passed on the command line and starts being
    inherited.
    """
    namespace = driver.eval_namespace(_args(driver, [*RUN, "--shot-seed", "7"]))

    assert namespace.split == "test"
    assert namespace.shots == 2
    assert namespace.shot_seed == 7
    assert namespace.limit is None
    assert namespace.task in TASKS

    # And 0 is what an arm gets when the driver is left alone, which is the seed the ceiling
    # ran at. Kept as a second assertion rather than the only one, for the reason above.
    assert driver.eval_namespace(_args(driver, RUN)).shot_seed == 0


def test_the_recorded_model_says_which_recipe_produced_the_weights(driver: Any) -> None:
    """The provenance field, and why it is not the path.

    ``run`` writes ``args.model`` into the record. The weights being scored are not what that
    path holds -- they went through a quantizer first -- so a bare path would label six
    different arms identically, and a later reader pairing them by ``model`` would pair GPTQ
    with AWQ.

    Turns red when: the qualification is dropped, or stops carrying the width, which is the
    part that distinguishes the two arms of the same method.
    """
    namespace = driver.eval_namespace(_args(driver, [*RUN, "--bits", "3"]))

    assert namespace.model == "/runs/lfm25/finetuned#gptq-3b-g128"
    assert namespace.label == "gptq-4b"


def test_the_decode_budget_is_never_defaulted_by_this_driver(driver: Any) -> None:
    """Unset here on purpose, because it is a measurement elsewhere.

    ``closure_budget.py`` reads the budget off the ceiling run's own closure distribution. A
    default in this parser would be a second guess at the number that script exists to
    measure -- and this campaign has already had a 5.50% headline that was measuring a decode
    cap rather than a model.

    Turns red when: ``--max-new-tokens`` acquires a default, which would silently take effect
    on every arm the flag was omitted from.
    """
    args = _args(driver, RUN)
    assert args.max_new_tokens is None

    namespace = driver.eval_namespace(args)
    # Absent from the namespace too, so the task spec supplies it and the record says so --
    # rather than this driver's guess arriving under the CLI's name.
    assert namespace.max_new_tokens is None


# --- the width accounting -------------------------------------------------------------


def test_a_group_size_that_does_not_divide_the_weight_is_refused(driver: Any) -> None:
    """Padding is not modelled, so it must not be reached.

    Group-wise quantization pads a contracted dimension that is not a multiple of the group
    size, and the accounting here charges ``numel // group_size`` groups -- which is low by
    one group per row when it does not divide. Low, in the direction that makes a baseline
    look cheaper than it is.

    ``charge`` is reached through the public function on a fake module tree rather than called
    directly, so the refusal is proven to be on the path the arms take.

    Turns red when: the divisibility check is dropped, or floor division is quietly replaced
    by a ceiling that pretends to model padding it has not measured.
    """
    import torch

    class _Odd(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(100, 8, bias=False)

    with pytest.raises(SystemExit, match="not a multiple of the group size"):
        _account(driver, _Odd(), bits=4)


def _account(driver: Any, model: Any, *, bits: int, group_size: int = 128) -> Any:
    """Run the driver's accounting against an already-built module tree.

    ``accounted_bytes`` builds its own reference from a checkpoint path, which a unit test has
    no business downloading. Patching the two names it builds that reference with is enough,
    and keeps the arithmetic under test rather than reimplemented.
    """
    import types

    fake_transformers = types.SimpleNamespace(
        AutoConfig=types.SimpleNamespace(from_pretrained=lambda _s: object()),
        AutoModelForCausalLM=types.SimpleNamespace(from_config=lambda _c: model),
    )
    saved = sys.modules.get("transformers")
    sys.modules["transformers"] = fake_transformers  # type: ignore[assignment]
    try:
        return driver.accounted_bytes("unused", bits, group_size)
    finally:
        if saved is None:
            del sys.modules["transformers"]
        else:
            sys.modules["transformers"] = saved


def test_the_expert_mass_is_charged_rather_than_left_at_fp16(driver: Any) -> None:
    """The measurement this driver exists for.

    An ``nn.Linear`` walk over this architecture reaches 8.5% of it. If the expert banks were
    not charged explicitly, the remaining 91.5% would be counted at 16 bits and a "4-bit" arm
    would account to roughly 15 bits -- or, worse, the walk would run after linearization and
    the number would look right while requiring a GPU to produce.

    Turns red when: the bank loop is dropped, or the banks stop being added to ``counted`` and
    so are charged twice -- once at ``bits`` and again in the fp16 remainder.
    """
    import torch

    class _Experts(torch.nn.Module):
        """A batched bank, named the way ``is_expert_container`` requires."""

        def __init__(self) -> None:
            super().__init__()
            self.gate_up_proj = torch.nn.Parameter(torch.zeros(4, 256, 128))
            self.down_proj = torch.nn.Parameter(torch.zeros(4, 128, 128))

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = torch.nn.Linear(128, 128, bias=False)
            self.experts = _Experts()

    banked = 4 * 256 * 128 + 4 * 128 * 128
    record = _account(driver, _Model(), bits=4)

    assert record["banked_params_quantized"] == banked
    assert record["quantized_params"] == banked + 128 * 128
    assert record["quantized_share"] == 1.0
    assert record["fp16_bits_share"] == 0.0
    # 4 bits of payload plus one fp16 scale and one 4-bit zero point per group of 128.
    assert record["accounted_bits"] == pytest.approx(4 + 20 / 128, abs=1e-4)


# --- saving ---------------------------------------------------------------------------


def test_a_three_bit_arm_refuses_to_be_saved(driver: Any) -> None:
    """compressed-tensors has no 3-bit packed format, so the directory would lie.

    ``save_pretrained(save_compressed=True)`` writes dequantized bf16 at any width it cannot
    pack. The file listing then reports ~16 bits per weight for an arm whose arithmetic is
    3-bit -- and a reader taking that byte count for the arm's footprint makes exactly the
    wrong-denominator error this driver was written to stop, one level further out.

    Turns red when: the width check is dropped or widened, which is tempting the first time
    someone wants a 3-bit directory to upload.
    """
    args = _args(
        driver,
        [
            "save",
            "--model",
            "/runs/lfm25/finetuned",
            "--save-to",
            "/out",
            "--method",
            "gptq",
            "--bits",
            "3",
        ],
    )
    with pytest.raises(SystemExit, match="packs 4 and 8 bits"):
        driver.do_save(args)


# --- the linearization gate -------------------------------------------------------------


class _PastTheLoadError(Exception):
    """Raised by the fake stack the moment a caller gets past the part being asserted."""


def _past_the_load(what: str) -> Any:
    def stop(*_a: Any, **_k: Any) -> Any:
        raise _PastTheLoadError(what)

    return stop


@contextlib.contextmanager
def _fake_stack(model: Any, *, surviving: int, seen: dict[str, Any]) -> Any:
    """Stand in for llm-compressor's detector and transformers' loader.

    Both are faked because the assertion under test is the driver's *reaction* to what the
    detector reports, and the detector's own agreement with this checkpoint is a GPU-or-host-RAM
    fact that ``baselines_lfm2.py linearize`` exists to establish separately.
    """
    import types

    calls = {"n": 0}

    def detect(_model: Any) -> list[Any]:
        calls["n"] += 1
        return [object()] * (22 if calls["n"] == 1 else surviving)

    # `oneshot` and `AutoTokenizer` are here only so a caller that goes *through* the load
    # -- `quantize` does -- reaches them and stops, loudly. Nothing past the load is faked,
    # because nothing past it is what these tests assert.
    llmc = types.ModuleType("llmcompressor")
    llmc.oneshot = _past_the_load("oneshot")  # type: ignore[attr-defined]

    modules = {
        "llmcompressor": llmc,
        "llmcompressor.modeling": types.ModuleType("llmcompressor.modeling"),
        "llmcompressor.modeling.moe": types.ModuleType("llmcompressor.modeling.moe"),
        "llmcompressor.modeling.moe.linearize": types.SimpleNamespace(
            get_non_linearized_moes=detect,
            linearize_moe=lambda _m: seen.setdefault("linearized", True),
        ),
        "transformers": types.SimpleNamespace(
            AutoConfig=types.SimpleNamespace(
                from_pretrained=lambda _s: types.SimpleNamespace(hidden_act="silu")
            ),
            AutoModelForCausalLM=types.SimpleNamespace(
                from_pretrained=lambda _s, **kw: (seen.update(kw), model)[1]
            ),
            AutoTokenizer=types.SimpleNamespace(from_pretrained=_past_the_load("tokenizer")),
        ),
    }
    saved = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)  # type: ignore[arg-type]
    printed: list[str] = []
    real_print = builtins.print
    builtins.print = lambda *a, **k: printed.append(str(a[0]))  # type: ignore[assignment]
    try:
        yield
    finally:
        builtins.print = real_print
        if printed:
            seen["printed"] = printed[-1]
        for name, module in saved.items():
            if module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = module


def test_a_surviving_bank_aborts_rather_than_quantizing_eight_percent(driver: Any) -> None:
    """The failure that succeeds if nobody checks.

    ``linearize_moe`` returns nothing and raises nothing. If it converted no banks -- a
    checkpoint whose module tree the detector's mappings do not match -- the recipe would run
    over the 8.5% reachable as ``nn.Linear``, finish, and emit an arm whose label says 4-bit
    and whose weights are 91.5% bf16.

    Turns red when: the post-conversion count is dropped, or is compared against the *before*
    count instead of zero, which passes whenever conversion is partial.
    """
    import torch

    with (
        _fake_stack(torch.nn.Linear(4, 4), surviving=3, seen={}),
        pytest.raises(SystemExit, match=r"8\.5%"),
    ):
        driver.load_linearized("/unused", device="cpu")


def test_the_gate_runs_on_the_device_it_was_asked_for(driver: Any) -> None:
    """``--device cpu`` is what makes this check runnable while the GPU is busy.

    Conversion is module surgery over weights it never reads, so the count it produces is the
    same on either device. A hardcoded ``cuda`` would make the cheapest guard in the driver the
    one that has to queue.

    Turns red when: ``device`` stops reaching ``from_pretrained``, which leaves the flag
    accepted and ignored -- an OOM on a box whose GPU is full, reported as a linearization
    failure.
    """
    import torch

    seen: dict[str, Any] = {}
    with _fake_stack(torch.nn.Linear(4, 4), surviving=0, seen=seen):
        args = _args(driver, ["linearize", "--model", "/unused", "--device", "cpu"])
        assert driver.do_linearize(args) == 0

    assert seen["device_map"] == "cpu"
    assert seen["linearized"] is True


def test_the_recipe_loads_on_the_device_the_panel_chose(driver: Any) -> None:
    """The flag the linearize gate already had, on the subcommand the panel actually runs.

    ``run`` is the only subcommand ``arms_lfm2`` invokes, and until now it was the one path
    in the panel that loaded onto ``cuda`` no matter what it was told -- ``--device`` existed
    on ``linearize`` alone. That made the seven-arm rehearsal impossible on the box the panel
    is scheduled on, whose GPU is held for hours by the fine-tune that produces the signal
    the panel scores.

    Turns red when: ``--device`` stops reaching ``load_linearized`` from ``quantize``, which
    leaves it parsed and ignored -- the failure mode where the rehearsal OOMs against a
    fine-tune rather than running beside it.
    """
    import torch

    seen: dict[str, Any] = {}
    with _fake_stack(torch.nn.Linear(4, 4), surviving=0, seen=seen):
        args = _args(
            driver,
            [
                "run",
                "--model",
                "/unused",
                "--label",
                "gptq_4b",
                "--method",
                "gptq",
                "--bits",
                "4",
                "--max-new-tokens",
                "1024",
                "--device",
                "cpu",
            ],
        )
        with pytest.raises(_PastTheLoadError):
            driver.do_run(args)

    assert seen["device_map"] == "cpu"


def test_the_gate_reports_the_parameter_share_not_just_a_module_count(driver: Any) -> None:
    """A module count says conversion happened; only the parameter share says it mattered.

    The claim this driver rests on is that linearization takes the recipe from 8.5% of the
    checkpoint to all of it. 2201 converted modules is consistent with that and also
    consistent with a mapping that reached most banks and skipped the widest one -- so the
    number that has to come out of the gate is the fraction of *parameters* now reachable as
    ``nn.Linear``, computed the same way ``visibility`` computes the 8.5%.

    Turns red when: the share is dropped back to a count, or is computed over the modules'
    parameter total rather than the model's -- which is 1.0 by construction and tells nobody
    anything.
    """
    import torch

    class _Partly(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.reached = torch.nn.Linear(10, 10, bias=False)
            self.missed = torch.nn.Parameter(torch.zeros(30, 10))

    seen: dict[str, Any] = {}
    with _fake_stack(_Partly(), surviving=0, seen=seen):
        driver.load_linearized("/unused", device="cpu")
    report = json.loads(seen["printed"])

    assert report["linear_params"] == 100
    assert report["params"] == 400
    assert report["linear_share"] == 0.25
