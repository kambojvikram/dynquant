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


def test_the_provenance_qualification_does_not_reach_the_tokenizer(driver: Any) -> None:
    """The field above stops being a path, and one thing downstream still reads it as one.

    ``eval`` defaults ``--tokenizer`` to ``--model``. Six of the seven arms pass a model
    object and a qualified string, so the default resolves ``<path>#gptq-4b-g128`` and
    ``from_pretrained`` rejects it as a Hub repo id -- after the calibration pass has been
    paid for, which is where the whole cost of a baseline arm is. It cost the seven-arm
    rehearsal its second arm, which is what the rehearsal is for.

    Turns red when: the tokenizer goes back to being defaulted, or is pointed at the
    qualified string rather than at the directory the weights were loaded from.
    """
    namespace = driver.eval_namespace(_args(driver, RUN))

    assert namespace.tokenizer == "/runs/lfm25/finetuned"
    assert "#" not in namespace.tokenizer
    assert "#" in namespace.model


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


def test_a_width_that_does_not_divide_32_refuses_to_be_saved(driver: Any) -> None:
    """A 3-bit directory would be packed, oversized and unreadable -- not bf16.

    This test used to assert the reason "compressed-tensors packs 4 and 8 bits", which is
    false: ``pack_to_int32`` takes 1 to 8 bits and round-trips 3-bit correctly. Measured, it
    packs ``32 // 3 == 10`` values per word for 3.2 stored bits per weight, and vLLM reads
    the tensor as ``Fraction(32, 3)`` -- 192 words per 2048-wide row against the 205
    written. So the refusal stands and its justification does not, and pinning the wrong
    reason is how a false claim about a dependency survives in three files at once.

    Turns red when: the check goes back to a hardcoded width list (2-bit divides 32 and must
    be allowed), or the message stops naming the layout disagreement that is the real
    blocker.
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
    with pytest.raises(SystemExit, match="does not divide 32") as caught:
        driver.do_save(args)
    message = str(caught.value)
    assert "3.2000 bits per weight" in message
    assert "192 words where 205 were written" in message
    assert "bf16" not in message


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


# --- the publication gate ----------------------------------------------------------------
#
# Linearization is what lets the recipe reach 91.5% of this model, and it is also what makes
# the resulting directory unreadable. The names that go to disk are the linearized ones;
# putting them back is a separate llm-compressor feature, ARCH_TO_2D_MAPPINGS, registered per
# architecture -- and this architecture has no entry. What that produces is not an error.
# transformers marks every expert tensor UNEXPECTED, rebuilds the banks from the config and
# returns a model with finite logits. Measured on a four-layer lfm2_moe through the shipped
# `save` subcommand: 108 packed expert tensors written, all 108 UNEXPECTED, both bank tensors
# MISSING per layer, and the reloaded bank at 32 distinct values in a group of 32 where 4-bit
# allows 16. `experiments/phase4/probe_linearized_save.py` is that measurement as a script.


@contextlib.contextmanager
def _mapping_registry(registered: set[str], asked: list[str]) -> Any:
    """Stand in for llm-compressor's per-architecture registry of load-time mappings.

    Faked rather than imported, because the gate's design is that it asks the dependency
    instead of hardcoding the answer. A test that imported the real predicate would assert
    today's contents of ``ARCH_TO_2D_MAPPINGS`` -- which is the one thing here expected to
    change, and the change is supposed to open the gate rather than turn a test red.
    """
    import types

    def has_linearize_load_mappings(model_type: str) -> bool:
        asked.append(model_type)
        return model_type in registered

    name = "llmcompressor.modeling.moe.conversion_mappings"
    saved = sys.modules.get(name)
    sys.modules[name] = types.SimpleNamespace(  # type: ignore[assignment]
        has_linearize_load_mappings=has_linearize_load_mappings
    )
    try:
        yield
    finally:
        if saved is None:
            del sys.modules[name]
        else:
            sys.modules[name] = saved


def _linearized_model(model_type: str) -> Any:
    """A module the gate can read a ``model_type`` off and ``load_linearized`` can count."""
    import types

    import torch

    model = torch.nn.Linear(4, 4)
    model.config = types.SimpleNamespace(model_type=model_type)
    return model


def test_a_linearized_bank_with_no_load_mapping_refuses_to_be_published(driver: Any) -> None:
    """The refusal names the measurement, not the suspicion.

    Every other guard in this driver stops something that would raise later. This one stops
    something that would *succeed*: a directory that writes, reloads, generates, and holds
    91.5% freshly initialized weights. So the message has to carry the number that shows it
    -- 32 distinct values in a 32-value group -- because "would not load correctly" is a
    claim a reader can dismiss and a count is not.

    Turns red when: the reason stops naming ARCH_TO_2D_MAPPINGS (which is where a reader
    goes to check whether it is still true), stops naming vLLM's ``('w1', 'w2', 'w3')`` (the
    other runtime, which fails the same way for the same reason), or stops pointing at the
    path that does work.
    """
    asked: list[str] = []
    with _mapping_registry(set(), asked), pytest.raises(SystemExit) as caught:
        driver.check_publishable(_linearized_model("lfm2_moe"), {"banks_before": 22})

    assert asked == ["lfm2_moe"]
    message = str(caught.value)
    assert "22 expert bank(s) were linearized" in message
    assert "'lfm2_moe' has no entry" in message
    assert "ARCH_TO_2D_MAPPINGS" in message
    assert "32 distinct values" in message
    assert "('w1', 'w2', 'w3')" in message
    assert "`dynquant quantize --map`" in message


def test_an_architecture_upstream_has_taught_opens_the_gate_untouched(driver: Any) -> None:
    """The gate is a question put to the dependency, so the dependency can answer it.

    ``deepseek_v4`` and ``qwen2_moe`` already have entries. If llm-compressor adds
    ``lfm2_moe``, the checkpoint round-trips and nothing in this repo should have to be
    edited for the refusal to stop firing -- and, more to the point, nobody should have to
    notice that it now refuses something that works.

    Turns red when: the predicate is replaced by a literal set of architectures, or by a
    negative list naming lfm2_moe, either of which pins today's upstream into our code.
    """
    asked: list[str] = []
    with _mapping_registry({"lfm2_moe"}, asked):
        assert driver.check_publishable(_linearized_model("lfm2_moe"), {"banks_before": 22}) is None

    assert asked == ["lfm2_moe"]


def test_a_model_that_was_never_linearized_is_not_asked_about_mappings(driver: Any) -> None:
    """No banks, no rename, no question to ask -- and asking would blame the wrong thing.

    A dense checkpoint through this driver linearizes nothing, so its names reach disk
    unchanged whatever the registry says. Gating it on ``has_linearize_load_mappings`` would
    refuse every architecture llm-compressor has no MoE entry for, which is nearly all of
    them, for a rename that did not happen.

    Turns red when: the ``banks`` term drops out of the condition, or the two terms swap so
    the registry is consulted first -- which is invisible until a dense model is saved.
    """
    asked: list[str] = []
    with _mapping_registry(set(), asked):
        assert driver.check_publishable(_linearized_model("llama"), {"banks_before": 0}) is None

    assert asked == []


def test_the_gate_runs_before_the_calibration_pass_it_would_waste(driver: Any) -> None:
    """Answerable from the model type alone, so it is answered before anything is spent.

    ``quantize`` reaches the tokenizer, 256 calibration rows and a full forward sweep over
    8 B parameters. Everything the gate needs is on the config object that came back from
    ``from_pretrained``, and the fake stack here stops at the tokenizer -- so a ``SystemExit``
    rather than ``_PastTheLoadError`` is the assertion that the order is right.

    Turns red when: the check moves after the tokenizer, or out of ``quantize`` into
    ``do_save`` past the call, which turns a free refusal into one that costs the
    calibration pass.
    """
    seen: dict[str, Any] = {}
    asked: list[str] = []
    args = _args(
        driver,
        ["save", "--model", "/unused", "--save-to", "/out", "--method", "gptq", "--bits", "4"],
    )
    with (
        _fake_stack(_linearized_model("lfm2_moe"), surviving=0, seen=seen),
        _mapping_registry(set(), asked),
        pytest.raises(SystemExit, match="ARCH_TO_2D_MAPPINGS"),
    ):
        driver.do_save(args)

    assert asked == ["lfm2_moe"]


def test_the_panel_is_not_gated_by_a_checkpoint_it_never_writes(driver: Any) -> None:
    """``run`` scores in memory, where the names never leave the module tree.

    Six of the seven arms go through this path and none of them writes a directory, so the
    round trip the gate is about does not happen to them. A gate on ``quantize`` rather than
    on publication would have made every baseline arm in the panel unrunnable on this
    architecture -- refused for a defect in an artifact they do not produce.

    Turns red when: ``for_publication`` stops defaulting to False, or the check moves out of
    the branch into ``quantize``'s body.
    """
    seen: dict[str, Any] = {}
    asked: list[str] = []
    with (
        _fake_stack(_linearized_model("lfm2_moe"), surviving=0, seen=seen),
        _mapping_registry(set(), asked),
        pytest.raises(_PastTheLoadError, match="tokenizer"),
    ):
        driver.do_run(_args(driver, RUN))

    assert asked == []


# --- AWQ smoothing mappings -------------------------------------------------------------
#
# llm-compressor has no entry for this architecture in either of its mapping registries, so
# without these the AWQ arms take the Llama defaults and either die on a half-matched target
# set or -- worse -- smooth nothing and score as AWQ anyway.

LFM25_8B = {
    # Attention at 2, 6, 10, 14, 18, 21; every other block is a short convolution.
    "layer_types": ["conv"] * 24,
    "num_dense_layers": 2,
    "num_experts": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
}
for _i in (2, 6, 10, 14, 18, 21):
    LFM25_8B["layer_types"][_i] = "full_attention"  # type: ignore[index]


def _config(**overrides: Any) -> Any:
    import types

    return types.SimpleNamespace(**{**LFM25_8B, **overrides})


def _module_names(config: Any) -> list[str]:
    """LFM2.5's module tree, as observed on the real checkpoint after ``linearize_moe``.

    Written out rather than derived from the mappings, because the mappings being consistent
    with themselves is not the question. Every name here was read off
    ``model.named_modules()`` on the box: the pre-mixer norm is ``operator_norm`` and the
    pre-FF norm is ``ffn_norm`` (neither is ``input_layernorm``), attention's output is
    ``out_proj`` (not ``o_proj``), and the dense blocks spell gate/up/down as w1/w3/w2.
    """
    names = ["model.embed_tokens", "model.norm", "lm_head"]
    for i, kind in enumerate(config.layer_types):
        layer = f"model.layers.{i}"
        if kind == "full_attention":
            names += [
                f"{layer}.self_attn.{p}"
                for p in ("q_proj", "k_proj", "v_proj", "out_proj", "q_layernorm", "k_layernorm")
            ]
        else:
            names += [f"{layer}.conv.{p}" for p in ("conv", "in_proj", "out_proj")]
        if i < config.num_dense_layers:
            names += [f"{layer}.feed_forward.{p}" for p in ("w1", "w3", "w2")]
        else:
            names.append(f"{layer}.feed_forward.gate")
            for e in range(config.num_experts):
                expert = f"{layer}.feed_forward.experts.{e}"
                names += [f"{expert}.{p}" for p in ("up_proj", "gate_proj", "down_proj")]
        names += [f"{layer}.operator_norm", f"{layer}.ffn_norm"]
    return names


def _selects(pattern: str, names: list[str]) -> list[str]:
    """``compressed_tensors._match_name``: ``re.match`` on the target minus its ``re:``."""
    import re

    return [n for n in names if re.match(pattern.removeprefix("re:"), n) is not None]


@pytest.fixture
def awq_mapping_class() -> Any:
    """A stand-in for ``AWQMapping``: two fields, which is all the driver reads.

    llm-compressor is not installed in CPU CI. What is under test here is which names the
    driver pairs with which, and that is decided before the dataclass is constructed.
    """
    import sys
    import types

    module = types.ModuleType("llmcompressor.modifiers.transform.awq")

    class AWQMapping:  # mirrors llm-compressor's own dataclass
        def __init__(self, smooth_layer: str, balance_layers: list[str]) -> None:
            self.smooth_layer = smooth_layer
            self.balance_layers = balance_layers

    module.AWQMapping = AWQMapping  # type: ignore[attr-defined]
    saved = {
        name: sys.modules.get(name)
        for name in (
            "llmcompressor",
            "llmcompressor.modifiers",
            "llmcompressor.modifiers.transform",
            "llmcompressor.modifiers.transform.awq",
        )
    }
    for name in saved:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["llmcompressor.modifiers.transform.awq"] = module
    try:
        yield AWQMapping
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def test_every_awq_mapping_selects_the_modules_this_architecture_actually_has(
    driver: Any, awq_mapping_class: Any
) -> None:
    """The defect the rehearsal found, stated as a property of the names.

    Each mapping's target set is all-or-nothing: ``match_modules_set`` accumulates matches
    per target and raises if it reaches the end of the tree holding a set that is missing
    one. So it is not enough that the balance layers exist -- the smooth layer has to exist
    in *exactly the blocks* the balance layers do. The Llama defaults fail precisely here:
    ``q/k/v_proj`` resolve, ``input_layernorm`` does not exist anywhere in this model, and
    the run dies on the residue after the calibration pass.

    Turns red when: a mapping names a norm this architecture does not have, or scopes one to
    a block kind whose balance layers live somewhere else -- an attention mapping that leaves
    the layer alternation off its ``operator_norm`` selects all 24 blocks while ``q_proj``
    selects 6, and that is the same raising case arrived at from the other side.
    """
    config = _config()
    names = _module_names(config)
    pairs = driver.awq_mappings(config)
    assert pairs, "no mappings at all is the silent-skip failure with an empty list"

    def blocks(target: str) -> set[str]:
        selected = _selects(target, names)
        assert selected, f"{target} selects nothing in this model"
        return {".".join(n.split(".")[:3]) for n in selected}

    for mapping, _ in pairs:
        governed = blocks(mapping.smooth_layer)
        for balance in mapping.balance_layers:
            # Set *equality*, in both directions, because the matcher's residue rule fails
            # both ways. A balance layer outside the smooth layer's blocks is an orphan; a
            # smooth layer covering blocks with no balance layer is the incomplete set that
            # raises -- and that is the one an unscoped `re:.*operator_norm$` produces,
            # matching all 24 blocks against six attention projections.
            assert blocks(balance) == governed, (
                f"{balance} and {mapping.smooth_layer} do not cover the same blocks"
            )


def test_the_predicted_set_count_is_read_off_the_config_not_assumed(
    driver: Any, awq_mapping_class: Any
) -> None:
    """The count is the guard against the *quiet* failure, so it has to be derived.

    A mapping that matches nothing is skipped with a debug line, not raised, and the arm
    finishes as round-to-nearest under an AWQ label. Predicting the number of resolved sets
    from ``layer_types``/``num_dense_layers``/``num_experts`` is what turns that into an
    abort -- and only if the prediction tracks the config rather than the model it was
    written against.

    Turns red when: a count is hardcoded, or the dense/MoE boundary is read from the wrong
    field -- flip ``num_dense_layers`` and a stale expectation keeps the old numbers.
    """
    counts = {m.smooth_layer: n for m, n in driver.awq_mappings(_config())}
    assert sorted(counts.values()) == [2, 2, 6, 18, 22, 704]
    # 704 is 22 MoE blocks x 32 experts: the expert pair yields per expert, not per block.
    assert max(counts.values()) == 22 * 32

    smaller = {m.smooth_layer: n for m, n in driver.awq_mappings(_config(num_experts=8))}
    assert max(smaller.values()) == 22 * 8
    shallow = driver.awq_mappings(_config(num_dense_layers=4))
    assert sorted(n for _, n in shallow) == [4, 4, 6, 18, 20, 20 * 32]


def test_grouped_query_attention_drops_the_pair_that_cannot_be_scaled(
    driver: Any, awq_mapping_class: Any
) -> None:
    """``v_proj -> out_proj`` only exists when the two ends have the same width.

    Under GQA ``v_proj`` emits ``kv_heads * head_dim`` rows and ``out_proj`` consumes
    ``heads * head_dim``; no per-channel scale divides one and multiplies the other. AWQ
    drops the pair for that reason, but its check is spelled ``balance_name.endswith
    (".o_proj")`` and this model calls the module ``out_proj`` -- so upstream's guard never
    fires here and ``_smooth`` reaches ``weight[-scales.size(0):]`` with four times as many
    scales as rows. LFM2.5-8B-A1B is 32 query heads over 8 key/value heads.

    Turns red when: the pair becomes unconditional -- which is a crash rather than a bad
    number, but a crash 256 calibration sequences into an 8 B model -- or when the condition
    is inverted and a model that *could* smooth ``out_proj`` silently stops.
    """
    grouped = [m.smooth_layer for m, _ in driver.awq_mappings(_config())]
    assert "re:.*self_attn.v_proj$" not in grouped

    even = [m.smooth_layer for m, _ in driver.awq_mappings(_config(num_key_value_heads=32))]
    assert "re:.*self_attn.v_proj$" in even
    assert len(even) == len(grouped) + 1


def test_a_config_from_another_architecture_is_refused_rather_than_matched(
    driver: Any, awq_mapping_class: Any
) -> None:
    """These names are LFM2's. Applied elsewhere they would match nothing, quietly.

    Turns red when: the config fields are read with ``getattr(..., default)``, which turns a
    Llama config into a mapping list scoped to zero layers instead of an abort.
    """
    import types

    with pytest.raises(SystemExit, match="LFM2 MoE stack"):
        driver.awq_mappings(types.SimpleNamespace(num_hidden_layers=32))


def _resolver_stack(sets_for: Any, seen: dict[str, Any]) -> Any:
    """Fake only ``match_modules_set``; the driver's reaction to it is what is asserted."""
    import contextlib
    import sys
    import types

    @contextlib.contextmanager
    def swap() -> Any:
        module = types.ModuleType("compressed_tensors.utils")
        module.match_modules_set = sets_for  # type: ignore[attr-defined]
        saved = {
            name: sys.modules.get(name)
            for name in ("compressed_tensors", "compressed_tensors.utils")
        }
        sys.modules.setdefault("compressed_tensors", types.ModuleType("compressed_tensors"))
        sys.modules["compressed_tensors.utils"] = module
        printed: list[str] = []
        real_print = builtins.print
        builtins.print = lambda *a, **k: printed.append(str(a[0]))  # type: ignore[assignment]
        try:
            yield
        finally:
            builtins.print = real_print
            if printed:
                seen["printed"] = printed[-1]
            for name, previous in saved.items():
                if previous is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous

    return swap()


# What is not an ``nn.Linear`` on the real checkpoint, and so is not the recipe's business
# nor AWQ's: the embedding, the four RMSNorms per block, and the depthwise short convolution.
NOT_LINEAR = ("embed_tokens", "norm", "q_layernorm", "k_layernorm", "conv.conv")


def _tiny_model(config: Any) -> Any:
    """A module tree with LFM2's names, ``nn.Linear`` exactly where the real one has one."""
    import torch

    root = torch.nn.Module()
    root.config = config
    holders: dict[str, torch.nn.Module] = {"": root}

    def holder(path: str) -> torch.nn.Module:
        if path not in holders:
            parent, _, leaf = path.rpartition(".")
            made = torch.nn.Module()
            holder(parent).add_module(leaf, made)
            holders[path] = made
        return holders[path]

    for name in _module_names(config):
        parent, _, leaf = name.rpartition(".")
        leafy = leaf.endswith("norm") or leaf in NOT_LINEAR or name.endswith(NOT_LINEAR)
        holder(parent).add_module(leaf, torch.nn.Module() if leafy else torch.nn.Linear(2, 2))
    return root


def test_a_mapping_that_resolves_fewer_sets_than_the_config_predicts_aborts(
    driver: Any, awq_mapping_class: Any
) -> None:
    """Half a resolution is the failure that still produces an accuracy number.

    Turns red when: the resolved-set count is collected and not compared, or compared with
    ``>=`` -- both leave a mapping that reached one block out of twenty-two looking fine.
    """
    import types

    config = _config(num_experts=2)
    model = types.SimpleNamespace(config=config, named_modules=lambda: [], modules=lambda: [])

    def one_set_each(_model: Any, targets: Any) -> Any:
        yield [[object()] for _ in targets]

    with (
        _resolver_stack(one_set_each, {}),
        pytest.raises(SystemExit, match=r"resolved 1 sets, config predicts"),
    ):
        driver.resolve_awq_mappings(model)


def test_a_partly_matched_target_set_names_the_mapping_that_did_not_fit(
    driver: Any, awq_mapping_class: Any
) -> None:
    """llm-compressor's own error says which keys matched; it does not say whose mapping.

    Turns red when: the ``ValueError`` is left to propagate -- the run still fails, with a
    message that lists three regexes and no way back to the mapping that produced them.
    """
    import types

    model = types.SimpleNamespace(config=_config(), named_modules=lambda: [], modules=lambda: [])

    def raising(_model: Any, _targets: Any) -> Any:
        raise ValueError("Found a final incomplete set with matches found for keys: ...")
        yield  # pragma: no cover

    with (
        _resolver_stack(raising, {}),
        pytest.raises(SystemExit, match=r"operator_norm.*do not describe this model"),
    ):
        driver.resolve_awq_mappings(model)


def test_resolving_every_mapping_and_smoothing_nothing_is_still_a_failure(
    driver: Any, awq_mapping_class: Any
) -> None:
    """The end state the whole guard exists for, reached the other way.

    Every mapping can resolve its predicted number of sets and still balance no module, if
    the matcher hands back empty match lists. That arm is round-to-nearest and its record
    says AWQ.

    Turns red when: coverage is inferred from the mapping count rather than from the modules
    the matcher actually returned.
    """
    import types

    import torch

    config = _config(num_experts=2)
    counts = [n for _, n in driver.awq_mappings(config)]
    model = types.SimpleNamespace(
        config=config,
        named_modules=lambda: [("lm_head", torch.nn.Linear(2, 2))],
    )
    remaining = iter(counts)

    def empty_sets(_model: Any, targets: Any) -> Any:
        for _ in range(next(remaining)):
            yield [[] for _ in targets]

    with (
        _resolver_stack(empty_sets, {}),
        pytest.raises(SystemExit, match="round-to-nearest under an AWQ label"),
    ):
        driver.resolve_awq_mappings(model)


def test_the_record_says_which_linears_go_through_unsmoothed(
    driver: Any, awq_mapping_class: Any
) -> None:
    """``conv.out_proj`` and ``lm_head`` have no linear producer, and that has to be visible.

    Every other Linear in the model is balanced by some mapping. These two are quantized
    without an activation-aware scale because there is nothing to fold the inverse into --
    a property of the architecture, not a miss. It goes in the arm's record so that the day
    it changes, it changes in a number somebody can see.

    ``self_attn.out_proj`` is here for a different reason -- grouped-query attention makes
    the ``v_proj`` pair unscalable -- and the two reasons reading the same in the record is
    the point: what the arm did not smooth is one number regardless of why.

    Turns red when: a mapping stops covering a family it used to cover. Drop the conv
    mapping and ``conv.in_proj`` joins this dict, which is 18 more unsmoothed projections
    than the record claims.
    """
    config = _config(num_experts=2)
    names = _module_names(config)
    model = _tiny_model(config)
    by_name = dict(model.named_modules())

    def real_enough(_model: Any, targets: Any) -> Any:
        """Group by block -- or by expert, for the one pair that lives inside one.

        This is what the real matcher's lowest-common-ancestor rule comes to on this tree,
        and it is a stand-in only for the grouping. That the grouping is *this* is checked
        against llm-compressor itself on the box, where the counts land at 6/6/18/2/2/22/704.
        """
        joined = "".join(targets)
        by_expert = "experts" in joined and "norm" not in joined
        buckets: dict[str, list[list[Any]]] = {}
        for column, target in enumerate(targets):
            for name in _selects(target, names):
                key = name.rsplit(".", 1)[0] if by_expert else ".".join(name.split(".")[:3])
                buckets.setdefault(key, [[] for _ in targets])[column].append(by_name[name])
        for _, bucket in sorted(buckets.items()):
            if all(bucket):
                yield bucket

    seen: dict[str, Any] = {}
    with _resolver_stack(real_enough, seen):
        mappings, report = driver.resolve_awq_mappings(model)

    assert len(mappings) == len(driver.awq_mappings(config))
    assert report["unsmoothed_linears"] == {
        "conv.out_proj": 18,
        "lm_head": 1,
        "self_attn.out_proj": 6,
    }
    assert report["smoothed_linears"] == report["linear_modules"] - 25
    assert json.loads(seen["printed"])["unsmoothed_linears"] == report["unsmoothed_linears"]


def test_the_mappings_reach_the_modifier_and_only_on_the_awq_arm(driver: Any) -> None:
    """A resolved mapping list that never reaches ``AWQModifier`` changes nothing at all.

    ``build_recipe`` is shared with two other experiments whose models llm-compressor
    already knows, so ``None`` has to keep meaning "use your own table" rather than "use an
    empty one" -- an empty list would disable smoothing everywhere in one keystroke.

    Turns red when: the parameter is accepted and dropped, or is passed to GPTQ/RTN, or
    ``None`` is forwarded into ``AWQModifier(mappings=None)`` -- which llm-compressor reads
    as an explicit empty override on some versions rather than as absence.
    """
    import inspect

    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    try:
        import _llmc
    finally:
        sys.path.pop(0)

    signature = inspect.signature(_llmc.build_recipe)
    assert signature.parameters["mappings"].default is None
    source = inspect.getsource(_llmc.build_recipe)
    assert "AWQModifier(mappings=mappings) if mappings is not None else AWQModifier()" in source
    assert "mappings" not in source.split('if method == "awq"')[0].split("scheme = ")[1]

    quantize = inspect.getsource(driver.quantize)
    assert (
        'if args.method == "awq":\n        mappings, smoothing = resolve_awq_mappings' in quantize
    )
    assert "mappings=mappings" in quantize


# --- carrying the recipe's own grid out ------------------------------------------------


@contextlib.contextmanager
def _offload_shim() -> Any:
    """Stand in for ``compressed_tensors.utils.align_module_device``.

    The real one moves an offloaded module's weights onto the execution device. Nothing in
    these tests is offloaded, so the behaviour under test is what the driver does with the
    numbers, not where they live -- and importing compressed-tensors to get a context
    manager would put a GPU-era dependency on CPU CI for no assertion.
    """
    import types

    made_parent = "compressed_tensors" not in sys.modules
    if made_parent:
        sys.modules["compressed_tensors"] = types.ModuleType("compressed_tensors")
    name = "compressed_tensors.utils"
    saved = sys.modules.get(name)
    sys.modules[name] = types.SimpleNamespace(  # type: ignore[assignment]
        align_module_device=lambda _m: contextlib.nullcontext()
    )
    try:
        yield
    finally:
        if saved is None:
            del sys.modules[name]
        else:
            sys.modules[name] = saved
        if made_parent:
            del sys.modules["compressed_tensors"]


def _on_a_grid(
    *, rows: int, in_features: int, group_size: int, bits: int, band: tuple[int, int]
) -> Any:
    """A weight that *is* ``scale * (q - zero)``, with codes confined to ``band``.

    The band is the point. A recipe that leaves every group spanning ``[0, qmax]`` cannot
    tell a carried grid from a re-fitted one -- min/max recovers the same step either way.
    GPTQ's compensation and AWQ's clipping search leave groups that do not span, and a
    narrow band is what those look like, so it is what the fixture builds.
    """
    import torch

    groups = -(-in_features // group_size)
    generator = torch.Generator().manual_seed(11)
    low, high = band
    codes = torch.randint(
        low, high + 1, (rows, in_features), generator=generator, dtype=torch.float32
    )
    # Pinned, so the band is exactly `band` rather than whatever the draw happened to hit.
    codes[:, 0], codes[:, 1] = low, high
    scale = torch.rand((rows, groups), generator=generator, dtype=torch.float32) * 0.1 + 0.01
    zero = torch.full((rows, groups), 3.0)
    wide_scale = scale.repeat_interleave(group_size, dim=1)[:, :in_features]
    wide_zero = zero.repeat_interleave(group_size, dim=1)[:, :in_features]
    return codes, scale, zero, wide_scale * (codes - wide_zero)


def _quantized(weight: Any, scale: Any, zero: Any, *, bits: int, symmetric: bool = False) -> Any:
    """A module shaped like one llm-compressor has finished with."""
    import types

    import torch

    module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    with torch.no_grad():
        module.weight.copy_(weight)
    module.register_buffer("weight_scale", scale)
    if zero is not None:
        module.register_buffer("weight_zero_point", zero)
    module.quantization_scheme = types.SimpleNamespace(
        weights=types.SimpleNamespace(num_bits=bits, symmetric=symmetric)
    )
    return module


def _holding(**modules: Any) -> Any:
    import torch

    return torch.nn.ModuleDict(modules)


def test_the_recipes_own_codes_are_carried_not_refitted_from_the_weights(driver: Any) -> None:
    """The whole reason this path exists, stated as the thing a refit would destroy.

    The weights handed over are on a grid whose groups occupy codes 5 through 9 of the 16 a
    4-bit width offers. Fitting a fresh grid to them -- which is what
    ``export_packed_checkpoint`` does unaided -- recovers a step four times narrower and
    puts the original levels between the new ones. Carrying the grid keeps the codes the
    recipe chose, and the test for that is that they are still 5 through 9.
    """
    codes, scale, zero, weight = _on_a_grid(
        rows=4, in_features=64, group_size=32, bits=4, band=(5, 9)
    )
    model = _holding(proj=_quantized(weight, scale, zero, bits=4))
    with _offload_shim():
        grids = driver.carried_grids(model, group_size=32)

    import torch

    grid = grids["proj"]
    assert set(grids) == {"proj"}
    assert grid["bits"] == 4
    assert torch.equal(grid["codes"], codes.to(torch.uint8)), "the carried codes moved"
    assert (int(grid["codes"].min()), int(grid["codes"].max())) == (5, 9), (
        "codes spanning the full width mean a grid was fitted here, not carried"
    )
    assert torch.equal(grid["offsets"], -scale * 3.0)


def test_a_carried_grid_reconstructs_the_weight_the_arm_was_scored_on(driver: Any) -> None:
    """Codes plus a float offset must land back on the levels the zero point described.

    Not bit-exact, and the docstring on ``from_codes`` says why: ``s * q + (-s * z)`` and
    ``s * (q - z)`` are the same number and round differently. The bound is in units of a
    code step, because that is the unit in which a *mapping* error would show up -- one
    step or more -- and this has to be far below it to be worth admitting.
    """
    import torch

    from dynquant.quant.tensor import QuantTensor

    _codes, scale, zero, weight = _on_a_grid(
        rows=4, in_features=64, group_size=32, bits=4, band=(5, 9)
    )
    model = _holding(proj=_quantized(weight, scale, zero, bits=4))
    with _offload_shim():
        grid = driver.carried_grids(model, group_size=32)["proj"]

    carried = QuantTensor.from_codes(
        grid["codes"],
        grid["scales"],
        grid["offsets"],
        bits=4,
        group_size=32,
        compute_dtype=torch.float32,
    )
    step = scale.abs().max()
    assert ((carried.dequantize().float() - weight).abs().max() / step).item() < 1e-5


def test_a_signed_band_is_carried_as_an_unsigned_code_and_an_offset(driver: Any) -> None:
    """A symmetric scheme has no zero point, and this format has no signed code.

    Both halves matter. compressed-tensors writes symmetric weights on ``[-8, 7]`` with no
    ``weight_zero_point`` buffer at all, and ``from_codes`` refuses anything outside
    ``[0, 15]``. Folding the shift into the offset moves no level, and the check is that
    the reconstruction still equals the weight.
    """
    import torch

    generator = torch.Generator().manual_seed(5)
    scale = torch.rand((3, 2), generator=generator) * 0.1 + 0.01
    codes = torch.randint(-8, 8, (3, 64), generator=generator, dtype=torch.float32)
    codes[:, 0], codes[:, 1] = -8, 7
    weight = scale.repeat_interleave(32, dim=1) * codes

    model = _holding(proj=_quantized(weight, scale, None, bits=4, symmetric=True))
    with _offload_shim():
        grid = driver.carried_grids(model, group_size=32)["proj"]

    assert torch.equal(grid["codes"], (codes + 8).to(torch.uint8))
    assert (int(grid["codes"].min()), int(grid["codes"].max())) == (0, 15)
    reconstructed = grid["scales"].repeat_interleave(32, dim=1) * grid["codes"].float()
    assert torch.allclose(reconstructed + grid["offsets"].repeat_interleave(32, dim=1), weight)


def test_a_weight_that_is_not_on_its_own_grid_refuses_to_be_carried(driver: Any) -> None:
    """The check that catches a convention disagreement instead of publishing one.

    ``materialize_quantization`` leaves ``weight == fake_quantize(weight)``, so recomputing
    the codes and reconstructing has exactly one degree of freedom: whether this file reads
    the library's convention the way the library writes it. Perturbing one weight off the
    grid stands in for that disagreement -- what a reader looking at the wrong end of a
    scale, or the wrong sign of a zero point, would produce.
    """
    _codes, scale, zero, weight = _on_a_grid(
        rows=4, in_features=64, group_size=32, bits=4, band=(5, 9)
    )
    weight = weight.clone()
    weight[2, 3] += 0.3 * scale[2, 0]
    model = _holding(proj=_quantized(weight, scale, zero, bits=4))
    with _offload_shim(), pytest.raises(SystemExit, match="does not reproduce"):
        driver.carried_grids(model, group_size=32)


def test_a_scale_that_is_not_per_group_along_the_input_is_refused(driver: Any) -> None:
    """A per-channel or per-tensor recipe has a different reader, and does not get this one."""
    import torch

    _codes, scale, _zero, weight = _on_a_grid(
        rows=4, in_features=64, group_size=32, bits=4, band=(5, 9)
    )
    model = _holding(proj=_quantized(weight, scale[:, :1].contiguous(), torch.zeros(4, 1), bits=4))
    with _offload_shim(), pytest.raises(SystemExit, match="per-group scales"):
        driver.carried_grids(model, group_size=32)


def test_a_model_the_recipe_never_touched_is_refused_rather_than_published_dense(
    driver: Any,
) -> None:
    """An empty grid set writes the unquantized checkpoint under a quantized name."""
    import torch

    with _offload_shim(), pytest.raises(SystemExit, match="no grid to carry"):
        driver.carried_grids(_holding(proj=torch.nn.Linear(64, 4)), group_size=32)


# --- and arranging it into banks by the rule the weights use ----------------------------


BANK_RULES: list[dict[str, Any]] = [
    {
        "linear": "l.{}.e.{}.gate_proj",
        "bank": "l.{}.e.gate_up_proj",
        "orientation": "as_stored",
        "splits": 2,
        "part": 0,
    },
    {
        "linear": "l.{}.e.{}.up_proj",
        "bank": "l.{}.e.gate_up_proj",
        "orientation": "as_stored",
        "splits": 2,
        "part": 1,
    },
    {
        "linear": "l.{}.e.{}.down_proj",
        "bank": "l.{}.e.down_proj",
        "orientation": "as_stored",
        "splits": 1,
        "part": 0,
    },
]
"""The shape ``expert_rules()`` derives, written out so this file does not need a model.

Not a second copy of the mapping: ``delinearize_state_dict`` is what is under test here, and
what it does with a rule is independent of which rule the installed llm-compressor produces.
``probe_linearize_mapping.py`` is what says these are the real ones, on the real library."""


def _expert_grids(*, experts: int, inter: int, hidden: int, bits: int = 4) -> Any:
    """One grid per linearized expert projection, values chosen to be identifiable."""
    import torch

    grids: dict[str, dict[str, Any]] = {}
    for expert in range(experts):
        for part, (leaf, rows, cols) in enumerate(
            [("gate_proj", inter, hidden), ("up_proj", inter, hidden), ("down_proj", hidden, inter)]
        ):
            tag = expert * 10 + part
            grids[f"l.0.e.{expert}.{leaf}"] = {
                "codes": torch.full((rows, cols), tag, dtype=torch.uint8),
                "scales": torch.full((rows, -(-cols // 4)), float(tag)),
                "offsets": torch.full((rows, -(-cols // 4)), -float(tag)),
                "bits": bits,
            }
    return grids


def test_the_codes_and_the_scales_are_stacked_by_the_rule_the_weights_are(driver: Any) -> None:
    """One assembler, three tensors, and the arrangement checked on all three.

    A bank's codes have to be the concatenation of its parts and the stack of its experts in
    the identical order its weights are, or the published checkpoint dequantizes to a model
    whose experts are permuted -- which loads, runs, and scores somewhere between the arm
    and noise. The scales carry the same arrangement with the group axis in place of the
    input axis, and are flattened to one row per output row because that is how the packer
    folds a bank's leading dimensions.
    """
    import torch

    grids = _expert_grids(experts=2, inter=3, hidden=4)
    banked, width = driver.banked_grids(object(), grids, BANK_RULES)

    assert width == 4
    assert set(banked) == {"l.0.e.gate_up_proj", "l.0.e.down_proj"}
    fused = banked["l.0.e.gate_up_proj"]
    assert tuple(fused["codes"].shape) == (2, 6, 4), "experts stacked, gate over up"
    for expert in range(2):
        assert torch.equal(fused["codes"][expert][:3], grids[f"l.0.e.{expert}.gate_proj"]["codes"])
        assert torch.equal(fused["codes"][expert][3:], grids[f"l.0.e.{expert}.up_proj"]["codes"])
    # [E, out, groups] flattened to [E * out, groups]: expert 1's up rows are the last three.
    assert tuple(fused["scales"].shape) == (12, 1)
    assert [float(v) for v in fused["scales"][:, 0]] == [0, 0, 0, 1, 1, 1, 10, 10, 10, 11, 11, 11]
    assert [float(v) for v in fused["offsets"][:, 0]] == [
        -0.0,
        -0.0,
        -0.0,
        -1,
        -1,
        -1,
        -10,
        -10,
        -10,
        -11,
        -11,
        -11,
    ]
    assert tuple(banked["l.0.e.down_proj"]["codes"].shape) == (2, 4, 3)


def test_a_recipe_that_used_two_widths_has_no_entry_for_the_bank_it_fills(driver: Any) -> None:
    """A bank is one tensor. Two widths inside it is not a checkpoint, it is a bug report."""
    grids = _expert_grids(experts=2, inter=3, hidden=4)
    grids["l.0.e.1.up_proj"]["bits"] = 3
    with pytest.raises(SystemExit, match=r"widths \[3, 4\]"):
        driver.banked_grids(object(), grids, BANK_RULES)


# --- and checking it against the weight it stands in for --------------------------------


def _carryable(driver: Any) -> Any:
    """A one-module banked grid plus the weight it is supposed to reconstruct."""
    _codes, scale, zero, weight = _on_a_grid(
        rows=4, in_features=64, group_size=32, bits=4, band=(5, 9)
    )
    model = _holding(proj=_quantized(weight, scale, zero, bits=4))
    with _offload_shim():
        grids = driver.carried_grids(model, group_size=32)
    banked, _width = driver.banked_grids(object(), grids, [])
    return banked, weight


def test_a_grid_that_reconstructs_its_weight_is_returned_with_the_gap_recorded(
    driver: Any,
) -> None:
    """The gap is a published number, not a tolerance nobody sees.

    ``scale * code + offset`` and ``scale * (q - zero)`` differ by the rounding of the
    offset into the storage dtype. Recording the measured worst case per run is what makes
    the difference between the directory and the scored arm a fact in the record rather
    than an assumption in a docstring.
    """
    banked, weight = _carryable(driver)
    drift: dict[str, float] = {}
    encode = driver.carrying_encoder(banked, group_size=32, drift=drift)

    quantized = encode("proj", weight, 4)
    assert quantized.bits == 4 and quantized.group_size == 32
    assert set(drift) == {"proj"}
    assert 0.0 <= drift["proj"] < driver.MAX_CARRY_DRIFT


def test_a_grid_arranged_differently_from_its_weight_fails_on_the_module(driver: Any) -> None:
    """The check that turns a rearrangement error into an exception instead of a score.

    The codes and the weight reach this encoder by different routes -- the integer route
    through the assembler, the float route through the state dict -- so a rule applied
    inconsistently to the two shows up as a reconstruction that is whole code steps away.
    Rolling the rows here is what a wrongly-stacked expert looks like at this scale.
    """
    banked, weight = _carryable(driver)
    drift: dict[str, float] = {}
    encode = driver.carrying_encoder(banked, group_size=32, drift=drift)

    with pytest.raises(SystemExit, match="code steps away"):
        encode("proj", weight.roll(1, dims=0), 4)


def test_a_module_the_recipe_never_quantized_is_refused_by_the_encoder(driver: Any) -> None:
    """The bit map and the grids come from one dict; a name in one only means two now."""
    banked, weight = _carryable(driver)
    encode = driver.carrying_encoder(banked, group_size=32, drift={})
    with pytest.raises(SystemExit, match="never quantized"):
        encode("somewhere.else", weight, 4)


def test_publish_takes_the_same_recipe_flags_the_scored_arm_ran(driver: Any) -> None:
    """A published directory has to come from the recipe whose row it is published under.

    ``publish`` reuses ``quant_flags`` rather than declaring its own, so this asserts the
    reuse survived: a width, group size or calibration size that ``run`` accepts and
    ``publish`` silently defaults differently would put a different model behind the number.
    """
    published = _args(
        driver,
        [
            "publish",
            "--model",
            "/runs/lfm25/finetuned",
            "--save-to",
            "/out/gptq4",
            "--method",
            "gptq",
            "--bits",
            "3",
            "--group-size",
            "64",
        ],
    )
    scored = _args(driver, [*RUN, "--group-size", "64", "--out", "/dev/null"])
    shared = ("method", "bits", "group_size", "calib_samples", "seq_len", "seed", "dtype")
    assert published.func is driver.do_publish
    assert published.pack_device == "auto"
    assert [getattr(published, f) for f in shared if f != "bits"] == [
        getattr(scored, f) for f in shared if f != "bits"
    ]


# --------------------------------------------------------------------------------------
# What the recipe leaves behind, and which name a tied table is published under.
#
# Both of these were found by running an actual llm-compressor recipe end to end
# (`experiments/phase4/probe_publish.py`) after the publish path had already shipped with
# thirteen green unit tests. Neither was reachable from the fixtures above: the first
# because a hand-built module never carries recipe scratch, the second because it takes
# `oneshot` mutating a live config to produce it.
# --------------------------------------------------------------------------------------


def _recipe_module(*, bias: bool = True) -> Any:
    """A module shaped like one a recipe has finished with, scratch tensors and all.

    ``weight_g_idx`` is GPTQ's column permutation and rides along with the other two on
    that arm. It is here so the filter is exercised against something it was not told
    about by name -- which is the whole point of subtracting rather than listing.
    """
    import types

    import torch

    module = torch.nn.Linear(8, 4, bias=bias)
    module.register_buffer("weight_scale", torch.ones(4, 1))
    module.register_buffer("weight_zero_point", torch.zeros(4, 1))
    module.register_buffer("weight_g_idx", torch.zeros(8, dtype=torch.int32))
    module.quantization_scheme = types.SimpleNamespace(
        weights=types.SimpleNamespace(num_bits=4, symmetric=False)
    )
    return module


def test_the_recipes_own_scales_are_not_published_as_model_weights(driver: Any) -> None:
    """The tensors compressed-tensors fitted with belong to neither side of the export.

    They are not merely extra. One hanging off a linearized expert trips the bank
    assembler's refusal -- correctly, since the derived rules say where a weight row goes
    and nothing about where a per-group scale would -- and one hanging off anything else
    survives into the output and makes ``load_state_dict(strict=True)`` reject the model.
    A weight and a bias are what a ``Linear`` owns; everything else arrived with the recipe.
    """
    model = _holding(q=_recipe_module(), plain=__import__("torch").nn.Linear(4, 4))

    assert driver.recipe_scratch(model) == {
        "q.weight_scale",
        "q.weight_zero_point",
        "q.weight_g_idx",
    }
    assert set(driver.recipe_weights(model)) == {
        "q.weight",
        "q.bias",
        "plain.weight",
        "plain.bias",
    }


def test_a_module_the_recipe_never_quantized_keeps_every_tensor_it_holds(driver: Any) -> None:
    """The filter asks the module whether a recipe touched it, not the leaf what it is named.

    A rotary cache, a norm's running statistic and an expert bank's routing counter are all
    non-architectural leaves on modules no recipe went near. Filtering by leaf name alone
    would take them out of the state dict and the strict load would then say they are
    missing -- the same failure as the one this helper exists to prevent, one module over.
    """
    import torch

    rope = torch.nn.Module()
    rope.register_buffer("inv_freq", torch.arange(4, dtype=torch.float32))
    model = _holding(q=_recipe_module(), rope=rope)

    assert "rope.inv_freq" not in driver.recipe_scratch(model)
    assert "rope.inv_freq" in driver.recipe_weights(model)


def _head_and_table(*, config_says: bool, share: bool, equal: bool) -> Any:
    """A model-shaped object holding an input embedding and an output head.

    The three knobs are independent on purpose, because the case that broke the publish
    path sets them in a combination no honest model would: a config saying untied over
    storage that is shared.
    """
    import types

    import torch

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(8, 4)
            self.lm_head = torch.nn.Linear(4, 8, bias=False)
            self.config = types.SimpleNamespace(tie_word_embeddings=config_says)

        def get_input_embeddings(self) -> Any:
            return self.model.embed_tokens

        def get_output_embeddings(self) -> Any:
            return self.lm_head

    model = _Model()
    if share:
        model.lm_head.weight = model.model.embed_tokens.weight
    elif equal:
        with torch.no_grad():
            model.lm_head.weight.copy_(model.model.embed_tokens.weight)
    return model


def test_a_tie_the_recipe_cleared_in_the_config_is_still_read_off_the_storage(
    driver: Any,
) -> None:
    """The measured shape of the bug: the flag says untied and the storage says otherwise.

    ``oneshot`` sets ``tie_word_embeddings`` to ``False`` on the live config while leaving
    ``lm_head`` and ``model.embed_tokens`` sharing one storage. It is describing the
    compressed checkpoint it would write, which carries two tables -- not the model in
    memory, which carries one, and which is the model being published. Reading the flag
    skips the rename, publishes the table under the head's name, and produces a directory
    whose embedding transformers reports missing, re-initializes at random, and then dies
    on in ``mark_tied_weights_as_initialized``.
    """
    report = driver.tie_report(_head_and_table(config_says=False, share=True, equal=False))

    assert report["tied"] is True
    assert report["config_says"] is False
    assert report["shared_storage"] is True
    assert (report["input"], report["output"]) == ("model.embed_tokens", "lm_head")


def test_two_storages_holding_the_same_numbers_are_as_tied_as_one(driver: Any) -> None:
    """Everything downstream is written from a state dict, where equal numbers are one table.

    A recipe that re-materializes the head as its own parameter and writes the same values
    back has not changed what gets published. Requiring a shared ``data_ptr`` here would
    refuse that model, and refusing it means the arm cannot be published at all.
    """
    report = driver.tie_report(_head_and_table(config_says=True, share=False, equal=True))

    assert (report["tied"], report["shared_storage"], report["values_equal"]) == (
        True,
        False,
        True,
    )


def test_a_head_holding_different_numbers_from_its_table_is_not_reported_tied(
    driver: Any,
) -> None:
    """The failure the ``declared_tie`` guard in ``do_publish`` is watching for.

    A checkpoint that declares its head tied and comes out of the recipe with two tables
    holding different weights was scored as a model 27% larger than its own byte accounting
    describes, with a table at a precision its label does not mention. Neither name is
    publishable, so ``do_publish`` refuses rather than picking one.
    """
    report = driver.tie_report(_head_and_table(config_says=True, share=False, equal=False))

    assert report["tied"] is False
    assert report["values_equal"] is False


def test_a_tied_table_is_published_under_the_name_the_loader_reads(driver: Any) -> None:
    """A recipe can only produce ``lm_head``; the format only reads ``model.embed_tokens``.

    llm-compressor targets ``Linear``, so the module it quantizes on a tied model is the
    head. The DynQuant format stores a tied table once, under the *input* embedding's name,
    and ``_tie_output_embedding`` then hands the head a view of it. A rename and not a
    second entry -- writing the table twice costs a quarter of a tied model for bytes the
    loader discards -- and the grid travels unchanged, because the two names are one tensor.
    """
    grid = {"codes": object()}
    moved = driver.under_the_input_table(
        _head_and_table(config_says=False, share=True, equal=False),
        {"lm_head": grid, "model.layers.0.mlp.up_proj": "other"},
    )

    assert set(moved) == {"model.embed_tokens", "model.layers.0.mlp.up_proj"}
    assert moved["model.embed_tokens"] is grid


def test_one_tied_table_carrying_two_grids_is_refused_rather_than_picked(driver: Any) -> None:
    """Two entries for one tensor is a question about which the export cannot answer."""
    with pytest.raises(SystemExit, match="two names for one tied"):
        driver.under_the_input_table(
            _head_and_table(config_says=False, share=True, equal=False),
            {"lm_head": {}, "model.embed_tokens": {}},
        )


def test_an_untied_head_keeps_the_name_it_was_quantized_under(driver: Any) -> None:
    """A model with two real tables publishes two, and the rename must not fire on it."""
    banked = {"lm_head": {}}
    assert (
        driver.under_the_input_table(
            _head_and_table(config_says=False, share=False, equal=False), banked
        )
        == banked
    )


def _checkpoint(tmp_path: Any, **fields: Any) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "config.json").write_text(json.dumps(fields), encoding="utf-8")
    return str(tmp_path)


def test_the_export_reads_the_checkpoints_config_and_not_the_recipes(
    driver: Any, tmp_path: Any
) -> None:
    """``model.config`` after a recipe is a description of a checkpoint nobody is writing.

    ``oneshot`` attaches a compressed-tensors ``quantization_config`` to it and clears
    ``tie_word_embeddings``. Handing that object to the model the publish path exports from
    produces two untied tables and a directory claiming a quantization format it is not in.
    The checkpoint on disk still says what the architecture actually is, so it is re-read.
    """
    source = _checkpoint(tmp_path, model_type="gpt2", tie_word_embeddings=True)

    config = driver.pristine_config(source)

    assert config.tie_word_embeddings is True
    assert getattr(config, "quantization_config", None) is None


def test_the_activation_linearize_moe_reads_is_supplied_without_touching_the_checkpoint(
    driver: Any, tmp_path: Any
) -> None:
    """LFM2's config omits ``hidden_act``; the expert modules built from it read one.

    Writing the field into the checkpoint would edit a directory five other arms load their
    weights from, so it is set on the in-memory object and nowhere else. The second half
    matters as much as the first: an architecture that *does* declare an activation must
    keep the one it declared rather than silently be rebuilt as SiLU.
    """
    source = _checkpoint(tmp_path, model_type="gpt2", tie_word_embeddings=True)
    before = (tmp_path / "config.json").read_bytes()

    assert driver.pristine_config(source).hidden_act == "silu"
    assert (tmp_path / "config.json").read_bytes() == before

    declared = _checkpoint(tmp_path / "llama", model_type="llama", hidden_act="gelu")
    assert driver.pristine_config(declared).hidden_act == "gelu"
