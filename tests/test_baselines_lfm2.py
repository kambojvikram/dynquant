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
