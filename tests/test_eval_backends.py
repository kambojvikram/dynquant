"""Scoring a task through a serving engine instead of ``model.generate``.

The campaign this exists for evaluates ~32 checkpoints through vLLM and compares the
numbers against the ``transformers`` arm. That comparison is only about the runtime if
everything else is identical, and the ways two engines quietly disagree about a
*prompt* all produce a plausible score rather than an error: one prepends its own BOS,
one truncates from the other end, one returns completions in a different order than it
was handed them. None of those raises. All of them land in a results table.

So the assertions here are about the boundary, not about generation quality. The fake
engine below returns whatever it is told to, in whatever order it is told to, and
records exactly what it was asked -- which is the only way to state "the vLLM arm saw
the same tokens as the transformers arm" as a test rather than as a hope.

vLLM itself is not installed on the development box and will not be on CI, so the
engine is faked at its documented surface: ``generate(prompts=[{"prompt_token_ids":
...}], sampling_params=..., use_tqdm=...)`` returning objects with ``request_id``,
``prompt_token_ids`` and ``outputs[0].token_ids``. That fake is a claim about vLLM's
API, and it is the one thing here a vLLM release could invalidate silently -- which is
why the real check is the S0/G4 gate script, run once against a real engine.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import pytest

torch = pytest.importorskip("torch")
# And transformers, which is *not* obvious from the imports below: nothing here names
# it. `_pair` drives both arms of the comparison, and the transformers arm reaches
# `greedy_generation_config`, which imports `GenerationConfig` at call time. The `test`
# CI job deliberately installs no transformers -- that is how it proves dynquant-core
# installs on a box that will never fine-tune -- so without this the whole file failed
# there with a bare ModuleNotFoundError raised from inside the harness.
#
# Skipping is only acceptable because it cannot become the silent kind: the
# `transformers-lines` job lists this file among those that must *run*, and fails if
# any of them skips. See .github/workflows/ci.yml.
transformers = pytest.importorskip("transformers")

from _decode_stub import StubModel, StubTokenizer  # noqa: E402

from dynquant.errors import DynQuantError  # noqa: E402
from dynquant.eval.backends import VllmBackend  # noqa: E402
from dynquant.eval.harness import EvalConfig, encode_prompts, generate_batched  # noqa: E402

# --------------------------------------------------------------------------
# A stand-in for vllm.LLM, plus the `vllm` module SamplingParams comes from
# --------------------------------------------------------------------------


@dataclass
class FakeCompletion:
    token_ids: list[int]


@dataclass
class FakeOutput:
    request_id: str
    prompt_token_ids: list[int]
    outputs: list[FakeCompletion]


@dataclass
class FakeEngine:
    """Replies with a fixed continuation, and records every request.

    ``shuffle`` and ``first_request_id`` exist because both are real vLLM behaviours
    a caller must survive: the engine's request counter persists across ``generate``
    calls, so the second call's ids do not start at zero, and nothing in the API
    contract promises the returned list is in submission order.

    ``echo`` makes the reply depend on the prompt -- the first token of it. A fixed
    reply cannot show a reordering bug, because every answer is then correct wherever
    it lands, which is the same reason a misalignment in the real thing produces a
    plausible results table instead of an error.
    """

    reply: list[int]
    shuffle: bool = False
    echo: bool = False
    first_request_id: int = 0
    seen: list[dict[str, Any]] = field(default_factory=list)

    def generate(self, *, prompts, sampling_params, use_tqdm):
        self.seen.append(
            {"prompts": prompts, "sampling_params": sampling_params, "use_tqdm": use_tqdm}
        )
        outputs = [
            FakeOutput(
                request_id=str(self.first_request_id + index),
                prompt_token_ids=list(prompt["prompt_token_ids"]),
                outputs=[
                    FakeCompletion(
                        token_ids=[prompt["prompt_token_ids"][0]] if self.echo else list(self.reply)
                    )
                ],
            )
            for index, prompt in enumerate(prompts)
        ]
        return list(reversed(outputs)) if self.shuffle else outputs


@dataclass
class FakeSamplingParams:
    """Every default here is deliberately *not* neutral, and deliberately not vLLM's.

    A fake that defaulted each penalty to the value the caller wants could not tell
    "pinned explicitly" from "not passed at all" -- the assertion would read 1.0 either
    way. The real ``SamplingParams`` does default these to the neutral values; that is
    precisely why they were left unpassed, and why a vLLM release moving one would have
    re-decoded the campaign with nothing to catch it.
    """

    n: int = 0
    temperature: float = 1.0
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.1
    presence_penalty: float = 0.5
    frequency_penalty: float = 0.5
    min_tokens: int = 8
    max_tokens: int | None = None
    stop: list[str] | None = None


@pytest.fixture(autouse=True)
def fake_vllm_module(monkeypatch):
    """``VllmBackend`` imports ``SamplingParams`` from ``vllm`` lazily, inside the call.

    Installing a stand-in module is what lets the greedy-decode assertions run at all
    on a box without vLLM; the import being lazy is itself part of the contract, since
    ``dynquant.eval`` has to stay importable there.
    """
    module = types.ModuleType("vllm")
    module.SamplingParams = FakeSamplingParams  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)
    return module


class Pair(NamedTuple):
    """Both arms of one comparison, with everything either of them touched."""

    direct: list[str]
    served: list[str]
    engine: FakeEngine
    model: StubModel
    tokenizer: StubTokenizer


def _pair(prompts, *, reply="7", **config_kwargs) -> Pair:
    """The same prompts through both backends, on one shared tokenizer.

    Shared deliberately: the stub's vocabulary is built lazily per instance, so two
    tokenizers assign different ids to the same word and any cross-arm comparison of
    ids would be meaningless.
    """
    tokenizer = StubTokenizer()
    config = EvalConfig(early_stop=False, **config_kwargs)

    direct_model = StubModel(tokenizer, reply)
    direct = generate_batched(direct_model, tokenizer, prompts, config)

    engine = FakeEngine(reply=tokenizer.encode_words(reply))
    served = generate_batched(VllmBackend(engine), tokenizer, prompts, config)
    return Pair(direct, served, engine, direct_model, tokenizer)


# --------------------------------------------------------------------------
# The boundary: identical prompts, identical answers
# --------------------------------------------------------------------------


def test_both_backends_are_handed_the_same_tokens() -> None:
    """The load-bearing claim of the whole vLLM eval path.

    Turns red when: the engine is sent strings instead of ids, or when either arm
    tokenizes with its own settings. Sending strings is the realistic regression --
    it is the shorter call -- and vLLM would then add its own BOS to a chat-templated
    prompt that already has one, for a few points of instruction following.
    """
    prompts = ["one Answer:", "a b c d Answer:", "mid size Answer:"]
    pair = _pair(prompts, batch_size=8)

    served = [
        pair.tokenizer.words_for(p["prompt_token_ids"]) for p in pair.engine.seen[0]["prompts"]
    ]
    assert served == [prompt.split() for prompt in prompts]

    # The transformers arm gets the same words, left-padded and length-sorted; strip
    # both and the two arms must be looking at the same prompts.
    direct = sorted([word for word in row if word] for row in pair.model.fed)
    assert direct == sorted(served)


def test_the_two_backends_produce_the_same_strings() -> None:
    """Same tokens in, same tokens out, so the decode cannot differ either.

    Turns red when: the vLLM path detokenizes with the engine's own settings rather
    than through the harness -- ``skip_special_tokens`` left on the default would
    leave ``<eos>`` in every answer, which the answer extractors then read past.
    """
    pair = _pair(["one Answer:", "a b c Answer:"], reply="42")
    assert pair.direct == pair.served == ["42", "42"]


def test_prompts_too_long_are_cut_at_the_front_for_the_engine_too() -> None:
    """Truncation happens before the boundary, so no backend can skip it.

    Turns red when: ``VllmBackend`` is given raw prompts and left to apply
    ``max_prompt_tokens`` itself, which vLLM does not do -- it would either error on
    ``max_model_len`` or silently score a different prompt.
    """
    prompt = "exemplar filler filler QUESTION options Answer:"
    pair = _pair([prompt], max_prompt_tokens=3)

    sent = pair.engine.seen[0]["prompts"][0]["prompt_token_ids"]
    assert pair.tokenizer.words_for(sent) == ["QUESTION", "options", "Answer:"]


def test_an_empty_prompt_list_never_reaches_the_engine() -> None:
    engine = FakeEngine(reply=[1])
    assert generate_batched(VllmBackend(engine), StubTokenizer(), [], EvalConfig()) == []
    assert engine.seen == []


# --------------------------------------------------------------------------
# Ordering and alignment
# --------------------------------------------------------------------------


def test_completions_returned_out_of_order_are_put_back() -> None:
    """Nothing in vLLM's contract promises submission order.

    Turns red when: ``_in_request_order`` stops sorting. The failure it prevents is
    the expensive one -- every answer scored against another problem's gold label,
    for a complete and plausible results table.
    """
    tokenizer = StubTokenizer()
    prompts = ["first Answer:", "second Answer:", "third Answer:"]
    ids = encode_prompts(tokenizer, prompts, EvalConfig()).ids

    engine = FakeEngine(reply=[], shuffle=True, echo=True)
    out = VllmBackend(engine).generate_ids(ids, EvalConfig())

    assert [tokenizer.words_for(row) for row in out] == [["first"], ["second"], ["third"]]


def test_request_ids_that_do_not_start_at_zero_are_still_sorted() -> None:
    """``LLM.generate``'s counter persists, so a second call's ids start mid-stream.

    Turns red when: ordering is done by checking for a permutation of ``range(n)``
    and falling back to positional order otherwise -- which would silently stop
    sorting from the second task onwards, i.e. on every arm but the first.
    """
    tokenizer = StubTokenizer()
    prompts = ["first Answer:", "second Answer:"]
    ids = encode_prompts(tokenizer, prompts, EvalConfig()).ids

    engine = FakeEngine(reply=[], shuffle=True, echo=True, first_request_id=4096)
    out = VllmBackend(engine).generate_ids(ids, EvalConfig())

    assert [tokenizer.words_for(row) for row in out] == [["first"], ["second"]]


def test_a_completion_answering_the_wrong_prompt_is_an_error() -> None:
    """The direct alignment check, independent of any ordering rule.

    Turns red when: ``_assert_aligned`` stops comparing the echoed prompt ids. An
    engine that reordered in a way the sort could not undo would otherwise produce a
    full results table scored against the wrong labels.
    """
    tokenizer = StubTokenizer()
    ids = encode_prompts(tokenizer, ["first Answer:", "second Answer:"], EvalConfig()).ids

    class Misaligned(FakeEngine):
        def generate(self, **kwargs):
            outputs = super().generate(**kwargs)
            outputs[0].prompt_token_ids = [999, 998]
            return outputs

    engine = Misaligned(reply=tokenizer.encode_words("x"))
    with pytest.raises(DynQuantError, match="different prompt"):
        VllmBackend(engine).generate_ids(ids, EvalConfig())


def test_a_dropped_completion_is_an_error() -> None:
    """Fewer answers than prompts shifts every later answer by one.

    Turns red when: the count check goes away. It cannot be left to the caller --
    ``generate_batched`` checks too, but only after the ids have been zipped.
    """
    tokenizer = StubTokenizer()
    ids = encode_prompts(tokenizer, ["a Answer:", "b Answer:"], EvalConfig()).ids

    class Lossy(FakeEngine):
        def generate(self, **kwargs):
            return super().generate(**kwargs)[:1]

    with pytest.raises(DynQuantError, match="1 completions for 2 prompts"):
        VllmBackend(Lossy(reply=[1])).generate_ids(ids, EvalConfig())


# --------------------------------------------------------------------------
# Decoding parameters
# --------------------------------------------------------------------------


def test_the_engine_is_asked_for_greedy_decoding() -> None:
    """Sampling turns a two-point effect into noise across every arm at once.

    Turns red when: the temperature drifts off zero, or ``n`` off one. Neither raises
    and both produce a full results table; a sampled run of the same checkpoint just
    disagrees with itself by more than the effect being measured.
    """
    params = _pair(["a Answer:"], max_new_tokens=64).engine.seen[0]["sampling_params"]

    assert params.temperature == 0.0
    assert params.n == 1
    assert params.top_p == 1.0
    assert params.top_k == -1
    assert params.max_tokens == 64


def test_no_penalty_is_applied_on_the_engine_arm_either() -> None:
    """The other half of the 23-point gap, pinned on the side that was already right.

    The transformers arm was the broken one -- it merged the checkpoint's
    ``repetition_penalty: 1.1`` -- and vLLM was correct only because
    ``SamplingParams`` happens to default these to neutral. "Correct by default" is
    not a property a campaign can rest on: an engine release that changed one would
    move every vLLM-scored arm at once, and the parity gate would then read the
    disagreement as a fault in the transformers path.

    Turns red when: any of the three penalties, or ``min_tokens``, stops being passed.
    ``FakeSamplingParams`` defaults them all to non-neutral values so that dropping the
    argument is distinguishable from pinning it.
    """
    params = _pair(["a Answer:"]).engine.seen[0]["sampling_params"]

    assert params.repetition_penalty == 1.0
    assert params.presence_penalty == 0.0
    assert params.frequency_penalty == 0.0
    assert params.min_tokens == 0, "no floor on the answer length; eos may come at once"


def test_the_two_arms_pin_the_same_settings_under_their_two_spellings() -> None:
    """The arms are only comparable where their settings correspond.

    Written as an explicit mapping because the libraries agree on neither the names
    nor the value that means "off": ``min_new_tokens=None`` and ``min_tokens=0`` are
    the same statement, and ``top_k`` is disabled by -1 in vLLM and by the vocabulary
    size in transformers. Deriving one from the other would assume the correspondence
    this asserts.

    Turns red when: one arm's neutral value is changed without the other's -- the
    failure mode the runtime-parity gate exists to detect, caught here for free
    instead of after an hour of GPU time.
    """
    from dynquant.eval.harness import NEUTRAL_DECODE

    params = _pair(["a Answer:"]).engine.seen[0]["sampling_params"]
    correspondence = {
        "repetition_penalty": ("repetition_penalty", params.repetition_penalty),
        "num_return_sequences": ("n", params.n),
        "min_new_tokens": ("min_tokens", None if params.min_tokens == 0 else params.min_tokens),
    }
    for hf_field, (vllm_field, vllm_value) in correspondence.items():
        assert vllm_value == NEUTRAL_DECODE[hf_field], (
            f"vLLM's {vllm_field} disagrees with transformers' {hf_field}"
        )


def test_stop_sequences_reach_the_engine_only_as_a_speed_setting() -> None:
    """``early_stop`` off means decode the whole budget, in both backends.

    The score cannot move either way -- the harness cuts the decoded text at the
    earliest stop sequence regardless -- so this asserts the two arms spend their
    tokens the same way, not that they score the same.
    """
    tokenizer = StubTokenizer()
    engine = FakeEngine(reply=tokenizer.encode_words("x"))
    backend = VllmBackend(engine)
    ids = encode_prompts(tokenizer, ["a Answer:"], EvalConfig()).ids

    backend.generate_ids(ids, EvalConfig(stop_sequences=("\n\n",), early_stop=True))
    backend.generate_ids(ids, EvalConfig(stop_sequences=("\n\n",), early_stop=False))

    assert engine.seen[0]["sampling_params"].stop == ["\n\n"]
    assert engine.seen[1]["sampling_params"].stop is None


def test_the_engines_own_progress_bar_is_off() -> None:
    """A per-arm tqdm in a 32-arm sweep is noise; the harness reports progress."""
    assert _pair(["a Answer:"]).engine.seen[0]["use_tqdm"] is False


def test_every_prompt_goes_in_one_call_not_in_batches() -> None:
    """``EvalConfig.batch_size`` is a padded-tensor knob and must not throttle vLLM.

    Turns red when: the backend starts chunking by ``batch_size``. Continuous
    batching is most of why the engine was chosen; 32-request waves give most of it
    back, and nothing about the result would look wrong.
    """
    prompts = [f"prompt {i} Answer:" for i in range(70)]
    engine = _pair(prompts, batch_size=8).engine

    assert len(engine.seen) == 1
    assert len(engine.seen[0]["prompts"]) == 70


def test_progress_is_reported_once_the_engine_is_done() -> None:
    tokenizer = StubTokenizer()
    ids = encode_prompts(tokenizer, ["a Answer:", "b Answer:"], EvalConfig()).ids
    seen: list[tuple[int, int]] = []

    VllmBackend(FakeEngine(reply=[1])).generate_ids(
        ids, EvalConfig(), progress=lambda done, total: seen.append((done, total))
    )
    assert seen == [(2, 2)]


# --------------------------------------------------------------------------
# Substitutability
# --------------------------------------------------------------------------


def test_a_task_cannot_tell_which_backend_it_was_given() -> None:
    """The property that keeps every task scorable through every runtime.

    Turns red when: a task starts reaching past ``generate_batched`` for
    ``model.device``, ``model.config`` or ``model.generate`` -- at which point the
    vLLM arm would need its own copy of that task, and the two copies would drift.
    """
    from dynquant.eval.casehold import CaseholdExample, evaluate_casehold

    examples = [
        CaseholdExample(
            citing_prompt=f"ctx {n}", holdings=("h0", "h1", "h2", "h3", "h4"), answer="1"
        )
        for n in ("one", "two")
    ]
    tokenizer = StubTokenizer()
    config = EvalConfig(early_stop=False, max_new_tokens=4)

    direct = evaluate_casehold(
        StubModel(tokenizer, "1"), tokenizer, examples, label="direct", config=config
    )
    served = evaluate_casehold(
        VllmBackend(FakeEngine(reply=tokenizer.encode_words("1"))),
        tokenizer,
        examples,
        label="served",
        config=config,
    )

    assert direct.hits == served.hits
    assert direct.accuracy == served.accuracy == 1.0
