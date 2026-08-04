"""IFEval instruction verifiers and the scoring arithmetic around them.

The scorer is the whole benchmark. IFEval has no gold answers -- a response is correct
if and only if a Python function says the constraints were met -- so a verifier that is
subtly wrong does not produce an error, it produces a *number*, and that number then
flows into a paired comparison and out into a claim about what 3-bit quantization cost.

So the 25 verifiers are each tested against a response that satisfies them and one that
does not, and the fidelity details that look like bugs (``keywords:existence`` matching
without word boundaries; ``length_constraints:number_paragraphs`` rejecting an empty
*interior* paragraph but tolerating empty ends) are pinned deliberately, because those
are precisely the lines a future reader will be tempted to "fix" into producing numbers
that are no longer IFEval's.

No model is loaded. Generation is stubbed; what is under test is the prompt, the
verifiers and the tallies.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from dynquant.errors import DynQuantError
from dynquant.eval._ifeval_instructions import (
    CAPABILITY_LANGDETECT,
    INSTRUCTION_IDS,
    Backends,
    backends,
    build_checker,
    count_words,
    missing_capabilities,
    scorer_fingerprint,
)
from dynquant.eval.harness import EvalConfig
from dynquant.eval.ifeval import (
    IfevalExample,
    build_prompt,
    evaluate_ifeval,
    load_ifeval,
    loose_variants,
)

# --------------------------------------------------------------------------
# The 25 verifiers
# --------------------------------------------------------------------------

# (instruction_id, kwargs, a response that follows it, a response that does not).
# Every registered id must appear here -- see `test_every_instruction_id_is_covered`,
# which is what stops a new id being added with no test behind it.
CASES: list[tuple[str, dict[str, Any], str, str]] = [
    (
        "keywords:existence",
        {"keywords": ["quantization", "kernel"]},
        "Quantization needs a kernel.",
        "Quantization alone.",
    ),
    (
        "keywords:frequency",
        {"keyword": "bit", "frequency": 2, "relation": "at least"},
        "bit and bit",
        "one bit only",
    ),
    (
        "keywords:forbidden_words",
        {"forbidden_words": ["float"]},
        "Everything here is integer.",
        "A float sneaks in.",
    ),
    (
        "keywords:letter_frequency",
        {"letter": "z", "let_frequency": 3, "let_relation": "at least"},
        "zzz",
        "zz",
    ),
    (
        "length_constraints:number_sentences",
        {"num_sentences": 2, "relation": "at least"},
        "One thing. Then another thing.",
        "Only one thing here",
    ),
    (
        "length_constraints:number_paragraphs",
        {"num_paragraphs": 2},
        "First half. *** Second half.",
        "First half. *** Second half. *** Third half.",
    ),
    (
        "length_constraints:number_words",
        {"num_words": 4, "relation": "less than"},
        "just three words",
        "this one has five words",
    ),
    (
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "second"},
        "Opening text.\n\nSecond paragraph starts here.",
        "Opening text.\n\nWrong opener here.",
    ),
    (
        "detectable_content:number_placeholders",
        {"num_placeholders": 2},
        "Send it to [address] on [date].",
        "Send it to [address].",
    ),
    (
        "detectable_content:postscript",
        {"postscript_marker": "P.S."},
        "The answer.\nP.S. one more thing",
        "The answer, with nothing after it.",
    ),
    (
        "detectable_format:number_bullet_lists",
        {"num_bullets": 2},
        "* first\n* second",
        "* first\n* second\n* third",
    ),
    (
        "detectable_format:constrained_response",
        {},
        "My answer is yes.",
        "Yes, definitely.",
    ),
    (
        "detectable_format:number_highlighted_sections",
        {"num_highlights": 2},
        "*one* and *two*",
        "*one* only",
    ),
    (
        "detectable_format:multiple_sections",
        {"section_spliter": "Section", "num_sections": 2},
        "Section 1 intro Section 2 body",
        "Section 1 intro only",
    ),
    (
        "detectable_format:json_format",
        {},
        '{"bits": 3}',
        "bits: 3",
    ),
    (
        "detectable_format:title",
        {},
        "<<A Real Title>>\nBody text.",
        "A Real Title\nBody text.",
    ),
    (
        "combination:two_responses",
        {},
        "first answer\n******\nsecond answer",
        "first answer\n******\nfirst answer",
    ),
    (
        "combination:repeat_prompt",
        {"prompt_to_repeat": "Explain quantization."},
        "Explain quantization. It maps floats to integers.",
        "It maps floats to integers.",
    ),
    (
        "startend:end_checker",
        {"end_phrase": "Any other questions?"},
        "That is all. Any other questions?",
        "That is all.",
    ),
    (
        "startend:quotation",
        {},
        '"the whole answer"',
        "the whole answer",
    ),
    (
        "change_case:capital_word_frequency",
        {"capital_frequency": 2, "capital_relation": "at least"},
        "GPTQ and AWQ compared",
        "GPTQ and awq compared",
    ),
    (
        "punctuation:no_comma",
        {},
        "no commas at all here",
        "commas, unfortunately",
    ),
]

LANGUAGE_CASES: list[tuple[str, dict[str, Any], str, str]] = [
    (
        "language:response_language",
        {"language": "fr"},
        "Bonjour, ceci est une réponse écrite entièrement en langue française.",
        "This response is written entirely in the English language instead.",
    ),
    (
        "change_case:english_capital",
        {},
        "THIS ENTIRE RESPONSE IS WRITTEN IN CAPITAL ENGLISH LETTERS THROUGHOUT.",
        "this entire response is written in lowercase english letters throughout.",
    ),
    (
        "change_case:english_lowercase",
        {},
        "this entire response is written in lowercase english letters throughout.",
        "THIS ENTIRE RESPONSE IS WRITTEN IN CAPITAL ENGLISH LETTERS THROUGHOUT.",
    ),
]

_HAS_LANGDETECT = CAPABILITY_LANGDETECT in backends().capabilities


@pytest.mark.parametrize(
    ("instruction_id", "kwargs", "good", "bad"), CASES, ids=[case[0] for case in CASES]
)
def test_verifier_separates_a_following_response_from_a_violating_one(
    instruction_id: str, kwargs: dict[str, Any], good: str, bad: str
) -> None:
    """Each verifier accepts what it should and rejects what it should.

    Both directions, always. A verifier that returned ``True`` unconditionally would
    pass a one-sided test and would silently award every model full marks on that
    constraint -- which reads as "quantization did no damage", the exact conclusion
    this campaign exists to test.
    """
    checker = build_checker(instruction_id, kwargs)
    assert checker(good) is True
    assert checker(bad) is False


@pytest.mark.skipif(not _HAS_LANGDETECT, reason="language constraints need langdetect")
@pytest.mark.parametrize(
    ("instruction_id", "kwargs", "good", "bad"),
    LANGUAGE_CASES,
    ids=[case[0] for case in LANGUAGE_CASES],
)
def test_language_verifier_separates_when_langdetect_is_present(
    instruction_id: str, kwargs: dict[str, Any], good: str, bad: str
) -> None:
    """The three langdetect-backed verifiers, when the dependency is installed."""
    checker = build_checker(instruction_id, kwargs)
    assert checker(good) is True
    assert checker(bad) is False


def test_every_instruction_id_is_covered_by_a_case() -> None:
    """The case table covers the registry exactly.

    Turns red when: an instruction id is added to the registry without a case, or a
    case names an id the registry does not have. The first is how an untested verifier
    ships; the second is how a renamed id silently stops being scored, since the
    dataset would then hit the unknown-id error at load.
    """
    covered = {case[0] for case in CASES} | {case[0] for case in LANGUAGE_CASES}
    assert covered == INSTRUCTION_IDS


def test_unknown_instruction_id_is_refused() -> None:
    """A dataset revision that adds a constraint type must not score as vacuously met."""
    with pytest.raises(DynQuantError, match="unknown IFEval instruction"):
        build_checker("detectable_format:haiku", {})


def test_missing_argument_names_the_instruction() -> None:
    """Turns red when: a verifier starts defaulting a missing argument.

    A ``num_words`` that quietly defaults to 0 makes "at least N words" true for every
    response, which is a full-marks constraint that nobody can see from the score.
    """
    with pytest.raises(DynQuantError, match="num_words"):
        build_checker("length_constraints:number_words", {"relation": "at least"})


def test_relation_outside_the_two_allowed_values_is_refused() -> None:
    with pytest.raises(DynQuantError, match="expected one of"):
        build_checker(
            "length_constraints:number_words", {"num_words": 3, "relation": "greater than"}
        )


def test_invalid_interpolated_pattern_names_its_instruction() -> None:
    """The section spliter goes into a regex raw, as upstream does.

    Turns red when: the guarded compile is replaced by a bare ``re.compile``. The
    failure still happens either way -- but at build time with the instruction named,
    rather than mid-scoring with a bare ``re.error``.
    """
    with pytest.raises(DynQuantError, match="invalid pattern"):
        build_checker(
            "detectable_format:multiple_sections", {"section_spliter": "Sec(", "num_sections": 2}
        )


# --------------------------------------------------------------------------
# Fidelity details that look like bugs
# --------------------------------------------------------------------------


def test_existence_matches_inside_words_but_forbidden_words_does_not() -> None:
    """The word-boundary asymmetry is upstream's, and it is load-bearing.

    ``keywords:existence`` searches the bare pattern, so "bat" is found inside
    "batting". ``keywords:forbidden_words`` anchors on word boundaries, so forbidding
    "bat" does not fail a response for saying "debate". Symmetrising either direction
    changes real scores: the first would get stricter, the second more forgiving.

    Turns red when: someone unifies the two, which is the obvious tidy-up.
    """
    assert build_checker("keywords:existence", {"keywords": ["bat"]})("batting order") is True
    assert build_checker("keywords:forbidden_words", {"forbidden_words": ["bat"]})("debate") is True


def test_empty_paragraph_at_an_end_is_tolerated_but_in_the_middle_is_not() -> None:
    """A leading or trailing ``***`` is a stray separator; an interior one is malformed.

    Turns red when: the paragraph splitter starts dropping every empty run, which would
    silently pass responses that emitted two separators in a row.
    """
    checker = build_checker("length_constraints:number_paragraphs", {"num_paragraphs": 2})
    assert checker("*** first *** second") is True
    assert checker("first *** *** second") is False


def test_nth_paragraph_indexes_the_unfiltered_list() -> None:
    """Blank paragraphs do not count toward the total but still occupy a position.

    Reproduced from upstream, where the count skips blanks and the index does not.
    Turns red when: the index is changed to walk the filtered list, which looks like
    the obvious repair and shifts every ``nth_paragraph`` verdict on responses with a
    blank line in them.
    """
    checker = build_checker(
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "second"},
    )
    # Three slots, one blank: two real paragraphs, but slot 2 is the blank one.
    assert checker("First.\n\n\n\nSecond paragraph.") is False


def test_word_count_is_the_upstream_regex_exactly() -> None:
    """``count_words`` needs no tokenizer substitute, unlike sentence splitting.

    Upstream uses ``RegexpTokenizer(r"\\w+")``, which is ``re.findall`` -- so the most
    common length constraint in the benchmark is bit-identical with or without NLTK,
    and the fallback splitter cannot move it.
    """
    assert count_words("don't count-hyphens as one") == 6
    assert count_words("") == 0


def test_fallback_sentence_splitter_keeps_abbreviations_and_initials_intact() -> None:
    """Only meaningful when NLTK is absent, which is the common case here.

    Turns red when: the abbreviation list is dropped. "Dr. Smith arrived. He left."
    would then count as four sentences, and every "at least N sentences" constraint
    would get easier by a factor of about two.
    """
    if backends().sentence_backend != "regex-sentences":
        pytest.skip("nltk punkt is installed; the fallback splitter is not in use")
    split = backends().split_sentences
    assert len(split("Dr. Smith arrived. He left.")) == 2
    assert len(split("J. R. R. Tolkien wrote it. Then he stopped.")) == 2
    # The list is deliberately short: "no." is a real sentence ending far more often
    # than it is an abbreviation, so it must still split.
    assert len(split("The answer is no. Then we moved on.")) == 2


# --------------------------------------------------------------------------
# Loose scoring
# --------------------------------------------------------------------------


def test_loose_variants_are_the_eight_upstream_rewrites() -> None:
    variants = loose_variants("Sure, here it is!\n*bold* body\nHope that helps")
    assert len(variants) == 8
    assert variants[0] == "Sure, here it is!\n*bold* body\nHope that helps"
    assert "*" not in variants[1]
    assert variants[2].startswith("*bold*")
    assert variants[3].endswith("body")
    assert variants[4] == "*bold* body"


def test_loose_forgives_a_preamble_that_strict_does_not() -> None:
    """The one behaviour that distinguishes the two headline metrics.

    A chat model opening with "Sure, here it is:" violates ``startend:quotation``
    outright; dropping the first line reveals a correctly quoted answer underneath.
    Turns red when: strict starts applying the rewrites, which would collapse the two
    metrics into one and inflate the strict number by roughly ten points.
    """
    example = _example(1, ("startend:quotation",), ({},))
    result = _score(['Sure, here it is:\n"the answer"'], [example])
    assert result.hits == [False]
    assert result.loose_hits == [True]


def test_an_empty_generation_follows_nothing() -> None:
    """Turns red when: a blank response starts passing constraints vacuously.

    ``punctuation:no_comma`` is true of the empty string and ``keywords:frequency``
    with "less than" is too, so a model that has collapsed into silence would otherwise
    score *above* one that answers imperfectly.
    """
    example = _example(1, ("punctuation:no_comma",), ({},))
    result = _score(["   \n  "], [example])
    assert result.hits == [False]
    assert result.loose_hits == [False]
    assert result.empty == 1


# --------------------------------------------------------------------------
# Tallies and pairing
# --------------------------------------------------------------------------


def test_prompt_level_needs_every_constraint_and_instruction_level_counts_each() -> None:
    """The two granularities are different questions about the same generation.

    One prompt, two constraints, one met: prompt-level scores 0, instruction-level
    scores a half. Turns red when: prompt-level starts counting partial credit, which
    would make the headline number look like the instruction-level one and roughly
    ten points too high.
    """
    example = _example(
        7,
        ("punctuation:no_comma", "length_constraints:number_words"),
        ({}, {"num_words": 50, "relation": "at least"}),
    )
    result = _score(["short answer without punctuation"], [example])

    assert result.hits == [False]
    assert result.accuracy == 0.0
    assert result.instruction_total == 2
    assert result.instruction_strict == 1
    assert result.instruction_strict_accuracy == 0.5
    assert result.instruction_hits == [[True, False]]


def test_hits_are_one_per_prompt_and_flat_hits_one_per_constraint() -> None:
    """The two paired vectors line up with what a McNemar test needs.

    ``hits`` pairs prompt-for-prompt; ``flat_instruction_hits`` pairs
    constraint-for-constraint and is the narrower test when a change moves a handful of
    constraints rather than whole prompts. Both are only valid if their lengths are
    properties of the *dataset*, which is what this pins.
    """
    examples = [
        _example(1, ("punctuation:no_comma",), ({},)),
        _example(
            2,
            ("punctuation:no_comma", "startend:quotation"),
            ({}, {}),
        ),
    ]
    result = _score(["clean", '"quoted"'], examples)

    assert len(result.hits) == len(examples)
    assert len(result.flat_instruction_hits()) == result.instruction_total == 3
    assert result.keys == [1, 2]


def test_instruction_and_kwargs_length_mismatch_is_refused() -> None:
    """Turns red when: the zip stops being strict.

    A silent truncation would check "at least 3 paragraphs" against the word-count
    arguments and produce a plausible number from misaligned pairs -- the worst failure
    available here, because it looks like a result.
    """
    example = _example(1, ("punctuation:no_comma", "startend:quotation"), ({},))
    with pytest.raises(DynQuantError, match="argument sets"):
        _score(["anything"], [example])


def test_checkers_are_built_before_a_single_token_is_generated() -> None:
    """A malformed dataset must cost seconds, not a full generation pass.

    Turns red when: checker construction moves below the generation call. On the
    phase-3 models a full IFEval pass is tens of GPU-minutes per arm, so discovering a
    bad kwarg afterwards is the difference between a typo and a wasted evening.
    """
    generated = False

    def spy(*_args: Any, **_kwargs: Any) -> list[str]:
        nonlocal generated
        generated = True
        return [""]

    example = _example(1, ("length_constraints:number_words",), ({"relation": "at least"},))
    with pytest.raises(DynQuantError, match="num_words"):
        _score(["anything"], [example], generate=spy)
    assert not generated


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------


def test_a_model_with_a_chat_template_is_prompted_through_it() -> None:
    """An instruct checkpoint given bare text continues it instead of answering.

    Turns red when: the template is dropped. The resulting scores are near the floor
    for every arm, which reads as catastrophic quantization damage in a table where
    the bf16 control has fallen just as far.
    """
    example = _example(1, ("punctuation:no_comma",), ({},))
    assert build_prompt(example, _tokenizer(chat_template="x")) == "<|user|>Write it.<|assistant|>"
    assert build_prompt(example, _tokenizer(chat_template=None)) == "Write it."


def test_prompt_style_is_recorded_on_the_result() -> None:
    """Base and instruct checkpoints are prompted differently, and that is not hidden.

    Turns red when: the field stops tracking the tokenizer. Two arms with different
    prompt styles are not comparable, and nothing in the accuracies themselves says so.
    """
    example = _example(1, ("punctuation:no_comma",), ({},))
    assert _score(["clean"], [example]).prompt_style == "chat-template"
    assert (
        _score(["clean"], [example], tokenizer=_tokenizer(chat_template=None)).prompt_style == "raw"
    )


def test_a_chat_template_forces_add_special_tokens_off() -> None:
    """The template already emits BOS; the tokenizer would emit a second one.

    Turns red when: the override is removed. Llama-3 and Gemma-3 prompts then carry a
    duplicated BOS, which costs a few points of instruction following, reports no
    error, and is the same size as the effect being measured.
    """
    seen: dict[str, Any] = {}

    def spy(
        _model: Any, _tok: Any, prompts: list[str], config: EvalConfig, **_kw: Any
    ) -> list[str]:
        seen["add_special_tokens"] = config.add_special_tokens
        return ["clean"] * len(prompts)

    example = _example(1, ("punctuation:no_comma",), ({},))
    _score(["clean"], [example], config=EvalConfig(add_special_tokens=True), generate=spy)
    assert seen["add_special_tokens"] is False


def test_default_config_leaves_room_for_length_constraints() -> None:
    """Turns red when: ``max_new_tokens`` drops toward a classification-sized budget.

    Many IFEval constraints are "at least 400 words". A cap that truncates the answer
    scores the cap, and it scores it identically for every arm, which flattens the
    comparison the campaign is built on.
    """
    from dynquant.eval.ifeval import DEFAULT_CONFIG

    assert DEFAULT_CONFIG.max_new_tokens >= 1024
    assert DEFAULT_CONFIG.add_special_tokens is False
    assert DEFAULT_CONFIG.stop_sequences == ()


# --------------------------------------------------------------------------
# Unverifiable constraints
# --------------------------------------------------------------------------


def test_missing_langdetect_refuses_to_produce_a_number(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither guess is acceptable, so the default is to refuse.

    Counting an uncheckable constraint as followed inflates; counting it as violated
    deflates. Both produce a number that looks ordinary. Turns red when: the default
    becomes a silent fallback.
    """
    _without_langdetect(monkeypatch)
    example = _example(1, ("language:response_language",), ({"language": "fr"},))
    with pytest.raises(DynQuantError, match="langdetect"):
        _score(["bonjour"], [example])


def test_dropping_unverifiable_prompts_records_which_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The escape hatch stays auditable.

    Turns red when: dropped keys stop being recorded. A dropped run is comparable only
    against another run that dropped the same prompts, and the accuracies alone give no
    hint that anything was removed.
    """
    _without_langdetect(monkeypatch)
    examples = [
        _example(1, ("language:response_language",), ({"language": "fr"},)),
        _example(2, ("punctuation:no_comma",), ({},)),
    ]
    # One generation, because only one prompt survives the drop -- the stub is
    # aligned to the prompts that were actually scored, as the real harness is.
    result = _score(["clean"], examples, on_unverifiable="drop")

    assert result.dropped == [1]
    assert result.total == 1
    assert result.keys == [2]
    assert result.hits == [True]


def test_missing_capabilities_only_reports_what_is_actually_needed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _without_langdetect(monkeypatch)
    assert missing_capabilities(["punctuation:no_comma"]) == frozenset()
    assert missing_capabilities(["change_case:english_lowercase"]) == {CAPABILITY_LANGDETECT}
    assert missing_capabilities([]) == frozenset()


def test_scorer_fingerprint_names_every_substitutable_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two arms scored under different fingerprints are not comparable.

    Turns red when: a backend is added without reaching the fingerprint. That is how a
    machine with NLTK and a machine without it end up in the same results table,
    differing by a sentence-splitting rule that nothing in the numbers reveals.
    """
    _without_langdetect(monkeypatch)
    fingerprint = scorer_fingerprint()
    assert fingerprint.startswith("ifeval/")
    assert "no-langdetect" in fingerprint
    assert backends().sentence_backend in fingerprint
    assert backends().word_backend in fingerprint


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_load_strips_the_null_padding_the_published_dataset_carries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrow pads every row to the union of all argument names, filling with ``None``.

    Turns red when: the strip is removed. A ``None`` reaching a verifier hits the
    missing-argument check and raises -- so this fails loudly rather than quietly, but
    it fails on every real dataset row, which is worth catching here instead of on the
    box.
    """
    rows = [
        {
            "key": 1001,
            "prompt": "Write it.",
            "instruction_id_list": ["punctuation:no_comma"],
            "kwargs": [{"num_words": None, "relation": None, "keywords": None}],
        }
    ]
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *_a, **_k: rows  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake)

    (example,) = load_ifeval()
    assert example.key == 1001
    assert example.instruction_ids == ("punctuation:no_comma",)
    assert example.kwargs == ({},)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _example(
    key: int, instruction_ids: tuple[str, ...], kwargs: tuple[dict[str, Any], ...]
) -> IfevalExample:
    return IfevalExample(
        key=key, prompt="Write it.", instruction_ids=instruction_ids, kwargs=kwargs
    )


def _tokenizer(*, chat_template: str | None) -> Any:
    """A tokenizer that renders a turn, or one that refuses to.

    ``chat_template=None`` makes ``apply_chat_template`` *raise*, which is what a real
    template-less tokenizer does -- transformers will not invent a turn structure for a
    base checkpoint. The double used to keep rendering while reporting no template,
    which no tokenizer does in either direction, and which mattered once the harness
    started deciding by asking rather than by reading the attribute.
    """

    class _Tok:
        def __init__(self) -> None:
            self.chat_template = chat_template

        def apply_chat_template(
            self, messages: list[dict[str, str]], *, tokenize: bool, add_generation_prompt: bool
        ) -> str:
            assert tokenize is False
            assert add_generation_prompt is True
            if chat_template is None:
                raise ValueError("tokenizer.chat_template is not set")
            return f"<|user|>{messages[0]['content']}<|assistant|>"

    return _Tok()


def _score(
    generations: list[str],
    examples: list[IfevalExample],
    *,
    tokenizer: Any = None,
    config: EvalConfig | None = None,
    generate: Any = None,
    on_unverifiable: str | None = None,
) -> Any:
    """Run the scorer over canned generations, with no model in sight.

    ``on_unverifiable`` defaults to *not passing the argument at all* rather than to
    repeating ``"raise"`` here. Restating the production default in the harness makes
    the refusal test pass no matter what the production default becomes, which is the
    one thing that test exists to notice.
    """
    import dynquant.eval.ifeval as module

    stub = generate or (lambda *_a, **_k: list(generations))
    extra = {} if on_unverifiable is None else {"on_unverifiable": on_unverifiable}
    original = module.generate_batched
    module.generate_batched = stub  # type: ignore[assignment]
    try:
        return evaluate_ifeval(
            model=None,
            tokenizer=tokenizer or _tokenizer(chat_template="x"),
            examples=examples,
            label="stub",
            config=config,
            **extra,  # type: ignore[arg-type]
        )
    finally:
        module.generate_batched = original  # type: ignore[assignment]


def _without_langdetect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the no-langdetect branch regardless of what is installed here.

    The dependency is optional, so the interesting branch is unreachable on a machine
    that has it -- and CI has it. Overriding the cached resolution is the only way to
    test the refusal path on the machine where it matters least.
    """
    resolved = backends()
    monkeypatch.setattr(
        "dynquant.eval._ifeval_instructions.backends",
        lambda: Backends(
            detect_language=None,
            split_sentences=resolved.split_sentences,
            sentence_backend=resolved.sentence_backend,
            tokenize_words=resolved.tokenize_words,
            word_backend=resolved.word_backend,
        ),
    )
