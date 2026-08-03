"""The 25 IFEval instruction verifiers.

Ported from ``google-research/instruction_following_eval/instructions.py``, which is
the definition of the benchmark rather than one implementation of it: the prompts say
"write at least 3 paragraphs" but what *counts* as a paragraph is whatever that file
says it is. So the port is deliberately literal, down to the regexes, including the
places where the original is arguably wrong (``keywords:existence`` searches without
word boundaries while ``keywords:forbidden_words`` searches with them). Fixing those
would produce numbers that are not IFEval numbers.

Two departures, both forced, both recorded in the scorer fingerprint so a run scored
one way cannot be silently compared against a run scored the other:

**Sentence splitting.** The original calls NLTK's trained ``punkt`` tokenizer. NLTK is
not a dependency here and ``punkt`` is a runtime *download*, which would put a network
call in the middle of a multi-hour evaluation. If NLTK and its data are importable we
use them and match the original exactly; otherwise a regex splitter with an
abbreviation list stands in. It disagrees with ``punkt`` on constructions like
"Dr. Smith arrived." -- rare in IFEval responses, but not never.

**Word tokenizing** for ``change_case:capital_word_frequency``, same story. Note that
the far more common ``length_constraints:number_words`` needs no substitute: the
original uses ``RegexpTokenizer(r"\\w+")``, which is exactly :func:`re.findall`, so
word counts are bit-identical either way.

Neither substitution threatens the campaign's claims, and it is worth being precise
about why rather than waving at it. Every claim rests on a *paired difference* between
two arms scored by the same scorer on the same prompts; a splitter that miscounts
"Dr. Smith" miscounts it identically for both arms, so the difference is untouched. It
is the *absolute* number that stops being leaderboard-comparable, which is why
:attr:`~dynquant.eval.ifeval.IfevalResult.scorer` is printed next to it.

**Missing ``langdetect``** is handled differently, because it is not an approximation:
three instruction types cannot be scored at all without it. Rather than guess, those
are reported as unverifiable and :func:`~dynquant.eval.ifeval.evaluate_ifeval` refuses
to fold them into a score. Counting them as followed inflates; counting them as
violated deflates; both look like results.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from dynquant._logging import get_logger
from dynquant.errors import DynQuantError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

__all__ = [
    "CAPABILITY_LANGDETECT",
    "INSTRUCTION_IDS",
    "Checker",
    "backends",
    "build_checker",
    "count_words",
    "missing_capabilities",
    "requirements_for",
    "scorer_fingerprint",
]

_log = get_logger(__name__)

CAPABILITY_LANGDETECT = "langdetect"
"""The only hard capability. Named rather than inlined because it appears in the
registry, in the error message, and in the fingerprint, and those must agree."""

_RELATIONS = ("less than", "at least")


@dataclass(frozen=True, slots=True)
class Checker:
    """One verifiable instruction, bound to its arguments.

    Callable rather than a bare closure so a failing instruction can be named in a
    prediction dump. When 3-bit Phi-4-mini drops eight points, the question is
    immediately *which* instructions it stopped following -- length constraints degrade
    long before JSON formatting does, and the two mean different things.
    """

    instruction_id: str
    predicate: Callable[[str], bool]

    def __call__(self, response: str) -> bool:
        return self.predicate(response)


# --------------------------------------------------------------------------
# Optional backends
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Backends:
    """Which implementation of each fuzzy operation this process resolved to."""

    detect_language: Callable[[str], str] | None
    split_sentences: Callable[[str], list[str]]
    sentence_backend: str
    tokenize_words: Callable[[str], list[str]]
    word_backend: str

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({CAPABILITY_LANGDETECT}) if self.detect_language else frozenset()


@lru_cache(maxsize=1)
def backends() -> Backends:
    """Resolve the optional backends once per process.

    Cached because the probing is not free and, more importantly, because a scorer
    that could change its mind partway through a run would produce a result set whose
    items were scored under different rules.
    """
    detect = _load_langdetect()
    split, sentence_backend = _load_sentence_splitter()
    tokenize, word_backend = _load_word_tokenizer()
    if detect is None:
        _log.info(
            "langdetect is not installed; %d IFEval instruction types cannot be scored "
            "(install it with `pip install langdetect`)",
            sum(1 for spec in _REGISTRY.values() if spec.requires),
        )
    return Backends(
        detect_language=detect,
        split_sentences=split,
        sentence_backend=sentence_backend,
        tokenize_words=tokenize,
        word_backend=word_backend,
    )


def scorer_fingerprint() -> str:
    """A short string identifying the scoring rules in force.

    Recorded on every result. Two arms scored under different fingerprints are not
    comparable, and the fingerprint is the only thing that makes that visible after
    the run -- the accuracies themselves look perfectly ordinary.
    """
    resolved = backends()
    langdetect = "langdetect" if resolved.detect_language else "no-langdetect"
    return f"ifeval/{resolved.sentence_backend}+{resolved.word_backend}+{langdetect}"


def _load_langdetect() -> Callable[[str], str] | None:
    try:
        import langdetect
    except ImportError:
        return None

    def detect(text: str) -> str:
        # Seeded, because langdetect samples internally and is otherwise
        # non-deterministic -- the same response can score differently on two runs of
        # the same arm, which would show up as quantization noise that is not there.
        langdetect.DetectorFactory.seed = 0
        return str(langdetect.detect(text))

    return detect


def _load_sentence_splitter() -> tuple[Callable[[str], list[str]], str]:
    try:
        import nltk

        probe = nltk.tokenize.sent_tokenize("A sentence. And another one.")
        if len(probe) == 2:
            return (lambda text: list(nltk.tokenize.sent_tokenize(text)), "nltk-punkt")
    # BLE001: deliberately blind, like the stop-criterion probe in `harness`. NLTK
    # fails here in several unrelated ways -- absent, importable but missing its
    # `punkt` data, or present with a renamed resource -- and every one of them has
    # the same right answer: use the fallback splitter and say so in the fingerprint.
    # A narrower catch would turn an approximation into a crashed evaluation.
    except Exception as exc:  # noqa: BLE001  # pragma: no cover -- environment-dependent
        _log.debug("nltk sentence tokenizer unavailable (%s); using the regex splitter", exc)
    return (_split_sentences_regex, "regex-sentences")


def _load_word_tokenizer() -> tuple[Callable[[str], list[str]], str]:
    try:
        import nltk

        if nltk.tokenize.word_tokenize("A sentence.") == ["A", "sentence", "."]:
            return (lambda text: list(nltk.tokenize.word_tokenize(text)), "nltk-words")
    except Exception as exc:  # noqa: BLE001  # pragma: no cover -- environment-dependent
        _log.debug("nltk word tokenizer unavailable (%s); using the regex tokenizer", exc)
    return (_tokenize_words_regex, "regex-words")


# SIM905 wants a list literal. A wrapped word list stays scannable when an entry is
# added or (as the note below records) argued back out; thirty quoted strings do not.
_ABBREVIATIONS = frozenset(
    """mr mrs ms dr prof sr jr st vs etc inc ltd co corp dept fig vol e.g i.e u.s u.k
    jan feb mar apr jun jul aug sep sept oct nov dec""".split()  # noqa: SIM905
)
"""Deliberately short. Every entry suppresses a sentence break, so a word that is
*usually* a real sentence ending costs more than it saves: "no." and "al." were both
dropped for that reason -- "The answer is no. Then we..." is far commoner in a chat
response than "No. 5" is."""

# The curly quotes are the point, not a typo RUF001 caught: a model emits U+2019 and
# U+201D far more often than a human types them, and a sentence that ends `word.”` has
# to close *after* the quote or every quoted line counts as two sentences.
_SENTENCE_BOUNDARY = re.compile(r"([.!?][\"'’”)\]]*)(\s+)")  # noqa: RUF001
_WORDISH = re.compile(r"\w+(?:['’.]\w+)*")  # noqa: RUF001


def _split_sentences_regex(text: str) -> list[str]:
    """Split on terminal punctuation, skipping abbreviations and initials."""
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY.finditer(text):
        head = text[start : match.end(1)]
        if _ends_with_abbreviation(head):
            continue
        stripped = head.strip()
        if stripped:
            sentences.append(stripped)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _ends_with_abbreviation(head: str) -> bool:
    if not head.endswith("."):
        return False
    last = head.split()[-1].rstrip(".").lower() if head.split() else ""
    # A single letter before a period is an initial -- "J. R. R. Tolkien" is one
    # sentence, and punkt gets this right too.
    return len(last) == 1 or last in _ABBREVIATIONS


def _tokenize_words_regex(text: str) -> list[str]:
    """Stand in for ``nltk.word_tokenize`` closely enough for the capitals count.

    Keeps internal periods and apostrophes attached so ``U.S.A.`` stays one token and
    is counted as one capitalised word rather than three, which is what punkt does.
    """
    return _WORDISH.findall(text)


def count_words(text: str) -> int:
    """The original's word count, exactly: ``RegexpTokenizer(r"\\w+")``."""
    return len(re.findall(r"\w+", text))


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Spec:
    factory: Callable[[str, Mapping[str, Any]], Callable[[str], bool]]
    requires: frozenset[str] = frozenset()


_REGISTRY: dict[str, _Spec] = {}


def _register(
    instruction_id: str, *, requires: Iterable[str] = ()
) -> Callable[
    [Callable[[str, Mapping[str, Any]], Callable[[str], bool]]],
    Callable[[str, Mapping[str, Any]], Callable[[str], bool]],
]:
    def decorate(
        factory: Callable[[str, Mapping[str, Any]], Callable[[str], bool]],
    ) -> Callable[[str, Mapping[str, Any]], Callable[[str], bool]]:
        _REGISTRY[instruction_id] = _Spec(factory=factory, requires=frozenset(requires))
        return factory

    return decorate


def build_checker(instruction_id: str, kwargs: Mapping[str, Any]) -> Checker:
    """Bind one instruction to its arguments.

    Raises rather than returning a checker that always fails, and the callers build
    every checker for the whole dataset *before* generating a single token. A missing
    kwarg or an unparseable regex should cost two seconds, not the six GPU-hours it
    would cost if it surfaced while scoring.
    """
    spec = _REGISTRY.get(instruction_id)
    if spec is None:
        raise DynQuantError(
            f"unknown IFEval instruction {instruction_id!r}; "
            f"the registry covers {len(_REGISTRY)} ids and this dataset needs one more"
        )
    unmet = spec.requires - backends().capabilities
    if unmet:
        raise DynQuantError(
            f"instruction {instruction_id!r} needs {', '.join(sorted(unmet))}, "
            "which is not installed"
        )
    return Checker(instruction_id=instruction_id, predicate=spec.factory(instruction_id, kwargs))


def requirements_for(instruction_id: str) -> frozenset[str]:
    spec = _REGISTRY.get(instruction_id)
    return spec.requires if spec else frozenset()


def missing_capabilities(instruction_ids: Iterable[str]) -> frozenset[str]:
    """Capabilities these instructions need that this process does not have."""
    needed = frozenset().union(*(requirements_for(i) for i in instruction_ids)) or frozenset()
    return needed - backends().capabilities


INSTRUCTION_IDS: frozenset[str] = frozenset()
"""Every id the registry can score. Populated at the end of the module."""


# --------------------------------------------------------------------------
# Argument access
# --------------------------------------------------------------------------


def _need(instruction_id: str, kwargs: Mapping[str, Any], name: str) -> Any:
    if kwargs.get(name) is None:
        raise DynQuantError(
            f"IFEval instruction {instruction_id!r} needs a {name!r} argument; got {sorted(kwargs)}"
        )
    return kwargs[name]


def _relation(instruction_id: str, kwargs: Mapping[str, Any], name: str) -> str:
    relation = str(_need(instruction_id, kwargs, name))
    if relation not in _RELATIONS:
        raise DynQuantError(
            f"IFEval instruction {instruction_id!r} got relation {relation!r}, "
            f"expected one of {_RELATIONS}"
        )
    return relation


def _compare(actual: int, threshold: int, relation: str) -> bool:
    return actual < threshold if relation == "less than" else actual >= threshold


def _compile(instruction_id: str, pattern: str, flags: int = 0) -> re.Pattern[str]:
    """Compile now, so a bad pattern names its instruction instead of a stack frame.

    Several instructions interpolate a dataset-supplied string straight into a regex,
    which is what the reference implementation does and therefore what has to happen
    here. Compiling at build time turns an unbalanced bracket into a two-second failure
    that says which prompt caused it, rather than a `re.error` raised while scoring.
    """
    try:
        return re.compile(pattern, flags)
    except re.error as exc:
        raise DynQuantError(
            f"IFEval instruction {instruction_id!r} produced an invalid pattern {pattern!r}: {exc}"
        ) from exc


# --------------------------------------------------------------------------
# keywords
# --------------------------------------------------------------------------


@_register("keywords:existence")
def _keyword_existence(instruction_id: str, kwargs: Mapping[str, Any]):
    # No word boundaries. The original searches for the bare pattern, so "existence"
    # is satisfied by "batting" when the keyword is "bat". Reproduced, not repaired.
    patterns = [
        _compile(instruction_id, keyword, re.IGNORECASE)
        for keyword in _need(instruction_id, kwargs, "keywords")
    ]
    return lambda response: all(pattern.search(response) for pattern in patterns)


@_register("keywords:frequency")
def _keyword_frequency(instruction_id: str, kwargs: Mapping[str, Any]):
    pattern = _compile(instruction_id, _need(instruction_id, kwargs, "keyword"), re.IGNORECASE)
    frequency = int(_need(instruction_id, kwargs, "frequency"))
    relation = _relation(instruction_id, kwargs, "relation")
    return lambda response: _compare(len(pattern.findall(response)), frequency, relation)


@_register("keywords:forbidden_words")
def _forbidden_words(instruction_id: str, kwargs: Mapping[str, Any]):
    # With word boundaries, unlike `keywords:existence` above. The asymmetry is in the
    # original and is load-bearing: forbidding "bat" must not fail a response for
    # saying "debate".
    patterns = [
        _compile(instruction_id, rf"\b{word}\b", re.IGNORECASE)
        for word in _need(instruction_id, kwargs, "forbidden_words")
    ]
    return lambda response: not any(pattern.search(response) for pattern in patterns)


@_register("keywords:letter_frequency")
def _letter_frequency(instruction_id: str, kwargs: Mapping[str, Any]):
    letter = str(_need(instruction_id, kwargs, "letter")).lower()
    frequency = int(_need(instruction_id, kwargs, "let_frequency"))
    relation = _relation(instruction_id, kwargs, "let_relation")
    return lambda response: _compare(Counter(response.lower())[letter], frequency, relation)


# --------------------------------------------------------------------------
# language
# --------------------------------------------------------------------------


@_register("language:response_language", requires=[CAPABILITY_LANGDETECT])
def _response_language(instruction_id: str, kwargs: Mapping[str, Any]):
    language = str(_need(instruction_id, kwargs, "language"))
    detect = backends().detect_language

    def check(response: str) -> bool:
        try:
            return detect(response) == language  # type: ignore[misc]
        # BLE001: langdetect raises its own exception type on text it cannot classify
        # (too short, no alphabetic content). The original catches it and counts the
        # instruction as *followed*, which is the behaviour being reproduced -- and
        # catching it by name would mean importing langdetect at module scope, which
        # is the thing this file is arranged to avoid.
        except Exception:  # noqa: BLE001
            return True

    return check


# --------------------------------------------------------------------------
# length constraints
# --------------------------------------------------------------------------


@_register("length_constraints:number_sentences")
def _number_sentences(instruction_id: str, kwargs: Mapping[str, Any]):
    num_sentences = int(_need(instruction_id, kwargs, "num_sentences"))
    relation = _relation(instruction_id, kwargs, "relation")
    split = backends().split_sentences
    return lambda response: _compare(len(split(response)), num_sentences, relation)


@_register("length_constraints:number_paragraphs")
def _number_paragraphs(instruction_id: str, kwargs: Mapping[str, Any]):
    num_paragraphs = int(_need(instruction_id, kwargs, "num_paragraphs"))
    splitter = re.compile(r"\s?\*\*\*\s?")

    def check(response: str) -> bool:
        paragraphs = splitter.split(response)
        count = len(paragraphs)
        for index, paragraph in enumerate(paragraphs):
            if paragraph.strip():
                continue
            # An empty run at either end is just a leading or trailing separator;
            # an empty one in the middle means a `***` with nothing between it and
            # the next, which is a malformed answer rather than a short one.
            if index in (0, len(paragraphs) - 1):
                count -= 1
            else:
                return False
        return count == num_paragraphs

    return check


@_register("length_constraints:number_words")
def _number_words(instruction_id: str, kwargs: Mapping[str, Any]):
    num_words = int(_need(instruction_id, kwargs, "num_words"))
    relation = _relation(instruction_id, kwargs, "relation")
    return lambda response: _compare(count_words(response), num_words, relation)


@_register("length_constraints:nth_paragraph_first_word")
def _nth_paragraph_first_word(instruction_id: str, kwargs: Mapping[str, Any]):
    num_paragraphs = int(_need(instruction_id, kwargs, "num_paragraphs"))
    nth = int(_need(instruction_id, kwargs, "nth_paragraph"))
    first_word = str(_need(instruction_id, kwargs, "first_word")).lower()
    punctuation = frozenset(".,?!'\"")

    def check(response: str) -> bool:
        paragraphs = re.split(r"\n\n", response)
        count = sum(1 for paragraph in paragraphs if paragraph.strip())
        if nth > count:
            return False
        # Indexed into the *unfiltered* list, as the original does: blank paragraphs
        # are excluded from the count but still occupy a position.
        paragraph = paragraphs[nth - 1].strip()
        if not paragraph:
            return False
        word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
        actual = ""
        for letter in word:
            if letter in punctuation:
                break
            actual += letter.lower()
        return count == num_paragraphs and actual == first_word

    return check


# --------------------------------------------------------------------------
# detectable content
# --------------------------------------------------------------------------


@_register("detectable_content:number_placeholders")
def _number_placeholders(instruction_id: str, kwargs: Mapping[str, Any]):
    num_placeholders = int(_need(instruction_id, kwargs, "num_placeholders"))
    pattern = re.compile(r"\[.*?\]")
    return lambda response: len(pattern.findall(response)) >= num_placeholders


@_register("detectable_content:postscript")
def _postscript(instruction_id: str, kwargs: Mapping[str, Any]):
    marker = str(_need(instruction_id, kwargs, "postscript_marker"))
    if marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + marker.lower() + r".*$"
    compiled = _compile(instruction_id, pattern, re.MULTILINE)
    return lambda response: bool(compiled.findall(response.lower()))


# --------------------------------------------------------------------------
# detectable format
# --------------------------------------------------------------------------


@_register("detectable_format:number_bullet_lists")
def _number_bullet_lists(instruction_id: str, kwargs: Mapping[str, Any]):
    num_bullets = int(_need(instruction_id, kwargs, "num_bullets"))
    star = re.compile(r"^\s*\*[^\*].*$", re.MULTILINE)
    dash = re.compile(r"^\s*-.*$", re.MULTILINE)
    return lambda response: len(star.findall(response)) + len(dash.findall(response)) == num_bullets


@_register("detectable_format:constrained_response")
def _constrained_response(instruction_id: str, kwargs: Mapping[str, Any]):
    del instruction_id, kwargs
    options = ("My answer is yes.", "My answer is no.", "My answer is maybe.")
    return lambda response: any(option in response.strip() for option in options)


@_register("detectable_format:number_highlighted_sections")
def _number_highlighted(instruction_id: str, kwargs: Mapping[str, Any]):
    num_highlights = int(_need(instruction_id, kwargs, "num_highlights"))
    single = re.compile(r"\*[^\n\*]*\*")
    double = re.compile(r"\*\*[^\n\*]*\*\*")

    def check(response: str) -> bool:
        count = sum(1 for match in single.findall(response) if match.strip("*").strip())
        count += sum(
            1
            for match in double.findall(response)
            if match.removeprefix("**").removesuffix("**").strip()
        )
        return count >= num_highlights

    return check


@_register("detectable_format:multiple_sections")
def _multiple_sections(instruction_id: str, kwargs: Mapping[str, Any]):
    spliter = str(_need(instruction_id, kwargs, "section_spliter"))
    num_sections = int(_need(instruction_id, kwargs, "num_sections"))
    # Interpolated raw, as the original does, so `Section` and `SECTION` behave
    # identically to upstream. Compiled here rather than per call so a spliter with
    # unbalanced regex metacharacters fails at build time.
    pattern = _compile(instruction_id, r"\s?" + spliter + r"\s?\d+\s?")
    return lambda response: len(pattern.split(response)) - 1 >= num_sections


@_register("detectable_format:json_format")
def _json_format(instruction_id: str, kwargs: Mapping[str, Any]):
    del instruction_id, kwargs

    def check(response: str) -> bool:
        value = (
            response.strip()
            .removeprefix("```json")
            .removeprefix("```Json")
            .removeprefix("```JSON")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        try:
            json.loads(value)
        except ValueError:
            return False
        return True

    return check


@_register("detectable_format:title")
def _title(instruction_id: str, kwargs: Mapping[str, Any]):
    del instruction_id, kwargs
    pattern = re.compile(r"<<[^\n]+>>")
    return lambda response: any(
        title.lstrip("<").rstrip(">").strip() for title in pattern.findall(response)
    )


# --------------------------------------------------------------------------
# combination
# --------------------------------------------------------------------------


@_register("combination:two_responses")
def _two_responses(instruction_id: str, kwargs: Mapping[str, Any]):
    del instruction_id, kwargs

    def check(response: str) -> bool:
        parts = response.split("******")
        valid: list[str] = []
        for index, part in enumerate(parts):
            if part.strip():
                valid.append(part)
            elif index not in (0, len(parts) - 1):
                return False
        # Two *different* responses: a model that emits the same answer twice with a
        # separator between them has not done what was asked.
        return len(valid) == 2 and valid[0].strip() != valid[1].strip()

    return check


@_register("combination:repeat_prompt")
def _repeat_prompt(instruction_id: str, kwargs: Mapping[str, Any]):
    to_repeat = str(_need(instruction_id, kwargs, "prompt_to_repeat")).strip().lower()
    return lambda response: response.strip().lower().startswith(to_repeat)


# --------------------------------------------------------------------------
# start / end
# --------------------------------------------------------------------------


@_register("startend:end_checker")
def _end_checker(instruction_id: str, kwargs: Mapping[str, Any]):
    end_phrase = str(_need(instruction_id, kwargs, "end_phrase")).strip().lower()
    return lambda response: response.strip().strip('"').lower().endswith(end_phrase)


@_register("startend:quotation")
def _quotation(instruction_id: str, kwargs: Mapping[str, Any]):
    del instruction_id, kwargs

    def check(response: str) -> bool:
        value = response.strip()
        return len(value) > 1 and value[0] == '"' and value[-1] == '"'

    return check


# --------------------------------------------------------------------------
# change case
# --------------------------------------------------------------------------


@_register("change_case:capital_word_frequency")
def _capital_word_frequency(instruction_id: str, kwargs: Mapping[str, Any]):
    frequency = int(_need(instruction_id, kwargs, "capital_frequency"))
    relation = _relation(instruction_id, kwargs, "capital_relation")
    tokenize = backends().tokenize_words

    def check(response: str) -> bool:
        capitals = sum(1 for word in tokenize(response) if word.isupper())
        return _compare(capitals, frequency, relation)

    return check


@_register("change_case:english_capital", requires=[CAPABILITY_LANGDETECT])
def _english_capital(instruction_id: str, kwargs: Mapping[str, Any]):
    del instruction_id, kwargs
    detect = backends().detect_language

    def check(response: str) -> bool:
        try:
            return response.isupper() and detect(response) == "en"  # type: ignore[misc]
        except Exception:  # noqa: BLE001  -- see `_response_language`
            return True

    return check


@_register("change_case:english_lowercase", requires=[CAPABILITY_LANGDETECT])
def _english_lowercase(instruction_id: str, kwargs: Mapping[str, Any]):
    del instruction_id, kwargs
    detect = backends().detect_language

    def check(response: str) -> bool:
        try:
            return response.islower() and detect(response) == "en"  # type: ignore[misc]
        except Exception:  # noqa: BLE001  -- see `_response_language`
            return True

    return check


# --------------------------------------------------------------------------
# punctuation
# --------------------------------------------------------------------------


@_register("punctuation:no_comma")
def _no_comma(instruction_id: str, kwargs: Mapping[str, Any]):
    del instruction_id, kwargs
    return lambda response: "," not in response


INSTRUCTION_IDS = frozenset(_REGISTRY)
