"""SLURP: name the intent of a *spoken* command, from the audio.

Why this task
-------------
The campaign fine-tunes an audio-native MoE, so the evaluation has to be one the
audio path is actually on. A text task scored on the same checkpoint would measure
the Thinker reading a transcript somebody else produced, and every arm would share
that transcript -- so the number would move with quantization damage but would say
nothing about the model that was fine-tuned.

SLURP is the spoken-language-understanding benchmark with the properties this
project screens for, in the order it screens for them:

* **Headroom.** Published supervised systems land in the high 70s / low 80s on
  intent accuracy from audio. A base instruct model prompted with the menu is far
  below that, which is the gap a fine-tune has to close. The GO/NO-GO band is
  checked by measurement before the fine-tune is booked, not assumed here --
  see :func:`load_slurp` and the screening protocol in the run log.
* **A low chance floor.** One in sixty-odd, against CaseHOLD's one in five. A
  model whose weights have been damaged cannot land on the right intent by luck,
  so the distance between "still works" and "broken" is nearly the whole scale.
* **Enough items to pair on.** The test split is large enough that a one-point
  difference is decidable by McNemar's test at the discordance rates this project
  measures (~7,850 items at d=0.10). CaseHOLD and Banking77 both cleared that bar
  and both produced promotable results; a 500-item audio set would not have.

Three rungs, one decode
-----------------------
A SLURP intent is a ``scenario_action`` pair -- ``email_query``,
``alarm_set``. So a single generation scores three nested questions at once:
the full intent, the scenario alone, and the action alone. The rungs cost nothing
extra and they separate two failures that a single number cannot: a model that has
lost the domain (scenario wrong) from one that has the domain and picks the wrong
verb in it (scenario right, action wrong). Only the full-intent vector is the
headline and only it is stored as ``hits``, because the paired test needs one
vector and a run that stored three would eventually have two arms compared on
different ones.

Why the exemplars are text and the query is audio
-------------------------------------------------
The few-shot prefix exists to teach the *output format* -- "answer with the number
only" -- and nothing about the format is audible. Rendering the exemplars as
transcripts costs about fifteen tokens each; rendering them as audio would cost
several hundred tokens each *and* run the audio encoder five times per scored item.
SLURP ships the gold transcript on every row, so the cheap version is free.

The thing this trades away is worth naming: the model never sees an example of the
audio-to-index mapping, only of the text-to-index one. That is the right trade for
a format prefix and it would be the wrong one if the shots were meant to teach the
task. They are not -- the fine-tune teaches the task, and the shots are held
identical across every arm so whatever they contribute is a constant.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch

from dynquant._logging import get_logger

from .harness import AudioPrompt, EvalBackend, EvalConfig, generate_batched, strip_reasoning

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "EXPECTED_INTENTS_SHA",
    "N_INTENTS",
    "SAMPLE_RATE",
    "SHUFFLE_SEED",
    "SlurpExample",
    "SlurpResult",
    "build_prompt",
    "decoded_audio",
    "deterministic_order",
    "evaluate_slurp",
    "extract_answer",
    "intent_menu",
    "load_slurp",
    "official_taxonomy",
    "split_intent",
]

_log = get_logger(__name__)

_SOURCE = "marcel-gohsen/slurp"
"""Where the *audio* comes from. Named here rather than at the call site so the one
thing that decides what is being measured is not a string in three files.

Not ``pswietojanski/slurp``, the dataset SLURP's own authors published: it no longer
resolves on the Hub. Several mirrors of the same release survive and they agree
exactly on the split sizes SLURP documents -- 50,628 / 8,690 / 13,078 recordings --
so what is being measured is unchanged. This one is chosen because it carries the
``devel`` split and the ``slurp_id`` that :data:`_ANNOTATION` is keyed on."""

_ANNOTATION = "https://raw.githubusercontent.com/pswietojanski/slurp/master/dataset/slurp/{}.jsonl"
"""Where the *labels* come from: SLURP's own annotation, keyed by ``slurp_id``.

Two sources rather than one because the mirrors' label column is unusable, and
unusable for a reason no mirror caused. SLURP's released annotation carries both an
``intent`` field and a ``scenario``/``action`` pair, and on 1,548 of its 72,396
recordings they disagree -- always by ``intent`` having lost the scenario, so
``iot_hue_lightoff`` is filed as ``hue_lightoff`` and, worse, nine different
``*_query`` intents are all filed as ``query``. A mirror that copies that field
inherits a taxonomy whose classes overlap, whose size differs per split (91 / 71 /
77 against the real 60), and in which a model answering "query" cannot be scored at
all. ``scenario`` and ``action`` are clean, so the joint pair is what this module
labels with, and the mirror's own ``intent`` column is read for nothing."""

SAMPLE_RATE = 16_000
"""What the audio is resampled to on load.

Fixed here and applied by ``datasets`` rather than left to the processor, because a
processor handed audio at the wrong rate does not raise -- it reads a clip as though
it were played at the wrong speed, which lands the score somewhere between chance
and correct and looks like a weak model."""

SHUFFLE_SEED = 0
"""Fixes the load order -- see :func:`load_slurp`. Both a prefix-sampling hazard and
a pairing requirement: `--limit` must sample the label space, and every arm must see
the same items in the same positions or the ``hits`` vectors pair item *i* of one
run against a different item of the next."""

N_INTENTS = 60
"""How many intents SLURP has: 18 scenarios crossed with the actions each admits.

Measured, not remembered. An earlier draft of this module wrote down 69 from memory
and the number was wrong; the taxonomy is now read out of the annotation by
:func:`official_taxonomy`, and this constant is the assertion that it still says what
it said -- so a mirror or a revision that moves it fails a load instead of quietly
re-scaling every accuracy published beside it."""

EXPECTED_INTENTS_SHA = "d04b663b407e9f5b5be80c9d11160c391c7b68f516c9da957aaca026138fc86d"
"""sha256 of the newline-joined intent menu this task was calibrated against.

The menu is the numbering an answer indexes into, so two arms scored against
different menus are not comparable however close their accuracies look -- and the
disagreement is silent, because the same index is a valid answer under either. This
digest is carried into every result record and re-checked on every load."""

FEWSHOT_STOP = "\n"
"""One newline, not the blank line Banking77 uses. The exemplars here are one line
each, so the model's own turn boundary is the end of the line."""

_LEADING_INDEX = re.compile(r"^\W*(\d+)")
_ANY_INDEX = re.compile(r"\d+")

_ID_FIELDS = ("slurp_id", "id")
"""Column names the mirrors have used for SLURP's own recording key, in the order
they are tried. This is the join to :data:`_ANNOTATION` and so the join to the
labels; a mirror carrying neither cannot be labelled and is refused."""


@dataclass(frozen=True, slots=True)
class SlurpExample:
    """One spoken command, with the menu it is numbered against."""

    audio: Callable[[], dict[str, Any]]
    """Returns ``{"array": ..., "sampling_rate": SAMPLE_RATE}`` for this row.

    A thunk rather than the array, so that loading a split does not decode it. The
    test set is ~13k clips; materialising every waveform to score the first 500 of
    them costs gigabytes and minutes for nothing. ``datasets`` keeps the table
    memory-mapped and decodes the row this closure asks for, so ``--limit`` pays for
    what it scores.
    """

    transcript: str
    """The gold transcription. Used only to render *exemplars* -- never for the item
    being scored, which the model has to hear."""

    intent: str
    """``scenario_action``, as the dataset spells it."""

    intents: tuple[str, ...]
    """The full menu, shared by every example from one load.

    Carried on the example rather than kept in a module global so that the numbering
    an answer indexes into travels with the item it numbers. The alternative -- a
    global filled by the last loader call -- is how two arms end up scored against
    two different menus with nothing in either record to say so.
    """

    @property
    def answer(self) -> str:
        """The gold index into :attr:`intents`, as a string."""
        return str(self.intents.index(self.intent))


@dataclass(slots=True)
class SlurpResult:
    """One measurement point: the intent, and the two halves it decomposes into."""

    label: str
    correct: int
    total: int
    unparseable: int
    """Generations naming no in-range index. Distinct from a wrong answer: the model
    has stopped answering rather than answered badly."""

    scenario_correct: int = 0
    action_correct: int = 0
    n_intents: int = 0
    intents_sha: str = ""
    """Which menu produced these numbers. Two arms whose digests differ were scored
    against different taxonomies and their accuracies are not comparable, however
    close they look."""

    hits: list[bool] = field(default_factory=list)
    """Per-item full-intent correctness, in dataset order. The headline vector and
    the only one -- see the module docstring on why the rungs do not each get one."""

    predictions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def chance(self) -> float:
        return 1.0 / self.n_intents if self.n_intents else 0.0

    @property
    def above_chance(self) -> float:
        return self.accuracy - self.chance

    def as_dict(self) -> dict[str, Any]:
        """The rungs the headline number does not carry."""
        return {
            "scenario_accuracy": self.scenario_correct / self.total if self.total else 0.0,
            "action_accuracy": self.action_correct / self.total if self.total else 0.0,
            "n_intents": self.n_intents,
            "intents_sha": self.intents_sha,
        }

    def summary(self) -> str:
        detail = self.as_dict()
        return (
            f"{self.label:<28} {self.accuracy:6.2%}  "
            f"({self.correct}/{self.total} intent, "
            f"{detail['scenario_accuracy']:.2%} scenario, "
            f"{detail['action_accuracy']:.2%} action, "
            f"{self.above_chance:+.2%} vs chance, {self.unparseable} unparseable)"
        )


def split_intent(intent: str) -> tuple[str, str]:
    """``"email_query"`` -> ``("email", "query")``.

    Split at the *first* underscore: scenarios are single words and actions are not
    (``query_details``, ``set_remove``), so splitting at the last one would move part
    of the action into the scenario and score both rungs wrong on exactly the
    multi-word actions a model is most likely to confuse.
    """
    scenario, _, action = intent.partition("_")
    return scenario, action


def intent_menu(intents: Sequence[str]) -> str:
    """The numbered menu, prepended to every prompt and to every training sequence.

    The model answers with an index and an index means nothing without the list that
    numbers it -- so this text is part of the prompt at every measurement point, and
    the fine-tune has to train against the same lines it is scored against.
    """
    header = (
        "Listen to the spoken command and classify it into one of these intents.\n"
        "Answer with the number only.\n"
    )
    return header + "\n".join(f"{index}. {intent}" for index, intent in enumerate(intents))


def _fingerprint(intents: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(intents).encode("utf-8")).hexdigest()


def official_taxonomy(
    *, cache_dir: str | None = None, splits: Sequence[str] = ("train", "devel", "test")
) -> tuple[tuple[str, ...], dict[int, str]]:
    """SLURP's label space and its ``slurp_id`` -> intent map, from SLURP's own files.

    Read across **all** splits, not the one being loaded. The menu has to be identical
    at fine-tune time and at eval time -- the model answers with a position in it -- and
    the splits do not each carry all 60 intents: ``train`` does, ``devel`` and ``test``
    are each missing one, and a different one. A menu derived per split would therefore
    renumber part of the taxonomy between training and scoring, and every index the
    model learned would name its neighbour.

    The intent is ``scenario_action`` and never the annotation's own ``intent`` field;
    see :data:`_ANNOTATION` for what is wrong with that field.
    """
    by_id: dict[int, str] = {}
    for split in splits:
        for row in _annotation_rows(split, cache_dir=cache_dir):
            by_id[int(row["slurp_id"])] = f"{row['scenario']}_{row['action']}"
    menu = tuple(sorted(set(by_id.values())))
    sha = _fingerprint(menu)
    if sha != EXPECTED_INTENTS_SHA:
        raise ValueError(
            f"SLURP's annotation yields {len(menu)} intents with digest {sha}, but this "
            f"task was calibrated against {N_INTENTS} intents with digest "
            f"{EXPECTED_INTENTS_SHA}. The index a model answers with names a different "
            f"intent under a different menu, so scores across the two are not comparable. "
            f"Pin the annotation revision, or re-screen and update EXPECTED_INTENTS_SHA."
        )
    if len(menu) != N_INTENTS:  # pragma: no cover - unreachable while the digest matches
        raise ValueError(f"{len(menu)} intents, expected {N_INTENTS}")
    return menu, by_id


def _annotation_rows(split: str, *, cache_dir: str | None = None) -> list[dict[str, Any]]:
    """One split of SLURP's annotation, fetched once and cached on disk.

    Cached because :func:`official_taxonomy` is called by the loader, by the trainer
    and by the scorer, and three fetches of the same immutable file per run is three
    chances for a campaign to half-fail on someone else's outage.
    """
    import json
    import urllib.request

    root = pathlib.Path(cache_dir) if cache_dir else pathlib.Path.home() / ".cache" / "dynquant"
    path = root / "slurp" / f"{split}.jsonl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(_ANNOTATION.format(split), timeout=120) as response:
            body = response.read().decode("utf-8")
        # Through a temporary name, so an interrupted fetch cannot leave a short file
        # that every later run then reads as the whole annotation.
        staged = path.with_suffix(".partial")
        staged.write_text(body, encoding="utf-8")
        staged.replace(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _id_column(dataset: Any) -> str:
    for candidate in _ID_FIELDS:
        if candidate in dataset.features:
            return candidate
    raise ValueError(
        f"{_SOURCE} carries no SLURP id column (tried {list(_ID_FIELDS)}); saw "
        f"{sorted(dataset.features)}. Without one its rows cannot be joined to SLURP's "
        f"annotation, and its own intent column is not usable -- see _ANNOTATION."
    )


def load_slurp(
    split: str = "test", *, cache_dir: str | None = None, seed: int = SHUFFLE_SEED
) -> list[SlurpExample]:
    """Load a SLURP split, deterministically shuffled, audio decoded on demand.

    The shuffle is not cosmetic. Splits on this Hub arrive grouped -- by speaker, by
    scenario, or both -- so any prefix of the raw order is a sample of one corner of
    the label space. ``--limit 500`` on such an order scores a handful of intents and
    reads as a destroyed model, which is exactly the failure a Banking77 smoke run
    produced before its loader started shuffling. A *fixed* seed rather than none,
    because the paired design compares ``hits`` position by position and an order
    that varied between arms would pair each item against a different one.

    The menu does not come from this split at all -- it is SLURP's whole taxonomy, so
    ``--limit`` changes which items are scored, and the *split* changes which items
    exist, but neither changes what any of them are scored against. See
    :func:`official_taxonomy`.
    """
    from datasets import Audio, load_dataset

    intents, by_id = official_taxonomy(cache_dir=cache_dir)

    raw = load_dataset(_SOURCE, split=split, cache_dir=cache_dir)
    audio_column = _audio_column(raw)
    raw = raw.cast_column(audio_column, Audio(sampling_rate=SAMPLE_RATE))

    # Iterated with the audio column dropped, and once: an id and a transcript are
    # needed for every row, and pulling them through the audio feature would decode
    # thirteen thousand waveforms to read thirteen thousand small integers.
    id_column = _id_column(raw)
    transcript_column = _transcript_column(raw)
    rows = list(raw.remove_columns([audio_column]))

    row_intents: list[str] = []
    unjoined: list[int] = []
    for row in rows:
        intent = by_id.get(int(row[id_column]))
        if intent is None:
            unjoined.append(int(row[id_column]))
        else:
            row_intents.append(intent)
    if unjoined:
        raise ValueError(
            f"{len(unjoined)} of {len(rows)} rows of {_SOURCE} split {split!r} carry a "
            f"{id_column} that SLURP's annotation does not define (first: {unjoined[:5]}). "
            f"The mirror and the annotation are describing different releases, and the "
            f"rows that did join would be scored against a menu the rest were not in."
        )

    examples = [
        SlurpExample(
            audio=_row_audio(raw, index, audio_column),
            transcript=str(row[transcript_column]).strip(),
            intent=intent,
            intents=intents,
        )
        for index, (row, intent) in enumerate(zip(rows, row_intents, strict=True))
    ]
    _log.info(
        "SLURP %s: %d items, %d of %d intents present, menu sha256 %s",
        split,
        len(examples),
        len(set(row_intents)),
        len(intents),
        EXPECTED_INTENTS_SHA,
    )
    return deterministic_order(examples, seed=seed)


def _row_audio(dataset: Any, index: int, column: str) -> Callable[[], dict[str, Any]]:
    """A thunk that decodes one row's audio when something asks for it."""

    def fetch() -> dict[str, Any]:
        return decoded_audio(dataset[index][column])

    return fetch


def decoded_audio(row_value: Any) -> dict[str, Any]:
    """One row's audio as ``{"array": mono float32, "sampling_rate": SAMPLE_RATE}``.

    Two shapes are accepted because ``datasets`` changed its mind: through 4.x an
    audio row decoded to that dict, and 5.x hands back a torchcodec ``AudioDecoder``
    instead. Adapting both is a dozen lines and keeps this module off a version floor
    on a library the trainer also depends on.

    Mono by averaging, because the encoder takes one channel and a stereo clip handed
    through as ``[2, n]`` is read as a clip of twice the length at half the pitch.

    The sample-rate check is the reason this is a function and not an attribute
    access. A processor given audio at the wrong rate does not raise; it reads the
    clip as though it were played at the wrong speed, and the resulting score sits
    between chance and correct and looks exactly like a weak model. The rate is
    requested from ``datasets`` at cast time and *verified* here, because the cast is
    a request and this is the only place the answer is visible.
    """
    import numpy as np

    if hasattr(row_value, "get_all_samples"):
        samples = row_value.get_all_samples()
        rate = int(samples.sample_rate)
        data = samples.data
        array = data.mean(dim=0) if data.ndim > 1 else data
        array = array.to("cpu", dtype=torch.float32).numpy()
    else:
        rate = int(row_value["sampling_rate"])
        array = np.asarray(row_value["array"], dtype=np.float32)
        if array.ndim > 1:
            array = array.mean(axis=0)

    if rate != SAMPLE_RATE:
        raise ValueError(
            f"{_SOURCE} decoded a clip at {rate} Hz, not the {SAMPLE_RATE} Hz the "
            f"encoder was trained on. Resampling was requested at load; something "
            f"downstream of that request did not honour it. Scoring on this would not "
            f"fail -- it would read every clip at the wrong speed and report a weak "
            f"model."
        )
    return {"array": array, "sampling_rate": rate}


def _audio_column(dataset: Any) -> str:
    for candidate in ("audio", "audio_file", "speech"):
        if candidate in dataset.features:
            return candidate
    raise ValueError(f"{_SOURCE} has no audio column; saw {sorted(dataset.features)}")


def _transcript_column(dataset: Any) -> str:
    for candidate in ("sentence", "transcript", "text", "utt"):
        if candidate in dataset.features:
            return candidate
    raise ValueError(f"{_SOURCE} has no transcript column; saw {sorted(dataset.features)}")


def deterministic_order(
    examples: list[SlurpExample], *, seed: int = SHUFFLE_SEED
) -> list[SlurpExample]:
    """``examples`` shuffled by a fixed seed. Separate from the loader so the property
    that matters -- a prefix is label-diverse, and two calls agree -- is testable
    without a network round trip."""
    import random

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def build_prompt(
    processor: Any,
    example: SlurpExample,
    shots: Sequence[SlurpExample] = (),
) -> AudioPrompt:
    """The prompt for one item: menu, text exemplars, then the clip to classify.

    Built through the processor's own chat template rather than assembled here,
    because the audio placeholder run inside the ids has to be exactly as long as
    the encoder's frame count and only the processor knows that arithmetic. Getting
    it wrong by one raises nothing -- it offsets the audio against the text and
    scores like a weak model.

    The clip goes into the content block as a bare array, not as the
    ``{"array", "sampling_rate"}`` dict :func:`decoded_audio` returns. A content
    block's ``audio`` is whatever ``transformers.audio_utils.load_audio`` accepts,
    and that is an ``np.ndarray`` or a string naming a file, a URL or a base64
    payload -- a dict raises. So the rate is not carried in the block at all; it
    is passed beside it, where the processor's audio kwargs read it.
    """
    lines = [f"{shot.transcript} -> {shot.answer}" for shot in shots]
    preamble = intent_menu(example.intents)
    if lines:
        preamble += "\n\nExamples, as text:\n" + "\n".join(lines)

    clip = example.audio()
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": preamble + "\n\nNow the spoken command:"},
                {"type": "audio", "audio": clip["array"]},
                {"type": "text", "text": "Intent:"},
            ],
        }
    ]
    encoded = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        # The rate the clip was *decoded* at, not the constant it was requested at.
        # `decoded_audio` has already refused anything else, so the two cannot
        # disagree here -- which is the point: the processor is handed a measured
        # number, and the place that could have lied about it raises instead.
        sampling_rate=clip["sampling_rate"],
    )
    payload = dict(encoded)
    ids = payload.pop("input_ids")
    payload.pop("attention_mask", None)
    return AudioPrompt(ids=tuple(int(token) for token in ids[0]), features=payload)


def extract_answer(text: str, n_intents: int) -> str | None:
    """The chosen index, or ``None`` if the generation names none in range.

    Prefers an integer at the very start, which is the format the menu asks for, and
    falls back to the first in-range integer anywhere, because a model answering
    ``"intent 41"`` has chosen and scoring that as a failure would charge a
    formatting wobble to quantization. Out-of-range integers are skipped rather than
    accepted, so a number naming no intent is unparseable and not merely wrong.
    """
    text = strip_reasoning(text)
    leading = _LEADING_INDEX.match(text.strip())
    if leading and int(leading.group(1)) < n_intents:
        return leading.group(1)
    for match in _ANY_INDEX.finditer(text):
        if int(match.group(0)) < n_intents:
            return match.group(0)
    return None


def evaluate_slurp(
    model: Any,
    processor: Any,
    examples: Sequence[SlurpExample],
    *,
    label: str,
    shots: Sequence[SlurpExample] = (),
    config: EvalConfig | None = None,
    progress: Callable[[int, int], None] | None = None,
    keep_predictions: int = 0,
) -> SlurpResult:
    """Score a model on spoken-intent classification, and on the two rungs below it.

    Args:
        model: a ``transformers`` multi-modal model, or an
            :class:`~dynquant.eval.harness.EvalBackend`. A raw model is wrapped in
            :class:`~dynquant.eval.omni.OmniThinkerBackend` here rather than by the
            caller, because the wrapping is a fact about this task -- its prompts
            carry encoder inputs, and a backend that batches ids alone would have to
            drop them.
        processor: the checkpoint's ``AutoProcessor``. Both the encoder *and* the
            decoder for this task: it builds the prompts and detokenizes the answers,
            so no second tokenizer can disagree with it about either.
        shots: text exemplars from the *train* split, identical at every measurement
            point, and excluded from the fine-tuning set -- an exemplar shown in every
            prompt and trained on is an answer handed to the model for free.
    """
    config = config or EvalConfig(
        max_new_tokens=8, stop_sequences=(FEWSHOT_STOP,), max_prompt_tokens=4096, batch_size=8
    )
    if FEWSHOT_STOP not in config.stop_sequences:
        config = replace(config, stop_sequences=(*config.stop_sequences, FEWSHOT_STOP))
    # The template emits its own control tokens, so the tokenizer must not add a
    # second BOS on top of them -- the same reason every chat-framed task sets this.
    config = replace(config, add_special_tokens=False)

    subset = list(examples[: config.limit] if config.limit else examples)
    if not subset:
        raise ValueError("no examples to score")
    menus = {example.intents for example in subset}
    if len(menus) != 1:
        raise ValueError(
            f"these examples carry {len(menus)} different intent menus. An index means a "
            f"different intent under each, so they cannot be scored in one run."
        )
    intents = subset[0].intents

    if not isinstance(model, EvalBackend):
        from .omni import OmniThinkerBackend

        model = OmniThinkerBackend(model, processor)

    prompts = [build_prompt(processor, example, shots) for example in subset]
    generations = generate_batched(model, processor, prompts, config, progress=progress)

    result = SlurpResult(
        label=label,
        correct=0,
        total=len(subset),
        unparseable=0,
        n_intents=len(intents),
        intents_sha=_fingerprint(intents),
    )
    for index, (example, text) in enumerate(zip(subset, generations, strict=True)):
        predicted = extract_answer(text, len(intents))
        if predicted is None:
            result.unparseable += 1
        hit = predicted is not None and predicted == example.answer
        result.correct += int(hit)
        result.hits.append(hit)

        gold_scenario, gold_action = split_intent(example.intent)
        if predicted is None:
            predicted_intent = None
            predicted_scenario = predicted_action = None
        else:
            predicted_intent = intents[int(predicted)]
            predicted_scenario, predicted_action = split_intent(predicted_intent)
        result.scenario_correct += int(predicted_scenario == gold_scenario)
        result.action_correct += int(predicted_action == gold_action)

        if index < keep_predictions:
            result.predictions.append(
                {
                    "generation": text.strip(),
                    "predicted": predicted,
                    "predicted_intent": predicted_intent,
                    "gold": example.answer,
                    "gold_intent": example.intent,
                    "transcript": example.transcript,
                    "correct": hit,
                }
            )

    _log.info("%s", result.summary())
    return result
