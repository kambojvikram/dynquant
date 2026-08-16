"""SLURP's label space: where it comes from, and the two ways it silently goes wrong.

The scoring risk on this task is not the extractor -- Banking77 already covers a
multi-digit index into a long menu, and SLURP's is the same machinery. It is the
*menu itself*, in two distinct ways that both score somewhere between chance and
correct without raising:

**The wrong column.** SLURP's released annotation carries an ``intent`` field and a
``scenario``/``action`` pair, and they disagree on 1,548 of 72,396 recordings --
always by ``intent`` having dropped the scenario. Nine different ``*_query`` intents
are all filed as bare ``query``, so a taxonomy built from that field has classes
that overlap, has a different size in every split (91 / 71 / 77 against the real
60), and cannot score a model that answers "query" at all. Every mirror on the Hub
copies the field, so this is not a mirror bug to shop around for.

**The wrong scope.** The menu is the numbering an answer indexes into, so it has to
be the same object at fine-tune time and at eval time. The splits do not each carry
all 60 intents -- ``devel`` and ``test`` are each missing one, and a different one --
so a menu derived from whichever split is loaded renumbers part of the taxonomy
between training and scoring, and every index the model learned names its neighbour.

No network and no dataset: the annotation is a fixture on disk, which is exactly
what :func:`dynquant.eval.slurp._annotation_rows` reads once it has been fetched.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import pytest

from dynquant.eval import slurp
from dynquant.eval.slurp import (
    EXPECTED_INTENTS_SHA,
    N_INTENTS,
    SAMPLE_RATE,
    SlurpExample,
    build_prompt,
    decoded_audio,
    deterministic_order,
    extract_answer,
    intent_menu,
    official_taxonomy,
    split_intent,
)

if TYPE_CHECKING:
    from pathlib import Path

# One row per prompt, in SLURP's own shape. `intent` is deliberately the *damaged*
# spelling on two of these -- that is the field the module must not read.
FIXTURE: dict[str, list[dict[str, Any]]] = {
    "train": [
        {"slurp_id": 1, "scenario": "weather", "action": "query", "intent": "query"},
        {"slurp_id": 2, "scenario": "email", "action": "query", "intent": "query"},
        {"slurp_id": 3, "scenario": "alarm", "action": "set", "intent": "alarm_set"},
    ],
    "devel": [
        {"slurp_id": 4, "scenario": "weather", "action": "query", "intent": "weather_query"},
    ],
    "test": [
        {"slurp_id": 5, "scenario": "email", "action": "query", "intent": "email_query"},
        {"slurp_id": 6, "scenario": "weather", "action": "query", "intent": "query"},
    ],
}

FIXTURE_MENU = ("alarm_set", "email_query", "weather_query")
FIXTURE_SHA = hashlib.sha256("\n".join(FIXTURE_MENU).encode("utf-8")).hexdigest()


@pytest.fixture
def annotation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A cache directory already holding an annotation, and constants that match it.

    Pre-populated rather than fetched, so the tests exercise the parsing and the
    guards and never the network. The constants are moved to the fixture's own menu
    because what is under test is the mechanism; that the *shipped* constants agree
    with the real corpus is a separate assertion below.
    """
    for split, rows in FIXTURE.items():
        path = tmp_path / "slurp" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    monkeypatch.setattr(slurp, "EXPECTED_INTENTS_SHA", FIXTURE_SHA)
    monkeypatch.setattr(slurp, "N_INTENTS", len(FIXTURE_MENU))
    return str(tmp_path)


def test_the_label_is_scenario_action_and_never_the_annotations_intent_field(
    annotation: str,
) -> None:
    """The load-bearing one.

    Four of the six fixture rows are ``*_query`` intents in three different scenarios,
    and three of them spell their ``intent`` field as bare ``query``. Reading that
    field would collapse them onto one class and lose the distinction the task is
    scored on; reading ``scenario``/``action`` keeps them apart.
    """
    menu, by_id = official_taxonomy(cache_dir=annotation)

    assert menu == FIXTURE_MENU
    assert by_id[1] == "weather_query"
    assert by_id[2] == "email_query"
    assert "query" not in menu


def test_the_menu_spans_every_split_not_the_one_being_loaded(annotation: str) -> None:
    """``alarm_set`` exists only in the fixture's train split.

    A menu derived from ``test`` alone would be two entries long, so ``weather_query``
    would be index 1 there and index 2 here -- and both are valid answers, so nothing
    downstream can tell which numbering a score was produced under. The taxonomy is
    therefore read across all splits, and the digest is what makes the narrower menu
    unreachable rather than merely discouraged: asking for one split raises.
    """
    everything, _ = official_taxonomy(cache_dir=annotation)

    assert everything == FIXTURE_MENU
    assert "alarm_set" in everything, "present only in the fixture's train split"
    assert everything.index("weather_query") == 2

    with pytest.raises(ValueError, match="not comparable"):
        official_taxonomy(cache_dir=annotation, splits=("test",))


def test_a_menu_that_is_not_the_calibrated_one_is_refused(annotation: str, tmp_path: Path) -> None:
    """The negative control: a mirror that moves under a campaign fails the load.

    Both digests go in the message, because the failure is only actionable if you can
    tell which of the two ends moved.
    """
    extra = {"slurp_id": 7, "scenario": "iot", "action": "wemo_off", "intent": "wemo_off"}
    path = tmp_path / "slurp" / "test.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n" + json.dumps(extra), encoding="utf-8")

    with pytest.raises(ValueError, match="not comparable") as raised:
        official_taxonomy(cache_dir=annotation)

    message = str(raised.value)
    assert FIXTURE_SHA in message, "the calibrated digest"
    assert "4 intents" in message, "what the annotation actually yields now"


def test_a_split_is_read_from_the_cache_and_not_refetched(
    annotation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cached annotation means no network, so an outage cannot half-fail a campaign.

    The loader, the trainer and the scorer each ask for the taxonomy; three fetches of
    the same immutable file is three chances to fail at a different point in a run.
    """

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("fetched an annotation that was already on disk")

    monkeypatch.setattr("urllib.request.urlopen", explode)

    first, _ = official_taxonomy(cache_dir=annotation)
    second, _ = official_taxonomy(cache_dir=annotation)

    assert first == second == FIXTURE_MENU


def test_an_interrupted_fetch_leaves_no_file_to_be_read_as_the_whole_annotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A short annotation is the worst possible artifact: it parses.

    It yields a smaller menu, a different digest, and -- if the digest guard were ever
    relaxed -- a silently renumbered taxonomy. So the fetch lands on a temporary name
    and is renamed only once complete, and a failure leaves the cache empty.
    """

    def die(*args: Any, **kwargs: Any) -> Any:
        raise OSError("connection reset")

    monkeypatch.setattr("urllib.request.urlopen", die)

    with pytest.raises(OSError, match="connection reset"):
        slurp._annotation_rows("test", cache_dir=str(tmp_path))

    assert not (tmp_path / "slurp" / "test.jsonl").exists()


def test_the_id_column_is_named_and_a_mirror_without_one_says_why() -> None:
    """The join key is the whole labelling strategy, so its absence is not a warning.

    A mirror with no SLURP id cannot be labelled at all -- its own intent column is
    the damaged one -- so the message points at the reason rather than at the column.
    """

    class Features:
        def __init__(self, *names: str) -> None:
            self.features = dict.fromkeys(names)

    assert slurp._id_column(Features("slurp_id", "id", "audio")) == "slurp_id"
    assert slurp._id_column(Features("id", "audio")) == "id"

    with pytest.raises(ValueError, match="cannot be joined") as raised:
        slurp._id_column(Features("audio", "intent", "sentence"))
    assert "intent" in str(raised.value), "names what it saw"


def test_the_shipped_constants_describe_slurps_published_taxonomy() -> None:
    """18 scenarios by the actions each admits, and the digest that pins the order.

    Not a restatement of the module: ``N_INTENTS`` is what a load is checked against
    and the digest is what a *result* is checked against, so a change to either that
    was not intended to re-scale every published accuracy fails here first.
    """
    assert N_INTENTS == 60
    assert EXPECTED_INTENTS_SHA == (
        "d04b663b407e9f5b5be80c9d11160c391c7b68f516c9da957aaca026138fc86d"
    )


def test_the_task_specs_chance_floor_is_its_modules_class_count() -> None:
    """The floor in ``TASKS`` is a literal, so something has to tie it to the taxonomy.

    It is a literal for a reason -- ``dynquant.eval`` imports torch and ``TASKS`` is on
    the CLI's startup path -- but this exact line read ``1.0 / 69`` until the taxonomy
    was measured, and 69 is not a number SLURP has ever had. A wrong floor does not
    change any accuracy; it changes ``above_chance``, which is what a screening
    decision is made on.
    """
    from dynquant.commands.evaluate import TASKS
    from dynquant.eval.banking77 import N_INTENTS as BANKING77_INTENTS

    assert TASKS["slurp"].chance == pytest.approx(1.0 / N_INTENTS)
    assert TASKS["banking77"].chance == pytest.approx(1.0 / BANKING77_INTENTS)


def test_an_intent_splits_at_the_first_underscore_so_multiword_actions_survive() -> None:
    """Scenarios are one word and actions are not.

    Splitting at the last underscore would move ``hue`` into the scenario and score
    both rungs wrong on exactly the intents a damaged model confuses first.
    """
    assert split_intent("email_query") == ("email", "query")
    assert split_intent("iot_hue_lightdim") == ("iot", "hue_lightdim")
    assert split_intent("audio_volume_mute") == ("audio", "volume_mute")


def test_the_menu_numbers_from_zero_and_the_gold_index_agrees_with_it() -> None:
    """An answer is a position, so the rendering and the gold have to be one function."""
    menu = intent_menu(FIXTURE_MENU)
    example = SlurpExample(
        audio=lambda: None, transcript="", intent="weather_query", intents=FIXTURE_MENU
    )

    assert "0. alarm_set" in menu
    assert "2. weather_query" in menu
    assert example.answer == "2"
    assert extract_answer("2", len(FIXTURE_MENU)) == example.answer


def test_out_of_range_indices_are_unparseable_rather_than_wrong() -> None:
    """Same accuracy, different diagnosis.

    A model naming index 60 of a 60-entry menu has lost the list it was given, which
    is a damage signature; a model naming index 3 has answered and been wrong.
    """
    assert extract_answer("59", N_INTENTS) == "59"
    assert extract_answer("60", N_INTENTS) is None
    assert extract_answer("intent 41", N_INTENTS) == "41"


def test_a_prefix_of_the_load_order_samples_the_label_space(annotation: str) -> None:
    """``--limit`` has to score a spread of intents, not one corner of the split.

    SLURP arrives grouped by speaker and scenario, so an unshuffled prefix names a
    handful of intents and reads as a destroyed model -- the failure a Banking77 smoke
    run produced before its loader started shuffling.
    """
    menu, _ = official_taxonomy(cache_dir=annotation)
    grouped = [
        SlurpExample(audio=lambda: None, transcript="", intent=intent, intents=menu)
        for intent in menu
        for _ in range(20)
    ]

    ordered = deterministic_order(grouped)
    prefix = {example.intent for example in ordered[:12]}

    assert len(prefix) == len(menu), "a 12-item prefix of 60 should reach every class"
    assert [e.intent for e in ordered] == [e.intent for e in deterministic_order(grouped)], (
        "two calls must agree, or hits vectors pair different items"
    )


class _Samples:
    """What `torchcodec` returns from `AudioDecoder.get_all_samples()`."""

    def __init__(self, data: Any, sample_rate: int) -> None:
        self.data = data
        self.sample_rate = sample_rate


class _Decoder:
    """A `datasets` 5 audio row: a decoder, not the array dict 4.x handed back."""

    def __init__(self, data: Any, sample_rate: int) -> None:
        self._samples = _Samples(data, sample_rate)

    def get_all_samples(self) -> _Samples:
        return self._samples


def test_both_shapes_datasets_has_handed_back_decode_to_the_same_array() -> None:
    """`datasets` 4 gave a dict and 5 gives a torchcodec decoder.

    Adapting both is what keeps this module off a version floor on a library the
    trainer also pulls -- and the two have to agree exactly, or an arm run before the
    upgrade and one run after are scored on different audio.
    """
    import numpy as np
    import torch

    wave = [0.0, 0.25, -0.5, 1.0]
    from_decoder = decoded_audio(_Decoder(torch.tensor([wave]), SAMPLE_RATE))
    from_dict = decoded_audio({"array": np.asarray(wave), "sampling_rate": SAMPLE_RATE})

    assert from_decoder["sampling_rate"] == from_dict["sampling_rate"] == SAMPLE_RATE
    assert from_decoder["array"].dtype == from_dict["array"].dtype == np.float32
    assert np.array_equal(from_decoder["array"], from_dict["array"])
    assert from_decoder["array"].tolist() == wave


def test_a_stereo_clip_is_averaged_and_not_passed_through_at_twice_the_length() -> None:
    """The encoder takes one channel.

    A `[2, n]` clip handed through unaveraged is read as a `2n` clip -- twice the
    duration at half the pitch -- which is a wrong answer and not an error.
    """
    import numpy as np
    import torch

    stereo = torch.tensor([[1.0, 1.0], [0.0, 2.0]])

    decoded = decoded_audio(_Decoder(stereo, SAMPLE_RATE))

    assert decoded["array"].shape == (2,)
    assert np.array_equal(decoded["array"], np.asarray([0.5, 1.5], dtype=np.float32))


def test_a_clip_at_the_wrong_rate_is_refused_rather_than_scored() -> None:
    """The failure this replaces is the quietest one on the task.

    Nothing downstream raises on a mis-rated clip: the processor reads it as the same
    words spoken at the wrong speed, the model answers badly, and the arm reports a
    weak model rather than a broken loader. The rate is *requested* at cast time and
    this is the only point where the answer is visible, so it is checked here.
    """
    import torch

    with pytest.raises(ValueError, match="not the 16000 Hz") as raised:
        decoded_audio(_Decoder(torch.zeros(1, 8), 8_000))

    assert "read every clip at the wrong speed" in str(raised.value)


class _Processor:
    """A processor that refuses what ``transformers`` refuses, and records the rest.

    The one method under test is ``apply_chat_template``, and the reason to fake it
    rather than to load a real one is that the real one weighs a checkpoint. The fake
    is only worth anything if it is strict in the same place: ``transformers`` feeds a
    content block's ``audio`` value to ``audio_utils.load_audio``, which takes an
    ``np.ndarray`` or a ``str`` and raises ``TypeError`` on anything else. That is
    reproduced exactly, because a lenient fake would accept the dict that broke every
    prompt on the real box and the test would have passed through the failure.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> dict[str, Any]:
        import numpy as np

        for message in conversation:
            for block in message["content"]:
                if block["type"] != "audio":
                    continue
                value = block["audio"]
                if not isinstance(value, np.ndarray | str):
                    raise TypeError(
                        "Incorrect format used for `audio`. Should be a numpy array or a `str`"
                    )
        self.calls.append({"conversation": conversation, **kwargs})
        # Two keys, so the split below is a real split: the ids come out and
        # everything else stays as the features that travel beside them.
        return {"input_ids": [[7, 8, 9]], "input_features": "spectrogram", "attention_mask": [[1]]}


def _example(
    intents: tuple[str, ...] = FIXTURE_MENU, *, intent: str = "email_query"
) -> SlurpExample:
    import numpy as np

    return SlurpExample(
        audio=lambda: {"array": np.zeros(4, dtype=np.float32), "sampling_rate": SAMPLE_RATE},
        transcript="send my mother an email",
        intent=intent,
        intents=intents,
    )


def test_the_clip_reaches_the_template_as_an_array_the_processor_accepts() -> None:
    """The shape that broke every prompt on the box, pinned.

    ``decoded_audio`` returns a dict because the rate has to be checked somewhere; a
    content block takes an array because that is what ``load_audio`` takes. Passing
    the dict through raised on the first scored item -- which is the good case, and
    the reason this is a cheap test rather than a lost run.
    """
    import numpy as np

    processor = _Processor()

    build_prompt(processor, _example(), shots=())

    blocks = processor.calls[0]["conversation"][0]["content"]
    audio = [block for block in blocks if block["type"] == "audio"]
    assert len(audio) == 1, "one clip per prompt: the item being scored"
    assert isinstance(audio[0]["audio"], np.ndarray)
    assert "sampling_rate" not in audio[0], "the rate travels beside the block, not in it"


def test_the_rate_the_clip_decoded_at_is_what_the_processor_is_told() -> None:
    """A processor left to its default is a processor that was never told.

    The default happens to be 16 kHz for this checkpoint, so passing nothing would
    score identically today and silently differently on a checkpoint whose encoder
    wants something else. The value passed is the one ``decoded_audio`` measured, not
    the constant it was requested at -- those cannot disagree, because the measurement
    refuses first.
    """
    processor = _Processor()

    build_prompt(processor, _example(), shots=())

    assert processor.calls[0]["sampling_rate"] == SAMPLE_RATE


def test_the_ids_come_out_and_everything_else_stays_beside_them() -> None:
    """``AudioPrompt`` is one object because its halves must not be zipped up wrong.

    The ids are the prompt; the encoder inputs are what the placeholder run in those
    ids is filled from. ``attention_mask`` is dropped because the harness rebuilds it
    per batch -- a per-example mask carried through would be the wrong length the
    moment two prompts are padded together.
    """
    processor = _Processor()

    prompt = build_prompt(processor, _example(), shots=())

    assert prompt.ids == (7, 8, 9)
    assert dict(prompt.features) == {"input_features": "spectrogram"}


def test_the_exemplars_are_text_and_only_the_scored_item_is_heard() -> None:
    """What the few-shot prefix costs, and what it is for.

    The shots teach the output format, and nothing about "answer with the number
    only" is audible. Rendering them as audio would run the encoder five times per
    scored item to teach what fifteen tokens of text teaches -- so they arrive as
    transcript-to-index lines in the preamble, and the audio blocks stay at one.
    """
    processor = _Processor()
    shots = [_example(intent="alarm_set"), _example(intent="weather_query")]

    build_prompt(processor, _example(), shots=shots)

    blocks = processor.calls[0]["conversation"][0]["content"]
    assert sum(block["type"] == "audio" for block in blocks) == 1
    preamble = blocks[0]["text"]
    for shot in shots:
        assert f"{shot.transcript} -> {shot.answer}" in preamble
