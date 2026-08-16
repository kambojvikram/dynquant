"""``dynquant export`` has to write a directory a processor can be loaded from.

The Qwen3-Omni packed checkpoints came out of ``export`` with a tokenizer and no
``processor_config.json``, and every evaluation pointed at them exited before it
read a single weight. What made it expensive to diagnose is that the error names
the *wrong file*: ``AutoProcessor`` falls through to the image-processor auto
path and raises about a missing ``preprocessor_config.json``, which the merged
bf16 source directory also does not have and which loads fine anyway.
``processor_config.json`` is the file that decides.

So the test that matters is not "does the copy loop copy" -- it is "are these the
names transformers actually looks for". They are asserted against
``transformers``' own module-level constants, because a fixture that repeats my
spelling would agree with me whether or not the spelling is real.
"""

from __future__ import annotations

import pytest

from dynquant import constants
from dynquant.commands.quantize import _copy_processor_sidecars


def test_sidecar_names_are_transformers_own_constants():
    """The negative control: read the names back off the dependency, not a fixture."""
    transformers = pytest.importorskip("transformers")
    processing_utils = pytest.importorskip("transformers.processing_utils")

    expected = {
        processing_utils.PROCESSOR_NAME,
        processing_utils.LEGACY_PROCESSOR_CHAT_TEMPLATE_FILE,
        processing_utils.AUDIO_TOKENIZER_NAME,
        transformers.utils.IMAGE_PROCESSOR_NAME,
    }
    missing = expected - set(constants.HF_PROCESSOR_SIDECARS)
    assert not missing, f"transformers looks for files the exporter does not copy: {missing}"


def test_video_sidecar_name_is_transformers_own():
    """Split out because the video processor moved modules and may move again."""
    video_utils = pytest.importorskip("transformers.video_processing_utils")
    assert video_utils.VIDEO_PROCESSOR_NAME in constants.HF_PROCESSOR_SIDECARS


def test_processor_config_is_the_one_that_decides():
    """Named on its own: it is the difference between a loadable directory and not."""
    assert "processor_config.json" in constants.HF_PROCESSOR_SIDECARS


def test_copies_present_sidecars_and_ignores_absent(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "processor_config.json").write_text('{"processor_class": "X"}', encoding="utf-8")
    (source / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()

    _copy_processor_sidecars(source, out)

    assert (out / "processor_config.json").read_text(encoding="utf-8") == '{"processor_class": "X"}'
    assert (out / "preprocessor_config.json").is_file()
    assert not (out / "chat_template.json").exists()


def test_never_overwrites_what_the_exporter_already_wrote(tmp_path):
    """``tokenizer.save_pretrained`` runs first; its output is the authority."""
    source = tmp_path / "src"
    source.mkdir()
    (source / "chat_template.json").write_text("stale", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    (out / "chat_template.json").write_text("fresh", encoding="utf-8")

    _copy_processor_sidecars(source, out)

    assert (out / "chat_template.json").read_text(encoding="utf-8") == "fresh"


def test_hub_id_source_is_a_no_op(tmp_path):
    """A source that is not a local directory is left to the loader's own hub fetch."""
    out = tmp_path / "out"
    out.mkdir()

    _copy_processor_sidecars(tmp_path / "Qwen" / "Qwen3-Omni-30B-A3B-Instruct", out)

    assert list(out.iterdir()) == []
