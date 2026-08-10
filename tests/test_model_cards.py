"""The card is the last place a number can drift and the first place anyone reads it.

Six directories of about 4 GB each will sit on the Hub with a README above them. Nobody
downloads one to check whether its README is right; the README is why they download it. So
the failures worth covering here are the ones that produce a card a reader believes:

* a number typed into the card once and left behind when the panel was re-run;
* a DynQuant arm whose card does not say its accuracy was measured in bf16 rather than from
  the directory being offered;
* a baseline arm whose card claims the size it was scored at while the directory on disk is
  2.3% larger, because the codes were carried into a container with a wider zero point;
* a comparison the table flagged for mixed expert arithmetic, printed on the card as a
  clean verdict -- the dispatch difference is 0.29x the effect being reported, so a flag
  dropped here is a result overstated by a quarter of itself;
* a card for the bf16 ceiling, which would describe a model this campaign did not make.

The panel here is `test_panel_table`'s, imported rather than copied. The two files test
opposite ends of one pipeline -- that module builds the table, this one renders it -- and a
second fixture would let them drift apart precisely when a field is added to the table,
which is the case that matters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_panel_table import (
    REPO_ROOT,
    SCRIPT,
    _all_indexed,
    _load,
    _run,
    _write_panel,
)

CARDS = REPO_ROOT / "experiments" / "phase4" / "model_cards.py"

#: The fine-tune's own record, transcribed from `s2_finetune.json` rather than invented.
#: The card reads six fields out of it and the shapes matter: `dataset` is one string with
#: `+` separators and not a list, `seconds` is a float the card divides by 3600, and
#: `train_loss` is formatted to four places. A tidied fixture would let a card that cannot
#: read the real file stay green.
FINETUNE = {
    "model": "LiquidAI/LFM2.5-8B-A1B",
    "dataset": "gretelai/synthetic_text_to_sql+Salesforce/wikisql+b-mc2/sql-create-context",
    "conversations_kept": 49905,
    "regime": "lora",
    "lora_rank": 32,
    "epochs": 1.0,
    "steps": 1560,
    "train_loss": 0.1113953912869478,
    "seconds": 24111.8,
    "commit": "d0d33f3bce6f3f59359ce704b16040c7e9ba78f5",
}


@pytest.fixture(scope="module")
def cards() -> Any:
    return _load("_dq_model_cards", CARDS)


@pytest.fixture(scope="module")
def table_mod() -> Any:
    return _load("_dq_panel_table_for_cards", SCRIPT)


def _built(table_mod: Any, out: Path) -> tuple[dict[str, Any], Path]:
    """Run the table over a panel and read back what `--json-out` wrote.

    Going through the file rather than through `--json` is deliberate: the file is what the
    card generator is actually pointed at, and a flag that prints correctly while writing
    something else would be invisible to every other test.
    """
    dest = out.parent / "table.json"
    _run(table_mod, out, "--json-out", str(dest))
    return json.loads(dest.read_text(encoding="utf-8")), dest


def _row(text: str, prefix: str) -> str:
    matched = [line for line in text.splitlines() if line.startswith(prefix)]
    assert len(matched) == 1, f"{prefix!r} appears {len(matched)} times"
    return matched[0]


def test_the_card_prints_the_tables_numbers_and_holds_none_of_its_own(
    cards: Any, table_mod: Any, tmp_path: Path
) -> None:
    """Nothing is typed, so changing the table has to change the card.

    The reason for the whole split. The panel is re-run every time an arm lands, over a day
    and a half, and any accuracy written into this file by hand would be correct on the
    afternoon it was written and wrong for the rest of the campaign -- silently, because a
    card that says 73.75% looks exactly as authoritative as one that says 82.07%.

    Turns red when: a headline number is inlined, cached, or recomputed from the records
    instead of read from the table.
    """
    out = _write_panel(tmp_path / "arms")
    table, _ = _built(table_mod, out)

    before = cards.card(table, "gptq_4b", FINETUNE, repo_prefix=None)
    assert "73.75%" in before

    for row in table["arms"]:
        if row["label"] == "gptq_4b":
            row["accuracy"] = 0.5
    after = cards.card(table, "gptq_4b", FINETUNE, repo_prefix=None)
    assert "50.00%" in after
    assert "73.75%" not in after


def test_each_arm_states_its_own_container_and_never_the_other_kinds(
    cards: Any, table_mod: Any, tmp_path: Path
) -> None:
    """Two different honest sentences, and each is a lie on the other kind of arm.

    A DynQuant arm was scored by encoding its widths back into bf16, so its accuracy did
    not come from the directory being downloaded -- that has to be said, on the arm this
    campaign is arguing for, on its own card. A baseline arm was scored under
    compressed-tensors and republished into DynQuant's wider container, so the directory
    weighs about 2.3% more than the size in its results table.

    Putting both bullets on every card would be the easy way to never be wrong and would
    tell a DynQuant reader their directory is oversized and a GPTQ reader their accuracy
    was measured somewhere else. Both false.

    Turns red when: a caveat becomes unconditional, or its condition stops reading `apply`
    and `kind` off the row.
    """
    out = _write_panel(tmp_path / "arms")
    table, _ = _built(table_mod, out)

    dq = cards.card(table, "dq_4b", FINETUNE, repo_prefix=None)
    baseline = cards.card(table, "gptq_4b", FINETUNE, repo_prefix=None)

    assert "measured in bf16, not from this directory" in dq
    assert "2.3% larger" not in dq

    assert "2.3% larger" in baseline
    assert "measured in bf16" not in baseline


def test_the_flag_the_table_raised_is_the_flag_the_card_prints(
    cards: Any, table_mod: Any, tmp_path: Path
) -> None:
    """The campaign's headline row carries a confound, and this is where it survives to.

    `4b DynQuant vs GPTQ` is the sentence this panel exists to write, and on the landed
    records the two arms are not both recorded as having run the same expert dispatch. The
    two available dispatches disagree on 1.24% of teacher-forced tokens, 0.29x the effect
    of quantizing to 4 bits. A card that reported `separated` without that is the most-read
    document in the campaign overstating its own result.

    The cleared case is half the test. Once every arm is re-scored onto one dispatch the
    flag has to go -- a mark that never clears is a mark a reader learns to ignore, and the
    re-score is being run specifically to remove it.

    Turns red when: the flag stops reaching the card, the footnote loses the magnitude that
    makes it interpretable, or the mark stays on a panel where nothing is flagged.
    """
    out = _write_panel(tmp_path / "arms")
    table, _ = _built(table_mod, out)

    flagged = cards.card(table, "dq_4b", FINETUNE, repo_prefix=None)
    assert "[^1]" in flagged
    assert "1.24%" in flagged and "0.29x" in flagged
    assert "[^1]: the two arms are not both recorded" in flagged

    _all_indexed(out)
    cleared, _ = _built(table_mod, out)
    clean = cards.card(cleared, "dq_4b", FINETUNE, repo_prefix=None)
    assert "[^1]" not in clean
    assert "1.24%" not in clean
    assert "separated" in clean, "clearing the flag must not clear the result"


def test_a_ceiling_and_an_unscored_arm_are_refused_rather_than_described(
    cards: Any, table_mod: Any, tmp_path: Path
) -> None:
    """Three ways to end up with a card for something that is not a published arm.

    The ceiling is the checkpoint the panel started from; a card for it would describe
    somebody else's model under this campaign's name. An arm still running has no accuracy,
    and a card is a claim about a measurement -- mid-panel that is the state four arms are
    in for a day. A label that is simply not in the table is a typo in `--only`, and
    guessing at it would write a card for the wrong arm.

    Turns red when: a refusal becomes a default, a placeholder, or an empty results row.
    """
    out = _write_panel(tmp_path / "arms", omit=("awq_3b",))
    table, _ = _built(table_mod, out)

    assert "bf16" not in cards.publishable(table)
    assert "awq_3b" not in cards.publishable(table)
    assert cards.publishable(table) == ["gptq_4b", "awq_4b", "dq_4b", "gptq_3b", "dq_3b"]

    with pytest.raises(SystemExit, match="ceiling"):
        cards.card(table, "bf16", FINETUNE, repo_prefix=None)
    with pytest.raises(SystemExit, match="no accuracy"):
        cards.card(table, "awq_3b", FINETUNE, repo_prefix=None)
    with pytest.raises(SystemExit, match="has no arm"):
        cards.card(table, "dq_2b", FINETUNE, repo_prefix=None)


def test_one_comparison_reads_identically_on_both_of_its_cards(
    cards: Any, table_mod: Any, tmp_path: Path
) -> None:
    """The same pair, rendered on two cards, has to be the same row.

    The tempting edit is to flip the delta so every card leads with a favourable number.
    It would be arithmetically fine and editorially fatal: the question text, the CI and
    the flip counts all read left to right, so half-reversing a row produces something that
    cannot be checked against the table it came from -- and the two cards would then
    disagree about the same measurement while both citing the same panel.

    Turns red when: a card reorients a comparison to suit the arm it is about.
    """
    out = _write_panel(tmp_path / "arms")
    table, _ = _built(table_mod, out)
    entry = next(
        e for e in table["head_to_head"] if (e["left"], e["right"]) == ("dq_4b", "gptq_4b")
    )

    winner = cards.card(table, "dq_4b", FINETUNE, repo_prefix=None)
    loser = cards.card(table, "gptq_4b", FINETUNE, repo_prefix=None)
    prefix = "| " + entry["question"].strip()

    assert _row(winner, prefix) == _row(loser, prefix)
    assert f"{entry['delta_points']:+.2f}" in _row(loser, prefix)


def test_a_card_is_written_beside_the_weights_and_never_where_there_are_none(
    cards: Any, table_mod: Any, tmp_path: Path, capsys: Any
) -> None:
    """A README with no weights under it is a published model that does not exist.

    `publish_panel.py` writes one directory per arm and can be run for a subset -- mid-panel
    it will be, because the expensive arms go first. So the card generator writes into the
    directories that are there and says which ones it skipped, rather than creating them.
    A card in an empty directory would be indexed, linked and downloadable as a 404.

    Turns red when: the writer starts creating its target, or skips silently.
    """
    out = _write_panel(tmp_path / "arms")
    _, table_path = _built(table_mod, out)
    finetune_path = tmp_path / "s2_finetune.json"
    finetune_path.write_text(json.dumps(FINETUNE), encoding="utf-8")

    published = tmp_path / "published"
    (published / "dq_4b").mkdir(parents=True)

    assert (
        cards.main(
            [
                "--table",
                str(table_path),
                "--finetune",
                str(finetune_path),
                "--out",
                str(published),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out

    written = (published / "dq_4b" / "README.md").read_text(encoding="utf-8")
    assert written.startswith("---" + "\n" + "base_model: LiquidAI/LFM2.5-8B-A1B")
    assert sorted(p.name for p in published.iterdir()) == ["dq_4b"]
    assert "gptq_4b: " in printed and "publish the arm first" in printed


def test_the_loader_named_is_the_one_that_can_open_the_directory(
    cards: Any, table_mod: Any, tmp_path: Path
) -> None:
    """A GPTQ arm in this panel does not load with a GPTQ loader, and the card must not imply it.

    Every arm here is written by DynQuant's exporter: the baselines' integer codes are
    carried into DynQuant's container, which is why they are 2.3% larger. That container is
    read by DynQuant's `HfQuantizer` and not by vLLM's native compressed-tensors path. A
    card headed "GPTQ 4-bit" that showed the usual AutoGPTQ snippet would send every reader
    to a loader that cannot open the file, and the error they would get is about the
    checkpoint rather than about the instructions.

    Turns red when: the usage block is genericised per method, or the repo id in it stops
    matching the id the arm is published under.
    """
    out = _write_panel(tmp_path / "arms")
    table, _ = _built(table_mod, out)
    text = cards.card(table, "gptq_4b", FINETUNE, repo_prefix="acme/lfm25-8b-a1b-text2sql")

    assert "dynquant.register_hf_quantizer()" in text
    assert "acme/lfm25-8b-a1b-text2sql-GPTQ-4bit" in text
    for wrong in ("AutoGPTQ", "auto-gptq", "GPTQModel", "AutoAWQ"):
        assert wrong not in text, f"{wrong} cannot open a DynQuant container"


def test_the_snippet_names_a_function_that_exists(
    cards: Any, table_mod: Any, tmp_path: Path
) -> None:
    """The one line on the card that a reader executes, checked against the package.

    ``import dynquant`` deliberately does not register the quantizer -- the import stays
    free of torch and transformers, so registration is an explicit second line. That makes
    the snippet load-bearing rather than decorative: without it ``from_pretrained`` reports
    an unknown quantization method and the reader concludes the checkpoint is broken.

    A card cannot verify itself, so this test does it: the symbol the snippet calls is
    resolved on the installed package. Renaming or folding the registration into import
    would leave every published README quietly wrong, on six repositories, with no failing
    build anywhere.

    Turns red when: ``register_hf_quantizer`` is renamed, moved off the top level, or the
    snippet drifts to some other entry point.
    """
    out = _write_panel(tmp_path / "arms")
    table, _ = _built(table_mod, out)
    text = cards.card(table, "dq_4b", FINETUNE, repo_prefix=None)

    import dynquant

    called = [
        line.split("dynquant.")[1].split("(")[0]
        for line in text.splitlines()
        if line.startswith("dynquant.")
    ]
    assert called == ["register_hf_quantizer"]
    assert callable(getattr(dynquant, called[0]))


def test_the_frontmatter_carries_no_tag_twice_and_invents_no_licence(
    cards: Any, table_mod: Any, tmp_path: Path
) -> None:
    """Two small things the Hub renders and nobody proofreads.

    A DynQuant arm's method name and the package tag are the same word, so the tag list
    builds ``dynquant`` twice and the Hub draws two identical chips -- the kind of detail
    that reads as carelessness about everything else on the page.

    The licence matters more. This file has no way to read the base model's terms, and a
    derivative inherits them; writing ``apache-2.0`` here because it is the common answer
    would be inventing a permission on somebody else's model. `other` plus a link to the
    base repo is the only honest thing a generator can say.

    Turns red when: the tag list stops deduplicating, or a concrete licence is asserted.
    """
    out = _write_panel(tmp_path / "arms")
    table, _ = _built(table_mod, out)
    front = cards.card(table, "dq_4b", FINETUNE, repo_prefix=None).split("---")[1]

    tags = [line[2:] for line in front.splitlines() if line.startswith("- ")]
    assert tags == list(dict.fromkeys(tags)), tags
    assert "dynquant" in tags and "4-bit" in tags

    assert "license: other" in front
    assert "license_link: https://huggingface.co/LiquidAI/LFM2.5-8B-A1B" in front
    for invented in ("apache-2.0", "mit", "llama"):
        assert invented not in front
