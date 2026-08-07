"""``--reuse-maps`` skips seven hours of allocation, so it has to refuse more than it accepts.

A moments-priced map costs about 1 h 45 m of CPU on an 8B checkpoint, and the S3 driver
builds six of them before it quantizes anything. A run that already has the maps and only
wants to quantize and evaluate should not rebuild them -- but the thing being added is a
resume guard, and a resume guard's failure mode is not that it declined to skip. It is that
existence-on-disk says nothing about *when* the file was written, so a map that predates
the stats file it names gets stapled into a fresh run and every number downstream describes
an allocation nobody asked for. That has happened in this repository before.

So the tests here are mostly negative. Each one builds a map that is wrong in exactly one
way a finished map cannot be inspected for, and asserts the guard rebuilds:

* **Stale.** The map is well-formed and names every right input; it is simply older than
  one of them. Nothing inside the file records this.
* **A different allocator.** ``rank`` and ``dq`` differ only in whether a sensitivity table
  was passed. Their bit widths are both plausible integers, so a map built by the wrong arm
  is undetectable from its contents -- the ``allocator`` field is the only witness.

The one positive case worth as much as the negatives is the *derived* variant. The arms are
allocated from the signal and shuffled variants, but those are rewritten by the same run
that reads them, so stamping a map against one asks whether the map predates a file this run
has just regenerated. The answer is always yes, it says nothing about whether the numbers
moved, and it is what made the first attempt at this rebuild five of six maps.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "scripts" / "run_s3_allocate.py"


@pytest.fixture(scope="module")
def driver():
    spec = importlib.util.spec_from_file_location("_dq_s3_driver", DRIVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dq_s3_driver"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bench(tmp_path: Path):
    """S2's sources, S3's derived variants, and a map that legitimately reuses.

    The two are kept apart deliberately. ``source_stats`` and ``source_moments`` are what
    the run is handed and what freshness is judged against; ``variant_stats`` is what the
    ``dq`` arm is allocated from and what the map names. Collapsing them into one file would
    make the distinction this guard turns on untestable.
    """
    model = tmp_path / "merged"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")

    source = tmp_path / "s2_stats"
    source.mkdir()
    source_stats = source / "dynquant_stats.json"
    source_stats.write_text("{}", encoding="utf-8")
    source_moments = source / "dynquant_moments.safetensors"
    source_moments.write_bytes(b"\x00")

    variant_stats = tmp_path / "dynquant_stats.signal.json"
    variant_stats.write_text("{}", encoding="utf-8")
    variant_moments = tmp_path / "dynquant_moments.signal.safetensors"
    variant_moments.write_bytes(b"\x00")

    save_map = tmp_path / "map.dq3.json"
    save_map.write_text(
        json.dumps(
            {
                "schema": "dynquant_allocation_v1",
                "model": str(model),
                "stats": str(variant_stats),
                "group_size": 128,
                "allocator": "sensitivity",
                "maps": {"3257925632": {"nbytes": 3257925632}},
            }
        ),
        encoding="utf-8",
    )
    _touch(save_map, newest(model, source_stats, source_moments) + 10)

    args = argparse.Namespace(
        reuse_maps=True,
        model=str(model),
        group_size=128,
        stats=str(source),
        moments=str(source_moments),
    )
    return argparse.Namespace(
        args=args,
        map=save_map,
        source_stats=source_stats,
        source_moments=source_moments,
        variant_stats=variant_stats,
        variant_moments=variant_moments,
        kwargs={
            "keys": ["3257925632"],
            "stats": str(variant_stats),
            "allocator": "sensitivity",
        },
    )


def newest(*paths: Path) -> float:
    return max(p.stat().st_mtime for p in paths)


def _touch(path: Path, when: float) -> None:
    os.utime(path, (when, when))


def test_a_map_newer_than_every_input_it_names_is_reused(driver, bench) -> None:
    """The positive case, so the negatives below are not passing for want of a match.

    Turns red when: the guard tightens to the point of never skipping, which would make
    ``--reuse-maps`` a no-op that still reads as an optimisation at the call site.
    """
    assert driver._reusable(bench.map, bench.args, **bench.kwargs) is not None


def test_a_map_older_than_the_signal_it_summarises_is_rebuilt(driver, bench) -> None:
    """The map is valid in every respect except that it cannot have been built from this.

    This is the whole reason the guard reads mtimes: re-running S2's signal extraction and
    then reusing yesterday's map produces a file that passes every structural check, names
    the right stats path, and describes an allocation computed from different numbers. Both
    sources are checked -- the moments alone decide every width the sensitivity allocator
    assigns, so a run that refreshed only those would otherwise slip through.

    Turns red when: the freshness comparison is dropped, or is made against the map's own
    directory rather than its inputs, or covers the stats but not the moments.
    """
    for source in (bench.source_stats, bench.source_moments):
        when = bench.map.stat().st_mtime
        _touch(source, when + 10)
        assert driver._reusable(bench.map, bench.args, **bench.kwargs) is None, source.name
        _touch(source, when - 10)


def test_a_map_older_than_the_variant_it_names_is_still_reused(driver, bench) -> None:
    """The variants are derived, not sources, and the run rewrites them before reading them.

    ``write_variants`` runs on every invocation, so both variants are newer than any map
    built from them the moment the driver starts. Stamping against one asks "is this map
    older than a file I wrote thirty seconds ago", which is always yes and says nothing
    about whether the numbers moved. The first version of this guard did exactly that and
    rebuilt five of six maps, spending the hours the flag exists to save.

    Turns red when: the derived variants are added back to the stamp set -- at which point
    ``--reuse-maps`` still passes every negative test above and silently saves nothing.
    """
    _touch(bench.variant_stats, bench.map.stat().st_mtime + 10)
    _touch(bench.variant_moments, bench.map.stat().st_mtime + 10)
    assert driver._reusable(bench.map, bench.args, **bench.kwargs) is not None

    # Stated directly as well, because the assertion above would also pass if the guard
    # stopped reading mtimes at all -- which the staleness tests would then catch, but only
    # after this one had gone quiet about the thing it exists to pin.
    assert driver._inputs_mtime(bench.args) < bench.variant_stats.stat().st_mtime


def test_a_map_older_than_the_checkpoint_is_rebuilt(driver, bench) -> None:
    """The model is an input too -- widths come from its shapes, not only from the stats.

    Turns red when: only the stats and moments are stamped against, which would miss a
    re-merged fine-tune entirely.
    """
    _touch(Path(bench.args.model) / "config.json", bench.map.stat().st_mtime + 10)
    assert driver._reusable(bench.map, bench.args, **bench.kwargs) is None


def test_a_map_built_by_the_other_allocator_is_rebuilt(driver, bench) -> None:
    """``rank`` and ``dq`` produce equally plausible integers; only the field tells them apart.

    A ``dq`` arm that reused the ``rank`` map would report the baseline's allocation under
    the headline's name, at matched bytes, with nothing anomalous in the widths to notice.

    Turns red when: the allocator field stops being checked, or stops being written.
    """
    payload = json.loads(bench.map.read_text(encoding="utf-8"))
    payload["allocator"] = "rank_product"
    when = bench.map.stat().st_mtime
    bench.map.write_text(json.dumps(payload), encoding="utf-8")
    _touch(bench.map, when)
    assert driver._reusable(bench.map, bench.args, **bench.kwargs) is None


def test_a_map_for_another_model_or_group_size_is_rebuilt(driver, bench) -> None:
    """Provenance mismatches, each of which would silently produce a wrong-model arm.

    The ``stats`` field is checked here rather than with the freshness checks because it is
    the only thing separating one arm's map from another's: ``shuf3`` and ``dq3`` name the
    same model, the same allocator and the same group size, and differ solely in which
    variant they were priced from.

    Turns red when: a field is dropped from the comparison, most plausibly ``group_size``,
    which does not appear in the arm's name and so has no other witness in the run.
    """
    for field, value in (("model", "/somewhere/else"), ("group_size", 64), ("stats", None)):
        payload = json.loads(bench.map.read_text(encoding="utf-8"))
        payload[field] = value
        when = bench.map.stat().st_mtime
        bench.map.write_text(json.dumps(payload), encoding="utf-8")
        _touch(bench.map, when)
        assert driver._reusable(bench.map, bench.args, **bench.kwargs) is None, field


def test_the_flag_is_off_by_default(driver, bench) -> None:
    """Without ``--reuse-maps`` nothing is skipped, however fresh the map looks.

    The default has to be rebuild: a driver that silently reused whatever was lying in its
    work directory would make every run's meaning depend on what a previous run left there.

    Turns red when: the flag's default flips, or the guard stops consulting it.
    """
    bench.args.reuse_maps = False
    assert driver._reusable(bench.map, bench.args, **bench.kwargs) is None


def test_rewriting_identical_bytes_leaves_the_mtime_alone(driver, tmp_path) -> None:
    """The variants are byte-stable across runs, which is what makes their content a fact.

    Both S3 variants are a deterministic function of the S2 stats and the seed, and the
    guard above declines to stamp against them on exactly that reasoning. Writing them
    unconditionally would no longer break the guard, but it would destroy the evidence:
    ``rewritten: false`` in the run record is the only thing that says the numbers the maps
    were priced from are the numbers on disk now.

    Turns red when: the variants go back to writing unconditionally, and the run record's
    ``rewritten`` flag becomes a constant true that attests to nothing.
    """
    target = tmp_path / "dynquant_stats.signal.json"

    assert driver._write_if_changed(target, lambda p: p.write_text("same", encoding="utf-8"))
    stamp = target.stat().st_mtime_ns

    assert not driver._write_if_changed(target, lambda p: p.write_text("same", encoding="utf-8"))
    assert target.stat().st_mtime_ns == stamp, "an identical write moved the timestamp"

    assert driver._write_if_changed(target, lambda p: p.write_text("other", encoding="utf-8"))
    assert target.read_text(encoding="utf-8") == "other"
    assert not list(tmp_path.glob("*.tmp")), "the scratch file outlived the write"
