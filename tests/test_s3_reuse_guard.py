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
    """A checkpoint, a stats file, a moments file and a map that legitimately reuses."""
    model = tmp_path / "merged"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    stats = tmp_path / "dynquant_stats.signal.json"
    stats.write_text("{}", encoding="utf-8")
    moments = tmp_path / "dynquant_moments.signal.safetensors"
    moments.write_bytes(b"\0")

    save_map = tmp_path / "map.dq3.json"
    save_map.write_text(
        json.dumps(
            {
                "schema": "dynquant_allocation_v1",
                "model": str(model),
                "stats": str(stats),
                "group_size": 128,
                "allocator": "sensitivity",
                "maps": {"3257925632": {"nbytes": 3257925632}},
            }
        ),
        encoding="utf-8",
    )
    _touch(save_map, newest(model, stats, moments) + 10)

    args = argparse.Namespace(reuse_maps=True, model=str(model), group_size=128)
    return argparse.Namespace(
        args=args,
        map=save_map,
        stats=str(stats),
        moments=str(moments),
        kwargs={
            "keys": ["3257925632"],
            "stats": str(stats),
            "allocator": "sensitivity",
            "moments": str(moments),
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


def test_a_map_older_than_its_stats_file_is_rebuilt(driver, bench) -> None:
    """The map is valid in every respect except that it cannot have been built from this.

    This is the whole reason the guard reads mtimes: re-running the signal extraction and
    then reusing yesterday's map produces a file that passes every structural check, names
    the right stats path, and describes an allocation computed from different numbers.

    Turns red when: the freshness comparison is dropped, or is made against the map's own
    directory rather than its inputs.
    """
    _touch(Path(bench.stats), bench.map.stat().st_mtime + 10)
    assert driver._reusable(bench.map, bench.args, **bench.kwargs) is None


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
    """The reuse guard reads mtimes, so the variants must only touch theirs on a change.

    Both S3 variants are a deterministic function of the S2 stats and the seed. Written
    unconditionally they are newer than every map derived from them on every run, and
    ``--reuse-maps`` can never answer yes for the three arms that read one -- which is
    exactly what the first attempt at this did, rebuilding five of six maps.

    Turns red when: the variants go back to writing unconditionally, in which case the
    reuse flag still exists, still passes its own tests, and quietly saves nothing.
    """
    target = tmp_path / "dynquant_stats.signal.json"

    assert driver._write_if_changed(target, lambda p: p.write_text("same", encoding="utf-8"))
    stamp = target.stat().st_mtime_ns

    assert not driver._write_if_changed(target, lambda p: p.write_text("same", encoding="utf-8"))
    assert target.stat().st_mtime_ns == stamp, "an identical write moved the timestamp"

    assert driver._write_if_changed(target, lambda p: p.write_text("other", encoding="utf-8"))
    assert target.read_text(encoding="utf-8") == "other"
    assert not list(tmp_path.glob("*.tmp")), "the scratch file outlived the write"
