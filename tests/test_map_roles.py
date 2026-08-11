"""The mechanism table has to come from the map, including the parts the map contradicts.

`map_roles.py` reads an exported allocation and prints role x width, so the report stops carrying
a thirty-two-cell table someone typed. That only helps if the tool is more trustworthy than the
typing was, and it starts out less so: it has a JSON of dotted names where the allocator had the
module tree, so every role in it is `role_of_name`, which the package documents as the last
resort. A generated table that is confidently wrong is worse than a transcribed one, because it
arrives with the authority of having been generated.

Three things keep it honest, and each is tested here.

The map records the true role on every module whose floor was breached, so those are checked and
a disagreement refuses. The map also records the floor that was *enforced*, which is not the
package default: `embedding` defaults to 4 and this campaign's model held it at 8, because the
embedding is tied to the LM head and `Policy.floor_for` takes the strictest floor across a tie.
Printing the default in a column headed "floor", one table above a breach row printing 8, is the
same defect as deriving the roles -- a computed value standing where a recorded one exists.

And where the map records nothing, the tool says so rather than implying coverage. A role is only
breached if the budget bound on it, so an unbreached role gets no second opinion at all, and this
model has one that needs it: 24 modules read as `attn.o` and 18 of them are the short-conv
block's output projection. Both floors are 4, so the widths are right and only the label is
wrong -- which is exactly the failure no amount of staring at the row would reveal.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "experiments" / "phase4" / "map_roles.py"
PANEL = REPO_ROOT / "experiments" / "phase4" / "results" / "s4-lfm25-panel" / "maps"

# Real module names from the LFM2.5-8B-A1B map, one per role the model produces. Synthesising
# names would test `role_of_name` against this file's guess at its rules rather than against the
# checkpoint the report is about.
EMBED = "model.embed_tokens"
ROUTER = "model.layers.10.feed_forward.gate"
EXPERT_DOWN = "model.layers.10.feed_forward.experts.down_proj"
ATTN_OUT = "model.layers.10.self_attn.o_proj"
CONV_OUT = "model.layers.0.conv.out_proj"


@pytest.fixture(scope="module")
def roles() -> Any:
    spec = importlib.util.spec_from_file_location("_dq_map_roles", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_dq_map_roles"] = module
    spec.loader.exec_module(module)
    return module


def _entry(bits: dict[str, int], violations: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "average_bits": 4.0,
        "nbytes": 1,
        "bits": bits,
        "histogram": {str(b): sum(1 for v in bits.values() if v == b) for b in set(bits.values())},
        "violations": violations or [],
    }


def _write(tmp_path: Path, name: str, entry: dict[str, Any]) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"maps": {"1": entry}}), encoding="utf-8")
    return path


def _run(roles: Any, *paths: Path) -> str:
    out = StringIO()
    argv = [arg for path in paths for arg in ("--map", str(path))]
    with redirect_stdout(out):
        assert roles.main(argv) == 0
    return out.getvalue()


def test_the_enforced_floor_beats_the_default(roles: Any, tmp_path: Path) -> None:
    """A tied embedding's floor is 8 and `DEFAULT_FLOOR_BITS` says 4.

    Turns red if the widths table goes back to reading the default table for a role the map
    already told it about -- which would put `embedding` two rows above `mlp.down` instead of at
    the top with the routers, and print a 4 that the breach block contradicts three lines later.
    """
    entry = _entry(
        {EMBED: 4, ROUTER: 8, EXPERT_DOWN: 2},
        [
            {
                "name": EMBED,
                "role": "embedding",
                "floor_bits": 8,
                "assigned_bits": 4,
                "num_params": 262_144_000,
            }
        ],
    )
    printed = _run(roles, _write(tmp_path, "dq_3b", entry))
    assert "| `embedding` | 8* |" in printed
    assert "was held to 8 bits, not the 4 of the default table" in printed
    # Sorted by floor, so the role the allocator protected most comes first.
    body = printed.split("|---|", 1)[1]
    assert body.index("`embedding`") < body.index("`moe.expert.down`")


def test_an_unbreached_role_falls_back_to_the_default(roles: Any, tmp_path: Path) -> None:
    """No violation, no recorded floor, no star -- and no pretence that one was checked."""
    printed = _run(roles, _write(tmp_path, "dq_4b", _entry({ROUTER: 8, EXPERT_DOWN: 2})))
    assert "| `moe.router` | 8 |" in printed
    assert "8*" not in printed
    assert "no floor breached" in printed


def test_two_maps_disagreeing_on_a_floor_refuse(roles: Any, tmp_path: Path) -> None:
    """Different floors for one role means different policies, so the columns are not comparable."""
    strict = _entry(
        {EMBED: 4, ROUTER: 8},
        [
            {
                "name": EMBED,
                "role": "embedding",
                "floor_bits": 8,
                "assigned_bits": 4,
                "num_params": 1,
            }
        ],
    )
    loose = _entry(
        {EMBED: 2, ROUTER: 8},
        [
            {
                "name": EMBED,
                "role": "embedding",
                "floor_bits": 4,
                "assigned_bits": 2,
                "num_params": 1,
            }
        ],
    )
    with pytest.raises(SystemExit) as raised:
        _run(roles, _write(tmp_path, "a", strict), _write(tmp_path, "b", loose))
    assert "different policies" in str(raised.value)


def test_a_name_derived_role_contradicting_the_recorded_one_refuses(
    roles: Any, tmp_path: Path
) -> None:
    """The only role check the format admits, and it has to stop the table rather than footnote it."""
    entry = _entry(
        {ROUTER: 8},
        [
            {
                "name": ROUTER,
                "role": "mlp.gate",
                "floor_bits": 4,
                "assigned_bits": 8,
                "num_params": 1,
            }
        ],
    )
    with pytest.raises(SystemExit) as raised:
        _run(roles, _write(tmp_path, "dq_4b", entry))
    assert "from the name alone it reads as" in str(raised.value)


def test_a_role_covering_two_blocks_is_disclosed(roles: Any, tmp_path: Path) -> None:
    """`attn.o` holds the short-conv output as well as attention's, and nothing else can say so.

    The disclosure is generic -- it groups a role's members by parent segment -- so it fires on
    any role assembled from two structurally different places without this file or the tool
    knowing what LFM2 is.
    """
    printed = _run(roles, _write(tmp_path, "dq_4b", _entry({ATTN_OUT: 4, CONV_OUT: 4, ROUTER: 8})))
    assert "`attn.o` is not one block: 1 under `conv`, 1 under `self_attn`." in printed
    assert "the label is the tool's, not the method's" in printed


def test_a_role_from_one_block_is_not_disclosed(roles: Any, tmp_path: Path) -> None:
    """The note has to be absent when there is nothing to disclose, or it stops being read."""
    printed = _run(roles, _write(tmp_path, "dq_4b", _entry({ATTN_OUT: 4, ROUTER: 8})))
    assert "not one block" not in printed


def test_a_crosstab_that_misses_a_module_refuses(roles: Any, tmp_path: Path) -> None:
    """The map counts its own widths, so the table has a second opinion on its own coverage."""
    entry = _entry({ROUTER: 8, EXPERT_DOWN: 2})
    entry["histogram"]["4"] = 7
    with pytest.raises(SystemExit) as raised:
        _run(roles, _write(tmp_path, "dq_4b", entry))
    assert "is not reading the `bits` block" in str(raised.value)


def test_maps_over_different_modules_refuse(roles: Any, tmp_path: Path) -> None:
    """A row comparing a width against a blank reads as a width of zero."""
    with pytest.raises(SystemExit) as raised:
        _run(
            roles,
            _write(tmp_path, "a", _entry({ROUTER: 8, EXPERT_DOWN: 2})),
            _write(tmp_path, "b", _entry({ROUTER: 8})),
        )
    assert "do not cover the same modules" in str(raised.value)


def test_a_file_with_no_maps_block_refuses(roles: Any, tmp_path: Path) -> None:
    path = tmp_path / "not-a-map.json"
    path.write_text(json.dumps({"bits": {ROUTER: 8}}), encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        _run(roles, path)
    assert "no `maps` block" in str(raised.value)


@pytest.mark.skipif(not PANEL.exists(), reason="the banked LFM2 panel is not in this checkout")
def test_the_banked_maps_produce_the_report_s_mechanism(roles: Any) -> None:
    """The real thing, end to end, because every refusal above is only worth what it catches here.

    These two maps are the ones §13 of the phase-4 report reads, and the three rows asserted are
    the mechanism it claims: the routers are held at 8 bits at both budgets while the expert
    down-projections go to their floor, and the 3-bit budget pays for that by breaching the tied
    embedding and fourteen of the twenty-two gate/up banks. A change that moves any of them is a
    change to the report's explanation of a nineteen-point margin.
    """
    printed = _run(roles, PANEL / "dq_4b.json", PANEL / "dq_3b.json")
    assert "| `moe.router` | 8 | 22 | 0 / 0 / 0 / 22 | 0 / 0 / 0 / 22 |" in printed
    assert "| `moe.expert.down` | 2 | 22 | 4 / 14 / 4 / 0 | 22 / 0 / 0 / 0 |" in printed
    assert "`dq_4b`: no floor breached" in printed
    assert "`dq_3b`: 15 floor(s) breached." in printed
    assert "| `embedding` | 8 | **4** | 1 | 0.26G |" in printed
    assert "| `moe.expert.gate_up` | 4 | **3** | 10 | 2.35G |" in printed
