"""Registration writes three containers by hand, so each write needs a test.

vLLM publishes ``register_quantization_config`` and
``register_weight_loader_v2_supported_method``; SGLang publishes neither, because
its quantization layer was forked from vLLM ``v0.5.5`` before those decorators
existed. So :func:`dynquant.integration.sglang_plugin.register` reaches into
module-level mutables, two of which are private by any reasonable reading. Tests
here answer, for each one: does the write land, does it land *twice* when
registration runs twice, and does a rename upstream produce a message that names
what moved.

Run against the stub in :mod:`_sglang_stub`, not a real SGLang, and
:mod:`test_sglang_stub_conformance` is the other half of that bargain -- see the
stub's docstring. The short version: SGLang ships manylinux-only wheels from 0.5.11,
so ``importorskip("sglang")`` would skip this file on every developer machine and
every CPU runner, which is where these particular mistakes get made.
"""

from __future__ import annotations

import sys

import pytest

from dynquant.constants import HF_QUANT_METHOD
from dynquant.integration.sglang_plugin import SGLangIncompatibleError, register

from _sglang_stub import REMOVED, fake_sglang


def test_the_method_becomes_resolvable_by_name():
    with fake_sglang() as quantization:
        register()
        assert HF_QUANT_METHOD in quantization.QUANTIZATION_METHODS

        from dynquant.integration.sglang_plugin.config import DynQuantConfig

        assert quantization.QUANTIZATION_METHODS[HF_QUANT_METHOD] is DynQuantConfig


def test_the_write_goes_to_the_dict_that_is_read():
    """0.5.16 split the registry; only one half is consulted at serve time.

    ``QUANTIZATION_METHODS = {**BASE_QUANTIZATION_METHODS}`` runs at import, so the
    copy is already taken before any plugin loads. A write to ``BASE_`` is a silent
    no-op that leaves ``--quantization dynquant`` unresolvable -- an easy mistake to
    make, because ``BASE_`` is the one that reads like the source of truth.
    """
    with fake_sglang() as quantization:
        register()
        assert HF_QUANT_METHOD not in quantization.BASE_QUANTIZATION_METHODS


def test_the_cli_accepts_the_flag():
    with fake_sglang():
        import sglang.srt.server_args as server_args

        register()
        assert HF_QUANT_METHOD in server_args.QUANTIZATION_CHOICES


def test_the_v2_weight_loader_is_opted_into():
    """The load-bearing write, and the one whose absence is hardest to diagnose.

    ``WEIGHT_LOADER_V2_SUPPORTED`` is a list of class-name strings, matched at
    ``linear.py:369`` and ``:1454`` to pick between two loaders. The v1 loader places
    shards with ``param.data.narrow(output_dim, ...)``, which assumes a rectangular
    parameter; DynQuant's are flat, so v1 has no way to express the layout and the
    parameter's own placement hooks are never reached. It fails rather than
    corrupts -- every v1 path asserts the shapes before copying -- but as a bare
    ``AssertionError`` inside a spawned scheduler subprocess. The negative test that
    runs those v1 bodies against our buffers lands with the linear method.
    """
    with fake_sglang():
        import sglang.srt.layers.linear as linear

        before = list(linear.WEIGHT_LOADER_V2_SUPPORTED)
        register()
        assert "DynQuantLinearMethod" in linear.WEIGHT_LOADER_V2_SUPPORTED
        # SGLang's own entries survive: this is an append to a shared list, and
        # replacing it would disable the v2 loader for GPTQ and AWQ.
        assert linear.WEIGHT_LOADER_V2_SUPPORTED[: len(before)] == before


@pytest.mark.parametrize("times", [2, 5])
def test_registering_repeatedly_changes_nothing(times):
    """``load_plugins()`` is guarded by a flag, but is reachable by several routes.

    It runs from the server entrypoint and again as the first statement of
    ``run_scheduler_process()`` -- the scheduler is spawned, not forked, so it has to.
    Any of those may be reached more than once in a process that imports SGLang more
    than once. ``add_quantization_method_choices`` is a bare ``list.extend`` with no
    dedup, so a second registration would put ``dynquant`` in ``--quantization``'s
    help text twice.
    """
    with fake_sglang() as quantization:
        import sglang.srt.layers.linear as linear
        import sglang.srt.server_args as server_args

        for _ in range(times):
            register()

        assert server_args.QUANTIZATION_CHOICES.count(HF_QUANT_METHOD) == 1
        assert linear.WEIGHT_LOADER_V2_SUPPORTED.count("DynQuantLinearMethod") == 1
        assert list(quantization.QUANTIZATION_METHODS).count(HF_QUANT_METHOD) == 1


# Each row is a way 0.5.x could move under us: the symbol goes away, or it stays and
# becomes something the write does not work on. Both reach the user as a traceback
# from inside a spawned scheduler subprocess, which is why they are converted here.
DRIFT = [
    ("sglang.srt.layers.quantization:QUANTIZATION_METHODS", REMOVED),
    ("sglang.srt.layers.quantization:QUANTIZATION_METHODS", ["awq", "gptq"]),
    ("sglang.srt.server_args:QUANTIZATION_CHOICES", REMOVED),
    ("sglang.srt.server_args:QUANTIZATION_CHOICES", {"awq", "gptq"}),
    ("sglang.srt.server_args:add_quantization_method_choices", REMOVED),
    ("sglang.srt.server_args:add_quantization_method_choices", "not callable"),
    ("sglang.srt.layers.linear:WEIGHT_LOADER_V2_SUPPORTED", REMOVED),
    ("sglang.srt.layers.linear:WEIGHT_LOADER_V2_SUPPORTED", ("a", "tuple")),
]


@pytest.mark.parametrize(("target", "replacement"), DRIFT)
def test_drift_raises_something_a_user_can_act_on(target, replacement):
    symbol = target.replace(":", ".")
    with fake_sglang(**{target: replacement}), pytest.raises(SGLangIncompatibleError) as excinfo:
        register()

    message = str(excinfo.value)
    assert symbol in message, message
    # The version, because "SGLang changed" is only actionable with a which.
    assert "sglang" in message
    assert "github.com/kambojvikram/dynquant/issues" in message


def test_drift_is_detected_before_anything_expensive_is_imported():
    """An incompatible SGLang should cost an AttributeError's worth of work.

    All four container checks run ahead of the ``config`` import, which pulls in the
    schema and the packing code behind it. On a machine where SGLang has moved, the
    right outcome is a fast readable error -- not several seconds of imports and then
    a fast readable error.
    """
    target = "sglang.srt.layers.linear:WEIGHT_LOADER_V2_SUPPORTED"
    with fake_sglang(**{target: REMOVED}):
        with pytest.raises(SGLangIncompatibleError):
            register()
        assert "dynquant.integration.sglang_plugin.config" not in sys.modules


def test_the_stub_leaves_no_trace():
    """The fixture's own contract, asserted rather than assumed.

    ``DynQuantConfig`` binds whichever ``QuantizationConfig`` was importable when its
    class body ran. A stub-based one surviving the context would be reused by
    :mod:`test_sglang_stub_conformance` on the Linux box, where a real SGLang is
    present -- and that test would then be comparing the stub against itself.

    Checked on both routes a later test could take, because they are not the same
    lookup: ``sys.modules`` and the attribute the import system leaves on the parent
    package.

    Compared against a snapshot rather than asserted absent. Absence is the right
    answer only where nothing has imported SGLang, and the machine where this test
    matters most is the other kind: there
    :mod:`test_sglang_stub_conformance` has already imported a real SGLang and built
    a real ``DynQuantConfig`` against it, and ``fake_sglang`` is required to
    *restore* both, not evict them. The snapshot still catches a leak -- a
    stub-built module left behind is a different object at the same key -- and it
    catches over-eager teardown as well, which "not in sys.modules" cannot.
    """
    import dynquant.integration.sglang_plugin as plugin

    def snapshot():
        # Top-level segment, not a prefix: `sglang_router` is a different distribution.
        modules = {
            name: module
            for name, module in sys.modules.items()
            if name.split(".")[0] == "sglang"
            or name == plugin.__name__
            or name.startswith(plugin.__name__ + ".")
        }
        return modules, getattr(plugin, "config", None)

    before = snapshot()

    with fake_sglang():
        register()
        assert "dynquant.integration.sglang_plugin.config" in sys.modules

    assert snapshot() == before
    assert sys.modules[plugin.__name__] is plugin
