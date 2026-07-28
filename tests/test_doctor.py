"""Backend selection, ``dynquant doctor``, and the CLI.

The scenario under test is the one most users hit first and the one that is easiest
to get wrong: **no compiled kernels, no GPU.** A quantizer that refuses to run
without CUDA would be useless on a laptop, and a doctor that reported FAIL on a
perfectly good CPU install would train users to ignore it. So the assertions here
are mostly about *not* failing -- while still failing loudly for the cases that
genuinely invalidate results.
"""

from __future__ import annotations

import json

import pytest

from dynquant.cli import main
from dynquant.constants import ENV_BACKEND
from dynquant.doctor import diagnose, render
from dynquant.errors import BackendUnavailableError
from dynquant.runtime import Backend, available_backends, backend_report, resolve_backend


@pytest.fixture
def no_backend_env(monkeypatch):
    """Remove any inherited $DYNQUANT_BACKEND and drop the probe cache.

    The cache is process-wide and lru_cached, so a test that changes the
    environment must invalidate it both before and after or it leaks into
    unrelated tests through whichever one happens to run first.
    """
    monkeypatch.delenv(ENV_BACKEND, raising=False)
    available_backends(refresh=True)
    yield
    available_backends(refresh=True)


# --------------------------------------------------------------------------
# Backend selection
# --------------------------------------------------------------------------


def test_torch_backend_is_always_available(no_backend_env):
    """The fallback must be unconditional. If it can fail, ``resolve_backend`` has
    a path that returns nothing and every caller needs a None check."""
    statuses = {status.backend: status for status in available_backends()}
    assert statuses[Backend.TORCH].available


def test_every_unavailable_backend_explains_itself(no_backend_env):
    """A bare "backend unavailable" sends the user to the issue tracker. Naming
    the missing piece and the command that installs it does not."""
    for status in available_backends():
        if not status.available:
            assert status.reason, f"{status.backend} is unavailable with no reason given"


def test_resolution_falls_back_without_raising(no_backend_env):
    assert resolve_backend() in set(Backend)


def test_explicitly_requested_missing_backend_raises(no_backend_env):
    """The rule that makes benchmarks trustworthy: asking for ``cuda`` and getting
    ``torch`` would turn a memory-savings measurement into a fabrication, so an
    unhonourable request is an error rather than a downgrade."""
    statuses = {status.backend: status for status in available_backends()}
    unavailable = [b for b, s in statuses.items() if not s.available]
    if not unavailable:
        pytest.skip("every backend is available on this machine")
    with pytest.raises(BackendUnavailableError):
        resolve_backend(unavailable[0])


def test_env_override_is_attributed_to_the_env_var(no_backend_env, monkeypatch):
    """The message has to say *where* the request came from -- a CI job that
    exports DYNQUANT_BACKEND in a base image is otherwise very hard to debug."""
    statuses = {status.backend: status for status in available_backends()}
    unavailable = [b for b, s in statuses.items() if not s.available]
    if not unavailable:
        pytest.skip("every backend is available on this machine")
    monkeypatch.setenv(ENV_BACKEND, unavailable[0].value)
    with pytest.raises(BackendUnavailableError) as excinfo:
        resolve_backend()
    assert ENV_BACKEND in str(excinfo.value)


def test_unknown_backend_name_lists_the_valid_ones(no_backend_env):
    with pytest.raises(BackendUnavailableError) as excinfo:
        resolve_backend("cudaa")
    message = str(excinfo.value)
    for backend in Backend:
        assert backend.value in message


def test_backend_report_is_json_serialisable(no_backend_env):
    """It ends up inside ``dynquant doctor --json`` and in bug reports, so a
    dataclass or an Enum leaking through would break the one path that exists for
    getting diagnostics out of a user's machine."""
    json.dumps(backend_report())


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def test_doctor_passes_on_a_cpu_only_install():
    """The headline property: a CPU-only, kernel-less install is a *valid* install
    for the quantization half of DynQuant, and doctor must say so."""
    report = diagnose()
    failures = [check for check in report.checks if check.failed]
    assert not failures, "\n".join(f"{c.name}: {c.message}" for c in failures)
    assert report.ok


def test_numerical_checks_actually_ran():
    """Guard against the self-check quietly becoming a no-op. These two need
    neither a GPU nor an extension, so they must never report ``skip``."""
    statuses = {check.name: check.status for check in diagnose().checks}
    assert statuses["packing"] == "ok"
    assert statuses["quantization-error"] == "ok"


def test_missing_kernels_is_not_a_failure():
    """``warn``/``skip`` for a missing extension, never ``fail``: it costs speed
    and memory, not correctness. Crying wolf here is how a doctor stops being
    read."""
    kernels = next(check for check in diagnose().checks if check.name == "kernels")
    assert kernels.status in {"ok", "warn", "skip"}


def test_report_round_trips_through_json():
    payload = json.dumps(diagnose().as_dict())
    assert json.loads(payload)["ok"] is True


def test_rendered_report_is_ascii_only():
    """This output gets pasted into GitHub issues and CI logs through terminals
    with every imaginable code page. A box-drawing character that renders as
    mojibake -- or raises UnicodeEncodeError on a cp1252 console mid-print --
    destroys the diagnostic at the moment it is needed."""
    text = render(diagnose())
    text.encode("ascii")  # raises if anything non-ASCII crept in
    assert "Result:" in text


def test_rendered_report_names_the_selected_backend():
    text = render(diagnose())
    assert resolve_backend().value in text


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_doctor_exits_zero(capsys):
    assert main(["doctor"]) == 0
    assert "Result: OK" in capsys.readouterr().out


def test_cli_doctor_json_is_parseable(capsys):
    assert main(["doctor", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert {"environment", "backends", "checks"} <= set(payload)


def test_cli_version_reports_the_contract_numbers(capsys):
    from dynquant._version import KERNEL_ABI_VERSION

    assert main(["version", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["kernel_abi"] == KERNEL_ABI_VERSION
    assert payload["dynquant-core"]


def test_bare_invocation_is_a_usage_error():
    """Exit 2, not 0. ``dynquant`` with no arguments succeeding silently reads as
    "the command worked" in a shell script."""
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
