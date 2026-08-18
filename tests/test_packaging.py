"""Packaging invariants, checked without building anything.

Every assertion here corresponds to a failure that is invisible in a source
checkout and only appears once someone runs `pip install`:

* a ``readme``/``license-files`` entry naming a file that does not exist -- the
  wheel build fails, and it fails for the person installing, not the person who
  wrote the declaration;
* the meta distribution pinning a ``dynquant-core`` version that was never
  released, so ``pip install dynquant`` resolves to nothing;
* a pre-commit hook or CI step pointing at a renamed script, which silently stops
  guarding whatever it guarded;
* a linter pinned to one version in CI and another in pre-commit, which produces
  failures contributors cannot reproduce.

The three-way version and pin checks are the same idea as ``test_abi.py``: when a
number has to be written down in more than one file, a lint is what keeps the
copies honest.
"""

from __future__ import annotations

import platform
from pathlib import Path
from unittest import mock

import pytest
import yaml

from dynquant._version import __version__

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = REPO_ROOT / "packages"
CORE = PACKAGES / "dynquant-core"
KERNELS = PACKAGES / "dynquant-kernels"
META = PACKAGES / "dynquant"

ALL_PACKAGES = (CORE, KERNELS, META)


def _load_toml_text(text: str) -> dict:
    try:
        import tomllib
    except ImportError:  # Python 3.10
        tomli = pytest.importorskip(
            "tomli", reason="TOML parsing needs Python 3.11+ or the tomli backport"
        )
        return tomli.loads(text)
    return tomllib.loads(text)


def _load_toml(path: Path) -> dict:
    return _load_toml_text(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def core_toml() -> dict:
    return _load_toml(CORE / "pyproject.toml")


@pytest.fixture(scope="module")
def kernels_toml() -> dict:
    return _load_toml(KERNELS / "pyproject.toml")


@pytest.fixture(scope="module")
def meta_toml() -> dict:
    return _load_toml(META / "pyproject.toml")


# --------------------------------------------------------------------------
# Files the build declares must exist
# --------------------------------------------------------------------------


@pytest.mark.parametrize("package", ALL_PACKAGES, ids=lambda p: p.name)
def test_declared_readme_and_license_exist(package: Path) -> None:
    """Both are declared in all three pyprojects, and a missing one is a build error.

    hatchling and scikit-build-core cannot reach outside their own project
    directory, which is why each package carries its own copy of the licence rather
    than pointing at the repository root.
    """
    data = _load_toml(package / "pyproject.toml")
    project = data["project"]

    readme = project["readme"]
    assert (package / readme).is_file(), f"{package.name} declares readme={readme!r}"

    for pattern in project["license-files"]:
        assert (package / pattern).is_file(), f"{package.name} declares license-files={pattern!r}"


@pytest.mark.parametrize("package", ALL_PACKAGES, ids=lambda p: p.name)
def test_licence_copies_are_identical(package: Path) -> None:
    """Per-package copies, one licence. A diverged copy is a licensing question that
    nobody will notice until it is a legal one."""
    root = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert (package / "LICENSE").read_text(encoding="utf-8") == root


# --------------------------------------------------------------------------
# Versions and pins
# --------------------------------------------------------------------------


def test_meta_version_matches_core(meta_toml: dict) -> None:
    """``dynquant==X`` and ``dynquant-core==X`` must mean the same code, because the
    meta package is the name users type and the one they report bugs against."""
    assert meta_toml["project"]["version"] == __version__


def test_meta_pins_core_exactly(meta_toml: dict) -> None:
    pins = [d for d in meta_toml["project"]["dependencies"] if d.startswith("dynquant-core")]
    assert pins == [f"dynquant-core=={__version__}"], pins


def test_meta_mirrors_every_core_extra(core_toml: dict, meta_toml: dict) -> None:
    """A user should never have to learn that ``dynquant-core`` exists as a separate
    name, which means every extra it offers has to be reachable from ``dynquant``."""
    core_extras = set(core_toml["project"]["optional-dependencies"])
    meta_extras = set(meta_toml["project"]["optional-dependencies"])
    missing = core_extras - meta_extras
    assert not missing, f"dynquant does not mirror dynquant-core's extras: {sorted(missing)}"

    for extra in core_extras:
        entries = meta_toml["project"]["optional-dependencies"][extra]
        assert f"dynquant-core[{extra}]=={__version__}" in entries, (
            f"dynquant[{extra}] must forward to dynquant-core[{extra}]=={__version__}, "
            f"got {entries}"
        )


def test_the_default_install_cannot_move_torch(meta_toml: dict) -> None:
    """``pip install dynquant`` must never change the torch the user already has.

    This is the regression guard for the defect that made 0.5.0's install story
    worse than no install story. ``dynquant-kernels`` is a compiled extension
    pinned to one torch minor (``torch>=2.7,<2.8``), and it was listed as an
    unconditional dependency here. pip does not decline a hard requirement it
    cannot otherwise satisfy -- it *downgrades*, so a machine carrying torch 2.13
    silently got torch 2.7.1 installed underneath it, torchvision left compiled
    against the outgoing version, and the next ``import transformers`` died on
    ``operator torchvision::nms does not exist``. The command reported success.

    So: nothing reachable from a bare ``pip install dynquant`` may carry a torch
    bound tighter than core's own open floor. The kernels stay available through
    ``dynquant[kernels]``, where the user asked for them.
    """
    deps = meta_toml["project"]["dependencies"]
    offenders = [d for d in deps if d.startswith("dynquant-kernels")]
    assert not offenders, (
        "dynquant-kernels must not be a default dependency: it pins a torch minor, "
        f"and pip satisfies such a pin by downgrading the user's torch. Got {offenders}"
    )
    assert deps == [f"dynquant-core=={__version__}"], (
        "the default install is core and nothing else; anything added here must be "
        f"proven not to constrain torch. Got {deps}"
    )


def test_kernels_is_pinned_by_range_not_exactly(meta_toml: dict) -> None:
    """The kernels wheel is rebuilt on its own cadence -- a new toolkit, a new torch
    minor -- none of which changes a line of Python. An exact pin would force a core
    release for every binary rebuild. Compatibility is the ABI handshake's job."""
    extras = meta_toml["project"]["optional-dependencies"]
    pins = [d for d in extras["kernels"] if d.startswith("dynquant-kernels")]
    assert len(pins) == 1, pins
    specifier, _, marker = pins[0].partition(";")
    assert "==" not in specifier, specifier
    assert ">=" in specifier and "<" in specifier, specifier
    # Marker-free on purpose, and this is the reverse of what the old dependency
    # asserted. As an extra the request is explicit, so a platform with no prebuilt
    # wheel should get the sdist build it asked for rather than silently nothing.
    assert not marker.strip(), (
        f"dynquant[kernels] should resolve everywhere; a marker makes it a no-op "
        f"on exactly the platforms whose users typed it deliberately. Got {marker!r}"
    )


def test_the_doctors_platform_test_matches_the_declared_wheel_platforms(
    meta_toml: dict,
) -> None:
    """``_prebuilt_wheel_exists`` decides which remedy ``dynquant doctor`` prints when
    the kernels are missing. If it drifts from where wheels are actually published,
    the doctor tells users on a wheel-less platform to run an install that cannot
    give them one -- bad advice from the command whose whole job is to explain why
    something did not work.

    The marker used to live on the meta-package's kernels dependency. That
    dependency is gone (see ``test_the_default_install_cannot_move_torch``), so the
    fact is declared under ``[tool.dynquant]`` instead -- the same claim, with no
    install-time side effect attached to stating it."""
    from dynquant.runtime.backend import _prebuilt_wheel_exists

    marker = meta_toml["tool"]["dynquant"]["prebuilt-wheel-marker"].replace('"', "'").strip()

    # Evaluate the real marker against synthetic environments and require the doctor's
    # predicate to agree on each. `packaging` ships with pip and is a test-time dep.
    from packaging.markers import Marker

    cases = [
        ({"platform_system": "Linux", "platform_machine": "x86_64"}, True),
        ({"platform_system": "Linux", "platform_machine": "aarch64"}, False),
        ({"platform_system": "Windows", "platform_machine": "AMD64"}, False),
        ({"platform_system": "Darwin", "platform_machine": "arm64"}, False),
    ]
    parsed = Marker(marker)
    for env, expected in cases:
        assert parsed.evaluate(env) is expected, (marker, env)

        with (
            mock.patch.object(platform, "system", lambda e=env: e["platform_system"]),
            mock.patch.object(platform, "machine", lambda e=env: e["platform_machine"]),
        ):
            assert _prebuilt_wheel_exists() is expected, env


def test_requires_python_agrees_across_packages(
    core_toml: dict, kernels_toml: dict, meta_toml: dict
) -> None:
    """A meta package that installs on 3.9 and then cannot resolve core is a worse
    error message than refusing 3.9 up front."""
    versions = {
        toml["project"]["name"]: toml["project"]["requires-python"]
        for toml in (core_toml, kernels_toml, meta_toml)
    }
    assert len(set(versions.values())) == 1, versions


# --------------------------------------------------------------------------
# Layout decisions worth freezing
# --------------------------------------------------------------------------


def test_the_repo_root_is_not_a_distribution() -> None:
    """The root pyproject holds shared tool configuration only. A ``[project]``
    table there would make the repository root pip-installable, and `pip install .`
    would then produce something that is not any of the three real packages."""
    root = _load_toml(REPO_ROOT / "pyproject.toml")
    assert "project" not in root
    assert "build-system" not in root
    assert {"tool"} == set(root)


def test_kernels_install_as_a_top_level_package(kernels_toml: dict) -> None:
    """``dynquant_kernels``, never ``dynquant.kernels``.

    A submodule shipped by a *different* distribution makes ``dynquant`` a split
    namespace package: an interrupted upgrade leaves a new core beside an old
    kernels half with nothing able to detect it, and the failure appears later as
    wrong numbers. A separate top-level name makes the ABI handshake the only
    coupling.
    """
    assert kernels_toml["tool"]["scikit-build"]["wheel"]["packages"] == ["src/dynquant_kernels"]
    assert (KERNELS / "src" / "dynquant_kernels" / "__init__.py").is_file()
    assert not (CORE / "src" / "dynquant" / "kernels").exists()


def test_the_meta_package_ships_no_code(meta_toml: dict) -> None:
    assert meta_toml["tool"]["hatch"]["build"]["targets"]["wheel"]["bypass-selection"] is True
    assert not (META / "src").exists()


def test_the_console_script_points_at_something_real(core_toml: dict) -> None:
    entry = core_toml["project"]["scripts"]["dynquant"]
    module, _, attribute = entry.partition(":")
    imported = __import__(module, fromlist=[attribute])
    assert callable(getattr(imported, attribute))


# --------------------------------------------------------------------------
# Guards that only work if their scripts are still where they were
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pre_commit_config() -> dict:
    return yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ci_workflow() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def ci_wheels_workflow() -> dict:
    return yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "wheels.yml").read_text(encoding="utf-8")
    )


def test_local_pre_commit_hooks_point_at_existing_scripts(pre_commit_config: dict) -> None:
    """A renamed script does not fail the hook -- it makes the hook a no-op that
    reports success, which is how a guard stops guarding without anyone noticing."""
    for repo in pre_commit_config["repos"]:
        if repo["repo"] != "local":
            continue
        for hook in repo["hooks"]:
            for token in hook["entry"].split():
                if token.endswith(".py"):
                    assert (REPO_ROOT / token).is_file(), f"hook {hook['id']}: missing {token}"


def test_ci_scripts_exist(ci_workflow: dict) -> None:
    referenced = {"scripts/check_no_confidential.py", "scripts/assert_doctor.py"}
    text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for script in referenced:
        assert script in text, f"CI no longer runs {script}"
        assert (REPO_ROOT / script).is_file()
    assert ci_workflow["name"] == "CI"


def test_ruff_version_agrees_between_ci_and_pre_commit(
    ci_workflow: dict, pre_commit_config: dict
) -> None:
    """Different versions in the two places means CI rejects what the hook accepted,
    and the contributor cannot reproduce the failure locally."""
    ci_version = ci_workflow["env"]["RUFF_VERSION"]
    revs = [repo["rev"] for repo in pre_commit_config["repos"] if "ruff-pre-commit" in repo["repo"]]
    assert revs == [f"v{ci_version}"], (revs, ci_version)


# --------------------------------------------------------------------------
# Release tooling
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stamp():
    """``scripts/stamp_kernel_version.py``, loaded by path -- ``scripts/`` is not a
    package, and this code only ever runs during a release, which is the worst
    possible time to discover it is broken."""
    import importlib.util

    path = REPO_ROOT / "scripts" / "stamp_kernel_version.py"
    spec = importlib.util.spec_from_file_location("_stamp_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_label_keeps_only_major_minor(stamp) -> None:
    """A cu12.6 build runs against any 12.x torch, and a torch patch release does not
    change the C++ ABI -- encoding either would multiply the matrix without
    describing a real incompatibility."""
    assert stamp.local_label("12.6", "2.7.1") == "cu126torch27"
    assert stamp.local_label("12.6", "2.7.1+cu126") == "cu126torch27"
    assert stamp.local_label("12.1", "2.4.1") == "cu121torch24"


def test_stamping_is_idempotent(stamp) -> None:
    """CI reuses one checkout across matrix cells. Appending twice would produce
    `0.1.0+cu126torch27+cu121torch24`, which is not a valid version at all."""
    once = stamp.stamped_version("0.1.0.dev0", "cu126torch27")
    assert once == "0.1.0.dev0+cu126torch27"
    assert stamp.stamped_version(once, "cu121torch24") == "0.1.0.dev0+cu121torch24"
    assert stamp.stamped_version(once, None) == "0.1.0.dev0"


def test_stamped_versions_are_valid_pep_440(stamp) -> None:
    from packaging.version import Version

    stamped = Version(stamp.stamped_version("0.1.0.dev0", stamp.local_label("12.6", "2.7.1")))
    assert stamped.local == "cu126torch27"
    # Same release, different build: this is what makes a range pin resolve to any
    # variant while an exact pin selects one.
    assert stamped.base_version == Version("0.1.0.dev0").base_version


def test_the_torch_pin_excludes_the_torch_that_broke_0_1_0(stamp) -> None:
    """The regression this pin exists for.

    0.1.0 shipped the kernels with an open ``torch>=2.4``, so on Linux
    ``pip install dynquant`` resolved a wheel built against torch 2.7.1+cu126
    alongside torch 2.13.0 and the CUDA 13 stack. Nothing warned: pip reported
    success and the extension failed its undefined-symbol import at runtime, which
    the loader turns into a silent fall back to the torch backend. So the assertion
    that matters is not the pin's spelling but that the resolver would now refuse
    the pairing."""
    from packaging.specifiers import SpecifierSet

    pin = stamp.torch_pin("2.7.1+cu126")
    assert pin == "torch>=2.7,<2.8"

    spec = SpecifierSet(pin.removeprefix("torch"))
    assert "2.13.0" not in spec, "the exact pairing that shipped broken in 0.1.0"
    assert "2.8.0" not in spec and "2.6.0" not in spec
    # Patch releases stay in: libtorch's C++ ABI moves per minor, and rejecting
    # 2.7.2 would send users to a source build for a wheel that loads fine.
    assert "2.7.0" in spec and "2.7.1" in spec and "2.7.9" in spec


def test_every_matrix_cell_stamps_a_torch_pin(ci_wheels_workflow: dict) -> None:
    """--torch is required by argparse, but the plain cell is the one that reaches
    PyPI and the one whose invocation was missing it. A workflow edit that drops it
    should fail here rather than at the next release."""
    jobs = ci_wheels_workflow["jobs"]
    steps = [s for j in jobs.values() for s in j.get("steps", []) if "run" in s]
    invocations = [
        line.strip()
        for s in steps
        for line in s["run"].splitlines()
        if "stamp_kernel_version.py" in line
    ]
    assert invocations, "no stamp invocations found; did the workflow move?"
    for line in invocations:
        assert "--torch" in line, line


def test_the_torch_pin_regex_matches_the_real_pyproject(stamp) -> None:
    """Companion to the version-file check below. If this regex stops matching, the
    script now raises instead of stamping nothing -- but only if the pattern and the
    file are checked against each other somewhere, which is here."""
    text = stamp.PYPROJECT_FILE.read_text(encoding="utf-8")
    stamped, count = stamp._RUNTIME_TORCH.subn(r"\g<1>torch>=9.9,<10.0\g<2>", text, count=1)
    assert count == 1, f"no runtime torch dependency found in {stamp.PYPROJECT_FILE}"

    # It must have rewritten the *runtime* dependency and left `[build-system]
    # requires` alone -- that one names torch too, and it is cibuildwheel's to set.
    parsed = _load_toml_text(stamped)
    assert parsed["project"]["dependencies"] == ["torch>=9.9,<10.0"]
    assert any(r.startswith("torch") for r in parsed["build-system"]["requires"]), (
        "the build-system requirement was rewritten or removed; it must not be touched"
    )


def test_the_stamp_regex_matches_the_real_version_file(stamp) -> None:
    """The script and scikit-build-core's metadata provider read the same line with
    two different regexes. If this one stops matching, the release silently keeps the
    previous version instead of failing."""
    text = stamp.VERSION_FILE.read_text(encoding="utf-8")
    match = stamp._ASSIGNMENT.search(text)
    assert match is not None, f"no version assignment found in {stamp.VERSION_FILE}"
    assert stamp.VERSION_FILE == (KERNELS / "src" / "dynquant_kernels" / "_version.py"), (
        "the script is pointing at a different file than this test checks"
    )
    provider = _load_toml(KERNELS / "pyproject.toml")["tool"]["scikit-build"]["metadata"]["version"]
    assert provider["input"] == "src/dynquant_kernels/_version.py"


def test_the_changelog_the_metadata_advertises_exists(core_toml: dict) -> None:
    url = core_toml["project"]["urls"]["Changelog"]
    assert url.endswith("CHANGELOG.md")
    assert (REPO_ROOT / "CHANGELOG.md").is_file()


def test_gitignore_still_blocks_the_reviewer_pdf() -> None:
    """The supplementary PDF is a confidential reviewer copy. Committing it -- even
    privately -- puts it one visibility flip away from publication, and git keeps it
    in history afterwards regardless. Patterns rather than a filename, so a
    re-download under a different name is still caught."""
    lines = {
        line.strip() for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    }
    assert "*.pdf" in lines
    assert "20710_*" in lines
