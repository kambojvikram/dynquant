#!/usr/bin/env bash
# Compile the kernels on the box and put the grouped GEMV under its own tests.
#
# Everything about `moe_grouped_gemv` that a laptop can check has been checked: the segment
# table's dtype and residency, the four fallbacks, fused-against-loop equality, the host-read
# counter, nine mutations. None of that is evidence about the `.cu` file, which has never been
# compiled. This is the script that turns the remaining claims into results, and it is written
# down rather than typed because the order matters and two of the steps are refusals.
#
#   local:  bash experiments/phase4/sync_clone.sh   (see that script; needs a bundle)
#   box:    bash experiments/phase4/build_kernels.sh
#
# Everything it refuses, it refuses before invoking a compiler.
set -euo pipefail

CLONE=${CLONE:-/workspace/dq-next}
PY=${PY:-/workspace/venv-llmc/bin/python}
JOBS=${JOBS:-$(nproc)}

say() { printf '%s\n' "$*" >&2; }
die() { printf 'refused: %s\n' "$*" >&2; exit 1; }

# 1. The same guard `sync_clone.sh` and `rescore_eager.sh` open with, and for the same reason
#    plus one of this script's own: a build writes into the venv the panel is importing from.
#    An in-place `pip install` while a driver is thirty hours into a seven-arm run replaces
#    modules under a live process, and the failure mode is not a clean crash.
#
#    An absent `pgrep` must not read as an absent driver. `if pgrep ...` on a shell without it
#    takes the false branch and builds, announcing the guard as passed.
if [ "${DRIVER_CHECKED:-}" != 1 ]; then
  command -v pgrep >/dev/null \
    || die "no pgrep on this shell, so the running-driver guard cannot run -- and a guard that
          cannot run is not a guard that passed. Check by hand (ps aux | grep arms_lfm2) and
          re-run with DRIVER_CHECKED=1 if nothing is scoring."
  if pgrep -af 'arms_lfm2\.py run' >/dev/null; then
    pgrep -af 'arms_lfm2\.py run' >&2
    die "the panel driver is still running. The GPU and this venv are its until it exits."
  fi
fi

# 2. A clone without the source is a clone that will build the *old* extension, pass every test
#    in the file, and report a green run for a kernel that is not in it. `test_abi.py` catches
#    the schema half of that; nothing catches "you never synced".
[ -f "$CLONE/packages/dynquant-kernels/csrc/moe/grouped_gemv.cu" ] \
  || die "$CLONE has no csrc/moe/grouped_gemv.cu. Run sync_clone.sh first -- building now would
          produce a green run for an extension that does not contain the kernel under test."

cd "$CLONE"
say "clone at $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD)"

# 3. Say which toolkit before using it. The last build on a box this shape picked up torch's
#    vendored nvcc -- 13.3, with headers stamped 13000 -- reached through `-isystem` ahead of
#    the system toolkit, and the mismatch surfaced hundreds of lines into a template error.
#    CMakeLists resolves that now; printing the inputs is how the next surprise gets named in
#    one line instead of read out of a traceback.
say "toolkit:  $(command -v nvcc || echo 'no nvcc on PATH') -- $(nvcc --version 2>/dev/null | tail -1 || true)"
say "torch:    $("$PY" -c 'import torch; print(torch.__version__, torch.version.cuda)')"
say "device:   $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)"

# 4. One architecture, not eight. The default list is right for a wheel and wrong for this: the
#    GEMV alone is 4 bits x 4 row tiles x 3 dtypes and the grouped kernel adds the same again,
#    so every extra `-real` target is another full instantiation sweep. Build for the card in
#    the box and nothing else.
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
ARCHS=${ARCHS:-${CC}-real}
say "building for $ARCHS with $JOBS jobs"

# `--no-deps`, deliberately: this venv holds a working llm-compressor and the transformers pin
# four arms were scored against. A resolver run here can move either, and the panel's records
# would then describe an environment that no longer exists.
DYNQUANT_CUDA_ARCHS="$ARCHS" CMAKE_BUILD_PARALLEL_LEVEL="$JOBS" \
  "$PY" -m pip install --no-build-isolation --no-deps --force-reinstall -v \
  ./packages/dynquant-kernels 2>&1 | tail -25

# 5. The binary's own ABI number, which is the one thing `test_abi.py` cannot ask for. That test
#    compares three *source* declarations so it can run on a CPU box; it is satisfied by three
#    files agreeing and says nothing about what was compiled. A stale build directory that
#    relinked an old object would pass it and fail here.
# The clone is passed in rather than assumed from the working directory. This step
# compares the freshly built binary against core's declaration of the ABI it expects,
# so which core answers decides what the comparison means -- and `dynquant` is not
# installed in this venv at all, so without the path the step dies on an import error
# after the compile has already cost ten minutes.
"$PY" - "$CLONE" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]) / "packages" / "dynquant-core" / "src"))

import torch
import dynquant_kernels
from dynquant._version import KERNEL_ABI_VERSION

print("core:", Path(__import__("dynquant").__file__).resolve())

assert dynquant_kernels.is_available(), dynquant_kernels.diagnostics()
built = torch.ops.dynquant.abi_version()
print(f"binary abi {built}, core expects {KERNEL_ABI_VERSION}")
assert built == KERNEL_ABI_VERSION, "the extension on disk is not the one this tree describes"
assert hasattr(torch.ops.dynquant, "moe_grouped_gemv"), "built, but without the grouped op"
print("grouped op present")
PY

# 6. The tests. Parity first and alone, because it is the only file with anything new to say --
#    if the grouped kernel disagrees with `gemv_kernel` band for band, nothing downstream is
#    worth reading.
say "--- grouped parity ---"
"$PY" -m pytest tests/test_kernels_parity.py -q --no-header -k grouped

say "--- the rest of the kernel surface ---"
"$PY" -m pytest tests/test_kernels_parity.py tests/test_abi.py tests/test_expert_bank.py \
  tests/test_experts_dispatch.py -q --no-header

# 7. Re-run the dispatch probe, which is now a different measurement than when it was written.
#    It reported 0.00% argmax disagreement between `dynquant_experts_forward` and `grouped_mm`
#    at the 8B's own MoE geometry -- and at the time, that forward was the Python loop. With
#    the extension present the same call goes through the kernel, so the probe re-answers its
#    own question about the thing that will actually serve. A packed bank scoring 0.00% by loop
#    and something else by kernel is the one failure the parity tests cannot see: they compare
#    bands, and this compares the model.
say "--- the dispatch probe, now running through the kernel ---"
"$PY" experiments/phase4/probe_experts_dispatch.py --scale wide

# What is still owed after all of this: a clock. Nothing above times anything, because there is
# no decode baseline on this card and a threshold invented here would be a number someone made
# up. The >=3x gate in P8 stays open, and the grouped kernel having no vectorized variant is the
# first thing standing in front of it.
