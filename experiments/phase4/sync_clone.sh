#!/usr/bin/env bash
# Move the local commits into the box's clone as a git fetch, not as fifty scp'd files.
#
# `rescore_eager.sh` refuses until the clone carries `--experts-impl`, and its instruction for
# getting there is a sentence: "sync arms_lfm2.py, panel_table.py and evaluate.py into it first."
# Fifty files across two trees have drifted since the panel was launched, so that sentence is a
# way to arrive at a clone that runs code no commit describes. This is the sentence as a command.
#
# A bundle rather than scp because of what it preserves. The clone stays a git repository, so
# `git log -1` on the box answers "what code produced this record" after the fact, which is the
# question every one of these reports has had to answer at least once. And a bundle carries its
# own basis: it will not apply to a clone that is not where this script thinks it is.
#
#   local:  git bundle create /tmp/s4-sync.bundle <pinned>..HEAD
#           scp -P <port> /tmp/s4-sync.bundle root@<host>:/workspace/
#   box:    bash sync_clone.sh
#
# Everything it refuses, it refuses before touching the clone.
set -euo pipefail

CLONE=${CLONE:-/workspace/dq-next}
BUNDLE=${BUNDLE:-/workspace/s4-sync.bundle}
PINNED=${PINNED:-4109dcc476da0b956187519d2a8fd5153d9e3406}

say() { printf '%s\n' "$*" >&2; }
die() { printf 'refused: %s\n' "$*" >&2; exit 1; }

# 1. The same refusal `rescore_eager.sh` opens with, and for a stronger reason here: that script
#    only reads the clone, this one rewrites it. A driver part-way through arm five loads the next
#    arm's code from disk when it gets there, so a sync under a running panel produces a panel
#    whose arms were scored by two different versions and no record of which was which.
#    And an absent `pgrep` must not look like an absent driver. `if pgrep ...` on a shell without
#    it takes the false branch and syncs, which is the failure this whole script exists to avoid,
#    announced as a passing check. So the tool is required, and the only way past it is a person
#    saying they looked.
if [ "${DRIVER_CHECKED:-}" != 1 ]; then
  command -v pgrep >/dev/null \
    || die "no pgrep on this shell, so the running-driver guard cannot run -- and a guard that
          cannot run is not a guard that passed. Check by hand (ps aux | grep arms_lfm2) and
          re-run with DRIVER_CHECKED=1 if nothing is scoring."
  if pgrep -af 'arms_lfm2\.py run' >/dev/null; then
    pgrep -af 'arms_lfm2\.py run' >&2
    die "the panel driver is still running. The clone is pinned until it exits."
  fi
fi

[ -d "$CLONE/.git" ] || die "$CLONE is not a git clone. A bundle has nothing to apply to."
[ -f "$BUNDLE" ] || die "no $BUNDLE. Build it locally and scp it here first:
          git bundle create s4-sync.bundle $PINNED..HEAD"

git -C "$CLONE" bundle verify "$BUNDLE" >/dev/null 2>&1 \
  || die "$BUNDLE does not verify against this clone. Either it is truncated, or it was built
          from a basis this clone does not have -- check that HEAD here is $PINNED."

# 2. A dirty tree is not a merge conflict waiting to happen, it is a finding. The panel's records
#    are reported as having been produced by a pinned commit. If the clone has uncommitted edits
#    then that claim is false, and overwriting them destroys the only evidence of what actually
#    ran. Stop and let a person read the diff.
if [ -n "$(git -C "$CLONE" status --porcelain)" ]; then
  git -C "$CLONE" status --short >&2
  die "the clone has uncommitted changes. The panel's records are attributed to a pinned commit
          and this says they are not. Read the diff before it is overwritten -- if it is
          incidental, stash it by hand and re-run."
fi

HEAD_NOW=$(git -C "$CLONE" rev-parse HEAD)
say "clone is at $HEAD_NOW"
[ "$HEAD_NOW" = "$PINNED" ] || say "note: that is not the expected pin $PINNED. The bundle verified
          against it, so the fetch is safe -- but the report saying which commit scored the panel
          is describing a different tree than this one."

say "fetching $(basename "$BUNDLE") into $CLONE"
# Detached first, because the previous run left the clone standing on the branch this fetch
# writes and git refuses to fetch into a checked-out ref. That refusal is correct and it makes
# the script single-use, which is backwards: the sync that carries a fix found by running the
# code is always the second one. The tree is already known clean -- guard 2 above refuses a dirty
# one -- so detaching moves nothing and loses nothing.
git -C "$CLONE" checkout --quiet --detach
# Not forced. A bundle whose HEAD is not a descendant of what is already here means the local
# history was rewritten under a campaign whose records are attributed to its commits, and the
# fetch refusing is the correct end of that.
git -C "$CLONE" fetch --no-tags "$BUNDLE" 'HEAD:refs/heads/s4-sync'
git -C "$CLONE" checkout --quiet s4-sync

say "clone is now at $(git -C "$CLONE" rev-parse --short HEAD) on branch s4-sync"
say "$(git -C "$CLONE" rev-list --count "$HEAD_NOW..HEAD") commit(s) applied"

# 3. Say what arrived, in the terms the next script checks. `rescore_eager.sh` dies on a clone
#    missing any of these, and discovering that after the driver has been killed and the card is
#    idle is the expensive order to discover it in.
for needed in arms_lfm2.py panel_table.py dispatch_delta.py probe_dispatch_agreement.py \
  rate_profile.py rescore_eager.sh build_kernels.sh; do
  if [ -f "$CLONE/experiments/phase4/$needed" ]; then
    say "  ok      experiments/phase4/$needed"
  else
    say "  MISSING experiments/phase4/$needed"
  fi
done
grep -q -- '--experts-impl' "$CLONE/experiments/phase4/arms_lfm2.py" \
  && say "  ok      arms_lfm2.py carries --experts-impl" \
  || say "  MISSING --experts-impl in arms_lfm2.py -- rescore_eager.sh will refuse"

# And the kernel source, checked here rather than only in build_kernels.sh. That script refuses
# a clone without it, which is the right refusal in the wrong place: by then the panel is over,
# the card is idle, and the answer is another bundle from a laptop. A bundle can verify, fetch
# and check out cleanly and still be a bundle cut before the kernel was written.
if [ -f "$CLONE/packages/dynquant-kernels/csrc/moe/grouped_gemv.cu" ]; then
  say "  ok      csrc/moe/grouped_gemv.cu"
else
  say "  MISSING csrc/moe/grouped_gemv.cu -- build_kernels.sh will refuse"
fi
