#!/usr/bin/env bash
# Re-score the three banked arms on the eager dispatch, without losing the grouped_mm side.
#
# The re-score overwrites bf16.json, dq_4b.json and dq_3b.json in place. Those three records
# are the only measurement anyone has of the dispatch `post_init` chose on this model, and
# nothing in the driver knows that. So the copy-aside is not a precaution here, it is step one
# of the measurement, and it belongs in a script rather than in someone's memory at the end of
# a thirty-hour panel.
#
# Everything this refuses, it refuses before touching a file. Run it, read what it says, and if
# it stops, the reason is the next thing to fix -- none of the stops are advisory.
set -euo pipefail

RUN=${RUN:-/workspace/runs/s4}
CLONE=${CLONE:-/workspace/dq-next}
PY=${PY:-/workspace/venv-llmc/bin/python}
PANEL="$RUN/panel"
KEEP="$RUN/panel_grouped_mm"

# `dynquant` is not installed into the venv -- every caller runs it from the clone's source
# tree, and `s4_panel.sh` is where that got exported for the panel. Inheriting it from the
# launching shell worked for as long as one person launched both from the same shell; from
# `ssh host bash rescore_eager.sh` there is no such shell. Derived from $CLONE so it cannot
# point at a different tree than the one every other line here reads.
export PYTHONPATH="$CLONE/packages/dynquant-core/src${PYTHONPATH:+:$PYTHONPATH}"

say() { printf '%s\n' "$*" >&2; }
die() { printf 'refused: %s\n' "$*" >&2; exit 1; }

# The driver, told apart from every command line that merely mentions it. `pgrep -f` matches a
# whole command line, and this campaign is full of lines carrying the driver's name without being
# it: a watcher polling for it, a `grep` over these scripts, the `ssh host bash -c '...'` wrapper
# that launches this very script with a diagnostic appended -- that last one refused a relaunch on
# 2026-08-09 with the GPU at 1 MiB and nothing scoring, and printed its own PID as the evidence.
# A guard that refuses cannot afford that: a false positive looks exactly like the true positive it
# exists to produce, so the only move left to the operator is to decide the guard is wrong.
#
# `pgrep -af` prints "PID cmdline", so field 2 is argv[0] and field 3 is argv[1]. A process
# *running* the driver has `arms_lfm2.py` in one of those two -- field 3 under an interpreter,
# which is how every script here launches it, and field 2 under its shebang. Anything that only
# mentions it has the name further along, behind a `bash -c`, a `grep` or an `ssh`. Matching on
# python in field 2 would work today and would silently miss the shebang form; position does not.
#
# A missing tool must not read as a missing driver: without `pgrep` or without `awk` this returns
# nothing, the caller takes the empty branch, and the guard announces itself as passed at the one
# moment it had something to say. So it refuses instead, and the way past it is a person looking.
driver_procs() {
  command -v pgrep >/dev/null && command -v awk >/dev/null \
    || die "no pgrep or no awk on this shell, so the running-driver guard cannot run -- and a
          guard that cannot run is not a guard that passed. Check by hand
          (ps aux | grep arms_lfm2) and re-run with DRIVER_CHECKED=1 if nothing is scoring."
  pgrep -af 'arms_lfm2\.py run' | awk '$2 ~ /arms_lfm2\.py$/ || $3 ~ /arms_lfm2\.py$/' || true
}

# 1. The panel owns the clone until it exits. It was launched from a pinned commit and pulling
#    the clone under a running driver swaps the code mid-panel for the arms not yet scored.
#    No DRIVER_CHECKED escape hatch here, unlike the other two: this is the script that starts a
#    driver, and overriding the guard wrongly means two of them writing the same panel directory.
#    The discriminator above is what makes an override unnecessary.
running=$(driver_procs)
if [ -n "$running" ]; then
  printf '%s\n' "$running" >&2
  die "the panel driver is still running. The clone is pinned until it exits."
fi

# 2. A second run of this script would copy the already-re-scored panel over the grouped_mm
#    side and leave two identical directories, which `dispatch_delta.py` would then correctly
#    refuse as "both records ran eager" -- correct, and one dispatch measurement too late.
[ -d "$PANEL" ] || die "$PANEL is not there"
if [ -e "$KEEP" ]; then
  die "$KEEP already exists. If the re-score has run, this is the grouped_mm side and
          overwriting it destroys the only copy. If it has not, move it aside by hand."
fi

# 3. The pinned commit predates --experts-impl. Running the re-score against it scores three
#    arms a second time on the same dispatch, burns eight to fifteen hours of a rented card,
#    and produces the zero `dispatch_delta.py` exists to refuse.
grep -q -- '--experts-impl' "$CLONE/experiments/phase4/arms_lfm2.py" \
  || die "$CLONE predates --experts-impl. Run sync_clone.sh against a bundle of the commits
          since the pin, then re-run this."
for needed in dispatch_delta.py probe_dispatch_agreement.py rate_profile.py; do
  [ -f "$CLONE/experiments/phase4/$needed" ] \
    || die "$CLONE has no experiments/phase4/$needed. Run sync_clone.sh first."
done

# 3b. The driver shells out to `python -m dynquant eval` once per arm and reports a failed import
#     the same way it reports a failed arm: exit 1, and on to the next one. Three arms fail in
#     under a second, the script runs to completion, and the only evidence is three "exit 1 after
#     0.0s" lines inside a log named for a fifteen-hour job. Worse, it fails *after* step 5 copies
#     the panel aside, so the retry then hits the `$KEEP already exists` guard and the operator
#     has to decide whether a copy they did not make is safe to delete. One import, checked here.
"$PY" -c 'import dynquant' 2>/dev/null \
  || die "\`$PY\` cannot import dynquant, and the driver shells out to \`python -m dynquant eval\`
          for every arm. PYTHONPATH is $PYTHONPATH -- check that $CLONE/packages/dynquant-core/src
          is a tree and not a stale path."

# 4. Snapshot the sampler's log before anything else appends to it. `/workspace/rate.sh` stamps
#    every 800-item progress line, which is the only length evidence this campaign has -- the eval
#    records keep no proxy for how many decode steps an arm took. The re-score adds three more arms
#    to that same log, and the box is not a volume, so the copy has to happen here rather than in
#    someone's memory. The profile is best-effort: naming the arms is positional and a resumed or
#    restarted arm changes the count, in which case `rate_profile.py` refuses rather than mislabel.
#    The refusal must not take the re-score down with it -- the raw log is the artifact that matters
#    and it is already copied by then.
RATE=${RATE:-/workspace/rate.log}
if [ -f "$RATE" ]; then
  cp -a "$RATE" "$RUN/rate.panel.log"
  say "snapshot: $(wc -l < "$RUN/rate.panel.log") stamped line(s) -> $RUN/rate.panel.log"
  "$PY" "$CLONE/experiments/phase4/rate_profile.py" "$RUN/rate.panel.log" \
    --arms "${RATE_ARMS:-awq_4b,dq_4b,gptq_3b,awq_3b,dq_3b}" \
    --out "$RUN/rate_profile.json" >/dev/null \
    || say "the profile refused. The snapshot is safe; re-run it by hand with the right --arms."
else
  say "no $RATE -- the sampler was not running, so this panel has no length evidence"
fi

# 5. Cheap, and it has to come first: it wants a free card, and the re-score takes the card
#    for eight to fifteen hours. It also answers the question the re-scored table will
#    rest on -- whether the linearised loop and eager are one class on a real 8B model,
#    which four places in the package assert from a four-layer CPU fp32 run. It gates
#    nothing. The re-score happens either way; what this changes is which caveat the
#    finished table carries.
say "three dispatches over 24 teacher-forced items, before the card is busy again"
cd "$CLONE"
"$PY" experiments/phase4/probe_dispatch_agreement.py \
  --model "$RUN/lfm25-8b-a1b.text2sql/merged" --items 24 --shots 2 --shot-seed 0 \
  --out "$RUN/dispatch_agreement.json" \
  || say "the probe failed. Nothing below depends on it; the pair it measures does."

say "copying the grouped_mm side aside: $PANEL -> $KEEP"
cp -a "$PANEL" "$KEEP"
say "$(find "$KEEP" -name '*.json' | wc -l) json file(s) preserved"

# 6. Stamp the re-score's own progress lines as they arrive, rather than polling a file the way
#    `/workspace/rate.sh` does. Two reasons it has to happen here and not afterwards. The sampler
#    resolves its target with `ls -t` *once*, outside its loop, so it is pinned to the panel's log
#    and will not follow this run -- without this, the re-score produces no length evidence at all.
#    And this is the run where length evidence is worth most: the same weights are scored on both
#    dispatches, so length is held fixed by construction and a per-block ratio measures the
#    dispatch multiplier directly, where the panel's profile could only bound it at 1.82.
#    `progress_printer` passes `flush=True`, so a stamp taken as the line arrives is the line's own
#    time and not a flush boundary -- this profile has none of the 15-second poll slop the panel's
#    carries.
RESCORE_LOG="$RUN/rescore.log"
stamp() { while IFS= read -r line; do printf '%s   %s\n' "$(date -u +%FT%TZ)" "$line"; done; }

# The order is dq_4b, dq_3b, bf16, and it is neither alphabetical nor incidental. `/workspace` on
# this box is not a volume, so a recycle at hour nine keeps whatever finished and loses the rest --
# and the three arms are not worth the same. `dq_4b` on eager is the headline: it is what turns the
# two comparisons the panel table currently flags -- DynQuant vs GPTQ and vs AWQ at 4 bits, the ones
# mixing expert arithmetic -- into same-dispatch comparisons, and it lands about three hours in
# rather than nine. `dq_3b` is the same comparison at 3 bits. `bf16` goes last because it is a
# ceiling: everything is compared against it, but no quantizer claim rests on it, and re-scoring it
# *measures* the dispatch effect rather than removing it from a margin. Losing bf16 to a recycle
# costs a measurement; losing dq_4b costs the result.
#
# Two invocations, because one cannot express that. `--rescore` is parsed into a frozenset and used
# as a membership test while the driver walks `plan_arms()`, whose order is fixed and opens with the
# ceiling -- so a single `--rescore dq_4b,dq_3b,bf16` runs bf16 first and spends the first three
# hours on the arm the paragraph above argues to run last. The list is not an order and writing it
# in the intended one does not make it one. Splitting the call is the whole fix: each invocation
# filters to arms the driver reaches in the order this campaign wants, `--resume` makes the second
# one skip what the first finished, and neither the driver nor the panel's semantics change during
# a re-score they are the subject of.
run_arms() {
  "$PY" experiments/phase4/arms_lfm2.py run \
    --model "$RUN/lfm25-8b-a1b.text2sql/merged" \
    --stats "$RUN/lfm25-8b-a1b.text2sql/stats/dynquant_stats.json" \
    --moments "$RUN/lfm25-8b-a1b.text2sql/stats/dynquant_moments.safetensors" \
    --out "$PANEL" --device cuda --limit 12000 --batch-size 32 \
    --resume --rescore "$1" --experts-impl eager 2>&1
}

say "re-scoring dq_4b, dq_3b, bf16 on eager -- stamped to $RESCORE_LOG"
cd "$CLONE"
{ run_arms dq_4b,dq_3b; run_arms bf16; } | stamp | tee "$RESCORE_LOG"

# 7. Best-effort for the same reason step 4 is: the arm names are positional. A resumed re-score,
#    or a `--rescore` list edited without editing RESCORE_ARMS, changes the arm count, and the
#    profile refuses rather than filing dq_3b's blocks under bf16's name. The stamped log is the
#    artifact; the json is a convenience over it.
"$PY" experiments/phase4/rate_profile.py "$RESCORE_LOG" \
  --arms "${RESCORE_ARMS:-dq_4b,dq_3b,bf16}" \
  --out "$RUN/rate_profile.rescore.json" >/dev/null \
  || say "the re-score profile refused. $RESCORE_LOG is intact; re-run it by hand with --arms."

say "the measurement the re-score paid for:"
"$PY" experiments/phase4/dispatch_delta.py --before "$KEEP" --after "$PANEL"
