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

say() { printf '%s\n' "$*" >&2; }
die() { printf 'refused: %s\n' "$*" >&2; exit 1; }

# 1. The panel owns the clone until it exits. It was launched from a pinned commit and pulling
#    the clone under a running driver swaps the code mid-panel for the arms not yet scored.
if pgrep -af 'arms_lfm2\.py run' >/dev/null; then
  pgrep -af 'arms_lfm2\.py run' >&2
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

say "re-scoring bf16, dq_4b, dq_3b on eager -- stamped to $RESCORE_LOG"
cd "$CLONE"
"$PY" experiments/phase4/arms_lfm2.py run \
  --model "$RUN/lfm25-8b-a1b.text2sql/merged" \
  --stats "$RUN/lfm25-8b-a1b.text2sql/stats/dynquant_stats.json" \
  --moments "$RUN/lfm25-8b-a1b.text2sql/stats/dynquant_moments.safetensors" \
  --out "$PANEL" --device cuda --limit 12000 --batch-size 32 \
  --resume --rescore bf16,dq_4b,dq_3b --experts-impl eager 2>&1 | stamp | tee "$RESCORE_LOG"

# 7. Best-effort for the same reason step 4 is: the arm names are positional. A resumed re-score,
#    or a `--rescore` list edited without editing RESCORE_ARMS, changes the arm count, and the
#    profile refuses rather than filing dq_3b's blocks under bf16's name. The stamped log is the
#    artifact; the json is a convenience over it.
"$PY" experiments/phase4/rate_profile.py "$RESCORE_LOG" \
  --arms "${RESCORE_ARMS:-bf16,dq_4b,dq_3b}" \
  --out "$RUN/rate_profile.rescore.json" >/dev/null \
  || say "the re-score profile refused. $RESCORE_LOG is intact; re-run it by hand with --arms."

say "the measurement the re-score paid for:"
"$PY" experiments/phase4/dispatch_delta.py --before "$KEEP" --after "$PANEL"
