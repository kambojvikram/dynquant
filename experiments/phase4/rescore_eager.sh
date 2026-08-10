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
  || die "$CLONE predates --experts-impl. Sync arms_lfm2.py, panel_table.py and
          dynquant/commands/evaluate.py into it first, then re-run this."
for needed in dispatch_delta.py probe_dispatch_agreement.py; do
  [ -f "$CLONE/experiments/phase4/$needed" ] \
    || die "$CLONE has no experiments/phase4/$needed. Sync the clone first."
done

# 4. Cheap, and it has to come first: it wants a free card, and the re-score takes the card
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

say "re-scoring bf16, dq_4b, dq_3b on eager"
cd "$CLONE"
"$PY" experiments/phase4/arms_lfm2.py run \
  --model "$RUN/lfm25-8b-a1b.text2sql/merged" \
  --stats "$RUN/lfm25-8b-a1b.text2sql/stats/dynquant_stats.json" \
  --moments "$RUN/lfm25-8b-a1b.text2sql/stats/dynquant_moments.safetensors" \
  --out "$PANEL" --device cuda --limit 12000 --batch-size 32 \
  --resume --rescore bf16,dq_4b,dq_3b --experts-impl eager

say "the measurement the re-score paid for:"
"$PY" experiments/phase4/dispatch_delta.py --before "$KEEP" --after "$PANEL"
