#!/bin/bash
# Time the merge under the panel's own flags, at three batch sizes, before seven arms run.
#
# Three questions, one script.
#
# Is the merge worth evaluating? The base model scored 57.75% on 400 of these problems at
# this decode budget. A merge below that is a fine-tune to investigate, not a checkpoint to
# quantize six ways -- and five minutes is a better place to learn that than a finished
# bf16 ceiling arm.
#
# How many items can the panel afford? `--limit` unset means the whole 21 729-item test
# split on each of seven arms, and nothing in this campaign has ever recorded seconds
# against an item count for text-to-SQL.
#
# And how much does batch size buy? It is deliberately *not* a pairing field -- left-padded
# decode perturbs the last bits of a logit, but the problems are the same, so raising it
# uniformly across seven arms is legitimate. It is the only lever that buys wall clock
# without costing statistical power, so it is measured rather than left at the task default.
set -u

RUN=/workspace/runs/s4/lfm25-8b-a1b.text2sql
MERGED=$RUN/merged
N=${N:-128}
BATCHES=${BATCHES:-"32 64 128"}

[ -d "$MERGED" ] || { echo "no merge at $MERGED" >&2; exit 1; }
if pgrep -f "run_s2_finetune.py --model lfm" > /dev/null; then
  echo "the fine-tune is still running; the merge is not final" >&2; exit 1
fi

export HF_HOME=/workspace/.hf_home
export PYTHONPATH=/workspace/dq-next/packages/dynquant-core/src
cd /workspace/dq-next || exit 1

for B in $BATCHES; do
  echo
  echo "### batch $B"
  # A batch that OOMs writes no record, so the summary glob below would read the one
  # a previous probe left and price the panel off it. Remove it first: a missing row
  # is a result, a stale row wearing this run's timestamp is not.
  rm -f "/workspace/runs/s4/probe.b$B.json"
  /workspace/venv-llmc/bin/python -m dynquant eval "$MERGED" \
    --task text2sql \
    --out "/workspace/runs/s4/probe.b$B.json" \
    --label "probe-b$B" \
    --device cuda \
    --split test \
    --shots 2 \
    --shot-seed 0 \
    --prompt-style chat \
    --max-new-tokens 1024 \
    --keep-predictions 0 \
    --batch-size "$B" \
    --limit "$N" || echo "batch $B FAILED (out of memory is a result, not an error)"
done

/workspace/venv-llmc/bin/python - "$N" <<'PY'
import json
import sys
from pathlib import Path

n = int(sys.argv[1])
rows = []
for path in sorted(Path("/workspace/runs/s4").glob("probe.b*.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    batch = int(path.stem.split(".b")[1])
    rows.append((batch, record))

print()
print("=== probe ===")
for batch, record in rows:
    seconds = float(record.get("seconds", 0.0))
    print(
        f"  batch {batch:<4} {record.get('accuracy', 0):7.2%}  "
        f"{seconds:8.1f} s  ->  {seconds / n:6.3f} s/item   "
        f"style={record.get('detail', {}).get('prompt_style')!r}"
    )
print("  (base model scored 57.75% on 400 of these; prompt_style must be 'chat')")

if rows:
    best_batch, best = min(rows, key=lambda r: float(r[1].get("seconds", 1e9)))
    per_item = float(best.get("seconds", 0.0)) / n
    print()
    print(f"  seven arms at batch {best_batch}, evaluation only -- quantization is extra:")
    print(f"  {'limit':>7}  {'hours':>7}   minimum difference Holm-6 can call, at 20% discordance")
    for limit in (4000, 6000, 8000, 12000, 21729):
        hours = per_item * limit * 7 / 3600
        mdd = 2.638 * (0.20 / limit) ** 0.5 * 100
        print(f"  {limit:>7}  {hours:7.1f}   {mdd:5.2f} pts")
    print()
    print("  Measured on Ministral-8B stored hits: two same-width quantizations of one model")
    print("  disagree on 18-20% of items (median), 3-bit pairs higher than 4-bit. The prior")
    print("  DynQuant-over-GPTQ headline was +1.54 pts, so a limit whose MDD exceeds that")
    print("  buys a null result at full price.")
PY
