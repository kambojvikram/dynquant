#!/bin/bash
# The seven-arm panel on the fine-tuned LFM2.5-8B-A1B merge.
#
# Gate first, launch second, and the gate is the point: the arms cannot tell a stats file
# that measured 91.5% of this model from one that measured 8.5% of it and wrote hundreds of
# attention entries anyway. Every check here is cheap; the thing it guards is seven GPU-hours.
#
#   ./s4_panel.sh                gate only, prints and exits
#   ./s4_panel.sh --go <LIMIT>   gate, then launch the panel detached over LIMIT items
set -u

RUN=/workspace/runs/s4/lfm25-8b-a1b.text2sql
MERGED=$RUN/merged
STATS=$RUN/stats/dynquant_stats.json
MOMENTS=$RUN/stats/dynquant_moments.safetensors
OUT=/workspace/runs/s4/panel
LOG=/workspace/runs/s4/panel.log

fail() { echo "GATE FAILED: $*" >&2; exit 1; }

# 1. The fine-tune is finished, not merely far along. A merge written by a live trainer is
#    a checkpoint of an unfinished model with the right name.
if pgrep -f "run_s2_finetune.py --model lfm" > /dev/null; then
  fail "the fine-tune is still running; the merge is not final"
fi

# 2. The merge exists and carries a tokenizer. The panel's baseline arms load the tokenizer
#    from the directory the weights came from, and a merge saved without one fails at the
#    second arm, after a 256-sample calibration pass over 8 B parameters.
[ -d "$MERGED" ] || fail "no merged checkpoint at $MERGED"
[ -f "$MERGED/config.json" ] || fail "$MERGED has no config.json"
ls "$MERGED"/*.safetensors > /dev/null 2>&1 || fail "$MERGED has no weights"
[ -f "$MERGED/tokenizer_config.json" ] || fail "$MERGED has no tokenizer"
[ -f "$MERGED/chat_template.jinja" ] || echo "WARNING: no chat_template.jinja beside the merge -- every arm will render a bare string and agree with itself about it"

# 3. The signal file is the finished run's, not a flush from the middle of it. This was
#    an mtime ordering -- stats must postdate the merge -- and that test is backwards by
#    construction: DynQuantCallback flushes inside trainer.train() and save_outputs writes
#    the merge after it returns, so on a correct run the signal file is always the OLDER of
#    the two and the warning fired on every correct run. A warning that always fires is one
#    nobody reads, and it sat directly above the check that guards seven GPU-hours.
#    Ask the file instead: every layer records grad_norm_count, the optimizer step it was
#    last written at, so the artifact can say for itself whether it is the finished one. The
#    denominator comes from the training log rather than a literal, and the comparison takes
#    max() across layers, not min() -- the embedding and 18 depthwise convs are frozen and
#    legitimately sit at zero.
[ -f "$STATS" ] || fail "no stats file at $STATS"
[ -f "$MOMENTS" ] || fail "no moments file at $MOMENTS"
echo "=== signal file is final ==="
/venv/main/bin/python /workspace/scratch/check_stats_final.py "$STATS" \
  /workspace/runs/s4/finetune.log || fail "the signal file is not the finished run's"

# 4. Every expert bank measured. This is the one the arms cannot detect for themselves.
echo "=== expert-bank coverage ==="
/venv/main/bin/python /workspace/scratch/check_banks.py "$MERGED" "$STATS" || fail "bank check errored"

# 4b. The merge merged. The probe answers this statistically and cannot: at 128 items the
#     binomial s.e. is 4.3 points, and an unfolded adapter lands the accuracy on the base
#     model's 57.75% -- the one value a threshold test has no power against. Compare the
#     weights instead.
echo "=== merged vs base ==="
/venv/main/bin/python /workspace/scratch/check_merged.py   /workspace/models/LFM2.5-8B-A1B "$MERGED" || fail "the merge is not a fine-tuned checkpoint"

# 5. The GPU is free. Seven arms at 8 B parameters do not share.
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

if [ "${1:-}" != "--go" ]; then
  echo
  echo "gate passed. re-run with --go <LIMIT> to launch the panel."
  exit 0
fi

# 6. The evaluation set is named, not defaulted. `--limit` unset means the whole scoreable
#    test split -- 16,143 items, not the 21,729 raw rows: the evaluator admits only sources
#    carrying INSERTs, golds leading with SELECT/WITH, and golds that return rows against
#    their own schema -- on each of seven arms. That is a wall-clock decision nobody made,
#    so the launcher refuses to make it silently. Run ./s4_probe.sh first.
LIMIT="${2:-}"
case "$LIMIT" in
  ''|*[!0-9]*) fail "--go needs an item count: ./s4_panel.sh --go 8000 [BATCH]. Unset means
      the whole 16,143-item scoreable split on all seven arms. Run ./s4_probe.sh first." ;;
esac

# 6b. And a batch size, because the probe measured one. Batch size is deliberately not a
#     pairing field -- left-padded decode perturbs the last bits of a logit, but the problems
#     are identical -- which makes raising it uniformly the only lever that buys wall clock
#     without costing statistical power. Left unset it falls to the task default, the one
#     value the probe did not time, and every hour-estimate would then describe a different
#     configuration from the one running. On this model bigger is slower: 0.709 s/item at 32
#     against 0.880 at 128, because a batch decodes until its longest member stops and
#     --max-new-tokens is 1024, so a wider batch waits on a longer straggler.
BATCH="${3:-32}"
case "$BATCH" in
  *[!0-9]*) fail "batch size must be numeric" ;;
esac

cd /workspace/dq-next || fail "no /workspace/dq-next"
git status --short | grep -q . && fail "dq-next is dirty; the panel must run committed code"

export HF_HOME=/workspace/.hf_home
export PYTHONPATH=/workspace/dq-next/packages/dynquant-core/src

echo "launching seven arms over $LIMIT items at batch $BATCH -> $OUT"
setsid nohup /workspace/venv-llmc/bin/python experiments/phase4/arms_lfm2.py run \
  --model "$MERGED" \
  --stats "$STATS" \
  --moments "$MOMENTS" \
  --out "$OUT" \
  --device cuda \
  --limit "$LIMIT" \
  --batch-size "$BATCH" \
  --resume \
  > "$LOG" 2>&1 < /dev/null &

sleep 10
pgrep -f "arms_lfm2.py run" | head -2
head -20 "$LOG"
