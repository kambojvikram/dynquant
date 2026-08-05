#!/usr/bin/env bash
# S1 headroom screen: base-model accuracy before any fine-tune.
#
# One (model, task) pair per `dynquant eval` invocation, each writing its own
# per-problem correctness vector so every later arm can be compared against it
# with a paired test rather than a difference of two summary numbers.
#
# The screen answers one question per pair: is there room for a quantization
# regression to show?  A pair above ~90% has none and leaves the headline.
set -u

BACKEND="${BACKEND:-vllm}"
OUT="${OUT:-/workspace/reports/s1}"
LOGS="${LOGS:-/workspace/logs/s1}"
PY=/venv/main/bin/python

# Open-weight models only.  meta-llama/Llama-3.1-8B-Instruct and
# google/gemma-3-4b-it are gated=manual: licence acceptance is per HF account
# and cannot be done from here, so they join the screen when a token arrives.
MODELS="${MODELS:-microsoft/Phi-4-mini-instruct mistralai/Ministral-8B-Instruct-2410}"
TASKS="${TASKS:-ifeval gsm8k humaneval mbpp}"

mkdir -p "$OUT" "$LOGS"
export HF_HOME=/workspace/.hf_home HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120

# Weights and datasets come from the cache, and a miss is an error rather than a
# download.  Two reasons, and the first one already cost four arms.
#
# vLLM's loader fetches `*.safetensors` from the Hub and only *then* filters against
# `model.safetensors.index.json`.  Ministral's repo carries a 16 GB
# `consolidated.safetensors` next to the four sharded files, so every arm re-pulled
# 16 GB it was about to discard -- and this link drops large transfers at around a
# gigabyte, so all four Ministral arms died in engine init with a RemoteProtocolError
# that named httpx and looked nothing like a loader bug.  Offline, `snapshot_download`
# returns the cached snapshot directly and vLLM sees only the four shards it loads.
#
# The second reason is that a scored run should not be doing network I/O at all: a
# stalled download mid-evaluation is a timing artefact inside a measurement, and a
# model nobody prefetched should fail in seconds rather than thirty minutes in.
# Prefetch is how models get here; this makes that a rule instead of a habit.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

cd /workspace/dynquant || exit 1

for model in $MODELS; do
  slug=$(echo "$model" | tr '/' '-' | tr '[:upper:]' '[:lower:]')
  for task in $TASKS; do
    dest="$OUT/$slug.$task.json"
    log="$LOGS/$slug.$task.log"
    if [ -s "$dest" ]; then
      echo "== skip $slug $task (have $dest)"
      continue
    fi
    # HumanEval and MBPP score by running the generated program.
    extra=""
    case "$task" in humaneval|mbpp) extra="--allow-execution" ;; esac

    echo "== $slug $task -> $dest"
    start=$SECONDS
    # shellcheck disable=SC2086
    "$PY" -u -m dynquant.cli eval "$model" \
      --task "$task" --backend "$BACKEND" \
      --label "s1:$slug:$task" --out "$dest" $extra \
      > "$log" 2>&1
    rc=$?
    echo "   rc=$rc  $((SECONDS - start))s"
    tail -3 "$log" | sed 's/^/   /'
  done
done

echo "== S1 screen done"
