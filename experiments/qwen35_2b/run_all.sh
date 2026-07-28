#!/bin/bash
# The whole four-point experiment, all six measurement arms, one task.
#
# This lives in the repository rather than on the machine that ran it. The GSM8K pass
# was driven by a script that existed only on a rented box, which meant the exact
# sequence that produced the published table was the one artifact not kept -- and a
# results table whose recipe cannot be re-read is a table nobody can check.
#
# Resumable, because six hours on a spot instance is long enough to lose. Each stage is
# skipped when its output already exists, and every skip is printed: a run that quietly
# reused a stale checkpoint would otherwise look exactly like a run that recomputed it.
# The stage-2 check tests for the model directory, not for the record -- a smoke run
# with --max-steps 5 writes a perfectly well-formed record, so trusting the record here
# would let a five-step model be evaluated as the fine-tuned arm.
#
# Usage::
#
#     DQ_TASK=casehold ./run_all.sh
#
#     # Every stage end to end in ~15 min, into a scratch run directory. Worth doing
#     # before the real thing: stages 4 and 5 consume the stats file stage 2 writes, so
#     # a schema problem between them is otherwise discovered three hours in. MAX_STEPS
#     # must exceed the callback's log_every or no stats are flushed and stage 4 has
#     # nothing to score.
#     DQ_TASK=casehold DQ_RUN_DIR=/workspace/runs/smoke_casehold \
#         LIMIT=32 MAX_STEPS=60 ./run_all.sh
set -u

TASK="${DQ_TASK:-casehold}"
export DQ_TASK="$TASK"
export PYTHONPATH=/workspace/dynquant-core/src
# The batch is right-padded to its longest row, and CaseHOLD rows run 135-640 tokens,
# so successive steps ask the allocator for quite different block sizes. Without this
# the worst-case batch fragments the cache and the allocator recovers by flushing it --
# which it survives, but pays for, once per unlucky batch over thousands of steps.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/venv/main/bin/python
# Exported, not just assigned: common.py resolves the same variable, and if the two
# disagreed the stages would write records into one directory and read them from another.
export DQ_RUN_DIR="${DQ_RUN_DIR:-/workspace/runs/qwen35_2b_${TASK}}"
RUNS="$DQ_RUN_DIR"
LIMIT="${LIMIT:-}"
LIM_ARG=""
[ -n "$LIMIT" ] && LIM_ARG="--limit $LIMIT"
STEP_ARG=""
[ -n "${MAX_STEPS:-}" ] && STEP_ARG="--max-steps $MAX_STEPS"

cd "$(dirname "$0")" || exit 1
mkdir -p "$RUNS"

run_stage() {
    # run_stage <marker path> <human label> <command...>
    local marker="$1" label="$2"
    shift 2
    echo ""
    echo "################  $label  ################"
    date -Is
    if [ -e "$marker" ]; then
        echo "SKIP: $marker already exists"
        return 0
    fi
    "$@" || {
        echo "ABORTING: $label failed" >&2
        exit 1
    }
}

echo "task=$TASK  runs=$RUNS  limit=${LIMIT:-none}"

run_stage "$RUNS/stage1_base.json" "1/6  base fp16" \
    $PY stage1_eval_base.py $LIM_ARG

# Identical recipe to the GSM8K arm -- 2 epochs, lr 1e-5 cosine, effective batch 32 --
# deliberately. The comparison this experiment rests on is "same recipe, different task
# headroom", so a step count tuned per task would introduce a second difference and
# make the two tables answer different questions.
# shellcheck disable=SC2086
run_stage "$RUNS/finetuned/config.json" "2/6  fine-tune (collects the signal map)" \
    $PY stage2_finetune.py $STEP_ARG

run_stage "$RUNS/stage3_finetuned.json" "3/6  fine-tuned fp16" \
    $PY evaluate.py --model "$RUNS/finetuned" --label "fine-tuned" \
    --name stage3_finetuned $LIM_ARG

run_stage "$RUNS/stage4_bitmaps.json" "4/6  score + allocate" \
    $PY stage4_allocate.py --targets 4.25 3.25

for spec in "4.25:stage5_4p25:" "4.25:stage5_uniform4b:--uniform 4" \
    "3.25:stage5_3p25:" "3.25:stage5_uniform3b:--uniform 3"; do
    target="${spec%%:*}"
    rest="${spec#*:}"
    name="${rest%%:*}"
    extra="${rest#*:}"
    # shellcheck disable=SC2086
    run_stage "$RUNS/$name.json" "5/6  quantize $target ${extra:-allocated}" \
        $PY stage5_quantize.py --target "$target" $extra $LIM_ARG
done

echo ""
echo "################  6/6  results table  ################"
date -Is
$PY results_table.py
