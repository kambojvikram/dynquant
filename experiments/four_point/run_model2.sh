#!/usr/bin/env bash
# The second model of the external comparison, from nothing to nine arms.
#
# Same task, same fine-tuning regime, same nine arms as the Mistral run -- the only
# thing that changes is the model. That is the point: a quantizer that beats GPTQ on
# one checkpoint has shown something about that checkpoint, and the second model is
# what turns it into a claim about the method.
#
# Qwen3.5-2B-Base is chosen for what it does *not* share with Mistral-7B-Instruct.
# Different family, different scale, hybrid linear/full attention rather than dense
# GQA, and -- the part that matters most to an allocator -- a **tied** embedding and LM
# head. On Mistral those are two 134 M tensors the allocator prices independently, and
# it spent 8 bits on the head. Here they are one tensor carrying ~27% of the model, so
# pricing the head up drags the embedding with it. If the comparison against GPTQ and
# AWQ survives that, it survives the structural case that is hardest for this method.
#
# LoRA r=32 at lr 1e-4 rather than the full fine-tune the earlier Qwen runs used: the
# fine-tuning regime is being held constant against the Mistral run, so that the
# quantization comparison is not also a comparison of two training setups.
#
# The task comes from the environment, defaulting to banking77
# -------------------------------------------------------------
# It was hardcoded, and hardcoding it was a mistake worth leaving a note about. The
# Mistral run is Banking77 because that is the task Banking77 headroom was measured on;
# the Qwen3.5-2B work in RESULTS.md is **CaseHOLD**, chosen for this model by screening
# base-model headroom, after GSM8K turned out to sit at the supervised ceiling and cost a
# full six-arm run to diagnose. Pairing each model with the task it was actually validated
# on is the design; running Qwen on Banking77 instead compares two models on one dataset,
# which is a different and weaker experiment.
#
# Both Qwen panels are kept rather than one deleted -- Banking77 base is 58.0% against
# 93.41% fine-tuned, so that run has real headroom and is valid work, just not the pairing
# the comparison is built on. ``$DQ_RUN_DIR`` defaults off the task so the two never share
# a directory, and every stage is guarded by an "is it already on disk" test, so a run that
# is interrupted resumes instead of repeating.
set -u

export HF_HOME=/workspace/.hf_home
export DQ_MODEL="${DQ_MODEL:-Qwen/Qwen3.5-2B-Base}"
export DQ_TASK="${DQ_TASK:-banking77}"
export DQ_RUN_DIR="${DQ_RUN_DIR:-/workspace/runs/qwen3_5_2b_base_$DQ_TASK}"
export PYTHONPATH=/workspace/dynquant-git/packages/dynquant-core/src

PY=/workspace/venv-cmp/bin/python
cd "$(dirname "$0")" || exit 1

step() { echo "########## $*"; }

# Headroom before the fine-tune, not after. A flat arm on this task once cost a full
# six-arm run to diagnose, and the cause was that the base model was already at
# ceiling -- which a 300-row screen would have shown in ten minutes. If this comes back
# near the ~93% supervised reference there is nothing for quantization damage to be
# read against and the model is the wrong choice, whatever the 7 B run showed.
if [ ! -f "$DQ_RUN_DIR/stage1_screen.json" ]; then
  step "screen: base accuracy on $DQ_TASK"
  $PY stage1_eval_base.py --limit 300 --name stage1_screen || exit 1
fi

if [ ! -d "$DQ_RUN_DIR/finetuned" ]; then
  step "fine-tune (LoRA r=32, lr 1e-4) with signal collection"
  $PY stage2_finetune.py --lora-rank 32 --lr 1e-4 || exit 1
fi

if [ ! -f "$DQ_RUN_DIR/stage4_bitmaps.json" ]; then
  step "allocate at 4.25 and 3.25 stored bits"
  $PY stage4_allocate.py --targets 4.25 3.25 || exit 1
fi

step "nine arms"
exec bash run_baselines.sh
