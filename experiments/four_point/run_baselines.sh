#!/usr/bin/env bash
# Every arm of the external comparison, in one environment, in one pass.
#
# The DynQuant arms are re-scored here rather than read out of the stage 5 records,
# and that is the whole reason this file exists instead of a hand-typed command list.
# The stage 5 numbers were measured under torch 2.13 / transformers 5.14; llm-compressor
# pins transformers back to 5.10, so a baseline measured in this venv and a DynQuant
# number read from disk would differ by one uncontrolled variable -- the decoder --
# in a table whose entire claim is that only the weights differ. Re-running the
# reference arms costs ~15 minutes each and removes the objection completely.
#
# Order is deliberate: cheapest arm of each family first, so a configuration error
# surfaces in minutes rather than after a 40-minute GPTQ pass.
set -u

export HF_HOME=/workspace/.hf_home
export DQ_MODEL="${DQ_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
export DQ_TASK="${DQ_TASK:-banking77}"
export PYTHONPATH=/workspace/dynquant-git/packages/dynquant-core/src

PY=/workspace/venv-cmp/bin/python
FP="$(dirname "$0")"
RUN_DIR="${DQ_RUN_DIR:-/workspace/runs/mistral_7b_instruct_v0_3_banking77}"
export DQ_RUN_DIR="$RUN_DIR"
FT="$RUN_DIR/finetuned"

cd "$FP" || exit 1

arm() {  # arm <name> <command...>
  local name="$1"; shift
  if [ -f "$RUN_DIR/$name.json" ]; then
    echo "########## $name -- already on disk, skipping"
    return
  fi
  echo "########## $name"
  if "$@"; then echo "ARM $name OK"; else echo "ARM $name FAILED"; fi
}

# The ceiling every quantized arm is measured against, and the two DynQuant points.
arm stage8_fp16 \
  $PY stage8_baselines.py eval --model "$FT" --name stage8_fp16 --label "fine-tuned bf16"

for T in 4.25 3.25; do
  arm "stage8_dq_${T/./p}" \
    $PY stage5_quantize.py --model "$FT" --target "$T" --name "stage8_dq_${T/./p}" \
        --label "DynQuant ${T}b"
done

# RTN first at each width: no calibration, so if it fails the problem is the recipe or
# the checkpoint, not the calibration set or the Hessian.
for BITS in 4 3; do
  for M in rtn gptq awq; do
    arm "stage8_${M}_${BITS}b" \
      $PY stage8_baselines.py run --method "$M" --bits "$BITS" --model "$FT" \
          --name "stage8_${M}_${BITS}b" --label "$M ${BITS}-bit g128"
  done
done

# Fourth family: a fixed non-uniform data type, quantized on load with no calibration.
# Last because it is the cheapest -- no calibration pass at all -- so it is the arm that
# can be appended to a finished run without re-running anything.
arm stage8_nf4_4b \
  $PY stage8_bnb.py --model "$FT" --name stage8_nf4_4b --label "bnb NF4 4-bit"

echo "########## all arms attempted"
ls -1 "$RUN_DIR"/stage8_*.json 2>/dev/null
