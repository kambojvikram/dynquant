#!/usr/bin/env bash
# The baselines again, with the tied embedding/LM-head tensor quantized.
#
# Only meaningful on a model that ties the two, and it refuses to run on one that does
# not. On an untied model -- Mistral-7B -- the LM head is 5.5% of the checkpoint and
# leaving it in fp16 costs the baselines 0.35 bits, which is a footnote. On Qwen3.5-2B
# ``embed_tokens`` and ``lm_head`` are one 508.6M tensor, 27% of the model, so the
# standard ``ignore=["lm_head"]`` recipe pins a quarter of the weights at fp16: "4-bit
# g128" accounts to 7.3605 bits and "3-bit" to 6.6253, against DynQuant's 4.2486 and
# 3.2494. The default panel therefore compares a model that kept 27% of itself in full
# precision against one that did not, and the accuracy column in it cannot be read as a
# statement about either method.
#
# With ``--include-head`` the same recipes land at 4.1597 and 3.1522 bits -- 2.1% and
# 3.0% *below* DynQuant's measured width. That is the panel that isolates the allocator,
# and the residual budget difference runs against DynQuant, so it cannot manufacture a
# win. Both panels are kept: the default one is a real result about the convention, and
# deleting it in favour of the flattering-to-nobody variant would hide the finding that
# a tied embedding is where the published recipes quietly stop compressing.
#
# bnb-NF4 is deliberately absent. Its exclusion list is chosen inside bitsandbytes'
# ``replace_with_bnb_linear`` rather than by our recipe, so an equivalent arm would mean
# reaching into that call; the 4-bit iso arm from run_isosize.sh already covers the
# equal-byte question for it.
set -u

export HF_HOME=/workspace/.hf_home
# CaseHOLD, not Banking77: Qwen3.5-2B's pairing in this comparison is the task its own
# headroom screen chose for it, and each model is paired with the task it was validated on
# rather than both models sharing one dataset. A Qwen/Banking77 run also exists and is real
# work, but it is a secondary panel -- so it has to be asked for by name here, not inherited
# from a default.
export DQ_MODEL="${DQ_MODEL:-Qwen/Qwen3.5-2B-Base}"
export DQ_TASK="${DQ_TASK:-casehold}"
export PYTHONPATH=/workspace/dynquant-git/packages/dynquant-core/src

PY=/workspace/venv-cmp/bin/python
RUN_DIR="${DQ_RUN_DIR:-/workspace/runs/qwen3_5_2b_base_$DQ_TASK}"
export DQ_RUN_DIR="$RUN_DIR"
FT="$RUN_DIR/finetuned"

cd "$(dirname "$0")" || exit 1

arm() {  # arm <name> <command...>
  local name="$1"; shift
  if [ -f "$RUN_DIR/$name.json" ]; then
    echo "########## $name -- already on disk, skipping"
    return
  fi
  echo "########## $name"
  if "$@"; then echo "ARM $name OK"; else echo "ARM $name FAILED"; fi
}

# Checked by object identity on a meta-device copy, not by reading
# ``config.tie_word_embeddings``. The flag says what the architecture intends; identity
# says what the loaded module tree actually shares, which is what the accounting and the
# recipe both see. A model whose flag is set but whose weights are separate would make
# this panel a duplicate of the default one at a different name.
#
# The test is ``head.weight is embed.weight``. It was ``id(head.weight) is id(embed.weight)``,
# which is a different thing and always False: ``id()`` returns an int, and ``is`` on two ints
# outside CPython's small-value cache compares the boxes, not the values. Measured on this
# checkpoint the broken form returned False while the tie was plainly there -- 508,559,360 of
# 1,881,825,088 parameters shared -- so the gate below fired its "nothing to do" branch and
# exited 0. Six arms skipped, exit status clean, log reading as a deliberate no-op. Compare
# tensors with ``is``, or ints with ``==``, and never the two crossed.
TIED=$(CUDA_VISIBLE_DEVICES= $PY - "$FT" <<'EOF'
import sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM

with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(sys.argv[1]))
embed = model.get_input_embeddings()
head = model.get_output_embeddings()
tied = head is not None and embed is not None and head.weight is embed.weight
share = embed.weight.numel() / sum({id(p): p.numel() for p in model.parameters()}.values())
print("yes" if tied else "no", f"{share:.4f}")
EOF
) || exit 1

echo "tied embedding check: $TIED (tied? share-of-params)"
case "$TIED" in
  yes*) ;;
  no*) echo "not a tied-embedding model -- the default panel is already comparable, nothing to do"
     exit 0 ;;
  # Anything that is neither is a broken probe, not an untied model, and the two must not
  # share an exit path: a silent skip is indistinguishable from a successful no-op.
  *) echo "!!! tie probe produced no verdict: '$TIED'" >&2
     exit 1 ;;
esac

for BITS in 4 3; do
  for M in rtn gptq awq; do
    arm "stage8_${M}_${BITS}b_head" \
      $PY stage8_baselines.py run --method "$M" --bits "$BITS" --model "$FT" \
          --include-head \
          --name "stage8_${M}_${BITS}b_head" --label "$M ${BITS}-bit g128 +head"
  done
done

echo "########## tied-head arms attempted"
ls -1 "$RUN_DIR"/stage8_*_head.json 2>/dev/null
