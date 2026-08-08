#!/bin/bash
# Wait for the fine-tune to finish, then gate and price -- and stop there.
#
# The launch decision stays a decision. This only removes the idle hour between the last
# training step and the first measurement: it runs the gate (including the expert-bank
# coverage check against the *final* stats file) and the 128-item timed probe, then exits
# with everything needed to choose `--limit`. It never starts the panel.
set -u

LOG=/workspace/runs/s4/after.log
exec > "$LOG" 2>&1

echo "watching for the fine-tune to exit"
while pgrep -f "run_s2_finetune.py --model lfm" > /dev/null; do
  sleep 60
done
echo "trainer process gone at $(date -u +%FT%TZ)"

# The trainer saves the merge before it exits, but the filesystem and the GPU both need a
# moment: an evaluation started while the last shard is still being flushed reads a
# truncated safetensors, and one started while the trainer's allocator is still releasing
# reads a GPU that is not free yet.
sleep 90

echo
echo "############ gate ############"
/workspace/s4_panel.sh || { echo "GATE REFUSED -- not probing"; exit 1; }

echo
echo "############ probe ############"
/workspace/s4_probe.sh
echo
echo "PROBE DONE at $(date -u +%FT%TZ)"
