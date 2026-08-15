#!/bin/bash
set -u
cd /workspace/dq-next && git fetch -q origin && git reset -q --hard origin/main
echo "COMMIT $(git rev-parse --short HEAD)"
PY=/venv/omni/bin/python
export PYTHONPATH=/workspace/dq-next/packages/dynquant-core/src:/workspace/dq-next/tests
export PYTHONUNBUFFERED=1
OUT=/workspace/compile-final.jsonl
rm -f "$OUT"
run () {
  local name="$1" model="$2" armflags="$3" mode="$4" cmode="$5"
  echo "##### $name $armflags compile=$mode mode=$cmode $(date -u +%H:%M:%SZ) #####"
  $PY /workspace/dq-next/experiments/phase4/moe_end_to_end.py "$model" \
    $armflags --reps 3 --cache-impl static --compile "$mode" --compile-mode "$cmode" \
    --out "$OUT" 2>&1 \
    | tr '\r' '\n' \
    | grep -aE 'decode_tok_s|naive_tok_s|dynamo_unique|warmup_s|cache_impl|compile_mode|disable_compile|decoded_cache_len|resident_mib|peak_mib_total|accounted_bits|Error|Traceback|RuntimeError|OutOfMemory' \
    | tail -20
  echo "rc=${PIPESTATUS[0]}"
}
for spec in "lfm:/dev/shm/lfm" "olmoe:/dev/shm/olmoe"; do
  name="${spec%%:*}"; model="${spec##*:}"
  run "$name" "$model" "--arm bf16" off reduce-overhead
  run "$name" "$model" "--arm bf16" auto default
  run "$name" "$model" "--arm bf16" auto reduce-overhead
  run "$name" "$model" "--arm dynquant --bits 3" off reduce-overhead
  run "$name" "$model" "--arm dynquant --bits 3" auto default
  run "$name" "$model" "--arm dynquant --bits 3" auto reduce-overhead
  run "$name" "$model" "--arm dynquant --bits 3" manual reduce-overhead
done
echo "COMPILE FINAL DONE $(date -u +%H:%M:%SZ)"
