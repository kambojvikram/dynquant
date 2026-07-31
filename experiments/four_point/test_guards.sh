#!/usr/bin/env bash
# Validate the two stage5 provenance guards.
#
# Every case must exit BEFORE load_model, or the test costs a 7B quantize and an eval. The first
# attempt at this did exactly that: it used runs/.../stage4_bitmaps.ft1.json as the "stale map",
# but that file is a *copy* made on Jul 30 11:50, so its mtime is younger than the Jul 29 weights
# and the guard correctly passed -- then the run proceeded to quantize and evaluate for real.
# So: build the stale artifact explicitly with `touch -d` on a scratch copy, and give every
# passing case a bogus --target so it returns rc=2 immediately after the check.
set -uo pipefail
cd /workspace/dynquant-git/experiments/four_point || exit 1

PY=/workspace/venv-cmp/bin/python
export PYTHONPATH=/workspace/dynquant-git/packages/dynquant-core/src
export HF_HOME=/workspace/.hf_home
MIS=/workspace/runs/mistral_7b_instruct_v0_3_banking77
TMP=/workspace/guardtest_tmp; mkdir -p "$TMP"

# A real map, then a copy backdated to before the weights it claims to describe.
FRESH="$MIS/stage4_bitmaps.json"
STALE="$TMP/stale_bitmaps.json"
cp "$FRESH" "$STALE"
WT=$($PY -c "
import pathlib
print(max(p.stat().st_mtime for p in pathlib.Path('$MIS/finetuned').rglob('*') if p.is_file()))")
touch -d "@$(python3 -c "print(int($WT) - 3600)")" "$STALE"
echo "weights newest mtime = $WT ; stale map backdated 1h before that"
echo

pass=0; fail=0
check () { if [ "$2" = "$3" ]; then echo "  PASS $1 (rc=$3)"; pass=$((pass+1));
           else echo "  FAIL $1 (expected rc=$2, got rc=$3)"; fail=$((fail+1)); fi; }
run () { OUT=$("$@" 2>&1); RC=$?; echo "$OUT" | grep -E "provenance|PROVENANCE|aborting|no bit map" | sed 's/^/     /'; return $RC; }

echo "=== 1. NEGATIVE: map older than the weights  -> must abort rc=3, never load"
export DQ_MODEL=mistralai/Mistral-7B-Instruct-v0.3 DQ_TASK=banking77
run $PY stage5_quantize.py --model "$MIS/finetuned" --bitmaps "$STALE" --target 3.25 --name _gt
check "stale map aborts" 3 $?
echo "$OUT" | grep -qi "loaded /workspace" && { echo "  FAIL it loaded the model anyway"; fail=$((fail+1)); } \
  || { echo "  PASS aborted before the model load"; pass=$((pass+1)); }

echo "=== 2. NEGATIVE: DQ_MODEL unset -> RUN_DIR resolves to the Qwen dir -> must abort rc=3"
unset DQ_MODEL; export DQ_TASK=banking77
run $PY stage5_quantize.py --model "$MIS/finetuned" --bitmaps "$FRESH" --target 3.25 --name _gt
check "wrong RUN_DIR aborts" 3 $?

echo "=== 3. POSITIVE: correct map + correct DQ_MODEL, bogus target stops after the check"
export DQ_MODEL=mistralai/Mistral-7B-Instruct-v0.3 DQ_TASK=banking77
run $PY stage5_quantize.py --model "$MIS/finetuned" --bitmaps "$FRESH" --target 99.9 --name _gt
check "clean pairing passes" 2 $?
echo "$OUT" | grep -q "provenance: ok" && { echo "  PASS reported ok"; pass=$((pass+1)); } \
  || { echo "  FAIL did not report ok"; fail=$((fail+1)); }

echo "=== 4. OVERRIDE: --allow-stale downgrades the stale map to a warning"
run $PY stage5_quantize.py --model "$MIS/finetuned" --bitmaps "$STALE" --target 99.9 --name _gt --allow-stale
check "--allow-stale proceeds" 2 $?

rm -rf "$TMP"
echo
echo "GUARDS: $pass passed, $fail failed"
ls /workspace/runs/*/*_gt* 2>/dev/null && echo "!!! a test wrote a record" || echo "no records written"
exit $fail
