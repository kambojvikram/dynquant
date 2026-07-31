#!/usr/bin/env bash
# DynQuant at the baselines' *measured* widths, so accuracy is compared at equal bytes.
#
# The main table compares DynQuant 3.25 against GPTQ "3-bit", and those are not the same
# budget. GPTQ's 3-bit g128 checkpoint keeps its embedding and LM head in fp16 and pays
# an fp16 scale plus a zero point per group of 128, which measures 3.6249 bits on
# Mistral-7B against DynQuant's 3.25. So the -0.81 point gap in that row is a method gap
# *plus* a 0.37-bit budget gap, and the row cannot say which. This script closes it from
# the other side: give DynQuant the same bytes the baseline actually spends and re-run.
#
# The targets are read out of the baseline records rather than written here, because the
# measured width depends on the model -- vocabulary share, tied embeddings, how much of
# the checkpoint the fp16 exclusions cover. On a model with a tied embedding carrying 27%
# of the parameters the baselines' fp16 share is far larger than on Mistral, so a
# hardcoded 3.6249 would silently be the wrong budget for the second model.
#
# One iso arm covers GPTQ, AWQ and RTN at a given width: all three use the same
# w{BITS}a16 g128 scheme and therefore the same accounted width. bnb-NF4 is close but
# not equal (4.5671 vs 4.5953 on Mistral) because its metadata is a per-64 absmax that is
# itself quantized, so it is compared against the 4-bit iso arm with that caveat noted.
set -u

export HF_HOME=/workspace/.hf_home
export DQ_MODEL="${DQ_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
export DQ_TASK="${DQ_TASK:-banking77}"
export PYTHONPATH=/workspace/dynquant-git/packages/dynquant-core/src

PY=/workspace/venv-cmp/bin/python
RUN_DIR="${DQ_RUN_DIR:-/workspace/runs/mistral_7b_instruct_v0_3_banking77}"
export DQ_RUN_DIR="$RUN_DIR"
FT="$RUN_DIR/finetuned"
MAPS="$RUN_DIR/stage4_bitmaps.json"

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

# The widths to match, taken from the arms that were actually measured.
read -r ISO4 ISO3 < <("$PY" - "$RUN_DIR" <<'EOF'
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
widths = []
for bits in (4, 3):
    # gptq/awq/rtn share a scheme and a width; whichever ran is the same number, so the
    # first one present is used and disagreement between them is a bug worth failing on.
    found = {}
    for method in ("gptq", "awq", "rtn"):
        path = run / f"stage8_{method}_{bits}b.json"
        if path.exists():
            found[method] = json.loads(path.read_text(encoding="utf-8"))["accounted_bits"]
    if not found:
        sys.exit(f"no {bits}-bit baseline record in {run}; nothing to match")
    if len(set(round(v, 4) for v in found.values())) != 1:
        sys.exit(f"{bits}-bit baselines disagree on accounted width: {found}")
    widths.append(f"{next(iter(found.values())):.4f}")
print(" ".join(widths))
EOF
) || exit 1

echo "matching baseline widths: 4-bit family = $ISO4 bits, 3-bit family = $ISO3 bits"

# The allocator formats its map keys to two decimals, so that is the key to ask stage 5
# for -- not the four-decimal target.
KEY4=$(printf '%.2f' "$ISO4")
KEY3=$(printf '%.2f' "$ISO3")

# stage4_allocate overwrites stage4_bitmaps.json, and the two already-measured arms are
# keyed inside it. So all four targets are emitted in one pass and the two existing maps
# are diffed against a backup. If the allocator is deterministic the measured arms keep
# their provenance; if it is not, the existing table is not reproducible and that is a
# finding, so the run stops rather than quietly replacing the maps under it.
if ! grep -q "\"$KEY4\"" "$MAPS" || ! grep -q "\"$KEY3\"" "$MAPS"; then
  cp "$MAPS" "$MAPS.bak" || exit 1
  echo "########## allocate at $ISO4 4.25 $ISO3 3.25"
  $PY stage4_allocate.py --targets "$ISO4" 4.25 "$ISO3" 3.25 || exit 1

  $PY - "$MAPS" <<'EOF' || exit 1
import json, sys
from pathlib import Path

new = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["maps"]
old = json.loads(Path(sys.argv[1] + ".bak").read_text(encoding="utf-8"))["maps"]
for key in sorted(old):
    if key not in new:
        sys.exit(f"regenerated maps lost target {key}")
    if new[key]["bits"] != old[key]["bits"]:
        differing = [n for n, b in old[key]["bits"].items() if new[key]["bits"].get(n) != b]
        sys.exit(
            f"allocator is not deterministic: target {key} moved {len(differing)} modules "
            f"({', '.join(differing[:3])}); the measured arms no longer match their map"
        )
print(f"allocator reproduced targets {', '.join(sorted(old))} bit-identically")
EOF
else
  echo "########## bit maps already carry $KEY4 and $KEY3 -- skipping allocation"
fi

arm "stage8_dq_iso${KEY4/./p}" \
  $PY stage5_quantize.py --model "$FT" --target "$KEY4" \
      --name "stage8_dq_iso${KEY4/./p}" --label "DynQuant ${KEY4}b iso"

arm "stage8_dq_iso${KEY3/./p}" \
  $PY stage5_quantize.py --model "$FT" --target "$KEY3" \
      --name "stage8_dq_iso${KEY3/./p}" --label "DynQuant ${KEY3}b iso"

echo "########## iso-size arms attempted"
ls -1 "$RUN_DIR"/stage8_dq_iso*.json 2>/dev/null
