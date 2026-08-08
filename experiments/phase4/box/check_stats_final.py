"""Is this the signal file the finished fine-tune wrote, or one from the middle of it?

The gate previously answered this with an mtime ordering: the stats file must postdate the
merge. That test is backwards by construction. DynQuantCallback flushes inside
trainer.train() and save_outputs writes the merge after it returns, so on a perfectly
correct run the signal file is always the older of the two and the warning always fires. A
warning that always fires is one nobody reads, and this one sat directly above the check
that guards seven GPU-hours.

The stats file records the answer directly. Every layer carries grad_norm_count, the number
of optimizer steps its Welford accumulator saw, so the file says for itself which step it
was written at. Take the maximum across layers, not the minimum: the embedding and the
eighteen depthwise convs are frozen and legitimately sit at zero, so a floor test would trip
on them on every correct run. The batched expert banks are *not* among them -- all 44 carry
the full count, measured through requires_grad_ on the raw parameters.

argv: <stats.json> <finetune.log> -- the log supplies the denominator, so nothing here is
hardcoded to one particular run length.
"""

import json
import re
import sys
from pathlib import Path

stats_path, log_path = Path(sys.argv[1]), Path(sys.argv[2])

layers = json.loads(stats_path.read_text(encoding="utf-8"))["layers"]
counts = [int(v.get("grad_norm_count", 0)) for v in layers.values()]
seen = max(counts) if counts else 0
zero = sum(1 for c in counts if c == 0)

# The *last* progress bar in the log is the shard writer -- "1/1 [00:11<00:00]" -- not the
# trainer. Taking found[-1] made the denominator 1, so the comparison below read
# "1560 < 1 - 60": false for every possible input, a guard that passes unconditionally. It
# was caught by reading its printed numbers, not its exit code. Take the largest denominator
# in the tail instead; the training bar owns it and the shard writer cannot.
with log_path.open("rb") as handle:
    handle.seek(0, 2)
    handle.seek(max(0, handle.tell() - 200000))
    tail = handle.read().decode("utf-8", "replace")
found = re.findall(r"(\d+)/(\d+) \[", tail)
total = max((int(m[1]) for m in found), default=0)

frozen = f"{zero} of {len(counts)} layers never had a gradient"
print(f"  stats written at step {seen} of {total} ({frozen})")

if not total:
    print("  could not read a step total from the log; not asserting")
    raise SystemExit(0)

SLACK = 60
if seen < total - SLACK:
    print(f"  REFUSED: this signal file is {total - seen} steps short of the end. It is a")
    print("  mid-training flush, and quantizing the final merge through it is the")
    print("  stale-artifact failure in its original shape.")
    raise SystemExit(1)
print("  final: the signal file and the merge come from the same finished run")
