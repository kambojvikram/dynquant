# Box-side orchestration for the phase-4 LFM2.5-8B-A1B campaign

These scripts run **on the GPU instance**, not here. Their paths are absolute into
`/workspace`, they call interpreters that exist only there (`/venv/main/bin/python` for
safetensors and transformers 5.14.1, `/workspace/venv-llmc/bin/python` for llmcompressor),
and none of them will do anything useful on a developer machine.

They are checked in anyway, for one reason: **`/workspace` on that box is not a volume.**
`workspace_is_volume = False`, so a recycle or a destroy takes the filesystem with it. Every
gate that decides whether seven GPU-hours are worth spending lived only there. Losing them
means the next campaign either re-derives the checks from the report prose or, more likely,
runs without them.

| file | what it decides |
| --- | --- |
| `s4_panel.sh` | the seven-arm launch gate: trainer stopped, merge and tokenizer present, signal file final, all expert banks measured, merge actually moved, GPU free, panel repo clean; then launches detached with an explicit `--limit` and `--batch-size` |
| `s4_probe.sh` | prices an arm before the panel commits to one: 128 items at batch 32/64/128 under the panel's own flags |
| `s4_after.sh` | post-run collection |
| `check_stats_final.py` | is this signal file the *finished* run's, by `grad_norm_count` rather than by mtime |
| `check_banks.py` | is every batched expert bank present in the stats file |
| `check_merged.py` | did the adapter actually fold into the merge, by weight displacement against the base |

Two of them exist because of specific failures, and the failures are the point:

- `check_stats_final.py` replaced an mtime ordering (`stats -nt merge`) that was **backwards by
  construction** -- the callback flushes inside `trainer.train()`, `save_outputs` writes the
  merge after it returns, so on a correct run the signal file is always older and the warning
  fired every time. Its own first draft then passed vacuously, because the last `N/M [` progress
  bar in the training log belongs to the shard writer (`1/1`), making the test `1560 < 1 - 60`.
  It takes `max()` over a tail window now, and it is proven in both directions: exit 0 on the
  real artifacts, exit 1 on the same file with the counts doctored down.
- `s4_probe.sh` removes each `probe.b$B.json` before the batch runs, because the summary reads
  them back through a glob -- so an out-of-memory batch would have written nothing and the
  summary would have priced the panel off a previous probe's record.

Keep them in sync by hand. There is no deploy step; `scp` to `/workspace` and
`/workspace/scratch` respectively.
