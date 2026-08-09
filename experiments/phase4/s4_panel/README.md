# Phase 4 panel records, pulled off the box as they land

`/workspace` on the vast.ai instance is **not a volume** — `workspace_is_volume = False`, so a
recycle or a destroy takes everything with it. Every artifact here was copied down while the panel
was still running, not collected at the end, and that is the rule rather than an accident of
timing: an arm costs about six hours of a rented card, and a record that only exists on the box is
a record one `vastai destroy` away from being re-earned.

What is here:

| file | what it is |
| --- | --- |
| `arms.json` | the panel's own plan: arm order, byte anchors, tolerance, launch stamp |
| `<arm>.json` | one eval record — accuracy, per-item `hits`, per-split breakdown, timings |
| `<arm>.quant.json` | the quantization manifest for that arm: measured bytes, accounted bits, per-role width histogram |
| `probe.b{32,64,128}.json` | the decode-cost probe that chose `--limit 12000` and `--batch-size 32` |
| `leak*.json` | train/test overlap scans for the text-to-SQL mixture |

The per-item `hits` vectors are the reason the records are worth their size. Every A/B in
[`panel_table.py`](../panel_table.py) is a McNemar test on stored hits rather than a comparison of
two accuracy scalars, which roughly halves the standard error of a difference — and it can only be
done after the fact if the hits were written down at scoring time.

The fine-tune's own outputs live beside this in
[`../s4_runs/lfm25-8b-a1b.text2sql/`](../s4_runs/lfm25-8b-a1b.text2sql), following the phase-3
layout. The LoRA adapter is **not** in the repository: it is 45 MB and regenerates the 16 GB merge,
so it is backed up outside the tree rather than committed.
