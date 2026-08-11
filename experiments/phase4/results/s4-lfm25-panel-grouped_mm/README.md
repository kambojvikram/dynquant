# The three records the panel superseded

`../s4-lfm25-panel/` is the panel of record and every arm in it indexes one expert at a time.
Three of its seven arms were scored twice to get there, and these are the first scoring.

The four baselines are quantized by `llm-compressor`, which linearises the batched expert banks
into per-expert `Linear` modules before it touches them -- 22 banks to 0, recorded in
`../s4-lfm25-panel/linearize_mapping.json`. The three arms that are not quantized by
`llm-compressor` -- `bf16`, `dq_4b`, `dq_3b` -- kept the checkpoint's own batched banks and ran
`grouped_mm`. So the first panel compared DynQuant against GPTQ and AWQ across a difference in
expert arithmetic, and a margin that size at 4 bits is inside what an arithmetic difference could
plausibly produce. Section 8 of the report measured dispatch-only disagreement at 1.24% on this
model, which is 0.29x the 4-bit fidelity gap -- large enough to matter, not large enough to
explain it, and an estimate either way.

Re-scoring the three arms under `--experts-impl eager` replaces the estimate with the thing
itself. It is not a relabelling: the records in `../s4-lfm25-panel/` carry
`"experts": {"found": "grouped_mm", "ran": "eager"}`, so the checkpoints are the same objects
these ones were and the only difference is how the experts were multiplied. The allocation did not
move -- `../s4-lfm25-panel/maps/` is unchanged and the driver reported "allocation unchanged" for
both DynQuant arms.

| arm | grouped_mm | eager | delta | items |
|---|---:|---:|---:|---:|
| `bf16` | 84.258% | 84.292% | +0.033 | +4 |
| `dq_4b` | 82.708% | 82.842% | +0.133 | +16 |
| `dq_3b` | 79.850% | 79.892% | +0.042 | +5 |

Dispatch was worth a tenth of a point at 4 bits, and it was worth it in the direction that makes
the confound harmless: DynQuant's margin over GPTQ went **up**, +0.64 to +0.78, when it stopped
having the arithmetic the confound would have credited it for. Over AWQ, +0.94 to +1.08. At 3
bits, +19.09 to +19.13. Nothing in the report's conclusions turns on the difference, which is the
useful outcome -- but it is now measured on the arms the claims are about rather than inferred
from a probe, and these three files are what it was measured against.

The other four records are not copied here. They never re-ran, so `../s4-lfm25-panel/` holds the
only copy and `sha256sum` says it is the same file. Duplicating them would suggest a second
measurement that does not exist.
