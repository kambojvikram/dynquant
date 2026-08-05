# S3 preparation: what fusion costs Phi-4-mini, and how far below its floors S3 asks it to go

**Measured 2026-08-05**, before S3's arms were built, on the two models phase 3 fine-tunes.
Script: [`experiments/phase3/s3_allocation/floor_headroom.py`](../../experiments/phase3/s3_allocation/floor_headroom.py).
Record: `floor_headroom.json` beside it. Both models are constructed on the meta device at
their checkpoints' real geometry, so only shapes and parameter counts are read and the whole
measurement runs in seconds.

## Why this was run before S3 and not after

Phase 2 ran on Qwen3.5-2B and Mistral-7B. Both spell every projection separately. Phi-4-mini
does not — `qkv_proj` and `gate_up_proj` are **55.1% of its parameters** — and fused
projections were bug 5 in the original supplement, where they fell through to a 3-bit MLP
catch-all. The package has a `phi3` plugin that partitions both tensors, but nothing in the
campaign had exercised it, and 18 GPU-hours of fine-tuning were already running against a
checkpoint whose quantization path had never been checked.

## What holds

**The partitions are right at the real geometry, not just the fixture's.** The arch matrix
runs at hidden 64 with a 4:2 head ratio; Phi-4-mini is hidden 3072 with 24:8, and it leaves
`head_dim` unset so the plugin's `hidden // heads` fallback is the path that fires. Both
boundaries land where `Phi3Attention` and `Phi3MLP` put them:

| tensor | out_features | partition |
|---|---|---|
| `qkv_proj` | 5120 = 24×128 + 2×(8×128) | Q `[0, 3072)`, K `[3072, 4096)`, V `[4096, 5120)` |
| `gate_up_proj` | 16384 = 2 × 8192 | gate `[0, 8192)`, up `[8192, 16384)` |

Pinned by `test_phi_partitions_hold_at_phi_4_mini_s_real_geometry`.

**The allocator descends from floors it cannot afford, on a fused model.** That was bug 4,
and every test guarding it runs on an unfused synthetic model. At a 3.25-bit target Phi lands
at 3.2492 with 90 floors breached and named; at 4.0 it lands at 3.9991 with 9. Scores drive
the result: against a shuffled control — the same score *distribution*, no correspondence to
modules — 69 of 129 modules take a different width at 3.25b. Pinned by
`test_soft_floors_reach_phi_s_fused_tensors`, which has to run at real geometry: at hidden 64
a group of 128 pads every row until the scale/offset metadata outweighs the payload, and the
allocator correctly refuses a 3.25-bit budget before it can descend at all.

## What does not hold, and changes how S3's table reads

### Phi and Ministral are not two samples of one condition

| | Phi-4-mini | Ministral-8B |
|---|---|---|
| params | 3.836 B | 8.020 B |
| fused | **55.1%** | 0% |
| floors cost | **4.4310 b** | 3.8159 b |
| headroom @ 3.25 b | **−1.1810 b** | −0.5659 b |
| headroom @ 4.00 b | **−0.4310 b** | +0.1841 b |
| modules moved vs shuffled @ 3.25 b | 69 / 129 | 137 / 254 |

At the same nominal target Phi is asked to go twice as far below its floors. Phase 2 found
that the signal only earns its keep once the role floors stop being affordable, so this is
the axis along which the two models differ most — and it predicts Phi should show DynQuant
*more* favourably than Ministral, for a reason that has nothing to do with the models being
better or worse suited to the method. At 4.0 bits the two are in different regimes outright:
Ministral has 0.18 bits of slack and only 15 breached floors, Phi is still 0.43 bits short.

### 0.21 of Phi's 4.43 bits is the allocator, not the checkpoint

`floor_bits` on a fused tensor is the **strictest** of its partitions' floors. That is the
right rule — the tensor gets one width, and it cannot go below what its most sensitive rows
need. But the role table prices `MLP_UP` at 3 bits and `MLP_GATE` at 4, so Phi's
up-projection rows — **0.805 B parameters, 21% of the model** — are charged a bit they do not
need, purely for sharing a tensor with the gate. Priced per row block the same floors cost
**4.2210 b** rather than 4.4310.

The cost is not primarily bytes. At a 3.25-bit target the allocator pushes `gate_up_proj` to
3 bits anyway, breaching the floor — which means the *gate* also sits at 3, the one FFN
tensor the paper's own ablation says will not tolerate it. Priced per row block, the same
bytes could buy gate 4 / up 2. So what fusion costs is the ability to spend a tensor's
allowance where it matters, across 55% of the model.

**This is a known gap, not a defect.** `allocate/` never reads `partitions`; the plan's P3
promises per-row-block widths ("gate rows 4-bit, up rows 3-bit") and `QuantTensor` already
carries `row_partitions`, so the format is ready and the allocator is not. Closing it is a
change to the allocator, the quantizer and the manifest, and it would break comparability
with phase 2's numbers. **Not started, and not to be started without a decision** — the
alternative is to run S3 as planned and report Phi's arms with this handicap stated, which
is what the numbers above are for.

`classify.py` already documents the parallel gap for MoE, where a partition scheme would have
to be defined over the expert axis. Dense fused tensors are the case where the partitions
exist and simply are not spent.

## Consequences for S3 as currently planned

1. Phi's arms should not be read as a replication of Ministral's at the same label. The
   nominal target is the same; the distance below floors is not.
2. Every Phi arm below 4.43 bits will emit a floor-violation report naming most of the model.
   That is the designed behaviour, and it is not a signal that anything is wrong.
3. The per-row allocation arms S3 was already going to run (with the shuffled-row control)
   are a *different* granularity change from the one described here: they subdivide rows
   within a role, where this would subdivide a tensor between two roles.
4. Phi's tied embedding contributes separately and is already understood — see
   `tied-embedding-breaks-baseline-bit-widths`. It is why Phi's floors start high before
   fusion is counted at all.
