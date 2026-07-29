# Mistral-7B-Instruct-v0.3 / Banking77 — run records

The records every number in
[`RESULTS-mistral7b-banking77.md`](../../experiments/four_point/RESULTS-mistral7b-banking77.md)
was read from. Written by `experiments/four_point/` on 2026-07-29 against a single
A100 80 GB, `DQ_MODEL=mistralai/Mistral-7B-Instruct-v0.3 DQ_TASK=banking77`.

They are committed for the same reason the headroom screen script is: a result whose
inputs live only on a rented box is not reproducible, it is just asserted. The box goes
away; this does not.

| file | what it holds |
|---|---|
| `signals.json` | the signal map — 226 layers, StatsFile schema, written by `DynQuantCallback` during the LoRA fine-tune. **This is the input to scoring and allocation**, so it alone reproduces the bit maps below without re-running the 60-minute fine-tune. |
| `stage1_base.json` | base bf16 eval, 36.27% |
| `stage2_finetune.json` | fine-tune config and timings |
| `stage3_finetuned.json` | fine-tuned bf16 eval, 94.51% |
| `stage4_bitmaps.json` | the allocated maps at both budgets |
| `stage5_*.json` | the four quantized arms' evals |
| `stage5_*_quant.json` | per-module widths, RMSE and clip statistics for each arm |

Each eval record carries a `hits` vector — the per-problem correct/incorrect
sequence — which is what makes the paired McNemar comparisons possible after the fact.
The order is the fixed-seed shuffle from `dynquant.eval.banking77`, identical across
arms; that is a correctness requirement, not a convenience, because a paired test
compares position by position.

Not committed: `dynquant_moments.safetensors` (10.7 MB), the raw accumulator
`signals.json` was reduced from, and the 14 GB merged fp16 checkpoint. `*.safetensors`
is gitignored, and neither is needed to reproduce anything in the write-up.

## Re-running allocation from these records

```bash
dynquant inspect <merged-model> --stats stats/mistral7b_banking77/signals.json --target 3.25
```
