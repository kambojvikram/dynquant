# The checkpoint was choosing the decode

*Record of the first G4 run: two decode defects, how far they reach, and — per the
correction below — what they turned out not to explain.*

> **Correction.** This report first named the checkpoint's `repetition_penalty` as the cause
> of the 23-point gap between the two arms. **It was not.** With the decode replaced, the
> transformers arm scored *exactly* 37.00 % again and the gap widened to 24.00 points. Both
> defects below are real and both fixes stand on their own terms — a "greedy" decode must not
> inherit the checkpoint author's chat settings, whatever it costs. But the gap had a third,
> unrelated cause, found by dumping the generations rather than by reasoning about
> mechanisms, and it is recorded in [`runtime-parity-gap.md`](runtime-parity-gap.md).

## What was run

`scripts/gate_runtime_parity.py` scores one checkpoint twice — once through
`model.generate`, once through a vLLM engine — and asks whether the two agree closely
enough that the campaign may report arms scored on either. The smoke run was
`Qwen/Qwen2.5-1.5B-Instruct`, GSM8K, `--limit 100`, on the RTX PRO 6000 box.

| arm | accuracy |
|---|---|
| vLLM | 60.00 % |
| transformers | 37.00 % |

Paired over the 100 problems: 30 both right, 7 transformers-only, 30 vLLM-only, 33 both
wrong. Delta −23.00 points, p = 1.9e−4.

Qwen publish ≈ 60 % for this checkpoint on GSM8K. The vLLM arm matched it. The
**transformers arm was the broken one**, which is the opposite of the direction a serving
gate is usually built to catch.

## Two defects, independent of each other

### 1. The checkpoint's chat preferences were merged into a "greedy" decode

`model.generate` merges the checkpoint's own `generation_config` *underneath* whatever
keyword arguments the call site passes. Every field the caller does not name is therefore
whatever the checkpoint author chose — and what they chose was chat sampling:

```json
{"do_sample": true, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
 "repetition_penalty": 1.1}
```

The old call site named `do_sample=False`, `num_beams=1`, `temperature=None`, `top_p=None`,
`top_k=None`. It did not name `repetition_penalty`, so **1.1 was applied to every greedy
generation on the transformers arm and to none on the vLLM arm**. A repetition penalty on a
GSM8K chain of thought punishes the model for re-using the numbers it is reasoning about —
which is a coherent mechanism for a lost score, reads as an explanation, and measured zero.
The re-run's transformers arm did not move by a single problem. The defect is real; the
attribution was not, and the difference between the two is a measurement.

The fix is not "also name `repetition_penalty`". `greedy_generation_config` now builds a
fresh `GenerationConfig`, which *replaces* the checkpoint's rather than layering over it, so
every field it does not name is the library's default. The token ids are the deliberate
exception — they are facts about the tokenizer, not decode settings, and dropping
`eos_token_id` would make every sequence run to `max_new_tokens`.

`NEUTRAL_DECODE` is a **tripwire, not the mechanism**: nineteen fields whose non-neutral
value would move a greedy score, each pinned to the value that means "not applied", asserted
against a fresh `GenerationConfig()`. A transformers release that changes one of those
defaults turns a test red instead of quietly re-decoding a campaign.

The vLLM arm had the mirror-image hole. Its `SamplingParams` left `repetition_penalty`,
`presence_penalty`, `frequency_penalty` and `min_tokens` unpassed because the engine's own
defaults are already neutral — true today, and nothing would have caught it changing. They
are now pinned, and a test maps the two arms' settings onto each other under their two
spellings (`num_return_sequences` ↔ `n`, `min_new_tokens` ↔ `min_tokens`).

### 2. The gate diagnosed the disagreement as a sample-size problem

Equivalence testing has three outcomes, and they are discriminated on whether the confidence
interval **excludes zero**, not on how wide it is:

1. interval inside ±δ → equivalent (a statistically significant but tiny delta passes);
2. interval excludes zero → the arms differ, and more data will not change that;
3. interval contains zero but exceeds ±δ → underpowered.

`_judge` checked width first, so case 2 was reported as case 3: "too few problems to tell".
Acting on that message means scoring *more* problems — the one action that cannot help, and
the expensive one. The three cases are now separate branches with distinct messages, and the
100-problem counts above are a test fixture.

## How far it reaches

A checkpoint only carries the defect if its `generation_config.json` sets a non-neutral
field. Measured, by fetching each file from the Hub:

| checkpoint | ships penalties? | affected |
|---|---|---|
| `Qwen/Qwen3.5-2B-Base` (phase 1 & 2) | **no `generation_config.json` at all** | no |
| `mistralai/Mistral-7B-Instruct-v0.3` (phase 2) | token ids only | no |
| `microsoft/Phi-4-mini-instruct` (phase 3) | token ids only | no |
| `mistralai/Ministral-8B-Instruct-2410` (phase 3) | token ids only | no |
| `Qwen/Qwen2.5-1.5B-Instruct` (G4 smoke only) | `repetition_penalty: 1.1` + sampling | **yes** |
| `google/gemma-3-4b-it`, `meta-llama/Llama-3.1-8B-Instruct` | not checked — gated, 401 without a token | unknown |

So **no published DynQuant number changes**: the phase-1 and phase-2 campaigns ran on a base
checkpoint that ships no generation config and on a Mistral instruct checkpoint that ships
only token ids.

Two limits on that statement, stated rather than glossed:

- Where a contaminated path *had* been used, a **paired A/B between two quantizations scored
  through it is still valid for the difference** — both arms carry the same penalty. Only
  absolute accuracy moves.
- The two gated phase-3 models could not be checked without an accepted licence. It no longer
  matters for correctness — the fix replaces the config rather than patching a field — but it
  is why the S1 screen must not be read as a check on this.

## What the re-run measured

The section this replaces said the `repetition_penalty` explanation was "a hypothesis with a
matching mechanism, not a measurement", and that it would become a result when the gate was
re-run and the two arms came back inside the bound. The gate was re-run on the same
checkpoint, same 100 problems, with the decode fixed:

| arm | before the fix | after the fix |
|---|---|---|
| transformers | 37.00 % | **37.00 %** |
| vLLM | 60.00 % | 61.00 % |
| delta | −23.00 | **−24.00**, 95 % CI [−34.78, −13.22], p = 7.0e−5 |

Not "narrowed less than hoped" — *unchanged*, to the problem. Whatever the penalty was doing
to those generations, it was not deciding whether they were right.

Two things this pins for later:

- **The fix stands regardless.** A greedy decode that silently inherits `do_sample: true`,
  `temperature: 0.7` and `top_p: 0.8` from the checkpoint is wrong whether or not it happens
  to cost points on one task and one model. `NEUTRAL_DECODE` still earns its place: it fails
  loudly on a transformers release that moves a default, which is a different risk from the
  one that was actually realised here.
- **A mechanism that fits is not a cause.** The penalty story was consistent with every
  number then in hand, and the way it was settled was not more reasoning but dumping the text
  both arms produced. See [`runtime-parity-gap.md`](runtime-parity-gap.md).

The smoke's `--limit 100` cannot certify equivalence at `--max-delta 1.0` in any case — 100
problems give an interval far wider than the bound — so the certificate still needs the full
1319, and that run is only worth its GPU time once a smoke comes back inside the bound.
