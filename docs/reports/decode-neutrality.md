# The checkpoint was choosing the decode

*Record of the G4 runs: three decode defects, how far they reach, how the right diagnosis
came to be retracted on the strength of a fix that never ran, and what the gate measures
now that all three are fixed.*

> **Correction, superseding the one this replaces.** This report first named the checkpoint's
> `repetition_penalty` as a cause of the gap between the two arms. A later revision retracted
> that in bold — the decode had been replaced, the transformers arm scored *exactly* 37.00 %
> again, and a mechanism that moves nothing is not a cause.
>
> **The retraction was wrong.** The penalty is a cause, worth 19 points. The fix could not
> take effect, for a reason no assertion in this repository could see: transformers 5.x
> refills a passed `GenerationConfig`'s unset fields from `model.generation_config`, so
> building a fresh config no longer replaces the checkpoint's. On the 4.x line it does, and
> the fix as written was correct there — which is why every test stayed green. Isolated on
> GSM8K problems 100–199, one field at a time: **42/100 as shipped, 61/100 with
> `repetition_penalty` pinned to 1.0**, against 61/100 on transformers 4.56.2 and 60/100 on
> vLLM. Pinning all nineteen `NEUTRAL_DECODE` fields gives the same 61 and the same
> per-problem hits.
>
> So there were three defects, not two, and all three are real: the penalty (this report),
> the sample-size misdiagnosis (this report), and the stop sequence
> ([`runtime-parity-gap.md`](runtime-parity-gap.md)). The stop sequence is what moved the
> smoke from 37 to 61; the penalty is what kept the *full* 1319-problem gate failing at
> −9.17 points after the smoke had come back clean. With the penalty actually neutralised the
> full gate reads −0.83 (p = 0.27) — [measured below](#what-the-fix-measured-at-full-scale),
> and still a FAIL, for a reason that is about GSM8K's size rather than about the decode.
>
> The method error is recorded rather than tidied away, because it is the more general one:
> **"the fix didn't move the number" has two readings, and only one of them was checked.**
> Either the cause was wrong, or the fix never reached the code path. Distinguishing them
> costs one assertion on the object the library actually used.

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
GSM8K chain of thought punishes the model for re-using the numbers it is reasoning about.
Note which three fields transformers reports as "not valid and may be ignored" under
`do_sample=False`: `temperature`, `top_p`, `top_k`. Those are sampling warpers and the
processor list never builds them. `repetition_penalty` is not a warper, applies under greedy,
and is not reported. The warning that looked like reassurance was the merge announcing
itself.

The first fix was "build a fresh `GenerationConfig`, which *replaces* the checkpoint's rather
than layering over it, so every field it does not name is the library's default". That
sentence is true on transformers 4.x and false on 5.x, where an unset field means "the caller
did not set this" and gets refilled from `model.generation_config`. Same source, same tests,
two behaviours — and the campaign ran on the line where it was false.

So `NEUTRAL_DECODE` **is** the mechanism now, not a tripwire on one: nineteen fields whose
non-neutral value would move a greedy score, each pinned explicitly in the returned config so
that neutrality is a property of this dict rather than of any library's defaults. The token
ids are the deliberate exception — they are facts about the tokenizer, not decode settings,
and dropping `eos_token_id` would make every sequence run to `max_new_tokens`.

It remains a tripwire as well: a release that moves one of those defaults turns a test red
instead of quietly re-decoding a campaign. What no formulation here can cover is a field
transformers *adds* later — `GenerationConfig` has no "give me nothing but greedy"
constructor — and the guard for that is the gate itself.

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
- The two gated phase-3 models could not be checked without an accepted licence. With the
  final fix it no longer matters for correctness — every neutral field is pinned by name, so
  what the checkpoint ships cannot reach the decode whatever it is — but it is why the S1
  screen must not be read as a check on this. Under the *first* fix it mattered a great deal,
  which is the point: "the fix is general, so the unchecked cases are covered" was an
  argument, and it was resting on a library behaviour that had changed.

## What the re-run measured, and what that turned out to mean

The gate was re-run on the same checkpoint, same 100 problems, with the decode "fixed":

| arm | before the fix | after the fix |
|---|---|---|
| transformers | 37.00 % | **37.00 %** |
| vLLM | 60.00 % | 61.00 % |
| delta | −23.00 | **−24.00**, 95 % CI [−34.78, −13.22], p = 7.0e−5 |

Not "narrowed less than hoped" — *unchanged*, to the problem. That was read as refutation and
written up as one. It was a null result about the **fix**, not about the cause: on
transformers 5.x the new config left `repetition_penalty` unset, `generate` refilled it from
the checkpoint, and the arm decoded exactly as before. A no-op cannot move a score, and a
no-op is what was measured.

Nothing available at the time distinguished the two readings. Every test was green — because
every test asserted on the config the harness *built*, which was already correct, rather than
on the one `generate` used. The smoke then came back 61.00/61.00 after the stop-sequence fix,
which looked like closure and was in fact the second piece of bad luck: block 0 is the only
100-problem block of GSM8K where these two arms agree.

## The isolation

The full 1319-problem gate failed at −9.17 points, and the block-wise breakdown put block 0
at −1.0 against a spread running to −18.0 elsewhere. Re-slicing block 100–199 into its own
run, one variable at a time, against vLLM's 60/100:

| arm | score | rules out |
|---|---|---|
| transformers, full run | 43/100 | — |
| transformers, re-sliced | 42/100 | batch composition |
| transformers, `batch_size=1` | 42/100 | padding |
| transformers, sdpa/float32 | 42/100 | precision |
| transformers **4.56.2**, sdpa | **61/100** | *the version* |
| transformers 5.5.3, sdpa | 42/100 | not a 5.14.1 regression — the whole 5.x line |
| 5.5.3, `generation_config` stripped | **61/100** | the checkpoint's file is the carrier |
| 5.5.3, `+repetition_penalty=1.0` only | **61/100** | one field, 19 points |
| 5.5.3, `+NEUTRAL_DECODE` (19 fields) | **61/100**, per-problem identical | the fix does nothing beyond neutralising |

Prompt token ids were verified byte-identical across the two venvs first (both `tokenizers`
0.22.2, same SHA, same per-prompt lengths), so "different transformers, different
tokenization" was eliminated before the version arms were believed. The neutralised 5.5.3
generations came back character-identical to 4.56.2's.

The dumped generations rule out the shapes that would have been visible in a summary line:
zero contain `"Question:"`, zero exceed 1200 characters, zero unparseable. They are simply
worse reasoning — problem 107, gold 3, answered 30 via "7 − 8 = 1 hour"; problem 109, gold
28, answered 32.00 by adding the cashback instead of subtracting it.

## What the fix measured at full scale

The gate was re-run on all 1319 GSM8K test problems with the nineteen fields pinned:

| | before the fix | after the fix |
|---|---|---|
| transformers | 53.15 % (701/1319) | **61.49 % (811/1319)** |
| vLLM | 62.32 % (822/1319) | 62.32 % (822/1319) |
| delta | −9.17 | **−0.83** |
| 95 % CI | [−11.92, −6.43] | [−2.19, +0.52] |
| p | 1.121e−10 | 0.2723 |
| agreement | 73.24 % | **93.71 %** (1236/1319) |
| discordant | 353 | 83 — 36 transformers-only, 47 vLLM-only |
| verdict | separated → FAIL | underpowered → FAIL |
| wall clock | tf 682.8 s / vLLM 20.6 s | tf 682.5 s / vLLM 20.7 s |

vLLM's 822 is byte-identical across the two runs, which is the internal check that only the
transformers path moved. The transformers arm gained exactly 110 problems.

**The fix works and the gate still fails**, now for the other reason. The residual −0.83 is
not distinguishable from zero (p = 0.27) and the split of the 83 discordant problems is 36/47
— consistent with unbiased noise rather than one arm being systematically worse, which is what
first-token divergence between two independent bf16 attention implementations produces.

The interval is 2.71 points wide against a ±1.00 bound, and that is now the binding
constraint. At 6.29 % discordance a ±1.00 half-width needs **n ≥ 2418** problems; GSM8K's test
split *is* 1319. Equivalently, holding n = 1319, discordance would have to fall to ≤ 45
problems (3.43 %) — below the floor greedy argmax flips put under it. So the gate's standing
advice, "score more", is unfollowable on this task. `_judge` now takes `exhausted=args.limit is
None` and, in that case, reports the two actions that do exist: make the runtimes agree more
closely, or set the bound to what the task can resolve.

The campaign consequence is the part worth carrying forward. **Cross-engine parity on GSM8K is
certifiable to about ±1.4 points, and DynQuant's phase-2 headline margin over GPTQ is +1.54.**
A comparison that scores one arm through vLLM and the other through transformers would spend
most of its claimed margin on runtime noise. The defensible arrangement is one engine per
comparison — vLLM throughout, which is also 33× faster — leaving this gate as a check that the
reference implementation still agrees, not as a licence to mix engines inside a headline.

## Why no test caught it, and what now does

The repository already had the assertion this needed —
`test_the_neutral_values_are_the_librarys_own_defaults`, checking
`GenerationConfig().repetition_penalty == 1.0`. It never fired, for two compounding reasons:

1. **The suite installs no transformers.** `pytest.importorskip("transformers")` skipped the
   entire decode module in CI while the campaign ran on 5.14.1. Skipped is not green.
2. **On 4.x the broken and fixed programs are indistinguishable.** Verified by mutation:
   reverting the fix fails four tests on 5.14.1 and *zero* on 4.56.2. Local development
   (4.53.2) could not have caught this at any level of test-writing effort.

So the guards are:

- `greedy_generation_config` pins all nineteen fields by name; the mutation above is what
  turns them red.
- `_library_neutral()` resolves each field's neutral value through both lines' conventions
  (attribute on 4.x, `_get_default_generation_params()` on 5.x), so the tripwire is a real
  assertion on either rather than `None == None`.
- `test_a_checkpoints_preferences_cannot_refill_the_built_config` replays the merge —
  `update(**shipped, defaults_only=True)` where that exists — instead of trusting the built
  object.
- A `transformers-lines` CI job runs the suite on 4.56.2 **and** 5.14.1 with transformers
  actually installed, asserts the pinned version is the one that got imported, and fails if
  any transformers-gated file skips.

The pins must track whatever the campaign runs. That is the whole process fix; the rest is
detail.
