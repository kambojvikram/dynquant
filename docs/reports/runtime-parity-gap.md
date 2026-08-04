# The stop sequence the model never wrote

*Why the two runtimes scored 24 points apart on the same 100 GSM8K problems, and what the
generations said when they were finally read.*

## Where this picks up

[`decode-neutrality.md`](decode-neutrality.md) records the first G4 run: the `transformers`
arm scored 37.00 % against vLLM's 60.00 % on `Qwen/Qwen2.5-1.5B-Instruct`, and the
checkpoint's `repetition_penalty: 1.1` — merged into a supposedly greedy decode — was named
as the cause. The decode was replaced with one that does not inherit anything. The gate was
re-run. The `transformers` arm scored **37.00 %**, unchanged, and the gap widened to 24.00
points because one vLLM answer flipped the other way.

At that point the score had said everything a score can say. The next step was to stop
reasoning about mechanisms and read the text.

## What the generations said

`diag_arm_gap.py` re-ran both arms on eight hand-picked problems — the six the arms
disagreed on, plus two both got right — with `keep_predictions` set so the text survived.
Three of the six losses looked like this (problem 10, gold 366):

| arm | chars | predicted | text |
|---|---|---|---|
| vLLM | 233 | **366** | `…26 = 366.\n\nTherefore, the answer is 366.` |
| transformers | 1001 | **25** | `…60 + 180 + 126 = 366 downloads.\n\nTherefore, the answer is 366. Question: There are 12 more green apples than red apples in a bowl.\nAnswer: … Total: 16 + 28 = 44 apples.\n\nTherefore, the answer is 44. Question: John has just begun working…` |

The `transformers` arm **solved the problem** — "the answer is 366" — and then kept going,
inventing two more problems and answering those too, until it hit `max_new_tokens=320`.
`extract_answer` found no `####` marker, fell back to "the last number in the text", and
returned a number from the third invented problem. Problems 6 (gold 260 → predicted 10) and
17 (gold 57500 → predicted −2) failed identically, at 1057 and 952 characters.

That is the worst shape a scoring bug can take. It is not an unparseable generation, which
the harness counts separately and which would have been visible in the summary line. It is a
confident wrong number that reads like bad arithmetic.

## The defect

`FEWSHOT_STOP` was `"\n\nQuestion:"`, and the reasoning behind it was sound as far as it
went: `build_prompt` joins its exemplars with a blank line, so a blank line followed by
`Question:` is the model's own turn boundary.

The model does not write it back. It writes:

```
Therefore, the answer is 366. Question: There are 12 more green apples…
```

One space, same line. Across the eight dumped generations the literal `"\n\nQuestion:"`
appears **zero** times and the bare `"Question:"` appears seven. So:

- `StopStringCriteria` never fired — generation ran to the full token budget.
- `_truncate` never matched — the run-on reached the extractor intact.

Both arms carried the same stop, so vLLM had the same hole. It was not hurt by it because
its generations ended at EOS before reaching for a second problem, which is the divergence
described below.

The fix is one line: `FEWSHOT_STOP = "Question:"`. Cutting on the bare word costs nothing —
a continuation that has written "Question:" has stopped answering either way — and it is
also what `lm-eval-harness` uses for this task. Truncated at that boundary, problem 10's
generation ends `…Therefore, the answer is 366.` and the fallback extractor returns 366.

## What is *not* the defect, checked rather than assumed

Two other explanations fitted the shape of the data and both were wrong.

**Batched, padded generation.** The `transformers` arm pads a 48-wide batch to its longest
prompt; vLLM pads nothing. Re-running the same eight problems at `batch_size=1`:

| batch size | correct (of 8) |
|---|---|
| 48 | 4 |
| 8 | 4 |
| 1 | 5 |

Real, and worth knowing — the six disputed problems went 0/6 at batch 48 and 3/6 at batch 1,
so padding costs something on this model. But it is not the mechanism: at `batch_size=1` the
arm still ran on past its own answers, and still lost problems the widened stop recovers.

**Different inputs.** Ruled out by construction, not by inspection: `VllmBackend` is handed
the same token ids the `transformers` arm gets, and `_assert_aligned` checks the ids vLLM
echoes back against the ids sent, per prompt. Both arms load `bfloat16`. Neither applies a
chat template.

## The residual: the arms diverge at the first token

With identical ids, identical weights and greedy decode, the two runtimes still produced
different text from **character 0** on seven of the eight problems:

```
problem 4  bs1 : 'First find the total number of cups of feed given away: 15 cups + 25…'
           vllm: 'In the morning, Wendi gives her flock of chickens 15 cups of feed. In…'
```

This is expected and is not a bug in either arm. After `Answer:` the model's first-token
distribution is close to tied between a dozen ways of opening a sentence; `sdpa` and
FlashAttention-2 reduce a 1111-token prefill differently in bf16; the argmax flips; and
greedy decoding amplifies one flipped token into an entirely different chain of thought.

What it means for the campaign is the part worth stating plainly: **a runtime-parity gate on
a generative task can never certify that two engines produce the same text.** It can only
certify that they produce the same *score* within a bound. Those are different claims, and
the second is the one the campaign needs — arms scored on either engine must be comparable,
not identical.

It also sets the ceiling on what the widened stop can fix. The stop removes a bias: run-ons
were a one-directional failure, always converting a solved problem into a wrong answer, and
they hit the `transformers` arm because that arm ran on. What remains after it is chaotic
divergence, which is unbiased and shrinks with problem count. Whether it shrinks *enough* to
clear `--max-delta 1.0` is the next measurement, not a prediction.

## Guards added

- `test_the_stop_matches_a_run_on_the_model_writes_inline` — the real generation above,
  shortened, asserted to extract 366 truncated and 44 untruncated. Red if `FEWSHOT_STOP`
  regains either newline.
- `test_the_task_adds_its_stop_to_a_config_that_arrives_without_one` — `evaluate_gsm8k` run
  end to end against the decode stub with an `EvalConfig` carrying `stop_sequences=()`, which
  is exactly what `gate_runtime_parity.py` builds. Red if the task stops merging its own stop
  into a caller's config.

Seven mutations run against the pair: six killed, one an equivalent mutant (the default
`EvalConfig`'s `stop_sequences` argument is redundant with the merge two lines below it, so
no test can distinguish the two programs).

## Why the harness could not have caught this

Nothing in the `transformers` path is wrong. The generations are exactly what the model
produced; the extractor did what its docstring says. Every component behaved as specified and
the score was still wrong by 24 points, because the specification of "where does an answer
end" was written from the prompt format rather than from what models emit.

The gate caught it. That is the whole argument for scoring the same checkpoint twice before
trusting either number — and note the direction: the gate exists to catch a *serving* engine
drifting from the reference, and what it found both times was the reference arm.
