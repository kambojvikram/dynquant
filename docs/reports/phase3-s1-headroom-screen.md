# Phase 3 · S1 — the headroom screen

**Measured 2026-08-04.** Instruct checkpoints at fp16, four benchmarks, before any fine-tune
is spent. Raw records: [`experiments/phase3/s1_headroom/`](../../experiments/phase3/s1_headroom/).

S1 exists because of a rule this project already paid to learn: *screen headroom before you
spend a fine-tune.* GSM8K's flat arms in campaign 1 cost a full six-arm run to diagnose, and
the diagnosis was that the base model was already at the supervised ceiling, so nothing
downstream could read quantization damage against a fine-tuning gain. Phase 3 changes the
model panel, the datasets and the benchmarks all at once, so every one of those benchmarks has
to be shown to have room *first*.

A headroom screen is cheap — eight arms, 7 minutes of GPU — and it caught two harness defects
that would have invalidated half the campaign. That is the second thing this report is about.

---

## 1. The panel, and what is missing from it

| model | params | tokenizer backend | in the screen |
|---|---|---:|---|
| `microsoft/Phi-4-mini-instruct` | 3.8 B | `TokenizersBackend` (Jinja) | ✅ |
| `mistralai/Ministral-8B-Instruct-2410` | 8.0 B | `MistralCommonBackend` (tekken) | ✅ |
| `meta-llama/Llama-3.1-8B-Instruct` | 8.0 B | — | ❌ **blocked** |
| `google/gemma-3-4b-it` | 4.3 B | — | ❌ **blocked** |

**The two missing models were a blocker, and are now a decision.** Both repositories are
`gated=manual` on the Hub: access requires accepting a licence *per Hugging Face account*,
through the web UI, and no token is configured on the box — not resolvable from the machine or
from the code. **On 2026-08-05 the panel was settled at two models** rather than held for a
token, so Llama-3.1-8B-Instruct and gemma-3-4b-it are out of phase 3.

What survives is a panel spanning fused projections (Phi, `qkv_proj` / `gate_up_proj`) against
unfused dense GQA (Ministral), two different tokenizer backends, and 2.1× of scale. What is
given up is the Llama family and the only model with a `sliding-window + full` alternating
stack. Both are stated as scope in every phase-3 claim rather than left implied. Adding a model
later costs one fine-tune and its arms, and invalidates nothing: every arm is compared against
its own model's bf16 ceiling.

## 2. Setup

One NVIDIA RTX PRO 6000 Blackwell Workstation Edition (97 887 MiB, driver 580.159.03), vLLM
0.26.0, torch 2.11.0+cu130, transformers 5.14.1, `dynquant-core` 0.2.0. Driver script:
[`s1_screen.sh`](../../experiments/phase3/s1_headroom/s1_screen.sh).

| task | split | shots | max new tokens | scored by |
|---|---|---:|---:|---|
| IFEval | `train` (541) | 0 | 1024 | prompt-strict, `regex-sentences+regex-words+langdetect` |
| GSM8K | `test` (1319) | 5 | 320 | exact match on the final number |
| HumanEval | (164) | 0 | 1024 | executed, `exec/linux/py3.12/rlimits/t=8s/m=4096MB` |
| MBPP | `test` (500) | 3 | 1024 | executed, same sandbox |

Greedy throughout. Every arm scores through **vLLM**, not `transformers` — the
one-engine-per-comparison rule from [`decode-neutrality.md`](decode-neutrality.md): cross-engine
GSM8K parity is certifiable only to ≈±1.4 points against a phase-2 margin of +1.54, so an
engine change inside a comparison is not affordable.

**Everything is read from cache.** `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` is set for two
reasons, and the first one already cost four arms. vLLM's loader globs `*.safetensors` from the
Hub and only *then* filters against `model.safetensors.index.json`; Ministral's repository ships
a 16 GB `consolidated.safetensors` beside the four sharded files, so every arm re-pulled 16 GB
it was about to discard, and this link drops large transfers at around a gigabyte. All four
Ministral arms died in engine init with an `httpcore.RemoteProtocolError` that named `httpx` and
looked nothing like a loader bug. The second reason is that a scored run should not be doing
network I/O at all: a stalled download mid-evaluation is a timing artefact inside a
measurement.

## 3. Results

Every arm below: **0 unscored, 0 empty generations**, correct framing recorded on the result.

| model | IFEval | GSM8K | HumanEval | MBPP |
|---|---:|---:|---:|---:|
| Phi-4-mini-instruct | 68.76 % | **83.17 %** | 77.44 % | 60.00 % |
| Ministral-8B-Instruct | 54.53 % | 80.89 % | **79.27 %** | 55.80 % |
| | 372 / 541 · 295 / 541 | 1097 / 1319 · 1067 / 1319 | 127 / 164 · 130 / 164 | 300 / 500 · 279 / 500 |

IFEval secondary metrics, which the campaign will report alongside prompt-strict because the
strict metric moves in steps of a whole prompt:

| model | prompt-strict | prompt-loose | instruction-strict | instruction-loose |
|---|---:|---:|---:|---:|
| Phi-4-mini | 68.76 % | 72.09 % | 77.10 % | 80.46 % |
| Ministral-8B | 54.53 % | 58.04 % | 64.39 % | 67.75 % |

(834 instructions across the 541 prompts, both models.)

**All eight arms have headroom.** The highest is Phi's GSM8K at 83.17 %, which leaves 16.8
points before the ceiling; the lowest, Ministral's IFEval at 54.53 %, leaves 45. Nothing is
near enough to 100 % that quantization damage would have nowhere to show, which is the single
question S1 was run to answer. **Verdict: all four benchmarks are usable for the headline.**

Two observations worth carrying forward:

- **The two models are not ranked consistently across the four tasks.** Phi wins IFEval by 14.2
  and MBPP by 4.2; Ministral wins HumanEval by 1.8; GSM8K is within 2.3. A per-task split like
  that is what makes a four-benchmark panel worth running instead of one benchmark plus three
  correlated ones.
- **MBPP is the tightest floor** (60.0 % / 55.8 %) and the one where a large quantization loss
  will approach the range where 3-shot prompt noise matters. It is kept, but it is the arm most
  likely to need the paired-hits treatment to separate anything.

---

## 4. What the screen caught

Two defects, both in the harness rather than in the models, both of which produce a *stable,
plausible, wrong number* rather than an error. They are the reason S1 ran three times, and the
losing arms are kept as controls because they price what each bug cost.

### Ministral-8B, one checkpoint, three framings

| framing | IFEval | empty | HumanEval | empty | MBPP | empty |
|---|---:|---:|---:|---:|---:|---:|
| raw text (bug 1) | 24.77 % | 195 / 541 | 70.73 % | 0 | 43.60 % | 0 |
| de-tokenized chat frame (bug 2) | 37.52 % | 84 / 541 | 23.17 % | 120 / 164 | 52.80 % | 1 |
| **chat frame as ids (correct)** | **54.53 %** | **0** | **79.27 %** | **0** | **55.80 %** | **0** |

Records: [`control_raw_framing/`](../../experiments/phase3/s1_headroom/control_raw_framing/),
[`control_detokenized_frame/`](../../experiments/phase3/s1_headroom/control_detokenized_frame/),
[`corrected/`](../../experiments/phase3/s1_headroom/corrected/).

**The cost of getting the framing wrong on this model is up to 56.1 points on HumanEval and
29.8 on IFEval** — between five and thirty times the effect size phase 2 was measuring
(+1.54). Neither wrong number is near zero.

### Bug 1 — an instruct checkpoint measured as a base checkpoint

Fixed in [`8336aab`](../../CHANGELOG.md). The harness decided whether to frame a prompt as a
chat turn by reading `tokenizer.chat_template`. That attribute belongs to the *Jinja-backed
implementation*, not to the capability. `AutoTokenizer` returns `MistralCommonBackend` for any
Mistral checkpoint shipping a `tekken.json` — Ministral among them — and that class leaves the
attribute `None` while `apply_chat_template` works and renders `<s>[INST]…[/INST]`.

So an instruct model was handed bare text, and an instruct model handed bare text *continues*
it rather than answering it: 24.77 % on IFEval with 195 of 541 generations empty.

`harness.chat_prompt_style` now decides by asking the tokenizer to render a turn and seeing
whether it can. That distinguishes "will not" from "cannot" — a base checkpoint keeps getting
the raw prompt, because it genuinely has no turn structure, and that remains a real difference
in measurement rather than a fallback to paper over.

### Bug 2 — the frame detected, then thrown away on the way to the model

Fixed in [`2b62907`](../../CHANGELOG.md). The more dangerous half, and it was *exposed* by
fixing bug 1. Having correctly decided Ministral is an instruct checkpoint, the harness rendered
the turn with `apply_chat_template(tokenize=False)` and re-tokenized that string. Rendering is
lossy for `MistralCommonBackend`: it emits the frame as the **characters** `<s>[INST]…[/INST]`,
and `tekken` never parses control tokens back out of user text — a deliberate injection guard,
not a bug. The frame survived rendering and died on encoding:

```
<s>[INST]Write a function add(a,b).[/INST]
  re-tokenized text →  17 ids, [1060, 1115, 110391, …]   no BOS, no [INST]
  tokenizer's own   →  10 ids, [1, 3, 18746, …, 4]       BOS, [INST], [/INST]
```

transformers warns about exactly this (`apply_chat_template(..., tokenize=False)` "is unsafe …
don't encode the output manually") in a line a batch evaluation buries. Handed the flattened
frame, the model mostly returned nothing at all: 120 of 164 HumanEval problems empty.

`harness.render_chat` removes the round trip rather than repairing it — it asks for
`tokenize=True` and returns token ids, and `encode_prompts` accepts a prompt that is already ids
and truncates it identically. "Re-tokenizing rendered text reproduces the render" is an
assumption no tokenizer promises to keep.

**Phi-4-mini's four arms were not re-run, and did not need to be.** The round trip is lossless
for a Jinja-backed tokenizer — verified on the box, the same ten ids either way — and a test
pins that property so the claim is not a one-off observation.

### What made both of them visible

`prompt_style`, recorded on every result. Bug 1 showed up as `raw` on one model and
`chat-template` on the other in the same table; there was no other signal. It is the argument
for keeping this class of provenance on every result — and also the limit of it, because bug 2
reported `chat-template` throughout and had to be found by comparing the ids the model received
against the ids the tokenizer produces for the same turn.

**Both fixes ship with regression tests that turn red on the specific reversal**, including a
`_MistralShaped` double that renders control tokens as text and then refuses to read them back
— the shape of the real class, not a hypothetical invented for a test. CI green on both commits.

---

## 5. Open items out of S1

1. **`HF_TOKEN` for the two gated models.** Requires a Hugging Face account that has accepted
   the Llama-3.1 and Gemma-3 licences. Nothing else in the campaign is blocked on it, but the
   panel is 2 models until it lands. A token would also lift the Hub rate limits that made the
   Ministral prefetch take five attempts.
2. **MBPP's floor.** 55.8 % is the lowest arm in the screen; watch whether 3-shot exemplar
   choice moves it more than quantization does.
3. **`langdetect` is a hard requirement for IFEval**, not an optional extra: 95 of 541 prompts
   carry constraints that cannot be scored without it, and the scorer refuses to produce a
   number rather than guess. `nltk`'s `punkt_tab` is *not* required — the scorer probes for it
   with a real call and records the regex fallback it chose in `detail.scorer`.
