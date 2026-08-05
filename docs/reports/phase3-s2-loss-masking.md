# Phase 3 · S2 — locating the assistant turns

**Measured 2026-08-04.** Two tokenizers, 3 000 Tulu-3 conversations each, before any of the
~30 GPU-hours of S2 fine-tuning is spent. Driver:
[`scripts/run_s2_finetune.py`](../../scripts/run_s2_finetune.py); dry-run records:
`/workspace/runs/s2probe/*/mask_census.json`.

Supervised fine-tuning on a chat mixture needs one thing that no dataset ships and no tokenizer
API reliably provides: **which token positions are the assistant's**. Everything else in S2 —
the mixture, the LR schedule, the signal hook — is standard. This is the part that can be
silently wrong, and a silently wrong loss mask does not fail. It trains a slightly worse model
and reports success, which is the worst possible outcome for a campaign whose entire purpose is
to measure a small difference between quantization arms on top of that model.

So the mask is measured, not assumed, and the driver refuses to train when the measurement does
not hold. This report is about what the measurement found: **the obvious method works on one of
the two tokenizers on the panel and drops 100 % of the data on the other**, three designs died
against real data, and the one that survived is verifiable against an independent one.

---

## 1. The obvious method, and the assumption inside it

Render `messages[:i]`, render `messages[:i+1]`, and the difference in length is turn `i`'s span.
It reads like arithmetic. It is not — it is an assumption:

> ids for `messages[:i+1]` begin with ids for `messages[:i]`.

Call it **prefix-stability**. No tokenizer promises it. A template that emits a
`<|im_start|>`-style opener per turn happens to have it; one that closes a finished document, or
merges turns, or strips whitespace at a boundary, does not. One token of drift shifts every
label after it by one position, and the run still completes.

`template` mode does exactly this walk and *checks* the assumption on every turn rather than
relying on it. That check is the whole point: on a tokenizer where the assumption holds it costs
two renders per turn and nothing else; where it fails, it fails loudly.

**The alternative API is worse than useless here.** `apply_chat_template(...,
return_assistant_tokens_mask=True)` looks purpose-built for this and is a trap: it requires
`{% generation %}` markers inside the Jinja template. Phi-4-mini's template has none, so the
call **succeeds** and returns a full-length mask of zeros. A driver that trusted it would
supervise nothing at all, train for 15 GPU-hours, and never emit a warning.

## 2. What the panel's two tokenizers actually do

| | `microsoft/Phi-4-mini-instruct` | `mistralai/Ministral-8B-Instruct-2410` |
|---|---|---|
| backend | `TokenizersBackend` (Jinja) | `MistralCommonBackend` (tekken) |
| assistant turn ends with | `<|end|>` — id **200020** | `</s>` — id **2** |
| `eos_token_id` | `<|endoftext|>` — id **199999** | 2 |
| prefix-stable per turn | ✅ | — cannot be asked |
| renders an assistant-final conversation | ✅ (and *closes* it) | ❌ **refuses** |
| `continue_final_message=True` | ✅ | ✅ |

Two rows of that table are the entire problem.

**`mistral_common` refuses to render any conversation ending in an assistant message.** With
`add_generation_prompt=False` it raises `InvalidMessageStructureException` ("Expected last role
User or Tool ... for serving"); with `True` it raises a `ValueError` naming
`continue_final_message`. It is validating a *serving request*, and every prefix the per-turn
walk needs is precisely the shape it rejects. Measured: **3 000 / 3 000 conversations dropped**,
one reason, `full render raised InvalidMessageStructureException`.

**Phi closes a conversation it is not asked to continue.** `add_generation_prompt=False` on a
finished conversation appends `<|endoftext|>`. So that render is *not* a prefix of the training
sequence — the sequence needs the next turn's header where the render has a document terminator.

## 3. `assemble` mode, and the constant it needs

What `mistral_common` *does* accept is the shape its own error message asks for, and the shape
assistant-final text has during training rather than serving: `continue_final_message=True`,
which renders the last turn **open** — header and content, no terminator, no following header.
Both backends accept it. That collapses the problem to two legal renders per assistant turn:

* **`prompt`** = `messages[:i]` + generation prompt — literally the serving request the model
  would receive before writing this turn. The supervised span starts at its end, so the
  assistant header is context, never a target.
* **`through`** = `messages[:i+1]` continued — the same thing plus this turn's content.

Neither contains the turn terminator, and the model must learn to emit it or it never stops.
(That lesson is already in this project's record: a GSM8K stop sequence the model never wrote
back cost 24 points, silently.)

**The terminator cannot be recovered by arithmetic.** No legal render on either tokenizer ever
ends *between* an assistant terminator and the next turn's opener — they are adjacent constants,
so every length equation recovers only their sum. Three synthetic renders split them:

| render | messages | ends with |
|---|---|---|
| `open` | `[user, assistant]`, continued | assistant content |
| `closed` | `[user, assistant, user]` + generation prompt | terminator **+** user frame |
| `lone` | `[user]` + generation prompt | the identical user frame |

`closed` and `lone` end in the same user frame, so their shared suffix reaches back exactly to
the terminator and stops — `lone` has a BOS there, `closed` has the terminator. Measured once
per tokenizer, cached: Phi `[200020]` `<|end|>`, Ministral `[2]` `</s>`.

Note which token that is **not**. Phi's `eos_token_id` is `<|endoftext|>` (199999), a different
token from the one that ends its turns. A design that reached for `eos_token_id` would have
trained the model to end every reply with the wrong token.

**Getting it wrong is not silent.** The assembled sequence must be a prefix of the *next* turn's
`prompt`, which the template renders itself. A wrong terminator therefore costs dropped rows,
never mistrained spans — the failure is converted from invisible to loud.

## 4. Three designs that passed unit tests and failed on real tokenizers

Recorded because the pattern matters more than any of them: each was internally consistent,
each passed a stub-based test suite, and each died on first contact with a real tokenizer.

1. **The terminator is `eos_token_id`.** Wrong token on Phi (§3).
2. **Tokenize the message content on its own and match it.** `mistral_common` strips trailing
   whitespace before framing, so `"...Yosemite National Park "` assembles with a space token
   (id 1032) the template never emits. 28 of 3 000 conversations, plus 2 whose assistant content
   is the empty string.
3. **Append a synthetic user turn to any prefix to make it renderable.** `mistral_common`
   **merges adjacent same-role messages** into one turn joined by `\n\n`, so appending a user
   turn to a prefix already ending in one produces a single merged turn and no frame at all.

Design 3 is the instructive one: **it passed a 3 000-row dry run at a 95 % success rate.** The
merge separator and the turn opener are each exactly one token, so the two errors cancelled to
zero and the arithmetic came out right for the wrong reason. A 95 % pass rate on a design that is
wrong is a far more dangerous result than a 0 % one.

The methodological correction, which is the point of this section: after two designs passed unit
tests and failed on the box, the third was prototyped **on the box against the real tokenizers
first** and only then ported into the module. It returned 196/200 on both tokenizers on its first
run.

## 5. Which mode, and who decides

Nothing in a tokenizer's config says which mode it needs. `--mask-mode auto` (the default)
measures: it masks a 32-row sample under both modes and takes whichever kept more, ties going to
`template` — the mode that asks the template itself.

| model | probe (of 32) | chosen |
|---|---|---|
| Phi-4-mini | template **30**, assemble **30** | `template` (tie) |
| Ministral-8B | template **0**, assemble **30** | `assemble` |

Phi's tie is not a coincidence and it is the cross-check: Phi is the one model **both** modes
handle, so `assemble` can be validated against a mode that reads spans out of the template's own
output.

**On 500 Tulu-3 rows, all 495 that both modes mask come back token-identical** — identical ids,
identical span starts, identical span ends on every span but the last. The single difference:
`template` mode's sequence carries a trailing `<|endoftext|>`, Phi's document terminator, inside
its final span (~0.2 % of supervised tokens). That is a difference of **scope, not alignment**.

It is left alone rather than reconciled. It is what each template says a *finished* conversation
is — Ministral has no equivalent token — and forcing either into the other's shape would train a
model on a frame it is not served under. This is also why the tie breaks toward `template`: each
model stays in its own template's native shape.

## 6. The dry runs

3 000 rows of `allenai/tulu-3-sft-mixture`, `--max-len 2048`.

| | Phi-4-mini | Ministral-8B |
|---|---:|---:|
| mode | `template` | `assemble` |
| kept | 2 890 | 2 875 |
| **dropped** | **3.67 %** | **4.17 %** |
| — over `--max-len` | 3.67 % (110) | 4.10 % (123) |
| — **unmaskable** | **0.00 %** | **0.07 %** (2) |
| tokens | 1 577 261 | 1 622 149 |
| supervised | 1 115 924 (70.8 %) | 1 143 701 (70.5 %) |

Ministral's 2 unmaskable rows are `InvalidAssistantMessageException` on empty assistant content —
a correct refusal by the tokenizer, and a conversation with nothing to supervise.

**The two drop classes have separate ceilings because they are different kinds of fact.**
"Too long" is a *budget*: a known, tunable function of `--max-len`, governed by
`--max-length-drop-rate` (0.15). Everything else is a *broken assumption about the tokenizer*,
governed by `--max-drop-rate` (0.05). Collapsing them into one number would let a template
defect hide behind a sequence-length setting. Both models are under both ceilings, and the
0.00 % / 0.07 % column is the one that licenses the run.

That the two models land within 0.5 points of each other on drop rate and within 0.3 points on
supervised fraction — through completely different tokenizers and two different masking
algorithms — is the strongest available evidence that both are measuring the same thing.

## 7. What this does not establish

* Nothing here says the fine-tune will help. It says the loss will be applied to the right
  tokens. S2's own gate is the eval delta against the S1 headroom numbers.
* The panel is still **two models, not four** — `meta-llama/Llama-3.1-8B-Instruct` and
  `google/gemma-3-4b-it` are `gated=manual` and remain blocked on a Hub token
  ([S1 §1](phase3-s1-headroom-screen.md)). Both would need their own probe; neither mode is
  assumed to generalize.
* The cross-check in §5 is available *only* on Phi. On Ministral, `assemble` is verified by its
  own internal consistency check — every turn's assembled prefix re-verified against the
  template's next render — not against a second independent implementation.
* 3 000 rows is a dry run. The full S2 mixture is larger, and the census is written on every
  real run for exactly this reason.

## 8. Verification

28 unit tests in [`tests/test_run_s2_finetune.py`](../../tests/test_run_s2_finetune.py), each
answering "what future diff turns this red?". The ones that carry this report's findings:

| test | what it pins |
|---|---|
| `test_only_assistant_content_is_supervised` | every label is `-100` or the id at its own index — the shift this design exists to prevent |
| `test_a_template_that_is_not_prefix_stable_is_dropped_not_mis_masked` | drift is dropped, never supervised at an offset |
| `test_drift_larger_than_a_terminator_is_still_rejected` | the terminator allowance is not a hole big enough to hide drift in |
| `test_assemble_mode_reproduces_what_the_template_would_have_rendered` | the two modes agree exactly where nothing intervenes |
| `test_the_two_modes_differ_only_by_the_document_terminator` | §5's cross-check, as a stub |
| `test_the_supervised_span_ends_where_the_model_has_to_stop` | the terminator is supervised on both tokenizer shapes |
| `test_a_terminator_that_cannot_be_measured_is_refused_not_guessed` | §3 never falls back to a guess |
| `test_a_wrong_eos_token_does_not_reach_the_supervised_span` | dead hypothesis 1 stays dead |
| `test_assemble_mode_masks_what_the_template_will_not_render` | the refusal *and* the same-role merge, i.e. dead hypothesis 3 |
| `test_the_mode_is_measured_against_the_tokenizer_not_read_off_it` | `auto` probes; it does not consult an attribute |

The `_MistralShaped` stub models all three real behaviours — same-role merge, assistant-final
refusal, open continuation — so the failures above are reproducible on CPU in milliseconds
without the 8 B checkpoint.
