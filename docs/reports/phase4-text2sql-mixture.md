# Phase 4 · the text-to-SQL mixture — building an evaluation that can be wrong

**Measured 2026-08-08.** Three corpora, mixed, screened before any GPU time is spent. Raw
records: [`experiments/phase4/out/`](../../experiments/phase4/out/). Screen driver:
[`screen_text2sql.py`](../../experiments/phase4/screen_text2sql.py). Loader and scorer:
[`text2sql.py`](../../packages/dynquant-core/src/dynquant/eval/text2sql.py),
[`text2sql_sources.py`](../../packages/dynquant-core/src/dynquant/eval/text2sql_sources.py).

Phase 4 quantizes `LiquidAI/LFM2.5-8B-A1B` six ways — GPTQ 4b/3b, AWQ 4b/3b, DynQuant 4b/3b —
against a bf16 ceiling, on text-to-SQL. This report is about the part that comes first and has
no GPU in it: **the benchmark**. Seven arms compared on a task that cannot distinguish a
correct answer from a broken one would produce a full results table and no information, and
every defect below was found by screening rather than by a number that looked wrong.

This is not a results report. Nothing here is a claim about LFM2.5 or about DynQuant.

---

## 1. Why three corpora, mixed

One text-to-SQL corpus measures one corpus's idea of SQL. The three chosen disagree in ways
that matter:

| source | schema | rows | what it contributes |
|---|---|---|---|
| `gretelai/synthetic_text_to_sql` | synthetic, 100 verticals | shipped as `INSERT`s in the context | joins, CTEs, window functions, set operations |
| `Salesforce/wikisql` | real Wikipedia tables | synthesised from table content | single-table, real-world column captions with spaces and punctuation |
| `b-mc2/sql-create-context` | bare `CREATE TABLE` | **none** | schema-grounding at scale; trains, cannot score |

The mixture is **balanced per source and round-robin interleaved**, not concatenated. A prefix
of a concatenation makes the headline an average weighted by whichever source survives its own
admission filter most often — which here would have been WikiSQL by a wide margin, and the
headline would have quietly become "single-table WikiSQL accuracy" while naming three corpora.
Pinned by
`test_the_mixture_is_interleaved_so_a_truncated_run_still_sees_every_source`.

`create-context` is admitted for **training only**. Its schemas hold no rows, so a query and a
query that returns nothing are the same observation — it can teach the mapping and cannot score
it. `resolve_sources` refuses it for `test` rather than leaving that to the caller.

---

## 2. The metric, and the three ways it lies

Scoring is **execution accuracy**: the schema is materialised in memory per item, gold and
prediction both run, result sets compared. String comparison would punish a correct query for
its formatting; execution does not.

Execution has one failure mode, and it has three shapes. **Two queries that both return nothing
compare equal.** An evaluation set made of items whose gold returns nothing scores `SELECT 0`
near the ceiling, and the table looks like a working benchmark.

| shape | what it looks like | why the obvious test misses it |
|---|---|---|
| (a) no rows in the database | `SELECT COUNT(*)` over an empty schema → `[(0,)]` | returns a row, so "did it return rows?" passes |
| (b) gold matches nothing | `SELECT AVG(x) WHERE <no match>` → `[(None,)]` | same |
| (c) the gold is not a query | `UPDATE t SET …` → no rows | same, and see §4 |

Admission therefore requires all of: **the database holds rows**, **the gold finds some**, and
**the result is not a single all-NULL/all-zero row**. The last clause is the one that is easy
to get wrong in the safe direction — `(0, 5)` is a legitimate answer and is not degenerate;
only an entire row of nulls and zeros is. Parametrised over both cases in
`test_eval_text2sql_sources.py`.

**Admission is asymmetric between splits, deliberately.** A gold that returns nothing is still
correct supervision, so `require_rows` is off for `train` and on for `test`. The row filter is
a property of what can be *scored*, not of what can be *learned*.

---

## 3. Two defects in SQLite semantics, found by screening

**`COLLATE NOCASE` on WikiSQL text columns.** WikiSQL's condition values are the annotator's
typing (`terrence ross`); the cells are Wikipedia's (`Terrence Ross`). SQLite `=` on `TEXT` is
case-sensitive, so **a third of WikiSQL golds matched nothing** and were being refused as
"the gold finds no rows" — a correct refusal for an incorrect reason, discarding a third of the
corpus. Declaring the collation on synthesised text columns took that from 33 % to 0.4 %.

**Type affinity on quoted numerics.** Condition literals are single-quoted unconditionally,
including against `REAL` columns: affinity applies to the comparison, so `"Year" = '2011'`
still matches `2011.0`, while unconditional quoting stops `007` and `1-2` from being reinterpreted
as numbers that are no longer the value the question asked about.

Neither of these is visible in an accuracy number. Both are visible in an admission rate.

---

## 4. The DML leak — a train/eval format mismatch, caught before the fine-tune

Gretel is a **SQL corpus, not a query corpus**: `UPDATE`, `INSERT`, `DELETE` and `CREATE`
golds are **10.2 % of its test rows and 11.3 % of its train rows** (measured, 2 000 shuffled
rows per split).

These were already absent from the evaluation set — but only as a side effect. A DML statement
runs cleanly and returns no rows, so the row filter caught it and filed it under
`empty_result`, "the gold matched nothing", which is a wrong diagnosis for a statement that was
never going to match anything.

**In training there is no row filter, so they survived.** `extract_sql` reads an answer out of
a generation by cutting at `SELECT` or `WITH`. A model taught on 11 % `UPDATE` answers emits
text the scorer cannot read at all — scored `unparseable`, on a metric whose floor is zero.
That is a fine-tune damaging the arm it is measured by, in a way no arm of the comparison could
have revealed, because **all seven arms would have shared it**.

Closed by `is_readable_query`, applied before execution and in *every* split, with its own
`not_a_query` tally so the refusal is labelled as what it is. The regex is anchored past
comments and whitespace, because an unanchored search accepts
`UPDATE t SET a = (SELECT max(b) FROM u)` on the strength of its subquery.

The screen was re-run afterwards. Admitted counts on the evaluation split did not move — those
rows were already excluded — and **Gretel's training admission fell from ~78 % to 68.8 %**,
which is the leak, measured.

---

## 5. Measured admission

2 000 shuffled rows per source per split. Shuffled under a fixed seed rather than taken as a
prefix: HF splits arrive label-sorted, and a prefix samples one region of the corpus.

**Evaluation split** (`require_rows=True`) — 2 796 admitted across the mixture:

| | gretel | wikisql |
|---|---:|---:|
| **admitted** | **1 026 (51.3 %)** | **1 770 (88.5 %)** |
| would not execute | 413 (20.6 %) | 0 |
| not a query (DML) | 204 (10.2 %) | 0 |
| schema holds no rows | 287 (14.3 %) | 0 |
| gold matched nothing | 43 (2.1 %) | 2 (0.1 %) |
| all-null/zero answer | 27 (1.4 %) | 218 (10.9 %) |
| over the 3 072-char cap | 0 | 10 (0.5 %) |

**Training split** (`require_rows=False`) — 5 326 admitted:

| | gretel | wikisql | create-context |
|---|---:|---:|---:|
| **admitted** | **1 376 (68.8 %)** | **1 988 (99.4 %)** | **1 962 (98.1 %)** |
| would not execute | 398 (19.9 %) | 0 | 38 (1.9 %) |
| not a query (DML) | 226 (11.3 %) | 0 | 0 |
| over the char cap | 0 | 12 (0.6 %) | 0 |

**Gretel's 20 % execution-failure rate is a property of the corpus, not of the harness**:
362 of 413 are `OperationalError` — the shipped context does not define everything the shipped
gold selects from. There is no repair for that from here; the rows are refused and counted.

Shape of what survives, evaluation split: Gretel contributes 531 basic, 249 aggregation,
135 single-join, 44 subquery, 43 window-function, 17 multi-join, 7 set-operation queries;
WikiSQL contributes 1 433 plain selects and 337 aggregates. The mixture is not uniformly easy
and it is not uniformly hard.

---

## 6. Three properties the harness holds by construction

**One evaluator.** The base model's headroom is measured through the registered CLI task —
`dynquant eval <model> --task text2sql --limit 400 --prompt-style chat` — the same path the six
quantized arms take. The screen deliberately does *not* implement its own; two evaluators can
disagree, and the one that would be wrong is the one nobody re-checks.

**One admission rule.** The screen imports the loader's `admit` rather than re-deriving it. A
screen with its own copy drifts from the loader the first time either changes, and is then a
verdict about a dataset nobody evaluates.

**One instruction string.** `instruction()` is called by the chat evaluation, the training-text
builder, and the SFT driver's row assembler. A model trained on one phrasing and asked in
another is being measured on the gap between them, and the gap is invisible in the output
because both halves look correct alone.

---

## 7. A defect in the harness itself

`text2sql` shipped with a loader, a scorer, a registry entry in `TASKS` and two test files —
**and could not be run.** `dynquant eval --task` carried a hand-written tuple of the other six
tasks, so argparse refused it with a usage error naming those six, which reads as a typo rather
than as the omission it was.

The choices are now derived from the registry, so a registered task cannot be unreachable, and
`test_every_registered_eval_task_is_reachable_from_the_command_line` turns red if the literal
list comes back.

Fixing it exposed a second one. `_TaskSpec` has separate `executes_code` and `takes_style`
capabilities, and a test asserted the `style` argument against `executes_code` — sound only
because every style-taking task so far also ran code. `text2sql` takes a prompt framing and
executes nothing, and is the first task to separate them. The production code was already
correct; the test was passing on a coincidence.

Both are the same shape as the defect in §4: **a check that was right for the wrong reason, and
stayed right until the first case that told the two reasons apart.**

---

---

## 8. The headroom screen found three more, and one of them was worth 34 points

The screen from §6 ran. It reported **5.50%** — 22 of 400, with 370 items marked "would not
run" and 3 as "no query". A number that low is either a model that cannot do the task or a
harness that cannot see the answer, and those two have very different consequences: the first
ends the campaign, the second ends the measurement.

It was the second, and the reason is a property of the model that nothing in the pipeline knew
about.

### LFM2.5-8B-A1B is a reasoning model, and there is no switch

`chat_template.jinja` carries `preserve_thinking | default(false)`, which looks like an
off-switch and is not one. All it governs is whether *prior* assistant turns keep their trace,
implemented as `content.split("</think>")[-1]`. For the turn being generated,
`add_generation_prompt` emits `<|im_start|>assistant\n` and stops — the model opens `<think>`
itself, on its own initiative, and no template flag reaches that decision. (The template lives
in a separate `.jinja` file; `tokenizer_config.json` has no `chat_template` key at all, which is
where transformers 5.x moved it.)

So the honest configuration is to let the model reason and read what comes after. Every one of
32 probe generations opened a trace.

### The extractor was reading the deliberation

`extract_sql` falls back to scanning for the first `SELECT`. Inside a reasoning trace, the first
`SELECT` is not an answer — it is a **candidate**, and a trace is precisely where the model
argues against its candidates. One probe generation contained:

```
SELECT Model number FROM ...;   But note that the column name is ...
```

The extractor took that query. The model had already rejected it two words later.

Measured on the 32 stored generations, with a control arm first:

| Arm | Correct | Would not run | Ran, wrong rows | No query |
|---|---|---|---|---|
| Rescored with the shipped extractor (control) | 2 | 30 | 0 | 0 |
| What the run itself recorded | 2 | 30 | 0 | 0 |
| Rescored with the trace cut at `</think>` | **13** | 5 | 1 | 13 |

**6.2% → 40.6%**, and the control is the load-bearing row. Rescoring offline reloads the items
from the loader; if that reload had produced a different item order than the generations were
made against, the second arm would be a comparison between two datasets rather than two
extractors. Requiring the control to reproduce the run's own tally *before* the delta is allowed
to mean anything is what makes the 34 points attributable.

The 13 "no query" are not failures of extraction. They are generations that never closed
`<think>` inside 256 new tokens — the model was still thinking when the budget ran out. So the
ceiling at that budget was 19/32 = 59.4%, and conditional on having answered at all the model
was right 13/19 = **68%**.

### The fix, and why it is safe to apply everywhere

`strip_reasoning` in `dynquant.eval.harness`, wired into all five extractors. Three cases:

- **No trace** — returned unchanged. This is every non-reasoning model, so adding this to an
  extractor *cannot* move a number already collected. That is a property, not a hope, and
  `test_stripping_reasoning_cannot_move_a_number_already_collected` is what holds it.
- **A closed trace** — everything after the **last** close tag. Last, not first, matching what
  the model's own template does when it strips a prior turn. The convention is the model's, not
  this package's.
- **An unclosed trace** — the empty string. The model never answered, so there is no answer
  region and the caller counts the item unparseable. Returning the trace instead would score the
  model on a query it had not finished arguing with, and would report *a decode budget* as a
  wrong answer — which in a quantization campaign reads as quantization damage.

The last case is the one that matters for what this report is for. Six arms will be compared
against each other on this metric. A too-small token budget that silently becomes "wrong answer"
is a bias with no error message, and it lands on whichever arm happens to be most verbose.

### Two more, both invisible to any A/B

**The chat prompt was not the prompt the record described.** `build_prompt`'s chat branch
discarded `shots` entirely, while the CLI resolved a two-shot pool, passed it, and wrote
`"shots": 2` into the result JSON *and* the run manifest. So a chat run recorded a two-shot
prompt and sent a zero-shot one. Fixing it moved the median prompt from 301 to 1228 tokens —
a factor of four that nobody would have noticed from the outputs.

This is a provenance error, not an accuracy one, and it is the kind that no comparison in the
campaign could surface: **every arm shared it**, so it cancels in every difference and the
record outlives the run. A stale `_log.info` claiming "chat framing ignores the few-shot
exemplars" was removed with it — it had become a lie about the code it sat in.

MBPP's chat branch still drops its shots. It is recorded here rather than fixed: correcting it
would invalidate MBPP arms already collected, and unlike text2sql it at least says so in the
log.

**`format_training_text` was exported, untested, and unusable.** It returned a chat-rendered
string where the other three tasks return `(prompt, completion)`. Two things were wrong. It was
the only training path in the package going through `apply_chat_template(..., tokenize=False)`
— a round trip transformers itself calls unsafe, and one that cost 120 of 164 HumanEval problems
on Ministral earlier in this campaign. And the signature did not match the `TaskSpec` protocol
the four-point experiment holds tasks in, so nothing could call it. It surfaced as a one-line
mypy complaint about returning `Any`.

All three are the same shape as §4 and §7: **not wrong answers, but wrong records** — and a
wrong record is worse, because a wrong answer eventually contradicts something.

### What this cost, and what it saved

It cost one 400-item run and a 32-item probe. It saved a fine-tune and six quantization arms
measured against a benchmark on which the base model appeared to score 5.50%, where the honest
figure is an order of magnitude higher — every subsequent comparison would have been made in the
noise floor of a broken extractor, and "quantization is harmless here" is exactly what that
looks like.

This is the rule from the GSM8K campaign holding a second time: **measure the base model before
spending a fine-tune.** The first time it caught a dataset with no headroom. This time it caught
an evaluator that could not read the model's output — and the same screen catches both.

---

## 9. Signal coverage on this architecture: 11.6% → 100%

Orthogonal to the eval work, and specific to this model being an MoE.

DynQuant's graph classifies **100%** of LFM2.5-8B-A1B's parameters. Signal *collection* reached
11.6% of them. An expert bank here is one module holding two 3-D parameters
(`gate_up_proj` `[E, 2I, H]`, `down_proj` `[E, H, I]`) rather than a `ModuleList` of `nn.Linear`,
and the tracker hooks modules. So all 44 banks were refused: **0.979 B of 8.468 B measured, 7.751 B
listed UNMEASURED** and allocated on a floor — 88.4% of the model quantized without a signal, in
a campaign whose entire claim is that the signal decides the allocation.

The fix needs no architecture-specific code, because the bank's own boundary settles the
assignment. Exactly two activations cross it and each weight owns exactly one:
`gate_up_proj` ← the bank input, `down_proj` ← the bank output. The other two activations are
locals of the bank's forward and never cross anything, so there is nothing to attribute and
nothing to guess. Ranking is within role and the two roles are distinct, so the two weights are
never priced against each other on saliency drawn from different tensors.

One hazard is specific to this path and would have been invisible. Bank parameters sit in no
optimizer, so `optimizer.zero_grad()` never reaches them, and a gradient left attached
accumulates across steps. Welford would then read a monotonically growing norm as *plasticity* —
and the arm would look like it worked. Collection releases the gradient itself; the test runs two
identical steps and asserts the variance is zero, which is the assertion that fails when the
release is removed.

Eleven cases, each checked by mutating the thing it covers: side swap, hook duplication, all
three estimator branches, the grad release, and detach leaving `requires_grad` or `grad` behind.
Two of them were initially inert and passed against their own mutant — both caught, both fixed,
both noted in the test file.

---

## 10. The baselines cannot see this architecture either

The same blindness that held signal collection to 11.6% (§9) applies to GPTQ and AWQ, and it
does more damage there, because their failure mode produces a checkpoint rather than a warning.

Measured by instantiating the model on the `meta` device — no weights, no GPU, so this is a
structural fact and cost nothing to establish:

| What | Parameters | Share |
|---|---|---|
| Total | 8 467 856 128 | 100% |
| Reachable as `nn.Linear` | 716 570 624 | **8.5%** |
| Held in batched expert banks | 7 751 073 792 | **91.5%** |

`llmcompressor`'s GPTQ and AWQ modifiers walk `nn.Linear` modules. On this model that is 8.5% of
the weights. Pointed at it unmodified, a "GPTQ 4-bit" run would quantize 8.5% of the checkpoint,
leave the other 91.5% in bfloat16, and **succeed** — emitting a directory labelled 4-bit that
weighs about 15.5 GB against bf16's 16.9 GB.

That number is the whole problem. It is not a wrong accuracy, it is a wrong *denominator*, and it
biases the comparison in DynQuant's favour: a DynQuant 4-bit arm at ~4.5 GB would appear to beat a
"GPTQ 4-bit" arm three times its size, and the natural reading of that table is that DynQuant won.
What actually happened is that GPTQ never ran on 91.5% of the model.

This is the third time this campaign has produced a baseline whose bytes were quietly wrong, and
the second with the same signature: `ignore=["lm_head"]` on a tied-embedding model made a "4-bit"
GPTQ checkpoint measure 7.36 bits. Both were caught by measuring bytes on disk rather than reading
the label on the config. The rule that catches this class is: **an arm's size is a measurement,
never a setting** — if a comparison quotes a bit width that was requested rather than weighed, it
is not yet a comparison.

### What a fair baseline requires, and why it turned out to be one line

Each bank is one `Lfm2MoeExperts` module holding `gate_up_proj [32, 3584, 2048]` and
`down_proj [32, 2048, 1792]`; 22 of the 24 layers carry one (`num_dense_layers: 2`). Giving GPTQ
and AWQ the whole model means materialising each bank as per-expert `nn.Linear` modules.

`llmcompressor` already does this — `modeling.moe.linearize.linearize_moe`, with
`load_quantizable_moe` as the loading context manager. It has registered *load* mappings for only
`deepseek_v4` and `qwen2_moe`, so `lfm2_moe` misses the memory-efficient path, but it documents a
fallback for exactly that case: load the 3-D checkpoint normally, then convert in place. Verified,
not assumed — `get_non_linearized_moes` finds **all 22** banks on this model, and the conversion
produces 96 `nn.Linear` per bank (32 experts × gate/up/down), 2112 in total.

One thing blocked it, and it is the second half of §9's story. `MoEConfig.from_config` reads the
activation from `config.hidden_act`, `hidden_activation`, or `mlp_hidden_act`; `lfm2_moe` carries
none of the three and the load dies with `AttributeError`. The fix is `config.hidden_act = "silu"`,
and that is **not** a guess: `Lfm2MoeExperts.__init__` hard-codes `self.act_fn = F.silu`. The value
is read off the model's own source rather than inferred from the family.

### The swap is bit-exact, and that had to be checked rather than argued

The GPTQ and AWQ arms can only be quantized through the linearized structure, so every number they
produce is measured on a module layout the DynQuant arms never use. If the swap perturbed the
arithmetic, the comparison would acquire a second variable and the bf16 ceiling would stop being a
shared reference point.

It does not. The reference forward already loops per expert and calls
`F.linear(current_state, self.gate_up_proj[expert_idx])` — indexing the first dimension of a
contiguous tensor yields contiguous memory, so replacing that slice with a `Linear`'s weight is the
same op on the same layout. Measured on a full-size bank with all 32 experts exercised:

| dtype | bit-identical | max abs delta |
|---|---|---|
| `float32` | yes | 0.0 |
| `bfloat16` | yes | 0.0 |

Bit-identical, not `allclose`. The weaker assertion would have passed over a changed reduction
order, and a reduction order that changes under one arm and not the others is exactly the kind of
difference that shows up later as an unexplained delta with no owner.

### Two consequences of MoE that survive the fix

Both are properties of the architecture rather than of this harness, and both have to be reported
next to the numbers:

- **Each expert sees a fraction of the calibration set.** Routing is 4-of-32, so an expert receives
  roughly an eighth of the tokens. GPTQ's Hessian and AWQ's activation scales are estimated per
  module, so both get an eighth of the statistics per expert that they would get on a dense model
  of the same width. That handicap is theirs, not something introduced here — but a DynQuant win
  that is partly a calibration-sparsity artefact must not be reported as a method advantage.
- **DynQuant is not exposed to it.** Its signal comes from a fine-tune-time hook, which sees every
  token routed to an expert across the whole run rather than a 512-sequence sample. This is the
  first point in the campaign where the fine-tune-time premise pays a dividend a post-hoc method
  cannot match by trying harder, and it should be claimed as exactly that — not as the allocator
  being better.

One more trap is already visible in the config: `tie_word_embeddings: true`. On a tied model,
`ignore=["lm_head"]` is what made a previous "4-bit" GPTQ checkpoint measure 7.36 effective bits,
because the shared tensor gets counted once and quantized never. The six arms must weigh their
bytes on disk, and none of them may ignore the head.

### The driver, and the one seam it needed

`experiments/phase4/baselines_lfm2.py` runs the arms. Four decisions in it are the §10 findings
turned into code, and each one is a refusal rather than a default:

- **It will not run without linearizing.** `get_non_linearized_moes` is called before and after
  `linearize_moe`, and a surviving bank aborts the run. The failure this guards is not a crash: a
  run where linearization silently did nothing is the 8.5% run, and it succeeds.
- **`ignore` is empty.** Not "we did not set it" -- a named module-level constant with the
  tied-embedding reason attached, because `["lm_head"]` is the convention and the convention is
  what produced 7.36 bits last time.
- **The expert mass is charged explicitly.** The accounting walks a `meta`-device reference and
  charges `nn.Linear` weights *and* the 3-D banks by name, rather than inferring the banks from a
  module walk. The walk necessarily runs before linearization; a walk that ran after would need a
  GPU to produce a number this has to give for free. Both contracted dimensions -- 2048 and 1792 --
  are multiples of 128, so groups divide evenly, and a group size that does not divide is a hard
  error rather than an accounting that silently omits padding it does not model.
- **A 3-bit arm cannot be saved.** `compressed-tensors` packs 4 and 8 bits; at any other width
  `save_pretrained(save_compressed=True)` writes dequantized bf16. The directory then reports ~16
  bits per weight for an arm whose arithmetic is 3-bit, which is the wrong-denominator error of this
  section one level further out -- so 3-bit arms are scored in memory and never written.

Run against the real config, the accounting closes exactly, which is the property worth having:
716 570 624 + 7 751 073 792 = 8 467 644 416 quantized, leaving 211 712 parameters -- the norms --
at fp16, 0.01% of the bits. The matched-byte targets every arm is held to:

| Width | Accounted bits | Bytes | vs bf16 |
|---|---|---|---|
| bf16 ceiling | 16 | 16 936 006 912 | — |
| 4-bit g128 | 4.157 | 4 399 629 312 | 3.85× |
| 3-bit g128 | 3.149 | 3 332 904 576 | 5.08× |

Scoring those arms needed one change outside the experiment. `dynquant eval` takes a path, and a
3-bit arm has none for the reason just given. `evaluate.run` now takes an optional in-memory model.
It is a parameter on the command rather than a second evaluator beside it because everything below
that line is what makes two numbers comparable -- the prompt, the shot prefix and its seed, the
decode settings, the scorer, the per-item hit vector, and the `PAIRING_FIELDS` guard a McNemar test
checks before it agrees to pair two records. `experiments/four_point` has its own `run_eval`, and
the cost of that is exactly this: its records cannot be paired against a `dynquant eval` record
without arguing that two implementations of the same settings agree. Passing a model *and* `--map`
is refused, because `--map` quantizes the model it is given and either behaviour on an
already-quantized one reports a doubly-quantized model under one method's name.

One llmcompressor fact is now in `experiments/_llmc.py` rather than copied per experiment, because
it is the kind of finding a drifting copy un-finds: `oneshot` fits scales and leaves the weight
tensor alone for every recipe except GPTQ. An in-process arm that skips the materialization step
scores bf16 weights with unused scales bolted on. That was found once by RTN returning
byte-identical predictions to bf16 -- and the dangerous case is AWQ, whose transform *does* rewrite
weights, so an unrounded AWQ arm produces plausibly-different numbers rather than suspiciously
identical ones.

The decode budget has no default in this driver. It is read off the ceiling run's own closure
distribution, and a default here would be a second guess at the number that measurement exists to
replace -- which is how §8's 5.50% happened.

---

## 11. Status

Done: the mixture, the admission rule, the metric, the screen, both splits measured, the DML leak
closed, the task reachable from the CLI, the reasoning-trace and shots defects fixed, and signal
collection reaching every parameter of this architecture.

The 5.50% headroom figure in §8 is **withdrawn** — it measured the extractor, not the model. It
is kept in this report because the diagnosis is the finding, and deleting the number would delete
the evidence that a campaign can be configured off one.

Also done since: the six-arm driver, the in-process eval seam it needed, and the accounting run
against the real config -- 8.46% / 91.54%, closing to the parameter, with matched-byte targets of
4 399 629 312 B at 4 bits and 3 332 904 576 B at 3 bits. What that buys is that every arm's size is
now a number this campaign computed from the checkpoint's own module tree, not a width someone
typed into a recipe.

Not done, in order: headroom re-measured with the fixes, at a budget generous enough that the
closure distribution comes out uncensored and the arms' budget can be *read* rather than guessed
(§8's 256 was censored -- closures were still arriving at the cap); `load_linearized` exercised on
the GPU, because the 22-banks-to-zero assertion is the one part of the driver that CPU tests cannot
reach; the fine-tune, with the expert mass measured; then the six arms at matched bytes -- GPTQ and
AWQ through the linearized structure verified bit-exact in §10, all three widths weighed rather
than requested; then the paired comparison over stored per-item hits.

§10 was written expecting linearization to be the largest remaining unknown in the phase. It is
not: llmcompressor already ships it, it detects all 22 banks here, and the swap is bit-identical.
What the section is worth keeping for is the 8.5% measurement -- had the arms been run without it,
they would have succeeded.

One thing already fixed on the strength of §10 and §9 together: the S2 driver could not turn expert
measurement on, and its default is off. It is now a tri-state flag whose *unset* value is an error
-- but only on a model that has banks, so the four dense panel models keep their existing commands.
Defaulting either way was wrong. Off writes a stats file with 88.4% of the checkpoint UNMEASURED
and allocated on role floors, which is a run that finishes and produces numbers; on silently
commits a gradient buffer for the whole expert mass, ~15.5 GB in bfloat16 here. A campaign whose
claim is that the signal decides the allocation cannot have the answer to "was the signal
collected?" depend on a default nobody typed.
