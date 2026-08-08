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

### The honest number, and where the arms' decode budget comes from

Re-run with all three fixes in and the cap raised to 1024 tokens, on the same 400 problems:

| | budget | correct | unparseable | wall |
|---|---|---|---|---|
| withdrawn | 256 | 22 / 400 = **5.50%** | — | — |
| measured | 1024 | 231 / 400 = **57.75%** | 10 | 4304.5 s |

Split by source: gretel 133/200, wikisql 98/200. Of the 169 misses, 85 are execution errors
against the schema and 49 are queries that ran and returned the wrong rows — which is the
distribution a model with real but imperfect SQL has, not the distribution of a broken
extractor.

The point of raising the cap was to read the arms' budget off the closure distribution rather
than guess it, and that read came back **censored**: 10 of 400 generations (2.5%) were still
deliberating when they hit 1024. Over the 390 that closed, total length is p50 277, p90 608,
p95 778, p99 1024; the deliberation prefix alone is p95 666 with a max of 965, and the answer
after it is p95 49, max 107. So the reasoning is what consumes the budget, and 256 never had a
chance — the median generation is longer than the entire cap that produced 5.50%.

Two decisions follow, and only one of them is the obvious one.

**The 57.75% stands as a headroom judgement even though it is censored.** Censoring is
one-directional here: the 10 unclosed generations can only ever be scored correct, never
incorrect-because-truncated-differently. So the true value is in [57.75%, 60.25%], and with 40
points of room to the ceiling that interval does not change any decision the screen exists to
make. Re-running at 2048 to close it would cost another ~2.4 h and buy a tighter bound on a
number nothing depends on.

**But this distribution is not the one that should set the arms' budget.** The six arms
quantize the *fine-tuned* model, and fine-tuning on text-to-SQL teaches direct answering — the
whole point of the mixture is that the target is a query, not a derivation. A base model's
deliberation length does not predict a fine-tuned model's. Reading 1280 off this table and
stamping it on the arms would be transferring a measurement across the one intervention
designed to invalidate it. The budget comes from the fine-tuned bf16 ceiling instead, measured
the same way; and if that run at 1024 comes back uncensored, the ceiling record *is* the
ceiling arm — same budget, so it pairs, at no extra cost.

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

### Measured on the real checkpoint: 8.5% → 100.0%

Run against `/workspace/models/LFM2.5-8B-A1B`, on CPU, so it did not have to queue behind the GPU:

| | before | after |
|---|---|---|
| Batched banks detected | 22 | **0** |
| `nn.Linear` modules | 89 | 2201 |
| Parameters reachable as `nn.Linear` | 716 570 624 | **8 467 644 416** |
| Share of the checkpoint | 8.46% | **100.00%** |

The parameter row is the one that matters. A module count says conversion happened; it is equally
consistent with a mapping that reached most banks and skipped the widest one. And 8 467 644 416 is
the same figure the `meta`-device accounting reached from the other direction -- an `nn.Linear` walk
over the *unconverted* tree plus an explicit charge for the 3-D banks. Two independent paths, one
number, so neither is resting on the other.

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

## 11. The contamination check that could not fire — 189 of 200

The S2 dry run of §10 wrote a census, and one line of it read
`"sources_overlapping_an_eval_task": []`. That is the driver's contamination guard, it has
been there since phase 3, and on this mixture it is worth nothing:

```python
_CONTAMINATING = ("gsm8k", "humaneval", "mbpp")
```

Those are phase 3's markers. No SQL corpus name contains any of them, so the guard could not
have named a source whatever the data held. **An empty result from a check that cannot fire
reads exactly like a mixture that passed** — the same shape as §4 and §7, and the same shape
as a determinism guard whose stated reason for firing is a hypothesis rather than a diagnosis.

The specific worry was not hypothetical. Evaluation draws `test` from Gretel and WikiSQL;
training draws `train` from those two **and** from `b-mc2/sql-create-context`, which ships a
single `train` split and is a community aggregate assembled from WikiSQL and Spider. If it was
built from all of WikiSQL rather than WikiSQL's training half, then evaluation questions are
inside the training mixture.

### What that would and would not invalidate, written down before the number arrived

So the number could not decide it retroactively:

* **The A/B stays valid.** Every arm — bf16 ceiling, GPTQ ×2, AWQ ×2, DynQuant ×2 — quantizes
  the *same* fine-tuned model. Contamination inflates all seven equally, and the paired test
  measures quantization damage regardless.
* **The absolute accuracy stops being a claim about text-to-SQL.** "The fine-tune reached X%"
  becomes "the fine-tune reached X%, of which some is recall."
* **And the headroom argument weakens**, because §8's 57.75% was measured on the base model
  with none of this, and the fine-tuned number would be measured with it. The gain between
  them would not be all learning.

### Measured

`experiments/phase4/leak_text2sql.py`, against the 400-item evaluation set (200 Gretel, 200
WikiSQL) and each training pool **in full** — the pool rather than the sample, because a
sample is what one seed draws and a pool overlap is what any seed can draw:

| training pool | rows | distinct questions | Gretel eval items present | WikiSQL eval items present |
|---|---:|---:|---:|---:|
| `gretelai/synthetic_text_to_sql` train | 100 000 | 99 805 | 0 / 200 | 0 / 200 |
| `Salesforce/wikisql` train | 56 355 | 56 105 | 0 / 200 | 1 / 200 |
| `b-mc2/sql-create-context` | 78 577 | 78 251 | 0 / 200 | **189 / 200** |

**94.5% of the WikiSQL evaluation half is in the aggregate.** Gretel is clean in both
directions. WikiSQL's own train split contributes one item — a question that appears on both
sides of its own boundary under a fold that ignores case and punctuation, which is either a
duplicate in the corpus or two questions about different tables that read identically. It is
dropped either way; that is the conservative direction and it costs one row.

In the 50 000-row mixture this run would actually have sampled at seed 0, **38 of the 200
WikiSQL items are present** — one fifth of half the benchmark, in the training set, unremarked.

### The scan's own false clean

The first version of the sampled arm reported **0 / 200** directly beneath a pool scan
reporting 189 / 200, and the two are not in conflict. It walked the driver's rendered
conversations and compared the *user turn* against the evaluation questions — but a user turn
is `instruction(item)`, the question wrapped in a schema and a directive, so the equality could
never hold. A clean sample was being reported by a comparison that had no way to come out any
other way.

That is the same failure as `_CONTAMINATING`, committed inside the tool written to expose it,
and it is recorded in that function's docstring rather than quietly fixed.

### Three decisions in the filter

**By question, not by provenance.** The aggregate does not record which upstream corpus each
row came from, so "drop the WikiSQL-derived rows" is not an operation the data supports.
Matching on the question is — and it is the stronger rule, because it catches an item that
reaches the mixture by any route, including a source not yet added.

**Against the whole test split, not the sampled items.** The banned set is every question in
Gretel's and WikiSQL's complete `test` splits: 21 729 rows, 21 681 distinct keys. Keyed on one
run's `--limit` and seed instead, the filter would be undone by changing either, with nothing
reporting that it had been. This is also why Gretel and WikiSQL lose rows below despite being
clean against the 400 *sampled* items — the 400 are a sample of 21 729.

**Before admission, not after.** A contaminated row must not consume a quota slot or be counted
as kept. Filtering afterwards trains on N−1 while reporting N, and the shortfall reads as an
admission rate.

The fold itself is case, punctuation and whitespace and nothing more. Deliberately not stemming
or similarity matching: this decides whether a training row is an evaluation item, and a looser
rule deletes legitimate training data for *resembling* the test set — a cost paid silently
against a benefit nobody can measure.

### What it removed

Re-running the dry run with the filter in place:

```
decontaminated: dropped 5 gretel rows that ask a question the evaluation asks
decontaminated: dropped 21 wikisql rows that ask a question the evaluation asks
decontaminated: dropped 3990 create-context rows that ask a question the evaluation asks
```

Roughly one create-context row in five of those read. The quotas were still met —
16 667 / 16 667 / 16 666, 49 905 conversations kept after masking, unchanged — so the
decontamination cost this run nothing but the rows it was supposed to cost.

### Both checks are now in the census, and it says what each is worth

```json
"sources_overlapping_an_eval_task": [],
"contamination_markers": ["gsm8k", "humaneval", "mbpp"],
"decontaminated": {"gretel": 5, "wikisql": 21, "create-context": 3990}
```

The empty list is kept, and the marker list is written beside it, because an empty result is
only as strong as the thing that produced it and a later reader cannot see the marker list from
the result. `decontaminated` is the check that can fire, reported per source rather than as a
boolean — a mixture that stops needing the filter and a filter that stops running produce the
same `{}` otherwise, and on this mixture the expected number is four thousand, so zero has to
be readable as suspicious.

The count is returned from `load_rows` beside the rows rather than stashed on the module: a
census reporting a stale number from a previous call would be worse than one reporting none.

---

## 12. Who sets the byte budget, and why it is not DynQuant

The panel's claim is "at matched bytes". Two of the three methods cannot take a size, so the
matching has a direction, and the direction is worth 2.3%.

GPTQ and AWQ take a *width*. The bytes fall out of what `compressed-tensors` writes: the payload
at `bits`, an fp16 scale per group, and a zero point packed at the weight's own width. DynQuant
takes a *size*, and its format writes an fp16 scale **and** an fp16 offset -- 32 bits per group
rather than `16 + bits`. Per parameter at a group of 128:

| | payload | per group | bits/param @4 | bits/param @3 |
|---|---|---|---|---|
| `compressed-tensors` (GPTQ, AWQ) | `bits` | `16 + bits` | 4.15625 | 3.1484 |
| DynQuant (`quant/pack.py:stored_bits`) | `bits`, word-aligned | 32 | 4.25 | 3.25 |
| | | | **+2.26%** | **+3.23%** |

Whole-model, through `baselines_lfm2.accounted_bytes`, that is **4,399,629,312 B** at 4 bits and
**3,332,904,576 B** at 3 -- the numbers in §10, and now the anchors.

Anchoring on DynQuant's uniform arm instead would have handed DynQuant 2.3% more bytes at 4 bits
and 3.2% more at 3 **inside the arm whose accuracy is the claim**. Nothing in the run would have
reported it: each accounting is correct about the format it describes, both would have printed a
tidy "matched" line, and the extra bytes would have arrived as accuracy.

So DynQuant is pinned to the baselines' byte count. `--target-size` accepts a bare integer, so the
pin is exact rather than rounded through a unit, and `dynquant inspect` keys the saved map on the
literal string it was passed -- which is why the allocation and the eval must format that number
the same way, and why a test asserts they do.

The overhead comes out of DynQuant's own payload, and that is the point rather than a concession:
a method that stores more metadata has fewer bits left for weights at the same footprint. That is
a real cost of the format and it should be paid in the comparison, not accounted around.

Charging both by one set of rules would be worse in either direction. By DynQuant's, the baselines
are billed for an offset `compressed-tensors` never writes; by the baselines', DynQuant writes
metadata it never paid for.

The orchestrator is `experiments/phase4/arms_lfm2.py`. It plans the ceiling first -- the only arm
that can fail for a reason unrelated to quantization -- then both widths of each method, reads each
DynQuant arm's realised size back out of the map the allocator wrote rather than from the request,
and refuses any arm more than 0.1% off its anchor. `--target-size` is a ceiling, so the drift that
actually happens is *downward*, which is why the check is on the absolute value: a signed one would
wave through the only failure that occurs.

Every arm is a subprocess of `sys.executable`, and the run refuses to start if llm-compressor is
not importable from it. The box has two environments and only one has it; per-arm interpreters
would score the baselines and the DynQuant arms under two transformers versions, which is a
difference in the measuring instrument reported as a difference between methods.

### `--resume` is where a panel stops being one panel

Seven arms is seven hours, so a crash in arm six must not re-spend arms one through five. But
`--resume` is also the only path by which a record enters the manifest without this run having
produced it, and a leftover record's entire claim to provenance is its filename. Two failures
follow, both of which reach the table rather than the console.

The first: the original reuse branch restored `record` and skipped everything else, so a resumed
arm's `nbytes` stayed `None`. The manifest row then read as an arm that never claimed a size
rather than one whose size stopped being checked -- and because `check_matched` was skipped with
it, a map that had drifted since would pass. Arms are now **weighed on both paths and run on one**:
the DynQuant arms are still read back out of their maps and still checked against the anchor, and
a resumed arm whose map has been deleted is refused rather than reported at its request.

The second is coarser. `eval_flags` makes the seven *commands* identical, which says nothing about
a record this run did not write; an arm kept from a 200-item smoke run resumes into a 400-item
panel as a valid file with a complete `hits` vector, and the McNemar that pairs them compares two
different problem sets. `check_pairable` now reads every record scored so far, after each arm rather
than once at the end -- a reused record from another run is then caught after the arm that exposed
it instead of after six more -- and compares them through
`dynquant.commands.evaluate._comparability` -- the eval command's own
flattener, private and imported anyway, because a second copy of the contract here would go blind
to exactly the field someone later adds to `PAIRING_FIELDS`. Delegating also keeps the check quiet
about everything that legitimately differs per arm: accuracy, runtime, packed size, the batch size
that fit in VRAM. The run stops before the manifest is written, since a manifest listing seven
unpairable arms is worse than no manifest at all -- it is the artefact the comparison reads.

That is as far as this got before the panel was staged, and it is not far enough. `_comparability`
is the right contract for the question *"do these two records describe the same problem set?"* and
it is the whole of what was guarding reuse. Every field in it -- task, backend, split, shots, shot
seed, limit -- answers that question and only that question. None of them names the model, and none
of them can be older than anything. The second half of this section returns to it.

### The one setting two identical commands can still disagree about

Wiring the panel's guard to `_comparability` exposed a hole in the contract itself, and closing
it is the reason this subsection exists rather than a footnote.

Every field in `PAIRING_FIELDS` comes off the command line, so seven arms launched by one driver
cannot differ in any of them. **`prompt_style` does not.** `--prompt-style auto` is the default and
it is answered by the tokenizer: `chat_prompt_style` asks the tokenizer to render a turn and reads
what comes back. A quantized checkpoint whose saved tokenizer lost its chat template therefore
resolves to `completion` where the ceiling resolved to `chat` -- both arms run clean, both commands
are byte-identical, and one of them was asked a different kind of question.

The size of that mistake is already measured in this project. On IFEval the same failure put
Ministral-8B-Instruct at **24.77%** with 195 of 541 generations empty, against Phi-4-mini's 68.76%
with none: low enough to look like a broken model, stable enough to look real. IFEval and the code
tasks record their resolved style for exactly that reason. `text2sql` -- the task this campaign runs
-- did not.

Three changes close it. `Text2SqlResult` gains a required `prompt_style`, set from the *resolved*
value rather than the requested one, because recording `"auto"` would be worse than recording
nothing: it would look like a matched setting. It goes into `as_dict`, which is what `dynquant eval`
writes into the record's `detail` block. And `DETAIL_PAIRING_FIELDS = ("prompt_style",)` reads it
back out of there, the same route `decode.max_new_tokens` takes.

One asymmetry was needed. Every other pairing field is written on every run, so `_compare` treats
its absence from *this* run's record as a bug in the command. The detail block is the task's: three
tasks carry none at all and the rest carry different keys, so absence has to mean "this task reports
no style". `_OPTIONAL_COMPARABILITY` names the exemption explicitly rather than testing for a
`detail.` prefix. Absence still refuses against a record that *has* a style -- a record written
before this change cannot be shown to match one written after it, and "unknown" is not "the same".
That costs a re-run of anything being compared against the older records; the alternative cost is a
wrong number in a table.

### The budget nobody was going to state

The two guards above make the seven records agree with each other. Neither of them makes the
records agree with the plan, and on the decode budget they did not.

§8 ends by deciding that the arms run at the budget the fine-tuned bf16 ceiling is cleared at,
and that 1024 is the number to try. The driver did not say so. `--max-new-tokens` defaulted to
`None`, and `eval_flags` passed the flag only when it was set -- deliberately, on the reasoning
that an inherited default is inherited identically by all seven arms and therefore pairs. That
reasoning is sound and the premise is false: there are two defaults. `dynquant eval` resolves an
unset budget through `_TaskSpec("text2sql").max_new_tokens`, which is **320**; the in-process
`DEFAULT_CHAT_CONFIG` used by direct callers says **384**. The panel routes through the CLI, so
all seven arms would have run at 320 -- consistently, pairably, and 704 tokens under the number
the ceiling was supposed to establish.

What that costs is not a small bias. A query cut mid-clause does not score as a near-miss; it
fails to parse, or parses and errors, so a binding budget is a floor under accuracy rather than a
tax on it. And it binds *unevenly*: the arm most likely to ramble is the most damaged one, which
is the arm the campaign is trying to measure. The result would have read as quantization damage
and been reported as such.

Three changes. The flag now defaults to **1024** in the panel's own parser and is passed
unconditionally, so the number is in the record rather than in a lookup table two packages away.
`check_uncensored` runs on the ceiling -- the first arm, by `plan_arms`' ordering -- and refuses
the panel if any generation was still deliberating at the cap, because a censored ceiling is a
roof the flag set rather than one the model set, and every arm underneath is then compared
beneath it. And `check_pairable` moved inside the loop, so a stale record is caught by the arm
that exposed it rather than after six more have been spent.

The censoring check is deliberately asymmetric: it fires for the ceiling and for nothing else. A
3-bit arm that stops closing its queries inside a budget bf16 cleared comfortably is not a defect
in the run, it is the finding -- `unfinished_reasoning` is already in the record so the table can
report it. Refusing that arm would delete the result. Refusing a censored *ceiling* preserves the
panel, and it costs one arm's rerun at an hour rather than the whole panel's at seven.

### What a 38 M-parameter rehearsal found in four minutes

The guards above are about arms disagreeing with each other. This one is about the panel not
running at all, and it was found by refusing to let the real model be the first thing that tried
the DynQuant path.

Before spending seven GPU-hours, the arm was rehearsed on a model built to be structurally
identical and numerically worthless: `lfm2_moe` from the real `config.json` with every dimension
shrunk to the smallest group-aligned value that keeps the tree -- 6 layers, 8 experts, 38 552 064
parameters, and the same 14 three-dimensional tensors. Weights random, task nonsense, four
problems, 32 new tokens, on the CPU. What it exercises is not the accuracy, it is the *plumbing*:
`inspect --save-map` then `eval --map`, the two commands the panel issues for a DynQuant arm and
the only two nothing else in the campaign had run against this architecture.

Allocation was healthy -- 20 046 144 B realised against a 20 054 016 B anchor, −0.039% drift, all
ten bank tensors priced, roles resolving to `moe.expert.gate_up`, `moe.expert.down` and `ssm.in`.
Then the eval refused the map that the inspect one command earlier had written:

```
error: the bit map names 10 module(s) this model does not have
       (model.layers.1.feed_forward.experts.down_proj, ...). Was it built for a different
       checkpoint?
```

It was not. The map was correct and the message was wrong, in the same way twice over.

**First: `named_modules` misses raw parameters, for the third time.** A batched expert bank keeps
its experts as bare 3-D `nn.Parameter`s, so `model.get_submodule("...experts.gate_up_proj")`
raises. The tracker learned that (it writes those names), the graph learned it (it classifies
them), the allocator learned it (it prices them), and `quantize_model` learned it (`_target_tensor`
resolves them). The pre-flight guard in front of all three -- `check_map_covers`, which also fronts
`quantize` and `export` -- still asked `get_submodule` and refused. On this checkpoint that is
91.5% of the parameters, phrased as a stale map. The fix is not a second branch in the guard: it
now asks the quantizer's own resolver (`resolves_to_weight`), so the two cannot answer differently
again. A guard whose whole job is to predict what the next stage will do should be *calling* the
next stage's resolver, not reimplementing it.

**Second, and larger: the packed runtime genuinely cannot hold these weights.** With the guard
fixed, the command got one step further and failed inside `pack_model`, because packing replaces
`Linear` and `Embedding` *modules* and there is no module here to replace. That is not a bug to
patch out; the grouped packed path is P8 and it does not exist yet. It does mean the default way
of scoring a DynQuant arm reaches 8.5% of this model.

So `dynquant eval` gained `--map-apply {pack,encode}`. `pack` is unchanged and stays the default:
it swaps modules onto the packed runtime and the memory figure is real. `encode` runs the identical
encoder at the identical widths and writes the reconstruction back in the compute dtype -- same
accuracy, fp16 residency -- which is the only mode that reaches a weight held as a parameter. The
two are pinned as bit-identical on a bf16 `Linear` where both apply, so substituting one for the
other is neutral to the number being measured. The MoE arms run `encode`; the record says so; and
every byte figure in this campaign still comes from the map the allocator priced and the anchor
check verified, never from what the model happens to be holding while it is scored.

The two failures also used to arrive with one message. `get_submodule` raises `AttributeError`
both for a name the model does not have and for a name that addresses a tensor, and the packed
runtime reported both as "not a module of this model" -- which sends someone to re-check a map
that is right. Those are separate messages now, and the tensor one names the mode that works.

### What the two anchors are, before either arm runs

Recorded now rather than after, because a claim about what the allocator did is worth more
when the shape of the problem was written down before it did it. Both figures come from the
checkpoint's own module tree on the meta device -- no weights, no signal file, no run.

The model has 8.4676 B quantizable parameters and 0.111 M it will not touch (eighteen depthwise
`[2048, 1, 3]` convolutions, classified `other` and left in fp16 -- 0.0013% of the model, and
correctly excluded rather than fed to a group-128 encoder with a contracted dimension of three).
Paying every role's default floor costs **29 700 587 520 bits**.

| anchor | budget | floors | |
|---|---|---|---|
| 4 bits (4 399 629 312 B) | 35 197 034 496 bits | 29 700 587 520 | **655 MiB free** |
| 3 bits (3 332 904 576 B) | 26 663 236 608 bits | 29 700 587 520 | **362 MiB short** |

So the two arms are not the same exercise at two sizes. At 4 bits the floors are affordable and
the greedy knapsack spends 655 MiB upward, which is the regime the score was designed for. At 3
bits **no assignment exists that honours every floor**, and the soft-floor descent -- the thing
added because the paper's allocator silently returned the floor map here and let the score do
nothing (bug 4) -- decides the map by choosing which floors to breach and in what order.

Stronger, and specific to this architecture: stripping *every other role* to the hard 2-bit
minimum releases 2.953 G bits against a 3.037 G-bit deficit. Even destroying the LM head, the
embedding, every attention projection and every routed expert's `down_proj` does not close it. The
expert `gate_up` banks -- 5 167 M parameters at a floor of 4 -- have to be breached at the 3-bit
anchor no matter what the signal says. What the signal decides is *which* of the twenty-two banks
and by how much, and that is the whole of the 3-bit result.

The census also settles an open question that had been carried in the wrong terms. The conv
block's `out_proj` matches the name table's `("out_proj", ATTN_O)` entry, so on this model the
`attn.o` role holds 24 modules of which **18 are convolutions and 6 are attention**. That had been
noted as a possible floor error; it is not one -- `attn.o` and `ssm.out` both floor at 4, so no
module gets a different width for it. What it does change is the *within-role percentile rank*:
eighteen conv output projections and six attention output projections are ranked against each
other as one population. Whether that distorts either is measurable once the signal file exists
and is not measurable before, so it is left as it is for this panel rather than changed on a
guess -- a role reassignment made now would move the bit map for reasons nobody could then
separate from the result.

The rehearsal cost four minutes of CPU. Both defects sit on the DynQuant arms, which `plan_arms`
schedules fourth and seventh, so in the real panel the first one would have arrived roughly four
hours in, after the ceiling and both GPTQ arms had been paid for, phrased as though the bit map
were stale. The general form: a cheap structural double of the model exercises every command the
expensive run will issue, and the only thing it cannot check is the answer.

### The table the panel lands in, written before it runs

The panel is seven arms and about seven hours, and the step after it is a table. Writing that
table afterwards means writing it under time pressure, against real numbers, with every
formatting decision made by whichever number happened to be in front of it. So it was written
first, against a synthetic panel: `experiments/phase4/panel_table.py`, nine tests, no GPU and no
model load. Three of its decisions are load-bearing enough to record here.

**The size column comes from `arms.json`, not from the model that was scored.** Four of the six
quantized arms are evaluated on a checkpoint `compressed-tensors` wrote, and the other two --
the DynQuant arms -- are evaluated by encoding the allocator's widths back into bf16 weights and
writing the reconstruction in place, because 91.5% of this model's parameters live in batched
expert banks that the module-swapping path cannot reach (`--map-apply encode`, §12). A DynQuant
arm is therefore *resident* at fp16 while *costing* the allocator's bytes, and a size column
filled by measuring the loaded model would print 16 bits for the arm whose compression is the
entire claim. The honest source is the manifest: each baseline's own format accounting at its
width, and for a DynQuant arm the byte count the allocator realised -- the same number
`check_matched` already held against the anchor. The fp16 row is the one derived figure,
`params * 2` with `params` read from a baseline's `.quant.json` side file. No literal parameter
count appears in the script; one written for this model would be silently wrong for the next.

**Drift is checked in both directions, and a panel that fails it prints no comparisons at all.**
The whole point of the anchors is that the accuracy differences are not confounded with size, so
an arm off its budget does not deserve a footnote -- it invalidates every row below it. The table
recomputes each arm's distance from its anchor, marks any arm past the 0.1% tolerance, and then
refuses. The sign matters more than it looks: `--target-size` is a *ceiling*, so the failure that
actually happens in a real run is an arm landing **under** budget, not over. An absolute-value
test catches that; a `> tolerance` test silently passes it. That exact mutation was the one that
initially survived the mutation harness, because the fixture only exercised positive drift. It is
now tested in both directions and the harness catches all eight.

**Twelve comparisons, two Holm-corrected families, and the verdict follows the adjusted p.**
Twelve tests at alpha=0.05 expect half a false positive, and the headline of this panel is one of
the twelve. The comparisons split into two blocks that answer different questions -- six head-to-
head at matched bytes (DynQuant vs GPTQ vs AWQ at each width) and six against the bf16 ceiling --
and each is Holm-corrected within its own block, with the block size printed so a reader who
rejects the split can multiply by twelve instead. Holm rather than Bonferroni because it controls
the same family-wise rate and is uniformly more powerful; discarding real findings buys no rigour.
The synthetic panel is built so this is not decorative: two of its 4-bit comparisons are
significant raw (p = 0.0309 and 0.0243) and **neither survives correction** (0.0972), while the
3-bit result does. A test pins that the printed verdict and the JSON field both follow the
adjusted number, so a future edit that quietly reads the raw one turns red.

Writing the reader also found two defects in the writer, which is the cheap half of the
rehearsal lesson. The driver built its manifest only after all seven arms succeeded, so a panel
that died at the fifth would have left six hours of scored records with no manifest -- and the
manifest is the only thing that says which record belongs to which arm at which budget. It is now
rewritten after every arm, with unscored arms carrying `"record": null`, which the table already
renders as a missing row rather than a missing comparison. The refusal path keeps its stronger
property: an arm the pairing guard rejects is *not* named in the manifest, so its record can sit
in the directory without being read back into the comparison that just refused it. Second, the
driver stores each record's path as it was given, so a run launched with a relative `--out` writes
paths that only resolve from the directory it ran in; the table now retries beside the manifest,
which cannot pick up a foreign file because the writer's own invariant is that a record is named
for its arm and lives in `out`. Read literally, a manifest copied off the box would have reported
`0/7 arms scored` for a panel that finished -- an answer, not an error. The fixture is now built
through the driver's own writer rather than a hand-copy of its keys, so the two cannot drift.

Fixing that in one place turned out to be the more dangerous half-fix. The manifest names three
kinds of path -- the record, the saved bit map, and the `.quant.json` side file the parameter
count comes from -- and only the record went through the new resolver. The arms would then all
score, so the panel reads as whole, while the fp16 ceiling loses the parameter count that
denominates every bits-per-param figure and each DynQuant arm loses its allocation block. Both
print as absence, and absence of a floor breach is precisely what §12 pre-registered 4 bits to
show: the prediction would have been "confirmed" by the reader failing to find the evidence. The
map is also the case a filename-only retry gets actively wrong rather than merely misses, because
maps live in `out/maps/<label>.json` and the record of the same arm lives in `out/<label>.json` --
so the retry reads the record, parses it, finds no `maps` key, and reports no allocation. The rule
is now the longest tail of the stored path that exists under `out`, so the more specific location
always wins. The test that was supposed to cover this rewrote only the record and asserted only
the arm count; it passed before the fix and after it. It now rewrites the map too and asserts the
parameter line and the breached role by name.

Two smaller things the table also does. It prints the allocation next to the accuracy -- average
bits, the width histogram, and the *names* of any roles whose floors the budget could not afford
-- because the §12 prediction is that 4 bits breaches nothing and 3 bits must breach the expert
`gate_up` banks, and neither fact is recoverable from the accuracy afterwards.

The histogram is where checking the reader against a real artefact paid, rather than against the
fixture written alongside it. `BitMap.histogram()` counts **modules** at each width, not
parameters -- its docstring said parameters, `_map_payload` named the field `params`, and the
table duly printed the counts through a billions-scale formatter. On a real map that is
`{"2": 5, "3": 4, "4": 21, "8": 5}` for a 38 M model; on the panel it will be a width holding 181
tensors, and it would have rendered as `0K`. The allocator's entire answer, printed as though it
had assigned nothing -- and printed, not raised, which is the only failure mode that matters for
a formatter. Fixed in three places at once, because a unit that is wrong in the docstring is
wrong wherever anyone read the docstring. The parameter mass is genuinely not in the saved map;
it is in `violations[].num_params`, which is where the question that needs it -- how much of the
model did a breached floor cost -- is actually asked, and that is the one place the table still
formats at billions scale.

That the fixture had agreed with the wrong unit is the second half of the lesson. It was
hand-written, so it asserted whatever the reader already believed. Rebuilding it through
`write_bit_maps` with a real `BitMap` and real `FloorViolation`s immediately produced a second
disagreement: the fixture's role string `moe_expert_gate`, which no allocation has ever emitted,
against the enum's actual `moe.expert.gate_up`. Neither defect needed the panel to run, and
neither would have been found by a test that only compared the writer to a copy of itself.

### The signal file, audited at 41% instead of at the end

The S2 gate is pre-registered: signal collection must reach all 44 expert-bank tensors, because
91.5% of this model's parameters live in batched banks and a stats file that misses them would
send the allocator into the panel with no information about the mass it is allocating. The gate
was written to run on the finished file. It ran on the partial one instead, because the callback
rewrites the stats file periodically -- so the structural question could be answered four hours
before the fine-tune lands, when a bad answer would still leave time to restart. **44 bank
tensors, 0 missing.**

One thing that looked like a second finding was not one, and the way it dissolved is worth a
line. Checking that the gate could even import its dependencies, `grep -c 'def canonical_name'`
against `dynquant/integration/peft_utils.py` returned 0, and the obvious reading was that the gate
would raise at the moment it was needed -- the fine-tune finished, the gate invoked, nothing
checked. The obvious reading was wrong: `peft_utils` re-exports the name from
`dynquant.graph.naming`, so the import resolves. Grepping for a *definition* does not test
importability, and the one-line check that settles it is to run the import. Recorded because the
mistake is the one this section is otherwise about, pointed the other way: a verification that
does not execute the thing it claims to verify can produce a false alarm as easily as a false
pass.

Auditing the whole file rather than just the banks then produced an apparent contradiction worth
recording, because it is the tied-embedding trap running in the harmless direction. The stats file
holds 152 modules. Enumerating the model's quantizable tensors gives 112, all 112 present -- and
their parameter mass is 8,728,346,624, which is **103.08%** of the model's 8,467,856,128
parameters. Covering more than everything is not a coverage result; it is a double count.
`tie_word_embeddings` is `True` and vocab x hidden is 262,144,000, so `embed_tokens` and `lm_head`
are one tensor enumerated under two names. `model.parameters()` deduplicates shared storage and
the audit did not.

The question that actually matters is whether the *anchors* share the error, since they define
every matched-byte comparison in the panel. They do not. Dividing each anchor by its accounted
bits-per-parameter backs out ~8.4685 G parameters -- the deduplicated total, not the 8.728 G
double count. Better than that, the two widths cross-confirm each other: the small excess over
the deduplicated total is the fp16 remainder, which is charged 16 bits while the denominator
charges 4.15625 or 3.1484375, and solving for it gives 211,684 parameters at 4 bits and 211,706
at 3 bits. Two independent anchors agreeing on the same unquantized remainder to 0.01% is a
stronger statement about the accounting than either number alone.

The remaining 40 modules are 18 `Conv1d` and 22 `Lfm2MoeTopKRouter` -- 112 + 40 = 152, closing
exactly. Both are tracked on purpose. The routers are the ones that would matter if they were
not: a router is a `Linear` named `gate`, it is the module whose corruption collapses routing
outright, and it is measured here rather than assumed.

### The panel's interpreter is not the fine-tune's, so both were tested

The fine-tune runs under transformers 5.14.1 / torch 2.11; the panel runs under
`venv-llmc`, which is transformers **5.10.1** / torch 2.12, because that is where
`llmcompressor` and its `lfm2_moe` support live. Two things follow, and neither was checked
before the panel was already written.

The first is forward compatibility. The trainer writes the checkpoint under 5.14.1 and the panel
reads it under 5.10.1, which is the newer-writes-older direction -- the one that actually breaks.
Tested rather than assumed, with the 38 M structural double: saved under 5.14.1, loaded under
5.10.1, forward pass finite, 38,552,064 parameters. It works, and the seven-hour panel does not
begin with a load error.

The second is the decode settings, and this is the trap from an earlier campaign. LFM2.5-8B-A1B's
`generation_config.json` ships `temperature: 0.2`, `top_k: 80`, and `repetition_penalty: 1.05` --
chat defaults, reasonable for chat, not what a benchmark should decode with. Temperature and
top-k are inert under greedy and transformers says so out loud (*"generation flags are not valid
and may be ignored"*). The repetition penalty is not inert: it is a logits **processor**, not a
sampling warper, so it rewrites the logits before the argmax whether or not sampling is on. That
is the field that cost 19 GSM8K points once already, on a checkpoint shipping 1.1.

`greedy_generation_config` pins it, along with the rest of `NEUTRAL_DECODE`. The question is
whether the pin still holds on 5.10.1, and the honest answer required a control. Generating three
ways on the double -- through the guard, through a fully pinned config, and through a config that
names `do_sample=False` but leaves the penalty unset -- the first attempt returned *all three
identical*, which reads like a pass and is worthless: with random weights a 1.05 penalty never
flips an argmax, so the comparison could not have detected a failure either. Repeating it at
`repetition_penalty: 5.0` separated them: guarded still equals fully-pinned, and both now differ
from the unset one. So the 5.x refill is real on the panel's exact interpreter, the guard defeats
it, and the earlier all-identical result was a fixture without resolution rather than a guard
doing its job. A tripwire that cannot fire is the thing this campaign keeps re-learning; here it
cost one extra command instead of a run.

### Seven arms, one checkpoint -- and the six the panel does not write

The disk question was worth settling before the panel rather than during it, because its failure
mode is losing six hours at the fifth arm. The arithmetic that raised it -- seven quantized copies
of a 17 GB checkpoint against 102 G free -- was projecting a cost the panel does not pay.
`save_pretrained` appears in exactly one place in `baselines_lfm2.py`, inside `do_save`, and the
arms driver invokes `run`, which quantizes into the live model and scores it in the same process.
No baseline checkpoint is written at all. The panel's footprint is the merged fine-tuned checkpoint
and seven small JSON records.

What that settles for the panel it opens for the campaign's other half. The stated goal is to
publish all six quantized variants; the panel produces a publishable checkpoint for none of them,
and the three reasons are different. GPTQ 4-bit and AWQ 4-bit *could* be saved and simply are not
-- `do_save` exists and works. GPTQ 3-bit and AWQ 3-bit cannot be saved through this path at all:
`save` refuses 3 bits on purpose, because `compressed-tensors` packs 4 and 8 only, so a 3-bit
`save_pretrained` would write a dequantized bf16 folder the size of the original and label it
3-bit -- wrong in the one direction nobody checks, since it loads and generates perfectly well. And
the two DynQuant arms run `--map-apply encode`, which writes the reconstruction back into fp16
weights rather than packing them, because 91.5% of this model's parameters are batched expert banks
and the packed path in `runtime/linear.py` replaces `nn.Linear` modules, which a bank is not. So the
Hub push is separate work with a real hole in it -- P8, the grouped packed path for batched banks --
rather than a byproduct of the panel. Recorded now so it is a scheduled task and not a discovery
made at upload time.

Read against the writers rather than projected, the hole has a shape and a floor. Two of the six
variants publish today: GPTQ 4-bit and AWQ 4-bit, packed, at the bytes their manifest claims. Two
cannot be written by anything in the tree: the 3-bit baselines, because the container does not have
a 3-bit form and the only thing `save_pretrained` could write is a full-size bf16 folder wearing a
3-bit label. The two DynQuant arms are the interesting case, because *which* writer they fail is
not the one the P8 note implies. `dynquant export` -- the packed writer, the one a server loads --
refuses them, and refuses them by name: `_resolve` catches the `AttributeError`, asks
`resolves_to_weight`, and says *"is a tensor, not a module -- a batched expert bank ... this map
cannot be exported yet"*. `dynquant quantize --map` does not refuse them. It encodes at the map's
widths and writes the values back in the compute dtype, which is a directory
`AutoModelForCausalLM.from_pretrained` reads with DynQuant not installed at all, holding exactly
the numbers the packed checkpoint would hold. So the DynQuant variants are publishable as
*accuracy*-faithful artifacts now and *size*-faithful artifacts only after P8 -- and the command
that writes them already prints the packed size beside the directory size on every run, for the
specific reason that the research supplement reported storage savings from this path while the
model it loaded was fp16 all along. Publishing the encoded form under a "4-bit" name without that
distinction printed on the card would be the supplement's own error, committed deliberately.

So the push is two variants that are ready, two that need P8 to be honest about their size, and two
that need a 3-bit container that does not exist. That is a decision about what to publish, not a
gap to be discovered at upload time.

### The chat template is in a file, not the tokenizer config

Every arm renders its prompts with `--prompt-style chat`, which asks the tokenizer to lay out a
turn. On this model the template is not in `tokenizer_config.json` -- there is no `chat_template`
key at all -- it is a sibling file, `chat_template.jinja`. The panel therefore depends on a
round-trip nothing in the campaign had exercised: the fine-tune saves a tokenizer beside the merge
under transformers 5.14.1, and all seven arms load it under 5.10.1. A template lost anywhere along
that path does not raise. `apply_chat_template` falls back to rendering a bare string, seven arms
render the same wrong prompt, and the A/B stays perfectly self-consistent while measuring a model
nobody prompted correctly -- which is the shape of §8's shots defect, one layer lower.

Checked rather than assumed, before the merge existed, by saving the base checkpoint's tokenizer
with the fine-tune's interpreter and rendering a turn with the panel's: `chat_template.jinja` is
written by the newer version and read by the older one, and the rendered turn is identical --
`<|startoftext|><|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n`, 77 characters, byte for
byte. It survives. The reason to write down a check that passed is that its failure would have been
invisible in every artifact the panel produces.

### Both anchors allocated four hours before the checkpoint existed

The tracker rewrites the stats file periodically, so the allocation could be run against the real
signal on CPU while the GPU was still training -- the same property that let the bank-coverage gate
run at 41% instead of at the end. The pre-registration above made two predictions about what the
allocator would do at the two anchors. Both are now tested against the real signal rather than
against arithmetic.

| | 4-bit anchor | 3-bit anchor |
|---|---|---|
| budget | 4 399 629 312 B | 3 332 904 576 B |
| allocated | 4 397 666 304 B | 3 331 526 656 B |
| drift | -0.045% | -0.041% |
| average bits | 4.1547 | 3.1475 |
| floor breaches | **0** | **15**, over 3.55 G params |
| 8-bit tier | 63 modules, 444 M params | 35 modules, 18 M params |
| 4-bit tier | 52 modules, 5.91 G params | 61 modules, 2.56 G params |
| 3-bit tier | 14 modules, 1.64 G params | 11 modules, 2.36 G params |
| 2-bit tier | 4 modules, 470 M params | 26 modules, 3.52 G params |

Both land under budget and neither by enough to matter. That is the accounting claim closing:
`stored_bits` now charges what `accounted_bytes` charges the baselines, and two independent
allocations agreeing with it to within 0.05% is a stronger statement than either number alone.
The floors alone cost **3.5075 average bits** on this model, which is why the two anchors are in
different regimes rather than at two points on one curve: 4.15 has room above the floor map and
3.15 cannot be reached without breaking it.

The 4-bit prediction was that nothing breaches, and nothing does. On its own that is the weaker of
the two claims -- an allocation with no breach is also what an allocator that never moved would
produce, which is exactly the supplement's headline defect. What separates them here is that the
4-bit map is not the floor map. It puts four expert `down_proj` banks at 2 bits and fourteen at 3,
against a `moe.expert.down` floor of 2, and spends the 2.11 G parameters that buys on lifting 63
small modules to 8. Those are legal widths, not breaches, and they are the allocator choosing.

The 3-bit prediction was that the expert `gate_up` banks must breach. Fourteen of the twenty-two
do: four dropped from a floor of 4 to 2 bits (939.5 M params) and ten to 3 (2.35 G). That is the
pre-registered fact and it held.

The fifteenth breach was not pre-registered and is the more interesting one. `model.embed_tokens`
lands at 4 bits against a floor of 8 -- 262.1 M parameters. Its *role* floor is 4; the 8 comes from
`tie_word_embeddings` being `True`, which `classify` says out loud in the log -- *tied weights share
one bit-width: model.embed_tokens == lm_head (one tensor, one decision)*. So the 3-bit DynQuant arm
ships a 4-bit output head. An earlier campaign recorded the tied-embedding trap running the other
way, where `ignore=["lm_head"]` let a nominally 4-bit baseline measure 7.36 bits; this is the same
structure with the sign flipped. The budget is tight enough that the allocator pays for the expert
mass out of the head, and because the tensor is used twice the byte saving is counted once while the
accuracy cost lands twice -- in the input representation and in the logits. Whether that trade is
right is what the panel measures. The reason to write it down now is that it is a prediction the
pre-registration *missed*: found in the results afterwards it would have been indistinguishable from
a story assembled after seeing the accuracy.

### The concordance reads 1.000, over 8.5% of the model

`dynquant inspect` exists because the supplement's allocator produced a complete, plausible bit map
while never reading the importance scores -- inverting every score changed 0 of 282 modules -- and
nothing in its output said so. Within-role concordance is the guard: over every pair of modules in
the same role that got different widths, how often is the wider one the higher-scoring one. At the
4-bit anchor it reports **138 of 138 pairs agreeing, 1.000**.

Read the per-role breakdown and the number says less than it appears to. The 138 pairs come from
`attn.o` (119), `ssm.in` (17), `mlp.gate` (1) and `mlp.up` (1). Not one pair is drawn from an expert
bank. The guard against "the signal never reached the allocator" is computed entirely inside the
8.46% of parameters the signal could measure, and it is silent about the other 91.54% -- including
every module whose floor the 3-bit anchor breaches. The concordance is true, and it is not evidence
about the mass of this model.

That is not a defect in the diagnostic; it is the diagnostic correctly declining to answer. Expert
banks are unmeasurable **by construction**, and the tracker documents why: a bank's forward spans two
matmuls and a non-linearity, so no module boundary yields the `dY = dW x` pairing the Kronecker form
needs. A Gram matrix built at the bank boundary would pair the bank's input with the bank's output
gradient and produce, in the tracker's own words, "a well-formed number for a factorisation that
does not exist." Confirmed against the real moments file: 178 tensors, exactly 89 modules x
{`input_sq`, `output_grad_sq`}, and zero bank keys. Banks are still scored -- they carry saliency and
exact parameter-gradient plasticity, `_collect_bank_grads` measuring each tensor from the side it
owns. They are scored by the rank-product proxy rather than by measured Gauss-Newton sensitivity,
and the two prices sit at different floors.

`_apply_fallback_scale` is what makes them comparable: it rescales the proxy onto the sensitivity
scale by a ratio of medians over rung-normalised prices -- each module's next-step value divided by
its width span -- because calibrating on raw step values would apply the width factor twice. On this
model the split is **89 modules measured (716 570 624 params, 8.46%) against 44 proxied
(7 751 073 792 params, 91.54%)**, and the multiplier is **1.807e-17**. It is the same number at both
anchors, which is the expected result and worth checking anyway: the rescale is a property of the
two price populations, not of the budget, so a multiplier that moved with the target would mean it
was being fitted to the answer.

That one constant decides where 91.5% of the parameters sit in a heap ordered against the other
8.5%, and the map shows how completely. At the 4-bit anchor **every module above 4 bits is a
measured one** -- all 63 in the 8-bit tier -- and **every module below 4 bits is a proxied one** --
all 4 at 2 bits, all 14 at 3. The 4-bit tier is where the two populations meet, 52 modules of which
26 were measured. At the 3-bit anchor the same boundary holds with the mass pushed down: the 35
modules at 8 bits are all measured, the 26 at 2 bits are all proxied. The allocator's decision
boundary runs almost exactly along the line between its two prices, which is either the correct
answer -- the expert mass genuinely is the cheapest place to buy bytes -- or an artifact of where
`1.807e-17` happened to land the proxy population. Nothing in this campaign yet distinguishes those,
and the panel does not distinguish them either; it measures the consequence.

Until this week that multiplier was not written anywhere. Not in the saved bit map, not in the panel
record, not in the manifest. It was a `logger.debug` line, in a run whose driver captures stdout at
WARNING.

Worse than absent, two reporting paths in `inspect` actively misdescribed it. The per-width score
statistics were computed over all members of a width group with unmeasured modules padded to zero,
so a group containing nothing but expert banks printed `score_min = score_median = score_max = 0.0`.
That is indistinguishable from a group the signal measured and found worthless -- and on this
architecture the 2- and 3-bit groups were the first kind while reading as the second, at both
anchors, which is 2.11 G parameters at 4 bits and 5.89 G at 3. The `narrowest` list had the same
error in a different shape: `ordering.get(name, 0.0)`, so the widest tensors in the model appeared
at the bottom of a ranked list carrying a score of zero, reading as "the allocator put 91.5% of the
model at the bottom" rather than "these were priced another way."

The fix is not to force a factorisation that does not exist. It is to make the split visible
wherever the widths are. `BitMap` now carries a `Pricing` record -- modules and parameters on each
side, and the multiplier -- which `_map_payload` writes into the saved map, `inspect` reports in both
JSON and rendered form, and `panel_table` prints beside the width histogram. The 4-bit dry run now
renders as:

```
   2b  n=  4     469.8M params  measured dL(2b)-dL(8b) per parameter: none of them measured
   3b  n= 14    1644.2M params  measured dL(2b)-dL(8b) per parameter: none of them measured
   4b  n= 52    5909.8M params  measured dL(2b)-dL(8b) per parameter min 3.853e-18  median 7.049e-18  max 1.19e-17 (over 26 measured)
   8b  n= 63     443.9M params  measured dL(2b)-dL(8b) per parameter min 5.068e-18  median 2.467e-16  max 5.984e-16
  priced: 89 measured, 44 from the score proxy (91.5% of parameters, rescaled by 1.807e-17)
```

Statistics over a width group are taken over the members the quantity actually covers, with a
`measured` count beside the module count and `None` rather than `0.0` where nothing was measured;
a group with no measured member says so. The allocation itself is unchanged -- both re-runs
reproduce 4.154741179776939 / 4 397 666 304 B and 3.1474946101794448 / 3 331 526 656 B exactly --
which is the point: this was an observability defect, not an allocation defect, and a change that
moved the numbers would have been the wrong change.

One distinction in that record was worth getting right rather than convenient. `scale` is `1.0` when
no rescaling was needed -- every module measured, or none of them -- because in both cases one
formula priced the whole heap and the order it produced means something. It is `None` only for the
case where the order does not: a mix of the two prices with no positive overlap to calibrate on,
which `_apply_fallback_scale` already warns about and then proceeds through. A first attempt marked
the no-moments run `None` as well, which would have filed the most trustworthy ordering the
allocator can produce under the same marker as the least trustworthy one. The run that hits the
uncalibrated path is precisely the run whose bit map should not be trusted, and in the artifacts it
looked identical to every other run.

Four new mutations against the table -- pricing line dropped from the allocation block, pricing not
carried out of the map, the proxied share reported as a module count, an uncalibrated scale rendered
as `1.000e+00` -- all caught, 17 of 17 overall. The module count is the mutation worth naming: 44 of
133 reads like an edge case and is 91.5% of the checkpoint, so the parameter share goes first in the
line and the count second.

### The whole driver, rehearsed on 38 M parameters, and the four arms it stopped

The rehearsal above ran the DynQuant arm alone. What had still never run was the *driver* --
seven arms in sequence, each record into the manifest, the manifest into the pairing guard and
the table. On the double it costs about four minutes on the CPU, beside the fine-tune that owns
the GPU. It failed four times, and all four would have failed the real panel; three of them fail
after the calibration pass, which is where the entire cost of a baseline arm sits.

**One: `run` could not load anywhere but the GPU.** `--device` existed on `linearize` and nowhere
else. `run` -- the only subcommand `arms_lfm2.py` invokes -- passed nothing, and `load_linearized`
defaulted to `device_map="cuda"`. That is why the rehearsal had been deferred rather than run: the
box's GPU is held for hours by the fine-tune that produces the signal being scored, so the one
machine the panel is scheduled on was the one machine the panel could not be rehearsed on. A
driver that can only be rehearsed on the hardware the real run needs is a driver that gets
rehearsed for the first time during the real run. `--device` is now a single flag on the arms
driver, threaded into the shared `eval_flags` so it reaches all seven arms -- a panel where one
arm loaded somewhere else is not a panel either. `dq_inspect_cmd` is deliberately left out of it:
allocation reads a stats file and a config, `inspect` already defaults to `cpu`, and routing the
one CPU-only step of the panel to the panel's device would put it on the GPU the passthrough
exists to keep clear. Four mutations, all caught, including the flag parsed and then ignored --
which would leave a CPU rehearsal OOMing against the fine-tune it was meant to run beside.

**Two: the provenance qualification reached the tokenizer.** Found at the second arm. The baseline
arms deliberately set `--model` to a non-path -- the qualified string `<merge>#gptq-4b-g128`, so
six records built from one checkpoint do not all claim to be the same weights. `evaluate.run`
defaults `--tokenizer` to `--model`, and handed that string to `from_pretrained`, which rejected it
as a Hub repo id. On the real panel that is four arms x a 256-sample pass over 8 B parameters, each
one dying at the tokenizer with the quantized weights already in memory and nothing written.
`eval_namespace` now states `--tokenizer` as the directory the weights were loaded from, and
`evaluate.run`'s docstring says the sentence that would have prevented it: the field a caller owns
is also the tokenizer default.

**Three: AWQ had no mappings for this architecture, and would not have said so.** AWQ divides a
linear's input channels by a per-channel scale and folds the inverse into whatever produced that
input. Which module that is comes from a per-architecture table, and `Lfm2MoeForCausalLM` is in
neither of llm-compressor's two: not the static `AWQ_MAPPING_REGISTRY`, and not the dynamic
builder, whose hybrid-stack shape is exactly right but whose `_get_hybrid_attention_config`
returns `None` unless `layer_types` contains `linear_attention` -- this model says `conv`. So the
Llama defaults applied, and on this architecture they are wrong in both halves of every block: the
pre-mixer norm is `operator_norm`, not `input_layernorm`; the pre-feed-forward norm is `ffn_norm`,
not `post_attention_layernorm`; and attention's output projection is `out_proj`, not `o_proj`.
`q/k/v_proj` matched, their smooth partner never did, and `match_modules_set` raised on the
incomplete set -- `awq_4b failed with exit code 1`, after the calibration pass.

The mappings themselves are seven regexes and were the easy half. The half that matters is the
number beside each one. A mapping that matches *nothing* does not raise: `_set_resolved_mappings`
logs a `debug` line and moves on, so the arm runs to completion, writes a record, and enters the
table as round-to-nearest wearing an AWQ label. That is the failure `materialize_quantization`
exists to prevent (§10) in a different disguise, and it wants the same treatment -- predict the
set count from the config, resolve it against the real tree with llm-compressor's own matcher, and
fail before the calibration pass rather than after it. The prediction is read off `layer_types`,
`num_dense_layers` and `num_experts`: six attention blocks, eighteen convolution blocks, two dense
feed-forwards, twenty-two MoE feed-forwards and thirty-two experts give **`[2, 2, 6, 18, 22, 704]`
sets across six mappings**. The 704 is the one that would have been quietly wrong --
`match_modules_set` yields when the lowest common ancestor of the matched set changes, so a pair
that sits inside one expert yields one set per *expert*, not one per layer. Checking with
llm-compressor's own matcher rather than a hand-rolled name scan is deliberate for the same
reason: a scan can agree with the regexes and disagree with the matcher, and the disagreement is
the whole defect. (`_match_name` calls `re.match`, not `fullmatch`, so every pattern is
`$`-anchored and the layer alternation is emitted widest-first -- `(21|18|...|2)` -- which makes
`(2|21)` matching `layers.21.` a property of the expression rather than of backtracking.)

**Four: AWQ's own grouped-query guard is inert on this model.** With the mappings resolving, the
arm reached `_apply_smoothing` and died -- `The size of tensor a (128) must match the size of
tensor b (256)`, at `_smooth`'s `weight[-scales.size(0):].div_(scales.view(-1, 1))`. The pair is
`v_proj -> out_proj`, and under grouped-query attention it cannot exist: `v_proj` emits
`num_key_value_heads x head_dim` rows and `out_proj` consumes `num_attention_heads x head_dim`, so
there is no per-channel scale that divides one and multiplies the other. LFM2.5-8B-A1B is 4:1 GQA,
32 heads against 8 KV heads. Upstream knows this and drops the pair -- but
`_check_layers_are_compatible` tests `balance_name.endswith(".o_proj")`, so on a model that spells
it `out_proj` the guard never fires. Ours is conditional on `kv_heads == heads` read from the
config, which is the same fact stated where it is true rather than where it happens to be spelled,
and the consequence is recorded rather than assumed: `self_attn.out_proj` is quantized without an
AWQ scale, and it appears in `unsmoothed_linears` beside `conv.out_proj` and `lm_head`, neither of
which has a linear producer to fold an inverse scale into.

The double is 2:1 GQA for the same reason the real model is 4:1 -- it is the config, shrunk. Both
AWQ arms now record **6 mappings, 145 Linear modules, 138 smoothed**, unsmoothed
`{conv.out_proj: 4, lm_head: 1, self_attn.out_proj: 2}`, and `145/145` weights materialized, which
is the shape §10 says an AWQ arm should have. Nine tests and ten mutations; the first pass caught
nine of ten, and the one that got through is worth naming. The selection test asserted that every
balance layer sits under a *selected* block, but not the converse -- so an attention-norm pattern
with its layer scope deleted, matching all 24 blocks against 6 attention blocks, passed. It is the
raising case from the other side. Set equality in both directions closes it, and the re-run caught
10 of 10.

The seven arms then ran end to end into `arms.json` and the table rendered: all seven rows, both
DynQuant width histograms, the pricing line, the floor-breach lists, and both Holm-corrected
McNemar blocks. Every accuracy is 0.0%, which is the correct answer -- 38 M random weights
answering four text-to-SQL problems in 32 new tokens. The rehearsal measures the plumbing. The
plumbing is what four of the real panel's GPU-hours would otherwise have been spent discovering,
one arm at a time, each time after the expensive part.

### The rehearsal passed `--limit 4`, and that is the one flag it could not rehearse

Every setting in `eval_flags` is stated on every arm's command line, including the ones that have
perfectly good defaults, for a reason §8 paid for: a setting left to a default is a setting two arms
can disagree about while their commands read identically. `--limit` is the exception. It is
forwarded only when it is not `None`, and an absent `--limit` means `dynquant eval` scores the
whole test split -- which on this benchmark is **16 143 items**, on each of seven arms.

Nothing about that is a bug in the arms. It is a wall-clock decision of the first magnitude that
nobody had made: the panel would have run until it was killed, or until the box was recycled with
nothing pushed off it, and either way the first three arms would have been thrown away.

The rehearsal could not have found it, and the reason is structural rather than an oversight. A
rehearsal is only worth running if it is cheap, so this one ran four problems at 32 new tokens --
it *supplied* the flag whose absence is the defect. A cheap rehearsal exercises every code path the
real run takes and is blind to precisely those settings that make the real run expensive, because
supplying them is what made it cheap. Worth stating once, because it applies to every rehearsal
this campaign will run: what the rehearsal passes on the command line is what it cannot test.

So the limit stops being a default and becomes an argument. The launcher refuses `--go` without an
item count and says why, which is the same move as §10's set-count prediction -- convert a silence into
a refusal at the cheapest point.

### Pricing the evaluation set: power first, then wall clock

Two constraints pull opposite ways, and they are not symmetric.

**Power binds.** This panel exists to separate arms that differ by one or two points, and an
underpowered panel is not a weaker conclusion -- it is a re-run of all seven arms, because `limit`
is a pairing field and a larger set cannot be bolted onto records scored at a smaller one. The
comparison is McNemar on the stored hit vectors, so the standard error of a paired difference is
`sqrt(d / n)` in the *discordance rate* `d`, not the much larger unpaired one; and the head-to-head
block is six comparisons under Holm, so the smallest *p* in it has to clear 0.05 / 6 = 0.0083, or
*z* = 2.64.

That leaves `d`. It was first written here as 8%, from memory, and it is wrong. Phase 3's stored hit
vectors answer it directly, and they were already on disk: over the twelve same-width pairs among
Ministral-8B's quantized arms, two quantizations of one model disagree on a **median 20.0% of IFEval
items and 18.0% of HumanEval items**, and the 3-bit pairs run higher than the 4-bit ones -- `dq3` vs
`rank3` at 25.7% against `rank4` vs `shuf4` at 11.6%. Quantization does not perturb a few hard items
at the margin; it reshuffles a fifth of the benchmark and mostly breaks even. At `d = 0.20` the
smallest difference Holm can call is:

| items | s.e. of the paired difference | smallest callable difference |
| ---: | ---: | ---: |
| 4 000 | 0.71 pts | 1.87 pts |
| 6 000 | 0.58 pts | 1.52 pts |
| 8 000 | 0.50 pts | 1.32 pts |
| 12 000 | 0.41 pts | 1.08 pts |
| 16 143 | 0.35 pts | 0.93 pts |

The last row is the ceiling, and it is worth saying why it is 16 143 and not the 21 729 this report
quotes elsewhere. Those are different populations and both numbers are right. 21 729 is the raw row
count of Gretel's and WikiSQL's `test` splits, which is the correct denominator for the
decontamination scan in section 6 -- a training row leaks whether or not the evaluator would have
admitted it. The evaluator admits fewer, on three documented conditions: the source must carry
`INSERT`s (so `create-context`, whose contexts are bare `CREATE TABLE`, is excluded from scoring
altogether and trains only -- two queries returning nothing compare equal, and `SELECT 0` would
score near the ceiling), the gold must lead with `SELECT` or `WITH` (9.8% of Gretel's are DML, which
the extractor cannot read back), and the gold must actually return rows against its own schema. That
leaves 16 143 scoreable items, which is what `--limit` unset means in practice and what the last row
prices.

The prior DynQuant-over-GPTQ headline was **+1.54 points**. So 4 000 items -- the number the 8%
figure endorsed two paragraphs ago -- buys a null result at full price, and the floor is **8 000**.
Recorded as a correction rather than quietly applied, because the shape is worth keeping: the
assumption was off by a factor of two and a half, it was the only input to the decision, and it
reversed it. Checking it cost one pass over records this campaign had already written.

**Wall clock is the other, and it was never measured.** No text-to-SQL eval in this campaign
recorded its seconds against an item count, so the cost of an arm on this model is unknown to
within an order of magnitude, and the honest response to an unknown that decides the run is to
measure it rather than to pick a limit that sounds reasonable. Hence `s4_probe.sh`: 128 items on
the merge under the panel's own flags, byte for byte, differing only in the number being measured.
It costs minutes and prices everything after it.

The probe sweeps batch size while it is there, because that is the only lever that buys wall clock
without costing power. `batch_size` is deliberately outside `PAIRING_FIELDS`: left-padded decode can
move the last bits of a logit, so two batch sizes are not guaranteed identical per-item outcomes,
but the prompts and the problems are the same and that is what pairing is about. Raising it
uniformly across all seven arms is therefore legitimate where shortening the decode budget or
thinning the item set is not. The task default is 32 on a 97 GiB card holding a 16.5 GB model whose
KV cache spans six attention layers out of twenty-four -- so 32 is a default, not a measurement, and
the probe measures 32, 64 and 128. An out-of-memory at 128 is a result and is reported as one.

The probe earns its place twice over, because it also answers a question that should be asked
before seven arms are spent on a checkpoint. The base model scored **57.75%** on 400 of these
problems at this decode budget (§8). A merge that lands below that is a fine-tune to investigate,
not a checkpoint to quantize six ways -- and finding that out from a five-minute probe is different
from finding it out from a bf16 ceiling arm that has already run.


**Measured.** The probe ran three batch sizes over the same 128 items and returned something the
sweep was not designed to find: on this model, bigger batches are *slower*.

| batch | accuracy | wall clock | per item |
| ---: | ---: | ---: | ---: |
| 32 | 86.72% | 90.7 s | **0.709 s** |
| 64 | 85.16% | 94.7 s | 0.740 s |
| 128 | 85.94% | 112.6 s | 0.880 s |

Nothing ran out of memory, and the ordering is monotone in the unexpected direction: 128 costs 24%
more per item than 32. The mechanism is the decode budget, not the card. A batch runs until its
*longest* member emits a stop token, and `--max-new-tokens` is 1024, so every member of a batch pays
for its straggler and widening the batch widens the chance of drawing one. Text-to-SQL sharpens
this, because the answer-length distribution is badly skewed -- most golds are one short `SELECT`, a
few are joins over subqueries -- so each additional lane of parallelism is bought against a longer
tail. The default of 32 turned out to be the right setting, but it is worth recording that it was
checked and why it wins: the reflex on a 97 GiB card holding a 16.5 GB model is to raise the batch
size, and here the reflex costs 24%.

The accuracy column carries no signal and must not be read as one. At n = 128 and p near 0.86 the
standard error of a single proportion is 3.1 points, so the 1.56-point spread across the three rows
sits well inside noise. That is the expected result for a lever deliberately left outside
`PAIRING_FIELDS` -- and had the spread been larger, the exclusion would have needed revisiting
before the panel, not after it.

**The fine-tune worked, decisively.** 86.72% against the base model's 57.75% at the same decode
budget on the same task (section 8). That is the gate the probe existed to be: a merge landing at or
below the base model is a training run to investigate, not a checkpoint to quantize six ways, and
five minutes is the right price to learn which one this is.

**So the panel runs 12 000 items at batch 32.** 12 000 items x 7 arms x 0.709 s is 16.5 hours of
scoring, plus the 45-64 s per-invocation overhead measured across 36 phase-3 records, plus GPTQ and
AWQ calibration, which nothing in this campaign has ever timed and which this panel will record for
the first time. It buys a smallest-callable-difference of **1.08 points** against a +1.54-point
headline: above the 8 000-item floor, one row short of the 0.93-point ceiling, and inside the
pre-committed budget of about sixteen hours. The remaining 4 143 items would buy 0.15 points of
resolution for another 5.7 hours, which is the wrong trade when the effect under test is more than
ten times the resolution already purchased.

One defect closed on the way there. The probe wrote `probe.b$B.json` and its summary read the
records back through a glob -- so a batch size that ran out of memory would have written no record,
and the summary would have silently priced the whole panel off a *previous* probe's file. The
records are now removed before each batch runs, on the same principle that produced
`check_resumable`: a missing row is a result, a stale row wearing this run's timestamp is not.


### The measured 8.46% is a moments gap, not a gradient gap

*This subsection replaces an earlier version that got the cause wrong. The reasoning is left
visible because the wrong version was plausible, was written down, and would have sent the next
experiment to the wrong place.*

The allocator prices **89 modules by the Gauss-Newton form (716 570 624 params, 8.46%) and 44
expert banks by the rescaled rank-product proxy (7 751 073 792 params, 91.54%)**. Two candidate
explanations, and the campaign briefly committed to the second.

The first is that the fine-tune never trained the banks. That part is true. LoRA with
`target_modules="all-linear"` reports **0.13% of 8.48 B trainable**, about 11 M adapter parameters;
wrapping the 2112 expert matrices at the configured rank of 32 would cost on the order of 260 M.
At runtime the banks are batched `nn.Parameter`s rather than `nn.Linear` children, so PEFT's module
walk cannot see them, and the checkpoint's per-expert storage (2134 tensors) makes this look untrue
on disk. The banks were frozen for the whole run.

The inference drawn from it -- that the pricing split *follows* from the freeze, because plasticity
is gradient-derived and a frozen parameter has no gradient -- is wrong, and the signal file says so
directly: **all 44 banks carry `grad_norm_count = 1560`**, every optimizer step, the same as every
trained module. Not updated is not the same as not differentiated. `--measure-expert-banks` calls
`requires_grad_(True)` on the banks for the duration of the run, autograd fills `.grad` for the
whole batched tensor, and `_collect_bank_grads` reads its norm and immediately releases it -- the
release being load-bearing, because these parameters sit in no optimizer and `zero_grad()` never
reaches them, so an unreleased `.grad` would accumulate monotonically and plasticity, being a
*variance* of gradient norms, would rank experts by how late in the run they were touched.

That path is not a degraded substitute. It is the **most** exact plasticity measurement in the
system: every other module is priced by `outer_exact`, which reconstructs `∇W = δxᵀ` on a bounded
256-token subsample, while the banks are priced from the gradient autograd actually computed over
every token. The only entries that never saw a gradient at all are `embed_tokens` and 18 depthwise
`conv.conv` modules of 6144 parameters each. Both halves of the DynQuant signal cover 100% of the
quantizable model.

What is actually missing is a different artifact. Gauss-Newton sensitivity is priced from
per-channel second moments in `dynquant_moments.safetensors`, and that file holds **178 tensors --
89 modules x {`input_sq`, `output_grad_sq`} -- and not one key containing `expert`**. So the banks
fall through to the proxy for want of moments, not for want of gradients. `--measure-expert-banks`
teaches the *stats tracker* about batched parameters; there is no equivalent for the *moments
collector*, which is the `named_modules`-misses-raw-parameters blind spot for the third time in
this project -- first costing bytes in the denominator, then costing LoRA coverage, now costing
sensitivity pricing.

The distinction matters for what the fix is. A parameter gradient is free: autograd computes it
whether or not anything consumes it, so plasticity needed only a flag. Moments are not, because the
Kronecker form needs the `x` and `δ` of a *single* matmul, and a batched MoE bank's forward runs
gate-up, a non-linearity and down inside one fused call -- there is no module boundary where that
pairing is exposed, which is why section 12 elsewhere calls the banks unmeasurable by construction
rather than unmeasured by omission. Reaching them means instrumenting inside the MoE forward and
then re-collecting the signal, which is another fine-tune-length run, not a collector patch.

So the honest description of this panel is: **full DynQuant signal on 100% of the model, measured
sensitivity pricing on 8.46% of it** -- a narrower and more accurate claim than "the signal is
untested on 91.5% of the parameters". The earlier version reached roughly the right cost estimate
for the follow-up through the wrong mechanism, which is the failure mode worth naming: it would
have sent the next experiment at the training recipe, where there is nothing to fix, instead of at
the moments collector, where there is.

### The calibration cost is worth measuring and not worth probing

The projection above prices the *eval* half. The four baseline arms also pay a 256-sample
calibration pass over 8 B parameters, which this campaign has never timed, and on this architecture
there is reason to think it is not small: GPTQ's cost is per module, not per parameter, and
linearizing the MoE turns 22 layers into 2112 expert matrices before the attention and conv blocks
are counted. AWQ resolves 704 expert sets. A dense 8 B model is a poor guide to either.

The obvious move is to probe it -- one GPTQ pass at 16 samples and one at 64 separates the fixed
per-module term from the per-sample term and extrapolates to 256. It is the wrong move, and the
reason is worth keeping. The calibration cost is **additive and independent of `--limit`**: it is
the same number whether the panel scores 8000 items or 16 143. Lowering the item count cannot buy
any of it back, and the item count is already floored at 8000 by the power calculation. So a
measurement taken before the launch cannot change the launch -- there is no value of the number
that selects a different limit, and dropping a baseline arm is not available because the six
variants are the deliverable. It would be information with no decision attached to it, bought with
two model loads and forty minutes.

What it can change is the *next* panel, and that is precisely what the timing added to `_run`
records for free while this one runs. The rule this leaves behind: probe a cost before committing
only when some value of it would change what you commit to. Otherwise instrument it and read it
afterwards.

The one number the phase 3 records do supply is the per-invocation overhead, and it supplies it
thirty-six times. Subtracting each record's own `seconds` from the gap to the previous record's
mtime gives 45-64 s across every arm and every task on Ministral-8B -- model load and task setup,
flat, with no arm paying more than another. Seven arms of that is six minutes, which is why it does
not appear in the projection.

### `--resume` can tell that a record exists; it cannot tell whose it is

The panel launches with `--resume`, and resume is the right default: an arm that dies at the fifth
of seven should not cost the four that finished. But the whole of what resume checks is
`record.is_file()`, and a file's existence carries no claim about which model produced it or when.

`check_pairable` looks like the guard against that and is not. It compares records through the eval
command's own `_comparability`, which is the contract a paired test needs -- task, backend, split,
shots, shot seed, limit. Every one of those fields describes the *problem set*. None of them names
the *model*, and none of them can be older than anything. So two records scored on two different
merges at identical settings pair perfectly, and a record allocated from the signal file as it was
before a rewrite pairs perfectly with one allocated after. Both produce a table that reports a
difference between quantizers and is measuring a difference in the checkpoint. This campaign has
already met that failure once, in a different shape: a skip-if-output-exists guard cannot see that
its output predates its input.

So `check_resumable` asks the two questions `_comparability` structurally cannot. The manifest
already records which model, signal file, moments file and group size the previous run was handed
-- it is rewritten after every arm precisely so a dead panel leaves one -- and a resume that
changes any of them is a new panel wearing the old one's directory, refused before the first arm.
Then every surviving record is compared against the mtimes of the inputs it is supposed to have
scored.

The freshness charge is per arm, and that asymmetry is the part worth writing down. The signal file
is charged only against the two DynQuant arms, because the four baselines never open it.
Regenerating the signal without retraining -- rerunning the bank census, fixing a key, extending
the moments -- invalidates two arms and leaves four standing, and a guard that condemned all six
would price the cheapest correct fix at a whole new panel. That is how a guard teaches people to
pass `--resume` less often instead of more carefully.

It refuses rather than repairs. Deleting the directory is one command and always right; deciding
which of seven records survived a change of inputs is neither. Two tests, three mutations, all
caught -- including the one that makes the stats charge unconditional, which is the mutation that
turns a correct guard into an annoying one.

---

## 13. Status

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

Also done since: headroom re-measured at 1024 tokens -- **57.75%**, uncensored enough to read a
decode budget off (§8) -- `load_linearized` exercised on the real checkpoint, and the training
mixture decontaminated against its own benchmark (§11), which removed 4016 rows the fine-tune
would otherwise have been trained on and scored against.

Also done since: both interpreters tested against each other -- a 5.14.1-written checkpoint
loads and runs under the panel's 5.10.1, and the greedy pin still defeats the checkpoint's
repetition penalty there, verified with a control after the first check turned out to be blind.
The signal file audited mid-run -- 44 expert-bank tensors with none missing,
152 modules closing exactly as 112 quantizable plus 18 conv plus 22 routers, and both anchors
cross-confirming the same fp16 remainder. And the table the panel lands in, written before the
panel runs -- ten tests and thirteen mutations against a synthetic seven-arm run, sizes read from the manifest rather than from
the fp16-resident scored model, drift refused in both directions, and twelve comparisons Holm-
corrected in two blocks with the verdict following the adjusted p. Checking that reader against a
real saved map, rather than against its own fixture, is what caught the width histogram being
counted in modules and printed in parameters.

Also done since: the disk question settled by reading the driver rather than projecting from
checkpoint sizes -- the baselines score in-process and write nothing, so the panel costs one
merged checkpoint and seven JSON records -- and the same read established that the Hub push is
separate work with a hole in it, since two of the six variants cannot be saved through
`compressed-tensors` at all and the two DynQuant arms encode rather than pack.

Also done since: both anchors allocated against the real signal file while the fine-tune was still
running -- **0 floor breaches at 4 bits and 15 at 3**, both within 0.05% of their budget, the
pre-registered `gate_up` breach confirmed at fourteen of twenty-two banks, and one breach the
pre-registration missed: the tied `embed_tokens` at 4 bits against a floor of 8, which is a 4-bit
LM head at the 3-bit anchor.

Also done since: which of the two prices chose those widths, recorded in the map rather than in a
debug log. **89 modules measured, 44 priced by the rescaled rank-product proxy -- 8.46% of
parameters against 91.54%** -- because a batched expert bank has no boundary at which the
Gauss-Newton form exists. The within-role concordance that guards against the supplement's headline
defect reads 1.000 at 4 bits and is computed entirely inside the measured 8.46%. Two `inspect`
paths that had been reporting the unmeasured mass as measured-and-worthless are fixed, four
mutations added, 17 of 17 caught, and the allocation is bit-identical before and after -- which is
what an observability fix should be.

Also done since: the driver itself, rehearsed end to end -- seven arms on a 38 M structural
double, into the manifest and out to the table -- which found **five more defects**, three of them
failing after the calibration pass: `run` was CUDA-only, the provenance qualification reached the
tokenizer, AWQ had no mappings for this architecture and would have been skipped silently rather
than raised, and upstream's own grouped-query guard is inert on a model that spells the output
projection `out_proj`. Nineteen tests and sixteen mutations across those four, all caught. The
fifth is not code and has no test: `--limit` is forwarded only when set, so the panel as staged
would have scored the whole 16 143-item split on every one of seven arms. The launcher now refuses
to launch without an item count, and `s4_probe.sh` measures what one costs before any are spent.
The sixth came from reading the resume path against a failure this campaign has already had:
`--resume` reuses any record whose file exists, and the pairing check that looks like it would
catch a foreign one reads only fields that describe the problem set, never the model and never a
date. `check_resumable` now refuses a directory whose manifest names different inputs, and any
record older than the inputs it claims to have scored -- charging the signal file against the two
arms that read it and not the four that do not. Two tests, three mutations, all caught.

Not done, in order: the fine-tune, with the expert mass measured; then the six arms at matched
bytes -- GPTQ and AWQ through the linearized structure verified bit-exact in §10, all three widths
weighed rather than requested; then the paired comparison over stored per-item hits.

One number in this report is now conditional on §11 rather than absolute. The fine-tuned model's
accuracy is a claim about text-to-SQL only because the filter ran; before it, 38 of the 200 WikiSQL
items were in the mixture at this seed and 189 of 200 were in the pool any seed draws from. The
paired comparison between arms never depended on that and still does not.

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
