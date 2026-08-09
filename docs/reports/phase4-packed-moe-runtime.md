# Phase 4 · the packed MoE runtime — making the artifact the same object as the measurement

**Measured 2026-08-08 / 2026-08-09.** Code:
[`runtime/linear.py`](../../packages/dynquant-core/src/dynquant/runtime/linear.py),
[`quant/tensor.py`](../../packages/dynquant-core/src/dynquant/quant/tensor.py),
[`quant/checkpoint.py`](../../packages/dynquant-core/src/dynquant/quant/checkpoint.py),
[`integration/hf_quantizer.py`](../../packages/dynquant-core/src/dynquant/integration/hf_quantizer.py).
Tests: [`tests/test_expert_bank.py`](../../tests/test_expert_bank.py) (20),
[`tests/test_hf_quantizer.py`](../../tests/test_hf_quantizer.py) (18),
[`tests/test_export_checkpoint.py`](../../tests/test_export_checkpoint.py).
Commits `a60d993`, `10846ca`, `18a8870`, `62cadf7`, `fac7624`.

The seven-arm panel scores the **encoder** — `quantize_model(in_place=True)`, which rewrites each
weight to its dequantized value and leaves an ordinary model behind. It does that because a
matched-byte panel needs one GPU pass per arm and nothing on disk. What a person downloads is the
**packer** — `export_packed_checkpoint` and the `DynQuantLinear` / `DynQuantExpertBank` runtime
that reads it back.

Those are two implementations of one format, and the entire evidentiary value of the panel rests
on them being the same object. This report is the five defects that separated them, found in the
order they hid behind each other, and the measurement that closes each one.

None of the five was visible from any accuracy number, and none was reachable from a laptop. The
first three needed the *genuine* `Lfm2MoeExperts` rather than the synthetic copy the unit tests
use. The fourth needed 4.4 GB actually written to disk, because it is the writer and the reader of
one format disagreeing and nothing smaller than a real export puts both in the same run. The fifth
needed a version of `transformers` this repository's own gates do not install.

There is a sixth item, and it is not a defect in this code. It is a claim this report made in its
own §6 — true of the model it was measured on, false at the scale it was then used to license,
and copied into five places before anyone re-measured it. §8.

Nothing here changes an arm's score. What it changes is whether the score describes something
downloadable.

---

## 1. Rank 3 is not a bank

`resolve_target` decides what a name in the bit map points at: a `Linear`, an `Embedding`, or a
bare `nn.Parameter` of rank 3 that is a batched expert bank. The rank-3 test was the whole test.

LFM2.5-8B-A1B is a hybrid: **18 of its 24 layers are short convolutions**, attention only at
layers 2, 6, 10, 14, 18 and 21. An `nn.Conv1d` kernel is `[channels, 1, width]` — rank 3, and
nothing whatever to do with experts. Slicing one "per expert" yields a `[1, 3]` strip of a
convolution and calls it a weight matrix.

The fix is that a bank now has to look like a bank in more than rank: the parameter's owner must
not be a `Conv1d`, and the last two extents must be a plausible `[out, in]`. The refusal names the
conv rather than the bank, which matters because the previous message would have sent anyone
reading it to the wrong subsystem.

On the genuine architecture, at tiny scale:

```
rank-3 parameters: 6 banks, 2 conv kernels
  model.layers.1.feed_forward.experts.down_proj (6, 64, 96)
  model.layers.1.feed_forward.experts.gate_up_proj (6, 192, 64)
  model.layers.0.conv.conv.weight (64, 1, 3)
  model.layers.2.conv.conv.weight (64, 1, 3)
[PASS] a genuine bank resolves
[PASS] a conv kernel is refused by its own reason
[PASS] and not by the bank's
```

This is the first of the three, and it is the one that made the other two reachable: until the
resolver stopped choking, no forward ever ran.

---

## 2. The dispatch that stopped calling the loop

The design of `DynQuantExpertBank` rests on one property of the parent, counted in
[the mixture report](phase4-text2sql-mixture.md#12): of the 52 `*Experts*` classes in transformers
5.14.1 that hold their experts as one `nn.Parameter`, **49 reach an expert by indexing it**, so a
module registered under the parameter's name can answer `bank[e]` and no architecture needs
enumerating.

That count is of what those classes' `forward` methods contain. On 5.14.1 it is not what runs.
`transformers.integrations.moe` decorates each of them with `@use_experts_implementation`, which
**replaces** `forward` (`moe.py:575-576`) with a dispatcher reading `config._experts_implementation`
against `ALL_EXPERTS_FUNCTIONS = {"deepgemm", "batched_mm", "grouped_mm", "sonicmoe"}`. The indexing
loop is only the `eager` fall-through to `original_forward`. The **default is `grouped_mm`**
(`modeling_utils.py:2099`), which routes to `_grouped_linear` → `_grouped_mm(input,
weight.transpose(-2, -1), offs=offs)` at `moe.py:374`.

So the packed bank was installed correctly, passed every test, and died at its first real forward
*inside transformers*:

```
AttributeError: 'DynQuantExpertBank' object has no attribute 'transpose'
```

A module is not a tensor, and the fast path wants the tensor whole.

### The fix, and why it is not a preference

`pack_model` and `_process_model_before_weight_loading` now both call `use_eager_experts(model)`
when they install a bank — through the model's own `set_experts_implementation`, not by writing the
config attribute past it — and `PackReport` records what it moved from, so the summary says so
instead of the operator discovering it at inference time.

Moving dispatches is only free if the dispatches agree, so that was measured rather than assumed.
On the genuine tiny LFM2-MoE in fp32, against the unquantized model:

| route | max abs Δ from fp32 |
|---|---|
| encoder, `grouped_mm` | 0.0100791 |
| encoder, `eager` | 0.0100791 |
| packed, `grouped_mm` | *refuses:* `'DynQuantExpertBank' object has no attribute 'transpose'` |
| packed, `eager` | 0.0100791 |

and directly between routes, **1.78814e-07** — one part in 56 000 of what quantization itself moves.
The dispatch is not a numerical choice. What it *is* is a speed choice: `grouped_mm` is the fast
path, one launch for all experts, and eager is a Python loop. That gap is exactly the P8 kernel's
job, and the right way to close it is now visible: register a DynQuant grouped GEMM **into**
`ALL_EXPERTS_FUNCTIONS` alongside those four, rather than fighting the dispatcher. `QuantTensor.rows()`
already returns the row band a grouped GEMM addresses by.

Five mutations of the switch were introduced — never switching, switching without recording it,
switching unconditionally, writing the config past the setter, and leaving the load path out — and
all five turned tests red.

---

## 3. One format, two encodings

The last one is the one that would have survived publication.

`pack_model` and `quantize_model` are the artifact and its yardstick. Given the same weights, the
same width and the same group size they must produce the same numbers, or the panel's accuracy
belongs to a model nobody can download. On the real class, in fp32, they did not:

```
max|bank[e] - encoder[e]| over every expert = 0.00901368
max|packed - encoder| (logits)              = 0.00167
(quantization itself moved the logits         0.0101)
```

16% of the quantization effect, in the *weights*, between two functions that are supposed to be one
function.

### Three copies of a rule, agreeing on everything anyone ships

The dtype scales and offsets are stored in was decided in three places:

| caller | rule | fp32 weight → |
|---|---|---|
| `runtime/linear.py` (packer) | the weight's own dtype | **fp32** |
| `quant/checkpoint.py` (exporter) | fp16 unless already fp16/bf16 | fp16 |
| `quant/quantizer.py` (encoder) | passed nothing; took `from_dense`'s fallback | fp16 |

On fp16 and bf16 all three agree. Every model anyone ships is fp16 or bf16. The disagreement could
only ever surface on an fp32 model, which is why it waited for a CPU test harness built around the
genuine architecture — and it surfaced there as a number that looked like a kernel bug.

This is the fourth instance in this project of *a second copy of a registry agrees until it
doesn't*, and the most expensive so far, because the two copies were the two things being compared
against each other.

### Which rule wins, and why it is not a numerical question

Settled on **16-bit metadata**, in one `storage_dtype()` that all four call sites share.

Not because fp16 is more accurate — it is strictly less — but because
[`allocate/budget.py`](../../packages/dynquant-core/src/dynquant/allocate/budget.py) prices every
quoted bit-width in this project against `metadata_bits: int = 16` ("an fp16 scale and an fp16
offset per 128"). fp32 scales put a model **0.25 bits/weight above its own manifest** at group 128.
The manifest is the number the method is judged on, and 166 tests already enforce the packer
against it — the first attempt at unification, on the weight's own dtype, turned all 166 red and
that is what identified the budget as the fourth copy of the rule.

### What that costs, and the one place it has to be paid

The objection to fp16 metadata is real but narrower than it looks. `QuantTensor.dequantize()`
returns the scale dtype when it is not told otherwise, so fp16 metadata in an fp32 model means fp16
weights entering an fp32 graph and `F.linear` raising on the mismatch.

It does not apply to a packed Linear: `ops.quantized_matmul` does `qt.dequantize(dtype=x.dtype)` and
follows the **activation**. It applies to exactly one caller — `DynQuantExpertBank.__getitem__`,
which is asked for a weight before any activation exists and so has nothing to follow.

That one is told instead. Both constructors know the dense weight's dtype (`from_parameter` at pack
time, `_shell` at load time) and pass it; the bank holds it in a zero-element **non-persistent
buffer**, so `nn.Module._apply` carries it through `.half()` and `.float()` the same way it carries
the scales, and it stays out of every `state_dict` — a bank loaded into an fp32 model should serve
fp32 whatever the model it was written from computed in.

### After

```
max|bank[e] - encoder[e]| over every expert = 0
packed(eager) vs encoder(eager)             = 0
packed(eager) vs encoder(grouped_mm)        = 1.78814e-07
```

Exact, not within a tolerance — the two paths are the same arithmetic on the same inputs, so the
honest bar is zero and any gap at all is two rules again. Four new tests hold it there; six
mutations were introduced across `storage_dtype`, the bank's constructor, `__getitem__`, the
`.to()`-following buffer, the whole-bank escape hatch and the load skeleton, and all six turned
tests red.

---

## 4. An artifact its own reader refused

The first three defects were found on a 4-layer, 6-expert, fp32 stand-in for the real class. The
fourth could not have been: it needs a genuine export, and the run that produced one wrote 4.4 GB
and then refused to read it.

```
== export ==
  wrote /workspace/runs/s4/export/dq_4b in 1394s
  on disk: 4,397,930,240 B in 2 shard(s)
  [PASS] the artifact lands within 1% of what the allocator priced
== the encoder, in place, as the yardstick ==
  encoded in 1493s
== load the artifact back ==
  [PASS] the config declares dynquant -- dynquant
Traceback (most recent call last):
  ...
dynquant.errors.DynQuantError: 'model.layers.10.feed_forward.gate' is a Lfm2MoeTopKRouter,
which owns a weight but is not a Linear or an Embedding, so the packed runtime has no
forward to replace
```

Twenty-three minutes of encoding, a size check that passed, and then a refusal that is *correct
about its own side*. The packer's rule is **a tensor exists to pack**. The runtime's rule is **a
module exists whose forward can be replaced**. Each is the right rule for its own half of the
format, they differ on exactly one class of target, and nobody owned the difference — the fifth
time in this project that two copies of one registry agreed until they didn't, and the worst shape
it has taken, because the disagreement does not surface as a wrong answer but as an artifact
rejected by its own reader.

### Which side moves

Not the writer. An `Lfm2MoeTopKRouter` owns `[num_experts, hidden]` and its parent calls
`F.linear` on the parameter, so there is no submodule to put anything in front of and the
memory-side rule cannot be satisfied by any module at all. Refusing on the write side would mean
the exporter silently skipping 22 tensors the allocator had already priced at 8 bits, and the
manifest's byte total would stop being the artifact's byte total — the one property a
matched-byte panel cannot lose.

So the loader moved. A target that satisfies the disk contract and not the memory contract is
dequantized once at load and held dense (`RestoredWeight`). On LFM2.5-8B-A1B that is 22 routers,
**2.9 MB of 4.4 GB** — disclosed rather than absorbed, because it is real: those routers occupy
bf16 in memory while occupying 8 bits on disk.

### The guard, and where it has to sit

`_refuse_what_no_loader_reads` runs over every name in the map before `out.mkdir`, and it *asks the
loader's own resolver* rather than restating the loader's rule, so there is still exactly one rule
available to be wrong. Before `mkdir`, because a refusal that leaves a directory behind is a
refusal someone will later mistake for a checkpoint.

Its test uses two names that genuinely have no reader on this architecture — a bare 2-D
parameter, and a `Conv1d` kernel, which is rank 3 like a bank and indexed by nothing, and which is
18 of this model's 24 layers rather than an exotic case — and asserts `not out.exists()` after
each refusal. Seven mutations of the restore path and two of the pre-flight, all nine red.

### What widening a loader costs downstream

A restored router is neither a packed module nor a dense `Linear`, and its weight is a buffer, so
it fell through every pass of the byte accounting at once and was counted **nowhere**. That is the
tied-embedding error running backwards: the same kind of walk that once double-counted a shared
table can also miss a tensor no module owns. `packed_bytes` now makes three passes — packed
modules, dense `Linear`/`Embedding`, and bare parameters on modules of any other class — and
reconciles against the model's own total rather than against the sum of the two it can see.

The general form is worth stating because it will recur: **when you widen a loader, check every
accounting pass it feeds.** A new kind of object is a new blind spot in every walk that enumerates
by type.

---

## 5. The tie `transformers` records twice

The artifact then loaded, and died at the very end of a load that had otherwise fully succeeded.

```
AttributeError: `lm_head.weight` is not an nn.Parameter
  ...
  transformers/modeling_utils.py, in mark_tied_weights_as_initialized
  transformers/modeling_utils.py, in _finalize_model_loading
```

A packed embedding registers no `weight`; the tied head reads its buffers. `tie_weights` was made a
no-op on the instance for exactly that reason, and until 5.14.1 that was the whole of the tie —
correct on 4.53.2 and on 5.10.1, both of which this campaign also runs. v5 records the same fact a
*second* way, in an `all_tied_weights_keys` mapping built at `post_init`, and
`mark_tied_weights_as_initialized` hands every name in it to `get_parameter`.

Silencing one reader of a fact does not change the fact, so the fact is corrected: entries naming
tensors the packed model no longer owns are dropped. Deliberately **pruned and not emptied** — a
model that ties something other than its head keeps that bookkeeping — and **both sides of each
pair are checked**, because the crash is on the target and the re-tie is on the source, so an entry
is unusable if either name has gone. `remove_duplicate=False` on both walks, since a tie
`transformers` has already established hides one of its two names from the default iteration, and
dropping an entry for being invisible rather than for being absent would untie a live pair.

Two things this cost, both worth naming.

**A green local gate is a statement about the installed dependency.** All four gates passed here on
`transformers` 4.53.2, which has no `mark_tied_weights_as_initialized` at all; three end-to-end
tests went red only on the box under 5.14.1. `from_pretrained` calls what the installed version
has, so an end-to-end test can only catch what the installed version does. The new test therefore
*seeds* the mapping — on v5 it overwrites v5's own value with itself, below v5 it reconstructs
what v5 would have built — and asserts the invariant wherever the suite runs, then calls the
real method where one exists. Four mutations, all red; `64 passed` on the box under 5.14.1, against
`3 failed, 60 passed` before the fix.

**A buffer is not a parameter.** The first version of that test kept its surviving pair as two
buffers, and went red on the box for a genuine reason in an unintended place: `get_parameter`
refuses a buffer outright. An entry the pruning *keeps* has to name something `transformers` can
resolve as a parameter — a constraint on the test's fixture, and a real constraint on any future
scheme that would keep a tie entry pointing at packed storage.

---

## 6. What the genuine architecture says now

End to end, in fp32, on `Lfm2MoeExperts` / `Lfm2MoeTopKRouter` built from a real `Lfm2MoeConfig`
under transformers 5.14.1 — resolve, pack, dispatch, forward, byte-count, export, `from_pretrained`,
compare:

```
[PASS] the expert holder is the genuine class -- Lfm2MoeExperts
[PASS] a genuine bank resolves / a conv kernel is refused by its own reason
[PASS] every bank is now a module -- 6/6
[PASS] every conv kernel is still a tensor -- 2 kernels
[PASS] the parameter was deregistered, not shadowed -- _parameters=[]
       _experts_implementation after packing: 'eager' (was 'grouped_mm')
[PASS] the packer and the encoder encode the same bank identically -- 0
[PASS] the packed bank answers the real forward like the encoder does -- matched=0
[PASS] changing dispatch is small against changing the weights -- 1.79e-07 vs 0.0101
[PASS] packing shrank what the byte counter sees -- 606,720 < 1,726,464
[PASS] nothing is counted twice or dropped -- 207,360 + 399,360 == 606,720
[PASS] the loader put a packed bank where the checkpoint said one was -- 6/6
[PASS] a bank survives the trip to disk and back exactly -- 0
```

**The honest limit, and the line that does not survive it.** This is the genuine *class*, at tiny
*scale*: 4 layers, 6 experts, fp32. The 8B has 44 banks, 24 layers, bf16 and a tied embedding, and
it is the bf16 that makes the scale-dtype defect invisible there — both rules return bf16, so the
8B was never mis-encoded. §7 puts the 8B through the export-and-reload path and discharges the
format claim outright.

One line above does not make the trip, and it is worth flagging where it is made rather than only
where it is corrected. `changing dispatch is small against changing the weights -- 1.79e-07 vs
0.0101` is an accurate measurement of a one-layer model and a false generalisation of one. At 24
layers a top-k router converts a small numeric difference into a discrete one and depth compounds
it. §8.

---

## 7. The real model, end to end

Everything above is the genuine *class* at tiny *scale*. LFM2.5-8B-A1B is 24 layers, 32 experts,
bf16, tied, and 18 of its 24 layers are convolutions. It has now been through the whole path
— price, export, reload, compare — on CPU beside the running panel.

**The bytes reconcile three ways, and each way answers a different question.**

```
priced     4,397,666,304 = packed 4,397,445,120 + conv 221,184
on disk    4,397,930,240 = priced + rank-1 205,056 + container 58,880
in memory  4,400,549,888 = packed + dense 3,104,768  (conv 221,184 + restored routers 2,883,584)
```

The allocator's price is what the manifest promises, the disk figure is what a downloader pays, and
the memory figure is what the model occupies while answering. They differ by 264 KB and 2.9 MB, and
both differences are named rather than absorbed: the rank-1 tensors the budget did not price, the
safetensors container, and the 22 routers that live at 8 bits on disk and bf16 in RAM because no
packed forward can stand where they stand (§4). The first of those is now priced — see §10,
where a 205 KB rounding error turns out to be 91.5% of a differently-shaped model.

**Every tensor in the map came back exactly.**

```
     1 x DynQuantEmbedding
    44 x DynQuantExpertBank
    66 x DynQuantLinear
    22 x Lfm2MoeTopKRouter
  compared 133 of 133; worst 0 at (none)
  [PASS] every module in the map was compared -- 0 unreachable: []
  [PASS] the download equals the thing the panel measured -- 0
```

The count matters as much as the zero. An earlier version of this comparison reported
`compared 44 ... worst 0` and read as a clean bill of health. It had walked the map with an
`elif hasattr(module, "dequantize")` fallback, and only `DynQuantExpertBank` carries that method,
so every `DynQuantLinear`, the packed embedding and all 22 restored routers went unchecked —
"worst 0" was a statement about the banks. Reaching the rest needs no production change, since
every packed module exposes `weight_qt` and `QuantTensor` dequantizes; what it needs is for the
comparison to assert its coverage against the map's own length. **A comparison that does not count
what it compared cannot tell you what it skipped.**

**And the logits are identical, once both sides are asked the same question.**

```
max|loaded(eager)       - encoder(eager)|      = 0        <- the format
max|encoder(grouped_mm) - encoder(eager)|      = 8.4375   <- the dispatch
max|loaded(eager)       - encoder(grouped_mm)| = 8.4375   <- what the first run reported
```

Exactly zero, not within a tolerance. The first run of this comparison reported the third number
and attributed it to the format, which was the wrong half of an experiment that had varied two
things at once: a packed bank can only be served by `eager`, so the loaded model is forced there,
while `post_init` leaves the encoder on `grouped_mm`. Holding dispatch fixed collapses the gap to
nothing. The second and third numbers coming out identical is the algebraic consequence of the
first being exactly zero, not a second piece of evidence.

---

## 8. The dispatch is not free, and it sits on the panel's main axis

`use_eager_experts` moves a packed model to `eager` because a packed bank answers `bank[e]` and
nothing else. It is called from `pack_model` and from nowhere else, so the encoder path never moves,
and the two halves of this campaign run different arithmetic over the same weights:

```
the panel's dq arms       encoder in place, dispatch left where post_init put it: grouped_mm
what a person downloads   packed, dispatch forced to eager
```

Three places licensed that as free, all from the single tiny-scale number above: this report's
§6, the docstring on `use_eager_experts` (*"eager and grouped_mm agree to 1.8e-07 on a tiny
LFM2-MoE"*), and `_ENCODER_REMEDY` (*"both give the same accuracy"*). It is repeated in §12 of the
mixture report and row 20 of the index, and guarded by a test called
`test_the_two_dispatches_agree_so_moving_between_them_is_free` -- one MoE layer, fp16, whose
"grouped" path is a hand-written dense-and-mask transcription that never calls
`torch._grouped_mm`. The mechanism below is structurally absent from that fixture, so the test
could not have caught this and was never going to. It is now
`test_the_fixtures_two_dispatches_are_transcriptions_of_each_other`, which is what it checks.

**What the 8B says.** Teacher-forced over every gold position of 24 real text-to-SQL items, four
arms from one load, dispatch and quantization varied separately:

```
A vs B   bf16: grouped_mm against eager      (dispatch, unquantized)             0.9876
C vs D   dq:   eager against grouped_mm      (download against panel)            0.9876
B vs C   eager: bf16 against dq              (quantization, the yardstick)       0.9567
A vs D   grouped_mm: bf16 against dq         (quantization as the panel saw it)  0.9660

dispatch disagreement 0.0124 against quantization 0.0433    = 0.29x
max|peak logit|       dispatch 2.0 against quantization 7.25 = 0.28x
```

Teacher-forced rather than free generation, because a greedy prefix diverges once and then
compounds, which measures the divergence point repeatedly instead of measuring a rate. Two
independent quantities land on the same ratio, which is reassuring about the measurement and
unhelpful about the conclusion. Against the panel's own numbers — bf16 84.26%, `dq_4b` 82.71%,
`gptq_4b` 82.07% — quantization costs **1.55 points** and the margin under discussion is
**0.64**. A dispatch effect at 0.29x of quantization is worth roughly **0.45 points** if accuracy
moved the way token agreement does. That is the same order as the margin.

**And it falls on exactly the wrong axis.** GPTQ and AWQ have no `*Experts` module left to
dispatch: `llm-compressor` linearises all 22 banks into 2,201 `Linear`s (`banks_before: 22,
banks_after: 0, linear_share: 1.0`), which *is* the eager arithmetic. bf16 and DynQuant kept their
banks and ran `grouped_mm`. So the panel compares **dq(grouped_mm) against gptq(eager)**, and the
dispatch difference is confounded with the quantizer difference on precisely the comparison that
carries the headline.

**What this does not establish.** The probe measures disagreement, not which side is more accurate,
so the sign is unknown; and 24 items of teacher-forced token agreement does not map linearly onto
exact-match over 12,000 generations. It does not show the margin is wrong. It shows the margin
cannot be defended without measuring it. A prior threshold of 0.25x was written into the probe
before it ran and 0.29x cleared it, but 0.29-against-0.25 is a coin flip on an invented bar; the
0.45-against-0.64 comparison is the argument.

**A note on the input, because it changed the answer by 60x.** An earlier version of this probe ran
on `arange(1, 65)` — arbitrary token ids — and found the two dispatches selecting different
experts in **79%** of slots, against 1.24% of tokens here. On out-of-distribution input a top-k
router is closer to uniform over its 32 experts, so selection is maximally unstable and the number
measures the input rather than the model. The synthetic run is worth keeping for one thing only: at
layer 2, the first MoE layer, fed by two dense layers, the two dispatches agree **256/256 with
routing-weight L1 exactly 0**, and disagreement then grows monotonically with depth to 7% by layer
23. That is the cascade hypothesis passing a test that could have killed it — had layer 2
disagreed, something upstream of the experts would have differed and the whole explanation would be
wrong.

**The fix is singular.** GPTQ and AWQ cannot be moved to `grouped_mm`; there is nothing left to
dispatch. `eager` is therefore the only dispatch all seven arms can share, and it is also the one a
downloader runs, so one re-run decontaminates the panel and validates the artifact claim at once:
the three arms that kept their banks — `bf16`, `dq_4b` and `dq_3b` — re-scored on
`eager` at the same anchors, the same 12,000 items, the same seed, paired against the stored
per-item hits so the comparison is a McNemar test and not two independent rates. The four
linearised arms need nothing: they already computed eager.

**And the code fix, so the next panel cannot straddle it.** A re-run repairs one campaign; what
allowed the campaign to be built this way was that no record said which computation produced its
number. `dynquant eval` now pins the dispatch to `eager` after the model is resolved — in `run`,
not in `_load_runtime`, because a baseline arrives through `model=` and a ceiling applies no map,
so that is the only point every arm passes through — and writes `{found, ran}` into the record.
`EXPERTS_PAIRING_FIELDS` then refuses to pair two arms that ran different arithmetic, exempt on
absence because a dense model has no dispatch and never will. `--experts-impl auto` restores the
model's own choice, which is what measuring the dispatch itself needs and what a panel must not
use. Both phase-4 drivers state the flag on every scoring command rather than inheriting it, for
the reason `eval_flags` already states the decode budget: a default that moves under a driver is a
difference between arms that the records still describe as shared.

**The general form.** A measurement whose conclusion holds at one scale and fails at another is not
a wrong measurement, and calling it one hides the actual failure. `1.79e-07` was true of a one-layer
model. What made it load-bearing was carrying it into a docstring, a remedy string, two reports and
an index row without re-measuring it on the object the claim was being used about — and a test
whose name asserts the general claim while its body asserts a hand-written fixture's fidelity to
itself. A one-layer fixture cannot exhibit a cascade. It can only be honest about not trying.

---

## 9. What this does and does not say about the panel

The format claim is now stronger than it was, and the accuracy claim is weaker.

**Stronger.** Before this work, "DynQuant 4-bit scores X on text-to-SQL" was a statement about a
transient object: the packed artifact could not be built for this architecture (§1), could not run
if built (§2), would not have matched on a model in fp32 (§3), was rejected by its own reader
(§4), and crashed at the end of a load that had otherwise succeeded (§5). It is now a statement
about a checkpoint whose bytes, weights and logits have each been checked equal to the thing that
was scored — 133 of 133 tensors at exactly 0, and logits at exactly 0.

**Weaker.** "Asked the same question" is the qualifier on that last zero, and until the eager arms
land it is doing real work. The panel's dq rows describe a computation the download does not
perform, and the four landed arms are not all on the same dispatch as one another.

The remaining gap after that is speed, not correctness, and it is the one the P8 kernel exists to
close. Worth noting what closing it would also do: a DynQuant grouped GEMM registered into
`ALL_EXPERTS_FUNCTIONS` serves a packed bank on the default dispatch, so nothing would need to move
at all, and this entire section becomes moot rather than merely resolved.

---

## 10. The denominator was smaller than the file

Section 7 named three differences and absorbed none of them. One has now been closed, and closing
it turned out to be a larger change than the 205,056 bytes that prompted it.

`classify_model` records every tensor it declines to classify in `ModelGraph.skipped`, with a
reason. Nothing then read that dictionary. `total_params()` summed the quantizable modules and the
ones floored at compute dtype; `Budget.from_target` divided by that sum and subtracted a fixed cost
built from the same two groups. So the refused tensors were documented and unpriced — present in
the artifact, absent from the arithmetic describing it.

**The size of the error depends entirely on what got refused, and the range is four orders of
magnitude.**

| what is refused | on which model | share of the model |
|---|---|---|
| 1-D norms and biases | LFM2.5-8B-A1B | 205,056 B of 4.4 GB, 0.005% |
| a batched expert bank, wrong orientation | a `gpt_oss`-shaped MoE | **91.5% of quantizable parameters** |

The first is why this looked like a rounding error and sat on the list as "price the rank-1
tensors". The second is what it actually is. `_expert_bank` refuses a bank whose input axis is not
last, because grouping along the wrong axis still round-trips and still reports a plausible
reconstruction error — there is no later symptom, so the refusal is correct. But a refused bank
is the *entire* MLP of every layer, and on an LFM2-class model that is 91.5% of the quantizable
parameters. A budget that leaves it out of the denominator is not off by a header; it is sizing a
different model.

**What changed.** `skipped` becomes `dict[str, SkippedTensor]` carrying `num_params` and `tied_to`
alongside the reason. `total_params()` gains a third term, `floor_cost_bits()` and
`Budget.from_target` charge the refused parameters at the 16-bit floor, and `dynquant inspect`
writes the count into the manifest beside `unquantized` rather than inside it — they are both
dense on disk, but one is a role's floor and the other is a tensor the graph declined to have an
opinion about, and on the MoE above the difference is most of the model.

All three refusal sites file through one recorder that gives the parameters to the first name and
records a `tied_to` on the rest. Two modules sharing one norm would otherwise be counted twice,
which is the tied-embedding error running in the opposite direction: that one made a shared tensor
27% of a model twice over, this one would make a shared norm 32 KB twice over. The mechanism is the
same and only the magnitude is forgiving.

**One term is still named and still not priced.** The safetensors container, 58,880 bytes on this
model, is a function of the tensor *names* and offsets — which do not exist until the allocation
being budgeted has been made. It cannot be priced from inside `Budget.from_target` without
allocating first, so a target is hit to within a header, and the module docstring says so rather
than letting a reader discover it from a `du` that is 0.001% off.

**A test that had been reading an accident.** `test_a_zero_score_is_cut_to_the_minimum_for_free`
pins the reason `percentile_ranks` maps to an *open* interval: a score of exactly zero prices every
cut as free, and on Qwen3.5-2B that put 20 of 187 modules at 2 bits and scored the model at 20.8%
against 65.1%. The test ran at an 8.0-bit target, where the allocator's downgrade pass never
executes at all. What it was actually observing was the *upgrade* pass, which accepts a zero-value
move when no priced move can afford the slack that remains — so whether the zero-scored module
finished below its peers depended on the size of that remainder. This change moved the remainder by
0.09% of the budget and the assertion flipped, with nothing about the handling of a zero having
changed.

That is worth recording as its own small lesson. **A test whose subject is a downgrade should not
be run at a target where nothing is downgraded.** It now runs at 4.0 — the target the incident in
its own docstring happened at — where the victim is cut *through its floor* to the minimum and is
the only module in the model there. Three assertions instead of one, and all three are about the
mechanism rather than about the leftovers.

**What this does not change.** The panel's four landed arms were allocated under the old
accounting, and on this model the correction is 205,056 bytes against a 4.4 GB anchor: 0.005%,
against a 0.1% match tolerance. The maps are not invalidated and are not being re-derived. The
eager re-score of section 8 reuses them through `--rescore`, precisely so that the dispatch is the
only thing that moves between the two measurements — this change is the second one that would
otherwise have ridden along.
