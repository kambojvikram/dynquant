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

And a seventh, which is the same writer pointed the other way. Four of the six promised variants
cannot be written in the container their own recipe produced. They can be written in this one,
because a foreign grid is a grid: the codes carry across unchanged and only the notation for the
offset differs. §12.

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

**And it falls on exactly the wrong axis.** GPTQ and AWQ have no *batched* bank left to
dispatch: `llm-compressor` linearises all 22 banks into 2,201 `Linear`s (`banks_before: 22,
banks_after: 0, linear_share: 1.0`), so the grouped kernel has no tensor to take and the arithmetic
is the loop. The `*Experts` modules and the config field both survive it, which matters for what the
records can say and is taken up in §11. bf16 and DynQuant kept their
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
linearised arms need nothing: they already computed the loop, which their own `banks_after: 0`
records and §11 separates from the distinct, weaker claim that the loop and `eager` agree
numerically at this scale.

**And the code fix, so the next panel cannot straddle it.** A re-run repairs one campaign; what
allowed the campaign to be built this way was that no record said which computation produced its
number. `dynquant eval` now pins the dispatch to `eager` after the model is resolved — in `run`,
not in `_load_runtime`, because a baseline arrives through `model=` and a ceiling applies no map,
so that is the only point every arm passes through — and writes `{found, ran}` into the record.
`EXPERTS_PAIRING_FIELDS` then refuses to pair two arms that ran different arithmetic, exempt on
absence because a dense model has no dispatch and never will — an exemption that also covers a
record written before the field existed, which is the one straddle it cannot see and §11 prints
instead. `--experts-impl auto` restores the
model's own choice, which is what measuring the dispatch itself needs and what a panel must not
use. Both phase-4 drivers state the flag on every scoring command rather than inheriting it, for
the reason `eval_flags` already states the decode budget: a default that moves under a driver is a
difference between arms that the records still describe as shared.

**And the clock says the same thing, on a different pair.** This section's title claims the
dispatch is not free and every measurement under it is about accuracy. The panel's own timings
supply the other half, and they were already paid for:

```
arm       dispatch      weights at run time            eval seconds   s/item   vs bf16
bf16      grouped_mm    bf16                              10,307.6     0.859     1.00x
dq_4b     grouped_mm    encoded back to compute dtype     10,011.2     0.834     0.97x
gptq_4b   the loop      compressed-tensors, 4-bit         19,805.0     1.650     1.92x
awq_4b    the loop      compressed-tensors, 4-bit         23,350.9     1.946     2.27x
```

Same 12,000 items, same batch size, same decode budget, same GPU, one arm after another. The two
that kept their banks scored in under three hours each; the two linearised into 2,201 `Linear`s took
five and a half. **The confound is that those are the same two arms.** Linearisation and on-the-fly
dequantization arrive together and no arm in this panel has one without the other. `dq_4b` cannot
help, because `apply: encode` writes the reconstruction back in the compute dtype, so it runs plain
bf16 matmuls and pays no dequantization at all.

**And the thing I first said bounded that confound does not bound it.** The revision of this
paragraph written a day earlier offered the 18% between `gptq_4b` and `awq_4b` as the ceiling —
"kernel choice alone, at identical dispatch and identical shapes," so differences of that order
exist and are an order below the gap. The two manifests say otherwise. Both arms are 4 bits at
`group_size` 128 over the same 2,201 modules for the same 4,399,629,312 bytes, which is not two
kernels; it is one. What actually differs is that AWQ is asymmetric where GPTQ is symmetric, so one
dequant subtracts a zero point the other does not, and that AWQ's smoothing moved every one of the
2,201 weights (`max_weight_delta: 2.89`), which changes what the model writes and therefore how long
it writes for against a 1,024-token cap. So the 18% prices one dequant scheme against another, plus
the length difference between two sets of weights. **A difference inside dequantization is not a
bound on dequantization**, and dequantizing against not dequantizing at all is the confound. The
number was real and the inference from it was not.

So the linearised arms cost 1.9—2.3x the banked ones and this panel cannot say how much of that is
the 22 grouped matmuls becoming 22 × 32 = 704 module calls per forward. What can say is two
measurements that were already on the list for other reasons, and both are cheap because both have
identical weights on either side and therefore no dequantization anywhere in them: `eager` against
`grouped_mm`, which the re-score produces, and the loop against `eager`, which is 24 teacher-forced
items on the bf16 model. Between them they isolate exactly the two halves that arrive together here.

**And this is the loop against `grouped_mm`, which is not the pair the rest of §8 is about.** The
accuracy number above is `eager` against `grouped_mm`. The clock here is the loop against
`grouped_mm`. Three names, three pairs, and this report has already been caught once carrying a
number across one of those boundaries. The missing clock is `eager` against `grouped_mm` — and
the re-score produces it for nothing, because `bf16`, `dq_4b` and `dq_3b` will each have been scored
twice over the same 12,000 items with the dispatch as the only difference. Whatever that costs is
the number a grouped kernel registered into `ALL_EXPERTS_FUNCTIONS` would be recovering, which is
the first time this campaign will have priced that work instead of asserting it is worth doing.

For nothing, but not for free: the re-score overwrites the records it re-scores, so the sequence is
`cp -a panel panel_grouped_mm` and then `dispatch_delta.py --before panel_grouped_mm --after panel`.
That script exists because a measurement contingent on remembering a `cp` is a measurement that gets
lost, and it carries one refusal worth naming here. If both records report the same dispatch it
prints no row. Two runs of the same computation pair into a delta of exactly zero at p = 1.0, and
that table is a clean demonstration that the dispatch is free — which is the claim this section
retracted, reproduced by the fix having silently not run. The zero is the most convincing output the
tool could produce and the least trustworthy, so it is refused instead of printed.

It also re-prices the re-score. Budgeting three arms at 2.8 hours each assumes `eager` is as cheap
as `grouped_mm`, and the premise of this whole section is that it is not: `eager` indexes one expert
at a time, like the loop. It should sit below the loop, since it indexes inside one batched tensor
rather than across 2,201 separate modules, so the bracket is **8.5 to 17 hours** and the low end is
the one with no argument behind it.

**Three bits is slower again, and that is a third thing.** `gptq_3b` is scoring at roughly 3.8
s/item against `gptq_4b`'s 1.65 — another 2.3x, at the same dispatch, the same runtime, the same
decode budget and the same batch size. Either `compressed-tensors` unpacks 3-bit more expensively
than 4-bit, or the 3-bit model has degraded into longer generations against the 1,024-token cap. If
it is the second then the cost of evaluating a baseline is partly a function of how badly that
baseline was hurt, which is a thing to know before budgeting a panel by arm count.

This paragraph also said the landing record would separate them, naming accuracy and
`detail.errored`. The four records already in hand say it will not. `errored` counts SQL that failed
to execute and it does not order the clock: `gptq_4b` has 187 and is the faster of the linearised
pair, `awq_4b` has 118 and is the slower. `unparseable` is 0 on all four and `unfinished_reasoning`
is 0 on all four. Nothing recorded is a length proxy, which is a gap in the record rather than an
answer.

**And the second hypothesis is easier to satisfy than it sounds.** Decoding is greedy and batched at
32, and a `generate` call runs until every sequence in its batch has stopped, so each batch costs
what its *longest* generation costs. A 3-bit model does not have to write longer answers on average
to double the clock — it has to fail to stop on one item in thirty-two. `unparseable: 0` does not
rule that out either: a generation truncated at the cap can still carry extractable SQL earlier in
its text, so it scores like any other item and says nothing about how long it ran.

Separating them wants a fixed-length teacher-forced forward on the 3-bit and 4-bit weights, where
nothing is generated and unpacking is the only thing left. The panel cannot supply it: `run` is
"quantize and score in one process (no checkpoint)" and no baseline weights survive the arm that
made them, so the price of asking is a re-quantization — about 32 minutes for GPTQ, 47 for AWQ, from
their own `quantize_seconds`.

**A file written for another reason answered it first, and for nothing.** A sampler on the box polls
`panel.log` every 15 seconds and stamps each new `[text2sql] N/12000` line, which turns the panel's
own progress into an interval profile: 800 items at a time, across three arms. Record mtimes name
those arms without inference — the first run of stamps ends at 13:32:29Z and `awq_4b.json` is written
at 13:32:23Z, the second ends at 16:30:47Z against `dq_4b.json` at 16:30:43Z, the third is `gptq_3b`
and still running. And the profile is readable at all only because every arm draws the same items in
the same order: seed 0, limit 12,000, and the four landed records agree to the item on their
per-source denominators, 3,063 gretel and 8,937 wikisql. Block *k* is the same 800 questions in all
three. `experiments/phase4/rate_profile.py` does the alignment; `rate.log` and `rate_profile.json`
are in `experiments/phase4/s4_panel/`.

| block ends at | 4000 | 4800 | 5600 | 6400 | 7200 | 8000 | 8800 | 9600 | 10400 | 11200 | 12000 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `dq_4b` (banked, 4b) | 0.77 | 0.71 | 0.71 | 0.71 | 0.62 | 0.69 | 0.75 | 1.56 | 0.83 | 0.73 | 0.58 |
| `awq_4b` (loop, 4b) | 1.76 | 1.50 | 1.44 | 2.70 | 1.22 | 1.37 | 1.43 | 4.73 | 2.46 | 1.33 | 2.18 |
| `gptq_3b` (loop, 3b) | 2.78 | 3.02 | 4.26 | 2.59 | 1.37 | | | | | | |
| `awq_4b` / `dq_4b` | 2.29 | 2.11 | 2.03 | **3.78** | 1.97 | 1.97 | 1.90 | 3.04 | 2.97 | **1.82** | 3.74 |
| `gptq_3b` / `awq_4b` | 1.58 | 2.01 | **2.95** | **0.96** | 1.12 | | | | | | |

Seconds per item.

**Block index is item index, not wall-clock index**, and that is what makes the table readable rather
than merely suggestive. The arms run one after another, so block 9600 of `awq_4b` was scored between
11:09 and 12:12 and block 9600 of `dq_4b` between 15:41 and 16:02 — four and a half hours apart. A
spike landing on the same block in both arms therefore cannot be one transient on the card; it has to
be the items. And 9600 is the slowest block in both arms by a wide margin in both, with 10400 in the
top three of both. Item cost is real and large on its own: `dq_4b`'s eleven blocks here range 0.58 to
1.56, a 2.7x swing on one set of weights at one dispatch.

Which is what makes the *unshared* spikes mean something. `awq_4b` is elevated at 6400 — 2.70 against
its own median of 1.50 — and at 12000, at 2.18, where `dq_4b` sits at 0.71 and 0.58, the second of
those being `dq_4b`'s fastest block of the eleven. Two arms diverging on the same 800 items, hours
apart, is not a fixed cost. It is the arms taking different numbers of decode steps on the same
questions.

The first ratio row is the dispatch question with the bit width held fixed, and it
is not flat: `awq_4b` costs between 1.82x and 3.78x what `dq_4b` costs depending on which 800 items
are in front of it, a 2.08x swing. A fixed per-forward cost — unpacking, or 704 module calls in place
of 22 grouped matmuls — is by construction the same work on every block and would be flat. So 1.82 is
a **ceiling** on the fixed component, against an aggregate of 2.55x over the same eleven blocks,
which leaves **at least 1.40x of the linearised arms' cost as decode steps rather than dispatch or
dequantization**. The ceiling holds only while `awq_4b` never takes *fewer* steps than `dq_4b` on a
block; that is not checkable from a clock, so the script prints the condition beside the number
rather than under it. What the profile does not do is split the 1.82 between linearisation and
dequantization. It narrows the total; the two identical-weight probes above are still the only
instruments for the split.

The second ratio row settles the 3-bit question, and settles it against the hypothesis this section
opened with. `awq_4b` and `gptq_3b` are **both linearised** — the loop is on both sides — so the only
differences are the bit width and the weights. Over their five shared blocks `gptq_3b` runs 1.58x,
2.01x and 2.95x the cost of `awq_4b`, and then **0.96x on block 6400** and **1.12x on 7200**. A fixed
per-forward unpack cost cannot be negative, so a block where the 3-bit arm is the *faster* one says
its excess everywhere else is not fixed work.

**The 0.96 should not be the number quoted, though**, and the reason is in the paragraph above it.
Block 6400 is `awq_4b`'s own anomaly — 2.70 where its median is 1.50, and `dq_4b` is flat there, so
it is not the items. Divide by an arm that was slow for an unexplained reason and the quotient
inherits the reason. Drop the block entirely and the conclusion survives on 7200, where all three
arms sit at or near their own minima and `awq_4b` against `dq_4b` reads 1.97 against a median of
2.1, which is an arm behaving normally. **1.12 is therefore the ceiling worth defending**: the
fixed component of three bits over four is at most 12% of a forward, against an aggregate of 1.62x
over the five shared blocks, so at least 1.45x of the 3-bit arm's excess is decode steps. The 0.96
says the same thing more strongly and rests on less.

Two things caveat that rather than withdraw it. `gptq_3b` differs from `awq_4b` in its weights as
well as its width — a different quantizer, and AWQ's smoothing moved all 2,201 — so "three bits" is
not the only difference between the sides. But the unpack hypothesis is a claim about *width*, and it
predicts a *flat* ratio whatever the weights are; a 2.6x swing across five blocks is the prediction
failing without needing the sub-1.0 block at all. And the arm is unfinished at five shared blocks,
with up to 15 seconds of poll slop on each stamp — against blocks of 1,095 to 4,636 s, under 1.4%.
What is still worth a re-quantization is the *size* of the fixed component rather than its existence,
and 1.12 already bounds it at a tenth of a forward. That is a smaller question than the one this
paragraph was written to ask, and it is no longer on the critical path for the panel's table.

**And it puts an argument under the top of the re-score bracket, which had none.** The 8.5-to-17
hours above came from doubling: `eager` might cost what `grouped_mm` costs, or it might cost what
the loop costs, and 2x was the shape of the second guess rather than a measurement. Two things
sharpen it. The re-score runs the *same weights* on both sides — `bf16`, `dq_4b` and `dq_3b`, scored
again — so generation length is held fixed by construction and the only multiplier in play is the
fixed per-forward one, which is the component the profile bounds. And that bound is 1.82, not 2, and
it is the bound for the **loop**, which `eager` should sit under. So the re-score is at most 1.82x
the banked arms' own seconds: 10,308 for `bf16` and 10,011 for `dq_4b` with `dq_3b` still to land,
which puts the ceiling near **15 hours** and the floor unchanged at 8.5. The floor is still the end
with no argument behind it.

**And the re-score can now check that bracket rather than merely spend it.** The box's sampler
resolves its target with `ls -t` once, outside its loop, so it is pinned to the panel's log and
would not have followed the re-score — the run with the most to say about dispatch cost would have
produced no length evidence at all. `rescore_eager.sh` therefore stamps its own lines as they
arrive instead of polling one, which is strictly the better instrument here: `progress_printer`
flushes, so the stamp is the line's own time and the 15-second slop the panel's profile carries is
simply absent. What comes back is not another bound. Length is held fixed by construction on both
sides, so the per-block ratio *is* the fixed multiplier, measured against the same 800 questions —
the number 1.82 is currently a ceiling for.

**The general form.** A measurement whose conclusion holds at one scale and fails at another is not
a wrong measurement, and calling it one hides the actual failure. `1.79e-07` was true of a one-layer
model. What made it load-bearing was carrying it into a docstring, a remedy string, two reports and
an index row without re-measuring it on the object the claim was being used about — and a test
whose name asserts the general claim while its body asserts a hand-written fixture's fidelity to
itself. A one-layer fixture cannot exhibit a cascade. It can only be honest about not trying.

### The cost is now avoidable, and the panel already paid it anyway

§8 established that moving a packed model to `eager` is not free and put it on the same axis as the
margins this panel reports. The obvious next question is whether the move is *necessary*, and it is
not. `eager` was never the requirement; indexing was. `grouped_mm_experts_forward` sorts tokens by
expert, computes segment offsets, masks expert-parallel sentinels and reduces once over the k axis —
and the only step in it a packed bank cannot answer is the one that hands the whole bank to
`torch._grouped_mm`. Substituting `bank[e]` for that read leaves every other step intact.

`dynquant.runtime.experts.dynquant_experts_forward` is that substitution, registered into
`ALL_EXPERTS_FUNCTIONS` under its own name. Measured on a four-layer model at LFM2.5-8B-A1B's real
MoE geometry — hidden 2048, `moe_intermediate` 1792, 32 experts, top-4, 256 tokens, bf16, all three
dispatches verifiably selected:

| against `grouped_mm` | max logit gap | argmax tokens disagreeing |
|---|---|---|
| `eager` | 0.62109375 | **1.95%** |
| `dynquant` | **0.0** | **0.00%** |

Against a maximum absolute logit of 4.125. At fp32 the same comparison gives 1.25e-06 and 0.0, which
is the control: the bf16 gap is an accumulation-order effect and not a different function. The 1.95%
brackets the 1.24% measured on the real 24-layer model, which is what makes four layers a defensible
stand-in for this particular question.

**The mechanism is narrower than "a different order", and that turned out to matter.** The first
attempt to test it asserted that slot order and expert order give different answers, and it failed
its own vacuity guard: `torch.sum` over a bf16 axis accumulates in an fp32 accumulator and rounds
**once**, so permuting the summands is nearly harmless. What `eager` does is different in kind — it
adds each expert's contribution into a bf16 output tensor as it walks the experts, rounding to bf16
**k times**. One rounding against k roundings is the whole effect, and the test now constructs it
exactly: one expert contributing 1.0 and three contributing 2^-9 each sum to 1.0078125 in one
reduction and to 1.0 in four.

**Two things went wrong on the way, and only one of them was the code.** At a tiny geometry — hidden
64, four experts, top-2 — all three dispatches produced bit-identical logits, and an earlier draft
of the probe divided by that zero and printed `infx closer`. The temptation was to read the zero as
a no-op, and two diagnostics refused it: all three dispatches were confirmed selected and
`grouped_mm` was confirmed to reach the real `torch._grouped_mm`. The zero was real; four experts at
hidden 64 simply cannot resolve the effect. The fix was to scale *width, expert count and k* rather
than depth, because those are what a single layer's reduction sees, and to make the probe **refuse**
on `eager == grouped_mm` instead of reporting a ratio against it.

The second was a genuine bug, and the probe could not have found it. Expert-parallel sentinels were
folded into bin 0 before the segment offsets were counted. Because the bands are offsets into a
*sorted* array, widening bin 0 by rows that physically sit at the tail displaces every later band
off its own rows — a two-expert case returned expert 0's answer for both. The `wide` probe routes no
sentinels, so it was blind to this by construction and its result stands; the two-expert unit test
caught it on the first run. This is the division of labour worth keeping: the probe says the two
agree on a real model, the unit tests say *why*, and the why is where the bug was. Six mutations of
the forward are now each caught by a named test.

**What this retires, and what it does not.** It retires the condition every packed figure carried —
a packed artifact's accuracy was comparable to a bf16 number only if the bf16 side was moved to
`eager` too. A downloaded checkpoint now runs `dynquant`, which is bit-identical to the default. It
does not retire the re-score, and the reason is the linearised baselines: `llm-compressor` rewrites
a GPTQ or AWQ arm's banks into per-expert `Linear` modules, so it computes the loop whatever the
config says and there is nothing left in it to dispatch. `eager` remains the only setting all seven
arms can share.

### What the banked arms actually ran, confirming what §11 predicted

This section measures a state two earlier paragraphs already called: §8 above records that the
baselines straddle the dispatch, and §11 records that `_comparability` cannot tell a null `experts`
key from a missing one. Neither had been checked against the banked records. Doing so is worth the
paragraph because a predicted failure and an observed one are different evidence, and because the
prediction was right about the mechanism while understating the reach.

Reading the five records for their dispatch gives `None` five times, and §11's distinction decides
what that means. The key is **absent**. The panel clone is pinned at `4109dcc`, which predates `_pin_experts_dispatch`,
`use_eager_experts` and `--experts-impl` alike: nothing pinned anything, and nothing recorded
anything. The panel venv carries transformers 5.10.1, whose `ALL_EXPERTS_FUNCTIONS` holds
`{batched_mm, deepgemm, grouped_mm, sonicmoe}` and whose default is `grouped_mm`. So:

| arm | what dispatched | how it got there |
|---|---|---|
| `bf16` | `grouped_mm` | `post_init` default, never moved |
| `dq_4b` | `grouped_mm` | `apply: encode` holds compute-dtype values, so no bank is packed and nothing forces a move |
| `gptq_4b`, `awq_4b`, `gptq_3b` | the per-expert loop | linearised by `llm-compressor`; the config is not consulted |

Two consequences, pulling in opposite directions. The **bf16-to-DynQuant margin is clean**: both
sides are on `grouped_mm`, so 84.26 → 82.71 is a quantization effect and nothing else. The
**DynQuant-to-baseline margins are not**: `dq_4b` at 82.71 against `gptq_4b` at 82.07 is 0.64 points
across a comparison that also changes dispatch, and the dispatch effect on this model is 1.24% of
teacher-forced tokens. The confound has no established sign — the earlier concordance measurement
says the two dispatches differ, not which is better — so this is not a direction to correct for. It
is a reason the 0.64 cannot be read as a quantizer result until the re-score puts both sides on
`eager`.

And the pairing guard was never going to catch it, which is the part §11 called and the index row
still overstates. That row says `EXPERTS_PAIRING_FIELDS` "refuses to pair two arms that ran
different arithmetic"; it refuses to pair two arms that *recorded* different arithmetic, and here
all five recorded nothing. `_comparability` treats an absent `experts` key as an exemption, which it
has to, because a dense model genuinely has no dispatch — so these arms pair with each other and
with everything scored after them, silently. The guard is not weaker than advertised; it is that an
arm which never had the field cannot be distinguished by a field. That is what the dispatch census
in `panel_table.py` prints, and why it names the arms in the third state rather than counting them.

What the observation adds to the prediction is the reach. §8 said four arms straddle; it is five of
five, because `bf16` is on the unpinned default too. So there is no arm in the banked panel whose
dispatch was chosen rather than inherited.

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
---

## 11. What a linearised baseline still carries

§8 planned one re-run: `bf16`, `dq_4b` and `dq_3b` re-scored on `eager`, the four
`llm-compressor` arms left alone because "they already computed eager." Both halves of that
sentence turn out to need work, and only one of them is about arithmetic.

**The structural claim was wrong in its mechanism.** §8 says GPTQ and AWQ "have no `*Experts`
module left to dispatch," and `_pin_experts_dispatch` was written to return `None` for them on that
basis — a model with nothing to dispatch, filed beside a dense one. `linearize_moe` replaces
*modules*. `_experts_implementation` lives on the *config*, which it never touches.

A four-layer LFM2-MoE built from this campaign's own config, on CPU, says so directly:

| | before `linearize_moe` | after |
|---|---|---|
| `config._experts_implementation` | `grouped_mm` | `grouped_mm` |
| `set_experts_implementation` callable | yes | yes |
| modules named `*.experts` | 3 | 3 |
| non-linearised banks (`get_non_linearized_moes`) | 3 | 0 |
| per-expert `Linear`s under `.experts` | 0 | 288 |

`use_eager_experts` then returns `'grouped_mm'` and leaves `'eager'`. So a re-scored baseline would
record `{found: grouped_mm, ran: eager}` exactly like an encoded DynQuant arm, and the two would
pair — on the strength of a config field, on a model where setting it is inert because the
grouped path has no batched tensor to take. The conclusion is right and the route to it is an
accident. The operative fact §8 needed is narrower than what it wrote: there is no *batched* bank
left, so the grouped kernel cannot run, so the arithmetic is the loop.

**The numerical claim, half measured.** "They already computed eager" was an inference from
structure: both index one expert at a time. The 1.24% figure is `eager` against `grouped_mm` and
says nothing about `eager` against a `ModuleList` of `Linear`s, so the §8 re-run rests on that
inference at its last step. The cheap half of the measurement now exists. Same four-layer model,
weights deep-copied before either side moved, one put on `eager` and the other through
`linearize_moe`, 4 × 48 random ids:

```
argmax agreement   1.0        max |delta|   0.0        bitwise identical   True
```

Bitwise, not close. Both sides run the same GEMM over the same contiguous weight in the same order,
and there is no numeric difference for a router to turn discrete or for 22 layers to compound —
which is the structural difference from the `eager`-vs-`grouped_mm` pair, where 1.8e-07 at one layer
became 1.24% of tokens at 24. The linearised model also still read `grouped_mm` in its config while
producing that output, which is the setting being inert stated as a measurement rather than as an
argument.

It is still a tiny CPU model in fp32, and §8's lesson is precisely that such a result does not
transfer. What it changes is the prior and the failure it would catch: a disagreement here would
have been decisive, because nothing at this scale compounds. The scaled check — teacher-forced
argmax over the same 24 items, bf16, on the real model — costs minutes beside the 5.5 hours an
arm's eval takes and runs after the panel. Until then the panel's claim is "all arms index one
expert at a time, and at small scale that is bit-identical arithmetic."

**And the records will not pair, correctly.** The four landed baselines were scored by a build that
predates the `experts` field, so they carry no key at all. After the re-run the three DynQuant-side
arms will carry `{grouped_mm, eager}` and `_comparability` will refuse them against the four:
`_ABSENT != 'eager'`. That is the guard doing its job — a record that cannot say what it ran
cannot be certified as having run the same thing — and it is not worth re-running the baselines
to fix, which would cost roughly 22 GPU-hours to change a field while changing no arithmetic. The
panel will therefore report a `NOT PAIRED` line whose content is provenance, not computation, and
the report has to say which.

**But provenance and recoverability are not the same thing, and the first draft of this section
conflated them.** It said nothing recovered what those four ran. Their own records do.
`baselines_lfm2.do_run` linearises, calibrates and scores *one object*: `quantize` returns the model
in memory and `score` hands that same object to `evaluate.run` with no `save_pretrained` and no
reload in between. The `banks_after: 0` sitting in `gptq_4b.quant.json` is therefore a count of the
weights that were then scored, not a claim about a checkpoint someone later opened — and a model
with no batched bank has no grouped kernel to take. The arithmetic was the loop, on the strength of
a number this campaign wrote down.

That changes which arms are actually unknown, and the answer is worth stating precisely because it
is not the four:

| | dispatch | on what evidence |
|---|---|---|
| `gptq_4b` `awq_4b` `gptq_3b` `awq_3b` | the loop | `banks_after: 0`, counted in the scoring process |
| `bf16` `dq_4b` `dq_3b` | unrecorded | nothing; structurally `grouped_mm`, but that is a fact about transformers 5.14.1's default, not about these runs |

The unknown set is exactly the re-score set. That is not a coincidence and it is a better argument
than the one this section opened with: the three arms being re-scored are the three whose arithmetic
no artifact states, and the four being left alone are left alone because their own quantizer records
answer the question. `panel_table.print_dispatch` renders the split, and it renders the evidence
— `loop (22 banks -> 0)` rather than a bare `loop` — because the count is the reason to
believe it. The guard is unchanged and still refuses: recovery reads a sibling file, pairing reads
the record, and a record is not certified by a neighbour that can speak for it.

**And the same guard would have killed the re-score on its second arm.** Worth stating separately
because it was found by asking a question the campaign had not asked — not "is this argument
sound" but "will the command I intend to type actually run" — and the answer was no.
`check_pairable` runs inside the arm loop, over every arm scored so far, and raises `SystemExit` on
any `_comparability` difference. Under `--rescore bf16,dq_4b,dq_3b --experts-impl eager` the first
arm is `bf16`, which scores again and writes `experts.ran = eager`; the second is `gptq_4b`, which
is reused and predates the field. The driver stops there. The three-hour ceiling re-score is spent
and not one of the DynQuant arms the re-score exists for is reached.

The fix is a distinction the panel had already drawn one layer up and the driver had not. Every
other member of the pairing contract names *a question that was asked* — task, split, shot seed,
decode budget — and two records disagreeing on one of those were scored over different items, so
their hit vectors do not line up element-wise and a McNemar across them is arithmetic on unrelated
vectors. `experts.ran` names *how the answer was computed*. The items are the same items in the same
order and the vectors pair; what a difference costs is the reading of the delta. That reading is
already priced, per comparison, by the `!` mark `panel_table` puts on a row whose two arms did not
demonstrably run the same arithmetic. So `check_pairable` now holds the experts block out of the
refusal — by name, from `EXPERTS_PAIRING_FIELDS`, so a field added there is held out with it
rather than turning fatal unannounced — and prints a note naming the straddling arms instead.
Everything else still raises, including when a `limit` difference arrives behind a dispatch
difference, and the message still leads with the `limit`: an operator reads it to decide which file
to delete.

Two things are worth not glossing. The first is that this weakens a guard, and the argument that it
is safe is that a record differing *only* in expert dispatch is by construction over the same
problem set — a stale record from another run would differ in `limit` or `split` or `shots` as
well, and those still stop the driver. The second is the smaller lesson: a plan that is sound end to
end can still be unrunnable, and the cost of finding that out is paid in whatever the run spends
before it hits the wall. Reading `check_resumable` established the re-score would *start*. It took
reading `check_pairable` to establish it would *finish*, and only the second question was load-bearing.

**And then the same question, asked once more, produced a worse answer than the first one.** The
driver was fixed so the re-score could finish. That left the table, and the table had the same bug
in a more expensive form. `panel_table`'s `pairable` is a single string for the whole directory:
`check_pairable` walks the arms, returns the first disagreement with `bf16`, and every row in every
block then short-circuits to `(records are not comparable)`. Not the affected rows — all of them.
After the eager re-score three arms carry `experts.ran` and four do not, the flag fires, and the
panel prints no delta, no interval and no p-value anywhere. Including `GPTQ vs AWQ`, a comparison
neither re-scored arm is part of. **The re-score would have cleared every caveat and deleted the
numbers the caveats were annotating.** Run against the four landed records with `bf16` and `dq_4b`
given the field they will have, the pre-fix script prints exactly that:

```
4b  DynQuant vs GPTQ         (records are not comparable)
4b  DynQuant vs AWQ          (records are not comparable)
4b  GPTQ vs AWQ              (records are not comparable)
```

Two things were wrong and they are worth separating, because only one of them is about dispatch.

**A panel is a set of pairs, and comparability is a property of a pair.** One stale arm anywhere in
the directory blanking comparisons it does not appear in is a defect on its own, independent of
anything to do with experts — it means a single bad `awq_3b` would cost the entire seven-arm
table rather than the two rows containing it. Comparability is now decided per comparison, and the
row names the field: `(not comparable: decode.max_new_tokens)`, not a bare "not comparable" that
sends someone to diff two 120 KB records.

**And `experts.ran` should not have been in the pairing contract at all.** This reverses a decision
§11 recorded two revisions ago, so here is the reasoning rather than just the outcome. It went in
because two arms on different dispatches produce a delta contaminated by dispatch, and that is
true. The error was the response. *Paired* has a technical meaning — the two hit vectors index
the same items in the same order — and a dispatch difference does not break it. The concern is
about what the delta **means**, which is a different thing from whether it can be computed, and the
panel already had somewhere to put it: the `!` mark and the priced footnote added one revision ago.
Having both mechanisms meant the strong one silently pre-empted the informative one. A number
carrying "these two arms are not known to have run the same arithmetic, and the gap between
dispatches is 0.29x the effect you are reading" is strictly more use than a blank row.

The rule now lives once, in `dynquant.commands.evaluate.problem_set_difference`, next to the field
tuples it subtracts from, because by this point three call sites had started growing their own copy
of the subtraction. `_comparability` still reports every difference including the dispatch; what
`problem_set_difference` answers is the narrower question of which differences stop a pairing.

**What the clean table then owes.** After the re-score all seven arms are on indexed arithmetic
— three recording `eager`, four recovered as the loop — so every mark clears and the panel
prints numbers with no caveat at all. That is the correct output and the most dangerous one this
campaign will produce, because the collapse holding it up is that `eager` and the linearised loop
are one class, and the evidence for that is a four-layer CPU fp32 model where the two were bitwise
identical. Bitwise is strong. §8 is also an agreement at small scale that did not survive to 8B.
So the census now prints that claim whenever both buckets are occupied, which is exactly the state
the re-score creates:

```
  2 arm(s) ran `eager` and 2 ran the linearised loop. The panel treats those
  as one class -- both index one expert at a time -- on a four-layer CPU fp32 model where the two
  were bitwise identical. Bitwise is strong, but section 8 of the report is an agreement at small
  scale that did not survive to 8B. Nothing below is marked for dispatch; that rests on this.
```

A table with no caveats has to say what it is resting on, and this one now does. The 8B measurement
remains owed.

**Which makes one absence worth printing.** `_comparability` reads `record.get("experts")` and
treats a non-dict as absent. A `null` — a dense model, no dispatch, none possible — and a
missing key — a record written before anyone asked — are the same value to it, and pair with
each other silently. That exemption has to exist for the dense case. So `panel_table.py` now prints
a dispatch census: the three states rendered distinctly, and the arms in the third named. It is the
only place in the panel where the difference between "nothing to dispatch" and "nobody recorded it"
appears, because the guard by construction cannot raise on it.

**The general form, and it is not the same one as §8's.** §8 was a measurement carried across
scales without being re-taken. This is a claim about one object justified by reasoning about
another: the sentence was about modules, the code read the config, and the two agreed on the answer
while disagreeing about everything else. A structural rewrite that leaves the configuration behind
is the ordinary case, not the exotic one — `named_modules` missing bare parameters and a
reconciling byte total that files every quantized tensor under "dense" are the same shape. The check
that would have caught it is cheap and was available the whole time: build the object, do the
rewrite, print the field.

---

## 12. The four variants that could not be published, and the container that publishes them

**Measured 2026-08-10.** Row 19 of the [experimental record](README.md) — written up as §12 of
[`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) — ends on a refusal. Four of the six
promised variants, `gptq_4b`, `awq_4b`, `gptq_3b` and `awq_3b`, cannot be written as
`compressed-tensors` checkpoints, for two reasons that have nothing to do with each other. The
recipe reaches 91.5% of this model by **renaming** its expert banks, and `ARCH_TO_2D_MAPPINGS`
registers the inverse of that rename for `deepseek_v4` and `qwen2_moe` and nothing else;
`lfm2_moe` linearizes through the generic protocol, so the surgery runs and its inverse does not
exist. And `compressed-tensors` packs `32 // 3 == 10` three-bit values per word, storing **3.2
bits against a label of 3** — 6.7% over, ~200 MiB, sixty-seven times the panel's 0.001 match
tolerance.

Neither is fixable here. The first needs a new upstream conversion mapping authored and merged;
the second is the storage format itself.

But the refusal was about a *container*, and the panel does not score containers. It scores
weights. This section is the bridge that takes the weights a baseline arm was scored on and writes
them into DynQuant's own packed format — bit-for-bit, not re-fitted — and the probe that proves
the directory holds what the arm held.

Commits `2d1d463`, `77a5fd4`, `7b8293c`, `d69de4c`, `d57c927`, `85077fa`, `87d380c`. Code:
[`experiments/phase4/baselines_lfm2.py`](../../experiments/phase4/baselines_lfm2.py),
[`quant/tensor.py`](../../packages/dynquant-core/src/dynquant/quant/tensor.py),
[`quant/checkpoint.py`](../../packages/dynquant-core/src/dynquant/quant/checkpoint.py). Probe:
[`experiments/phase4/probe_publish.py`](../../experiments/phase4/probe_publish.py).

### The one thing that makes this possible, and it is an accident of notation

DynQuant stores a group as `scale * code + offset`, with a **float offset and no integer zero
point**. The ecosystem stores `scale * (q - zero)`. Those are the same set of representable
values: put `code = q` and `offset = -scale * zero` and the two grids coincide exactly, for every
group, at every width. Not approximately — the codes are the same integers and the reconstruction
is the same product plus the same sum.

So the import is a *renaming of parameters*, not a re-quantization. That is the whole reason this
is worth doing rather than merely possible.

### Why the obvious route is wrong

The obvious route is to hand the recipe's dequantized weights to `dynquant export` and let it fit
its own grid. It would produce a directory. It would not produce **this** directory.

Min/max fitting recovers a group's original scale only when the group still occupies both ends of
its code range. A round-to-nearest arm mostly does. GPTQ's error compensation moves weights off
their own grid on purpose, and AWQ's clipping search deliberately narrows the band — both leave
groups spanning, say, codes 4 through 11 of a 16-wide range, and re-fitting those recovers a step
too small and a different set of integers.

The probe runs both, on the same weights, in the same process. The re-fitted export is not a
strawman — it is the same code path with the carrying encoder swapped out — and it is the control
that makes the carried column mean something:

| arm | codes carried | codes re-fitted |
|---|---|---|
| rtn-4b | **0.0000** | 0.5180 |
| rtn-3b | **0.0000** | 0.5031 |
| gptq-4b | **0.0000** | 0.5070 |
| gptq-3b | **0.0000** | 0.5031 |
| awq-4b | **0.0681** | 0.4483 |
| awq-3b | **0.0240** | 0.3987 |

Units are **code steps**: how far the published weight sits from the scored weight, measured in
the arm's own quantization step. Half a step is as wrong as rounding can be, and the re-fit sits
near there on every arm — including `rtn`, which was supposed to be the easy case. The carried
path is not close to the scored weights on the four symmetric arms; it is equal to them.

The two AWQ rows are not zero, and the reason is the format rather than the carry. An asymmetric
grid's offset is `scale * (qmin - zero)` with `zero` a nonzero integer, and that product is not
exactly representable in the dtype the offset is stored in — where a symmetric arm's `scale * -8`
is, being a power of two times the scale. 0.068 of a step is the bf16 rounding of a number up to
sixteen times the scale; it is a fifth of the `MAX_CARRY_DRIFT = 0.125` budget and a seventh of
what re-fitting the same weights costs. Storing the offset in fp32 would remove it and would widen
every group in the container by two bytes, which is not a trade worth making for 7% of a code
step.

### The four pieces

1. **The inverse of `linearize_moe`** (`2d1d463`, `77a5fd4`) — derived by measurement rather than
   read off a docstring, because there is no docstring. Both banks are `[E, out, in]` as stored,
   no transpose; `gate_up_proj[e]` takes `gate_proj` in rows `0:inter` and `up_proj` in rows
   `inter:2*inter`; `down_proj[e]` takes the whole slice. Proven by a bit-for-bit round trip on a
   rectangular geometry, chosen so the two ends of a fused bank cannot be confused by shape.
2. **`QuantTensor.from_codes`** (`7b8293c`) — a constructor that accepts integers and a grid
   instead of floats and a search.
3. **An `encoder` seam on `export_packed_checkpoint`** (`d69de4c`), plus `_check_encoder_agrees`,
   which re-derives what the encoder claimed and refuses a disagreement. The exporter's own
   accounting, tying, sharding and manifest are unchanged — this is the same writer the DynQuant
   arms use, which is the point.
4. **`carried_grids` / `banked_grids` / `carrying_encoder` / `do_publish`** (`d57c927`) — read
   every quantized module's codes and scales off the recipe, re-stack the linearized experts back
   into banks by the same rule the weights follow, and hand the exporter a grid per name.
   `MAX_CARRY_DRIFT` refuses any module whose carried grid does not reconstruct its own weight.

### What the end-to-end probe found that twelve unit tests could not

`do_publish` shipped green: four gates, twelve tests, a mutation sweep. It had three defects. All
three needed an actual `llm-compressor` run to reach, and
[`probe_publish.py`](../../experiments/phase4/probe_publish.py) is that run — quantize a
four-layer `lfm2_moe`, publish it, reload the directory through the real `HfQuantizer`, compare
against the weights the in-process arm was scored on.

**The recipe's own scratch was being published as model weights.** `delinearize_state_dict` was
handed `model.state_dict()`, which after a recipe carries `weight_scale`, `weight_zero_point` and,
on the GPTQ arm, `weight_g_idx`. On a linearized expert those trip the bank assembler's refusal —
correctly, since the derived rules say where a weight *row* goes and nothing about where a
per-group scale would. Everywhere else they survive into the output and make the strict load
reject the model. The fixtures build modules by hand and so never carry scratch; nothing smaller
than a recipe produces it.

The fix subtracts rather than lists. `recipe_scratch` asks each quantized module what it holds
beyond a weight and a bias, and drops that. A hard-coded list of `weight_scale`,
`weight_zero_point` and `weight_g_idx` would have been a second copy of `compressed-tensors`' set
of artifacts — which is exactly the defect two subsections below. The filter is allowed to be
liberal because it is not the check: `do_publish` loads the result with `strict=True`, so dropping
one key too many fails as loudly as keeping one too few.

**The tied table was published under a name the loader does not read.** The DynQuant format stores
a tied table **once, under the input embedding's name**; `_tie_output_embedding` then replaces the
head with a `DynQuantLinear` holding no tensors of its own. A recipe targets `Linear`, so the
module it quantizes is `lm_head`, and a directory keyed that way loads with
`model.embed_tokens.weight | MISSING`, gets a random re-initialization, and dies in
`mark_tied_weights_as_initialized` with `AttributeError: DynQuantLinear has no attribute 'weight'`.
Note the order: the load **reports** the missing table and continues; the exception arrives later,
from somewhere else, about something else.

The first fix for this did not fire, and the reason is worth more than the fix. It gated on
`config.tie_word_embeddings`, and **`oneshot` sets that to `False`** while leaving the storage
shared. Measured identically on every arm:

```
{'config_says': False, 'input': 'model.embed_tokens', 'output': 'lm_head',
 'tied': True, 'shared_storage': True, 'values_equal': True}
```

The config is not wrong. It is true of a different object — the compressed checkpoint `oneshot`
would have written, which carries two tables — and false of the model in memory, which carries one
and is the model being published. `tie_report` now derives the tie from storage and reports the
flag beside it as `config_says`; `pristine_config` re-reads the checkpoint's own config so the
export is not built from the recipe's mutated one, which would otherwise also carry a stale
`quantization_config` describing a format the directory is not in.

This is §11 running backwards. There, a structural rewrite left the configuration behind — the
modules changed and the config did not. Here the configuration was rewritten to describe an
artifact nobody was writing, and the modules did not change. Both are the same instrument: build
the object, do the operation, print the field.

### The third defect, and it is the family this campaign keeps rediscovering

The two arms above are round-to-nearest and GPTQ, and both are **symmetric**. AWQ is not, and the
first asymmetric arm refused at the carry check:

```
model.layers.0.conv.in_proj: scale * (q - zero) does not reproduce the materialized weight
(max |delta| = 6.976e-02)
```

That message offers two branches — the weights were never rounded, or this reader's idea of the
convention has diverged from the library's — and, per this campaign's own standing lesson, a
guard's stated reason for firing is a hypothesis and not a diagnosis. So it was measured. On that
module, at 4 bits, group 32:

| what was asked of the weight | what came back |
|---|---|
| largest fractional part of `weight / scale`, in steps | **0.0377** |
| elements sitting on the assumed bottom rail `qmin = 0` | **7,443 of 12,288** |
| range of `weight_zero_point` | **-4 … 3**, 8 distinct |

The first row settles the first branch: the weight *is* on a lattice of its own scale, to within
4% of a step, so it was rounded and the "never quantized" reading is false. The third row says
what the second branch was. `compressed-tensors` puts **every** integer scheme on the signed band
`[-2^(b-1), 2^(b-1)-1]`, whether or not it is symmetric, and lets an asymmetric scheme ride that
same band with a *signed* zero point — hence a zero point of -4 through 3, which is meaningless on
an unsigned range. This reader had worked the range out for itself instead, and its rule —
unsigned `[0, 2^b - 1]`, switching to signed only when the scheme said symmetric — agreed with the
library on every symmetric arm and clamped 61% of the first asymmetric weight onto the bottom
rail.

The fix is one line: `calculate_range(scheme, device)` is `compressed-tensors`' own function and
is now what the carrier calls. The rest of `carried_grids` is derived-and-then-checked — a wrong
derivation announces itself against the materialized weight — but the range cannot be, because it
is an *input* to that derivation. It is the one piece that has to be imported rather than
verified, so it is now imported.

The fixtures had encoded the same wrong convention, which is why twelve tests were green over it.
They now build codes on the signed band, so every one of them is a guard; and a new test drives an
unusual range through the stub and asserts the reader honours it, which is the difference between
"asked the library" and "worked it out and happened to agree". Fourteen mutations of the publish
path, all caught.

The shape is the one this repository already enumerates six times over — a duplicated task list, a
duplicated name resolver that narrowed twice, a width list guarding on the wrong criterion, and a
writer and a reader of one format that each held half a rule. This instance is a step further out.
It is not a second copy of *our* registry; it is a second copy of a **dependency's arithmetic**,
which is the hardest kind to notice, because nothing inside this repository is able to contradict
it. Only the first input the two definitions disagree about can, and that input was the fifth arm
of the six.

### One more thing the probe found, which is upstream and is not ours

The AWQ arm on this tiny model **is not deterministic**. A randomly initialized `lfm2_moe` in
bfloat16 produces non-finite activations on random calibration ids; AWQ skips every mapping whose
parent forward is not finite; and how many it skips moves run to run — 21, 20, 13 and 6 observed
across four runs of identical code on identical weights. Two failure modes follow from the tail of
that distribution, both in `llm-compressor`: `_log_error_metrics` divides by the number of
mappings that survived, so **all skipped** is a `ZeroDivisionError`; and a mapping whose grid
search finds no finite loss raises outright. The probe retries, and the retry count is the honest
way to report an arm that is stable in outcome and unstable in getting there.

None of this touches the 8B, whose `awq_4b` and `awq_3b` arms calibrate on real text and ran
clean. It is a property of calibrating a *random* model, and it is recorded because it cost an
afternoon to tell apart from a defect in this code. What the same arm did need on the real model
is unrelated and is a bridge concern: the stock AWQ mappings smooth q/k/v against an
`input_layernorm`, LFM2 names that `operator_norm`, and `_set_resolved_mappings` raises on the
incomplete set. The probe therefore takes the driver's own `resolve_awq_mappings` — 6 mappings, 53
linear modules, 48 smoothed, the 5 unsmoothed named — so the arm it publishes is the arm the panel
scored rather than a differently-calibrated one.

### What the published directory costs, and it is not the arm's byte count

The container is exactly `bits + 32 / group_size` bits per parameter — an fp16 scale and an fp16
offset per group and nothing else. Predicted to the byte on all six probe arms: 202,720 B at 4
bits and 162,176 B at 3, group 32, over the same 324,352 quantized parameters. At the panel's
group size of 128 that is **4.25 bits at 4 and 3.25 at 3**.

`compressed-tensors` stores 4.15625 at 4 bits — the same codes, an fp16 scale, and a *4-bit* zero
point where DynQuant carries an fp16 offset. Twenty bits per group of 128 against DynQuant's
thirty-two. So a republished `gptq_4b` holds the same weights to the code and occupies **0.09375
bits/param more**: 2.3%, about **99 MB** against this model's 8,467,856,128 parameters. At 3 bits
it goes the other way — 3.25 against a container whose codes alone cost 3.2 before any scale — so
the width that could not be published honestly at all is the one where this container is both
correct and smaller.

A model card for a republished baseline therefore carries two numbers: the bytes the arm was
scored at, and the bytes of the directory. They are not the same number, and the difference is the
container, not the model.

### The gap the carry check does not close, and the label that was standing in for it

The carry check compares the codes written to disk against the codes held by the model in memory.
It proves the *export* is faithful. It says nothing about where that model came from — and where
it came from is a second calibration pass, because the panel never serialized a first one. `run`
loads the checkpoint, applies the recipe, scores the result in process, and writes a JSON record.
There is no checkpoint at the end of an arm. So `publish` re-runs the recipe, and the only thing
connecting the directory it writes to the row in the table was the label on the directory.

That is not a small gap on this campaign. A GPTQ pass over 8 B parameters takes 32 minutes and an
AWQ pass 47; both were run on a box whose GPU is shared with a panel, over days, from a shell.
A directory named `gptq_4b` that came out of a pass with `--calib-samples 128` instead of 256, or
against a `--model` one directory over, is an ordinary artifact of a long campaign and an
extraordinary thing to notice by reading a filename.

`publish --scored <arm>.quant.json` replaces the label with a comparison against the arm's own
record. It runs in two halves, and the split is the point:

- **Before the recipe:** `method`, `bits`, `group_size`, `ignore`, `seq_len`, `source`. Every one
  is readable from the namespace before anything loads, so learning that `--bits` disagrees with
  the record costs a second rather than half an hour of a contended GPU.
- **After the recipe:** `calib_samples`, `materialized_modules`, `weights_moved`,
  `max_weight_delta`, `probe_unique_values_per_row`, `accounted_bits`, `accounted_bytes`,
  `quantized_params`, `banked_params_quantized`, `params`. Compared exactly. There is no tolerance
  and no override flag — a disagreement here is the finding that the published weights are not the
  scored weights, and what to do about that is a decision, not a default.

**The byte accounting alone could never have done this**, and the two 4-bit arms already on record
are the proof:

| field | `gptq_4b` | `awq_4b` |
|---|---|---|
| `accounted_bits` | 4.1565 | 4.1565 |
| `accounted_bytes` | 4,399,629,312 | 4,399,629,312 |
| `quantized_params` | 8,467,644,416 | 8,467,644,416 |
| `banked_params_quantized` | 7,751,073,792 | 7,751,073,792 |
| `materialized_modules` | 2,201 | 2,201 |
| `weights_moved` | **0** | **2,201** |
| `max_weight_delta` | **0.0** | **2.890625** |

Every number describing how large the result is agrees to the byte, because both are the same
architecture at the same width and the same group size. Publishing one arm's weights under the
other's row would leave all of it intact, and a size check would pass. What separates them is what
the recipe *did* to the weights: AWQ's smoothing moves all 2,201 modules by up to 2.89 before
quantizing, GPTQ moves none. So the three fingerprint fields are load-bearing rather than
decorative, and a test says so directly — trim `SCORED_WEIGHTS` down to the accounting and that
test goes red.

A record written before a field existed is still worth comparing against, so an absent field is
skipped — and counted. The published metadata carries `fields_compared` beside `fields_available`,
because a directory matched on six fields should not be filed as a directory matched on ten. The
asymmetry is deliberate and is the whole difference between a coverage gap and a defect: a field
the *record* does not carry is a gap; a field the record carries and this pass did not produce is
a disagreement.

### The last document, and the only one anyone reads before downloading 4 GB

Everything above produces directories. What a reader meets first is the README above one of them,
and that file is the last place in this pipeline where a number can drift without anything going
red. The panel is re-run every time an arm lands — five times so far, over a day and a half — so a
card written by hand in between is correct on the afternoon it is typed and wrong afterwards,
silently, because a README that says 73.75% looks exactly as authoritative as one that says 82.07%.

`experiments/phase4/model_cards.py` therefore types no number. Each card is assembled from two
files the runs themselves produced:

| input | what it supplies |
|---|---|
| `panel_table.py --json-out` | every arm's size, bits per parameter and execution match; every head-to-head with its Holm-adjusted p, its CI and its verdict; the per-source split |
| `s2_finetune.json` | base model, the three training datasets, the regime, step count, loss, commit |

Reading the **table** rather than the seven records is the whole point of the split.
`panel_table.py` already decides which arms are comparable, corrects a family of six with Holm
step-down, and marks the comparisons whose two arms are not known to have run the same expert
arithmetic. A card that re-derived any of that from `panel/*.json` would be a second
implementation of the statistics — the failure this campaign has now found in eight places — and
this one would agree with the table until the first panel where it did not, with the disagreement
surfacing on the Hub rather than in a terminal.

#### The caveats are generated, not remembered

Three of them are load-bearing here, and each is emitted because of something on the arm's row
rather than because it is on a checklist:

- **`apply == "encode"` → the accuracy was not measured from this directory.** A DynQuant arm is
  scored by encoding its allocated widths back into bf16, because 91.5% of this model's parameters
  are batched expert banks and the scoring path applies widths in memory rather than writing a
  17 GB decoded copy per arm. Same encoder, same widths, same values — but the packed container
  was not separately scored, and the card says so on the arm this campaign is arguing for.
- **`kind != "dq"` → this directory is about 2.3% larger than the size in its own results table.**
  The recipe's integer codes are carried across exactly into a container that spends a full bf16
  zero per group where compressed-tensors packs the zero to the weight width.
- **a flagged comparison → `[^1]` and the number that earns it.** The two expert dispatches
  available for this architecture disagree on **1.24%** of teacher-forced tokens, **0.29x** the
  effect of quantizing to 4 bits. A flagged row's delta carries a term that is not the
  quantization method, so the verdict is what the stored per-item hits say and not yet a statement
  about quantization alone.

Putting all three on every card would be the easy way never to be wrong, and would tell a DynQuant
reader their directory is oversized and a GPTQ reader their accuracy was measured somewhere else.
Both false. The conditions are the content.

#### What writing the generator found

`as_json` was dropping two fields on the way out: `question`, the row label, and
`same_arithmetic`, which the printed table shows as a trailing `!`. That was survivable while the
only consumer was a person reading a terminal — the flag was two blocks up in the dispatch census.
It is not survivable once six Hub READMEs are built from the payload, because the card would then
publish `separated` with its confound stripped off. Both fields are carried now, and `--json-out`
writes the same payload `--json` prints, constructed once: a second construction for the file
would be a second copy of the table, which is the thing the split exists to prevent.

The generator refuses three things rather than describing them: the `bf16` ceiling (a card for it
would describe a model this campaign did not make), an arm with no accuracy (mid-panel that is
four arms for a day, and a card is a claim about a measurement), and a label the table does not
carry. It writes into directories that already hold weights and names the ones it skipped — a
README with no weights under it is a published model that does not exist.

Ten tests, in `tests/test_model_cards.py` and one addition to `tests/test_panel_table.py`, reusing
the panel fixture rather than copying it so the two ends of the pipeline cannot drift apart. The
ones that carry weight: mutating a number in the table changes the card; each arm carries its own
container caveat and never the other kind's; the flag survives into the card **and clears** when
every arm is re-scored onto one dispatch; a comparison reads identically on both of its cards with
the sign not flipped; and the usage snippet's `dynquant.register_hf_quantizer` is resolved against
the installed package — `import dynquant` deliberately does not register, so that line is
load-bearing, and renaming it would leave six published READMEs quietly wrong with no failing
build anywhere.

None of that would have found the two defects that a real table did. Running the generator
against the five arms that have actually landed -- rather than only against the synthetic
fixture the tests use -- produced a card whose headline sentence counted "a seven-arm panel"
as a **typed literal**, the one thing this file exists to forbid, and whose results table
silently **dropped the two arms still scoring**: five rows under a sentence claiming seven,
with nothing to tell a reader the others exist. A fixture cannot reach either, because a
fixture is written complete. Mid-panel is not an edge case here -- the expensive arms publish
first while the cheap ones are still running, so it is the state most of these cards would
have been written in. The count is derived from the table now, and an unscored arm keeps its
row and says what it is.

### Status, stated as what is not yet true

- All six arms — `rtn`, `gptq` and `awq` at 4 and 3 bits — publish and reload within **0.068 code
  steps** of the weights the in-process arm was scored on, with six packed expert banks, nothing
  unmatched, no dense expert parameters left behind, finite logits, and the tie renamed onto
  `model.embed_tokens`.
- This is a **four-layer** model. What remains between here and the 8B is scale, not model class —
  the same architecture, the same linearization, the same banks — but it has not been run, and it
  cannot be until the panel releases the GPU.
- One leg of that has since been run at 8B, on CPU, while the panel kept the card. A DynQuant arm
  does not re-calibrate: it exports a saved map, and `export --map maps/dq_4b.json --map-key
  4399629312 --dry-run` is the whole of that command shape except the write. Against the real
  merged checkpoint it resolves the key, matches **133** banked tensor names, and predicts
  **4.096 GiB at 4.1547 average bits, "as recorded"** — the arm's own row. The names, the key and
  the accounting are therefore not in question at scale; what is still untested at 8B is the write
  itself, and `export` compares the bytes it wrote against that prediction on its own.
- The two arm kinds weigh different amounts against their rows and neither is a defect. A map arm
  was priced in DynQuant's container and exports into it, so it should land on its figure. A
  recipe arm was scored under compressed-tensors at 4 + 20/128 bits and republishes by carrying
  the identical codes into a container that spends a full bf16 zero per group, 4 + 32/128 — about
  **2.3%**, or **+99 MB** at the 4-bit anchor, for the same numbers on dequantization.
- The published artifact loads through **DynQuant's** `HfQuantizer`, not through vLLM's native
  `compressed-tensors` path. Row 19's "yes — vLLM and transformers" is not restored by this and is
  not claimed. What is restored is that the weights a person downloads are the weights that were
  scored — provided `--scored` is given the arm's record. Without it the directory is published on
  the strength of matching flags, which is a claim about the inputs and not about the weights.
- The cards have now been generated from the **real** table, which is where two defects the
  synthetic panel could not reach came from. The five landed arms produce a `table.json` and the
  four scored quantized arms produce cards; running them found the headline sentence counting
  "a seven-arm panel" as a **typed literal**, and the results table silently **dropping the two
  arms still scoring** — five rows under a sentence claiming seven, with nothing to tell a reader
  the others exist. The count is derived now and an unscored arm keeps its row. Nothing uploads,
  and the Hub push is a separate decision and a separate command.
- The publish path now carries **nineteen** mutations, each with the test it is expected to redden.
  Five of them are the scored check: running the recipe before reading the record, accepting the
  flag without acting on it, skipping a field this pass did not produce, counting coverage over
  the wrong side of the comparison, and trimming the fingerprint down to the byte accounting.
