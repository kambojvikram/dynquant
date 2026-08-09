# Phase 4 · the packed MoE runtime — making the artifact the same object as the measurement

**Measured 2026-08-08 / 2026-08-09.** Code:
[`runtime/linear.py`](../../packages/dynquant-core/src/dynquant/runtime/linear.py),
[`quant/tensor.py`](../../packages/dynquant-core/src/dynquant/quant/tensor.py),
[`integration/hf_quantizer.py`](../../packages/dynquant-core/src/dynquant/integration/hf_quantizer.py).
Tests: [`tests/test_expert_bank.py`](../../tests/test_expert_bank.py) (20).
Commits `a60d993`, `10846ca`, `18a8870`.

The seven-arm panel scores the **encoder** — `quantize_model(in_place=True)`, which rewrites each
weight to its dequantized value and leaves an ordinary model behind. It does that because a
matched-byte panel needs one GPU pass per arm and nothing on disk. What a person downloads is the
**packer** — `export_packed_checkpoint` and the `DynQuantLinear` / `DynQuantExpertBank` runtime
that reads it back.

Those are two implementations of one format, and the entire evidentiary value of the panel rests
on them being the same object. This report is the three defects that separated them, found in the
order they hid behind each other, and the measurement that closes each one. None of the three was
visible from any accuracy number. All three were found by running the *genuine*
`Lfm2MoeExperts` — not the synthetic copy the unit tests use.

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

## 4. What the genuine architecture says now

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

**The honest limit.** This is the genuine *class*, at tiny *scale*: 4 layers, 6 experts, fp32. The
8B has 44 banks, 24 layers, bf16 and a tied embedding, and it is the bf16 that makes the scale-dtype
defect invisible there — both rules return bf16, so the 8B was never mis-encoded. What the 8B has
not yet been through is the export-and-reload path itself, which is running on CPU beside the panel
as of this writing; until that lands, the claim is "the format round-trips exactly on the real
architecture" and not "on the real model."

---

## 5. What this does and does not say about the panel

It does not revise any arm's score. The panel's DynQuant arms run `--map-apply encode`, the encoder
path, and the encoder was never the wrong one — on bf16 it agreed with the packer all along.

What it changes is the standing of those scores. Before this work, "DynQuant 4-bit scores X on
text-to-SQL" was a statement about a transient object; the packed artifact could not be built for
this architecture (§1), could not run if built (§2), and would not have matched on a model in fp32
(§3). It is now a statement about a checkpoint whose bytes, weights and logits have each been
checked equal to the thing that was scored.

The remaining gap between measurement and artifact is speed, not correctness, and it is the one the
P8 kernel exists to close.
