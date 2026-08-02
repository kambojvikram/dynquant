# SGLang integration plan

Serve DynQuant checkpoints on SGLang without patching SGLang, the same way
`dynquant.integration.vllm_plugin` does for vLLM.

Written against SGLang `131bd51b0` (2026-08-01), **re-verified 2026-08-02 against the
released `sglang==0.5.16` wheel**; line citations below are the 0.5.16 ones. Every
claim cites the file it came from; re-check the citations before starting, because
none of these surfaces are covered by a compatibility guarantee.

Two claims did not survive re-verification and are corrected in place — see S3.
Both were in the direction of "SGLang hands us more than vLLM does"; it does not.

### Install constraint, found during re-verification

SGLang stopped shipping `py3-none-any` after 0.5.10.post1. 0.5.16 publishes
**manylinux_2_34 wheels only** (cp310–cp313 × x86_64/aarch64), so a Windows or macOS
`pip install sglang` silently resolves to 0.5.10.post1 — a release that predates the
plugin system entirely and has no `srt/plugins/` at all. Consequences:

- Every SGLang-importing test skips off Linux. Unlike the vLLM plugin, whose
  `geometry.py`/`schema.py` are laptop-testable, S2–S4's shim tests can only *run* on
  the Linux box. This is the reason S1 exists, and it raises the stakes on it: the
  framework-free half is the only half that gets exercised on a laptop.
- Do not read API surfaces out of a locally-installed SGLang on a dev box. It will be
  the stale one and it will disagree.

## Why this is mostly a port and not a rewrite

SGLang's quantization layer is a fork of vLLM's — `base_config.py` still carries
`Adapted from .../vllm/v0.5.5/...` in its header, and `parameter.py` still names its
classes `BasevLLMParameter`. The consequences:

| Surface | vLLM | SGLang | Work |
|---|---|---|---|
| Config ABC | `QuantizationConfig` | identical, incl. `override_quantization_method` | import swap |
| Linear method ABC | `LinearMethodBase` | identical | import swap |
| Parameter classes | `BasevLLMParameter` &co. | same names | **signature drift, see S4** |
| Layer dispatch | `get_quant_method(layer, prefix)` | same, from Linear / FusedMoE / VocabParallelEmbedding / RadixAttention | none |
| Checkpoint-driven selection | `_verify_quantization` | same, ported | none |
| `from_config` payload | the `quantization_config` dict | the same single dict, but with `packed_modules_mapping` **injected into it** | ~~simplification~~ **see S3** |
| Method registration | `@register_quantization_config` | **no public API** | S2 |
| v2 weight loader opt-in | `register_weight_loader_v2_supported_method()` | **no public API** | S2 + S8 |

### The three containers a plugin must mutate

| Container | Location (0.5.16) | Public setter |
|---|---|---|
| `QUANTIZATION_METHODS` | `layers/quantization/__init__.py:139` | none — read live by `get_quantization_config` and by `ModelConfig._verify_quantization` (`configs/model_config.py:1409`), so in-place insert works |
| `QUANTIZATION_CHOICES` | `server_args.py:153` | **yes** — `add_quantization_method_choices()` at `server_args.py:359` |
| `WEIGHT_LOADER_V2_SUPPORTED` | `layers/linear.py:58` | none — a list of class-name *strings*, matched at `linear.py:368` and `:1450` |

`QUANTIZATION_METHODS` is built as `{**BASE_QUANTIZATION_METHODS}` (0.5.16 split the
dict in two to let NPU/MPS/CPU override entries). Insert into `QUANTIZATION_METHODS`,
the one both readers actually consult — writing to `BASE_` would be a no-op, since the
copy has already been taken by the time any plugin runs.

`WEIGHT_LOADER_V2_SUPPORTED` is the load-bearing one. The v1 loader places shards
with `param.data.narrow(output_dim, ...)`; on DynQuant's flat buffers that narrows
into the wrong rows and copies **without raising**. Getting this wrong produces a
model that loads, serves, and is quietly wrong.

### Ordering is already correct

`load_plugins()` (`srt/plugins/__init__.py:103`) runs before anything reads the
containers, in every process that matters:

- `launch_server.py:66` — before `prepare_server_args` at `:68`, so argparse sees our choice
- `cli/serve.py:99` — before `prepare_server_args` at `:139`, same reason
- `entrypoints/engine.py:212` — first thing in `Engine.__init__`
- `entrypoints/engine.py:790` — defensive re-entry
- `managers/scheduler.py:4590` — the **first statement** of `run_scheduler_process()`,
  which is the one that matters: the scheduler is a *spawned* process, so it does not
  inherit the parent's registry mutations, and it is where `ModelConfig` (`:578`) and
  the quantization-config resolution actually run

One cosmetic gap: `sglang serve --help` calls `prepare_server_args(["--help"])` at
`cli/serve.py:74`, which is *before* the `load_plugins()` at `:99`. So `dynquant` will
not appear in the `--help` choice list even though `--quantization dynquant` works.
Not worth a patch; worth not being confused by.

That last one is better than vLLM, which relies on fork inheritance for the manual
`import my_plugin` path. `load_plugins()` is idempotent (`_plugins_loaded` guard) but
may still be called more than once per process, so `register()` must be re-entrant.

---

## Phases

Each phase has a gate. S1–S4 need no GPU.

### S0 — Environment, on the vast.ai box

SGLang pins `torch==2.11.0` (`python/pyproject.toml:78`) — the same torch minor as
`/workspace/venv-vllm`. `dynquant_kernels 0.1.2` was built against that ABI, so it
should import unchanged; the ABI guard will say so either way.

1. `venv-sglang` with python 3.12, torch 2.11.0+cu130, `sglang[srt]`.
2. `pip install --no-build-isolation` the existing `dynquant-core` (editable) and the
   already-built `dynquant_kernels` wheel from `/workspace/vllm_parity/`.
3. `DYNQUANT_BACKEND=cuda python -c "from dynquant.runtime.backend import resolve_backend; print(resolve_backend())"`
   — must print `Backend.CUDA`, not fall back.
4. Serve the **unquantized** Qwen2.5-0.5B through SGLang and generate.

**Gate:** kernels import in the new venv with `Backend.CUDA`; SGLang serves fp16.
If the ABI guard rejects the wheel, rebuild with `/workspace/vllm_parity/build_kernels.sh`
pointed at the new venv — the three traps in its comments still apply.

### S1 — Extract the framework-free core

`geometry.py` (547 lines), `schema.py` (310) and `fuse.py` (45) import nothing from
vLLM. That is 902 of 1660 lines, and it is the half that fails silently rather than
loudly.

`fuse.py` was not in the original list and belongs there for a reason stronger than
tidiness: it registers a **process-global** `torch.library` op, `dynquant::fused_shard_concat`.
Two copies under two plugin packages is a duplicate-registration error the moment
anything imports both — which a parity harness comparing the two backends in one
process would do immediately. It also has to move on the merits: SGLang compiles with
inductor too, so the `split(cat(...))` cancellation the op exists to defeat is not a
vLLM-specific bug.

1. Move all three to `dynquant/integration/serving_common/`.
2. `vllm_plugin/` imports them from the new location; **no behavior change**.
3. Rename `tests/test_vllm_geometry.py` → `test_serving_geometry.py`, same for schema
   and fuse.

**Gate:** existing suite green, and the A100 four-arm parity sweep reproduces its
previous numbers exactly (fp16 0.006304 / dynquant 0.009147 / effect 2.461067). A
pure move must not move a decimal place.

### S2 — Registration shim

`dynquant/integration/sglang_plugin/__init__.py`:

```python
def register() -> None:
    from sglang.srt.layers.linear import WEIGHT_LOADER_V2_SUPPORTED
    from sglang.srt.layers.quantization import QUANTIZATION_METHODS
    from sglang.srt.server_args import add_quantization_method_choices

    from dynquant.integration.sglang_plugin.config import DynQuantConfig

    QUANTIZATION_METHODS.setdefault("dynquant", DynQuantConfig)
    if "dynquant" not in QUANTIZATION_CHOICES:
        add_quantization_method_choices(["dynquant"])
    if "DynQuantLinearMethod" not in WEIGHT_LOADER_V2_SUPPORTED:
        WEIGHT_LOADER_V2_SUPPORTED.append("DynQuantLinearMethod")
```

Two of those three writes touch names SGLang does not promise. Guard them: check each
symbol exists and is the expected type, and on failure raise naming the installed
SGLang version and what changed — not an `AttributeError` three frames deep in a
scheduler subprocess.

`pyproject.toml`:

```toml
[project.entry-points."sglang.srt.plugins"]
dynquant = "dynquant.integration.sglang_plugin:register"
```

**Gate:** unit test with the three containers faked — `register()` is idempotent under
repeated calls, and the version guard fires (with a readable message) when any symbol
is missing or the wrong type.

### S3 — Config port

`sglang_plugin/config.py`, from `vllm_plugin/config.py`:

- base classes from `sglang.srt.layers.quantization.base_config` and
  `sglang.srt.layers.linear`

**Correction 1 — `from_config` is single-arg, and the mapping does not arrive by
itself.** `QuantizationConfig.from_config(cls, config: Dict[str, Any])`
(`base_config.py:163`) is byte-identical to vLLM's, so our existing implementation
ports unchanged. What differs is the *caller*: SGLang mutates the dict on the way in,

```python
# model_loader/weight_utils.py:278  (and again at :345 for the sidecar path)
hf_quant_config["packed_modules_mapping"] = packed_modules_mapping
return quant_cls.from_config(hf_quant_config)
```

but nothing then copies that key onto the instance. `QuantizationConfig.__init__`
sets `self.packed_modules_mapping = dict()` (`base_config.py:131`) and the only
writer of `update_packed_modules_mapping()` in the whole tree is
`models/deepseek_v2.py:2707`. Neither `GPTQConfig` nor `AWQConfig` reads the key,
because neither needs fusion structure to build a layer — DynQuant does, since
`resolve_shards` is how a fused `qkv_proj` learns it is three modules at three widths.

So the SGLang config must lift it itself:

```python
self.packed_modules_mapping = config.get("packed_modules_mapping", {}) or {}
```

Left out, every fused layer resolves to no shards and takes the unquantized path
against a checkpoint full of packed words. Worth a direct test: `from_config` on a
dict carrying the key must expose it on the instance.

- `get_quant_method`: `LinearBase` → `DynQuantLinearMethod`; `VocabParallelEmbedding` →
  our embedding method; `FusedMoE` → deferred to S7; `RadixAttention` → `None`.
- `VocabParallelEmbedding` (but not `ParallelLMHead`) requires the method to implement
  `embedding()` — `vocab_parallel_embedding.py:305-311`, same rule as vLLM.
- `get_scaled_act_names()` is still `@abstractmethod` in SGLang (`base_config.py:230`)
  though vLLM has dropped it. Return `[]`. Omitting it makes `DynQuantConfig`
  abstract, and the failure surfaces as an unrelated-looking `TypeError` at
  instantiation inside a scheduler subprocess.
- `apply_vllm_mapper` is named `apply_weight_name_mapper(self, hf_to_sglang_mapper)`
  here (`base_config.py:244`). Same `WeightsMapper`, same body; rename only.

**Correction 2 — `override_quantization_method` is polled for every checkpoint, not
just ours, and it is a hijack risk rather than a cheapness concern.** The detection
loop asks *every* registered config, ours included, about *every* model someone
serves:

```python
# configs/model_config.py:1409
for _, method in QUANTIZATION_METHODS.items():
    quantization_override = method.override_quantization_method(quant_cfg, self.quantization)
    if quantization_override:
        quant_method = quantization_override
        self.quantization = quantization_override
        break
```

First truthy answer wins and breaks the loop. Returning anything but `None` for a
checkpoint that is not ours would silently redirect someone else's GPTQ model into
DynQuant's loader — and because we are inserted into a dict, our position in that
iteration order is not something we control. Inheriting the base's `return None`
(`base_config.py:171`) is therefore the correct implementation, and the test that
matters is the negative one: pass GPTQ, AWQ and bitsandbytes `quantization_config`
dicts and assert `None` for each.

**Gate, CPU only:** construct `ModelConfig` on the exported checkpoint with no
`--quantization` and assert `quantization == "dynquant"`, resolved from `config.json`.

### S4 — Parameter and linear port (highest risk)

The one real semantic difference. vLLM's parameter classes read tensor-parallel state
from globals; SGLang passes it in:

```python
# sglang/srt/layers/parameter.py:145   (_ColumnvLLMParameter)
def load_column_parallel_weight(self, loaded_weight, tp_rank, use_presharded_weights=False)
# :293  (RowvLLMParameter)
def load_row_parallel_weight(self, loaded_weight, tp_rank, use_presharded_weights=False)
# :226  (_ColumnvLLMParameter) -- tp_rank positional here too
def load_qkv_weight(self, loaded_weight, tp_rank, use_presharded_weights=False, **kwargs)
```

Confirmed against 0.5.16. Our flat-buffer overrides must accept and honour those
instead of calling `get_tensor_model_parallel_rank()`. A subclass that silently
ignores the kwarg still runs — and shards wrong on every rank but 0.

Two asymmetries to copy exactly, both visible in `parameter.py`:

- `load_merged_column_weight` (`:177`) takes `tp_rank` **through `**kwargs`**
  (`kwargs.get("tp_rank")`), not positionally, unlike its three siblings. A uniform
  `(self, loaded_weight, tp_rank, ...)` signature across all four would break it.
- `PerTensorScaleParameter` (`:406`, `:417`) *pops* `tp_rank` and
  `use_presharded_weights` before delegating. That is the shape of an override that
  legitimately ignores rank; ours is not one, and the S4 gate exists to prove it.

Also: `create_weights` takes an extra `skip_block_quant_check` kwarg
(`linear.py:330`, forwarded at `:365`; `ColumnParallelLinear` at `:961`/`:1013`) —
only `fp8.py` reads it, so accepting and ignoring it via `**extra_weight_attrs` is
correct, but the signature must not choke on it. SGLang uses `copy_with_check` rather
than a bare `copy_`; and there are `_is_cpu` padding branches we should assert we
never enter.

**Gate, CPU only:**
1. Geometry oracle tests at TP ∈ {1, 2, 4} exercising the `tp_rank` argument
   specifically — assert rank *n* receives rows `[n·shard, (n+1)·shard)`.
2. A negative test proving the v2 registration is load-bearing: with
   `WEIGHT_LOADER_V2_SUPPORTED` **not** patched, loading must produce different bytes
   than the v2 path. If that test passes trivially, S2's third write is dead code and
   the whole design is wrong.

### S5 — First real serve

```bash
python -m sglang.launch_server --model-path /workspace/vllm_parity/ckpt \
  --tokenizer-path <snapshot> --dtype float16 --mem-fraction-static 0.35 \
  --max-total-tokens 4096 --disable-cuda-graph --port 8299
```

No `--quantization` flag — resolution must come from `config.json`.

**Gate:** `/generate` returns coherent text; the log line from
`model_runner.py:1062` reports `dynquant`.

### S6 — Parity sweep

Reuse `04_compare.py` unchanged. Six arms instead of four, one process each,
`DYNQUANT_BACKEND=cuda` exported so an unavailable backend raises rather than
silently falling back to Triton:

| arm | purpose |
|---|---|
| direct fp16 vs direct packed | **the yardstick** — how far quantization moves the model at all |
| vLLM fp16 vs direct fp16 | vLLM runtime control |
| vLLM packed vs direct packed | vLLM serving gap |
| **SGLang fp16 vs direct fp16** | SGLang runtime control |
| **SGLang packed vs direct packed** | SGLang serving gap |

Assert checkpoint identity first — all 504 buffers across 168 modules `torch.equal`
between `pack_model` and the exported safetensors — *before* measuring anything.
Report teacher-forced argmax agreement, not greedy prefixes; a divergence at token 9
changes every token after it, which is why greedy was 0/12 identical on the fp16
control too.

**Gate:** SGLang serving gap of the same order as vLLM's, and orders below the
quantization effect. A gap near the *effect* means we are serving different weights,
not a numerically noisier runtime.

### S7 — MoE and tensor parallel

Gated on the equivalent vLLM results, because a layout bug found there would
invalidate this port. SGLang's MoE interface is the one place that genuinely diverged:
`FusedMoEMethodBase.apply(layer, dispatch_output) -> CombineInput` with a
`MoeRunnerConfig` and a required `get_triton_quant_info()`
(`base_config.py:85-120`), against vLLM's `apply(layer, router, x, router_logits)`.
Budget this as new code, not a port.

### S8 — Upstream

One PR to SGLang: a `register_weight_loader_v2_supported_method()` helper mirroring
`vllm/model_executor/layers/linear.py:69`. It closes a real API gap — SGLang's own
[quantization contribution guide](https://docs.sglang.io) documents only the in-tree
path — and we are the demonstrated caller.

SGLang has no AGENTS.md and no AI-disclosure policy, but `.claude/rules/` applies to
code we contribute: `msgspec.Struct` over `@dataclass`, no defensive
`getattr`/`hasattr`, keyword arguments, and `unit-test-admission.md` — every test must
answer "what future diff turns this red?". The negative test from S4 qualifies as a
completeness/negative-branch contract; a happy-path "it registers" test does not.

---

## Prerequisite

Not started until the vLLM side is actually tested: accuracy through vLLM (CaseHOLD on
Qwen3.5-2B, paired McNemar against the stored direct-run hits), TP>1 on a real model,
and MoE on a real model. All three can invalidate the layout this port copies.

### Status, 2026-08-02

**Accuracy — done, both execution paths.** Qwen3.5-2B at 3.25 bits, CaseHOLD, 5,314
items, per-item hits stored so every comparison is paired (McNemar, exact two-sided):

| arm | accuracy | vs direct run | 95% CI | agree |
|---|---|---|---|---|
| direct run (`p2_bodyonly`) | 86.96% (4621) | — | — | — |
| vLLM, eager | 86.96% (4621) | +0.00, p=1.0000 | [−0.13, +0.13] | 5,302 / 5,314 |
| vLLM, inductor + FULL CUDA graphs | 87.00% (4623) | +0.04, p=0.7905 | [−0.10, +0.18] | 5,300 / 5,314 |
| fine-tuned bf16 | 89.74% (4769) | −2.75, p<0.0001 | [−3.45, −2.04] | — |

The bound only means something beside the last row: the serving gap is under a fifth of
a point where quantization itself costs 2.75. Getting the compiled path there needed
[`fuse.py`](../packages/dynquant-core/src/dynquant/integration/vllm_plugin/fuse.py) —
inductor cancels `split(cat(...))`, and vLLM's piecewise boundaries record the stride
from before the cancellation. **This is the finding SGLang inherits**: any port that
joins per-shard matmuls hits it wherever the serving framework splits a compiled graph
and records strides across the split. Do not port the `torch.cat`.

**TP>1 — placement verified, engine run outstanding.** `tests/test_vllm_tp_placement.py`
builds both ranks of a `tp_size=2` layer and requires them to reassemble the checkpoint
exactly: fused qkv with three different widths, replicated KV heads, `gate_up_proj`, and
row-parallel word-axis splits at all four bit widths, plus the two refusals (a split off
a group boundary, a fused row-parallel layer). What is *not* covered is a real two-rank
engine, which needs a second GPU the current box does not have. The uncovered part is
vLLM's all-reduce, not DynQuant's arithmetic, but it is uncovered and should be run
before this port claims TP support.

**MoE — out of scope, and now genuinely fails closed.** DynQuant does not serve a fused
MoE through vLLM; the packed grouped GEMM is P8. So there is no MoE layout for this port
to copy, and the S-phases should not invent one. Testing the refusal found it was not
firing: through vLLM 0.25 `FusedMoE` was the class that owned expert weights, in 0.26 it
is a factory returning a `MoERunner` and the object passed to `get_quant_method` is a
`RoutedExperts`. The name-keyed probe matched nothing, and because `None` means "use
vLLM's unquantized method", the failure mode was fp16 experts built against packed
weights rather than an error. The probe now keys on the defining module and is swept
over every MoE class in the installed vLLM. **SGLang needs the same guard**, keyed on
whatever SGLang's equivalent is, and the same sweep — its MoE layer has been
reorganised at least as often.
