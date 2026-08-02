# SGLang integration plan

Serve DynQuant checkpoints on SGLang without patching SGLang, the same way
`dynquant.integration.vllm_plugin` does for vLLM.

Written against SGLang `131bd51b0` (2026-08-01). Every claim below cites the file
it came from; re-check the citations before starting, because none of these
surfaces are covered by a compatibility guarantee.

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
| `from_config` payload | the `quantization_config` dict | the dict **plus `packed_modules_mapping` and `hf_config`** | simplification |
| Method registration | `@register_quantization_config` | **no public API** | S2 |
| v2 weight loader opt-in | `register_weight_loader_v2_supported_method()` | **no public API** | S2 + S8 |

### The three containers a plugin must mutate

| Container | Location | Public setter |
|---|---|---|
| `QUANTIZATION_METHODS` | `layers/quantization/__init__.py:143` | none — read live by `get_quantization_config` and by `ModelConfig._verify_quantization` (`configs/model_config.py:1372`), so in-place insert works |
| `QUANTIZATION_CHOICES` | `server_args.py:137` | **yes** — `add_quantization_method_choices()` at `server_args.py:359` |
| `WEIGHT_LOADER_V2_SUPPORTED` | `layers/linear.py:57` | none — a list of class-name *strings*, matched at `linear.py:369` and `:1453` |

`WEIGHT_LOADER_V2_SUPPORTED` is the load-bearing one. The v1 loader places shards
with `param.data.narrow(output_dim, ...)`; on DynQuant's flat buffers that narrows
into the wrong rows and copies **without raising**. Getting this wrong produces a
model that loads, serves, and is quietly wrong.

### Ordering is already correct

`load_plugins()` (`srt/plugins/__init__.py:103`) runs before anything reads the
containers, in every process that matters:

- `cli/serve.py:99` — before `prepare_server_args`, so argparse sees our choice
- `entrypoints/engine.py:220` — before `ServerArgs` construction, by explicit comment
- `entrypoints/engine.py:1013` — defensive re-entry
- `managers/scheduler.py:4781` — **in the spawned scheduler process**, which is where
  `ModelConfig` and `_get_quantization_config` actually run

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

`geometry.py` (419 lines) and `schema.py` (288) import nothing from vLLM. That is 707
of 1384 lines, and it is the half that fails silently rather than loudly.

1. Move both to `dynquant/integration/serving_common/`.
2. `vllm_plugin/` imports them from the new location; **no behavior change**.
3. Rename `tests/test_vllm_geometry.py` → `test_serving_geometry.py`, same for schema.

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
- `from_config` now receives `packed_modules_mapping` and `hf_config`
  (`model_loader/weight_utils.py:281-282`). Consume the mapping instead of
  re-deriving fusion from module names — delete that derivation on this path.
- `get_quant_method`: `LinearBase` → `DynQuantLinearMethod`; `VocabParallelEmbedding` →
  our embedding method; `FusedMoE` → deferred to S7; `RadixAttention` → `None`.
- `VocabParallelEmbedding` (but not `ParallelLMHead`) requires the method to implement
  `embedding()` — `vocab_parallel_embedding.py:305-311`, same rule as vLLM.
- `override_quantization_method` returns `None`. It is called on **every** registered
  config during detection (`model_config.py:1459`), so it must be cheap and total.

**Gate, CPU only:** construct `ModelConfig` on the exported checkpoint with no
`--quantization` and assert `quantization == "dynquant"`, resolved from `config.json`.

### S4 — Parameter and linear port (highest risk)

The one real semantic difference. vLLM's parameter classes read tensor-parallel state
from globals; SGLang passes it in:

```python
# sglang/srt/layers/parameter.py:145
def load_column_parallel_weight(self, loaded_weight, tp_rank, use_presharded_weights=False)
# and :293 load_row_parallel_weight(self, loaded_weight, tp_rank, use_presharded_weights=False)
```

and the call sites pass `tp_rank=self.tp_rank, tp_size=self.tp_size` explicitly
(`linear.py:866-878`). Our flat-buffer overrides must accept and honour those instead
of calling `get_tensor_model_parallel_rank()`. A subclass that silently ignores the
kwarg still runs — and shards wrong on every rank but 0.

Also: `create_weights` takes an extra `skip_block_quant_check` kwarg
(`linear.py:358-370`); SGLang uses `copy_with_check` rather than a bare `copy_`; and
there are `_is_cpu` padding branches we should assert we never enter.

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
