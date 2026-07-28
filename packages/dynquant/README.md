# dynquant

**Mixed-precision LLM quantization that decides bit-widths from your fine-tune's own training dynamics.**

```bash
pip install dynquant
dynquant doctor
```

This distribution contains no code. It is the one name to install, and it pulls in:

* [`dynquant-core`](https://pypi.org/project/dynquant-core/) — the Python half:
  signal collection hook, role classification, scoring, allocation, packing, CLI.
  Installs anywhere, no compiler required.
* [`dynquant-kernels`](https://pypi.org/project/dynquant-kernels/) — prebuilt CUDA
  kernels, where a wheel exists for your platform. Without them everything still
  works on the reference backend; you lose inference speed and the VRAM saving,
  not correctness.

## Usage

Collect signals during the fine-tune you were going to run anyway:

```python
from transformers import Trainer
from dynquant import DynQuantCallback

trainer = Trainer(model=model, ..., callbacks=[DynQuantCallback("stats/")])
trainer.train()
```

Allocate and pack:

```bash
dynquant quantize ./merged --stats stats/dynquant_stats.json --target 3.0 -o ./q3
```

Load through plain transformers:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./q3")
```

Full documentation: <https://github.com/dynquant/dynquant>

## Extras

```bash
pip install 'dynquant[train]'     # transformers, peft, trl, datasets
pip install 'dynquant[eval]'      # lm-eval-harness
pip install 'dynquant[triton]'    # portability fallback for ROCm / newer GPUs
pip install 'dynquant[kernels]'   # force the compiled kernels (builds from sdist)
```

## License

Apache-2.0.
