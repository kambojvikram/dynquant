# dynquant-core

The pure-Python half of [DynQuant](https://github.com/kambojvikram/dynquant):
training-dynamics-driven mixed-precision LLM quantization.

Most users should `pip install dynquant`, which adds the prebuilt CUDA kernels.
Install `dynquant-core` directly when you want the quantization pipeline without
any binary dependency — CPU-only machines, Windows, macOS, ARM, and CI.

## What is in here

* `dynquant.signals` — the training-time hook and the stats file format
* `dynquant.graph` — architecture-generic module role classification
* `dynquant.quant` — group-wise n-bit packing and the quantizer driver
* `dynquant.runtime` — backend selection and the packed `Linear`
* `dynquant.cli` — the `dynquant` command, including `dynquant doctor`

Everything except inference speed works with no compiler and no GPU. The
reference (`torch`) backend dequantizes to compute, so it saves no memory and is
not fast — but it is the oracle the CUDA kernels are tested against, and it is
what makes quantizing a checkpoint possible on a laptop.

## Install

```bash
pip install dynquant-core            # quantize anywhere
pip install 'dynquant-core[hf]'      # + transformers, accelerate
pip install 'dynquant-core[train]'   # + peft, trl, datasets
pip install 'dynquant-core[dev]'     # + pytest, ruff, mypy, docs
```

## License

Apache-2.0.
