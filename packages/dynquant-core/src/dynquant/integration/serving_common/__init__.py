"""The half of a serving plugin that does not know which server it is in.

vLLM and SGLang need the same three things from DynQuant, and none of the three
has an opinion about the framework:

* :mod:`~dynquant.integration.serving_common.schema` -- reading the per-module bit
  map out of ``config.json`` and answering "which modules does this fused layer
  contain?"
* :mod:`~dynquant.integration.serving_common.geometry` -- where each shard's packed
  words live inside one flat parameter, which is the arithmetic that makes a fused
  projection with disagreeing widths expressible at all.
* :mod:`~dynquant.integration.serving_common.fuse` -- an opaque custom op that stops
  inductor from cancelling ``split(cat(...))``.

Splitting these out is not tidiness. It is where the risk is.

SGLang's quantization layer is a fork of vLLM's -- ``base_config.py`` still carries
``Adapted from .../vllm/v0.5.5/...`` -- so the two plugins' framework halves are
import swaps and shallow signature drift, and they fail *loudly*: a renamed base
class is an ``ImportError`` at registration. This half fails quietly. A shard offset
that is wrong by one group produces a model that loads, serves, and answers slightly
worse, which is indistinguishable from quantization loss and is the failure mode that
would survive a smoke test.

Second, and more practically: SGLang stopped shipping pure-Python wheels after
0.5.10.post1 and vLLM never did, so neither plugin's framework half can even be
imported on a development laptop. Everything in this package is importable with
nothing but torch, which makes it the only part whose arithmetic gets checked outside
a GPU box. Keep it that way -- an import of vllm or sglang anywhere below this
package silently deletes that property.

``fuse.py`` additionally *must* live here rather than be copied: it registers
``dynquant::fused_shard_concat`` in the process-global ``torch.library`` namespace,
and a second copy under another plugin package raises on the import that follows the
first.
"""

from __future__ import annotations

from dynquant.integration.serving_common.fuse import fused_shard_concat
from dynquant.integration.serving_common.geometry import (
    FusedPackedGeometry,
    ShardPlan,
    ShardSpec,
    TensorParallelSplit,
    match_shards_to_partitions,
    row_parallel_split,
)
from dynquant.integration.serving_common.schema import (
    CHECKPOINT_FORMAT,
    SCHEMA_VERSION,
    ModuleQuantSpec,
    QuantizationConfigSchema,
    expand_fused_prefix,
)

__all__ = [
    "CHECKPOINT_FORMAT",
    "SCHEMA_VERSION",
    "FusedPackedGeometry",
    "ModuleQuantSpec",
    "QuantizationConfigSchema",
    "ShardPlan",
    "ShardSpec",
    "TensorParallelSplit",
    "expand_fused_prefix",
    "fused_shard_concat",
    "match_shards_to_partitions",
    "row_parallel_split",
]
