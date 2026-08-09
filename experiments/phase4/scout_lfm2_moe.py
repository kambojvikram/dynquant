"""Can DynQuant see `Lfm2MoeForCausalLM` at all, before any GPU time is spent on it?

LFM2.5-8B-A1B is the first target that is hybrid *and* sparse at once: 24 layers of
which 18 are LFM2 short-conv operators and 6 are attention, with 32 experts at top-4
above all but the first two. Three of this repository's recorded failures are latent in
that description, and each one produces a checkpoint that loads and answers rather than
an error:

* **A router at the MLP catch-all.** `mlp.gate` is a Linear that matches neither
  `gate_proj` nor the attention list, and a 3-bit router does not route. The structural
  test is `out_features == num_experts` with an `experts` sibling; this checks it fires.
* **Weights that are not modules.** The conv operator's kernel is plausibly a bare
  `nn.Parameter`, and `named_modules` cannot see one -- so it is quantized by nothing and
  counted in no byte total, which makes every "average bits" number a lie in the
  optimistic direction. Grouped-expert weights `[E, out, in]` have the same shape.
* **A tied embedding.** `tie_word_embeddings` is true and the vocabulary is 128 000 x
  2048, so `lm_head` and `embed_tokens` are one tensor. Counting it twice, or ignoring
  the head while it carries the embedding, is what made a "4-bit" baseline measure 7.36.

None of this needs the 8B checkpoint. A config with the real `layer_types` pattern,
`num_experts` and `num_dense_layers` at small widths reproduces the module tree exactly,
which is the only thing under test here.

    python experiments/phase4/scout_lfm2_moe.py
"""

from __future__ import annotations

import json
from collections import Counter

import torch
import transformers

from dynquant.graph.classify import classify_model
from dynquant.graph.roles import DEFAULT_FLOOR_BITS, ModuleRole

REPO = "LiquidAI/LFM2.5-8B-A1B"


def tiny() -> tuple[torch.nn.Module, transformers.PretrainedConfig]:
    """The real architecture at toy widths: same layer pattern, same expert count.

    Widths are shrunk and the layer list truncated to two repeats of the real
    `conv/conv/full_attention` motif plus the two dense layers, because what is being
    read is the module *tree* -- names, nesting and which leaves are parameters rather
    than modules. None of that depends on `hidden_size`.
    """
    cfg = transformers.AutoConfig.from_pretrained(REPO)
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.moe_intermediate_size = 32
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.layer_types = ["conv", "conv", "full_attention", "conv", "conv", "full_attention"]
    cfg.num_hidden_layers = len(cfg.layer_types)
    cfg.num_dense_layers = 2
    cfg.vocab_size = 256
    # The real special-token ids are five digits and `nn.Embedding` asserts on a
    # `padding_idx` outside the table, so they have to shrink with the vocabulary.
    cfg.pad_token_id, cfg.bos_token_id, cfg.eos_token_id = 0, 1, 2
    # The config comes back with the model because classification *reads* it, and
    # reading the real one against toy shapes is not a smaller version of the same
    # test -- it is a different, wrong one. `bank_orientation` decides which axis of
    # an expert tensor is the input dimension by matching its extents against
    # `hidden_size` and `moe_intermediate_size`; hand it 2048/1792 while the tensor
    # is [32, 64, 32] and nothing matches, so it returns UNKNOWN and the bank is
    # refused. That is what made this script report 8.5% coverage on a model the
    # graph in fact carries in full.
    with torch.device("meta"):
        return transformers.AutoModelForCausalLM.from_config(cfg), cfg


def main() -> None:
    cfg = transformers.AutoConfig.from_pretrained(REPO)
    print(
        f"{REPO}: {cfg.model_type}  {cfg.num_hidden_layers} layers, "
        f"{cfg.num_experts} experts top-{cfg.num_experts_per_tok}, "
        f"{cfg.num_dense_layers} dense, tied={cfg.tie_word_embeddings}"
    )
    print("layer_types:", json.dumps(Counter(cfg.layer_types)))
    print()

    model, tiny_cfg = tiny()

    # --- what a module walk sees, and what it misses -------------------------------
    module_leaves = {
        name: type(m).__name__
        for name, m in model.named_modules()
        if not any(True for _ in m.children()) and any(True for _ in m.parameters(recurse=False))
    }
    owned = {
        f"{mod}.{p}" if mod else p
        for mod, m in model.named_modules()
        for p, _ in m.named_parameters(recurse=False)
        if mod in module_leaves
    }
    orphans = sorted(set(dict(model.named_parameters())) - owned)

    print(f"leaf modules holding parameters: {len(module_leaves)}")
    kinds = Counter(module_leaves.values())
    print(f"  by class: {dict(kinds)}")
    print(f"parameters owned by NO leaf module: {len(orphans)}")
    for name in orphans[:12]:
        print(f"    {name:<60} {tuple(dict(model.named_parameters())[name].shape)}")
    print()

    # The custom leaves are where the risk is: none of them is an `nn.Linear`, and
    # between them they hold every expert in the model.
    print("what the custom leaves hold:")
    custom = {"Lfm2MoeExperts", "Lfm2MoeTopKRouter", "Lfm2MoeShortConv", "Conv1d"}
    seen: set[str] = set()
    for name, module in model.named_modules():
        kind = type(module).__name__
        if kind not in custom or kind in seen:
            continue
        seen.add(kind)
        print(f"  {name}  ({kind})")
        for param, tensor in module.named_parameters(recurse=False):
            print(f"    .{param:<22} {tuple(tensor.shape)}")
    print()

    # --- what the classifier makes of it -------------------------------------------
    graph = classify_model(model, config=tiny_cfg)
    by_role: Counter[str] = Counter()
    for info in graph.modules.values():
        by_role[info.role.name] += 1
    print(f"classified modules: {len(graph.modules)}   quantizable: {len(graph.quantizable())}")
    for role, count in sorted(by_role.items(), key=lambda kv: -kv[1]):
        floor = DEFAULT_FLOOR_BITS.get(ModuleRole[role])
        print(f"  {role:<18} {count:>4}   floor={floor}")

    # A batched expert bank is keyed by its *parameter* name, not the module's, so
    # the bank never appears in `graph.modules` even when both its tensors do.
    # Subtracting module names alone reported every bank as invisible.
    covered_leaves = {name.rsplit(".", 1)[0] for name in graph.modules} | set(graph.modules)
    unseen = sorted(set(module_leaves) - covered_leaves)
    print(f"\nleaf modules the graph does not carry: {len(unseen)}")
    for name in unseen:
        print(f"    {name:<58} {module_leaves[name]}")

    print(f"\nclassified OTHER: {len(graph.unclassified())}")
    for name in graph.unclassified()[:20]:
        print(f"    {name}")

    routers = [n for n, i in graph.modules.items() if i.role is ModuleRole.MOE_ROUTER]
    print(f"\nrouters found: {len(routers)}  (expect one per MoE layer)")
    for name in routers[:8]:
        print(f"    {name}")

    # --- the number that decides the campaign ---------------------------------------
    #
    # Counts are meaningless here; mass is the whole question. Anything the graph does
    # not carry is quantized by nothing *and* absent from the byte denominator, so a run
    # that skips it still prints a confident "3.25 average bits" -- computed over the
    # tensors it did compress. The full config on the meta device costs no memory and
    # gives the exact figure.
    print("\n" + "=" * 78)
    with torch.device("meta"):
        full = transformers.AutoModelForCausalLM.from_config(cfg)
    full_graph = classify_model(full, config=cfg)
    carried = set(full_graph.modules)

    params = dict(full.named_parameters())
    total = sum(t.numel() for t in params.values())
    covered = 0
    missed: Counter[str] = Counter()
    for name, tensor in params.items():
        owner = name.rsplit(".", 1)[0]
        # Two spellings, because the graph keys `nn.Linear` weights by the *module*
        # (`...q_proj`, whose tensor is `...q_proj.weight`) and batched expert banks
        # by the *parameter* (`...experts.gate_up_proj`, which owns no module of its
        # own). Testing only the first spelling counts 91% of a sparse model as
        # missing; testing only the second counts every dense weight as missing.
        if owner in carried or name in carried:
            covered += tensor.numel()
        elif tensor.ndim >= 2:  # norms and biases are never quantized by anyone
            missed[type(full.get_submodule(owner)).__name__] += tensor.numel()

    refused_params = full_graph.skipped_params()
    print(
        f"  refused with a reason  : {len(full_graph.skipped)} tensors, "
        f"{refused_params / 1e6:.2f} M params priced dense"
    )
    for skipped_name, entry in list(full_graph.skipped.items())[:4]:
        print(f"    {skipped_name}: {entry.reason[:110]}")

    print(f"{REPO} at full size: {total / 1e9:.2f} B parameters")
    print(f"  carried by the graph : {covered / 1e9:>6.2f} B  ({covered / total * 100:.1f} %)")
    for kind, n in missed.most_common():
        print(f"  NOT carried, {kind:<20}: {n / 1e9:>6.2f} B  ({n / total * 100:.1f} %)")


if __name__ == "__main__":
    main()
