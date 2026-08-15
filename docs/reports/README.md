# The experimental record

Every experiment run on DynQuant since the package was built, what it measured, and where the
full record lives. Nothing here is taken from the paper; everything is measured on this
repository's own code. Questions 1–6 were measured on a single NVIDIA A100 80GB PCIe;
from question 7 the box is one NVIDIA RTX PRO 6000 Blackwell Workstation Edition (97 887 MiB,
driver 580.159.03). Nothing is measured across the two — every comparison in a given campaign
was run on one machine.

There are eighteen campaigns — seventeen of them measured on a GPU, the eighteenth an audit of
how all seventeen counted their bytes. They answer twenty-eight questions, in this order — phase 4
answers ten of them, because whether a benchmark can read damage, whether the model has
already seen its answers, whose bytes “matched bytes” means, which of two prices chose the
widths, whether the driver that runs the arms runs at all, how many of its variants can be
published, whether the model it scores is the one a person would download, what the margin
it finally reports is a difference *in*, whether the nineteen points it reports at 3 bits
belong to the signal or to the shape of the map, and whether any of it survives on a dense
model of another family are ten separate failures:

| # | question | verdict | full record |
|---|---|---|---|
| 1 | Does a signal-driven allocator beat a same-size uniform one? | **Only after the score was replaced.** The published rank-product score *lost* by 2.03 pts; a measured Gauss–Newton sensitivity wins by **+10.29** | [`RESULTS.md`](../../experiments/four_point/RESULTS.md) |
| 2 | Does that hold on a different model, scale, architecture and training regime? | **Yes, qualitatively; not in magnitude** | [`RESULTS-mistral7b-banking77.md`](../../experiments/four_point/RESULTS-mistral7b-banking77.md) |
| 3 | Does it beat what people actually ship — GPTQ, AWQ, RTN, bnb-NF4? | **Wins at 2.42×, ties at 3.8×, lost at 4.9×** | [phase 1 PDF](https://github.com/kambojvikram/dynquant/releases/latest/download/dynquant-phase1-external-comparison.pdf) · [record](../../experiments/four_point/RESULTS-external-comparison.md) |
| 4 | Can the 3-bit loss be reversed without adopting GPTQ's mechanism? | **Held pending a control.** +1.54 over GPTQ at 7.4 % fewer bytes, *p* < 0.0001 — but that GPTQ arm was fitted **symmetric** where DynQuant was asymmetric, and on Mistral-7B that difference alone was worth **69.4 points** (§23). `gptq_3b_head` sits 1.71 points under its own fp16 ceiling and the claimed margin over it is 1.54 | [phase 2 PDF](https://github.com/kambojvikram/dynquant/releases/latest/download/dynquant-phase2-beating-gptq-3bit.pdf) |
| 5 | Do inference servers hold the same quantized weights the direct run holds? | **Yes, on both vLLM and SGLang** — after a real defect was found and fixed | [`serving-parity.md`](serving-parity.md) |
| 6 | Do the phase-3 benchmarks have room for quantization damage to show? | **Yes, all four** — 54.5–83.2 %, no arm near ceiling; two harness defects caught | [`phase3-s1-headroom-screen.md`](phase3-s1-headroom-screen.md) |
| 7 | Does the S2 fine-tune know which tokens are the assistant's? | **Yes on both tokenizers, ≤0.07 % unmaskable** — after the obvious method dropped 100 % of the data on one of them | [`phase3-s2-loss-masking.md`](phase3-s2-loss-masking.md) |
| 8 | What does fusion cost the one panel model that has it? | **0.21 of 4.43 floor bits** — and the two panel models are in different regimes at the same target | [`phase3-s3-fused-floors.md`](phase3-s3-fused-floors.md) |
| 9 | Did the S2 fine-tune produce a signal map worth spending? | **Yes, with one caveat** — `embed_tokens` scores on nothing and pays 67.8 % of the floor shortfall | [`phase3-s2-phi-signal-map.md`](phase3-s2-phi-signal-map.md) |
| 10 | Is S3's shuffled control actually ablating the signal? | **No — it was a null by construction.** 0 of 129 modules differed from the treatment; it permuted a file the allocator had stopped reading | [`phase3-s3-null-control.md`](phase3-s3-null-control.md) |
| 11 | Did the second S2 fine-tune produce a signal map worth spending? | **Yes, and it exposed a ranker defect** — a singleton role group scores 0.5 against itself, costing the baseline a whole bit on 6.7 % of the model | [`phase3-s2-ministral-signal-map.md`](phase3-s2-ministral-signal-map.md) |
| 12 | Does S3's resume guard know a fresh map from a stale one? | **No, twice over** — it compared against a file it regenerates, and called a metadata reordering a content change | [`phase3-s3-reuse-guard.md`](phase3-s3-reuse-guard.md) |
| 13 | At matched bytes, what are the allocator and the signal each worth? | **+4.31 over uniform at 3.25 bits, split 1.91 allocator / 2.41 signal — and nothing at 4.25, where the maps diverge just as far and buy nothing** | [`phase3-s3-allocation.md`](phase3-s3-allocation.md) |
| 14 | Can the phase-4 text-to-SQL benchmark tell a correct answer from a broken one? | **Only after four defects were removed** — two SQLite-semantics bugs discarding a third of one corpus, a DML block teaching an unscoreable answer format, and a registered task argparse refused | [`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) |
| 15 | Does the training mixture contain the questions the arms are scored on? | **Yes — 189 of the 200 WikiSQL evaluation items are questions in `b-mc2/sql-create-context`**, and the guard that reported the mixture clean could not have fired | [`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §11 |
| 16 | “At matched bytes” — whose bytes? | **The baselines’** — DynQuant’s own format costs 4.25 bits/param at 4 bits against `compressed-tensors`’ 4.15625, so anchoring on its uniform arm would hand it **2.3% more bytes inside the arm whose accuracy is the claim** | [`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §12 |
| 17 | Which of DynQuant’s two prices chose the widths, and on how much of the model? | **44 of 133 modules — 91.54% of parameters — by the rank-product proxy rescaled by 1.807e-17**, because a batched expert bank has no boundary where the Gauss–Newton form exists; the concordance guard reads 1.000 over the other 8.46% | [`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §12 |
| 18 | Does the seven-arm panel driver run? | **Not until six defects were removed** — three of them fail *after* the calibration pass, one would have finished and entered the table as round-to-nearest wearing an AWQ label, the fifth would have run all seven arms over the whole 16 143-item test split because `--limit` is forwarded only when set, and the sixth would have let `--resume` staple in a record scored on another checkpoint | [`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §12 |
| 19 | How many of the six quantized variants can actually be published? | **None of the four baselines in the container that was promised; all six in DynQuant's own** — `dynquant export` refused a batched expert bank and said the packed *format* could not hold one; the format always could, and the refusal came from a second copy of the name resolver -- which then refused a second time, on a router, because the copy had been narrowed twice. Both DynQuant arms now write packed at their manifest bytes, size-honest and not yet loadable -- but *not loadable* turned out to mean transformers skipping the unknown `quant_method` and **returning a randomly initialised model with no exception**, measured identically on 4.53.2, 5.10.1 and 5.14.1, none of which has entry-point discovery. An `HfQuantizer` now makes that a hard error, round-trips a dense model to 4.9e-4 of the encoder, and stops a tied `lm_head` crashing on a packed embedding that has no `.weight`. And *needs the grouped path* was two blockers read as one: the kernel that makes batched experts fast, and the object that lets them be held at all. The second is Python -- the parent reaches an expert by **indexing**, so a module registered under the parameter's own name intercepts it and dequantizes 10.5 MiB of a 336 MiB bank per hit. That is 91.5% of this model, and it was also 91.5% missing from the byte denominator, which walked modules and so never saw a tensor no module owns. The two 3-bit baselines were refused for a false reason as well: `compressed-tensors` packs 1-8 bits and round-trips 3-bit fine, but at `32 // 3` values per word it stores **3.2 bits against a label of 3**, and vLLM sizes the same tensor as `Fraction(32, 3)` -- 192 words per 2048-wide row where 205 were written. The count held and the identity flipped. `gptq_4b` and `awq_4b` were the row's "yes -- vLLM and transformers", and they are the two that cannot be published: the recipe reaches 91.5% of this model by **renaming** the banks, `ARCH_TO_2D_MAPPINGS` registers the inverse for `deepseek_v4` and `qwen2_moe` and nothing else, and `lfm2_moe` linearizes through the generic protocol -- so the surgery runs and its inverse does not exist. Measured through the shipped `save` on a four-layer model: 108 packed expert tensors written, all 108 `UNEXPECTED` on reload, both banks `MISSING`, **no exception**, finite logits, and the reloaded bank at **32 distinct values in a 32-value group** where 4-bit allows 16 -- against 13 for the same instrument on a `dynquant quantize` directory. vLLM keys its expert loader on `("w1", "w2", "w3")`, so it does not find them either. `do_save` now refuses on llm-compressor's own predicate, before the calibration pass, and the panel is untouched because `run` never serializes. §12 of [`phase4-packed-moe-runtime.md`](phase4-packed-moe-runtime.md) then answered this in a different container rather than by fixing the promised one -- read row 20 for it, and note that it is measured on a four-layer `lfm2_moe`: the real 8B has not been published yet | [`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §12 |
| 20 | Is the model the panel scores the model a person would download? | **It was five defects away, and the fifth was in this report** — a rank-3 test that could not tell a batched expert bank from a `Conv1d` kernel, on a model where **18 of 24 layers are conv**; a packed bank that installed cleanly and then died *inside transformers* at `weight.transpose(-2, -1)`, because 5.14.1 replaces every `*Experts.forward` with a dispatcher whose **default is `grouped_mm`**, so the indexing loop the whole design rests on is not what runs; and the packer and the in-place encoder disagreeing by **0.0082** on the same fp32 bank — 16% of what quantization itself moves — because the scale dtype had **three definitions** that agree on fp16 and bf16 and therefore on every model anyone ships. All three are now zero, or exactly 0 where exactness is the honest bar, verified against the genuine `Lfm2MoeExperts` — which then found a fourth, an export whose own reader refused it, and a fifth that was this report's own claim: it called the dispatch move **free** on the strength of that one-layer 1.79e-07, and at 8B the two dispatches disagree on **1.24% of teacher-forced tokens**, **0.29x** the quantization effect, on the same axis as the panel's margins. Corrected in five places and pinned in code. §11 corrects §8's mechanism: a linearised baseline keeps `_experts_implementation` and its `*Experts` modules, so "no dispatch left" was reasoning about modules while the code read the config — and it corrects itself in turn: the four baselines are not unrecoverable, their own `banks_after: 0` was counted in the process that scored them, so the genuinely unrecorded arms are `bf16`/`dq_4b`/`dq_3b`, which is exactly the re-score set. Then twice more, from asking whether the re-score would *finish* rather than whether it was justified: the driver's guard would have killed it on arm two, and the table's would have blanked **every row in every block** — the re-score clearing every caveat and deleting the numbers they annotate. Comparability is a property of a pair, and `experts.ran` is not part of it: *paired* means the hit vectors index the same items, which a dispatch difference does not break. It is priced on the row instead, and the resulting caveat-free table now prints the one claim holding it up. §8 also finally prices its own title: the linearised arms cost **1.9—2.3x** the banked ones over the same 12,000 items, which re-brackets the re-score at **8.5—17 hours** and makes it the first `eager`-against-`grouped_mm` clock this campaign will have taken. That paragraph then corrected itself within the day: it had offered the 18% between GPTQ and AWQ as a bound on the dequantization confound, calling it "kernel choice alone" — but the two manifests are the same 4 bits at the same group size for the same 4,399,629,312 bytes, so there is no second kernel, only asymmetric-against-symmetric dequant and the length difference two sets of weights produce. **A difference inside dequantization does not bound dequantization**, so the split stays unmeasured, and the two probes that can split it are the re-score and 24 teacher-forced items on bf16 — both with identical weights on either side, hence no dequant in either. The 3-bit half then answered itself from a file written for another reason. The box's progress sampler stamps every 800-item line, so the panel's own log is an interval profile over the same items in the same order — which the four records confirm to the item, 3,063 gretel and 8,937 wikisql each. Aligned block-for-block, linearised-against-banked swings **1.82—3.78x** where a fixed per-forward cost would be flat, capping that cost at 1.82 against an aggregate of 2.55 and leaving **at least 1.40x as decode steps**; and `gptq_3b` against `awq_4b` — both linearised, so the dispatch is held fixed and only the width and weights move — swings **1.12—2.95x** over five shared blocks. A fixed unpack cost is the same work every block, so a 2.6x swing falsifies it outright and **1.12 is the ceiling worth defending** — at most 12% of a forward against an aggregate of 1.62x, leaving at least 1.45x as decode steps. There is a block reading 0.96, which would settle it more strongly still, but it divides by `awq_4b`'s own anomaly — 2.70 where that arm's median is 1.50 and `dq_4b` is flat — and a quotient inherits the reason its denominator was slow, so it is not the number quoted. Either way the 3-bit slowdown is generation length, settled for free where this report had priced it at a re-quantization. The 1.82 also puts an argument under the top of the re-score bracket, which had none: that re-score holds the weights fixed on both sides, so length cannot move and the fixed multiplier is the only one in play — at most 1.82x the banked arms' own seconds and for the *loop* at that, with `eager` under it, which lands the ceiling near **15 hours** rather than 17. §8 then answers its own title: the move it priced is **not required**. `dynquant_experts_forward` indexes a packed bank from inside the grouped path and measures **bit-identical** to `grouped_mm` at bf16 and fp32 — 0.00% of argmax tokens against `eager`'s **1.95%**, at the 8B's own MoE geometry — so the condition every packed figure carried is retired rather than restated, and the pin survives only for linearised baselines, which have nothing left to dispatch. Two false starts are kept because they were the informative part: a tiny geometry where all three dispatches were bit-identical and an earlier draft divided by that zero to print `infx closer` — the zero was real, proven by confirming each dispatch ran, and the fix was to scale width and `k` rather than depth — and a sentinel-offset bug the probe was structurally blind to and a two-expert unit test caught immediately, since bands index a *sorted* array and one over-wide bin displaces every band after it. Six mutations of the forward, six named tests. Then the banked records were finally read for their dispatch, confirming what §11 predicted and widening it: the `experts` key is **absent**, not null, so `_comparability` exempts all five, and the straddle is five of five including `bf16` — no arm in the banked panel chose its dispatch. §10 closes the last of §7's three named byte gaps: the tensors classification *refuses* were recorded with a reason and left out of the denominator, which is **205 KB of 4.4 GB here** and **91.5% of the model** on a MoE whose banks are refused for orientation. §12 answers row 19 in a different container. A foreign grid is a grid: `scale * (q - zero)` *is* `scale * code + offset` with `code = q` and `offset = -scale * zero`, so the four variants `compressed-tensors` cannot hold can be written as DynQuant with the codes carried **unchanged** rather than re-fitted. All six baseline arms of a four-layer `lfm2_moe` published and reloaded through the real `HfQuantizer`: rtn and gptq at **0.0000 code steps** from the weights the in-process arm was scored on, awq at **0.0681** and **0.0240** against a 0.125 budget, where re-fitting the same weights -- the same code path with the encoder swapped -- moves them **0.399--0.518**. Three defects nothing smaller than a real recipe could reach: the recipe's own `weight_scale` / `weight_zero_point` / `weight_g_idx` published as model weights; the tied table written under `lm_head` where the loader reads `model.embed_tokens`, because the first fix gated on `config.tie_word_embeddings` and **`oneshot` sets that to `False` while the storage stays shared** -- a config true of the compressed checkpoint it never wrote; and the first *asymmetric* arm refusing at 6.976e-02 with the weight on its own lattice to **0.0377** of a step, because the reader had worked the integer range out for itself as unsigned `[0, 2^b-1]` where `compressed-tensors` puts *every* integer scheme on the signed band and lets asymmetric ride it with a signed zero point (**-4..3** measured), clamping **7,443 of 12,288** elements onto a rail that does not exist. Imported now, not derived -- the same shape as the six duplicated-registry cases below, one step further out, since a second copy of a *dependency's arithmetic* is a copy nothing in this repository can contradict. And the container is not free: **4.25 bits at 4 and 3.25 at 3** against `compressed-tensors`' 4.15625, so a republished `gptq_4b` costs **+99 MB** for identical codes while a republished 3-bit is *smaller* than the container that could not hold it honestly at all. And the carry check answers only half of it: it proves the export is faithful to the model in memory, but that model is a *second* calibration pass, because `run` scores in process and writes a record rather than a checkpoint. `publish --scored <arm>.quant.json` compares the second pass against the arm's own record -- six flags before the recipe, ten weight fingerprints after -- and the byte accounting alone could not have done it: `gptq_4b` and `awq_4b` agree on every parameter count and differ only in what the recipe did to the weights (**0** modules moved against **2,201**, by **0.0** against **2.890625**) — though the byte figures this row used to offer as agreeing, **4.1565** bits and **4,399,629,312** bytes, do *not* agree, per question 28: the size column was blind to symmetric-against-asymmetric, the one axis separating those two arms. The last document is the one anyone actually reads, so it is generated too: `model_cards.py` builds each arm's README from `panel_table --json-out` and the fine-tune's own record, typing no number, and refuses the ceiling, an unscored arm and an unknown label rather than describing them. Writing it found the table dropping `question` and `same_arithmetic` on the way into json -- survivable while the only reader was a terminal with the dispatch census two blocks up, and not once six Hub READMEs are built from that payload, since the card would publish `separated` with its **0.29x** confound stripped off. Three caveats are emitted from the row rather than from a checklist -- scored-in-bf16 for a map arm, **+2.3%** on disk for a recipe arm, `[^1]` for a flagged pair -- because carrying all three on every card would tell a DynQuant reader their directory is oversized and a GPTQ reader their accuracy came from somewhere else, and both are false. §13 takes the other half of §8: the dispatch is settled, the *loop* is not, and its cost was never arithmetic. `_segment_offsets` returned a Python list, and building one is a device-to-host copy — **44 fences per token** on 22 MoE layers holding two banks each. The copy is nanoseconds; what it costs is that a forward containing a host read cannot be CUDA-graph captured and cannot be traced under `fullgraph=True`. The table is an `[E + 1]` **int32 device tensor** now, the grid comes from `seg_offsets.shape` and the values from two `__ldg` loads, and the whole function traces because `bincount` and `cumsum` are shape-determined — **except `bincount` is not.** It sizes its output from a host read of `input.max()`, and `minlength` raises the floor on that size without removing the read, so one fence per bank per layer survived *on the fused path* and this row's own counter could not see it: `bincount` never calls `.tolist()`, so an exact counter answered a narrower question than the section claimed. It is a `scatter_add_` into a fixed `[E + 1]` buffer now, found by CUDA-graph capture rather than by counting — see row 24. Removing a fence changes no output, so the property is asserted by *counting* `.tolist()` on 1-D int32 tensors: two per layer on the loop path, and the two left on the fused path are the stand-in kernel's own, standing in for the device's loads — the caller takes none. ABI 3 is additive by design: `KERNEL_ABI_VERSION` 3, `MIN_KERNEL_ABI_VERSION` still **2**, and the runtime asks the op table rather than the number, because an ABI-2 wheel serves every model it served yesterday and refusing it trades a slower correct answer for none. Four supported configurations get the loop rather than an exception, the quietest being a transposed bank — not a crash but the wrong expert's rows at the right shape. The nine-mutation run ended at zero survivors only after the informative failure: `expert_ids % num_experts` survived three rounds because **both paths read the same table**, and an equality assertion between two consumers of one input cannot see that the input is wrong. It took a known answer — three experts and one sentinel, where folding the sentinel in moves expert 2's row into the band before it and token 1 comes back 3.0 where the answer is 5.0. And the ABI's *third* declaration was found the way it was meant to be, at `assert 2 == 3`, by a lint that reads the last one from source text so it runs on a CPU box. And the header's own bit-exactness claim was **wrong about which thing it was identical to**, caught by reading rather than by a red GPU test: `dynquant::gemv` is two kernels, the grouped loop is `gemv_kernel` line for line, and `gemv_vec_kernel` -- the one `gemv` picks for every geometry a transformer contains -- is already pinned at **2e-3, not zero**, by this file's own two-paths test. So an exact assertion against the *op* would have failed on the shapes that matter and passed on the leftovers. The claim names the kernel now and the test splits: CPU exact and parametrized, CUDA one subprocess under `DYNQUANT_GEMV_SCALAR=1`. What is still not claimed: **`grouped_gemv.cu` has never been compiled**, there is no vectorized variant so a busy expert decodes at the general path's bandwidth, and no speedup exists yet -- only a counter. The first clause is retired by row 24: it compiles and its five parity tests are 23 passed at first run. The other two stand. | [`phase4-packed-moe-runtime.md`](phase4-packed-moe-runtime.md) |
| 21 | Is the 4-bit margin a difference in method, or in how closely each arm tracks the ceiling — and was it the expert dispatch all along? | **Fidelity, one identity produces both of its signs, and the dispatch was worth a tenth of a point in the *other* direction** — split by source the +0.78 over GPTQ is +1.24 on wikisql and -0.59 on gretel, which Cochran's Q calls heterogeneous; but the bf16 ceiling is 88.50% on wikisql against 72.02% on gretel, so on this panel *source* and *hard* are the same column. Stratify instead by whether the **ceiling** got the item right — a label owing nothing to either compared arm, so McNemar stays valid inside each half — and the margin varies far more than sampling explains, **+1.26** over 10 115 items against **-1.80** over 1 885, Q=11.70 at Holm 0.00125. The sign flip is a point estimate and the negative cell no longer separates on its own (Holm 0.126), so the claim rests on the spread, which is the stronger test of the thing actually claimed. The account is one measurement: a hit is a boolean, so `accuracy = c·f_right + (1-c)·(1-f_wrong)` is an **identity**, and DynQuant agrees with the ceiling on **95.30%** of items against GPTQ's 93.96% (**+1.34**, *p*=1.01e-07) while the two baselines do not separate from each other on fidelity (+0.37, *p*=0.177) any more than on accuracy (*p*=0.272). An arm that tracks the ceiling more closely inherits its right answers *and* its wrong ones, so it **must** win the 84% and lose the 16%: the second stratum is the first's fidelity delta with its sign flipped, identical *p*, discordant counts swapped. Every 4-bit *fidelity* margin is now flat across the two sources, and the whole per-source spread in *accuracy* is the identity applied to two mixtures — gretel is 28.0% ceiling-wrong against wikisql's 11.5%, so the term DynQuant loses is weighted two and a half times more heavily there, and the four rows sum to the observed margins exactly. **Two controls this row used to lean on did not reproduce**: GPTQ-vs-AWQ was heterogeneous across the four crossed cells at Q=8.73 and is now consistent at Q=4.41 (Holm 0.22), and its fidelity margin *changed sign* between sources (+1.76 / -0.13) and now does not (+1.31 / +0.04, Holm 0.125). Both were within-noise structure at Holm ≈ 0.015 across a six-row block, which is roughly how often that block is entitled to produce one. What replaces them is mechanical rather than empirical — opposite signs across difficulty are what fidelity *is* when read as accuracy — and the 3-bit baseline pair still carries real cell structure (Q=94.46 on a +2.84 pooled margin). **And the confound is closed.** `rescore_eager.sh` re-scored `bf16`, `dq_4b` and `dq_3b` under the baselines' expert kernel, and the records carry `experts: {found: grouped_mm, ran: eager}` — so the first panel's banked arms really did dispatch `grouped_mm` and this is a measurement, not a relabelling. Dispatch moves **0.85–1.76% of items**, close to §8's 1.24% estimate in magnitude, but **53 up against 49 down** on bf16: near-symmetric noise, which is the one thing §8 could not supply and no argument was going to. Removing it made every margin **larger** — 4b +0.64→**+0.78** vs GPTQ and +0.94→**+1.08** vs AWQ — which is the wrong direction for a confound that had been supplying the effect. Both panels are committed, so the comparison no longer depends on a box whose `/workspace` is not a volume | [`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §13 |
| 22 | At 3 bits DynQuant beats GPTQ by nineteen points — is that the signal, or the shape of the map? | **Measured — and then halved by a control: nine of the nineteen points were a GPTQ default rather than a map, and against the recovered baseline the signal carries 98.3% of what is left.** At 3 332 904 576 B — 5.08x compression, every arm within 0.05% of the anchor — GPTQ scores 60.76% and AWQ 57.92% against a bf16 ceiling of 84.29%; DynQuant scores **79.89%**, **+19.13** over GPTQ (+2781/-485) and **+21.98** over AWQ (+3150/-513), both at *p* below double precision — but that GPTQ was fitted *symmetric*, and the control at the end of this row re-fits it. It is the only 3-bit arm still emitting SQL: 213 unparseable generations against 1 008 and 1 523. Four control arms decompose that margin, run at the same anchor with the same flags and dispatch, differing only in what the allocator was shown: `shuffle` permutes the driving quantity within role, `flat` draws the same permutation and then sets every score to 1.0, `table` keeps the measured table by identity and sets every score to 1.0, `uniform` flattens every score and consults no sensitivity table. They form **two** nesting chains rather than one ladder — `shuffle → flat → uniform` and `table → flat → uniform` — and `shuffle` and `table` are the incomparable pair. `dq_3b_shuf` scores 79.12%, `dq_3b_flat` **80.30%** and `dq_3b_unif` **70.42%**, giving four rungs that partition the margin exactly in raw counts (92 − 141 + 1186 + 1159 = 2296): within-role placement **+0.77** [+0.27, +1.26], the score's magnitude over a table permuted the same way **−1.18** [−1.62, −0.73], the measured `dL` table permuted but present **+9.88** [+9.16, +10.60], and floors-plus-knapsack with no signal at all **+9.66** [+8.74, +10.58]. **The signal is +9.47 of the +19.13, 49.5%; the signal-free allocator is 50.5% — but that bottom rung is measured against the symmetric GPTQ, and almost none of the signal's half is the score.** Flattening every score *gains* 1.18 points over permuting it, and the real arm does not separate from the flat one at all (**−0.41** [−0.89, +0.07], *p* = 0.101, this section's only null result), so the plasticity-times-saliency ranking is not distinguishable from a constant on this model while the measured `dL` pricing carries +9.88 on its own. **The `table` chain is the one to read it off, because each of its two rungs moves one channel and nothing else:** the score is **−0.48** [−0.95, −0.02] (*p* = 0.0465, the largest in the family and fragile to it) and the measured table is **+9.96** [+9.24, +10.68], summing to the same 1 137 items. `dq_3b_tabl` scores **80.38%**, the best 3-bit arm in the panel, above the shipped `dq_3b`'s 79.89%. And the score's whole effect is on the expert banks: **16 of 133 widths separate the two arms and every one is a `feed_forward.experts.gate_up_proj`** — the 22 modules whose price *is* the rank-product proxy, with no measured `dL` behind it. Meanwhile the table's within-role *placement* is worth nothing (`tabl` − `flat` = +0.07, *p* = 0.71, four widths apart); its role-level magnitude is the whole +9.96. The byte edge over the two permuted controls is **zero**, not the 202 KB first reported here: `dq_3b`'s map was priced before commit `709b0c1` gave a price to the 61 rank-1 tensors the graph refuses (101,120 parameters, 202,240 B), and re-deriving it under the current checkout leaves all 133 widths, the histogram and all 15 floor breaches unchanged while raising the price to **3,331,728,896 B** — the shuffle and flat arms' figure to the byte. Only the uniform arm's edge survives, at exactly 1 MiB, so the rungs are read at matched bytes rather than at a discount. **The mechanism is not what the map's headline rows suggested.** All 22 routers sit at 8 bits — but `MOE_ROUTER` is a *structural* floor, so the allocator was never free to breach it, and all 22 expert down-projections sit at their own floor of 2; both are identical under every null. What the signal actually decides is **`attn.k` and `attn.v` held at 8 bits against a floor of 4**, which the uniform arm drops to 3–4 — and which the flat arm *keeps* at 8, so that decision is bought by the measured table's role-level magnitude and not by the ranking. The shuffle moved 20 of 133 widths, left the width histogram identical, and reproduced the floor-breach shape exactly (15 modules, 10 at 3b, 4 at 2b, one embedding) — good control, small measurement, and one of **four seeded draws**: byte-identical maps at 3,331,728,896 B every time, deltas of +0.77 (@0), +0.77 (@1), **+1.20** (@2) and **+0.40** (@3, Holm *p* = 0.11, **not separated**), mean +0.78 with sample SD 0.33 — a between-draw spread the size of the within-draw paired SE, and seeds 0 and 1 score an identical 9,495 on *different items* (926 vs 1,016 discordant), so an equal accuracy is not an equal map. The nesting holds the headline steady: the signal rungs sum to `dq_3b` − `dq_3b_unif` = **+9.47** for every seed, so the draw moves only the boundary between them, and with it the one-rung figure the Ministral comparison is read against (2.1 % to 6.3 % of the +19.13, not 4.0 %; 4.1 % to 12.4 % of the +9.64 the control leaves). Uniform moved 65 and grew the breached mass from 41.9% to 50.2% of parameters. Not established: the score rung separates at *p* = 0.0465 on one model, one task and one budget, which supports "not worth points" and not "reliably harmful"; four draws bound the permutation spread but not the spread's own error; and GPTQ/AWQ handed DynQuant's own bit map is still unbuilt. And 49.5% is not a third point beside the 12% on Qwen3.5-2B and 56% on Ministral-8B, but for the opposite reason to the one first written here: re-checked in the source, the two earlier campaigns used **two different controls**. Qwen's is `dict.fromkeys(scores, 0.5)` over an allocator with no sensitivity table — this ladder's `uniform`, so 12% is **both rungs**; Ministral's is the within-role permutation — this ladder's `shuffle`, so 56% is **the first alone**. Two series of two, then, read against the recovered baseline: 12% against **98.3%** here over allocator terms of +22.62 and **+0.17**, and 56% against **8.0%** here over +1.91 and +8.87 — both monotone in the direction those reports predicted, and neither convertible to the other without running the arm that campaign did not. The denominators also differ: both earlier shares divide by `dq` − `rtn`, uniform-width rounding, where this panel has no RTN arm and bottoms at an *asymmetric* GPTQ — a stronger floor than either, which makes the 98.3% as inflated as a share can get. **The control has now run.** `gptq_3b_asym_noao` changes one flag at the same anchor, `symmetric=False` with act-order off: GPTQ recovers 60.76% → **70.25%**, worth **+9.49** [+8.65, +10.33] on 1944/805 flips, so DynQuant's margin is **+9.64** [+8.88, +10.41] at Holm 6.8e-133 and the ladder reconciles onto it exactly (−0.48 + 9.96 + 0.17 = +9.64, and −58 + 1 195 + 20 = +1 157 items). The signal-free allocator, which this row said stands ten points above GPTQ, stands **+0.17** [−0.66, +0.99] above the recovered one at *p* = **0.706** — the panel's least separated comparison, with fidelity agreeing at +0.42 (*p* = 0.33). That +0.17 is two opposite results cancelling: **−7.87** on gretel against **+2.92** on wikisql at Q = 123.66, the largest heterogeneity in the panel, on a mixture that is 74.5% wikisql. The Holm family grew from six to nine and no verdict flipped. Act-order stays unmeasured here and must not be assumed to recover further — on the Mistral panel it took the same asymmetric grid from 76.08% down to 3.99%. | [`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §13.4, §13.5 |
| 23 | Does phase 4's 3-bit result hold on a dense model of another family? | **The compression does; the margin does not, and the control says the 68-point gap was the grid.** At matched bytes on Mistral-7B-Instruct-v0.3 DynQuant scores 75.22% against AWQ's 74.16% — **+1.06, Holm *p* = 0.744, not separated** — and at the 4-bit anchor every comparison returns 1.000. GPTQ collapsed to 6.68%; re-run at the same anchor with **an asymmetric grid and nothing else changed** it scores **76.08%**, so **+69.4 points is the zero point alone** and DynQuant does not separate from it (**−0.86, *p* = 0.186**) — while activation ordering on top of that same grid collapses the arm a second time, to **3.99%**, below the symmetric default, so one of the three GPTQ configurations at this anchor produces a working model at all. On agreement with bf16 the tie breaks against DynQuant: 90.30% against 93.36%, **−3.06 at *p* = 8.1e−07** | [`phase4-mistral7b-text2sql.md`](phase4-mistral7b-text2sql.md) §8 |
| 24 | Do the CUDA sources compile, and is the grouped MoE kernel correct on real silicon? | **Yes on both, and the one failure in 654 was the reference rather than a kernel.** Built for `sm_89` on an L4 against system nvcc 12.4 and torch 2.11.0+cu126 — the first time `grouped_gemv.cu` has been compiled anywhere — and its parity suite is **23 of 23** at first run. The rest of the surface came back 653 passed / 1 failed, on **CPU**, at the widest geometry: both `gemv_cpu` and `moe_grouped_gemv_cpu` materialised the dequantized weight through the `dequant` op, which rounds its store to the *scales'* dtype, so an fp16 rounding landed on every weight element in front of a sum over `in_features` of them. That grows like `sqrt(K)` against a flat tolerance and the kernel does not have it, because `gemv_kernel` never materialises the weight and applies scale and offset in fp32. Priced as the fraction of the parity tolerance each path actually spends, over every geometry × width × row count: CPU **1.041 → 0.000**, CUDA **0.043 → 0.043**, the CUDA column byte-identical on both sides, which is what says the defect was never in the kernel. The CPU column also rises monotonically with `in_features` — 0.032 at K=4 to 1.041 at K=3072 — while CUDA is flat at 0.02–0.04, so the margin was a function of geometry on one device and not the other. Two things this cost: the first probe of it counted `|d| > atol` and reported 20 of 20 CPU seeds failing where pytest reported one in 654, because `assert_close` allows `atol + rtol*|expected|` — that figure is discarded; and the fix makes the CPU parity case an *unpacker-agreement* test with the matmul held in common, so the kernel-against-reference question it is named for is answered on CUDA alone. The four gates never saw any of it: the module is `importorskip`‑ed whole on the Windows box, and the local gate is 2238 passed / 14 skipped both before and after. Pinned by a test that asserts the spend, not the value — the reverted route measures 1.04 against a bar of 0.25 — and the parity docstring's claim that the margin was geometry-independent is corrected in place, since it is the sentence a reader would have used to call this flakiness. Both of that campaign's closing caveats were then retired on the same box. **Timing**: at decode the grouped kernel is **4.0×–15.9×** the per-expert loop as it ships, which meets P8's ≥3× gate, and 3.1×–5.7× against that same loop over an already-dense bank — so the win is one launch over segments rather than skipped dequantization — while it crosses over as `rows` grows and loses **8.3×** at 2048 rows on the widest bank, which is the prefill split stated in milliseconds instead of in principle. The sweep also found a defect in its own denominator: `DynQuantExpertBank.__getitem__` reached the pure-torch *reference* dequantizer even with the kernels loaded, a median **4.01×** the loop never had to pay, so the first draft's 29×–209× was mostly a defect in the baseline; the fix is pinned by a test that asserts the *call* rather than the output, because on CPU both paths are numerically identical and a value test would stay green through a revert. **Sanitizers**: all four tools, 0 errors and 0 hazards, over a 108-launch grouped workload and over the parity suite — which needed `--target-processes all`, and the reason is now a measurement rather than an assertion: an out-of-bounds gather one subprocess away reports `3 errors` under `all` and prints **no `ERROR SUMMARY` at all** under the default, so the number of grouped launches that had ever run under the tool was one, not twenty-three. `racecheck` cost 2348 s against `memcheck`'s 104 s, and an earlier attempt at it was killed at fifteen minutes on the assumption it had hung — it had not, it needed thirty-nine. Then a whole model ran through it, which is the third caveat retired and two findings the benchmark could not have produced. **`LFM2.5-8B-A1B` decodes through `grouped_gemv.cu` coherently at 3 and at 4 bits**, at **1.95x** the per-expert loop at 4 bits and **2.66x** at 3 — **3 bits is 3.0% faster than 4** on the grouped path, 31.58 against 32.52 tok/s, while the loop loses **24%** going from 4 bits to 3, so the advantage grows exactly where this project's margins are. An earlier draft of this row read *width-independent, 31.64 against 31.55*; that was a `torch.bincount` host read flattening both widths by a fixed per-step cost, and the graph-capture clause at the end of this row is where it was found. It is **0.95x** bf16's rate at **3.76x** less resident memory, and the loop at that same memory is 0.49x, so what the kernel removes is not a choice between fast and small but the halving that packing otherwise costs. The model-level 1.95x is not the sweep's 8.55x at this geometry, and the gap is **Amdahl, not a shortfall** — solving for it puts the expert banks at roughly half a decode step, stated as the inference it is rather than quoted as the model's number. The memory figure was nearly published 67% wrong: peak over load-and-pack is **7167.5 MiB** against a resident **4295.7**, because the clipping search keeps a dense copy on the GPU, and a server loading an exported checkpoint pays only the second — recorded properly it lands **6.4 MiB** from `packed_bytes` on a 4.3 GB model, and 6.6 at 3 bits, which is P6's *peak VRAM ≈ manifest size* met against the allocator rather than predicted from the bit map. Running it also found what reading it had not: **`dynquant eval --map-apply pack` cannot reach the packed runtime on this family at all**, because `MOE_ROUTER` carries an 8-bit floor rather than an exclusion, so the router is in every map however the map was made and `pack_model` refuses it by class — the harness filters it caller-side and reports the 22 routers and 1,441,792 parameters that stay dense rather than calling the share small. Two harness errors are kept because they were the informative part: a missing chat template made the bf16 arm return fluent contentless loops that timed identically, so a coherence claim read off it would have described the harness; and a comment asserting the routers were *0.05% of this model* was a number nothing had measured, and is 0.017%. **Then P8's last clause closed, and closing it falsified a claim two committed reports had published.** The first capture attempt refused — not on the loop, on the **fused** path — because `torch.bincount` reads `input.max()` on the host and `minlength` does not spare it (bisected one primitive at a time: `sort`, `cumsum` and `zeros(E+1).scatter_add_` all capture; `bincount` refuses at `minlength=E` and at `2E`). Rebuilt as a `scatter_add_`, one MoE block captures and replays at **3.25x** eager at one token and **3.65x** at 3 bits, 1.15x at 8 tokens and 1.00x at 64 — and the *removed* milliseconds fall with it, 0.371 to 0.179 to 0.016, so what replay takes out is the launch cost **not already hidden** behind GPU work, which is the shape a decode-only claim should have. The per-expert loop refuses capture at every width, its `.tolist()` being the trip count rather than a fence, so the two paths are on opposite sides of a line no tuning crosses. The fence was also live in every packed step this row timed: re-run, `bf16` and the built-in `eager` control move 0.3% and 0.1% while packed 3-bit moves **+3.1%**, which is what corrects the width claim above — two post-fix readings 1.6% apart against one pre-fix reading, suggestive rather than settled. Still not claimed: no vectorized variant, one card, **one model family** rather than the Mixtral and Qwen3-MoE the gate names, and **the 3.25x is one MoE block, not a captured decode step** — 22 x 0.371 ms = 8.2 ms against a measured 31.67 ms step is an upper bound, not a prediction — **which row 25 then went and measured, at 76–82% rather than 26%** | [`kernel-first-compile.md`](kernel-first-compile.md) |
| 25 | Does the whole packed model capture as a CUDA graph, and what does replay buy on a real decode step? | **Yes, and through `torch.compile` rather than through capture code: 4.94x at 4 bits and 5.45x at 3, with zero graph breaks.** Section 24 left the model-level number as an upper bound — 22 MoE layers x 0.371 ms against a 31.67 ms step, so at most 26%, *an upper bound, not a prediction*. Measured on the packed LFM2.5-8B-A1B it is **76–82%** of a decode step, because a block has a handful of launches around real arithmetic and a model has 24 layers of them around the same arithmetic. A cacheless whole-model forward replays at 3.98x / 4.34x **bit-identically**; a real decode step under the default `DynamicCache` replays at 4.03x and returns **the wrong token** — `max_abs_delta` 15.33, argmax flips, nothing raised — because a graph records addresses and `torch.cat` growth abandons them. `torch.compile(mode="reduce-overhead")` refuses that container outright, inductor's `cudagraph_trees` guarding the case the hand capture walked into silently, and **the difference between the two is entirely whether someone wrote a correctness check**. The static cache — the container a capture wants — then died on a device-side `index_copy_` assert, and **four explanations of that were written down and falsified one at a time**: the 18-convolution / 6-attention layer mix (all 24 layers *are* in the cache, enumerated); this file's decode position (an in-bounds position asserts identically); the graph (one **eager** step with no capture in the process asserts too); and the packed runtime (dense bf16 gives a byte-identical traceback). The cause is two lines of `transformers`: `generate` sizes a static cache at `max_length - 1`, and `StaticLayer.update` **ignores the `cache_position` its caller passes**, writing at a device-resident cursor it advances itself — so a prefill of N tokens leaves the cache both N long and exactly N full and the next step runs off the end at *any* position. There was never a position that worked, and a `DynamicCache` has no capacity to run off, which is why it hid this for three rounds and returned a wrong answer instead. With `max_cache_len` raised by 192 slots every arm captures: hand-rolled 4.207x / 4.556x, compiled **4.936x / 5.452x** with **0 graph breaks** across 111 packed modules, the grouped MoE kernel, and both layer families in one graph — the first model-scale evidence that `custom_op` plus `register_fake` holds, and P8's *graph replay removes measurable launch overhead* closed. **The supported path beats the hand-rolled reference at both widths**, Inductor fusing inside the graph as well as capturing it, so the arm written to be the yardstick is the one that loses. No static arm agreed with eager to zero (0.375–2.906), so they were re-run with a control that takes **a second eager forward** and compares the two eager runs to each other — a decode step mutates the cache it reads, so consecutive eager forwards need not agree either. On three of four arms **the control is larger than the quantity it controls**; on the fourth it is 2.28 against 2.906; all four agree on the argmax in both directions, against the `DynamicCache` arm's 15.33 with a flipped argmax. Two internal checks say the timing is structural: **removed** time is width-invariant across all four static arms (24.76–26.44 ms, no ordering by width) while **remaining** time tracks width (3-bit faster in both arms), which is what removing launches rather than work looks like. One cost the speedup hides: `cache_writes` is **116** for a 50-iteration run because every call advances the static cursor, replays included — a captured step is replayable only as often as the cache has spare slots. Still not claimed: not tokens per second, not long context (the removable fraction falls as the cursor grows and 448 slots is the favourable end), not bit-exactness (which fails for eager against itself), and one card, one model family, one prompt | [`kernel-first-compile.md`](kernel-first-compile.md) §14 |
| 26 | Does the grouped kernel's >=3x hold on a second MoE family, and where does the margin come from? | **Yes — 3.180x on OLMoE-1B-7B-0125-Instruct at 3 bits, wider than LFM2.5's 2.66x, and the widening is the launch-bound story arriving from the other side.** Everything in rows 23-25 came off one checkpoint. P8's gate names Mixtral-8x7B / Qwen3-MoE; neither fits an L4 at bf16, so the clause was answered with a family that is genuinely a different test rather than one that shares a name: 64 experts against 32, top-8 against top-4, expert intermediate 1024 against 1792, 16 full-attention layers against 6 attention plus 18 short-convolution, a plain `nn.Linear` router against a custom router class, untied embeddings against tied. What it shares is the batched `[E, out, in]` bank the kernel consumes. The packer needed nothing added: **98 modules packed, 16 routers left dense** (2,097,152 params, 0.030% of the model), 0 tied, 0 skipped, `accounted_bits` **3.2515**, and the generic structural classifier reached `mlp.gate` through `out_features == num_experts` with an `experts` sibling — the test P3 was written around, exercised end to end for the first time on a family it was not developed against. Decode: bf16 **41.48**, per-expert loop **11.59**, grouped **36.86** tok/s — **3.180x the loop** (3.167x in an earlier process, so it reproduces), 0.889x bf16. Resident **2685.4 MiB** against a manifest `packed_bytes` of 2,679.8 MiB — **5.6 MiB apart, 0.21%**, closing P6's *peak VRAM ~ manifest size* on a second family; the bf16 ratio is **4.914x**, identical to LFM2.5's at four significant figures because both land at 3.2515 accounted bits. Coherent at 3 bits on both prompts in all three arms, and the two packed arms are **byte-identical on the first prompt** and divergent on the second, which is what substituting one kernel at the dispatch should give. The interesting column is why the margin is *wider* here. OLMoE is the smaller model and its bf16 arm is faster (41.48 against 33.25), yet **its loop is slower in absolute terms** — 11.59 against 12.23 while reading 17% fewer expert parameters per token. A loop doing less work in more time is launch-bound, and the geometry says by how much: **384 expert matmuls per token against 264**, each doing **43% less arithmetic**, so roughly 1.75x the launch overhead per unit of work — while the grouped kernel issues one launch per layer regardless of expert count. Row 25 measured that overhead directly at 76-82% of a decode step; this row watches the gap widen exactly where the launches get smaller, which is what a launch-bound explanation predicts and a bandwidth-bound one does not. The column that moves the other way moves honestly: grouped recovers 0.889x of bf16 here against 0.978x on LFM2.5, because a 2048x1024 expert matrix gives the quantized GEMV less arithmetic to hide dequantization behind. One harness bug is recorded because a comparison would have hidden it: OLMoE ships **`use_cache: false`** in its config, the explicit `GenerationConfig` had left that one field unset, and 128 tokens without a KV cache is quadratic — all three arms would have paid it equally, so the **ratio** would have survived and the **rate** would not. The first instrumentation for it read `model.config.use_cache` and **measured nothing**, reporting `False` next to three arms that demonstrably used a cache, because `generate` is governed by the per-call `GenerationConfig`; it was replaced by a four-token probe that asks `generate` to hand the cache back, recorded as `decoded_cache_len` and reading **30** on every arm. Not claimed: two families are not a trend, the cross-family arithmetic is a sanity check rather than a controlled comparison (only the within-model grouped-against-loop ratios are controlled), coherent generation on two prompts is not an evaluation, and `--map-apply pack` still cannot reach this path because routers carry an 8-bit floor rather than an exclusion | [`kernel-first-compile.md`](kernel-first-compile.md) §15 |
| 27 | Is the packed path actually faster than bf16, or only faster than its own eager baseline? | **Both, and which one you get is a single flag.** Rows 24-26 timed the grouped kernel against the per-expert loop and against bf16 *eager*; row 25 measured graph replay on one model at one width. Neither answered whether that speedup belongs to the packed path or is what `torch.compile` gives anything on this card. Fourteen arms at commit `d92ee05` — LFM2.5-8B-A1B and OLMoE-1B-7B × {bf16, uniform 3-bit} × the compile ladder, `--reps 3 --cache-impl static` throughout so the cache is fixed and only the compiler moves. **The compiled bf16 control is the point of the run**: compiling buys bf16 **15.6%** and **14.3%**, and buys the packed path **4.96×** and **4.67×** — 31.57 — 156.57 and 35.41 — 165.20 tok/s. So row 26's number was never *`torch.compile` is fast on an L4*; it is the packed path carrying a per-step launch and Python surface, once per module, that bf16 does not have. The consequence is a **sign change**: uncompiled, DynQuant decodes *slower* than bf16 (**0.967×** and **0.873×**); compiled, it decodes **4.145×** and **3.563×** the compiled bf16 arm — the whole distance between *packing costs 3-13% of decode rate* and *packing is four times faster* is one flag, and every ratio against bf16 earlier in this campaign (0.95× at row 24, 0.889× at row 26) is the eager end of that pair. Running `--compile-mode default` against `reduce-overhead` splits the win into what Inductor fused and what the cudagraph captured: on the packed path **3.116× / 1.592×** and **3.430× / 1.360×**, on bf16 1.099/1.052 and 1.086/1.053 — so **fusion is the larger half**, which a capture-only measurement could not see, and the two columns are *with graphs* and *without* rather than *launches* and *work*, because a fused dequant-into-GEMV removes launches too. **Memory does not move at all**: resident is identical across all three compile settings to the tenth of a MiB, **3 286.7** and **2 685.4** MiB against manifests of 3 280.1 and 2 679.8 — 0.20% and 0.21% apart, P6's *peak VRAM ≈ manifest size* on two families at once — and `peak_mib_total` equals `peak_mib_loaded` on every packed arm while exceeding it on every bf16 arm, which is what a decode that never materialises a dense weight looks like. `dynamo_unique_graphs` is read back from the counter rather than inferred: **0** on every arm that asked for eager, **2** on every arm that did not. `decoded_cache_len` reads **26** and **30** on all twelve arms — prompts of 23 and 27 tokens plus the three writes a four-token generation makes — where before `d92ee05` the same field read 54 and 58, `prompt + 31`, the *warmup's* fill: `generate` keeps one static cache on the model, so a reference held past the probe reported the timed run. Constant across compile modes is the evidence it now measures the probe; the same fix corrects a sentence in §15 that read the 30 as *26 prompt tokens plus 4 generated*. **The two `manual` arms exit 1 on both families**, same line, same error: `accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`. Compiling `forward` underneath `generate`'s loop leaves the loop reading tensors the next replay overwrote — the hazard row 25's hand capture met while **returning a wrong token with nothing raised**. Here it raises, so the supported path is the one that neither lies nor raises and is also the fastest arm. Not claimed: two families, one card, one prompt pair, three reps, nothing bounding long context; **these are not the panel's checkpoints** (uniform 3-bit, routers dense, `accounted_bits` 3.251 and 3.2515, not the allocator's 3.1488) and no accuracy was scored, so nothing here revises a panel number; and `warmup_s` is **22.4-24.5 s for bf16 against 7.98-9.03 s packed** — the model with 111 substituted modules and a custom op compiles three times faster, in both families and both modes, with no account of it here | [`kernel-first-compile.md`](kernel-first-compile.md) §16 |
| 28 | Do the arms these panels call byte-matched hold the same number of bytes? | **No — every symmetric arm was charged for a zero point it never stores.** `meta_bits = 16 + bits` existed in **two** independent copies, the second with a written comment defending it: charging only the asymmetric arm would make the baselines differ by a *convention* rather than by their weights. The argument inverts — arms differ in a size column because they **store** different things, and charging both the maximum is what imposes a convention. `compressed-tensors` writes `weight_zero_point` **only when the grid is asymmetric**, measured rather than read: a symmetric checkpoint holds **0** such tensors, an asymmetric one **186**, each `I32 [2,16]` = 1 024 bits = exactly `groups × bits`. GPTQ and RTN are symmetric by default in this repository and always have been, so **every GPTQ and RTN arm this project has published** was over-charged **~0.7%** of its width, in the direction that flatters DynQuant: `gptq_4b` **4.1565 → 4.1253** b, `gptq_3b` **3.1488 → 3.1253**, Mistral's **4.3760 → 4.3453** and **3.3869 → 3.3639**. Exactly one arm corrects the other way — `gptq_3b_asym` was never charged for its `weight_g_idx`, **+0.163%**. The serious part is `anchor_bytes`, which computes **one** budget per width: every DynQuant arm in phase 4 was sized on the asymmetric figure and then scored against a symmetric arm, carrying **+0.713%**, **+0.708%**, **+0.693%** and **+0.654%** more bytes — all **6.5–7.1×** the panels' own **0.1%** tolerance, all one way. So **DynQuant-against-GPTQ was not byte-matched on any phase-4 panel**; DynQuant-against-AWQ was, and is untouched, which is why every panel still carries one honest external baseline. Nothing was re-quantized and no accuracy figure moves — this is a denominator. Found by a smoke test refusing to print rather than by review, and its assertion's *stated* reason for firing was the wrong hypothesis while the thing it actually tested was right: **a control that varies an axis needs every column able to see that axis**; a blind column does not produce an obviously broken table, it produces one whose numbers are individually plausible and whose comparison measures nothing. Ninth duplicated-registry case in this campaign and the second where the duplicated thing is a *dependency's* arithmetic. Not claimed: the panels are **not** re-run, so those GPTQ comparisons stand with the gap stated rather than closed; and `stage8_bnb.py` holds a third copy of `accounted_bytes` for NF4, which does not take this term and has not been audited against bitsandbytes' own storage | [`byte-accounting-zero-point.md`](byte-accounting-zero-point.md) |

The method itself — signals, sensitivity estimator, allocator, encoder, format, packed
runtime, kernels — is documented end to end in the
[**whitepaper**](https://github.com/kambojvikram/dynquant/releases/latest/download/dynquant-whitepaper.pdf),
which also carries the kernel and VRAM measurements described in §6 below.

Every PDF linked from this page is a **release asset**, not a file in the tree: the xelatex
source sits next to this index and is what gets reviewed, and the built PDF is attached to the
release. The repository-wide `*.pdf` ignore exists to keep a confidential document out of git
history and is not worth a hole for a build artifact. So each link points at
`releases/latest/download/…`, which means a release that ships without re-attaching a rebuilt
PDF gives a reader a 404 rather than a stale document — the loud failure, on purpose.

---

## 1. The allocator campaign — Qwen3.5-2B-Base, GSM8K and CaseHOLD

[`experiments/four_point/RESULTS.md`](../../experiments/four_point/RESULTS.md) · 1 302 lines

The experiment the package was built to make possible: collect a signal map during a real
fine-tune, let the scorer and allocator choose per-module bit widths from it, quantize, and
measure the same benchmark at every stage.

It resolved in three stages, and the first one was a negative result against the method as
published.

**Stage one.** At a ~4.25-bit budget, where the score changes only 14–16 of 187 module
widths, score-driven allocation is indistinguishable from a same-size uniform control
(GSM8K +1.36, p = 0.20; CaseHOLD −0.13, p = 0.56). At ~3.25 bits, where it changes 70–79 of
them, allocation **loses**: −7.88 on GSM8K (p = 1.1e−10) and −2.03 on CaseHOLD (p = 5.8e−05).
The more the rank-product score drove the widths, the more it cost.

**Stage two — the diagnosis and the fix.** The published score
`Rank(plasticity) × Rank(saliency)` multiplies a factor that correlates with measured damage
by one that correlates **against** it (Spearman ρ = −0.21 over 187 modules), and then compares
modules by ordinal rank when the allocator's decision is cardinal. Replacing it with a
measured per-module sensitivity, `Σ E[δ_r²] E[x_c²] (W − Q_b(W))²`, collected by the same
tracker and priced per **move** rather than per module, turns the same budget on the same
checkpoint from −2.03 into **+10.29 points** (84.08 % vs 73.79 %, p = 1.3e−81). Quantization
damage at 3.25 bits falls from 16.73 points to **4.40**.

**Stage three — packed on real kernels.** Accuracy exactly unchanged (the same 4 468 of 5 314
problems), which retroactively licenses every simulated-quantization number in the document.
See §6.

**Task selection is part of the result.** GSM8K is the paper's own task and the fine-tune did
not move it (−0.99, p = 0.48) — not because the fine-tune failed but because the base model
was already at the supervised ceiling, so its table cannot read quantization damage against a
fine-tuning gain. CaseHOLD, chosen by measuring base-model headroom *first*
([`experiments/screen/`](../../experiments/screen/)), gained **+53.35 points** from the
identical recipe. Against that gain the quantized arms become legible: ~4.25 bits keeps 96 %
of it, ~3.25 bits keeps 69 %.

Four defects were found and fixed by this campaign, each written up where it occurred: every
role's least-important module was free to destroy; the paired test was unrecoverable after
the GPU-hours were spent; fallback calibration double-counted the width span; a tied embedding
table arrived on the GPU twice.

## 2. Transfer — Mistral-7B-Instruct-v0.3, Banking77

[`experiments/four_point/RESULTS-mistral7b-banking77.md`](../../experiments/four_point/RESULTS-mistral7b-banking77.md) · 388 lines

Four conditions changed at once against campaign 1: 1.88 B → 7.25 B, hybrid linear/full
attention with a **tied** embedding → dense GQA with an **untied** one, CaseHOLD → Banking77,
full fine-tune → **LoRA r=32**. Changing four things at once is normally how you learn
nothing; it is right here because the question is not "which of these mattered" but "does the
method transfer at all", and a method that only works on one 2 B model with a tied embedding
under a full fine-tune is not a method.

What replicated:

1. The allocator helps where damage exists and not otherwise — null at 4.25 bits on both
   models, separated at 3.25 bits on both.
2. The **reallocation direction** — bits out of MLP `up_proj`/`down_proj`, into attention
   `k_proj`/`v_proj`. Two runs sharing no model, task, architecture family or training regime
   agreeing on which roles are worth paying for is evidence the signal measures transformer
   structure rather than one checkpoint.
3. The current default scorer is on the right side of zero end to end, which had never been
   shown before this run.

What did not transfer, and could not: the magnitude. A 7 B model at 3 bits loses 2.53 points
to uniform quantization where the 2 B lost 14.69. The method's value scales with the damage —
which means it matters most exactly where compression is most aggressive, and that a bigger
model buys robustness that reduces how much any allocator can add.

## 3. Phase 1 — against GPTQ, AWQ, RTN and bitsandbytes NF4

[**PDF**](https://github.com/kambojvikram/dynquant/releases/latest/download/dynquant-phase1-external-comparison.pdf) · [LaTeX](dynquant-phase1-external-comparison.tex) ·
records: [`REPORT-quantization-comparison.md`](../../experiments/four_point/REPORT-quantization-comparison.md) (934 lines),
[`RESULTS-external-comparison.md`](../../experiments/four_point/RESULTS-external-comparison.md) (1 540 lines)

Two models, two tasks, one harness, one process. 45 arms across three panels. Each model is
fine-tuned on the task its own headroom screen selected, then quantized six ways from that one
checkpoint, and every arm scores the same held-out split in the same order. Baselines are
produced with `llm-compressor` (GPTQ, AWQ, RTN) and `bitsandbytes` (NF4).

| model | task | test rows | chance | base | fine-tuned |
|---|---|---:|---:|---:|---:|
| Mistral-7B-Instruct-v0.3 | Banking77, 77-way | 3 080 | 1.30 % | 33.7 % | 94.38 % |
| Qwen3.5-2B-Base | CaseHOLD, 5-way | 5 314 | 20.0 % | 32.33 % | 89.74 % |

Seven findings, in descending order of how much they should change your mind:

1. **At 2.42× compression DynQuant is lossless and GPTQ is not.** At byte-identical size
   (1.4512 vs 1.4514 GiB) DynQuant scores 89.71 % against GPTQ's 88.97 % — **+0.73, p = 0.017** —
   and is statistically indistinguishable from bf16 (−0.04, p = 0.905) where GPTQ is not
   (−0.77, p = 0.012). One tier, one task, one model.
2. **At 3.8× compression DynQuant ties everything.** 11 of 12 comparisons fail to separate;
   the twelfth favours DynQuant. On the tied-embedding model the tie is achieved at **1.73×
   fewer bytes** than the baselines' shipped configuration. The tie is the result; the size is
   the win.
3. **At 4.9× compression GPTQ wins, on all three panels** — −1.34 (p = 4.2e−04), −1.04
   (p = 0.011), −0.78 (p = 7.1e−03) — simultaneously 3.1 % smaller and 1.34 points better.
   This became the brief for phase 2.
4. **DynQuant beats AWQ at 3 bits by 2.5–3.3 points on both Qwen tasks**, at half the bytes.
5. **The margin over naive rounding is 88 % architectural prior and 12 % training signal.**
   At 3.25 bits DynQuant beats byte-matched RTN by 25.78 points; holding allocator, graph,
   floors and budget fixed and replacing only the signal scores with a constant splits that
   into **22.62 points of allocation and 3.16 of signal** (p = 1.9e−15). That constant is
   `dict.fromkeys(scores, 0.5)` over an allocator with no sensitivity table — which §22 identifies
   as the *far* end of its ladder, not the near one — so 3.16 is the whole signal term and 12 % is
   the figure comparable to §22's 49.5 %.
6. **Above ~4 bits the training signal is worth nothing measurable** — 12 of 187 widths moved,
   +0.19 points, p = 0.15. The operating rule: DynQuant earns its keep only when the budget is
   tight enough that the role floors cannot all be paid for.
7. **The accuracies survive real kernels and the bytes are exact; the speed is not there yet.**

The single most important methodological point in this report is §2, the accounting problem:
GPTQ and AWQ's conventional `ignore=["lm_head"]` makes a nominally 4-bit arm measure 7.36
bits on a tied-embedding model, so any comparison that does not match bytes is comparing
different sizes.

## 4. Phase 2 — reversing the 3-bit loss without copying GPTQ

[**PDF**](https://github.com/kambojvikram/dynquant/releases/latest/download/dynquant-phase2-beating-gptq-3bit.pdf) · [LaTeX](dynquant-phase2-beating-gptq-3bit.tex) ·
record: [`RESULTS-external-comparison.md` §Phase 2](../../experiments/four_point/RESULTS-external-comparison.md)

Measured 2026-07-30. One panel: Qwen3.5-2B-Base / CaseHOLD, 5 314 items, every comparison a
paired exact McNemar test on stored per-item hits.

| | arm | acc % | bytes | avg bits | vs `gptq_3b_head` | p |
|---|---|---:|---:|---:|---:|---:|
| before | `dq_3p25` (shipped) | 86.70 | 764 290 013 | 3.2494 | −1.34 | 0.0004 |
| tie | `p2_wc_agg` (per-module body) | 88.54 | 708 087 808 | 3.0104 | +0.51 | 0.16 (ns) |
| **after** | **`p2_rb_agg`** (per-row body) | **89.57** | **708 087 808** | 3.0104 | **+1.54** | **< 0.0001** |

+2.87 points end to end at 7.4 % fewer bytes than the shipped arm. The fp16 ceiling on this
panel is 89.74 %, so the final arm gives up **0.17 points against full precision at 3.01
stored bits**.

**The constraint.** The obvious fix for "rounds naively" is GPTQ's own — propagate each
column's quantization error into the not-yet-quantized columns via an inverse Hessian. It was
rejected for two reasons: it would make the method a reimplementation of GPTQ with a
preprocessing step, and it requires a calibration dataset, which is the exact dependency the
fine-tune-time hook exists to remove. Every lever below acts on a component DynQuant already
owns, and every input is a moment the `transformers` hook already accumulates online.

Four levers, each isolated at byte-identical budgets: a Gauss–Newton sensitivity model, a
row-partitioned tied embedding, an `E[x²]`-weighted clip objective, and per-row body
allocation.

**The central finding is not the headline.** Allocation granularity is a *multiplier on the
signal, not a gain of its own*: with the row widths shuffled, per-row allocation **loses 1.28
points** at identical bytes, and wins 2.31 against that control when given the signal's
ordering. Any granularity change must ship with its shuffled control.

## 5. Serving parity — vLLM and SGLang

[`serving-parity.md`](serving-parity.md)

The only experiment in this project that runs the model through something other than
`transformers`. A serving integration can load a checkpoint, start, answer, and be wrong: the
failure mode is not a crash but a model that silently dropped its packed tensors.

Both servers reproduce the direct GPU run to **100 % teacher-forced top-1 agreement** at all
108 scored positions, with a mean absolute logprob difference of 0.009147 (vLLM) and 0.006822
(SGLang) against fp16 controls of 0.006304 and 0.006896 — so the serving gap is 1.45× and
**0.99×** the gap each runtime already has on an unquantized model, and 269× / 361× smaller
than quantization's own effect on the same numbers (2.461067). Neither run passes a
`--quantization` flag or patches the server.

It found a real defect. SGLang reads `getattr(model_class, "packed_modules_mapping", {})`,
and on 0.5.16 that attribute is absent from **172 of 210** model files — including
`Qwen2ForCausalLM`, which fuses q/k/v inside `load_weights` anyway. The result was a server
that started, answered, and computed with uninitialised buffers. Fixed in `9e4c6ed` with a
bounded fallback plus a guard that makes the failure loud forever after.

One number is not yet explained and is flagged as open: on fp16 the two servers generate
identically for 12/12 prompts, on the quantized checkpoint only 10/12.

## The packed runtime, VRAM and the kernels

Not a separate document — measured inside campaigns 1–3 and written up in the
[whitepaper](https://github.com/kambojvikram/dynquant/releases/latest/download/dynquant-whitepaper.pdf) §"The packed runtime" and in
[`RESULTS.md` §"Running it packed"](../../experiments/four_point/RESULTS.md). Collected here
because it answers its own question: *do the accounted bytes exist on the GPU, and what do
they cost?*

**Accuracy through the kernels: exactly unchanged.** The same 4 468 of 5 314 problems, packed
and simulated. Not "within noise" — identical, because both paths quantize through the same
search over the same grid, so anything but an exact match is a kernel bug rather than a
rounding difference. On Mistral the two arms differ by 5 and 1 predictions out of 3 080.

**VRAM: the manifest was right to 0.03 %.** 0.7237 GiB resident against 3.5052 GiB at bf16
(**4.84×**); the allocator predicted 764 317 696 bytes and the live model holds 764 530 560.
This is the claim the paper's own Appendix F concedes it cannot make. At batch 32 the same
model is only 1.52× smaller as a process, because the KV cache dominates — quantization buys
headroom in proportion to how much of your memory is weights.

**Decode is slightly slower, and the reason is measured rather than guessed.** 0.90× at
batch 1, 0.60× at batch 32 (above `gemv_max_rows() == 8` the runtime falls back to
dequantize-then-`F.linear`). Profiling both arms through one script puts in-model matmul time
at **1.42× faster** packed, with the entire drop in GPU-busy time accounted for by that one
change — but a decode step on this model issues ~2 000 kernel launches, leaves the GPU idle
~70 % of the time, and spends 12 % of its wall clock in matmuls, so the 1.04 ms won is
outweighed by the host-side cost of dispatching 187 packed modules from Python. CUDA Graphs
(P8) are what remove that.

**The kernel in isolation.** Three rounds of optimization took it from 0.64–1.83× bf16 to
**1.09–2.56×**, and from 26–41 % of achievable HBM bandwidth to **10–98 %**. The ≥70 % gate is
met at 8 bits (98 %), three points short at 4 bits (67 %), and missed at 3 and 2 bits (52 %,
36 %). What remains is issue-bound rather than bandwidth-bound — measured against a computed
instruction-issue floor of ~141 µs against 237 µs achieved — and needs the tensor-core path
(P7) rather than more of the same. Round 2 is the most useful measurement in that section: a
−11.7 % instruction count bought only −2.5 % time, which is how the ceiling was located.

**Verification.** 417 kernel-parity tests across every geometry × width × M × dtype, and all
four `compute-sanitizer` tools — memcheck, initcheck, racecheck, synccheck — report 0 errors
and 0 hazards over that suite.

## 6. Phase 3 — S1, the headroom screen

[`phase3-s1-headroom-screen.md`](phase3-s1-headroom-screen.md)

The first pre-flight check of phase 3, applying a rule this project had already paid to learn:
*screen headroom before spending a fine-tune.* GSM8K's flat arms in campaign 1 cost a full
six-arm run to diagnose, and the diagnosis was that the base model already sat at the supervised
ceiling — so nothing downstream could read quantization damage against a fine-tuning gain. Phase
3 changes the models, the datasets and the benchmarks at once, so every benchmark has to be shown
to have room first. Eight arms, seven minutes of GPU.

**All eight have room.** Phi-4-mini-instruct scores 68.76 % IFEval, 83.17 % GSM8K, 77.44 %
HumanEval and 60.00 % MBPP; Ministral-8B-Instruct 54.53 %, 80.89 %, 79.27 % and 55.80 %. The
highest arm leaves 16.8 points before the ceiling and the lowest 45, so nothing is near enough to
100 % that damage would have nowhere to show — the single question S1 was run to answer. The two
models also rank differently on three of the four tasks, which is the argument for a
four-benchmark panel rather than one benchmark and three correlated with it. MBPP is the tightest
floor and the arm most likely to need paired hits to separate anything.

**Two of the four candidate models are not in the panel, and that is scope rather than an
omission.** `meta-llama/Llama-3.1-8B-Instruct` and `google/gemma-3-4b-it` are `gated=manual` on
the Hub — per-account licence acceptance through a web UI, not resolvable from the box — so on
2026-08-05 the panel was settled at two models rather than held for a token. What survives spans
fused projections against unfused dense GQA, two tokenizer backends and 2.1× of scale; what is
given up is the Llama family and the only alternating sliding-window stack. Every arm is scored
against its own model's bf16 ceiling, so adding a model later invalidates none of it.

**The screen's other product was two harness defects**, both returning a stable, plausible, wrong
number rather than an error. The harness decided whether to frame a prompt as a chat turn by
reading `tokenizer.chat_template` — an attribute of the *Jinja-backed implementation*, not of the
capability. `AutoTokenizer` hands back `MistralCommonBackend` for any Mistral checkpoint shipping
a `tekken.json`, and that class leaves the attribute `None` while `apply_chat_template` works, so
an instruct model was measured as a base model: **24.77 %** IFEval with 195 of 541 generations
empty, because an instruct checkpoint handed bare text continues it instead of answering it. The
second bug detected the frame and then discarded it on the way to the model, taking HumanEval to
**23.17 %** with 120 of 164 empty. Correctly framed, the same checkpoint scores 54.53 % and
79.27 %. **Getting the framing wrong is worth up to 56.1 points on HumanEval and 29.8 on IFEval**
— five to thirty times the +1.54 effect size phase 2 was built to measure. Both losing arms are
kept as controls, because a number that large is only harmless once it is priced.

## 7. Phase 3 — S2, locating the assistant turns

[`phase3-s2-loss-masking.md`](phase3-s2-loss-masking.md)

The second pre-flight check of phase 3, and the same rule as §"Task selection precedes
fine-tuning" applied one level down: verify the loss mask before spending 30 GPU-hours on it.
A wrong mask does not fail — it trains a slightly worse model and reports success, which is
fatal to a campaign whose purpose is to measure small differences *on top of* that model.

The obvious method — locate turn `i` by the length of the render of `messages[:i]` — assumes
each rendered prefix is a token prefix of the next. Nothing promises that. `mistral_common`,
behind Ministral-8B, refuses to render *any* assistant-final conversation at all (it is
validating a serving request), so the walk drops **3 000 of 3 000** rows; Phi-4-mini appends a
document terminator to a conversation it is not asked to continue, so its "prefix" isn't one.
The mode that works renders each assistant turn *open* with `continue_final_message=True` and
closes it with a terminator measured from three synthetic renders — `<|end|>` (200020) on Phi,
which is **not** its `eos_token_id`. Which mode a tokenizer needs is not an attribute anywhere,
so `--mask-mode auto` measures both on 32 real rows and takes the winner.

Result on 3 000 Tulu-3 rows each: **0.00 %** unmaskable on Phi, **0.07 %** on Ministral (2 rows
of empty assistant content), 70.8 % / 70.5 % of tokens supervised. Phi accepts both modes, and
on 500 rows the two agree on every id and every span start, differing only by that one document
terminator.

Three designs died on real tokenizers after passing a stub suite; the most dangerous **passed a
3 000-row dry run at 95 %**, because `mistral_common` merges adjacent same-role turns and the
merge separator and turn opener are each one token, so two errors cancelled. That is why the
fourth design was prototyped against the real tokenizers before being written into the module.

The same run's census flags a GSM8K-derived source at **5.2 % of the mixture**, against a
headline that includes GSM8K. Reporting it and keeping it is defensible for arm-versus-arm —
both sides train on the same mixture — but that argument does not cover *test* leakage, which
would make the score recall rather than reasoning and compress the range quantization damage
has to show up in. Settled by looking: a 13-gram index over all 1 319 test questions, scanned
against the run's own 50 000 rows, flags **2 items and 0 usable duplicates** — one shared word
problem skeleton with different quantities, one WildChat conversation quoting "Janet's ducks".
0.15 points of headroom either way, against a +1.54 effect size. GSM8K stays; the claim that
its post-SFT number measures *general* math gain does not.

---

## 8. Phase 3 — S3, what fusion costs the model that has it

Full report: [phase3-s3-fused-floors.md](phase3-s3-fused-floors.md). Measured before S3's arms
were built, because phase 2 ran only on models that spell every projection separately and
Phi-4-mini does not: `qkv_proj` and `gate_up_proj` are **55.1 % of its parameters**.

What holds: the row partitions land correctly at the checkpoint's real geometry — 24:8 GQA
and an unset `head_dim`, neither of which the hidden-64 fixture produces — and the allocator
descends from floors it cannot afford on a fused model, hitting 3.2492 against a 3.25 target
with 90 breaches named and 69 of 129 modules moving against a shuffled-score control. That is
bug 4's guard, which until now had only ever run on an unfused synthetic model.

What does not: **the two panel models are in different regimes at the same nominal target.**
Phi's floors cost **4.43 average bits** against Ministral's **3.82**, so a 3.25-bit arm asks
Phi to go 1.18 bits below its floors and Ministral 0.57 — and phase 2 found the signal only
earns its keep once floors stop being affordable. Of Phi's 4.43, **0.21 bits is fusion rather
than architecture**: a fused tensor takes the strictest of its partitions' floors, so 0.805 B
up-projection parameters are charged the SwiGLU gate's 4 bits instead of their own 3. The
allocator never reads `partitions`, though `QuantTensor` already carries them — the format is
ready and the allocator is not. Left open deliberately; closing it would break comparability
with phase 2.

---

## 9. Phase 3 — S2 arm 1, verifying the signal map before spending it

Full report: [phase3-s2-phi-signal-map.md](phase3-s2-phi-signal-map.md). Phi-4-mini × Tulu-3
finished in **12.68 h**, 1501 optimizer steps, final train loss **0.6747**. A run that completes
is not a run that produced something, so the map is checked before S3 reads it.

Structure is sound: 130 modules (2 + 32×4) under canonical names, the tie recorded,
`grad_norm_count` equal to the optimizer-step count on every module that has one (bug 10 stays
fixed), and `forward_calls` **uniformly 12 004** — the check that gradient checkpointing stayed
off, without which one part of the model's saliency would sit on a squared EMA decay and rank
against the rest on different terms. Saliency spans 428.9×.

One module scores on nothing: `model.embed_tokens`, for **three** independent reasons — no
`δxᵀ` to form under `outer_exact` (an embedding's gradient is a scatter-add), a guard that
discards its measured saliency along with the missing plasticity, and a role group of size one,
where a percentile rank is 0.5 by construction. It is 16% of the model and **pays 67.8% of the
3.25-bit floor shortfall by itself**. Forcing its score across the full range moves it 2 → 4
bits, so the neutrality is not structurally harmless; substituting its tied partner `lm_head`'s
row — the same tensor, measured — scores 0.9264 and lands on the same 3 bits.

**Corrected 2026-08-06.** That was originally read as *costs this checkpoint nothing*, off one
tensor's width. The budget is shared: at the same substitution the map moves **12 / 5 / 11 / 8**
other modules at 3.25 / 4.0 / 4.25 / 4.5, at byte totals equal to within 0.02%. The tied tensor
holds its width and the tail is rewritten around it. The verifier now reports
`other_modules_moved` beside `changes_width`.

---

## 10. Phase 3 — S3, the control that was ablating nothing

Full report: [phase3-s3-null-control.md](phase3-s3-null-control.md). Found by running the S3
driver's CPU path on real weights, before any arm was quantized.

S3's `shuf` arm permutes the signal within role and re-allocates at the same byte count; the
`dq` − `shuf` gap is what the campaign calls the signal's contribution. The driver permuted
the **stats** file and then passed `--moments` the real, unpermuted sidecar. Since the
Gauss-Newton estimator was added, `move_value` prices every width change from the measured
sensitivity table whenever the module has one and only falls back to the stats-derived score
when it does not — and Phi's sidecar covers **all 129** quantizable modules. The permuted
score was therefore never read. Confirmed on the pre-fix run rather than argued: `shuf3` and
`dq3` differ on **0 of 129 modules**, hash identically, and land on the same byte count and
the same 3.249487442337263 average bits — while `rank3`, which allocates from the score with
no moments at all, moves 65. The control would have reported *the signal does not matter*,
silently, with both arms exiting 0.

Fixed by deriving one permutation and applying it to both artifacts, grouped by
`(role, in-channels, out-channels)` — a no-op on a dense model, and there because a channel
vector of the wrong length does not raise; it makes the module drop out of the sensitivity
table, so the control would ablate coverage as well as correspondence. Nothing downstream had run, so no
reported number changes; §8's 69-of-129 figure is unaffected, its allocation never loading
moments at all. The transferable rule: **a control is defined by what the treatment reads,
not by what it is named after**, and an ablation arm whose map is identical to its treatment's
has not run.

---

## 11. Phase 3 — S2 arm 2, the tensor ranked against nobody

Full report: [phase3-s2-ministral-signal-map.md](phase3-s2-ministral-signal-map.md).
Ministral-8B × Tulu-3 finished in **11.23 h**, 1492 optimizer steps, final train loss
**0.6335**. Structure is sound on all four properties: 254 modules (2 + 36×7) under canonical
names, `grad_norm_count` = 1492 everywhere it exists, `forward_calls` uniformly 23 865,
saliency spanning 1221×.

§9 predicted what would be different: Ministral is untied, so `lm_head` is its own quantizable
tensor **in a role group of one**. It has a complete measurement — 1492 gradient observations,
**rank 1 of 254 on saliency** (1.78× the runner-up) and **rank 6 of 254 on plasticity**, so both
signals agree and this is not a tensor that merely has large activations for scale reasons — and
the shipped per-role ranker computes its percentile against a set containing only itself and
returns **0.5**. Ranked globally, the same measurements score **0.9783**.

At the headline 3.25-bit target that is a whole bit on 6.7% of an 8B model: **3 bits shipped
against 4 bits measured**, paid for by 24 projections dropping 4 b → 3 b at an *identical* byte
total and identical 3.2500 average bits. At 4.0 b the head's own width does not move and **23
other modules do**. With `model.embed_tokens` — same 536.9 M parameters, no gradient signal at
all because the `outer_exact` hook never registers on a frozen embedding fed integer ids —
**13.4% of the model is allocated on a number reflecting no comparison**.

Controlled, because a greedy knapsack near its ceiling could just be chaotic: forcing the same
0.9783 onto 24 ordinary projections moves a **median of 0** modules (max 8 at 3.25, max 2 at
4.0). Nor is it a size effect — `embed_tokens` is the same size and moves **0** at 4.0, because
it already sits at its floor and is not a knapsack candidate, while `lm_head` sits four bits
below its 8-bit floor and its ROI position decides the tail. **The neutral score bites hardest
where the tensor is large and its floor is breached**, which is the regime the headline targets
are defined by.

**Corrected 2026-08-07 — which allocator reads the score.** Those numbers come from
`verify_signal_map.py`, which allocates without a sensitivity table: the **rank-product** path,
which is the `rank` *baseline* arm. `allocate_bits` prices a module from measured sensitivity
wherever the channel moments cover it and from the score only where they do not. Ministral's
sidecar covers 253 of 254 modules. So at the 3.25-bit anchor, all four arms landing on exactly
3 257 925 632 bytes: `lm_head` gets **3 b under rank-product and 4 b under `dq`** — the headline
arm reaches the width the measurement implies on its own, and the singleton handicap costs the
baseline, not `dq`. The tensor that *does* carry a neutral score into the headline arm is
`model.embed_tokens`: no moments, untied, no partner to borrow from, and `dq` puts it at **2 b**
where rank-product left it at 3. Phi's tie is the contrast — it routes `lm_head`'s moments onto
the same tensor, so Phi's `dq3` gives it **4 b**. Untying removes the coverage, not just the
check. And `shuf` cannot ablate either tensor: within-role permutation is a fixed point on a
group of one, so both are assigned identically to `dq` by construction and neither is among the
39 of 254 modules the arms differ on.

**The 4.25-bit anchor corroborates it from the other side.** Both anchors are now complete and
byte-matched (4 260 364 288 B, +0 B widest drift). At 4.25 every role floor is affordable — all
three allocating arms hit **zero violations** — so `rank4` hands `lm_head` its full **8 b** floor
and `model.embed_tokens` its **4 b**, the same widths the sensitivity arms pick. The neutral 0.5
costs the baseline nothing there. The defect is only visible where the allocator is *forced to
choose*, which is §7's finding restated in the negative: the signal earns its keep once the role
floors stop being affordable, and so does the flaw in how it is ranked. Where the allocators part
at 4.25 is instead the **8-bit tail** — `rank4` widens 2 modules, `dq4` widens 37, paid for by
dropping 62 to 3 b against rank-product's 43. The control ablates at both anchors (39 of 254 at
3.25, 28 at 4.25) and is blind to both singletons at both. All of that is asserted against the
committed maps in `tests/test_s3_allocation_arms.py`.

Which map is better is an eval question and an S3 arm. The scorer was **not** changed —
that would move every model's allocation including phase 2's published comparisons, and it is
a decision, not a fix. The *verifier* was: it now enumerates singleton role groups, reports
`informed` apart from `scored`, and measures whole-map movement rather than one width, which
is what forced §9's correction above.

---

## 12. Phase 3 — S3, a resume guard that measured the wrong thing

Full report: [phase3-s3-reuse-guard.md](phase3-s3-reuse-guard.md). Found while clearing the
way for the S3 quantize-and-eval sweep, before any of it was launched.

Six maps are allocated before a single weight is written, three of them priced from the
channel moments at about 1 h 45 m of CPU each — seven hours before the GPU has anything to do.
`--reuse-maps` skips that, and its whole burden is telling a current map from a stale one,
because existence on disk says nothing about *when* a file was written. It got that wrong in
both directions at once.

**It stamped freshness against a file the same run regenerates.** The maps' mtimes were
compared to the stats file each map *names* — but for the three signal arms that is a derived
variant the driver writes at the top of every invocation. The comparison asked *is this map
older than a file I wrote thirty seconds ago*, answered yes, and rebuilt five of six. The fix
stamps against the run's sources — S2's stats and moments and the merged checkpoint — and
leaves the variants their real job, which is the **identity** witness separating `shuf3` from
`dq3`, two arms that name the same model, allocator and group size.

**And it called a metadata reordering a content change.** With the first defect fixed,
`arms.json`'s `rewritten: false` is the only record that a reused map was priced from the
numbers now on disk — and it was a constant `true`. Root-caused, not waved off:
`safetensors.torch.save_file` serialises `__metadata__` from a Rust hash map, and Rust seeds
each map instance separately, so **two writes of the same object in one process differ**.
Measured on the Ministral moments: both 11 519 008 B, payload byte-identical, headers equal as
parsed mappings, 506 tensors with **0 differing values**, 254 metadata entries equal as a
mapping — and the first differing byte at offset **43**, where the key order splits. So
`sha256(file)` is not a content identity for a safetensors artifact, and every hash-based
cache, dedup or provenance check built on one reports a change every time while distinguishing
nothing.

Verified end to end before the sweep launched: 7 of 7 maps reused, both variants
`rewritten: false`, widest byte drift **+0 B** at both anchors. Independently, the seven maps
on the box are sha256-identical to the seven committed — including `map.rank3.json`, which the
aborted dry run had rebuilt from scratch, which is a determinism proof nobody designed.
Nothing downstream had run, so no reported number changes. The transferable rule: **a resume
guard must compare against what it reads, not against what it writes** — and a comparison that
is too strict does not fail safe, it just stops distinguishing anything.

## 13. Phase 3 — S3, what the allocator and the signal are each worth

Full report: [phase3-s3-allocation.md](phase3-s3-allocation.md). Ministral-8B × Tulu-3, eight
quantized arms over two budgets and four benchmarks, plus a bf16 ceiling — **36 eval cells**,
each arm at an anchor byte-identical to its siblings (3 257 925 632 B at 3.25, 4 260 364 288 B
at 4.25, widest drift +0 B).

**At 3.25 bits the method works. At 4.25 nothing does.** `dq3` beats uniform by **+4.31** mean
points and has the best mean at both budgets, but at 4.25 all sixteen comparisons sit between
p = 0.22 and p = 1.00 — uniform is already within 6.15 points of bf16 and the floors stop
binding (182 breaches at 3.25 against 1 at 4.25). §11 predicted exactly this from the
allocation side: **the allocator earns its keep only where the floors stop being affordable.**

**But not for the reason the first draft gave.** It said the 4.25 maps converge; they do not.
`dq4` differs from uniform on **99 of 254 modules** — more than `dq3`'s 96 — and promotes 37 to
8 bits where uniform has none. The null is not a degenerate allocation scoring like uniform
because it *is* uniform; it is a heavy reallocation that buys nothing measurable. The exception
is `rank4`, 82 % identical to uniform with 45 modules moved, which really did stop allocating —
a different null with a different cause, hiding under the same p-values. The correction came
from reading the committed maps instead of inferring their shape from floor counts and eval
discordance.

**The signal's whole footprint is 39 modules of 254.** `dq3` and `shuf3` share allocator,
budget and byte total, so their disagreements are exactly what the measured signal moved: 15 %
of modules over 24 of 36 layers, concentrated in `o_proj` (17), `q_proj` (11) and `gate_proj`
(9), two-way at matched bytes — 14 promoted 2→3 paid for by 11 demoted 3→2. That is what
+2.41 mean points and the +4.78 on GSM8K are bought with.

**The signal is 56 % of the 3.25-bit margin — where on Qwen3.5-2B it was 12 %.** The
decomposition is `dq − rtn = +4.31 = allocator +1.91 + signal +2.41`. Same anchor, same module
granularity, opposite split. Both are real: on Qwen the uniform baseline was catastrophic and
almost any reallocation recovered most of a 25.78-point gap, so knapsack mechanics dominated;
here uniform is merely mediocre and *which* modules to protect is proportionally more of a
smaller answer. **The signal's share grows as the allocator's structural advantage shrinks**,
so the 12 % figure is scoped to its campaign, not retracted. Re-checked at §22, and the two shares are
*not* the same quantity: this one splits at a within-role shuffle, Qwen's splits at a
constant-score allocator with no sensitivity table. 56 % is one rung of §22's ladder and 12 % is
all of it, so the campaigns pair off as 56 %-against-4.0 % and 12 %-against-49.5 % — both monotone
in the stated direction, and neither able to be re-read on the other's definition without running
the arm it did not. The 4.25 row is deliberately left
undecomposed — dividing a +0.74 margin that is inside noise prints "the signal is 139 % of it",
and `s3_table.py` now refuses to compute the share unless something clears p < 0.05.

**Two of forty-eight paired tests survive Bonferroni**, and the important one is the control
comparison: `dq3` − `shuf3` on GSM8K, **+4.78, p = 4.8 × 10⁻⁵**. At identical bytes with the
same allocator, pricing from the measured signal beats pricing from a within-role permutation
of it — and the control is not a strawman, `shuf3` beats uniform by +1.91 on its own. The other
survivor is `rank3` − `rtn3` on HumanEval at +14.02. `dq3` − `rtn3` on HumanEval (+11.59,
p = 0.0034) does **not** clear correction and is reported as record, not result.

**Negatives, stated as such.** The published rank-product score is a coin flip — it wins
HumanEval by +14.02 and loses GSM8K and IFEval at 3.25, then does nothing at 4.25 — the fourth
campaign in which it fails to reliably beat doing nothing. MBPP separates no arm anywhere (all
four 3.25-bit arms within 1.0 point, discordant counts split evenly); it reads damage, not
allocation, and belongs in no headline that claims an allocator difference. And **no arm wins
all four tasks at 3.25**, so the mean is never quoted without the per-task table beside it.

## 14. Phase 4 — the text-to-SQL mixture, before any of it is scored

[`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) · screened 2026-08-08

Phase 4 quantizes `LiquidAI/LFM2.5-8B-A1B` six ways against a bf16 ceiling. This entry is the
benchmark those seven arms will be compared on, and it is here as its own campaign because
building it found four things, none of which would have shown up as a number that looked wrong.

**Execution accuracy has one failure mode with three shapes.** Two queries that both return
nothing compare equal, so a set of items whose gold returns nothing scores `SELECT 0` near the
ceiling. `COUNT(*)` over an empty schema returns `[(0,)]` and an unmatched `AVG` returns
`[(None,)]` — both pass a naive "did it return rows?" test. Admission requires the database to
hold rows, the gold to find some, and the answer not to be a single all-NULL/all-zero row.

**A third of WikiSQL was being discarded for the wrong reason.** Its condition values are the
annotator's typing, its cells are Wikipedia's, and SQLite `=` on `TEXT` is case-sensitive.
33 % of golds matched nothing and were refused as "gold finds no rows" — a correct refusal with
an incorrect cause. Declaring `COLLATE NOCASE` took it to 0.4 %.

**Gretel is a SQL corpus, not a query corpus.** 10.2 % of test and 11.3 % of train golds are
`UPDATE`/`INSERT`/`DELETE`/`CREATE`. The evaluation already excluded them, but as
`empty_result` — the wrong diagnosis. Training has no row filter, so they survived, and the
scorer reads an answer by cutting at `SELECT`: an 11 % `UPDATE` diet teaches a response format
scored `unparseable` on a zero-floor metric, identically across all seven arms. Closed before
the fine-tune; Gretel's training admission fell from ~78 % to 68.8 %, which is the leak.

**`text2sql` was fully implemented and unreachable.** `dynquant eval --task` carried a
hand-written copy of the registry, so argparse refused the task with a usage error naming the
other six. Choices now derive from the registry. Fixing it exposed a test asserting the
`style` capability against `executes_code` — passing on a coincidence until the first task that
takes a framing and runs nothing.

Measured admission: 2 796 items on the evaluation split, 5 326 on the training split, balanced
per source and round-robin interleaved so a truncated run still sees every corpus.

The same report carries three later screens. The base model reaches **57.75 %** on the mixture
once it is given room to finish reasoning, so there is headroom for damage to show — the first
attempt read 5.50 %, and 34 of the missing points were a truncated reasoning trace rather than a
model that cannot do the task. And the architecture is only partly visible: DynQuant's signal
collection reached **11.6 %** of its parameters and `llmcompressor`'s GPTQ and AWQ reach
**8.5 %**, because 91.5 % of the weights live in batched expert banks that are not `nn.Linear`
modules. Linearizing the banks takes both to 100 %, verified bit-exact against the original
forward.

## 15. Phase 4 — the benchmark's answers were already in the training set

[`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §11 · measured 2026-08-08

The S2 driver has carried a contamination check since phase 3 and it reported nothing on this
mixture. It could not have reported anything: its markers are `("gsm8k", "humaneval", "mbpp")`,
and no SQL corpus name contains one. **An empty result from a check that cannot fire reads
exactly like a mixture that passed.**

Measured against each training pool in full: Gretel is clean in both directions, WikiSQL's own
train split contributes 1 of 200, and `b-mc2/sql-create-context` — a community aggregate
assembled from WikiSQL and Spider, shipping a single `train` split — holds **189 of the 200
WikiSQL evaluation questions**. In the 50 000-row mixture this run would have sampled at seed 0,
38 of those 200 are present.

What that does and does not invalidate was written down before the number arrived, so the number
did not get to decide it. **The A/B stays valid**: all seven arms quantize the same fine-tuned
model, so contamination inflates them equally and the paired test still measures quantization
damage. What it would have cost is the *absolute* accuracy as a claim about text-to-SQL, and the
base→fine-tuned difference as a claim about learning.

**The scan that found it first reported clean.** Its sampled arm compared the rendered *user
turn* — the question wrapped in a schema and a directive — against the bare evaluation question,
so `0 / 200` printed directly beneath the pool scan's `189 / 200`. That is the same failure as
`_CONTAMINATING`, committed inside the tool written to expose it; it is recorded in that
function's docstring rather than quietly fixed.

The filter matches on the **question**, not provenance — the aggregate does not record which
upstream corpus a row came from, and a question match catches an item arriving by any route.
It matches against Gretel's and WikiSQL's **whole** test splits (21 729 rows, 21 681 distinct
keys) rather than the 400 sampled items, so changing `--limit` or the seed cannot undo it
silently. And it drops **before admission**, so a contaminated row never consumes a quota slot
and is never counted as kept. It removed 5 Gretel, 21 WikiSQL and 3 990 create-context rows with
every quota still met.

Both checks are now in the census, and it says what each is worth: the empty
`sources_overlapping_an_eval_task` is kept, the marker list is written beside it, and
`decontaminated` reports per source rather than as a boolean — because on this mixture the
expected number is four thousand, so zero has to be readable as suspicious rather than as clean.

## 16. Phase 4 — the matched-byte anchor has a direction, and it is worth 2.3%

[`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §12 · decided 2026-08-08

GPTQ and AWQ take a width and the bytes fall out of the format; DynQuant takes a size. So "at
matched bytes" has a direction, and the two formats do not cost the same: `compressed-tensors`
writes an fp16 scale plus a zero point at the weight's own width, `16 + bits` per group of 128,
while DynQuant writes an fp16 scale **and** an fp16 offset, a flat 32. At 4 bits that is 4.25
bits per parameter against **4.15625**; at 3 bits, 3.25 against **3.1484**.

Anchoring the panel on DynQuant's uniform arm would therefore have given it **2.3% more bytes at
4 bits and 3.2% more at 3, inside the arm whose accuracy is the claim** — and nothing would have
reported it, because each accounting is correct about the format it describes and both would have
printed a matched line.

DynQuant is pinned to the baselines' counts: **4 399 629 312 B** and **3 332 904 576 B**. The
overhead comes out of its own payload, which is the point rather than a concession — a format that
stores more metadata has fewer bits left for weights at the same footprint, and that is a real
cost. Charging both sides by one set of rules would be wrong in either direction: by DynQuant's,
the baselines are billed for an offset they never write; by the baselines', DynQuant writes
metadata it never paid for.

Two smaller decisions fall out of it. Each DynQuant arm's realised size is read back from the map
the allocator wrote rather than taken from the request, because `--target-size` is a *ceiling* and
the drift that actually happens is downward — so the tolerance check is on the absolute value,
where a signed one would wave through the only failure that occurs. And every arm is a subprocess
of one interpreter, refusing to start if llm-compressor is not importable from it: per-arm
environments would score the baselines and the DynQuant arms under two transformers versions,
which is a difference in the measuring instrument reported as a difference between methods.

Wiring the panel's own guard through the eval command's `_comparability` then exposed a hole in the
contract. Every field in `PAIRING_FIELDS` comes off the command line, so seven arms from one driver
cannot differ in any of them — but **`prompt_style` does not**. `--prompt-style auto` is answered by
the tokenizer, so a quantized checkpoint whose saved tokenizer lost its chat template is asked
bare-text questions while the ceiling is asked chat questions, with byte-identical commands on both.
That failure is already measured here: it put Ministral-8B-Instruct at 24.77% on IFEval against
Phi-4-mini's 68.76%. IFEval and the code tasks record their resolved style; `text2sql` did not.
`DETAIL_PAIRING_FIELDS` now reads it out of the record's `detail` block, where a task's own metrics
already go, and absence is exempted from the "this run always writes it" guard but still refuses
against a record that has one — "unknown" is not "the same".

The same pass caught the panel about to run at a budget nobody chose. `--max-new-tokens` was left
unset on the reasoning that an inherited default is inherited identically by all seven arms; there
are two defaults, the CLI's task spec says **320** and the in-process chat config says **384**, and
the panel routes through the CLI. All seven would have been consistent, pairable, and 704 tokens
under the 1024 the ceiling was meant to establish — and a truncated query does not near-miss, it
fails to parse, so a binding budget is a floor under accuracy that binds hardest on the most
damaged arm. The panel now states 1024 on every command, refuses a *ceiling* that was still
deliberating at the cap, and pairs after each arm rather than after all seven. A censored
quantized arm is left alone: that one is the finding, not a defect in the run.

Then the DynQuant arm was rehearsed on a 38 M-parameter `lfm2_moe` built from the real
`config.json` with every dimension shrunk — same module tree, same fourteen 3-D tensors, random
weights, four problems, CPU, four minutes. Allocation was healthy (−0.039% off its anchor, all ten
bank tensors priced) and then `eval --map` refused the map `inspect --save-map` had just written,
as "10 module(s) this model does not have". Two defects, both on the arms scheduled fourth and
seventh, both of which would have surfaced roughly four hours into the real seven-hour panel. The
first is **"named_modules misses raw parameters"** in its third location: the
pre-flight guard `check_map_covers` — which also fronts `quantize` and `export` — resolved names
with `get_submodule` while the quantizer behind it resolved them correctly, so it refused maps the
next stage would have applied, for 91.5% of this checkpoint. It now calls the quantizer's own
resolver, because a guard that predicts what the next stage will do should call that stage's
resolver rather than reimplement it. The second is not a bug: the **packed runtime cannot hold a
weight that is not a module**, and the grouped path is P8. So `dynquant eval` gained
`--map-apply {pack,encode}` — `pack` unchanged and still the default, `encode` running the identical
encoder and writing the reconstruction back in the compute dtype, pinned bit-identical to `pack` on
a bf16 `Linear` where both apply. The MoE arms encode, the record says which mode ran, and the byte
figures still come from the allocator's priced map rather than from what the scored model holds.
The general lesson is the rehearsal, not either fix: a structural double costs minutes and issues
every command the expensive run will, and the only thing it cannot check is the answer.

The step after the panel is a table, so the table was written before the panel — nine tests and an
eight-mutation harness against a synthetic seven-arm run, no GPU and no model load. Its size column
reads the manifest rather than the scored model, because the DynQuant arms are resident at fp16
under `--map-apply encode` and a measured column would print 16 bits for the arm whose compression
is the claim. It re-checks every arm's drift from its anchor **in both directions** and, past 0.1%,
refuses to print the comparisons at all rather than footnote them — a panel that is not byte-matched
cannot support a table, and the signed version of that test was the one mutation that initially
survived, because under-budget is the failure that actually happens. And its twelve comparisons are
Holm-corrected inside two blocks — six head-to-head at matched bytes, six against the ceiling — with
the block sizes printed and the verdict following the adjusted p: in the fixture, two 4-bit
comparisons are significant raw (0.0309, 0.0243) and neither survives (0.0972), which is the case
the correction exists for. Writing it first is the cheap half of the same lesson as the rehearsal:
the formatting decisions get made against a panel that can be re-run in seconds, not against the one
that cost seven hours.

## 17. Phase 4 — the concordance reads 1.000, over 8.5% of the model

[`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §12 · measured 2026-08-08

The allocator prices a module two ways. Where the fine-tune left channel moments it uses measured
Gauss–Newton sensitivity; where it did not, the rank-product proxy, rescaled onto the sensitivity
scale by a ratio of medians. On LFM2.5-8B-A1B that split is **89 modules measured against 44
proxied — 8.46% of parameters against 91.54%**, because a batched expert bank's forward spans two
matmuls and a non-linearity, so no module boundary yields the `dY = dW x` pairing the Kronecker
form needs. The banks are not unmeasured by omission; they are unmeasurable by construction, and
178 moment tensors covering exactly 89 modules with zero bank keys is the confirmation.

That makes the rescale multiplier — **1.807e-17** here, identical at both anchors, as a property of
the two price populations rather than of the budget — the constant that decides where 91.5% of the
parameters sit in a heap ordered against the other 8.5%. The maps show how completely: at the
4-bit anchor every module above 4 bits is a measured one and every module below 4 bits is a
proxied one. Until this campaign that number appeared in no artifact — not the saved map, not the
panel record, not the manifest. It was a `logger.debug` line in a run whose driver captures stdout
at WARNING.

The within-role concordance that exists to catch the supplement's headline defect — an allocator
that produced a plausible bit map while never reading the scores — reports **138 of 138 pairs
agreeing, 1.000** at the 4-bit anchor, and every one of those pairs comes from `attn.o`, `ssm.in`,
`mlp.gate` or `mlp.up`. None comes from an expert bank. The guard is true and it is silent about
the mass of the model, including every module whose floor the 3-bit anchor breaches.

Two `inspect` paths were worse than silent. Per-width score statistics were computed over all
members of a group with unmeasured modules padded to zero, so a group of nothing but expert banks
printed `min = median = max = 0.0` — indistinguishable from a group the signal measured and found
worthless, on 2.11 G parameters at the 4-bit anchor and 5.89 G at the 3-bit one. `narrowest` had
the same error in a different shape, showing the widest tensors in the model at the bottom of a
ranked list with a score of zero. `BitMap` now carries a `Pricing` record that `_map_payload`
writes into the saved map and `panel_table` prints beside the width histogram; width statistics
cover only the members the quantity covers; and the allocation is bit-identical before and after,
which is what an observability fix should be. Four mutations added, 17 of 17 caught — including
the one that reports the proxied share as a module count, since 44 of 133 reads like an edge case
and is 91.5% of the checkpoint.


## 18. Phase 4 — seven arms on 38 M parameters, and the six they stopped

[`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §12 · measured 2026-08-08

The panel is seven arms over one 8 B checkpoint: a bf16 ceiling, GPTQ and AWQ at two widths each,
and DynQuant at two anchors. Nothing had ever run the *driver* — arms in sequence, records into the
manifest, manifest into the pairing guard and the table. Rehearsing it on a structurally identical
38 M double (the same `lfm2_moe` config with every dimension shrunk, random weights, four problems,
32 new tokens, CPU) costs about four minutes a pass. It failed four times, and reading the driver against the real split and the real resume path afterwards found two more the rehearsal could not reach. Reading the
driver against the real split afterwards found a fifth the rehearsal could not have reached.

**`run` could only load on the GPU.** `--device` existed on `linearize` alone; the one subcommand
the driver invokes hardcoded `device_map="cuda"`. The box's GPU is held for hours by the fine-tune
that produces the signal being scored, so the only machine the panel is scheduled on was the one
machine it could not be rehearsed on — which is why the rehearsal had been deferred rather than
run.

**The provenance qualification reached the tokenizer.** Baseline arms set `--model` to
`<merge>#gptq-4b-g128` so six records from one checkpoint do not all claim to be the same weights;
`--tokenizer` defaults to `--model`, and `from_pretrained` rejected it as a Hub repo id. It fails
after the calibration pass — on the real panel, four arms × a 256-sample pass over 8 B parameters,
each dying with the quantized weights already in memory.

**AWQ had no mappings for this architecture, and would not have said so.** `Lfm2MoeForCausalLM` is
in neither of llm-compressor's two registries — the dynamic hybrid-stack builder is the right shape
but requires `linear_attention` in `layer_types`, and this model says `conv` — so the Llama defaults
applied, and they are wrong in both halves of every block (`operator_norm` not `input_layernorm`,
`ffn_norm` not `post_attention_layernorm`, `out_proj` not `o_proj`). The visible failure is a raise
on a partially matched set. The invisible one is the reason the fix is a *count* and not a regex: a
mapping that matches **nothing** does not raise. It is logged at `debug` and skipped, and the arm
finishes as round-to-nearest under an AWQ label. Predicting the set count from `layer_types`,
`num_dense_layers` and `num_experts` — `[2, 2, 6, 18, 22, 704]` over six mappings — turns that
silence into a pre-calibration abort. The 704 is the one that would have been quietly wrong:
`match_modules_set` groups by lowest common ancestor, so an expert-local pair yields one set per
expert, not one per layer.

**Upstream's grouped-query guard is inert here.** `v_proj → out_proj` cannot be smoothed under
GQA — `v_proj` emits `kv_heads × head_dim` rows and `out_proj` consumes `heads × head_dim`.
llm-compressor drops the pair, but its check reads `balance_name.endswith(".o_proj")`, so on this
model `_smooth` reached `weight[-scales.size(0):]` with 256 scales for 128 rows. The pair is now
conditional on `kv_heads == heads` from the config, and the consequence — `self_attn.out_proj`
quantized unsmoothed — is written into the record rather than assumed.

Both AWQ arms now report **6 mappings, 138 of 145 Linear modules smoothed**, the seven that are not
named by suffix, and 145/145 weights rounded. Nineteen tests and sixteen mutations across the four
defects, all caught — after the first mutation pass caught nine of ten. The miss is the instructive
one: the selection test checked that every balance layer sits under a selected block but not the
converse, so deleting a mapping's layer scope, which selects all 24 blocks where 6 belong, passed.
Set equality in both directions closes it.

All seven arms then ran into the table. Every accuracy is 0.0%, which is correct and meaningless —
38 M random weights answering four text-to-SQL problems. The rehearsal measures plumbing, and the
plumbing is what four of the real panel's GPU-hours would otherwise have been spent discovering,
one arm at a time, each time after the expensive part.

**And the fifth, which the rehearsal supplied rather than tested.** Every setting in `eval_flags` is
stated on every arm's command line, because a setting left to a default is one two arms can disagree
about while their commands read identically. `--limit` is the exception: it is forwarded only when
set, and unset means the whole test split — **16 143 items**, on each of seven arms. The rehearsal
ran four problems at 32 new tokens, so it *passed the flag whose absence is the defect*. That is
general, and worth stating once: a rehearsal is only worth running if it is cheap, and what it
passes on the command line to make itself cheap is exactly what it cannot test. The launcher now
refuses `--go` without an item count, and a 128-item timed probe prices an arm before seven are
spent — which also catches a merge that landed below the 57.75% the base model scored, from five
minutes rather than from a finished bf16 ceiling.

**And the sixth, which no rehearsal could reach, because it needs a second run.** The panel launches
with `--resume`, and the whole of what resume checks is that the record file exists. `check_pairable`
looks like the guard against a foreign one and is not: it compares records through the fields a
paired test needs — task, backend, split, shots, shot seed, limit — every one of which describes the
problem set, and none of which names the model or can be older than anything. Two records scored on
two different merges at identical settings pair perfectly. `check_resumable` now refuses a directory
whose manifest names different inputs, and any record older than the inputs it claims to have
scored, charging the signal file against the two DynQuant arms and not the four baselines that never
open it — because condemning all six would price the cheapest correct fix at a whole new panel,
which is how a guard teaches people to pass `--resume` less carefully rather than more.

## 19. Phase 4 — the refusal that blamed the format

[`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §12 · fixed 2026-08-09

**The state on 2026-08-09, superseded.** Six variants were to be published and only two could
be written. The refusal was answered a day later by changing container rather than by fixing the
promised one -- §12 of [`phase4-packed-moe-runtime.md`](phase4-packed-moe-runtime.md), and row 20
above. What follows is the refusal as it stood, because what was wrong with it is the part worth
keeping: it blamed the format, and the format was never the problem.

The two 3-bit baselines have no container — `compressed-tensors` has no 3-bit packed form, so
the only thing `save_pretrained` could write is a full-size bf16 folder wearing a 3-bit label.
The two DynQuant arms were blocked by something else: `dynquant export` raised on every batched
MoE expert bank, saying the packed format could not represent one. That was written into the
report as scheduled kernel work, and it was wrong.

`QuantTensor` had carried `logical_shape` for exactly this case all along. An `[E, out, in]` bank
flattens to `[E x out, in]` rows, packs, accounts for its bytes and restores rank three on
dequantize. What refused it was a **second copy of the name resolver**: the export path had built
its own on `get_submodule`, which cannot reach a bare 3-D `nn.Parameter`, while `quantize_model`
had been resolving both cases correctly through `target_tensor`. The blind spot arrived wearing
the costume of a capability limit, because whoever wrote the refusal was reasoning from the same
wrong copy — the third instance of this shape in the package, after the eval task argparse
refused and the test that asserted `style` against `executes_code`.

Verified before the claim changed: a `[8, 256, 128]` bank exported at 4 bits writes
`.qweight (2048, 16)`, `.scales (2048, 1)` and `.offsets (2048, 1)` — 139 264 bytes for
262 144 parameters, **4.25 bits per weight** — with `logical_shape` preserved and the raw
parameter key absent from the shards. The fix had its own trap, mutation-checked by removing the
line that avoids it: a bank belongs to no tie-alias group, so unless the export loop marks the
bank's own state-dict key consumed, the dense pass writes every value again at fp16 and the
quantized directory comes out larger than the checkpoint it compressed.

Half the original claim survives and is the half worth keeping separate. The packed *runtime*
still cannot load a bank — it swaps `Linear` and `Embedding` modules and there is none to swap
— so a DynQuant directory is **size-honest today and loadable after P8**. Two facts about one
artifact, and both now travel with the directory as `ExportReport.banks` and as `expert_banks` in
the manifest, because whoever uploads the folder will have the folder and not the report.

Run against the real architecture rather than a grafted Llama, the same duplicate refused a second
time one line lower: an `Lfm2MoeTopKRouter` owns a plain `[num_experts, hidden]` weight and is not
a `Linear`, so a whitelist `target_tensor` never had rejected five routers the pre-flight guard
accepted and the quantizer encodes. **A copy narrows, and this one narrowed twice** — so the
duplicate is gone rather than widened again. The full 38 M map then exports whole at
**4.1576 average bits against a predicted 4.1585**, and the packed prediction reconciles to the
byte. Only rank-1 tensors go unpriced: 205 056 B on the real model, 0.0047% of the 4-bit
anchor and 21x inside the panel's match tolerance, and the same set `llm-compressor` counts as its
211 712 fp16 parameters.

## 20. Phase 4 — the artifact and the measurement were five defects apart

[`phase4-packed-moe-runtime.md`](phase4-packed-moe-runtime.md) · fixed 2026-08-09, extended 2026-08-10

The panel scores the **encoder**, `quantize_model(in_place=True)`, because a matched-byte panel
needs one GPU pass per arm and nothing on disk. What a person downloads is the **packer**. Those
are two implementations of one format, and the panel’s evidentiary value is entirely the claim
that they are the same object. Run against the genuine `Lfm2MoeExperts` rather than the synthetic
copy the unit tests use, they were not, in three ways that hid behind each other.

**Rank 3 is not a bank.** `resolve_target` accepted any rank-3 `nn.Parameter` as a batched expert
bank. LFM2.5-8B-A1B is a hybrid whose **18 of 24 layers are short convolutions**, and an
`nn.Conv1d` kernel is `[channels, 1, width]` — rank 3, and a `[1, 3]` strip of it per "expert".

**The indexing loop is not what runs.** The design rests on a counted property: 49 of the 52
`*Experts*` classes in transformers 5.14.1 reach an expert by indexing their bank. That is a count
of what those classes’ `forward` methods contain, and `integrations/moe.py` **replaces** each one
with a dispatcher reading `config._experts_implementation` against `ALL_EXPERTS_FUNCTIONS`. The
loop is only the `eager` entry; the default is `grouped_mm`, which hands the bank whole to
`torch._grouped_mm`. So the bank installed correctly, passed every test, and raised
`'DynQuantExpertBank' object has no attribute 'transpose'` at its first real forward. Both the pack
path and the load path now move the model through its own `set_experts_implementation`.

**And the move is not free, which this report said it was.** The line above used to read *"the
move is free: eager and `grouped_mm` differ by 1.79e-07 against a quantization effect of 0.0101"*
— a true measurement of a 4-layer 6-expert fp32 model, copied into five places and used
to license a claim about an 8B one. On LFM2.5-8B-A1B, teacher-forced over 24 real text-to-SQL
items, the two dispatches disagree on **1.24% of tokens** against a quantization effect of 4.33%:
**0.29x**, confirmed independently by peak-logit deltas at 0.28x. The mechanism is that a top-k
router turns a last-bit numeric difference into a *different set of experts* — layer 2
routes bit-identically, layer 23 agrees on 7% of its slots. Four arms of the panel had already
landed straddling it, because `llm-compressor` linearises an expert bank into per-expert `Linear`
modules and a baseline therefore computes eager while an encoded dq arm computes `grouped_mm`.
`dynquant eval` now pins every arm to `eager` — the only dispatch a linearised baseline
and a packed artifact can both run — and records the dispatch in the record, where
`EXPERTS_PAIRING_FIELDS` refuses to pair two arms that ran different arithmetic.

**Two corrections to that, one of them to the sentence above.** The pairing field refuses to pair
two arms that *recorded* different arithmetic, and the banked arms recorded nothing: the panel clone
is pinned at `4109dcc`, which predates the pin, the flag and `use_eager_experts` alike, so the
`experts` key is **absent** and `_comparability` exempts it. Checked against the five records rather
than assumed, the straddle is also wider than "four arms" — it is five of five, `bf16` included, so
no arm in the banked panel had its dispatch chosen rather than inherited. The bf16-to-DynQuant
margin survives that intact, both sides being on `grouped_mm`; the DynQuant-to-baseline margins do
not, and the re-score is what settles them.

**And the pin is no longer what a packed artifact needs.** The observation that the right home for
the P8 kernel is *inside* `ALL_EXPERTS_FUNCTIONS` turned out to be the fix rather than a plan for
one: `dynquant_experts_forward` is `grouped_mm`'s own forward — same sort, same offsets, same
sentinel mask, same single reduction over the k axis — with `bank[e]` substituted for the whole-bank
read. At the 8B's MoE geometry it is **bit-identical** to `grouped_mm` at bf16 and fp32 where
`eager` disagrees on **1.95% of argmax tokens**, so a downloaded checkpoint no longer trades
arithmetic for a packed bank. The pin stays on `eager` for panels only, and only because a
linearised baseline has nothing left in it to dispatch. What is still owed is the fast path.

**One format, two encodings.** The packer and the encoder encoded the same fp32 bank 0.0082 apart.
Three copies of the scale-dtype rule: the packer used the weight’s own dtype, the exporter used
fp16 unless already half, the encoder passed nothing and inherited the exporter’s. All three agree
on fp16 and bf16 — so on every model anyone ships — and the disagreement waited for an fp32
test model. The **fourth** instance of *a second copy of a registry agrees until it doesn’t*, and
the most expensive, because the two copies were the two things being compared. Settled on 16-bit
metadata in one shared `storage_dtype()`, not on accuracy grounds but because `budget.py` prices
every quoted bit-width against `metadata_bits: int = 16` and fp32 scales would put a model 0.25
bits/weight above its own manifest; the first attempt, on the weight’s own dtype, turned 166 tests
red and that is what identified the budget as the fourth copy. The one caller that genuinely needs
the parent’s dtype — `DynQuantExpertBank.__getitem__`, which is asked for a weight before any
activation exists — is told at construction and holds it in a non-persistent buffer so `.half()`
carries it.

Afterwards, on the genuine class: `max|bank[e] - encoder[e]| = 0`, packed-vs-encoder logits `= 0`,
and a bank exported and reloaded through `from_pretrained` returns `0`. Eleven mutations across the
two fixes, all eleven red. The limit is scale, not fidelity: this is the real *class* at 4 layers
and 6 experts in fp32, and the 8B’s own export-and-reload is running beside the panel.

**And the same writer, pointed the other way.** Four of the six promised variants cannot be written
in the container their own recipe produced — `lfm2_moe` has no registered inverse for the expert
rename, and 3 bits does not divide 32. Both are properties of that container, neither is a property
of the numbers, and DynQuant's own format holds the numbers exactly: `scale * (q - zero)` is
`scale * code + offset` with `code = q`, so a baseline's grid is *carried* rather than re-fitted.
Re-fitting is the control and it is not a strawman — the same code path with the encoder swapped —
and it moves the weights 0.399—0.518 code steps where carrying moves them 0.0000 on every symmetric
arm. The residual on the two AWQ arms, 0.0681 and 0.0240, is the bf16 offset dtype rather than the
carry: a symmetric arm's `scale * -8` is exactly representable and an asymmetric arm's
`scale * -11` is not. Getting there cost three defects that only a real `llm-compressor` run could
reach, and the third is the family this record keeps enumerating — a range worked out here instead
of asked of the library, green against every symmetric arm and wrong about the first asymmetric one.
What is still owed is the 8B: this is the real architecture at four layers, and the publish path has
not been run at scale because the panel has the GPU.

**And the loop that was correct and still cost something.** §8 retired its own premise — the
packed forward measures bit-identical to `grouped_mm`, so no arm's number depends on the
dispatch — which settles correctness and leaves the loop. Thirty-two `F.linear` calls per bank,
and worse, a Python list of segment offsets: building one is a device-to-host copy, and this model
has 22 MoE layers holding two banks each, so **44 synchronizations per token**. The copy is
nanoseconds and irrelevant; the fence is neither, because a forward containing a host read cannot
be captured as a CUDA graph and cannot be traced under `fullgraph=True`, which are the two things
the packed runtime needs next. The table is an `[E + 1]` int32 device tensor now and the launch
geometry comes from its *shape*. Since removing a synchronization changes no output, the property
is asserted by counting `.tolist()` rather than by comparing results — every equality test in the
file passes on a version that computes the list and throws it away. The mutation run made the same
point from the other side: `expert_ids % num_experts` survived three rounds of fused-against-loop
comparison because both paths read the same table, and it took a known answer with one sentinel
and three experts to kill it. The kernel itself is written and **not yet compiled**; its
band-for-band bit-exactness holds on CPU by construction and is unverified on CUDA, and no speedup
is claimed.

## 21. Phase 4 — the margin is fidelity, and the confound turned out to be worth a tenth of a point against it

[`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §13 · measured 2026-08-10, re-measured 2026-08-11

The panel's headline is DynQuant +0.78 points over GPTQ at 4 bits, and that number is not one
number. By source it is +1.24 on wikisql and -0.59 on gretel. But the two sources differ in more
than identity — the bf16 ceiling is 88.50% on one and 72.02% on the other — so a source-wise
test on this panel cannot separate *which corpus* from *how hard*.

**Stratifying by the ceiling's own answer costs nothing and tells them apart.** The label owes
nothing to either compared arm, so McNemar stays valid inside each half, and the ceiling answered
every item, so no `sources.json` is needed. The margin varies with difficulty far more than
sampling explains — +1.26 where bf16 is right against -1.80 where it is wrong, Q=11.70 at Holm
0.00125 — and its point estimate is negative among the items the ceiling missed. The earlier
reading said it *changes sign*; that cell read -2.22 at Holm 0.0342 on the first panel and reads
-1.80 at Holm 0.126 on the second, so the sign is no longer separated on its own and the claim
rests on the spread instead. Which is the stronger test of what is actually being claimed.

**One measurement accounts for both signs.** A quantized arm either matches the ceiling on an item
or flips it, so `accuracy = c·f_right + (1-c)·(1-f_wrong)` is an identity rather than a model.
DynQuant agrees with bf16 on 95.30% of items against GPTQ's 93.96%; the two baselines do not
separate from each other on fidelity or on accuracy. An arm that tracks the ceiling more closely
inherits its right answers and its wrong ones, so winning the 84% and losing the 16% is forced.
The difficulty split is that identity being read, not independent evidence for it. It also prices
the source spread without appealing to the sources: every 4-bit *fidelity* margin is flat across
gretel and wikisql, and gretel is 28.0% ceiling-wrong against wikisql's 11.5%, so the term
DynQuant loses is weighted two and a half times more heavily there. The four rows sum to the
observed margins exactly.

**Two controls this section used to lean on did not reproduce.** It read GPTQ-vs-AWQ moving across
the four crossed cells at Q=8.73 as evidence that cell structure is a property of the *mixture*
rather than of the method under test; that row is now consistent at Q=4.41, Holm 0.22. It also
reported the baselines' *fidelity* margin changing sign between sources, +1.76 on gretel against
-0.13 on wikisql; it is now +1.31 and +0.04, consistent. Both sat at Holm ≈ 0.015 across a six-row
block, which is about how often such a block is entitled to produce one. Losing them costs an
argument and improves the position: at 4 bits the baseline-only comparison is flat everywhere while
both comparisons involving DynQuant are not, so the structure is specific to the rows with
DynQuant in them — and it has a mechanical explanation that needs no control, since opposite signs
across difficulty are what fidelity looks like when it is read as accuracy. Real baseline cell
structure is still available on this mixture; it is at 3 bits, where GPTQ and AWQ separate by only
+2.84 pooled and spread at Q=94.46.

**And the confound is closed by measurement.** The section used to end one confound wide: the
DynQuant arms ran `grouped_mm` while the linearised baselines ran `eager`, and §8 measured those
dispatches disagreeing on 1.24% of teacher-forced tokens — 0.29x the fidelity gap, on the same
axis. `rescore_eager.sh` re-scored `bf16`, `dq_4b` and `dq_3b` under the baselines' kernel. The
records carry `experts: {found: grouped_mm, ran: eager}`, so the first panel's banked arms did
dispatch `grouped_mm` and this is a measurement rather than a relabelling; the allocation did not
move and `maps/` is byte-identical. Dispatch changes 102 of bf16's 12 000 items — 0.85%, close to
§8's estimate in magnitude — but **53 up against 49 down**. Direction is what §8 could not supply
and no argument was going to, and near-symmetric noise at that rate cannot manufacture a systematic
1.34-point gap. Taking it away made every margin *larger*: 4b +0.64→+0.78 against GPTQ and
+0.94→+1.08 against AWQ. A confound that had been supplying the effect would have shrunk it.

One thing the re-score cannot be mined for. The eager arms finished about three times faster at the
same `--batch-size 32` on the same items, which looks like a dispatch cost and is not admissible as
one: the two panels ran two days apart, no arm ran in both windows under one dispatch, and a
uniform factor across three arms is what a slower window and a slower kernel both look like.

The statistics and the tables are both generated. `panel_table.py` cuts fifteen comparisons three
ways and carries all three into the json the model cards read; `report_tables.py` formats that
payload into the markdown this section prints, so the rows arrive with the Holm family size
attached rather than losing it between a terminal and a report. Both panels are committed under
[`experiments/phase4/results/`](../../experiments/phase4/results/) — the eager one is the panel of
record and the grouped one keeps its three superseded records — so the comparison no longer depends
on a box whose `/workspace` is not a volume.

## 22. Phase 4 — nineteen points at 3 bits, and the control that turned them into nine

[`phase4-text2sql-mixture.md`](phase4-text2sql-mixture.md) §13.4, §13.5 · measured 2026-08-11, control 2026-08-15

At 3 bits the panel stops comparing degradations and starts comparing a degradation against a
collapse. At 3 332 904 576 bytes — 5.08x compression, every arm within 0.05% of the anchor — GPTQ
scores 60.76% and AWQ 57.92% against the bf16 ceiling's 84.29%. DynQuant scores **79.89%**: +19.13
over GPTQ, +21.98 over AWQ, both at *p* below double precision, and -4.40 from the ceiling it was
encoded out of. It is the only 3-bit arm still emitting SQL, 213 unparseable generations against
1 008 and 1 523.

**The fidelity instrument reads that as a difference in kind.** `gptq_3b` agrees with bf16 on
86.68% of the items bf16 got *wrong* and only 69.60% of the ones it got right — where-wrong above
where-right, which no arm still tracking a ceiling can produce; it "agrees" on the failures by
being broadly wrong, and the identity returns its 60.76% exactly. `awq_3b` is the same shape.
`dq_3b` keeps the normal ordering, 91.40 against 81.86, which is why both terms of its
decomposition point the same way and there is no cancellation left to do.

**What the allocator did is legible, and the table is generated.** `map_roles.py` reads the
exported maps: all **22 routers held at 8 bits at both budgets**, the widest width in the table on
the smallest tensors in the model, while the budget tightens a full bit per parameter around them —
paid for by **all 22 expert down-projections at 2 bits**, 0.94 G parameters at the cheapest width.
A router that mis-ranks its top-k sends a token to the wrong expert and no downstream precision
recovers it, which is the failure a uniform recipe has no way to avoid. The 4-bit map breached no
floor at all; the 3-bit map breached 15 — the tied embedding 8→4 and fourteen `gate_up` banks 4→2
or 3 — which is the soft-floor mechanism doing what P4 specified instead of returning the floor map
and reporting failure.

Running the tool for the first time found two defects that hand transcription would have carried
into the report. `embedding` shows floor 8, not the 4 in `DEFAULT_FLOOR_BITS`, because the tie to
`lm_head` raises it and `Policy.floor_for` takes the strictest floor across a tie — the first
version printed the default in a column headed "floor" three lines above a breach row printing 8.
And `attn.o` is 24 modules where `attn.q` is 6, because 18 of them are the short-conv block's
`out_proj`; both carry floor 4 so the widths are right and only the label is broad. The disclosure
is generic — it groups a role's members by parent segment — so it would fire on any architecture
without the tool knowing what LFM2 is.

**And the controls ran.** Three matched-byte arms at the same anchor, differing from `dq_3b`
only in what the allocator was shown — `shuffle` permutes the driving quantity within role,
`flat` draws the same permutation and then sets every score to 1.0, `uniform` flattens every
score and consults no sensitivity table — chain into four rungs that partition the margin exactly
in raw counts, 92 − 141 + 1186 + 1159 = 2296 items: within-role placement **+0.77** [+0.27,
+1.26], the score's magnitude over a table permuted the same way **−1.18** [−1.62, −0.73], the
measured `dL` table permuted but present **+9.88** [+9.16, +10.60], and floors-plus-knapsack with
no signal at all **+9.66** [+8.74, +10.58] — that last rung measured against a *symmetric*
GPTQ, which the control below splits into +9.49 of scheme and **+0.17** of allocator shape. **The
fine-tune signal is +9.47 of the +19.13 against that baseline, 49.5%; against a GPTQ allowed its
own asymmetric grid the margin is +9.64 and the signal's share is 98.3%.** All three controls
spent *more* bytes than the real arm, so every byte edge in the block runs against DynQuant. A
method with no training-time hook, given this package's role floors and soft-floor knapsack,
scores 70.42%: ten points over a symmetric GPTQ, and **not separated** from a recovered one
(*p* = 0.706).

**And the signal's half is the pricing, not the ranking.** The middle rung is negative: setting
every score to 1.0 *gains* 1.18 points over permuting it, 450 items flipping to `dq_3b_flat`
against 309. The real arm does not separate from the flat one at all — **−0.41** [−0.89, +0.07]
on 404/453 flips, *p* = 0.101, the only comparison in the section that fails to separate. So the
plasticity-times-saliency ranking is not distinguishable from a constant on this model at this
budget, a *permuted* ranking is measurably worse than none, and the measured `dL` table carries
+9.88 on its own — more than the whole signal is worth, because the score channel hands 1.18 of
it back. Flattening the score also pulls the width histogram in from the extremes, `{2: 26, 3: 11,
4: 61, 8: 35}` to `{2: 22, 3: 19, 4: 57, 8: 35}`, while its median relative error gets *worse*
(0.10385 against 0.10305) — so the ranking is not being paid back in fidelity either.

**A single within-role null would have reported the small half as the whole.** On the first rung
alone the share is 4.0% of the +19.13 (8.0% of the +9.64 the control below leaves), and the reason
is structural rather than statistical: a within-role permutation cannot move a bit from one role
to another, so it can never price the decision that holds `attn.k` and `attn.v` four bits above
their floor. The shuffle moved 20 of 133 widths, left the width histogram identical and reproduced
the floor-breach shape exactly — which is what makes it a faithful control and exactly why it
measures so little. Uniform moved 65 and grew the breached mass from 41.9% to 50.2% of parameters.

**The two earlier campaigns sit at opposite ends of this ladder, one each.** The re-check went the
other way from the reports that prompted it, which had said both computed the share the same way.
Qwen's control is `stage4_allocate.py`'s `uniform = dict.fromkeys(scores, 0.5)`, over an allocator
that predates the moments path and so consults no sensitivity table — this section's `uniform`
exactly, and its 12 % is therefore both rungs. Ministral's is `s3_maps.py`'s `shuf`, the
within-role permutation — this section's `shuffle`, and its 56 % is the first rung alone. So there
are two series of two, read against the recovered baseline the control below supplies: **12 % on
Qwen against 98.3 % here**, over allocator terms of +22.62 and **+0.17**, and **56 % on Ministral
against 8.0 % here**, over +1.91 and +8.87 — where that 8.0 % is one of four seeded draws and the
four span 4.1 % to 12.4 %. Both run in the direction those reports predicted, the share growing as
the allocator's structural advantage shrinks — though two points are monotone whatever they are,
and the denominators do not match: Qwen and Ministral divide by `dq` − `rtn`, uniform-width
rounding, where this panel has no RTN arm and bottoms at an asymmetric GPTQ instead — a stronger
floor than either, which is what makes the 98.3% as inflated as a share can get. Closed since:
`apply_null` gives `flat` the same within-role permutation it gives `shuffle`, so the +9.88 priced
a permuted table against no table — `dq_3b_tabl` keeps the real `dL` rows by identity and flattens
only the score, and it prices the table at **+9.96** with the score channel at **−0.48**. Still
open: the permutation spread is four draws, enough to see it is real and not enough to bound it;
GPTQ and AWQ handed *DynQuant's* bit map remains unbuilt, which would price the allocator against
the quantizer rather than against itself; and an RTN arm at this anchor, which the control below
makes the load-bearing one.

**And the control this section was written without has now run, and it falsifies the sentence
above about ten points.** `gptq_3b_asym_noao` re-fits GPTQ at the same 3 332 904 576 B anchor with
the same calibration set and group size, changing one flag: `symmetric=False`, act-order left off.
The flag alone is worth **+9.49** [+8.65, +10.33] on 1944/805 flips — GPTQ goes 60.76% —
**70.25%**, SQL errors 1 008 — 586, exact matches 5 127 — 6 500 — so DynQuant's margin at 3 bits
is **+9.64** [+8.88, +10.41] rather than +19.13, still separated at Holm 6.8e-133. And the
signal-free allocator, which this section said would stand ten points above GPTQ, stands **+0.17**
[−0.66, +0.99] above the recovered one at *p* = **0.706**, the least separated comparison in the
panel, with fidelity agreeing (+0.42, *p* = 0.33). The ladder reconciles exactly onto the new
baseline: −0.48 + 9.96 + 0.17 = **+9.64**, and −58 + 1 195 + 20 = **+1 157** items. So the
fine-tune signal is **98.3%** of what is left — a stronger claim about the method, on a smaller
margin. **But the +0.17 is two opposite results cancelling**: split by source it is **−7.87** on
gretel and **+2.92** on wikisql at Q = 123.66, the largest heterogeneity anywhere in this panel,
and the mixture is 74.5% wikisql; split by difficulty it is the only *consistent* row of the three
(+0.35 / −0.80, Q = 1.12, neither stratum separating). The Holm family grew from six to nine, so
every previously published adjusted *p* in this panel moved — no verdict flips, and §13.5
tabulates both. Act-order remains unmeasured here and must not be assumed to recover further: on
the Mistral panel, act-order on top of the *same* asymmetric grid took the arm from 76.08% down to
**3.99%**.

## 23. Phase 4 — the same panel on a dense model, where the nineteen points do not reproduce

[`phase4-mistral7b-text2sql.md`](phase4-mistral7b-text2sql.md) §8 · measured 2026-08-12

Every phase-4 number above was measured on one mixture-of-experts checkpoint. This campaign runs
the same seven arms — bf16, and GPTQ, AWQ and DynQuant at a 4-bit and a 3-bit byte anchor — on
**Mistral-7B-Instruct-v0.3**: dense, 7.25 G parameters, no expert banks and nothing for the
proxy price to cover, fine-tuned two epochs on a Gretel + WikiSQL + SQL-create-context mixture and
scored by execution match over 2 454 held-out problems split evenly across Gretel, WikiSQL and
**Spider**.

**At the 4-bit anchor nothing separates.** 3 964 674 048 bytes, 3.66x compression: GPTQ 78.28%,
bf16 78.16%, DynQuant 78.08%, AWQ 77.91%. All six pairwise comparisons return an adjusted *p* of
1.000 — including DynQuant against the ceiling it was encoded out of. A panel reporting a winner
here would be reporting noise, and this one does not.

**At the 3-bit anchor the collapse was the grid, and the ordering does not survive the control.**
3 068 534 784 bytes, 4.73x compression. As the panel first read: DynQuant **75.22%**, AWQ 74.16%,
GPTQ 6.68%. The control then ran GPTQ again at the same anchor with **an asymmetric grid and
nothing else changed**, and it came back at **76.08%** — **+69.4 points from the zero point
alone**, McNemar *p* ~ 0, 1 725 discordant split 11/1 714. Against that arm DynQuant is **−0.86,
*p* = 0.186, not separated**, and AWQ is −1.92 at *p* = 0.0054. The second half of the control,
activation ordering on top of that grid, recovers nothing and collapses the arm again to **3.99%**
— below the symmetric default it was meant to improve on — so one of three GPTQ configurations at
this anchor produces a working model. Among the arms that do: **DynQuant separates from nothing —
not from AWQ, not from the recovered GPTQ — while AWQ does separate, below the recovered GPTQ.**
Where a separated result does exist it goes against DynQuant: agreement with bf16 is 90.30%
against the recovered GPTQ's 93.36%, **−3.06 at *p* = 8.1e−07**. The +1.06 over AWQ is *p* =
0.149, adjusted 0.744 — it never cleared significance, and §22's nineteen points do not reproduce
on a dense model. What stands alone is the compression: 4.73x retaining **96.2%** of the
fine-tuned model's accuracy, with 35.0% of the model allocated below its role floors and all 57
breaches reported by name. The Spider claim does not stand: the asymmetric GPTQ arm leads there
too, 487 of 818 against DynQuant's 480.

**Every GPTQ arm this project has published was fitted symmetric.** Not an inference — `git log
-p` shows `symmetric=(method != "awq")` hardcoded in `_llmc.py` from its first commit, and the
banked side files carry no `symmetric` key because the flag postdates the arms. The LFM2.5 panel
in §22 has since run the same control — GPTQ recovers 60.76% to 70.25% there, DynQuant's margin
survives at +9.64, and the signal-free allocator's does not — which leaves the comparison
uncontrolled under the phase-2 headline on Qwen3.5-2B/CaseHOLD, where `gptq_3b_head` sits **1.71
points under its fp16 ceiling** and the DynQuant margin claimed over it is **1.54**. Those numbers
are not wrong; they are unattributed. `--symmetric yes|no|auto` and `--actorder` now exist, every
arm records the scheme it was fitted at, `panel_table` carries it into the table, and arms that
predate the flags are recovered through one shared rule and labelled as recoveries rather than
guessed at. **Until the control runs on that panel, the phase-2 headline may not be cited as
"DynQuant beats GPTQ at 3 bits."** The published model cards carry that caveat and now name the
arm that isolates it, emitted off the kinds actually present in the panel rather than off a
hardcoded list. The card generator also lost a sentence it had been carrying from the MoE
campaign, which would have told a reader of a dense model that 91.5% of its parameters were
batched expert banks.

**The 3-bit map in §7.2 was first read off the wrong checkpoint.** The feasibility dry run
allocated over the *base* Hub model; the panel allocated over the fine-tuned merge. The first
draft reported 56 floor breaches and widths {2:8, 3:109, 4:108, 8:1}; the map that was actually
quantized has **57 breaches and {2:7, 3:111, 4:107, 8:1}**, and within-role concordance is 560 of
561 rather than the dry run's 554 of 558. The allocator is deterministic — the disagreement was an
input, not a seed — and the diff became a measurement of its own: fine-tuning moves 6 of 226
widths at the 4-bit anchor with the width histogram preserved exactly, and 4 of 226 at the 3-bit
one. The incident is written up in §7.3 rather than patched out of §7.2.

Three arms are published, public, on 2026-08-13: the bf16 fine-tune and both DynQuant
variants, under `VikramPal/mistral-7b-instruct-v0.3-text2sql-{bf16,DynQuant-4bit,DynQuant-3bit}`.
The baselines are not, and the reason is structural rather than thrift: a DynQuant arm's map
*is* the artifact its score came from, while GPTQ and AWQ score in process and keep no
checkpoint, so republishing them means re-running the recipe.

## 24. The kernels compile, and the failure was in the reference

[`kernel-first-compile.md`](kernel-first-compile.md)

`gemv.cu` and `dequant.cu` had been compiled and measured before — the bandwidth and VRAM
numbers above come from them. `grouped_gemv.cu` had not, anywhere, and the packed-MoE report
says so as its own closing caveat. This campaign builds the whole extension from source on an
L4 at `sm_89` (system nvcc 12.4.131, torch 2.11.0+cu126) and runs the suite.

The grouped kernel is correct at first run: **23 of 23** parity cases, each against
`QuantTensor.dequantize()` per expert band, so reading the right band at the wrong offset fails
rather than returns plausible numbers. The rest of the surface came back **653 passed, 1 failed**
— on CPU, at the widest geometry, one element in 260.

The failure was real and it was in the *reference*. `gemv_cpu` and `moe_grouped_gemv_cpu` both
materialised the dequantized weight through the `dequant` op, which rounds its store to the
scales' dtype; with fp16 scales that puts one fp16 rounding on every weight element in front of a
sum over `in_features` of them, and the error grows like `sqrt(K)` against a flat tolerance. The
kernel has no such error — it never materialises the weight and applies scale and offset in fp32.

Priced as the fraction of the parity tolerance each path spends, over every geometry × width ×
row count: CPU **1.041 → 0.000**, CUDA **0.043 → 0.043**, with the CUDA rows byte-identical on
both sides of the fix. The CPU column rises monotonically with `in_features` (0.032 at K=4 to
1.041 at K=3072) where CUDA is flat at 0.02–0.04 — the margin was a function of geometry on one
device and not the other, which is what an accumulating reference looks like.

Two costs are recorded rather than smoothed. The first probe of this counted `|d| > atol` and
reported 20 of 20 CPU seeds failing where pytest reported one in 654, because `assert_close`
allows `atol + rtol*|expected|`; that figure is discarded. And the fix turns the CPU parity case
into an *unpacker-agreement* test with the matmul held in common, so the
kernel-against-reference question it is named for is answered on CUDA alone.

None of the four gates could have caught it: `tests/test_kernels_parity.py` is
`importorskip`-ed whole on a machine with no compiled extension, which is every machine those
gates have run on. The local gate reads 2238 passed / 14 skipped both before and after.

Both closing caveats were retired afterwards, on the same box and the same build, and written up
as sections 10 and 11 of the campaign report rather than folded into it — so the report still
reads in the order the work happened.

The grouped kernel is **4.0×–15.9×** the per-expert loop at decode, which is P8's ≥3× gate met on
the denominator that is hardest to argue with, and **3.1×–5.7×** against the same loop over a bank
that is already dense — so the win is one launch over segments, not skipped dequantization. It
crosses over as tokens per bank grow and loses **8.3×** at 2048 rows on the widest geometry, which
is the prefill split stated in measured milliseconds instead of in principle.

The sweep's own denominator was the second finding. `DynQuantExpertBank.__getitem__` reached the
pure-torch *reference* dequantizer even on a box with the kernels loaded — a median **4.01×**, up
to 46.13×, that the loop never had to pay — so the first draft's 29×–209× was mostly a defect in
the baseline rather than a property of grouping. The fix is two lines; the guard is a test that
asserts the *call* and not the output, since on CPU the two paths are numerically identical and a
value comparison would have stayed green through a revert. The old cost stayed in the sweep as
its own column, so a revert shows up as the loop climbing to meet it.

All four `compute-sanitizer` tools then ran on this build: 0 errors and 0 hazards over the
108-launch grouped workload and over the parity suite. Establishing that took a correction the
first framing got wrong twice. The default `--target-processes application-only` does not follow
a spawned subprocess, which is how the suite runs, so the number of grouped launches that had
ever been instrumented was **one**, not twenty-three — and that is now demonstrated rather than
argued: the same out-of-bounds gather reports `ERROR SUMMARY: 3 errors` directly and under
`--target-processes all`, and under the default one subprocess away prints no `ERROR SUMMARY`
line at all.

An end-to-end model has now run through it. `LFM2.5-8B-A1B` decodes through `grouped_gemv.cu`
coherently at 3 and 4 bits, at **1.95x** the per-expert loop at 4 bits and **2.66x** at 3 -- and
3 bits is **3.0% faster than 4** on the grouped path (31.58 against 32.52 tok/s) while the loop
loses 24% going from 4 bits to 3, so its advantage grows exactly where this project's margins
are. Against bf16 it is 0.95x the rate at **3.76x** less resident memory at 4 bits and 0.98x at
**4.91x** less at 3. That memory
figure is the one this run nearly got wrong: peak over load-and-pack is 7167.5 MiB against a
resident 4295.7, because packing keeps a dense copy on the GPU, and only resident is what a
server pays -- measured that way it sits **6.4 MiB** from the byte accounting on a 4.3 GB model,
which is P6's VRAM gate met against the allocator rather than predicted from the bit map. Running
it also found what reading it had not: `dynquant eval --map-apply pack` **cannot reach the packed
runtime on this model family**, because `MOE_ROUTER` carries an 8-bit floor rather than an
exclusion, so the router is in every map however the map was made, and `pack_model` refuses it by
class.

Then P8's graph clause closed, and closing it falsified a claim two committed reports had
published. `torch.bincount` sizes its output from `input.max()` read **on the host**, and
`minlength` raises the floor on that size without removing the read -- so a fence survived inside
`_segment_offsets` on the **fused** path, the very path the removal above was about, and the first
capture attempt refused. It survived because that removal asserted absence by *counting*
`.tolist()`, which `bincount` never calls: an exact counter answering a narrower question than its
section claimed. Rebuilt as a `scatter_add_` into a fixed `[E + 1]` buffer, one MoE block captures
and replays at **3.25x** eager at one token and **3.65x** at 3 bits, 1.15x at 8 tokens and 1.00x
at 64 -- and the *removed* milliseconds fall too, 0.371 to 0.179 to 0.016, so what replay takes
out is the launch cost **not already hidden** behind GPU work, which is the shape a decode-only
claim should have. The per-expert loop refuses capture at every width, its `.tolist()` being the
trip count rather than a fence. The fence was also live in every packed step timed above: re-run,
`bf16` and the built-in `eager` control move 0.3% and 0.1% while packed 3-bit moves **+3.1%**,
which is what corrects the width claim. Still not claimed: no vectorized grouped variant, one
architecture, one model family rather than the Mixtral and Qwen3-MoE the gate names, and the
3.25x is **one MoE block, not a captured decode step** -- 22 x 0.371 ms = 8.2 ms against a
measured 31.67 ms step is an upper bound, not a prediction.

## 25. The whole packed model captures, and the supported path is the fastest arm

Section 24 closed on an upper bound rather than a measurement — 22 MoE layers x 0.371 ms of
removed launch latency against a 31.67 ms decode step, so at most 26%, *an upper bound, not a
prediction*. The L4 was still rented, so the bound was tested instead of quoted, on the packed
LFM2.5-8B-A1B rather than on one synthetic block.

It is not 26%. A whole-model forward at sequence length 1 — every module the model owns, no
cache — captures and replays at **3.98x** at 4 bits and **4.34x** at 3, 29.63 ms down to 7.44
and 29.52 down to 6.80, bit-identical to a fresh eager forward after a new token is written into
the captured input buffer. On a real decode step the removed fraction is **76-82%**. A block has
a handful of launches around real arithmetic; a model has 24 layers of them around the same
arithmetic, which is why one block could not predict this.

The arm that matters took four wrong explanations to reach. A **real** decode step at position
255, cache built by an actual prefill, captures under the default `DynamicCache` and replays at
**4.03x** — and produces the wrong token: `max_abs_delta` 15.33, argmax disagrees, nothing
raised and nothing warned. A CUDA graph records *addresses*, and a `DynamicCache` grows by
`torch.cat`, so replay reads buffers the cache has already abandoned. `torch.compile(mode=
"reduce-overhead")` refused the same container outright — inductor's `cudagraph_trees` guards
exactly that case — and the static cache, which is the container a capture actually wants, died
on a device-side `index_copy_` assert inside `generate` before a step was taken.

Four explanations of that assert were written down and each was falsified by an observation
rather than by argument: the model's 18-convolution / 6-attention layer mix (refuted — all 24
layers are present and initialized in the cache); this file's own decode position (refuted — an
in-bounds position asserts identically); the graph (refuted — one **eager** step with no capture
anywhere in the process asserts too); and the packed runtime (refuted — a dense bf16 model
produces a byte-identical traceback). The cause is two lines of `transformers`: `generate` sizes
a static cache at `max_length - 1`, and `StaticLayer.update` **ignores the `cache_position` its
caller passes**, writing at a device-resident cursor it advances itself. A prefill of N tokens
therefore leaves a static cache both N long and exactly N full, and the next step runs off the
end at *any* position. There was never a position that worked. A `DynamicCache` has no capacity
to run off, which is why it hid this for three rounds and returned a wrong answer instead.

The fix is one argument — ask `generate` for `max_cache_len` above what the prefill will fill —
and with 192 spare slots every arm goes through:

| arm | 4-bit | 3-bit | graph breaks |
|---|---|---|---|
| hand-captured `step` | 4.207x | 4.556x | — |
| `torch.compile(mode="reduce-overhead")` | **4.936x** | **5.452x** | **0** |

**Zero graph breaks** across 111 packed modules, the grouped MoE kernel, 18 convolution and 6
attention layers, all in one graph — the first evidence at model scale that `custom_op` plus
`register_fake` holds, and it closes P8's *graph replay removes measurable launch overhead*. The
supported path also **beats** the hand-rolled reference at both widths, because Inductor fuses
inside the graph as well as capturing it. The arm written to be the yardstick is the one that
loses, which is the outcome that argues for shipping the compiled path rather than capture code.

None of the four static arms agreed with eager to zero — 0.375 to 2.906 — so the arms were
re-run with a control that takes **a second eager forward** and compares the two eager runs to
each other. A decode step mutates the cache it reads, so consecutive eager forwards need not
agree either, and without that number a graph-vs-eager delta cannot be attributed to the graph.
On three of the four arms the control is **larger** than the quantity it controls (2.31 against
1.69; 2.00 against 1.19; 0.25 against 0.375 the other way); on the fourth it is 2.28 against
2.906. All four agree on the argmax in both directions. Against the `DynamicCache` arm's 15.33
with a flipped argmax, that is two orders of magnitude of separation.

One number the speedup does not show: `cache_writes` is **116** for a 50-iteration run, because
every call advances the static cursor — replays included, the increment being a device op inside
the recording. A captured decode step is replayable only as many times as the cache has spare
slots.

Still not claimed: not tokens per second, since one step in isolation is not a generation loop;
not long context, since the removable fraction falls as the cursor grows and 448 slots is the
favourable end; not bit-exactness, which does not hold for eager against itself; and one card,
one model family, one prompt.

## 26. A second MoE family, and the margin widens where the launches get smaller

Rows 23 through 25 all came off one checkpoint. P8's last open clause asks for a second MoE
family, naming Mixtral-8x7B and Qwen3-MoE; at 47B and 30B parameters neither fits an L4 with room
for a bf16 baseline arm, so the clause was answered with the nearest thing that is genuinely a
different test rather than the nearest thing that shares a name.

`allenai/OLMoE-1B-7B-0125-Instruct` differs from LFM2.5-8B-A1B in every dimension the classifier
and the allocator have to reason about: 64 experts against 32, top-8 against top-4, expert
intermediate 1024 against 1792, 16 full-attention layers against 6 attention plus 18
short-convolution, a plain `nn.Linear` router against a bespoke router class, untied embeddings
against tied. What it shares is the batched `[E, out, in]` expert bank, which is the thing the
grouped kernel actually consumes — so it varies the router and the routing density without
varying what is under test.

Nothing had to be added for it. **98 modules packed, 16 routers left dense** — 2,097,152
parameters, 0.030% of the model — 0 tied, 0 skipped, `accounted_bits` **3.2515**, and the expert
banks moved from `grouped_mm` to the DynQuant dispatch exactly as they do on LFM2.5. The generic
structural classifier found `mlp.gate` through `out_features == num_experts` with an `experts`
sibling, which is the test P3 was designed around and the first time a family it was not
developed against has exercised it end to end.

Three arms in one process, on the same packed weights, 128 greedy tokens: bf16 **41.48**,
per-expert loop **11.59**, grouped kernel **36.86** tok/s. That is **3.180x the loop** — P8's
≥3x gate, met on a second family — and 0.889x of bf16. An earlier run before the cache probe was
added measured 3.167x, so it reproduces across processes. Resident memory after load is **2685.4
MiB** against a manifest `packed_bytes` of 2,679.8 MiB: **5.6 MiB apart, 0.21%**, which closes
P6's *peak VRAM ≈ manifest size* on a second family too. The bf16 memory ratio is **4.914x**,
matching LFM2.5's to four significant figures because both land at 3.2515 accounted bits. All
three arms generate coherently at 3 bits, and the two packed arms produce **byte-identical text
on the first prompt** while diverging partway through the second — the expected signature of
substituting one kernel at the dispatch and letting greedy decoding amplify a sub-margin
floating-point difference.

The margin being *wider* here than LFM2.5's 2.66x is the part worth reading. OLMoE is the smaller
model and its bf16 arm is duly faster, 41.48 against 33.25 — yet **its per-expert loop is slower
in absolute terms**, 11.59 against 12.23, while reading 17% fewer expert parameters per token. A
loop that does less work in more time is launch-bound, and the geometry says by how much: **384
expert matmuls per token against 264**, each doing **43% less arithmetic**, so roughly 1.75x the
launch overhead per unit of work. The grouped kernel issues one launch per layer regardless of
expert count and pays none of it. Row 25 measured that same overhead head-on, freezing a decode
step into one graph and removing 76–82% of it; this row watches the gap between the two paths
widen precisely where the launches get smaller, which is what a launch-bound account predicts and
a bandwidth-bound one does not.

The column that moves the other way moves honestly. Against bf16 the grouped path recovers 0.889x
here where it recovers 0.978x on LFM2.5, because a 2048x1024 expert matrix gives the quantized
GEMV less arithmetic to hide the dequantization behind. Both readings describe the same geometry.

One harness bug is recorded because a comparison would have concealed it. OLMoE ships
**`use_cache: false`** in its own `config.json`; LFM2.5 ships `true`. The harness builds an
explicit `GenerationConfig` and had simply not named that field, so 128 tokens would have been
decoded quadratically. All three arms would have paid it identically, so the **ratio** would have
survived intact and the **rate** would have been a number about a configuration nobody runs. The
first instrumentation written for it was itself wrong: it recorded `model.config.use_cache`, which
reports the checkpoint's static declaration and read `False` next to three arms that demonstrably
used a cache — `generate` is governed by the per-call `GenerationConfig`, not by `model.config`.
It was replaced with an observation rather than a declaration: a four-token probe that asks
`generate` to hand the cache back, recorded as `decoded_cache_len`, reading **30** on all three
arms.

What this does not establish: two families are not a trend, and the launch-count reading drawn
through them is consistent with both plus row 25's direct measurement rather than fitted to them.
The cross-family arithmetic is a sanity check, not a controlled comparison — the two models differ
in attention, vocabulary, depth and embedding tying, and only the within-model ratios are
controlled. Coherent generation on two prompts is a viability smoke test, not an evaluation.
And `dynquant eval --map-apply pack` still cannot reach this runtime on any MoE, because routers
carry an 8-bit structural floor rather than an exclusion, so both packed arms here quantize
uniformly at 3 bits.

## 27. The packed path against bf16, once both of them are compiled

Rows 24 through 26 timed the grouped kernel against the per-expert loop and against bf16 *eager*,
and row 25 measured graph replay on one model at one width. What none of them measured is the
control that decides whether any of it is a serving result: **bf16 compiled the same way**.
Without it a 4.96× could be the packed path, or could be what `torch.compile` gives anything on an
L4.

Fourteen arms at commit `d92ee05` settle it — two MoE families × {bf16, uniform 3-bit} × the
compile ladder, `--reps 3 --cache-impl static` on every arm so the cache is held fixed and the
compiler is the only thing that moves, with `dynamo_unique_graphs` read back from the counter so
each arm reports what happened rather than what was asked for. Compiling buys bf16 **15.6%** and
**14.3%**. It buys the packed path **4.96×** and **4.67×** — 31.57 to 156.57 tok/s on
LFM2.5-8B-A1B and 35.41 to 165.20 on OLMoE-1B-7B. Same card, same script, same cache, same commit.
What the compiler removes is a per-step launch and Python surface paid once per module that bf16
never had to pay.

The reading a deployment takes off that is a sign change. **Uncompiled, DynQuant decodes slower
than bf16** — 0.967× and 0.873×. **Compiled, it decodes 4.145× and 3.563× the compiled bf16 arm.**
Same weights, same kernel, same card on both sides of that, separated by one flag; and every ratio
against bf16 earlier in this campaign is the eager end of the pair.

Running `default` against `reduce-overhead` splits the win. On the packed path Inductor's fusion
is worth **3.116×** and **3.430×** and the cudagraph the remaining **1.592×** and **1.360×**; on
bf16 both halves are near-inert. Row 25 put launch overhead at 76-82% of a packed decode step by
capturing it; this splits the same quantity the other way and says fusion is the larger half —
with the caveat that `default` fuses and therefore removes launches too, so the columns are *with
graphs* and *without*, not *launches* and *work*.

Memory is unchanged by any of it: resident is identical across all three compile settings within
an arm, **3 286.7** and **2 685.4** MiB against manifests of 3 280.1 and 2 679.8 — 0.20% and 0.21%
apart, which is P6's *peak VRAM ≈ manifest size* on two families at once. `peak_mib_total` equals
`peak_mib_loaded` on every packed arm and exceeds it on every bf16 arm.

Two smaller things the run fixed or found. `decoded_cache_len` now reads **26** and **30** on all
twelve arms, prompt plus the three writes a four-token generation makes, where before `d92ee05` it
read the *warmup's* fill because `generate` keeps one static cache on the model and the probe
returned a live reference; being constant across compile modes is what says it measures the probe.
And both `--compile manual` arms exit 1 on both families with `accessing tensor output of
CUDAGraphs that has been overwritten by a subsequent run` — the hazard row 25's hand capture met
silently, raised here instead, on the arm kept as a check on the supported path rather than as
anything that ships.

What this does not establish: two families and one card, one prompt pair, three reps, nothing
bounding long context. The packed arms are a uniform 3-bit map with routers left dense at
`accounted_bits` 3.251 and 3.2515, **not** the phase 4 allocator's 3.1488, and no accuracy was
scored in this run, so nothing here revises a panel number. And `warmup_s` is 22.4-24.5 s on the
bf16 arms against 7.98-9.03 s on the packed ones — the model with 111 substituted modules and a
custom op in the graph compiles three times faster, reproducibly, and this file has no account of
why.

## 28. Every symmetric baseline was charged for a zero point it never stores

Seventeen campaigns above are byte comparisons, and this one asks whether the bytes were counted
right. They were not, in two independent copies of one line: `meta_bits = 16 + bits`, charged to
every arm. `compressed-tensors` writes `weight_zero_point` **only when the grid is asymmetric**,
which the format's own source says and a two-checkpoint probe then measured -- **0** such tensors
symmetric against **186** asymmetric, each `I32 [2,16]` = 1 024 bits = exactly `groups x bits`.
GPTQ and RTN are symmetric by default here, so every GPTQ and RTN arm this project has published
was over-charged **~0.7%** of its width, in the direction that makes the baseline look more
expensive than it is.

Restated rather than re-run, since nothing was re-quantized and no accuracy moves: `gptq_4b`
**4.1565 -> 4.1253** bits, `gptq_3b` **3.1488 -> 3.1253**, Mistral's **4.3760 -> 4.3453** and
**3.3869 -> 3.3639**. One arm corrects upward -- `gptq_3b_asym` never paid for its `weight_g_idx`,
**+0.163%**.

The consequence is not the restatement. `anchor_bytes` derived **one** budget per width, so every
DynQuant arm in phase 4 was sized on the asymmetric figure and then scored against a symmetric
arm: **+0.713%**, **+0.708%**, **+0.693%**, **+0.654%** more bytes, all **6.5-7.1x** the panels'
own **0.1%** tolerance and all one way. **DynQuant-against-GPTQ was not byte-matched on any
phase-4 panel.** DynQuant-against-AWQ was, and is untouched, so every panel still carries one
external baseline the comparison is honest against. The panels are not re-run; the affected
sentences are corrected in place with the gap stated.

It was found by a smoke test refusing to print -- two GPTQ arms 22 accuracy points apart
accounting to the same 3.1522 bits -- and the assertion's *stated* reason for firing was the wrong
hypothesis while the thing it actually tested was right. The lesson is the one worth keeping:
**a control that varies an axis needs every column able to see that axis.** A blind column does
not produce an obviously broken table; it produces one whose numbers are individually plausible
and whose comparison measures nothing. `phase4-packed-moe-runtime.md` names
asymmetric-against-symmetric as the difference between two arms in the same paragraph where it
offers their identical byte count as evidence.

Fixed by one definition: `_llmc.stored_meta_bits`, taking `symmetric` and `actorder`, called by
both stages, with `--symmetric auto` resolved once per run and read by both the arm record and
the size column. `anchor_bytes` now returns the cheaper scheme, so DynQuant is pinned under every
baseline rather than between them, and each baseline is held to the size its own scheme predicts.
Ten tests, the load-bearing one asserting the two schemes **cannot** account to the same width at
any of 2/3/4/8 bits -- a property, because the broken version satisfied every equality anyone
would have thought to write.

## Conventions that apply to every campaign

**Paired tests on stored per-item hits.** Every arm stores which items it got right, so every
A/B is an exact McNemar test rather than a comparison of two independent proportions. The
standard error of the difference roughly halves — which promoted a +1.07 to p = 0.0002 and
refused to promote a +0.51 over GPTQ.

**Stored bits, not payload bits.** Every "average bits" figure in every table is total stored
bytes divided by parameter count — scales, offsets and packing overhead included — so the
manifest number is the on-disk number. Comparisons against baselines are made at matched
*bytes*, never at matched nominal width.

**Task selection precedes fine-tuning.** Base-model accuracy is screened before a fine-tune is
spent, because a task the model is already at ceiling on cannot read quantization damage
against a fine-tuning gain. GSM8K's flat arms cost a full six-arm run to diagnose.

**Negative results stay in.** The rank-product score losing by 2.03 points, the +0.51 that
never became significant, the shuffled-row control that loses 1.28, the missed bandwidth gate,
the decode slowdown, and the unexplained 10/12 in §5 are all in the record at the same weight
as the wins.

## Where the raw records live

| path | contents |
|---|---|
| [`experiments/four_point/`](../../experiments/four_point/) | the four campaign records and every stage script (`stage1`–`stage8`, `p2_*`, `run_*.sh`) |
| [`experiments/screen/`](../../experiments/screen/) | the base-model headroom screen that selects a task |
| [`stats/`](../../stats/) | the signal maps collected by the fine-tune hooks |
| [`experiments/phase3/`](../../experiments/phase3/) | the phase-3 campaign: the headroom screen, the S2 signal maps, and S3's allocation maps and 36 eval cells with per-item hits |
| [`experiments/phase4/`](../../experiments/phase4/) | the phase-4 campaign: the text-to-SQL admission screen for both splits, per source, with the refusal broken down by cause. Panel artifacts pulled off the non-volume box while it ran live in [`s4_panel/`](../../experiments/phase4/s4_panel/) (records with per-item hits, quant manifests, decode probes, leakage scans) and [`s4_runs/`](../../experiments/phase4/s4_runs/) (the signal file and the measured expert-bank moments). The finished seven-arm panel is banked in [`results/`](../../experiments/phase4/results/) — `s4-lfm25-panel/` is the panel of record, scored with every arm on one expert arithmetic, and `s4-lfm25-panel-grouped_mm/` keeps the three records it superseded so the delta stays checkable. |
| [`docs/format-spec.md`](../format-spec.md) | the checkpoint format contract these experiments write and read |
| [`docs/legacy-audit.md`](../legacy-audit.md) | what was wrong with the supplementary code, defect by defect |
| [`decode-neutrality.md`](decode-neutrality.md) | the checkpoint's own `generation_config` reaching a "greedy" decode: how the phase-3 G4 gate found it, which campaigns it does and does not touch, and why the fix took two attempts — the first was correct on transformers 4.x and inert on the 5.x the campaign runs. Ends with what the fixed gate measures: −0.83 points, and a ±1.00 bound GSM8K is too small to resolve |
| [`runtime-parity-gap.md`](runtime-parity-gap.md) | the other half: a GSM8K stop sequence the model never wrote back, generations running on into invented problems, and the two explanations that fitted the data and were wrong (padded batching, different inputs) |
| [`docs/sglang-integration-plan.md`](../sglang-integration-plan.md) | the SGLang plugin design and its S0–S8 staging |
| [`CHANGELOG.md`](../../CHANGELOG.md) | every change, in order, with the reasoning |
