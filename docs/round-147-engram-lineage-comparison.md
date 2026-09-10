# Round 147: DeepSeek Engram / Qwen3.8-Next PLE / DeepSeek-V4.1-Flash vs our graft

> Date: 2026-09-10
> Trigger: DeepSeek-V4.1-Flash was open-sourced today
> (`deepseek-ai/DeepSeek-V4.1-Flash`, HF `createdAt=2026-09-10T02:17:58Z`);
> it contains an Engram module.  This note compares the four implementations
> and extracts the ideas that are actually relevant to the pure-PLE graft.
>
> Sources: official V4.1 config + `inference/engram.py` + tech report, the
> DeepSeek Engram paper/repo, the Qwen3.8-Flash-Next config + official
> `transformers` `qwen4_exp` code, our repo, and two community references
> (`ortegaalfredo/ngram-knowledge-injector`, `QingGo/engram-peft`).

## 1. Timeline

| Date | Event |
|---|---|
| 2022 | N-Grammer: early hash n-gram embedding work cited by the Engram paper |
| 2025 | SCONE, Gemma 3n/4 PLE (Per-Layer Embeddings, token-keyed, every layer) |
| **2026-01-12** | DeepSeek Engram paper: *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs* (arXiv:2601.07372; v2 2026-07-12; ACL 2026) |
| **2026-08-24/26** | Qwen3.8-Flash-Next (`model_type: qwen4_exp`) open-sourced: first production-scale 51.2B PLE table |
| **2026-09-08…10** | our pure-PLE graft experiments, queues, and standard held-out evaluation |
| **2026-09-10 (today)** | DeepSeek-V4.1-Flash open-sourced: 552B backbone + 196B Engram parameters |

The original paper is the source of the design; Qwen is the first large
open-weight production deployment; V4.1 is the largest Engram deployment so
far.  Our project is a *graft* of the Qwen table onto a different, much smaller
frozen backbone, not a reproduction of any of the three.

## 2. Four implementations at a glance

| | DeepSeek Engram paper (Engram-27B) | Qwen3.8-Flash-Next PLE | DeepSeek-V4.1-Flash Engram | our pure-PLE graft |
|---|---|---|---|---|
| Backbone / active compute | 26.7B MoE, iso-FLOPs with Dense-4B (~4B active) | ~125B MoE + 4B MTP (~180B total, 6B active) | 552B MoE (~748B total with Engram), 8B active prefill / 16B decode | Qwen3.5-0.8B, frozen, 0.8B active |
| Memory params | 5.7B embedding module (21% of total) | 51.2B table | 196B (2 modules, ~36% of backbone) | 51.2B frozen table |
| Memory / active compute | ~1.4× | ~8.5× | ~24.5× | **~64×** |
| N-gram orders | {2,3} | {2,3} | {2,3,4} | {2,3} (frozen Qwen table) |
| Heads | 8 per order | 8 per order (16 rows/token) | 8 per order (24 rows/token) | 16 rows/token |
| Head dim / e_t width | d_mem 1280 (paper config) | 160 × 16 = 2560 | 256 × 24 = 6144 per module | 160 × 16 = 2560 |
| Injection layers | 2 and 15 (1-indexed) | 2 (0-indexed, single layer) | 1 and 14 (0-indexed) | 2 (0-indexed, single layer) |
| Gate | RMSNorm dot product, signed sqrt + sigmoid; branch-specific keys, shared value | same, per HC branch (hc=4) | same + **learnable per-dim q/k weights**; FP8 keys | official Qwen gate, no learnable gate bias; custom reader has `gate_bias`/`gate_override` |
| Short causal conv | kernel 4, dilation = max n-gram order, zero-init | kernel 4, dilation 3 | **removed** (performance vs complexity) | reused from official Qwen reader |
| Table training | Adam, 5× LR, no weight decay; conv zero-init | trained jointly with backbone (Adam, weight decay disabled) | momentum update + Sinkhorn balancing, 5× LR, FP8 tables | **frozen, no gradients** |
| Systems | deterministic addressing, host-memory offload | host RAM / NVMe, prefetch overlaps layer-1 compute; `ple_layer_ids=[2]` chosen for systems | RDMA host prefetch, tables sharded across rank groups; layers 1/14 balance pipeline memory | rows in `/dev/shm`, Store-I 128 shards, lazy e_t cache |
| Reported outcome | MMLU +3.4, CMMLU +4.0, BBH +5.0, ARC-C +3.7, HumanEval +3.0, MATH +2.4, MQ-NIAH 84.2→97.0 | loss improves; knowledge/Chinese benchmarks improve, MATH peaks then regresses; fixed-budget table did not beat MoE-only | new model, Engram is one component of the long-context/KV-compression story | raw BoolQ format repair; no standard knowledge gain; chat reader slightly hurts vs no-reader |

Numbers for the paper are the paper's; Qwen/V4.1 numbers are from the official
config/tech report.  The Qwen ablation details (placement, fixed-budget,
token-normalization experiments) are also summarized in the community
`ortegaalfredo/ngram-knowledge-injector` research report.

## 3. The common core

All four implementations share the same primitive:

1. **Deterministic n-gram hashing.**  For token `x_t`, build suffix n-grams over
   the last `n-1` compressed/normalized token IDs.  Each (n-gram order, head)
   owns a prime-sized bucket range; row id = `hash(gram) % prime + offset`.
   Hash = multiplicative-XOR with odd multipliers.  The Engram paper compresses
   the vocab (23% reduction); Qwen's official code resets the n-gram history at
   EOS (`_shift_right_ignore_eos`); V4.1 compresses 129,280 → 99,092 tokens and
   treats image spans as `DEAD`.
2. **Concatenate retrieved head vectors** into `e_t` (Qwen 2560, V4.1 6144 per
   module).
3. **Context-aware gate.**  The current hidden state is the query, `e_t` is the
   key/value source:
   `alpha = sigmoid(signed_sqrt(RMSNorm(h) · RMSNorm(W_k e_t) / sqrt(d)))`.
   The gate is intended to close when the memory contradicts the context.
4. **Gated value + short depthwise causal conv + SiLU + residual.**
5. **Multi-branch sharing.**  One shared value projection `W_v`, multiple
   branch-specific key projections `W_k^(m)`.
6. **Deterministic addressing** enables host-memory / NVMe offload with
   prefetch overlapping earlier-layer compute.

Qwen3.8-Next's PLE is essentially the Engram design with a single layer,
`{2,3}`-grams, `d_mem=2560`, and a 51.2B table; V4.1 is the design scaled to
two layers, `{2,3,4}`-grams, 24 heads, 6144-dim `e_t`, FP8 tables, and no short
conv.

## 4. Differences that actually matter for us

### 4.1 Co-training vs frozen-table grafting (the biggest one)

The paper, Qwen, and V4.1 all **train the memory table together with the
backbone** (paper: Adam 5× LR no weight decay; Qwen: Adam no weight decay;
V4.1: momentum + Sinkhorn, 5× LR, FP8).  Our setting is fundamentally
different:

```
Qwen3.8 backbone + Qwen PLE table   (co-trained)  ->  Qwen reader/gate
Qwen3.5-0.8B backbone (frozen) + frozen Qwen PLE table
    -> trainable cross-space reader (15.2M params) only
```

The table rows live in Qwen's residual space; our reader must translate
Qwen's source space (2560, 4 HC branches) into Qwen3.5's target space (1024,
single residual stream).  Neither paper addresses this; it is the core reason
our graft is hard.

### 4.2 Memory/compute ratio

Engram-27B: 5.7B memory / ~4B active ≈ 1.4×.
Qwen3.8-Flash-Next: 51.2B / 6B active ≈ 8.5×.
V4.1-Flash: 196B / 8B active ≈ 24.5×.
Our graft: 51.2B / 0.8B active ≈ **64×**.

The paper's U-shaped sparsity-allocation law says there is an optimal
compute/memory balance; our ratio is far outside all three systems.  Even with
perfect row lookup, a 0.8B model has to integrate a memory whose intended
compute is 6–8B active.  Qwen's own fixed-budget ablation (table reduces loss
but does not beat MoE-only on downstream benchmarks) is a warning that table
capacity alone is not enough.

### 4.3 Cross-space bridge and branch approximation

* Our `OfficialSourceQwenReader` keeps the official `key_proj`, `value_proj`,
  `norm_key`, `norm_query`, `norm_conv`, `conv1d` frozen, and trains
  `query_bridge` (1024 → 1024 → 10240) and `out_proj` (2560 → 1024 → 1024,
  zero-init last layer).  No trained model in the DeepSeek/Qwen line needs
  this bridge.
* Qwen/V4.1 gate four hyper-connection branches with per-branch keys and a
  shared value.  Our target backbone has a single residual stream; we
  instantiate the four source branches internally and sum them before
  `out_proj`, which is a lossy approximation.

### 4.4 Gate selectivity

The paper's case study shows the gate is selective (it correlates with semantic
pattern matching).  Our SFT reader's gate saturates open (mean 0.77–0.98), so
always-on injection adds noise on TriviaQA/NQ under chat.  V4.1 adds learnable
per-dimension `q_weight`/`k_weight` on top of the normalized dot product; the
paper/V4.1 use per-branch keys.  Those are exactly the missing degrees of
freedom for our graft.

### 4.5 N-gram coverage

Qwen's table has `{2,3}`-grams and 16 heads of 160 dims.  V4.1 adds
`{2,3,4}`-grams, 24 heads, 256 dims.  If entity/relation knowledge lives
substantially in 4-grams, our frozen Qwen table simply does not contain it.
The paper's 27B result also used `{2,3}`-grams, so this is not fatal by itself,
but it is a coverage ceiling worth auditing.

## 5. What we can borrow (prioritized)

### P0 — cheap, high expected information

1. **Dual-layer injection (layers 2 + middle).**
   DeepSeek: layers 2 and 15 (1-indexed); V4.1: layers 1 and 14 (0-indexed);
   the paper's mechanism analysis says the benefit comes from early intervention
   + rich late-stage context for gating.  Keep the PLE table frozen, train a
   second reader at a middle layer of Qwen3.5-0.8B (e.g. 12–16), evaluate on
   the standard 1500 raw/chat.  Qwen chose a single layer for *systems*
   reasons plus a co-trained table; our graft is not that setting.

2. **Gate selectivity.**
   * add V4.1-style learnable per-dim `q_weight`/`k_weight` (init 1, used as a
     product) to the official gate;
   * add a per-branch gate bias and/or a per-head (16) value scale;
   * keep `--gate-reg-weight` and add a token-type mask: gate=0 on
     `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, padding; optionally mask the
     system/role prefix.
   * Track gate mean/open-fraction on BoolQ vs TriviaQA/NQ; the target is a
     gate that closes on chat open-QA tokens and opens on BoolQ-style tokens.

3. **Head/order ablation.**
   Run only 2-gram heads, only 3-gram heads, and per-head zeroing.  Combine
   with the standard-oracle unique-correct items to see which heads produce the
   `real_only` wins.  This is a direct coverage/collision diagnostic.

### P1 — diagnostic, may change the thesis

4. **Hot-row table adaptation upper bound.**
   Freeze all but the rows touched by the training corpus (or top-k hot rows)
   and update them with SparseAdam / momentum at 5× LR, as in the paper
   (V4.1 uses momentum + Sinkhorn at 5× LR).  If this closes the oracle gap,
   the bottleneck is table↔backbone co-adaptation, not reader alignment.  This
   is explicitly **not** pure grafting; it is the diagnostic that tells us
   whether pure grafting can ever work.

5. **Memory cap / allocation experiment.**
   Use fewer heads, only high-frequency rows, or a distilled smaller table to
   move the memory/compute ratio toward the paper's range.  A 51.2B table on a
   0.8B backbone is 64× active compute; a smaller PLE may be better utilized
   and less noisy.  Also audit collision/coverage of the eval n-grams (which
   rows are shared, how often they occur).

### P2 — systems / training hygiene (not the current bottleneck)

6. **Stateful n-gram cache** across prefill/decode (V4.1 `NgramHashState`);
   our `ple_hash.py` is stateless per call.  Useful for long generations and
   for avoiding O(T) rehash.
7. **Store-P packed `e_t`** (one 2560-byte read per unique n-gram key instead of
   16 scattered row reads) via EngramDB; useful for faster training/feature
   generation, not for quality.
8. **FP8 + 5× LR / sparse optimizer** if we ever update table rows.

## 6. What not to borrow

* **Remove the short conv** (V4.1): V4.1's table is co-trained, FP8, and its
  inference stack is different; the paper/Qwen keep the conv and zero-init it.
* **HC multi-branch expansion**: our target backbone is single-stream and
  frozen; emulate with multi-layer injection instead.
* **Tokenizer compression**: Qwen reports no consistent gains, and our table is
  already keyed to Qwen's tokenizer/compression; not actionable.
* **Huge table scaling**: more frozen memory cannot fix a 64× allocation
  mismatch.
* **A second router**: per the project preference, improve the existing gate
  instead.

## 7. Decision rules

* If dual-layer + selective gate recovers `real > chat-no-reader` on the
  standard 1500, continue pure grafting and study layer/head placement.
* If it does not, but hot-row adaptation does, the frozen table is the
  bottleneck; pivot to a “tiny co-adapted PLE” rather than pure grafting.
* If neither works, stop scaling SFT data and write up the negative result:
  standard held-out evidence says the raw gain is format repair and the
  remaining oracle headroom is routing/selectivity, not reader capacity.

## 8. References

* DeepSeek-V4.1-Flash: <https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash>
  (`config.json`, `inference/engram.py`, `DeepSeek_V41_Tech_Report.pdf`).
  Key facts: `engram_layer_ids=[1,14]`, `engram_max_ngram_size=4`,
  `engram_num_embeddings=[384006168, 384016682]`,
  `engram_head_dim=256`, `engram_n_heads=8`, `engram_compressed_vocab_size=99092`,
  FP8 tables, 196B Engram parameters.
* DeepSeek Engram paper + code: <https://github.com/deepseek-ai/Engram>,
  arXiv:2601.07372; 27B experiment: 5.7B embedding module, layers 2 and 15,
  max n-gram 3, 8 heads, dim 1280, Adam 5× LR, no weight decay.
* Qwen3.8-Flash-Next: <https://huggingface.co/Qwen/Qwen3.8-Flash-Next>
  (`ngram_size=3`, `heads_per_ngram=8`, `ngram_vocab_size_base=20M`,
  `ple_embed_dim=2560`, `ple_layer_ids=[2]`, `split_ngram_parts=128`,
  `ple_conv_kernel_size=4`; 320M rows × 160 = 51.2B table).
* Qwen official modeling code: `transformers/models/qwen4_exp/modeling_qwen4_exp.py`
  (installed 5.6.0); our frozen extraction is
  `src/qwen35_ple/official_ple_snapshot.py`.
* Community research report with Qwen config/ablations/lineage:
  <https://github.com/ortegaalfredo/ngram-knowledge-injector/blob/main/docs/qwen3-next-ngram-research.md>.
* Paper-aligned PEFT implementation and cross-model notes:
  <https://github.com/QingGo/engram-peft/blob/master/docs/paper_alignment.md>.
* Disk-first PLE/Engram store: <https://github.com/QingGo/EngramDB>.
* Tensorized Engram / hash-collision critique: arXiv:2606.08347.
