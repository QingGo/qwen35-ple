# Round 146: standard held-out final results, cross-file oracle, queue completion

> Remote: AutoDL RTX 4090 24GB (`connect.nmb1.seetacloud.com:19236`)
> Date: 2026-09-10 (queues finished 08:54 / 10:18 CST)
> Scope: close out the overnight queues, decide H1/H2/H3, and record the
> standard held-out 1500-item evidence with v2 answer extraction.

## 1. Queue status

Both overnight queues finished cleanly:

| Marker | Timestamp | Contents |
|---|---|---|
| `outputs/LARGE_SFT_DONE` | 2026-09-10 08:54 | large mixed50, warmup250, full raw 1500 eval (3 seeds), eval-600 oracle/report |
| `outputs/CHAT_SFT_DONE` | 2026-09-10 10:18 | chat-SFT mixed50 + warmup250, full chat 1500 eval (3 seeds), oracle/report |
| `outputs/OVERNIGHT_REPORT.md` | 2026-09-10 10:18 | regenerated report |

API errors in the queue logs (now fixed):

* `analyze_oracle_upper_bound.py` failed on the standard-eval directories with
  `KeyError: 'no-reader'` / `KeyError: 'real'`.
* Cause: Phase 0 writes the no-reader arm as `phase1-full.json`, so the mode was
  `full`, not `no-reader`; and the standard 1500 runs store modes in separate
  files, while the oracle analyzer expected a single matrix file per seed.
* Fixes in Round 146:
  * mode aliases (`full`/`none`/`baseline` → `no-reader`);
  * `_unique_counts` returns `{}` when `real` or `no-reader` is absent;
  * a standard held-out summary script
    (`scripts/summarize_standard_heldout.py`);
  * a cross-file merge for the standard oracle analysis (raw / chat matrix
    files) so the oracle numbers below are reproducible.

## 2. Standard held-out 1500 (v2 extraction metrics)

Evaluation set: `data/qa-standard/eval.jsonl` = 500 BoolQ + 500 TriviaQA
(rc.wikipedia) + 500 NQ-open.  Metrics: BoolQ `extracted_exact`, open QA
`extracted_contains`.  Reader: 6000-item `sft-large-mixed50` /
`sft-chat-mixed50`, layer 2, official Qwen source reader + trainable
`query_bridge` / `out_proj`.

### 2.1 Raw completion prompt

| Mode | Seed | BoolQ | TriviaQA | NQ | Mean |
|---|---:|---:|---:|---:|---:|
| no-reader | 0 | 0.6720 | 0.1280 | 0.0620 | 0.2873 |
| real | 0 | 0.7740 | 0.1420 | 0.0460 | 0.3207 |
| real | 1 | 0.6900 | 0.1340 | 0.0360 | 0.2867 |
| real | 2 | 0.7520 | 0.1620 | 0.0400 | 0.3180 |
| **real mean** | – | **0.7387** | **0.1460** | **0.0407** | **0.3084 ± 0.0154** |
| real − no-reader | – | +0.0667 | +0.0180 | −0.0213 | **+0.0211** |

### 2.2 Chat template

| Mode | Seed | BoolQ | TriviaQA | NQ | Mean |
|---|---:|---:|---:|---:|---:|
| no-reader | 0 | 0.7220 | 0.1800 | 0.0680 | 0.3233 |
| real | 0 | 0.6980 | 0.1440 | 0.0420 | 0.2947 |
| real | 1 | 0.7560 | 0.1440 | 0.0420 | 0.3140 |
| real | 2 | 0.7620 | 0.1560 | 0.0360 | 0.3180 |
| **real mean** | – | **0.7387** | **0.1480** | **0.0400** | **0.3089 ± 0.0102** |
| real − no-reader | – | +0.0167 | −0.0320 | −0.0280 | **−0.0144** |

### 2.3 300-item subset (directly comparable with the chat ablation)

| Train \\ Eval | Raw prompt | Chat template |
|---|---:|---:|
| no reader | 0.2800 | 0.3333 |
| raw-trained reader | 0.3133 ± 0.0136 | 0.3167 (old 88-item SFT ablation) |
| chat-trained reader | not run | 0.3044 ± 0.0129 |

The 1500-item result says the same thing as the 300-item ablation, with less
noise:

* raw eval: reader +0.021 mean, but driven by BoolQ format repair (+0.067) and
  with a 0.034 seed spread;
* chat eval: reader −0.014 mean, with TriviaQA −0.032 and NQ −0.028; chat
  no-reader already captures the format/elicitation gain;
* **training the reader on chat data does not restore a PLE gain** (C < chat
  no-reader).  H1 (“reader was simply not trained on chat-formatted data”) is
  not supported by this counterfactual.
* H2 is partly supported: the raw-reader gain is largely format/elicitation,
  not new knowledge.
* H3 is refined below by the standard cross-file oracle.

## 3. Standard 1500 cross-file oracle

The standard eval stores no-reader in `phase1-full.json` and real in
`phase1-real-seed*.json`.  For the oracle we duplicate the no-reader arm under
each real seed and run the analyzer on the merged matrix.  Control was not run
for the standard 1500, so only `oracle(real/no)` is available.

### 3.1 Raw completion

| Policy | Mean | Per-seed |
|---|---:|---|
| always real | 0.3084 | 0.3207 / 0.2867 / 0.3180 |
| always no-reader | 0.2873 | – |
| oracle(real/no) | **0.3689** | 0.3813 / 0.3440 / 0.3813 |

Unique-correct counts (summed over 3 seeds; 1500 items each):

| Task | both real/no | real only | no-reader only | all wrong |
|---|---:|---:|---:|---:|
| BoolQ | 863 | **245** | 145 | 247 |
| TriviaQA | 119 | **100** | 73 | 1208 |
| NQ | 39 | 22 | **54** | 1385 |

### 3.2 Chat template

| Policy | Mean | Per-seed |
|---|---:|---|
| always real | 0.3089 | 0.2947 / 0.3140 / 0.3180 |
| always no-reader | 0.3233 | – |
| oracle(real/no) | **0.3822** | 0.3947 / 0.3727 / 0.3793 |

Unique-correct counts (summed over 3 seeds):

| Task | both real/no | real only | no-reader only | all wrong |
|---|---:|---:|---:|---:|
| BoolQ | 910 | **198** | 173 | 219 |
| TriviaQA | 173 | 49 | **97** | 1181 |
| NQ | 42 | 18 | **60** | 1380 |

Interpretation:

* There is real routing headroom on the standard 1500 (raw +0.06, chat +0.06
  over the best fixed policy), so the reader is **not** content-free.
* The pattern is task-specific: BoolQ has more real-only wins (reader helps
  yes/no format); chat TriviaQA/NQ have more no-reader-only wins (always-on
  injection hurts open QA under chat).
* A deployable gain therefore needs **selectivity** (per-token/per-task
  gating), not a stronger always-open reader.  The current gate saturates open
  after SFT (mean 0.77–0.98 in the earlier gate stats), which fits this failure
  mode.
* The oracle is label-seeing and optimistic; the actual capturable gain is
  smaller and needs a learned gate.

## 4. Decision / next experiments

1. **Gate selectivity first.**  Train the existing gate, but add cheap
   selectivity capacity/objectives: per-dim query/key weights (V4.1 style),
   per-branch/per-head gate bias, token-type mask for special/role tokens,
   and/or entropy/sparsity regularization.  Do not add a new router.
2. **Dual-layer injection.**  Keep the frozen PLE table and train a second
   reader at a middle layer (DeepSeek Engram uses layers 2 and 15; V4.1 uses
   layers 1 and 14).  This is the most direct architectural borrowing from the
   DeepSeek line.
3. **Head/order ablation.**  Use only 2-gram or only 3-gram heads, or learn
   per-head scales, to see which heads carry the BoolQ vs TriviaQA/NQ signal.
4. **Table adaptation upper bound (diagnostic).**  Freeze almost all rows and
   train only the hot rows touched by the SFT corpus (SparseAdam / momentum,
   5× LR as in the Engram recipe).  This distinguishes content mismatch from
   reader alignment.
5. **Memory/compute allocation.**  The frozen Qwen table is 51.2B parameters
   grafted onto a 0.8B backbone (64× the active compute).  Audit coverage and
   try a capped/subsampled PLE (fewer heads or hot rows) to test whether a
   smaller, better-utilized memory is stronger.

## 5. Artifacts

* `scripts/summarize_standard_heldout.py`
* `scripts/analyze_oracle_upper_bound.py` (mode aliases + missing-mode guard)
* `outputs/standard-heldout-summary.md`
* `outputs/standard-heldout-summary-300.md`
* `outputs/oracle-analysis/qa-standard-raw-matrix.{json,md}`
* `outputs/oracle-analysis/qa-standard-chat-matrix.{json,md}`
* `outputs/OVERNIGHT_REPORT.md`
