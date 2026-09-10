# Round 148: format vs content — gold-answer NLL and the oracle-routing probe

> Date: 2026-09-10
> Remote: AutoDL RTX 4090, tmux session `round148a`
> Goal: decide whether reader-only training injects world knowledge, or
> whether the standard-held-out gains are format/elicitation.

## 1. Design

Two format-insensitive diagnostics on the standard 1500-item held-out set
(BoolQ / TriviaQA rc.wikipedia / NQ-open, 500 each):

1. **Gold-answer NLL** (`--qa-gold-nll`): teacher-forced NLL of the gold
   answer tokenization after the exact prompt protocol used for generation.
   * `real`: real frozen PLE rows;
   * `control`: shuffled PLE rows (same reader, same training recipe);
   * `no-reader`: no PLE injection;
   * head/order ablations: keep only 2-gram heads or only 3-gram heads.
   The key comparison is **real vs control** — both arms receive the same
   reader/Q-SFT format adaptation, so the difference isolates PLE content.

2. **Oracle-routing probe**: layer-2 hidden states from the frozen backbone
   (the same hidden state the reader uses as its query) are used to predict,
   item by item, which arm the label-seeing oracle would choose
   (`real` correct & `no-reader` wrong vs the converse).  Features: last-token
   hidden, mean-pooled hidden, and their concatenation.  Model: closed-form
   ridge probe (fast; replaced the initial slow logistic-regression version).
   Cross-validation is stratified by task; leave-one-task-out tests transfer.

## 2. Gold-answer NLL

### 2.1 Raw completion prompt

| Run | BoolQ | TriviaQA | NQ | Overall |
|---|---:|---:|---:|---:|
| no-reader | 12.3765 | 7.3230 | 6.5292 | 8.7429 |
| real seed0 | 0.5304 | 2.9765 | 3.4357 | 2.3142 |
| real seed1 | 0.6124 | 3.0316 | 3.4281 | 2.3574 |
| real seed2 | 0.5854 | 3.0442 | 3.3159 | 2.3152 |
| control seed0 | 0.5333 | 3.0033 | 3.4183 | 2.3183 |
| control seed1 | 0.5687 | 3.0732 | 3.4472 | 2.3630 |
| control seed2 | 0.6119 | 3.0657 | 3.4277 | 2.3684 |
| keep2gram (seed0) | 0.5153 | 2.9678 | 3.4354 | 2.3062 |
| keep3gram (seed0) | 2.0960 | 4.7048 | 3.9064 | 3.5690 |

Paired **real − control** (positive = real lower NLL):

| Seed | BoolQ | TriviaQA | NQ | Overall | SEM |
|---|---:|---:|---:|---:|---:|
| 0 | +0.0029 | +0.0267 | −0.0174 | **+0.0040** | 0.0091 |
| 1 | −0.0437 | +0.0415 | +0.0190 | **+0.0056** | 0.0110 |
| 2 | +0.0265 | +0.0215 | +0.1118 | **+0.0533** | 0.0111 |
| mean | −0.005 | +0.030 | +0.038 | **+0.021** | – |

Format effect (no-reader → real): **~6.43 nats**.
Content effect (real → control): **~0.00–0.05 nats**, inconsistent across seeds.

### 2.2 Chat template

| Run | BoolQ | TriviaQA | NQ | Overall |
|---|---:|---:|---:|---:|
| no-reader | 9.1666 | 5.4706 | 5.5242 | 6.7205 |
| real seed0 | 0.6158 | 3.0831 | 3.3347 | 2.3445 |
| real seed1 | 0.5382 | 3.1039 | 3.4256 | 2.3559 |
| real seed2 | 0.5374 | 3.0965 | 3.3994 | 2.3444 |
| control seed0 | 0.6248 | 3.1213 | 3.3177 | 2.3546 |
| control seed1 | 0.5710 | 3.1724 | 3.4069 | 2.3834 |
| control seed2 | 0.6030 | 3.0866 | 3.3639 | 2.3512 |
| keep2gram (seed0) | 0.6481 | 3.0751 | 3.3402 | 2.3545 |
| keep3gram (seed0) | 0.6949 | 3.0573 | 3.3865 | 2.3796 |

Paired **real − control** (positive = real lower NLL):

| Seed | BoolQ | TriviaQA | NQ | Overall | SEM |
|---|---:|---:|---:|---:|---:|
| 0 | +0.0089 | +0.0382 | −0.0170 | **+0.0101** | 0.0078 |
| 1 | +0.0328 | +0.0685 | −0.0187 | **+0.0275** | 0.0100 |
| 2 | +0.0657 | −0.0099 | −0.0356 | **+0.0067** | 0.0093 |
| mean | +0.036 | +0.032 | −0.024 | **+0.015** | – |

Format effect (no-reader → real): **~4.37 nats**.
Content effect: **~0.01–0.03 nats**, again inconsistent by task.

### 2.3 Head/order ablation

Raw seed0:

| Variant | BoolQ | TriviaQA | NQ | Overall | Δ vs full | SEM |
|---|---:|---:|---:|---:|---:|---:|
| full | 0.5304 | 2.9765 | 3.4357 | 2.3142 | – | – |
| keep2gram | 0.5153 | 2.9678 | 3.4354 | 2.3062 | **+0.0081** | 0.0033 |
| keep3gram | 2.0960 | 4.7048 | 3.9064 | 3.5690 | **−1.2548** | 0.0311 |

Chat seed0:

| Variant | Overall | Δ vs full | SEM |
|---|---:|---:|---:|
| full | 2.3445 | – | – |
| keep2gram | 2.3545 | **−0.0099** | 0.0039 |
| keep3gram | 2.3796 | **−0.0350** | 0.0088 |

Interpretation: under this reader, **2-gram heads carry almost all of the
gold-answer likelihood signal**; 3-gram heads are neutral-to-slightly-harmful
in raw and only marginally useful in chat (~0.035 nats, still tiny compared
with the 4–6 nat format effect).

## 3. Oracle-routing probe

Raw features (layer-3 hidden states = output of decoder block 2, the reader's
query):

| Feature | Oracle n | AUC | Acc | Majority |
|---|---:|---:|---:|---:|
| last | 214 | 0.5089 | 0.5561 | 0.6028 |
| mean | 214 | 0.5044 | 0.5935 | 0.6028 |
| last+mean | 214 | 0.5064 | 0.5981 | 0.6028 |

Leave-one-task-out oracle AUC: BoolQ 0.4307, TriviaQA 0.4020, NQ 0.3571
(below chance).  Direct correctness probes are much higher (reader-correct
AUC 0.8150, no-reader-correct 0.7390), i.e. hidden states predict each arm's
difficulty/format success, but **not the relative advantage**.

Chat:

| Feature | Oracle n | AUC | Acc | Majority |
|---|---:|---:|---:|---:|
| last | 181 | 0.5631 | 0.4972 | 0.5525 |
| mean | 181 | 0.6300 | 0.5304 | 0.5525 |
| last+mean | 181 | 0.6353 | 0.5414 | 0.5525 |

Leave-one-task-out oracle AUC: BoolQ 0.4374, TriviaQA 0.4965, NQ 0.4364
(no transfer).  Direct correctness AUC: reader 0.8206, no-reader 0.7772.

Interpretation:

* In raw, a linear item-level "use PLE or not" signal is **not decodable**
  from the layer-2 hidden state (AUC ≈ 0.51).
* In chat there is a weak within-task signal (AUC ≈ 0.63 for mean-pooled
  features), but it does not transfer across tasks (leave-one-task-out ≈
  0.44–0.50).  This is task-specific difficulty, not a general routing rule.
* The oracle headroom measured earlier (+0.06) is therefore **not accessible
  to a simple linear gate on this hidden state**.

## 4. Conclusions

1. **Reader-only pure PLE grafting does not inject world knowledge.**
   The standard held-out generation gain is format/elicitation; gold-answer
   NLL shows a 4–6 nat format effect and only a ~0.02 nat content effect.
2. **Control (shuffled PLE rows) matches real on gold-answer NLL**, confirming
   that the reader learns format/calibration rather than table content.
3. **The oracle headroom is not learnable by a simple gate** on the current
   layer-2 query; chat shows only weak, task-specific signal.
4. **2-gram heads carry most of the usable signal**; 3-gram heads are largely
   neutral or harmful for this reader.  This points to a practical recipe
   (2-gram-only) but also reinforces that the signal is format-like.
5. The remaining hypotheses are:
   * **reader capacity** — the official source projections (`key_proj`,
     `value_proj`, norms, conv) are frozen; unfreezing them is the last
     reader-only lever;
   * **table content/co-adaptation** — hot-row adaptation is the decisive
     content upper bound;
   * **allocation** — 51.2B frozen memory on 0.8B active compute is an
     extreme point of the Engram scaling law.

## 5. Next

* Round 148-B: train `OfficialSourceQwenReader` with the official source
  projections **unfrozen** on the same 6000-item mixed50 recipe, 1 real +
  1 control seed first; evaluate standard 1500 raw + chat and real-vs-control
  gold NLL.  If the content effect does not grow, reader capacity is not the
  bottleneck.
* If reader capacity fails, run the hot-row table-adaptation upper bound
  (SparseAdam / momentum, 5× LR, freeze all but hot rows).  If that also
  fails, the pure frozen-table route is falsified for world-knowledge gains.
* Infra: tmux queue; ridge probe (40 s instead of >5 min); queue script probe
  inputs fixed to use the generation eval JSONs (which carry `qa_exact`),
  not the NLL-only outputs.

## 6. Artifacts

* `scripts/run_round148a.sh`
* `scripts/summarize_gold_nll.py`
* `scripts/extract_layer_hidden.py`
* `scripts/train_oracle_routing_probe.py`
* `outputs/round148a/gold-nll-raw.md`
* `outputs/round148a/gold-nll-chat.md`
* `outputs/round148a/probe-raw.md`
* `outputs/round148a/probe-chat.md`
* `outputs/round148a/features-raw.npz`
* `outputs/round148a/features-chat.npz`
