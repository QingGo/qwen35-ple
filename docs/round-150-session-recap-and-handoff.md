# Round 150: session recap and handoff

> Date: 2026-09-10 23:55 CST
> Remote: `ssh -p 19236 root@connect.nmb1.seetacloud.com`
> Purpose: self-contained handoff before context compression.  Read this first
> when resuming.

## 0. One-paragraph state

We are testing whether an external n-gram memory (Qwen3.8-Flash-Next PLE /
DeepSeek Engram-style table) can improve a frozen or lightly adapted
Qwen3.5-0.8B backbone.  The frozen-table + reader-only route has now been
falsified for **world-knowledge transfer**: standard held-out gains are
format/elicitation, shuffled-row control matches real, oracle routing is not
linearly learnable, and unfreezing the official source projections did not
produce a content effect.  The current experiment (Round 149) is the full
fine-tuning row of the backbone co-adaptation 2×2; preliminary results show
that the full-FT mixed50 recipe collapses open QA (TriviaQA/NQ ≈ 0.002), so
the recipe must be softened or replaced by LoRA before the interaction test is
meaningful.

## 1. Mission / north star

Layered goal (details in `docs/round-149-systematic-roadmap-and-tech-debt.md`):

1. **Scientific**: map the boundary conditions under which external n-gram
   memory can be read by a small model (frozen → backbone co-adaptation →
   table co-adaptation → scale/space matching).
2. **Engineering (edge)**: 0.8B-class active compute + 51.2B FP8 PLE table on
   UFS/SD/NVMe via EngramDB Store-P + 256MB–2GB RAM hot cache + async
   prefetch + lightweight backbone adaptation (LoRA / full FT) + hot-swap
   knowledge patches.
3. **Platform**: one source of truth per layer (hash/rowid: engram-peft/Qwen;
   storage: EngramDB; reader/gate: `qwen35_ple.reader`; backbone adaptation:
   HF PEFT; serving: EngramDB bundle; evaluation: locked standard suite +
   v2 scoring + gold NLL).

## 2. Timeline of this round

| Stage | What happened | State |
|---|---|---|
| 148-A | `--qa-gold-nll`, `--qa-head-mask`, gold-NLL summary, layer-hidden extraction, ridge oracle-routing probe; standard 1500 raw/chat | done, committed `c5370d1` |
| 148-B | Unfroze official source projections (`key/value/norm/conv`), table still frozen; real + control; standard 1500 raw | done, committed `1af395c` |
| Edge correction | Accepted EngramDB disk-first design: 51.2GB table on flash is capacity-feasible; IO/latency/endurance are the real constraints | documented |
| Downloads | Qwen3.5-4B (8.8G, hidden 2560) and 2B (4.3G, hidden 2048) via ModelScope-first downloader | done |
| Roadmap | Gates G0–G4, tech debt, sibling-project borrowing plan | done, committed `385997a` |
| 149 full-FT row | 0.8B × {no PLE, real, control} with full fine-tuning, 500 steps mixed50, standard 1500 raw generation + gold NLL | done; full-FT recipe collapses open QA; committed queue `47f7338` |

## 3. Key findings

### 3.1 Format vs content (Round 148-A)

Standard 1500, gold-answer NLL (lower = better):

| Protocol | no-reader | real (3 seeds) | control (3 seeds) | real − control |
|---|---:|---:|---:|---:|
| raw | 8.743 | 2.314 / 2.357 / 2.315 | 2.318 / 2.363 / 2.368 | **+0.004 / +0.006 / +0.053** |
| chat | 6.720 | 2.345 / 2.356 / 2.344 | 2.355 / 2.383 / 2.351 | **+0.010 / +0.028 / +0.007** |

* Format/elicitation effect: ~4.4–6.4 nats.
* Real PLE content effect: ~0.015–0.021 nat overall; inconsistent across seeds
  and tasks.
* Control (shuffled PLE rows) matches real → the reader learns format and
  output calibration, not table content.

Head/order ablation:

```text
raw seed0:
  full      2.314
  keep2gram 2.306  (+0.008 vs full)
  keep3gram 3.569  (−1.255 vs full)
chat seed0:
  keep2gram −0.010, keep3gram −0.035
```

2-gram heads carry most of the usable signal.

### 3.2 Oracle-routing probe (Round 148-A)

* Raw: oracle choice n=214, AUC 0.504–0.509; leave-one-task-out 0.36–0.43
  (below chance).
* Chat: n=181, AUC 0.563–0.635, but leave-one-task-out 0.44–0.50 (no
  cross-task transfer).
* Direct arm-correctness probes are much higher (0.66–0.82), i.e. hidden
  states predict difficulty/format success, not the relative PLE advantage.
* Conclusion: the oracle headroom (+0.06) is not accessible to a simple linear
  item-level gate on the layer-2 hidden state.

### 3.3 Reader capacity (Round 148-B)

* Standard 1500 raw generation: no-reader mean 0.2873; unfrozen-source real
  mean 0.3273 (BoolQ 0.782 / TriviaQA 0.156 / NQ 0.044); unfrozen control
  mean 0.3120 (0.748 / 0.146 / 0.042).
* Gold NLL: real 2.3133, control 2.3113, `real − control = −0.0019`
  (SEM 0.0094).
* Conclusion: unfreezing the official source projections does **not** create a
  content effect.  Reader capacity / source-space alignment is not the
  bottleneck.

### 3.4 Full fine-tuning row (Round 149, complete)

| Arm | BoolQ | TriviaQA | NQ | Mean | Gold NLL overall |
|---|---:|---:|---:|---:|---:|
| fullft-nople | 0.608 | 0.002 | 0.002 | 0.2040 | 5.8927 |
| fullft-real | 0.608 | 0.002 | 0.002 | 0.2040 | 5.9023 |
| fullft-control | 0.608 | 0.004 | 0.002 | 0.2047 | 5.9535 |

Paired gold NLL: `fullft-real vs fullft-control = +0.0511 ± 0.0129` overall
(TriviaQA +0.112, NQ +0.048, BoolQ −0.006), but generation is identical
(0.608 / ~0.002 / ~0.002).  `fullft-real vs fullft-nople = −0.0096 ± 0.0118`.

Interpretation: the full-FT recipe (500 steps, mixed50 QA SFT, LR 1e-4 on
0.8B) causes catastrophic open-QA collapse.  All three arms collapse to the
same BoolQ-only regime, so the interaction is **untestable with this recipe**.
The small NLL difference is inside the collapsed regime and not matched by
generation.  Need a gentler adaptation (LoRA, lower LR, fewer steps, replay,
partial FT, or early stopping) before concluding anything about backbone
co-adaptation.  This is a strong argument for doing LoRA next, not full FT.

### 3.5 Engram/V4.1/Qwen comparison

* Paper, Qwen, V4.1 all co-train table + backbone.
* Paper: iso-parameter/iso-FLOPs gains; layers 2 & 15, 4-gram not required,
  Adam 5× LR, U-shaped allocation.
* Qwen: extra-parameter regime; loss improves monotonically, downstream mixed
  (knowledge/Chinese strongest, MATH regresses past 20×); fixed-budget does not
  beat MoE-only; single layer 2 chosen partly for prefetch/systems.
* V4.1: 552B backbone + 196B Engram, 8B active prefill, layers 1/14,
  {2,3,4}-grams, 8 heads, head_dim 256, FP8 tables, no short conv,
  momentum+Sinkhorn updates; public report has no isolated Engram ablation.
* Our setting violates the co-training and allocation regimes: memory/active
  ~64×, hidden 1024 vs source 2560, single stream vs HC 4.

### 3.6 Edge feasibility correction

* The 51.2B table does not need to be RAM-resident.
* EngramDB Store-P packs e_t into one 2560B record per token; hot cache +
  async prefetch on UFS/SD/NVMe makes capacity a solved problem.
* Remaining risks: random-read latency/tail, IOPS during prefill, FTL/thermal
  behavior, RAM hot-cache budget.
* Our training/eval currently uses `/dev/shm` rows, which hides this IO cost;
  an EngramDB disk-serving benchmark is required before making edge claims.

## 4. Attempts and artifacts

Implemented this round:

```text
scripts/run_phase0.py
  --qa-gold-nll, --qa-head-mask, --unfreeze-official-source,
  --finetune-backbone
scripts/summarize_gold_nll.py
scripts/extract_layer_hidden.py
scripts/train_oracle_routing_probe.py
scripts/run_round148a.sh
scripts/run_round148b.sh
scripts/run_round149_fullft.sh
scripts/download_modelscope_model.py
scripts/download_qwen35_models.sh
```

Outputs (remote):

```text
outputs/round148a/         gold NLL, features, probe
outputs/round148b/         unfrozen-source reader eval + gold NLL
outputs/round149/          full-FT row (complete; recipe collapses open QA)
outputs/sft-unfrozen-mixed50/  trained unfrozen readers
outputs/qa-standard-eval-large-mixed50/
outputs/qa-standard-eval-chat-large/
```

Models:

```text
/root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B
/root/autodl-tmp/qwen35-ple/models/Qwen3.5-2B   (4.3G, hidden 2048)
/root/autodl-tmp/qwen35-ple/models/Qwen3.5-4B   (8.8G, hidden 2560)
```

Commits:

```text
c5370d1  round-148: gold-answer NLL, head ablation, oracle-routing probe
1af395c  round-148b: unfrozen official-source reader capacity diagnostic
385997a  round-149: systematic roadmap, tech debt, and ModelScope downloads
47f7338  round-149: full-finetune backbone x frozen PLE interaction queue
```

## 5. Pitfalls and lessons

1. **Leading-space NLL mismatch**: SFT trains answer tokens without a leading
   space; the first gold-NLL implementation added `" " + answer`, which
   penalized reader arms.  Fixed; golden convention documented.
2. **Probe input bug**: `run_round148a.sh` passed NLL-only outputs (no
   `qa_exact`) to the correctness-label loader.  Fixed to use the generation
   eval JSONs.
3. **Cross-file oracle mode naming**: standard eval stores no-reader as
   `phase1-full.json`; analyzer needed `full → no-reader` alias and
   cross-file merging.
4. **Full-FT collapse**: full fine-tuning on the mixed50 recipe destroys open
   QA; do not use it to judge PLE content.  LoRA/gentler FT first.
5. **Temporary LoRA decision**: no `peft` installed; full-FT support was
   implemented as the first co-adaptation probe.  Prefer HF PEFT for the
   actual LoRA row; if unavailable, a minimal in-place LoRA shim is acceptable
   but should be clearly temporary.
6. **Remote repo drift**: the remote repo is dirty/behind local; manual tar
   sync is used.  Keep local as source of truth; sync changed scripts before
   queue launches.
7. **SSH/background**: `nohup` over SSH can hang the tool; use tmux.  Long
   `sleep` + SSH polling can be interrupted; poll in shorter steps.
8. **Disk pressure**: data disk was at 12G free; old phase2/warmup outputs were
   cleaned to 26G, then 4B/2B downloads consumed ~14G (now ~13G free).  Old
   large checkpoints may need cleanup before the next model save.
9. **Custom 62 overestimation**: never use it as a deployable claim; standard
   1500 held-out is primary.
10. **Rows**: `/dev/shm` is lost on reboot; use
    `scripts/ensure_qwen38_rows.sh` with the venv python.  Persistent rows are
    at `/root/autodl-tmp/qwen35-ple/qwen38-rows`.

## 6. Completed / unfinished

Completed:

* 148-A and 148-B experiments, summaries, analysis scripts, tests.
* Standard 1500 v2 generation + gold NLL infrastructure.
* Oracle-routing probe (ridge, cross-validated).
* Round 149 roadmap and technical-debt inventory.
* ModelScope download of Qwen3.5-4B/2B.
* Full-FT code path and queue; no-PLE and real arms finished.

Unfinished / next:

1. Round 149 control arm (running) and final summary.
2. Commit the 148-B and Round 149 result docs (this handoff covers the text).
3. LoRA row of the 2×2 (preferred next experiment after the full-FT recipe
   problem) with standard 1500 raw/chat + gold NLL.
4. Qwen3.5-4B frozen graft (G0) once GPU free.
5. EngramDB disk-serving microbenchmark (G3).
6. Hot-row table adaptation (G2) if LoRA row fails to show interaction.
7. Small PLE Pareto (G4) and/or tiny co-trained PLE if frozen transfer is
   falsified.
8. Unified runner + batched eval + artifact manifest (engineering debt).

## 7. Future plan (gates)

```text
G0 scale/space: Qwen3.5-4B + frozen PLE
   pass -> 0.8B failure is scale/hidden mismatch

G1 backbone co-adaptation 2x2:
   {frozen, LoRA, full FT} x {no PLE, real, control}
   pass -> real > control under tuned backbone (>=2 pts or >=0.05 nat)
   current status: full-FT recipe collapsed open QA; switch to LoRA/gentler FT

G2 table co-adaptation: hot-row SparseAdam/momentum 5x LR
   pass -> small co-adapted PLE is the product path
   fail -> frozen cross-model world-knowledge transfer falsified

G3 edge serving: EngramDB Store-P on NVMe/USB SSD + hot cache + prefetch
   pass -> disk-first edge deployment is engineering-feasible

G4 small PLE Pareto: 2-gram-only / quantized / pruned / distilled / co-trained
   -> smallest memory that preserves any genuine quality gain
```

Pre-registered thresholds:

```text
content effect: real − control >=2 accuracy points or >=0.05 nat
selectivity: learned gate recovers >=30% of oracle(real/no) gap, cross-task
edge: decode >=20 tok/s NVMe, >=5 tok/s USB SSD, RAM <=4GB excluding table,
      p99 without large stalls, hot-swap in seconds
```

## 8. Handoff checklist

Environment:

```text
ssh -p 19236 root@connect.nmb1.seetacloud.com
/root/autodl-tmp/qwen35-ple/repo          # code (dirty remote; local is source of truth)
/root/autodl-tmp/qwen35-ple/venv          # Python env
/root/autodl-tmp/qwen35-ple/models        # Qwen3.5-0.8B/2B/4B + tokenizer
/root/autodl-tmp/qwen35-ple/qwen38-rows   # persistent PLE rows (48G)
/dev/shm/qwen38-rows                      # fast rows (129 files, ~48G)
/root/autodl-tmp/qwen35-ple/outputs       # experiment outputs
/root/autodl-tmp/qwen35-ple/logs          # logs
```

Running now:

```text
Round 149 finished at 2026-09-10 23:56; no experiment queue is running.
tmux session learn: unrelated old session (ignore or kill)
disk: /root/autodl-tmp ~13G free (models consumed the cleanup headroom)
GPU: RTX 4090 24GB, idle
```

First actions on resume:

1. Read `outputs/round149/gold-nll-raw.md` and
   `outputs/round149/{fullft-nople,fullft-real,fullft-control}.json`.
2. Commit the full-FT result note and this handoff (this doc is the note).
3. Implement/run the **LoRA row** (preferred) or a much gentler full-FT recipe
   (lower LR, fewer steps, replay, partial FT, early stopping) because the
   current full-FT recipe is invalid for judging PLE interaction.
4. Run the Qwen3.5-4B frozen graft (G0) on standard 1500 raw + gold NLL.
5. Start the EngramDB disk-serving benchmark (G3).
6. If LoRA still shows no interaction: hot-row table adaptation (G2), then
   small co-trained PLE (G4) or format-prior positioning.
7. Do not start a new large pivot before G0/G1/G2 produce a clear stop signal;
   update the docs and commit at every gate.
8. Disk caution: 4B model + 2B model are on disk; clean old outputs/checkpoints
   before saving new full models.  Prefer LoRA adapters and in-process
   evaluation over saving full backbones.

## 9. Reference docs

```text
docs/round-143-oracle-upper-bound-and-overnight-queue.md
docs/round-144-strategic-roadmap.md
docs/round-145-standard-heldout-chat-template-and-format-matched-experiments.md
docs/round-146-standard-heldout-final-and-cross-file-oracle.md
docs/round-147-engram-lineage-comparison.md
docs/round-148-format-vs-content-nll-and-oracle-routing-probe.md
docs/round-149-systematic-roadmap-and-tech-debt.md
```
