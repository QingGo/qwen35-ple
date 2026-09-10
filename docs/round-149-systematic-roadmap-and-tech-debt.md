# Round 149: systematic roadmap and technical debt after the frozen-graft experiments

> Date: 2026-09-10
> Scope: turn the last three sessions of evidence into a stable research and
> engineering plan.  The plan must make the next experiments cheaper, keep the
> claims honest, and integrate with the sibling projects instead of forking
> them.

## 0. TL;DR

* Our scientific question is now precise: **can a small frozen/semi-frozen
  backbone learn to use a large external n-gram memory, and under what
  conditions does that improve general performance?**
* Frozen cross-model reader-only grafting is not enough: gold-answer NLL shows
  a 4–6 nat format effect and only ~0.02 nat content effect; control (shuffled
  PLE rows) matches real.  Unfreezing the official source projections did not
  change that.
* The next decisive levers are **backbone co-adaptation (LoRA/full FT × frozen
  PLE 2×2)**, **table co-adaptation (hot-row adaptation)**, and
  **architecture/scale matching (Qwen3.5-4B, hidden 2560 = PLE source space)**.
* EngramDB makes the edge story plausible: the 51.2B table does not need to
  live in RAM; Store-P + hot cache + async prefetch on UFS/SD is the right
  deployment shape.  Capacity is solved; IO/latency/endurance must be measured.
* We need a single runner, batched eval, artifact registry, locked eval suite,
  and one source of truth per layer (hash, storage, reader/gate, adaptation,
  serving) to avoid the drift that already caused one metric bug and one queue
  bug this session.

## 1. Ultimate goal

### 1.1 Scientific goal

Map the boundary conditions under which external n-gram memory (Engram/PLE)
transfers to a small model:

| Question | Falsifiable form |
|---|---|
| Can a frozen PLE table be read by a frozen backbone? | standard held-out `real > control` ≥2 pts or ≥0.05 nat |
| If not, does backbone co-adaptation unlock it? | 2×2 interaction `real − control` under LoRA/full FT |
| If not, does table co-adaptation unlock it? | hot-row adaptation `real − control` ≥0.05 nat |
| Is the failure a scale/space mismatch? | Qwen3.5-4B (hidden 2560) frozen graft effect size |
| Does the mechanism need multi-branch/HC? | single-stream vs dual-layer/per-head gate ablation |

Both the positive and the negative outcome are deliverables, as long as the
boundary is measured.

### 1.2 Engineering goal (edge)

An edge-deployable memory-augmented small model:

```text
0.8B-class active compute
  + 51.2B FP8 PLE table on UFS/SD/NVMe via EngramDB Store-P
  + 256MB–2GB RAM hot cache + async prefetch
  + lightweight backbone adaptation (LoRA / full FT, merged at inference)
  + reader/gate trained to query the external memory
  + hot-swappable knowledge patches (row-level, no full retraining)
```

Target metrics (to be pre-registered before the next product phase):

```text
quality: standard 1500 + reasoning/long-context not worse than no-PLE baseline
decode:  ≥20 tok/s on NVMe-class storage, ≥5 tok/s on USB SSD/SD (0.8B)
RAM:     model + cache ≤ 4GB excluding the flash-resident table
p99:     no large stall beyond the model compute budget
update:  hot-swap a fact in seconds without touching backbone weights
```

### 1.3 Platform goal

A reproducible stack with one source of truth per layer:

```text
hash/rowid      -> engram-peft / Qwen official semantics + golden tests
storage         -> EngramDB Store-I/Store-P, badges, prefetch, bundles
reader/gate     -> our qwen35_ple.reader (trainable adapters)
backbone adapt  -> HF PEFT (LoRA/DoRA) or full FT, not a custom trainer fork
serving         -> EngramDB bundle + llama.cpp/vLLM integration boundary
evaluation      -> locked standard suite + v2 scoring + gold NLL
```

## 2. What this session found

### 2.1 Scientific findings

* **Format, not knowledge.**  Standard held-out gold-answer NLL:
  no-reader → SFT reader improves by 4.4–6.4 nats; `real − control` is only
  ~0.015–0.053 nat.  Control (shuffled PLE rows) matches real.
* **Oracle routing is not linearly learnable.**  Raw AUC ≈ 0.51, leave-one-task
  out below chance; chat AUC ≈ 0.63 within task but no cross-task transfer.
  Direct arm correctness is predictable (0.66–0.82), relative advantage is not.
* **Reader capacity is not the bottleneck.**  Round 148-B unfroze the official
  source projections; generation `real − control` ≈ +1.5 pts overall (mostly
  BoolQ), but gold NLL `real − control = −0.002`.  The content effect did not
  materialize.
* **2-gram heads dominate.**  Raw keep2gram ≈ full; keep3gram is much worse.
  In chat, removing 3-gram heads costs only ~0.035 nat.
* **Data scaling does not help.**  88 → 6000 SFT items: no standard open-QA
  gain; custom 62 had overestimated the effect.
* **All three Engram systems co-train table + backbone.**  Paper (iso-param /
  iso-FLOPs), Qwen (extra-parameter regime; fixed-budget no clear downstream
  gain), V4.1 (196B Engram; public report lacks isolated ablation).  Our frozen
  cross-space graft is the only setting without co-adaptation.
* **Memory/compute ratio is extreme.**  Our graft is ~64× active compute;
  paper ~1.4×, Qwen ~8.5×, V4.1 ~24.5×.  Hidden is 1024 vs source 2560.
* **Edge capacity is an IO problem, not a capacity problem.**  EngramDB
  Store-P packs e_t into one 2560B record; 51.2GB on flash is feasible with
  hot cache + prefetch; measured raw numbers are promising, but our current
  training/eval uses `/dev/shm` and hides the edge IO cost.

### 2.2 Technical debt discovered

| Area | Debt | Impact | Fix |
|---|---|---|---|
| Metrics | leading-space continuation mismatch between SFT and gold NLL | inflated/deflated arms | fixed; golden test now covers the convention |
| Metrics | generation accuracy conflates format and knowledge | past conclusions mixed | gold NLL + real-vs-control as primary content test |
| Metrics | no pre-registered thresholds; custom 62 overestimated | wrong decisions | thresholds in docs; locked standard eval |
| Evaluation | one 1500 test set, no separate validation; risk of tuning on test | slow leakage risk | locked test; hold out a validation split |
| Evaluation | no reasoning/long-context/code suite | task-signature blind spots | add small BBH/ARC/MQ-NIAH subsets |
| Statistics | 1 seed for many arms; no CIs in early tables | noisy decisions | SEM/paired tests + 3 seeds for claims |
| Cross-file analysis | mode names (`full` vs `no-reader`), per-file matrix assumptions | oracle failures | alias fix; cross-file merge helper |
| Queue bugs | probe fed NLL outputs lacking `qa_exact` | Round 148-A probe failed | fixed; queue artifact checks |
| Remote code | dirty repo, manual tar sync, remote behind local | metric bugs like the space mismatch | clean checkout or sync script + CI |
| Infra | per-arm process/model reload, batch=1, GPU ~30% | 3 min/arm, slow sweeps | batched eval core + single-process runner |
| Artifacts | no manifest/registry; outputs scattered; disk fills | manual archaeology | manifest + retention policy |
| Storage | 48GB rows duplicated in `/dev/shm` and persistent disk; lost on reboot | restore work | restore script exists; keep manifest + optional single copy policy |
| Integration | EngramDB/engram-peft versions and APIs not pinned in research flow | drift | pin versions, golden tests, adapter seam |
| Edge | training/eval uses RAM rows, not Store-P on disk | edge claim unvalidated | disk-serving microbenchmark |
| Tooling | local `.venv` lacks torch; probe tests skipped locally | CI-only coverage | optional torch extra or skip markers |
| Ops | AutoDL `shutdown` wrapper ignored time arguments | instance shutdown accident | only console scheduled shutdown; documented |

## 3. Development plan and gates

### 3.1 Gate G0 — scale/space matching (cheap, downloads in progress)

* Qwen3.5-4B (hidden 2560, exact PLE source hidden) and Qwen3.5-2B (2048).
* Same frozen table, same reader recipe, standard 1500 raw.
* Pass: `real > control` ≥2 pts or ≥0.05 nat.
* Meaning: frozen graft failed at 0.8B because of hidden/scale mismatch.

### 3.2 Gate G1 — backbone co-adaptation 2×2 (priority: edge)

|  | no PLE | PLE real | PLE control |
|---|---|---|---|
| frozen backbone | baseline | pure graft | placebo |
| LoRA | strong baseline | hybrid | placebo |
| full FT | strong baseline | hybrid | placebo |

* Same data (standard 6000 mixed50), standard 1500 raw/chat, gold NLL.
* Pass: in a tuned row, `real − control` ≥2 pts or ≥0.05 nat.
* Meaning: external memory needs backbone co-adaptation; edge hybrid is viable.
* If fail: PLE is redundant for general performance; pivot to table-side or
  small co-trained PLE.

### 3.3 Gate G2 — table co-adaptation (hot rows)

* Freeze 99%+ rows; train only rows touched by the SFT corpus (or per-row
  low-rank deltas), SparseAdam/momentum 5× LR, reader trained too.
* Pass: `real > control` on open QA.
* Meaning: content co-adaptation is the missing ingredient; small co-adapted
  PLE becomes the product path.
* If fail: frozen cross-model world-knowledge transfer is falsified.

### 3.4 Gate G3 — edge serving (EngramDB)

* 0.8B + 51.2B Store-P on NVMe and USB SSD; 256MB/1GB/4GB hot cache;
  deterministic rowid prefetch one layer ahead.
* Compare with `/dev/shm` rows for quality parity.
* Pass: decode tok/s and p99 within the edge budget; quality unchanged.
* Meaning: the disk-first edge story is real.
* If fail: need smaller table (2-gram-only, quantized, distilled) or a
  different memory hierarchy.

### 3.5 Gate G4 — small PLE / memory-quality Pareto

* 2-gram-only, quantized (FP8/4-bit), pruned/distilled table; tiny co-trained
  PLE from scratch at 1B/5B/20B.
* Goal: find the smallest memory that preserves the quality gain.
* This is the actual edge product shape.

### 3.6 Workstreams

```text
W1 Experiment core: unified runner, config registry, status.json, batched eval
W2 Evaluation: locked standard suite, gold NLL, CIs, contamination audit,
   reasoning/long-context subsets
W3 Adaptation: frozen/LoRA/full FT + reader/gate + optional hot rows
W4 Memory: EngramDB Store-P, hot cache, prefetch, edge benchmark, small PLE
W5 Integration/release: bundles/manifests, version pinning, docs, CI, ModelScope
   downloads, artifact retention
```

## 4. What to borrow from similar projects, without conflict

| Project | Borrow | Do not duplicate | Integration seam |
|---|---|---|---|
| EngramDB | Store-I/Store-P, badge layout, hot cache, prefetch, bundle manifest, disk benchmarks | another storage engine or row format | `engramdb-python`; `serving/bundle.py`, `serving/adapter.py` |
| engram-peft | canonical EngramConfig, hash/gate/short-conv semantics, PEFT wrappers | a second hash/gate implementation | pinned package + cross-repo golden tests |
| Qwen official modeling | PLE layer math, hash mapping, HC/conv, layer placement, ablations | reverse-engineering from scratch | `official_ple_snapshot.py`, parity tests |
| ortegaalfredo ngram-knowledge-injector | row-level patch/hot-swap, exact row addressing, GGUF metadata | a separate patch format | patch files + EngramDB update path |
| llama.cpp-NLTM / qwen4exp | PLE offload/runtime integration, mmap, host row indices | embedding research loops into the runtime | serving bundle / GGUF export |
| vLLM / SGLang PLE offload | async prefetch, pinned host memory, offload scheduling, metrics | forking a serving engine | plugin/offload path |
| DeepSeek Engram paper | U-shaped allocation, 4-gram/multi-head, SparseAdam 5× LR, gating math | blindly scaling to our regime | configs and training recipes |
| TN-gram / tensorized Engram | factorization and collision mitigation, quality per byte | replacing hash without golden tests | optional compressed table backend |
| HF PEFT | LoRA/DoRA configs, training loops, merging | a custom LoRA implementation | `peft` package + unified trainer |
| RAG / retrieval projects | knowledge baselines, evaluation protocols | conflating retrieval with PLE | baseline arm in the standard suite |

Conflict-avoidance rules:

1. One source of truth per layer; adapters at boundaries.
2. Golden parity tests across repos; pin `engramdb-python` and `engram-peft`
   versions in CI and remote envs.
3. Shared evaluation protocol; no repo-local metric definitions for claims.
4. One unified trainer for frozen/LoRA/full FT + reader/gate + hot rows.
5. Storage is always EngramDB; research code never invents a row format.
6. Serving is always a bundle; runtime integrations are plugins.

## 5. Stability and operations

* Remote repo: clean checkout or a single `sync_and_run.sh`; never hand-edit
  remote scripts without syncing the local commit.
* Queues: tmux, not `nohup &`; status.json per run; skip completed artifacts.
* Disk: keep 26GB+ free; retention policy for old readers; keep summaries,
  delete `partial/`, `backup/`, and superseded checkpoints.
* Rows: persistent copy + `/dev/shm` copy; restore script; manifest.
* Downloads: ModelScope first, HF fallback; record checksums in a manifest.
* Ops: AutoDL console scheduled shutdown only; never container shutdown args.
* CI: lint subset, shell syntax, tests, cross-repo golden smoke; add new
  scripts to the lint list.

## 6. Immediate next steps (Round 149+)

1. Finish Qwen3.5-4B/2B downloads (ModelScope, tmux `qwen35dl`).
2. Commit Round 148-B results and this roadmap.
3. Build the unified 2×2 runner: frozen/LoRA/full FT × no/real/control on
   0.8B, standard 1500 raw/chat + gold NLL.
4. Run the 4B scale/space test once downloads finish.
5. Implement the EngramDB disk-serving microbenchmark (Store-P, hot cache,
   prefetch) and compare with `/dev/shm`.
6. If G1 passes: build the edge hybrid bundle and measure RAM/tok/s.
   If G1 fails: run hot-row adaptation (G2); if G2 fails, pivot to small
   co-trained PLE (G4) or reframe PLE as a format prior.
7. Only then consider RL / other large pivots; keep them as baselines, not
   replacements for the current question.

## 7. Decision rules

```text
G0 pass -> frozen graft is scale/space-limited; continue edge hybrid with 4B
G1 pass -> backbone co-adaptation unlocks the memory; build edge hybrid
G2 pass -> table co-adaptation is required; build small co-adapted PLE
G1/G2 fail -> frozen cross-model world-knowledge transfer is falsified;
              pivot to small co-trained PLE or format-prior positioning
G3 pass -> edge deployment is engineering-feasible
G3 fail -> compress/distill the table or change the memory hierarchy
```

Every gate has a pre-registered effect-size threshold, a control arm, and a
stopping rule.  No gate is allowed to be rescued by adding a new module or by
changing the evaluation set.
