# Phase 2 layer2 smoke artifacts

Run: 2026-09-09, AutoDL RTX 4090

Config:

```text
corpus  : PURE_WIKI
seed    : 0
modes   : real control no-reader
steps   : 500
layer   : 2  (official Qwen3.8 PLE layer)
QA      : instruction-style prompt, 150 items, 96 new tokens
```

Key result:

```text
PPL: real 22.924 < control 24.810 < no-reader 37.590

TriviaQA:
  real 0.860 > control 0.700 > no-reader 0.480

BoolQ:
  real 0.240 < control 0.280 << no-reader 0.760

Overall gate: FAIL (BoolQ severe regression)
```

Files:

- `phase1-PURE_WIKI.json` raw JSON is local-only.
- `summary-v2.json` / `summary-v2.md`
- `gates-v2.json` / `gates-v2.md`
