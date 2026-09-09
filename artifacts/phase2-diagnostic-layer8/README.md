# Phase 2 layer8 diagnostic artifacts

Run: 2026-09-09, AutoDL RTX 4090

Config:

```text
corpora : PURE_WIKI PURE_CODE FW_STEM
modes   : real control no-reader
seeds   : 0 1 2
steps   : 500
layer   : 8  (incorrect: official PLE layer is 2)
QA      : instruction-style prompt, 150 items, 96 new tokens
```

Key result:

```text
PPL gate: PASS
Task gate: FAIL
BoolQ severe regression on PURE_WIKI / FW_STEM
PURE_CODE TriviaQA below control
```

Files:

- `phase1-*.json.gz` if present; raw JSONs are local-only.
- `summary-v2.json` / `summary-v2.md`
- `gates-v2.json` / `gates-v2.md`
- `sha256.txt`
