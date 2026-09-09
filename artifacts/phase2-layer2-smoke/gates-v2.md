# Phase 2 gate report

- protocol: `v2`
- task metric: `auto`
- overall pass: `False`

## PPL gate

- real < control corpora: 1 / 1
- real < no-reader corpora: 1 / 1
- real < control seeds: 1 / 1
- real < no-reader seeds: 1 / 1
- pass: `True`

## Task gate

| Corpus | Task | metric | real | control | no-reader | candidate | severe |
|---|---|---|---:|---:|---:|---|---|
| PURE_WIKI | boolq | extracted_exact | 0.240 | 0.280 | 0.760 | False | True |
| PURE_WIKI | triviaqa | extracted_contains | 0.860 | 0.700 | 0.480 | True | False |
| PURE_WIKI | nq | extracted_contains | 0.100 | 0.080 | 0.100 | False | False |

- task gate pass: `False`
- overall pass: `False`
