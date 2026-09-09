# Phase 2 gate report

- protocol: `v2`
- task metric: `auto`
- overall pass: `False`

## PPL gate

- real < control corpora: 1 / 1
- real < no-reader corpora: 1 / 1
- real < control seeds: 3 / 3
- real < no-reader seeds: 3 / 3
- pass: `True`

## Task gate

| Corpus | Task | metric | real | control | no-reader | candidate | severe |
|---|---|---|---:|---:|---:|---|---|
| PURE_WIKI | boolq | extracted_exact | 0.313 | 0.133 | 0.760 | False | True |
| PURE_WIKI | triviaqa | extracted_contains | 0.807 | 0.887 | 0.480 | False | True |
| PURE_WIKI | nq | extracted_contains | 0.087 | 0.093 | 0.100 | False | False |

- task gate pass: `False`
- overall pass: `False`
