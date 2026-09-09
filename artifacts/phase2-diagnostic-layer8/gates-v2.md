# Phase 2 gate report

- protocol: `v2`
- task metric: `auto`
- overall pass: `False`

## PPL gate

- real < control corpora: 3 / 3
- real < no-reader corpora: 3 / 3
- real < control seeds: 9 / 9
- real < no-reader seeds: 9 / 9
- pass: `True`

## Task gate

| Corpus | Task | metric | real | control | no-reader | candidate | severe |
|---|---|---|---:|---:|---:|---|---|
| FW_STEM | boolq | extracted_exact | 0.620 | 0.587 | 0.760 | False | True |
| FW_STEM | triviaqa | extracted_contains | 0.620 | 0.613 | 0.480 | True | False |
| FW_STEM | nq | extracted_contains | 0.073 | 0.080 | 0.100 | False | False |
| PURE_CODE | boolq | extracted_exact | 0.733 | 0.653 | 0.760 | False | False |
| PURE_CODE | triviaqa | extracted_contains | 0.527 | 0.700 | 0.480 | False | True |
| PURE_CODE | nq | extracted_contains | 0.060 | 0.080 | 0.100 | False | False |
| PURE_WIKI | boolq | extracted_exact | 0.313 | 0.133 | 0.760 | False | True |
| PURE_WIKI | triviaqa | extracted_contains | 0.807 | 0.887 | 0.480 | False | True |
| PURE_WIKI | nq | extracted_contains | 0.087 | 0.093 | 0.100 | False | False |

- task gate pass: `False`
- overall pass: `False`
