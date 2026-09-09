# Phase 2 gate report

- protocol: `v2`
- task metric: `extracted_exact`
- overall pass: `False`

## PPL gate

- real < control corpora: 5 / 6
- real < no-reader corpora: 6 / 6
- real < control seeds: 15 / 18
- real < no-reader seeds: 18 / 18
- pass: `True`

## Task gate

| Corpus | Task | real | control | no-reader | candidate | severe |
|---|---|---:|---:|---:|---|---|
| FW_CODE | boolq | 0.693 | 0.680 | 0.220 | True | False |
| FW_CODE | triviaqa | 0.020 | 0.020 | 0.020 | False | False |
| FW_CODE | nq | 0.000 | 0.000 | 0.000 | False | False |
| FW_STEM | boolq | 0.640 | 0.647 | 0.220 | False | False |
| FW_STEM | triviaqa | 0.040 | 0.067 | 0.020 | False | False |
| FW_STEM | nq | 0.000 | 0.000 | 0.000 | False | False |
| PURE_CODE | boolq | 0.613 | 0.533 | 0.220 | True | False |
| PURE_CODE | triviaqa | 0.000 | 0.007 | 0.020 | False | False |
| PURE_CODE | nq | 0.000 | 0.000 | 0.000 | False | False |
| PURE_FINEWEB | boolq | 0.373 | 0.487 | 0.220 | False | True |
| PURE_FINEWEB | triviaqa | 0.013 | 0.007 | 0.020 | False | False |
| PURE_FINEWEB | nq | 0.000 | 0.000 | 0.000 | False | False |
| PURE_STEM | boolq | 0.727 | 0.667 | 0.220 | True | False |
| PURE_STEM | triviaqa | 0.000 | 0.027 | 0.020 | False | False |
| PURE_STEM | nq | 0.000 | 0.000 | 0.000 | False | False |
| PURE_WIKI | boolq | 0.147 | 0.093 | 0.220 | False | True |
| PURE_WIKI | triviaqa | 0.007 | 0.020 | 0.020 | False | False |
| PURE_WIKI | nq | 0.000 | 0.000 | 0.000 | False | False |

- task gate pass: `False`
- overall pass: `False`
