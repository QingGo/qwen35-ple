# Paper Evaluation Card

> Version: 2026-09-06
> Purpose: track reproducibility, artifacts, and experiment coverage for the current paper effort.

---

## 1. Environment

| Item | Value |
|---|---|
| Model | Qwen3.5-0.8B |
| Memory | Qwen3.8-Flash-Next PLE / n-gram addressable memory |
| GPU | NVIDIA GTX 1070 8GB |
| Python | 3.11 / 3.13 remote venv |
| PyTorch | 2.6.0+cu124 (remote) |
| Repo | QingGo/qwen35-ple |

---

## 2. Real datasets

| Dataset | Source | Current usage |
|---|---|---|
| HumanEval | `openai/openai_humaneval` | 5 / 20+ problems |
| TriviaQA | `mandarjoshi/trivia_qa` RC | 100 questions |

Synthetic/local-only datasets are explicitly labelled as such in the paper.

---

## 3. Baseline methods

| Baseline | Status |
|---|---|
| BM25 RAG | ✅ |
| kNN-LM (small hidden-state) | ✅ |
| NGM (official hook) | ✅ / in progress |
| MemSFT | not yet |
| TokenMem | existing prototype, not in final table |
| Memory Grafting | not yet |

---

## 4. Metrics

| Metric | Where used |
|---|---|
| teacher-forced NLL | local continuation, paired projector |
| hit@1 | local continuation |
| pass@1 | HumanEval |
| exact match | TriviaQA |
| repetition rate | HumanEval / TriviaQA |
| bootstrap 95% CI | 5-seed projector |
| multi-system answer logprob | joint system table |

---

## 5. Reproducibility checklist

- [x] Fixed random seeds
- [x] Paired evaluation rows
- [x] Bootstrap CIs
- [x] Commands in docs / manifest
- [ ] Public model/adapter weights
- [ ] Public memory table (or sample)
- [ ] Docker image build tested
- [ ] CPU/efficiency numbers
- [ ] LLM-as-judge agreement with human

---

## 6. Artifacts (planned/public)

| Artifact | Status |
|---|---|
| Source code | ✅ public repo |
| CI | ✅ |
| Repro manifest | ✅ |
| Dockerfile | ✅ created |
| Evaluation card | ✅ this file |
| Adapter weights | ❌ |
| Dataset preprocessing scripts | ✅ |
| HumanEval/TriviaQA outputs | ✅ in outputs/ |
