# Paper Artifacts

This directory documents the public artifact status for the paper.

## Released

- Source code: [`qwen35-ple`](https://github.com/QingGo/qwen35-ple)
- Typst paper: `paper.typ`
- Compiled PDF: `paper.pdf`
- Docker image definition: `Dockerfile`
- Evaluation card: `docs/evaluation-card-paper.md`
- Reproducibility manifest: `docs/reproducibility-manifest.md`
- Evidence package: `docs/round-115-paper-evidence-package.md`
- Public Hugging Face artifact repo:
  [`DefEki/qwen35-ple-auditable-ngram-memory`](https://huggingface.co/DefEki/qwen35-ple-auditable-ngram-memory)
  - PLE projector checkpoints (10k, seeds 0–4)
  - Purified OPSD MoRA adapters (seeds 0–2)
  - Projector datasets (1k/10k)
  - Fusion/router/token-policy configs
  - Reproducibility scripts, evaluation card, checksums

## Not yet released / blocked

- Full HumanEval/TriviaQA final result tables are being updated as the GPU
  evidence runs complete.
- CPU deployment benchmark is still being measured.
- Quantized weights are not yet published.

## Intended release format

- Projector JSON: `ple-projector-10k-seed{0..4}.projector.json`
- Adapter directories: `cap1-purified-mora-80`, etc.
- Datasets: `data/ple-projector-dataset-1k.jsonl`, `data/ple-projector-dataset-10k.jsonl`
- Checksum manifest: `artifact-manifest.json` + `SHA256SUMS`
