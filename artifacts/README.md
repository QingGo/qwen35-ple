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

## Not yet released / blocked

- PLE projector checkpoints (`outputs/*.projector.json`) are large and not yet
  committed publicly.
- Purified OPSD adapter weights are not yet hosted.
- CPU deployment benchmark is not yet included.

## Intended release format

- Projector JSON: `ple-projector-10k-seed{0,1,2}.projector.json`
- Adapter directories: `cap1-purified-mora-80`, etc.
- Datasets: `data/ple-projector-dataset-1k.jsonl`, `data/ple-projector-dataset-10k.jsonl`
