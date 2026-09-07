#!/usr/bin/env python3
"""Build and optionally upload the public qwen35-ple artifact release.

This script collects the paper-relevant artifacts (PLE projector checkpoints,
Purified MoRA adapters, configs, datasets, evaluation docs) into a clean
directory, writes checksums and a metadata manifest, and can upload the bundle
to Hugging Face with ``--upload``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

REPO_FILES = {
    "projector": [
        "outputs/ple-projector-10k-seed0.projector.json",
        "outputs/ple-projector-10k-seed1.projector.json",
        "outputs/ple-projector-10k-seed2.projector.json",
        "outputs/ple-projector-10k-seed3.projector.json",
        "outputs/ple-projector-10k-seed4.projector.json",
    ],
    "adapter_dirs": [
        "outputs/cap1-purified-mora-80",
        "outputs/cap1-purified-mora-80-s1",
        "outputs/cap1-purified-mora-80-s2",
    ],
    "datasets": [
        "data/ple-projector-dataset-1k.jsonl",
        "data/ple-projector-dataset-10k.jsonl",
    ],
    "configs": [
        "configs/ngram-fusion-router.json",
        "configs/ngram-fusion-router-projector-open.json",
        "configs/ngram-fusion-router-projector-policy.json",
        "configs/ngram-fusion-router-token-policy.json",
        "configs/token-ple-policy.json",
    ],
    "docs": [
        "docs/evaluation-card-paper.md",
        "docs/reproducibility-manifest.md",
        "docs/round-115-paper-evidence-package.md",
        "docs/round-121-ultimate-goal-tech-debt-plan.md",
    ],
    "scripts": [
        "scripts/run_llm_judge.py",
        "scripts/run_humaneval_real_ablation.py",
        "scripts/run_triviaqa_real_eval.py",
        "scripts/run_humaneval_passk.py",
        "scripts/train_ple_projector.py",
        "scripts/build_ple_projector_dataset.py",
        "scripts/analyze_ple_projector_paired.py",
        "scripts/run_sensitivity_sweep.sh",
        "scripts/run_ple_evidence_p0.py",
        "scripts/build_judge_input.py",
        "scripts/build_hf_artifact_release.py",
    ],
    "root_files": [
        "Dockerfile",
        "Makefile",
        "AGENTS.md",
        "docs/round-122-phase-ab-progress.md",
    ],
    "results": [
        "outputs/humaneval-real-50-fast.json",
        "outputs/triviaqa-real-200.json",
        "outputs/llm-judge-humaneval50.json",
        "outputs/llm-judge-triviaqa200.json",
        "outputs/ple-projector-paired-analysis-5seed.json",
        "outputs/humaneval-passk-10x3.json",
        "outputs/cpu-bench.json",
        "outputs/cpu-bench-quant.json",
        "outputs/code-gen-projector-open.json",
        "outputs/code-gen-projector-policy.json",
        "outputs/code-gen-projector-policy2.json",
    ],
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_tree(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--bundle", default="artifacts-release")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--hf-repo", default="DefEki/qwen35-ple-auditable-ngram-memory")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    bundle = Path(args.bundle)
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)

    manifest: dict[str, object] = {
        "schema": "qwen35-ple-artifact-release-v1",
        "repo": "https://github.com/QingGo/qwen35-ple",
        "date": subprocess.run(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "git_commit": subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "files": [],
    }

    # Projector JSONs
    for rel in REPO_FILES["projector"]:
        src = root / rel
        if src.exists():
            dst = bundle / "projectors" / src.name
            _copy_tree(src, dst)
            manifest["files"].append({"path": str(dst.relative_to(bundle)), "sha256": _sha256(dst), "bytes": dst.stat().st_size})

    # Adapter directories (keep their internal README/config/files)
    for rel in REPO_FILES["adapter_dirs"]:
        src = root / rel
        if src.exists():
            dst = bundle / "adapters" / src.name
            _copy_tree(src, dst)
            for p in sorted(dst.rglob("*")):
                if p.is_file():
                    manifest["files"].append({"path": str(p.relative_to(bundle)), "sha256": _sha256(p), "bytes": p.stat().st_size})

    # Datasets
    for rel in REPO_FILES["datasets"]:
        src = root / rel
        if src.exists():
            dst = bundle / "datasets" / src.name
            _copy_tree(src, dst)
            manifest["files"].append({"path": str(dst.relative_to(bundle)), "sha256": _sha256(dst), "bytes": dst.stat().st_size})

    # Configs
    for rel in REPO_FILES["configs"]:
        src = root / rel
        if src.exists():
            dst = bundle / "configs" / Path(rel).name
            _copy_tree(src, dst)
            manifest["files"].append({"path": str(dst.relative_to(bundle)), "sha256": _sha256(dst), "bytes": dst.stat().st_size})

    # Docs
    for rel in REPO_FILES["docs"]:
        src = root / rel
        if src.exists():
            dst = bundle / "docs" / Path(rel).name
            _copy_tree(src, dst)
            manifest["files"].append({"path": str(dst.relative_to(bundle)), "sha256": _sha256(dst), "bytes": dst.stat().st_size})

    # Scripts
    for rel in REPO_FILES["scripts"]:
        src = root / rel
        if src.exists():
            dst = bundle / "scripts" / Path(rel).name
            _copy_tree(src, dst)
            manifest["files"].append({"path": str(dst.relative_to(bundle)), "sha256": _sha256(dst), "bytes": dst.stat().st_size})

    # Root-level release files
    for rel in REPO_FILES["root_files"]:
        src = root / rel
        if src.exists():
            dst = bundle / "root" / Path(rel).name
            _copy_tree(src, dst)
            manifest["files"].append({"path": str(dst.relative_to(bundle)), "sha256": _sha256(dst), "bytes": dst.stat().st_size})
        elif Path(rel).suffix == ".md":
            # Keep docs under docs/ when a top-level file is missing.
            src2 = root / "docs" / Path(rel).name
            if src2.exists():
                dst = bundle / "docs" / Path(rel).name
                _copy_tree(src2, dst)
                manifest["files"].append({"path": str(dst.relative_to(bundle)), "sha256": _sha256(dst), "bytes": dst.stat().st_size})

    # Evaluation result files
    for rel in REPO_FILES["results"]:
        src = root / rel
        if src.exists():
            dst = bundle / "results" / Path(rel).name
            _copy_tree(src, dst)
            manifest["files"].append({"path": str(dst.relative_to(bundle)), "sha256": _sha256(dst), "bytes": dst.stat().st_size})

    manifest_path = bundle / "artifact-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # SHA256SUMS
    sums = []
    for entry in manifest["files"]:
        sums.append(f"{entry['sha256']}  {entry['path']}")
    (bundle / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")

    readme = bundle / "README.md"
    readme.write_text(
        """---
language:
  - en
license: apache-2.0
tags:
  - n-gram
  - external-memory
  - small-language-models
  - auditable-memory
  - ple
  - engram
  - low-resource
library_name: custom
pipeline_tag: text-generation
---

# qwen35-ple Auditable N-Gram Memory Artifacts

This release contains the reproducible artifacts for the paper
**Auditable N-Gram Memory for Small Language Models**.

## Links

- Code repository: https://github.com/QingGo/qwen35-ple
- Paper source: https://github.com/QingGo/qwen35-ple/blob/main/paper/paper.typ
- Compiled PDF: https://github.com/QingGo/qwen35-ple/blob/main/paper.pdf
- Evaluation card: https://github.com/QingGo/qwen35-ple/blob/main/docs/evaluation-card-paper.md
- Reproducibility manifest: https://github.com/QingGo/qwen35-ple/blob/main/docs/reproducibility-manifest.md

## Contents

- `projectors/`: PLE Projector checkpoints (10k training, seeds 0-4).
- `adapters/`: Purified OPSD MoRA adapters (seeds 0-2).
- `datasets/`: PLE projector local-continuation datasets (1k and 10k).
- `configs/`: N-gram fusion router / token policy / projector configs.
- `scripts/`: Evaluation, training, fairness/sensitivity scripts.
- `results/`: HumanEval, TriviaQA, pass@k, LLM judge, sensitivity, CPU benchmark results.
- `docs/`: Evaluation card, reproducibility manifest, evidence notes.
- `artifact-manifest.json` / `SHA256SUMS`: checksums and provenance.

## Use

```bash
# Reproduce PLE projector training (example)
python scripts/train_ple_projector.py \
  --model <Qwen3.5-0.8B> \
  --dataset datasets/ple-projector-dataset-10k.jsonl \
  --max-train-samples 7000 --max-eval-samples 300 --steps 100 \
  --device cuda --seed 0 --output projector-seed0.json
```

## License

Artifacts are released under Apache-2.0 unless noted otherwise.
Model weights remain subject to the upstream Qwen license.
""",
        encoding="utf-8",
    )

    print(f"[artifact] wrote bundle at {bundle} with {len(manifest['files'])} files")
    if args.upload:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise SystemExit("HF_TOKEN is required for --upload")
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(repo_id=args.hf_repo, repo_type="model", exist_ok=True, private=args.private)
        api.upload_folder(folder_path=str(bundle), repo_id=args.hf_repo, repo_type="model", commit_message="qwen35-ple auditable n-gram memory artifacts")
        print(f"[artifact] uploaded to https://huggingface.co/{args.hf_repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
