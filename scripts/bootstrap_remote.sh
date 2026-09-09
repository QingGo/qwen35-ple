#!/usr/bin/env bash
# One-command post-reboot bootstrap for the qwen35-ple AutoDL instance.
#
# It does NOT install a fresh environment.  It verifies the persistent assets
# that should already exist on the data disk, ensures qwen38-rows is available
# (extracting from ModelScope only when necessary), writes a reproducibility
# manifest, and prints the next diagnostic command.
#
# Typical use after expanding the data disk to ~100GB:
#
#   bash /root/autodl-tmp/qwen35-ple/repo/scripts/bootstrap_remote.sh \
#     --pull
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

ROOT="/root/autodl-tmp/qwen35-ple"
VENV="$ROOT/venv"
MODEL_DIR="$ROOT/models/Qwen3.5-0.8B"
TOKENIZER_DIR="$ROOT/models/Qwen3.8-Flash-Next-FP8-tokenizer"
ROWS_DIR="/dev/shm/qwen38-rows"
PERSISTENT_ROWS_DIR="$ROOT/qwen38-rows"
DOWNLOAD_DIR="$ROOT/tmp"
MANIFEST="$ROOT/remote-manifest.json"
PYTHON=""
PULL=0
SKIP_ROWS=0
SKIP_TOKENIZER_DOWNLOAD=0
FORCE_EXTRACT=0
TOKENIZER_REPO="Qwen/Qwen3.8-Flash-Next-FP8"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --repo) REPO_DIR="$2"; shift 2 ;;
    --venv) VENV="$2"; shift 2 ;;
    --model-dir) MODEL_DIR="$2"; shift 2 ;;
    --tokenizer-dir) TOKENIZER_DIR="$2"; shift 2 ;;
    --rows-dir) ROWS_DIR="$2"; shift 2 ;;
    --persistent-rows-dir) PERSISTENT_ROWS_DIR="$2"; shift 2 ;;
    --download-dir) DOWNLOAD_DIR="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --tokenizer-repo) TOKENIZER_REPO="$2"; shift 2 ;;
    --pull) PULL=1; shift ;;
    --skip-rows) SKIP_ROWS=1; shift ;;
    --skip-tokenizer-download) SKIP_TOKENIZER_DOWNLOAD=1; shift ;;
    --force-extract) FORCE_EXTRACT=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PYTHON" ]]; then
  if [[ -x "$VENV/bin/python" ]]; then
    PYTHON="$VENV/bin/python"
  else
    PYTHON="python3"
  fi
fi

echo "=== [bootstrap] qwen35-ple remote bootstrap ==="
date
echo "root=$ROOT"
echo "repo=$REPO_DIR"
echo "venv=$VENV"
echo "python=$PYTHON"
echo "model=$MODEL_DIR"
echo "tokenizer=$TOKENIZER_DIR"
echo "rows=$ROWS_DIR"
echo "persistent_rows=$PERSISTENT_ROWS_DIR"

if [[ ! -d "$REPO_DIR/.git" ]]; then
  echo "[bootstrap] ERROR: repo checkout not found at $REPO_DIR" >&2
  exit 1
fi

if [[ "$PULL" == "1" ]]; then
  echo "[bootstrap] pulling repo ..."
  git -C "$REPO_DIR" pull --ff-only
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "[bootstrap] ERROR: python not executable: $PYTHON" >&2
  exit 1
fi

echo "=== [bootstrap] versions ==="
"$PYTHON" - <<'PY'
import sys
print("python", sys.version.split()[0])
for name in ("torch", "transformers", "tokenizers", "numpy", "engramdb"):
    try:
        import importlib.metadata
        print(name, importlib.metadata.version(name))
    except Exception as exc:  # noqa: BLE001
        print(name, "MISSING", type(exc).__name__)
PY

echo "=== [bootstrap] disk ==="
df -hT "$ROOT" /dev/shm 2>/dev/null || true

echo "=== [bootstrap] model check ==="
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "[bootstrap] ERROR: missing $MODEL_DIR/config.json" >&2
  exit 1
fi
if ! ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
  echo "[bootstrap] ERROR: no model.safetensors* in $MODEL_DIR" >&2
  exit 1
fi
echo "[bootstrap] model ok"

echo "=== [bootstrap] tokenizer check ==="
TOKENIZER_FILES=(tokenizer.json tokenizer_config.json vocab.json merges.txt config.json)
missing_tokenizer=0
for name in "${TOKENIZER_FILES[@]}"; do
  if [[ ! -f "$TOKENIZER_DIR/$name" ]]; then
    missing_tokenizer=1
    if [[ "$SKIP_TOKENIZER_DOWNLOAD" == "1" ]]; then
      echo "[bootstrap] ERROR: missing $TOKENIZER_DIR/$name" >&2
      exit 1
    fi
    echo "[bootstrap] downloading tokenizer file: $name"
    mkdir -p "$TOKENIZER_DIR"
    curl --http1.1 -L --fail --retry 10 --retry-delay 3 --retry-all-errors \
      --connect-timeout 30 -C - \
      -o "$TOKENIZER_DIR/$name" \
      "https://modelscope.cn/models/${TOKENIZER_REPO}/resolve/master/$name"
  fi
done
if [[ "$missing_tokenizer" == "1" ]]; then
  echo "[bootstrap] tokenizer files restored"
else
  echo "[bootstrap] tokenizer ok"
fi

if [[ "$SKIP_ROWS" == "0" ]]; then
  echo "=== [bootstrap] qwen38-rows ==="
  ENSURE_ARGS=(
    --rows-dir "$ROWS_DIR"
    --persistent-dir "$PERSISTENT_ROWS_DIR"
    --download-dir "$DOWNLOAD_DIR"
    --python "$PYTHON"
  )
  if [[ "$FORCE_EXTRACT" == "1" ]]; then
    ENSURE_ARGS+=(--force-extract)
  fi
  bash "$REPO_DIR/scripts/ensure_qwen38_rows.sh" "${ENSURE_ARGS[@]}"
else
  echo "[bootstrap] rows check skipped"
fi

echo "=== [bootstrap] manifest ==="
"$PYTHON" "$REPO_DIR/scripts/remote_manifest.py" \
  --output "$MANIFEST" \
  --root "$ROOT" \
  --repo-dir "$REPO_DIR" \
  --venv "$VENV" \
  --model-dir "$MODEL_DIR" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --rows-dir "$ROWS_DIR" \
  --rows-manifest "$ROWS_DIR/manifest.json"

echo "=== [bootstrap] next ==="
cat <<EOF
Diagnostic run example:

  cd $REPO_DIR
  export PYTHONPATH=$REPO_DIR/src
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  bash scripts/run_phase2_diagnostic.sh \\
    --pyenv $PYTHON \\
    --rows-dir $ROWS_DIR \\
    --model-dir $ROOT/models/qwen38_ple \\
    --model $MODEL_DIR \\
    --output-dir $ROOT/outputs/phase2-diagnostic
EOF
