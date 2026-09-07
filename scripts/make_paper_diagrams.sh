#!/usr/bin/env bash
# Regenerate Typst/Fletcher architecture and pipeline diagrams as SVG files.
# These diagrams are committed as SVG so paper.pdf can be built without network access.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export TYPST_PACKAGE_CACHE_PATH="$ROOT/paper/vendor/typst/packages"

DIAGRAMS=(
  "$ROOT/paper/figures/architecture_diagram.typ"
  "$ROOT/paper/figures/inference_pipeline.typ"
)

for src in "${DIAGRAMS[@]}"; do
  echo "compiling $(basename "$src")"
  typst compile "$src" "${src%.typ}.svg"
done
