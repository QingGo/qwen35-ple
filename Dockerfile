# Minimal reproducibility container for qwen35-ple paper experiments.
#
# This image does not bundle model weights / datasets (they are large and may
# require separate licensing).  It provides the Python environment and scripts
# needed to run the experimental harness on a CUDA machine.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential rustc cargo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml uv.lock ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests
COPY docs ./docs

# Install project deps and dev tools.
RUN pip install --no-cache-dir uv && uv sync --all-groups --frozen

CMD ["bash"]
