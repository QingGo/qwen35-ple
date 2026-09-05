#!/usr/bin/env python3
"""Smoke test for the HTTP RAG serving endpoint.

Starts the standard-library HTTP server in a subprocess, waits for /health,
queries /answer, and shuts the server down.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    port = 8791
    cmd = [
        sys.executable,
        str(repo / "scripts" / "serve_rag_http.py"),
        "--model",
        "data/models/Qwen3.5-0.8B",
        "--corpus",
        "data/sources/wikitext.jsonl",
        "--mode",
        "bm25",
        "--max-docs",
        "20",
        "--device",
        "cuda",
        "--max-new-tokens",
        "16",
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(40):
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    health = json.loads(r.read().decode())
                break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        else:
            print("server did not become healthy", flush=True)
            return 1

        q = urllib.parse.quote("What is the capital of France?")
        with urllib.request.urlopen(base + f"/answer?q={q}", timeout=120) as r:
            result = json.loads(r.read().decode())
        print("health:", health, flush=True)
        print("answer keys:", sorted(result.keys()), flush=True)
        print("answer preview:", result.get("answer", "")[:200], flush=True)
        print("SMOKE_OK", flush=True)
        return 0
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
