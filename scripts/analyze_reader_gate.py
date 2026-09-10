#!/usr/bin/env python3
"""Measure per-token PLE gate statistics on QA prompts.

The Phase 2 diagnostic showed that the reader helps TriviaQA/NQ but hurts
BoolQ.  A natural hypothesis is that the PLE gate opens on BoolQ passages and
injects irrelevant rows.  This script loads a saved reader checkpoint, installs
the reader hook, and records ``reader.last_gate`` on the same QA prompts used
by the evaluation harness.

Outputs per-task gate means, maxima, and open fractions, plus per-item rows so
the same JSON can be joined with correctness data later.

Usage::

    PYTHONPATH=src python scripts/analyze_reader_gate.py \
      --model /root/autodl-tmp/qwen35-ple/models/Qwen3.5-0.8B \
      --reader-checkpoint outputs/phase2-diagnostic-layer2/reader-PURE_WIKI-real-seed0.pt \
      --rows-dir /dev/shm/qwen38-rows \
      --model-dir /root/autodl-tmp/qwen35-ple/models/qwen38_ple \
      --qa-file data/phase1/kb-wiki/qa.eval.jsonl \
      --output outputs/gate-stats-real-seed0.json \
      --markdown outputs/gate-stats-real-seed0.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from qwen35_ple.eval.prompting import format_qa_prompt
from qwen35_ple.reader import install_reader_hook
from qwen35_ple.reader_registry import load_reader_with_extra
from qwen35_ple.real_ple import resolve_ple_weight_scale, rowids_from_tokens


def _install_torch_compat() -> None:
    for name, alias in [
        ("uint16", "int16"),
        ("uint32", "int32"),
        ("uint64", "int64"),
    ]:
        if not hasattr(torch, name):
            setattr(torch, name, getattr(torch, alias))
    if not hasattr(torch, "get_default_device"):
        torch.get_default_device = lambda: torch.device("cpu")
    if not hasattr(torch, "set_default_device"):
        torch.set_default_device = lambda device: None
    _orig = torch.is_autocast_enabled

    def _autocast(device_type=None):
        return _orig()

    torch.is_autocast_enabled = _autocast
    if not hasattr(torch.nn, "RMSNorm"):

        class _RMSNorm(torch.nn.Module):
            def __init__(self, dim: int, eps: float = 1e-6) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(dim))
                self.eps = eps

            def forward(self, x):
                return (
                    x
                    * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                    * self.weight
                )

        torch.nn.RMSNorm = _RMSNorm


def _load_qa(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    records: list[dict] = []
    if text.lstrip().startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise SystemExit(f"{path}: expected a JSON list")
        records = data
    else:
        for line in text.splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    out = []
    for item in records:
        if "question" not in item or "answer" not in item:
            raise SystemExit(f"{path}: item missing question/answer")
        out.append(
            {
                "task": str(item.get("task", "qa")),
                "question": str(item["question"]),
                "answer": str(item["answer"]),
            }
        )
    return out


class _EtStore:
    def __init__(self, rows_dir: str, scale: float) -> None:
        import engramdb

        self.store = engramdb.Store(
            rows_dir,
            shards=128,
            rows_per_shard=2_500_012,
            width=160,
        )
        self.scale = float(scale)

    def fetch(self, ids: list[int]) -> np.ndarray:
        import engramdb

        rowids = rowids_from_tokens(np.asarray(ids, dtype=np.int64))
        arr = engramdb.fetch_e_t_tensor(
            self.store,
            rowids.reshape(-1).tolist(),
            scale=self.scale,
            num_heads=16,
            head_dim=160,
            dtype=None,
            out_dtype=None,
        )
        return arr.reshape(len(ids), 2560).numpy()

    def close(self) -> None:
        self.store.close()


def _load_model(model_path: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32
    )
    if next(model.parameters()).dtype != torch.float32:
        model = model.to(torch.float32)
    model.eval()
    if device != "cpu":
        model = model.to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    return tokenizer, model


def _gate_stats(gate: torch.Tensor) -> dict[str, float]:
    """gate: [B, T, hc, 1] or [B, T, branches]."""
    values = gate.detach().float().reshape(-1)
    if values.numel() == 0:
        return {
            "mean": float("nan"),
            "max": float("nan"),
            "open_frac": float("nan"),
            "frac_gt_0p1": float("nan"),
            "frac_gt_0p5": float("nan"),
            "frac_gt_0p9": float("nan"),
        }
    return {
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
        "open_frac": float((values > 0.5).float().mean().item()),
        "frac_gt_0p1": float((values > 0.1).float().mean().item()),
        "frac_gt_0p5": float((values > 0.5).float().mean().item()),
        "frac_gt_0p9": float((values > 0.9).float().mean().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reader-checkpoint", required=True)
    parser.add_argument("--rows-dir", required=True)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--qa-file", required=True)
    parser.add_argument("--qa-prompt-template", default="Question: {question}\nAnswer:")
    parser.add_argument(
        "--qa-boolq-prompt-template",
        default="Question: {question}\nAnswer with one word, Yes or No:",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()

    _install_torch_compat()
    scale = resolve_ple_weight_scale(model_dir=args.model_dir, scale=args.scale)
    tokenizer, model = _load_model(args.model, args.device)
    reader, _extra = load_reader_with_extra(
        args.reader_checkpoint, device=args.device
    )
    reader.eval()
    install_reader_hook(model, args.layer, reader)

    items = _load_qa(Path(args.qa_file))
    store = _EtStore(args.rows_dir, scale)
    rows: list[dict] = []
    try:
        for idx, item in enumerate(items):
            text = format_qa_prompt(
                item["question"],
                task=item.get("task"),
                template=args.qa_prompt_template,
                boolq_template=args.qa_boolq_prompt_template,
            )
            ids = list(tokenizer.encode(text, add_special_tokens=False))
            if len(ids) > args.max_tokens:
                ids = ids[-args.max_tokens :]
            if not ids:
                continue
            e_t = store.fetch(ids)
            device = next(model.parameters()).device
            input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            model._current_ple_e_t = (
                torch.from_numpy(e_t).float().to(device).unsqueeze(0)
            )
            with torch.no_grad():
                model(input_ids=input_ids)
            gate = getattr(reader, "last_gate", None)
            if gate is None:
                raise SystemExit(
                    "reader did not expose last_gate; "
                    "use a checkpoint from the current reader implementation"
                )
            stats = _gate_stats(gate)
            contrib = getattr(model, "_last_reader_contribution", None)
            hidden_t = getattr(model, "_last_reader_hidden", None)
            contrib_norm = float("nan")
            hidden_norm = float("nan")
            contrib_ratio = float("nan")
            contrib_abs = float("nan")
            if contrib is not None and hidden_t is not None:
                c = contrib.detach().float()
                h = hidden_t.detach().float()
                contrib_norm = float(c.norm(dim=-1).mean().item())
                hidden_norm = float(h.norm(dim=-1).mean().item())
                contrib_ratio = contrib_norm / max(hidden_norm, 1e-8)
                contrib_abs = float(c.abs().mean().item())
            # Split the prompt into "body" and "tail" to see whether the gate
            # opens on the passage or only near the question/answer marker.
            g = gate.detach().float()
            if g.dim() == 4:
                g = g.squeeze(0).squeeze(-1)  # [T, hc]
            elif g.dim() == 3:
                g = g.squeeze(0)  # [T, hc]
            elif g.dim() == 2:
                g = g.squeeze(0)
            g = g.reshape(g.shape[0], -1)
            tail = min(32, g.shape[0])
            rows.append(
                {
                    "task": item["task"],
                    "question": item["question"][:200],
                    "prompt_len": len(ids),
                    "gate_mean": stats["mean"],
                    "gate_max": stats["max"],
                    "gate_open_frac": stats["open_frac"],
                    "gate_frac_gt_0p1": stats["frac_gt_0p1"],
                    "gate_frac_gt_0p5": stats["frac_gt_0p5"],
                    "gate_frac_gt_0p9": stats["frac_gt_0p9"],
                    "gate_mean_tail32": float(g[-tail:].mean().item()),
                    "gate_mean_body": float(g[:-tail].mean().item())
                    if g.shape[0] > tail
                    else float(g.mean().item()),
                    "contrib_norm": contrib_norm,
                    "hidden_norm": hidden_norm,
                    "contrib_ratio": contrib_ratio,
                    "contrib_abs_mean": contrib_abs,
                }
            )
            if (idx + 1) % 10 == 0:
                print(f"  gate stats {idx + 1}/{len(items)}", flush=True)
    finally:
        store.close()

    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    aggregate: dict[str, Any] = {}
    for task, task_rows in sorted(by_task.items()):
        aggregate[task] = {
            "n": len(task_rows),
            "gate_mean": float(np.mean([r["gate_mean"] for r in task_rows])),
            "gate_max": float(np.mean([r["gate_max"] for r in task_rows])),
            "gate_open_frac": float(
                np.mean([r["gate_open_frac"] for r in task_rows])
            ),
            "gate_mean_tail32": float(
                np.mean([r["gate_mean_tail32"] for r in task_rows])
            ),
            "gate_mean_body": float(
                np.mean([r["gate_mean_body"] for r in task_rows])
            ),
            "gate_frac_gt_0p1": float(
                np.mean([r["gate_frac_gt_0p1"] for r in task_rows])
            ),
            "gate_frac_gt_0p5": float(
                np.mean([r["gate_frac_gt_0p5"] for r in task_rows])
            ),
            "gate_frac_gt_0p9": float(
                np.mean([r["gate_frac_gt_0p9"] for r in task_rows])
            ),
            "contrib_norm": float(np.mean([r["contrib_norm"] for r in task_rows])),
            "hidden_norm": float(np.mean([r["hidden_norm"] for r in task_rows])),
            "contrib_ratio": float(np.mean([r["contrib_ratio"] for r in task_rows])),
            "contrib_abs_mean": float(
                np.mean([r["contrib_abs_mean"] for r in task_rows])
            ),
        }
    report = {
        "model": args.model,
        "reader_checkpoint": args.reader_checkpoint,
        "layer": args.layer,
        "scale": scale,
        "qa_file": args.qa_file,
        "aggregate": aggregate,
        "rows": rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.markdown:
        lines = [
            "# PLE gate statistics",
            "",
            f"- model: `{args.model}`",
            f"- reader: `{args.reader_checkpoint}`",
            f"- layer: {args.layer}",
            f"- QA: `{args.qa_file}` ({len(rows)} items)",
            "",
            "| Task | n | gate mean | gate max | frac>0.1 | frac>0.5 | frac>0.9 | tail-32 | body | contrib ratio | contrib abs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for task, entry in aggregate.items():
            lines.append(
                f"| {task} | {entry['n']} | {entry['gate_mean']:.4f} | "
                f"{entry['gate_max']:.4f} | {entry['gate_frac_gt_0p1']:.4f} | "
                f"{entry['gate_frac_gt_0p5']:.4f} | {entry['gate_frac_gt_0p9']:.4f} | "
                f"{entry['gate_mean_tail32']:.4f} | {entry['gate_mean_body']:.4f} | "
                f"{entry['contrib_ratio']:.4f} | {entry['contrib_abs_mean']:.4f} |"
            )
        lines += [
            "",
            "> `contrib ratio` is the mean per-token reader contribution norm divided",
            "> by the mean hidden-state norm.  A low gate mean can still break a task",
            "> if a few high-gate tokens or a large value/output scale produce a large",
            "> relative contribution.",
            "",
        ]
        md_path = Path(args.markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
