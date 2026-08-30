"""A0/A1 ablation evaluation protocol.

This module is intentionally small and dependency-free.  Real evaluation
harnesses (knowledge recall, long-context, reasoning) should emit a JSON file
matching :class:`EvalResult`; this module compares A0 vs A1 and writes a
human-readable report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalResult:
    model: str
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> "EvalResult":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            model=str(data.get("model", Path(path).stem)),
            metrics={str(k): float(v) for k, v in data.get("metrics", {}).items()},
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class AblationComparison:
    baseline: EvalResult
    treatment: EvalResult

    @property
    def metric_names(self) -> list[str]:
        return sorted(set(self.baseline.metrics) | set(self.treatment.metrics))

    def delta(self, metric: str) -> float | None:
        if metric not in self.baseline.metrics or metric not in self.treatment.metrics:
            return None
        return self.treatment.metrics[metric] - self.baseline.metrics[metric]

    def relative_delta(self, metric: str) -> float | None:
        base = self.delta(metric)
        if base is None or self.baseline.metrics[metric] == 0:
            return None
        return base / abs(self.baseline.metrics[metric])

    def to_report(self) -> str:
        lines = [
            "# A0/A1 消融对比",
            "",
            f"- Baseline (A0): {self.baseline.model}",
            f"- Treatment (A1): {self.treatment.model}",
            "",
            "| Metric | A0 | A1 | Δ | Δ% |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in self.metric_names:
            a0 = self.baseline.metrics.get(name)
            a1 = self.treatment.metrics.get(name)
            delta = self.delta(name)
            rel = self.relative_delta(name)
            lines.append(
                f"| {name} | {a0 if a0 is not None else 'N/A'} "
                f"| {a1 if a1 is not None else 'N/A'} "
                f"| {delta if delta is not None else 'N/A'} "
                f"| {rel if rel is not None else 'N/A'} |"
            )
        return "\n".join(lines)


def load_comparison(a0_path: str | Path, a1_path: str | Path) -> AblationComparison:
    return AblationComparison(
        baseline=EvalResult.from_json(a0_path),
        treatment=EvalResult.from_json(a1_path),
    )
