#!/usr/bin/env python3
"""Train an oracle-routing probe on frozen layer hidden states.

Question: is "use the PLE reader on this item" decodable from the same hidden
state the reader uses as its query?  We build labels from the standard-eval
generation runs:

* oracle target: 1 = reader correct & no-reader wrong, 0 = no-reader correct &
  reader wrong (ambiguous items are dropped);
* direct targets: reader correctness and no-reader correctness on all items.

The probe never sees the PLE reader; it only sees layer hidden states from the
base backbone.  A high AUC would mean selective gating is learnable in
principle; an AUC near 0.5 means the current gate has no signal to exploit.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

TASKS = ("boolq", "triviaqa", "nq")


def _load_answer_helpers():
    try:
        from qwen35_ple.eval.answers import score_answer_v2
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("run with PYTHONPATH=src") from exc
    return score_answer_v2


def _answers(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results") or []
    for entry in results:
        qa = (entry or {}).get("qa_exact") or {}
        if isinstance(qa.get("answers"), list):
            return list(qa["answers"])
    summary = data.get("summary") or {}
    for entry in summary.values():
        for detail in (entry or {}).get("details", []) or []:
            qa = (detail or {}).get("qa_exact") or {}
            if isinstance(qa.get("answers"), list):
                return list(qa["answers"])
    raise SystemExit(f"no qa_exact answers in {path}")


def _correctness(path: Path) -> dict[tuple[str, str], bool]:
    scorer = _load_answer_helpers()
    out: dict[tuple[str, str], bool] = {}
    for row in _answers(path):
        task = str(row.get("task", "unknown"))
        metric = "extracted_exact" if task == "boolq" else "extracted_contains"
        scores = scorer(
            str(row.get("generated", "")), str(row.get("answer", "")), task=task
        )
        out[(task, str(row.get("question", "")))] = bool(scores[metric])
    return out


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    pos = np.sort(score[y == 1])
    neg = np.sort(score[y == 0])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    lt = np.searchsorted(neg, pos, side="left").sum()
    le = np.searchsorted(neg, pos, side="right").sum()
    return float((lt + 0.5 * (le - lt)) / (len(pos) * len(neg)))


def _fit_ridge(
    X: np.ndarray, y: np.ndarray, alpha: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form ridge probe; fast enough for a 1500-item diagnostic."""
    x = X.astype(np.float64)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-6] = 1.0
    xs = (x - mu) / sd
    target = y.astype(np.float64)
    xm = xs.mean(axis=0)
    ym = target.mean()
    xc = xs - xm
    yc = target - ym
    gram = xc.T @ xc + alpha * np.eye(xc.shape[1])
    w = np.linalg.solve(gram, xc.T @ yc)
    b = ym - xm @ w
    return w, b, mu, sd


def _predict_ridge(
    w: np.ndarray, b: float, mu: np.ndarray, sd: np.ndarray, X: np.ndarray
) -> np.ndarray:
    x = (X.astype(np.float64) - mu) / sd
    return x @ w + b


def _stratified_folds(
    labels: np.ndarray, groups: np.ndarray, n_folds: int, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    assignment = np.full(len(labels), -1, dtype=np.int64)
    buckets: dict[tuple[str, int], list[int]] = {}
    for idx, (label, group) in enumerate(zip(labels, groups)):
        buckets.setdefault((str(group), int(label)), []).append(idx)
    for indices in buckets.values():
        rng.shuffle(indices)
        for offset, idx in enumerate(indices):
            assignment[idx] = offset % n_folds
    folds = []
    for fold in range(n_folds):
        test = np.where(assignment == fold)[0]
        train = np.where(assignment != fold)[0]
        if len(test) and len(train):
            folds.append((train, test))
    return folds


def _cross_val(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int
) -> dict[str, Any]:
    folds = _stratified_folds(y, groups, n_folds)
    scores = np.full(len(y), np.nan, dtype=np.float64)
    for fold_idx, (train, test) in enumerate(folds):
        w, b, mu, sd = _fit_ridge(X[train], y[train])
        scores[test] = _predict_ridge(w, b, mu, sd, X[test])
    mask = np.isfinite(scores)
    y_masked = y[mask]
    out: dict[str, Any] = {
        "n": int(mask.sum()),
        "auc": _auc(y_masked, scores[mask]),
        "acc": float(((scores[mask] >= 0.0) == y_masked).mean()),
        "majority": float(max(y_masked.mean(), 1.0 - y_masked.mean())),
    }
    per_task = {}
    for task in TASKS:
        tmask = mask & (groups == task)
        if tmask.sum() and len(np.unique(y[tmask])) > 1:
            per_task[task] = {
                "n": int(tmask.sum()),
                "auc": _auc(y[tmask], scores[tmask]),
            }
    out["per_task"] = per_task
    return out


def _leave_one_group_out(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> dict[str, Any]:
    out = {}
    for task in TASKS:
        test = groups == task
        train = ~test
        if not test.any() or not train.any():
            continue
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            continue
        w, b, mu, sd = _fit_ridge(X[train], y[train])
        scores = _predict_ridge(w, b, mu, sd, X[test])
        out[task] = {
            "n": int(test.sum()),
            "auc": _auc(y[test], scores),
            "acc": float(((scores >= 0.0) == y[test]).mean()),
        }
    return out


def _fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def build_markdown(
    protocol: str,
    feature_path: Path,
    feature_sizes: dict[str, int],
    oracle: dict[str, Any],
    direct: dict[str, Any],
    loo: dict[str, Any],
) -> str:
    lines = [
        f"# Oracle-routing probe ({protocol})",
        "",
        f"- features: `{feature_path}`",
        f"- feature sizes: {feature_sizes}",
        "",
        "## Oracle choice (reader-only vs no-reader-only)",
        "",
        "| Feature | n | AUC | Acc | Majority |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, entry in oracle.items():
        lines.append(
            f"| {name} | {entry['n']} | {_fmt(entry['auc'])} | "
            f"{_fmt(entry['acc'])} | {_fmt(entry['majority'])} |"
        )
    lines += ["", "### Per-task AUC", "", "| Feature | BoolQ | TriviaQA | NQ |", "|---|---:|---:|---:|"]
    for name, entry in oracle.items():
        cells = [
            _fmt(entry.get("per_task", {}).get(task, {}).get("auc", float("nan")))
            for task in TASKS
        ]
        lines.append(f"| {name} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += [
        "",
        "## Direct correctness probes (5-fold CV)",
        "",
        "| Target | Feature | n | AUC | Acc | Majority |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for target, entries in direct.items():
        for name, entry in entries.items():
            lines.append(
                f"| {target} | {name} | {entry['n']} | {_fmt(entry['auc'])} | "
                f"{_fmt(entry['acc'])} | {_fmt(entry['majority'])} |"
            )
    if loo:
        lines += [
            "",
            "## Leave-one-task-out (oracle choice)",
            "",
            "| Held-out task | n | AUC | Acc |",
            "|---|---:|---:|---:|",
        ]
        for task, entry in loo.items():
            lines.append(
                f"| {task} | {entry['n']} | {_fmt(entry['auc'])} | {_fmt(entry['acc'])} |"
            )
    lines += [
        "",
        "> AUC near 0.5 means the hidden state carries no linearly decodable",
        "> signal for selective PLE use.  n is the number of unambiguous items.",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--no-reader", type=Path, required=True)
    parser.add_argument("--real", type=Path, action="append", required=True)
    parser.add_argument("--protocol", default="raw")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)

    npz = np.load(args.features, allow_pickle=False)
    last = npz["last"].astype(np.float32)
    mean = npz["mean"].astype(np.float32)
    tasks = npz["tasks"].astype(str)
    questions = npz["questions"].astype(str)
    n = len(tasks)

    no_map = _correctness(args.no_reader)
    real_maps = [_correctness(path) for path in args.real]
    no_labels = np.asarray(
        [int(no_map.get((str(t), str(q)), 0)) for t, q in zip(tasks, questions)],
        dtype=np.int64,
    )
    real_labels = np.stack(
        [
            np.asarray(
                [int(m.get((str(t), str(q)), 0)) for t, q in zip(tasks, questions)],
                dtype=np.int64,
            )
            for m in real_maps
        ]
    )
    real_majority = (real_labels.mean(axis=0) > 0.5).astype(np.int64)

    def oracle_labels(real: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keep = real != no_labels
        return real[keep], keep

    feature_matrices = {
        "last": last,
        "mean": mean,
        "last+mean": np.concatenate([last, mean], axis=1),
    }

    oracle: dict[str, Any] = {}
    for name, X in feature_matrices.items():
        y, keep = oracle_labels(real_majority)
        oracle[name] = _cross_val(X[keep], y, tasks[keep], args.folds)
        oracle[name]["n_seeds"] = len(args.real)

    direct: dict[str, Any] = {"reader": {}, "no-reader": {}}
    for name, X in feature_matrices.items():
        direct["reader"][name] = _cross_val(X, real_majority, tasks, args.folds)
        direct["no-reader"][name] = _cross_val(X, no_labels, tasks, args.folds)

    y_oracle, keep = oracle_labels(real_majority)
    loo = _leave_one_group_out(
        feature_matrices["last+mean"][keep], y_oracle, tasks[keep]
    )

    payload = {
        "protocol": args.protocol,
        "features": str(args.features),
        "n_items": n,
        "oracle": oracle,
        "direct": direct,
        "leave_one_task_out": loo,
        "real_seeds": [str(path) for path in args.real],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        build_markdown(
            args.protocol,
            args.features,
            {name: int(X.shape[1]) for name, X in feature_matrices.items()},
            oracle,
            direct,
            loo,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"oracle": oracle, "direct": direct}, indent=2))
    print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
