#!/usr/bin/env python3
"""Aggregate rare-token QA logit-patch results into Phase-A tables.

Consumes the JSON produced by ``mechanism_logit_patch.py`` and the benchmark
metadata produced by ``build_rare_kb.py``.  Reports mean answer log-probability
per condition for rare and common items, plus real-vs-control / real-vs-no-reader
gaps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CONDITIONS = ["no-reader", "real", "control", "random", "zero"]


def _load_benchmark(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("items", [])
    return data


def _load_patch(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("results", [])
    return data


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return sum(vals) / len(vals)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--logits", required=True)
    parser.add_argument("--output", default="outputs/rare-kb-logit-summary.json")
    args = parser.parse_args()

    items = _load_benchmark(args.benchmark)
    results = _load_patch(args.logits)
    if len(items) != len(results):
        print(
            f"warning: benchmark={len(items)} results={len(results)}; "
            "matching by index may be misaligned"
        )

    groups: dict[str, list[tuple[dict, dict]]] = {
        "rare": [],
        "common": [],
    }
    for item, res in zip(items, results, strict=False):
        if isinstance(res, dict) and "conditions" in res:
            cond = res["conditions"]
        elif isinstance(res, dict):
            cond = res
        else:
            continue
        group = "rare" if item.get("is_rare", False) else "common"
        groups[group].append((item, cond))

    summary: dict = {}
    for group, pairs in groups.items():
        entry: dict = {
            "n": len(pairs),
            "conditions": {},
            "deltas": {},
        }
        for cond in CONDITIONS:
            vals = []
            for _item, c in pairs:
                v = c.get(cond, {}).get("answer_logprob")
                if v is not None:
                    vals.append(float(v))
            entry["conditions"][cond] = {
                "n": len(vals),
                "mean": _mean(vals),
            }
        # Deltas are paired per question.
        for a, b in [
            ("real", "control"),
            ("real", "no-reader"),
            ("control", "no-reader"),
            ("real", "random"),
            ("real", "zero"),
        ]:
            diffs = []
            for _item, c in pairs:
                va = c.get(a, {}).get("answer_logprob")
                vb = c.get(b, {}).get("answer_logprob")
                if va is not None and vb is not None:
                    diffs.append(float(va) - float(vb))
            entry["deltas"][f"{a}_minus_{b}"] = {
                "n": len(diffs),
                "mean": _mean(diffs),
            }
        summary[group] = entry

    # Also split by original task where available.
    task_groups: dict[str, dict[str, list[float]]] = {}
    for item, cond in groups["rare"] + groups["common"]:
        task = str(item.get("task", "other"))
        task_groups.setdefault(task, {})
        for cond_name in CONDITIONS:
            v = cond.get(cond_name, {}).get("answer_logprob")
            if v is not None:
                task_groups[task].setdefault(cond_name, []).append(float(v))
    task_summary: dict = {}
    for task, by_cond in task_groups.items():
        task_summary[task] = {
            cond: _mean(vals) for cond, vals in sorted(by_cond.items())
        }
    summary["by_task"] = task_summary

    # Split by original source (used to separate Alpaca-style language tasks
    # from the core QA-expanded knowledge set).
    source_groups: dict[str, dict[str, list[float]]] = {}
    for item, cond in groups["rare"] + groups["common"]:
        source = str(item.get("source", "other"))
        source_groups.setdefault(source, {})
        for cond_name in CONDITIONS:
            v = cond.get(cond_name, {}).get("answer_logprob")
            if v is not None:
                source_groups[source].setdefault(cond_name, []).append(float(v))
    summary["by_source"] = {
        source: {cond: _mean(vals) for cond, vals in sorted(by_cond.items())}
        for source, by_cond in sorted(source_groups.items())
    }

    # Source x rarity matrix.
    source_rare_groups: dict[str, dict[str, dict[str, list[float]]]] = {}
    for item, cond in groups["rare"] + groups["common"]:
        source = str(item.get("source", "other"))
        group = "rare" if item.get("is_rare", False) else "common"
        source_rare_groups.setdefault(source, {}).setdefault(group, {})
        for cond_name in CONDITIONS:
            v = cond.get(cond_name, {}).get("answer_logprob")
            if v is not None:
                source_rare_groups[source][group].setdefault(cond_name, []).append(float(v))
    summary["by_source_rare"] = {
        source: {
            group: {
                cond: _mean(vals) for cond, vals in sorted(by_cond.items())
            }
            for group, by_cond in sorted(groups_by_rare.items())
        }
        for source, groups_by_rare in sorted(source_rare_groups.items())
    }
    summary["n_total"] = len(items)
    summary["n_results"] = len(results)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
