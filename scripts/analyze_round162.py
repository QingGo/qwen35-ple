#!/usr/bin/env python3
"""Round-162 analysis: format distribution, paired item-level contrasts, verdict.

Reads the six arm JSONs produced by ``run_phase0.py`` (one mode each), applies
:mod:`scripts.classify_format` to every generation, and reports

* the **full label distribution** per arm (overall and per task),
* **paired item-level** contrasts (same items across arms, so the comparison is
  not two independent marginals) with SEM and McNemar counts,
* the **domain-lexical fingerprint** that decides whether a format move tracks
  the training corpus,
* the **magnitude control** (injected-norm / hidden-norm at the injection layer),
* a **mechanical verdict** implementing the round-162 pre-registration.

The decision rule is fixed in advance and uses the ``--ple-off`` vs
``no-reader`` pair as the empirical noise floor: those two arms are the same
mechanism (contribution identically zero), so whatever separates them is noise,
and a 1-vs-2-vs-3 spread at that level is not evidence of content dependence.

Usage::

    python scripts/analyze_round162.py \
      --arm wiki=outputs/round162/arm-wiki.json \
      --arm code=outputs/round162/arm-code.json \
      ... --out outputs/round162/analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from classify_format import (
    LABELS,
    PREFIX_CLASSES,
    classify,
    describe_token,
    domain_fingerprint,
    first_token_key,
    load_arm,
    prefix_class,
    sub_label,
)

# Pre-registered arm roles (round-162 section 0.2).
TRAINED = ("wiki", "code", "stem")
INJECTING = ("wiki", "code", "stem", "wiki-shuf")
ZERO_INJECTION = ("ple-off", "no-reader")

# One-sided directional predictions: (arm A, arm B, marker) means "A > B".
DIRECTIONAL = (
    ("code", "wiki", "code"),
    ("stem", "wiki", "math"),
    ("wiki", "code", "prose"),
)


def _sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    return float(x.std(ddof=1) / math.sqrt(x.size))


def _paired(a: list[float], b: list[float]) -> dict[str, float]:
    """Paired difference a-b on item-aligned vectors."""
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    if av.shape != bv.shape:
        raise SystemExit(f"paired vectors differ in length: {av.shape} vs {bv.shape}")
    d = av - bv
    return {
        "n": int(d.size),
        "mean_diff": float(d.mean()) if d.size else float("nan"),
        "sem": _sem(d),
        "n_a_only": int(np.sum((av > 0) & (bv == 0))),
        "n_b_only": int(np.sum((bv > 0) & (av == 0))),
    }


def _mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact binomial p-value for discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # exact two-sided binomial with p=0.5
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def _one_sided_p(diff_mean: float, sem: float) -> float:
    if not math.isfinite(sem) or sem <= 0:
        return float("nan")
    z = diff_mean / sem
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def _tv(dist_a: dict[str, float], dist_b: dict[str, float]) -> float:
    """Total-variation distance between two label distributions."""
    return float(0.5 * sum(abs(dist_a.get(k, 0.0) - dist_b.get(k, 0.0)) for k in LABELS))


def _tv_generic(dist_a: dict[str, float], dist_b: dict[str, float]) -> float:
    """Total-variation distance over the union of the two supports."""
    keys = set(dist_a) | set(dist_b)
    return float(
        0.5 * sum(abs(dist_a.get(k, 0.0) - dist_b.get(k, 0.0)) for k in keys)
    )


def _norm_counts(values: list[str]) -> dict[str, float]:
    n = len(values)
    out: dict[str, float] = {}
    for v in values:
        out[v] = out.get(v, 0.0) + 1.0
    return {k: v / n for k, v in out.items()} if n else {}


def _joint(labels: list[str], prefixes: list[str]) -> dict[str, float]:
    """Normalised joint distribution over (label, prefix) pairs."""
    out: dict[str, float] = {}
    for label, prefix in zip(labels, prefixes):
        key = f"{label}|{prefix}"
        out[key] = out.get(key, 0.0) + 1.0
    n = len(labels)
    return {k: v / n for k, v in out.items()} if n else {}


def build(name_to_path: dict[str, Path]) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    for name, path in name_to_path.items():
        arm = load_arm(path)
        answers = arm["answers"]
        texts = [a.get("generated", "") for a in answers]
        labels = [classify(t) for t in texts]
        subs = [sub_label(t) for t in texts]
        prefixes = [prefix_class(t) for t in texts]
        first_tokens = [first_token_key(a) for a in answers]
        fps = [domain_fingerprint(t) for t in texts]
        arms[name] = {
            "source": str(path),
            "mode": arm["mode"],
            "ple_off": bool(arm["ple_off"]),
            "answers": answers,
            "labels": labels,
            "subs": subs,
            "prefixes": prefixes,
            "first_tokens": first_tokens,
            "fp": fps,
            "metrics": arm["metrics"],
            "gold_metrics": arm["gold_metrics"],
            "tasks": [a.get("task") for a in answers],
        }

    names = list(arms)
    out: dict[str, Any] = {"n_arms": len(names), "arms": {}}

    for name, arm in arms.items():
        n = len(arm["labels"])
        counts: dict[str, int] = {label: 0 for label in LABELS}
        for label in arm["labels"]:
            counts[label] += 1
        dist = {k: (counts[k] / n if n else 0.0) for k in LABELS}
        sub_counts: dict[str, int] = {}
        for s in arm["subs"]:
            sub_counts[s] = sub_counts.get(s, 0) + 1
        prefix_counts: dict[str, int] = {k: 0 for k in PREFIX_CLASSES}
        for p in arm["prefixes"]:
            prefix_counts[p] = prefix_counts.get(p, 0) + 1
        first_counts: dict[str, int] = {}
        for tok in arm["first_tokens"]:
            first_counts[tok] = first_counts.get(tok, 0) + 1
        per_task: dict[str, Any] = {}
        for task in sorted({t for t in arm["tasks"] if t}):
            idx = [i for i, t in enumerate(arm["tasks"]) if t == task]
            t_labels = [arm["labels"][i] for i in idx]
            t_subs = [arm["subs"][i] for i in idx]
            t_prefix = [arm["prefixes"][i] for i in idx]
            tc: dict[str, int] = {label: 0 for label in LABELS}
            for label in t_labels:
                tc[label] += 1
            sc: dict[str, int] = {}
            for s in t_subs:
                sc[s] = sc.get(s, 0) + 1
            pc: dict[str, int] = {k: 0 for k in PREFIX_CLASSES}
            for p in t_prefix:
                pc[p] = pc.get(p, 0) + 1
            per_task[task] = {
                "n": len(idx),
                "counts": tc,
                "dist": {k: tc[k] / len(idx) for k in LABELS},
                "sub_counts": sc,
                "sub_dist": {k: v / len(idx) for k, v in sc.items()},
                "prefix_counts": pc,
                "prefix_dist": {k: pc[k] / len(idx) for k in PREFIX_CLASSES},
                "n_generated_mean": float(
                    np.mean([arm["answers"][i].get("n_generated", 0) for i in idx])
                ),
            }
        fp = arm["fp"]
        out["arms"][name] = {
            "source": arm["source"],
            "mode": arm["mode"],
            "ple_off": arm["ple_off"],
            "n": n,
            "counts": counts,
            "dist": dist,
            "sub_counts": sub_counts,
            "sub_dist": {k: v / n for k, v in sub_counts.items()},
            "prefix_counts": prefix_counts,
            "prefix_dist": {k: prefix_counts[k] / n for k in PREFIX_CLASSES},
            "n_generated_mean": float(
                np.mean([a.get("n_generated", 0) for a in arm["answers"]])
            )
            if arm["answers"]
            else float("nan"),
            "n_generated_median": float(
                np.median([a.get("n_generated", 0) for a in arm["answers"]])
            )
            if arm["answers"]
            else float("nan"),
            "first_token_counts": first_counts,
            "first_token_top": sorted(
                first_counts.items(), key=lambda kv: -kv[1]
            )[:8],
            "per_task": per_task,
            "domain": {
                k: float(np.mean([f[k] for f in fp])) if fp else float("nan")
                for k in ("code", "math", "prose")
            },
            "run_phase0": arm["metrics"],
            "run_phase0_gold": arm["gold_metrics"],
        }

    # ---- pairwise paired contrasts -------------------------------------
    pairs: dict[str, Any] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if len(arms[a]["labels"]) != len(arms[b]["labels"]):
                continue
            entry: dict[str, Any] = {}
            for label in LABELS:
                ia = [1.0 if x == label else 0.0 for x in arms[a]["labels"]]
                ib = [1.0 if x == label else 0.0 for x in arms[b]["labels"]]
                st = _paired(ia, ib)
                st["mcnemar_p"] = _mcnemar_p(st["n_a_only"], st["n_b_only"])
                entry[f"label_{label}"] = st
            for marker in ("code", "math", "prose"):
                va = [f[marker] for f in arms[a]["fp"]]
                vb = [f[marker] for f in arms[b]["fp"]]
                st = _paired(va, vb)
                st["one_sided_p"] = _one_sided_p(st["mean_diff"], st["sem"])
                entry[f"domain_{marker}"] = st
            for pclass in PREFIX_CLASSES:
                ia = [1.0 if x == pclass else 0.0 for x in arms[a]["prefixes"]]
                ib = [1.0 if x == pclass else 0.0 for x in arms[b]["prefixes"]]
                st = _paired(ia, ib)
                st["mcnemar_p"] = _mcnemar_p(st["n_a_only"], st["n_b_only"])
                entry[f"prefix_{pclass}"] = st
            entry["tv_label_dist"] = _tv(
                out["arms"][a]["dist"], out["arms"][b]["dist"]
            )
            entry["tv_prefix_dist"] = _tv_generic(
                out["arms"][a]["prefix_dist"], out["arms"][b]["prefix_dist"]
            )
            # Joint (taxonomy label x leading-surface) distribution: the
            # finest format descriptor that is still a *format* descriptor.
            entry["tv_joint_dist"] = _tv_generic(
                _joint(arms[a]["labels"], arms[a]["prefixes"]),
                _joint(arms[b]["labels"], arms[b]["prefixes"]),
            )
            entry["tv_first_token_dist"] = _tv_generic(
                _norm_counts(arms[a]["first_tokens"]),
                _norm_counts(arms[b]["first_tokens"]),
            )
            entry["n_generated"] = _paired(
                [a.get("n_generated", 0) for a in arms[a]["answers"]],
                [a.get("n_generated", 0) for a in arms[b]["answers"]],
            )
            entry["first_token_identical_rate"] = float(
                np.mean(
                    [
                        1.0 if x == y else 0.0
                        for x, y in zip(arms[a]["first_tokens"], arms[b]["first_tokens"])
                    ]
                )
            )
            pairs[f"{a}|{b}"] = entry
    out["pairs"] = pairs

    # ---- directional predictions ---------------------------------------
    def pair_oriented(a: str, b: str) -> dict[str, Any] | None:
        """Return the (a - b) contrast whichever key order ``pairs`` holds."""
        if f"{a}|{b}" in pairs:
            return pairs[f"{a}|{b}"]
        if f"{b}|{a}" in pairs:
            flipped: dict[str, Any] = {}
            for key, val in pairs[f"{b}|{a}"].items():
                if isinstance(val, dict):
                    val = {**val, "mean_diff": -val["mean_diff"]}
                flipped[key] = val
            return flipped
        return None

    directional: dict[str, Any] = {}
    for a, b, marker in DIRECTIONAL:
        p = pair_oriented(a, b)
        if p is None:
            continue
        st = dict(p[f"domain_{marker}"])
        st["one_sided_p"] = _one_sided_p(st["mean_diff"], st["sem"])
        directional[f"{a}>{b}:{marker}"] = st
    out["directional"] = directional

    # ---- mechanical verdict -------------------------------------------
    def _pair(a: str, b: str) -> dict[str, Any] | None:
        return pair_oriented(a, b)

    def _spread(arm_names: tuple[str, ...], label: str) -> tuple[float, float]:
        """Max |paired mean diff| and its SEM over all pairs within a group."""
        vals = []
        for i, a in enumerate(arm_names):
            for b in arm_names[i + 1 :]:
                p = _pair(a, b)
                if p is None:
                    continue
                vals.append((abs(p[f"label_{label}"]["mean_diff"]), p[f"label_{label}"]["sem"], f"{a}|{b}"))
        if not vals:
            return float("nan"), float("nan")
        worst = max(vals, key=lambda kv: kv[0])
        return worst[0], worst[1]

    floor_scaffold, floor_sem = _spread(ZERO_INJECTION, "chat_scaffold")
    floor_tv = _pair(ZERO_INJECTION[0], ZERO_INJECTION[1])
    floor_tv = floor_tv["tv_label_dist"] if floor_tv else float("nan")
    spread_scaffold, spread_sem = _spread(TRAINED, "chat_scaffold")
    tv_candidates = [
        pairs[f"{a}|{b}"]["tv_label_dist"]
        for i, a in enumerate(TRAINED)
        for b in TRAINED[i + 1 :]
        if f"{a}|{b}" in pairs
    ]
    spread_tv = max(tv_candidates) if tv_candidates else float("nan")

    def _spread_metric(arm_names: tuple[str, ...], metric: str) -> tuple[float, str]:
        """Max value of a pair-level metric within a group, with its pair name."""
        vals = [
            (pairs[f"{a}|{b}"][metric], f"{a}|{b}")
            for i, a in enumerate(arm_names)
            for b in arm_names[i + 1 :]
            if f"{a}|{b}" in pairs
        ]
        if not vals:
            return float("nan"), ""
        return max(vals, key=lambda kv: kv[0])

    def _delta_spread_metric(arm_names: tuple[str, ...], metric: str) -> float:
        vals = [
            abs(pairs[f"{a}|{b}"][metric]["mean_diff"])
            for i, a in enumerate(arm_names)
            for b in arm_names[i + 1 :]
            if f"{a}|{b}" in pairs
        ]
        return max(vals) if vals else float("nan")

    injecting_vs_zero = []
    for a in INJECTING:
        for b in ZERO_INJECTION:
            p = _pair(a, b)
            if p is None:
                continue
            injecting_vs_zero.append(
                {
                    "pair": f"{a}|{b}",
                    "scaffold_diff": p["label_chat_scaffold"]["mean_diff"],
                    "scaffold_sem": p["label_chat_scaffold"]["sem"],
                    "scaffold_p": p["label_chat_scaffold"]["mcnemar_p"],
                    "tv": p["tv_label_dist"],
                    "tv_joint": p["tv_joint_dist"],
                    "tv_first_token": p["tv_first_token_dist"],
                    "first_token_identical_rate": p["first_token_identical_rate"],
                }
            )

    co_primary = {}
    for metric in ("tv_label_dist", "tv_prefix_dist", "tv_joint_dist", "tv_first_token_dist"):
        floor_v, _ = _spread_metric(ZERO_INJECTION, metric)
        spread_v, spread_pair = _spread_metric(TRAINED, metric)
        co_primary[metric] = {
            "noise_floor": floor_v,
            "trained_123_spread": spread_v,
            "trained_123_spread_pair": spread_pair,
            "ratio": (spread_v / floor_v) if floor_v and math.isfinite(floor_v) and floor_v > 0 else None,
        }
    # Item-level prefix contrast: the concrete round-156 shape is
    # "newline_lead" (zero injection) vs "space_lead" (injection).
    co_primary["n_generated_absdiff"] = {
        "noise_floor": _delta_spread_metric(ZERO_INJECTION, "n_generated"),
        "trained_123_spread": _delta_spread_metric(TRAINED, "n_generated"),
        "trained_123_spread_sem": max(
            [
                pairs[f"{a}|{b}"]["n_generated"]["sem"]
                for i, a in enumerate(TRAINED)
                for b in TRAINED[i + 1 :]
                if f"{a}|{b}" in pairs
            ]
            or [float("nan")]
        ),
    }
    co_primary["newline_lead_absdiff"] = {
        "noise_floor": _delta_spread_metric(ZERO_INJECTION, "prefix_newline_lead"),
        "trained_123_spread": _delta_spread_metric(TRAINED, "prefix_newline_lead"),
        "trained_123_spread_sem": max(
            [
                pairs[f"{a}|{b}"]["prefix_newline_lead"]["sem"]
                for i, a in enumerate(TRAINED)
                for b in TRAINED[i + 1 :]
                if f"{a}|{b}" in pairs
            ]
            or [float("nan")]
        ),
    }

    out["verdict"] = {
        "arms_present": sorted(arms),
        "missing_preregistered_arms": sorted(
            {*TRAINED, *INJECTING, *ZERO_INJECTION} - set(arms)
        ),
        "noise_floor_scaffold_absdiff": floor_scaffold,
        "noise_floor_scaffold_sem": floor_sem,
        "noise_floor_tv": floor_tv,
        "trained_123_spread_scaffold": spread_scaffold,
        "trained_123_spread_scaffold_sem": spread_sem,
        "trained_123_spread_tv": spread_tv,
        "co_primary": co_primary,
        "injecting_vs_zero": injecting_vs_zero,
        "decision": None,
        "decision_co_primary": None,
    }
    # Mechanical decision, pre-registered:
    #   * 1/2/3 differ from each other if their scaffold spread exceeds both the
    #     zero-injection noise floor and 3 SEM;
    #   * otherwise the content-dependence claim is dead.
    complete = not out["verdict"]["missing_preregistered_arms"]
    if math.isfinite(spread_scaffold) and math.isfinite(floor_scaffold) and complete:
        content_dependent = bool(
            spread_scaffold > max(floor_scaffold, 3.0 * (spread_sem if math.isfinite(spread_sem) else 0.0))
        )
        out["verdict"]["decision"] = (
            "CONTENT_DEPENDENT" if content_dependent else "PERTURBATION_ARTIFACT"
        )
    # Co-primary: same rule applied to the joint (label x leading-surface)
    # distribution, which is the finest descriptor that is still about *format*
    # rather than about content.  Reported separately so that a coarse-taxonomy
    # null cannot hide a real, sub-taxonomy format move -- and so that a
    # coarse-taxonomy "hit" cannot be an artifact of one bucket.
    joint = co_primary["tv_joint_dist"]
    if complete and math.isfinite(joint["trained_123_spread"]) and math.isfinite(joint["noise_floor"]):
        out["verdict"]["decision_co_primary"] = (
            "CONTENT_DEPENDENT"
            if joint["trained_123_spread"] > max(joint["noise_floor"], 0.0)
            else "PERTURBATION_ARTIFACT"
        )
    return out


def _fmt_pct(x: float) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:+.4f}"


def render_markdown(res: dict[str, Any]) -> str:
    lines: list[str] = []
    known = ("wiki", "code", "stem", "wiki-shuf", "ple-off", "no-reader")
    order = [n for n in known if n in res["arms"]] + [
        n for n in res["arms"] if n not in known
    ]
    lines.append("### 每臂格式分布（n = 每臂生成条数）\n")
    header = "| arm | n | " + " | ".join(LABELS) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(LABELS) + 2))
    for name in order:
        a = res["arms"][name]
        cells = " | ".join(f"{a['dist'][k]:.3f}" for k in LABELS)
        lines.append(f"| `{name}` | {a['n']} | {cells} |")
    lines.append("")
    lines.append("### raw_continuation 细分 / 领域词法指纹\n")
    lines.append("| arm | terse | long | domain:code | domain:math | domain:prose |")
    lines.append("|---|---|---|---|---|---|")
    for name in order:
        a = res["arms"][name]
        lines.append(
            f"| `{name}` | {a['sub_dist'].get('terse', 0.0):.3f} | "
            f"{a['sub_dist'].get('long', 0.0):.3f} | {a['domain']['code']:.3f} | "
            f"{a['domain']['math']:.3f} | {a['domain']['prose']:.3f} |"
        )
    lines.append("")
    lines.append("### 逐任务脚手架率 / 空串率\n")
    lines.append("| arm | task | n | chat_scaffold | raw_cont | empty | terse |")
    lines.append("|---|---|---|---|---|---|---|")
    for name in order:
        for task, t in res["arms"][name]["per_task"].items():
            lines.append(
                f"| `{name}` | {task} | {t['n']} | {t['dist']['chat_scaffold']:.3f} | "
                f"{t['dist']['raw_continuation']:.3f} | {t['dist']['empty']:.3f} | "
                f"{t['sub_dist'].get('terse', 0.0):.3f} |"
            )
    lines.append("")
    lines.append("### 前导面 / 首个生成 token（比分类法更细的格式指纹）\n")
    lines.append("| arm | newline_lead | space_lead | no_lead | empty | first-token top3 (id:count) |")
    lines.append("|---|---|---|---|---|---|")
    for name in order:
        a = res["arms"][name]
        top = ", ".join(
            f"{tok}({describe_token(tok)}):{cnt}" for tok, cnt in a["first_token_top"][:3]
        )
        lines.append(
            f"| `{name}` | {a['prefix_dist']['newline_lead']:.3f} | "
            f"{a['prefix_dist']['space_lead']:.3f} | {a['prefix_dist']['no_lead']:.3f} | "
            f"{a['prefix_dist']['empty']:.3f} | {top} |"
        )
    lines.append("")
    lines.append("| arm | mean n_generated | median | distinct_rate (run_phase0) |")
    lines.append("|---|---|---|---|")
    for name in order:
        a = res["arms"][name]
        dr = a["run_phase0"].get("qa_fmt_distinct_rate")
        lines.append(
            f"| `{name}` | {a['n_generated_mean']:.2f} | {a['n_generated_median']:.1f} | "
            f"{dr if dr is not None else float('nan'):.4f} |"
        )
    lines.append("")
    lines.append("### 判决量\n")
    v = res["verdict"]
    lines.append(
        f"* 5/6 噪声地板（脚手架率 |Δ|）: {v['noise_floor_scaffold_absdiff']:.4f} "
        f"(SEM {v['noise_floor_scaffold_sem']:.4f}, TV {v['noise_floor_tv']:.4f})"
    )
    lines.append(
        f"* 1/2/3 内部最大 |Δ|（脚手架率）: {v['trained_123_spread_scaffold']:.4f} "
        f"(SEM {v['trained_123_spread_scaffold_sem']:.4f}, TV {v['trained_123_spread_tv']:.4f})"
    )
    lines.append(f"* 主判决（粗分类法）: **{v['decision']}**")
    lines.append("")
    lines.append("| co-primary 指标 | 5/6 噪声地板 | 1/2/3 最大 | 最大所在对 | 倍数 |")
    lines.append("|---|---|---|---|---|")
    for metric, e in v["co_primary"].items():
        ratio = f"{e['ratio']:.2f}x" if e.get("ratio") else "n/a"
        lines.append(
            f"| {metric} | {e['noise_floor']:.4f} | {e['trained_123_spread']:.4f} | "
            f"{e.get('trained_123_spread_pair', '')} | {ratio} |"
        )
    lines.append(f"\n* 副判决（label×前导面 联合 TV）: **{v['decision_co_primary']}**")
    lines.append("")
    lines.append("### 注入 vs 零注入（逐条配对，McNemar 精确 p）\n")
    lines.append("| pair | Δ scaffold | SEM | McNemar p | TV(label) | TV(joint) | TV(first tok) | first-tok 相同率 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for e in v["injecting_vs_zero"]:
        lines.append(
            f"| {e['pair']} | {_fmt_pct(e['scaffold_diff'])} | {e['scaffold_sem']:.4f} | "
            f"{e['scaffold_p']:.3g} | {e['tv']:.4f} | {e['tv_joint']:.4f} | "
            f"{e['tv_first_token']:.4f} | {e['first_token_identical_rate']:.4f} |"
        )
    lines.append("")
    lines.append("### 方向性预测（配对，单侧 p）\n")
    lines.append("| prediction | Δ mean | SEM | one-sided p | n_a_only | n_b_only |")
    lines.append("|---|---|---|---|---|---|")
    for key, st in res["directional"].items():
        lines.append(
            f"| {key} | {st['mean_diff']:+.4f} | {st['sem']:.4f} | {st['one_sided_p']:.3g} | "
            f"{st['n_a_only']} | {st['n_b_only']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--out", default=None, help="write the full analysis JSON here")
    parser.add_argument("--markdown", default=None, help="write the rendered tables here")
    args = parser.parse_args(argv)

    name_to_path = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--arm must be NAME=PATH, got {spec!r}")
        name_to_path[name] = Path(path)

    res = build(name_to_path)
    md = render_markdown(res)
    print(md)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"[round162] wrote {args.out}")
    if args.markdown:
        Path(args.markdown).write_text(md + "\n")
        print(f"[round162] wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
