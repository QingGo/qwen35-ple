"""Round 168 Stage 1.5g: apply the pre-registered rung-3 (row content) rule.

The rule is frozen in ``docs/round-168-stage1.5g-preregistration.md`` section 3
before any aligned number was computed, and is reproduced here verbatim:

    recovery R = probe_raw_rows top-1 / count_trigram top-1
                 (identical candidate set K, identical position set)
    S          = shuffled-row probe top-1

    IF probe_raw_rows top-1 <= max(S, majority) + 0.005
        => ROWS_DEAD
    ELSE IF R >= 0.50
        => ROWS_CARRY_THE_TRIGRAM
    ELSE IF R >= 0.20
        => ROWS_DEGRADED
    ELSE
        => ROWS_NEARLY_DEAD

Section 4.1 adds the binding sensitivity clause: both ``K = 5000`` (primary,
comparable to round-161's ~59%) and ``K = 1000`` must be reported, and if they
disagree **the more conservative verdict wins**.  Section 4.2 adds the
amendment that ``ROWS_DEAD`` may only be declared when *neither* the linear nor
the MLP probe beats ``max(shuffled, majority) + 0.005`` -- a single weak probe
is not evidence that the rows are empty, because a linear probe's R is a LOWER
bound (TD-9).

What this rung answers and what it does NOT: a high R only says the rows carry a
recoverable trigram prior.  It says nothing about usefulness -- that was already
decided against by 1.5c (replacement is THIN/EMPTY) and 1.5e (complementarity
8/8 ABSENT).

Torch-free: this module only reads the JSON written by
``scripts/probe_table_next_token.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CARRY_FLOOR",
    "DEAD_MARGIN",
    "DEGRADED_FLOOR",
    "PRIMARY_K",
    "SENSITIVITY_K",
    "SEVERITY",
    "VerdictInputs",
    "apply_rule",
    "collect_inputs",
    "recovery",
    "render_markdown",
    "top1",
    "verdict_for_k",
]

#: Rule constants, frozen in the pre-registration section 3.  ``0.50`` is the
#: "consistent with round-161's ~0.59" lower edge; ``0.20`` is "still has real
#: signal"; ``0.005`` is the resolvable increment over the shuffled control.
CARRY_FLOOR = 0.50
DEGRADED_FLOOR = 0.20
DEAD_MARGIN = 0.005
#: Section 4.1: report both, decide on the more conservative of the two.
PRIMARY_K = 5000
SENSITIVITY_K = 1000
#: Worst-to-best.  "More conservative" in section 4.1 means the LOWER label.
SEVERITY = {
    "ROWS_DEAD": 0,
    "ROWS_NEARLY_DEAD": 1,
    "ROWS_DEGRADED": 2,
    "ROWS_CARRY_THE_TRIGRAM": 3,
}

#: Section 3.1, frozen: what each outcome licenses.
ACTIONS = {
    "ROWS_CARRY_THE_TRIGRAM": (
        "ladder rung 3 takes the aligned R; the paper's conclusion becomes "
        "'the loss is localised in rung 4 (read-out)', with rungs 1-3 all contentful"
    ),
    "ROWS_DEGRADED": (
        "ladder rung 3 takes R < 0.5; the conclusion is 'the rows themselves already "
        "lost part of it', so rung 4 is not the only bottleneck"
    ),
    "ROWS_NEARLY_DEAD": (
        "rung 3 is empty: this is a POSITIVE rejection of the co-trained small-table "
        "route (even the row content is not there), and it explains why the forty zeros "
        "need not be attributed to the read-out"
    ),
    "ROWS_DEAD": (
        "rung 3 is empty AND the linear probe cannot beat the shuffled control: this is "
        "a POSITIVE rejection of the co-trained small-table route (even the row content "
        "is not there), and it explains why the forty zeros need not be attributed to "
        "the read-out"
    ),
}


@dataclass
class VerdictInputs:
    """Everything the rule reads, per K, plus provenance."""

    per_k: dict[int, dict[str, Any]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


def top1(report: dict[str, Any], name: str, frame: str = "eval") -> float | None:
    """Read ``results[name][frame].top1``, or None when the key is absent."""
    results = report.get("results")
    if not isinstance(results, dict):
        return None
    entry = results.get(name)
    if not isinstance(entry, dict):
        return None
    frame_d = entry.get(frame)
    if not isinstance(frame_d, dict):
        return None
    value = frame_d.get("top1")
    return None if value is None else float(value)


def recovery(probe_report: dict[str, Any]) -> float | None:
    """R = probe_raw_rows top-1 / count_trigram top-1, or None if undefined."""
    probe = top1(probe_report, "probe_raw_rows")
    count = top1(probe_report, "count_trigram")
    if probe is None or count is None or count <= 0.0:
        return None
    return probe / count


def verdict_for_k(probe_report: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen section 3 chain to one probe run.

    The linear and MLP probes are written by the same run, so both live in
    ``probe_report``.  When ``--no-mlp`` was used the MLP keys are absent, the
    section 4.2 amendment cannot be satisfied, and ``ROWS_DEAD`` is withheld
    (the run falls through to the R chain, which is the conservative direction).
    """
    linear = top1(probe_report, "probe_raw_rows")
    count = top1(probe_report, "count_trigram")
    shuffled = top1(probe_report, "control_shuffled_train_rows")
    shuffled_eval = top1(probe_report, "control_shuffled_eval_rows")
    majority = top1(probe_report, "majority_train_prior")
    if linear is None or count is None or majority is None:
        return {"label": "INCOMPLETE", "reason": "probe JSON is missing a required result"}
    if count <= 0.0:
        return {"label": "INCOMPLETE", "reason": "count_trigram top-1 is zero: R undefined"}

    # Section 2 names the shuffled-row control as the mandatory one; the
    # label-permutation control is reported but is not part of the floor.
    if shuffled is None:
        return {"label": "INCOMPLETE", "reason": "shuffled-row control is missing"}
    floor = max(shuffled, majority)

    mlp = top1(probe_report, "probe_mlp_raw_rows")
    mlp_shuf = top1(probe_report, "control_shuffled_train_rows_mlp")
    mlp_maj = top1(probe_report, "majority_train_prior")
    mlp_floor = None
    if mlp is not None and mlp_maj is not None:
        mlp_floor = max(mlp_shuf if mlp_shuf is not None else mlp_maj, mlp_maj)

    linear_dead = linear <= floor + DEAD_MARGIN
    # Section 4.2: DEAD needs BOTH probes at the floor.  A skipped MLP therefore
    # cannot produce DEAD.
    mlp_dead = mlp is not None and mlp_floor is not None and mlp <= mlp_floor + DEAD_MARGIN

    if linear_dead and mlp_dead:
        label = "ROWS_DEAD"
        reason = (
            f"linear {linear:.4f} and MLP {mlp:.4f} are both at or below the shuffled/"
            f"majority floor + {DEAD_MARGIN}"
        )
    else:
        r = linear / count
        if r >= CARRY_FLOOR:
            label = "ROWS_CARRY_THE_TRIGRAM"
            reason = f"R = {r:.4f} >= {CARRY_FLOOR}"
        elif r >= DEGRADED_FLOOR:
            label = "ROWS_DEGRADED"
            reason = f"{DEGRADED_FLOOR} <= R = {r:.4f} < {CARRY_FLOOR}"
        else:
            label = "ROWS_NEARLY_DEAD"
            reason = f"R = {r:.4f} < {DEGRADED_FLOOR}"
        if linear_dead and not mlp_dead:
            if mlp is None:
                reason += (
                    "; the linear probe is at the floor, but the MLP probe was not run, "
                    "and the section 4.2 amendment forbids ROWS_DEAD on one probe alone"
                )
            else:
                reason += (
                    "; the linear probe is at the floor but the MLP probe is not, so by "
                    "the section 4.2 amendment this is NOT ROWS_DEAD (a linear R is a "
                    "lower bound)"
                )

    return {
        "label": label,
        "reason": reason,
        "probe_top1": linear,
        "count_trigram_top1": count,
        "shuffled_rows_top1": shuffled,
        "shuffled_eval_top1": shuffled_eval,
        "majority_top1": majority,
        "floor": floor,
        "recovery_R": linear / count,
        "mlp_top1": mlp,
        "mlp_floor": mlp_floor,
        "linear_at_floor": bool(linear_dead),
        "mlp_at_floor": bool(mlp_dead) if mlp is not None else None,
    }


def collect_inputs(
    per_k: dict[int, dict[str, Any]],
    *,
    provenance: dict[str, Any] | None = None,
) -> VerdictInputs:
    """Normalise ``{K: probe_report}`` into :class:`VerdictInputs`."""
    out = VerdictInputs(per_k={}, provenance=dict(provenance or {}))
    for k, report in per_k.items():
        out.per_k[int(k)] = report
    if PRIMARY_K not in out.per_k:
        out.problems.append(
            f"the primary K={PRIMARY_K} run is missing; section 4.1 requires it"
        )
    if SENSITIVITY_K not in out.per_k:
        out.problems.append(
            f"the sensitivity K={SENSITIVITY_K} run is missing; section 4.1 requires it"
        )
    return out


def apply_rule(inputs: VerdictInputs) -> dict[str, Any]:
    """Decide on the more conservative of the reported K values (section 4.1)."""
    per_k: dict[str, Any] = {}
    for k, report in sorted(inputs.per_k.items()):
        per_k[str(k)] = verdict_for_k(report)

    labels = [v["label"] for v in per_k.values()]
    if not labels or all(label == "INCOMPLETE" for label in labels):
        final = "INCOMPLETE"
        reason = "no K produced a decidable verdict"
    else:
        decided = [v for v in per_k.values() if v["label"] != "INCOMPLETE"]
        worst = min(decided, key=lambda v: SEVERITY[v["label"]])
        # A missing K cannot be compared, so an incomplete panel degrades to the
        # conservative side rather than silently deciding on one K.
        final = worst["label"]
        if any(v["label"] == "INCOMPLETE" for v in per_k.values()):
            final = "INCOMPLETE"
            reason = (
                "at least one reported K was undecidable; section 4.1 compares the K "
                "values, so an incomplete panel cannot fix a verdict"
            )
        elif len({v["label"] for v in decided}) > 1:
            reason = (
                f"the K values disagree ({', '.join(sorted({v['label'] for v in decided}))}); "
                f"section 4.1 takes the more conservative, i.e. {final}"
            )
        else:
            reason = f"all reported K agree: {final}"

    return {
        "verdict": final,
        "reason": reason,
        "agreement": len({v["label"] for v in per_k.values()}) <= 1,
        "action": ACTIONS.get(final, "the panel is incomplete; the rule does not apply"),
        "per_k": per_k,
        "problems": list(inputs.problems),
        "provenance": dict(inputs.provenance),
        "frozen_rule": {
            "carry_floor": CARRY_FLOOR,
            "degraded_floor": DEGRADED_FLOOR,
            "dead_margin": DEAD_MARGIN,
            "primary_k": PRIMARY_K,
            "sensitivity_k": SENSITIVITY_K,
        },
        "scope_note": (
            "R only says the rows carry a recoverable trigram prior. It says nothing "
            "about usefulness: 1.5c found replacement THIN/EMPTY and 1.5e found "
            "complementarity 8/8 ABSENT."
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    """Human-readable report; the JSON is authoritative."""
    lines: list[str] = []
    lines.append("# Stage 1.5g rung-3 (row content), aligned")
    lines.append("")
    lines.append(f"**{result['verdict']}**")
    lines.append("")
    lines.append(f"> {result['reason']}")
    lines.append("")
    lines.append("## Per-K numbers")
    lines.append("")
    lines.append(
        "| K | probe top-1 | count trigram top-1 | R = probe/count | floor "
        "(max(shuffled, majority)) | MLP top-1 | verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for k, v in sorted(result["per_k"].items(), key=lambda kv: int(kv[0]), reverse=True):
        if v.get("label") == "INCOMPLETE":
            lines.append(f"| {k} | - | - | - | - | - | INCOMPLETE: {v.get('reason', '')} |")
            continue
        mlp = v.get("mlp_top1")
        lines.append(
            f"| {k} | {v['probe_top1']:.4f} | {v['count_trigram_top1']:.4f} | "
            f"{v['recovery_R']:.4f} | {v['floor']:.4f} | "
            f"{'-' if mlp is None else f'{mlp:.4f}'} | {v['label']} |"
        )
    lines.append("")
    lines.append(f"**Action (frozen in section 3.1):** {result['action']}")
    lines.append("")
    lines.append(f"* {result['scope_note']}")
    if result["problems"]:
        lines.append("")
        lines.append("## Problems")
        lines.append("")
        for p in result["problems"]:
            lines.append(f"* {p}")
    lines.append("")
    lines.append("## Frozen rule")
    lines.append("")
    rule = result["frozen_rule"]
    lines.append(
        f"* carry floor R >= {rule['carry_floor']}; degraded floor R >= "
        f"{rule['degraded_floor']}; dead margin {rule['dead_margin']}"
    )
    lines.append(
        f"* primary K = {rule['primary_k']}, sensitivity K = {rule['sensitivity_k']}; "
        "disagreement resolves to the more conservative verdict"
    )
    return "\n".join(lines) + "\n"
