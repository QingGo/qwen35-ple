"""Round 167 Stage 2.2: apply the pre-registered gate-selectivity decision rule.

The rule is frozen in ``docs/round-167-stage2-gate-selectivity-preregistration.md``
section 4 before any number was read, and is reproduced here verbatim:

    O(mode) = gate_open_frac_all_entries  (share of RAW gate entries above 0.5)
    M(mode, arm) = mean of TriviaQA and NQ exact-match under the chat template

    IF anti-no-op preconditions fail            => UNDERPOWERED / DISEASE_NOT_REPRODUCED
    ELSE IF O(per_dim) >= 0.90                  => GATE_STILL_SATURATED
    ELSE IF O(per_dim) <= O(scalar) - 0.20
            AND M(per_dim, real) - M(scalar, real) >= +0.02
            AND M(per_dim, real) - M(per_dim, control) >= +0.02
                                                => GATE_SELECTIVITY_HELPS
    ELSE IF O(per_dim) <= O(scalar) - 0.20
            AND |M(per_dim, real) - M(scalar, real)| < 0.02
                                                => GATE_SELECTIVITY_NO_EFFECT
    ELSE                                        => PARTIAL

The question it answers: round 165/167 found injection to be **always-on and
harmful** -- the scalar gate sits saturated so the memory writes into the
residual stream at every position regardless of whether the row carries
anything.  Is that saturation the *cause*, and does a per-dimension gate that
can actually close fix it?  The three thresholds (0.20 / 0.02 / 0.90) were fixed
in advance.

Torch-free: this module only reads JSON written by the GPU queue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CHAT_SUFFIX",
    "DISEASE_NOT_REPRODUCED_BELOW",
    "DISEASE_OPEN_FRAC_RANGE",
    "MIN_INJECTION_FRACTION",
    "OPEN_FRAC_DROP",
    "SATURATED_FRAC",
    "VerdictInputs",
    "apply_rule",
    "collect_inputs",
    "render_markdown",
]

#: The always-on regime the intervention is aimed at is the *chat* template.
CHAT_SUFFIX = "-chat"
#: Rule constants, frozen in the pre-registration.
OPEN_FRAC_DROP = 0.20
EFFECT_MARGIN = 0.02
SATURATED_FRAC = 0.90
#: Pre-registration section 3: a reader that never injected makes the arm a
#: silent no-op (the round-161 lesson), so the run is VOID rather than null.
MIN_INJECTION_FRACTION = 1e-9
#: The disease round-146 diagnosed, quoted verbatim in the pre-registration
#: section 0: "The current gate saturates open after SFT (mean 0.77-0.98)".
#: So a HIGH open fraction is the disease; a scalar arm below 0.5 means the
#: retrain did not reproduce it and there is nothing to fix.
DISEASE_OPEN_FRAC_RANGE = (0.77, 0.98)
DISEASE_NOT_REPRODUCED_BELOW = 0.5

#: Tasks whose exact match the rule averages.
RULE_TASKS = ("qa_triviaqa_em", "qa_nq_em")


@dataclass
class VerdictInputs:
    """Everything the rule reads, with provenance per field."""

    modes: list[str] = field(default_factory=list)
    open_frac: dict[str, float] = field(default_factory=dict)
    real_mean: dict[str, float] = field(default_factory=dict)
    control_mean: dict[str, float] = field(default_factory=dict)
    per_seed: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    n_seeds: dict[str, int] = field(default_factory=dict)
    injected: dict[str, int] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "modes": self.modes,
            "open_frac": self.open_frac,
            "real_mean": self.real_mean,
            "control_mean": self.control_mean,
            "n_seeds": self.n_seeds,
            "injected": self.injected,
            "per_seed": self.per_seed,
            "missing": self.missing,
        }


def _injection_count(report: dict[str, Any]) -> int | None:
    """Injection count from a run_phase0 report, or ``None`` if absent.

    ``run_phase0`` records it under ``results[i]["ple_injection"]["injected"]``.
    The first version of this loader looked for a top-level ``injected`` key,
    found nothing, defaulted to zero and returned UNDERPOWERED for a run that had
    injected 5,268 times -- a false verdict produced entirely by reading a field
    that does not exist.  Absence must be reported, never defaulted.
    """
    for result in report.get("results") or []:
        injection = result.get("ple_injection")
        if isinstance(injection, dict) and isinstance(injection.get("injected"), int):
            return int(injection["injected"])
    top = report.get("injected")
    return int(top) if isinstance(top, int) else None


def _metric_mean(metrics: dict[str, Any], tasks: tuple[str, ...] = RULE_TASKS) -> float | None:
    values = [metrics[t] for t in tasks if isinstance(metrics.get(t), (int, float))]
    if not values:
        return None
    return float(sum(values) / len(values))


def collect_inputs(
    eval_reports: dict[tuple[str, str, int], dict[str, Any]],
    gate_reports: dict[tuple[str, int], dict[str, Any]],
    *,
    modes: list[str],
    seeds: list[int],
) -> VerdictInputs:
    """Assemble the rule's inputs from the queue's raw JSON payloads.

    ``eval_reports`` is keyed by ``(mode, arm, seed)`` and must hold the **chat**
    report; ``gate_reports`` is keyed by ``(mode, seed)``.
    """
    out = VerdictInputs(modes=list(modes), n_seeds={m: 0 for m in modes})
    for mode in modes:
        gate_fracs: list[float] = []
        injections: list[int] = []
        for seed in seeds:
            gate = gate_reports.get((mode, seed))
            if gate is None:
                out.missing.append(f"gate:{mode}:seed{seed}")
                continue
            aggregate = gate.get("aggregate") or {}
            fracs = [
                float(row["gate_open_frac_all_entries"])
                for row in aggregate.values()
                if isinstance(row.get("gate_open_frac_all_entries"), (int, float))
            ]
            if fracs:
                gate_fracs.append(sum(fracs) / len(fracs))
        out.open_frac[mode] = (
            float(sum(gate_fracs) / len(gate_fracs)) if gate_fracs else float("nan")
        )

        for arm in ("real", "control"):
            values: list[float] = []
            rows: list[dict[str, Any]] = []
            for seed in seeds:
                report = eval_reports.get((mode, arm, seed))
                if report is None:
                    out.missing.append(f"eval:{mode}:{arm}:seed{seed}")
                    continue
                summary = report.get("summary") or {}
                details = (summary.get(arm) or {}).get("details") or []
                if not details:
                    out.missing.append(f"eval:{mode}:{arm}:seed{seed}")
                    continue
                count = _injection_count(report)
                if count is None:
                    out.missing.append(f"injection:{mode}:{arm}:seed{seed}")
                else:
                    injections.append(count)
                metrics = (details[0].get("qa_exact") or {}).get("metrics") or {}
                value = _metric_mean(metrics)
                if value is None:
                    out.missing.append(f"metrics:{mode}:{arm}:seed{seed}")
                    continue
                values.append(value)
                rows.append({
                    "seed": seed,
                    "rule_mean": value,
                    "triviaqa_em": metrics.get("qa_triviaqa_em"),
                    "nq_em": metrics.get("qa_nq_em"),
                })
            target = out.real_mean if arm == "real" else out.control_mean
            target[mode] = float(sum(values) / len(values)) if values else float("nan")
            out.per_seed[f"{mode}:{arm}"] = rows
            if values:
                out.n_seeds[mode] = max(out.n_seeds.get(mode, 0), len(values))
        out.injected[mode] = sum(injections)
    return out


def apply_rule(inputs: VerdictInputs) -> dict[str, Any]:
    """The frozen rule.  Returns the verdict plus every quantity it read."""
    modes = inputs.modes
    if "scalar" not in modes or "per_dim" not in modes:
        raise ValueError("the rule compares the scalar and per_dim arms")
    o_scalar = inputs.open_frac.get("scalar", float("nan"))
    o_per_dim = inputs.open_frac.get("per_dim", float("nan"))
    quantities = {
        "O": dict(inputs.open_frac),
        "M_real": dict(inputs.real_mean),
        "M_control": dict(inputs.control_mean),
        "injected": dict(inputs.injected),
        "n_seeds": dict(inputs.n_seeds),
        "thresholds": {
            "open_frac_drop": OPEN_FRAC_DROP,
            "effect_margin": EFFECT_MARGIN,
            "saturated_frac": SATURATED_FRAC,
        },
        "missing": list(inputs.missing),
    }

    # --- anti-no-op preconditions (section 3) -----------------------------
    if inputs.missing:
        return {
            "label": "INCOMPLETE",
            "reason": f"inputs missing: {', '.join(inputs.missing[:6])}",
            **quantities,
        }
    no_injection = [m for m in modes if inputs.injected.get(m, 0) <= 0]
    if no_injection:
        return {
            "label": "UNDERPOWERED",
            "reason": (
                f"no injection recorded for {', '.join(no_injection)}: the arm is a "
                "silent no-op, so real==control by construction (round-161)"
            ),
            **quantities,
        }
    if o_scalar < DISEASE_NOT_REPRODUCED_BELOW:
        return {
            "label": "DISEASE_NOT_REPRODUCED",
            "reason": (
                f"the scalar arm's gate_open_frac_all_entries is {o_scalar:.4f} < "
                f"{DISEASE_NOT_REPRODUCED_BELOW}: round-146's disease is saturation "
                f"at {DISEASE_OPEN_FRAC_RANGE[0]}-{DISEASE_OPEN_FRAC_RANGE[1]}, so "
                "this retrain does not reproduce it and there is nothing to fix"
            ),
            **quantities,
        }

    # --- the rule itself (section 4) -------------------------------------
    if not (o_per_dim < SATURATED_FRAC):
        return {
            "label": "GATE_STILL_SATURATED",
            "reason": (
                f"per_dim gate_open_frac_all_entries = {o_per_dim:.4f} >= "
                f"{SATURATED_FRAC}: a per-dimension gate is just as open, so the "
                "gate is not the mechanism"
            ),
            **quantities,
        }

    m_per_real = inputs.real_mean.get("per_dim", float("nan"))
    m_sca_real = inputs.real_mean.get("scalar", float("nan"))
    m_per_ctrl = inputs.control_mean.get("per_dim", float("nan"))
    selective = o_per_dim <= o_scalar - OPEN_FRAC_DROP
    gains_real = m_per_real - m_sca_real
    gains_ctrl = m_per_real - m_per_ctrl
    quantities["selectivity"] = {
        "open_frac_drop": o_scalar - o_per_dim,
        "is_selective": bool(selective),
        "gain_vs_scalar": float(gains_real),
        "gain_vs_control": float(gains_ctrl),
    }

    if selective and gains_real >= EFFECT_MARGIN and gains_ctrl >= EFFECT_MARGIN:
        return {
            "label": "GATE_SELECTIVITY_HELPS",
            "reason": (
                f"O(per_dim) {o_per_dim:.4f} <= O(scalar) {o_scalar:.4f} - "
                f"{OPEN_FRAC_DROP}, and the chat-template TriviaQA/NQ mean gains "
                f"{gains_real:+.4f} over scalar and {gains_ctrl:+.4f} over its own "
                "control: the selectivity restores a CONTENT-RELATED gain"
            ),
            **quantities,
        }
    if selective and abs(gains_real) < EFFECT_MARGIN:
        return {
            "label": "GATE_SELECTIVITY_NO_EFFECT",
            "reason": (
                f"O(per_dim) {o_per_dim:.4f} <= O(scalar) {o_scalar:.4f} - "
                f"{OPEN_FRAC_DROP}, but the metric moves by only {gains_real:+.4f} "
                f"(< {EFFECT_MARGIN}): the gate is genuinely more selective and it "
                "does not matter"
            ),
            **quantities,
        }
    return {
        "label": "PARTIAL",
        "reason": (
            f"no branch matched: O(per_dim) = {o_per_dim:.4f}, O(scalar) = "
            f"{o_scalar:.4f}, gain vs scalar {gains_real:+.4f}, gain vs control "
            f"{gains_ctrl:+.4f}.  Report arm by arm; do not merge into one story"
        ),
        **quantities,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = ["# Stage 2.2 gate selectivity — pre-registered verdict", ""]
    lines.append(f"**{result['label']}**")
    lines.append("")
    lines.append(f"> {result['reason']}")
    lines.append("")
    lines.append("## Quantities the rule read")
    lines.append("")
    lines.append("| mode | O (gate_open_frac_all_entries) | M(chat, triviaqa+nq) real | "
                 "M control | seeds | injected |")
    lines.append("|---|---|---|---|---|---|")
    for mode in result["O"]:
        lines.append(
            f"| {mode} | {result['O'][mode]:.4f} | {result['M_real'][mode]:.4f} | "
            f"{result['M_control'][mode]:.4f} | {result['n_seeds'].get(mode, 0)} | "
            f"{result['injected'].get(mode, 0)} |"
        )
    sel = result.get("selectivity")
    if sel:
        lines.append("")
        lines.append(f"* open-fraction drop: {sel['open_frac_drop']:+.4f} "
                     f"(selective: {sel['is_selective']})")
        lines.append(f"* gain vs scalar: {sel['gain_vs_scalar']:+.4f}")
        lines.append(f"* gain vs the arm's own control: {sel['gain_vs_control']:+.4f}")
    if result["missing"]:
        lines.append("")
        lines.append(f"* missing inputs: {', '.join(result['missing'])}")
    lines.append("")
    lines.append("## Thresholds (frozen in the pre-registration)")
    lines.append("")
    for key, value in result["thresholds"].items():
        lines.append(f"* `{key}` = {value}")
    lines.append("")
    return "\n".join(lines)
