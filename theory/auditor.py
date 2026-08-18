"""Cascade Auditor: budget-aware fusion diagnosis for a pool of correlated verifiers.

Point a verifier pool at the auditor (verdict data, or a small probe) and it returns the
fusion-efficiency picture and a recommendation among four actions:

    SCALE      call the SAME verifier again        cheap, but useless once its self-n_eff saturates
    DIVERSIFY  add a complementary verifier         the only in-pool way to gain independent evidence
    STOP       accept; more verification not worth its cost   (Theorem 4)
    ESCALATE   go outside the pool (stronger/human) the ONLY way past the blind-spot ceiling (Theorem 2)

Core quantities (fusion language):
  n_eff              effective number of INDEPENDENT verifiers the pool currently delivers
  Verification Debt  D = calls - n_eff : calls paid that bought no independent evidence
  Verification Efficiency  eta = n_eff / calls in [0,1]
  correlation tax    solo_catch(v) - marginal_catch(v | survivors) : value lost to redundancy
  blind-spot ceiling 1 - pi0 : reliability no in-pool cascade can exceed (Theorem 2)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional

import numpy as np

from theory.betabinom_mixture import fit_mixture
from theory.cascade_design import (
    solo_catch, marginal_greedy, cascade_reliability, budgeted_greedy_stop,
    reliability_ceiling,
)


@dataclass
class VerifierReport:
    name: str
    solo_catch: float
    self_rho_v: float          # within-verifier verdict correlation (repetition redundancy)
    self_n_eff_at: float       # effective independent calls if you repeat it `repeat_depth` times
    repeat_depth: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AuditReport:
    verifiers: list
    per_verifier: list
    optimal_order: list
    reliability_curve: list
    blind_spot_ceiling: float
    lam: float
    target_reliability: Optional[float]
    k_star: int
    reliability_at_k_star: float
    calls_if_naive: int
    verification_debt: float
    verification_efficiency: float
    recommendation: str
    rationale: str
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["per_verifier"] = [v.to_dict() if hasattr(v, "to_dict") else v for v in self.per_verifier]
        return d


def _independent_neff(reliability: float, per_call_catch: float) -> float:
    """How many INDEPENDENT verifiers of catch p reach this reliability: R = 1-(1-p)^neff."""
    p = min(max(per_call_catch, 1e-6), 1 - 1e-6)
    R = min(max(reliability, 1e-9), 1 - 1e-9)
    return float(np.log(1 - R) / np.log(1 - p))


def audit_pool(
    verdict_data: dict,
    verifiers: list,
    costs: Optional[list] = None,
    lam: float = 0.03,
    target_reliability: Optional[float] = None,
    repeat_depth: int = 50,
) -> AuditReport:
    """verdict_data[item][verifier] = list of accept booleans (True=accept a true error)."""
    items = [it for it in verdict_data if all(verdict_data[it].get(v) for v in verifiers)]
    m = len(verifiers)
    costs = [1.0] * m if costs is None else list(costs)

    # per-item accept matrix (marginal rates)
    A = np.array([[float(np.mean(verdict_data[it][v])) for v in verifiers] for it in items])

    # per-verifier self-repetition redundancy (SCALE diagnosis) via Beta-Binomial
    per_v = []
    for j, v in enumerate(verifiers):
        counts = [(sum(verdict_data[it][v]), len(verdict_data[it][v])) for it in items]
        mm = np.array([c[0] for c in counts]); KK = np.array([c[1] for c in counts])
        fit = fit_mixture(mm, KK, compute_ci=False)
        deff = 1.0 + (repeat_depth - 1) * fit.rho_v
        per_v.append(VerifierReport(
            name=v, solo_catch=float(1 - A[:, j].mean()),
            self_rho_v=float(fit.rho_v),
            self_n_eff_at=float(repeat_depth / deff), repeat_depth=repeat_depth))

    order = marginal_greedy(A)
    rel = cascade_reliability(A, order)
    ceiling = reliability_ceiling(A)
    stop = budgeted_greedy_stop(A, lam=lam, costs=costs)
    k_star = stop["stop_depth"]
    rel_star = stop["stop_reliability"]

    # verification debt at the naive operating point (use the whole pool)
    calls_naive = m
    pbar = float(np.mean([per_v[j].solo_catch for j in order[:max(k_star, 1)]]))
    neff = _independent_neff(rel_star, max(pbar, 1e-6)) if rel_star > 0 else 0.0
    debt = max(calls_naive - neff, 0.0)
    eff = neff / calls_naive if calls_naive else float("nan")

    # 4-way recommendation
    best_marginal_util = max(stop["marginal_utilities"]) if stop["marginal_utilities"] else 0.0
    remaining_after_stop = m - k_star
    if target_reliability is not None and ceiling < target_reliability - 1e-9:
        rec = "ESCALATE"
        rationale = (f"pool blind-spot ceiling {ceiling:.2f} < target {target_reliability:.2f}; "
                     f"Theorem 2 says no in-pool cascade can reach the target — add a decorrelated "
                     f"or stronger out-of-pool source (human / different modality).")
    elif target_reliability is not None and rel_star >= target_reliability - 1e-9:
        rec = "STOP"
        rationale = (f"target {target_reliability:.2f} met at k*={k_star} (reliability {rel_star:.2f}); "
                     f"further verification is below its cost (Theorem 4).")
    elif remaining_after_stop > 0 and rel_star < ceiling - 0.01:
        rec = "DIVERSIFY"
        rationale = (f"in-pool complementary verifiers still pay: adding {verifiers[order[k_star]]} "
                     f"raises reliability toward the ceiling {ceiling:.2f}.")
    else:
        rec = "STOP"
        rationale = (f"marginal value of every remaining verifier is below cost λ={lam}; "
                     f"stop at k*={k_star} (reliability {rel_star:.2f}).")

    warn = ""
    if remaining_after_stop > 0:
        warn = (f"adding the remaining {remaining_after_stop} verifier(s) / more same-model calls is "
                f"statistically useless here: marginal reliability < λ·cost.")

    return AuditReport(
        verifiers=list(verifiers),
        per_verifier=per_v,
        optimal_order=[verifiers[j] for j in order],
        reliability_curve=[round(float(x), 4) for x in rel],
        blind_spot_ceiling=round(float(ceiling), 4),
        lam=lam, target_reliability=target_reliability,
        k_star=k_star, reliability_at_k_star=round(float(rel_star), 4),
        calls_if_naive=calls_naive,
        verification_debt=round(float(debt), 2),
        verification_efficiency=round(float(eff), 3),
        recommendation=rec, rationale=rationale, warning=warn,
    )


def format_report(r: AuditReport) -> str:
    lines = ["=" * 68, "CASCADE AUDITOR — budget-aware fusion of correlated verifiers", "=" * 68]
    lines.append(f"{'verifier':14s} {'solo':>6s} {'self_rho_v':>11s} {'self_n_eff@'+str(r.per_verifier[0].repeat_depth):>13s}")
    for v in r.per_verifier:
        lines.append(f"{v.name:14s} {v.solo_catch:>6.3f} {v.self_rho_v:>11.3f} {v.self_n_eff_at:>13.2f}")
    lines.append("-" * 68)
    lines.append(f"optimal ordering        : {' > '.join(r.optimal_order)}")
    lines.append(f"reliability curve       : {r.reliability_curve}")
    lines.append(f"blind-spot ceiling 1-π0 : {r.blind_spot_ceiling}")
    lines.append(f"optimal depth k*        : {r.k_star}  (reliability {r.reliability_at_k_star}, λ={r.lam})")
    lines.append(f"Verification Debt       : {r.verification_debt} calls bought no independent evidence")
    lines.append(f"Verification Efficiency : {r.verification_efficiency}  (n_eff / calls)")
    lines.append(f">>> RECOMMENDATION: {r.recommendation}")
    lines.append(f"    {r.rationale}")
    if r.warning:
        lines.append(f"    ⚠ {r.warning}")
    lines.append("=" * 68)
    return "\n".join(lines)
