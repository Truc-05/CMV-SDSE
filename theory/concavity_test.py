from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np

EPS = 1e-12


@dataclass
class DecayFit:
    name: str
    slope: float
    intercept: float
    r_squared: float
    log_likelihood: float
    n_params: int
    aic: float
    n_points: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConcavityTestResult:
    k_values: np.ndarray
    survival: np.ndarray
    fit_exponential: DecayFit
    fit_power: DecayFit
    preferred_model: str
    delta_aic: float
    tail_exponent_b: float
    per_gate_alpha: float
    n_excluded_saturated: int
    is_heterogeneous: bool
    below_odds_law_fraction: float = float("nan")
    max_gap_from_odds_law: float = float("nan")
    odds_law_is_upper_bound: bool = False
    n_odds_law_points: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_values": np.asarray(self.k_values).tolist(),
            "survival": np.asarray(self.survival).tolist(),
            "fit_exponential": self.fit_exponential.to_dict(),
            "fit_power": self.fit_power.to_dict(),
            "preferred_model": self.preferred_model,
            "delta_aic": self.delta_aic,
            "tail_exponent_b": self.tail_exponent_b,
            "per_gate_alpha": self.per_gate_alpha,
            "n_excluded_saturated": self.n_excluded_saturated,
            "is_heterogeneous": self.is_heterogeneous,
            "below_odds_law_fraction": self.below_odds_law_fraction,
            "max_gap_from_odds_law": self.max_gap_from_odds_law,
            "odds_law_is_upper_bound": self.odds_law_is_upper_bound,
            "n_odds_law_points": self.n_odds_law_points,
            "note": self.note,
        }


def empirical_log_odds(reliability: np.ndarray) -> np.ndarray:
    r = np.asarray(reliability, dtype=float)
    return np.log(r / (1.0 - r))


def compare_to_odds_law(k_values: Sequence[int], reliability: Sequence[float],
                        odds_law_log_odds: Sequence[float],
                        tolerance: float = 1e-9,
                        saturation_eps: float = 1e-9) -> tuple[float, float, bool, int]:
    r = np.asarray(reliability, dtype=float)
    ref = np.asarray(odds_law_log_odds, dtype=float)
    usable = (r > saturation_eps) & (r < 1.0 - saturation_eps) & np.isfinite(ref)
    if usable.sum() == 0:
        return float("nan"), float("nan"), False, 0
    obs = empirical_log_odds(r[usable])
    gap = ref[usable] - obs
    below = float(np.mean(gap >= -tolerance))
    return below, float(np.max(gap)), bool(below >= 1.0), int(usable.sum())


def _weighted_linear_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                         name: str) -> DecayFit:
    w = weights / weights.sum()
    x_bar = np.sum(w * x)
    y_bar = np.sum(w * y)
    sxx = np.sum(w * (x - x_bar) ** 2)
    sxy = np.sum(w * (x - x_bar) * (y - y_bar))
    slope = sxy / sxx if sxx > 0 else float("nan")
    intercept = y_bar - slope * x_bar

    pred = intercept + slope * x
    resid = y - pred
    ss_res = np.sum(weights * resid ** 2)
    ss_tot = np.sum(weights * (y - y_bar) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    n = len(x)
    sigma2 = max(ss_res / max(weights.sum(), 1.0), EPS)
    ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0)
    n_params = 3
    aic = 2 * n_params - 2 * ll

    return DecayFit(
        name=name, slope=float(slope), intercept=float(intercept),
        r_squared=float(r_squared), log_likelihood=float(ll),
        n_params=n_params, aic=float(aic), n_points=int(n),
    )


def run_concavity_test(
    k_values: Sequence[int],
    reliability: Sequence[float],
    n: Sequence[int] | None = None,
    delta_aic_threshold: float = 2.0,
    odds_law_log_odds: Sequence[float] | None = None,
) -> ConcavityTestResult:
    k = np.asarray(k_values, dtype=float)
    r = np.clip(np.asarray(reliability, dtype=float), 0.0, 1.0)
    counts = (np.full_like(k, 1000.0) if n is None
              else np.asarray(n, dtype=float))

    survival = 1.0 - r
    usable = survival > EPS
    n_excluded = int((~usable).sum())

    k_u = k[usable]
    s_u = survival[usable]
    c_u = counts[usable]

    if len(k_u) < 3:
        empty = DecayFit("none", float("nan"), float("nan"), float("nan"),
                         float("nan"), 3, float("nan"), int(len(k_u)))
        return ConcavityTestResult(
            k_values=k_u, survival=s_u, fit_exponential=empty, fit_power=empty,
            preferred_model="undetermined", delta_aic=float("nan"),
            tail_exponent_b=float("nan"), per_gate_alpha=float("nan"),
            n_excluded_saturated=n_excluded, is_heterogeneous=False,
            note=("fewer than 3 unsaturated points; every error was caught at small k "
                  "so the tail carries no information"),
        )

    below, max_gap, is_bound, n_bound = (float("nan"), float("nan"), False, 0)
    if odds_law_log_odds is not None:
        below, max_gap, is_bound, n_bound = compare_to_odds_law(
            k_values, reliability, odds_law_log_odds)

    log_s = np.log(s_u)
    weights = c_u * s_u * (1.0 - s_u) + EPS

    fit_exp = _weighted_linear_fit(k_u, log_s, weights, "exponential_in_k")
    fit_pow = _weighted_linear_fit(np.log(k_u), log_s, weights, "power_in_log_k")

    delta = fit_exp.aic - fit_pow.aic
    if delta > delta_aic_threshold:
        preferred = "power_in_log_k"
    elif delta < -delta_aic_threshold:
        preferred = "exponential_in_k"
    else:
        preferred = "indistinguishable"

    note = ""
    if n_excluded:
        note = (f"{n_excluded} of {len(k)} k points had zero survivors and were dropped; "
                "log survival is undefined there")

    return ConcavityTestResult(
        k_values=k_u,
        survival=s_u,
        fit_exponential=fit_exp,
        fit_power=fit_pow,
        preferred_model=preferred,
        delta_aic=float(delta),
        tail_exponent_b=float(-fit_pow.slope),
        per_gate_alpha=float(np.exp(fit_exp.slope)),
        n_excluded_saturated=n_excluded,
        is_heterogeneous=bool(preferred == "power_in_log_k"),
        below_odds_law_fraction=below,
        max_gap_from_odds_law=max_gap,
        odds_law_is_upper_bound=is_bound,
        n_odds_law_points=n_bound,
        note=note,
    )
