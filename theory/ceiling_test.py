from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import betaln

EPS = 1e-12


def survival_beta(k: np.ndarray, a: float, b: float) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    return np.exp(betaln(a + k, b) - betaln(a, b))


def survival_mixture(k: np.ndarray, pi0: float, a: float, b: float) -> np.ndarray:
    return pi0 + (1.0 - pi0) * survival_beta(k, a, b)


def reliability_mixture(k: np.ndarray, pi0: float, a: float, b: float) -> np.ndarray:
    return 1.0 - survival_mixture(k, pi0, a, b)


@dataclass
class MixtureFit:
    pi0: float
    a: float
    b: float
    log_likelihood: float
    n_points: int
    n_observations: int
    converged: bool
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CeilingTestResult:
    fit_full: MixtureFit
    fit_null: MixtureFit
    blind_spot_mass: float
    blind_spot_ci: tuple[float, float]
    likelihood_ratio: float
    p_value: float
    is_ceiling_detected: bool
    tail_exponent_b: float
    asymptotic_reliability: float
    min_gates_used: int
    k_max_observed: int
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fit_full"] = self.fit_full.to_dict()
        payload["fit_null"] = self.fit_null.to_dict()
        return payload


def _counts(k_values: Sequence[int], reliability: Sequence[float],
            n: Sequence[int] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = np.asarray(k_values, dtype=float)
    r = np.clip(np.asarray(reliability, dtype=float), 0.0, 1.0)
    if n is None:
        counts = np.full_like(k, 1000.0)
    else:
        counts = np.asarray(n, dtype=float)
    caught = np.rint(r * counts)
    return k, caught, counts


def _neg_log_likelihood(params: np.ndarray, k: np.ndarray, caught: np.ndarray,
                        counts: np.ndarray, free_pi0: bool) -> float:
    if free_pi0:
        logit_pi0, log_a, log_b = params
        logit_pi0 = float(np.clip(logit_pi0, -60.0, 60.0))
        pi0 = 1.0 / (1.0 + np.exp(-logit_pi0))
    else:
        log_a, log_b = params
        pi0 = 0.0
    # Bound the concentration: past a+b ~ 1e6 the Beta-survival ratio loses all precision
    # and the likelihood degenerates (an optimiser will exploit the overflow region).
    a = np.exp(np.clip(log_a, -30.0, 13.8))
    b = np.exp(np.clip(log_b, -30.0, 13.8))
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return np.inf
    r = reliability_mixture(k, pi0, a, b)
    r = np.clip(r, EPS, 1.0 - EPS)
    ll = np.sum(caught * np.log(r) + (counts - caught) * np.log1p(-r))
    return -ll if np.isfinite(ll) else np.inf


def _fit(k: np.ndarray, caught: np.ndarray, counts: np.ndarray,
         free_pi0: bool) -> MixtureFit:
    best = None
    starts = [(-2.0, 0.0, 0.0), (-4.0, 0.7, -0.7), (0.0, -0.7, 0.7), (-1.0, 1.1, 0.4)]
    for start in starts:
        x0 = np.array(start if free_pi0 else start[1:], dtype=float)
        res = minimize(
            _neg_log_likelihood, x0, args=(k, caught, counts, free_pi0),
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000, "maxfev": 20000},
        )
        if best is None or res.fun < best.fun:
            best = res

    if free_pi0:
        pi0 = float(1.0 / (1.0 + np.exp(-float(np.clip(best.x[0], -60.0, 60.0)))))
        a, b = float(np.exp(best.x[1])), float(np.exp(best.x[2]))
    else:
        pi0 = 0.0
        a, b = float(np.exp(best.x[0])), float(np.exp(best.x[1]))

    return MixtureFit(
        pi0=pi0, a=a, b=b,
        log_likelihood=float(-best.fun),
        n_points=int(len(k)),
        n_observations=int(counts.sum()),
        converged=bool(best.success),
        message=str(best.message),
    )


def _profile_ci(k: np.ndarray, caught: np.ndarray, counts: np.ndarray,
                fit: MixtureFit, level: float = 0.95) -> tuple[float, float]:
    cutoff = fit.log_likelihood - 0.5 * stats.chi2.ppf(level, 1)
    grid = np.unique(np.clip(np.concatenate([
        np.linspace(0.0, 1.0, 201),
        np.linspace(max(fit.pi0 - 0.05, 0.0), min(fit.pi0 + 0.05, 1.0), 101),
    ]), 0.0, 1.0 - 1e-9))

    inside = []
    for pi0 in grid:
        def nll(params):
            a, b = np.exp(np.clip(params, -30.0, 13.8))
            if a <= 0 or b <= 0:
                return np.inf
            r = np.clip(reliability_mixture(k, pi0, a, b), EPS, 1.0 - EPS)
            ll = np.sum(caught * np.log(r) + (counts - caught) * np.log1p(-r))
            return -ll if np.isfinite(ll) else np.inf

        res = minimize(nll, np.array([np.log(fit.a), np.log(fit.b)]),
                       method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 4000})
        if -res.fun >= cutoff:
            inside.append(pi0)
    if not inside:
        return (float("nan"), float("nan"))
    return (float(min(inside)), float(max(inside)))


def run_ceiling_test(
    k_values: Sequence[int],
    reliability: Sequence[float],
    n: Sequence[int] | None = None,
    min_gates_for_fit: int = 10,
    alpha_level: float = 0.05,
    compute_ci: bool = True,
) -> CeilingTestResult:
    k, caught, counts = _counts(k_values, reliability, n)

    if len(k) < 4:
        empty = MixtureFit(float("nan"), float("nan"), float("nan"), float("nan"),
                           len(k), int(counts.sum()), False, "too few k points")
        return CeilingTestResult(
            fit_full=empty, fit_null=empty, blind_spot_mass=float("nan"),
            blind_spot_ci=(float("nan"), float("nan")), likelihood_ratio=float("nan"),
            p_value=float("nan"), is_ceiling_detected=False,
            tail_exponent_b=float("nan"), asymptotic_reliability=float("nan"),
            min_gates_used=min_gates_for_fit, k_max_observed=int(k.max()) if len(k) else 0,
            note="need at least 4 distinct k values to identify the mixture",
        )

    fit_full = _fit(k, caught, counts, free_pi0=True)
    fit_null = _fit(k, caught, counts, free_pi0=False)

    lr = 2.0 * (fit_full.log_likelihood - fit_null.log_likelihood)
    lr = max(lr, 0.0)
    p_value = 0.5 * float(stats.chi2.sf(lr, 1)) if lr > 0 else 1.0

    ci = _profile_ci(k, caught, counts, fit_full) if compute_ci else (float("nan"), float("nan"))

    detected = bool(
        p_value < alpha_level
        and int(k.max()) >= min_gates_for_fit
        and (np.isnan(ci[0]) or ci[0] > 0.0)
    )

    note = ""
    if int(k.max()) < min_gates_for_fit:
        note = (f"largest k observed is {int(k.max())}, below the required "
                f"{min_gates_for_fit}; ceiling not declared regardless of fit")

    return CeilingTestResult(
        fit_full=fit_full,
        fit_null=fit_null,
        blind_spot_mass=fit_full.pi0,
        blind_spot_ci=ci,
        likelihood_ratio=lr,
        p_value=p_value,
        is_ceiling_detected=detected,
        tail_exponent_b=fit_full.b,
        asymptotic_reliability=1.0 - fit_full.pi0,
        min_gates_used=min_gates_for_fit,
        k_max_observed=int(k.max()),
        note=note,
    )
