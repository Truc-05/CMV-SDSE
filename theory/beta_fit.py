from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats
from scipy.optimize import minimize


@dataclass
class BetaFitResult:
    a: float
    b: float
    method: str
    n_obs: int
    log_likelihood: float


def _sanitize(alphas: Sequence[float], eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(alphas, dtype=float)
    arr = arr[~np.isnan(arr)]
    arr = np.clip(arr, eps, 1.0 - eps)
    return arr


def fit_beta_moments(alphas: Sequence[float]) -> BetaFitResult:
    arr = _sanitize(alphas)
    mean = arr.mean()
    var = arr.var(ddof=1) if len(arr) > 1 else 1e-6
    var = max(var, 1e-8)
    common = mean * (1 - mean) / var - 1
    a = max(mean * common, 1e-3)
    b = max((1 - mean) * common, 1e-3)
    ll = float(np.sum(stats.beta.logpdf(arr, a, b)))
    return BetaFitResult(a=a, b=b, method="moments", n_obs=len(arr), log_likelihood=ll)


def fit_beta_mle(alphas: Sequence[float]) -> BetaFitResult:
    arr = _sanitize(alphas)
    moments = fit_beta_moments(arr)

    def neg_log_likelihood(params):
        a, b = params
        if a <= 0 or b <= 0:
            return np.inf
        return -np.sum(stats.beta.logpdf(arr, a, b))

    res = minimize(
        neg_log_likelihood,
        x0=[moments.a, moments.b],
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000},
    )
    a, b = res.x
    a = max(a, 1e-3)
    b = max(b, 1e-3)
    ll = float(-res.fun)
    return BetaFitResult(a=a, b=b, method="mle", n_obs=len(arr), log_likelihood=ll)


def rho_v_from_beta(fit: BetaFitResult) -> float:
    return 1.0 / (fit.a + fit.b + 1.0)


def expected_reliability_beta(k: np.ndarray, fit: BetaFitResult) -> np.ndarray:
    from scipy.special import betaln

    k = np.asarray(k, dtype=float)
    log_e_alpha_k = betaln(fit.a + k, fit.b) - betaln(fit.a, fit.b)
    e_alpha_k = np.exp(log_e_alpha_k)
    return 1.0 - e_alpha_k