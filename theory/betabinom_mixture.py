"""Item-level Beta-Binomial ceiling-mixture estimator for correlated verifier cascades.

This is the inferential backbone of the project. It replaces the earlier curve-fitting
tests (``concavity_test`` / ``ceiling_test``), which summed a binomial log-likelihood
across the k points of the *cumulative* reliability curve. Because every item passes
through all k gates, those k points come from the same items and are almost perfectly
autocorrelated; summing a per-k binomial likelihood treats one item measured k times as
k independent items and inflates the effective sample size by ~k. The likelihood-ratio
statistics and profile intervals built on it are anticonservative (see
``theory/simulation.py`` for the calibration evidence).

The fix is to model the data at the level it was actually generated. Under the theory's
exchangeability assumption (Aksu 2026b, de Finetti representation) the k gate verdicts on
one item are conditionally i.i.d. given a latent per-item accept probability ``alpha_i``,
so the *order* of verdicts carries no information and the per-item sufficient statistic is
the pair ``(m_i, K_i)`` -- the number of "unwanted" verdicts among the ``K_i`` decided
gates. For an error item, an unwanted verdict is a *false accept* (a miss); for a clean
item it is a *false alarm* (a reject). The three nested theory models become:

    M1  Binomial(K, mu)                          independent, homogeneous gates: S(k)=mu^k
    M2  Beta-Binomial(K, a, b)                   heterogeneous, no ceiling; rho_v=1/(a+b+1)
    M3  pi0 * 1[m=K] + (1-pi0) * BetaBinom(K,a,b) blind-spot / false-alarm ceiling mass pi0

Each item contributes exactly one likelihood term, so the effective sample size is the
number of items, not items x gates. The intraclass (intra-item) correlation of the gate
verdicts is exactly the Beta-Binomial correlation ``rho_v = 1/(a+b+1)``, which is also the
design-effect parameter, so H1-H4 all read off a single fit:

    H1 (non-independence / concavity)  LRT of M2 vs M1  (overdispersion, rho_v > 0)
    H2 (dependence ladder)             compare rho_v across pairing pools
    H3 (ceiling)                       LRT of M3 vs M2  (excess mass at m=K)
    H4 (effective gates)               n_eff(K) = K / (1 + (K-1) rho_v)

Both LRTs test a parameter on the boundary of the space (rho_v = 0 is a+b -> inf; pi0 = 0
is the edge of [0,1]), so the null distribution of the statistic is the 50:50 mixture of a
point mass at 0 and chi^2_1 (Chernoff 1954; Self & Liang 1987):  p = 0.5 * P(chi^2_1 > LR).
Confidence intervals use profile likelihood, not Wald, because the parameters sit near a
boundary where the quadratic approximation is poor.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Optional, Sequence

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, logsumexp

EPS = 1e-12
# Concentration a+b is capped so rho_v >= ~1e-6; distinguishing rho_v below that from 0 is
# neither meaningful nor numerically stable (betaln loses precision for huge a,b).
MAX_CONCENTRATION = 1e6
_LOG_S_MAX = float(np.log(MAX_CONCENTRATION))
_CHI2_1_95 = float(stats.chi2.ppf(0.95, 1))


# --------------------------------------------------------------------------------------
# probability mass functions (log scale)
# --------------------------------------------------------------------------------------
def _log_choose(K: np.ndarray, m: np.ndarray) -> np.ndarray:
    return gammaln(K + 1.0) - gammaln(m + 1.0) - gammaln(K - m + 1.0)


def binom_logpmf(m: np.ndarray, K: np.ndarray, mu: float) -> np.ndarray:
    mu = float(np.clip(mu, EPS, 1.0 - EPS))
    return _log_choose(K, m) + m * np.log(mu) + (K - m) * np.log1p(-mu)


def betabinom_logpmf(m: np.ndarray, K: np.ndarray, a: float, b: float) -> np.ndarray:
    # As the concentration a+b grows the Beta-Binomial converges to Binomial(K, a/(a+b)).
    # Past ~1e6 the betaln differences lose all floating-point precision and the "pmf"
    # numerically degenerates to 1 for every m (log-pmf -> 0), which an optimiser will
    # happily exploit. Fall back to the exact limiting Binomial there.
    if a + b > MAX_CONCENTRATION:
        return binom_logpmf(m, K, a / (a + b))
    return _log_choose(K, m) + betaln(a + m, b + K - m) - betaln(a, b)


def _ceiling_logpmf(m: np.ndarray, K: np.ndarray, pi0: float, a: float, b: float) -> np.ndarray:
    """log[ pi0 * 1[m==K] + (1-pi0) * BetaBinom(m|K,a,b) ]."""
    pi0 = float(np.clip(pi0, 0.0, 1.0 - EPS))
    bb = betabinom_logpmf(m, K, a, b) + np.log1p(-pi0)
    at_ceiling = np.isclose(m, K)
    if not np.any(at_ceiling):
        return bb
    out = bb.copy()
    ceil_term = np.full(np.sum(at_ceiling), np.log(max(pi0, EPS)))
    out[at_ceiling] = np.logaddexp(bb[at_ceiling], ceil_term)
    return out


# --------------------------------------------------------------------------------------
# survival curve implied by a fit (for plotting / odds-law comparison)
# --------------------------------------------------------------------------------------
def survival_from_params(k: np.ndarray, pi0: float, a: float, b: float) -> np.ndarray:
    """S(k) = P(a random item's first k gates all give the unwanted verdict).

    S(k) = pi0 + (1-pi0) * E_Beta[alpha^k],  E_Beta[alpha^k] = B(a+k,b)/B(a,b).
    """
    k = np.asarray(k, dtype=float)
    e_alpha_k = np.exp(betaln(a + k, b) - betaln(a, b))
    return pi0 + (1.0 - pi0) * e_alpha_k


# --------------------------------------------------------------------------------------
# parameterisation helpers  (mean/concentration is far better conditioned than (a,b))
# --------------------------------------------------------------------------------------
def _ab_from(logit_mu: float, log_s: float) -> tuple[float, float]:
    mu = 1.0 / (1.0 + np.exp(-np.clip(logit_mu, -60.0, 60.0)))
    s = np.exp(np.clip(log_s, -30.0, _LOG_S_MAX))  # concentration a+b, capped for stability
    return mu * s, (1.0 - mu) * s


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


# --------------------------------------------------------------------------------------
# fit results
# --------------------------------------------------------------------------------------
@dataclass
class ModelFit:
    name: str
    n_params: int
    log_likelihood: float
    aic: float
    params: dict[str, float]
    converged: bool
    n_items: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MixtureInference:
    n_items: int
    n_gates_mean: float
    k_max: int

    fit_binomial: ModelFit          # M1
    fit_betabinom: ModelFit         # M2
    fit_ceiling: ModelFit           # M3

    # H1 -- non-independence / concavity (M2 vs M1)
    lr_heterogeneity: float
    p_heterogeneity: float
    delta_aic: float                # AIC(M1) - AIC(M2); >0 favours the heterogeneous (power) law
    preferred_decay: str
    is_heterogeneous: bool

    # dispersion / design effect (from M2)
    mu_hat: float                   # mean unwanted-verdict rate
    a_hat: float
    b_hat: float
    rho_v: float
    rho_v_ci: tuple[float, float]
    design_effect_at_k_max: float
    n_eff_at_k_max: float

    # H3 -- ceiling (M3 vs M2)
    blind_spot_mass: float          # pi0
    blind_spot_ci: tuple[float, float]
    lr_ceiling: float
    p_ceiling: float
    ceiling_detected: bool
    min_gates_for_fit: int

    # odds-law upper-bound check, derived from the fitted curve
    below_odds_law_fraction: float
    odds_law_is_upper_bound: bool

    note: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("fit_binomial", "fit_betabinom", "fit_ceiling"):
            payload[key] = getattr(self, key).to_dict()
        return payload


# --------------------------------------------------------------------------------------
# individual model fits
# --------------------------------------------------------------------------------------
def _fit_binomial(m: np.ndarray, K: np.ndarray) -> ModelFit:
    # closed-form MLE for the pooled mean
    mu = float(np.clip(m.sum() / max(K.sum(), EPS), EPS, 1.0 - EPS))
    ll = float(np.sum(binom_logpmf(m, K, mu)))
    return ModelFit("binomial", 1, ll, 2 * 1 - 2 * ll, {"mu": mu}, True, len(m))


def _nll_betabinom(theta: np.ndarray, m: np.ndarray, K: np.ndarray) -> float:
    a, b = _ab_from(theta[0], theta[1])
    if a <= 0 or b <= 0 or not np.isfinite(a) or not np.isfinite(b):
        return np.inf
    ll = float(np.sum(betabinom_logpmf(m, K, a, b)))
    return -ll if np.isfinite(ll) else np.inf


def _fit_betabinom(m: np.ndarray, K: np.ndarray, mu_start: float) -> ModelFit:
    logit_mu0 = float(np.log(mu_start / (1.0 - mu_start)))
    best = None
    for log_s0 in (-1.0, 0.0, 1.5, 3.0, 5.0):
        x0 = np.array([logit_mu0, log_s0], dtype=float)
        res = minimize(_nll_betabinom, x0, args=(m, K), method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000, "maxfev": 20000})
        if best is None or res.fun < best.fun:
            best = res
    a, b = _ab_from(best.x[0], best.x[1])
    ll = float(-best.fun)
    return ModelFit("beta_binomial", 2, ll, 2 * 2 - 2 * ll,
                    {"a": float(a), "b": float(b), "mu": float(a / (a + b)),
                     "concentration": float(a + b)}, bool(best.success), len(m))


def _nll_ceiling(theta: np.ndarray, m: np.ndarray, K: np.ndarray) -> float:
    pi0 = _sigmoid(theta[0])
    a, b = _ab_from(theta[1], theta[2])
    if a <= 0 or b <= 0 or not np.isfinite(a) or not np.isfinite(b):
        return np.inf
    ll = float(np.sum(_ceiling_logpmf(m, K, pi0, a, b)))
    return -ll if np.isfinite(ll) else np.inf


def _fit_ceiling(m: np.ndarray, K: np.ndarray, bb: ModelFit) -> ModelFit:
    logit_mu0 = float(np.log(bb.params["mu"] / (1.0 - bb.params["mu"])))
    log_s0 = float(np.log(max(bb.params["concentration"], 1e-3)))
    best = None
    for logit_pi0 in (-4.0, -2.0, -1.0, 0.0):
        x0 = np.array([logit_pi0, logit_mu0, log_s0], dtype=float)
        res = minimize(_nll_ceiling, x0, args=(m, K), method="Nelder-Mead",
                       options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000, "maxfev": 20000})
        if best is None or res.fun < best.fun:
            best = res
    pi0 = _sigmoid(best.x[0])
    a, b = _ab_from(best.x[1], best.x[2])
    ll = float(-best.fun)
    return ModelFit("ceiling_mixture", 3, ll, 2 * 3 - 2 * ll,
                    {"pi0": float(pi0), "a": float(a), "b": float(b)},
                    bool(best.success), len(m))


# --------------------------------------------------------------------------------------
# profile-likelihood intervals
# --------------------------------------------------------------------------------------
def _profile_ci_rho(m: np.ndarray, K: np.ndarray, ll_hat: float,
                    mu_hat: float, level: float = 0.95) -> tuple[float, float]:
    """Profile CI for rho_v = 1/(concentration+1), profiling out mu at each rho."""
    cutoff = ll_hat - 0.5 * stats.chi2.ppf(level, 1)
    logit_mu0 = float(np.log(mu_hat / (1.0 - mu_hat)))
    rho_grid = np.clip(np.concatenate([
        np.linspace(1e-4, 0.999, 200),
    ]), 1e-6, 1.0 - 1e-6)
    inside = []
    for rho in rho_grid:
        s = 1.0 / rho - 1.0  # concentration a+b
        log_s = np.log(max(s, EPS))

        def nll_mu(x):
            a, b = _ab_from(x[0], log_s)
            if a <= 0 or b <= 0:
                return np.inf
            v = float(np.sum(betabinom_logpmf(m, K, a, b)))
            return -v if np.isfinite(v) else np.inf

        res = minimize(nll_mu, np.array([logit_mu0]), method="Nelder-Mead",
                       options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 4000})
        if -res.fun >= cutoff:
            inside.append(rho)
    if not inside:
        return (float("nan"), float("nan"))
    return (float(min(inside)), float(max(inside)))


def _profile_ci_pi0(m: np.ndarray, K: np.ndarray, ll_hat: float,
                    a_hat: float, b_hat: float, level: float = 0.95) -> tuple[float, float]:
    cutoff = ll_hat - 0.5 * stats.chi2.ppf(level, 1)
    logit_mu0 = float(np.log((a_hat / (a_hat + b_hat)) / (1.0 - a_hat / (a_hat + b_hat))))
    log_s0 = float(np.log(max(a_hat + b_hat, 1e-3)))
    grid = np.unique(np.clip(np.concatenate([
        np.linspace(0.0, 0.6, 121), np.linspace(0.0, 1.0, 51),
    ]), 0.0, 1.0 - 1e-9))
    inside = []
    for pi0 in grid:
        def nll_ab(x):
            a, b = _ab_from(x[0], x[1])
            if a <= 0 or b <= 0:
                return np.inf
            v = float(np.sum(_ceiling_logpmf(m, K, pi0, a, b)))
            return -v if np.isfinite(v) else np.inf

        res = minimize(nll_ab, np.array([logit_mu0, log_s0]), method="Nelder-Mead",
                       options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 6000})
        if -res.fun >= cutoff:
            inside.append(pi0)
    if not inside:
        return (float("nan"), float("nan"))
    return (float(min(inside)), float(max(inside)))


# --------------------------------------------------------------------------------------
# odds-law upper-bound check from the fitted curve
# --------------------------------------------------------------------------------------
def _odds_law_check(pi0: float, a: float, b: float, k_max: int) -> tuple[float, bool]:
    """Under independence the reliability curve is the tangent at k=1 (Odds Law); the
    correlated curve must lie on or below it. Compare model-implied reliability to the
    Odds-Law reliability anchored at the k=1 survival."""
    if k_max < 2:
        return float("nan"), False
    ks = np.arange(1, k_max + 1, dtype=float)
    surv = survival_from_params(ks, pi0, a, b)
    surv = np.clip(surv, EPS, 1.0 - EPS)
    rel = 1.0 - surv
    # Odds Law: log-odds grows linearly with slope fixed by the k=1 point
    lo1 = np.log(rel[0] / (1.0 - rel[0]))
    odds_law_lo = lo1 * ks  # anchored so that k=1 matches; slope = single-gate increment
    obs_lo = np.log(rel / (1.0 - rel))
    below = float(np.mean(obs_lo <= odds_law_lo + 1e-9))
    return below, bool(below >= 1.0 - 1e-9)


# --------------------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------------------
def fit_mixture(
    accept_counts: Sequence[int],
    n_gates: Sequence[int],
    min_gates_for_fit: int = 10,
    alpha_level: float = 0.05,
    delta_aic_threshold: float = 2.0,
    compute_ci: bool = True,
) -> MixtureInference:
    """Fit the nested Binomial / Beta-Binomial / ceiling-mixture models to per-item counts.

    Parameters
    ----------
    accept_counts : m_i, the number of unwanted verdicts for item i (false accepts for an
                    error item; false alarms for a clean item).
    n_gates       : K_i, the number of *decided* gates for item i.
    """
    m = np.asarray(accept_counts, dtype=float)
    K = np.asarray(n_gates, dtype=float)
    ok = (K > 0) & np.isfinite(m) & np.isfinite(K)
    m, K = m[ok], K[ok]
    n_items = int(len(m))
    k_max = int(K.max()) if n_items else 0
    warnings: list[str] = []

    if n_items < 4:
        empty = ModelFit("none", 0, float("nan"), float("nan"), {}, False, n_items)
        return MixtureInference(
            n_items=n_items, n_gates_mean=float(K.mean()) if n_items else 0.0, k_max=k_max,
            fit_binomial=empty, fit_betabinom=empty, fit_ceiling=empty,
            lr_heterogeneity=float("nan"), p_heterogeneity=float("nan"),
            delta_aic=float("nan"), preferred_decay="undetermined", is_heterogeneous=False,
            mu_hat=float("nan"), a_hat=float("nan"), b_hat=float("nan"),
            rho_v=float("nan"), rho_v_ci=(float("nan"), float("nan")),
            design_effect_at_k_max=float("nan"), n_eff_at_k_max=float("nan"),
            blind_spot_mass=float("nan"), blind_spot_ci=(float("nan"), float("nan")),
            lr_ceiling=float("nan"), p_ceiling=float("nan"), ceiling_detected=False,
            min_gates_for_fit=min_gates_for_fit,
            below_odds_law_fraction=float("nan"), odds_law_is_upper_bound=False,
            note="fewer than 4 items; the mixture is not identifiable",
            warnings=["n_items < 4"],
        )

    m1 = _fit_binomial(m, K)
    m2 = _fit_betabinom(m, K, mu_start=max(min(m1.params["mu"], 0.99), 0.01))

    # M1 is the concentration -> inf limit of M2, so M2 can never fit worse. If the
    # optimiser slipped, adopt the M1 solution (mu, rho_v ~ 0) as the M2 point.
    if m2.log_likelihood < m1.log_likelihood - 1e-6:
        a_lim = m1.params["mu"] * MAX_CONCENTRATION
        b_lim = (1.0 - m1.params["mu"]) * MAX_CONCENTRATION
        m2 = ModelFit("beta_binomial", 2, m1.log_likelihood, 2 * 2 - 2 * m1.log_likelihood,
                      {"a": a_lim, "b": b_lim, "mu": m1.params["mu"],
                       "concentration": MAX_CONCENTRATION}, True, n_items)

    m3 = _fit_ceiling(m, K, m2)

    # M3 must dominate M2 in likelihood (nested); guard against optimiser slips
    if m3.log_likelihood < m2.log_likelihood - 1e-6:
        m3 = ModelFit("ceiling_mixture", 3, m2.log_likelihood, 2 * 3 - 2 * m2.log_likelihood,
                      {"pi0": 0.0, "a": m2.params["a"], "b": m2.params["b"]}, True, n_items)

    # H1: heterogeneity / non-independence (M2 vs M1), boundary at rho_v = 0
    lr_h = max(2.0 * (m2.log_likelihood - m1.log_likelihood), 0.0)
    p_h = 0.5 * float(stats.chi2.sf(lr_h, 1)) if lr_h > 0 else 1.0
    delta_aic = m1.aic - m2.aic  # >0: heterogeneous (power-law tail) preferred

    a_hat, b_hat = m2.params["a"], m2.params["b"]
    rho_v = 1.0 / (a_hat + b_hat + 1.0)

    if delta_aic > delta_aic_threshold:
        preferred = "power_in_log_k"
    elif delta_aic < -delta_aic_threshold:
        preferred = "exponential_in_k"
    else:
        preferred = "indistinguishable"

    # H3: ceiling (M3 vs M2), boundary at pi0 = 0
    lr_c = max(2.0 * (m3.log_likelihood - m2.log_likelihood), 0.0)
    p_c = 0.5 * float(stats.chi2.sf(lr_c, 1)) if lr_c > 0 else 1.0
    pi0 = m3.params["pi0"]

    rho_ci = (float("nan"), float("nan"))
    pi0_ci = (float("nan"), float("nan"))
    if compute_ci:
        rho_ci = _profile_ci_rho(m, K, m2.log_likelihood, m2.params["mu"])
        pi0_ci = _profile_ci_pi0(m, K, m3.log_likelihood, a_hat, b_hat)

    ceiling_detected = bool(
        p_c < alpha_level and k_max >= min_gates_for_fit
        and (np.isnan(pi0_ci[0]) or pi0_ci[0] > 0.0)
    )
    note = ""
    if k_max < min_gates_for_fit:
        note = (f"largest K observed is {k_max}, below the required {min_gates_for_fit}; "
                "ceiling not declared regardless of fit")

    deff = 1.0 + (k_max - 1) * rho_v
    n_eff = k_max / deff if deff > 0 else float("nan")

    below_frac, is_bound = _odds_law_check(pi0, a_hat, b_hat, k_max)

    if not m2.converged:
        warnings.append("beta-binomial fit did not report convergence")
    frac_at_ceiling = float(np.mean(np.isclose(m, K)))
    if frac_at_ceiling == 0.0:
        warnings.append("no item reached m=K; pi0 is only bounded above, not identified")

    return MixtureInference(
        n_items=n_items, n_gates_mean=float(K.mean()), k_max=k_max,
        fit_binomial=m1, fit_betabinom=m2, fit_ceiling=m3,
        lr_heterogeneity=lr_h, p_heterogeneity=p_h, delta_aic=delta_aic,
        preferred_decay=preferred, is_heterogeneous=bool(preferred == "power_in_log_k"),
        mu_hat=m2.params["mu"], a_hat=a_hat, b_hat=b_hat,
        rho_v=rho_v, rho_v_ci=rho_ci,
        design_effect_at_k_max=deff, n_eff_at_k_max=n_eff,
        blind_spot_mass=pi0, blind_spot_ci=pi0_ci,
        lr_ceiling=lr_c, p_ceiling=p_c, ceiling_detected=ceiling_detected,
        min_gates_for_fit=min_gates_for_fit,
        below_odds_law_fraction=below_frac, odds_law_is_upper_bound=is_bound,
        note=note, warnings=warnings,
    )
