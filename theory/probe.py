"""Cheap-probe prediction: how well do R verifier verdicts predict the full cascade?

The theory (Han 2026) states that R repeated verdicts per instance identify the first R
moments of the latent false-accept distribution G, and that "two verdicts identify rho_v".
This turns the cascade into a decision tool: pay for a few verifier calls, fit the correlated
model, and predict the reliability at any depth k -- including depths you never ran. The
practical payoff is knowing, before spending the compute, whether more gates will help.

We contrast two extrapolations from the same R-verdict probe:
  * correlated  -- fit Beta-Binomial(a,b) on the R-verdict counts, predict 1 - E[alpha^k];
  * independent -- fit a single accept rate mu on the R verdicts, predict 1 - mu^k (Odds Law).
The theory predicts the independent extrapolation badly over-states reliability at large k;
this module lets you check that on real held-out depths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from theory.betabinom_mixture import fit_mixture, survival_from_params


@dataclass
class ProbeResult:
    R: int
    n_items: int
    rho_v: float
    mu: float
    a: float
    b: float
    k_grid: np.ndarray
    pred_correlated: np.ndarray      # predicted reliability(k), Beta-Binomial probe
    pred_independent: np.ndarray     # predicted reliability(k), independence (Odds Law)

    def to_dict(self) -> dict:
        return {
            "R": self.R, "n_items": self.n_items, "rho_v": self.rho_v, "mu": self.mu,
            "a": self.a, "b": self.b,
            "k_grid": np.asarray(self.k_grid).tolist(),
            "pred_correlated": np.asarray(self.pred_correlated).tolist(),
            "pred_independent": np.asarray(self.pred_independent).tolist(),
        }


def probe_from_counts(accept_counts_R, R: int, k_grid) -> ProbeResult:
    """Fit both models on per-item accept counts out of the first R gates; extrapolate.

    accept_counts_R : m_i, unwanted-verdict count among the first R decided gates of item i.
    """
    m = np.asarray(accept_counts_R, dtype=float)
    n = len(m)
    K = np.full(n, R)
    fit = fit_mixture(m, K, min_gates_for_fit=max(R, 2), compute_ci=False)
    k = np.asarray(k_grid, dtype=float)

    pred_corr = 1.0 - survival_from_params(k, 0.0, fit.a_hat, fit.b_hat)
    mu = float(m.sum() / max(R * n, 1))
    pred_indep = 1.0 - np.clip(mu, 1e-12, 1 - 1e-12) ** k

    return ProbeResult(R=R, n_items=n, rho_v=fit.rho_v, mu=mu,
                       a=fit.a_hat, b=fit.b_hat, k_grid=k,
                       pred_correlated=pred_corr, pred_independent=pred_indep)


def holdout_error(pred: np.ndarray, observed: np.ndarray, k_grid, R: int) -> float:
    """Mean absolute error between a prediction and the observed reliability at held-out
    depths k > R (the depths the probe never saw)."""
    k = np.asarray(k_grid)
    mask = k > R
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs(np.asarray(pred)[mask] - np.asarray(observed)[mask])))
