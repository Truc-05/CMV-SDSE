"""Null-model and recovery tests for the item-level Beta-Binomial mixture estimator."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from theory.betabinom_mixture import fit_mixture, MAX_CONCENTRATION


def _simulate(rng, n, K, pi0, a, b):
    m = np.empty(n, dtype=int)
    for i in range(n):
        alpha = 1.0 if rng.random() < pi0 else rng.beta(a, b)
        m[i] = int((rng.random(K) < alpha).sum())
    return m, np.full(n, K)


def _ab_from_rho(rho, mu):
    s = 1.0 / rho - 1.0
    return mu * s, (1.0 - mu) * s


def test_independent_gates_recovered_as_low_correlation():
    rng = np.random.default_rng(1)
    # pure Binomial: alpha ~ Beta with huge concentration is effectively fixed
    a, b = _ab_from_rho(1e-4, 0.3)
    m, K = _simulate(rng, 400, 40, 0.0, a, b)
    fit = fit_mixture(m, K, min_gates_for_fit=10, compute_ci=False)
    assert fit.rho_v < 0.05
    assert fit.preferred_decay in ("exponential_in_k", "indistinguishable")
    assert not fit.ceiling_detected


def test_heterogeneous_recovered_and_prefers_power():
    rng = np.random.default_rng(2)
    a, b = _ab_from_rho(0.40, 0.35)
    m, K = _simulate(rng, 500, 50, 0.0, a, b)
    fit = fit_mixture(m, K, min_gates_for_fit=10, compute_ci=False)
    assert abs(fit.rho_v - 0.40) < 0.08
    assert fit.preferred_decay == "power_in_log_k"
    assert fit.p_heterogeneity < 0.05
    assert not fit.ceiling_detected


def test_real_ceiling_is_detected_and_recovered():
    rng = np.random.default_rng(3)
    a, b = _ab_from_rho(0.35, 0.4)
    for pi0 in (0.15, 0.30):
        m, K = _simulate(rng, 600, 50, pi0, a, b)
        fit = fit_mixture(m, K, min_gates_for_fit=10, compute_ci=True)
        assert fit.ceiling_detected, f"missed ceiling pi0={pi0}"
        assert abs(fit.blind_spot_mass - pi0) < 0.07
        assert fit.blind_spot_ci[0] <= pi0 <= fit.blind_spot_ci[1]


def test_ceiling_not_declared_below_min_gates():
    rng = np.random.default_rng(4)
    a, b = _ab_from_rho(0.35, 0.4)
    m, K = _simulate(rng, 400, 5, 0.20, a, b)
    fit = fit_mixture(m, K, min_gates_for_fit=10, compute_ci=False)
    assert not fit.ceiling_detected
    assert "below the required" in fit.note


def test_no_false_ceiling_under_strong_heterogeneity():
    # regression guard for the concentration-overflow bug: strongly heterogeneous, no
    # ceiling. The saturating survival curve must NOT be mistaken for a blind-spot mass.
    rng = np.random.default_rng(5)
    a, b = _ab_from_rho(0.65, 0.32)
    detections = 0
    reps = 20
    for _ in range(reps):
        m, K = _simulate(rng, 40, 20, 0.0, a, b)
        fit = fit_mixture(m, K, min_gates_for_fit=10, compute_ci=False)
        assert fit.a_hat < MAX_CONCENTRATION and fit.b_hat < MAX_CONCENTRATION
        detections += int(fit.p_ceiling < 0.05)
    assert detections <= 3, f"ceiling false-positive rate too high: {detections}/{reps}"


def test_nesting_monotone_in_likelihood():
    rng = np.random.default_rng(6)
    a, b = _ab_from_rho(0.5, 0.3)
    m, K = _simulate(rng, 200, 30, 0.1, a, b)
    fit = fit_mixture(m, K, compute_ci=False)
    # M1 <= M2 <= M3 in log-likelihood (each nests the previous)
    assert fit.fit_betabinom.log_likelihood >= fit.fit_binomial.log_likelihood - 1e-6
    assert fit.fit_ceiling.log_likelihood >= fit.fit_betabinom.log_likelihood - 1e-6


def test_design_effect_and_n_eff_consistent():
    rng = np.random.default_rng(7)
    a, b = _ab_from_rho(0.5, 0.3)
    m, K = _simulate(rng, 300, 40, 0.0, a, b)
    fit = fit_mixture(m, K, compute_ci=False)
    deff = 1.0 + (fit.k_max - 1) * fit.rho_v
    assert abs(fit.design_effect_at_k_max - deff) < 1e-6
    assert abs(fit.n_eff_at_k_max - fit.k_max / deff) < 1e-6
    assert 1.0 <= fit.n_eff_at_k_max <= fit.k_max
