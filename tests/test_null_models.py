import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from theory.ceiling_test import run_ceiling_test, survival_beta, reliability_mixture
from theory.concavity_test import run_concavity_test

K_GRID = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50]
N = [4000] * len(K_GRID)


def independent_reliability(alpha, k):
    return 1.0 - np.asarray(alpha, dtype=float) ** np.asarray(k, dtype=float)


def test_independent_gates_prefer_exponential_decay():
    for alpha in (0.5, 0.6, 0.75, 0.9):
        rel = independent_reliability(alpha, K_GRID)
        res = run_concavity_test(K_GRID, list(rel), n=N)
        assert res.preferred_model == "exponential_in_k"
        assert not res.is_heterogeneous
        assert abs(res.per_gate_alpha - alpha) < 0.02


def test_independent_gates_show_no_ceiling():
    for alpha in (0.5, 0.6, 0.75, 0.9):
        rel = independent_reliability(alpha, K_GRID)
        res = run_ceiling_test(K_GRID, list(rel), n=N, min_gates_for_fit=10,
                               compute_ci=False)
        assert not res.is_ceiling_detected
        assert res.blind_spot_mass < 0.01


def test_heterogeneous_without_ceiling_shows_no_ceiling():
    for a, b in [(2.0, 3.0), (3.0, 0.8), (1.0, 1.0), (0.5, 2.0)]:
        rel = 1.0 - survival_beta(np.array(K_GRID), a, b)
        res = run_ceiling_test(K_GRID, list(rel), n=N, min_gates_for_fit=10,
                               compute_ci=False)
        assert not res.is_ceiling_detected, f"false ceiling for Beta({a},{b})"
        assert res.blind_spot_mass < 0.01


def test_heterogeneous_prefers_power_decay():
    for a, b in [(2.0, 3.0), (3.0, 0.8), (1.0, 1.0)]:
        rel = 1.0 - survival_beta(np.array(K_GRID), a, b)
        res = run_concavity_test(K_GRID, list(rel), n=N)
        assert res.preferred_model == "power_in_log_k"
        assert res.is_heterogeneous


def test_real_ceiling_is_detected():
    for pi0 in (0.05, 0.15, 0.30):
        rel = reliability_mixture(np.array(K_GRID), pi0, 2.0, 3.0)
        res = run_ceiling_test(K_GRID, list(rel), n=N, min_gates_for_fit=10,
                               compute_ci=False)
        assert res.is_ceiling_detected
        assert abs(res.blind_spot_mass - pi0) < 0.01


def test_ceiling_not_declared_below_min_gates():
    short_k = [1, 2, 3, 4, 5]
    rel = reliability_mixture(np.array(short_k), 0.15, 2.0, 3.0)
    res = run_ceiling_test(short_k, list(rel), n=[4000] * len(short_k),
                           min_gates_for_fit=10, compute_ci=False)
    assert not res.is_ceiling_detected
    assert "below the required" in res.note


def test_saturated_tail_is_reported_not_silently_used():
    rel = [0.9, 0.99, 1.0, 1.0, 1.0, 1.0]
    res = run_concavity_test([1, 2, 3, 4, 5, 6], rel, n=[100] * 6)
    assert res.n_excluded_saturated == 4
    assert res.preferred_model == "undetermined"


def test_odds_law_is_its_own_upper_bound():
    from theory.concavity_test import compare_to_odds_law
    from theory.odds_law import odds_law_log_odds, odds_law_reliability

    ks = np.array([1, 2, 3, 4, 5, 6, 8, 10])
    rel = odds_law_reliability(ks, 0.3, 0.9, 0.1)
    ref = odds_law_log_odds(ks, float(np.log(0.3 / 0.7)), 0.9, 0.1)
    below, max_gap, is_bound, n_pts = compare_to_odds_law(list(ks), list(rel), ref)
    assert n_pts > 0
    assert abs(max_gap) < 1e-6, f"self-comparison should be exact, got {max_gap}"
    assert is_bound
    assert below == 1.0


def test_trichotomy_needs_false_alarms_to_exist():
    from theory.net_utility import net_utility_curve

    k = [1, 2, 3, 5, 8, 13, 21]
    catch = [0.40, 0.58, 0.68, 0.78, 0.84, 0.87, 0.88]

    flat = net_utility_curve(k, catch, [0.0] * len(k))
    assert flat.k_dagger is None
    assert "untestable" in flat.note

    rising = net_utility_curve(k, catch, [0.02, 0.05, 0.09, 0.16, 0.27, 0.42, 0.60])
    assert rising.k_dagger is not None
    assert rising.harmful_region_detected
    assert rising.k_best == 5
