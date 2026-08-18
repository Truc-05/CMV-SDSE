"""Tests for the marginal-value cascade-design algorithm."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from theory.cascade_design import (
    cascade_reliability, solo_catch, accuracy_greedy, marginal_greedy,
    optimal_order, reliability_ceiling, estimate_accept_matrix, compare_designs,
)


def _random_A(rng, n=200, m=5):
    # per-item difficulty makes verifiers correlated (shared latent), like real data
    diff = rng.beta(0.5, 0.5, size=n)          # item hardness
    skill = rng.uniform(-1, 1, size=m)
    A = np.clip(diff[:, None] + 0.3 * skill[None, :] + 0.1 * rng.standard_normal((n, m)), 0.02, 0.98)
    return A


def test_reliability_matches_product_formula():
    rng = np.random.default_rng(0)
    A = _random_A(rng)
    order = [2, 0, 4]
    rel = cascade_reliability(A, order)
    expect = 1.0 - np.prod(A[:, order[:3]], axis=1).mean()
    assert abs(rel[-1] - expect) < 1e-12


def test_marginal_gains_are_non_increasing_submodular():
    rng = np.random.default_rng(1)
    A = _random_A(rng)
    order = marginal_greedy(A)
    rel = cascade_reliability(A, order)
    gains = np.diff(np.concatenate([[0.0], rel]))
    # diminishing returns: each greedy pick's gain <= previous pick's gain
    assert np.all(np.diff(gains) <= 1e-9)


def test_marginal_greedy_dominates_accuracy_greedy():
    rng = np.random.default_rng(2)
    for _ in range(20):
        A = _random_A(rng)
        ra = cascade_reliability(A, accuracy_greedy(A))
        rm = cascade_reliability(A, marginal_greedy(A))
        # at every depth, ours is at least as reliable as the accuracy order
        assert np.all(rm >= ra - 1e-9)


def test_greedy_within_1_minus_1_over_e_of_optimal():
    rng = np.random.default_rng(3)
    for _ in range(10):
        A = _random_A(rng, m=5)
        for k in (2, 3):
            g = cascade_reliability(A, marginal_greedy(A, budget=k))[-1]
            o = cascade_reliability(A, optimal_order(A, budget=k))[-1]
            # (1 - 1/e) guarantee on the *gain* over the empty set (reliability from 0)
            assert g >= (1 - 1 / np.e) * o - 1e-9


def test_flip_when_strong_verifier_is_correlated():
    # v0: strong but perfectly correlated with v1 (same blind spots)
    # v2: weaker but independent -> should be picked before v1 by marginal greedy
    n = 300
    rng = np.random.default_rng(4)
    hard = rng.random(n) < 0.5                      # half the items are a shared blind spot
    a0 = np.where(hard, 0.98, 0.05)                 # strong: catches easy, misses hard
    a1 = np.where(hard, 0.98, 0.10)                 # strong, same blind spot as v0
    a2 = np.clip(0.55 + 0.05 * rng.standard_normal(n), 0.02, 0.98)  # weaker, independent
    A = np.column_stack([a0, a1, a2])
    sc = solo_catch(A)
    assert sc[0] > sc[2] and sc[1] > sc[2]          # v2 is the weakest solo
    order = marginal_greedy(A)
    # after the first strong verifier, the independent-but-weaker v2 is preferred over its
    # correlated twin -> the flip
    assert order[1] == 2, f"expected the weak independent verifier picked 2nd, got {order}"


def test_probe_recovers_the_order_from_few_verdicts():
    rng = np.random.default_rng(5)
    A = _random_A(rng, n=150, m=4)
    # simulate verdict lists by sampling from A
    verdicts = [[[bool(rng.random() < A[i, v]) for _ in range(30)] for v in range(4)]
                for i in range(150)]
    full = marginal_greedy(A)
    A2 = estimate_accept_matrix(verdicts, 150, 4, R=2)
    probe = marginal_greedy(A2)
    # the top pick should agree; probe order should be a decent match
    assert probe[0] == full[0]


def test_stopping_marginal_utility_non_increasing():
    # Theorem 4: greedy marginal gains (hence utilities) are non-increasing.
    rng = np.random.default_rng(10)
    from theory.cascade_design import budgeted_greedy_stop
    A = _random_A(rng, n=300, m=6)
    res = budgeted_greedy_stop(A, lam=0.02)
    us = res["marginal_utilities"]
    assert all(us[i] >= us[i + 1] - 1e-9 for i in range(len(us) - 1))


def test_stopping_fires_where_gain_drops_below_cost():
    from theory.cascade_design import budgeted_greedy_stop, cascade_reliability, marginal_greedy
    rng = np.random.default_rng(11)
    A = _random_A(rng, n=300, m=6)
    lam = 0.05
    res = budgeted_greedy_stop(A, lam=lam)
    # every kept gate had marginal utility > 0; the first dropped gate had utility <= 0
    us = res["marginal_utilities"]
    k = res["stop_depth"]
    assert all(us[t] > 0 for t in range(k))
    if k < len(us):
        assert us[k] <= 0


def test_repeat_reliability_saturates_below_one_under_heterogeneity():
    # same-model repetition plateaus below 1 when some items are always accepted (blind spot)
    from theory.cascade_design import repeat_reliability
    a = np.array([0.02, 0.02, 1.0, 1.0, 0.5])   # two permanent blind spots
    r1, r10, r100 = (repeat_reliability(a, c) for c in (1, 10, 100))
    assert r1 < r10 <= r100
    assert r100 <= 0.6 + 1e-9   # capped by the 2/5 blind-spot mass


def test_ceiling_is_upper_bound():
    rng = np.random.default_rng(6)
    A = _random_A(rng)
    ceil = reliability_ceiling(A)
    cmp = compare_designs(A, [f"v{i}" for i in range(A.shape[1])])
    assert max(cmp.reliability_marginal) <= ceil + 1e-9
