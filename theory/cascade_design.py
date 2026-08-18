"""Verifier-cascade construction by conditional marginal value (not solo accuracy).

The point of the paper. A serial cascade returns an answer only if every gate accepts, so a
true error is CAUGHT if any selected verifier rejects it. Under the latent-difficulty model
(each item i has a per-verifier accept probability a_{iv}; verifiers are conditionally
independent given the item), the probability that item i survives a set S of verifiers is the
product of its accept probabilities, and the cascade's reliability is

    R(S) = mean_i [ 1 - prod_{v in S} a_{iv} ].

Two facts drive the design:

1. R(S) is a MONOTONE SUBMODULAR set function. The marginal gain of adding verifier w to S is
       R(S + w) - R(S) = mean_i [ (prod_{v in S} a_{iv}) * (1 - a_{iw}) ]
   i.e. the expected catch of w *among the errors that survive S*. The prefactor
   prod_{v in S} a_{iv} only shrinks as S grows, so the marginal gain is non-increasing in S
   (diminishing returns) and non-negative (monotone). Hence greedy -- repeatedly add the
   verifier with the largest marginal gain on the current survivors -- achieves at least
   (1 - 1/e) of the optimal k-subset for every budget k (Nemhauser-Wolsey-Fisher 1978).

2. A verifier's marginal gain is its CONDITIONAL catch on the survivors, which is its solo
   catch times how independent it is from the errors S already lets through. So ordering by
   solo accuracy is wrong: a high-accuracy verifier that is correlated with the survivors can
   be worth less than a low-accuracy verifier that is complementary. ``accuracy_greedy`` (the
   standard "pick the best judge") is not submodular-optimal and ``marginal_greedy`` provably
   dominates it in expectation.

The accept matrix A is exactly what the cheap probe estimates (a few verdicts per verifier per
item), so the whole design is operable from a small probe budget -- see ``estimate_accept_matrix``.

A hard ceiling remains: R(S) <= 1 - mean_i prod_{v in pool} a_{iv}; if a mass of items has
a_{iv}=1 for every available verifier (a shared blind spot), no ordering crosses it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cascade_reliability(A: np.ndarray, order) -> np.ndarray:
    """Reliability after each prefix of ``order``.

    A : (n_items, n_verifiers) per-item accept (miss) probabilities.
    order : sequence of verifier column indices.
    Returns an array of length len(order): reliability after 1, 2, ... gates.
    """
    surv = np.ones(A.shape[0])
    out = []
    for j in order:
        surv = surv * A[:, j]
        out.append(1.0 - surv.mean())
    return np.array(out)


def solo_catch(A: np.ndarray) -> np.ndarray:
    """Per-verifier standalone catch rate = 1 - mean accept."""
    return 1.0 - A.mean(axis=0)


def accuracy_greedy(A: np.ndarray) -> list[int]:
    """Baseline: order verifiers by descending solo catch (the leaderboard order)."""
    return list(np.argsort(-solo_catch(A)))


def marginal_greedy(A: np.ndarray, budget: int | None = None) -> list[int]:
    """Our algorithm: greedily add the verifier with the largest catch on current survivors.

    Monotone-submodular greedy, so the length-k prefix is within (1 - 1/e) of the best k-subset.
    """
    n, m = A.shape
    budget = m if budget is None else min(budget, m)
    surv = np.ones(n)
    chosen: list[int] = []
    remaining = set(range(m))
    for _ in range(budget):
        # marginal gain of v = expected catch on survivors = mean_i surv_i * (1 - a_iv)
        best = max(remaining, key=lambda v: float((surv * (1.0 - A[:, v])).mean()))
        chosen.append(best)
        surv = surv * A[:, best]
        remaining.discard(best)
    return chosen


def optimal_order(A: np.ndarray, budget: int | None = None) -> list[int]:
    """Brute-force best prefix by exhaustive search (only for small pools; sanity/oracle)."""
    import itertools
    n, m = A.shape
    budget = m if budget is None else min(budget, m)
    best_seq, best_val = None, -np.inf
    for perm in itertools.permutations(range(m), budget):
        val = 1.0 - np.prod(A[:, list(perm)], axis=1).mean()
        if val > best_val:
            best_val, best_seq = val, list(perm)
    # extend to a full order deterministically for a complete ranking
    rest = [v for v in accuracy_greedy(A) if v not in best_seq]
    return best_seq + rest


def repeat_reliability(a_v: np.ndarray, c: int) -> float:
    """Reliability of calling one verifier `c` times (same-model repetition baseline).

    a_v : per-item accept (miss) probabilities of the single verifier.
    """
    return float(1.0 - np.mean(np.asarray(a_v, dtype=float) ** c))


def budgeted_greedy_stop(A: np.ndarray, lam: float, costs=None,
                         budget: int | None = None) -> dict:
    """Cost-benefit greedy with optimal stopping (Theorem 4).

    Adds the verifier maximizing marginal-gain / cost; records the marginal utility
    u_t = Δ_t − λ·c_t; stops at the first t with u_t ≤ 0. Returns the order up to the stop,
    the reliability there, the utility, and the marginal-utility trajectory.
    """
    n, m = A.shape
    costs = np.ones(m) if costs is None else np.asarray(costs, dtype=float)
    budget = m if budget is None else min(budget, m)
    surv = np.ones(n)
    chosen: list[int] = []
    remaining = set(range(m))
    us: list[float] = []
    rel = 0.0
    stop_at = None
    for t in range(budget):
        # pick by cost-benefit ratio; evaluate its raw marginal utility for the stop test
        gains = {v: float((surv * (1.0 - A[:, v])).mean()) for v in remaining}
        w = max(remaining, key=lambda v: gains[v] / costs[v])
        delta = gains[w]
        u = delta - lam * costs[w]
        us.append(u)
        if u <= 0 and stop_at is None:
            stop_at = t  # optimal prefix is the first `t` gates (before this unprofitable add)
        chosen.append(w)
        surv = surv * A[:, w]
        rel = 1.0 - surv.mean()
        remaining.discard(w)
    if stop_at is None:
        stop_at = len(chosen)
    stop_order = chosen[:stop_at]
    stop_surv = np.ones(n)
    for v in stop_order:
        stop_surv = stop_surv * A[:, v]
    stop_rel = float(1.0 - stop_surv.mean())
    stop_cost = float(sum(costs[v] for v in stop_order))
    return {
        "order_full": chosen,
        "marginal_utilities": us,
        "stop_depth": stop_at,
        "stop_order": stop_order,
        "stop_reliability": stop_rel,
        "stop_utility": stop_rel - lam * stop_cost,
    }


def optimal_per_k(A: np.ndarray) -> list[float]:
    """Best achievable reliability at each depth k = max over all k-subsets (small pools only)."""
    import itertools
    m = A.shape[1]
    out = []
    for k in range(1, m + 1):
        out.append(max(1.0 - np.prod(A[:, list(c)], axis=1).mean()
                       for c in itertools.combinations(range(m), k)))
    return out


def reliability_ceiling(A: np.ndarray) -> float:
    """The best reliability any ordering of the full pool can reach (blind-spot wall)."""
    return float(1.0 - np.prod(A, axis=1).mean())


def estimate_accept_matrix(verdict_lists, n_items: int, n_verifiers: int,
                           R: int | None = None, prior: float = 1.0) -> np.ndarray:
    """Probe estimate of A from a few verdicts per (item, verifier).

    verdict_lists[i][v] is a list of booleans (True = accept). With R set, only the first R
    verdicts are used (the probe budget). A Beta(prior, prior) shrinkage keeps 0/K estimates
    off the boundary so the greedy is stable at tiny R.
    """
    A = np.full((n_items, n_verifiers), 0.5)
    for i in range(n_items):
        for v in range(n_verifiers):
            vs = verdict_lists[i][v]
            if R is not None:
                vs = vs[:R]
            if not vs:
                continue
            acc = sum(1 for x in vs if x)
            A[i, v] = (acc + prior) / (len(vs) + 2 * prior)
    return A


def greedy_from_probe_rounds(rounds: np.ndarray, budget: int | None = None) -> list[int]:
    """Theorem-3 estimator: greedy driven by marginal gains estimated DIRECTLY from joint
    probe rounds, with no per-cell rate estimation.

    rounds : boolean array (R, n_items, m) — rounds[r,i,v] = True iff verifier v accepted item i
             in probe round r. Each row is one joint pass over all verifiers.
    The marginal gain of w at S is estimated as the fraction of (round, item) pairs where every
    verifier in S accepted and w rejected — an unbiased estimate of E_i[P_S(i)(1-a_{iw})].
    """
    rounds = np.asarray(rounds, dtype=bool)
    R, n, m = rounds.shape
    budget = m if budget is None else min(budget, m)
    # survive[r,i] = all chosen verifiers accepted in round r on item i
    survive = np.ones((R, n), dtype=bool)
    chosen: list[int] = []
    remaining = set(range(m))
    for _ in range(budget):
        # gain(w) = mean over (r,i) of survive & (w rejects)
        best = max(remaining,
                   key=lambda w: float((survive & (~rounds[:, :, w])).mean()))
        chosen.append(best)
        survive = survive & rounds[:, :, best]
        remaining.discard(best)
    return chosen


@dataclass
class DesignComparison:
    verifiers: list
    solo_catch: dict
    accuracy_order: list
    marginal_order: list
    optimal_order: list
    reliability_accuracy: list
    reliability_marginal: list
    reliability_optimal: list
    ceiling: float

    def to_dict(self) -> dict:
        return {
            "verifiers": list(self.verifiers),
            "solo_catch": self.solo_catch,
            "accuracy_order": [self.verifiers[j] for j in self.accuracy_order],
            "marginal_order": [self.verifiers[j] for j in self.marginal_order],
            "optimal_order": [self.verifiers[j] for j in self.optimal_order],
            "reliability_accuracy": list(map(float, self.reliability_accuracy)),
            "reliability_marginal": list(map(float, self.reliability_marginal)),
            "reliability_optimal": list(map(float, self.reliability_optimal)),
            "ceiling": self.ceiling,
        }


def compare_designs(A: np.ndarray, verifier_names) -> DesignComparison:
    ao = accuracy_greedy(A)
    mo = marginal_greedy(A)
    oo = optimal_order(A)
    sc = solo_catch(A)
    return DesignComparison(
        verifiers=list(verifier_names),
        solo_catch={verifier_names[j]: float(sc[j]) for j in range(len(verifier_names))},
        accuracy_order=ao, marginal_order=mo, optimal_order=oo,
        reliability_accuracy=list(cascade_reliability(A, ao)),
        reliability_marginal=list(cascade_reliability(A, mo)),
        reliability_optimal=list(cascade_reliability(A, oo)),
        ceiling=reliability_ceiling(A),
    )
