"""Cluster-bootstrap uncertainty for cascade curves.

The reliability curve r(k) and the false-alarm curve f(k) are cumulative: every point is
computed from the same items passing through more and more gates. The k points are
therefore not independent, so a per-k Wilson or binomial interval understates the
uncertainty and, worse, ignores that neighbouring k share items. The honest interval
resamples the *items* (the independent unit) with replacement and recomputes the whole
curve each time -- a cluster (block) bootstrap with the item as the cluster.
"""

from __future__ import annotations

import numpy as np


def cumulative_curve(first_event_gate: np.ndarray, k_max: int) -> np.ndarray:
    """Fraction of items whose first flagged gate is <= k, for k = 1..k_max.

    ``first_event_gate[i]`` is the 1-indexed gate at which item i was first flagged
    (rejected), or any value > k_max (e.g. k_max+1) if it was never flagged.
    """
    ks = np.arange(1, k_max + 1)
    fe = np.asarray(first_event_gate)[:, None]
    return (fe <= ks[None, :]).mean(axis=0)


def bootstrap_curve_ci(
    first_event_gate: np.ndarray,
    k_max: int,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 20260701,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster-bootstrap CI for a cumulative curve.

    Returns (point, lo, hi), each of length k_max. ``point`` is the observed curve.
    """
    fe = np.asarray(first_event_gate)
    n = len(fe)
    point = cumulative_curve(fe, k_max)
    if n == 0:
        nan = np.full(k_max, np.nan)
        return nan, nan, nan
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, k_max), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = cumulative_curve(fe[idx], k_max)
    lo = np.quantile(draws, (1 - level) / 2, axis=0)
    hi = np.quantile(draws, 1 - (1 - level) / 2, axis=0)
    return point, lo, hi


def bootstrap_statistic_ci(
    values: np.ndarray,
    statistic,
    n_boot: int = 500,
    level: float = 0.95,
    seed: int = 20260701,
) -> tuple[float, float, float]:
    """Generic item-level bootstrap CI for a scalar statistic of a per-item array.

    ``values`` is an (n_items, ...) array; ``statistic`` maps a resampled slice to a float.
    Returns (point, lo, hi). NaN bootstrap replicates are dropped before taking quantiles.
    """
    values = np.asarray(values)
    n = len(values)
    point = float(statistic(values))
    if n == 0:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    reps = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        reps[b] = statistic(values[idx])
    reps = reps[np.isfinite(reps)]
    if reps.size == 0:
        return point, float("nan"), float("nan")
    lo = float(np.quantile(reps, (1 - level) / 2))
    hi = float(np.quantile(reps, 1 - (1 - level) / 2))
    return point, lo, hi
