from __future__ import annotations

import numpy as np


def _clip01(p: float, eps: float = 1e-9) -> float:
    return min(max(p, eps), 1.0 - eps)


def per_gate_log_odds_increment(true_positive_rate: float, false_positive_rate: float) -> float:
    tpr = _clip01(true_positive_rate)
    fpr = _clip01(false_positive_rate)
    return np.log(tpr / fpr)


def odds_law_log_odds(
    k: np.ndarray,
    prior_log_odds: float,
    true_positive_rate: float,
    false_positive_rate: float,
) -> np.ndarray:
    increment = per_gate_log_odds_increment(true_positive_rate, false_positive_rate)
    k = np.asarray(k, dtype=float)
    return prior_log_odds + increment * k


def log_odds_to_probability(log_odds: np.ndarray) -> np.ndarray:
    log_odds = np.asarray(log_odds, dtype=float)
    return 1.0 / (1.0 + np.exp(-log_odds))


def odds_law_reliability(
    k: np.ndarray,
    prior_probability: float,
    true_positive_rate: float,
    false_positive_rate: float,
) -> np.ndarray:
    p0 = _clip01(prior_probability)
    prior_log_odds = np.log(p0 / (1.0 - p0))
    log_odds = odds_law_log_odds(k, prior_log_odds, true_positive_rate, false_positive_rate)
    return log_odds_to_probability(log_odds)