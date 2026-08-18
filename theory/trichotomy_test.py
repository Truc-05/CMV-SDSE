from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class TrichotomyResult:
    k_values: np.ndarray
    marginal_gain: np.ndarray
    k_dagger: Optional[int]
    regime_before: str
    regime_after: str
    harmful_region_detected: bool
    smoothing_window: int


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def find_k_dagger(
    k_values: Sequence[int],
    reliability: Sequence[float],
    smoothing_window: int = 1,
    persistence_required: int = 2,
) -> TrichotomyResult:
    k = np.asarray(k_values, dtype=float)
    r = np.asarray(reliability, dtype=float)

    order = np.argsort(k)
    k = k[order]
    r = r[order]

    r_smooth = _smooth(r, smoothing_window)
    k_smooth = k[: len(r_smooth)] if smoothing_window <= 1 else k[smoothing_window - 1 :][: len(r_smooth)]

    marginal_gain = np.diff(r_smooth)
    k_dagger = None

    for i in range(len(marginal_gain) - persistence_required + 1):
        window = marginal_gain[i : i + persistence_required]
        if np.all(window < 0):
            k_dagger = int(k_smooth[i])
            break

    harmful_detected = k_dagger is not None

    return TrichotomyResult(
        k_values=k_smooth,
        marginal_gain=marginal_gain,
        k_dagger=k_dagger,
        regime_before="gates_help" if harmful_detected or len(marginal_gain) else "undetermined",
        regime_after="gates_harm" if harmful_detected else "no_harmful_regime_observed",
        harmful_region_detected=harmful_detected,
        smoothing_window=smoothing_window,
    )
