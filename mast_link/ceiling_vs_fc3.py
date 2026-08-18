from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .mast_fc3_reference import MASTFC3Reference
from theory.ceiling_test import CeilingTestResult
from theory.trichotomy_test import TrichotomyResult


@dataclass
class CeilingFC3Comparison:
    measured_blind_spot_mass: float
    blind_spot_ci: tuple
    ceiling_p_value: float
    ceiling_detected: bool
    fc3_rate: float
    ratio: float
    absolute_gap: float
    interpretation: str
    regime_relative_to_k_dagger: Optional[str]
    caveat: str = (
        "FC3 in MAST is the share of annotated failure flags labelled task-verification, "
        "measured on a different population by a fallible annotator. Blind-spot mass here is "
        "the share of known errors no gate ever rejects. The two are not the same estimand; "
        "treat any agreement as suggestive, never as validation."
    )


def compare_ceiling_to_fc3(
    ceiling_result: CeilingTestResult,
    fc3_reference: MASTFC3Reference,
    operating_k: Optional[int] = None,
    trichotomy_result: Optional[TrichotomyResult] = None,
) -> CeilingFC3Comparison:
    measured = ceiling_result.blind_spot_mass
    fc3_rate = fc3_reference.fc3_rate

    if measured != measured:
        ratio = float("nan")
        gap = float("nan")
        interpretation = "ceiling_not_estimable"
    elif not ceiling_result.is_ceiling_detected:
        gap = measured - fc3_rate
        ratio = measured / fc3_rate if fc3_rate > 0 else float("nan")
        interpretation = "no_ceiling_distinguishable_from_zero_comparison_not_meaningful"
    else:
        gap = measured - fc3_rate
        ratio = measured / fc3_rate if fc3_rate > 0 else float("nan")
        if abs(gap) <= 0.03:
            interpretation = "blind_spot_mass_consistent_with_fc3_rate"
        elif measured > fc3_rate:
            interpretation = "measured_blind_spot_exceeds_observed_fc3_rate"
        else:
            interpretation = "measured_blind_spot_below_observed_fc3_rate"

    regime = None
    if trichotomy_result is not None and operating_k is not None:
        if trichotomy_result.k_dagger is None:
            regime = "below_or_at_k_dagger_not_observed_in_range"
        elif operating_k < trichotomy_result.k_dagger:
            regime = "operating_below_k_dagger_gates_helping"
        elif operating_k == trichotomy_result.k_dagger:
            regime = "operating_at_k_dagger"
        else:
            regime = "operating_above_k_dagger_gates_harming"

    return CeilingFC3Comparison(
        measured_blind_spot_mass=measured,
        blind_spot_ci=tuple(ceiling_result.blind_spot_ci),
        ceiling_p_value=ceiling_result.p_value,
        ceiling_detected=ceiling_result.is_ceiling_detected,
        fc3_rate=fc3_rate,
        ratio=ratio,
        absolute_gap=gap,
        interpretation=interpretation,
        regime_relative_to_k_dagger=regime,
    )
