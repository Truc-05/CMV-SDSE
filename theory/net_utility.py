from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional, Sequence

import numpy as np


@dataclass
class NetUtilityCurve:
    k_values: np.ndarray
    catch_rate: np.ndarray
    false_alarm_rate: np.ndarray
    net_utility: np.ndarray
    false_alarm_cost: float
    k_dagger: Optional[int]
    k_best: Optional[int]
    max_net_utility: float
    net_at_k_max: float
    harmful_region_detected: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("k_values", "catch_rate", "false_alarm_rate", "net_utility"):
            payload[key] = np.asarray(getattr(self, key)).tolist()
        return payload


def net_utility_curve(
    k_values: Sequence[int],
    catch_rate: Sequence[float],
    false_alarm_rate: Sequence[float],
    false_alarm_cost: float = 1.0,
    persistence_required: int = 2,
) -> NetUtilityCurve:
    k = np.asarray(k_values, dtype=float)
    catch = np.asarray(catch_rate, dtype=float)
    fa = np.asarray(false_alarm_rate, dtype=float)

    order = np.argsort(k)
    k, catch, fa = k[order], catch[order], fa[order]

    net = catch - false_alarm_cost * fa
    gains = np.diff(net)

    k_dagger = None
    for i in range(len(gains) - persistence_required + 1):
        if np.all(gains[i:i + persistence_required] < 0):
            k_dagger = int(k[i])
            break

    k_best = int(k[int(np.argmax(net))]) if len(net) else None

    note = ""
    if np.allclose(fa, 0.0):
        note = ("false alarm rate is zero at every k, so the net utility is just the catch "
                "rate and cannot turn down; the trichotomy is untestable on this data")

    return NetUtilityCurve(
        k_values=k,
        catch_rate=catch,
        false_alarm_rate=fa,
        net_utility=net,
        false_alarm_cost=float(false_alarm_cost),
        k_dagger=k_dagger,
        k_best=k_best,
        max_net_utility=float(np.max(net)) if len(net) else float("nan"),
        net_at_k_max=float(net[-1]) if len(net) else float("nan"),
        harmful_region_detected=k_dagger is not None,
        note=note,
    )


def net_utility_cost_frontier(
    k_values: Sequence[int],
    catch_rate: Sequence[float],
    false_alarm_rate: Sequence[float],
    costs: Sequence[float],
    prevalence: Optional[float] = None,
) -> "list[dict]":
    """Sweep the false-alarm cost and report the optimal cascade depth at each cost.

    The single fixed ``false_alarm_cost`` in ``net_utility_curve`` hides that the trichotomy
    (whether more gates help or harm) depends entirely on how a false alarm is priced
    relative to a catch. This sweep exposes the whole frontier and, when ``prevalence`` is
    given (e.g. the MAST FC3 base rate), weights the two error types by how often each item
    type actually occurs:

        U(k) = prevalence * reliability(k) - (1 - prevalence) * cost * false_alarm(k)

    so ``k_best`` is the depth a rational operator at that base rate and cost would choose.
    With ``prevalence=None`` the errors are weighted equally (the raw catch-minus-cost*fa).
    """
    k = np.asarray(k_values, dtype=float)
    catch = np.asarray(catch_rate, dtype=float)
    fa = np.asarray(false_alarm_rate, dtype=float)
    order = np.argsort(k)
    k, catch, fa = k[order], catch[order], fa[order]

    if prevalence is None:
        w_catch, w_fa = 1.0, 1.0
    else:
        w_catch, w_fa = float(prevalence), 1.0 - float(prevalence)

    rows = []
    for c in costs:
        net = w_catch * catch - c * w_fa * fa
        i_best = int(np.argmax(net))
        gains = np.diff(net)
        k_dagger = None
        for i in range(len(gains) - 1):
            if gains[i] < 0 and gains[i + 1] < 0:
                k_dagger = int(k[i])
                break
        rows.append({
            "false_alarm_cost": float(c),
            "prevalence": float(prevalence) if prevalence is not None else None,
            "k_best": int(k[i_best]),
            "max_net_utility": float(net[i_best]),
            "net_at_k1": float(net[0]),
            "net_at_k_max": float(net[-1]),
            "k_dagger": k_dagger,
            "gates_help": bool(k[i_best] > k[0]),
        })
    return rows
