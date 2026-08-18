"""Regression tests for output isolation and the greedy marginal-value policy.

Guards two things that must not silently break:
  1. raw-run dirs are isolated by context condition (evidence never overwrites truncate);
  2. the greedy policy is driven by decision utility, not dependence — a near-constant
     ("dead"), low-correlation verifier that catches nothing is never chosen, while a
     complementary verifier that catches errors the set misses is.
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cascade.paths import raw_run_dir, table_path
from theory.marginal_value import greedy_marginal_value, phase_diagram


# --------------------------------------------------------------- analysis-output isolation
def test_table_path_isolates_context():
    t = table_path("cascade_theory_report.csv", "truncate")
    e = table_path("cascade_theory_report.csv", "evidence")
    assert t.endswith("cascade_theory_report.csv") and "__ctx_" not in t   # canonical, unsuffixed
    assert e.endswith("cascade_theory_report__ctx_evidence.csv")           # never clobbers truncate
    assert t != e
    assert table_path("x.csv") == table_path("x.csv", "truncate")          # default is truncate


# ----------------------------------------------------------------------------- isolation
def test_raw_run_dir_isolates_context():
    t = raw_run_dir("3.3", "truncate")
    e = raw_run_dir("3.3", "evidence")
    assert t != e, "truncate and evidence must resolve to different dirs"
    assert t.endswith("mast__3_3__ctx_truncate")
    assert e.endswith("mast__3_3__ctx_evidence")
    assert raw_run_dir("3.3") == t, "default context mode is truncate"


# ----------------------------------------------------------------------------- synthetic
def _rec(item, gates):
    return {"failure_mode_id": "FM-X", "ground_truth_is_error": True,
            "item_id": item, "n_gates": 50, "gates": gates}


def _synthetic_df():
    """3 verifiers on 20 error items:
       strong  catches items 0-9   (solo 10)
       comp    catches items 10-17 (complementary; covers what strong misses)
       dead    always accepts -> catches nothing, ~constant (low correlation, zero utility)
       items 18-19 are caught by nobody (forces an explicit STOP)."""
    rows = []
    for i in range(20):
        gates = []
        gates += [{"verifier_id": "strong", "accepted": not (i < 10), "latency_seconds": 1.0}
                  for _ in range(10)]
        gates += [{"verifier_id": "comp", "accepted": not (10 <= i < 18), "latency_seconds": 1.0}
                  for _ in range(10)]
        gates += [{"verifier_id": "dead", "accepted": True, "latency_seconds": 1.0}
                  for _ in range(10)]
        rows.append(_rec(f"it{i}", gates))
    return pd.DataFrame(rows)


def test_greedy_prefers_utility_over_low_correlation():
    mv = greedy_marginal_value(_synthetic_df(), lam=0.1)

    # strongest solo starts the set
    init = mv[(mv.step == 0) & mv.selected].iloc[0]
    assert init.candidate == "strong"

    # the complementary verifier is added (positive marginal value), not the dead one
    div = mv[mv.action == "DIVERSIFY"]
    assert list(div.candidate) == ["comp"], "only the complementary source should diversify"
    assert div.iloc[0].delta > 0

    # the dead, low-correlation verifier has zero marginal value and is never diversified in
    dead = mv[mv.candidate == "dead"]
    assert (dead.delta == 0).all()
    assert "DIVERSIFY" not in set(dead.action)

    # honest zero: a zero point estimate must NOT report a degenerate [0,0] CI — the Wilson
    # upper bound stays > 0 (we cannot prove true Delta is exactly zero from a finite sample)
    assert (dead[dead.denom > 0].delta_ci_high > 0).all()

    # coverage of {18,19} is impossible -> the policy must STOP, not keep adding
    assert "STOP" in set(mv.action)


# ----------------------------------------------------------------------- real FM-3.3 data
def _load_fm33_truncate():
    files = glob.glob(f"{raw_run_dir('3.3', 'truncate')}/*.jsonl")
    if not files:
        pytest.skip("FM-3.3 truncate raw data not present")
    recs = [json.loads(l) for p in files for l in open(p) if l.strip()]
    df = pd.DataFrame(recs)
    return df[df["n_gates"] == 50]


def test_greedy_trace_on_real_fm33():
    mv = greedy_marginal_value(_load_fm33_truncate(), lam=0.1)
    sel = mv[mv.selected]

    assert sel[sel.step == 0].iloc[0].candidate == "mistral7b"           # best solo
    s1 = sel[sel.step == 1].iloc[0]
    assert s1.candidate == "llama31" and s1.action == "DIVERSIFY"        # useful, added

    # the dead 3B verifier: zero marginal value, never diversified in
    l3 = mv[mv.candidate == "llama32_3b"]
    assert len(l3) and (l3.delta == 0).all()
    assert "DIVERSIFY" not in set(l3.action)

    # utility-greedy converges to the 2-verifier set (mistral7b + llama31), then stops
    assert (mv.action == "DIVERSIFY").sum() == 1


def test_phase_diagram_regions():
    ph = phase_diagram(_load_fm33_truncate())

    def act(r, lam):
        return ph[(ph.cost_ratio_fa_over_miss == r) & (ph.lambda_cost == lam)].iloc[0].first_action

    # cheap false alarms + free compute -> add a diverse verifier; both costs high -> stop at one
    assert act(0.0, 0.0) == "DIVERSIFY"
    assert act(4.0, 0.8) == "STOP"
    # STOP is a *prediction* of the budget-aware policy: once false alarms are dear, it dominates
    assert (ph[ph.cost_ratio_fa_over_miss >= 2.0].first_action == "STOP").all()
