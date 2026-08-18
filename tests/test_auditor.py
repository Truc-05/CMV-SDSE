"""Tests for the Cascade Auditor 4-way recommendation."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theory.auditor import audit_pool


def _verdicts(rng, n, verifiers, blind_frac, base_miss):
    """Build verdict_data with a shared blind-spot set (ceiling = 1 - blind_frac)."""
    blind = rng.random(n) < blind_frac
    data = {}
    for i in range(n):
        data[f"i{i}"] = {}
        for v in verifiers:
            if blind[i]:
                verdicts = [True] * 8                      # always accepts (blind spot)
            else:
                verdicts = [bool(rng.random() < base_miss) for _ in range(8)]
            data[f"i{i}"][v] = verdicts
    return data


def test_escalate_when_target_above_ceiling():
    rng = np.random.default_rng(0)
    data = _verdicts(rng, 200, ["a", "b", "c"], blind_frac=0.3, base_miss=0.2)
    r = audit_pool(data, ["a", "b", "c"], lam=0.02, target_reliability=0.95)
    assert r.blind_spot_ceiling < 0.95
    assert r.recommendation == "ESCALATE"


def test_stop_when_target_reached():
    rng = np.random.default_rng(1)
    data = _verdicts(rng, 200, ["a", "b", "c"], blind_frac=0.02, base_miss=0.3)
    r = audit_pool(data, ["a", "b", "c"], lam=0.02, target_reliability=0.6)
    assert r.reliability_at_k_star >= 0.6 - 1e-6
    assert r.recommendation == "STOP"


def test_recommendation_is_one_of_four():
    rng = np.random.default_rng(2)
    data = _verdicts(rng, 150, ["a", "b"], blind_frac=0.1, base_miss=0.25)
    r = audit_pool(data, ["a", "b"], lam=0.02, target_reliability=0.7)
    assert r.recommendation in {"SCALE", "DIVERSIFY", "STOP", "ESCALATE"}
    assert 0.0 <= r.verification_efficiency <= 1.0
    assert r.verification_debt >= 0.0
    assert r.blind_spot_ceiling <= 1.0


def test_ceiling_matches_shared_blind_mass():
    rng = np.random.default_rng(3)
    data = _verdicts(rng, 400, ["a", "b", "c"], blind_frac=0.25, base_miss=0.1)
    r = audit_pool(data, ["a", "b", "c"], lam=0.02)
    # ceiling ~ 1 - blind_frac (the always-accept mass); allow noise
    assert abs(r.blind_spot_ceiling - 0.75) < 0.08
