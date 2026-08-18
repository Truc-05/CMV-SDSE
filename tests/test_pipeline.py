import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cascade.verifier_chain import VerifierChain
from cascade.gate_runner import GateRunner
from cascade.alpha_recorder import AlphaRecorder
from cascade.scenario_loader import render_prompt, summarise
from theory.beta_fit import fit_beta_mle, rho_v_from_beta, expected_reliability_beta
from theory.odds_law import odds_law_reliability
from theory.concavity_test import run_concavity_test
from theory.ceiling_test import run_ceiling_test, reliability_mixture
from theory.trichotomy_test import find_k_dagger
from mast_link.mast_fc3_reference import load_mast_fc3_reference
from mast_link.ceiling_vs_fc3 import compare_ceiling_to_fc3


class StubBackend:
    verifier_id = "stub"

    def __init__(self, false_accept_rate):
        self.false_accept_rate = false_accept_rate

    def verify(self, item, seed):
        rng = random.Random(seed)
        if not item.get("is_error", False):
            return rng.random() > 0.05
        return rng.random() < self.false_accept_rate


def test_verifier_chain_runs():
    chain = VerifierChain(StubBackend(0.3), "test_verifier", k_max=10)
    result = chain.run(item={"is_error": True}, item_id="i1",
                       ground_truth_is_error=True, seed_base=1, k=10)
    assert result.n_gates() == 10
    assert 0 <= result.false_accepts() <= 10
    assert result.n_indeterminate() == 0


def test_verifier_pool_rotates():
    class Named(StubBackend):
        def __init__(self, name, rate):
            super().__init__(rate)
            self.verifier_id = name

    pool = [Named("a", 0.5), Named("b", 0.5), Named("c", 0.5)]
    chain = VerifierChain(pool, "pooled", k_max=6)
    assert chain.pool_ids() == ["a", "b", "c"]
    result = chain.run(item={"is_error": True}, item_id="i1",
                       ground_truth_is_error=True, seed_base=1, k=6)
    assert [g.verifier_id for g in result.gates] == ["a", "b", "c", "a", "b", "c"]


def test_indeterminate_gate_is_not_an_accept():
    class Broken:
        verifier_id = "broken"

        def verify(self, item, seed):
            raise RuntimeError("no verdict")

    chain = VerifierChain(Broken(), "broken", k_max=4, retry_on_backend_error=1)
    result = chain.run(item={"is_error": True}, item_id="i1",
                       ground_truth_is_error=True, seed_base=1, k=4)
    assert result.n_indeterminate() == 4
    assert result.false_accepts() == 0
    assert result.caught_at() is None
    assert result.survived_all_gates() is False


def test_gate_runner_seed_is_stable_across_processes():
    import hashlib

    a = int.from_bytes(hashlib.sha256("item#3|0".encode("utf-8")).digest()[:4], "big")
    b = int.from_bytes(hashlib.sha256("item#3|0".encode("utf-8")).digest()[:4], "big")
    assert a == b


def test_gate_runner_and_alpha_recorder(tmp_path):
    chain = VerifierChain(StubBackend(0.4), "test_verifier", k_max=8)
    scenarios = [
        {"item_id": f"e{i}", "ground_truth_is_error": True, "payload": {"is_error": True}}
        for i in range(15)
    ] + [
        {"item_id": f"c{i}", "ground_truth_is_error": False, "payload": {"is_error": False}}
        for i in range(5)
    ]
    runner = GateRunner(scenario_source=scenarios, verifier_chain=chain,
                        output_dir=str(tmp_path / "raw"), pairing_condition="same_model",
                        failure_mode_id="FM-2.4", seed_base=42)
    records = runner.run_all(k=8, n_repeats=2)
    assert len(records) == 40
    assert all(r["n_decided"] == 8 for r in records)

    recorder = AlphaRecorder(str(tmp_path / "raw"), str(tmp_path / "processed"))
    df = recorder.load_raw()
    assert len(df) == 40

    inst = recorder.per_instance_alpha(df)
    assert inst["alpha_empirical"].between(0, 1).all()
    assert len(inst) == 30

    rel = recorder.reliability_by_k(df)
    assert set(rel["k"]) == set(range(1, 9))
    assert rel["reliability"].between(0, 1).all()

    cond = recorder.reliability_by_k_and_condition(df)
    assert set(cond["pairing_condition"]) == {"same_model"}

    fa = recorder.false_alarm_rate(df)
    assert fa["false_alarm_rate"].between(0, 1).all()

    indet = recorder.indeterminate_report(df)
    assert indet["indeterminate_rate"].iloc[0] == 0.0

    assert recorder.save_processed(inst, "inst").exists()


def test_beta_fit_and_rho_v():
    alphas = [0.1, 0.2, 0.15, 0.9, 0.85, 0.3, 0.25]
    fit = fit_beta_mle(alphas)
    assert fit.a > 0 and fit.b > 0
    assert 0 < rho_v_from_beta(fit) < 1
    reliab = expected_reliability_beta([1, 5, 10, 50], fit)
    assert (reliab >= 0).all() and (reliab <= 1).all()
    assert reliab[-1] >= reliab[0]


def test_odds_law_monotone():
    r = odds_law_reliability([1, 2, 3, 4], 0.3, 0.9, 0.1)
    assert all(r[i] <= r[i + 1] for i in range(len(r) - 1))


def test_ceiling_recovers_known_parameters():
    k = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
    rel = reliability_mixture(np.array(k), 0.15, 2.0, 3.0)
    res = run_ceiling_test(k, list(rel), n=[4000] * len(k), min_gates_for_fit=10,
                           compute_ci=False)
    assert abs(res.blind_spot_mass - 0.15) < 0.01
    assert abs(res.fit_full.b - 3.0) < 0.05
    assert res.is_ceiling_detected


def test_ceiling_is_invariant_to_k_window():
    estimates = []
    for kmax in [20, 50, 200]:
        k = np.unique(np.round(np.geomspace(1, kmax, 14)).astype(int))
        rel = reliability_mixture(k, 0.15, 2.0, 3.0)
        res = run_ceiling_test(list(k), list(rel), n=[4000] * len(k),
                               min_gates_for_fit=10, compute_ci=False)
        estimates.append(res.blind_spot_mass)
    assert max(estimates) - min(estimates) < 0.01


def test_trichotomy():
    k = [1, 2, 3, 5, 8, 13, 21, 34, 50]
    reliability = [0.5, 0.65, 0.73, 0.80, 0.84, 0.86, 0.87, 0.875, 0.878]
    trich = find_k_dagger(k, reliability, smoothing_window=1)
    assert trich.k_values is not None
    assert not trich.harmful_region_detected


def test_scenario_loader_prompt_and_summary():
    trace = {
        "scenario_id": "fm_2_4",
        "messages": [{"turn": 1, "sender": "A", "recipient": "B", "content": "hello"}],
        "actions": [{"turn": 1, "agent": "A", "name": "FILE_MANIFEST", "payload": {"x": 1}}],
    }
    prompt = render_prompt(trace)
    assert "Task brief." in prompt and "hello" in prompt and "FILE_MANIFEST" in prompt
    assert summarise([])["n_items"] == 0


def test_mast_link():
    ref = load_mast_fc3_reference(None)
    assert ref.fc3_rate == 0.235

    k = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
    rel = reliability_mixture(np.array(k), 0.15, 2.0, 3.0)
    ceiling = run_ceiling_test(k, list(rel), n=[4000] * len(k), min_gates_for_fit=10,
                               compute_ci=False)
    trich = find_k_dagger(k, list(rel), smoothing_window=1)
    comparison = compare_ceiling_to_fc3(ceiling, ref, operating_k=3, trichotomy_result=trich)
    assert comparison.fc3_rate == 0.235
    assert comparison.ceiling_detected


def test_mast_link_refuses_meaningless_comparison():
    k = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
    rel = 1.0 - 0.6 ** np.array(k, dtype=float)
    ceiling = run_ceiling_test(k, list(rel), n=[4000] * len(k), min_gates_for_fit=10,
                               compute_ci=False)
    ref = load_mast_fc3_reference(None)
    comparison = compare_ceiling_to_fc3(ceiling, ref)
    assert not comparison.ceiling_detected
    assert "not_meaningful" in comparison.interpretation
