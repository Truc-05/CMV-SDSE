"""Monte-Carlo calibration and power study for the cascade estimators.

A novel estimator is only worth its conclusions if it recovers the truth on data where the
truth is known. This module simulates verifier cascades from the three nested theory models
at the actual pilot design (n=34 items, K=20 gates) and the target full-sweep design
(n=40, K=50), then measures, for the item-level Beta-Binomial estimator
(``theory.betabinom_mixture``):

  * bias and RMSE of rho_v, pi0, mu;
  * finite-sample size of the H1 (heterogeneity) and H3 (ceiling) likelihood-ratio tests;
  * power of H1 and H3 at each design;
  * coverage of the profile-likelihood intervals for rho_v and pi0.

It also runs the *old* curve-based ceiling test (``theory.ceiling_test``) on the very same
simulated cascades and reports its ceiling-detection rate under a no-ceiling truth. Because
the old test sums a per-k binomial likelihood over the cumulative reliability curve -- k
autocorrelated points from the same items -- it treats one item measured through K gates as
K independent observations and its LRT is anticonservative. Quantifying that inflation is
the empirical justification for moving to the item-level likelihood.

Everything here is deterministic given the seed and uses no LLM backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from theory.betabinom_mixture import fit_mixture
from theory.ceiling_test import run_ceiling_test


@dataclass(frozen=True)
class Scenario:
    name: str
    pi0: float
    a: float
    b: float

    @property
    def rho_v(self) -> float:
        return 1.0 / (self.a + self.b + 1.0)

    @property
    def mu(self) -> float:
        return self.a / (self.a + self.b)


def scenario_from_rho(name: str, rho_v: float, mu: float, pi0: float = 0.0) -> Scenario:
    """Build a scenario from the interpretable (rho_v, mu) pair instead of (a, b)."""
    s = 1.0 / rho_v - 1.0            # concentration a + b
    return Scenario(name=name, pi0=pi0, a=mu * s, b=(1.0 - mu) * s)


# Scenarios span the null (independence), mild and strong heterogeneity calibrated to the
# real pilot (rho_v ~ 0.68, miss rate ~ 0.32), and a genuine ceiling.
DEFAULT_SCENARIOS = [
    # pure Binomial (rho_v ~ 0): measures the finite-sample SIZE of the H1 test
    scenario_from_rho("binomial_null", rho_v=1e-5, mu=0.32, pi0=0.0),
    scenario_from_rho("mild_hetero", rho_v=0.20, mu=0.32, pi0=0.0),
    # calibrated to the real pilot; no ceiling -> measures the SIZE of the H3 ceiling test
    scenario_from_rho("pilot_like", rho_v=0.68, mu=0.32, pi0=0.0),
    # a genuine blind-spot ceiling -> measures H3 power and pi0 recovery
    scenario_from_rho("ceiling", rho_v=0.40, mu=0.40, pi0=0.15),
]

# (n_items, K_gates)
DEFAULT_DESIGNS = [(34, 20), (40, 50)]


def simulate_sequences(rng: np.random.Generator, n: int, K: int,
                       sc: Scenario) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate n items, each passing through K exchangeable gates.

    Returns
    -------
    m       : accept counts per item (the item-level sufficient statistic).
    Karr    : K for each item.
    caught  : first-rejection gate per item (1-indexed), or K+1 if the item survives all
              gates -- used to build the cumulative reliability curve the old test needs.
    """
    m = np.empty(n, dtype=int)
    caught = np.empty(n, dtype=int)
    for i in range(n):
        if rng.random() < sc.pi0:
            alpha = 1.0                       # blind spot: every gate accepts
        else:
            alpha = rng.beta(sc.a, sc.b)
        seq = rng.random(K) < alpha           # True = accept (an unwanted verdict)
        m[i] = int(seq.sum())
        rejects = np.where(~seq)[0]
        caught[i] = int(rejects[0] + 1) if rejects.size else K + 1
    return m, np.full(n, K), caught


def reliability_curve(caught: np.ndarray, K: int, n: int) -> tuple[list[int], list[float], list[int]]:
    ks = list(range(1, K + 1))
    rel = [float(np.mean(caught <= k)) for k in ks]
    return ks, rel, [n] * K


def _summarise(records: list[dict], sc: Scenario, n: int, K: int,
               coverage: bool) -> dict:
    df = pd.DataFrame(records)
    row = {
        "scenario": sc.name, "n_items": n, "K": K,
        "rho_v_true": sc.rho_v, "pi0_true": sc.pi0, "mu_true": sc.mu,
        "n_reps": len(df),
        "rho_v_mean": df["rho_v"].mean(),
        "rho_v_bias": df["rho_v"].mean() - sc.rho_v,
        "rho_v_rmse": float(np.sqrt(np.mean((df["rho_v"] - sc.rho_v) ** 2))),
        "pi0_mean": df["pi0"].mean(),
        "pi0_bias": df["pi0"].mean() - sc.pi0,
        "mu_bias": df["mu"].mean() - sc.mu,
        # H1 heterogeneity LRT: size if rho_true~0, else power
        "h1_reject_rate": float(np.mean(df["p_hetero"] < 0.05)),
        # H3 ceiling LRT, new item-level likelihood
        "h3_reject_rate_new": float(np.mean(df["p_ceiling_new"] < 0.05)),
        # H3 ceiling LRT, old curve-based (k-summed) likelihood -- same data
        "h3_reject_rate_old": float(np.mean(df["p_ceiling_old"] < 0.05)),
    }
    if coverage:
        row["rho_v_ci_coverage"] = float(np.mean(df["rho_covered"]))
        row["pi0_ci_coverage"] = float(np.mean(df["pi0_covered"]))
        row["rho_v_ci_width"] = float(df["rho_ci_width"].mean())
    return row


def calibration_run(
    scenarios: Sequence[Scenario] = DEFAULT_SCENARIOS,
    designs: Sequence[tuple[int, int]] = DEFAULT_DESIGNS,
    n_reps: int = 400,
    coverage_reps: int = 150,
    seed: int = 20260807,
    min_gates_for_fit: int = 10,
) -> pd.DataFrame:
    """Run the full calibration grid. Returns one row per (scenario, design)."""
    rng = np.random.default_rng(seed)
    rows = []
    for sc in scenarios:
        for (n, K) in designs:
            do_cov = coverage_reps > 0
            reps = max(n_reps, coverage_reps) if do_cov else n_reps
            records = []
            for r in range(reps):
                m, Karr, caught = simulate_sequences(rng, n, K, sc)
                want_ci = do_cov and r < coverage_reps
                fit = fit_mixture(m, Karr, min_gates_for_fit=min_gates_for_fit,
                                  compute_ci=want_ci)
                # old curve-based ceiling test on the same cascade
                ks, rel, counts = reliability_curve(caught, K, n)
                old = run_ceiling_test(ks, rel, n=counts,
                                       min_gates_for_fit=min_gates_for_fit, compute_ci=False)
                rec = {
                    "rho_v": fit.rho_v, "pi0": fit.blind_spot_mass, "mu": fit.mu_hat,
                    "p_hetero": fit.p_heterogeneity,
                    "p_ceiling_new": fit.p_ceiling,
                    "p_ceiling_old": old.p_value,
                }
                if want_ci:
                    lo, hi = fit.rho_v_ci
                    plo, phi = fit.blind_spot_ci
                    rec["rho_covered"] = bool(lo <= sc.rho_v <= hi) if lo == lo else False
                    rec["pi0_covered"] = bool(plo <= sc.pi0 <= phi) if plo == plo else False
                    rec["rho_ci_width"] = (hi - lo) if lo == lo else float("nan")
                records.append(rec)
            rows.append(_summarise(records, sc, n, K, coverage=do_cov))
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(prog="theory.simulation")
    parser.add_argument("--reps", type=int, default=400)
    parser.add_argument("--coverage-reps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--out", default="results/theory_fits/estimator_calibration.csv")
    args = parser.parse_args(argv)

    print(f"running calibration: reps={args.reps} coverage_reps={args.coverage_reps} ...")
    table = calibration_run(n_reps=args.reps, coverage_reps=args.coverage_reps, seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    table.to_csv(args.out, index=False)

    show = ["scenario", "n_items", "K", "rho_v_true", "rho_v_bias", "pi0_true", "pi0_bias",
            "h1_reject_rate", "h3_reject_rate_new", "h3_reject_rate_old"]
    if "rho_v_ci_coverage" in table.columns:
        show += ["rho_v_ci_coverage", "pi0_ci_coverage"]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table[show].to_string(index=False))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
