"""Cheap re-probing vs zero-shot transfer (offline, frozen verdicts).

When a policy trained on task A is moved to task B it can lose utility (worst at rho=0.25,
mean transfer regret 0.130). Question: does a TINY probe on B recover the native policy?

For each target family we compare, at a given rho:
  zero-shot transfer (A->B)  : source policy applied as-is (avg over the 5 sources).
  R-round re-probe           : ignore the source; re-select on B using only the first R gates
                               per item, then evaluate on B's full data.
  full recalibration         : select on all of B (= native, regret 0).

Regret is always U_native(B) - U(policy on B). Reuses ra.greedy_normalized / ra._cematrix_capped,
so it reconciles with the transfer matrix (zero-shot mean = 0.130 at rho=0.25). No GPU.
Writes results/tables/rev_reprobe_transfer.csv.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from theory.marginal_value import _prep
import reviewer_analyses as ra

T = "results/tables"
FAMS = list(ra.FAM.keys())
BIG = 10 ** 9


def arrays(df, R, models, ei, ci):
    cat, fa = ra._cematrix_capped(df, R)                 # per-(model,item) majority over first R gates
    Ce = {m: np.array([cat.get(m, {}).get(i, 0) for i in ei], bool) for m in models}
    Cf = {m: np.array([fa.get(m, {}).get(i, 0) for i in ci], bool) for m in models}
    return Ce, Cf


def run(rho):
    # native / zero-shot / all utility evaluations use the SAME _prep arrays as the transfer
    # table (so zero-shot reconciles to 0.130 at rho=0.25). The capped data is used ONLY to
    # derive the re-probe's selected model set.
    dfs = {B: ra.load(slug) for B, slug in ra.FAM.items()}
    P = {B: _prep(dfs[B], 0.5) for B in FAMS}

    def U(B, S):
        c, f = ra.cascade(S, P[B]["Ce"], P[B]["Cf"]); return c - rho * f

    Snat = {B: ra.greedy_normalized(P[B]["Ce"], P[B]["Cf"], P[B]["cost"], w_f=rho, lam=0.1, gate=True)
            for B in FAMS}
    nat = {B: U(B, Snat[B]) for B in FAMS}

    def select_capped(B, R):
        models = P[B]["models"]
        cat, fa = ra._cematrix_capped(dfs[B], R)
        ei = sorted({i for m in models for i in cat.get(m, {})})
        ci = sorted({i for m in models for i in fa.get(m, {})})
        Ce = {m: np.array([cat.get(m, {}).get(i, 0) for i in ei], bool) for m in models}
        Cf = {m: np.array([fa.get(m, {}).get(i, 0) for i in ci], bool) for m in models}
        return ra.greedy_normalized(Ce, Cf, P[B]["cost"], w_f=rho, lam=0.1, gate=True)

    def summarize(name, gates, regs):
        r = np.array(regs)
        return dict(method=name, target_gates=gates, mean_regret=round(float(r.mean()), 3),
                    median=round(float(np.median(r)), 3), max=round(float(r.max()), 3),
                    frac_le_0p05=round(float((r <= 0.05).mean()), 2))

    rows = [summarize("zero-shot transfer (A->B)", 0,
                      [nat[B] - U(B, Snat[A]) for B in FAMS for A in FAMS if A != B])]
    for R in (1, 2, 5):
        rows.append(summarize(f"{R}-round re-probe on B", R,
                              [nat[B] - U(B, select_capped(B, R)) for B in FAMS]))
    rows.append(summarize("full recalibration", 50, [0.0] * len(FAMS)))
    D = pd.DataFrame(rows); D.insert(0, "rho", rho)
    return D


if __name__ == "__main__":
    out = pd.concat([run(r) for r in (0.25, 0.5, 1.0)], ignore_index=True)
    out.to_csv(f"{T}/rev_reprobe_transfer.csv", index=False)
    for rho in (0.25, 0.5, 1.0):
        print(f"=== rho={rho} ===")
        print(out[out.rho == rho].drop(columns="rho").to_string(index=False), "\n")
    print("wrote results/tables/rev_reprobe_transfer.csv")
