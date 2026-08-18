"""Cross-dataset transfer of CMV selection (offline, frozen verdicts).

Question: is conditional complementarity a property of the verifier PAIR (transfers across
task distributions) or of the task distribution (does not transfer)?

For each source family C we freeze the CMV-SDSE selection S_C (greedy on C's full data), then
evaluate S_C on every target family T. Cell (C,T) = utility on T of the set chosen on C; the
diagonal is native (test-on-test). Transfer regret = U_T(native) - U_T(S_C) >= 0. We also track
the first diversifier per family (by residual catch given the solo) and whether it keeps positive
residual catch when moved to another family. Reuses the paper primitives (reconciles to U=0.486).
Writes results/tables/rev_transfer_matrix.csv, rev_transfer_diversifier.csv.
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


def prep_all():
    P = {}
    for fam, slug in ra.FAM.items():
        d = _prep(ra.load(slug), 0.5)
        P[fam] = d
    return P


def transfer(rho=1.0):
    P = prep_all()
    sel = {}      # family -> frozen greedy selection (models)
    for C in FAMS:
        sel[C] = ra.greedy_normalized(P[C]["Ce"], P[C]["Cf"], P[C]["cost"], w_f=1.0, lam=0.1, gate=True)
    # utility of a model set on a family
    def U_on(fam, S):
        c, f = ra.cascade(S, P[fam]["Ce"], P[fam]["Cf"])
        return c - rho * f
    rows = []
    for C in FAMS:
        row = {"calibrate_on": C, "selected": "+".join(sel[C])}
        for Ttgt in FAMS:
            row[Ttgt] = round(U_on(Ttgt, sel[C]), 3)
        rows.append(row)
    M = pd.DataFrame(rows).set_index("calibrate_on")
    native = {t: U_on(t, sel[t]) for t in FAMS}
    # transfer regret matrix (native_T - transferred)
    reg = []
    for C in FAMS:
        rr = {"calibrate_on": C}
        for Ttgt in FAMS:
            rr[Ttgt] = round(native[Ttgt] - M.loc[C, Ttgt], 3)
        reg.append(rr)
    R = pd.DataFrame(reg).set_index("calibrate_on")
    offdiag = [R.loc[C, Ttgt] for C in FAMS for Ttgt in FAMS if C != Ttgt]
    M.to_csv(f"{T}/rev_transfer_matrix.csv")
    print(f"=== Utility transfer matrix U_T(S_C) at rho={rho} (rows=calibrate, cols=test) ===")
    print(M.to_string(), "\n")
    print("native diagonal:", {k: round(v, 3) for k, v in native.items()})
    print(f"off-diagonal transfer regret: mean={np.mean(offdiag):.3f}  "
          f"median={np.median(offdiag):.3f}  max={np.max(offdiag):.3f}  "
          f"frac<=0.05: {np.mean(np.array(offdiag)<=0.05):.2f}\n")
    return M, R, native


def diversifier_transfer():
    """First diversifier per family by residual catch given the solo; does it keep positive
    residual catch when moved to another family? (Complementarity as a pair-vs-distribution test.)"""
    P = prep_all()
    solo, div = {}, {}
    for C in FAMS:
        Ce = P[C]["Ce"]; models = P[C]["models"]
        s = max(models, key=lambda m: Ce[m].mean())
        surv = ~Ce[s]
        cand = [m for m in models if m != s]
        d = max(cand, key=lambda m: Ce[m][surv].mean() if surv.any() else 0.0)
        solo[C], div[C] = s, d
    rows = []
    for C in FAMS:
        row = {"family": C, "solo": solo[C], "diversifier": div[C]}
        # residual catch of C's diversifier on each target T, given T's own solo
        for Ttgt in FAMS:
            CeT = P[Ttgt]["Ce"]; sT = solo[Ttgt]
            survT = ~CeT[sT]
            dcatch = float(CeT[div[C]][survT].mean()) if survT.any() and div[C] in CeT else 0.0
            row[Ttgt] = round(dcatch, 3)
        rows.append(row)
    D = pd.DataFrame(rows).set_index("family")
    D.to_csv(f"{T}/rev_transfer_diversifier.csv")
    print("=== Diversifier transfer: residual catch of family C's diversifier on family T "
          "(given T's solo) ===")
    print(D.to_string(), "\n")
    return D


def regret_vs_regime(rhos=(1.0, 0.5, 0.25)):
    P = prep_all()
    out = []
    for rho in rhos:
        sel = {C: ra.greedy_normalized(P[C]["Ce"], P[C]["Cf"], P[C]["cost"], w_f=rho, lam=0.1, gate=True)
               for C in FAMS}
        def U_on(fam, S):
            c, f = ra.cascade(S, P[fam]["Ce"], P[fam]["Cf"]); return c - rho * f
        native = {t: U_on(t, sel[t]) for t in FAMS}
        reg = [native[Tt] - U_on(Tt, sel[C]) for C in FAMS for Tt in FAMS if C != Tt]
        n_distinct = len({tuple(sel[C]) for C in FAMS})
        out.append(dict(rho=rho, distinct_selections=n_distinct,
                        mean_regret=round(float(np.mean(reg)), 3),
                        median_regret=round(float(np.median(reg)), 3),
                        max_regret=round(float(np.max(reg)), 3),
                        frac_le_0p05=round(float(np.mean(np.array(reg) <= 0.05)), 2)))
    D = pd.DataFrame(out)
    D.to_csv(f"{T}/rev_transfer_regret.csv", index=False)
    print("=== transfer regret vs operating regime ===")
    print(D.to_string(index=False), "\n")
    return D


if __name__ == "__main__":
    transfer()
    diversifier_transfer()
    regret_vs_regime()
    print("wrote results/tables/rev_transfer_{matrix,diversifier,regret}.csv")
