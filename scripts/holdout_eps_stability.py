"""#9: utility-stability of the held-out selection.

Reviewer point: exact-set agreement is low on some families (MBPP 0.01), which looks
alarming. But the paper's claim is about DECISION UTILITY, not set identity. This replays
the exact held-out split loop of reviewer_holdout_escalate.holdout (same seed, same
cal/test sizes, same frozen controller) and reports, per family, the fraction of splits
whose selected-set test utility is within epsilon of the test-oracle utility:

    P(U_selected >= U_oracle - eps) = P(regret <= eps),  eps in {0.02, 0.05}.

If this probability is high while sel_agree is low, the exact set is unstable but the
decision utility is stable -- exactly the paper's thesis. Non-destructive: writes a new
CSV, does not touch rev_holdout_cmv.csv.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import reviewer_analyses as ra
from theory.marginal_value import _prep
import reviewer_holdout_escalate as rh   # reuse the EXACT helpers + constants (main-guarded)

EPS = [0.02, 0.05]
# The frozen held-out pool is the four <=14B core verifiers. Since the k=3 escalate and
# k=50 redundancy runs added mistral_small_24b verdicts into the same family dirs, _prep
# would otherwise pick it up as a 5th model and change the selection -- restrict to CORE
# so this reconciles EXACTLY with the frozen rev_holdout_cmv.csv.
CORE = ["qwen7b", "llama31", "llama32_3b", "mistral7b"]


def main():
    rows = []
    for fam, slug in ra.FAM.items():
        P = _prep(ra.load(slug), 0.5)
        models = [m for m in CORE if m in P["Ce"]]
        Ce = {m: P["Ce"][m] for m in models}
        Cf = {m: P["Cf"][m] for m in models}
        cost = {m: P["cost"][m] for m in models}
        ne, nc = P["ne"], P["nc"]
        rng = np.random.RandomState(20260815)          # identical seed -> identical splits
        reg, agree = [], []
        for _ in range(rh.N_SPLITS):
            ep = rng.permutation(ne); cp = rng.permutation(nc)
            cal_e, tst_e = ep[:rh.CAL_ERR], ep[rh.CAL_ERR:]
            cal_c, tst_c = cp[:rh.CAL_CLN], cp[rh.CAL_CLN:]
            Ce_cal, Cf_cal = rh.sub(Ce, cal_e), rh.sub(Cf, cal_c)
            Ce_tst, Cf_tst = rh.sub(Ce, tst_e), rh.sub(Cf, tst_c)
            S = ra.greedy_normalized(Ce_cal, Cf_cal, cost, w_f=1.0, lam=0.1, gate=True)
            tU = rh.cascade_U(S, Ce_tst, Cf_tst, rh.RHO)[2]
            oS, _, _, oU = rh.best_subset(Ce_tst, Cf_tst, models, rh.RHO)
            reg.append(oU - tU)
            agree.append(int(set(S) == set(oS)))
        reg = np.array(reg)
        rows.append(dict(family=fam,
                         mean_regret=round(float(reg.mean()), 3),
                         sel_agree=round(float(np.mean(agree)), 3),
                         **{f"P(regret<={e})": round(float((reg <= e + 1e-9).mean()), 3) for e in EPS}))
    D = pd.DataFrame(rows)
    macro = {c: round(D[c].mean(), 3) for c in D.columns if c != "family"}
    macro["family"] = "MACRO"
    D = pd.concat([D, pd.DataFrame([macro])], ignore_index=True)
    out = "results/tables/rev_holdout_eps.csv"
    D.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print("=== #9 held-out utility stability (200 splits, rho=1) ===")
    print(D.to_string(index=False))
    print("\nRead: exact-set agreement can be low (e.g. MBPP) while P(regret<=eps) stays high")
    print("=> the SET is unstable but the decision UTILITY is stable.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
