"""Wilson significance-gate ablation (offline, held-out): does the gate CAUSE the out-of-sample edge?

Everything identical (same cost term lambda, same residual-mass value V) except the gate on/off:
  gate=True  -- admit v only if V_hat(v|S)>0 AND Wilson-LB(residual catch)>0   (CMV-SDSE)
  gate=False -- admit v whenever V_hat(v|S)>0                                   (ablation)

Per cost regime rho in {0.25,0.5,1,2}, over 200 held-out splits (50err/25cln calibration ->
100err/50cln test, the tab:holdout protocol), macro over the six families, we report:
  in-sample U, held-out U, avg #sources, test catch/FA,
  over-acquisition rate = of the non-initial acquisitions (each had V_hat>0 on calibration),
                          the fraction whose ACTUAL held-out marginal utility V_test <= 0,
  selection stability   = fraction of splits whose selected set equals the family's modal set.
No model/API calls. Writes results/tables/rev_gate_ablation.csv.
"""
from __future__ import annotations
import sys
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from theory.marginal_value import _prep
import reviewer_analyses as ra
from fusion_selection_baselines import ges

T = "results/tables"
RHOS = (0.25, 0.5, 1.0, 2.0)
NSPL, CALE, CALC = 200, 50, 25


def U(Ce, Cf, S, rho):
    if not S:
        return 0.0
    c, f = ra.cascade(S, Ce, Cf); return c - rho * f


def overacq(S, Ce_t, Cf_t, rho):
    """(bad, total) non-initial acquisitions: bad = held-out marginal utility <= 0."""
    bad = tot = 0
    for i in range(1, len(S)):
        tot += 1
        if U(Ce_t, Cf_t, S[:i + 1], rho) - U(Ce_t, Cf_t, S[:i], rho) <= 0:
            bad += 1
    return bad, tot


def run():
    fams = {f: _prep(ra.load(s), 0.5) for f, s in ra.FAM.items()}
    # three selectors on identical splits: CMV no-gate, CMV+gate, and GES (for the mechanism)
    SEL = {
        "CMV without gate":  lambda Cc, Fc, cost, rho: ra.greedy_normalized(Cc, Fc, cost, w_f=rho, lam=0.1, gate=False),
        "CMV + Wilson gate": lambda Cc, Fc, cost, rho: ra.greedy_normalized(Cc, Fc, cost, w_f=rho, lam=0.1, gate=True),
        "Greedy ensemble (GES)": lambda Cc, Fc, cost, rho: ges(Cc, Fc, rho),
    }
    rows = []
    for rho in RHOS:
        agg = {g: defaultdict(list) for g in SEL}
        oa = {g: [0, 0] for g in SEL}
        stab = {g: [] for g in SEL}
        for fam, P in fams.items():
            Ce, Cf, cost = P["Ce"], P["Cf"], P["cost"]; ne, nc = P["ne"], P["nc"]
            rng = np.random.RandomState(20260815)
            per = {g: defaultdict(list) for g in SEL}
            sets = {g: [] for g in SEL}
            for _ in range(NSPL):
                ep, cp = rng.permutation(ne), rng.permutation(nc)
                ce, te, cc, tc = ep[:CALE], ep[CALE:], cp[:CALC], cp[CALC:]
                Ce_c = {m: Ce[m][ce] for m in Ce}; Cf_c = {m: Cf[m][cc] for m in Cf}
                Ce_t = {m: Ce[m][te] for m in Ce}; Cf_t = {m: Cf[m][tc] for m in Cf}
                for g, fn in SEL.items():
                    S = fn(Ce_c, Cf_c, cost, rho)
                    per[g]["ins"].append(U(Ce_c, Cf_c, S, rho))
                    per[g]["held"].append(U(Ce_t, Cf_t, S, rho))
                    per[g]["nver"].append(len(set(S)))
                    c, f = ra.cascade(S, Ce_t, Cf_t)
                    per[g]["catch"].append(c); per[g]["fa"].append(f)
                    b, t = overacq(S, Ce_t, Cf_t, rho); oa[g][0] += b; oa[g][1] += t
                    sets[g].append(tuple(sorted(set(S))))
            for g in SEL:
                for k in ("ins", "held", "nver", "catch", "fa"):
                    agg[g][k].append(round(float(np.mean(per[g][k])), 3))   # per-family, as in tab:holdout
                stab[g].append(Counter(sets[g]).most_common(1)[0][1] / NSPL)
        for name in SEL:
            rows.append(dict(
                rho=rho, method=name,
                insample_U=round(np.mean(agg[name]["ins"]), 3),
                heldout_U=round(np.mean(agg[name]["held"]), 3),
                n_sources=round(np.mean(agg[name]["nver"]), 3),
                test_catch=round(np.mean(agg[name]["catch"]), 3),
                test_fa=round(np.mean(agg[name]["fa"]), 3),
                overacq_rate=(round(oa[name][0] / oa[name][1], 3) if oa[name][1] else float("nan")),
                n_acquisitions=oa[name][1],
                stability=round(np.mean(stab[name]), 3)))
    D = pd.DataFrame(rows)
    D.to_csv(f"{T}/rev_gate_ablation.csv", index=False)
    for rho in RHOS:
        print(f"=== rho={rho} ===")
        print(D[D.rho == rho].drop(columns="rho").to_string(index=False), "\n")
    print("wrote results/tables/rev_gate_ablation.csv")
    return D


if __name__ == "__main__":
    run()
