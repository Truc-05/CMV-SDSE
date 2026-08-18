"""Information-fusion / source-selection baselines, on the SAME verdicts and budget as CMV-SDSE.

Two canonical selectors from the fusion / ensemble-selection literature, the closest prior art to
conditional-marginal-value acquisition:

  GES  -- Greedy Ensemble Selection (Caruana, Niculescu-Mizil, Crew & Ksikes, ICML 2004):
          forward hill-climbing that repeatedly adds (with replacement) the verifier maximising the
          fused decision metric U = catch - rho*FA on the acquired verdicts; stops when no add helps.
          No compute term, no significance gate.
  mRMR -- max-Relevance min-Redundancy (Peng, Long & Ding, IEEE TPAMI 2005): greedily add the source
          with the highest mutual information with the ground truth minus its mean MI with the sources
          already chosen. The principled form of "accurate but diverse" selection.

Both run on the frozen Ce/Cf (reconciles to CMV-SDSE U=0.486). We compare in-sample at rho=1, across
cost regimes, and out-of-sample (held-out calibration->test) against CMV-SDSE. Writes
results/tables/rev_fusion_selection.csv (in-sample) and rev_fusion_selection_heldout.csv.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from theory.marginal_value import _prep
import reviewer_analyses as ra

T = "results/tables"


def U(Ce, Cf, S, rho):
    c, f = ra.cascade(S, Ce, Cf); return c - rho * f


def ges(Ce, Cf, rho):
    """Caruana greedy ensemble selection on the OR-cascade metric U=catch-rho*FA."""
    models = list(Ce); S = []
    while True:
        base = U(Ce, Cf, S, rho) if S else -1e9
        cand, cbest = None, base
        for m in models:
            u = U(Ce, Cf, S + [m], rho)
            if u > cbest + 1e-12:
                cbest, cand = u, m
        if cand is None:
            break
        S.append(cand)
    return S


def _mi(a, b):
    """Mutual information (nats) between two binary arrays over the same items."""
    mi = 0.0
    for va in (0, 1):
        for vb in (0, 1):
            pab = np.mean((a == va) & (b == vb))
            if pab <= 0:
                continue
            pa = np.mean(a == va); pb = np.mean(b == vb)
            mi += pab * np.log(pab / (pa * pb))
    return float(mi)


def mrmr(Ce, Cf, rho=None):
    """Peng et al. mRMR: relevance = MI(verdict; truth); redundancy = mean MI with selected."""
    models = list(Ce)
    ne = len(next(iter(Ce.values()))); nc = len(next(iter(Cf.values())))
    V = {m: np.concatenate([Ce[m], Cf[m]]).astype(int) for m in models}
    Y = np.concatenate([np.ones(ne, int), np.zeros(nc, int)])
    rel = {m: _mi(V[m], Y) for m in models}
    S = [max(models, key=lambda m: rel[m])]
    while len(S) < len(models):
        best, bscore = None, 0.0                      # stop when no positive mRMR score
        for m in models:
            if m in S:
                continue
            red = np.mean([_mi(V[m], V[u]) for u in S])
            if rel[m] - red > bscore:
                bscore, best = rel[m] - red, m
        if best is None:
            break
        S.append(best)
    return S


METHODS = {"Ensemble selection (GES)": ges, "mRMR": mrmr}


def insample():
    # (a) full table at rho=1 ; (b) utility across regimes
    fam_rows, reg_rows = [], []
    for fam, slug in ra.FAM.items():
        P = _prep(ra.load(slug), 0.5); Ce, Cf, cost = P["Ce"], P["Cf"], P["cost"]
        sels = {name: fn(Ce, Cf, 1.0) for name, fn in METHODS.items()}
        sels["CMV-SDSE"] = ra.greedy_normalized(Ce, Cf, cost)
        for name, S in sels.items():
            c, f = ra.cascade(S, Ce, Cf)
            fam_rows.append(dict(family=fam, policy=name, catch=c, fa=f,
                                 nver=len(set(S)), U=c - 1.0 * f))
        for rho in (0.25, 1.0, 2.0):
            s = {name: fn(Ce, Cf, rho) for name, fn in METHODS.items()}
            s["CMV-SDSE"] = ra.greedy_normalized(Ce, Cf, cost, w_f=rho, lam=0.1, gate=True)
            for name, S in s.items():
                reg_rows.append(dict(family=fam, rho=rho, policy=name, U=U(Ce, Cf, S, rho)))
    A = (pd.DataFrame(fam_rows).groupby("policy")[["catch", "fa", "nver", "U"]].mean().round(3)
         .reindex(["Ensemble selection (GES)", "mRMR", "CMV-SDSE"]))
    A.to_csv(f"{T}/rev_fusion_selection.csv")
    R = (pd.DataFrame(reg_rows).groupby(["policy", "rho"])["U"].mean().round(3)
         .unstack("rho").reindex(["Ensemble selection (GES)", "mRMR", "CMV-SDSE"]))
    R.to_csv(f"{T}/rev_fusion_selection_regimes.csv")   # across-regime cols for tab:fusionselect (was print-only)
    print("=== in-sample acquisition at rho=1 (macro over 6 families) ===\n", A, "\n")
    print("=== in-sample utility across regimes ===\n", R, "\n")
    return A, R


def heldout(rho=1.0, nspl=200, cale=50, calc=25):
    macro = defaultdict(list)
    for fam, slug in ra.FAM.items():
        P = _prep(ra.load(slug), 0.5); Ce, Cf, cost = P["Ce"], P["Cf"], P["cost"]
        ne, nc = P["ne"], P["nc"]; rng = np.random.RandomState(20260815)
        acc = defaultdict(list)
        for _ in range(nspl):
            ep, cp = rng.permutation(ne), rng.permutation(nc)
            ce, te = ep[:cale], ep[cale:]; cc, tc = cp[:calc], cp[calc:]
            Ce_c = {m: Ce[m][ce] for m in Ce}; Cf_c = {m: Cf[m][cc] for m in Cf}
            Ce_t = {m: Ce[m][te] for m in Ce}; Cf_t = {m: Cf[m][tc] for m in Cf}
            sels = {name: fn(Ce_c, Cf_c, rho) for name, fn in METHODS.items()}
            sels["CMV-SDSE"] = ra.greedy_normalized(Ce_c, Cf_c, cost, w_f=rho, lam=0.1, gate=True)
            for name, S in sels.items():
                acc[name].append(U(Ce_t, Cf_t, S, rho))
        for name in acc:
            macro[name].append(round(float(np.mean(acc[name])), 3))   # per-family, as in tab:holdout
    D = pd.DataFrame({"policy": list(macro), "heldout_test_U": [round(np.mean(v), 3) for v in macro.values()]})
    D = D.set_index("policy").reindex(["Ensemble selection (GES)", "mRMR", "CMV-SDSE"])
    D.to_csv(f"{T}/rev_fusion_selection_heldout.csv")
    print(f"=== held-out test utility (rho={rho}, {nspl} splits, macro) ===\n", D, "\n")
    return D


if __name__ == "__main__":
    insample()
    heldout()
    print("wrote results/tables/rev_fusion_selection{,_heldout}.csv")
