"""Third batch: the remaining reviewer analyses, all from existing frozen runs.
  H1. leave-one-family-out (LOFO)   -- do the two headline results survive dropping any family?
  H2. leave-one-model-out  (LOMO)   -- is any single verifier load-bearing?
  H3. per-model breakdown            -- is one model dragging the average?
  E2. per-family fusion winner       -- does "no universal rule" hold per family, not just macro?
  B2. Verification Debt / Efficiency -- calls vs effective sources
  F2. probe cost vs final utility    -- does the R-round probe controller match full-data utility?
Writes results/tables/rev3_*.csv.
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from theory.marginal_value import _prep, _wilson
from reviewer_analyses import load, cascade, greedy_normalized, FAM, _cematrix_capped

T = "results/tables"


# ---------- H1. leave-one-family-out ----------
def lofo():
    d = pd.read_csv(f"{T}/redundancy_invariant_7family.csv")
    rows = []
    for drop in ["(none)"] + d.family.tolist():
        rem = d if drop == "(none)" else d[d.family != drop]
        g = rem.dropna(subset=["div_unique"])
        rows.append(dict(dropped=drop,
                         neff_min=round(rem.sm_n_eff50.min(), 2),
                         neff_max=round(rem.sm_n_eff50.max(), 2),
                         neff_mean=round(rem.sm_n_eff50.mean(), 3),
                         diversify_pooled=f"{int(g.div_unique.sum())}/{int(g.div_denom.sum())}",
                         families_ci_low_pos=int((g.div_ci_low > 0).sum())))
    L = pd.DataFrame(rows)
    L.to_csv(f"{T}/rev3_lofo.csv", index=False)
    print("=== H1. leave-one-family-out ===\n", L.to_string(index=False), "\n")


# ---------- H2. leave-one-model-out ----------
def lomo(rho=1.0):
    full, per = [], defaultdict(list)
    for fam, slug in FAM.items():
        P = _prep(load(slug), 0.5); Ce, Cf, cost, models = P["Ce"], P["Cf"], P["cost"], P["models"]
        c, f = cascade(greedy_normalized(Ce, Cf, cost), Ce, Cf); full.append(c - rho * f)
        for m in models:
            keep = [x for x in models if x != m]
            Ce2 = {x: Ce[x] for x in keep}; Cf2 = {x: Cf[x] for x in keep}; co2 = {x: cost[x] for x in keep}
            c, f = cascade(greedy_normalized(Ce2, Cf2, co2), Ce2, Cf2)
            per[m].append(c - rho * f)
    rows = [dict(removed="(full pool)", cmv_U=round(float(np.mean(full)), 3))]
    for m, us in per.items():
        rows.append(dict(removed=m, cmv_U=round(float(np.mean(us)), 3),
                         drop_vs_full=round(float(np.mean(us) - np.mean(full)), 3)))
    L = pd.DataFrame(rows)
    L.to_csv(f"{T}/rev3_lomo.csv", index=False)
    print("=== H2. leave-one-model-out (macro CMV-SDSE U, rho=1) ===\n", L.to_string(index=False), "\n")


# ---------- H3. per-model breakdown ----------
def per_model():
    lat = defaultdict(list)
    catch, fa, seldiv, selstart = defaultdict(list), defaultdict(list), defaultdict(int), defaultdict(int)
    for fam, slug in FAM.items():
        df = load(slug); P = _prep(df, 0.5); Ce, Cf, cost, models = P["Ce"], P["Cf"], P["cost"], P["models"]
        for m in models:
            catch[m].append(Ce[m].mean()); fa[m].append(Cf[m].mean())
        S = greedy_normalized(Ce, Cf, cost, w_f=0.0, lam=0.0)  # miss-dominated regime: diversification fires
        selstart[S[0]] += 1
        for m in S[1:]:
            seldiv[m] += 1
        for _, r in df.iterrows():
            for g in r["gates"]:
                if g.get("latency_seconds"):
                    lat[g["verifier_id"]].append(g["latency_seconds"])
    allm = sorted(catch)
    rows = [dict(model=m, standalone_catch=round(float(np.mean(catch[m])), 3),
                 standalone_fa=round(float(np.mean(fa[m])), 3),
                 median_latency_s=round(float(np.median(lat[m])), 2),
                 times_solo=selstart.get(m, 0), times_diversifier=seldiv.get(m, 0)) for m in allm]
    L = pd.DataFrame(rows)
    L.to_csv(f"{T}/rev3_per_model.csv", index=False)
    print("=== H3. per-model breakdown (over 6 families) ===\n", L.to_string(index=False), "\n")


# ---------- E2. per-family fusion winner ----------
def fusion_winner():
    fb = pd.read_csv(f"{T}/fusion_baselines.csv")
    short = {"single-best verifier (CV)": "single-best", "Youden-weighted vote (CV)": "weighted",
             "pattern-Bayes stack (CV)": "stack", "MAJORITY (>half)": "MAJORITY",
             "Dawid-Skene EM (unsup)": "Dawid-Skene", "OR (cascade / paper rule)": "OR",
             "AND (unanimous)": "AND"}
    fb = fb[fb.method.isin(short)]
    rows = []
    for fam, g in fb.groupby("family"):
        w = g.loc[g.youden.idxmax()]
        rows.append(dict(family=fam, best_rule_at_rho1=short[w.method], youden=round(w.youden, 3)))
    L = pd.DataFrame(rows)
    L.to_csv(f"{T}/rev3_fusion_winner.csv", index=False)
    print("=== E2. per-family best fusion rule at rho=1 ===\n", L.to_string(index=False), "\n")


# ---------- B2. verification debt / efficiency ----------
def debt():
    d = pd.read_csv(f"{T}/redundancy_invariant_7family.csv")
    d = d.assign(verification_debt=(50 - d.sm_n_eff50).round(2),
                 efficiency=(d.sm_n_eff50 / 50).round(4))
    d[["family", "sm_n_eff50", "verification_debt", "efficiency"]].to_csv(f"{T}/rev3_debt.csv", index=False)
    print("=== B2. verification debt (50 - n_eff) and efficiency (n_eff/50) ===\n",
          d[["family", "sm_n_eff50", "verification_debt", "efficiency"]].to_string(index=False), "\n")


# ---------- F2. probe cost vs final utility ----------
def probe_utility(Rs=(1, 2, 3, 5, 10, 50), rho=1.0):
    rows = []
    for fam, slug in FAM.items():
        df = load(slug); models = _prep(df, 0.5)["models"]; cost = _prep(df, 0.5)["cost"]
        for R in Rs:
            cat, faa = _cematrix_capped(df, R)
            items_e = sorted({it for m in models for it in cat.get(m, {})})
            items_c = sorted({it for m in models for it in faa.get(m, {})})
            Ce = {m: np.array([cat.get(m, {}).get(it, 0) for it in items_e], bool) for m in models}
            Cf = {m: np.array([faa.get(m, {}).get(it, 0) for it in items_c], bool) for m in models}
            S = greedy_normalized(Ce, Cf, cost)
            c, f = cascade(S, Ce, Cf)
            rows.append(dict(family=fam, R=R, U=c - rho * f))
    P = pd.DataFrame(rows)
    Pm = P.groupby("R").U.mean().round(3)
    Pm.to_csv(f"{T}/rev3_probe_utility.csv")
    print("=== F2. controller utility from an R-round probe (macro, rho=1) ===\n", Pm, "\n")


if __name__ == "__main__":
    lofo(); lomo(); per_model(); fusion_winner(); debt(); probe_utility()
    print("wrote results/tables/rev3_*.csv")
