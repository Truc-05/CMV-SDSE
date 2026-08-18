"""Second batch of reviewer analyses from the existing frozen runs:
  C. value-function component ablation (full CMV vs catch-only / no-FA / no-cost)
  G. paired bootstrap CI + effect size for the acquisition-policy utility gaps
  J. real compute cost (per-verifier latency) for each acquisition policy
Uses the NORMALISED decision value of Eq (cmv): V = [1-R]Δcatch - w_f[1-F]Δfa - λ c_v,
so the controller here matches the paper exactly. Writes results/tables/rev_*.csv.
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

T = "results/tables"
FAM = {"ARC": "mcqa__arc__ctx_truncate", "MMLU": "mcqa__mmlu__ctx_truncate",
       "CSQA": "mcqa__csqa__ctx_truncate", "TruthfulQA": "mcqa__truthfulqa__ctx_truncate",
       "GSM8K": "gsm8k__gsm8k__ctx_truncate", "MBPP": "mbpp__mbpp__ctx_truncate"}
RNG = np.random.RandomState(0)


def load(slug):
    recs = [json.loads(l) for p in glob.glob(f"data/raw_cascade_runs/{slug}/*.jsonl")
            for l in open(p) if l.strip()]
    df = pd.DataFrame(recs)
    return df[df["n_gates"] == 50]


def cascade(models, Ce, Cf):
    if not models:
        return 0.0, 0.0
    c = np.zeros(len(next(iter(Ce.values()))), bool)
    fa = np.zeros(len(next(iter(Cf.values()))), bool)
    for m in models:
        c |= Ce[m]; fa |= Cf[m]
    return float(c.mean()), float(fa.mean())


def greedy_normalized(Ce, Cf, cost, w_f=1.0, lam=0.1, gate=True):
    """Greedy on the normalised value V = [1-R]Δcatch - w_f[1-F]Δfa - λ c_v (Eq. cmv)."""
    models = list(Ce)
    start = max(models, key=lambda v: Ce[v].mean() - w_f * Cf[v].mean() - lam * cost[v])
    S = [start]
    cc = Ce[start].copy(); cf = Cf[start].copy()
    while True:
        R, F = cc.mean(), cf.mean()
        se, sc = ~cc, ~cf
        best, bestV = None, 1e-9
        for v in models:
            if v in S:
                continue
            dcatch = Ce[v][se].mean() if se.any() else 0.0
            dfa = Cf[v][sc].mean() if sc.any() else 0.0
            V = (1 - R) * dcatch - w_f * (1 - F) * dfa - lam * cost[v]
            if gate:
                k, n = int(Ce[v][se].sum()), int(se.sum())
                if not (n > 0 and _wilson(k, n)[0] > 0):
                    continue
            if V > bestV:
                bestV, best = V, v
        if best is None:
            break
        S.append(best); cc |= Ce[best]; cf |= Cf[best]
    return S


# ---------- C. value-function component ablation ----------
def component_ablation(rho=1.0):
    variants = {"full CMV": dict(w_f=1.0, lam=0.1, gate=True),
                "no false-alarm term": dict(w_f=0.0, lam=0.1, gate=True),
                "no compute term": dict(w_f=1.0, lam=0.0, gate=True),
                "catch-only (no FA, no cost)": dict(w_f=0.0, lam=0.0, gate=True),
                "no significance gate": dict(w_f=1.0, lam=0.1, gate=False)}
    rows = []
    for fam, slug in FAM.items():
        P = _prep(load(slug), 0.5)
        Ce, Cf, cost = P["Ce"], P["Cf"], P["cost"]
        for name, kw in variants.items():
            S = greedy_normalized(Ce, Cf, cost, **kw)
            c, f = cascade(S, Ce, Cf)
            rows.append(dict(family=fam, variant=name, catch=c, fa=f, ncalls=len(S), U=c - rho * f))
    A = pd.DataFrame(rows)
    Am = A.groupby("variant")[["catch", "fa", "ncalls", "U"]].mean().round(3)
    Am = Am.reindex(["full CMV", "no false-alarm term", "no compute term",
                     "catch-only (no FA, no cost)", "no significance gate"])
    Am.to_csv(f"{T}/rev_component_ablation.csv")
    print("=== C. value-function component ablation (rho=1, macro) ===\n", Am, "\n")
    return A


# ---------- G. paired bootstrap CI + effect size ----------
def bootstrap_paired(rho=1.0, B=2000):
    """Per-family utility of CMV vs each baseline; paired bootstrap over the six families."""
    fams = list(FAM); perU = {}
    for fam, slug in FAM.items():
        P = _prep(load(slug), 0.5); Ce, Cf, cost, models = P["Ce"], P["Cf"], P["cost"], P["models"]
        solo = max(models, key=lambda m: Ce[m].mean())
        def U(S): c, f = cascade(S, Ce, Cf); return c - rho * f
        cmv = greedy_normalized(Ce, Cf, cost)
        acc = sorted(models, key=lambda m: Ce[m].mean(), reverse=True)[:3]
        # low-correlation order
        S = [solo]; rest = [m for m in models if m != solo]
        while rest:
            def cw(m):
                rv = np.concatenate([Ce[m], Cf[m]]).astype(float)
                so = np.zeros(len(rv), bool); off = len(Ce[m])
                for u in S:
                    so[:off] |= Ce[u]; so[off:] |= Cf[u]
                so = so.astype(float)
                return float(np.corrcoef(rv, so)[0, 1]) if rv.std() and so.std() else 0.0
            nxt = min(rest, key=cw); S.append(nxt); rest.remove(nxt)
        lowc = S[:3]
        perU[fam] = dict(cmv=U(cmv), accuracy=U(acc), lowcorr=U(lowc), samemodel=U([solo]))
    dfU = pd.DataFrame(perU).T
    rows = []
    for base in ["accuracy", "lowcorr", "samemodel"]:
        d = (dfU["cmv"] - dfU[base]).values
        boot = [np.mean(RNG.choice(d, len(d), replace=True)) for _ in range(B)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        cohen = d.mean() / (d.std(ddof=1) + 1e-9)
        rows.append(dict(comparison=f"CMV-SDSE − {base}", mean_dU=round(float(d.mean()), 3),
                         ci_low=round(float(lo), 3), ci_high=round(float(hi), 3),
                         cohen_d=round(float(cohen), 2)))
    R = pd.DataFrame(rows)
    R.to_csv(f"{T}/rev_bootstrap.csv", index=False)
    print("=== G. paired bootstrap of ΔU (CMV − baseline), 6 families ===\n", R.to_string(index=False), "\n")
    return R


# ---------- J. real compute cost via per-verifier latency ----------
def latency_analysis():
    med = defaultdict(list)
    for fam, slug in FAM.items():
        for _, r in load(slug).iterrows():
            for g in r["gates"]:
                t = g.get("latency_seconds"); v = g.get("verifier_id")
                if t:
                    med[v].append(t)
    medlat = {v: float(np.median(ts)) for v, ts in med.items()}
    rows = []
    for fam, slug in FAM.items():
        P = _prep(load(slug), 0.5); Ce, Cf, cost, models = P["Ce"], P["Cf"], P["cost"], P["models"]
        solo = max(models, key=lambda m: Ce[m].mean())
        cmv = greedy_normalized(Ce, Cf, cost)
        pol = {"same-model (k=50)": [solo] * 50, "cross-family (all 3)": models[:3],
               "CMV-SDSE": cmv}
        for name, S in pol.items():
            lat = sum(medlat.get(m, 0.0) for m in S)
            c, f = cascade(list(dict.fromkeys(S)), Ce, Cf)
            rows.append(dict(family=fam, policy=name, calls=len(S),
                             wall_seconds=round(lat, 2), catch=round(c, 3), U=round(c - f, 3)))
    L = pd.DataFrame(rows)
    Lm = L.groupby("policy")[["calls", "wall_seconds", "catch", "U"]].mean().round(2)
    Lm = Lm.reindex(["same-model (k=50)", "cross-family (all 3)", "CMV-SDSE"])
    Lm.to_csv(f"{T}/rev_latency.csv")
    print("=== J. real compute cost (median per-verifier latency, macro) ===\n", Lm, "\n")
    print("median latency (s) per verifier:", {k: round(v, 2) for k, v in sorted(medlat.items())})
    return Lm


if __name__ == "__main__":
    component_ablation()
    bootstrap_paired()
    latency_analysis()
    print("wrote results/tables/rev_{component_ablation,bootstrap,latency}.csv")
