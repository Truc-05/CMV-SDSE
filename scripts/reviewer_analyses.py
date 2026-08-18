"""Reviewer-facing analyses computed from the EXISTING frozen runs (no new benchmarks):
  A. acquisition-policy ablation  (CMV-SDSE vs accuracy / low-correlation / random / same-model)
  B. budget curve                 (catch and utility vs. number of verifiers acquired)
  E. controller regret            (best fixed fusion rule minus controller, per cost ratio)
  F. probe-size sensitivity       (does an R-round probe recover the full selection order?)
All macro-averaged over the six identical-setup confirmatory families. Writes results/tables/rev_*.csv.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from theory.marginal_value import _prep, _wilson, greedy_marginal_value

T = "results/tables"
FAM = {"ARC": "mcqa__arc__ctx_truncate", "MMLU": "mcqa__mmlu__ctx_truncate",
       "CSQA": "mcqa__csqa__ctx_truncate", "TruthfulQA": "mcqa__truthfulqa__ctx_truncate",
       "GSM8K": "gsm8k__gsm8k__ctx_truncate", "MBPP": "mbpp__mbpp__ctx_truncate"}
RNG = np.random.RandomState(0)


def load(slug, keep_all_k=False):
    recs = [json.loads(l) for p in glob.glob(f"data/raw_cascade_runs/{slug}/*.jsonl")
            for l in open(p) if l.strip()]
    df = pd.DataFrame(recs)
    # Default keeps only the frozen k=50 pools (reproducibility of the frozen tables).
    # keep_all_k=True is needed for probes collected at a different k -- e.g. the gemma
    # diversity probe at k=3, which the #3 counterexample scan must see. Without it the
    # scan silently drops gemma2_9b and reports a stale "NONE found".
    return df if keep_all_k else df[df["n_gates"] == 50]


def cascade(models, Ce, Cf):
    """OR-cascade catch (TPR) and false-alarm (FPR) for an ordered/unordered model list."""
    if not models:
        return 0.0, 0.0
    c = np.zeros(len(next(iter(Ce.values()))), bool)
    fa = np.zeros(len(next(iter(Cf.values()))), bool)
    for m in models:
        c |= Ce[m]; fa |= Cf[m]
    return float(c.mean()), float(fa.mean())


def corr_with(m, S, Ce, Cf):
    rv = np.concatenate([Ce[m], Cf[m]]).astype(float)
    so = np.zeros(len(rv), bool)
    off = len(Ce[m])
    for u in S:
        so[:off] |= Ce[u]; so[off:] |= Cf[u]
    so = so.astype(float)
    return float(np.corrcoef(rv, so)[0, 1]) if rv.std() and so.std() else 0.0


def order_accuracy(models, Ce, Cf):
    return sorted(models, key=lambda m: Ce[m].mean(), reverse=True)


def order_lowcorr(models, Ce, Cf):
    solo = max(models, key=lambda m: Ce[m].mean())
    S = [solo]; rest = [m for m in models if m != solo]
    while rest:
        nxt = min(rest, key=lambda m: corr_with(m, S, Ce, Cf))
        S.append(nxt); rest.remove(nxt)
    return S


def greedy_normalized(Ce, Cf, cost, w_f=1.0, lam=0.1, gate=True):
    """Greedy on the normalised value V = [1-R]Δcatch - w_f[1-F]Δfa - λ c_v (Eq. cmv)."""
    models = list(Ce)
    start = max(models, key=lambda v: Ce[v].mean() - w_f * Cf[v].mean() - lam * cost[v])
    S = [start]; cc = Ce[start].copy(); cf = Cf[start].copy()
    while True:
        R, F = cc.mean(), cf.mean(); se, sc = ~cc, ~cf
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


# ---------- A. acquisition ablation + B. budget curve ----------
def acquisition(rho=1.0, K=3):
    abl, budg = [], []
    for fam, slug in FAM.items():
        df = load(slug); P = _prep(df, 0.5)
        models, Ce, Cf = P["models"], P["Ce"], P["Cf"]
        solo = max(models, key=lambda m: Ce[m].mean())
        solo_c, solo_f = cascade([solo], Ce, Cf)
        cmv = greedy_normalized(Ce, Cf, P["cost"])
        orders = {"CMV-SDSE": cmv, "accuracy": order_accuracy(models, Ce, Cf),
                  "low-correlation": order_lowcorr(models, Ce, Cf)}
        # random: average over permutations
        rand_rows = []
        for _ in range(30):
            perm = list(RNG.permutation(models))
            rand_rows.append(perm)
        # budget curve k=1..len(models)
        for name, order in list(orders.items()) + [("random", None)]:
            for k in range(1, len(models) + 1):
                if name == "random":
                    cs = np.mean([cascade(p[:k], Ce, Cf) for p in rand_rows], axis=0)
                    c, f = float(cs[0]), float(cs[1])
                else:
                    c, f = cascade(order[:k], Ce, Cf)
                budg.append(dict(family=fam, policy=name, k=k, catch=c, fa=f, U=c - rho * f))
        # same-model reference (repeats add nothing): flat at solo
        for k in range(1, len(models) + 1):
            budg.append(dict(family=fam, policy="same-model", k=k,
                             catch=solo_c, fa=solo_f, U=solo_c - rho * solo_f))
        # ablation table: each policy at min(K, stop) verifiers; CMV at its stop
        def at(order, kk):
            c, f = cascade(order[:kk], Ce, Cf); return c, f, kk
        cmv_c, cmv_f, cmv_k = at(cmv, len(cmv))
        rand_k = min(K, len(models))
        rc = np.mean([cascade(p[:rand_k], Ce, Cf) for p in rand_rows], axis=0)
        rows = {
            "random": (float(rc[0]), float(rc[1]), rand_k),
            "accuracy": at(orders["accuracy"], min(K, len(models))),
            "low-correlation": at(orders["low-correlation"], min(K, len(models))),
            "same-model": (solo_c, solo_f, 1),
            "CMV-SDSE": (cmv_c, cmv_f, cmv_k),
        }
        for name, (c, f, kk) in rows.items():
            abl.append(dict(family=fam, policy=name, catch=c, fa=f, ncalls=kk,
                            U=c - rho * f, residual=c - solo_c))
    A = pd.DataFrame(abl); B = pd.DataFrame(budg)
    Am = A.groupby("policy")[["catch", "fa", "ncalls", "U", "residual"]].mean().round(3)
    Am = Am.reindex(["same-model", "random", "accuracy", "low-correlation", "CMV-SDSE"])
    Am.to_csv(f"{T}/rev_acquisition_ablation.csv")
    B.groupby(["policy", "k"])[["catch", "fa", "U"]].mean().round(4).to_csv(f"{T}/rev_budget_curve.csv")
    print("=== A. acquisition ablation (macro, rho=1) ===\n", Am, "\n")
    return Am, B


def Ce_of(df):
    return {m: v.mean() for m, v in _prep(df, 0.5)["Ce"].items()}


# ---------- E. regret ----------
def regret():
    cs = pd.read_csv(f"{T}/fusion_cost_sweep.csv")
    cs["regret"] = (cs.best_fixed_U - cs.controller_U).round(3)
    cs[["cost_ratio_fa_over_miss", "best_fixed_rule", "best_fixed_U",
        "controller_U", "regret"]].to_csv(f"{T}/rev_regret.csv", index=False)
    print("=== E. controller regret ===\n", cs[["cost_ratio_fa_over_miss", "regret"]].to_string(index=False), "\n")


# ---------- F. probe-size sensitivity ----------
def _cematrix_capped(df, R):
    """Per-(model,item) majority verdict using only the first R decided gates."""
    from collections import defaultdict
    catch, fa = defaultdict(dict), defaultdict(dict)
    for _, r in df.iterrows():
        seen = defaultdict(list)
        for g in r["gates"]:
            v = g.get("verifier_id"); a = g.get("accepted")
            if a in (True, False) and len(seen[v]) < R:
                seen[v].append(a is True)
        for v, acc in seen.items():
            if not acc:
                continue
            rej = 1 if (sum(acc) / len(acc)) < 0.5 else 0
            (catch if r["ground_truth_is_error"] else fa)[v][r["item_id"]] = rej
    return catch, fa


def probe_sensitivity(Rs=(1, 2, 3, 5, 10)):
    out = []
    for fam, slug in FAM.items():
        df = load(slug)
        full = _prep(df, 0.5); models = full["models"]
        Cef = full["Ce"]
        solo_full = max(models, key=lambda m: Cef[m].mean())
        missed = ~Cef[solo_full]
        full_order = sorted([m for m in models if m != solo_full],
                            key=lambda m: Cef[m][missed].mean() if missed.any() else 0, reverse=True)
        for R in Rs:
            cat, _ = _cematrix_capped(df, R)
            items = sorted({it for m in models for it in cat.get(m, {})})
            idx = {it: i for i, it in enumerate(items)}
            Ce = {m: np.array([cat.get(m, {}).get(it, 0) for it in items], bool) for m in models}
            solo = max(models, key=lambda m: Ce[m].mean())
            miss = ~Ce[solo]
            order = sorted([m for m in models if m != solo],
                           key=lambda m: Ce[m][miss].mean() if miss.any() else 0, reverse=True)
            top_agree = int(order[0] == full_order[0]) if order and full_order else 0
            solo_agree = int(solo == solo_full)
            out.append(dict(family=fam, R=R, solo_match=solo_agree, top_diversifier_match=top_agree))
    P = pd.DataFrame(out)
    Pm = P.groupby("R")[["solo_match", "top_diversifier_match"]].mean().round(3)
    Pm.to_csv(f"{T}/rev_probe_sensitivity.csv")
    print("=== F. probe-size sensitivity (fraction of families matching full-data order) ===\n", Pm, "\n")
    return Pm


if __name__ == "__main__":
    acquisition()
    regret()
    probe_sensitivity()
    print("wrote results/tables/rev_{acquisition_ablation,budget_curve,regret,probe_sensitivity}.csv")
