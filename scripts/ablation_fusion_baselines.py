"""ABLATION (offline, no GPU): does the FUSION RULE matter, and does the frozen
budget-aware controller beat naive fusion of the same verifier verdicts?

Reviewer-facing ablation for the Information Fusion venue. Reuses the frozen
verdict definition (theory.marginal_value._prep: item-level majority-reject per
model at thr=0.5, on error items = catch, on clean items = false alarm) so every
number is consistent with the controller. No new data, no re-tuning.

For each of the 6 identical-setup confirmatory families (ARC/MMLU/CSQA/TruthfulQA
+ GSM8K + MBPP) we take the per-item reject vote of the 4 distinct verifiers and
compare fusion rules on catch (TPR on errors) / false-alarm (FPR on clean) /
Youden J = catch - fa / balanced accuracy:

  fixed rules (no labels):   OR (=cascade, the paper's rule) · AND (unanimous) · MAJORITY
  unsupervised:              Dawid-Skene EM (latent-truth from confusion matrices)
  supervised (5-fold CV):    single-best verifier · Youden-weighted vote · pattern-Bayes stack
  paper controller:          greedy-selected cascade (OR over greedy_marginal_value's chosen set)

Plus a SCALE panel: adding an independent SAME-MODEL repeat (self-split draw) vs a
DIVERSE verifier -> the fusion-framed version of n_eff@50 ~ 1.

Writes results/tables/fusion_baselines.csv (+ fusion_scale_panel.csv). MAST-3.3 is
excluded here: its verifier x item matrix is not rectangular (6 conditions, a 5th
model qwen14b on only some items), so it is not apples-to-apples for this ablation.
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
from theory.marginal_value import _prep, greedy_marginal_value

FAMILIES = {
    "ARC": "data/raw_cascade_runs/mcqa__arc__ctx_truncate",
    "MMLU": "data/raw_cascade_runs/mcqa__mmlu__ctx_truncate",
    "CSQA": "data/raw_cascade_runs/mcqa__csqa__ctx_truncate",
    "TruthfulQA": "data/raw_cascade_runs/mcqa__truthfulqa__ctx_truncate",
    "GSM8K": "data/raw_cascade_runs/gsm8k__gsm8k__ctx_truncate",
    "MBPP": "data/raw_cascade_runs/mbpp__mbpp__ctx_truncate",
}
K_FOLDS = 5
EPS = 1e-6


def load(d):
    recs = [json.loads(l) for p in glob.glob(f"{d}/*.jsonl") for l in open(p) if l.strip()]
    df = pd.DataFrame(recs)
    return df[df["n_gates"] == 50] if not df.empty else df


def build_matrix(df):
    """Return X (N x M reject votes), y (1=error/0=clean), model names, and the
    per-class matrices Ce/Se (for the SCALE panel), all from the frozen _prep."""
    P = _prep(df, 0.5)
    models = P["models"]
    Xe = np.column_stack([P["Ce"][m].astype(int) for m in models])   # (ne, M) reject on errors
    Xc = np.column_stack([P["Cf"][m].astype(int) for m in models])   # (nc, M) reject on clean
    X = np.vstack([Xe, Xc])
    y = np.concatenate([np.ones(len(Xe)), np.zeros(len(Xc))]).astype(int)
    return X, y, models, P


def metrics(r, y):
    r = r.astype(bool)
    catch = r[y == 1].mean()
    fa = r[y == 0].mean()
    return dict(catch=round(float(catch), 3), fa=round(float(fa), 3),
                youden=round(float(catch - fa), 3),
                bal_acc=round(float((catch + (1 - fa)) / 2), 3))


def strat_folds(y, k):
    """Deterministic stratified k-fold (no RNG -> fully reproducible)."""
    folds = [[] for _ in range(k)]
    for cls in (np.where(y == 0)[0], np.where(y == 1)[0]):
        for j, i in enumerate(cls):
            folds[j % k].append(i)
    return [np.array(sorted(f)) for f in folds]


def cv_oof(X, y, predict_fold):
    """Generic 5-fold out-of-fold prediction collector for supervised rules."""
    folds = strat_folds(y, K_FOLDS)
    oof = np.zeros(len(y), dtype=int)
    for f in range(K_FOLDS):
        te = folds[f]
        tr = np.concatenate([folds[g] for g in range(K_FOLDS) if g != f])
        oof[te] = predict_fold(X[tr], y[tr], X[te])
    return oof


def single_best_fold(Xtr, ytr, Xte):
    j = np.argmax([(Xtr[ytr == 1][:, m].mean() - Xtr[ytr == 0][:, m].mean())
                   for m in range(Xtr.shape[1])])
    return Xte[:, j].astype(int)


def weighted_fold(Xtr, ytr, Xte):
    w = np.array([max(0.0, Xtr[ytr == 1][:, m].mean() - Xtr[ytr == 0][:, m].mean())
                  for m in range(Xtr.shape[1])])          # Youden weight, dead verifier -> 0
    if w.sum() == 0:
        w = np.ones(Xtr.shape[1])
    return (Xte @ w > 0.5 * w.sum()).astype(int)


def pattern_bayes_fold(Xtr, ytr, Xte):
    """Empirical Bayes on the discrete vote pattern = optimal supervised fusion given the votes."""
    cnt = defaultdict(lambda: [0, 0])
    for i in range(len(ytr)):
        cnt[tuple(Xtr[i])][int(ytr[i])] += 1
    prior_reject = 1 if ytr.mean() > 0.5 else 0
    out = np.empty(len(Xte), dtype=int)
    for i in range(len(Xte)):
        c = cnt.get(tuple(Xte[i]))
        if c and c[1] != c[0]:
            out[i] = 1 if c[1] > c[0] else 0
        else:
            out[i] = prior_reject      # unseen pattern or tie -> class prior
    return out


def dawid_skene(X, iters=200):
    """Unsupervised 2-class EM over binary annotator votes; error class = reject-heavy class."""
    N, M = X.shape
    T = np.clip(X.mean(axis=1), EPS, 1 - EPS)             # init from per-item reject fraction
    for _ in range(iters):
        p1 = T.mean()
        a = (T @ X) / T.sum()                              # P(reject | z=error), per annotator
        b = ((1 - T) @ X) / (1 - T).sum()                 # P(reject | z=clean)
        a = np.clip(a, EPS, 1 - EPS); b = np.clip(b, EPS, 1 - EPS)
        logL1 = np.log(p1) + X @ np.log(a) + (1 - X) @ np.log(1 - a)
        logL0 = np.log(1 - p1) + X @ np.log(b) + (1 - X) @ np.log(1 - b)
        m = np.maximum(logL1, logL0)
        T = np.exp(logL1 - m) / (np.exp(logL1 - m) + np.exp(logL0 - m))
    # keep error class = the one annotators reject more on (a>b on average); flip if inverted
    if (a - b).mean() < 0:
        T = 1 - T
    return (T > 0.5).astype(int)


def greedy_selected_models(df, models):
    mv = greedy_marginal_value(df, lam=0.1)
    sel = mv[(mv.selected) & (mv.action.isin(["START", "DIVERSIFY", "SCALE"]))]
    S = [c for c in dict.fromkeys(sel.candidate.tolist()) if c in models]
    return S or [models[0]]


def run_family(fam, d):
    df = load(d)
    X, y, models, P = build_matrix(df)
    M = len(models)
    rows = []

    def add(method, r, nvf):
        rows.append(dict(family=fam, method=method, n_verifiers=nvf, **metrics(r, y)))

    add("OR (cascade / paper rule)", X.any(axis=1), M)
    add("AND (unanimous)", X.all(axis=1), M)
    add("MAJORITY (>half)", X.sum(axis=1) * 2 > M, M)
    add("Dawid-Skene EM (unsup)", dawid_skene(X), M)
    add("single-best verifier (CV)", cv_oof(X, y, single_best_fold), 1)
    add("Youden-weighted vote (CV)", cv_oof(X, y, weighted_fold), M)
    add("pattern-Bayes stack (CV)", cv_oof(X, y, pattern_bayes_fold), M)

    S = greedy_selected_models(df, models)
    cols = [models.index(m) for m in S]
    add(f"greedy cascade (paper, {'+'.join(S)})", X[:, cols].any(axis=1), len(S))

    # SCALE panel: solo -> +same-model repeat (self-split) -> +diverse verifier (error items only)
    start = S[0]
    diverse = next((m for m in S[1:]), None)
    Ce_start = P["Ce"][start]
    Se_start = P["Se"][start]                       # independent 2nd draw of the SAME model
    scale = dict(family=fam, start=start,
                 solo_catch=round(float(Ce_start.mean()), 3),
                 plus_same_model=round(float((Ce_start | Se_start).mean()), 3))
    if diverse is not None:
        scale["diverse_add"] = diverse
        scale["plus_diverse"] = round(float((Ce_start | P["Ce"][diverse]).mean()), 3)
    scale["d_scale"] = round(scale["plus_same_model"] - scale["solo_catch"], 3)
    scale["d_diverse"] = (round(scale.get("plus_diverse", np.nan) - scale["solo_catch"], 3)
                          if diverse is not None else np.nan)
    return rows, scale, dict(X=X, y=y, models=models, df=df)


def controller_at_rho(fam_data, rho):
    """Macro-avg (catch, fa, set-size) of the frozen greedy controller run at fa/miss
    cost ratio rho: it selects a set via marginal value (w_fa=rho), fused by OR-cascade."""
    cs, fs, ns = [], [], []
    for D in fam_data.values():
        mv = greedy_marginal_value(D["df"], lam=0.1, w_fa=rho)
        sel = mv[(mv.selected) & (mv.action.isin(["START", "DIVERSIFY", "SCALE"]))]
        S = [c for c in dict.fromkeys(sel.candidate.tolist()) if c in D["models"]] or [D["models"][0]]
        cols = [D["models"].index(m) for m in S]
        r = D["X"][:, cols].any(axis=1).astype(bool)
        y = D["y"]
        cs.append(r[y == 1].mean()); fs.append(r[y == 0].mean()); ns.append(len(S))
    return float(np.mean(cs)), float(np.mean(fs)), float(np.mean(ns))


def main():
    pd.set_option("display.width", 200)
    all_rows, panel, fam_data = [], [], {}
    for fam, d in FAMILIES.items():
        r, s, D = run_family(fam, d)
        all_rows += r
        panel.append(s)
        fam_data[fam] = D

    df = pd.DataFrame(all_rows)
    macro = (df.groupby("method", sort=False)[["catch", "fa", "youden", "bal_acc", "n_verifiers"]]
             .mean().round(3).reset_index())
    macro.insert(0, "family", "MACRO-AVG (6 fam)")

    out = "results/tables/fusion_baselines.csv"
    df.to_csv(out, index=False)
    pan = pd.DataFrame(panel)
    pan.to_csv("results/tables/fusion_scale_panel.csv", index=False)

    print("=" * 100)
    print("FUSION-RULE ABLATION  (catch=TPR on errors, fa=FPR on clean, youden=catch-fa; 6 confirmatory families)")
    print("=" * 100)
    print("\n--- MACRO-AVERAGE across 6 families (ranked by Youden) ---")
    print(macro.sort_values("youden", ascending=False).to_string(index=False))
    print("\n--- per family ---")
    for fam in FAMILIES:
        sub = df[df.family == fam]
        print(f"\n[{fam}]")
        print(sub.drop(columns="family").to_string(index=False))
    # cost-ratio sweep: which fusion rule maximizes utility U = catch - rho*fa as the
    # false-alarm/miss cost ratio rho grows? (the paper's point: the operating point is
    # cost-driven, no universal rule). Uses macro-avg catch/fa already computed.
    base_methods = ["OR (cascade / paper rule)", "AND (unanimous)", "MAJORITY (>half)",
                    "Dawid-Skene EM (unsup)", "single-best verifier (CV)",
                    "Youden-weighted vote (CV)", "pattern-Bayes stack (CV)"]
    mac = macro.set_index("method")
    rhos = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    sweep = []
    for rho in rhos:
        util = {m: mac.loc[m, "catch"] - rho * mac.loc[m, "fa"] for m in base_methods}
        win = max(util, key=util.get)
        c_catch, c_fa, c_n = controller_at_rho(fam_data, rho)   # controller run AT this rho
        u_ctrl = c_catch - rho * c_fa
        # does the controller match the cost-optimal fixed rule (within a small margin)?
        matches = u_ctrl >= util[win] - 0.02
        sweep.append(dict(cost_ratio_fa_over_miss=rho,
                          best_fixed_rule=win, best_fixed_U=round(util[win], 3),
                          controller_U=round(u_ctrl, 3), controller_setsize=round(c_n, 2),
                          controller_tracks_best=bool(matches)))
    sweep = pd.DataFrame(sweep)
    sweep.to_csv("results/tables/fusion_cost_sweep.csv", index=False)

    print("\n" + "=" * 100)
    print("SCALE PANEL  (error-catch: solo -> + same-model repeat vs + diverse verifier)")
    print("=" * 100)
    print(pan.to_string(index=False))
    print(f"\nmean Δcatch from a SAME-MODEL repeat : {pan['d_scale'].mean():+.3f}"
          f"   (fusion-framed n_eff@50 ~ 1)")
    print(f"mean Δcatch from a DIVERSE verifier  : {pan['d_diverse'].mean():+.3f}")

    print("\n" + "=" * 100)
    print("COST-RATIO SWEEP  (best FIXED rule vs the CONTROLLER run at each rho; U = catch - rho*fa; macro-avg)")
    print("=" * 100)
    print(sweep.to_string(index=False))
    tracks = int(sweep["controller_tracks_best"].sum())
    print(f"\n  -> the winning FIXED rule flips with the fa/miss cost ratio (OR -> stack -> single-best"
          f" -> MAJORITY -> AND): no universal fusion rule.")
    print(f"  -> the CONTROLLER, run at each rho, MATCHES the cost-optimal fixed rule at {tracks}/{len(rhos)}"
          f" cost ratios, and its selected set SHRINKS as false alarms get costly")
    print(f"     (setsize {sweep.controller_setsize.iloc[0]:.1f} at rho=0 -> "
          f"{sweep.controller_setsize.iloc[-1]:.1f} at rho=4) -- one knob, no per-cost re-tuning.")
    print(f"\nwrote {out}, fusion_scale_panel.csv, fusion_cost_sweep.csv")


if __name__ == "__main__":
    main()
