"""Joint (source, topology) acquisition controller  -- the co-optimisation the paper
currently leaves to future work (nd.md lines 914-916:  V_g(v|S)=U_g(S u {v})-U_g(S)).

The frozen CMV-SDSE controller optimises the marginal value of the DISJUNCTIVE (reject-any /
OR) cascade and only *later* compares other fusion rules.  A reviewer's fair objection: why is
acquisition optimised for OR while the other operators are evaluated separately?  Its sharpest
symptom is rho=4, where the OR controller cannot represent the conjunctive (AND) optimum and
its utility turns negative (U=-0.034 vs AND +0.093, regret 0.127).

This script builds the JOINT controller.  State = (S, g).  At each step it considers, over the
five fusion topologies G = {OR, AND, MAJORITY, weighted, stack}, both
    - adding a new source v under topology g:   marginal = U_g(S u {v}) - U_{g_cur}(S) - lam*c_v
    - re-fusing the CURRENT set under g:         marginal = U_g(S)       - U_{g_cur}(S)
and commits the highest positive cost-adjusted marginal, stopping when none is positive.

DECISIVE experiment (the reason this is not a rho=4 patch): the learned topologies (weighted,
stack) are LEARNED.  Selecting a topology on the same verdicts used to score it would be oracle
/ in-sample topology selection.  So everything -- which sources, which topology, and the learned
topology's parameters -- is chosen on a CALIBRATION split, FROZEN, and evaluated on a disjoint
HELD-OUT test split, mirroring the RQ6 protocol (cale=50 errors, calc=25 clean, 200 splits).
In-calibration topology scoring for the learned rules uses 5-fold CV so the greedy is not fooled
by their in-sample fit; the chosen rule is then refit on the full calibration for deployment.

Ablation ladder (held-out test U per cost regime rho):
    fixed-OR CMV        -- the frozen controller (OR selection, OR fusion)
    topology-only       -- OR-selected set, best topology chosen on calibration
    joint source+topology

Writes results/tables/joint_topology_heldout.csv and joint_topology_insample.csv.
6 rectangular confirmatory families (MAST-3.3 excluded, as in the fusion ablation).
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
from ablation_fusion_baselines import weighted_fold, pattern_bayes_fold, strat_folds

T = "results/tables"
LAM = 0.1                      # compute-cost weight, same as the frozen controller
TOL = 1e-9
RHOS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
G = ("OR", "AND", "MAJORITY", "weighted", "stack")     # the 5 topologies the controller may pick
FIXED = {"OR", "AND", "MAJORITY"}


# ---------- fusion rules on a column subset --------------------------------------------------
def _fixed_predict(Xcols, g):
    """OR / AND / MAJORITY reject vector on an (N x k) subset of the vote matrix."""
    k = Xcols.shape[1]
    if g == "OR":
        return Xcols.any(axis=1)
    if g == "AND":
        return Xcols.all(axis=1)
    if g == "MAJORITY":
        return Xcols.sum(axis=1) * 2 > k        # strict >half, ties -> pass (matches ablation)
    raise ValueError(g)


def _learned_predict(Xtr, ytr, Xte, g):
    """Fit a learned topology on (Xtr,ytr), predict on Xte (columns already subset)."""
    if g == "weighted":
        return weighted_fold(Xtr, ytr, Xte).astype(bool)
    if g == "stack":
        return pattern_bayes_fold(Xtr, ytr, Xte).astype(bool)
    raise ValueError(g)


def _U(reject, y, rho):
    catch = reject[y == 1].mean() if (y == 1).any() else 0.0
    fa = reject[y == 0].mean() if (y == 0).any() else 0.0
    return float(catch - rho * fa)                      # U = catch - rho*fa


def U_test(cols, g, Xtr, ytr, Xte, yte, rho):
    """Held-out utility: fit g on calibration columns `cols`, evaluate on test."""
    if not cols:
        return 0.0
    if g in FIXED:
        return _U(_fixed_predict(Xte[:, cols], g), yte, rho)
    return _U(_learned_predict(Xtr[:, cols], ytr, Xte[:, cols], g), yte, rho)


def U_cal(cols, g, Xtr, ytr, rho, folds):
    """In-calibration utility used to SELECT. Fixed rules: direct (label-free). Learned rules:
    5-fold out-of-fold CV so the greedy is not rewarded for in-sample overfit."""
    if not cols:
        return 0.0
    if g in FIXED:
        return _U(_fixed_predict(Xtr[:, cols], g), ytr, rho)
    oof = np.zeros(len(ytr), dtype=bool)
    for f in range(len(folds)):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != f])
        oof[te] = _learned_predict(Xtr[tr][:, cols], ytr[tr], Xtr[te][:, cols], g)
    return _U(oof, ytr, rho)


def _components(reject, y, rho):
    """(catch, fa, U) for a reject vector."""
    catch = reject[y == 1].mean() if (y == 1).any() else 0.0
    fa = reject[y == 0].mean() if (y == 0).any() else 0.0
    return float(catch), float(fa), float(catch - rho * fa)


def predict_test(cols, g, Xtr, ytr, Xte):
    """Frozen reject vector on test: fixed rules label-free; learned rules fit on full calibration."""
    if not cols:
        return np.zeros(len(Xte), dtype=bool)
    if g in FIXED:
        return _fixed_predict(Xte[:, list(cols)], g)
    return _learned_predict(Xtr[:, list(cols)], ytr, Xte[:, list(cols)], g)


def predict_cal(cols, g, Xtr, ytr, folds):
    """Selection-time reject vector on calibration (CV out-of-fold for learned rules)."""
    if not cols:
        return np.zeros(len(ytr), dtype=bool)
    if g in FIXED:
        return _fixed_predict(Xtr[:, list(cols)], g)
    oof = np.zeros(len(ytr), dtype=bool)
    for f in range(len(folds)):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != f])
        oof[te] = _learned_predict(Xtr[tr][:, list(cols)], ytr[tr], Xtr[te][:, list(cols)], g)
    return oof


# ---------- the joint controller (selection happens on calibration only) ---------------------
def joint_greedy(Xtr, ytr, cost, rho, folds):
    """Greedy over the joint (source-set, topology) lattice. Returns (cols, g)."""
    M = Xtr.shape[1]
    solo = [float(Xtr[ytr == 1][:, m].mean() - rho * Xtr[ytr == 0][:, m].mean() - LAM * cost[m])
            for m in range(M)]
    s0 = int(np.argmax(solo))                       # forced start = best single source at rho
    S, g_cur = [s0], "OR"
    U_cur = U_cal(S, "OR", Xtr, ytr, rho, folds)
    while True:
        best, best_marg = None, TOL
        # (a) add a new source v under topology g
        for v in range(M):
            if v in S:
                continue
            for g in G:
                u = U_cal(S + [v], g, Xtr, ytr, rho, folds)
                marg = u - U_cur - LAM * cost[v]
                if marg > best_marg:
                    best_marg, best = marg, ("add", v, g, u)
        # (b) re-fuse the current set under a different topology (zero compute cost)
        for g in G:
            if g == g_cur:
                continue
            u = U_cal(S, g, Xtr, ytr, rho, folds)
            marg = u - U_cur
            if marg > best_marg:
                best_marg, best = marg, ("switch", None, g, u)
        if best is None:
            break
        kind, v, g, u = best
        if kind == "add":
            S.append(v)
        g_cur, U_cur = g, u
    return S, g_cur


def or_selected_cols(Ce_c, Cf_c, cost_d, models, rho):
    """The frozen OR controller's selected set (reuses ra.greedy_normalized), as column idx."""
    S = ra.greedy_normalized(Ce_c, Cf_c, cost_d, w_f=rho, lam=LAM, gate=True)
    return [models.index(m) for m in dict.fromkeys(S) if m in models]


def _mats(Ce, Cf, models, e_idx, c_idx):
    """Vote matrix X (err rows then clean rows) over the given item indices, and labels y."""
    Xe = np.column_stack([Ce[m][e_idx].astype(int) for m in models])
    Xc = np.column_stack([Cf[m][c_idx].astype(int) for m in models])
    X = np.vstack([Xe, Xc])
    y = np.concatenate([np.ones(len(e_idx), int), np.zeros(len(c_idx), int)])
    return X, y


def heldout(nspl=200, cale=50, calc=25, per_family_csv=False):
    """Full held-out analysis. Per (family, rho, rung): test catch/fa/U, calibration U (the
    selection-time estimate), cal->test optimism gap, selected-set size, and topology share.
    Rungs: fixed_OR (frozen controller) -> topology_only (OR set, best topology on cal) ->
    joint (co-select source + topology). Macro table stays compatible with calc_sensitivity."""
    rec = defaultdict(list)                          # (fam,rho,rung,metric) -> [over splits]
    gc = defaultdict(lambda: defaultdict(int))       # (fam,rho,rung) -> {g: count}
    gpick = defaultdict(lambda: defaultdict(int))    # rho -> g -> count (joint, macro)
    for fam, slug in ra.FAM.items():
        P = _prep(ra.load(slug), 0.5)
        models, Ce, Cf, cost_d = P["models"], P["Ce"], P["Cf"], P["cost"]
        ne, nc = P["ne"], P["nc"]
        cost = np.array([cost_d[m] for m in models])
        rng = np.random.RandomState(20260815)
        for _ in range(nspl):
            ep, cp = rng.permutation(ne), rng.permutation(nc)
            ce, te = ep[:cale], ep[cale:]; cc, tc = cp[:calc], cp[calc:]
            Xtr, ytr = _mats(Ce, Cf, models, ce, cc)
            Xte, yte = _mats(Ce, Cf, models, te, tc)
            folds = strat_folds(ytr, 5)
            Ce_c = {m: Ce[m][ce] for m in models}; Cf_c = {m: Cf[m][cc] for m in models}
            for rho in RHOS:
                cols_or = or_selected_cols(Ce_c, Cf_c, cost_d, models, rho) or [int(np.argmax(
                    [Ce[m][ce].mean() for m in models]))]
                g2 = max(G, key=lambda g: U_cal(cols_or, g, Xtr, ytr, rho, folds))
                cols_j, gj = joint_greedy(Xtr, ytr, cost, rho, folds)
                for rung, cols, g in (("fixed_OR", cols_or, "OR"),
                                      ("topology_only", cols_or, g2),
                                      ("joint", cols_j, gj)):
                    ct, ft, ut = _components(predict_test(cols, g, Xtr, ytr, Xte), yte, rho)
                    _, _, uc = _components(predict_cal(cols, g, Xtr, ytr, folds), ytr, rho)
                    rec[(fam, rho, rung, "catch")].append(ct)
                    rec[(fam, rho, rung, "fa")].append(ft)
                    rec[(fam, rho, rung, "U")].append(ut)
                    rec[(fam, rho, rung, "Ucal")].append(uc)
                    rec[(fam, rho, rung, "setsize")].append(len(set(cols)))
                    gc[(fam, rho, rung)][g] += 1
                full = list(range(len(models)))
                rec[(fam, rho, "oracle", "U")].append(
                    max(U_test(full, g, Xtr, ytr, Xte, yte, rho) for g in G))
                gpick[rho][gj] += 1
    # ---- per-family long table ----
    M = lambda k: float(np.mean(rec[k]))
    pf_rows = []
    for (fam, rho, rung), gcount in gc.items():
        tot = sum(gcount.values())
        pf_rows.append(dict(
            family=fam, rho=rho, rung=rung,
            catch=round(M((fam, rho, rung, "catch")), 3), fa=round(M((fam, rho, rung, "fa")), 3),
            U_test=round(M((fam, rho, rung, "U")), 3), U_cal=round(M((fam, rho, rung, "Ucal")), 3),
            opt_gap=round(M((fam, rho, rung, "U")) - M((fam, rho, rung, "Ucal")), 3),
            setsize=round(M((fam, rho, rung, "setsize")), 2),
            g_mode=max(gcount, key=gcount.get),
            nonOR_share=round(sum(v for k, v in gcount.items() if k != "OR") / tot, 2)))
    PF = pd.DataFrame(pf_rows).sort_values(["rho", "rung", "family"])
    if per_family_csv:
        PF.to_csv(f"{T}/joint_topology_perfamily.csv", index=False)
    # ---- macro table (compatible with calc_sensitivity) ----
    rows = []
    for rho in RHOS:
        d = {rung: round(float(np.mean([M((fam, rho, rung, "U")) for fam in ra.FAM])), 3)
             for rung in ("fixed_OR", "topology_only", "joint")}
        d["oracle_fullpool"] = round(float(np.mean([M((fam, rho, "oracle", "U")) for fam in ra.FAM])), 3)
        d["regret_joint"] = round(d["oracle_fullpool"] - d["joint"], 3)
        d["regret_fixed_OR"] = round(d["oracle_fullpool"] - d["fixed_OR"], 3)
        tot = sum(gpick[rho].values())
        d["joint_AND_share"] = round(gpick[rho].get("AND", 0) / tot, 2)
        d["joint_nonOR_share"] = round(sum(gpick[rho][g] for g in G if g != "OR") / tot, 2)
        rows.append(dict(rho=rho, **d))
    D = pd.DataFrame(rows).set_index("rho")
    D.to_csv(f"{T}/joint_topology_heldout.csv")
    pd.set_option("display.width", 220)
    print(f"=== HELD-OUT macro test U ({nspl} splits) ===")
    print(D[["fixed_OR", "topology_only", "joint", "oracle_fullpool",
             "regret_fixed_OR", "regret_joint", "joint_AND_share"]], "\n")
    if per_family_csv:
        for rho in (1.0, 4.0):
            print(f"--- per-family @ rho={rho} (test catch/fa/U, cal U, optimism, set size, topology) ---")
            print(PF[PF.rho == rho][["family", "rung", "catch", "fa", "U_test", "U_cal",
                                     "opt_gap", "setsize", "g_mode", "nonOR_share"]].to_string(index=False), "\n")
    return D


def insample():
    """Sanity: does the joint controller close the rho=4 gap when it can see everything?
    (No calibration/test split; folds=trivial one-fold so learned rules are in-sample.)"""
    per = defaultdict(lambda: defaultdict(list))
    for fam, slug in ra.FAM.items():
        P = _prep(ra.load(slug), 0.5)
        models, Ce, Cf, cost_d = P["models"], P["Ce"], P["Cf"], P["cost"]
        cost = np.array([cost_d[m] for m in models])
        X, y = _mats(Ce, Cf, models, np.arange(P["ne"]), np.arange(P["nc"]))
        folds = strat_folds(y, 5)
        Ce_d = {m: Ce[m] for m in models}; Cf_d = {m: Cf[m] for m in models}
        for rho in RHOS:
            cols_or = or_selected_cols(Ce_d, Cf_d, cost_d, models, rho) or [0]
            cols_j, gj = joint_greedy(X, y, cost, rho, folds)
            per[rho]["fixed_OR"].append(U_test(cols_or, "OR", X, y, X, y, rho))
            per[rho]["joint"].append(U_test(cols_j, gj, X, y, X, y, rho))
            per[rho][f"joint_g"].append(gj)
    rows = []
    for rho in RHOS:
        rows.append(dict(rho=rho,
                         fixed_OR=round(float(np.mean(per[rho]["fixed_OR"])), 3),
                         joint=round(float(np.mean(per[rho]["joint"])), 3),
                         joint_g_mode=max(set(per[rho]["joint_g"]), key=per[rho]["joint_g"].count)))
    D = pd.DataFrame(rows).set_index("rho")
    D.to_csv(f"{T}/joint_topology_insample.csv")
    print("=== in-sample sanity (joint should dominate fixed-OR, esp. rho=4) ===\n", D, "\n")
    return D


def calc_sensitivity(calcs=(25, 35, 45, 55), nspl=200):
    """TEST (do not assume) whether the residual rho=4 held-out gap is bound by the finite
    clean-calibration set used to estimate a conjunctive rule's false-alarm rate. Vary ONLY
    calc (clean cal items; nc=75, cale=50 fixed). AND-share is the clean diagnostic -- it is a
    calibration-side quantity, unconfounded by shrinking test size. If AND-share rises and
    regret falls with calc, the gap is estimation-bound; if flat, report residual regret as a
    plain limitation."""
    rows = []
    for c in calcs:
        D = heldout(nspl=nspl, calc=c)
        for rho in (2.0, 4.0):
            r = D.loc[rho]
            rows.append(dict(calc=c, test_clean=75 - c, rho=rho,
                             joint_U=r["joint"], fixed_OR_U=r["fixed_OR"],
                             regret_joint=r["regret_joint"], regret_fixed_OR=r["regret_fixed_OR"],
                             AND_share=r["joint_AND_share"]))
    S = pd.DataFrame(rows)
    S.to_csv(f"{T}/joint_topology_calc_sensitivity.csv", index=False)
    print("\n" + "=" * 90)
    print("CALIBRATION-SIZE SENSITIVITY (vary clean-cal count; cale=50 fixed): is rho=4 estimation-bound?")
    print("=" * 90)
    for rho in (4.0, 2.0):
        print(f"\n rho={rho}:")
        print(S[S.rho == rho][["calc", "test_clean", "AND_share", "joint_U",
                               "fixed_OR_U", "regret_joint"]].to_string(index=False))
    return S


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "main"
    if mode == "sens":
        calc_sensitivity(nspl=int(sys.argv[2]) if len(sys.argv) > 2 else 200)
    else:
        nspl = int(sys.argv[1]) if len(sys.argv) > 1 else 200
        insample()
        heldout(nspl=nspl, per_family_csv=True)
    print("wrote results/tables/joint_topology_*.csv")
