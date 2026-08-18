"""Adversarial-review substantiation for the joint controller. Adds two controls the
main analysis lacks, to test reviewer attacks B and C. No paper edits; prints only.

B (topology-only is trivial because the OR set is a singleton): add a NON-degenerate
   control -- best fusion topology applied to the FULL pool, chosen on calibration
   (`topo_fullpool`). If it matches/beats `joint`, then joint's SOURCE selection adds
   nothing beyond 'switch topology on everything', and the coupling claim weakens.
C (no observed degradation): per (family,split) paired difference joint - fixed_OR;
   report the fraction of cases where joint is strictly worse and a paired 95% CI.
"""
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from theory.marginal_value import _prep
import reviewer_analyses as ra
from ablation_fusion_baselines import strat_folds
from joint_topology_controller import (_mats, U_test, U_cal, joint_greedy,
                                        or_selected_cols, RHOS, G, LAM)


def run(nspl=200):
    per = defaultdict(lambda: defaultdict(list))     # rho -> policy -> [per-family means]
    diffs = defaultdict(list)                        # rho -> [joint-fixedOR over all fam,split]
    for fam, slug in ra.FAM.items():
        P = _prep(ra.load(slug), 0.5)
        models, Ce, Cf, cost_d = P["models"], P["Ce"], P["Cf"], P["cost"]
        ne, nc = P["ne"], P["nc"]
        cost = np.array([cost_d[m] for m in models]); full = list(range(len(models)))
        rng = np.random.RandomState(20260815)
        acc = defaultdict(lambda: defaultdict(list))
        for _ in range(nspl):
            ep, cp = rng.permutation(ne), rng.permutation(nc)
            ce, te = ep[:50], ep[50:]; cc, tc = cp[:25], cp[25:]
            Xtr, ytr = _mats(Ce, Cf, models, ce, cc)
            Xte, yte = _mats(Ce, Cf, models, te, tc)
            folds = strat_folds(ytr, 5)
            Ce_c = {m: Ce[m][ce] for m in models}; Cf_c = {m: Cf[m][cc] for m in models}
            M = len(models)
            for rho in RHOS:
                cols_or = or_selected_cols(Ce_c, Cf_c, cost_d, models, rho) or [0]
                u_or = U_test(cols_or, "OR", Xtr, ytr, Xte, yte, rho)
                cols_j, gj = joint_greedy(Xtr, ytr, cost, rho, folds)
                u_j = U_test(cols_j, gj, Xtr, ytr, Xte, yte, rho)
                # NON-degenerate control: best topology on the FULL pool, picked on calibration
                g_fp = max(G, key=lambda g: U_cal(full, g, Xtr, ytr, rho, folds))
                u_fp = U_test(full, g_fp, Xtr, ytr, Xte, yte, rho)
                # compute-charged utility U_lambda = U - lam*|S| (the controller's ACTUAL objective)
                acc[rho]["fixed_OR"].append(u_or); acc[rho]["joint"].append(u_j)
                acc[rho]["topo_fullpool"].append(u_fp)
                acc[rho]["joint_Ulam"].append(u_j - LAM * len(set(cols_j)))
                acc[rho]["fp_Ulam"].append(u_fp - LAM * M)
                acc[rho]["fixedOR_Ulam"].append(u_or - LAM * len(set(cols_or)))
                acc[rho]["joint_sz"].append(len(set(cols_j)))
                diffs[rho].append(u_j - u_or)
        for rho in RHOS:
            for pol, xs in acc[rho].items():
                per[rho][pol].append(float(np.mean(xs)))
    print("U = catch - rho*FA (paper metric).  U_lam = U - 0.1*|S| (controller's objective).\n")
    print(f"{'rho':>5} | {'fixOR':>7} {'joint':>7} {'topoFP':>7} {'jt>FP':>6} | "
          f"{'fixOR_l':>8} {'joint_l':>8} {'topoFP_l':>8} {'jt>FP_l':>7} | "
          f"{'jt|S|':>5} {'deg%':>5} {'diff[95%CI]':>18}")
    for rho in RHOS:
        g = lambda k: np.mean(per[rho][k])
        d = np.array(diffs[rho]); deg = 100 * float(np.mean(d < -1e-9))
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{rho:>5} | {g('fixed_OR'):>7.3f} {g('joint'):>7.3f} {g('topo_fullpool'):>7.3f} "
              f"{str(g('joint') > g('topo_fullpool')):>6} | "
              f"{g('fixedOR_Ulam'):>8.3f} {g('joint_Ulam'):>8.3f} {g('fp_Ulam'):>8.3f} "
              f"{str(g('joint_Ulam') > g('fp_Ulam')):>7} | "
              f"{g('joint_sz'):>5.2f} {deg:>4.1f}% [{lo:.3f},{hi:.3f}]")
    rows = []
    for rho in RHOS:
        g = lambda k: round(float(np.mean(per[rho][k])), 3)
        d = np.array(diffs[rho])
        rows.append(dict(rho=rho, U_fixedOR=g("fixed_OR"), U_topoFP=g("topo_fullpool"),
                         U_joint=g("joint"), J_fixedOR=g("fixedOR_Ulam"), J_topoFP=g("fp_Ulam"),
                         J_joint=g("joint_Ulam"), joint_setsize=g("joint_sz"),
                         deg_rate=round(float(np.mean(d < -1e-9)), 3),
                         paired_lo=round(float(np.percentile(d, 2.5)), 3),
                         paired_hi=round(float(np.percentile(d, 97.5)), 3)))
    import pandas as pd
    pd.DataFrame(rows).to_csv("results/tables/joint_topology_controls.csv", index=False)
    print("\ntopo_fullpool = best topology on ALL verifiers, calibration-picked (non-oracle, uses full compute).")
    print("deg% = fraction of (family,split) where joint < fixed_OR (strict).")
    print("wrote results/tables/joint_topology_controls.csv")


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
