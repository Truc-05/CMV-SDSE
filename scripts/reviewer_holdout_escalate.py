"""Two reviewer-requested analyses, computed from the EXISTING frozen verdicts (no new GPU):

  H. Held-out CMV calibration -> test.  Split each family's items into DISJOINT
     calibration and test sets (D_cal cap D_test = empty).  Estimate Delta_catch/Delta_fa and run
     the FROZEN CMV-SDSE controller on D_cal only; freeze the selected set; evaluate on the
     unseen D_test.  Repeat over many random splits.  Answers: does CMV generalise, or does it
     look good only because selection saw the evaluation items?

  X. ESCALATE validation.  Construct the exhausted state the controller's 4th action is defined
     for: in-pool = {solo verifier}, the only in-pool lever is SCALE (same-model repeats, which
     saturate at n_eff ~ 1), and the catch target is unmet.  Compare STOP, SCALE, and ESCALATE
     (admit an out-of-pool source).  Includes a genuine 7B -> 14B escalation on MAST-3.3 using the
     held-out qwen14b checkpoint.

Reuses theory.marginal_value._prep/_wilson and scripts.reviewer_analyses.{load,cascade,
greedy_normalized,order_accuracy,order_lowcorr} so every number reconciles with the paper tables
(full-data macro CMV-SDSE utility = 0.486).  Writes results/tables/rev_holdout_cmv.csv,
rev_escalate.csv, rev_escalate_qwen14b.csv.
"""
from __future__ import annotations
import glob, json, sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from theory.marginal_value import _prep, _wilson
import reviewer_analyses as ra

T = "results/tables"
RHO = 1.0            # balanced regime (matches the paper headline)
RHO_MISS = 0.25      # miss-dominated regime where a hard catch target is the priority
N_SPLITS = 200
CAL_ERR, CAL_CLN = 50, 25     # -> 100 error / 50 clean held out for test (150/75 total)
TARGET = 0.90        # catch-reliability target for the ESCALATE experiment


# ---------------- helpers on index-restricted item sets ----------------
def sub(C, idx):
    return {m: C[m][idx] for m in C}


def cascade_U(S, Ce, Cf, rho):
    c, f = ra.cascade(S, Ce, Cf)
    return c, f, c - rho * f


def best_subset(Ce, Cf, models, rho):
    """Oracle: subset with max utility, evaluated on whatever items Ce/Cf hold."""
    best, bestU, bc, bf = None, -1e9, 0.0, 0.0
    for k in range(1, len(models) + 1):
        for combo in combinations(models, k):
            c, f, U = cascade_U(list(combo), Ce, Cf, rho)
            if U > bestU:
                best, bestU, bc, bf = list(combo), U, c, f
    return best, bc, bf, bestU


# ---------------- H. held-out CMV calibration -> test ----------------
def holdout(rho=RHO):
    rows = []
    for fam, slug in ra.FAM.items():
        df = ra.load(slug); P = _prep(df, 0.5)
        Ce, Cf, cost, models = P["Ce"], P["Cf"], P["cost"], P["models"]
        ne, nc = P["ne"], P["nc"]
        rng = np.random.RandomState(20260815)     # fixed seed -> reproducible splits
        acc = defaultdict(list)
        for _ in range(N_SPLITS):
            ep = rng.permutation(ne); cp = rng.permutation(nc)
            cal_e, tst_e = ep[:CAL_ERR], ep[CAL_ERR:]
            cal_c, tst_c = cp[:CAL_CLN], cp[CAL_CLN:]
            Ce_cal, Cf_cal = sub(Ce, cal_e), sub(Cf, cal_c)
            Ce_tst, Cf_tst = sub(Ce, tst_e), sub(Cf, tst_c)
            # FROZEN controller selects on calibration only
            S = ra.greedy_normalized(Ce_cal, Cf_cal, cost, w_f=1.0, lam=0.1, gate=True)
            cU = cascade_U(S, Ce_cal, Cf_cal, rho)[2]         # in-sample (calibration) utility
            tc, tf, tU = cascade_U(S, Ce_tst, Cf_tst, rho)    # out-of-sample (test) utility
            # oracle chosen ON TEST -> regret of the frozen (calibration) choice
            oS, _, _, oU = best_subset(Ce_tst, Cf_tst, models, rho)
            solo_cal = max(models, key=lambda m: Ce_cal[m].mean())
            solo_tst = max(models, key=lambda m: Ce_tst[m].mean())
            acc["test_catch"].append(tc); acc["test_fa"].append(tf); acc["test_U"].append(tU)
            acc["cal_U"].append(cU); acc["set_size"].append(len(S)); acc["regret"].append(oU - tU)
            acc["sel_agree"].append(int(set(S) == set(oS)))
            acc["solo_agree"].append(int(solo_cal == solo_tst))
            # held-out baselines, selected on calibration, scored on test
            acc["U_samemodel"].append(cascade_U([solo_cal], Ce_tst, Cf_tst, rho)[2])
            acq = ra.order_accuracy(models, Ce_cal, Cf_cal)[:3]
            lcq = ra.order_lowcorr(models, Ce_cal, Cf_cal)[:3]
            acc["U_accuracy"].append(cascade_U(acq, Ce_tst, Cf_tst, rho)[2])
            acc["U_lowcorr"].append(cascade_U(lcq, Ce_tst, Cf_tst, rho)[2])
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        s = {k: float(np.std(v)) for k, v in acc.items()}
        rows.append(dict(family=fam, **{k: round(m[k], 3) for k in
                    ["test_catch", "test_fa", "test_U", "cal_U", "set_size", "regret",
                     "sel_agree", "solo_agree", "U_samemodel", "U_accuracy", "U_lowcorr"]},
                    test_U_sd=round(s["test_U"], 3)))
    D = pd.DataFrame(rows)
    macro = {c: round(D[c].mean(), 3) for c in D.columns if c != "family"}
    macro["family"] = "MACRO"
    D = pd.concat([D, pd.DataFrame([macro])], ignore_index=True)
    D.to_csv(f"{T}/rev_holdout_cmv.csv", index=False)
    print(f"=== H. held-out CMV calibration->test (rho={rho}, {N_SPLITS} splits, "
          f"cal={CAL_ERR}err/{CAL_CLN}cln -> test={150-CAL_ERR}err/{75-CAL_CLN}cln) ===")
    print(D.to_string(index=False), "\n")
    return D


# ---------------- X. ESCALATE validation ----------------
def escalate(rho_report=RHO_MISS, target=TARGET, base="qwen7b"):
    """Fixed deployed in-pool = {base}; its only lever is SCALE (same-model, which saturates
    at n_eff~1). When same-model evidence tops out below the target, the pool is exhausted and
    the 4th action is the only exit. ESCALATE admits out-of-(base)-pool verifiers greedily by
    marginal catch. base=qwen7b is the single most-selected solo across the six families."""
    rows = []
    for fam, slug in ra.FAM.items():
        df = ra.load(slug); P = _prep(df, 0.5)
        Ce, Cf, cost, models = P["Ce"], P["Cf"], P["cost"], P["models"]
        solo = base if base in models else max(models, key=lambda m: Ce[m].mean())
        R_stop, F_stop = Ce[solo].mean(), Cf[solo].mean()
        # SCALE: same-model repeats saturate (n_eff~1); catch is flat by construction of Ce
        # (Ce[solo] is already the majority over all of solo's gates). Report it explicitly.
        R_scale, F_scale = R_stop, F_stop
        # ESCALATE: greedily admit out-of-pool verifiers by marginal catch on survivors
        S = [solo]; cov = Ce[solo].copy(); cf = Cf[solo].copy(); added = []
        while cov.mean() < target:
            rest = [m for m in models if m not in S]
            if not rest:
                break
            surv = ~cov
            best = max(rest, key=lambda m: Ce[m][surv].mean() if surv.any() else 0.0)
            if not surv.any() or Ce[best][surv].mean() <= 0:
                break
            S.append(best); cov |= Ce[best]; cf |= Cf[best]; added.append(best)
        R_esc, F_esc = cov.mean(), cf.mean()
        rows.append(dict(
            family=fam, solo=solo, target=target,
            R_stop=round(float(R_stop), 3), R_scale=round(float(R_scale), 3),
            R_escalate=round(float(R_esc), 3),
            stop_meets=bool(R_stop >= target), scale_meets=bool(R_scale >= target),
            escalate_meets=bool(R_esc >= target),
            U_stop=round(float(R_stop - rho_report * F_stop), 3),
            U_escalate=round(float(R_esc - rho_report * F_esc), 3),
            dU=round(float((R_esc - rho_report * F_esc) - (R_stop - rho_report * F_stop)), 3),
            escalated_to="+".join(added) if added else "(none)", n_added=len(added)))
    D = pd.DataFrame(rows)
    macro = dict(family="MACRO", solo="", target=target,
                 R_stop=round(D.R_stop.mean(), 3), R_scale=round(D.R_scale.mean(), 3),
                 R_escalate=round(D.R_escalate.mean(), 3),
                 stop_meets=int(D.stop_meets.sum()), scale_meets=int(D.scale_meets.sum()),
                 escalate_meets=int(D.escalate_meets.sum()),
                 U_stop=round(D.U_stop.mean(), 3), U_escalate=round(D.U_escalate.mean(), 3),
                 dU=round(D.dU.mean(), 3), escalated_to="", n_added=round(D.n_added.mean(), 2))
    D = pd.concat([D, pd.DataFrame([macro])], ignore_index=True)
    D.to_csv(f"{T}/rev_escalate.csv", index=False)
    print(f"=== X. ESCALATE (in-pool={{solo}}, target catch={target}, utility at rho={rho_report}) ===")
    print(D.to_string(index=False), "\n")
    return D


# ---------------- X2. genuine 7B -> 14B escalation on MAST-3.3 ----------------
def escalate_qwen14b(rho_report=RHO_MISS):
    """qwen_flip pool = (qwen7b, qwen14b, mistral7b) on MAST-3.3. In-pool = {qwen7b};
    ESCALATE target = the stronger qwen14b checkpoint (held out of the 7-8B confirmatory pool)."""
    p = "data/raw_cascade_runs/mast__3_3__ctx_truncate/FM-3.3__qwen_flip__qwen_flip.jsonl"
    recs = [json.loads(l) for l in open(p) if l.strip() and json.loads(l).get("n_gates") == 50]
    # per-(model,item) majority reject, split by class
    catch, fa = defaultdict(dict), defaultdict(dict)
    for r in recs:
        mk = defaultdict(lambda: [0, 0])
        for g in r["gates"]:
            v, a = g.get("verifier_id"), g.get("accepted")
            if a in (True, False):
                mk[v][1] += 1
                if a is False:            # reject
                    mk[v][0] += 1
        for v, (rej, n) in mk.items():
            if n:
                (catch if r["ground_truth_is_error"] else fa)[v][r["item_id"]] = int(rej / n > 0.5)
    models = ["qwen7b", "qwen14b", "mistral7b"]
    ei = sorted({it for v in models for it in catch.get(v, {})})
    ci = sorted({it for v in models for it in fa.get(v, {})})
    Ce = {v: np.array([catch.get(v, {}).get(it, 0) for it in ei], bool) for v in models}
    Cf = {v: np.array([fa.get(v, {}).get(it, 0) for it in ci], bool) for v in models}
    out = []
    base = "qwen7b"
    R0, F0 = Ce[base].mean(), Cf[base].mean()
    out.append(("STOP  (qwen7b solo)", R0, F0, R0 - rho_report * F0))
    out.append(("SCALE (qwen7b x50, same-model)", R0, F0, R0 - rho_report * F0))
    for tgt in ["qwen14b", "mistral7b"]:
        cov = Ce[base] | Ce[tgt]; cf = Cf[base] | Cf[tgt]
        out.append((f"ESCALATE (+{tgt})", cov.mean(), cf.mean(), cov.mean() - rho_report * cf.mean()))
    D = pd.DataFrame([dict(action=a, catch=round(float(c), 3), fa=round(float(f), 3),
                           U=round(float(u), 3)) for a, c, f, u in out])
    D.to_csv(f"{T}/rev_escalate_qwen14b.csv", index=False)
    print(f"=== X2. genuine 7B->14B escalation on MAST-3.3 (n_err={len(ei)}, n_cln={len(ci)}, "
          f"utility at rho={rho_report}) ===")
    print(D.to_string(index=False), "\n")
    return D


if __name__ == "__main__":
    holdout()
    escalate()
    escalate_qwen14b()
    print("wrote results/tables/rev_{holdout_cmv,escalate,escalate_qwen14b}.csv")
