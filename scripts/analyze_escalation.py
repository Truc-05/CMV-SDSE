"""Compose the ESCALATE-to-external strategies OFFLINE, once the external verifier's verdicts exist.

Reads, per confirmatory family, the local pool verdicts (n_gates==50) AND the external verifier's
verdicts (the low-k same_model_<model> run; default mistral_small_24b), aligns by item_id, scores:

  Local only            : CMV-SDSE on the local pool, no external call.
  Always external       : the external verifier on every item (external calls = 100%).
  Random escalation     : escalate a random fraction (matched to CMV's rate) to the external.
  CMV-SDSE -> external  : local first; call the external ONLY on items the local verdict ACCEPTS
                          (the ones that might hide a missed error), then OR the two verdicts.

The point: does CMV-SDSE reach near Always-external reliability while paying for the external
verifier on only a fraction of items? Mirrors the validated qwen14b escalation logic. No new compute.
Writes results/tables/rev_escalate_<model>.csv.
"""
from __future__ import annotations
import glob, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import reviewer_analyses as ra

T = "results/tables"
# external verifier id to escalate to (override on the CLI):
#   python scripts/analyze_escalation.py mistral_small_24b
DS = sys.argv[1] if len(sys.argv) > 1 else "mistral_small_24b"
CANON_LOCAL = ["qwen7b", "llama31", "llama32_3b", "mistral7b"]   # the frozen local pool
RHO = 0.25          # miss-dominated regime, as in the ESCALATE experiment (sec:escalate_exp)
RNG = np.random.RandomState(0)


def extract(slug):
    """{model:{item:reject}} for error and clean items, over ALL models present in the dir
    (local at n_gates==50 and the external model at low k both keyed by their gate verifier_id)."""
    catch, fa = defaultdict(dict), defaultdict(dict)
    for p in glob.glob(f"data/raw_cascade_runs/{slug}/*.jsonl"):
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            mk = defaultdict(lambda: [0, 0])
            for g in r["gates"]:
                v, a = g.get("verifier_id"), g.get("accepted")
                if a in (True, False):
                    mk[v][1] += 1
                    if a is False:
                        mk[v][0] += 1
            tgt = catch if r["ground_truth_is_error"] else fa
            for v, (rej, n) in mk.items():
                if n:
                    tgt[v][r["item_id"]] = int(rej / n > 0.5)
    return catch, fa


def aligned(slug):
    catch, fa = extract(slug)
    if DS not in catch:
        return None
    local = [m for m in CANON_LOCAL if m in catch]     # ignore any other external in the dir
    use = local + [DS]
    ei = sorted(set.intersection(*[set(catch[m]) for m in use]))
    ci = sorted(set.intersection(*[set(fa[m]) for m in use])) if all(fa.get(m) for m in use) else []
    Ce = {m: np.array([catch[m][i] for i in ei], bool) for m in use}
    Cf = {m: np.array([fa[m].get(i, 0) for i in ci], bool) for m in use}
    return dict(local=local, Ce=Ce, Cf=Cf, ne=len(ei), nc=len(ci))


def score(rho=RHO):
    rows = []
    missing = []
    for fam, slug in ra.FAM.items():
        A = aligned(slug)
        if A is None:
            missing.append(fam); continue
        Ce, Cf, local = A["Ce"], A["Cf"], A["local"]
        cost = {m: 1.0 for m in Ce}
        # local CMV-SDSE selection (uniform cost); its OR-cascade verdict per item
        S = ra.greedy_normalized({m: Ce[m] for m in local}, {m: Cf[m] for m in local}, cost)
        loc_e = np.zeros(A["ne"], bool); loc_c = np.zeros(A["nc"], bool)
        for m in S:
            loc_e |= Ce[m]; loc_c |= Cf[m]

        def U(c, f):
            return c - rho * f

        # (1) Local only
        rows.append(dict(family=fam, strategy="Local only", ext_pct=0.0,
                         catch=loc_e.mean(), fa=loc_c.mean(), U=U(loc_e.mean(), loc_c.mean())))
        # (2) Always external
        rows.append(dict(family=fam, strategy="Always external", ext_pct=1.0,
                         catch=Ce[DS].mean(), fa=Cf[DS].mean(), U=U(Ce[DS].mean(), Cf[DS].mean())))
        # (4) CMV-SDSE -> external: call DS only where local ACCEPTS; then OR
        esc_e = ~loc_e            # error items the local verdict accepts (missed) -> escalate
        esc_c = ~loc_c            # clean items the local verdict accepts -> escalate
        fin_e = loc_e | (esc_e & Ce[DS]); fin_c = loc_c | (esc_c & Cf[DS])
        ext_pct = (esc_e.sum() + esc_c.sum()) / (A["ne"] + A["nc"])
        rows.append(dict(family=fam, strategy="CMV-SDSE -> external", ext_pct=ext_pct,
                         catch=fin_e.mean(), fa=fin_c.mean(), U=U(fin_e.mean(), fin_c.mean())))
        # (3) Random escalation at the same external rate (avg over draws)
        rc, rf, ru = [], [], []
        for _ in range(200):
            me = RNG.rand(A["ne"]) < ext_pct
            mc = RNG.rand(A["nc"]) < ext_pct
            fe = loc_e | (me & Ce[DS]); fc = loc_c | (mc & Cf[DS])
            rc.append(fe.mean()); rf.append(fc.mean()); ru.append(U(fe.mean(), fc.mean()))
        rows.append(dict(family=fam, strategy="Random escalation", ext_pct=ext_pct,
                         catch=np.mean(rc), fa=np.mean(rf), U=np.mean(ru)))
    if missing:
        print(f"MISSING {DS} verdicts for: {missing}\n"
              f"Collect them first: scripts/run_mistral24b_escalation.sh", file=sys.stderr)
        if len(missing) == len(ra.FAM):
            return None
    D = pd.DataFrame(rows)
    M = (D.groupby("strategy")[["ext_pct", "catch", "fa", "U"]].mean().round(3)
         .reindex(["Local only", "Always external", "Random escalation", "CMV-SDSE -> external"]))
    out = f"{T}/rev_escalate_{DS}.csv"
    M.to_csv(out)
    print(f"=== ESCALATE-to-{DS} strategies (macro over {D.family.nunique()} families, "
          f"utility at rho={rho}) ===")
    print(M.to_string(), "\n")
    print(f"wrote {out}")
    return M


if __name__ == "__main__":
    score()
