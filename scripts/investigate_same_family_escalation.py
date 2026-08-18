"""#8 investigation (OFFLINE): more same-family escalation pairs, to move the
"escalation value = complementarity, not scale" claim past N=1 (qwen7b->qwen14b on MAST).

For each same-family pair (small -> larger same-family model) present in the frozen verdicts,
on the error items where the small model is the deployed solo, compare:
  - escalate to the LARGER SAME-FAMILY model   (scale, correlated)
  - escalate to the best CROSS-FAMILY model     (complementarity)
by resulting OR-cascade catch. The thesis holds if cross-family >= same-family lift.

24B verdicts are pinned to pairing_condition=="same_model_mistral24b" (the k=3 escalate
verdicts used in the paper), NOT the in-progress k=50 redundancy run.
"""
from __future__ import annotations
import glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import reviewer_analyses as ra

FAMILY_OF = {"qwen7b": "qwen", "qwen14b": "qwen", "llama31": "llama",
             "llama32_3b": "llama", "mistral7b": "mistral", "mistral_small_24b": "mistral"}
PIN_PAIRING = {"mistral_small_24b": "same_model_mistral24b"}   # ignore k=50 redun records
# (small -> larger same-family); label
PAIRS = [("llama32_3b", "llama31", "llama 3B->8B"),
         ("mistral7b", "mistral_small_24b", "mistral 7B->24B"),
         ("qwen7b", "qwen14b", "qwen 7B->14B")]
FAMS = dict(ra.FAM); FAMS["MAST-3.3"] = "mast__3_3__ctx_truncate"


def extract(slug):
    catch, fa = defaultdict(dict), defaultdict(dict)
    for p in glob.glob(f"data/raw_cascade_runs/{slug}/*.jsonl"):
        for l in open(p):
            if not l.strip():
                continue
            r = json.loads(l)
            mk = defaultdict(lambda: [0, 0])
            for g in r["gates"]:
                v, a = g.get("verifier_id"), g.get("accepted")
                if a not in (True, False):
                    continue
                if v in PIN_PAIRING and r.get("pairing_condition") != PIN_PAIRING[v]:
                    continue    # 24B: only the k=3 escalate verdicts
                mk[v][1] += 1
                if a is False:
                    mk[v][0] += 1
            tgt = catch if r["ground_truth_is_error"] else fa
            for v, (rej, n) in mk.items():
                if n:
                    tgt[v][r["item_id"]] = int(rej / n > 0.5)
    return catch, fa


def main():
    rows = []
    for fam, slug in FAMS.items():
        catch, _ = extract(slug)
        for small, large, lab in PAIRS:
            if small not in catch or large not in catch:
                continue
            items = sorted(set(catch[small]) & set(catch[large]))
            if len(items) < 10:
                continue
            cs = np.array([catch[small][i] for i in items], bool)
            cl = np.array([catch[large][i] for i in items], bool)
            samefam = float((cs | cl).mean())
            # best cross-family escalation target present on the same items
            cross_best, cross_who = None, None
            for v in catch:
                if FAMILY_OF.get(v) == FAMILY_OF[small] or v in (small, large):
                    continue
                cv = np.array([catch[v].get(i, 0) for i in items], bool)
                res = float((cs | cv).mean())
                if cross_best is None or res > cross_best:
                    cross_best, cross_who = res, v
            rows.append(dict(family=fam, pair=lab, n=len(items),
                             small_catch=round(float(cs.mean()), 3),
                             same_family=round(samefam, 3),
                             cross_family=round(cross_best, 3) if cross_best is not None else np.nan,
                             cross_who=cross_who,
                             scale_lift=round(samefam - float(cs.mean()), 3),
                             cross_lift=round((cross_best or 0) - float(cs.mean()), 3)))
    D = pd.DataFrame(rows)
    out = "results/tables/rev_same_family_escalation.csv"
    D.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print(D.to_string(index=False))
    # focus: cases where the small solo is genuinely weak (<0.6), where escalation matters
    weak = D[D.small_catch < 0.60]
    print("\n=== where the deployed solo is weak (catch<0.6) — escalation actually matters ===")
    if len(weak):
        print(weak.to_string(index=False))
        print(f"\nmean scale-lift (same-family) {weak.scale_lift.mean():+.3f}  "
              f"vs mean cross-family-lift {weak.cross_lift.mean():+.3f}  "
              f"(N={len(weak)} weak-solo cases across {weak.pair.nunique()} same-family pairs)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
