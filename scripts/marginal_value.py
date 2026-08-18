"""Conditional marginal verification value (utility), the complement to n_eff (dependence).

For a base set S (here a single base verifier, read from a pool that also contains the
candidate, so items + run conditions match) and candidate i:

    catch_v(x)  = v majority-rejects error x   (miss_prop_v(x) < 0.5)
    Delta_i(S)  = #{x : S misses x AND i catches x} / #{x : S misses x}
    V_i(S)      = Delta_i(S) - lambda * C_i        (C_i = relative median latency)

Read-only over cached jsonl. The point: rank candidates by dependence (corr / n_eff) vs by
utility (Delta) and show the orders DISAGREE — low correlation need not mean useful.
"""
from __future__ import annotations
import glob, json
from collections import defaultdict
import numpy as np

RAW = "data/raw_cascade_runs/mast__3_3__ctx_truncate"


def load_pool(pool):
    recs = []
    for p in glob.glob(f"{RAW}/*{pool}*.jsonl"):
        for line in open(p):
            line = line.strip()
            if line and (r := json.loads(line)).get("pairing_condition") == pool \
                    and r.get("n_gates") == 50:
                recs.append(r)
    return recs


def missprop(recs, model, is_error=True):
    """{item: miss_prop} for one model on error(or clean) items."""
    mk = defaultdict(lambda: [0, 0])
    for r in recs:
        if bool(r["ground_truth_is_error"]) != is_error:
            continue
        for g in r["gates"]:
            if g.get("verifier_id") != model or g.get("accepted") not in (True, False):
                continue
            mk[r["item_id"]][1] += 1
            if g["accepted"] is True:
                mk[r["item_id"]][0] += 1
    return {it: v[0] / v[1] for it, v in mk.items() if v[1] > 0}


def latency(recs, model):
    xs = [g.get("latency_seconds") for r in recs for g in r["gates"]
          if g.get("verifier_id") == model and g.get("latency_seconds")]
    return float(np.median(xs)) if xs else float("nan")


def delta(base_mp, cand_mp):
    """Delta_cand(base): among items base MISSES (majority-accept), fraction cand CATCHES."""
    common = set(base_mp) & set(cand_mp)
    base_miss = [it for it in common if base_mp[it] >= 0.5]
    if not base_miss:
        return float("nan"), 0, 0
    uniq = sum(1 for it in base_miss if cand_mp[it] < 0.5)
    return uniq / len(base_miss), uniq, len(base_miss)


def corr(a_mp, b_mp):
    common = sorted(set(a_mp) & set(b_mp))
    if len(common) < 3:
        return float("nan")
    return float(np.corrcoef([a_mp[i] for i in common], [b_mp[i] for i in common])[0, 1])


# --- key comparison: add a candidate to base = llama31, each from a pool holding both ---
BASE = "llama31"
CANDS = [  # (candidate, pool_containing_both, pooled_n_eff_from_theory_report)
    ("llama32_3b", "same_family", 7.39),
    ("mistral7b",  "diverse_matched", 3.35),
    ("qwen7b",     "cross_family_3", 4.62),
]

print(f"{'='*90}\nMARGINAL VALUE of adding a candidate to base S={{{BASE}}}")
print(f"{'candidate':<12}{'corr(base,c)':>13}{'pool_n_eff':>11}{'Delta_c(S)':>12}"
      f"{'uniq/den':>11}{'rel_cost':>10}{'V(lam=.5)':>11}")
rows = []
for cand, pool, neff in CANDS:
    recs = load_pool(pool)
    bmp, cmp_ = missprop(recs, BASE), missprop(recs, cand)
    d, uniq, den = delta(bmp, cmp_)
    c = corr(bmp, cmp_)
    cost = latency(recs, cand) / latency(recs, BASE)
    V = d - 0.5 * cost
    rows.append((cand, c, neff, d, uniq, den, cost, V))
    print(f"{cand:<12}{c:>+13.3f}{neff:>11.2f}{d:>12.3f}{f'{uniq}/{den}':>11}"
          f"{cost:>10.2f}{V:>+11.3f}")

print("\nRANKINGS DISAGREE (the whole point):")
by_dep = sorted(rows, key=lambda r: r[1])          # least correlated first = 'most diverse'
by_util = sorted(rows, key=lambda r: -r[3])        # highest Delta first = 'most useful'
print("  dependence-only would pick (least corr):", [r[0] for r in by_dep])
print("  utility says pick (highest Delta)       :", [r[0] for r in by_util])

# --- repeated-source case: 2nd same-model draw (STOP signal) ---
print(f"\n{'='*90}\nREPEATED SOURCE: value of a 2nd qwen7b draw (same_model, self-split)")
recs = load_pool("same_model")
half = defaultdict(lambda: [[0, 0], [0, 0]])  # item -> [[m,k]_firsthalf, [m,k]_secondhalf]
for r in recs:
    if not bool(r["ground_truth_is_error"]):
        continue
    gs = [g for g in r["gates"] if g.get("accepted") in (True, False)]
    for j, g in enumerate(gs):
        h = 0 if j < len(gs) / 2 else 1
        half[r["item_id"]][h][1] += 1
        if g["accepted"] is True:
            half[r["item_id"]][h][0] += 1
h1 = {it: v[0][0] / v[0][1] for it, v in half.items() if v[0][1] > 0}
h2 = {it: v[1][0] / v[1][1] for it, v in half.items() if v[1][1] > 0}
d, uniq, den = delta(h1, h2)
print(f"  corr(1st half, 2nd half) = {corr(h1, h2):+.3f}   Delta(2nd|1st misses) = {d:.3f}  ({uniq}/{den})")
print("  -> a repeated qwen draw catches ~none of what the first missed: STOP / SWITCH.")
