"""Offline dependence audit for a verifier pool (no model calls).

Decomposes a pairing_condition's raw gates by verifier_id and asks whether a low
pooled rho_v / high n_eff is REAL complementarity or an artifact of
pairing/seed/order/item-subset. Read-only over the cached jsonl.

Usage: python scripts/audit_pool_dependence.py [pool ...]   (default: same_family)
"""
from __future__ import annotations
import glob, json, sys
from collections import defaultdict
import numpy as np

RAW = "data/raw_cascade_runs/mast__3_3__ctx_truncate"


def load(pool):
    recs = []
    for p in glob.glob(f"{RAW}/*{pool}*.jsonl"):
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("pairing_condition") == pool and r.get("n_gates") == 50:
                recs.append(r)
    return recs


def gate_frame(recs, is_error):
    """rows: (item_id, model, accepted_bool) for decided gates of the requested class."""
    rows = []
    for r in recs:
        if bool(r["ground_truth_is_error"]) != is_error:
            continue
        for g in r["gates"]:
            acc = g.get("accepted")
            if acc not in (True, False):
                continue
            rows.append((r["item_id"], g.get("verifier_id"), bool(acc)))
    return rows


def per_item_missprop(recs):
    """{model: {item: miss_prop}} and {model: {item:(m,k)}} on ERROR items.
    miss = accepted True on an error (false-accept)."""
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # model->item->[m,k]
    for r in recs:
        if not bool(r["ground_truth_is_error"]):
            continue
        for g in r["gates"]:
            acc = g.get("accepted")
            if acc not in (True, False):
                continue
            mk = agg[g.get("verifier_id")][r["item_id"]]
            mk[1] += 1
            if acc is True:
                mk[0] += 1
    return agg


def summarise(pool):
    recs = load(pool)
    models = sorted({g.get("verifier_id") for r in recs for g in r["gates"]})
    print(f"\n{'='*78}\nPOOL: {pool}   models={models}   records={len(recs)}")

    # --- gate balance + per-model miss / false-alarm ---
    err = gate_frame(recs, True)
    cln = gate_frame(recs, False)
    print(f"{'model':<14}{'err_gates':>10}{'miss_rate':>11}{'clean_gates':>12}{'fa_rate':>9}")
    for mdl in models:
        eg = [a for (_, m, a) in err if m == mdl]
        cg = [a for (_, m, a) in cln if m == mdl]
        miss = np.mean(eg) if eg else float("nan")
        fa = np.mean([not a for a in cg]) if cg else float("nan")
        print(f"{mdl:<14}{len(eg):>10}{miss:>11.3f}{len(cg):>12}{fa:>9.3f}")

    # --- per-item miss propensity, heterogeneity, blind spots ---
    agg = per_item_missprop(recs)
    items_all = sorted({it for mdl in agg for it in agg[mdl]})
    print(f"\nerror items: {len(items_all)}")
    prop = {}  # model -> {item: miss_prop}
    for mdl in models:
        prop[mdl] = {it: mk[0] / mk[1] for it, mk in agg[mdl].items() if mk[1] > 0}
        vals = np.array(list(prop[mdl].values()))
        blind = int(sum(1 for it, mk in agg[mdl].items() if mk[0] == mk[1]))  # never caught
        print(f"  {mdl:<12} mean_miss={vals.mean():.3f}  SD_across_items={vals.std():.3f}"
              f"  solo_blind(items always-missed)={blind}/{len(vals)}")

    # pooled (all gates) per-item heterogeneity + pooled blind spot
    pooled = {}
    pooled_blind = 0
    for it in items_all:
        m = sum(agg[mdl][it][0] for mdl in models if it in agg[mdl])
        k = sum(agg[mdl][it][1] for mdl in models if it in agg[mdl])
        if k > 0:
            pooled[it] = m / k
            if m == k:
                pooled_blind += 1
    pv = np.array(list(pooled.values()))
    print(f"  {'POOLED':<12} mean_miss={pv.mean():.3f}  SD_across_items={pv.std():.3f}"
          f"  pooled_blind(all-models-miss)={pooled_blind}/{len(pv)}")

    # --- pairwise complementarity (2-model pools) ---
    if len(models) == 2:
        a, b = models
        common = sorted(set(prop[a]) & set(prop[b]))
        pa = np.array([prop[a][it] for it in common])
        pb = np.array([prop[b][it] for it in common])
        r = np.corrcoef(pa, pb)[0, 1] if len(common) > 2 else float("nan")
        ca, cb = pa < 0.5, pb < 0.5   # majority-catch
        both_c = int(np.sum(ca & cb)); both_m = int(np.sum(~ca & ~cb))
        a_only = int(np.sum(ca & ~cb)); b_only = int(np.sum(~ca & cb))
        print(f"\npairwise({a},{b}) on {len(common)} shared items:")
        print(f"  Pearson corr of per-item miss_prop = {r:+.3f}   (low/neg => complementary)")
        print(f"  majority-catch: both={both_c}  {a}-only={a_only}  {b}-only={b_only}  both-miss={both_m}")

    # --- confound checks: order / seed / coverage ---
    print("\nconfound checks:")
    r0 = next(r for r in recs if bool(r["ground_truth_is_error"]))
    order = [g.get("verifier_id") for g in r0["gates"]]
    print(f"  gate order (one item, first 12): {order[:12]}")
    # coverage balance per error item
    bal = []
    for r in recs:
        if not bool(r["ground_truth_is_error"]):
            continue
        c = defaultdict(int)
        for g in r["gates"]:
            c[g.get("verifier_id")] += 1
        bal.append(tuple(sorted(c.items())))
    uniq_bal = set(bal)
    print(f"  per-item gate counts by model (distinct patterns): {sorted(uniq_bal)[:3]}"
          f"{' ...' if len(uniq_bal) > 3 else ''}")
    # seed uniqueness
    seeds = [g.get("seed") for r in recs for g in r["gates"] if g.get("seed") is not None]
    print(f"  seeds: {len(seeds)} total, {len(set(seeds))} distinct "
          f"({'all unique' if len(seeds) == len(set(seeds)) else 'REUSED seeds present'})")
    return {it for it in items_all}


def main():
    pools = sys.argv[1:] or ["same_family"]
    itemsets = {p: summarise(p) for p in pools}
    # cross-pool item-subset confound: are pools evaluated on the same error items?
    if len(itemsets) > 1:
        print(f"\n{'='*78}\nITEM-SET OVERLAP (error items) across pools:")
        ks = list(itemsets)
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                A, B = itemsets[ks[i]], itemsets[ks[j]]
                print(f"  {ks[i]} vs {ks[j]}: |A|={len(A)} |B|={len(B)} "
                      f"shared={len(A & B)} jaccard={len(A & B)/len(A | B):.2f}")


if __name__ == "__main__":
    main()
