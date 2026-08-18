"""Decompose observed blindness into context-induced vs intrinsic, + a selection-bias check.

Compares the same mode x pool across context conditions, each resolved to its isolated
raw-run dir (mast__<mode>__ctx_<condition>) via cascade.paths.raw_run_dir:

    python scripts/context_decomposition.py --conditions truncate evidence --modes 3.1 3.2

For each mode x pool x condition it refits (blind-spot mass, n_eff@50, miss rate) on the error
items SHARED across conditions, then reports
    beta_context = beta(truncate) - beta(evidence)     (context-induced blindness)
    n_eff invariance across conditions                 (intrinsic redundancy)
Finally a SELECTION-BIAS check: using the MAST metadata (trace length, failure modes, MAS,
benchmark), it compares the traces too long to fit either context (>131k chars) against the ones
retained — so "matched non-truncated subset" is defended, not just asserted.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theory.betabinom_mixture import fit_mixture
from cascade.paths import raw_run_dir

CTX_CAP_CHARS = 131072  # 32768 tokens * 4 chars/token — the hard qwen2.5 context limit


def load_counts(mode, pool, context_mode="truncate"):
    out = {}
    for p in glob.glob(f"{raw_run_dir(mode, context_mode)}/*{pool}*.jsonl"):
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("pairing_condition") != pool or r.get("n_gates") != 50 \
                    or not r["ground_truth_is_error"]:
                continue
            acc = sum(1 for g in r["gates"] if g.get("accepted") is True)
            dec = sum(1 for g in r["gates"] if g.get("accepted") in (True, False))
            if dec > 0:
                out[r["item_id"]] = (acc, dec)
    return out


def fit_on(ids, counts):
    m = np.array([counts[i][0] for i in ids]); K = np.array([counts[i][1] for i in ids])
    f = fit_mixture(m, K, compute_ci=False)
    return {"n": len(ids), "mu_miss": round(float(f.mu_hat), 3),
            "rho_v": round(float(f.rho_v), 3), "n_eff50": round(float(f.n_eff_at_k_max), 2),
            "blind": round(float(f.blind_spot_mass), 3)}


def selection_bias_check(all_ids):
    """Compare traces that exceed the context cap (excluded) vs retained, on MAST metadata."""
    try:
        from cascade.mast_loader import resolve_path, FULL_FILENAME, _load_json, MODE_IDS
        path = resolve_path(FULL_FILENAME, None)
        recs = _load_json(path)
    except Exception as e:  # noqa
        print(f"[selection-bias] MAST metadata unavailable ({e}); skipping.")
        return
    by_id = {}
    for r in recs:
        iid = (f"{r.get('mas_name')}|{r.get('benchmark_name')}|"
               f"{r.get('llm_name', 'na')}|{r.get('trace_id')}")
        traj = r.get("trace", {})
        traj = traj.get("trajectory", "") if isinstance(traj, dict) else str(traj or "")
        ann = r.get("mast_annotation", {})
        by_id[iid] = {"chars": len(traj), "mas": r.get("mas_name"),
                      "bench": r.get("benchmark_name"),
                      "n_modes": sum(int(ann.get(m, 0)) for m in MODE_IDS)}
    rows = [by_id[i] for i in all_ids if i in by_id]
    if not rows:
        print("[selection-bias] no metadata matches.")
        return
    df = pd.DataFrame(rows)
    df["excluded"] = df["chars"] > CTX_CAP_CHARS
    print("\n=== SELECTION-BIAS CHECK (excluded >131k chars vs retained) ===")
    g = df.groupby("excluded").agg(n=("chars", "size"), median_chars=("chars", "median"),
                                   mean_n_modes=("n_modes", "mean")).reset_index()
    print(g.to_string(index=False))
    print("top MAS among excluded:", dict(df[df.excluded]["mas"].value_counts().head(3)))
    print("=> if excluded traces are not concentrated in a particular failure subtype / MAS and "
          "differ only in length, the matched-subset analysis is defensible.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=["truncate", "evidence"],
                    help="context modes to compare; each resolves via cascade.paths.raw_run_dir")
    ap.add_argument("--modes", nargs="*", default=["3.1", "3.2"])
    ap.add_argument("--pools", nargs="*", default=["same_model"])
    args = ap.parse_args()
    conds = list(args.conditions)

    rows, all_ids = [], set()
    for mode in args.modes:
        for pool in args.pools:
            cc = {ctx: load_counts(mode, pool, ctx) for ctx in conds}
            shared = set.intersection(*[set(c) for c in cc.values()]) if all(cc.values()) else set()
            if len(shared) < 4:
                print(f"skip {mode}/{pool}: shared items={len(shared)}"); continue
            all_ids |= shared
            rec = {"mode": mode, "pool": pool, "n_shared": len(shared)}
            for lab in conds:
                f = fit_on(shared, cc[lab])
                for k, v in f.items():
                    if k != "n":
                        rec[f"{k}_{lab}"] = v
            rows.append(rec)

    if not rows:
        print("no comparable data yet"); return
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/context_decomposition.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(df.to_string(index=False))
    labels = list(conds)
    if "truncate" in labels and "evidence" in labels:
        print("\n=== DECOMPOSITION ===")
        for _, r in df.iterrows():
            bc = r["blind_truncate"] - r["blind_evidence"]
            print(f"  {r['mode']}/{r['pool']}: beta_context = {r['blind_truncate']:.2f} - "
                  f"{r['blind_evidence']:.2f} = {bc:+.2f} | "
                  f"n_eff {r['n_eff50_truncate']:.2f} -> {r['n_eff50_evidence']:.2f} "
                  f"(invariant? {abs(r['n_eff50_truncate']-r['n_eff50_evidence'])<1.5})")
    selection_bias_check(all_ids)
    print("\nwrote results/tables/context_decomposition.csv")


if __name__ == "__main__":
    main()
