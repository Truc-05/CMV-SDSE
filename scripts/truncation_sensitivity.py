"""Truncation sensitivity: original (truncated) vs de-truncated reruns, on the SAME items.

    python scripts/truncation_sensitivity.py --orig truncate --new evidence --modes 3.1 3.2

Locks the confound before any 3-mode table is frozen. For each mode x pool it recomputes
rho_v, n_eff@50, blind-spot mass and reliability@50 from BOTH runs, restricted to the error
items present in both (so the comparison is item-matched, not sample-matched). The claim we
must preserve is directional, not the exact numbers:
    n_eff(same) << 50,  n_eff(diverse) > n_eff(same),  and ideally  pi_blind(diverse) < pi_blind(same).
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


def load(mode, pool, context_mode):
    """Return {item_id: {'is_error':bool, 'm':accepts, 'K':decided}} for one mode/pool run."""
    out = {}
    for p in glob.glob(f"{raw_run_dir(mode, context_mode)}/*{pool}*.jsonl"):
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("pairing_condition") != pool or r.get("n_gates") != 50:
                continue
            acc = sum(1 for g in r["gates"] if g.get("accepted") is True)
            dec = sum(1 for g in r["gates"] if g.get("accepted") in (True, False))
            if dec > 0:
                out[r["item_id"]] = {"is_error": bool(r["ground_truth_is_error"]),
                                     "m": acc, "K": dec}
    return out


def summarise(context_mode, mode, pool):
    items = load(mode, pool, context_mode)
    err = {k: v for k, v in items.items() if v["is_error"]}
    if len(err) < 4:
        return None
    m = np.array([v["m"] for v in err.values()]); K = np.array([v["K"] for v in err.values()])
    f = fit_mixture(m, K, compute_ci=False)
    return {"n": len(err), "ids": set(err), "items": err,
            "mu_miss": float(f.mu_hat), "rho_v": float(f.rho_v),
            "n_eff50": float(f.n_eff_at_k_max), "blind": float(f.blind_spot_mass)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="truncate", help="baseline context mode")
    ap.add_argument("--new", default="evidence", help="comparison context mode")
    ap.add_argument("--modes", nargs="*", default=["3.1", "3.2"])
    ap.add_argument("--pools", nargs="*", default=["same_model", "cross_family_3"])
    args = ap.parse_args()

    rows = []
    for mode in args.modes:
        for pool in args.pools:
            o = summarise(args.orig, mode, pool)
            n = summarise(args.new, mode, pool)
            if o is None or n is None:
                print(f"skip {mode}/{pool}: missing data (orig={o is not None}, new={n is not None})")
                continue
            shared = o["ids"] & n["ids"]
            # item-matched refit on shared items
            def refit(src):
                mm = np.array([src["items"][i]["m"] for i in shared])
                KK = np.array([src["items"][i]["K"] for i in shared])
                f = fit_mixture(mm, KK, compute_ci=False)
                return f.mu_hat, f.rho_v, f.n_eff_at_k_max, f.blind_spot_mass
            om = refit(o); nm = refit(n)
            rows.append({
                "mode": mode, "pool": pool, "n_shared": len(shared),
                "mu_miss_orig": round(om[0], 3), "mu_miss_new": round(nm[0], 3),
                "rho_v_orig": round(om[1], 3), "rho_v_new": round(nm[1], 3),
                "n_eff50_orig": round(om[2], 2), "n_eff50_new": round(nm[2], 2),
                "blind_orig": round(om[3], 3), "blind_new": round(nm[3], 3),
            })
    if not rows:
        print("no comparable data yet"); return
    df = pd.DataFrame(rows)
    df.to_csv("results/tables/truncation_sensitivity.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.to_string(index=False))
    # directional checks
    print("\nDIRECTION CHECKS (must hold in BOTH orig and new):")
    for mode in args.modes:
        sm = df[(df["mode"] == mode) & (df["pool"] == "same_model")]
        cf = df[(df["mode"] == mode) & (df["pool"] == "cross_family_3")]
        if len(sm) and len(cf):
            for tag, col in [("orig", "n_eff50_orig"), ("new", "n_eff50_new")]:
                s, c = sm[col].iloc[0], cf[col].iloc[0]
                b_s = sm[f"blind_{tag}"].iloc[0]; b_c = cf[f"blind_{tag}"].iloc[0]
                print(f"  {mode} [{tag}]: n_eff same={s:.2f} << 50 ? {s<10} | diverse {c:.2f} > same ? {c>s} "
                      f"| blind diverse {b_c:.2f} < same {b_s:.2f} ? {b_c<b_s}")
    print("\nwrote results/tables/truncation_sensitivity.csv")


if __name__ == "__main__":
    main()
