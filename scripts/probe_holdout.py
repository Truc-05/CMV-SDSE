"""Flagship experiment: can an R-verdict probe predict the full cascade on REAL data?

    python scripts/probe_holdout.py --target 3.3 --pool same_model

Fits the correlated (Beta-Binomial) and independent (Odds-Law) models on only the first R gate
verdicts per error item, extrapolates the reliability curve to the full observed depth, and
scores both against the held-out depths. Writes:
  results/tables/probe_holdout.csv        -- per-R rho_v, predicted vs observed reliability at
                                             K, and held-out MAE for both extrapolations
  results/figures/fig4_probe.png          -- observed curve + the two probe extrapolations
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theory.probe import probe_from_counts, holdout_error
from theory.uncertainty import bootstrap_curve_ci

BLUE, ORANGE, GREY, GREEN = "#2a6f97", "#e08a1e", "#8a8f98", "#2e8b57"


def load_error_items(target: str, pool: str | None, k_gates: int | None):
    slug = f"mast__{target.replace('.', '_')}__ctx_truncate"
    rows = []
    for p in glob.glob(f"data/raw_cascade_runs/{slug}/*.jsonl"):
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    err = [r for r in rows if r["ground_truth_is_error"]
           and (pool is None or r.get("pairing_condition") == pool)
           and (k_gates is None or r.get("n_gates") == k_gates)]
    return err


def decided_accepts(gates):
    """Per-item list of decided-gate accept booleans, in order."""
    return [g["accepted"] for g in gates if g.get("accepted") in (True, False)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="3.3")
    ap.add_argument("--pool", default="same_model")
    ap.add_argument("--k-gates", type=int, default=None,
                    help="restrict to one cascade-depth cohort (e.g. 50); avoids mixing "
                         "runs of different K under the same pool")
    ap.add_argument("--probe-Rs", type=int, nargs="*", default=[2, 3, 5])
    args = ap.parse_args(argv)

    err = load_error_items(args.target, args.pool, args.k_gates)
    if not err:
        print("no error items found for that target/pool/k-gates", file=sys.stderr)
        return 1

    seqs = [decided_accepts(r.get("gates", [])) for r in err]
    seqs = [s for s in seqs if len(s) >= 1]
    K = max(len(s) for s in seqs)
    k_grid = np.arange(1, K + 1)

    # observed reliability curve (first correct-reject gate per item) + cluster-bootstrap band
    first_reject = []
    for r in err:
        ca = r.get("caught_at")
        first_reject.append(int(ca) if ca not in (None,) and ca == ca else K + 1)
    obs, lo, hi = bootstrap_curve_ci(np.array(first_reject), K, n_boot=2000, seed=20260701)

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    table = []
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(k_grid, obs, "o-", color=BLUE, ms=4, zorder=5, label="observed (all gates)")
    ax.fill_between(k_grid, lo, hi, color=BLUE, alpha=0.15, zorder=1,
                    label="95% cluster bootstrap")

    probe_colors = {2: GREEN, 3: "#6a4c93", 5: "#c0392b"}
    for R in args.probe_Rs:
        m_R = np.array([sum(1 for a in s[:R] if a is True) for s in seqs if len(s) >= R])
        if len(m_R) < 4:
            continue
        pr = probe_from_counts(m_R, R, k_grid)
        obs_at_K = float(obs[-1])
        in_band = bool(lo[-1] <= pr.pred_correlated[-1] <= hi[-1])
        table.append({
            "pool": args.pool, "probe_R": R, "n_items": pr.n_items,
            "rho_v_hat": round(pr.rho_v, 4), "mu_hat": round(pr.mu, 4),
            "pred_rel_at_K_correlated": round(float(pr.pred_correlated[-1]), 4),
            "pred_rel_at_K_independent": round(float(pr.pred_independent[-1]), 4),
            "observed_rel_at_K": round(obs_at_K, 4),
            "correlated_pred_in_band": in_band,
            "heldout_mae_correlated": round(holdout_error(pr.pred_correlated, obs, k_grid, R), 4),
            "heldout_mae_independent": round(holdout_error(pr.pred_independent, obs, k_grid, R), 4),
        })
        c = probe_colors.get(R, GREY)
        ax.plot(k_grid, pr.pred_correlated, "--", color=c, lw=1.8,
                label=f"R={R} probe, correlated ($\\hat\\rho_v$={pr.rho_v:.2f})")
        ax.plot(k_grid, pr.pred_independent, ":", color=c, lw=1.4, alpha=0.8,
                label=f"R={R} probe, independent (Odds Law)")

    ax.set_xlabel("gates k"); ax.set_ylabel("reliability  P(error caught by gate $\\leq k$)")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"A {args.probe_Rs[0]}-verdict probe predicts the full cascade "
                 f"({args.pool}, K={K})")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    out_fig = "results/figures/fig4_probe.png"
    fig.savefig(out_fig, dpi=150); plt.close(fig)

    df = pd.DataFrame(table)
    out_csv = "results/tables/probe_holdout.csv"
    df.to_csv(out_csv, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.to_string(index=False))
    print(f"\nwrote {out_csv} and {out_fig}")
    print("\nRead: 'correlated' should track observed and land in-band at k=K; 'independent' "
          "should over-predict reliability (fail high). The gap is the theory's practical claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
