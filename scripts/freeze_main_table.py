"""Freeze the main empirical table: 3 failure modes x {same_model, cross_family} with
rho_v (+CI), n_eff@50, blind-spot (+CI), miss rate, and the context-robustness columns.

Primary condition = middle-truncated at the balanced budget: data/raw_cascade_runs/mast__<mode>__ctx_truncate.
Context robustness (same_model, FM-3.1/3.2) reads the evidence-preserving run
(mast__<mode>__ctx_evidence): n_eff and blind-spot under full context, and beta_context.
Both dirs resolve through cascade.paths.raw_run_dir so context conditions never collide.

Writes results/tables/main_results_3mode.csv and prints a Markdown table for the paper.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theory.betabinom_mixture import fit_mixture
from cascade.paths import raw_run_dir


def fit_pool(mode, pool, context_mode="truncate", ci=True):
    m, K = [], []
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
                m.append(acc); K.append(dec)
    if len(m) < 4:
        return None
    f = fit_mixture(np.array(m), np.array(K), compute_ci=ci)
    return f


def main():
    rows = []
    for mode in ["3.1", "3.2", "3.3"]:
        for pool in ["same_model", "cross_family_3"]:
            f = fit_pool(mode, pool)  # primary condition = truncate
            if f is None:
                continue
            row = {
                "mode": f"FM-{mode}", "pool": pool, "n": f.n_items,
                "miss_rate": round(f.mu_hat, 3),
                "rho_v": round(f.rho_v, 3),
                "rho_v_CI": f"[{f.rho_v_ci[0]:.2f}, {f.rho_v_ci[1]:.2f}]",
                "n_eff@50": round(f.n_eff_at_k_max, 2),
                "blind_spot": round(f.blind_spot_mass, 3),
                "blind_CI_hi": round(f.blind_spot_ci[1], 2),
            }
            # context robustness (same_model, 3.1/3.2)
            if pool == "same_model" and mode in ("3.1", "3.2"):
                fe = fit_pool(mode, pool, "evidence", ci=False)
                if fe is not None:
                    row["n_eff@50_evid"] = round(fe.n_eff_at_k_max, 2)
                    row["beta_context"] = round(f.blind_spot_mass - fe.blind_spot_mass, 3)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("results/tables/main_results_3mode.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(df.to_string(index=False))

    # markdown for the paper
    print("\n### Markdown\n")
    cols = ["mode", "pool", "n", "miss_rate", "rho_v", "rho_v_CI", "n_eff@50",
            "blind_spot", "blind_CI_hi"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in df.iterrows():
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    print("\ncontext robustness (same_model): "
          + "; ".join(f"{r['mode']} n_eff {r['n_eff@50']}→{r.get('n_eff@50_evid','—')}, "
                      f"β_context={r.get('beta_context','—')}"
                      for _, r in df.iterrows() if r["pool"] == "same_model" and "beta_context" in r))
    print("\nwrote results/tables/main_results_3mode.csv")


if __name__ == "__main__":
    main()
