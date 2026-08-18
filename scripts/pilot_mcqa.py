"""Offline analysis of the MC-QA benchmark runs (no GPU). The decisive replication check.

Per benchmark: candidate universe, same_model n_eff (redundancy), greedy sequence with
Delta_catch(+Wilson CI)/Delta_FA/V, the (fa/miss cost, lambda) phase regions, and the
cross-benchmark INVARIANT — a zero-unique-catch candidate is never diversified in,
regardless of its correlation. Writes only mcqa-specific files; MAST tables untouched.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cascade.paths import raw_run_dir
from theory.betabinom_mixture import fit_mixture
from theory.marginal_value import greedy_marginal_value, phase_diagram, _prep

BENCHES = ["arc", "mmlu", "csqa", "truthfulqa"]
CODE = {"DIVERSIFY": "D", "SCALE": "S", "STOP": "T", "NA": "-"}
TRACE = ["step", "cand_type", "candidate", "delta", "delta_ci_low", "delta_ci_high",
         "delta_fa", "unique_catches", "denom", "rel_cost", "V", "action"]


def load(bench):
    recs = [json.loads(l) for p in glob.glob(f"{raw_run_dir(bench, 'truncate', 'mcqa')}/*.jsonl")
            for l in open(p) if l.strip()]
    df = pd.DataFrame(recs)
    return df[df["n_gates"] == 50] if not df.empty else df


def same_model_neff(df):
    sm = df[(df.pairing_condition == "same_model") & (df.ground_truth_is_error)]
    m, K = [], []
    for _, r in sm.iterrows():
        acc = sum(1 for g in r["gates"] if g.get("accepted") is True)
        dec = sum(1 for g in r["gates"] if g.get("accepted") in (True, False))
        if dec:
            m.append(acc); K.append(dec)
    if len(m) < 4:
        return None
    f = fit_mixture(np.array(m), np.array(K), compute_ci=True)
    return f.rho_v, f.n_eff_at_k_max, f.mu_hat


def main():
    pd.set_option("display.width", 200)
    for b in BENCHES:
        df = load(b)
        print("=" * 88)
        if df.empty:
            print(f"[{b}] no data yet")
            continue
        P = _prep(df, 0.5)
        print(f"[{b}] universe={P['models']}  n_err={P['ne']} n_clean={P['nc']}")

        nf = same_model_neff(df)
        if nf:
            print(f"  same_model redundancy: rho_v={nf[0]:.3f}  n_eff@50={nf[1]:.2f}  miss={nf[2]:.3f}"
                  "   (expect n_eff ~1-2 => repeated same-model is redundant)")

        mv = greedy_marginal_value(df, lam=0.1)
        print("  greedy (selected):")
        print(mv[mv.selected][TRACE].to_string(index=False))

        # invariant = a candidate is ADDED (DIVERSIFY/SCALE selected) only when its Δcatch is
        # significant (Wilson CI-low > 0). Check per-row, not "ever had Δ=0 anywhere".
        added = mv[mv.selected & mv.action.isin(["DIVERSIFY", "SCALE"])]
        bad = added[added.delta_ci_low <= 0]
        div = sorted(set(added.candidate))
        never_zero = sorted(set(mv[(mv.cand_type == "NEW") & (mv.delta == 0)].candidate) - set(div))
        print(f"  INVARIANT: added={div}  zero-Δ candidates correctly-never-added={never_zero}  "
              f"added-with-nonsignificant-Δ={list(bad.candidate)}  "
              f"({'OK' if bad.empty else 'VIOLATED'})")

        ph = phase_diagram(df)
        piv = ph.pivot(index="cost_ratio_fa_over_miss", columns="lambda_cost",
                       values="first_action").replace(CODE)
        print("  phase (rows=fa/miss, cols=λ; D/S/T):")
        print(piv.to_string())

        mv.to_csv(f"results/tables/mcqa_marginal_{b}.csv", index=False)
        ph.to_csv(f"results/tables/mcqa_phase_{b}.csv", index=False)


if __name__ == "__main__":
    main()
