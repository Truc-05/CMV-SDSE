"""Offline analysis of the GSM8K confirmatory run (no GPU) — the reasoning-family replication.

Answers ONLY the three pre-registered / frozen questions (no re-tuning, no new claims):
  Q1  same-model redundancy: is n_eff@50 still ~1 when errors are multi-step reasoning failures?
  Q2  useful diversity: does at least one diverse verifier have Δcatch > 0 (Wilson CI-low > 0)?
  Q3  controller invariant: is a zero-unique-catch (useless) verifier still never added?

Reuses the frozen estimator/controller unchanged. Writes gsm8k marginal/phase tables.
Blind-spot π₀ is reported descriptively only (secondary, closed — see C11).
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

CODE = {"DIVERSIFY": "D", "SCALE": "S", "STOP": "T", "NA": "-"}
TRACE = ["step", "cand_type", "candidate", "delta", "delta_ci_low", "delta_ci_high",
         "delta_fa", "unique_catches", "denom", "rel_cost", "V", "action"]


def load():
    d = raw_run_dir("gsm8k", "truncate", "gsm8k")
    recs = [json.loads(l) for p in glob.glob(f"{d}/*.jsonl") for l in open(p) if l.strip()]
    df = pd.DataFrame(recs)
    return df[df["n_gates"] == 50] if not df.empty else df


def same_model_fit(df):
    sm = df[(df.pairing_condition == "same_model") & (df.ground_truth_is_error)]
    m, K = [], []
    for _, r in sm.iterrows():
        acc = sum(1 for g in r["gates"] if g.get("accepted") is True)
        dec = sum(1 for g in r["gates"] if g.get("accepted") in (True, False))
        if dec:
            m.append(acc); K.append(dec)
    if len(m) < 4:
        return None
    return fit_mixture(np.array(m, float), np.array(K, float), compute_ci=True)


def main():
    pd.set_option("display.width", 200)
    df = load()
    print("=" * 88)
    if df.empty:
        print("[gsm8k] no data yet — run scripts/run_gsm8k_confirmatory.sh first")
        return
    P = _prep(df, 0.5)
    print(f"[gsm8k] universe={P['models']}  n_err={P['ne']} n_clean={P['nc']}")

    # ---- Q1: same-model redundancy ----
    f = same_model_fit(df)
    q1 = None
    if f:
        q1 = f.n_eff_at_k_max
        print(f"\nQ1 same_model redundancy: rho_v={f.rho_v:.3f}  n_eff@50={f.n_eff_at_k_max:.2f}  "
              f"miss={f.mu_hat:.3f}   [expect ~1 => repeated same-model still redundant]")
        print(f"   (secondary/descriptive) blind-spot pi0={f.blind_spot_mass:.3f} "
              f"CI={tuple(round(x,3) for x in f.blind_spot_ci)} p_ceiling={f.p_ceiling:.3f}")

    # ---- Q2 + Q3: greedy controller ----
    mv = greedy_marginal_value(df, lam=0.1)
    print("\ngreedy (selected):")
    print(mv[mv.selected][TRACE].to_string(index=False))

    added = mv[mv.selected & mv.action.isin(["DIVERSIFY", "SCALE"])]
    bad = added[added.delta_ci_low <= 0]                      # added despite CI-low<=0
    div = added[added.cand_type == "NEW"]
    useful_div = div[div.delta_ci_low > 0]
    never_zero = sorted(set(mv[(mv.cand_type == "NEW") & (mv.delta == 0)].candidate)
                        - set(added.candidate))
    q2 = not useful_div.empty
    q3 = bad.empty
    print(f"\nQ2 useful diverse verifier (Δcatch CI-low>0): {q2}  "
          f"-> {list(useful_div.candidate)} " +
          (f"(CI-low {useful_div.delta_ci_low.round(3).tolist()})" if q2 else ""))
    print(f"Q3 controller invariant (no useless verifier added): {q3}  "
          f"zero-Δ correctly-never-added={never_zero}  added-with-nonsig-Δ={list(bad.candidate)}")

    ph = phase_diagram(df)
    piv = ph.pivot(index="cost_ratio_fa_over_miss", columns="lambda_cost",
                   values="first_action").replace(CODE)
    print("\nphase (rows=fa/miss, cols=λ; D/S/T):")
    print(piv.to_string())

    mv.to_csv("results/tables/gsm8k_marginal.csv", index=False)
    ph.to_csv("results/tables/gsm8k_phase.csv", index=False)

    print("\n" + "=" * 88)
    print("PRE-REGISTERED VERDICT (GSM8K / reasoning family):")
    print(f"  Q1 same-model n_eff@50 ~1 ......... {q1:.2f}" if q1 else "  Q1 .. n/a")
    print(f"  Q2 useful diversity (Δcatch>0) .... {q2}")
    print(f"  Q3 controller rejects useless ..... {q3}")
    print("  direction HELD" if (q1 and q1 < 1.6 and q2 and q3)
          else "  direction NOT fully held — inspect above")


if __name__ == "__main__":
    main()
