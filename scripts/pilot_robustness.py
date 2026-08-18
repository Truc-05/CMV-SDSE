"""Within-mode Stage-2 robustness pilot on cached FM-3.1 / FM-3.2 data (no GPU / no model calls).

Reports, per failure mode: candidate universe, greedy sequence, Delta_catch (+Wilson CI),
Delta_FA, V, the (fa/miss cost, lambda) phase regions, and the cross-mode INVARIANT that a
useless (zero unique-catch) candidate is never diversified in — regardless of correlation.
Writes only mode-specific files; the shared FM-3.3 tables are left untouched.
"""
import glob
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cascade.paths import raw_run_dir
from theory.marginal_value import greedy_marginal_value, phase_diagram, _prep

CODE = {"DIVERSIFY": "D", "SCALE": "S", "STOP": "T", "NA": "-"}
TRACE_COLS = ["step", "cand_type", "candidate", "delta", "delta_ci_low", "delta_ci_high",
              "delta_fa", "unique_catches", "denom", "rel_cost", "V", "action"]


def load(mode):
    recs = [json.loads(l) for p in glob.glob(f"{raw_run_dir(mode, 'truncate')}/*.jsonl")
            for l in open(p) if l.strip()]
    df = pd.DataFrame(recs)
    return df[df["n_gates"] == 50] if not df.empty else df


def main():
    pd.set_option("display.width", 200)
    for mode in ["3.1", "3.2"]:
        df = load(mode)
        print("=" * 84)
        if df.empty:
            print(f"FM-{mode}: no truncate data")
            continue
        P = _prep(df, 0.5)
        print(f"FM-{mode}  universe={P['models']}  n_err={P['ne']}  n_clean={P['nc']}"
              "   (NOTE: pools differ from FM-3.3 — compare direction, not magnitude)")

        mv = greedy_marginal_value(df, lam=0.1)
        print("\ngreedy trace (selected winners):")
        print(mv[mv.selected][TRACE_COLS].to_string(index=False))

        print("\nstep-1 full ranking (diverse NEW vs repeated SCALE):")
        s1 = mv[mv.step == 1]
        if not s1.empty:
            print(s1[TRACE_COLS[1:]].to_string(index=False))

        zero = sorted(set(mv[mv.delta == 0].candidate))
        div = sorted(set(mv[mv.action == "DIVERSIFY"].candidate))
        bad = [c for c in zero if c in div]
        print(f"\nINVARIANT  zero-Δcatch candidates={zero}  diversified-in={div}  "
              f"-> useless-but-selected={bad}  ({'OK' if not bad else 'VIOLATED'})")

        ph = phase_diagram(df)
        piv = ph.pivot(index="cost_ratio_fa_over_miss", columns="lambda_cost",
                       values="first_action").replace(CODE)
        print("\nphase (rows=fa/miss cost, cols=λ; D=diversify S=scale T=stop):")
        print(piv.to_string())

        mv.to_csv(f"results/tables/pilot_marginal_{mode}.csv", index=False)
        ph.to_csv(f"results/tables/pilot_phase_{mode}.csv", index=False)


if __name__ == "__main__":
    main()
