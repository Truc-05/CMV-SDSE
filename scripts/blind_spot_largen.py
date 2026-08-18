"""C11 — blind-spot / ceiling (pi0) re-fit on the confirmatory large-n MCQA data (no GPU).

The secondary claim is: a mass pi0 of error items is false-accepted by EVERY gate of a
same-model cascade (an intrinsic shared blind spot), and a DIVERSE pool breaks part of it.
This claim is only allowed to be stated as *existence* when, per family:

    (i)  the profile-likelihood CI_95(pi0) EXCLUDES 0, and
    (ii) the atom model (M3, ceiling mixture) beats the no-atom model (M2, Beta-Binomial)
         by the boundary-corrected LRT  (p_ceiling < 0.05).

Otherwise we report an upper bound only. We fit same_model (the claim) and the diverse pools
(cross_family_3, same_family) side by side; the contrast we want is pi0(same_model) high with
CI-low>0 and pi0(diverse) sharply lower.

Read-only over frozen large-n runs. Writes results/tables/blind_spot_largen.csv.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theory.betabinom_mixture import fit_mixture

FAMILIES = {
    "MAST-3.3": "data/raw_cascade_runs/mast__3_3__ctx_truncate/*.jsonl",
    "ARC": "data/raw_cascade_runs/mcqa__arc__ctx_truncate/*.jsonl",
    "MMLU": "data/raw_cascade_runs/mcqa__mmlu__ctx_truncate/*.jsonl",
    "CSQA": "data/raw_cascade_runs/mcqa__csqa__ctx_truncate/*.jsonl",
    "TruthfulQA": "data/raw_cascade_runs/mcqa__truthfulqa__ctx_truncate/*.jsonl",
}
POOLS = ["same_model", "cross_family_3", "same_family"]


def load_mK(pattern, pool):
    """Per-item (m, K): m = false-accepts (unwanted verdicts) on an ERROR item over K decided gates."""
    recs = [json.loads(l) for p in glob.glob(pattern) for l in open(p) if l.strip()]
    recs = [r for r in recs if r.get("n_gates") == 50
            and r.get("pairing_condition") == pool and r.get("ground_truth_is_error")]
    m, K = [], []
    for r in recs:
        acc = sum(1 for g in r["gates"] if g.get("accepted") is True)
        dec = sum(1 for g in r["gates"] if g.get("accepted") in (True, False))
        if dec:
            m.append(acc); K.append(dec)
    return np.array(m, dtype=float), np.array(K, dtype=float)


def row_for(fam, pool, m, K):
    n = len(m)
    if n < 4:
        return None
    at_ceiling = int(np.sum(np.isclose(m, K)))          # items every gate false-accepted
    f = fit_mixture(m, K, compute_ci=True)
    lo, hi = f.blind_spot_ci
    ci_excludes_0 = bool(np.isfinite(lo) and lo > 0.0)
    return {
        "family": fam, "pool": pool, "n": n,
        "raw_frac_at_ceiling": round(at_ceiling / n, 3), "n_at_ceiling": at_ceiling,
        "pi0_hat": round(float(f.blind_spot_mass), 3),
        "pi0_ci_lo": None if not np.isfinite(lo) else round(float(lo), 3),
        "pi0_ci_hi": None if not np.isfinite(hi) else round(float(hi), 3),
        "lr_ceiling": round(float(f.lr_ceiling), 2),
        "p_ceiling": None if not np.isfinite(f.p_ceiling) else round(float(f.p_ceiling), 4),
        "ci_excludes_0": ci_excludes_0,
        # existence = both conditions; else at most an upper bound
        "atom_supported": bool((f.p_ceiling < 0.05) if np.isfinite(f.p_ceiling) else False),
        "existence_claim_ok": bool(ci_excludes_0 and np.isfinite(f.p_ceiling) and f.p_ceiling < 0.05),
    }


def main():
    pd.set_option("display.width", 220)
    rows = []
    for fam, pat in FAMILIES.items():
        for pool in POOLS:
            m, K = load_mK(pat, pool)
            r = row_for(fam, pool, m, K)
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)
    out = "results/tables/blind_spot_largen.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))

    sm = df[df.pool == "same_model"]
    print("\n--- SAME-MODEL blind-spot verdict (the claim) ---")
    for _, r in sm.iterrows():
        verdict = ("EXISTENCE OK" if r.existence_claim_ok
                   else "UPPER-BOUND ONLY (CI touches 0 or atom LRT n.s.)")
        print(f"  {r.family:10s} n={int(r.n):3d}  pi0_hat={r.pi0_hat:.3f}  "
              f"CI=[{r.pi0_ci_lo}, {r.pi0_ci_hi}]  p_ceiling={r.p_ceiling}  -> {verdict}")

    n_ok = int(sm.existence_claim_ok.sum())
    print(f"\nsame-model families with CI-low>0 AND atom LRT: {n_ok}/{len(sm)}")

    # diverse-vs-same contrast (does diversification break the blind spot?)
    print("\n--- diversification breaks the blind spot? (raw ceiling mass, same->diverse) ---")
    for fam in FAMILIES:
        sub = df[df.family == fam].set_index("pool")
        if "same_model" not in sub.index:
            continue
        base = sub.loc["same_model", "raw_frac_at_ceiling"]
        parts = [f"same={base:.3f}"]
        for p in ("cross_family_3", "same_family"):
            if p in sub.index:
                parts.append(f"{p}={sub.loc[p, 'raw_frac_at_ceiling']:.3f}")
        print(f"  {fam:10s} " + "  ".join(parts))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
