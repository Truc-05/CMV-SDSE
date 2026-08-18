"""C11 power analysis — how many ITEMS to detect the same-model blind-spot atom (if it is real)?

At n=150 the same-model ceiling CI touches 0 on every family: the atom (M3) is confounded with
the heavy Beta-Binomial tail (M2), so the LRT is non-significant. This asks the pre-registered
question: *assuming the fitted atom is the true generating process*, how many error items are
needed for detection power >= 0.8?

Method: parametric bootstrap under the fitted M3 (pi0_hat, a_hat, b_hat) of each same_model family.
For each candidate n, simulate B datasets (atom w.p. pi0 -> m=K=50; else alpha~Beta(a,b),
m~Binom(50,alpha)), fit the mixture, and count detections (boundary-corrected LRT p_ceiling<0.05,
the fast equivalent of CI_95(pi0)-low>0). This is OPTIMISTIC: it conditions on the atom being real
with size pi0_hat; if the truth is a no-atom Beta-Binomial, no n ever detects it.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theory.betabinom_mixture import fit_mixture, _fit_binomial, _fit_betabinom, _fit_ceiling

FAMILIES = {
    "ARC": "data/raw_cascade_runs/mcqa__arc__ctx_truncate/*.jsonl",
    "MMLU": "data/raw_cascade_runs/mcqa__mmlu__ctx_truncate/*.jsonl",
    "CSQA": "data/raw_cascade_runs/mcqa__csqa__ctx_truncate/*.jsonl",
    "TruthfulQA": "data/raw_cascade_runs/mcqa__truthfulqa__ctx_truncate/*.jsonl",
}
K_GATES = 50
N_GRID = [150, 300, 500, 1000, 2000]
B = 120
SEED = 20260810


def fit_same_model(pattern):
    recs = [json.loads(l) for p in glob.glob(pattern) for l in open(p) if l.strip()]
    recs = [r for r in recs if r.get("n_gates") == 50
            and r.get("pairing_condition") == "same_model" and r.get("ground_truth_is_error")]
    m, K = [], []
    for r in recs:
        acc = sum(1 for g in r["gates"] if g.get("accepted") is True)
        dec = sum(1 for g in r["gates"] if g.get("accepted") in (True, False))
        if dec:
            m.append(acc); K.append(dec)
    m, K = np.array(m, float), np.array(K, float)
    m1 = _fit_binomial(m, K)
    m2 = _fit_betabinom(m, K, mu_start=max(min(m1.params["mu"], 0.99), 0.01))
    m3 = _fit_ceiling(m, K, m2)
    return m3.params["pi0"], m3.params["a"], m3.params["b"]


def simulate(n, pi0, a, b, rng):
    is_atom = rng.random(n) < pi0
    alpha = rng.beta(a, b, size=n)
    m = rng.binomial(K_GATES, alpha)
    m[is_atom] = K_GATES
    return m.astype(float), np.full(n, K_GATES, dtype=float)


def power_at(n, pi0, a, b, rng, B):
    hits = 0
    for _ in range(B):
        m, K = simulate(n, pi0, a, b, rng)
        f = fit_mixture(m, K, compute_ci=False)   # LRT only -> fast
        if np.isfinite(f.p_ceiling) and f.p_ceiling < 0.05:
            hits += 1
    return hits / B


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for fam, pat in FAMILIES.items():
        pi0, a, b = fit_same_model(pat)
        print(f"[{fam}] fitted atom: pi0={pi0:.3f}  a={a:.3f} b={b:.3f}  "
              f"(rho_v={1/(a+b+1):.3f})", flush=True)
        for n in N_GRID:
            pw = power_at(n, pi0, a, b, rng, B)
            rows.append({"family": fam, "pi0_hat": round(pi0, 3), "n_items": n,
                         "detect_power": round(pw, 3)})
            print(f"    n={n:5d}  power={pw:.3f}", flush=True)
    df = pd.DataFrame(rows)
    out = "results/tables/blind_spot_power.csv"
    df.to_csv(out, index=False)
    piv = df.pivot(index="family", columns="n_items", values="detect_power")
    print("\ndetection power (LRT p<0.05) vs n_items:")
    print(piv.to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
