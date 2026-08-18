"""Same-model redundancy n_eff@50 for the open 24B (Mistral-Small-24B), OFFLINE.

Reads the k=50 verdicts collected under pool `redun_mistral24b` (pairing_condition
"redun_mistral24b", n_gates==50, errors-only) and fits the same Beta-Binomial mixture
used for the frozen 5/7-family redundancy table. Tests whether the redundancy null
(n_eff@50 ~ 1) holds ABOVE the 14B tier. Does NOT touch any frozen artifact: it filters
pairing_condition=="redun_mistral24b" (the frozen same_model_neff filters =="same_model").

Writes results/tables/redun_mistral24b_neff.csv.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from theory.betabinom_mixture import fit_mixture

PAIRING = "redun_mistral24b"
FAMILIES = {
    "MAST-3.3":   "data/raw_cascade_runs/mast__3_3__ctx_truncate/*redun_mistral24b*.jsonl",
    "ARC":        "data/raw_cascade_runs/mcqa__arc__ctx_truncate/*redun_mistral24b*.jsonl",
    "MMLU":       "data/raw_cascade_runs/mcqa__mmlu__ctx_truncate/*redun_mistral24b*.jsonl",
    "CSQA":       "data/raw_cascade_runs/mcqa__csqa__ctx_truncate/*redun_mistral24b*.jsonl",
    "TruthfulQA": "data/raw_cascade_runs/mcqa__truthfulqa__ctx_truncate/*redun_mistral24b*.jsonl",
    "GSM8K":      "data/raw_cascade_runs/gsm8k__gsm8k__ctx_truncate/*redun_mistral24b*.jsonl",
    "MBPP":       "data/raw_cascade_runs/mbpp__mbpp__ctx_truncate/*redun_mistral24b*.jsonl",
}
FROZEN_BAND = (1.06, 1.21)   # 7-family same-model n_eff@50 band (<=14B pool)


def neff24(pattern):
    recs = [json.loads(l) for p in glob.glob(pattern) for l in open(p) if l.strip()]
    recs = [r for r in recs if r.get("n_gates") == 50
            and r.get("pairing_condition") == PAIRING and r.get("ground_truth_is_error")]
    m, K = [], []
    for r in recs:
        acc = sum(1 for g in r["gates"] if g.get("accepted") is True)
        dec = sum(1 for g in r["gates"] if g.get("accepted") in (True, False))
        if dec:
            m.append(acc); K.append(dec)
    if len(m) < 4:
        return None
    f = fit_mixture(np.array(m), np.array(K), compute_ci=True)
    # n_eff@50 monotone-decreasing in rho_v -> map the rho_v CI through n_eff(rho)=50/(1+49 rho)
    lo_r, hi_r = f.rho_v_ci
    neff = lambda r: 50.0 / (1.0 + 49.0 * r)
    return dict(n=len(m), rho_v=round(f.rho_v, 3),
                rho_v_ci=(round(lo_r, 3), round(hi_r, 3)),
                # keep full precision: rounding to 3dp here then .2f in the figure double-rounds
                # boundary values (e.g. CSQA 1.525025 -> 1.525 -> "1.52" instead of "1.53").
                n_eff50=round(f.n_eff_at_k_max, 6),
                n_eff50_ci=(round(neff(hi_r), 3), round(neff(lo_r), 3)),
                miss=round(f.mu_hat, 3))


def main():
    rows = []
    for fam, pat in FAMILIES.items():
        r = neff24(pat)
        if r is None:
            print(f"[skip] {fam}: no k=50 redun_mistral24b records yet ({pat})", file=sys.stderr)
            continue
        rows.append(dict(family=fam, **r))
    if not rows:
        print("No data. Run scripts/run_mistral24b_redundancy.sh first.", file=sys.stderr)
        return
    D = pd.DataFrame(rows)
    out = "results/tables/redun_mistral24b_neff.csv"
    D.to_csv(out, index=False)
    print("=== Mistral-Small-24B (open, 24B) same-model n_eff@50 ===")
    print(D.to_string(index=False))
    nmin, nmax = D.n_eff50.min(), D.n_eff50.max()
    print(f"\n24B n_eff@50 range: [{nmin:.2f}, {nmax:.2f}]   (frozen <=14B 7-family band: {FROZEN_BAND})")
    inband = (D.n_eff50 <= 1.5).all()   # ~1 means redundancy null holds
    print("VERDICT:", "redundancy null HOLDS above 14B (n_eff@50 ~ 1)" if inband
          else "n_eff@50 materially > 1 for the 24B -- redundancy null does NOT extend")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
