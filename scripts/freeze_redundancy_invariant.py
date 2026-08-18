"""FROZEN summary: the redundancy-invariant replication across the 5 evaluated
benchmark/task families (MAST FM-3.3 + ARC/MMLU/CSQA/TruthfulQA).

Two claims, one table:
  (1) same-model redundancy: n_eff@50 stays ~1 (repeated same-model gates add
      almost no independent evidence), and
  (2) the controller invariant: which verifier is diversified in is NOT set by any
      fixed model/family label -- it is the conditional marginal catch on THIS task
      (Wilson CI-low > 0). Zero-unique-catch candidates are never added.

Reporting discipline: this is "replicated across five evaluated benchmark/task
families", NOT "works on any benchmark/model". Do not over-generalise.

Read-only over frozen runs; writes results/tables/redundancy_invariant_5family.csv.
Does NOT touch the controller.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theory.betabinom_mixture import fit_mixture

# family label -> raw glob of same-model runs (ctx_truncate, canonical n_gates==50)
FAMILIES = {
    "MAST-3.3": "data/raw_cascade_runs/mast__3_3__ctx_truncate/*same_model*.jsonl",
    "ARC": "data/raw_cascade_runs/mcqa__arc__ctx_truncate/*same_model*.jsonl",
    "MMLU": "data/raw_cascade_runs/mcqa__mmlu__ctx_truncate/*same_model*.jsonl",
    "CSQA": "data/raw_cascade_runs/mcqa__csqa__ctx_truncate/*same_model*.jsonl",
    "TruthfulQA": "data/raw_cascade_runs/mcqa__truthfulqa__ctx_truncate/*same_model*.jsonl",
}
# controller-invariant evidence comes straight from the already-frozen MCQA greedy tables
MCQA_MARGINAL = {
    "ARC": "results/tables/mcqa_marginal_arc.csv",
    "MMLU": "results/tables/mcqa_marginal_mmlu.csv",
    "CSQA": "results/tables/mcqa_marginal_csqa.csv",
    "TruthfulQA": "results/tables/mcqa_marginal_truthfulqa.csv",
}


def same_model_neff(pattern):
    recs = [json.loads(l) for p in glob.glob(pattern) for l in open(p) if l.strip()]
    recs = [r for r in recs if r.get("n_gates") == 50
            and r["pairing_condition"] == "same_model" and r["ground_truth_is_error"]]
    m, K = [], []
    for r in recs:
        acc = sum(1 for g in r["gates"] if g.get("accepted") is True)
        dec = sum(1 for g in r["gates"] if g.get("accepted") in (True, False))
        if dec:
            m.append(acc); K.append(dec)
    if len(m) < 4:
        return None
    f = fit_mixture(np.array(m), np.array(K), compute_ci=True)
    return dict(n=len(m), rho_v=f.rho_v, n_eff50=f.n_eff_at_k_max, miss=f.mu_hat)


def greedy_invariant(path):
    d = pd.read_csv(path)
    solo = d[d.action == "START"].iloc[0]
    added = d[(d.selected) & (d.action.isin(["DIVERSIFY", "SCALE"])) & (d.cand_type == "NEW")]
    # dead = NEW candidates whose realised unique marginal catch is exactly zero
    dead = sorted(set(d[(d.cand_type == "NEW") & (d.delta == 0)].candidate) - set(added.candidate))
    bad = added[added.delta_ci_low <= 0]  # added despite non-significant Wilson CI-low
    div = added.iloc[0] if not added.empty else None
    return dict(
        solo_verifier=solo.candidate,
        solo_catch=float(solo.delta),
        diversified_in=None if div is None else div.candidate,
        div_unique=None if div is None else int(div.unique_catches),
        div_denom=None if div is None else int(div.denom),
        div_ci_low=None if div is None else float(div.delta_ci_low),
        dead_never_added=",".join(dead),
        invariant_ok=bool(bad.empty),
    )


def main():
    rows = []
    for fam, pat in FAMILIES.items():
        nf = same_model_neff(pat)
        row = {"family": fam, "n": nf["n"], "sm_rho_v": round(nf["rho_v"], 3),
               "sm_n_eff50": round(nf["n_eff50"], 2), "sm_miss": round(nf["miss"], 3)}
        if fam in MCQA_MARGINAL:
            row.update(greedy_invariant(MCQA_MARGINAL[fam]))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = "results/tables/redundancy_invariant_5family.csv"
    df.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    neff = df["sm_n_eff50"]
    print(f"\nsame-model n_eff@50 range across 5 families: [{neff.min():.2f}, {neff.max():.2f}]")
    div = df[df.div_unique.notna()]
    print(f"controller invariant OK on all MCQA families: {bool(df.invariant_ok.dropna().all())}"
          f"  (pooled diversify evidence {int(div.div_unique.sum())}/{int(div.div_denom.sum())} residual)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
