"""FROZEN summary EXTENDED to seven evaluated benchmark/task families.

Adds the two confirmatory large-n families to the frozen five:
  reasoning  -> GSM8K   (code path: gsm8k__gsm8k__ctx_truncate + gsm8k_marginal.csv)
  code       -> MBPP    (code path: mbpp__mbpp__ctx_truncate  + mbpp_marginal.csv)

Same two claims, same table, same reporting discipline as
freeze_redundancy_invariant.py -- this ONLY appends the two new rows so the
five-family artifact stays byte-for-byte for provenance. Reuses the frozen
helpers (same_model_neff, greedy_invariant) unchanged; the controller is not
touched. Writes results/tables/redundancy_invariant_7family.csv.

Reporting discipline: "replicated across seven evaluated benchmark/task
families" (MCQA x4 + MAST-3.3 + GSM8K reasoning + MBPP code), NOT "any
benchmark/model". Do not over-generalise.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# import the frozen helpers without modifying the 5-family script
spec = importlib.util.spec_from_file_location(
    "freeze5", ROOT / "scripts" / "freeze_redundancy_invariant.py")
freeze5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(freeze5)

# five frozen families + two confirmatory large-n families (reasoning, code)
FAMILIES = dict(freeze5.FAMILIES)
FAMILIES["GSM8K"] = "data/raw_cascade_runs/gsm8k__gsm8k__ctx_truncate/*same_model*.jsonl"
FAMILIES["MBPP"] = "data/raw_cascade_runs/mbpp__mbpp__ctx_truncate/*same_model*.jsonl"

MARGINAL = dict(freeze5.MCQA_MARGINAL)
MARGINAL["GSM8K"] = "results/tables/gsm8k_marginal.csv"
MARGINAL["MBPP"] = "results/tables/mbpp_marginal.csv"


def main():
    rows = []
    for fam, pat in FAMILIES.items():
        nf = freeze5.same_model_neff(pat)
        row = {"family": fam, "n": nf["n"], "sm_rho_v": round(nf["rho_v"], 3),
               "sm_n_eff50": round(nf["n_eff50"], 2), "sm_miss": round(nf["miss"], 3)}
        if fam in MARGINAL:
            row.update(freeze5.greedy_invariant(MARGINAL[fam]))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = "results/tables/redundancy_invariant_7family.csv"
    df.to_csv(out, index=False)

    pd.set_option("display.width", 220)
    print(df.to_string(index=False))
    neff = df["sm_n_eff50"]
    print(f"\nsame-model n_eff@50 range across 7 families: [{neff.min():.2f}, {neff.max():.2f}]")
    div = df[df.div_unique.notna()]
    print(f"controller invariant OK on all families with greedy: "
          f"{bool(df.invariant_ok.dropna().all())}  "
          f"(pooled diversify evidence {int(div.div_unique.sum())}/{int(div.div_denom.sum())} residual)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
