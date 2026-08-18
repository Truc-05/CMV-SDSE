"""#3 investigation: is there a verifier that is individually STRONG, LOW-correlation
with the best solo, yet USELESS (near-zero conditional marginal catch)?

If the only never-added verifier is the weak `llama32_3b`, the "diversity != value"
story risks collapsing to the trivial "weak = useless". A strong-but-diverse-yet-useless
verifier would be the non-trivial counterexample. This scans the verdicts offline.

The gemma2_9b probe is collected at k=3 (its own diversity run), while the frozen solos
are k=50. `ra.load(..., keep_all_k=True)` is REQUIRED to see it: the default loader keeps
only n_gates==50 and silently drops the k=3 gemma rows -- which is why the earlier run
reported a stale "NONE found". Any candidate surfaced at k!=50 is a PROBE result pending a
protocol-matched (k=50) confirmation, and is flagged as such below; do not headline it.

Per family, per core verifier v: standalone catch, FA, correlation(phi) with the best solo,
and conditional marginal catch given the best solo (fraction of the solo's MISSED errors that
v newly catches). Prints candidates for "strong + diverse + useless".
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import reviewer_analyses as ra
from theory.marginal_value import _prep

CORE = ["qwen7b", "llama31", "llama32_3b", "mistral7b", "gemma2_9b"]  # +gemma2_9b (#3 shot)


def phi(a, b):
    a = a.astype(float); b = b.astype(float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def verifier_kmap(slug):
    """Per-verifier gate count (n_gates), so the k=3-probe vs k=50-solo mismatch is explicit.

    Handles both record schemas: per-verifier rows (frozen pools carry a top-level
    verifier_id) and per-item rows (gemma/redun pools carry a `gates` list)."""
    km = {}
    for p in glob.glob(f"data/raw_cascade_runs/{slug}/*.jsonl"):
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            ng = r.get("n_gates")
            if r.get("verifier_id"):
                km[r["verifier_id"]] = ng
            for g in r.get("gates", []) or []:
                if g.get("verifier_id"):
                    km[g["verifier_id"]] = ng
    return km


def main():
    rows = []
    for fam, slug in ra.FAM.items():
        # keep_all_k=True -> include the gemma2_9b k=3 probe (default loader drops it).
        P = _prep(ra.load(slug, keep_all_k=True), 0.5,
                  exclude=frozenset({"mistral_small_24b"}))  # keep gemma2_9b (#3 probe)
        Ce, Cf = P["Ce"], P["Cf"]
        km = verifier_kmap(slug)
        models = [m for m in CORE if m in Ce]
        solo = max(models, key=lambda m: Ce[m].mean())          # best standalone catch
        miss = ~Ce[solo]                                         # errors the solo misses
        for m in models:
            marg = float(Ce[m][miss].mean()) if miss.any() else 0.0
            rows.append(dict(
                family=fam, verifier=m, k=km.get(m),
                catch=round(float(Ce[m].mean()), 3),
                fa=round(float(Cf[m].mean()), 3),
                is_best_solo=(m == solo),
                solo_missed=int(miss.sum()),
                rho_with_solo=round(phi(Ce[m], Ce[solo]), 3),
                marginal_catch_given_solo=round(marg, 3),
            ))
    D = pd.DataFrame(rows)
    out = "results/tables/rev_diversity_counterexample.csv"
    D.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print(D.to_string(index=False))

    # Candidate = non-solo verifier that is STRONG (catch >= 0.6), DIVERSE (rho < 0.6),
    # yet USELESS (marginal <= 0.10). Contrast with the dead-weak case.
    cand = D[(~D.is_best_solo) & (D.catch >= 0.60) & (D.rho_with_solo < 0.60)
             & (D.marginal_catch_given_solo <= 0.10)]
    print("\n=== #3 candidates: STRONG (catch>=.6) + DIVERSE (rho<.6) + USELESS (marg<=.10) ===")
    print(cand.to_string(index=False) if len(cand) else "  NONE found in existing data.")
    probe = cand[cand.k != 50]
    if len(probe):
        print("\n  !! PROBE-ONLY (k != 50) -- NOT confirmed. Re-run at k=50 before headlining:")
        for _, r in probe.iterrows():
            print(f"     {r.family}/{r.verifier}: k={r.k}, marg={r.marginal_catch_given_solo} "
                  f"on {r.solo_missed} residual errors (0/{r.solo_missed} has a wide CI).")
    print("\n=== dead-weak reference (llama32_3b) ===")
    print(D[D.verifier == "llama32_3b"][["family", "catch", "rho_with_solo",
          "marginal_catch_given_solo"]].to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
