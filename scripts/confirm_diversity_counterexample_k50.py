"""#3 CONFIRM (k=50 only): the strong+diverse+useless counterexample, protocol-matched.

The scan in investigate_diversity_counterexample.py SURFACES candidates using the k=3 gemma
probe (via keep_all_k=True). But _per_model_reject pools every gate for a (verifier_id,
item_id) across ALL raw files, so once the k=50 redun_gemma verdicts exist alongside the k=3
same_model_gemma probe, the scan's gemma row silently mixes 3+50=53 gates -- a contaminated
"k=50" point. This script re-derives the CSQA/gemma2_9b confirmation from the k=50 redun_gemma
file ALONE, item-aligned against the frozen k=50 solo (qwen7b), so the number that goes in the
paper is genuinely protocol-matched. Read-only over raw gate records; no model calls.
"""
from __future__ import annotations
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "data/raw_cascade_runs/mcqa__csqa__ctx_truncate"
GEMMA_K50 = DIR / "CSQA__redun_gemma__redun_gemma.jsonl"      # gemma2_9b at k=50 (the confirm)
SOLO_K50 = DIR / "CSQA__same_model__same_model.jsonl"          # frozen solo qwen7b at k=50


def per_item_reject(path, verifier):
    """item_id -> (is_error, majority_reject) using ONLY `verifier`'s gates in THIS file."""
    acc = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if "ground_truth_is_error" not in r:
            continue
        cell = acc.setdefault(r["item_id"], [0, 0, bool(r["ground_truth_is_error"])])
        for g in r.get("gates", []) or []:
            if g.get("verifier_id") != verifier:
                continue
            a = g.get("accepted")
            if a in (True, False):
                cell[1] += 1
                cell[0] += int(a is True)
    return {i: (e, (ok / dec) < 0.5) for i, (ok, dec, e) in acc.items() if dec > 0}


def phi(a, b):
    import statistics as st
    if len(a) < 2 or st.pstdev(a) == 0 or st.pstdev(b) == 0:
        return 0.0
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return cov / (da * db) if da and db else 0.0


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    d = 1 + z2 / n
    c = (p + z2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    gem = per_item_reject(GEMMA_K50, "gemma2_9b")
    solo = per_item_reject(SOLO_K50, "qwen7b")
    shared = [i for i in gem if i in solo]
    errs = [i for i in shared if gem[i][0]]

    gem_catch = sum(gem[i][1] for i in errs) / len(errs)
    solo_catch = sum(solo[i][1] for i in errs) / len(errs)
    rho = phi([int(gem[i][1]) for i in shared], [int(solo[i][1]) for i in shared])

    missed = [i for i in errs if not solo[i][1]]                 # errors the solo lets through
    gem_saves = [i for i in missed if gem[i][1]]                 # of those, gemma catches
    marg = len(gem_saves) / len(missed) if missed else 0.0
    lo, hi = wilson(len(gem_saves), len(missed))

    out = ROOT / "results/tables/rev_diversity_counterexample_confirm.csv"
    out.write_text(
        "family,verifier,k,catch,solo,solo_catch,rho_with_solo,solo_missed,"
        "marginal_catch,marg_ci_low,marg_ci_high\n"
        f"CSQA,gemma2_9b,50,{gem_catch:.3f},qwen7b,{solo_catch:.3f},{rho:.3f},"
        f"{len(missed)},{marg:.3f},{lo:.3f},{hi:.3f}\n"
    )
    print("=== #3 CONFIRM (CSQA / gemma2_9b, k=50 only, vs frozen solo qwen7b) ===")
    print(f"  items shared             {len(shared)}  (errors {len(errs)})")
    print(f"  gemma2_9b standalone catch {gem_catch:.3f}   (STRONG)")
    print(f"  qwen7b solo catch          {solo_catch:.3f}")
    print(f"  phi(gemma, solo)           {rho:.3f}   (moderate correlation)")
    print(f"  solo misses                {len(missed)} error items")
    print(f"  gemma catches of those     {len(gem_saves)}")
    print(f"  conditional marginal catch {marg:.3f}   Wilson95 [{lo:.3f}, {hi:.3f}]   (USELESS)")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
