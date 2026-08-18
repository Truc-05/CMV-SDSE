#!/usr/bin/env bash
# CONFIRMATORY large-n MCQA run — same FROZEN hyperparameters as the pilot, only n scaled up.
# Methodology is locked (docs/METHODOLOGY_FREEZE.md): --k 50, ctx_truncate, canonical
# errors-only, add-rule Wilson CI-low(Δcatch)>0, universe {llama31,llama32_3b,mistral7b,qwen7b}.
# NO controller changes, NO re-tuning. This script only (a) preserves the frozen pilot, then
# (b) collects fresh large-n raw. Analyse offline afterwards.
#
# WHY the archive step: cascade/gate_runner.py opens the raw jsonl in APPEND mode ("a"),
# so re-running into the live dirs would concatenate onto the pilot records (double-count).
# We move the pilot raw + frozen pilot tables aside first, so the large-n run writes clean files.
#
# Item property: the loader shuffles with a fixed seed and takes the first (n_err+n_ok), so the
# large-n sample is a SUPERSET (prefix) of the pilot's items — a genuine scale-up, not a new draw.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/alex/venvs/env/bin/python

# ---- the ONLY knob: sample size. ARC-Challenge validation is the binding constraint
# (~295 usable questions after extraction), so uniform N_ERR+N_OK must stay < ~295.
# 150/75 => ~3.75x the pilot (was 40/20); residual-for-diversity ~4-22/bench => properly powered.
# MMLU/CSQA/TruthfulQA can go higher if you want per-bench power (raise inside the loop).
N_ERR=150
N_OK=75

BENCHES=(arc mmlu csqa truthfulqa)
POOLS=(same_model cross_family_3 same_family)   # all 3 needed to reconstruct the 4-model universe

# ---- (a) preserve the frozen pilot (raw + tables) into a timestamped archive ----
STAMP=$(date +%Y%m%d_%H%M%S)
ARCH="data/raw_cascade_runs/_archive_pilot_mcqa_${STAMP}"
mkdir -p "$ARCH" "results/tables/_archive_pilot_mcqa_${STAMP}"
for bench in "${BENCHES[@]}"; do
  d="data/raw_cascade_runs/mcqa__${bench}__ctx_truncate"
  [ -d "$d" ] && mv "$d" "$ARCH/" && echo "archived raw: $d -> $ARCH/"
done
for f in results/tables/mcqa_marginal_*.csv results/tables/mcqa_phase_*.csv \
         results/tables/redundancy_invariant_5family.csv; do
  [ -f "$f" ] && cp "$f" "results/tables/_archive_pilot_mcqa_${STAMP}/" && echo "archived table: $f"
done
echo "=== pilot preserved under *_${STAMP}; large-n will write fresh dirs ==="

# ---- (b) collect fresh large-n raw ----
for bench in "${BENCHES[@]}"; do
  for pool in "${POOLS[@]}"; do
    echo "=== $(date +%H:%M:%S)  mcqa/$bench  pool=$pool  (n_err=$N_ERR n_ok=$N_OK) ==="
    $PY -u run_cascade.py --source mcqa --benchmark "$bench" --pool "$pool" \
        --k 50 --n-errors "$N_ERR" --n-non-errors "$N_OK" --skip-analyse
  done
done

echo ""
echo "=== ALL CONFIRMATORY RUNS DONE.  Analyse (offline, no GPU): ==="
echo "  $PY scripts/pilot_mcqa.py                    # per-bench greedy + phase + n_eff"
echo "  $PY scripts/freeze_redundancy_invariant.py   # merged 5-family table"
echo ""
echo "# Optional — scale the 5th family (MAST-3.3) too, if you want confirmatory redundancy there:"
echo "#   for p in same_model cross_family_3 same_family; do"
echo "#     $PY -u run_cascade.py --source mast --mast-target 3.3 --pool \$p --k 50 --repeats 4 --skip-analyse"
echo "#   done   # (MAST scales via --repeats; archive its raw first, same append caveat)"
