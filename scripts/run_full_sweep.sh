#!/usr/bin/env bash
# Full preregistered sweep — RUN THIS YOURSELF (author does not execute it).
# Offline analysis is already done; this is the LLM-backend part that needs a GPU/cluster.
#
# Runtime arithmetic (per pool):  calls = n_items * k * repeats
#   n_items = n_errors + n_non_errors = 40 + 20 = 60 ; k = 50 ; repeats = 1
#   -> 60 * 50 * 1 = 3000 calls/pool.  At ~20 s/call serial => ~16.7 h/pool, ~50 h for 3 pools.
#   Parallelise: Ollama serves concurrent requests; run the three pools at once and/or raise
#   OLLAMA_NUM_PARALLEL, or put one pool per GPU. 4x concurrency => ~12 h wall clock total.
#
# Why repeats=1 is enough: the item-level Beta-Binomial estimator uses the k=50 gate verdicts
# per item as the repeated verdicts the theory needs to identify the moments of G (rho_v and
# the low moments). Extra repeats multiply cost linearly and add little; only raise them if you
# want to separate within-item sampling noise at small k.
#
# Prereqs: `ollama serve` running; pull the models the pools use (all already present on the
# dev box): qwen2.5:7b, llama3.1:8b, llama3.2:3b, gemma2:9b, mistral:7b.
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET=3.3            # keep 3.3 for comparability with the pilot; add `fc3` for the full FC3 group
K=50
NERR=40
NCLEAN=20
REPEATS=1

# 0) sanity: plan the calls without touching the backend
for POOL in same_model same_family cross_family; do
  python run_cascade.py --pool "$POOL" --source mast --mast-target "$TARGET" \
    --k "$K" --n-errors "$NERR" --n-non-errors "$NCLEAN" --repeats "$REPEATS" --dry-run
done

# 1) H2 dependence ladder: rho_v(same_model) >= rho_v(same_family) >= rho_v(cross_family)
#    Run the three pools (foreground here; parallelise as noted above).
for POOL in same_model same_family cross_family; do
  echo ">>> running pool=$POOL"
  python run_cascade.py --pool "$POOL" --source mast --mast-target "$TARGET" \
    --k "$K" --n-errors "$NERR" --n-non-errors "$NCLEAN" --repeats "$REPEATS"
done
# each run appends to data/raw_cascade_runs/mast__3_3__ctx_truncate/ and re-runs analyse at the end.
# NOTE: the three pools share the same scope dir + failure_mode_id, so their records are
# distinguished by pairing_condition — analyse groups by it, giving one report row per pool,
# which is exactly the H2 comparison.

# 2) sensitivity — label noise (prereg §5) and transcript truncation (prereg §6)
python run_cascade.py --label-noise --source mast                         # writes results/tables/label_noise.json
python run_cascade.py --pool cross_family --source mast --mast-target "$TARGET" \
  --k "$K" --n-errors "$NERR" --n-non-errors "$NCLEAN" --mast-max-chars 6000   # tight  crop
python run_cascade.py --pool cross_family --source mast --mast-target "$TARGET" \
  --k "$K" --n-errors "$NERR" --n-non-errors "$NCLEAN" --mast-max-chars 24000  # loose crop
# compare rho_v / pi0 between the two crops; report the sensitivity.

# 3) regenerate the offline artefacts from the new data
python run_cascade.py --analyse-only --source mast --mast-target "$TARGET" --k-gates 50  # canonical population: exclude stale k!=50 pilots
python -m theory.simulation --reps 400 --coverage-reps 120   # refresh calibration if K/n changed
python scripts/make_figures.py

echo "done. See results/tables/cascade_theory_report.csv (one row per pairing_condition = H2)."
