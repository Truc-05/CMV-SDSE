#!/usr/bin/env bash
# Benchmark #2 replication: run the frozen verifier pools on 4 non-MAST MC-QA benchmarks.
# Data collection only (--skip-analyse); analyse offline afterwards with scripts/pilot_mcqa.py.
# Each (benchmark, pool) is item-matched (same 60 items via seed_base), isolated in
# data/raw_cascade_runs/mcqa__<bench>__ctx_truncate/. ~3000 calls/pool (~14 min); 4*3 pools.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/alex/venvs/env/bin/python

BENCHES=(arc mmlu csqa truthfulqa)
POOLS=(same_model cross_family_3 same_family)   # n_eff | 3-model greedy | +llama32_3b (weak?)

for bench in "${BENCHES[@]}"; do
  for pool in "${POOLS[@]}"; do
    echo "=== $(date +%H:%M:%S)  mcqa/$bench  pool=$pool ==="
    $PY -u run_cascade.py --source mcqa --benchmark "$bench" --pool "$pool" \
        --k 50 --n-errors 40 --n-non-errors 20 --skip-analyse
  done
done
echo "=== ALL MCQA RUNS DONE — now: $PY scripts/pilot_mcqa.py ==="
