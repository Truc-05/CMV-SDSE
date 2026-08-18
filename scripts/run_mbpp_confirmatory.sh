#!/usr/bin/env bash
# CONFIRMATORY MBPP run — a QUALITATIVELY DIFFERENT task family (Python code generation) under
# the SAME FROZEN hyperparameters as the MCQA/GSM8K confirmatory: --k 50, ctx_truncate, canonical
# errors-only, add-rule Wilson CI-low(Δcatch)>0, verifier universe {qwen7b,llama31,llama32_3b,
# mistral7b}. NO controller changes, NO re-tuning. Errors are REAL buggy programs harvested from
# an OUT-OF-POOL solver (gemma2:9b) and labelled by UNIT-TEST execution — generate them first:
#     $PY scripts/gen_mbpp_solutions.py
#
# mbpp is a fresh source dir (data/raw_cascade_runs/mbpp__mbpp__ctx_truncate), so unlike the MCQA
# rerun there is no pilot to archive. Run ONCE (gate_runner appends: a second run would
# double-count — delete the dir to redo). Analyse offline afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/alex/venvs/env/bin/python

N_ERR=150
N_OK=75
POOLS=(same_model cross_family_3 same_family)   # same three as the MCQA/GSM8K confirmatory

# guard: refuse to append onto an existing run
D="data/raw_cascade_runs/mbpp__mbpp__ctx_truncate"
if [ -d "$D" ] && ls "$D"/*.jsonl >/dev/null 2>&1; then
  echo "ERROR: $D already has raw runs. Delete it to re-collect (gate_runner appends)." >&2
  exit 1
fi

for pool in "${POOLS[@]}"; do
  echo "=== $(date +%H:%M:%S)  mbpp  pool=$pool  (n_err=$N_ERR n_ok=$N_OK k=50) ==="
  $PY -u run_cascade.py --source mbpp --pool "$pool" \
      --k 50 --n-errors "$N_ERR" --n-non-errors "$N_OK" --skip-analyse
done

echo ""
echo "=== MBPP CONFIRMATORY DONE.  Analyse (offline, no GPU): ==="
echo "  $PY scripts/pilot_mbpp.py     # same_model n_eff@50, greedy Δcatch, controller invariant"
