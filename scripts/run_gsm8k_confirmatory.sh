#!/usr/bin/env bash
# CONFIRMATORY GSM8K run — a QUALITATIVELY DIFFERENT task family (multi-step reasoning) under
# the SAME FROZEN hyperparameters as the MCQA confirmatory: --k 50, ctx_truncate, canonical
# errors-only, add-rule Wilson CI-low(Δcatch)>0, verifier universe {qwen7b,llama31,llama32_3b,
# mistral7b}. NO controller changes, NO re-tuning. Errors are REAL reasoning failures harvested
# from an OUT-OF-POOL solver (gemma2:9b), labelled vs gold — generate them first:
#     $PY scripts/gen_gsm8k_solutions.py
#
# gsm8k is a fresh source dir (data/raw_cascade_runs/gsm8k__gsm8k__ctx_truncate), so unlike the
# MCQA rerun there is no pilot to archive. Run ONCE (gate_runner appends: a second run would
# double-count — delete the dir to redo). Analyse offline afterwards.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/alex/venvs/env/bin/python

N_ERR=150
N_OK=75
POOLS=(same_model cross_family_3 same_family)   # same three as the MCQA confirmatory

# guard: refuse to append onto an existing run
D="data/raw_cascade_runs/gsm8k__gsm8k__ctx_truncate"
if [ -d "$D" ] && ls "$D"/*.jsonl >/dev/null 2>&1; then
  echo "ERROR: $D already has raw runs. Delete it to re-collect (gate_runner appends)." >&2
  exit 1
fi

for pool in "${POOLS[@]}"; do
  echo "=== $(date +%H:%M:%S)  gsm8k  pool=$pool  (n_err=$N_ERR n_ok=$N_OK k=50) ==="
  $PY -u run_cascade.py --source gsm8k --pool "$pool" \
      --k 50 --n-errors "$N_ERR" --n-non-errors "$N_OK" --skip-analyse
done

echo ""
echo "=== GSM8K CONFIRMATORY DONE.  Analyse (offline, no GPU): ==="
echo "  $PY scripts/pilot_gsm8k.py     # same_model n_eff@50, greedy Δcatch, controller invariant"
