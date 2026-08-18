#!/usr/bin/env bash
# =============================================================================
# #3 counterexample shot: collect gemma2:9b verdicts (Google family = genuinely
# LOW-correlation with Qwen/Llama/Mistral, and 9B = not weak) on the SAME items as
# the six confirmatory families. Question: is a STRONG + DIVERSE verifier USELESS
# (near-zero conditional marginal catch)? If yes -> the non-trivial diversity!=value
# counterexample the reviewer asked for. If it is useful -> honest negative data point.
#
# Pool same_model_gemma=[gemma2_9b] -> pairing_condition="same_model_gemma" (its own
# files; the frozen same_model_neff filters =="same_model", so this is invisible to it).
#
# RUN AFTER the k=50 redundancy run frees the GPU:
#   nohup bash scripts/run_gemma_diversity.sh > gemma_diversity.log 2>&1 &
# ANALYSE: $PY scripts/investigate_diversity_counterexample.py
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/alex/venvs/env/bin/python

K=3
N_ERR=150
N_OK=75
POOL=same_model_gemma

run () {  # $1=source $2=benchmark(or "") $3=raw-dir-label
  local dir="data/raw_cascade_runs/$3"
  if ls "$dir"/*"__${POOL}__${POOL}.jsonl" >/dev/null 2>&1; then
    echo "SKIP $3: ${POOL} already collected"
    return
  fi
  echo "=== $(date +%H:%M:%S)  gemma  $3  (k=$K n_err=$N_ERR n_ok=$N_OK) ==="
  if [ -n "$2" ]; then
    $PY -u run_cascade.py --source "$1" --benchmark "$2" --pool "$POOL" \
        --k "$K" --n-errors "$N_ERR" --n-non-errors "$N_OK" --skip-analyse
  else
    $PY -u run_cascade.py --source "$1" --pool "$POOL" \
        --k "$K" --n-errors "$N_ERR" --n-non-errors "$N_OK" --skip-analyse
  fi
}

run mcqa  arc        mcqa__arc__ctx_truncate
run mcqa  mmlu       mcqa__mmlu__ctx_truncate
run mcqa  csqa       mcqa__csqa__ctx_truncate
run mcqa  truthfulqa mcqa__truthfulqa__ctx_truncate
run gsm8k ""         gsm8k__gsm8k__ctx_truncate
run mbpp  ""         mbpp__mbpp__ctx_truncate

echo ""
echo "=== gemma verdicts collected. Analyse #3 offline:"
echo "    $PY scripts/investigate_diversity_counterexample.py"
