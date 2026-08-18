#!/usr/bin/env bash
# =============================================================================
# GPU0 variant of run_gemma_diversity.sh: identical collection, but pinned to the
# free GPU0 via configs/models_gpu0.yaml (ollama host 11435, CUDA_VISIBLE_DEVICES=1
# -> physical GPU0). Runs alongside the 24B redundancy job on 11437/GPU1 without
# contending for VRAM. See run_gemma_diversity.sh header for the #3 rationale.
#
# RUN:  nohup bash scripts/run_gemma_diversity_gpu0.sh > gemma_diversity.log 2>&1 &
# ANALYSE: $PY scripts/investigate_diversity_counterexample.py
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/alex/venvs/env/bin/python

K=3
N_ERR=150
N_OK=75
POOL=same_model_gemma
CFG=configs/models_gpu0.yaml

run () {  # $1=source $2=benchmark(or "") $3=raw-dir-label
  local dir="data/raw_cascade_runs/$3"
  if ls "$dir"/*"__${POOL}__${POOL}.jsonl" >/dev/null 2>&1; then
    echo "SKIP $3: ${POOL} already collected"
    return
  fi
  echo "=== $(date +%H:%M:%S)  gemma-gpu0  $3  (k=$K n_err=$N_ERR n_ok=$N_OK) ==="
  if [ -n "$2" ]; then
    $PY -u run_cascade.py --source "$1" --benchmark "$2" --pool "$POOL" \
        --models-config "$CFG" --k "$K" --n-errors "$N_ERR" --n-non-errors "$N_OK" --skip-analyse
  else
    $PY -u run_cascade.py --source "$1" --pool "$POOL" \
        --models-config "$CFG" --k "$K" --n-errors "$N_ERR" --n-non-errors "$N_OK" --skip-analyse
  fi
}

run mcqa  arc        mcqa__arc__ctx_truncate
run mcqa  mmlu       mcqa__mmlu__ctx_truncate
run mcqa  csqa       mcqa__csqa__ctx_truncate
run mcqa  truthfulqa mcqa__truthfulqa__ctx_truncate
run gsm8k ""         gsm8k__gsm8k__ctx_truncate
run mbpp  ""         mbpp__mbpp__ctx_truncate

echo ""
echo "=== gemma verdicts collected (GPU0). Analyse #3 offline:"
echo "    $PY scripts/investigate_diversity_counterexample.py"
