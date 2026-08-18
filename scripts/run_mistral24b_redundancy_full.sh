#!/usr/bin/env bash
# =============================================================================
# FULL 24B same-model redundancy (k=50) at PARITY with the frozen <=14B table:
#   6 confirmatory families @ n=150  +  MAST-3.3 @ n=40 (all available FM-3.3 errors)
#   = 7/7 families. Supersedes the earlier bounded 3-family n=50 check.
#
# Isolation unchanged: pool redun_mistral24b -> pairing_condition="redun_mistral24b",
# invisible to the frozen same_model_neff (=="same_model") and to _prep (probe-excluded).
# Deliberate post-freeze run (user-authorised). ~16h on a free GPU.
#
# RUN:  nohup bash scripts/run_mistral24b_redundancy_full.sh > mistral24b_redundancy_full.log 2>&1 &
# ANALYSE: $PY scripts/analyze_redun24b.py
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/alex/venvs/env/bin/python
K=50
POOL=redun_mistral24b

# Fresh n=150 collection -> remove the n=50 preliminary redun files (NOT the escalate
# same_model_mistral24b verdicts, which have a different name).
find data/raw_cascade_runs -name "*${POOL}*.jsonl" -print -delete
echo "cleared prior ${POOL} files; collecting fresh at parity n."

conf () {  # $1=source $2=benchmark(or "") $3=dir-label   -- confirmatory families @ n=150
  echo "=== $(date +%H:%M:%S)  redun24b-full  $3  (k=$K n_err=150) ==="
  if [ -n "$2" ]; then
    $PY -u run_cascade.py --source "$1" --benchmark "$2" --pool "$POOL" \
        --k "$K" --n-errors 150 --n-non-errors 0 --skip-analyse
  else
    $PY -u run_cascade.py --source "$1" --pool "$POOL" \
        --k "$K" --n-errors 150 --n-non-errors 0 --skip-analyse
  fi
}

conf mcqa  arc        mcqa__arc__ctx_truncate
conf mcqa  mmlu       mcqa__mmlu__ctx_truncate
conf mcqa  csqa       mcqa__csqa__ctx_truncate
conf mcqa  truthfulqa mcqa__truthfulqa__ctx_truncate
conf gsm8k ""         gsm8k__gsm8k__ctx_truncate
conf mbpp  ""         mbpp__mbpp__ctx_truncate

# MAST-3.3 (real multi-agent traces): only 40 FM-3.3 error items exist -> n=40 (matches frozen)
echo "=== $(date +%H:%M:%S)  redun24b-full  MAST-3.3  (k=$K n_err=40) ==="
$PY -u run_cascade.py --source mast --mast-target 3.3 --pool "$POOL" \
    --k "$K" --n-errors 40 --n-non-errors 0 --skip-analyse

echo ""
echo "=== full 7-family 24B redundancy collected. Analyse offline (no GPU):"
echo "    $PY scripts/analyze_redun24b.py"
