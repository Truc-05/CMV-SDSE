#!/usr/bin/env bash
# =============================================================================
# SAME-MODEL REDUNDANCY of the open 24B (Mistral-Small-24B) at k=50, to measure its
# own n_eff@50 and test whether the redundancy null (n_eff@50 ~ 1) holds ABOVE 14B.
# Bounded check on 3 task-diverse families: ARC (knowledge), GSM8K (reasoning), MBPP (code).
#
# ISOLATION: pool "redun_mistral24b" -> pairing_condition="redun_mistral24b" -> the frozen
# redundancy analysis (same_model_neff filters pairing_condition=="same_model") CANNOT see it,
# and it writes fresh *redun_mistral24b* files, leaving the k=3 escalate verdicts untouched.
# This is a DELIBERATE post-freeze run (user-authorised) to drop the "<=14B" scope on redundancy.
#
# RUN:  nohup bash scripts/run_mistral24b_redundancy.sh > mistral24b_redundancy.log 2>&1 &
# ANALYSE (offline, after):  $PY scripts/analyze_redun24b.py
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/alex/venvs/env/bin/python

K=50                 # 50 same-model repeats per item -> n_eff@50
N_ERR=50             # errors-only (redundancy fit uses ground_truth_is_error)
N_OK=0
POOL=redun_mistral24b

run () {  # $1=source  $2=benchmark(or "")  $3=raw-dir-label
  local dir="data/raw_cascade_runs/$3"
  if ls "$dir"/*"__${POOL}__${POOL}.jsonl" >/dev/null 2>&1; then
    echo "SKIP $3: ${POOL} already collected (delete the file to redo)"
    return
  fi
  echo "=== $(date +%H:%M:%S)  redun24b  $3  (k=$K n_err=$N_ERR) ==="
  if [ -n "$2" ]; then
    $PY -u run_cascade.py --source "$1" --benchmark "$2" --pool "$POOL" \
        --k "$K" --n-errors "$N_ERR" --n-non-errors "$N_OK" --skip-analyse
  else
    $PY -u run_cascade.py --source "$1" --pool "$POOL" \
        --k "$K" --n-errors "$N_ERR" --n-non-errors "$N_OK" --skip-analyse
  fi
}

run mcqa  arc   mcqa__arc__ctx_truncate
run gsm8k ""    gsm8k__gsm8k__ctx_truncate
run mbpp  ""    mbpp__mbpp__ctx_truncate

echo ""
echo "=== 24B k=50 redundancy verdicts collected. Analyse offline (no GPU):"
echo "    $PY scripts/analyze_redun24b.py"
