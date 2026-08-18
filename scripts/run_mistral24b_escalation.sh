#!/usr/bin/env bash
# =============================================================================
# Collect Mistral-Small-24B (local 24B open verifier) verdicts on the SAME items
# as the six confirmatory families, at low --k. Solo-collect one verdict per item; the
# ESCALATE strategies are composed OFFLINE afterwards. Local GPU only, no API key.
# Lifts the pool past the 14B ceiling with a public param count.
#
# PREP:  ollama pull mistral-small3.1:24b     # ~14GB 4-bit; on 2x16GB Ollama splits it
# RUN:
#   nohup bash scripts/run_mistral24b_escalation.sh > mistral24b_escalation.log 2>&1 &
#   tail -f mistral24b_escalation.log
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/alex/venvs/env/bin/python

K=3                             # verdicts per item (majority); K=1 = cheapest/fastest
N_ERR=150
N_OK=75
POOL=same_model_mistral24b      # Mistral-24B solo (defined in configs/models.yaml)

# SAFETY: frozen analyses filter n_gates==50, so a low-k run is invisible to them. Refuse k=50.
if [ "$K" = "50" ]; then
  echo "ERROR: K=50 would contaminate the n_gates==50 analyses. Use K<=10 (default 3)." >&2
  exit 1
fi

run () {  # $1=source  $2=benchmark(or "")  $3=raw-dir-label
  local dir="data/raw_cascade_runs/$3"
  if ls "$dir"/*"__${POOL}__${POOL}.jsonl" >/dev/null 2>&1; then
    echo "SKIP $3: ${POOL} already collected (delete the file to redo — gate_runner appends)"
    return
  fi
  echo "=== $(date +%H:%M:%S)  mistral24b  $3  (k=$K n_err=$N_ERR n_ok=$N_OK) ==="
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
echo "=== Mistral-24B verdicts collected. Compose ESCALATE strategies offline (no GPU):"
echo "    $PY scripts/analyze_escalation.py mistral_small_24b   # -> rev_escalate_mistral_small_24b.csv"
