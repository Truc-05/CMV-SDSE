#!/usr/bin/env bash
# End-to-end MBPP family: harvest labelled solutions (out-of-pool gemma2:9b) NOW — safe to run
# concurrently with the GSM8K cascade because gemma2 lands on the free GPU and is in no verifier
# pool — then, once the GSM8K confirmatory frees the verifier GPUs, run the MBPP confirmatory.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/alex/venvs/env/bin/python

echo "=== $(date +%H:%M:%S)  [1/3] harvest MBPP solutions (gemma2:9b, GPU0) ==="
$PY scripts/gen_mbpp_solutions.py

echo "=== $(date +%H:%M:%S)  [2/3] wait for GSM8K confirmatory to free the verifier GPUs ==="
GPID=$(cat gsm8k_confirmatory.pid 2>/dev/null || echo "")
if [ -n "$GPID" ]; then
  while kill -0 "$GPID" 2>/dev/null; do sleep 30; done
fi
echo "    GSM8K done; proceeding."

echo "=== $(date +%H:%M:%S)  [3/3] MBPP confirmatory cascade ==="
bash scripts/run_mbpp_confirmatory.sh

echo "=== $(date +%H:%M:%S)  MBPP FULL DONE.  Analyse: $PY scripts/pilot_mbpp.py ==="
