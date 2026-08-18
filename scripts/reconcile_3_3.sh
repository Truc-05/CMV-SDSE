#!/usr/bin/env bash
# (c) Reconcile FM-3.3 population: regenerate both tables on canonical n_gates==50.
# CPU-only (no GPU / no ollama) — just refits cached jsonl.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== (1) theory report @ n_gates==50 ==="
/home/alex/venvs/env/bin/python -u run_cascade.py --analyse-only --source mast --mast-target 3.3 --k-gates 50

echo "=== (2) main table (freeze) ==="
/home/alex/venvs/env/bin/python -u scripts/freeze_main_table.py

echo "=== (3) self-check ==="
/home/alex/venvs/env/bin/python -u scripts/_reconcile_check.py
