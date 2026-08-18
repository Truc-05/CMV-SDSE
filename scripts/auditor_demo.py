"""Run the Cascade Auditor on the real MAST verifier pools."""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theory.auditor import audit_pool, format_report


def load_verdicts(target="3.3"):
    slug = f"mast__{target.replace('.', '_')}__ctx_truncate"
    data = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(f"data/raw_cascade_runs/{slug}/*.jsonl"):
        for line in open(p):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r["ground_truth_is_error"] or r["n_gates"] != 50:
                continue
            for g in r["gates"]:
                v = g.get("verifier_id")
                if g.get("accepted") in (True, False):
                    data[r["item_id"]][v].append(g["accepted"] is True)
    return data


def main():
    data = load_verdicts()
    pool = ["qwen7b", "qwen14b", "llama31", "llama32_3b", "mistral7b"]
    # per-call cost proportional to model size (bigger = pricier)
    costs = {"qwen7b": 1.0, "qwen14b": 2.0, "llama31": 1.1, "llama32_3b": 0.4, "mistral7b": 1.0}

    print("\n### Full pool, target reliability 0.95 (unreachable -> should ESCALATE)")
    r = audit_pool(data, pool, costs=[costs[v] for v in pool], lam=0.03, target_reliability=0.95)
    print(format_report(r))

    print("\n### Full pool, target reliability 0.80 (reachable -> STOP/DIVERSIFY)")
    r2 = audit_pool(data, pool, costs=[costs[v] for v in pool], lam=0.03, target_reliability=0.80)
    print(format_report(r2))

    print("\n### Same-model pool {qwen7b} only — the SCALE trap")
    r3 = audit_pool(data, ["qwen7b"], lam=0.03, target_reliability=0.80)
    print(format_report(r3))


if __name__ == "__main__":
    main()
