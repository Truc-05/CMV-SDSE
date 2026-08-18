"""Build the (item x verifier) accept matrix from the real runs and compare cascade designs.

    python scripts/cascade_design_demo.py

Writes:
  results/tables/cascade_design.csv   -- solo catch, the two orders, reliability at each depth
  results/figures/fig5_cascade_design.png
Also runs the probe-driven version (R=2 verdicts) to show the algorithm needs almost no data.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theory.cascade_design import (
    compare_designs, cascade_reliability, marginal_greedy, accuracy_greedy,
    estimate_accept_matrix, optimal_per_k,
)

VERIFIERS = ["qwen7b", "llama31", "llama32_3b", "mistral7b"]
PRETTY = {"qwen7b": "Qwen2.5-7B", "llama31": "Llama3.1-8B",
          "llama32_3b": "Llama3.2-3B", "mistral7b": "Mistral-7B"}
BLUE, ORANGE, GREEN = "#2a6f97", "#e08a1e", "#2e8b57"


def build_matrix(target="3.3"):
    slug = f"mast__{target.replace('.', '_')}__ctx_truncate"
    rows = []
    for p in glob.glob(f"data/raw_cascade_runs/{slug}/*.jsonl"):
        for line in open(p):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # per item -> per verifier -> list of accept booleans (error items, k=50)
    data: dict = {}
    for r in rows:
        if not r["ground_truth_is_error"] or r["n_gates"] != 50:
            continue
        it = r["item_id"]
        d = data.setdefault(it, {v: [] for v in VERIFIERS})
        for g in r["gates"]:
            v = g.get("verifier_id")
            if v in VERIFIERS and g.get("accepted") in (True, False):
                d[v].append(g["accepted"] is True)
    items = [it for it, d in data.items() if all(len(d[v]) > 0 for v in VERIFIERS)]
    A = np.array([[float(np.mean(data[it][v])) for v in VERIFIERS] for it in items])
    verdicts = [[data[it][v] for v in VERIFIERS] for it in items]  # verdicts[i][col] = list
    return A, verdicts, items


def main():
    A, verdicts, items = build_matrix()
    n = len(items)
    cmp = compare_designs(A, VERIFIERS)

    opt_perk = optimal_per_k(A)

    # probe-driven order from only R=2 verdicts per (item, verifier)
    A_probe = estimate_accept_matrix(verdicts, n, len(VERIFIERS), R=2)
    probe_order = marginal_greedy(A_probe)

    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # table
    d = cmp.to_dict()
    rows = []
    for step in range(len(VERIFIERS)):
        rows.append({
            "step": step + 1,
            "accuracy_greedy_pick": d["accuracy_order"][step],
            "reliability_accuracy": round(d["reliability_accuracy"][step], 4),
            "marginal_greedy_pick": d["marginal_order"][step],
            "reliability_marginal": round(d["reliability_marginal"][step], 4),
            "optimal_reliability": round(opt_perk[step], 4),
        })
    tbl = pd.DataFrame(rows)
    tbl.to_csv("results/tables/cascade_design.csv", index=False)

    print(f"n items = {n}")
    print("solo catch:", {k: round(v, 3) for k, v in d["solo_catch"].items()})
    print("accuracy-greedy order:", [PRETTY[v] for v in d["accuracy_order"]])
    print("marginal-greedy order:", [PRETTY[v] for v in d["marginal_order"]])
    print("probe (R=2) order    :", [PRETTY[VERIFIERS[j]] for j in probe_order])
    print("ceiling (blind-spot wall):", round(cmp.ceiling, 4))
    print(tbl.to_string(index=False))

    # figure
    ks = np.arange(1, len(VERIFIERS) + 1)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(ks, opt_perk, "-", color="#444", lw=1.2, label="optimal (best k-subset)")
    ax.plot(ks, d["reliability_accuracy"], "s--", color=ORANGE, lw=1.8,
            label="accuracy-greedy (leaderboard order)")
    ax.plot(ks, d["reliability_marginal"], "o-", color=BLUE, lw=2.0,
            label="marginal-greedy (ours, submodular)")
    ax.plot(ks, cascade_reliability(A, probe_order), ":", color=GREEN, lw=1.6,
            label="ours, driven by a 2-verdict probe")
    ax.axhline(cmp.ceiling, color="#b23a48", ls=":", lw=1.2)
    ax.text(len(VERIFIERS), cmp.ceiling + 0.005, "blind-spot ceiling", ha="right",
            va="bottom", fontsize=7, color="#b23a48")
    # annotate the flip step
    for step in range(len(VERIFIERS)):
        if d["accuracy_order"][step] != d["marginal_order"][step]:
            ax.annotate(f"flip: ours picks {PRETTY[d['marginal_order'][step]]}\n"
                        f"(worst solo) not {PRETTY[d['accuracy_order'][step]]}",
                        xy=(step + 1, d["reliability_marginal"][step]),
                        xytext=(step + 1 - 1.4, d["reliability_marginal"][step] - 0.12),
                        fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))
            break
    ax.set_xlabel("cascade depth k (verifiers called)")
    ax.set_ylabel("reliability  P(error caught)")
    ax.set_xticks(ks)
    ax.set_title("Order verifiers by marginal value, not accuracy")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig("results/figures/fig5_cascade_design.png", dpi=150)
    plt.close(fig)
    print("\nwrote results/tables/cascade_design.csv and results/figures/fig5_cascade_design.png")


if __name__ == "__main__":
    main()
