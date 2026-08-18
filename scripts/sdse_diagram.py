"""Flagship figure: the Scale / Diversify / Stop / Escalate decision map.

Axes: x = complementary evidence available in the pool (blind-spot disjointness χ, ~ n_eff),
      y = required reliability (target).
The pool's achievable ceiling C(χ) rises with complementarity (Theorem 2: only genuinely new,
decorrelated evidence lifts the ceiling). The four regions:

  ESCALATE  target above the achievable ceiling — no in-pool cascade reaches it (Thm 2)
  DIVERSIFY target reachable AND complementary verifiers still pay (add a different source)
  SCALE     target reachable but only same-source repeats available and still informative
            (low self-correlation) — the narrow, usually-a-trap regime
  STOP      target already met by the current cascade (Theorem 4)

Real MAST pools are placed as anchors at their measured (complementarity, ceiling).
"""
from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theory.cascade_design import reliability_ceiling
from theory.betabinom_mixture import fit_mixture

C_ESC, C_DIV, C_SCALE, C_STOP = "#b23a48", "#e08a1e", "#8a6d3b", "#2a6f97"


def measured_pool(target, pool):
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
                if g.get("accepted") in (True, False):
                    data[r["item_id"]][g["verifier_id"]].append(g["accepted"] is True)
    items = [it for it in data if all(data[it].get(v) for v in pool)]
    A = np.array([[float(np.mean(data[it][v])) for v in pool] for it in items])
    ceil = reliability_ceiling(A)
    # complementarity proxy: how much the pool ceiling exceeds the best single verifier
    best_solo = float((1 - A.mean(axis=0)).max())
    chi = float(np.clip((ceil - best_solo) / max(1 - best_solo, 1e-6), 0, 1))
    return chi, ceil


def main():
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    x = np.linspace(0, 1, 400)
    # achievable ceiling rises with complementarity (schematic, monotone concave)
    base = 0.15
    ceiling = base + (0.99 - base) * (1 - (1 - x) ** 2.2)

    ax.fill_between(x, ceiling, 1.0, color=C_ESC, alpha=0.5)          # ESCALATE (above ceiling)
    ax.fill_between(x, 0, ceiling, where=(x >= 0.22), color=C_DIV, alpha=0.35)   # DIVERSIFY
    ax.fill_between(x, 0, ceiling, where=(x < 0.22), color=C_SCALE, alpha=0.35)  # SCALE (narrow)
    # STOP band: low target (already met) — a hatched strip near the bottom
    ax.fill_between(x, 0, 0.12, color=C_STOP, alpha=0.30)
    ax.plot(x, ceiling, color="k", lw=1.6)
    ax.text(0.5, 0.985, "achievable ceiling  C(χ)  — Theorem 2", ha="center", va="top", fontsize=8)

    ax.text(0.55, 0.93, "ESCALATE", color="white", fontsize=13, weight="bold", ha="center")
    ax.text(0.60, 0.45, "DIVERSIFY", color="#7a4a10", fontsize=13, weight="bold", ha="center")
    ax.text(0.10, 0.42, "SCALE\n(trap)", color="#4a3a1a", fontsize=10, weight="bold", ha="center")
    ax.text(0.5, 0.06, "STOP  (target already met — Theorem 4)", color="white",
            fontsize=9, weight="bold", ha="center")

    # real MAST pools as anchors
    anchors = [
        ("same_model {qwen}", ["qwen7b"]),
        ("same_family {llama}", ["llama32_3b", "llama31"]),
        ("cross_family {q,l,m}", ["qwen7b", "llama31", "mistral7b"]),
        ("full pool (5)", ["qwen7b", "qwen14b", "llama31", "llama32_3b", "mistral7b"]),
    ]
    for name, pool in anchors:
        try:
            chi, ceil = measured_pool("3.3", pool)
        except Exception:
            continue
        ax.scatter([chi], [ceil], s=60, color="k", zorder=5)
        ax.annotate(f"{name}\nceiling={ceil:.2f}", (chi, ceil), textcoords="offset points",
                    xytext=(8, -2), fontsize=7.5)

    ax.set_xlabel("complementary evidence available  χ  (blind-spot disjointness ≈ n_eff)")
    ax.set_ylabel("required reliability (target)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Budget-aware fusion of correlated verifiers: Scale / Diversify / Stop / Escalate")
    ax.legend(handles=[Patch(color=C_ESC, alpha=.6, label="ESCALATE (out-of-pool source)"),
                       Patch(color=C_DIV, alpha=.5, label="DIVERSIFY (add complementary verifier)"),
                       Patch(color=C_SCALE, alpha=.5, label="SCALE (repeat — usually a trap)"),
                       Patch(color=C_STOP, alpha=.5, label="STOP (target met)")],
              fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig("results/figures/fig7_sdse_decision.png", dpi=150)
    plt.close(fig)
    print("wrote results/figures/fig7_sdse_decision.png")
    for name, pool in anchors:
        try:
            chi, ceil = measured_pool("3.3", pool)
            print(f"  {name:22s} χ≈{chi:.2f}  ceiling={ceil:.3f}")
        except Exception as e:
            print(f"  {name}: {e}")


if __name__ == "__main__":
    main()
