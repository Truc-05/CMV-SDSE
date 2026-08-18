"""24B same-model redundancy across seven families (k=50) as a 3-panel composite.

Reads results/tables/redun_mistral24b_neff.csv (written by scripts/analyze_redun24b.py)
and draws one row of three horizontal-lollipop panels, matching the house style of
fig_sevenfamily but for the out-of-pool 24B checkpoint (Mistral-Small-24B):
  (a) n_eff@50  with 95% CI, against the frozen <=14B null band [1.06,1.21]
  (b) intra-item correlation rho_v  with 95% CI
  (c) same-model miss rate
Writes figures/fig_redun24b.{pdf,png}. Reads only the isolated 24B table, so the
frozen <=14B artifacts are untouched.
"""
from __future__ import annotations
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
T, OUT = ROOT / "results/tables", ROOT / "figures"

# house palette (matches scripts/make_figures.py)
BLUE, CRIMSON, GREEN, AMBER = "#2a6f97", "#b23a48", "#2e8b57", "#e08a1e"
GREY, LGREY = "#8a8f98", "#c9ccd1"
INK, MUTE, GRID = "#2b2b2b", "#6b7178", "#ececec"
plt.rcParams.update({"savefig.dpi": 170, "figure.dpi": 170, "font.family": "DejaVu Sans",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 10.2, "xtick.labelsize": 9.4,
    "ytick.labelsize": 9.8, "legend.fontsize": 8.6, "axes.linewidth": 0.9})

FAM7 = ["MAST-3.3", "ARC", "MMLU", "CSQA", "TruthfulQA", "GSM8K", "MBPP"]
BAND = (1.06, 1.21)  # frozen <=14B same-model n_eff@50 band


def _ci(s):  # "(1.864, 2.86)" -> (1.864, 2.86)
    return ast.literal_eval(s) if isinstance(s, str) else (np.nan, np.nan)


def lollipop(ax, vals, cis, labels, x0, xlim, title, xlabel, dotcol=BLUE, band=None):
    y = np.arange(len(labels))
    if band:
        ax.axvspan(band[0], band[1], color=BLUE, alpha=0.10, zorder=0,
                   label=f"≤14B null band {list(band)}")
    ax.axvline(x0, color=MUTE, ls=":", lw=1, zorder=1)
    ax.hlines(y, x0, vals, color=LGREY, lw=2.4, zorder=2)
    if cis is not None:
        lo = np.array([v - c[0] for v, c in zip(vals, cis)])
        hi = np.array([c[1] - v for v, c in zip(vals, cis)])
        ax.errorbar(vals, y, xerr=[lo, hi], fmt="none", ecolor=dotcol,
                    elinewidth=1.3, capsize=3, zorder=3)
    ax.scatter(vals, y, s=92, color=dotcol, zorder=4)
    for xv, yv in zip(vals, y):
        ax.annotate(f"{xv:.2f}", (xv, yv), (0, 10), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(*xlim); ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_title(title, color=INK, pad=6)
    ax.set_xlabel(xlabel, color=INK)
    ax.grid(True, axis="x", color=GRID, lw=0.8, zorder=0)
    if band:
        lg = ax.legend(loc="lower right", framealpha=0.95, edgecolor=LGREY)
        lg.get_frame().set_linewidth(0.8)


def main():
    d = pd.read_csv(T / "redun_mistral24b_neff.csv").set_index("family").reindex(FAM7).reset_index()
    d = d.iloc[::-1].reset_index(drop=True)  # MAST-3.3 on top, like fig_seven
    fams = d.family.tolist()
    neff, neff_ci = d.n_eff50.tolist(), [_ci(s) for s in d.n_eff50_ci]
    rho, rho_ci = d.rho_v.tolist(), [_ci(s) for s in d.rho_v_ci]
    miss = d.miss.tolist()

    f, axs = plt.subplots(1, 3, figsize=(13.8, 4.6), sharey=True, layout="constrained")
    lollipop(axs[0], neff, neff_ci, fams, 1.0, (0.95, 3.15),
             "(a)  same-model $n_{eff}$@50 (24B)",
             "effective verifiers  $n_{eff}$@50   (independent = 50)", band=BAND)
    lollipop(axs[1], rho, rho_ci, fams, 0.0, (0, 1.02),
             "(b)  intra-item correlation ρ_v", "correlation  ρ_v")
    lollipop(axs[2], miss, None, fams, 0.0, (0, 0.72),
             "(c)  same-model miss rate", "miss rate", dotcol=AMBER)

    f.suptitle("24B (Mistral-Small-24B) same-model redundancy across seven families ($k{=}50$)",
               color=INK, fontsize=13)
    for ext in ("pdf", "png"):
        f.savefig(OUT / f"fig_redun24b.{ext}", facecolor="white")
    plt.close(f)
    print("wrote figures/fig_redun24b.{pdf,png}")
    print(d[["family", "n_eff50", "rho_v", "miss"]].to_string(index=False))


if __name__ == "__main__":
    main()
