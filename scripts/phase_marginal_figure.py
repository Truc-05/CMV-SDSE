"""Render the controller's SCALE / DIVERSIFY / STOP phase diagram.

Axes: compute price lambda (x) vs false-alarm/miss cost ratio (y); each cell is the first
post-S0 action of the greedy under V = Delta_catch - (fa/miss)*Delta_fa - lambda*cost.
Read-only over the FM-3.3 truncate raw runs.
"""
import glob
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cascade.paths import raw_run_dir
from theory.marginal_value import phase_diagram

ORDER = {"DIVERSIFY": 0, "SCALE": 1, "STOP": 2}
COLORS = ["#2f7d4f", "#e0a52e", "#8a8f98"]  # green / amber / grey


def load(mode="3.3"):
    recs = [json.loads(l) for p in glob.glob(f"{raw_run_dir(mode, 'truncate')}/*.jsonl")
            for l in open(p) if l.strip()]
    df = pd.DataFrame(recs)
    return df[df["n_gates"] == 50]


def main():
    ph = phase_diagram(load())
    for mode, g in ph.groupby("failure_mode_id"):
        piv = g.pivot(index="cost_ratio_fa_over_miss", columns="lambda_cost",
                      values="first_action")
        Z = np.vectorize(lambda a: ORDER.get(a, 2))(piv.values)
        fig, ax = plt.subplots(figsize=(6.6, 4.7))
        ax.imshow(Z, cmap=ListedColormap(COLORS), vmin=0, vmax=2, aspect="auto", origin="lower")
        ax.set_xticks(range(len(piv.columns)), piv.columns)
        ax.set_yticks(range(len(piv.index)), piv.index)
        ax.set_xlabel("compute price  λ")
        ax.set_ylabel("false-alarm / miss cost ratio")
        ax.set_title(f"Verifier-controller phase diagram — {mode}\n"
                     r"$V=\Delta_{catch}-(fa/miss)\cdot\Delta_{fa}-\lambda\,C$")
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                ax.text(j, i, piv.values[i, j][0], ha="center", va="center",
                        color="white", fontsize=11, fontweight="bold")
        ax.legend(handles=[Patch(facecolor=COLORS[ORDER[k]], label=k) for k in ORDER],
                  loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.tight_layout()
        out = f"results/figures/fig_marginal_phase_{mode.replace('-', '_')}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("wrote", out)


if __name__ == "__main__":
    main()
