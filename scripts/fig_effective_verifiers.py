"""fig8: 50 verifier calls buy only 1-5 effective independent verifiers (across 3 failure modes)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

df = pd.read_csv("results/tables/main_results_3mode.csv")
modes = ["FM-3.1", "FM-3.2", "FM-3.3"]
SM = "#b23a48"; CF = "#2a6f97"

fig, ax = plt.subplots(figsize=(7.6, 4.6))
x = np.arange(len(modes)); w = 0.36
sm = [float(df[(df["mode"] == m) & (df["pool"] == "same_model")]["n_eff@50"].iloc[0]) for m in modes]
cf = [float(df[(df["mode"] == m) & (df["pool"] == "cross_family_3")]["n_eff@50"].iloc[0]) for m in modes]

ax.axhline(50, color="k", ls="--", lw=1.2)
ax.text(len(modes) - 0.5, 50, "nominal: 50 calls", ha="right", va="bottom", fontsize=8)
b1 = ax.bar(x - w/2, sm, w, color=SM, label="same-model (repeat one verifier)")
b2 = ax.bar(x + w/2, cf, w, color=CF, label="cross-family (diverse verifiers)")
for b, vals in [(b1, sm), (b2, cf)]:
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width()/2, v + 0.6, f"{v:.1f}",
                ha="center", fontsize=8)
ax.set_yscale("log")
ax.set_ylim(0.8, 70)
ax.set_yticks([1, 2, 5, 10, 20, 50]); ax.set_yticklabels([1, 2, 5, 10, 20, 50])
ax.set_xticks(x); ax.set_xticklabels(modes)
ax.set_ylabel("effective independent verifiers  $n_{eff}$  (log scale)")
ax.set_title("50 verifier calls ≠ 50 independent verifiers\n"
             "same-model repetition ≈ 1–2 effective; diversity recovers 2–5 — never 50",
             fontsize=10)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("results/figures/fig8_effective_verifiers.png", dpi=150)
plt.close(fig)
print("wrote results/figures/fig8_effective_verifiers.png")
print("same_model n_eff:", dict(zip(modes, [round(v, 2) for v in sm])))
print("cross_family n_eff:", dict(zip(modes, [round(v, 2) for v in cf])))
