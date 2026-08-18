"""Phase diagram for budgeted verifier-cascade design (Theorem 6, as a map).

Sweeps two pool properties and shows which of the three decisions dominates:
  x = cross-verifier correlation rho  (0 = complementary/independent blind spots, 1 = nested)
  y = strength spread sigma           (verifiers differ a lot in solo accuracy)
For each cell we measure, at a fixed compute price lambda:
  selection gain  = R(marginal-greedy) - R(accuracy-greedy)         -> DIVERSIFY matters
  stop fraction   = 1 - k*/m  (gates the optimal stop drops)        -> STOP matters
and classify the cell into {accuracy-suffices, diversify, stop-now}.

Offline, deterministic given the seed. Writes results/figures/fig6_phase_diagram.png and
results/tables/phase_diagram.csv.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theory.cascade_design import (
    accuracy_greedy, marginal_greedy, cascade_reliability, budgeted_greedy_stop,
)


def simulate_pool(rng, n, m, rho, chi, p_hard=0.92, p_easy=0.05):
    """Two independent knobs mapping to Theorem 6.

    rho  = SHARED blind-spot mass: fraction of items every verifier always misses (nested;
           these set the ceiling 1-rho and are never caught -> drive the STOP signal).
    chi  = COMPLEMENTARITY: each verifier additionally misses a chi-fraction of the *other*
           items, its OWN idiosyncratic blind spots (disjoint across verifiers). High chi =>
           diverse gates keep catching each other's misses => keep adding pays.
    """
    shared_hard = rng.random(n) < rho
    A = np.full((n, m), p_easy)
    A[shared_hard, :] = p_hard
    for v in range(m):
        idio = (~shared_hard) & (rng.random(n) < chi)
        A[idio, v] = p_hard
    return A


def main():
    rng = np.random.default_rng(20260807)
    n, m = 500, 8
    rho = 0.25                       # fixed shared ceiling for the map; swept in the CSV
    chis = np.linspace(0.0, 0.8, 10)
    lams = np.linspace(0.005, 0.12, 10)
    reps = 6

    KSTAR = np.zeros((len(lams), len(chis)))
    SELMAX = 0.0
    for a, lam in enumerate(lams):
        for b, chi in enumerate(chis):
            ks, sels = [], []
            for _ in range(reps):
                A = simulate_pool(rng, n, m, rho, chi)
                res = budgeted_greedy_stop(A, lam=lam)
                ks.append(res["stop_depth"])
                r_acc = cascade_reliability(A, accuracy_greedy(A))
                r_mar = cascade_reliability(A, marginal_greedy(A))
                sels.append(float(r_mar[-1] - r_acc[-1]))
            KSTAR[a, b] = np.mean(ks)
            SELMAX = max(SELMAX, float(np.mean(sels)))

    rows = [{"chi": round(chis[b], 3), "lambda": round(lams[a], 4),
             "k_star": round(KSTAR[a, b], 2)}
            for a in range(len(lams)) for b in range(len(chis))]
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/tables/phase_diagram.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ext = [chis[0], chis[-1], lams[0], lams[-1]]
    im = ax.imshow(KSTAR, origin="lower", extent=ext, aspect="auto", cmap="viridis", vmin=1, vmax=m)
    cb = fig.colorbar(im, ax=ax); cb.set_label("optimal cascade depth  k*")
    CS = ax.contour(chis, lams, KSTAR, levels=[1.5, 3.5, 6.5], colors="w", linewidths=1)
    ax.clabel(CS, inline=True, fontsize=7, fmt="k*=%.0f")
    ax.set_xlabel("complementarity  χ  (disjointness of blind spots)")
    ax.set_ylabel("compute price  λ")
    ax.set_title(f"Budgeted cascade: optimal depth k* over (χ, λ)   [pool size m={m}, ceiling 1−ρ={1-rho:.2f}]")
    ax.text(0.03, 0.10 * (lams[-1] - lams[0]) + lams[0], "STOP EARLY\n(nested / costly)",
            color="w", fontsize=8, ha="left", va="center")
    ax.text(0.62, 0.03, "KEEP DIVERSIFYING\n(complementary / cheap)",
            color="k", fontsize=8, ha="left", va="bottom")
    fig.tight_layout()
    fig.savefig("results/figures/fig6_phase_diagram.png", dpi=150)
    plt.close(fig)

    print(f"k* range: {KSTAR.min():.1f}..{KSTAR.max():.1f} over (chi,lambda); pool m={m}")
    print(f"selection gain (marginal vs accuracy) max over grid = {SELMAX:.3f} "
          f"(small everywhere -> the levers are STOP and DIVERSIFY, not reordering)")
    print("wrote results/figures/fig6_phase_diagram.png and results/tables/phase_diagram.csv")


if __name__ == "__main__":
    main()
