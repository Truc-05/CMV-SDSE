"""Cost--performance Pareto figure for budget-aware verifier fusion.

x = real compute per decision (median wall-clock seconds, log scale)
y = decision utility U = catch - rho*FA at rho=1

One point per acquisition policy (Same-model k=50, Random, Accuracy, Low-correlation,
CMV-SDSE), macro-averaged over the six confirmatory families, using the SAME primitives
as the paper tables (reconciles to same-model 30.40s/U0.233, CMV-SDSE 0.26s/U0.486).
A policy is Pareto-optimal if no other policy is both cheaper and higher-utility; the
non-dominated set and the region CMV-SDSE dominates are drawn. Writes results/tables/
rev_pareto.csv and figures/fig_pareto.{pdf,png}.
"""
from __future__ import annotations
import glob, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from theory.marginal_value import _prep
import reviewer_analyses as ra

OUT = "figures"; T = "results/tables"; RHO = 1.0
# house palette (matches scripts/appendix_figures.py)
BLUE, CRIMSON, GREEN, AMBER, PURPLE, GREY, LGREY = \
    "#2a6f97", "#b23a48", "#2e8b57", "#e08a1e", "#6a4c93", "#8a8f98", "#c9ccd1"
INK, MUTE, GRID = "#2b2b2b", "#6b7178", "#ececec"
plt.rcParams.update({"savefig.dpi": 170, "figure.dpi": 170,
    "figure.constrained_layout.use": True, "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12.5, "axes.labelsize": 10.5, "xtick.labelsize": 9.6,
    "ytick.labelsize": 9.6, "legend.fontsize": 9, "axes.linewidth": 0.9})


def median_latencies():
    lat = defaultdict(list)
    for slug in ra.FAM.values():
        for p in glob.glob(f"data/raw_cascade_runs/{slug}/*.jsonl"):
            for l in open(p):
                if not l.strip():
                    continue
                r = json.loads(l)
                if r.get("n_gates") != 50:
                    continue
                for g in r["gates"]:
                    if g.get("latency_seconds"):
                        lat[g["verifier_id"]].append(g["latency_seconds"])
    return {m: float(np.median(v)) for m, v in lat.items()}


def compute(rho=RHO):
    LAT = median_latencies()
    RNG = np.random.RandomState(0)
    rows = defaultdict(lambda: defaultdict(list))
    for fam, slug in ra.FAM.items():
        df = ra.load(slug); P = _prep(df, 0.5)
        Ce, Cf, cost, models = P["Ce"], P["Cf"], P["cost"], P["models"]
        solo = max(models, key=lambda m: Ce[m].mean())

        def rec(name, sel, calls, wall):
            c, f = ra.cascade(sel, Ce, Cf)
            rows[name]["U"].append(c - rho * f)
            rows[name]["calls"].append(calls)
            rows[name]["wall"].append(wall)

        rec("Same-model (k=50)", [solo], 50, 50 * LAT[solo])
        acc = ra.order_accuracy(models, Ce, Cf)[:3]
        lcq = ra.order_lowcorr(models, Ce, Cf)[:3]
        rec("Accuracy-ranked", acc, 3, sum(LAT[m] for m in acc))
        rec("Low-correlation", lcq, 3, sum(LAT[m] for m in lcq))
        # random: mean over permutations of a 3-verifier draw
        for _ in range(30):
            perm = list(RNG.permutation(models))[:3]
            rec("Random", perm, 3, sum(LAT[m] for m in perm))
        cmv = ra.greedy_normalized(Ce, Cf, cost)
        rec("CMV-SDSE", cmv, len(cmv), sum(LAT[m] for m in cmv))

    order = ["Same-model (k=50)", "Random", "Accuracy-ranked", "Low-correlation", "CMV-SDSE"]
    out = []
    for name in order:
        out.append(dict(policy=name,
                        wall_seconds=round(float(np.mean(rows[name]["wall"])), 2),
                        calls=round(float(np.mean(rows[name]["calls"])), 1),
                        U=round(float(np.mean(rows[name]["U"])), 3)))
    D = pd.DataFrame(out)
    # Pareto flag: not dominated by any other (cheaper-or-equal AND higher-or-equal U)
    dom = []
    for i, a in D.iterrows():
        dominated = any((b.wall_seconds <= a.wall_seconds and b.U >= a.U and j != i and
                         (b.wall_seconds < a.wall_seconds or b.U > a.U)) for j, b in D.iterrows())
        dom.append(not dominated)
    D["pareto"] = dom
    D.to_csv(f"{T}/rev_pareto.csv", index=False)
    print(D.to_string(index=False))
    return D


def plot(D):
    COL = {"Same-model (k=50)": CRIMSON, "Random": GREY, "Accuracy-ranked": AMBER,
           "Low-correlation": PURPLE, "CMV-SDSE": BLUE}
    MRK = {"Same-model (k=50)": "s", "Random": "^", "Accuracy-ranked": "D",
           "Low-correlation": "v", "CMV-SDSE": "o"}
    ORDER = ["Same-model (k=50)", "Random", "Accuracy-ranked", "Low-correlation", "CMV-SDSE"]
    # cumulative build: (a) repetition -> (b) + naive diversification -> (c) + CMV-SDSE.
    # Same axes across panels so the eye tracks one coordinate space as points accumulate.
    DIVERSIFY = {"Random", "Accuracy-ranked", "Low-correlation"}
    STAGES = [
        ("(a)", "repetition only",         {"Same-model (k=50)"},                          False),
        ("(b)", "+ naive diversification", {"Same-model (k=50)"} | DIVERSIFY,               False),
        ("(c)", "+ CMV-SDSE",              set(ORDER),                                      True),
    ]
    pw = float(D.loc[D.policy == "CMV-SDSE", "wall_seconds"].iloc[0])
    pu = float(D.loc[D.policy == "CMV-SDSE", "U"].iloc[0])

    f, axs = plt.subplots(1, 3, figsize=(13.8, 4.6), sharey=True, layout="constrained")

    for ax, (tag, sub, shown, full) in zip(axs, STAGES):
        ax.set_xscale("log")
        ax.set_xlim(0.16, 55); ax.set_ylim(0.15, 0.58)
        ax.set_xticks([0.25, 0.5, 1, 2, 5, 10, 30])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        if ax is axs[0]:
            ax.set_ylabel(r"decision utility  $U=\mathrm{catch}-\rho\,\mathrm{FA}$  ($\rho{=}1$)", color=INK)
        if full:  # dominance region + Pareto frontier revealed only in the last panel
            ax.axhspan(0.15, pu, color=BLUE, alpha=0.045, zorder=0)
            ax.axvline(pw, color=LGREY, lw=1.0, ls=":", zorder=1)
            ax.axhline(pu, color=LGREY, lw=1.0, ls=":", zorder=1)
            ax.plot([pw, pw, 60], [0.56, pu, pu], color=BLUE, lw=1.6, ls="--", zorder=2)
        for _, r in D.iterrows():
            if r.policy not in shown:
                continue
            s = 90 + 4.5 * r.calls
            ax.scatter(r.wall_seconds, r.U, s=s, color=COL[r.policy], marker=MRK[r.policy],
                       ec="white", lw=1.1, zorder=4)
        ax.set_title(f"{tag}  {sub}", color=INK, fontsize=11.5, pad=6)
        # one annotation per panel, carrying that stage's beat
        if tag == "(a)":
            ax.annotate("50 calls · 30 s\n$U$ only 0.233", (30.40, 0.233), (-10, 30),
                        textcoords="offset points", fontsize=8.6, color=CRIMSON, ha="right",
                        arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1.1))
        elif tag == "(b)":
            ax.annotate("3-verifier policies over-acquire:\nmore FA, lower $U$", (1.36, 0.20), (30, 44),
                        textcoords="offset points", fontsize=8.4, color=MUTE, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", color=LGREY, lw=1.0))
        else:
            ax.annotate("cheapest and\nhighest utility", (pw, pu), (16, 4), textcoords="offset points",
                        fontsize=8.8, color=BLUE, ha="left", va="bottom",
                        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))

    # shared legend across the bottom (all five policies + frontier)
    handles = [Line2D([], [], marker=MRK[p], color="white", markerfacecolor=COL[p],
                      markeredgecolor="white", markersize=10, ls="",
                      label=f"{p}  ($U${float(D.loc[D.policy == p, 'U'].iloc[0]):.3f}, "
                            f"{int(D.loc[D.policy == p, 'calls'].iloc[0])}×)")
               for p in ORDER]
    handles.append(Line2D([], [], color=BLUE, lw=1.6, ls="--", label="Pareto frontier (CMV-SDSE)"))
    lg = f.legend(handles=handles, loc="outside lower center", ncol=3, framealpha=0.96,
                  edgecolor=LGREY, fontsize=9, columnspacing=1.4, handletextpad=0.5)
    lg.get_frame().set_linewidth(0.8)

    f.suptitle("Cost--performance frontier: CMV-SDSE dominates every acquisition policy",
               color=INK, fontsize=13)
    axs[1].set_xlabel("compute per decision — median wall-clock (s, log scale)", color=INK, fontsize=10.5)
    for ext in ("pdf", "png"):
        f.savefig(f"{OUT}/fig_pareto.{ext}", facecolor="white")
    plt.close(f); print("wrote figures/fig_pareto.{pdf,png}")


if __name__ == "__main__":
    import matplotlib.ticker  # noqa
    plot(compute())
