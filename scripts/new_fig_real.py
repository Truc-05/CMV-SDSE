"""new_fig_real.py -- three polished 1x3 figures, ALL numbers read from the frozen result CSVs.

    source /home/alex/venvs/env/bin/activate
    python scripts/new_fig_real.py            # you run this; reads results/tables/*.csv only

  figure_A_recovery       (a) residual recovery per family  (b) fraction recovered  (c) n_eff@50
  figure_B_generalization (a) cal->test utility  (b) held-out CMV vs GES/mRMR  (c) joint vs base across rho
  figure_D_joint_policy   (a) selected topology  (b) selected |S|  (c) joint gain over fixed-OR @rho=4

Sources: redundancy_invariant_7family, joint_topology_perfamily, joint_topology_controls,
rev_fusion_selection_heldout.  Outputs to figure4/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
OUT = ROOT / "figure4"; OUT.mkdir(exist_ok=True)

# --------------------------------------------------------------- refined modern palette --------
BLUE, BLUE_L = "#1d317e", "#dbe4ff"
ORANGE, ORANGE_L = "#763d17", "#ffe8cc"
TEAL, CORAL, RED = "#225c4a", "#ea9595", "#fa5252"
MUTE, GREY = "#ced4da", "#868e96"
INK, SUB, BASE, GRIDC = "#212529", "#495057", "#dee2e6", "#f1f3f5"
LILAC = "#e5dbff"

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 13.5,
    "axes.titlesize": 14.5, "axes.titleweight": "bold", "axes.labelsize": 13,
    "axes.edgecolor": BASE, "axes.linewidth": 1.0, "text.color": INK,
    "xtick.color": SUB, "ytick.color": SUB, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRIDC, "grid.linewidth": 1.0, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white",
    "axes.facecolor": "white", "legend.frameon": False, "legend.fontsize": 11,
})
FAM = ["ARC", "MMLU", "CSQA", "TruthfulQA", "GSM8K", "MBPP"]
N = 150


def _base(ax, title, xl="", yl="", grid="x"):
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(length=3, colors=SUB)
    ax.grid(axis="x", visible=(grid in ("x", "both")))
    ax.grid(axis="y", visible=(grid in ("y", "both")))
    ax.set_title(title, loc="left", pad=10, color=INK)
    if xl:
        ax.set_xlabel(xl)
    if yl:
        ax.set_ylabel(yl)


def _cellborders(ax, ncol, nrow):
    ax.set_xticks(np.arange(-.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nrow, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.4)
    ax.tick_params(which="minor", length=0)


def _save(fig, name, sup):
    fig.suptitle(sup, x=0.006, ha="left", y=1.0, fontsize=16, fontweight="bold", color=INK)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.84, bottom=0.14, wspace=0.30)
    for e in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{e}")
    plt.close(fig); print("wrote", OUT / f"{name}.png")


# =================================================================== FIGURE A ===================
def figure_A():
    d = pd.read_csv(T / "redundancy_invariant_7family.csv").set_index("family").reindex(FAM)
    miss = d.div_denom.astype(float).values                # errors the best solo misses
    rec = d.div_unique.astype(float).values                # residual recovered by the diversifier
    solo = N - miss
    still = miss - rec
    y = np.arange(len(FAM))[::-1]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    # (a) recovery flow, stacked
    ax[0].barh(y, solo, height=0.62, color=MUTE, edgecolor="white", linewidth=1.6, zorder=3)
    ax[0].barh(y, rec, left=solo, height=0.62, color=BLUE, edgecolor="white", linewidth=1.6, zorder=3)
    ax[0].barh(y, still, left=solo + rec, height=0.62, color=CORAL, edgecolor="white", linewidth=1.6, zorder=3)
    for yi, r, m in zip(y, rec, still):                    # one clean label per row: +recovered / missed
        ax[0].annotate(f"+{int(round(r))} recovered · {int(round(m))} left", (N, yi), (9, 0),
                       textcoords="offset points", va="center", color=SUB, fontsize=10.5)
    ax[0].set_yticks(y); ax[0].set_yticklabels(FAM); ax[0].set_xlim(0, N + 55)
    ax[0].legend(handles=[Line2D([], [], marker="s", ls="", color=MUTE, ms=11, label="best solo"),
                          Line2D([], [], marker="s", ls="", color=BLUE, ms=11, label="recovered by diversifier"),
                          Line2D([], [], marker="s", ls="", color=CORAL, ms=11, label="still missed")],
                 loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, columnspacing=1.1)
    _base(ax[0], "(a) Residual error recovery", f"error items (of {N})")
    # (b) fraction of residual recovered — lollipop
    frac = rec / miss
    ax[1].hlines(y, 0, frac, color=BASE, lw=3, zorder=2)
    ax[1].scatter(frac, y, s=150, color=TEAL, edgecolors="white", linewidths=1.4, zorder=4)
    for f_, yi in zip(frac, y):
        ax[1].annotate(f"{f_:.2f}", (f_, yi), (9, 0), textcoords="offset points", va="center",
                       fontsize=11.5, color=INK, fontweight="bold")
    ax[1].set_yticks(y); ax[1].set_yticklabels(FAM); ax[1].set_xlim(0, 1)
    _base(ax[1], "(b) Fraction of residual recovered", "recovered / residual")
    # (c) n_eff@50 lollipop + null band
    ne = d.sm_n_eff50.values
    ax[2].axvspan(1.06, 1.21, color=BLUE_L, zorder=0, label="null band [1.06, 1.21]")
    ax[2].hlines(y, 1.0, ne, color=BASE, lw=3, zorder=2)
    ax[2].scatter(ne, y, s=150, color=BLUE, edgecolors="white", linewidths=1.4, zorder=4)
    for xv, yi in zip(ne, y):
        ax[2].annotate(f"{xv:.2f}", (xv, yi), (9, 0), textcoords="offset points", va="center",
                       fontsize=11.5, color=INK, fontweight="bold")
    ax[2].set_yticks(y); ax[2].set_yticklabels(FAM); ax[2].set_xlim(0.95, 1.52)
    ax[2].legend(loc="lower right")
    _base(ax[2], "(c) Effective verifiers  $n_{\\mathrm{eff}}$@50", "$n_{\\mathrm{eff}}$@50")
    _save(fig, "figure_A_recovery", "Figure A.  Residual error recovery — what CMV acquisition buys")


# =================================================================== FIGURE B ===================
def figure_B():
    pf = pd.read_csv(T / "joint_topology_perfamily.csv")
    j1 = pf[(pf.rung == "joint") & (pf.rho == 1.0)].set_index("family").reindex(FAM)
    y = np.arange(len(FAM))[::-1]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    # (a) calibration -> frozen test (dumbbell, optimism gap)
    uc, ut = j1.U_cal.values, j1.U_test.values
    for yi, a, b in zip(y, uc, ut):
        ax[0].plot([a, b], [yi, yi], color=BASE, lw=3.2, zorder=2, solid_capstyle="round")
    ax[0].scatter(uc, y, s=120, color=MUTE, edgecolors="white", linewidths=1.3, zorder=4)
    ax[0].scatter(ut, y, s=150, color=BLUE, edgecolors="white", linewidths=1.3, zorder=5)
    ax[0].set_yticks(y); ax[0].set_yticklabels(FAM)
    ax[0].legend(handles=[Line2D([], [], marker="o", ls="", color=MUTE, ms=11, label="calibration"),
                          Line2D([], [], marker="o", ls="", color=BLUE, ms=11, label="frozen test")],
                 loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
    _base(ax[0], "(a) Calibration → frozen-test utility  (ρ=1)", "utility  $U$")
    # (b) held-out selectors — CMV highlighted
    h = pd.read_csv(T / "rev_fusion_selection_heldout.csv").set_index("policy")
    order = [p for p in ["Ensemble selection (GES)", "mRMR", "CMV-SDSE"] if p in h.index]
    short = {"Ensemble selection (GES)": "GES", "mRMR": "mRMR", "CMV-SDSE": "CMV-SDSE"}
    vals = [float(h.loc[p, "heldout_test_U"]) for p in order]
    cols = [BLUE if "CMV" in p else MUTE for p in order]
    ax[1].bar(range(len(order)), vals, width=0.58, color=cols, edgecolor="white", linewidth=1.6, zorder=3)
    for i, (v, p) in enumerate(zip(vals, order)):
        ax[1].annotate(f"{v:.3f}", (i, v), (0, 5), textcoords="offset points", ha="center",
                       fontsize=12, fontweight="bold", color=(BLUE if "CMV" in p else SUB))
    ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels([short[p] for p in order])
    ax[1].set_ylim(0, max(vals) * 1.18)
    _base(ax[1], "(b) Held-out utility vs selectors  (ρ=1)", yl="held-out test  $U$", grid="y")
    # (c) joint vs base across regime — joint emphasised, improvement shaded
    c = pd.read_csv(T / "joint_topology_controls.csv")
    x = np.arange(len(c)); xl = [("0" if r == 0 else f"{r:g}") for r in c.rho]
    ax[2].axhline(0, color=BASE, lw=1)
    ax[2].fill_between(x, c.U_fixedOR, c.U_joint, where=(c.U_joint >= c.U_fixedOR),
                       color=BLUE_L, zorder=1)
    ax[2].plot(x, c.U_fixedOR, "-o", color=MUTE, lw=2, ms=6, zorder=3, label="fixed-OR (base)")
    ax[2].plot(x, c.U_topoFP, "-s", color=ORANGE, lw=2, ms=6, zorder=3, label="topology, full pool")
    ax[2].plot(x, c.U_joint, "-o", color=BLUE, lw=3.4, ms=9, zorder=4, label="joint")
    ax[2].set_xticks(x); ax[2].set_xticklabels(xl)
    ax[2].legend(loc="lower left")
    _base(ax[2], "(c) Joint vs base across cost regime", "false-alarm / miss ratio  ρ", "utility  $U$", grid="y")
    _save(fig, "figure_B_generalization", "Figure B.  Calibration → frozen-test generalization")


# =================================================================== FIGURE D ===================
def figure_D():
    pf = pd.read_csv(T / "joint_topology_perfamily.csv")
    j = pf[pf.rung == "joint"]
    rhos = sorted(j.rho.unique())
    topo = j.pivot(index="family", columns="rho", values="g_mode").reindex(FAM)[rhos]
    size = j.pivot(index="family", columns="rho", values="setsize").reindex(FAM)[rhos]
    ut = j.pivot(index="family", columns="rho", values="U_test").reindex(FAM)
    fo = pf[pf.rung == "fixed_OR"].pivot(index="family", columns="rho", values="U_test").reindex(FAM)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    xl = [("0" if r == 0 else f"{r:g}") for r in rhos]; y = np.arange(len(FAM)); nc = len(rhos)
    # (a) selected topology
    code = topo.where(topo.isin(["OR", "AND"]), "other").replace({"OR": 0, "AND": 1, "other": 2}).astype(float).values
    ax[0].imshow(code, cmap=ListedColormap([BLUE_L, ORANGE, LILAC]), vmin=0, vmax=2, aspect="auto")
    for i in range(len(FAM)):
        for k in range(nc):
            g = str(topo.values[i, k])
            ax[0].text(k, i, g, ha="center", va="center", fontsize=10,
                       color="white" if g == "AND" else INK, fontweight="bold" if g == "AND" else "normal")
    ax[0].set_xticks(range(nc), xl); ax[0].set_yticks(y, FAM)
    _cellborders(ax[0], nc, len(FAM)); ax[0].grid(False, which="major")
    ax[0].set_title("(a) Selected fusion topology", loc="left", pad=10); ax[0].set_xlabel("ρ")
    ax[0].legend(handles=[Line2D([], [], marker="s", ls="", mfc=BLUE_L, mec=BASE, ms=13, label="OR"),
                          Line2D([], [], marker="s", ls="", color=ORANGE, ms=13, label="AND")],
                 loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=2)
    # (b) selected set size
    im = ax[1].imshow(size.values, cmap="Blues", aspect="auto", vmin=1, vmax=2)
    for i in range(len(FAM)):
        for k in range(nc):
            v = size.values[i, k]
            ax[1].text(k, i, f"{v:.1f}", ha="center", va="center", fontsize=10,
                       color="white" if v >= 1.6 else INK, fontweight="bold")
    ax[1].set_xticks(range(nc), xl); ax[1].set_yticks(y, [])
    _cellborders(ax[1], nc, len(FAM)); ax[1].grid(False, which="major")
    ax[1].set_title("(b) Selected set size  |S|", loc="left", pad=10); ax[1].set_xlabel("ρ")
    fig.colorbar(im, ax=ax[1], fraction=0.045, pad=0.03)
    # (c) joint gain over fixed-OR at rho=4, sorted
    gain = (ut[4.0] - fo[4.0])
    gain = gain.sort_values()
    yy = np.arange(len(gain))
    ax[2].barh(yy, gain.values, height=0.6, color=[BLUE if g > 1e-6 else MUTE for g in gain.values],
               edgecolor="white", linewidth=1.6, zorder=3)
    for g, yi in zip(gain.values, yy):
        ax[2].annotate(f"+{g:.3f}" if g > 1e-6 else "0", (max(g, 0), yi), (8, 0),
                       textcoords="offset points", va="center", fontsize=11.5, fontweight="bold",
                       color=(BLUE if g > 1e-6 else SUB))
    ax[2].set_yticks(yy); ax[2].set_yticklabels(gain.index); ax[2].axvline(0, color=BASE, lw=1)
    ax[2].set_xlim(-0.01, float(gain.max()) * 1.25 + 0.01)
    _base(ax[2], "(c) Joint gain over fixed-OR  (ρ=4)", "Δ held-out utility")
    _save(fig, "figure_D_joint_policy", "Figure D.  Joint policy — topology, set size, and payoff by cost regime")


if __name__ == "__main__":
    figure_A(); figure_B(); figure_D()
    print("\nThree polished 1x3 figures (real data) written to:", OUT)
