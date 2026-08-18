"""Extra appendix figures for the table-only reviewer analyses (from rev*/rev3* CSVs).
Single files (no combining). Muted palette, consistent style.
  fig_component  · fig_lomo · fig_probeutil · fig_bootstrap · fig_latency · fig_permodel
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

T, OUT = "results/tables", "figures"
os.makedirs(OUT, exist_ok=True)
BLUE, CRIMSON, GREEN, AMBER, PURPLE, GREY, LGREY = "#2a6f97", "#b23a48", "#2e8b57", "#e08a1e", "#6a4c93", "#8a8f98", "#c9ccd1"
INK, MUTE, GRID = "#2b2b2b", "#6b7178", "#ececec"
plt.rcParams.update({"savefig.dpi": 170, "figure.dpi": 170, "figure.constrained_layout.use": True,
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.6, "ytick.labelsize": 9.6, "legend.fontsize": 9, "axes.linewidth": 0.9,
    "figure.facecolor": "white", "axes.facecolor": "white"})


def style(ax, xl="", yl="", ti="", grid="y", loc=None):
    ax.spines[["top", "right"]].set_visible(False); ax.spines[["left", "bottom"]].set_color(LGREY)
    ax.tick_params(colors=MUTE, length=3)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color(INK)
    if grid in ("y", "both"):
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    if grid in ("x", "both"):
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    if xl:
        ax.set_xlabel(xl, color=INK)
    if yl:
        ax.set_ylabel(yl, color=INK)
    if ti:
        ax.set_title(ti, color=INK, pad=9)
    if loc:
        lg = ax.legend(loc=loc, framealpha=0.95, edgecolor=LGREY); lg.get_frame().set_linewidth(0.8)


def save(f, name):
    f.savefig(f"{OUT}/{name}.png", facecolor="white"); plt.close(f); print("wrote", name)


def fig_component():
    d = pd.read_csv(f"{T}/rev_component_ablation.csv")
    order = ["full CMV", "no false-alarm term", "no compute term", "catch-only (no FA, no cost)", "no significance gate"]
    d = d.set_index("variant").reindex(order).reset_index()
    short = ["full\nCMV", "no FA\nterm", "no cost\nterm", "catch-\nonly", "no\ngate"]
    x = np.arange(len(d)); w = 0.26
    f, ax = plt.subplots(figsize=(8.2, 4.5))
    for off, col, key, lab in [(-w, GREEN, "catch", "catch"), (0, CRIMSON, "fa", "false alarm"), (w, BLUE, "U", "utility U")]:
        bars = ax.bar(x + off, d[key], w, color=col, ec="white", lw=0.6, label=lab, zorder=3)
        for b, v in zip(bars, d[key]):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, b.get_height()), (0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(short); ax.set_ylim(0, 1.05)
    style(ax, yl="rate / utility (ρ=1)", ti="Value-function component ablation", loc="upper right")
    save(f, "fig_component")


def fig_lomo():
    d = pd.read_csv(f"{T}/rev3_lomo.csv")
    d = d[d.removed != "(full pool)"]; full = 0.486
    d = d.sort_values("cmv_U"); y = np.arange(len(d))
    f, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.axvline(full, color=MUTE, ls="--", lw=1.2, label="full pool (0.49)")
    cols = [CRIMSON if u < full - 1e-6 else BLUE for u in d.cmv_U]
    ax.hlines(y, d.cmv_U, full, color=LGREY, lw=2.2, zorder=2)
    ax.scatter(d.cmv_U, y, s=110, color=cols, zorder=4)
    for u, yi in zip(d.cmv_U, y):
        ax.annotate(f"{u:.3f}", (u, yi), (-9, 0), textcoords="offset points", ha="right", va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(d.removed); ax.set_xlim(0.30, 0.52)
    style(ax, xl="CMV-SDSE utility with this verifier removed (ρ=1)", ti="Leave-one-model-out", grid="x", loc="upper left")
    save(f, "fig_lomo")


def fig_probeutil():
    d = pd.read_csv(f"{T}/rev3_probe_utility.csv")
    f, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.axhline(d.U.iloc[-1], color=GREY, ls=":", lw=1.6, label="full data (R=50)")
    ax.plot(range(len(d)), d.U, "-o", color=BLUE, lw=2.6, ms=8, zorder=4, label="R-round probe")
    ax.set_xticks(range(len(d))); ax.set_xticklabels(d.R)
    ax.set_ylim(0.40, 0.55)
    ax.annotate("a 1–2 round probe\nalready matches full data", (1, d.U.iloc[1]), (30, -34),
                textcoords="offset points", fontsize=9, color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
    style(ax, xl="probe rounds  R", yl="controller utility  U (ρ=1)", ti="Probe cost vs final utility", loc="lower right")
    save(f, "fig_probeutil")


def fig_bootstrap():
    d = pd.read_csv(f"{T}/rev_bootstrap.csv")
    d = d.iloc[::-1]; y = np.arange(len(d))
    f, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.axvspan(-0.05, 0, color=GREY, alpha=0.10, zorder=0); ax.axvline(0, color=CRIMSON, ls="--", lw=1.3)
    ax.errorbar(d.mean_dU, y, xerr=[d.mean_dU - d.ci_low, d.ci_high - d.mean_dU], fmt="o",
                color=BLUE, ecolor=LGREY, elinewidth=2.4, capsize=5, ms=10, zorder=4)
    for mu, yi, cd in zip(d.mean_dU, y, d.cohen_d):
        ax.annotate(f"{mu:.2f}   (d={cd})", (d.ci_high.iloc[yi], yi), (10, 0),
                    textcoords="offset points", va="center", fontsize=8.8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(d.comparison); ax.set_xlim(-0.05, 0.5)
    style(ax, xl="utility gap  ΔU  (95% paired bootstrap CI)", ti="CMV-SDSE beats every baseline (6 families)", grid="x")
    save(f, "fig_bootstrap")


def fig_latency():
    d = pd.read_csv(f"{T}/rev_latency.csv")
    cmap = {"same-model (k=50)": CRIMSON, "cross-family (all 3)": AMBER, "CMV-SDSE": BLUE}
    f, ax = plt.subplots(figsize=(7.2, 4.4))
    for _, r in d.iterrows():
        ax.scatter(r.wall_seconds, r.U, s=210, color=cmap.get(r.policy, GREY), ec="white", lw=1, zorder=4)
        ax.annotate(f"{r.policy}\n{r.wall_seconds:.2f}s, {int(r.calls)} call(s)", (r.wall_seconds, r.U),
                    (10, 8), textcoords="offset points", fontsize=8.6, color=INK)
    ax.set_xscale("log"); ax.set_xlim(0.15, 60); ax.set_ylim(0.15, 0.55)
    ax.annotate("", (0.26, 0.49), (30, 0.23), arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4, ls="--"))
    ax.text(2.0, 0.42, "117× less compute,\nhigher utility", color=BLUE, fontsize=9)
    style(ax, xl="wall-clock compute per decision (s, log)", yl="utility  U (ρ=1)", ti="Real compute cost vs utility")
    save(f, "fig_latency")


def fig_permodel():
    d = pd.read_csv(f"{T}/rev3_per_model.csv")
    f, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot([0, 1], [0, 1], color=LGREY, ls=":", lw=1.4, zorder=1)
    for _, r in d.iterrows():
        dead = r.model == "llama32_3b"
        ax.scatter(r.standalone_fa, r.standalone_catch, s=120 + r.median_latency_s * 260,
                   color=(CRIMSON if dead else BLUE), ec="white", lw=1, zorder=4, alpha=0.9)
        ax.annotate(r.model, (r.standalone_fa, r.standalone_catch), (9, 6), textcoords="offset points",
                    fontsize=9, color=(CRIMSON if dead else INK))
    ax.set_xlim(0, 0.75); ax.set_ylim(0, 1.0)
    ax.text(0.02, 0.93, "marker size ∝ latency;  llama32_3b = dead 3B (low catch)", fontsize=8.4, color=MUTE)
    style(ax, xl="standalone false-alarm rate", yl="standalone catch rate", ti="Per-verifier heterogeneity", grid="both")
    save(f, "fig_permodel")


if __name__ == "__main__":
    fig_component(); fig_lomo(); fig_probeutil(); fig_bootstrap(); fig_latency(); fig_permodel()
    print("\nDONE — 6 appendix figures.")
