"""make_figures2.py -- redesigned composites for paper figures 2, 4, 5, 11, 12, 13.

    source /home/alex/venvs/env/bin/activate
    python scripts/make_figures2.py            # you run this; no network / no model calls

Each figure is rebuilt as ONE combined image with its (a)/(b)/(c) panels in order, from the same
real CSVs the originals use, but with distinctive chart types (a shaded redundancy band, a bubble
Pareto with a dominance region, dumbbells, forest/CI dot-plots, a decay staircase, null-band
lollipops) instead of look-alike line charts. Large fonts, generous spacing, no overlapping text.

Outputs to figure2/ :  fig02_motivation, fig04_pareto, fig05_selection, fig11_saturation,
fig12_diversity, fig13_sevenfamily  (each .png 300dpi + .pdf), and figures_all.pdf (6 pages, in
figure order). Panel letters a,b,c are preserved so the manuscript captions still match.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "figure2"
OUT.mkdir(exist_ok=True)

# --------------------------------------------------------------- modern, CVD-safe palette -----
# modern palette (Open Color): indigo / coral / teal / violet — clean and contemporary
BLUE, ORANGE, AQUA, VIOLET = "#4c6ef5", "#f76707", "#0ca678", "#7048e8"
RED, YELLOW, GREEN, MAG = "#f03e3e", "#f59f00", "#2f9e44", "#e64980"
INK, SECOND, MUTE = "#212529", "#495057", "#909296"
GRID, BASE, SURF = "#e9ecef", "#ced4da", "#ffffff"
BLUE_FILL, ORANGE_FILL, GREEN_FILL = "#dbe4ff", "#ffe8cc", "#d3f9d8"

mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 14.5, "axes.titlesize": 15.5, "axes.labelsize": 14, "axes.titleweight": "bold",
    "xtick.labelsize": 12.5, "ytick.labelsize": 12.5, "legend.fontsize": 12,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": BASE,
    "xtick.color": SECOND, "ytick.color": SECOND, "axes.linewidth": 1.1,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False,
})
FAM7 = ["MAST-3.3", "ARC", "MMLU", "CSQA", "TruthfulQA", "GSM8K", "MBPP"]
POOL = {"qwen7b", "llama31", "mistral7b", "llama32_3b"}


def _fin(ax, title="", xlabel="", ylabel="", grid="y"):
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
    ax.tick_params(length=3.5, colors=SECOND)
    ax.grid(axis="y", visible=(grid in ("y", "both")))
    ax.grid(axis="x", visible=(grid in ("x", "both")))
    if title:
        ax.set_title(title, color=INK, loc="left", pad=9)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _letters(axs):                                  # fold (a)/(b)/(c) INTO the title (no overlap)
    for a, t in zip(axs, "abc"):
        a.set_title(f"({t})  {a.get_title()}", loc="left", color=INK)


def _wilson_lo(k, n, z=1.959963984540054):
    if n <= 0:
        return 0.0
    p = k / n; z2 = z * z; d = 1 + z2 / n
    c = (p + z2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / d
    return max(0.0, c - h)


def _neff(k, rho):
    return k / (1 + (k - 1) * rho)


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# ============================================================ shared panels ====================
def _panel_neff(ax):
    d = pd.read_csv(T / "redundancy_invariant_7family.csv").set_index("family").reindex(FAM7)
    ks = np.arange(1, 51)
    curves = np.vstack([_neff(ks, r) for r in d["sm_rho_v"].values])   # 7 families x 50
    lo, hi, mid = curves.min(0), curves.max(0), np.median(curves, 0)
    ax.plot(ks, ks, color=BASE, lw=2.2, ls=(0, (5, 4)), zorder=2)
    ax.annotate("independent ideal  $n_{\\mathrm{eff}}=k$", (43, 43), xytext=(-8, 10),
                textcoords="offset points", ha="right", va="bottom", color=MUTE, fontsize=12.5)
    ax.fill_between(ks, lo, hi, color=BLUE_FILL, zorder=1)
    ax.plot(ks, mid, color=BLUE, lw=3.2, zorder=4, label="same-model (7 families)")
    ax.plot(ks, _neff(ks, 0.20), color=ORANGE, lw=3.0, zorder=3, label="cross-family pool")
    ax.scatter([50], [mid[-1]], s=95, color=BLUE, edgecolors="white", linewidths=1.5, zorder=6)
    ax.annotate(f"50 calls → {mid[-1]:.1f}", (50, mid[-1]), (-12, 16), textcoords="offset points",
                ha="right", color=BLUE, fontweight="bold", fontsize=13)
    ax.set_xlim(1, 52); ax.set_ylim(0, 52)
    ax.legend(loc="upper left")
    _fin(ax, "50 calls ≈ one independent verifier", "verifier calls  $k$",
         "effective verifiers  $n_{\\mathrm{eff}}(k)$", grid="both")


def _panel_diversity(ax):
    d = pd.read_csv(T / "rev_diversity_counterexample.csv")
    d["is_best_solo"] = d["is_best_solo"].map({True: True, False: False, "True": True, "False": False})
    cand = d[(~d.is_best_solo) & (d.verifier.isin(POOL))]
    dead = cand[cand.verifier == "llama32_3b"]
    live = cand[cand.verifier != "llama32_3b"]
    ax.axhspan(-0.03, 0.03, color=GRID, zorder=0)
    ax.axhline(0, color=BASE, lw=1.0, zorder=1)
    ax.scatter(live.rho_with_solo, live.marginal_catch_given_solo, s=95, color=BLUE,
               edgecolors="white", linewidths=1.0, zorder=4, label="pool candidate")
    ax.scatter(dead.rho_with_solo, dead.marginal_catch_given_solo, s=115, color=RED,
               edgecolors="white", linewidths=1.0, zorder=5, label="llama3.2-3B (dead)")
    g = d[(d.family == "CSQA") & (d.verifier == "gemma2_9b")]
    if len(g):
        gx, gy = float(g.rho_with_solo.iloc[0]), float(g.marginal_catch_given_solo.iloc[0])
        ax.scatter([gx], [gy], s=340, marker="*", color=YELLOW, edgecolors=INK, linewidths=1.1,
                   zorder=6, label="gemma2-9B · CSQA (k=50)")
        ax.annotate("accurate & decorrelated,\nyet 0 residual catch", (gx, gy), (12, 44),
                    textcoords="offset points", fontsize=12, color="#b45309", ha="left",
                    arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1.6))
    if len(dead):
        ax.annotate("statistically diverse,\ndecision-useless",
                    (float(dead.rho_with_solo.mean()), 0.0), (14, 52), textcoords="offset points",
                    fontsize=12, color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    ax.set_xlim(-0.03, 1.02); ax.set_ylim(-0.06, 0.74)
    ax.legend(loc="upper right")
    _fin(ax, "Diversity ≠ decision value", "dependence: correlation with the solo",
         "conditional marginal catch")


def _panel_gate(ax):
    d = pd.read_csv(T / "rev_diversity_counterexample.csv")
    d["is_best_solo"] = d["is_best_solo"].map({True: True, False: False, "True": True, "False": False})
    c = d[(~d.is_best_solo) & (d.verifier.isin(POOL))].copy()
    n = c.solo_missed.astype(float).clip(lower=1)
    k = (c.marginal_catch_given_solo * n).round().astype(int)
    c["lo"] = [_wilson_lo(int(kk), int(nn)) for kk, nn in zip(k, n)]
    dead = (c.verifier == "llama32_3b").values
    ax.axhspan(1e-6, 0.9, color=GREEN_FILL, zorder=0)
    ax.axhline(0, color=RED, lw=1.6, ls=(0, (5, 3)), zorder=2)
    ax.scatter(c.marginal_catch_given_solo.values[~dead], c.lo.values[~dead], s=95, color=BLUE,
               edgecolors="white", linewidths=1.0, zorder=4, label="admitted (CI-low > 0)")
    ax.scatter(c.marginal_catch_given_solo.values[dead], c.lo.values[dead], s=115, color=RED,
               edgecolors="white", linewidths=1.0, zorder=5, label="dead verifier (origin)")
    ax.text(0.03, 0.88, "ADD region  (Wilson CI-low > 0)", transform=ax.transAxes, color=GREEN,
            fontsize=13, fontweight="bold")
    ax.set_xlim(-0.03, 0.8); ax.set_ylim(-0.03, 0.6)
    ax.legend(loc="lower right")
    _fin(ax, "The value the controller acts on",
         "conditional marginal catch  $\\Delta_{\\mathrm{catch}}$", "Wilson lower bound")


# ============================================================ FIG 2 — motivation ===============
def fig02(fig=None):
    own = fig is None
    if own:
        fig = plt.figure(figsize=(19.5, 6.0))
    axs = fig.subplots(1, 3)
    _panel_neff(axs[0]); _panel_diversity(axs[1]); _panel_gate(axs[2])
    _letters(axs)
    fig.suptitle("Figure 2   Why more calls and more diversity are not enough",
                 x=0.008, ha="left", y=0.975, fontsize=17, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.13, wspace=0.28)
    if own:
        _save(fig, "fig02_motivation")


# ============================================================ FIG 4 — Pareto (bubble build) ====
def fig04(fig=None):
    own = fig is None
    if own:
        fig = plt.figure(figsize=(19.5, 6.0))
    axs = fig.subplots(1, 3, sharex=True, sharey=True)
    d = pd.read_csv(T / "rev_pareto.csv")
    d["pareto"] = d["pareto"].map({True: True, False: False, "True": True, "False": False})
    sm = d[d.policy.str.startswith("Same-model")]
    base3 = d[(~d.pareto) & (~d.policy.str.startswith("Same-model"))]
    cmv = d[d.pareto]
    size = lambda c: 90 + 26 * np.sqrt(c.astype(float))
    stages = [("repetition only", [(sm, MUTE)]),
              ("+ naive diversification", [(sm, MUTE), (base3, ORANGE)]),
              ("+ CMV-SDSE", [(sm, MUTE), (base3, ORANGE), (cmv, BLUE)])]
    cx, cy = float(cmv.wall_seconds.iloc[0]), float(cmv.U.iloc[0])
    for i, (ax, (subt, layers)) in enumerate(zip(axs, stages)):
        if i == 2:                                    # dominance region on the final panel
            ax.fill_between([cx, 60], 0.12, cy, color=BLUE_FILL, zorder=0)
            ax.text(cx * 2.3, 0.155, "region CMV-SDSE\ndominates", color=BLUE, fontsize=11.5)
        for sub, col in layers:
            ax.scatter(sub.wall_seconds, sub.U, s=size(sub.calls), color=col,
                       edgecolors="white", linewidths=1.4, zorder=4)
        ax.set_xscale("log"); ax.set_xlim(0.18, 60); ax.set_ylim(0.12, 0.56)
        ax.set_xticks([0.2, 1, 5, 30]); ax.set_xticklabels(["0.2", "1", "5", "30"])
        _fin(ax, subt, "compute per decision  (s, log)",
             "utility  $U=\\mathrm{catch}-\\rho\\,\\mathrm{FA}$" if i == 0 else "", grid="both")
    a = axs[2]
    a.annotate("CMV-SDSE\n(1 verifier)", (cx, cy), (14, -4), textcoords="offset points",
               color=INK, fontweight="bold", fontsize=13)
    a.annotate("same-model ×50\n(≈117× compute)", (float(sm.wall_seconds.iloc[0]), float(sm.U.iloc[0])),
               (-10, 14), textcoords="offset points", ha="right", color=SECOND, fontsize=12)
    a.annotate("3-verifier panels", (float(base3.wall_seconds.mean()), float(base3.U.min())),
               (0, -36), textcoords="offset points", ha="center", color="#d9480f", fontsize=12,
               arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    _letters(axs)
    fig.suptitle("Figure 4   Cost–performance frontier   (marker area ∝ verifier calls)",
                 x=0.008, ha="left", y=0.975, fontsize=17, fontweight="bold")
    fig.subplots_adjust(left=0.055, right=0.99, top=0.85, bottom=0.13, wspace=0.10)
    if own:
        _save(fig, "fig04_pareto")


# ============================================================ FIG 5 — selection evidence =======
def fig05(fig=None):
    own = fig is None
    if own:
        fig = plt.figure(figsize=(19.5, 6.2))
    axs = fig.subplots(1, 3)
    # (a) GSM8K first greedy step: Delta_catch with CI, coloured by selected
    gm = pd.read_csv(T / "gsm8k_marginal.csv")
    s1 = gm[gm.step == 1].copy()
    s1["lab"] = s1.cand_type.str.title() + ": " + s1.candidate
    s1 = s1.sort_values("delta")
    y = np.arange(len(s1))
    sel = s1.selected.map({True: True, False: False, "True": True, "False": False}).values
    a = axs[0]
    a.axvline(0, color=RED, ls=(0, (5, 3)), lw=1.6, zorder=1)
    a.hlines(y, s1.delta_ci_low, s1.delta_ci_high, color=BASE, lw=3.0, zorder=2)
    a.scatter(s1.delta, y, s=150, c=np.where(sel, BLUE, MUTE), edgecolors="white",
              linewidths=1.2, zorder=4)
    a.set_yticks(y); a.set_yticklabels(s1.lab); a.set_xlim(-0.05, 1.02)
    a.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=11, label="added"),
                      Line2D([], [], marker="o", ls="", color=MUTE, ms=11, label="not added (CI touches 0)")],
             loc="lower right")
    _fin(a, "First greedy step (GSM8K)", "conditional marginal catch  $\\Delta_{\\mathrm{catch}}$", grid="x")
    # (b) useful diversification significant on every family
    d = pd.read_csv(T / "redundancy_invariant_7family.csv").set_index("family")
    d = d.reindex([f for f in FAM7 if f != "MAST-3.3"]).dropna(subset=["div_unique"]).iloc[::-1]
    frac = (d.div_unique / d.div_denom).values; ylo = d.div_ci_low.values; yy = np.arange(len(d))
    b = axs[1]
    b.axvline(0, color=BASE, lw=1.0)
    b.hlines(yy, ylo, frac, color=BASE, lw=3.0, zorder=2)
    b.scatter(ylo, yy, marker="|", s=360, color=INK, linewidths=2.4, zorder=3)
    b.scatter(frac, yy, s=150, color=AQUA, edgecolors="white", linewidths=1.2, zorder=4)
    for f_, yi in zip(frac, yy):
        b.annotate(f"{f_:.2f}", (f_, yi), (11, 0), textcoords="offset points", va="center",
                   fontsize=12.5, color=INK, fontweight="bold")
    b.set_yticks(yy); b.set_yticklabels(d.index); b.set_xlim(-0.02, 0.95)
    b.annotate("Wilson lower bound", (float(ylo.min()), len(d) - 1), (4, 22),
               textcoords="offset points", fontsize=11.5, color=SECOND)
    _fin(b, "Useful diversification is significant", "survivor errors caught by the added verifier", grid="x")
    # (c) repetition vs diversity dumbbell
    sp = pd.read_csv(T / "fusion_scale_panel.csv").iloc[::-1]
    yy = np.arange(len(sp)); c = axs[2]
    for yi, s0, pdv in zip(yy, sp.solo_catch, sp.plus_diverse):
        c.plot([s0, pdv], [yi, yi], color=BASE, lw=3.0, zorder=2, solid_capstyle="round")
    c.scatter(sp.solo_catch, yy, s=120, color=MUTE, edgecolors="white", linewidths=1.0, zorder=4)
    c.scatter(sp.plus_same_model, yy, s=95, color=RED, edgecolors="white", linewidths=1.0, zorder=6)
    c.scatter(sp.plus_diverse, yy, s=150, color=BLUE, edgecolors="white", linewidths=1.0, zorder=5)
    c.set_yticks(yy); c.set_yticklabels(sp.family); c.set_xlim(0.35, 1.03)
    c.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTE, ms=11, label="solo"),
                      Line2D([], [], marker="o", ls="", color=RED, ms=10, label="+ same-model (+0.003)"),
                      Line2D([], [], marker="o", ls="", color=BLUE, ms=11, label="+ diverse (+0.079)")],
             loc="lower left")
    _fin(c, "Repetition sits still; diversity moves", "error-catch rate", grid="x")
    _letters(axs)
    fig.suptitle("Figure 5   Conditional marginal value selects complementary verifiers",
                 x=0.008, ha="left", y=0.975, fontsize=17, fontweight="bold")
    fig.subplots_adjust(left=0.11, right=0.99, top=0.85, bottom=0.13, wspace=0.42)
    if own:
        _save(fig, "fig05_selection")


# ============================================================ FIG 11 — saturation ==============
def _band(ax, x, y, lo, hi, color, fill, label, marker="o"):
    ax.fill_between(x, lo, hi, color=fill, zorder=1)
    ax.plot(x, y, "-", marker=marker, color=color, lw=3.0, ms=6, zorder=4, label=label)


def fig11(fig=None):
    own = fig is None
    if own:
        fig = plt.figure(figsize=(19.5, 6.0))
    axs = fig.subplots(1, 3)
    _panel_neff(axs[0]); axs[0].set_title("Effective verifiers saturate", loc="left")
    rel = pd.read_csv(T / "reliability_by_k.csv")
    r = rel[(rel.pairing_condition == "cross_family_3") & (rel.failure_mode_id == "FM-3.3")].sort_values("k")
    _band(axs[1], r.k, r.reliability, r.ci_low, r.ci_high, BLUE, BLUE_FILL, "empirical (95% band)")
    axs[1].plot(r.k, 1 - 0.20 ** r.k, color=MUTE, ls=":", lw=2.2, label="independent bound")
    axs[1].set_ylim(0, 1.03); axs[1].legend(loc="lower right")
    _fin(axs[1], "Catch reliability plateaus by k≈6", "cascade depth  $k$", "P(error caught by gate ≤ k)")
    fa = pd.read_csv(T / "false_alarm_by_k.csv")
    fq = fa[(fa.pairing_condition == "cross_family_3") & (fa.failure_mode_id == "FM-3.3")].sort_values("k")
    _band(axs[2], fq.k, fq.false_alarm_rate, fq.ci_low, fq.ci_high, ORANGE, ORANGE_FILL,
          "empirical (95% band)", marker="s")
    axs[2].set_ylim(0, 1.03); axs[2].legend(loc="lower right")
    _fin(axs[2], "False alarms plateau at a floor", "cascade depth  $k$", "P(clean item flagged by gate ≤ k)")
    _letters(axs)
    fig.suptitle("Figure 11   Same-model verification saturates with cascade depth",
                 x=0.008, ha="left", y=0.975, fontsize=17, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.13, wspace=0.26)
    if own:
        _save(fig, "fig11_saturation")


# ============================================================ FIG 12 — count/diversity/value ==
def fig12(fig=None):
    own = fig is None
    if own:
        fig = plt.figure(figsize=(19.5, 6.0))
    axs = fig.subplots(1, 3)
    rel = pd.read_csv(PROC / "reliability_by_k.csv")
    s = rel[(rel.pairing_condition == "same_model") & (rel.failure_mode_id == "FM-3.3")].sort_values("k")
    k = s.k.values[:12]
    marg = np.diff(np.concatenate([[0.0], s.reliability.values]))[:12].clip(min=0)
    a = axs[0]
    a.bar(k, marg, width=0.82, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
    a.bar([k[0]], [marg[0]], width=0.82, color=ORANGE, edgecolor="white", linewidth=0.8, zorder=4)
    if len(marg) > 1:
        a.annotate("after the 1st call,\nrepetition adds ≈ 0", (2.7, marg[1]), (66, 66),
                   textcoords="offset points", fontsize=12.5, color=SECOND,
                   arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.3))
    a.set_xticks(range(1, 13, 2))
    _fin(a, "Count ≠ value (repetition saturates)", "same-model call index  $k$", "marginal catch at call $k$")
    _panel_diversity(axs[1]); _panel_gate(axs[2])
    _letters(axs)
    fig.suptitle("Figure 12   Count ≠ diversity ≠ value",
                 x=0.008, ha="left", y=0.975, fontsize=17, fontweight="bold")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.13, wspace=0.28)
    if own:
        _save(fig, "fig12_diversity")


# ============================================================ FIG 13 — seven families ==========
def _lolli(ax, vals, labels, x0, xlim, title, xlabel, band=None, color=BLUE):
    y = np.arange(len(labels))
    if band:
        ax.axvspan(band[0], band[1], color=BLUE_FILL, zorder=0, label=f"null band [{band[0]}, {band[1]}]")
    ax.axvline(x0, color=BASE, ls=(0, (2, 3)), lw=1.2, zorder=1)
    ax.hlines(y, x0, vals, color=BASE, lw=3.0, zorder=2)
    ax.scatter(vals, y, s=150, color=color, edgecolors="white", linewidths=1.2, zorder=4)
    for xv, yv in zip(vals, y):
        ax.annotate(f"{xv:.2f}", (xv, yv), (11, 0), textcoords="offset points", va="center",
                    fontsize=12.5, color=INK, fontweight="bold")
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(*xlim)
    ax.grid(axis="y", visible=False)
    if band:
        ax.legend(loc="lower right")
    _fin(ax, title, xlabel, grid="x")


def fig13(fig=None):
    own = fig is None
    if own:
        fig = plt.figure(figsize=(19.5, 6.0))
    axs = fig.subplots(1, 3)
    d = pd.read_csv(T / "redundancy_invariant_7family.csv").set_index("family").reindex(FAM7)
    fams = FAM7[::-1]
    _lolli(axs[0], d.sm_n_eff50.values[::-1], fams, 1.0, (0.95, 1.5),
           "Effective verifiers  $n_{\\mathrm{eff}}$@50", "$n_{\\mathrm{eff}}$@50   (independent = 50)",
           band=(1.06, 1.21), color=BLUE)
    _lolli(axs[1], d.sm_rho_v.values[::-1], fams, 0.0, (0, 1.08),
           "Intra-item correlation  $\\rho_v$", "correlation  $\\rho_v$", color=VIOLET)
    _lolli(axs[2], d.sm_miss.values[::-1], fams, 0.0, (0, 1.0),
           "Same-model miss rate", "miss rate", color=ORANGE)
    _letters(axs)
    fig.suptitle("Figure 13   Same-model redundancy replicates across seven families",
                 x=0.008, ha="left", y=0.975, fontsize=17, fontweight="bold")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.85, bottom=0.13, wspace=0.36)
    if own:
        _save(fig, "fig13_sevenfamily")


BUILDERS = [("fig02_motivation", fig02), ("fig04_pareto", fig04), ("fig05_selection", fig05),
            ("fig11_saturation", fig11), ("fig12_diversity", fig12), ("fig13_sevenfamily", fig13)]

if __name__ == "__main__":
    for _, fn in BUILDERS:
        fn()
    with PdfPages(OUT / "figures_all.pdf") as pdf:
        for _, fn in BUILDERS:
            fig = plt.figure(figsize=(19.5, 6.1))
            fn(fig=fig)
            pdf.savefig(fig); plt.close(fig)
    print("wrote", OUT / "figures_all.pdf")
    print("\nAll combined figures written to:", OUT)
