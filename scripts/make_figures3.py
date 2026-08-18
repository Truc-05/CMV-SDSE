"""make_figures3.py -- redesigned paper figures, one distinct silhouette per claim.

    source /home/alex/venvs/env/bin/activate
    python scripts/make_figures3.py            # you run this; no network / no model calls

Rebuilds six figures from the same real CSVs, but each with a chart type chosen for its claim:
  fig_pareto          -> ONE wide Pareto hero (dominance region + direct annotations)
  fig_redun24b        -> ONE wide paired-shift plot (<=14B -> 24B), core-band highlighted
  fig_saturation      -> 3 panels: ribbon growth | catch staircase | false-alarm plateau
  fig_obs2_diversity  -> 3 panels: decay lollipop | region scatter | decision threshold
  fig_sevenfamily     -> 3 panels sharing family rows: interval | bars | dots (a profile strip)
  fig_sign_befor      -> 3 panels: forest | ranked bars+CI | dumbbell

Files are named exactly as in nd.md (fig_<name>.pdf/.png) so they drop into figures/ to replace.
Written to figure3/ ; also figure3/figures_all.pdf (6 pages). Big fonts, generous spacing.
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
OUT = ROOT / "figure3"
OUT.mkdir(exist_ok=True)

# ------------------------------------------------- modern palette (Open Color) ----------------
BLUE, ORANGE, TEAL, VIOLET = "#132981", "#915832", "#82c7b3", "#18186c"
RED, YELLOW, GREEN, PINK = "#f03e3e", "#f3f554", "#2f529e", "#4988e6"
INK, SECOND, MUTE = "#212529", "#495057", "#909296"
GRID, BASE, SURF = "#e9ecef", "#ced4da", "#ffffff"
BLUE_FILL, ORANGE_FILL, GREEN_FILL, GREY_FILL = "#dbe4ff", "#ffe8cc", "#d3f9d8", "#f1f3f5"
AMBER_DK, ORANGE_DK = "#b45309", "#d9480f"

mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 14.5, "axes.titleweight": "bold",
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12.5,
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


def _letters(axs):
    for a, t in zip(axs, "abcdef"):
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


def _save(fig, name, suptitle):
    fig.suptitle(suptitle, x=0.008, ha="left", y=0.975, fontsize=17, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


# ================================================================ FIG 4 — Pareto hero ==========
def fig_pareto(fig=None):
    own = fig is None
    if own:
        fig = plt.figure(figsize=(13.6, 5.4))
    ax = fig.subplots(1, 1)
    d = pd.read_csv(T / "rev_pareto.csv")
    d["pareto"] = d["pareto"].map({True: True, False: False, "True": True, "False": False})
    sm = d[d.policy.str.startswith("Same-model")].iloc[0]
    base3 = d[(~d.pareto) & (~d.policy.str.startswith("Same-model"))]
    cmv = d[d.pareto].iloc[0]
    cx, cy = float(cmv.wall_seconds), float(cmv.U)
    ax.fill_between([cx, 100], 0.1, cy, color=GREY_FILL, zorder=0)
    ax.text(2.6, 0.15, "DOMINATED REGION", color=MUTE, fontsize=14, fontweight="bold")
    ax.scatter([sm.wall_seconds], [sm.U], s=560, color=MUTE, edgecolors="white", linewidths=1.6, zorder=4)
    ax.scatter(base3.wall_seconds, base3.U, s=180, color=ORANGE, edgecolors="white", linewidths=1.4, zorder=4)
    ax.scatter([cx], [cy], s=680, marker="*", color=BLUE, edgecolors=INK, linewidths=1.2, zorder=6)
    ax.annotate(f"CMV-SDSE\n≈{sm.wall_seconds / cx:.0f}× less compute · +{cy - sm.U:.2f} utility",
                (cx, cy), (18, -6), textcoords="offset points", color=INK, fontweight="bold", fontsize=14)
    ax.annotate("same-model ×50", (sm.wall_seconds, sm.U), (-12, 16), textcoords="offset points",
                ha="right", color=SECOND, fontsize=13)
    ax.annotate("naive 3-verifier panels\n(random · accuracy · low-correlation)",
                (float(base3.wall_seconds.mean()), float(base3.U.min())), (0, -44),
                textcoords="offset points", ha="center", color=ORANGE_DK, fontsize=12.5,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
    ax.set_xscale("log"); ax.set_xlim(0.18, 60); ax.set_ylim(0.1, 0.56)
    ax.set_xticks([0.2, 0.5, 1, 2, 5, 10, 30]); ax.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10", "30"])
    _fin(ax, "", "compute per decision  (s, log scale)  →",
         "decision utility  $U=\\mathrm{catch}-\\rho\\,\\mathrm{FA}$  ↑", grid="both")
    fig.subplots_adjust(left=0.075, right=0.98, top=0.88, bottom=0.15)
    if own:
        _save(fig, "fig_pareto", "Figure 4   Cost–performance frontier — CMV-SDSE is the sole non-dominated policy")


# ================================================================ FIG 6 — 24B paired shift =====
def fig_redun24b(fig=None):
    own = fig is None
    if own:
        fig = plt.figure(figsize=(13.6, 6.4))
    ax = fig.subplots(1, 1)
    a = pd.read_csv(T / "redundancy_invariant_7family.csv").set_index("family")
    b = pd.read_csv(T / "redun_mistral24b_neff.csv").set_index("family")
    fams = [f for f in FAM7 if f in b.index]
    df = pd.DataFrame({"n14": a.loc[fams, "sm_n_eff50"], "n24": b.loc[fams, "n_eff50"]}).sort_values("n24")
    y = np.arange(len(df))
    ax.axvspan(1.06, 1.21, color=BLUE_FILL, zorder=0)
    ax.text(1.135, len(df) - 0.35, "core ≤14B band\n[1.06, 1.21]", ha="center", va="top",
            color=BLUE, fontsize=13, fontweight="bold")
    for yi, (n14, n24) in enumerate(zip(df.n14, df.n24)):
        ax.annotate("", xy=(n24, yi), xytext=(n14, yi),
                    arrowprops=dict(arrowstyle="-|>", color=BASE, lw=2.8, mutation_scale=20), zorder=2)
    ax.scatter(df.n14, y, s=160, color=MUTE, edgecolors="white", linewidths=1.4, zorder=4, label="≤14B pool")
    ax.scatter(df.n24, y, s=185, color=ORANGE, edgecolors="white", linewidths=1.4, zorder=5, label="24B checkpoint")
    for yi, n14 in zip(y, df.n14):
        ax.annotate(f"{n14:.2f}", (n14, yi), (-12, 0), textcoords="offset points", ha="right",
                    va="center", color=SECOND, fontsize=12)
    for yi, n24 in zip(y, df.n24):
        ax.annotate(f"{n24:.2f}", (n24, yi), (13, 0), textcoords="offset points", ha="left",
                    va="center", color=ORANGE_DK, fontweight="bold", fontsize=13.5)
    ax.text(2.52, 0.15, "50 nominal calls →\nonly 1.4–2.3 effective sources",
            ha="right", va="bottom", color=INK, fontsize=13.5)
    ax.set_yticks(y); ax.set_yticklabels(df.index); ax.set_xlim(1.0, 2.6); ax.set_ylim(-0.6, len(df) - 0.3)
    ax.legend(loc="lower right")
    _fin(ax, "", "effective independent verifiers  $n_{\\mathrm{eff}}$@50   →", grid="x")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.12)
    if own:
        _save(fig, "fig_redun24b", "Figure 6   Redundancy persists beyond 14B — the magnitude shifts with the checkpoint")


# ================================================================ FIG 11 — saturation ==========
def fig_saturation(fig=None):
    own = fig is None
    if own:
        fig = plt.figure(figsize=(20, 6.1))
    axs = fig.subplots(1, 3)
    # (a) ribbon growth: independence ideal vs same-model band hugging the floor
    d = pd.read_csv(T / "redundancy_invariant_7family.csv").set_index("family").reindex(FAM7)
    ks = np.arange(1, 51)
    curves = np.vstack([_neff(ks, r) for r in d["sm_rho_v"].values])
    lo, hi, mid = curves.min(0), curves.max(0), np.median(curves, 0)
    a = axs[0]
    a.fill_between(ks, mid, ks, color=GREY_FILL, zorder=0)                 # the giant wasted gap
    a.plot(ks, ks, color=BASE, lw=2.4, ls=(0, (5, 4)), zorder=2)
    a.annotate("independent ideal  $n_{\\mathrm{eff}}=k$", (44, 44), (-8, 10),
               textcoords="offset points", ha="right", color=MUTE, fontsize=12.5)
    a.fill_between(ks, lo, hi, color=BLUE_FILL, zorder=1)
    a.plot(ks, mid, color=BLUE, lw=3.4, zorder=4, label="same-model (7 families)")
    a.plot(ks, _neff(ks, 0.20), color=ORANGE, lw=3.0, zorder=3, label="cross-family pool")
    a.annotate("", xy=(50, 50), xytext=(50, mid[-1]), arrowprops=dict(arrowstyle="<->", color=INK, lw=2.0))
    a.text(47.5, 26, f"{50 - mid[-1]:.0f} of 50 calls\nadd no evidence", ha="right", color=INK, fontsize=12.5)
    a.set_xlim(1, 52); a.set_ylim(0, 52); a.legend(loc="upper left")
    _fin(a, "Nominal 50 vs effective ~1", "verifier calls  $k$", "effective verifiers  $n_{\\mathrm{eff}}(k)$", grid="both")
    # (b) catch saturation staircase + marker
    rel = pd.read_csv(T / "reliability_by_k.csv")
    r = rel[(rel.pairing_condition == "cross_family_3") & (rel.failure_mode_id == "FM-3.3")].sort_values("k")
    b = axs[1]
    b.fill_between(r.k, r.ci_low, r.ci_high, step="post", color=BLUE_FILL, zorder=1)
    b.step(r.k, r.reliability, where="post", color=BLUE, lw=3.4, zorder=4)
    b.scatter(r.k, r.reliability, s=60, color=BLUE, edgecolors="white", linewidths=1.0, zorder=5)
    b.axvline(6, color=INK, lw=2.0, ls=(0, (4, 3)), zorder=3)
    b.text(6.4, 0.12, "SATURATION\n≈ 6 calls", color=INK, fontsize=13, fontweight="bold")
    b.set_ylim(0, 1.03)
    _fin(b, "Catch saturates early (staircase)", "cascade depth  $k$", "P(error caught by gate ≤ k)")
    # (c) false-alarm plateau
    fa = pd.read_csv(T / "false_alarm_by_k.csv")
    fq = fa[(fa.pairing_condition == "cross_family_3") & (fa.failure_mode_id == "FM-3.3")].sort_values("k")
    plateau = float(fq.false_alarm_rate.iloc[-1])
    c = axs[2]
    c.axhspan(plateau - 0.05, 1.0, color=ORANGE_FILL, zorder=0)
    c.text(2, plateau + 0.03, "false alarms never vanish", color=ORANGE_DK, fontsize=12.5, fontweight="bold")
    c.step(fq.k, fq.false_alarm_rate, where="post", color=ORANGE, lw=3.4, zorder=4)
    c.scatter(fq.k, fq.false_alarm_rate, s=60, color=ORANGE, edgecolors="white", linewidths=1.0, zorder=5)
    c.set_ylim(0, 1.03)
    _fin(c, "False alarms plateau at a floor", "cascade depth  $k$", "P(clean item flagged by gate ≤ k)")
    _letters(axs)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.13, wspace=0.24)
    if own:
        _save(fig, "fig_saturation", "Figure 11   Evidence saturates: nominal effort, early catch plateau, residual false alarms")


# ================================================================ FIG 12 — count/div/value =====
def _region_scatter(ax):
    d = pd.read_csv(T / "rev_diversity_counterexample.csv")
    d["is_best_solo"] = d["is_best_solo"].map({True: True, False: False, "True": True, "False": False})
    cand = d[(~d.is_best_solo) & (d.verifier.isin(POOL))]
    dead = cand[cand.verifier == "llama32_3b"]; live = cand[cand.verifier != "llama32_3b"]
    xm, ym = 0.5, 0.15
    ax.axhspan(ym, 0.75, color=GREEN_FILL, zorder=0)
    ax.axhspan(-0.06, ym, color=GREY_FILL, zorder=0)
    ax.axvline(xm, color=BASE, lw=1.0, zorder=1); ax.axhline(ym, color=BASE, lw=1.0, zorder=1)
    ax.text(0.02, 0.72, "valuable complement", color=GREEN, fontsize=12, fontweight="bold", va="top")
    ax.text(0.02, 0.02, "diverse but useless", color=RED, fontsize=12, fontweight="bold", va="bottom")
    ax.text(0.98, 0.02, "redundant + useless", color=MUTE, fontsize=12, fontweight="bold", va="bottom", ha="right")
    ax.scatter(live.rho_with_solo, live.marginal_catch_given_solo, s=100, color=BLUE,
               edgecolors="white", linewidths=1.0, zorder=4, label="pool candidate")
    ax.scatter(dead.rho_with_solo, dead.marginal_catch_given_solo, s=120, color=RED,
               edgecolors="white", linewidths=1.0, zorder=5, label="llama3.2-3B (dead)")
    g = d[(d.family == "CSQA") & (d.verifier == "gemma2_9b")]
    if len(g):
        gx, gy = float(g.rho_with_solo.iloc[0]), float(g.marginal_catch_given_solo.iloc[0])
        ax.scatter([gx], [gy], s=360, marker="*", color=YELLOW, edgecolors=INK, linewidths=1.1,
                   zorder=6, label="gemma2-9B · CSQA")
        ax.annotate("accurate & decorrelated,\nyet 0 residual value", (gx, gy), (14, 40),
                    textcoords="offset points", fontsize=12, color=AMBER_DK,
                    arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1.6))
    ax.set_xlim(-0.05, 1.02); ax.set_ylim(-0.06, 0.75); ax.legend(loc="upper right")
    _fin(ax, "Diversity ≠ value", "dependence: correlation with the solo", "conditional marginal catch")


def _threshold_plot(ax):
    d = pd.read_csv(T / "rev_diversity_counterexample.csv")
    d["is_best_solo"] = d["is_best_solo"].map({True: True, False: False, "True": True, "False": False})
    c = d[(~d.is_best_solo) & (d.verifier.isin(POOL))].copy()
    n = c.solo_missed.astype(float).clip(lower=1)
    k = (c.marginal_catch_given_solo * n).round().astype(int)
    c["lo"] = [_wilson_lo(int(kk), int(nn)) for kk, nn in zip(k, n)]
    dead = (c.verifier == "llama32_3b").values
    ax.axhspan(0.0, 0.62, color=GREEN_FILL, zorder=0)
    ax.axhspan(-0.2, 0.0, color=GREY_FILL, zorder=0)
    ax.axhline(0, color=RED, lw=2.2, zorder=2)
    ax.text(0.77, 0.03, "ACQUIRE", color=GREEN, fontsize=14, fontweight="bold", ha="right", va="bottom")
    ax.text(0.77, -0.03, "REJECT (don't buy)", color=SECOND, fontsize=13, ha="right", va="top")
    ax.scatter(c.marginal_catch_given_solo.values[~dead], c.lo.values[~dead], s=110, color=BLUE,
               edgecolors="white", linewidths=1.0, zorder=4)
    ax.scatter(c.marginal_catch_given_solo.values[dead], c.lo.values[dead], s=130, color=RED,
               edgecolors="white", linewidths=1.0, zorder=5)
    ax.set_xlim(-0.03, 0.8); ax.set_ylim(-0.12, 0.6)
    _fin(ax, "The controller's decision threshold", "conditional marginal catch  $\\Delta_{\\mathrm{catch}}$",
         "Wilson lower bound  (CI-low)")


def fig_obs2_diversity(fig=None):
    own = fig is None
    if own:
        fig = plt.figure(figsize=(20, 6.1))
    axs = fig.subplots(1, 3)
    # (a) decay lollipop: marginal catch of the k-th same-model call
    rel = pd.read_csv(PROC / "reliability_by_k.csv")
    s = rel[(rel.pairing_condition == "same_model") & (rel.failure_mode_id == "FM-3.3")].sort_values("k")
    marg = np.diff(np.concatenate([[0.0], s.reliability.values]))[:8].clip(min=0)
    y = np.arange(len(marg))[::-1]
    a = axs[0]
    a.hlines(y, 0, marg, color=BASE, lw=3.4, zorder=2)
    cols = [ORANGE if i == 0 else BLUE for i in range(len(marg))]
    a.scatter(marg, y, s=[240 if i == 0 else 130 for i in range(len(marg))], c=cols,
              edgecolors="white", linewidths=1.2, zorder=4)
    for mi, yi in zip(marg, y):
        a.annotate(f"{mi:.2f}", (mi, yi), (12, 0), textcoords="offset points", va="center",
                   fontsize=12.5, color=INK, fontweight="bold")
    a.set_yticks(y); a.set_yticklabels([f"call {i + 1}" for i in range(len(marg))])
    a.set_xlim(0, max(marg) * 1.25 + 0.02)
    a.text(max(marg) * 0.5, y[1] + 0.3, "diminishing returns\ncollapse after call 1", color=SECOND, fontsize=12.5)
    _fin(a, "Count ≠ value (repetition saturates)", "marginal catch at call $k$", grid="x")
    a.grid(axis="y", visible=False)
    _region_scatter(axs[1]); _threshold_plot(axs[2])
    _letters(axs)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.85, bottom=0.13, wspace=0.24)
    if own:
        _save(fig, "fig_obs2_diversity", "Figure 12   Count ≠ diversity ≠ value: repetition, then diversity, then conditional value")


# ================================================================ FIG 13 — family profile strip
def fig_sevenfamily(fig=None):
    own = fig is None
    if own:
        fig = plt.figure(figsize=(20, 6.1))
    axs = fig.subplots(1, 3, sharey=True)
    d = pd.read_csv(T / "redundancy_invariant_7family.csv").set_index("family").reindex(FAM7)
    y = np.arange(len(FAM7))[::-1]
    # (a) n_eff@50 — interval/range plot with null band (most prominent)
    a = axs[0]
    a.axvspan(1.06, 1.21, color=BLUE_FILL, zorder=0, label="null band [1.06, 1.21]")
    a.scatter(d.sm_n_eff50.values, y, s=170, color=BLUE, edgecolors="white", linewidths=1.4, zorder=4)
    for xv, yv in zip(d.sm_n_eff50.values, y):
        a.annotate(f"{xv:.2f}", (xv, yv), (12, 0), textcoords="offset points", va="center",
                   fontsize=12.5, color=INK, fontweight="bold")
    a.set_yticks(y); a.set_yticklabels(FAM7); a.set_xlim(0.95, 1.5); a.legend(loc="lower right")
    _fin(a, "Effective verifiers  $n_{\\mathrm{eff}}$@50", "$n_{\\mathrm{eff}}$@50   (independent = 50)", grid="x")
    a.grid(axis="y", visible=False)
    # (b) correlation — horizontal bars
    b = axs[1]
    b.barh(y, d.sm_rho_v.values, height=0.6, color=VIOLET, edgecolor="white", linewidth=1.0, zorder=3)
    for xv, yv in zip(d.sm_rho_v.values, y):
        b.annotate(f"{xv:.2f}", (xv, yv), (-8, 0), textcoords="offset points", ha="right",
                   va="center", fontsize=12, color="white", fontweight="bold")
    b.set_xlim(0, 1.0)
    _fin(b, "Intra-item correlation  $\\rho_v$", "correlation  $\\rho_v$", grid="x")
    b.grid(axis="y", visible=False)
    # (c) miss rate — dots on a track
    c = axs[2]
    c.hlines(y, 0, d.sm_miss.values, color=GRID, lw=6, zorder=1)
    c.scatter(d.sm_miss.values, y, s=170, color=ORANGE, edgecolors="white", linewidths=1.4, zorder=4)
    for xv, yv in zip(d.sm_miss.values, y):
        c.annotate(f"{xv:.2f}", (xv, yv), (12, 0), textcoords="offset points", va="center",
                   fontsize=12, color=ORANGE_DK, fontweight="bold")
    c.set_xlim(0, 1.0)
    _fin(c, "Same-model miss rate", "miss rate", grid="x")
    c.grid(axis="y", visible=False)
    _letters(axs)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.85, bottom=0.13, wspace=0.12)
    if own:
        _save(fig, "fig_sevenfamily", "Figure 13   Same-model redundancy replicates across seven families (read each row as a profile)")


# ================================================================ FIG 5 — selection evidence ===
def fig_sign_befor(fig=None):
    own = fig is None
    if own:
        fig = plt.figure(figsize=(20, 6.3))
    axs = fig.subplots(1, 3)
    # (a) forest: GSM8K first step, selected=solid / rejected=hollow
    gm = pd.read_csv(T / "gsm8k_marginal.csv")
    s1 = gm[gm.step == 1].copy()
    s1["lab"] = s1.cand_type.str.title() + ": " + s1.candidate
    s1 = s1.sort_values("delta"); y = np.arange(len(s1))
    sel = s1.selected.map({True: True, False: False, "True": True, "False": False}).values
    a = axs[0]
    a.axvline(0, color=RED, ls=(0, (5, 3)), lw=1.8, zorder=1)
    a.hlines(y, s1.delta_ci_low, s1.delta_ci_high, color=BASE, lw=3.2, zorder=2)
    a.scatter(s1.delta[sel], y[sel], s=170, color=BLUE, edgecolors="white", linewidths=1.2, zorder=5)
    a.scatter(s1.delta[~sel], y[~sel], s=170, facecolors="none", edgecolors=MUTE, linewidths=2.4, zorder=4)
    a.set_yticks(y); a.set_yticklabels(s1.lab); a.set_xlim(-0.05, 1.02)
    a.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=12, label="added (solid)"),
                      Line2D([], [], marker="o", ls="", mfc="none", mec=MUTE, mew=2, ms=12, label="rejected (hollow)")],
             loc="lower right")
    _fin(a, "Candidate selection (GSM8K, step 1)", "conditional marginal catch  $\\Delta_{\\mathrm{catch}}$", grid="x")
    # (b) ranked horizontal bars + CI whisker
    d = pd.read_csv(T / "redundancy_invariant_7family.csv").set_index("family")
    d = d.reindex([f for f in FAM7 if f != "MAST-3.3"]).dropna(subset=["div_unique"])
    d["frac"] = d.div_unique / d.div_denom
    d = d.sort_values("frac"); yy = np.arange(len(d))
    b = axs[1]
    b.barh(yy, d.frac.values, height=0.62, color=TEAL, edgecolor="white", linewidth=1.0, zorder=3)
    b.hlines(yy, d.div_ci_low.values, d.frac.values, color=INK, lw=2.2, zorder=4)
    b.scatter(d.div_ci_low.values, yy, marker="|", s=260, color=INK, linewidths=2.4, zorder=5)
    for fr, yi in zip(d.frac.values, yy):
        b.annotate(f"{fr:.2f}", (fr, yi), (10, 0), textcoords="offset points", va="center",
                   fontsize=12, color=INK, fontweight="bold")
    b.set_yticks(yy); b.set_yticklabels(d.index); b.set_xlim(0, 0.95)
    b.annotate("whisker = Wilson lower bound", (float(d.div_ci_low.min()), len(d) - 1), (2, 24),
               textcoords="offset points", fontsize=11.5, color=SECOND)
    _fin(b, "Useful diversification is significant", "survivor errors caught by the added verifier", grid="x")
    b.grid(axis="y", visible=False)
    # (c) dumbbell: solo -> +repeat -> +diverse
    sp = pd.read_csv(T / "fusion_scale_panel.csv").iloc[::-1]; yy = np.arange(len(sp))
    c = axs[2]
    for yi, s0, pdv in zip(yy, sp.solo_catch, sp.plus_diverse):
        c.plot([s0, pdv], [yi, yi], color=BASE, lw=3.4, zorder=2, solid_capstyle="round")
    c.scatter(sp.solo_catch, yy, s=130, color=MUTE, edgecolors="white", linewidths=1.0, zorder=4)
    c.scatter(sp.plus_same_model, yy, s=100, color=RED, edgecolors="white", linewidths=1.0, zorder=6)
    c.scatter(sp.plus_diverse, yy, s=160, color=BLUE, edgecolors="white", linewidths=1.0, zorder=5)
    c.set_yticks(yy); c.set_yticklabels(sp.family); c.set_xlim(0.35, 1.03)
    c.legend(handles=[Line2D([], [], marker="o", ls="", color=MUTE, ms=12, label="solo"),
                      Line2D([], [], marker="o", ls="", color=RED, ms=11, label="+ same-model (+0.003)"),
                      Line2D([], [], marker="o", ls="", color=BLUE, ms=12, label="+ diverse (+0.079)")],
             loc="lower left")
    _fin(c, "Repetition sits still; diversity moves", "error-catch rate", grid="x")
    c.grid(axis="y", visible=False)
    _letters(axs)
    fig.subplots_adjust(left=0.11, right=0.99, top=0.85, bottom=0.13, wspace=0.42)
    if own:
        _save(fig, "fig_sign_befor", "Figure 5   Conditional marginal value selects complementary verifiers")


BUILDERS = [("fig_pareto", fig_pareto, (13.6, 5.4)), ("fig_redun24b", fig_redun24b, (13.6, 6.4)),
            ("fig_saturation", fig_saturation, (20, 6.1)), ("fig_obs2_diversity", fig_obs2_diversity, (20, 6.1)),
            ("fig_sevenfamily", fig_sevenfamily, (20, 6.1)), ("fig_sign_befor", fig_sign_befor, (20, 6.3))]
TITLES = {"fig_pareto": "Figure 4   Cost–performance frontier — CMV-SDSE is the sole non-dominated policy",
          "fig_redun24b": "Figure 6   Redundancy persists beyond 14B — the magnitude shifts with the checkpoint",
          "fig_saturation": "Figure 11   Evidence saturates: nominal effort, early catch plateau, residual false alarms",
          "fig_obs2_diversity": "Figure 12   Count ≠ diversity ≠ value: repetition, then diversity, then conditional value",
          "fig_sevenfamily": "Figure 13   Same-model redundancy replicates across seven families (read each row as a profile)",
          "fig_sign_befor": "Figure 5   Conditional marginal value selects complementary verifiers"}

if __name__ == "__main__":
    for name, fn, _ in BUILDERS:
        fn()
    with PdfPages(OUT / "figures_all.pdf") as pdf:
        for name, fn, size in BUILDERS:
            fig = plt.figure(figsize=size)
            fn(fig=fig)
            fig.suptitle(TITLES[name], x=0.008, ha="left", y=0.975, fontsize=17, fontweight="bold")
            pdf.savefig(fig); plt.close(fig)
    print("wrote", OUT / "figures_all.pdf")
    print("\nAll redesigned figures written to:", OUT)
