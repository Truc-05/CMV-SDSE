"""Regenerate ALL paper figures as SEPARATE panels, grouped (3 or 6 per group).
The author combines each group's a/b/c(/d/e/f) panels into one composite; this script
does NOT montage. Panels within a group share the same size and style.

Naming:  fig_<group>_<letter>.png   (e.g. fig_dmap_a … fig_dmap_f)
Single-panel figures:  fig_<name>.png

GROUPS  (combine the letters of each into figures/<combined>.png):
  fig_dmap_{a..f}       -> fig_decision_maps.png   controller (λ,ρ) map, ONE PER FAMILY (all 6)
  fig_sat_{a,b,c}       -> fig_saturation.png       n_eff(k) all families · reliability(k) · false-alarm(k)
  fig_div_{a,b,c}       -> fig_obs2_diversity.png    count≠value · diversity≠value · selection gate
  fig_seven_{a,b,c}     -> fig_sevenfamily.png       n_eff@50 · rho_v · miss  (all 7 families)
  fig_blind_{a,b,c}     -> fig_blindspot.png          pi0 · power · raw-vs-atom (all families)
SINGLES (already one file, no combining):
  fig_architecture · fig_landscape · fig_significance · fig_beforeafter · fig_regime · fig_ablation · fig_budget
Run:  ~/venvs/env/bin/python scripts/make_figures.py
"""
import ast
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theory.marginal_value import _prep, _wilson

T, OUT = "results/tables", "figures"
os.makedirs(OUT, exist_ok=True)

# ---- muted editorial palette ----
BLUE, CRIMSON, GREEN = "#2a6f97", "#b23a48", "#2e8b57"
AMBER, PURPLE, TEAL = "#e08a1e", "#6a4c93", "#3a8f8f"
GREY, LGREY = "#8a8f98", "#c9ccd1"
INK, MUTE, GRID, BAND = "#2b2b2b", "#6b7178", "#ececec", "#e6eef3"

FAM = {"ARC": "mcqa__arc__ctx_truncate", "MMLU": "mcqa__mmlu__ctx_truncate",
       "CSQA": "mcqa__csqa__ctx_truncate", "TruthfulQA": "mcqa__truthfulqa__ctx_truncate",
       "GSM8K": "gsm8k__gsm8k__ctx_truncate", "MBPP": "mbpp__mbpp__ctx_truncate"}
PHASE = {"ARC": "mcqa_phase_arc", "MMLU": "mcqa_phase_mmlu", "CSQA": "mcqa_phase_csqa",
         "TruthfulQA": "mcqa_phase_truthfulqa", "GSM8K": "gsm8k_phase", "MBPP": "mbpp_phase"}
FAM7 = ["MAST-3.3", "ARC", "MMLU", "CSQA", "TruthfulQA", "GSM8K", "MBPP"]

plt.rcParams.update({
    "savefig.dpi": 170, "figure.dpi": 170, "figure.constrained_layout.use": True,
    "savefig.bbox": "standard", "font.family": "DejaVu Sans",
    "font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 10.5,
    "xtick.labelsize": 9.6, "ytick.labelsize": 9.6, "legend.fontsize": 9,
    "axes.linewidth": 0.9, "figure.facecolor": "white", "axes.facecolor": "white",
})


def rgba(h, a):
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255, a)


def new(size):
    return plt.subplots(figsize=size)


def finish(ax, title="", xlabel="", ylabel="", grid="y", legend_loc=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LGREY)
    ax.tick_params(colors=MUTE, length=3)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color(INK)
    if grid in ("y", "both"):
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    if grid in ("x", "both"):
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK)
    if title:
        ax.set_title(title, color=INK, pad=9)
    if legend_loc:
        lg = ax.legend(loc=legend_loc, framealpha=0.95, edgecolor=LGREY, fancybox=False)
        lg.get_frame().set_linewidth(0.8)


def save(f, name):
    f.savefig(f"{OUT}/{name}.png", facecolor="white")
    plt.close(f)
    print("wrote", name)


def load(slug):
    recs = [json.loads(l) for p in glob.glob(f"data/raw_cascade_runs/{slug}/*.jsonl")
            for l in open(p) if l.strip()]
    df = pd.DataFrame(recs)
    return df[df["n_gates"] == 50]


# =================================================================== SCHEMATICS
def schematic_architecture():
    f, ax = plt.subplots(figsize=(10.6, 3.7)); ax.set_xlim(0, 10.6); ax.set_ylim(0, 3.7); ax.axis("off")
    def box(x, y, w, h, s, fc, fs=9.5, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                     lw=1.1, edgecolor=LGREY, facecolor=fc, zorder=2))
        ax.text(x + w / 2, y + h / 2, s, ha="center", va="center", fontsize=fs, color=INK,
                zorder=3, weight=("bold" if bold else "normal"))
    def arr(x1, y1, x2, y2, c=INK, rad=0.0, lw=1.6):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                     color=c, lw=lw, zorder=1, connectionstyle=f"arc3,rad={rad}"))
    y0, h, w = 2.0, 0.95, 1.28; xs = [0.15, 1.75, 3.35, 4.95, 6.85, 8.45]
    box(xs[0], y0, w, h, "input\n(answer /\ntrace)", "#eef1f4", 9)
    box(xs[1], y0, w, h, "OBSERVE\ngate", rgba(AMBER, .20), 9.5, True)
    box(xs[2], y0, w, h, "verifier\npool  $\\mathcal{V}$", "#eef1f4")
    box(xs[3], y0, 1.55, h, "measurement\n$n_{eff}\\!\\to\\!\\Delta_{catch}\\!\\to\\!V$", rgba(BLUE, .15), 9)
    box(xs[4], y0, w, h, "controller\nCMV-SDSE", rgba(BLUE, .28), 9.5, True)
    box(xs[5], y0, w, h, "fusion", "#eef1f4")
    box(xs[5], 0.35, w, h, "output\n+ audit", rgba(GREEN, .18))
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        arr(xs[a] + (1.55 if a == 3 else w), y0 + h / 2, xs[b], y0 + h / 2)
    arr(xs[3] + 1.55, y0 + h / 2, xs[4], y0 + h / 2); arr(xs[4] + w, y0 + h / 2, xs[5], y0 + h / 2)
    arr(xs[5] + w / 2, y0, xs[5] + w / 2, 0.35 + h)
    arr(xs[4] + w / 2, y0 - 0.02, xs[2] + w / 2, y0 - 0.02, BLUE, -0.45, 1.4)
    ax.text((xs[2] + xs[4]) / 2 + 0.4, y0 - 1.02, "SCALE / DIVERSIFY  (acquire more)", ha="center", fontsize=8.3, color=BLUE)
    arr(xs[1] + w / 2, y0 + h, xs[1] + w / 2, y0 + h + 0.5, CRIMSON, 0, 1.3)
    ax.text(xs[1] + w / 2, y0 + h + 0.62, "fail $\\to$ FIX-CONTEXT", ha="center", fontsize=8.3, color=CRIMSON)
    ax.text(xs[4] + w / 2, y0 + h + 0.32, "STOP / ESCALATE", ha="center", fontsize=8.3, color=INK)
    f.savefig(f"{OUT}/fig_architecture.png", facecolor="white", bbox_inches="tight"); plt.close(f)
    print("wrote fig_architecture")


def schematic_landscape():
    f, ax = new((7.4, 5.0)); x = np.linspace(0, 1, 400); ceil = 0.15 + 0.84 * (1 - (1 - x) ** 2.2); xd = 0.22
    ax.fill_between(x, ceil, 1.0, color=CRIMSON, alpha=0.42)
    ax.fill_between(x, 0, ceil, where=(x >= xd), color=AMBER, alpha=0.32)
    ax.fill_between(x, 0, ceil, where=(x < xd), color=GREY, alpha=0.30)
    ax.fill_between(x, 0, 0.12, color=BLUE, alpha=0.28); ax.plot(x, ceil, color=INK, lw=1.8)
    ax.text(0.03, 0.80, "achievable ceiling  C(χ)", ha="left", fontsize=8.5, color=INK, style="italic")
    ax.text(0.42, 0.93, "ESCALATE", color="white", fontsize=13, weight="bold", ha="center")
    ax.text(0.66, 0.44, "DIVERSIFY", color="#7a4a10", fontsize=13, weight="bold", ha="center")
    ax.text(0.105, 0.40, "SCALE\n(trap)", color="#3f3f3f", fontsize=10, weight="bold", ha="center")
    ax.text(0.5, 0.055, "STOP  (target met)", color="white", fontsize=9.5, weight="bold", ha="center")
    for nm, chi, cy, off, ha, va in [("same-model", 0.02, 0.14, (9, -3), "left", "top"),
            ("same-family", 0.07, 0.56, (9, -3), "left", "top"),
            ("cross-family", 0.55, 0.84, (0, -14), "center", "top"),
            ("full pool", 0.62, 0.86, (13, -4), "left", "center")]:
        ax.scatter([chi], [cy], s=42, color=INK, zorder=5)
        ax.annotate(f"{nm}\nceiling={cy:.2f}", (chi, cy), off, textcoords="offset points", fontsize=7.8, color=INK, ha=ha, va=va)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    finish(ax, "Scale / Diversify / Stop / Escalate — decision map",
           xlabel="complementary evidence in pool  χ  (≈ $n_{eff}$)", ylabel="required reliability (target)", grid="none")
    save(f, "fig_landscape")


# ========================================= GROUP: decision maps (6 per family)
def group_decision_maps():
    ACT = {"DIVERSIFY": 2, "SCALE": 1, "STOP": 0}
    CMAP = ListedColormap([rgba(GREY, .28), rgba(AMBER, .55), rgba(BLUE, .55)])
    for letter, fam in zip("abcdef", FAM):
        d = pd.read_csv(f"{T}/{PHASE[fam]}.csv")
        piv = d.pivot(index="cost_ratio_fa_over_miss", columns="lambda_cost", values="first_action")
        M = piv.map(lambda a: ACT.get(a, 0)).values
        f, ax = new((4.3, 3.9))
        ax.imshow(M, cmap=CMAP, vmin=0, vmax=2, aspect="auto", origin="lower")
        ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=8.5)
        ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(colors=MUTE, length=0)
        s0 = d.s0.mode().iat[0]
        ax.set_title(f"({letter})  {fam}   solo: {s0}", color=INK, pad=8)
        ax.set_xlabel("compute weight  λ", color=INK); ax.set_ylabel("fa / miss ratio  ρ", color=INK)
        # shared legend once (top-left panel), so the combined figure reads cleanly
        if letter == "a":
            ax.legend(handles=[Patch(color=rgba(BLUE, .55), label="DIVERSIFY"),
                               Patch(color=rgba(AMBER, .55), label="SCALE"),
                               Patch(color=rgba(GREY, .28), label="STOP")],
                      loc="upper right", framealpha=0.97, edgecolor=LGREY, fontsize=8)
        save(f, f"fig_dmap_{letter}")


# ========================================= GROUP: saturation (3 panels)
def group_saturation():
    ks = np.arange(1, 51); neff = lambda k, r: k / (1 + (k - 1) * r)
    # (a) n_eff(k) with same-model ρ_v of ALL seven families + independent ideal
    d = pd.read_csv(f"{T}/redundancy_invariant_7family.csv")
    f, ax = new((7.0, 4.4))
    ax.plot(ks, ks, color=GREY, ls="--", lw=1.8, label="independent (ideal)")
    for _, r in d.iterrows():
        ax.plot(ks, neff(ks, r.sm_rho_v), color=CRIMSON, lw=1.6, alpha=0.55)
    ax.plot([], [], color=CRIMSON, lw=1.8, label="same-model, 7 families (ρ=0.82–0.94)")
    ax.plot(ks, neff(ks, 0.20), color=BLUE, lw=2.4, label="cross-family (ρ=0.20)")
    ax.scatter([50], [50], s=24, color=GREY, zorder=5); ax.annotate("50", (50, 50), (-16, 2), textcoords="offset points", fontsize=9, color=MUTE)
    ax.set_xlim(1, 55); ax.set_ylim(0, 53)
    finish(ax, "(a)  50 calls carry the evidence of ~1", xlabel="verifier calls  k",
           ylabel="effective verifiers  $n_{eff}(k)$", legend_loc="upper left")
    save(f, "fig_sat_a")
    # (b) reliability(k) + band, (c) false-alarm(k) + band  (cross-family FM-3.3)
    rel = pd.read_csv(f"{T}/reliability_by_k.csv")
    r = rel[(rel.pairing_condition == "cross_family_3") & (rel.failure_mode_id == "FM-3.3")].sort_values("k")
    f, ax = new((7.0, 4.4))
    ax.fill_between(r.k, r.ci_low, r.ci_high, color=BLUE, alpha=0.16, label="95% cluster bootstrap")
    ax.plot(r.k, 1 - 0.20 ** r.k, color=GREY, ls=":", lw=1.8, label="independent upper bound")
    ax.plot(r.k, r.reliability, "-o", color=BLUE, lw=2.4, ms=5, label="empirical reliability")
    ax.set_ylim(0, 1.02)
    finish(ax, "(b)  catch reliability saturates by k≈6", xlabel="cascade depth  k",
           ylabel="P(error caught by gate ≤ k)", legend_loc="lower right")
    save(f, "fig_sat_b")
    fal = pd.read_csv(f"{T}/false_alarm_by_k.csv")
    fa = fal[(fal.pairing_condition == "cross_family_3") & (fal.failure_mode_id == "FM-3.3")].sort_values("k")
    f, ax = new((7.0, 4.4))
    ax.fill_between(fa.k, fa.ci_low, fa.ci_high, color=CRIMSON, alpha=0.16, label="95% cluster bootstrap")
    ax.plot(fa.k, fa.false_alarm_rate, "-s", color=CRIMSON, lw=2.4, ms=5, label="empirical false-alarm rate")
    ax.set_ylim(0, 1.02)
    finish(ax, "(c)  false alarms plateau at a floor", xlabel="cascade depth  k",
           ylabel="P(clean item flagged by gate ≤ k)", legend_loc="lower right")
    save(f, "fig_sat_c")


# ========================================= GROUP: diversity ≠ value (3 panels)
def _obs2_points():
    P, first = [], []
    for fam in FAM:
        df = load(FAM[fam]); pr = _prep(df, 0.5); Ce, Cf, models = pr["Ce"], pr["Cf"], pr["models"]
        for _, r in df[(df.pairing_condition == "same_model") & (df.ground_truth_is_error)].iterrows():
            j, idx = 0, None
            for g in r["gates"]:
                if g.get("accepted") in (True, False):
                    j += 1
                    if g["accepted"] is False:
                        idx = j; break
            first.append(idx)
        solo = max(models, key=lambda m: Ce[m].mean()); missed = ~Ce[solo]
        rs = np.concatenate([Ce[solo], Cf[solo]]).astype(float)
        for v in models:
            if v == solo:
                continue
            rv = np.concatenate([Ce[v], Cf[v]]).astype(float)
            corr = float(np.corrcoef(rv, rs)[0, 1]) if rv.std() and rs.std() else 0.0
            dc = float(Ce[v][missed].mean()) if missed.any() else 0.0
            k, n = int(Ce[v][missed].sum()), int(missed.sum())
            lo = _wilson(k, n)[0] if n > 0 else 0.0
            P.append((corr, dc, lo, v == "llama32_3b"))
    return np.array(P, dtype=object), first


def _obs2_gemma_point():
    """The #3 non-trivial counterexample: gemma2_9b on CSQA at the protocol-matched k=50 --
    accurate + moderately decorrelated yet zero residual catch. Read the k=50 verdicts ONLY
    (redun_gemma file for gemma, same_model file for the frozen solo qwen7b); do NOT route
    through _prep, which pools the k=3 gemma probe into a contaminated 53-gate point and drops
    gemma as an out-of-pool probe anyway. Mirrors scripts/confirm_diversity_counterexample_k50.py.
    Returns (corr_with_solo, marginal_catch, standalone_catch)."""
    d = "data/raw_cascade_runs/mcqa__csqa__ctx_truncate"

    def reject(path, vid):
        acc = {}
        for l in open(path):
            if not l.strip():
                continue
            r = json.loads(l)
            if "ground_truth_is_error" not in r:
                continue
            cell = acc.setdefault(r["item_id"], [0, 0, bool(r["ground_truth_is_error"])])
            for g in r.get("gates", []) or []:
                if g.get("verifier_id") == vid and g.get("accepted") in (True, False):
                    cell[1] += 1
                    cell[0] += int(g["accepted"] is True)
        return {i: (e, (ok / dec) < 0.5) for i, (ok, dec, e) in acc.items() if dec > 0}

    gem = reject(f"{d}/CSQA__redun_gemma__redun_gemma.jsonl", "gemma2_9b")
    solo = reject(f"{d}/CSQA__same_model__same_model.jsonl", "qwen7b")
    shared = [i for i in gem if i in solo]
    errs = [i for i in shared if gem[i][0]]
    a = np.array([int(gem[i][1]) for i in shared], float)
    b = np.array([int(solo[i][1]) for i in shared], float)
    corr = float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else 0.0
    catch = float(np.mean([gem[i][1] for i in errs]))
    missed = [i for i in errs if not solo[i][1]]
    marg = float(np.mean([gem[i][1] for i in missed])) if missed else 0.0
    return corr, marg, catch


def group_diversity():
    P, first = _obs2_points()
    corr = np.array([p[0] for p in P]); dc = np.array([p[1] for p in P])
    lo = np.array([p[2] for p in P]); dead = np.array([p[3] for p in P]); col = np.where(dead, CRIMSON, BLUE)
    K = 12; tot = len(first); marg = np.array([sum(1 for x in first if x == k) / tot for k in range(1, K + 1)])
    # (a) count != value
    f, ax = new((7.2, 4.6)); ax.bar(range(1, K + 1), marg, color=CRIMSON, ec="white", lw=0.6, width=0.72, zorder=3)
    ax.set_xticks(range(1, K + 1, 2))
    ax.annotate("after the 1st call,\nextra same-model calls add ≈ 0", (2.6, marg[1]), (40, 60),
                textcoords="offset points", fontsize=8.8, color=MUTE, arrowprops=dict(arrowstyle="->", color=MUTE, lw=1))
    finish(ax, "(a)  count ≠ value  (repetition saturates)", xlabel="same-model call index  k", ylabel="marginal catch at call k")
    save(f, "fig_div_a")
    # (b) diversity != value
    f, ax = new((7.2, 4.6)); ax.axhspan(-0.02, 0.03, color=GREY, alpha=0.12, zorder=0)
    ax.scatter(corr, dc, s=85, c=col, ec="white", lw=0.8, zorder=4); ax.axhline(0, color=MUTE, ls=":", lw=1)
    ax.annotate("weak, diverse,\nuseless", (corr[dead].min(), dc[dead][np.argmin(corr[dead])]), (16, 34),
                textcoords="offset points", fontsize=8.6, color=CRIMSON, arrowprops=dict(arrowstyle="->", color=CRIMSON, lw=1))
    # the #3 non-trivial counterexample: an ACCURATE, moderately-decorrelated verifier on the zero line
    gx, gy, gcatch = _obs2_gemma_point()
    ax.scatter([gx], [gy], s=230, marker="*", c=AMBER, ec=INK, lw=0.9, zorder=6)
    ax.annotate(f"accurate 9B (catch {gcatch:.2f}),\nstill 0 residual catch", (gx, gy), (-8, 40),
                textcoords="offset points", fontsize=8.6, color="#7a4a10", ha="right",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.2))
    ax.set_xlim(-0.05, 1.0); ax.set_ylim(-0.03, 0.75)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="candidate"),
                       Line2D([], [], marker="o", ls="", color=CRIMSON, label="llama32_3b (dead 3B)"),
                       Line2D([], [], marker="*", ls="", color=AMBER, mec=INK, mew=0.7, ms=12,
                              label="gemma2_9b · CSQA (k=50)")],
              loc="upper left", framealpha=0.95, edgecolor=LGREY)
    finish(ax, "(b)  diversity ≠ value", xlabel="dependence: correlation with solo", ylabel="conditional marginal catch")
    save(f, "fig_div_b")
    # (c) selection gate
    f, ax = new((7.2, 4.6)); ax.axhspan(0, 0.9, color=GREEN, alpha=0.05, zorder=0)
    ax.scatter(dc, lo, s=85, c=col, ec="white", lw=0.8, zorder=4); ax.axhline(0, color=CRIMSON, ls="--", lw=1.3)
    ax.text(0.03, 0.86, "added: Wilson CI-low > 0", color=GREEN, fontsize=9, transform=ax.transAxes)
    ax.set_xlim(-0.03, 1.0)
    finish(ax, "(c)  the value the controller uses", xlabel="conditional marginal catch  $\\Delta_{catch}$", ylabel="Wilson CI-low")
    save(f, "fig_div_c")


# ========================================= GROUP: seven-family (3 lollipops)
def _lollipop(name, xs, labels, x0, xlim, title, xlabel, band=None, dotcol=CRIMSON):
    y = np.arange(len(labels)); f, ax = new((7.2, 4.6))
    if band:
        ax.axvspan(band[0], band[1], color=BLUE, alpha=0.10, zorder=0, label=f"null band {list(band)}")
    ax.axvline(x0, color=MUTE, ls=":", lw=1); ax.hlines(y, x0, xs, color=LGREY, lw=2.4, zorder=2)
    ax.scatter(xs, y, s=92, color=dotcol, zorder=4)
    for xv, yv in zip(xs, y):
        ax.annotate(f"{xv:.2f}", (xv, yv), (9, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(*xlim)
    finish(ax, title, xlabel=xlabel, grid="x", legend_loc=("lower right" if band else None))
    save(f, name)


def group_sevenfamily():
    d = pd.read_csv(f"{T}/redundancy_invariant_7family.csv").set_index("family").reindex(FAM7).reset_index()
    fams = d.family.tolist()[::-1]
    _lollipop("fig_seven_a", d.sm_n_eff50.tolist()[::-1], fams, 1.0, (0.95, 1.45),
              "(a)  same-model $n_{eff}$@50 (7 families)", "effective verifiers  $n_{eff}$@50   (independent = 50)", band=(1.06, 1.21))
    _lollipop("fig_seven_b", d.sm_rho_v.tolist()[::-1], fams, 0.0, (0, 1.05),
              "(b)  intra-item correlation ρ_v", "correlation  ρ_v")
    _lollipop("fig_seven_c", d.sm_miss.tolist()[::-1], fams, 0.0, (0, 1.0),
              "(c)  same-model miss rate", "miss rate", dotcol=AMBER)


# ============================= SINGLE: significance (2 subplots, one file)
def single_significance():
    f, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.6))
    gm = pd.read_csv(f"{T}/gsm8k_marginal.csv"); s1 = gm[gm.step == 1].copy().sort_values("delta")
    s1["lab"] = s1.cand_type + ":" + s1.candidate; y = np.arange(len(s1)); cols = [BLUE if sel else GREY for sel in s1.selected]
    a1.errorbar(s1.delta, y, xerr=[s1.delta - s1.delta_ci_low, s1.delta_ci_high - s1.delta], fmt="none",
                ecolor=LGREY, elinewidth=1.8, capsize=3, zorder=3)
    a1.scatter(s1.delta, y, c=cols, s=80, zorder=5); a1.axvline(0, color=MUTE, ls=":", lw=1)
    a1.set_yticks(y); a1.set_yticklabels(s1.lab); a1.set_xlim(-0.03, 1.0)
    a1.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="added (CI-low > 0)"),
                       Line2D([], [], marker="o", ls="", color=GREY, label="not added (Δ≈0)")],
              loc="lower right", framealpha=0.95, edgecolor=LGREY)
    finish(a1, "(a)  a diverse verifier that catches nothing (GSM8K)", xlabel="conditional marginal catch  $\\Delta_{catch}$", grid="x")
    d = pd.read_csv(f"{T}/redundancy_invariant_7family.csv").dropna(subset=["div_unique"]).iloc[::-1]
    d["frac"] = d.div_unique / d.div_denom; fams = d.family.tolist(); y = np.arange(len(fams))
    a2.axvline(0, color=MUTE, ls=":", lw=1); a2.hlines(y, d.div_ci_low, d.frac, color=LGREY, lw=2.4, zorder=2)
    a2.scatter(d.div_ci_low, y, marker="|", s=260, color=INK, lw=2, zorder=4); a2.scatter(d.frac, y, s=90, color=BLUE, zorder=5)
    for fr, yi in zip(d.frac, y):
        a2.annotate(f"{fr:.2f}", (fr, yi), (9, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    a2.set_yticks(y); a2.set_yticklabels(fams); a2.set_xlim(-0.02, 0.95)
    finish(a2, "(b)  useful diversification is significant (every family)", xlabel="survivor errors caught by the added verifier", grid="x")
    save(f, "fig_significance")


# ========================================= GROUP: blind spot (3 panels)
def group_blindspot():
    bl = pd.read_csv(f"{T}/blind_spot_largen.csv").groupby("family", sort=False).first().reset_index()
    order = ["MAST-3.3", "ARC", "MMLU", "CSQA", "TruthfulQA"]; bl = bl.set_index("family").reindex(order).reset_index()
    y = np.arange(len(bl))
    f, ax = new((7.2, 4.6)); ax.axvspan(-0.03, 0.0, color=CRIMSON, alpha=0.10, zorder=0)
    ax.errorbar(bl.pi0_hat, y, xerr=[bl.pi0_hat - bl.pi0_ci_lo, bl.pi0_ci_hi - bl.pi0_hat], fmt="o",
                color=BLUE, ecolor=LGREY, elinewidth=2.2, capsize=4, ms=8, zorder=4)
    ax.axvline(0, color=CRIMSON, ls="--", lw=1.3); ax.set_yticks(y); ax.set_yticklabels(bl.family); ax.set_xlim(-0.03, 0.8)
    ax.text(0.985, 0.06, "every CI touches 0 → not identified", transform=ax.transAxes, ha="right", fontsize=8.8, color=MUTE)
    finish(ax, "(a)  blind-spot atom not identified", xlabel="blind-spot mass  π₀  (95% CI)", grid="x")
    save(f, "fig_blind_a")
    bp = pd.read_csv(f"{T}/blind_spot_power.csv"); f, ax = new((7.2, 4.6)); ax.axhspan(0, 0.8, color=CRIMSON, alpha=0.06, zorder=0)
    cols = {"ARC": BLUE, "MMLU": AMBER, "CSQA": GREEN, "TruthfulQA": CRIMSON}
    for fam, s in bp.groupby("family", sort=False):
        ax.plot(s.n_items, s.detect_power, "-o", color=cols.get(fam, GREY), lw=2.2, ms=5, label=fam)
    ax.axhline(0.8, color=MUTE, ls="--", lw=1.1); ax.text(2000, 0.83, "0.8 power", ha="right", fontsize=8.5, color=MUTE)
    ax.set_xscale("log"); ax.set_ylim(0, 1.05)
    finish(ax, "(b)  identifiability wall, not sample size", xlabel="items  n (log)", ylabel="detection power", legend_loc="lower right")
    save(f, "fig_blind_b")
    bl2 = pd.read_csv(f"{T}/blind_spot_largen.csv").groupby("family", sort=False).first().reset_index().set_index("family").reindex(order).reset_index()
    f, ax = new((7.2, 4.6)); x = np.arange(len(bl2)); w = 0.38
    ax.bar(x - w / 2, bl2.raw_frac_at_ceiling, w, color=GREY, ec="white", lw=0.6, label="raw mass at ceiling", zorder=3)
    ax.bar(x + w / 2, bl2.pi0_hat, w, color=BLUE, ec="white", lw=0.6, label="fitted atom  π̂₀", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(bl2.family, rotation=20, ha="right"); ax.legend(frameon=False, loc="upper right")
    finish(ax, "(c)  raw mass vs identified atom", ylabel="fraction of error items")
    save(f, "fig_blind_c")


# ========================================= SINGLES: beforeafter / regime / ablation / budget
def single_beforeafter():
    sp = pd.read_csv(f"{T}/fusion_scale_panel.csv"); fams = sp.family.tolist()[::-1]; y = np.arange(len(fams))
    solo = sp.solo_catch.tolist()[::-1]; psame = sp.plus_same_model.tolist()[::-1]; pdiv = sp.plus_diverse.tolist()[::-1]
    f, ax = new((7.2, 4.6))
    for yi, s, pd_ in zip(y, solo, pdiv):
        ax.plot([s, pd_], [yi, yi], color=BLUE, lw=3, alpha=0.45, zorder=2, solid_capstyle="round")
    ax.scatter(solo, y, s=74, color=GREY, zorder=4); ax.scatter(psame, y, s=52, color=CRIMSON, zorder=6); ax.scatter(pdiv, y, s=84, color=BLUE, zorder=5)
    ax.set_yticks(y); ax.set_yticklabels(fams); ax.set_xlim(0.35, 1.02)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=GREY, label="solo verifier"),
                       Line2D([], [], marker="o", ls="", color=CRIMSON, label="+ same-model repeat"),
                       Line2D([], [], marker="o", ls="", color=BLUE, label="+ diverse verifier")],
              loc="center left", framealpha=0.95, edgecolor=LGREY)
    finish(ax, "Diversity moves the needle; repetition sits on top", xlabel="error-catch rate", grid="x")
    save(f, "fig_beforeafter")


def single_regime():
    cs = pd.read_csv(f"{T}/fusion_cost_sweep.csv"); x = np.arange(len(cs)); rho = cs.cost_ratio_fa_over_miss.tolist()
    short = {"OR (cascade / paper rule)": ("OR", BLUE), "pattern-Bayes stack (CV)": ("stack", TEAL),
             "single-best verifier (CV)": ("single", GREEN), "MAJORITY (>half)": ("MAJ", AMBER), "AND (unanimous)": ("AND", CRIMSON)}
    f, ax = new((7.4, 5.0)); edges = np.concatenate([[x[0] - 0.5], (x[:-1] + x[1:]) / 2, [x[-1] + 0.5]])
    for i, rl in enumerate(cs.best_fixed_rule):
        nm, col = short.get(rl, (rl, GREY)); ax.axvspan(edges[i], edges[i + 1], color=col, alpha=0.16)
        ax.text((edges[i] + edges[i + 1]) / 2, 1.02, nm, ha="center", va="bottom", fontsize=9, color=col)
    ax.plot(x, cs.best_fixed_U, "--s", color=GREY, lw=1.8, ms=6, label="best fixed rule", zorder=3)
    ax.plot(x, cs.controller_U, "-o", color=INK, lw=2.4, ms=7, label="controller (one knob)", zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(rho); ax.set_ylim(-0.1, 1.12)
    finish(ax, "No universal fusion rule — winner is cost-driven", xlabel="false-alarm / miss cost ratio  ρ",
           ylabel="utility  U = catch − ρ·fa", legend_loc="lower left")
    save(f, "fig_regime")


def single_ablation():
    fb = pd.read_csv(f"{T}/fusion_baselines.csv")
    short = {"OR (cascade / paper rule)": "OR\n(cascade)", "pattern-Bayes stack (CV)": "pattern-Bayes\nstack",
             "single-best verifier (CV)": "single-best", "Youden-weighted vote (CV)": "weighted\nvote",
             "MAJORITY (>half)": "MAJORITY", "Dawid-Skene EM (unsup)": "Dawid-\nSkene", "AND (unanimous)": "AND\n(unanimous)"}
    agg = fb[fb.method.isin(short)].groupby("method")[["catch", "fa", "youden"]].mean().sort_values("youden", ascending=False)
    names = [short[m] for m in agg.index]; x = np.arange(len(agg)); w = 0.26
    f, ax = plt.subplots(figsize=(13.0, 4.7))
    for off, col, key, lab in [(-w, GREEN, "catch", "catch (TPR ↑)"), (0, CRIMSON, "fa", "false alarm (FPR ↓)"), (w, BLUE, "youden", "Youden J = catch − fa")]:
        bars = ax.bar(x + off, agg[key], w, color=col, ec="white", lw=0.6, label=lab, zorder=3)
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8.2, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0, 1.22)
    ax.legend(loc="upper right", ncol=3, framealpha=0.95, edgecolor=LGREY)
    finish(ax, "Fusion-rule ablation — catch, false alarm, net value  (macro-avg, 6 families)", ylabel="rate", grid="y")
    save(f, "fig_ablation")


def single_budget():
    b = pd.read_csv(f"{T}/rev_budget_curve.csv"); piv = lambda c: b.pivot(index="k", columns="policy", values=c)
    sty = {"CMV-SDSE": (BLUE, "-o", 2.8), "accuracy": (CRIMSON, "-s", 2.0), "low-correlation": (AMBER, "-^", 2.0),
           "random": (GREY, "-D", 1.8), "same-model": (GREEN, "--", 2.0)}
    f, (a1, a2) = plt.subplots(1, 2, figsize=(11.6, 4.4))
    for ax, col, ti, yl in [(a1, "U", "(a)  more budget, less utility for naive acquisition", "utility  U = catch − ρ·fa  (ρ=1)"),
                            (a2, "catch", "(b)  catch rises but at a false-alarm cost", "catch (TPR)")]:
        Z = piv(col)
        for pol, (c, m, lw) in sty.items():
            ax.plot(Z.index, Z[pol], m, color=c, lw=lw, ms=6, label=pol)
        if col == "U":
            kb = Z["CMV-SDSE"].idxmax(); ax.scatter([kb], [Z["CMV-SDSE"].max()], s=180, facecolor="none", edgecolor=BLUE, lw=2, zorder=6)
            ax.annotate("CMV-SDSE stops here\n(1 verifier, peak)", (kb, Z["CMV-SDSE"].max()), (26, -6), textcoords="offset points", fontsize=8.6, color=BLUE)
        ax.set_xticks([1, 2, 3, 4]); finish(ax, ti, xlabel="verifiers acquired (budget)", ylabel=yl, legend_loc="center right")
    save(f, "fig_budget")


if __name__ == "__main__":
    schematic_architecture(); schematic_landscape()
    group_decision_maps()
    group_saturation()
    group_diversity()
    group_sevenfamily()
    group_blindspot()
    single_significance(); single_beforeafter(); single_regime(); single_ablation(); single_budget()
    print("\nDONE. Combine each fig_<group>_<letter> set into figures/<combined>.png (see header).")
